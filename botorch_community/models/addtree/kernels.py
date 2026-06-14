# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""AddTree covariance kernel for GPyTorch.

The AddTree kernel is

.. math::
    k(x, x') = c \cdot \sum_{v \in \mathrm{tree}}
        k_{\Delta,v}(\mathrm{flag}_v(x), \mathrm{flag}_v(x'))
        \cdot k_{\mathrm{RBF},v}(\mathrm{data}_v(x), \mathrm{data}_v(x'))

where the sum runs over every node ``v`` in the BFS-ordered tree.
``flag_v`` is the node's ``local_id`` slot in the BFS encoding (or the
sentinel ``-1`` if the node is off-path), and ``data_v`` is the
contiguous block of continuous-parameter slots immediately following it.

This module provides:

* :class:`AddTreeDeltaKernel` -- a stationary 1-d kernel that returns
  ``1`` if its inputs are equal and ``0`` otherwise. The categorical
  factor of the AddTree sum.
* :func:`build_addtree_kernel` -- assembles the additive sum from an
  :class:`~botorch_community.models.addtree.space.AddTreeSpace`.

References:

.. [Ma2020addtree]
    X. Ma and M. Blaschko. Additive Tree-Structured Covariance Function
    for Conditional Parameter Spaces in Bayesian Optimization. AISTATS
    2020.

Contributor: maxc01
"""

from __future__ import annotations

import math
from functools import reduce

import torch

from botorch_community.models.addtree.space import AddTreeSpace
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from torch import Tensor


__all__ = ["AddTreeDeltaKernel", "build_addtree_kernel"]


# ``metric_bounds = [(-7, 4)]`` (george-AddTree paper) means
# ``log(metric) in [-7, 4]``, where ``metric = lengthscale**2``. So the
# plain-lengthscale interval is ``[exp(-3.5), exp(2)] ~= [0.030, 7.39]``.
_LS_LOWER = math.exp(-3.5)
_LS_UPPER = math.exp(2.0)
# ``ConstantKernel(-0.69, bounds=[(-7, 4)])`` -> ``c in [exp(-7), exp(4)]``.
_OS_LOWER = math.exp(-7.0)
_OS_UPPER = math.exp(4.0)
# Initial values from the AddTree paper.
_LS_INIT = math.sqrt(0.5)
_OS_INIT = math.exp(-0.69)


class AddTreeDeltaKernel(Kernel):
    r"""Hard delta kernel: ``K(x1, x2) = 1 if x1 == x2 else 0``.

    Stationary, no learnable parameters, not differentiable. Designed to
    be multiplied by a continuous kernel (e.g. :class:`RBFKernel`) on a
    disjoint set of ``active_dims``. Equality is exact tensor equality
    along the (typically 1-d) active dimensions; the AddTree encoding
    only stores integer-valued floats in the dimensions this kernel
    operates on, so this is well-defined.
    """

    has_lengthscale = False
    is_stationary = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        if last_dim_is_batch:
            # GPyTorch convention: move the last dim into a leading batch dim.
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        if diag:
            eq = (x1 == x2).all(dim=-1)
            return eq.to(x1.dtype)
        eq = (x1.unsqueeze(-2) == x2.unsqueeze(-3)).all(dim=-1)
        return eq.to(x1.dtype)


def build_addtree_kernel(
    space: AddTreeSpace,
    *,
    lengthscale_init: float = _LS_INIT,
    outputscale_init: float = _OS_INIT,
    lengthscale_bounds: tuple[float, float] = (_LS_LOWER, _LS_UPPER),
    outputscale_bounds: tuple[float, float] = (_OS_LOWER, _OS_UPPER),
) -> ScaleKernel:
    r"""Build the AddTree kernel for an :class:`AddTreeSpace`.

    Args:
        space: The :class:`AddTreeSpace` defining the tree.
        lengthscale_init: Initial lengthscale for per-node RBF kernels.
            Defaults to ``sqrt(0.5)`` (matches the AddTree paper).
        outputscale_init: Initial value of the outer ``ScaleKernel``.
            Defaults to ``exp(-0.69)``.
        lengthscale_bounds: ``(lower, upper)`` interval constraint for
            the RBF lengthscale. Defaults match the paper's
            ``metric_bounds = [(-7, 4)]`` translated from log-metric to
            plain lengthscale (i.e. ``[exp(-3.5), exp(2)]``).
        outputscale_bounds: ``(lower, upper)`` constraint for the
            ``ScaleKernel.outputscale``.

    Returns:
        A :class:`ScaleKernel` wrapping the additive sum
        ``Sum_v DeltaKernel(active_dims=[flag_v]) *
                 RBFKernel(active_dims=cont_dims_v)`` over all nodes.
    """
    ls_constraint = Interval(*lengthscale_bounds)
    os_constraint = Interval(*outputscale_bounds)

    components: list[Kernel] = []
    for rec in space._all_records:
        delta = AddTreeDeltaKernel(active_dims=(rec.flag_index,))
        if not rec.cont_indices:
            components.append(delta)
            continue

        rbf = RBFKernel(
            ard_num_dims=len(rec.cont_indices),
            active_dims=tuple(rec.cont_indices),
            lengthscale_constraint=ls_constraint,
        )
        # Initialise lengthscale strictly inside (lo, hi).
        ls = max(lengthscale_bounds[0] * 1.01, lengthscale_init)
        ls = min(lengthscale_bounds[1] * 0.99, ls)
        with torch.no_grad():
            rbf.lengthscale = torch.full(
                (1, len(rec.cont_indices)),
                ls,
                dtype=torch.get_default_dtype(),
            )
        components.append(delta * rbf)

    additive = reduce(lambda a, b: a + b, components)
    scaled = ScaleKernel(additive, outputscale_constraint=os_constraint)
    os = max(outputscale_bounds[0] * 1.01, outputscale_init)
    os = min(outputscale_bounds[1] * 0.99, os)
    with torch.no_grad():
        scaled.outputscale = os
    return scaled
