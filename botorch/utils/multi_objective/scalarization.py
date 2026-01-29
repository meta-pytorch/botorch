#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helper utilities for constructing scalarizations.

References

.. [Knowles2005]
    J. Knowles, "ParEGO: a hybrid algorithm with on-line landscape approximation
    for expensive multiobjective optimization problems," in IEEE Transactions
    on Evolutionary Computation, vol. 10, no. 1, pp. 50-66, Feb. 2006.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.transforms import normalize
from torch import Tensor


def get_chebyshev_scalarization(
    weights: Tensor, Y: Tensor, alpha: float = 0.05
) -> Callable[[Tensor, Tensor | None], Tensor]:
    r"""Construct an augmented Chebyshev scalarization.

    The augmented Chebyshev scalarization is given by
        g(y) = max_i(w_i * y_i) + alpha * sum_i(w_i * y_i)

    where the goal is to minimize g(y) in the setting where all objectives y_i are
    to be minimized. Since the default in BoTorch is to maximize all objectives,
    this method constructs a Chebyshev scalarization where the inputs are first
    multiplied by -1, so that all objectives are to be minimized. Then, it computes
    g(y) (which should be minimized), and returns -g(y), which should be maximized.

    Minimizing an objective is supported by passing a negative
    weight for that objective. To make all w * y's have the same sign
    such that they are comparable when computing max(w * y), outcomes of minimization
    objectives are shifted from [0,1] to [-1,0].

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A ``m``-dim tensor of weights.
            Positive for maximization and negative for minimization.
        Y: A ``n x m``-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1] or [-1,0]. If ``n=0``, then outcomes
            are left unnormalized.
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, -0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    # the chebyshev_obj assumes all objectives should be minimized, so
    # multiply Y by -1
    Y = -Y
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    def chebyshev_obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
        product = weights * Y
        return product.max(dim=-1).values + alpha * product.sum(dim=-1)

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0
    if Y.shape[-2] == 0:
        if minimize.any():
            raise UnsupportedError(
                "negative weights (for minimization) are only supported if "
                "Y is provided."
            )
        # If there are no observations, we do not need to normalize the objectives

        def obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
            # multiply the scalarization by -1, so that the scalarization should
            # be maximized
            return -chebyshev_obj(Y=-Y)

        return obj
    # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m.
    Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    def obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(-Y, bounds=Y_bounds)
        # If minimizing an objective, convert Y_normalized values to [-1,0],
        # such that min(w*y) makes sense, we want all w*y's to be positive
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        # multiply the scalarization by -1, so that the scalarization should
        # be maximized
        return -chebyshev_obj(Y=Y_normalized)

    return obj
