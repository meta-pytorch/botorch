# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Acquisition-function optimisation for the AddTree conditional space.

This is a **thin** wrapper around BoTorch's
:func:`~botorch.optim.optimize.optimize_acqf_mixed`: the AddTree spec
already exposes its ``fixed_features_list``, so optimisation is a single
call.

Sign convention:
    BoTorch maximises; AddTree's original acquisition (LCB) minimises.
    Negate observed objective values when constructing the
    :class:`~botorch_community.models.addtree.model.AddTreeGP` and pass a
    maximisation acquisition function (the default
    :class:`~botorch.acquisition.UpperConfidenceBound` here).

Contributor: maxc01
"""

from __future__ import annotations

from typing import Any

from botorch.acquisition import (
    AcquisitionFunction,
    qUpperConfidenceBound,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf_mixed

from botorch_community.models.addtree.model import AddTreeGP
from torch import Tensor


__all__ = ["optimize_addtree_acqf"]


def optimize_addtree_acqf(
    model: AddTreeGP,
    *,
    acqf: AcquisitionFunction | None = None,
    beta: float = 1.0,
    q: int = 1,
    num_restarts: int = 2,
    raw_samples: int = 128,
    **optimize_acqf_mixed_kwargs: Any,
) -> tuple[Tensor, Tensor]:
    r"""Optimise an acquisition function over the AddTree space.

    Args:
        model: A fitted :class:`AddTreeGP`. The space is read from
            ``model.addtree_space``.
        acqf: Optional acquisition function. If ``None``, defaults to
            :class:`~botorch.acquisition.UpperConfidenceBound` with the
            supplied ``beta``.
        beta: ``beta`` for the default UCB acquisition. Ignored if
            ``acqf`` is provided.
        q: Batch size (number of candidates). q > 1 is supported for
            q-batch acquisitions; sequential greedy is used internally.
        num_restarts: Multistart count for L-BFGS-B (forwarded).
        raw_samples: Sobol initialisation samples per fixed-features
            entry (forwarded).
        **optimize_acqf_mixed_kwargs: Forwarded to
            :func:`optimize_acqf_mixed`.

    Returns:
        ``(candidates, acq_values)`` exactly as
        :func:`optimize_acqf_mixed` returns; ``candidates`` has shape
        ``(q, space.dim)``.
    """
    space = model.addtree_space
    if acqf is None:
        if q == 1:
            acqf = UpperConfidenceBound(model=model, beta=beta, maximize=True)
        else:
            # q > 1 needs an MC acquisition that supports X_pending.
            acqf = qUpperConfidenceBound(model=model, beta=beta)

    # Move bounds to the model's dtype/device so optimize_acqf_mixed is
    # happy.
    train_X = model.train_inputs[0]
    bounds = space.bounds.to(dtype=train_X.dtype, device=train_X.device)

    return optimize_acqf_mixed(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features_list=list(space.fixed_features_list),
        **optimize_acqf_mixed_kwargs,
    )
