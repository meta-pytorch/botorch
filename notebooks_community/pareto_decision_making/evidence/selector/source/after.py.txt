# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""A posteriori Pareto decisions using knee regions and stated preferences.

Contributor: @M-Colley.

Implements equations (8)--(13) of Schmitt, Hoffmann, Rodemann, and Adamy
(2022), "Incorporating Human Preferences in Decision Making for Dynamic
Multi-Objective Optimization in Model Predictive Control",
https://doi.org/10.3390/inventions7030046.

The method chooses among available outcomes. It does not learn a utility
function, select information-gathering experiments, or guarantee safety.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


def _validate_y(Y: Tensor) -> None:
    if not isinstance(Y, Tensor) or not Y.is_floating_point():
        raise TypeError("Y must be a floating point torch.Tensor.")
    if Y.ndim < 2 or Y.numel() == 0:
        raise ValueError("Y must have nonempty shape (..., n, m).")
    if not torch.isfinite(Y).all():
        raise ValueError("Y must contain only finite values.")


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number in [0, 1].")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    return float(alpha)


def _pareto(costs: Tensor) -> Tensor:
    # Reuse upstream's memory-bounded dominance and duplicate handling.
    return is_non_dominated(costs, maximize=False, deduplicate=True)


def _prepare(Y: Tensor, alpha: float, maximize: bool) -> tuple[Tensor, Tensor]:
    _validate_y(Y)
    alpha = _alpha(alpha)
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    costs = -Y if maximize else Y
    pareto = _pareto(costs)
    # Keep rectangular tensors instead of extracting a different-length front
    # per batch. All batch dimensions are processed in tensor operations.
    valid = pareto.unsqueeze(-1)
    lower = costs.masked_fill(~valid, math.inf).amin(dim=-2, keepdim=True)
    upper = costs.masked_fill(~valid, -math.inf).amax(dim=-2, keepdim=True)
    # Scale before subtracting: the difference of finite endpoints can overflow.
    scale = torch.maximum(lower.abs(), upper.abs()).clamp_min(1)
    lower_scaled, upper_scaled = lower / scale, upper / scale
    span = upper_scaled - lower_scaled
    Z = (costs / scale - lower_scaled) / torch.where(
        span > 0, span, torch.ones_like(span)
    )
    Z = torch.where(span > 0, Z, torch.zeros_like(Z))
    norm_squared = Z.square().sum(dim=-1).masked_fill(~pareto, -math.inf)
    q = norm_squared.argmax(dim=-1, keepdim=True)
    sums = Z.sum(dim=-1)
    distances = (sums.gather(-1, q) - sums) / math.sqrt(Y.shape[-1])
    maximum = distances.masked_fill(~pareto, -math.inf).amax(dim=-1, keepdim=True)
    return pareto & (distances >= alpha * maximum), Z


def knee_region(Y: Tensor, *, alpha: float = 0.5, maximize: bool = True) -> Tensor:
    r"""Identify the knee region of a finite Pareto approximation.

    Dominated outcomes and duplicate rows after their first occurrence are
    excluded before per-objective range normalization. The signed distance
    from a plane through the normalized row with largest Euclidean norm is
    compared with ``alpha * maximum_distance`` (Schmitt et al., equations
    8--11). Flat objectives contribute zero. Geometry depends on the supplied
    front approximation; changes in its observed ranges can change decisions.

    Args:
        Y: A finite floating-point tensor of shape ``batch_shape x n x m``.
        alpha: Relative distance threshold in ``[0, 1]``. Larger values narrow
            the knee region. Even zero can exclude negative-distance rows.
        maximize: Whether all objectives are maximized. Set to False for costs.
            Orient columns beforehand when objective directions differ.

    Returns:
        A boolean tensor of shape ``batch_shape x n`` on the input device,
        indexed by the original rows. Discrete masks are not differentiable.
    """
    return _prepare(Y, alpha, maximize)[0]


def select_preferred(
    Y: Tensor,
    preferences: Tensor | Sequence[float],
    *,
    alpha: float = 0.5,
    maximize: bool = True,
) -> Tensor:
    r"""Select one knee-region outcome using a supporting preference plane.

    Minimizes the preference-weighted normalized costs over the knee region
    (Schmitt et al., equation 13). Weights represent stated priorities, not
    probabilities or promised shares of achieved outcomes. Filtering can
    exclude a preferred extreme even for a single nonzero weight.

    Args:
        Y: A finite floating-point tensor of shape ``batch_shape x n x m``.
        preferences: Finite nonnegative weights of shape ``(..., m)``, with
            at least one positive entry per vector. The leading dimensions
            must broadcast to ``batch_shape``. Positive rescaling is immaterial.
        alpha: Relative knee distance threshold in ``[0, 1]``.
        maximize: Whether all objectives are maximized. Set to False for costs.

    Returns:
        Int64 original-row indices of shape ``batch_shape`` on ``Y.device``.
        Equal scores choose the first original row. Indices are discrete;
        this function is not a differentiable acquisition or utility model.
    """
    mask, Z = _prepare(Y, alpha, maximize)
    p = torch.as_tensor(preferences, device=Y.device, dtype=Y.dtype)
    if p.ndim == 0 or p.shape[-1] != Y.shape[-1]:
        raise ValueError("preferences must have shape (..., m).")
    if not torch.isfinite(p).all() or (p < 0).any() or (p.amax(-1) <= 0).any():
        raise ValueError("preferences must be finite, nonnegative, and nonzero.")
    try:
        p = torch.broadcast_to(p, (*Y.shape[:-2], Y.shape[-1]))
    except RuntimeError as exc:
        raise ValueError("preferences do not broadcast to Y's batch shape.") from exc
    p = p / p.amax(dim=-1, keepdim=True)
    scores = (Z * p.unsqueeze(-2)).sum(dim=-1).masked_fill(~mask, math.inf)
    return scores.argmin(dim=-1)
