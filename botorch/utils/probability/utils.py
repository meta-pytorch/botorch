#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator

from functools import lru_cache
from math import pi
from numbers import Number
from typing import Any

import torch
from botorch.utils.safe_math import logdiffexp
from numpy.polynomial.legendre import leggauss as numpy_leggauss
from torch import BoolTensor, LongTensor, Tensor

CaseNd = tuple[Callable[[], BoolTensor], Callable[[BoolTensor], Tensor]]

_log_2 = math.log(2)
_sqrt_pi = math.sqrt(pi)
_inv_sqrt_pi = 1 / _sqrt_pi
_inv_sqrt_2pi = 1 / math.sqrt(2 * pi)
_inv_sqrt_2 = 1 / math.sqrt(2)
_neg_inv_sqrt_2 = -_inv_sqrt_2
_log_sqrt_2pi = math.log(2 * pi) / 2
STANDARDIZED_RANGE: tuple[float, float] = (-1e6, 1e6)
_log_two_inv_sqrt_2pi = _log_2 - _log_sqrt_2pi  # = log(2 / sqrt(2 * pi))


def case_dispatcher(
    out: Tensor,
    cases: Iterable[CaseNd] = (),
    default: Callable[[BoolTensor], Tensor] = None,
) -> Tensor:
    r"""Basic implementation of a tensorized switching case statement.

    Args:
        out: Tensor to which case outcomes are written.
        cases: Iterable of function pairs (pred, func), where `mask=pred()` specifies
            whether `func` is applicable for each entry in `out`. Note that cases are
            resolved first-come, first-serve.
        default: Optional `func` to which all unclaimed entries of `out` are dispatched.
    """
    active = None
    for closure, func in cases:
        pred = closure()
        if not pred.any():
            continue

        mask = pred if (active is None) else pred & active
        if not mask.any():
            continue

        if mask.all():  # where possible, use Ellipsis to avoid indexing
            out[...] = func(...)
            return out

        out[mask] = func(mask)
        if active is None:
            active = ~mask
        else:
            active[mask] = False

        if not active.any():
            break

    if default is not None:
        if active is None:
            out[...] = default(...)
        elif active.any():
            out[active] = default(active)

    return out


@lru_cache(maxsize=None)
def get_constants(
    values: Number | Iterator[Number],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Returns scalar-valued Tensors containing each of the given constants.
    Used to expedite tensor operations involving scalar arithmetic. Note that
    the returned Tensors should not be modified in-place."""
    if isinstance(values, Number):
        return torch.full((), values, dtype=dtype, device=device)

    return tuple(torch.full((), val, dtype=dtype, device=device) for val in values)


def get_constants_like(
    values: Number | Iterator[Number],
    ref: Tensor,
) -> Tensor | Iterator[Tensor]:
    return get_constants(values, device=ref.device, dtype=ref.dtype)


def gen_positional_indices(
    shape: torch.Size,
    dim: int,
    device: torch.device | None = None,
) -> Iterator[torch.LongTensor]:
    ndim = len(shape)
    _dim = ndim + dim if dim < 0 else dim
    if _dim >= ndim or _dim < 0:
        raise ValueError(f"dim={dim} invalid for shape {shape}.")

    cumsize = shape[_dim + 1 :].numel()
    for i, s in enumerate(reversed(shape[: _dim + 1])):
        yield torch.arange(0, s * cumsize, cumsize, device=device)[(...,) + i * (None,)]
        cumsize *= s


def build_positional_indices(
    shape: torch.Size,
    dim: int,
    device: torch.device | None = None,
) -> LongTensor:
    return sum(gen_positional_indices(shape=shape, dim=dim, device=device))


@lru_cache(maxsize=None)
def leggauss(deg: int, **tkwargs: Any) -> tuple[Tensor, Tensor]:
    x, w = numpy_leggauss(deg)
    return torch.as_tensor(x, **tkwargs), torch.as_tensor(w, **tkwargs)


def ndtr(x: Tensor) -> Tensor:
    r"""Standard normal CDF."""
    half, neg_inv_sqrt_2 = get_constants_like((0.5, _neg_inv_sqrt_2), x)
    return half * torch.erfc(neg_inv_sqrt_2 * x)


def phi(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    inv_sqrt_2pi, neg_half = get_constants_like((_inv_sqrt_2pi, -0.5), x)
    return inv_sqrt_2pi * (neg_half * x.square()).exp()


def log_phi(x: Tensor) -> Tensor:
    r"""Logarithm of standard normal pdf"""
    log_sqrt_2pi, neg_half = get_constants_like((_log_sqrt_2pi, -0.5), x)
    return neg_half * x.square() - log_sqrt_2pi


def log_ndtr(x: Tensor) -> Tensor:
    """Implementation of log_ndtr that remedies problems of torch.special's version
    for large negative x, where the torch implementation yields Inf or NaN gradients.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(ndtr(x)).
    """
    if not (x.dtype == torch.float32 or x.dtype == torch.float64):
        raise TypeError(
            f"log_Phi only supports torch.float32 and torch.float64 "
            f"dtypes, but received {x.dtype=}."
        )
    neg_inv_sqrt_2, log_2 = get_constants_like((_neg_inv_sqrt_2, _log_2), x)
    return log_erfc(neg_inv_sqrt_2 * x) - log_2


def log_erfc(x: Tensor) -> Tensor:
    """Computes the logarithm of the complementary error function in a numerically
    stable manner. The GitHub issue https://github.com/pytorch/pytorch/issues/31945
    tracks progress toward moving this feature into PyTorch in C++.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(erfc(x)).
    """
    if not (x.dtype == torch.float32 or x.dtype == torch.float64):
        raise TypeError(
            f"log_erfc only supports torch.float32 and torch.float64 "
            f"dtypes, but received {x.dtype=}."
        )
    is_pos = x > 0
    x_pos = x.masked_fill(~is_pos, 0)
    x_neg = x.masked_fill(is_pos, 0)
    return torch.where(
        is_pos,
        torch.log(torch.special.erfcx(x_pos)) - x_pos.square(),
        torch.log(torch.special.erfc(x_neg)),
    )


def log_erfcx(x: Tensor) -> Tensor:
    """Computes the logarithm of the complementary scaled error function in a
    numerically stable manner. The GitHub issue tracks progress toward moving this
    feature into PyTorch in C++: https://github.com/pytorch/pytorch/issues/31945.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(erfcx(x)).
    """
    is_pos = x > 0
    x_pos = x.masked_fill(~is_pos, 0)
    x_neg = x.masked_fill(is_pos, 0)
    return torch.where(
        is_pos,
        torch.special.erfcx(x_pos).log(),
        torch.special.erfc(x_neg).log() + x.square(),
    )


def standard_normal_log_hazard(x: Tensor) -> Tensor:
    """Computes the logarithm of the hazard function of the standard normal
    distribution, i.e. `log(phi(x) / Phi(-x))`.

    Args:
        x: A tensor of any shape, with either float32 or float64 dtypes.

    Returns:
        A Tensor of the same shape `x`, containing the values of the logarithm of the
        hazard function evaluated at `x`.
    """
    # NOTE: using _inv_sqrt_2 instead of _neg_inv_sqrt_2 means we are computing Phi(-x).
    a, b = get_constants_like((_log_two_inv_sqrt_2pi, _inv_sqrt_2), x)
    return a - log_erfcx(b * x)


def log_prob_normal_in(a: Tensor, b: Tensor) -> Tensor:
    r"""Computes the probability that a standard normal random variable takes a value
    in \[a, b\], i.e. log(Phi(b) - Phi(a)), where Phi is the standard normal CDF.
    Returns accurate values and permits numerically stable backward passes for inputs
    in [-1e100, 1e100] for double precision and [-1e20, 1e20] for single precision.
    In contrast, a naive approach is not numerically accurate beyond [-10, 10].

    Args:
        a: Tensor of lower integration bounds of the Gaussian probability measure.
        b: Tensor of upper integration bounds of the Gaussian probability measure.

    Returns:
        Tensor of the log probabilities.
    """
    if not (a < b).all():
        raise ValueError("Received input tensors a, b for which not all a < b.")
    # if abs(b) > abs(a), we use Phi(b) - Phi(a) = Phi(-a) - Phi(-b), since the
    # right tail converges to 0 from below, leading to digit cancellation issues, while
    # the left tail of log_ndtr is well behaved and results in large negative numbers
    rev_cond = b.abs() > a.abs()  # condition for reversal of inputs
    if rev_cond.any():
        c = torch.where(rev_cond, -b, a)
        b = torch.where(rev_cond, -a, b)
        a = c  # after we updated b, can assign c to a
    return logdiffexp(log_a=log_ndtr(a), log_b=log_ndtr(b))


def swap_along_dim_(
    values: Tensor,
    i: int | LongTensor,
    j: int | LongTensor,
    dim: int,
    buffer: Tensor | None = None,
) -> Tensor:
    r"""Swaps Tensor slices in-place along dimension `dim`.

    When passed as Tensors, `i` (and `j`) should be `dim`-dimensional tensors
    with the same shape as `values.shape[:dim]`. The xception to this rule occurs
    when `dim=0`, in which case `i` (and `j`) should be (at most) one-dimensional
    when passed as a Tensor.

    Args:
        values: Tensor whose values are to be swapped.
        i: Indices for slices along dimension `dim`.
        j: Indices for slices along dimension `dim`.
        dim: The dimension of `values` along which to swap slices.
        buffer: Optional buffer used internally to store copied values.

    Returns:
        The original `values` tensor.
    """
    dim = values.ndim + dim if dim < 0 else dim
    if dim and (
        (isinstance(i, Tensor) and i.ndim) or (isinstance(j, Tensor) and j.ndim)
    ):
        # Handle n-dimensional batches of heterogeneous swaps via linear indexing
        if isinstance(i, Tensor) and i.shape != values.shape[:dim]:
            raise ValueError("Batch shapes of `i` and `values` do not match.")

        if isinstance(j, Tensor) and j.shape != values.shape[:dim]:
            raise ValueError("Batch shapes of `j` and `values` do not match.")

        pidx = build_positional_indices(
            shape=values.shape[: dim + 1], dim=-2, device=values.device
        )

        swap_along_dim_(
            values.view(-1, *values.shape[dim + 1 :]),
            i=(pidx + i).view(-1),
            j=(pidx + j).view(-1),
            dim=0,
            buffer=buffer,
        )

    else:
        # Base cases: homogeneous swaps and 1-dimenensional heterogeneous swaps
        if isinstance(i, Tensor) and i.ndim > 1:
            raise ValueError("Tensor `i` must be at most 1-dimensional when `dim=0`.")

        if isinstance(j, Tensor) and j.ndim > 1:
            raise ValueError("Tensor `j` must be at most 1-dimensional when `dim=0`.")

        if dim:
            ctx = tuple(slice(None) for _ in range(dim))
            i = ctx + (i,)
            j = ctx + (j,)

        if buffer is None:
            buffer = values[i].clone()
        else:
            buffer.copy_(values[i])

        values[i] = values[j]
        values[j] = buffer

    return values


def compute_log_prob_feas_from_bounds(
    con_lower_inds: Tensor,
    con_upper_inds: Tensor,
    con_both_inds: Tensor,
    con_lower: Tensor,
    con_upper: Tensor,
    con_both: Tensor,
    means: Tensor,
    sigmas: Tensor,
) -> Tensor:
    r"""Compute logarithm of the feasibility probability for each batch of mean/sigma.

    Args:
        means: A `(b) x m`-dim Tensor of means.
        sigmas: A `(b) x m`-dim Tensor of standard deviations.
        con_lower_inds: 1d Tensor of indices con_lower applies to
            in the second dimension of means and sigmas.
        con_upper_inds: 1d Tensor of indices con_upper applies to
            in the second dimension of means and sigmas.
        con_both_inds: 1d Tensor of indices con_both applies to
            in the second dimension of means and sigmas.
        con_lower: 1d Tensor of lower bounds on the constraints
            equal in dimension to con_lower_inds.
        con_upper: 1d Tensor of upper bounds on the constraints
            equal in dimension to con_upper_inds.
        con_both: 2d Tensor of "both" bounds on the constraints
            equal in length to con_both_inds.

    Returns:
        A `(b)`-dim tensor of log feasibility probabilities
    """
    # indices are integers, so we don't cast the type
    con_upper_inds = con_upper_inds.to(device=means.device)
    con_lower_inds = con_lower_inds.to(device=means.device)
    con_both_inds = con_both_inds.to(device=means.device)

    con_lower = con_lower.to(device=means.device, dtype=means.dtype)
    con_upper = con_upper.to(device=means.device, dtype=means.dtype)
    con_both = con_both.to(device=means.device, dtype=means.dtype)

    log_prob = torch.zeros_like(means[..., 0])
    if len(con_lower_inds) > 0:
        i = con_lower_inds
        dist_l = (con_lower - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_ndtr(-dist_l).sum(dim=-1)  # 1 - Phi(x) = Phi(-x)
    if len(con_upper_inds) > 0:
        i = con_upper_inds
        dist_u = (con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_ndtr(dist_u).sum(dim=-1)
    if len(con_both_inds) > 0:
        i = con_both_inds
        con_lower, con_upper = con_both[:, 0], con_both[:, 1]
        # scaled distance to lower and upper constraint boundary:
        dist_l = (con_lower - means[..., i]) / sigmas[..., i]
        dist_u = (con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_prob_normal_in(a=dist_l, b=dist_u).sum(dim=-1)
    return log_prob


def percentile_of_score(data: Tensor, score: Tensor, dim: int = -1) -> Tensor:
    """Compute the percentile rank of `score` relative to `data`.
    For example, if this function returns 70 then 70% of the
    values in `data` are below `score`.

    This implementation is based on `scipy.stats.percentileofscore`,
    with `kind='rank'` and `nan_policy='propagate'`, which is the default.

    Args:
        data: A `... x n x output_shape`-dim Tensor of data.
        score: A `... x 1 x output_shape`-dim Tensor of scores.

    Returns:
        A `... x output_shape`-dim Tensor of percentile ranks.
    """
    # based on scipy.stats.percentileofscore
    left = torch.count_nonzero(data < score, dim=dim)
    right = torch.count_nonzero(data <= score, dim=dim)
    plus1 = left < right
    perct = (left + right + plus1) * (50.0 / data.shape[dim])
    # perct shape: `... x output_shape`
    # fill in nans due to current trial progression being nan
    nan_mask = torch.broadcast_to(torch.isnan(score.squeeze(dim)), perct.shape)
    perct[nan_mask] = torch.nan
    # fill in nans due to previous trial progressions being nan
    nan_mask = torch.broadcast_to(torch.any(torch.isnan(data), dim=dim), perct.shape)
    perct[nan_mask] = torch.nan
    return perct
