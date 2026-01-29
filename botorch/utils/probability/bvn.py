#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Methods for computing bivariate normal probabilities and statistics.

.. [Genz2004bvnt]
    A. Genz. Numerical computation of rectangular bivariate and trivariate normal and
    t probabilities. Statistics and Computing, 2004.

.. [Muthen1990moments]
    B. Muthen. Moments of the censored and truncated bivariate normal distribution.
    British Journal of Mathematical and Statistical Psychology, 1990.
"""

from __future__ import annotations

from math import pi as _pi

import torch
from botorch.exceptions import UnsupportedError
from botorch.utils.probability.utils import (
    case_dispatcher,
    leggauss,
    ndtr as Phi,
    phi,
    STANDARDIZED_RANGE,
)
from botorch.utils.safe_math import (
    div as safe_div,
    exp as safe_exp,
    mul as safe_mul,
    sub as safe_sub,
)
from torch import Tensor

# Some useful constants
_2pi = 2 * _pi
_sqrt_2pi = _2pi**0.5
_inv_2pi = 1 / _2pi


def bvn(r: Tensor, xl: Tensor, yl: Tensor, xu: Tensor, yu: Tensor) -> Tensor:
    r"""A function for computing bivariate normal probabilities.

    Calculates ``P(xl < x < xu, yl < y < yu)`` where ``x`` and ``y`` are bivariate
    normal with unit variance and correlation coefficient ``r``. See Section 2.4 of
    [Genz2004bvnt]_.

    This method uses a sign flip trick to improve numerical performance. Many of
    ``bvnu``s internal branches rely on evaluations ``Phi(-bound)``. For
    ``a < b < 0``, the term ``Phi(-a) - Phi(-b)`` goes to zero faster than
    ``Phi(b) - Phi(a)`` because ``finfo(dtype).epsneg`` is typically much larger
    than ``finfo(dtype).tiny``. In these cases, flipping the sign can prevent
    situations where ``bvnu(...) - bvnu(...)`` would otherwise be zero due to
    round-off error.

    Args:
        r: Tensor of correlation coefficients.
        xl: Tensor of lower bounds for ``x``, same shape as ``r``.
        yl: Tensor of lower bounds for ``y``, same shape as ``r``.
        xu: Tensor of upper bounds for ``x``, same shape as ``r``.
        yu: Tensor of upper bounds for ``y``, same shape as ``r``.

    Returns:
        Tensor of probabilities ``P(xl < x < xu, yl < y < yu)``.

    """
    if not (r.shape == xl.shape == xu.shape == yl.shape == yu.shape):
        raise UnsupportedError("Arguments to `bvn` must have the same shape.")

    # Sign flip trick
    flip_x = xl.abs() > xu  # is xl more negative than xu is positive?
    flip_y = yl.abs() > yu
    flip = (flip_x & (~flip_y | yu.isinf())) | (flip_y & (~flip_x | xu.isinf()))
    if (
        flip.any()
    ):  # symmetric calls to ``bvnu`` below makes swapping bounds unnecessary
        sign = 1 - 2 * flip.to(dtype=r.dtype)
        xl = sign * xl  # becomes ``-xu`` if flipped
        xu = sign * xu  # becomes ``-xl``
        yl = sign * yl  # becomes ``-yu``
        yu = sign * yu  # becomes ``-yl``
    # Computing the sum of terms this way instead of simply adding them
    # stops gradients from being computed for terms that are zero.
    # This tends to improve gradient stability, because these zeros usually
    # started as non-finite values that were zeroed out by safe math.
    terms = bvnu(r, xl, yl), -bvnu(r, xl, yu), -bvnu(r, xu, yl), bvnu(r, xu, yu)
    res = torch.zeros_like(terms[0])
    for term in terms:
        if (term != 0).any():
            res = res + term
    return res.clip(min=0, max=1)


def bvnu(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
    r"""Solves for ``P(x > h, y > k)`` where ``x`` and ``y`` are standard bivariate
    normal random variables with correlation coefficient ``r``. In [Genz2004bvnt]_,
    this is (1)

        `L(h, k, r) = P(x < -h, y < -k) \
        = 1/(a 2\pi) \int_{h}^{\infty} \int_{k}^{\infty} f(x, y, r) dy dx,`

    where ``f(x, y, r) = e^{-1/(2a^2) (x^2 - 2rxy + y^2)}`` and ``a = (1 - r^2)^{1/2}``.

    [Genz2004bvnt]_ report the following integation scheme incurs a maximum of 5e-16
    error when run in double precision: if ``|r| >= 0.925``, use a 20-point quadrature
    rule on a 5th order Taylor expansion; else, numerically integrate in polar
    coordinates using no more than 20 quadrature points.

    Args:
        r: Tensor of correlation coefficients.
        h: Tensor of negative upper bounds for ``x``, same shape as ``r``.
        k: Tensor of negative upper bounds for ``y``, same shape as ``r``.

    Returns:
        A tensor of probabilities ``P(x > h, y > k)``.
    """
    if not (r.shape == h.shape == k.shape):
        raise UnsupportedError("Arguments to `bvnu` must have the same shape.")
    lower, upper = STANDARDIZED_RANGE
    x_free = h < lower
    y_free = k < lower
    return case_dispatcher(
        out=torch.empty_like(r),
        cases=(  # Special cases admitting closed-form solutions
            (lambda: (h > upper) | (k > upper), lambda mask: 0),
            (lambda: x_free & y_free, lambda mask: 1),
            (lambda: x_free, lambda mask: Phi(-k[mask])),
            (lambda: y_free, lambda mask: Phi(-h[mask])),
            (lambda: r == 0, lambda mask: Phi(-h[mask]) * Phi(-k[mask])),
            (  # For |r| >= 0.925, use a Taylor approximation
                lambda: r.abs() >= 0.925,
                lambda m: _bvnu_taylor(r[m], h[m], k[m]),
            ),
        ),  # For |r| < 0.925, integrate in polar coordinates.
        default=lambda mask: _bvnu_polar(r[mask], h[mask], k[mask]),
    )


def _bvnu_polar(
    r: Tensor, h: Tensor, k: Tensor, num_points: int | None = None
) -> Tensor:
    r"""Solves for ``P(x > h, y > k)`` by integrating in polar coordinates as

        `L(h, k, r) = \Phi(-h)\Phi(-k) + 1/(2\pi) \int_{0}^{sin^{-1}(r)} f(t) dt \
        f(t) = e^{-0.5 cos(t)^{-2} (h^2 + k^2 - 2hk sin(t))}`

    For details, see Section 2.2 of [Genz2004bvnt]_.
    """
    if num_points is None:
        mar = r.abs().max()
        num_points = 6 if mar < 0.3 else 12 if mar < 0.75 else 20

    x, w = leggauss(num_points, dtype=r.dtype, device=r.device)
    x = x + 1
    asin_r = 0.5 * torch.asin(r)
    sin_asrx = (asin_r.unsqueeze(-1) * x).sin()

    _h = h.unsqueeze(-1)
    _k = k.unsqueeze(-1)
    vals = safe_exp(
        safe_sub(safe_mul(sin_asrx, _h * _k), 0.5 * (_h.square() + _k.square()))
        / (1 - sin_asrx.square())
    )
    probs = Phi(-h) * Phi(-k) + _inv_2pi * asin_r * (vals @ w)
    return probs.clip(min=0, max=1)  # necessary due to "safe" handling of inf


def _bvnu_taylor(r: Tensor, h: Tensor, k: Tensor, num_points: int = 20) -> Tensor:
    r"""Solves for ``P(x > h, y > k)`` via Taylor expansion.

    Per Section 2.3 of [Genz2004bvnt]_, the bvnu equation (1) may be rewritten as

        `L(h, k, r) = L(h, k, s) - s/(2\pi) \int_{0}^{a} f(x) dx \
        f(x) = (1 - x^2){-1/2} e^{-0.5 ((h - sk)/ x)^2} e^{-shk/(1 + (1 - x^2)^{1/2})},`

    where ``s = sign(r)`` and ``a = sqrt(1 - r^{2})``. The term ``L(h, k, s)`` is
    analytic. The second integral is approximated via Taylor expansion. See Sections
    2.3 and 2.4 of [Genz2004bvnt]_.
    """
    x, w = leggauss(num_points, dtype=r.dtype, device=r.device)
    x = x + 1

    s = 2 * (r > 0).to(r) - 1  # sign of ``r`` where sign(0) := 1
    sk = s * k
    skh = sk * h
    comp_r2 = 1 - r.square()

    a = comp_r2.clip(min=0).sqrt()
    b = safe_sub(h, sk)
    b2 = b.square()
    c = (4 - skh) / 8.0
    d = (12 - skh) / 80.0

    # ---- Solve for ``L(h, k, s)``
    int_from_0_to_s = case_dispatcher(
        out=torch.empty_like(r),
        cases=[(lambda: r > 0, lambda mask: Phi(-torch.maximum(h[mask], k[mask])))],
        default=lambda mask: (Phi(sk[mask]) - Phi(h[mask])).clip(min=0),
    )

    # ---- Solve for ``s/(2\pi) \int_{0}^{a} f(x) dx``
    # Analytic part
    _a0 = -0.5 * (safe_div(b2, comp_r2) + skh)
    _a1 = c * (1 - d * b2) / 3.0
    _a2 = 1 - b2 * _a1
    abs_b = b.abs()
    analytic_part = torch.subtract(  # analytic part of solution
        a * (_a2 + comp_r2 * _a1 + c * d * comp_r2.square()) * safe_exp(_a0),
        _sqrt_2pi * Phi(safe_div(-abs_b, a)) * abs_b * _a2 * safe_exp(-0.5 * skh),
    )

    # Quadrature part
    _b2 = b2.unsqueeze(-1)
    _skh = skh.unsqueeze(-1)
    _q0 = 0.25 * comp_r2.unsqueeze(-1) * x.square()
    _q1 = (1 - _q0).sqrt()
    _q2 = -0.5 * (_b2 / _q0 + _skh)

    _b2 = b2.unsqueeze(-1)
    _c = c.unsqueeze(-1)
    _d = d.unsqueeze(-1)
    _t1 = 1 + _c * _q0 * (1 + 5 * _d * _q0)
    _t2 = safe_exp(-0.5 * _q0 / (1 + _q1).square() * _skh) / _q1
    vals = safe_exp(_q2) * torch.subtract(_t1, _t2)
    mask = _q2 > -100
    if not mask.all():
        vals[~mask] = 0
    quadrature_part = -0.5 * a * (vals @ w)

    # Return ``P(x > h, y > k)``
    int_from_0_to_a = _inv_2pi * s * (analytic_part + quadrature_part)
    return (int_from_0_to_s - int_from_0_to_a).clip(min=0, max=1)


def bvnmom(
    r: Tensor,
    xl: Tensor,
    yl: Tensor,
    xu: Tensor,
    yu: Tensor,
    p: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Computes the expected values of truncated, bivariate normal random variables.

    Let ``x`` and ``y`` be a pair of standard bivariate normal random variables having
    correlation ``r``. This function computes ``E([x,y] \| [xl,yl] < [x,y] < [xu,yu])``.

    Following [Muthen1990moments]_ equations (4) and (5), we have

        `E(x \| [xl, yl] < [x, y] < [xu, yu]) \
        = Z^{-1} \phi(xl) P(yl < y < yu \| x=xl) - \phi(xu) P(yl < y < yu \| x=xu),`

    where ``Z = P([xl, yl] < [x, y] < [xu, yu])`` and ``\phi`` is the standard
    normal PDF.

    Args:
        r: Tensor of correlation coefficients.
        xl: Tensor of lower bounds for ``x``, same shape as ``r``.
        xu: Tensor of upper bounds for ``x``, same shape as ``r``.
        yl: Tensor of lower bounds for ``y``, same shape as ``r``.
        yu: Tensor of upper bounds for ``y``, same shape as ``r``.
        p: Tensor of probabilities ``P(xl < x < xu, yl < y < yu)``, same shape as ``r``.

    Returns:
        ``E(x \| [xl, yl] < [x, y] < [xu, yu])`` and
        ``E(y \| [xl, yl] < [x, y] < [xu, yu])``.
    """
    if not (r.shape == xl.shape == xu.shape == yl.shape == yu.shape):
        raise UnsupportedError("Arguments to `bvn` must have the same shape.")

    if p is None:
        p = bvn(r=r, xl=xl, xu=xu, yl=yl, yu=yu)

    corr = r[..., None, None]
    istd = (1 - corr.square()).rsqrt()
    lower = torch.stack([xl, yl], -1)
    upper = torch.stack([xu, yu], -1)
    bounds = torch.stack([lower, upper], -1)
    deltas = safe_mul(corr, bounds)

    # Compute densities and conditional probabilities
    density_at_bounds = phi(bounds)
    prob_given_bounds = Phi(
        safe_mul(istd, safe_sub(upper.flip(-1).unsqueeze(-1), deltas))
    ) - Phi(safe_mul(istd, safe_sub(lower.flip(-1).unsqueeze(-1), deltas)))

    # Evaluate Muthen's formula
    p_diffs = -(density_at_bounds * prob_given_bounds).diff().squeeze(-1)
    moments = (1 / p).unsqueeze(-1) * (p_diffs + r.unsqueeze(-1) * p_diffs.flip(-1))
    return moments.unbind(-1)
