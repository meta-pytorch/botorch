#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Special implementations of mathematical functions that
solve numerical issues of naive implementations.

.. [Maechler2012accurate]
    M. Mächler. Accurately Computing log (1 - exp (-| a|))
        Assessed by the Rmpfr package. Technical report, 2012.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from botorch.exceptions import UnsupportedError
from botorch.utils.constants import get_constants_like
from torch import finfo, Tensor
from torch.nn.functional import softplus

_log2 = math.log(2)
_inv_sqrt_3 = math.sqrt(1 / 3)

TAU = 1.0  # default temperature parameter for smooth approximations to non-linearities
ALPHA = 2.0  # default alpha parameter for the asymptotic power decay of _pareto


# Unary ops
def exp(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    maxexp = get_constants_like(math.log(info.max) - 1e-4, x)
    return torch.exp(x.clip(max=maxexp), **kwargs)


def log(x: Tensor, **kwargs) -> Tensor:
    info = finfo(x.dtype)
    return torch.log(x.clip(min=info.tiny), **kwargs)


# Binary ops
def add(a: Tensor, b: Tensor, **kwargs) -> Tensor:
    _0 = get_constants_like(0, a)
    case = a.isinf() & b.isinf() & (a != b)
    return torch.where(case, _0, a + b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(0, a)
    case = (a.isinf() & b.isinf()) & (a == b)
    return torch.where(case, _0, a - b)


def div(a: Tensor, b: Tensor) -> Tensor:
    _0, _1 = get_constants_like(values=(0, 1), ref=a)
    case = ((a == _0) & (b == _0)) | (a.isinf() & b.isinf())
    return torch.where(case, torch.where(a != b, -_1, _1), a / torch.where(case, _1, b))


def mul(a: Tensor, b: Tensor) -> Tensor:
    _0 = get_constants_like(values=0, ref=a)
    case = (a.isinf() & (b == _0)) | (b.isinf() & (a == _0))
    return torch.where(case, _0, a * torch.where(case, _0, b))


def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    log2 = get_constants_like(values=_log2, ref=x)
    is_small = -log2 < x  # x < 0
    return torch.where(
        is_small,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


def log1pexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 + exp(x)).
    See [Maechler2012accurate]_ for details.
    """
    mask = x <= 18
    return torch.where(
        mask,
        (lambda z: z.exp().log1p())(x.masked_fill(~mask, 0)),
        (lambda z: z + (-z).exp())(x.masked_fill(mask, 0)),
    )


def logexpit(X: Tensor) -> Tensor:
    """Computes the logarithm of the expit (a.k.a. sigmoid) function."""
    return -log1pexp(-X)


def logplusexp(a: Tensor, b: Tensor) -> Tensor:
    """Computes log(exp(a) + exp(b)) similar to logsumexp."""
    ab = torch.stack(torch.broadcast_tensors(a, b), dim=-1)
    return logsumexp(ab, dim=-1)


def logdiffexp(log_a: Tensor, log_b: Tensor) -> Tensor:
    """Computes log(b - a) accurately given log(a) and log(b).
    Assumes, log_b > log_a, i.e. b > a > 0.

    Args:
        log_a (Tensor): The logarithm of a, assumed to be less than log_b.
        log_b (Tensor): The logarithm of b, assumed to be larger than log_a.

    Returns:
        A Tensor of values corresponding to log(b - a).
    """
    log_a, log_b = torch.broadcast_tensors(log_a, log_b)
    is_inf = log_b == -torch.inf  # implies log_a == -torch.inf by assumption
    return log_b + log1mexp(log_a - log_b.masked_fill(is_inf, 0.0))


def logsumexp(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    """Version of logsumexp that has a well-behaved backward pass when
    x contains infinities.

    In particular, the gradient of the standard torch version becomes NaN
    1) for any element that is positive infinity, and 2) for any slice that
    only contains negative infinities.

    This version returns a gradient of 1 for any positive infinities in case 1, and
    for all elements of the slice in case 2, in agreement with the asymptotic behavior
    of the function.

    Args:
        x: The Tensor to which to apply ``logsumexp``.
        dim: An integer or a tuple of integers, representing the dimensions to reduce.
        keepdim: Whether to keep the reduced dimensions. Defaults to False.

    Returns:
        A Tensor representing the log of the summed exponentials of ``x``.
    """
    return _inf_max_helper(torch.logsumexp, x=x, dim=dim, keepdim=keepdim)


def _inf_max_helper(
    max_fun: Callable[[Tensor], Tensor],
    x: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool,
) -> Tensor:
    """Helper function that generalizes the treatment of infinities for approximations
    to the maximum operator, i.e., ``max(X, dim, keepdim)``. At the point of writing of
    this function, it is used to define ``logsumexp`` and ``fatmax``.

    Args:
        max_fun: The function that is used to smoothly penalize the difference of an
            element to the true maximum.
        x: The Tensor on which to compute the smooth approximation to the maximum.
        dim: The dimension(s) to reduce over.
        keepdim: Whether to keep the reduced dimension. Defaults to False.

    Returns:
        The Tensor representing the smooth approximation to the maximum over the
        specified dimensions.
    """
    M = x.amax(dim=dim, keepdim=True)
    is_inf_max = torch.logical_and(*torch.broadcast_tensors(M.isinf(), x == M))
    has_inf_max = _any(is_inf_max, dim=dim, keepdim=True)

    y_inf = x.masked_fill(~is_inf_max, 0.0)
    M_no_inf = M.masked_fill(M.isinf(), 0.0)
    y_no_inf = x.masked_fill(has_inf_max, 0.0) - M_no_inf

    res = torch.where(
        has_inf_max,
        y_inf.sum(dim=dim, keepdim=True),
        M_no_inf + max_fun(y_no_inf, dim=dim, keepdim=True),
    )
    # NOTE: Using ``sum`` instead of ``squeeze`` because PyTorch < 2.0 does not support
    # tuple ``dim`` arguments. ``sum`` and ``squeeze`` are equivalent here because the
    # ``dim`` dimensions have length one after the reductions in the previous lines.
    # TODO: Replace ``sum`` with ``squeeze`` once PyTorch >= 2.0 is required.
    return res if keepdim else res.sum(dim=dim)


def _any(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    """Extension of torch.any, which supports reducing over tuples of dimensions.

    Args:
        x: The Tensor to reduce over.
        dim: An integer or a tuple of integers, representing the dimensions to reduce.
        keepdim: Whether to keep the reduced dimensions. Defaults to False.

    Returns:
        The Tensor corresponding to ``any`` over the specified dimensions.
    """
    if isinstance(dim, tuple):
        for d in dim:
            x = x.any(dim=d, keepdim=True)
    else:
        x = x.any(dim, keepdim=True)
    return x if keepdim else x.squeeze(dim)


def logmeanexp(X: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    """Computes ``log(mean(exp(X), dim=dim, keepdim=keepdim))``.

    Args:
        X: Values of which to compute the logmeanexp.
        dim: The dimension(s) over which to compute the mean.
        keepdim: If True, keeps the reduced dimensions.

    Returns:
        A Tensor of values corresponding to ``log(mean(exp(X), dim=dim))``.
    """
    n = X.shape[dim] if isinstance(dim, int) else math.prod(X.shape[i] for i in dim)
    return logsumexp(X, dim=dim, keepdim=keepdim) - math.log(n)


def log_softplus(x: Tensor, tau: float | Tensor = TAU) -> Tensor:
    """Computes the logarithm of the softplus function with high numerical accuracy.

    Args:
        x: Input tensor, should have single or double precision floats.
        tau: Decreasing tau increases the tightness of the
            approximation to ReLU. Non-negative and defaults to 1.0.

    Returns:
        Tensor corresponding to ``log(softplus(x))``.
    """
    check_dtype_float32_or_float64(x)
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    # cutoff chosen to achieve accuracy to machine epsilon
    upper = 16 if x.dtype == torch.float32 else 32
    lower = -15 if x.dtype == torch.float32 else -35
    mask = x / tau > lower
    return torch.where(
        mask,
        softplus(x.masked_fill(~mask, lower), beta=(1 / tau), threshold=upper).log(),
        x / tau + tau.log(),
    )


def smooth_amax(
    X: Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    tau: float | Tensor = 1.0,
) -> Tensor:
    """Computes a smooth approximation to ``max(X, dim=dim)``, i.e the maximum
    value of ``X`` over dimension ``dim``, using the logarithm of the
    ``l_(1/tau)`` norm of ``exp(X)``. Note that when ``X = log(U)`` is the
    *logarithm* of an acquisition utility ``U``,

    ``logsumexp(log(U) / tau) * tau = log(sum(U^(1/tau))^tau)``
    ``= log(norm(U, ord=(1/tau))``

    Args:
        X: A Tensor from which to compute the smoothed amax.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be
            positive.

    Returns:
        A Tensor of smooth approximations to ``max(X, dim=dim)``.
    """
    # consider normalizing by log_n = math.log(X.shape[dim]) to reduce error
    return logsumexp(X / tau, dim=dim, keepdim=keepdim) * tau  # ~ X.amax(dim=dim)


def smooth_amin(
    X: Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    tau: float | Tensor = 1.0,
) -> Tensor:
    """A smooth approximation to ``min(X, dim=dim)``, similar to ``smooth_amax``."""
    return -smooth_amax(X=-X, dim=dim, keepdim=keepdim, tau=tau)


def check_dtype_float32_or_float64(X: Tensor) -> None:
    if X.dtype != torch.float32 and X.dtype != torch.float64:
        raise UnsupportedError(
            f"Only dtypes float32 and float64 are supported, but received {X.dtype}."
        )


def log_fatplus(x: Tensor, tau: float | Tensor = TAU) -> Tensor:
    """Computes the logarithm of the fat-tailed softplus.

    NOTE: Separated out in case the complexity of the ``log`` implementation increases
    in the future.
    """
    return fatplus(x, tau=tau).log()


def fatplus(x: Tensor, tau: float | Tensor = TAU) -> Tensor:
    """Computes a fat-tailed approximation to ``ReLU(x) = max(x, 0)`` by linearly
    combining a regular softplus function and the density function of a Cauchy
    distribution. The coefficient ``alpha`` of the Cauchy density is chosen to guarantee
    monotonicity and convexity.

    Args:
        x: A Tensor on whose values to compute the smoothed function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A Tensor of values of the fat-tailed softplus.
    """

    def _fatplus(x: Tensor) -> Tensor:
        alpha = 1e-1  # guarantees monotonicity and convexity (TODO: ref + Lemma 4)
        return softplus(x) + alpha * cauchy(x)

    return tau * _fatplus(x / tau)


def fatmax(
    x: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    tau: float | Tensor = TAU,
    alpha: float = ALPHA,
) -> Tensor:
    """Computes a smooth approximation to amax(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed maximum.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be positive.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A Tensor of smooth approximations to ``amax(X, dim=dim)`` with a fat tail.
    """

    def max_fun(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
        return tau * _pareto(-x / tau, alpha=alpha).sum(dim=dim, keepdim=keepdim).log()

    return _inf_max_helper(max_fun=max_fun, x=x, dim=dim, keepdim=keepdim)


def fatmin(
    x: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    tau: float | Tensor = TAU,
    alpha: float = ALPHA,
) -> Tensor:
    """Computes a smooth approximation to amin(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed minimum.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to min operator, becomes tighter as tau goes to 0. Needs to be positive.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A Tensor of smooth approximations to ``amin(X, dim=dim)`` with a fat tail.
    """
    return -fatmax(-x, dim=dim, keepdim=keepdim, tau=tau, alpha=alpha)


def fatmaximum(
    a: Tensor, b: Tensor, tau: float | Tensor = TAU, alpha: float = ALPHA
) -> Tensor:
    """Computes a smooth approximation to torch.maximum(a, b) with a fat tail.

    Args:
        a: The first Tensor from which to compute the smoothed component-wise maximum.
        b: The second Tensor from which to compute the smoothed component-wise maximum.
        tau: Temperature parameter controlling the smoothness of the approximation. A
            smaller tau corresponds to a tighter approximation that leads to a sharper
            objective landscape that might be more difficult to optimize.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A smooth approximation of torch.maximum(a, b).
    """
    return fatmax(
        torch.stack(torch.broadcast_tensors(a, b), dim=-1),
        dim=-1,
        keepdim=False,
        tau=tau,
        alpha=alpha,
    )


def fatminimum(
    a: Tensor, b: Tensor, tau: float | Tensor = TAU, alpha: float = ALPHA
) -> Tensor:
    """Computes a smooth approximation to torch.minimum(a, b) with a fat tail.

    Args:
        a: The first Tensor from which to compute the smoothed component-wise minimum.
        b: The second Tensor from which to compute the smoothed component-wise minimum.
        tau: Temperature parameter controlling the smoothness of the approximation. A
            smaller tau corresponds to a tighter approximation that leads to a sharper
            objective landscape that might be more difficult to optimize.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A smooth approximation of torch.minimum(a, b).
    """
    return -fatmaximum(-a, -b, tau=tau, alpha=alpha)


def log_fatmoid(X: Tensor, tau: float | Tensor = 1.0) -> Tensor:
    """Computes the logarithm of the fatmoid. Separated out in case the implementation
    of the logarithm becomes more complex in the future to ensure numerical stability.
    """
    return fatmoid(X, tau=tau).log()


def fatmoid(X: Tensor, tau: float | Tensor = 1.0) -> Tensor:
    """Computes a twice continuously differentiable approximation to the Heaviside
    step function with a fat tail, i.e. ``O(1 / x^2)`` as ``x`` goes to -inf.

    Args:
        X: A Tensor from which to compute the smoothed step function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A tensor of fat-tailed approximations to the Heaviside step function.
    """
    X = X / tau
    m = _inv_sqrt_3  # this defines the inflection point
    return torch.where(
        X < 0,
        2 / 3 * cauchy(X - m),
        1 - 2 / 3 * cauchy(X + m),
    )


def cauchy(x: Tensor) -> Tensor:
    """Computes a Lorentzian, i.e. an un-normalized Cauchy density function."""
    return 1 / (1 + x.square())


def _pareto(x: Tensor, alpha: float, check: bool = True) -> Tensor:
    """Computes a rational polynomial that is
    1) monotonically decreasing for ``x > 0``,
    2) is equal to 1 at ``x = 0``,
    3) has a first and second derivative of 1 at ``x = 0``, and
    4) has an asymptotic decay of ``O(1 / x^alpha)``.
    These properties make it possible to use the function to define a smooth and
    fat-tailed approximation to the maximum, which enables better gradient propagation,
    see ``fatmax`` for details.

    Args:
        x: The input tensor.
        alpha: The exponent of the asymptotic decay.
        check: Whether to check if the input tensor only contains non-negative values.

    Returns:
        The tensor corresponding to the rational polynomial with the stated properties.
    """
    if check and (x < 0).any():
        raise ValueError("Argument `x` must be non-negative.")
    alpha = alpha / 2  # so that alpha stands for the power decay
    # choosing beta_0, beta_1 so that first and second derivatives at x = 0 are 1.
    beta_1 = 2 * alpha
    beta_0 = alpha * beta_1
    return (beta_0 / (beta_0 + beta_1 * x + x.square())).pow(alpha)


def sigmoid(X: Tensor, log: bool = False, fat: bool = False) -> Tensor:
    """A sigmoid function with an optional fat tail and evaluation in log space for
    better numerical behavior. Notably, the fat-tailed sigmoid can be used to remedy
    numerical underflow problems in the value and gradient of the canonical sigmoid.

    Args:
        X: The Tensor on which to evaluate the sigmoid.
        log: Toggles the evaluation of the log sigmoid.
        fat: Toggles the evaluation of the fat-tailed sigmoid.

    Returns:
        A Tensor of (log-)sigmoid values.
    """
    Y = log_fatmoid(X) if fat else logexpit(X)
    return Y if log else Y.exp()
