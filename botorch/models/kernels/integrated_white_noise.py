#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Module for the integrated white noise covariance kernel.

This module provides a single kernel parameterized by the ``order`` of
integration, corresponding to repeatedly integrated white noise (equivalently,
integrated Brownian motion). For inputs s, t >= 0 the covariance of white noise
integrated ``order`` times has the closed form

.. math::

    k_p(s, t) = \frac{1}{((p-1)!)^2}
        \sum_{j=0}^{p-1} \binom{p-1}{j}
        \frac{(\max - \min)^{p-1-j} \, \min^{p+j}}{p + j}

where :math:`p` is the order, :math:`\min = \min(s, t)` and
:math:`\max = \max(s, t)`. The first three orders recover the familiar forms:

- ``order=1`` (Wiener process / Brownian motion):
    k(s, t) = min(s, t)
- ``order=2`` (integrated Brownian motion):
    k(s, t) = min(s, t)^2 * (3 * max(s, t) - min(s, t)) / 6
- ``order=3`` (twice integrated Brownian motion):
    k(s, t) = min(s, t)^3 * (min^2 - 5 * min * max + 10 * max^2) / 120

All inputs are assumed non-negative (representing time indices).
"""

from __future__ import annotations

import math

import torch
from gpytorch.kernels import Kernel
from torch import Tensor


class IntegratedWhiteNoiseKernel(Kernel):
    r"""
    Computes a covariance matrix based on the integrated white noise kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

    This is the covariance of white noise integrated ``order`` (:math:`p`)
    times, equivalently :math:`(p-1)`-times integrated Brownian motion. For
    :math:`s, t \ge 0`,

    .. math::

        k_p(s, t) = \frac{1}{((p-1)!)^2}
            \sum_{j=0}^{p-1} \binom{p-1}{j}
            \frac{(\max - \min)^{p-1-j} \, \min^{p+j}}{p + j}

    where :math:`\min = \min(s, t)` and :math:`\max = \max(s, t)`. The first
    three orders recover the familiar closed forms:

    - :math:`p = 1`: :math:`k(s, t) = \min(s, t)` (Wiener process / Brownian
      motion).
    - :math:`p = 2`: :math:`k(s, t) = \min^2 (3 \max - \min) / 6` (integrated
      Brownian motion, :math:`X(t) = \int_0^t B(u) \, du`).
    - :math:`p = 3`:
      :math:`k(s, t) = \min^3 (\min^2 - 5 \min \max + 10 \max^2) / 120`
      (twice integrated Brownian motion).

    .. note::

        This kernel assumes non-negative inputs (representing time indices).
        For inputs with negative values, the behavior may not correspond to
        integrated white noise.

    .. note::

        This kernel does not have a `lengthscale` parameter. To add a scaling
        parameter, decorate this kernel with a
        :class:`gpytorch.kernels.ScaleKernel`.

    Example:
        >>> x = torch.rand(10, 1)  # 10 time points in [0, 1]
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(
        ...     IntegratedWhiteNoiseKernel(order=2)
        ... )
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> batch_x = torch.rand(2, 10, 1)  # Batch of time points
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(
        ...     IntegratedWhiteNoiseKernel(order=2)
        ... )
        >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, order: int = 1, **kwargs) -> None:
        r"""Initialize the kernel.

        Args:
            order: The order of integration :math:`p \ge 1`. ``order=1`` is
                Brownian motion (Wiener process), ``order=2`` is integrated
                Brownian motion, and so on.
            kwargs: Keyword arguments passed to
                :class:`gpytorch.kernels.Kernel` (e.g. ``batch_shape``,
                ``active_dims``).
        """
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(
                f"IntegratedWhiteNoiseKernel requires an integer order >= 1, but "
                f"got order={order!r}."
            )
        super().__init__(**kwargs)
        self.order = order
        # Precompute the constant denominator ((p-1)!)^2 and the per-term
        # (binomial coefficient, exponents, divisor) tuples for j = 0..p-1.
        # Dividing by the denominator is numerically more stable than
        # multiplying by its floating-point reciprocal.
        self._denominator: int = math.factorial(order - 1) ** 2
        self._terms: list[tuple[int, int, int, int]] = [
            (math.comb(order - 1, j), order - 1 - j, order + j, order + j)
            for j in range(order)
        ]

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        r"""
        Compute the integrated white noise covariance between x1 and x2.

        Args:
            x1: First set of data, shape `(... x n x 1)`.
            x2: Second set of data, shape `(... x m x 1)`.
            diag: If True, return only the diagonal of the covariance matrix.

        Returns:
            Tensor: The covariance matrix of shape `(... x n x m)` or diagonal
                of shape `(... x n)` if diag=True.
        """
        # Validate that inputs are 1D (time indices)
        if x1.shape[-1] != 1 or x2.shape[-1] != 1:
            raise ValueError(
                f"IntegratedWhiteNoiseKernel requires 1D inputs (last dimension "
                f"must be 1), but got x1.shape[-1]={x1.shape[-1]} and "
                f"x2.shape[-1]={x2.shape[-1]}. The integrated white noise "
                f"covariance is only defined for scalar time indices."
            )

        # Squeeze the last dimension to get scalar time values
        x1_squeezed = x1.squeeze(-1)
        x2_squeezed = x2.squeeze(-1)

        if diag:
            # For diagonal, x1 and x2 should be equal, so min = max = x. Only the
            # j = p-1 term survives: k(t, t) = t^(2p-1) / ((p-1)!^2 * (2p-1)).
            return x1_squeezed ** (2 * self.order - 1) / (
                self._denominator * (2 * self.order - 1)
            )

        # Compute pairwise min and max with shapes broadcasting to (..., n, m).
        x1_expanded = x1_squeezed.unsqueeze(-1)
        x2_expanded = x2_squeezed.unsqueeze(-2)
        min_val = torch.minimum(x1_expanded, x2_expanded)
        max_val = torch.maximum(x1_expanded, x2_expanded)
        diff = max_val - min_val

        result = torch.zeros_like(min_val)
        for coeff, diff_exp, min_exp, divisor in self._terms:
            result = result + coeff * diff**diff_exp * min_val**min_exp / divisor
        return result / self._denominator
