#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from botorch.exceptions import UnsupportedError
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors import Prior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class LinearTruncatedFidelityKernel(Kernel):
    r"""GPyTorch Linear Truncated Fidelity Kernel.

    Computes a covariance matrix based on the Linear truncated kernel between
    inputs ``x_1`` and ``x_2`` for up to two fidelity parmeters:

        K(x_1, x_2) = k_0 + c_1(x_1, x_2)k_1 + c_2(x_1,x_2)k_2 + c_3(x_1,x_2)k_3

    where

    - ``k_i(i=0,1,2,3)`` are Matern kernels calculated between non-fidelity
        parameters of ``x_1`` and ``x_2`` with different priors.
    - ``c_1=(1 - x_1[f_1])(1 - x_2[f_1]))(1 + x_1[f_1] x_2[f_1])^p`` is the kernel
        of the the bias term, which can be decomposed into a determistic part
        and a polynomial kernel. Here ``f_1`` is the first fidelity dimension and
        ``p`` is the order of the polynomial kernel.
    - ``c_3`` is the same as ``c_1`` but is calculated for the second fidelity
        dimension ``f_2``.
    - ``c_2`` is the interaction term with four deterministic terms and the
        polynomial kernel between ``x_1[..., [f_1, f_2]]`` and
        ``x_2[..., [f_1, f_2]]``.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = LinearTruncatedFidelityKernel()
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = LinearTruncatedFidelityKernel(batch_shape = torch.Size([2]))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 10 x 10)
    """

    def __init__(  # noqa C901
        self,
        fidelity_dims: list[int],
        dimension: int | None = None,
        power_prior: Prior | None = None,
        power_constraint: Interval | None = None,
        nu: float = 2.5,
        lengthscale_prior_unbiased: Prior | None = None,
        lengthscale_prior_biased: Prior | None = None,
        lengthscale_constraint_unbiased: Interval | None = None,
        lengthscale_constraint_biased: Interval | None = None,
        covar_module_unbiased: Kernel | None = None,
        covar_module_biased: Kernel | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            fidelity_dims: A list containing either one or two indices specifying
                the fidelity parameters of the input.
            dimension: The dimension of ``x``. Unused if ``active_dims`` is specified.
            power_prior: Prior for the power parameter of the polynomial kernel.
                Default is ``None``.
            power_constraint: Constraint on the power parameter of the polynomial
                kernel. Default is ``Positive``.
            nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2,
                or 5/2. Unused if both ``covar_module_unbiased`` and
                ``covar_module_biased`` are specified.
            lengthscale_prior_unbiased: Prior on the lengthscale parameter of Matern
                kernel ``k_0``. Default is ``Gamma(1.1, 1/20)``.
            lengthscale_constraint_unbiased: Constraint on the lengthscale parameter
                of the Matern kernel ``k_0``. Default is ``Positive``.
            lengthscale_prior_biased: Prior on the lengthscale parameter of Matern
                kernels ``k_i(i>0)``. Default is ``Gamma(5, 1/20)``.
            lengthscale_constraint_biased: Constraint on the lengthscale parameter
                of the Matern kernels ``k_i(i>0)``. Default is ``Positive``.
            covar_module_unbiased: Specify a custom kernel for ``k_0``. If omitted,
                use a ``MaternKernel``.
            covar_module_biased: Specify a custom kernel for the biased parts
                ``k_i(i>0)``. If omitted, use a ``MaternKernel``.
            batch_shape: If specified, use a separate lengthscale for each batch of
                input data. If ``x1`` is a ``batch_shape x n x d`` tensor, this should
                be ``batch_shape``.
            active_dims: Compute the covariance of a subset of input dimensions. The
                numbers correspond to the indices of the dimensions.
        """
        if dimension is None and kwargs.get("active_dims") is None:
            raise UnsupportedError(
                "Must specify dimension when not specifying active_dims."
            )
        n_fidelity = len(fidelity_dims)
        if len(set(fidelity_dims)) != n_fidelity:
            raise ValueError("fidelity_dims must not have repeated elements")
        if n_fidelity not in {1, 2}:
            raise UnsupportedError(
                "LinearTruncatedFidelityKernel accepts either one or two"
                "fidelity parameters."
            )
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of 0.5, 1.5, or 2.5")

        super().__init__(**kwargs)
        self.fidelity_dims = fidelity_dims
        if power_constraint is None:
            power_constraint = Positive()

        if lengthscale_prior_unbiased is None:
            lengthscale_prior_unbiased = GammaPrior(3, 6)

        if lengthscale_prior_biased is None:
            lengthscale_prior_biased = GammaPrior(6, 2)

        if lengthscale_constraint_unbiased is None:
            lengthscale_constraint_unbiased = Positive()

        if lengthscale_constraint_biased is None:
            lengthscale_constraint_biased = Positive()

        self.register_parameter(
            name="raw_power",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        self.register_constraint("raw_power", power_constraint)

        if power_prior is not None:
            self.register_prior(
                "power_prior",
                power_prior,
                lambda m: m.power,
                lambda m, v: m._set_power(v),
            )

        if self.active_dims is not None:
            dimension = len(self.active_dims)

        if covar_module_unbiased is None:
            covar_module_unbiased = MaternKernel(
                nu=nu,
                batch_shape=self.batch_shape,
                lengthscale_prior=lengthscale_prior_unbiased,
                ard_num_dims=dimension - n_fidelity,
                lengthscale_constraint=lengthscale_constraint_unbiased,
            )

        if covar_module_biased is None:
            covar_module_biased = MaternKernel(
                nu=nu,
                batch_shape=self.batch_shape,
                lengthscale_prior=lengthscale_prior_biased,
                ard_num_dims=dimension - n_fidelity,
                lengthscale_constraint=lengthscale_constraint_biased,
            )

        self.covar_module_unbiased = covar_module_unbiased
        self.covar_module_biased = covar_module_biased

    @property
    def power(self) -> Tensor:
        return self.raw_power_constraint.transform(self.raw_power)

    @power.setter
    def power(self, value: Tensor) -> None:
        self._set_power(value)

    def _set_power(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_power)
        self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if params.get("last_dim_is_batch", False):
            raise NotImplementedError(
                "last_dim_is_batch not yet supported by LinearTruncatedFidelityKernel"
            )

        power = self.power.view(*self.batch_shape, 1, 1)
        active_dimsM = torch.tensor(
            [i for i in range(x1.size(-1)) if i not in self.fidelity_dims],
            device=x1.device,
        )
        if len(active_dimsM) == 0:
            raise RuntimeError(
                "Input to LinearTruncatedFidelityKernel must have at least one "
                "non-fidelity dimension."
            )
        x1_ = x1.index_select(dim=-1, index=active_dimsM)
        x2_ = x2.index_select(dim=-1, index=active_dimsM)
        covar_unbiased = self.covar_module_unbiased(x1_, x2_, diag=diag)
        covar_biased = self.covar_module_biased(x1_, x2_, diag=diag)

        # clamp to avoid numerical issues
        fd_idxr0 = torch.full(
            (1,), self.fidelity_dims[0], dtype=torch.long, device=x1.device
        )
        x11_ = x1.index_select(dim=-1, index=fd_idxr0).clamp(0, 1)
        x21t_ = x2.index_select(dim=-1, index=fd_idxr0).clamp(0, 1)
        if not diag:
            x21t_ = x21t_.transpose(-1, -2)
        cross_term_1 = (1 - x11_) * (1 - x21t_)
        bias_factor = cross_term_1 * (1 + x11_ * x21t_).pow(power)

        if len(self.fidelity_dims) > 1:
            # clamp to avoid numerical issues
            fd_idxr1 = torch.full(
                (1,), self.fidelity_dims[1], dtype=torch.long, device=x1.device
            )
            x12_ = x1.index_select(dim=-1, index=fd_idxr1).clamp(0, 1)
            x22t_ = x2.index_select(dim=-1, index=fd_idxr1).clamp(0, 1)
            x1b_ = torch.cat([x11_, x12_], dim=-1)
            if diag:
                x2bt_ = torch.cat([x21t_, x22t_], dim=-1)
                k = (1 + (x1b_ * x2bt_).sum(dim=-1, keepdim=True)).pow(power)
            else:
                x22t_ = x22t_.transpose(-1, -2)
                x2bt_ = torch.cat([x21t_, x22t_], dim=-2)
                k = (1 + x1b_ @ x2bt_).pow(power)

            cross_term_2 = (1 - x12_) * (1 - x22t_)
            bias_factor += cross_term_2 * (1 + x12_ * x22t_).pow(power)
            bias_factor += cross_term_2 * cross_term_1 * k

        if diag:
            bias_factor = bias_factor.view(covar_biased.shape)

        return covar_unbiased + bias_factor * covar_biased
