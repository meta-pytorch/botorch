#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor


class ExponentialDecayKernel(Kernel):
    r"""GPyTorch Exponential Decay Kernel.

    Computes a covariance matrix based on the exponential decay kernel
    between inputs ``x_1`` and ``x_2`` (we expect ``d = 1``):

        K(x_1, x_2) = w + beta^alpha / (x_1 + x_2 + beta)^alpha.

    where ``w`` is an offset parameter, ``beta`` is a lenthscale parameter, and
    ``alpha`` is a power parameter.
    """

    has_lengthscale = True

    def __init__(
        self,
        power_prior: Prior | None = None,
        offset_prior: Prior | None = None,
        power_constraint: Interval | None = None,
        offset_constraint: Interval | None = None,
        **kwargs,
    ):
        r"""
        Args:
            lengthscale_constraint: Constraint to place on lengthscale parameter.
                Default is ``Positive``.
            lengthscale_prior: Prior over the lengthscale parameter.
            power_constraint: Constraint to place on power parameter. Default is
                ``Positive``.
            power_prior: Prior over the power parameter.
            offset_constraint: Constraint to place on offset parameter. Default is
                ``Positive``.
            active_dims: List of data dimensions to operate on. ``len(active_dims)``
                should equal ``num_dimensions``.
        """
        super().__init__(**kwargs)

        if power_constraint is None:
            power_constraint = Positive()
        if offset_constraint is None:
            offset_constraint = Positive()

        self.register_parameter(
            name="raw_power",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        self.register_parameter(
            name="raw_offset",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        if power_prior is not None:
            self.register_prior(
                "power_prior",
                power_prior,
                lambda m: m.power,
                lambda m, v: m._set_power(v),
            )
        self.register_constraint("raw_power", offset_constraint)

        if offset_prior is not None:
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda m: m.offset,
                lambda m, v: m._set_offset(v),
            )

        self.register_constraint("raw_offset", offset_constraint)

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

    @property
    def offset(self) -> Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        offset = self.offset
        power = self.power
        if not params.get("diag", False):
            offset = offset.unsqueeze(-1)  # unsqueeze enables batch evaluation
            power = power.unsqueeze(-1)  # unsqueeze enables batch evaluation
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        diff = self.covar_dist(x1_, -x2_, **params)
        res = offset + (diff + 1).pow(-power)
        return res
