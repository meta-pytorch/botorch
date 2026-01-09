#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.probability.lin_ess import LinearEllipticalSliceSampler
from botorch.utils.probability.mvnxpb import MVNXPB
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal


class TruncatedMultivariateNormal(MultivariateNormal):
    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        bounds: Tensor = None,
        solver: MVNXPB | None = None,
        sampler: LinearEllipticalSliceSampler | None = None,
        validate_args: bool | None = None,
    ):
        r"""Initializes an instance of a TruncatedMultivariateNormal distribution.

        Let `x ~ N(0, K)` be an `n`-dimensional Gaussian random vector. This class
        represents the distribution of the truncated Multivariate normal random vector
        `x | a <= x <= b`.

        Args:
            loc: A mean vector for the distribution, `batch_shape x event_shape`.
            covariance_matrix: Covariance matrix distribution parameter.
            precision_matrix: Inverse covariance matrix distribution parameter.
            scale_tril: Lower triangular, square-root covariance matrix distribution
                parameter.
            bounds: A `batch_shape x event_shape x 2` tensor of strictly increasing
                bounds for `x` so that `bounds[..., 0] < bounds[..., 1]` everywhere.
            solver: A pre-solved MVNXPB instance used to approximate the log partition.
            sampler: A LinearEllipticalSliceSampler instance used for sample generation.
            validate_args: Optional argument to super().__init__.
        """
        if bounds is None:
            raise SyntaxError("Missing required argument `bounds`.")
        elif bounds.shape[-1] != 2:
            raise ValueError(
                f"Expected bounds.shape[-1] to be 2 but bounds shape is {bounds.shape}"
            )
        elif torch.gt(*bounds.unbind(dim=-1)).any():
            raise ValueError("`bounds` must be strictly increasing along dim=-1.")

        super().__init__(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            validate_args=validate_args,
        )
        self.bounds = bounds
        self._solver = solver
        self._sampler = sampler

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Approximates the true log probability."""
        neg_inf = float("-inf")
        inbounds = torch.logical_and(
            (self.bounds[..., 0] < value).all(-1),
            (self.bounds[..., 1] > value).all(-1),
        )
        if inbounds.any():
            return torch.where(
                inbounds,
                super().log_prob(value) - self.log_partition,
                neg_inf,
            )
        return torch.full(value.shape[: -len(self.event_shape)], neg_inf)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:  # noqa: B008
        r"""Draw samples from the Truncated Multivariate Normal.

        Args:
            sample_shape: The shape of the samples.

        Returns:
            The (sample_shape x batch_shape x event_shape) tensor of samples.
        """
        num_samples = sample_shape.numel() if sample_shape else 1
        return self.loc + self.sampler.draw(n=num_samples).view(*sample_shape, -1)

    @property
    def log_partition(self) -> Tensor:
        return self.solver.log_prob

    @property
    def solver(self) -> MVNXPB:
        if self._solver is None:
            self._solver = MVNXPB(
                covariance_matrix=self.covariance_matrix,
                bounds=self.bounds - self.loc.unsqueeze(-1),
            )
            self._solver.solve()
        return self._solver

    @property
    def sampler(self) -> LinearEllipticalSliceSampler:
        if self._sampler is None:
            eye = torch.eye(
                self.scale_tril.shape[-1],
                dtype=self.scale_tril.dtype,
                device=self.scale_tril.device,
            )

            A = torch.concat([-eye, eye])
            b = torch.concat(
                [
                    self.loc - self.bounds[..., 0],
                    self.bounds[..., 1] - self.loc,
                ],
                dim=-1,
            ).unsqueeze(-1)

            self._sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                covariance_root=self.scale_tril,
            )
        return self._sampler

    def expand(
        self, batch_shape: Sequence[int], _instance: TruncatedMultivariateNormal = None
    ) -> TruncatedMultivariateNormal:
        new = self._get_checked_instance(TruncatedMultivariateNormal, _instance)
        super().expand(batch_shape=batch_shape, _instance=new)

        new.bounds = self.bounds.expand(*new.batch_shape, *self.event_shape, 2)
        new._sampler = None  # does not implement `expand`
        new._solver = (
            None if self._solver is None else self._solver.expand(*batch_shape)
        )
        return new

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f", bounds: {self.bounds.shape})"
