#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Posterior module to be used with GPyTorch models.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.posteriors.base_samples import _reshape_base_samples_non_interleaved
from botorch.posteriors.torch import TorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator import settings as linop_settings
from linear_operator.operators import (
    BlockDiagLinearOperator,
    DenseLinearOperator,
    LinearOperator,
    SumLinearOperator,
)
from torch import Tensor
from torch.distributions import Normal

if TYPE_CHECKING:
    from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover


class GPyTorchPosterior(TorchPosterior):
    r"""A posterior based on GPyTorch's multi-variate Normal distributions."""

    distribution: MultivariateNormal

    def __init__(self, distribution: MultivariateNormal) -> None:
        r"""A posterior based on GPyTorch's multi-variate Normal distributions.

        Args:
            distribution: A GPyTorch MultivariateNormal (single-output case) or
                MultitaskMultivariateNormal (multi-output case).
        """
        super().__init__(distribution=distribution)
        self._is_mt: bool = isinstance(distribution, MultitaskMultivariateNormal)

    @property
    def mvn(self) -> MultivariateNormal:
        r"""Expose the distribution as a backwards-compatible attribute."""
        return self.distribution

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples."""
        return self.distribution.batch_shape + self.distribution.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        ``base_sample_shape``. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        if self._is_mt:
            return (0, -2)
        else:
            return (0, -1)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given ``sample_shape``.
        """
        base_shape = self.distribution.batch_shape + self.distribution.event_shape
        if not self._is_mt:
            base_shape += torch.Size([1])
        return sample_shape + base_shape

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: A Tensor of ``N(0, I)`` base samples of shape
                ``sample_shape x base_sample_shape``, typically obtained from
                a ``Sampler``. This is used for deterministic optimization.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )
        if self._is_mt:
            base_samples = _reshape_base_samples_non_interleaved(
                mvn=self.distribution,
                base_samples=base_samples,
                sample_shape=sample_shape,
            )
        with ExitStack() as es:
            if linop_settings._fast_covar_root_decomposition.is_default():
                es.enter_context(linop_settings._fast_covar_root_decomposition(False))
            samples = self.distribution.rsample(
                sample_shape=sample_shape, base_samples=base_samples
            )
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        with ExitStack() as es:
            if linop_settings._fast_covar_root_decomposition.is_default():
                es.enter_context(linop_settings._fast_covar_root_decomposition(False))
            samples = self.distribution.rsample(sample_shape=sample_shape)
        # make sure there always is an output dimension
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        mean = self.distribution.mean
        if not self._is_mt:
            mean = mean.unsqueeze(-1)
        return mean

    @property
    def variance(self) -> Tensor:
        r"""The posterior variance."""
        variance = self.distribution.variance
        if not self._is_mt:
            variance = variance.unsqueeze(-1)
        return variance

    def quantile(self, value: Tensor) -> Tensor:
        r"""Compute the quantiles of the marginal distributions."""
        if value.numel() > 1:
            return torch.stack([self.quantile(v) for v in value], dim=0)
        marginal = Normal(loc=self.mean, scale=self.variance.sqrt())
        return marginal.icdf(value)

    def density(self, value: Tensor) -> Tensor:
        r"""The probability density of the marginal distributions."""
        if value.numel() > 1:
            return torch.stack([self.density(v) for v in value], dim=0)
        marginal = Normal(loc=self.mean, scale=self.variance.sqrt())
        return marginal.log_prob(value).exp()


def _validate_scalarize_inputs(weights: Tensor, m: int) -> None:
    if weights.ndim > 1:
        raise BotorchTensorDimensionError("`weights` must be one-dimensional")
    if m != weights.size(0):
        raise RuntimeError(
            f"Output shape not equal to that of weights. Output shape is {m} and "
            f"weights are {weights.shape}"
        )


def scalarize_posterior_gpytorch(
    posterior: GPyTorchPosterior,
    weights: Tensor,
    offset: float = 0.0,
) -> tuple[Tensor, Tensor | LinearOperator]:
    r"""Helper function for ``scalarize_posterior``, producing a mean and
    variance.

    This mean and variance are consumed by ``scalarize_posterior`` to produce
    a ``GPyTorchPosterior``.

    Args:
        posterior: The posterior over ``m`` outcomes to be scalarized.
            Supports ``t``-batching.
        weights: A tensor of weights of size ``m``.
        offset: The offset of the affine transformation.

    Returns:
        The transformed (single-output) posterior. If the input posterior has
            mean ``mu`` and covariance matrix ``Sigma``, this posterior has mean
            ``weights^T * mu`` and variance ``weights^T Sigma w``.

    Example:
        Example for a model with two outcomes:

        >>> X = torch.rand(1, 2)
        >>> posterior = model.posterior(X)
        >>> weights = torch.tensor([0.5, 0.25])
        >>> mean, cov = scalarize_posterior_gpytorch(posterior, weights=weights)
        >>> mvn = MultivariateNormal(mean, cov)
        >>> new_posterior = GPyTorchPosterior
    """
    mean = posterior.mean
    q, m = mean.shape[-2:]
    _validate_scalarize_inputs(weights=weights, m=m)
    batch_shape = mean.shape[:-2]
    mvn = posterior.distribution
    cov = mvn.lazy_covariance_matrix if mvn.islazy else mvn.covariance_matrix

    if m == 1:  # just scaling, no scalarization necessary
        new_mean = offset + (weights[0] * mean).view(*batch_shape, q)
        new_cov = weights[0] ** 2 * cov
        return new_mean, new_cov

    new_mean = offset + (mean @ weights).view(*batch_shape, q)

    if q == 1:
        new_cov = weights.unsqueeze(-2) @ (cov @ weights.unsqueeze(-1))
    else:
        # we need to handle potentially different representations of the multi-task mvn
        if mvn._interleaved:
            w_cov = weights.repeat(q).unsqueeze(0)
            sum_shape = batch_shape + torch.Size([q, m, q, m])
            sum_dims = (-1, -2)
        else:
            # special-case the independent setting
            if isinstance(cov, BlockDiagLinearOperator):
                new_cov = SumLinearOperator(
                    *[
                        cov.base_linear_op[..., i, :, :] * weights[i].pow(2)
                        for i in range(cov.base_linear_op.size(-3))
                    ]
                )
                return new_mean, new_cov

            w_cov = torch.repeat_interleave(weights, q).unsqueeze(0)
            sum_shape = batch_shape + torch.Size([m, q, m, q])
            sum_dims = (-2, -3)

        cov_scaled = w_cov * cov * w_cov.transpose(-1, -2)
        # TODO: Do not instantiate full covariance for LinearOperators
        # (ideally we simplify this in GPyTorch:
        # https://github.com/cornellius-gp/gpytorch/issues/1055)
        if isinstance(cov_scaled, LinearOperator):
            cov_scaled = cov_scaled.to_dense()
        new_cov = cov_scaled.view(sum_shape).sum(dim=sum_dims[0]).sum(dim=sum_dims[1])
        new_cov = DenseLinearOperator(new_cov)

    return new_mean, new_cov


def scalarize_posterior(
    posterior: GPyTorchPosterior | PosteriorList,
    weights: Tensor,
    offset: float = 0.0,
) -> GPyTorchPosterior:
    r"""Affine transformation of a multi-output posterior.

    Args:
        posterior: The posterior over ``m`` outcomes to be scalarized.
            Supports ``t``-batching. Can be either a ``GPyTorchPosterior``,
            or a ``PosteriorList`` that contains GPyTorchPosteriors all with q=1.
        weights: A tensor of weights of size ``m``.
        offset: The offset of the affine transformation.

    Returns:
        The transformed (single-output) posterior. If the input posterior has
            mean ``mu`` and covariance matrix ``Sigma``, this posterior has mean
            ``weights^T * mu`` and variance ``weights^T Sigma w``.

    Example:
        Example for a model with two outcomes:

        >>> X = torch.rand(1, 2)
        >>> posterior = model.posterior(X)
        >>> weights = torch.tensor([0.5, 0.25])
        >>> new_posterior = scalarize_posterior(posterior, weights=weights)
    """
    # GPyTorchPosterior case
    if hasattr(posterior, "distribution"):
        mean, cov = scalarize_posterior_gpytorch(posterior, weights, offset)
        mvn = MultivariateNormal(mean, cov)
        return GPyTorchPosterior(mvn)

    # PosteriorList case
    if not hasattr(posterior, "posteriors"):
        raise NotImplementedError(
            "scalarize_posterior only works with a posterior that has an attribute "
            "`distribution`, such as a GPyTorchPosterior, or a posterior that contains "
            "sub-posteriors in an attribute `posteriors`, as in a PosteriorList."
        )

    mean = posterior.mean
    q, m = mean.shape[-2:]

    _validate_scalarize_inputs(weights, m)
    batch_shape = mean.shape[:-2]

    if q != 1:
        raise NotImplementedError(
            "scalarize_posterior only works with a PosteriorList if each sub-posterior "
            "has q=1."
        )

    means = [post.mean for post in posterior.posteriors]
    if {mean.shape[-1] for mean in means} != {1}:
        raise NotImplementedError(
            "scalarize_posterior only works with a PosteriorList if each sub-posterior "
            "has one outcome."
        )

    new_mean = offset + (mean @ weights).view(*batch_shape, q)
    new_cov = (posterior.variance @ (weights**2))[:, None]
    mvn = MultivariateNormal(new_mean, new_cov)
    return GPyTorchPosterior(mvn)
