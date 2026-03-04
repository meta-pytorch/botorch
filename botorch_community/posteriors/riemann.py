#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract base module for all botorch posteriors.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.posteriors.posterior import Posterior
from botorch.sampling.get_sampler import _get_sampler_mvn, GetSampler
from botorch.sampling.normal import NormalMCSampler
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor


class BoundedRiemannPosterior(Posterior):
    batch_range = (0, -1)

    """
    A single variate bounded Riemann posterior.
    """

    def __init__(self, borders, probabilities):
        r"""Bounded Riemann Posterior.

        A posterior distribution represented by a piecewise constant probability density
        function over a bounded domain. The domain is divided into buckets defined by
        borders, with each bucket having an associated probability.

        Args:
            borders: A tensor of shape ``(num_buckets + 1,)`` defining the boundaries of
                the buckets. Must be monotonically increasing.
            probabilities: A tensor of shape ``(b?, q?, num_buckets)`` defining the
                probability mass in each bucket. Must sum to 1 in the last dim.
        """

        assert torch.allclose(
            probabilities.sum(-1),
            torch.tensor(1.0, device=probabilities.device, dtype=probabilities.dtype),
            atol=1e-3,
        ), f"Probabilities must sum to 1, but sum to {probabilities.sum()}."

        self.borders = borders
        self.probabilities = probabilities
        self.cumprobs = torch.cumsum(self.probabilities, -1)

    def integrate(self, ag_integrate_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        r"""Integrate over the posterior, by calculating
            $$\int_{min}^{max} ag(y) posterior(y) dy$$.

        Args:
            ag_integral_fn: A vectorized function that integrates ag from lower_bound
                to upper_bound, that is, $$\int_{lower_bound}^{upper_bound} ag(y) dy$$.

        Returns:
            Tensor: The integral of the posterior.
        """
        all_lower = self.borders[:-1]
        all_upper = self.borders[1:]
        bucket_results = ag_integrate_fn(all_lower, all_upper)
        return (bucket_results * (self.probabilities / (all_upper - all_lower))).sum(-1)

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        sample_shape = sample_shape if sample_shape is not None else torch.Size([1])
        base_samples = torch.randn(
            sample_shape + self.probabilities.shape[:-1],
            device=self.probabilities.device,
        )
        return self.rsample_from_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """
        base_samples are N(0, I) samples, as this posterior is registered
        with the IIDNormalSampler below. Alternatively it could be registered
        with a uniform sampler in which case the transformation to uniform RVs
        could be avoided. Shape of base_samples is (nsamp, b?, q).
        """
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise ValueError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )
        # convert base samples from N(O, I) to Uniform.
        U = torch.distributions.Normal(0, 1).cdf(base_samples)
        # Convert U to Riemann samples.
        Z = self.icdf(U)  # (nsamp, b?, q, 1)
        return Z

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of the base samples required to draw from the posterior."""
        return self.probabilities.shape[:-1]

    @property
    def device(self) -> torch.device:
        r"""The torch device of the distribution."""
        return self.borders.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self.borders.dtype

    @property
    def mean(self):
        r"""The mean of the posterior distribution."""
        bucket_widths = self.borders[1:] - self.borders[:-1]
        bucket_means = self.borders[:-1] + bucket_widths / 2
        return (self.probabilities @ bucket_means).unsqueeze(-1)

    @property
    def mean_of_square(self) -> torch.Tensor:
        """Computes E[x^2]."""
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        return (self.probabilities @ bucket_mean_of_square).unsqueeze(-1)

    @property
    def variance(self) -> torch.Tensor:
        """Computes the variance via Var[x] = E[x^2] - E[x]^2."""
        return self.mean_of_square - self.mean.square()

    def confidence_region(
        self, confidence_level: float = 0.95
    ) -> tuple[Tensor, Tensor]:
        r"""
        Compute the lower and upper bounds of the confidence region.
        Args:
            confidence_level: The probability between the bounds
                of the confidence interval/region.
                Use .954 for 2 sigma of a normal distribution.
        """
        side_probs = (1.0 - confidence_level) / 2
        lower = self.icdf(side_probs).squeeze()
        upper = self.icdf(1.0 - side_probs).squeeze()
        return lower, upper

    def icdf(
        self,
        value: float | Tensor,
    ) -> Tensor:
        r"""Inverse cdf (with gradients).
        Use value to get the index of the bucket that contains the value
        and then interpolate between the left and right borders of the bucket

        Args:
            value: The value at which to evaluate the inverse CDF.
                Either a float, or a tensor with shape is (b', b?, q), where
                probabilities has shape (b?, q, num_buckets).

        Returns:
            The inverse CDF of the posterior at the given value(s).
            The shape of the return is (b', b?, q, 1), with a trailing
            dimension.
        """
        if not torch.is_tensor(value):
            # Promote to a (b', b?, q) tensor
            value = torch.tensor(value, device=self.device, dtype=self.dtype)
            value = value.expand(*self.probabilities.shape[:-1]).unsqueeze(0)
        value = value.movedim(0, -1)  # (b?, q, b')

        index = torch.searchsorted(self.cumprobs, value)  # (b?, q, b')

        left_border = self.borders[index]  # (b?, q, b')
        right_border = self.borders[index + 1]

        bucket_width = right_border - left_border
        right_cum_probs = torch.gather(self.cumprobs, -1, index)
        prob_width = torch.gather(self.probabilities, -1, index)

        bucket_proportion_remaining = (right_cum_probs - value) / prob_width
        result = (
            right_border - bucket_proportion_remaining * bucket_width
        )  # (b?, q, b')

        # reshape back to (b', b?, q, 1)
        result = result.movedim(-1, 0).unsqueeze(-1)
        return result


class MultivariateRiemannPosterior(BoundedRiemannPosterior):
    def __init__(self, borders, probabilities, correlation_matrix) -> None:
        """
        A multi-variate bounded Riemann posterior using a Gaussian copula.

        Uses BoundedRiemannPosterior for marginal distributions, and then MVN
        correlation structure via the Gaussian copula.

        Args:
            borders: A tensor of shape ``(num_buckets + 1,)`` defining the boundaries of
                the buckets. Must be monotonically increasing.
            probabilities: A tensor of shape ``(b?, q, num_buckets)`` defining the
                probability mass in each bucket. Must sum to 1 in the last dim.
            correlation_matrix: The Guassian correlation matrix, (b?, q, q).
        """
        super().__init__(borders=borders, probabilities=probabilities)
        self.correlation_matrix = correlation_matrix

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        """
        Sample from the posterior using base samples.

        base_samples are N(0, I) samples, as this posterior is registered
        with the IIDNormalSampler below. This is also necessary for the use
        of the Gaussian copula.

        Args:
            sample_shape: Shape of samples.
            base_samples: (nsamp, b?, q) standard normal samples.

        Returns: Samples from copula, shape (nsamp, b?, q, 1)
        """
        # Construct MVN
        mvn = MultivariateNormal(
            mean=torch.zeros_like(self.mean.squeeze(-1)),
            covariance_matrix=self.correlation_matrix,
        )
        # Draw samples
        samples = mvn.rsample(
            sample_shape=sample_shape, base_samples=base_samples
        ).squeeze(-1)  # (nsamp, b?, q)
        # Convert N(0, 1) marginals to Uniform samples.
        U = torch.distributions.Normal(0, 1).cdf(samples)  # (nsamp, b?, q)
        Z = self.icdf(U)  # (nsamp, b?, q, 1)
        return Z


@GetSampler.register(BoundedRiemannPosterior)
def _get_sampler_riemann(
    posterior: BoundedRiemannPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> NormalMCSampler:
    return _get_sampler_mvn(
        posterior=posterior,
        sample_shape=sample_shape,
        seed=seed,
    )
