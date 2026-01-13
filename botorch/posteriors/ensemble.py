#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Ensemble posteriors. Used in conjunction with ensemble models.
"""

from __future__ import annotations

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor
from torch.distributions.multinomial import Multinomial


class EnsemblePosterior(Posterior):
    r"""Ensemble posterior, that should be used for ensemble models that compute
    eagerly a finite number of samples per X value as for example a deep ensemble
    or a random forest."""

    def __init__(self, values: Tensor, weights: Tensor | None = None) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior as
                a ``(b) x s x q x m`` tensor where ``m`` is the output size of the
                model and ``s`` is the ensemble size.
            weights: Optional weights for the ensemble members as a tensor of shape
                ``(s,)``. If None, uses uniform weights.
        """
        if values.ndim < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        self.values = values
        self._weights = weights.to(values) if weights is not None else None
        # Pre-compute normalized weights and mixture properties for efficiency
        self._mixture_dims = list(range(self.values.ndim - 2))
        self._normalized_weights = self._compute_normalized_weights()
        self._normalized_mixture_weights = self._compute_normalized_mixture_weights()

    @property
    def ensemble_size(self) -> int:
        r"""The size of the ensemble"""
        return self.values.shape[-3]

    @property
    def mixture_size(self) -> int:
        r"""The total number of elements in the mixture dimensions"""
        return self.values.shape[:-2].numel()

    def _compute_normalized_weights(self) -> Tensor:
        r"""Compute and cache normalized weights."""
        if self._weights is not None:
            return self._weights / self._weights.sum(dim=-1, keepdim=True)
        else:
            return (
                torch.ones(
                    self.ensemble_size,
                    dtype=self.dtype,
                    device=self.device,
                )
                / self.ensemble_size
            )

    def _compute_normalized_mixture_weights(self) -> Tensor:
        r"""Compute and cache normalized mixture weights."""
        if self._weights is not None:
            unnorm_weights = self._weights.expand(self.values.shape[:-2])
            return unnorm_weights / unnorm_weights.sum(
                dim=self._mixture_dims, keepdim=True
            )
        else:
            return (
                torch.ones(
                    self.values.shape[:-2],
                    dtype=self.dtype,
                    device=self.device,
                )
                / self.mixture_size
            )

    @property
    def weights(self) -> Tensor:
        r"""The weights of the individual models in the ensemble.
        uniformly weighted by default."""
        return self._normalized_weights

    @property
    def mixture_weights(self) -> Tensor:
        r"""The weights of the individual models in the ensemble.
        uniformly weighted by default, and normalized over ensemble and
        batch dimensions of the model."""
        return self._normalized_mixture_weights

    @property
    def mixture_dims(self) -> list[int]:
        r"""The mixture dimensions of the posterior. For ensemble posteriors,
        this includes all dimensions except the last two (query points and outputs)."""
        return self._mixture_dims

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return self.values.dtype

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a ``(b) x n x m``-dim Tensor."""
        # Weighted average across ensemble dimension
        return (self.values * self.weights[..., None, None]).sum(dim=-3)

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a ``(b) x n x m``-dim Tensor.

        Computed as the weighted sample variance across the ensemble outputs.

        This treats weights as probability weights (normalized to sum to 1) and
        computes the unbiased weighted sample variance using the formula:
        Var = Σ(w_i * (x_i - μ)²) / (1 - Σw_i²)
        where the sum over w_i² is taken over the ensemble dimension only.
        Source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean under
        "Reliability Weights".
        """
        if self.ensemble_size == 1:
            return torch.zeros_like(self.values.squeeze(-3))

        # Add dimensions for query points and outputs to enable broadcasting
        weights = self.weights[..., None, None]
        squared_deviations = (self.values - self.mean.unsqueeze(-3)) ** 2
        return (weights * squared_deviations).sum(dim=-3) / (1 - (weights**2).sum())

    @property
    def mixture_mean(self) -> Tensor:
        r"""The mixture mean of the posterior as a ``(b) x n x m``-dim Tensor.

        Computed as the weighted average across the ensemble outputs.
        """
        return (self.values * self.mixture_weights[..., None, None]).sum(
            dim=self.mixture_dims
        )

    @property
    def mixture_variance(self) -> Tensor:
        r"""The mixture variance of the posterior as a ``(b) x n x m``-dim Tensor.

        Computed as the weighted sample variance across the ensemble outputs.

        This treats weights as probability weights (normalized to sum to 1) and
        computes the unbiased weighted sample variance using the formula:
        Var = Σ(w_i * (x_i - μ)²) / (1 - Σw_i²) where w_i is normalized over the
        entire mixture, and the sum over w_i² is taken over all mixture dimensions.
        Source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean under
        "Reliability Weights".
        """

        # Add dimensions for query points and outputs to enable broadcasting
        weights = self.mixture_weights[..., None, None]
        squared_deviations = (self.values - self.mixture_mean.unsqueeze(-3)) ** 2
        return (weights * squared_deviations).sum(dim=self.mixture_dims) / (
            1 - (weights**2).sum()
        )

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given ``sample_shape``.
        """
        return sample_shape + self.values.shape[:-3] + self.values.shape[-2:]

    @property
    def batch_shape(self) -> torch.Size:
        return self.values.shape[:-3]

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Based on the sample shape, base samples are generated and passed to
        ``rsample_from_base_samples``.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if sample_shape is None or len(sample_shape) == 0:
            sample_shape = torch.Size([1])

        # NOTE This occasionally happens in Hypervolume evals when there
        # are no points which improve over the reference point. In this case, we
        # create a posterior for all the points which improve over the reference,
        # which is an empty set.
        if self.values.numel() == 0:
            return torch.empty(
                *self._extended_shape(sample_shape=sample_shape),
                device=self.device,
                dtype=self.dtype,
            )

        base_samples = (
            Multinomial(
                probs=self.mixture_weights,
            )
            .sample(sample_shape=sample_shape)
            .argmax(dim=-1)
        )
        return self.rsample_from_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: A Tensor of indices as base samples of shape
                ``sample_shape``, typically obtained from ``IndexSampler``.
                This is used for deterministic optimization. The predictions of
                the ensemble corresponding to the indices are then sampled.


        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        # Check that the first dimensions of base_samples match sample_shape
        if base_samples.shape != sample_shape + self.batch_shape:
            raise ValueError(
                f"Sample_shape={sample_shape + self.batch_shape} does not match "
                f"the leading dimensions of base_samples.shape={base_samples.shape}."
            )

        if self.batch_shape:
            # Values is always going to be 4-dimensional with this reshape,
            # even if we have more than one batch dimension
            values = self.values.reshape(
                ((self.batch_shape.numel(),) + self.values.shape[-3:])
            )

            # Collapse the base samples to enable index selecting along the
            # ensemble dim (dim -3)
            batch_numel = self.batch_shape.numel()
            collapsed_base_samples = base_samples.reshape(sample_shape + (batch_numel,))

            # First dimension is just 1, 2, 3, ..., batch_shape.numel() -1 to flatten
            # the first dimension and extract one index

            # second dimension extracts the ensemble member, for each element in the
            # entire batch shape
            return values[torch.arange(batch_numel), collapsed_base_samples].reshape(
                self._extended_shape(sample_shape=sample_shape)
            )
        return self.values[base_samples]
