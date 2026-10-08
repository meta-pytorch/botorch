#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler to be used with ``EnsemblePosteriors`` to enable
deterministic optimization of acquisition functions with ensemble models.
"""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.sampling.base import MCSampler
from torch import Tensor
from torch.distributions.multinomial import Multinomial


class IndexSampler(MCSampler):
    r"""A sampler that calls ``posterior.rsample_from_base_samples`` to
    generate the samples via index base samples."""

    def forward(self, posterior: EnsemblePosterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The ensemble posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        self._construct_base_samples(posterior=posterior)
        samples = posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape, base_samples=self.base_samples
        )
        return samples

    def _construct_base_samples(self, posterior: EnsemblePosterior) -> None:
        r"""Constructs base samples as indices to sample with them from
        the Posterior.

        The same ensemble member is used for all batch dimensions of a given MC
        sample, i.e., the batch dimensions are collapsed (as for the normal samplers
        with ``batch_range``). This ensures that a candidate produces the same
        acquisition value regardless of its position in the t-batch, and that the
        sample average approximation does not change with the t-batch size (e.g.,
        between raw-sample screening and the optimization restarts).

        Args:
            posterior: The ensemble posterior to construct the base samples
                for.
        """
        target_shape = self.sample_shape + posterior.batch_shape
        if self.base_samples is None or self.base_samples.shape != target_shape:
            weights = posterior.weights
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                if weights.ndim == 1:
                    base_samples = (
                        Multinomial(probs=weights)
                        .sample(sample_shape=self.sample_shape)
                        .argmax(dim=-1)
                        .view(self.sample_shape + (1,) * len(posterior.batch_shape))
                        .expand(target_shape)
                    )
                else:  # batched ensemble weights, draw per batch
                    base_samples = (
                        Multinomial(probs=posterior.mixture_weights)
                        .sample(sample_shape=self.sample_shape)
                        .argmax(dim=-1)
                    )
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover

    def _update_base_samples(
        self, posterior: EnsemblePosterior, base_sampler: IndexSampler
    ) -> None:
        r"""Null operation just needed for compatibility with
        ``CachedCholeskyAcquisitionFunction``."""
        pass
