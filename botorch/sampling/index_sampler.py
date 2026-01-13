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

        Args:
            posterior: The ensemble posterior to construct the base samples
                for.
        """
        if (
            self.base_samples is None
            or self.base_samples.shape != self.sample_shape + posterior.batch_shape
        ):
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                base_samples = (
                    Multinomial(
                        probs=posterior.mixture_weights,
                    )
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
