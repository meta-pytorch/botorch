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

from typing import Any

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.sampling.base import MCSampler
from torch import Tensor
from torch.distributions.multinomial import Multinomial


class IndexSampler(MCSampler):
    r"""A sampler that calls ``posterior.rsample_from_base_samples`` to
    generate the samples via index base samples.

    Legacy checkpoints do not record the probabilities associated with their
    cached indices. These draws are retained, and the first posterior supplies
    only a reference for detecting subsequent probability changes; the original
    probabilities cannot be verified.
    """

    _version = 2

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
        the Posterior. Categorical indices depend on the posterior weights, so
        cached indices must be refreshed when those probabilities change.

        Args:
            posterior: The ensemble posterior to construct the base samples
                for.
        """
        weights = posterior.mixture_weights
        target_shape = self.sample_shape + posterior.batch_shape
        if (
            self.base_samples is not None
            and self.base_samples.device != posterior.device
        ):
            self.to(device=posterior.device)  # pragma: nocover
        cached_weights = getattr(self, "_base_sample_weights", None)
        reference_weights = (
            cached_weights
            if cached_weights is not None
            else getattr(self, "_legacy_sample_weights", None)
        )
        if (
            self.base_samples is not None
            and self.base_samples.shape == target_shape
            and reference_weights is None
        ):
            # Retain legacy draws with unknown probability context. The first
            # posterior is a reference for future changes, not recovered metadata.
            self.register_buffer(
                "_legacy_sample_weights", weights.detach().clone(), persistent=False
            )
            reference_weights = self._legacy_sample_weights
        if (
            self.base_samples is None
            or self.base_samples.shape != target_shape
            or reference_weights is None
            or not torch.equal(reference_weights, weights)
        ):
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                base_samples = (
                    Multinomial(
                        probs=weights,
                    )
                    .sample(sample_shape=self.sample_shape)
                    .argmax(dim=-1)
                )
            self.register_buffer("base_samples", base_samples)
            self.register_buffer("_base_sample_weights", weights.detach().clone())
            self.register_buffer("_legacy_sample_weights", None, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Allocate dynamic buffers without modifying the incoming state dict.
        # Absent probability metadata explicitly denotes a legacy, unknown context.
        for name in ("base_samples", "_base_sample_weights"):
            value = state_dict.get(prefix + name)
            if isinstance(value, Tensor):
                current = getattr(self, name, None)
                if current is None:
                    self.register_buffer(name, torch.empty_like(value))
                elif name == "_base_sample_weights" and current.shape != value.shape:
                    self.register_buffer(name, current.new_empty(value.shape))
            elif name == "_base_sample_weights":
                self.register_buffer(name, None)
        self.register_buffer("_legacy_sample_weights", None, persistent=False)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _update_base_samples(
        self, posterior: EnsemblePosterior, base_sampler: IndexSampler
    ) -> None:
        r"""Null operation just needed for compatibility with
        ``CachedCholeskyAcquisitionFunction``."""
        pass
