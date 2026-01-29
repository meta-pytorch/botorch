#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A ``SamplerList`` for sampling from a ``PosteriorList``.
"""

from __future__ import annotations

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors.posterior_list import PosteriorList
from botorch.sampling.base import MCSampler
from torch import Tensor
from torch.nn import ModuleList


class ListSampler(MCSampler):
    def __init__(self, *samplers: MCSampler) -> None:
        r"""A list of samplers for sampling from a ``PosteriorList``.

        Args:
            samplers: A variable number of samplers. This should include
                a sampler for each posterior.
        """
        super(MCSampler, self).__init__()
        self.samplers = ModuleList(samplers)
        self._validate_samplers()

    def _validate_samplers(self) -> None:
        r"""Checks that the samplers share the same sample shape."""
        sample_shapes = [s.sample_shape for s in self.samplers]
        if not all(sample_shapes[0] == ss for ss in sample_shapes):
            raise UnsupportedError(
                "ListSampler requires all samplers to have the same sample shape."
            )

    @property
    def sample_shape(self) -> torch.Size:
        r"""The sample shape of the underlying samplers."""
        self._validate_samplers()
        return self.samplers[0].sample_shape

    def forward(self, posterior: PosteriorList) -> Tensor:
        r"""Samples from the posteriors and concatenates the samples.

        Args:
            posterior: A ``PosteriorList`` to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        samples_list = [
            s(posterior=p) for s, p in zip(self.samplers, posterior.posteriors)
        ]
        return posterior._reshape_and_cat(tensors=samples_list)

    def _update_base_samples(
        self, posterior: PosteriorList, base_sampler: ListSampler
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        This is used in CachedCholeskyAcquisitionFunctions to ensure consistency.

        Args:
            posterior: The posterior for which the base samples are constructed.
            base_sampler: The base sampler to retrieve the base samples from.
        """
        self._instance_check(base_sampler=base_sampler)
        for s, p, bs in zip(self.samplers, posterior.posteriors, base_sampler.samplers):
            s._update_base_samples(posterior=p, base_sampler=bs)
