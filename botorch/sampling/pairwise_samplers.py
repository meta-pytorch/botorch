#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from torch import Tensor


class PairwiseMCSampler(MCSampler):
    r"""
    Abstract class for Pairwise MC Sampler.

    This sampler will sample pairwise comparisons. It is to be used together
    with PairwiseGP and BoTorch acquisition functions (e.g., qKnowledgeGradient)

    """

    def __init__(self, max_num_comparisons: int = None, seed: int = None) -> None:
        r"""
        Args:
            max_num_comparisons: Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons
            seed: The seed for np.random.seed. If omitted, use a random seed.
                May be overwritten by sibling classes or subclasses.
        """
        self.max_num_comparisons = max_num_comparisons
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior and make comparisons

        Args:
            posterior: The Posterior to sample from.
                The returned samples are expected to have output dimension of 1.

        Returns:
            Posterior sample pairwise comparisons.
        """
        samples = super().forward(posterior)
        np.random.seed(self.seed)

        s_n = samples.shape[-2]  # candidate number per batch
        if s_n < 2:
            raise RuntimeError("Number of samples < 2, cannot make comparisons")

        # TODO: Don't instantiate a generator
        all_pairs = np.array(list(combinations(range(s_n), 2)))
        if self.max_num_comparisons is None:
            comp_n = len(all_pairs)
        else:
            comp_n = min(self.max_num_comparisons, len(all_pairs))

        comp_pairs = all_pairs[
            np.random.choice(range(len(all_pairs)), comp_n, replace=False)
        ]
        s_comps_size = torch.Size((*samples.shape[:-2], comp_n, 2))
        s_v = samples.view(-1, s_n)

        idx1, idx2 = comp_pairs[:, 0], comp_pairs[:, 1]
        prefs = (s_v[:, idx1] > s_v[:, idx2]).long().cpu()
        cpt = comp_pairs.T
        c1 = np.choose(prefs, cpt)
        c2 = np.choose(1 - prefs, cpt)
        s_comps = torch.stack([c1, c2], dim=-1).reshape(s_comps_size)

        return s_comps


class PairwiseIIDNormalSampler(PairwiseMCSampler, IIDNormalSampler):
    def __init__(
        self,
        sample_shape: torch.Size,
        seed: int | None = None,
        max_num_comparisons: int = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            sample_shape: The ``sample_shape`` of the samples to generate.
            seed: The seed for the RNG. If omitted, use a random seed.
            max_num_comparisons:  Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons.
            kwargs: Catch-all for deprecated arguments.
        """
        PairwiseMCSampler.__init__(
            self, max_num_comparisons=max_num_comparisons, seed=seed
        )
        IIDNormalSampler.__init__(self, sample_shape=sample_shape, seed=seed, **kwargs)


class PairwiseSobolQMCNormalSampler(PairwiseMCSampler, SobolQMCNormalSampler):
    def __init__(
        self,
        sample_shape: torch.Size,
        seed: int | None = None,
        max_num_comparisons: int = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            sample_shape: The ``sample_shape`` of the samples to generate.
            seed: The seed for the RNG. If omitted, use a random seed.
            max_num_comparisons:  Max number of comparisons drawn within samples.
                If None, use all possible pairwise comparisons.
            kwargs: Catch-all for deprecated arguments.
        """
        PairwiseMCSampler.__init__(
            self, max_num_comparisons=max_num_comparisons, seed=seed
        )
        SobolQMCNormalSampler.__init__(
            self, sample_shape=sample_shape, seed=seed, **kwargs
        )
