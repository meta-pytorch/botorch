#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.sampling.index_sampler import IndexSampler
from botorch.utils.testing import BotorchTestCase


class TestIndexSampler(BotorchTestCase):
    def test_index_sampler(self):
        # Basic usage.
        posterior = EnsemblePosterior(
            values=torch.randn(torch.Size((50, 16, 1, 1))).to(self.device)
        )
        posterior_batch_shape = posterior.batch_shape
        sampler = IndexSampler(sample_shape=torch.Size((128,)))
        samples = sampler(posterior)
        self.assertTrue(samples.shape == torch.Size((128, 50, 1, 1)))
        self.assertTrue(sampler.base_samples.max() < 16)
        self.assertTrue(sampler.base_samples.min() >= 0)
        # check deterministic nature
        samples2 = sampler(posterior)
        self.assertAllClose(samples, samples2)
        # test construct base samples
        sampler = IndexSampler(sample_shape=torch.Size((4, 128)), seed=42)
        self.assertTrue(sampler.base_samples is None)
        sampler._construct_base_samples(posterior=posterior)
        (
            self.assertTrue(
                sampler.base_samples.shape
                == (torch.Size((4, 128))) + posterior_batch_shape
            )
        )
        self.assertTrue(
            sampler.base_samples.device.type
            == posterior.device.type
            == self.device.type
        )
        base_samples = sampler.base_samples
        sampler = IndexSampler(sample_shape=torch.Size((4, 128)), seed=42)
        sampler._construct_base_samples(posterior=posterior)
        self.assertAllClose(base_samples, sampler.base_samples)
