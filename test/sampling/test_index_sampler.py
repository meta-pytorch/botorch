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

    def test_index_sampler_collapses_batch_dims(self):
        # The same candidate replicated across t-batches must produce the same
        # samples, and the samples must not depend on the t-batch size.
        values = torch.randn(8, 1, 1, device=self.device)
        sampler = IndexSampler(sample_shape=torch.Size((16,)), seed=0)
        samples = sampler(EnsemblePosterior(values=values.expand(4, 8, 1, 1)))
        self.assertEqual(samples.shape, torch.Size((16, 4, 1, 1)))
        self.assertTrue((samples == samples[:, :1]).all())
        samples_2 = sampler(EnsemblePosterior(values=values.expand(2, 3, 8, 1, 1)))
        self.assertEqual(samples_2.shape, torch.Size((16, 2, 3, 1, 1)))
        self.assertTrue((samples_2 == samples[:, :1].unsqueeze(1)).all())
        # Batched (per-batch) ensemble weights are sampled per batch.
        weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device)
        posterior = EnsemblePosterior(
            values=torch.stack([torch.zeros(2, 1, 1), torch.ones(2, 1, 1)], dim=1).to(
                self.device
            ),
            weights=weights,
        )
        samples_w = IndexSampler(sample_shape=torch.Size((8,)), seed=0)(posterior)
        self.assertTrue((samples_w[:, 0] == 0).all())
        self.assertTrue((samples_w[:, 1] == 1).all())
