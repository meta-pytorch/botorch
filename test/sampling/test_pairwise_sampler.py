#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pairwise_samplers import (
    PairwiseIIDNormalSampler,
    PairwiseSobolQMCNormalSampler,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal


def _get_test_posterior(device, n=3, dtype=torch.float, batched=False):
    mean = torch.zeros(n, device=device, dtype=dtype)
    cov = torch.eye(n, device=device, dtype=dtype)
    if batched:
        cov = cov.repeat(3, 1, 1)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


class TestPairwiseIIDNormalSampler(BotorchTestCase):
    def test_forward(self):
        for dtype in (torch.float, torch.double):
            sampler = PairwiseIIDNormalSampler(sample_shape=torch.Size([4]), seed=1234)
            self.assertEqual(sampler.seed, 1234)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertAllClose(samples, samples2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=new_dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseIIDNormalSampler(
                sample_shape=torch.Size([4]), max_num_comparisons=2
            )
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))


class TestPairwiseSobolQMCNormalSampler(BotorchTestCase):
    def test_forward(self):
        for dtype in (torch.float, torch.double):
            sampler = PairwiseSobolQMCNormalSampler(
                sample_shape=torch.Size([4]), seed=1234
            )
            self.assertEqual(sampler.seed, 1234)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertAllClose(samples, samples2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior(
                device=self.device, dtype=new_dtype, batched=True
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 3, 2]))
            # ensure error is rasied when number of points are < 2
            posterior = _get_test_posterior(device=self.device, n=1, dtype=dtype)
            with self.assertRaises(RuntimeError):
                sampler(posterior)

            # check max_num_comparisons
            sampler = PairwiseSobolQMCNormalSampler(
                sample_shape=torch.Size([4]), max_num_comparisons=2
            )
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 2]))


class TestPairwiseSamplerConvention(BotorchTestCase):
    def test_winner_first_and_rng_isolation(self):
        # Utilities are well separated, so the preferred point (listed first,
        # following the PairwiseGP convention) must have the larger utility.
        mean = torch.arange(4, device=self.device, dtype=torch.double)
        cov = 1e-6 * torch.eye(4, device=self.device, dtype=torch.double)
        posterior = GPyTorchPosterior(MultivariateNormal(mean, cov))
        for sampler_cls in (PairwiseIIDNormalSampler, PairwiseSobolQMCNormalSampler):
            sampler = sampler_cls(sample_shape=torch.Size([3]), seed=0)
            np.random.seed(123)
            expected = np.random.rand()
            np.random.seed(123)
            comps = sampler(posterior)
            # the global NumPy RNG must not be reseeded by the sampler
            self.assertEqual(np.random.rand(), expected)
            self.assertEqual(comps.shape, torch.Size([3, 6, 2]))
            self.assertEqual(comps.device, posterior.device)
            self.assertTrue((comps[..., 0] > comps[..., 1]).all())
