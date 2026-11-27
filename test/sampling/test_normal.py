#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import (
    IIDNormalSampler,
    NormalMCSampler,
    SobolQMCNormalSampler,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator


def _get_test_posterior(device, dtype=torch.float):
    mean = torch.zeros(2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


def _get_test_posterior_batched(device, dtype=torch.float):
    mean = torch.zeros(3, 2, device=device, dtype=dtype)
    cov = torch.eye(2, device=device, dtype=dtype).repeat(3, 1, 1)
    mvn = MultivariateNormal(mean, cov)
    return GPyTorchPosterior(mvn)


class TestNormalMCSampler(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            NormalMCSampler(sample_shape=torch.Size([4]))

    def test_update_base_samples(self):
        # The base samples of ``base_sampler`` are re-used for the first points of the
        # new posterior, for both one- and multi-dimensional sample shapes.
        for cls, sample_shape, dtype in itertools.product(
            (IIDNormalSampler, SobolQMCNormalSampler),
            (torch.Size([4]), torch.Size([2, 3])),
            (torch.float, torch.double),
        ):
            base_sampler = cls(sample_shape=sample_shape, seed=0)
            base_sampler(_get_test_posterior(device=self.device, dtype=dtype))
            # a batched posterior over three points
            mean = torch.zeros(3, 3, device=self.device, dtype=dtype)
            cov = torch.eye(3, device=self.device, dtype=dtype).repeat(3, 1, 1)
            posterior = GPyTorchPosterior(MultivariateNormal(mean, cov))
            sampler = cls(sample_shape=sample_shape, seed=1)
            sampler._update_base_samples(posterior=posterior, base_sampler=base_sampler)
            self.assertEqual(sampler.base_samples.shape, sample_shape + (1, 3))
            self.assertTrue(
                torch.equal(
                    sampler.base_samples[..., :2],
                    base_sampler.base_samples.unsqueeze(-2),
                )
            )
            samples = sampler(posterior)
            self.assertEqual(samples.shape, sample_shape + (3, 3, 1))


class TestIIDNormalSampler(BotorchTestCase):
    def test_forward(self):
        for dtype in (torch.float, torch.double):
            sampler = IIDNormalSampler(sample_shape=torch.Size([4]), seed=1234)
            self.assertEqual(sampler.seed, 1234)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertAllClose(samples, samples2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=new_dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            # ensure this works with a different batch_range
            sampler.batch_range_override = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))


class TestSobolQMCNormalSampler(BotorchTestCase):
    def test_forward(self):
        for dtype in (torch.float, torch.double):
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([4]), seed=1234)
            self.assertEqual(sampler.seed, 1234)
            # check samples non-batched
            posterior = _get_test_posterior(device=self.device, dtype=dtype)
            samples = sampler(posterior)
            self.assertEqual(samples.shape, torch.Size([4, 2, 1]))
            # ensure samples are the same
            samples2 = sampler(posterior)
            self.assertAllClose(samples, samples2)
            # ensure this works with a differently shaped posterior
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            # ensure this works when changing the dtype
            new_dtype = torch.float if dtype == torch.double else torch.double
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=new_dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))
            # ensure this works with a different batch_range
            sampler.batch_range_override = (-3, -1)
            posterior_batched = _get_test_posterior_batched(
                device=self.device, dtype=dtype
            )
            samples_batched = sampler(posterior_batched)
            self.assertEqual(samples_batched.shape, torch.Size([4, 3, 2, 1]))

    def test_unsupported_dimension(self):
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        maxdim = torch.quasirandom.SobolEngine.MAXDIM + 1
        mean = torch.zeros(maxdim)
        cov = DiagLinearOperator(torch.ones(maxdim))
        mvn = MultivariateNormal(mean, cov)
        posterior = GPyTorchPosterior(mvn)
        with self.assertRaises(UnsupportedError) as e:
            sampler(posterior)
            self.assertIn(f"Requested: {maxdim}", str(e.exception))
