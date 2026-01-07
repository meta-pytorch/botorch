#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.posteriors.posterior_list import PosteriorList
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.testing import BotorchTestCase, MockPosterior


class TestListSampler(BotorchTestCase):
    def test_list_sampler(self):
        # Test initialization.
        sampler = ListSampler(
            IIDNormalSampler(sample_shape=torch.Size([2])),
            StochasticSampler(sample_shape=torch.Size([2])),
        )
        self.assertIsInstance(sampler.samplers[0], IIDNormalSampler)
        self.assertIsInstance(sampler.samplers[1], StochasticSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([2]))

        # Test validation.
        with self.assertRaisesRegex(UnsupportedError, "all samplers to have the "):
            ListSampler(
                StochasticSampler(sample_shape=torch.Size([2])),
                StochasticSampler(sample_shape=torch.Size([3])),
            )

        # Test basic usage.
        org_samples = torch.rand(1, 5)
        p1 = MockPosterior(samples=org_samples[:, :2])
        p2 = MockPosterior(samples=org_samples[:, 2:])
        p_list = PosteriorList(p1, p2)
        samples = sampler(p_list)
        self.assertAllClose(samples, org_samples.repeat(2, 1, 1))

        # Test _update_base_samples.
        sampler = ListSampler(
            IIDNormalSampler(sample_shape=torch.Size([2])),
            SobolQMCNormalSampler(sample_shape=torch.Size([2])),
        )
        sampler2 = ListSampler(
            IIDNormalSampler(sample_shape=torch.Size([2])),
            SobolQMCNormalSampler(sample_shape=torch.Size([2])),
        )
        with mock.patch.object(
            sampler.samplers[0], "_update_base_samples"
        ) as update_0, mock.patch.object(
            sampler.samplers[1], "_update_base_samples"
        ) as update_1:
            sampler._update_base_samples(posterior=p_list, base_sampler=sampler2)
        update_0.assert_called_once_with(
            posterior=p1, base_sampler=sampler2.samplers[0]
        )
        update_1.assert_called_once_with(
            posterior=p2, base_sampler=sampler2.samplers[1]
        )
