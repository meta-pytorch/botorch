#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.utils.priors import BetaPrior
from botorch.utils.testing import BotorchTestCase
from gpytorch.priors.utils import BUFFERED_PREFIX
from torch.distributions import Beta


class TestBetaPrior(BotorchTestCase):
    def test_init(self):
        prior = BetaPrior(1.2, 0.9)
        self.assertAlmostEqual(prior.concentration1.item(), 1.2)
        self.assertAlmostEqual(prior.concentration0.item(), 0.9)
        self.assertIsNone(prior._transform)

    def test_init_with_transform(self):
        prior = BetaPrior(2.0, 3.0, transform=torch.sigmoid)
        self.assertIs(prior._transform, torch.sigmoid)

    def test_log_prob_matches_torch(self):
        prior = BetaPrior(1.2, 0.9)
        ref = Beta(torch.tensor(1.2), torch.tensor(0.9))
        x = torch.rand(5, 4)
        self.assertTrue(torch.allclose(prior.log_prob(x), ref.log_prob(x)))

    def test_log_prob_with_transform(self):
        def transform(x):
            return x.clamp(0.01, 0.99)

        prior = BetaPrior(2.0, 2.0, transform=transform)
        prior_no_transform = BetaPrior(2.0, 2.0)
        x = torch.rand(3, 2)
        lp = prior.log_prob(x)
        self.assertEqual(lp.shape, torch.Size([3, 2]))
        # Values at boundaries should differ due to clamping
        x_boundary = torch.tensor([[0.001, 0.999]])
        self.assertFalse(
            torch.allclose(
                prior.log_prob(x_boundary),
                prior_no_transform.log_prob(x_boundary),
            )
        )

    def test_log_prob_batch(self):
        prior = BetaPrior(1.5, 2.5)
        x = torch.rand(7, 3, 2)
        lp = prior.log_prob(x)
        self.assertEqual(lp.shape, torch.Size([7, 3, 2]))

    def test_rsample(self):
        prior = BetaPrior(1.2, 0.9)
        samples = prior.rsample(torch.Size([10, 5]))
        self.assertEqual(samples.shape, torch.Size([10, 5]))
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples <= 1))

    def test_state_dict_roundtrip(self):
        prior = BetaPrior(1.2, 0.9)
        sd = prior.state_dict()
        self.assertIn(f"{BUFFERED_PREFIX}concentration1", sd)
        self.assertIn(f"{BUFFERED_PREFIX}concentration0", sd)

        prior2 = BetaPrior(999.0, 999.0)
        prior2.load_state_dict(sd)
        self.assertAlmostEqual(prior2.concentration1.item(), 1.2)
        self.assertAlmostEqual(prior2.concentration0.item(), 0.9)

    def test_expand(self):
        prior = BetaPrior(1.2, 0.9)
        expanded = prior.expand(torch.Size([3, 2]))
        self.assertEqual(expanded.concentration1.shape, torch.Size([3, 2]))
        self.assertEqual(expanded.concentration0.shape, torch.Size([3, 2]))

    def test_expand_preserves_transform(self):
        prior = BetaPrior(1.2, 0.9, transform=torch.sigmoid)
        expanded = prior.expand(torch.Size([3, 2]))
        self.assertIs(expanded._transform, torch.sigmoid)
        self.assertEqual(expanded._validate_args, prior._validate_args)
