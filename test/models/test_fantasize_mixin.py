#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from botorch.models.model import FantasizeMixin
from botorch.utils.testing import BotorchTestCase


class RecordingModel(FantasizeMixin):
    num_outputs = 1
    batch_shape = torch.Size()
    likelihood = None

    def transform_inputs(self, X, **kwargs):
        return X

    def posterior(self, X, **kwargs):
        return SimpleNamespace(mean=X[..., :1])

    def condition_on_observations(self, X, Y, **kwargs):
        return SimpleNamespace(X=X, Y=Y, noise=kwargs["noise"])


class TestFantasyNoiseShape(BotorchTestCase):
    def test_noise_excludes_every_sample_axis(self) -> None:
        for sample_shape in [torch.Size([3]), torch.Size([2, 3]), torch.Size()]:
            for points in [0, 4]:
                with self.subTest(sample_shape=sample_shape, points=points):
                    X = torch.zeros(2, points, 1, device=self.device)
                    noise = torch.full((2, 1, 1), 0.1, device=self.device)
                    sampler = Mock()
                    sampler.sample_shape = sample_shape
                    sampler.side_effect = lambda p: p.mean.expand(
                        sample_shape + p.mean.shape
                    )
                    result = RecordingModel().fantasize(
                        X, sampler, observation_noise=noise
                    )
                    self.assertEqual(
                        result.Y.shape, sample_shape + torch.Size([2, points, 1])
                    )
                    self.assertEqual(result.noise.shape, torch.Size([2, points, 1]))
                    self.assertAllClose(result.noise, noise.expand(2, points, 1))
