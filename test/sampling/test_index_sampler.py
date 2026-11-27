#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest import mock

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

    def test_index_sampler_changed_weights(self):
        for dtype in (torch.float, torch.double):
            for sample_shape in (torch.Size([1]), torch.Size([2, 3])):
                with self.subTest(dtype=dtype, sample_shape=sample_shape):
                    values = torch.arange(12, dtype=dtype, device=self.device).view(
                        3, 2, 2
                    )
                    sampler = IndexSampler(sample_shape=sample_shape, seed=0)
                    for member in (0, 2, 1):
                        weights = torch.zeros(3, dtype=dtype, device=self.device)
                        weights[member] = 1
                        posterior = EnsemblePosterior(values=values, weights=weights)
                        samples = sampler(posterior)
                        expected = values[member].expand(
                            sample_shape + torch.Size([2, 2])
                        )
                        self.assertAllClose(samples, expected)
                        cached = sampler.base_samples
                        # Equal probabilities retain the cached draw; new values do not
                        # require new indices.
                        posterior = EnsemblePosterior(
                            values=values + 1, weights=weights
                        )
                        self.assertAllClose(sampler(posterior), expected + 1)
                        self.assertIs(sampler.base_samples, cached)

    def test_index_sampler_changed_weights_match_fresh_draw(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        sampler = IndexSampler(sample_shape=torch.Size([128]), seed=1)
        sampler(EnsemblePosterior(values, weights=values.new_tensor([0.2, 0.3, 0.5])))
        posterior = EnsemblePosterior(
            values, weights=values.new_tensor([0.6, 0.3, 0.1])
        )
        expected = IndexSampler(sample_shape=sampler.sample_shape, seed=sampler.seed)(
            posterior
        )
        self.assertAllClose(sampler(posterior), expected)

    def test_index_sampler_changed_ensemble_size(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        sampler = IndexSampler(sample_shape=torch.Size([4]), seed=0)
        sampler(EnsemblePosterior(values, weights=values.new_tensor([0.0, 0.0, 1.0])))
        posterior = EnsemblePosterior(values[:1])
        self.assertAllClose(sampler(posterior), values[:1].expand(4, 1, 1))

    def test_index_sampler_probability_metadata_state_dict(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        posterior = EnsemblePosterior(
            values, weights=values.new_tensor([0.2, 0.3, 0.5])
        )
        sampler = IndexSampler(sample_shape=torch.Size([128]), seed=0)
        expected = sampler(posterior)
        state = deepcopy(sampler.state_dict())
        unchanged = deepcopy(state)
        restored = IndexSampler(sample_shape=sampler.sample_shape, seed=29)
        restored.load_state_dict(state, strict=True)
        self.assertAllClose(restored.base_samples, sampler.base_samples)
        with mock.patch("botorch.sampling.index_sampler.Multinomial") as draw:
            rng = torch.random.get_rng_state()
            self.assertAllClose(restored(posterior), expected)
            self.assertAllClose(restored(posterior), expected)
            draw.assert_not_called()
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(state.keys(), unchanged.keys())
        for key in state:
            self.assertTrue(torch.equal(state[key], unchanged[key]))
        # The same probability tensor may change in place.
        posterior.mixture_weights.copy_(values.new_tensor([0.0, 1.0, 0.0]))
        self.assertAllClose(restored(posterior), values[1].expand_as(expected))

    def test_index_sampler_legacy_state_dict(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        posterior = EnsemblePosterior(values)
        indices = torch.tensor([2, 0, 1, 2], device=self.device)
        legacy = {"base_samples": indices.clone()}
        for primed in (False, True):
            with self.subTest(primed=primed):
                restored = IndexSampler(sample_shape=torch.Size([4]), seed=29)
                if primed:
                    restored(posterior)
                restored.load_state_dict(legacy, strict=True)
                self.assertIsNone(restored._base_sample_weights)
                with mock.patch("botorch.sampling.index_sampler.Multinomial") as draw:
                    self.assertAllClose(restored(posterior), values[indices])
                    self.assertAllClose(restored(posterior), values[indices])
                    draw.assert_not_called()
                # The first-use reference cannot recover the original probabilities.
                # Saving again must retain that unknown status.
                self.assertIsNone(restored._base_sample_weights)
                self.assertNotIn("_base_sample_weights", restored.state_dict())
                self.assertEqual(list(legacy), ["base_samples"])
                self.assertTrue(torch.equal(legacy["base_samples"], indices))
                changed = EnsemblePosterior(
                    values, weights=values.new_tensor([0.0, 1.0, 0.0])
                )
                self.assertAllClose(restored(changed), values[1].expand(4, 1, 1))

    def test_index_sampler_populated_state_dict_buffers(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        posterior = EnsemblePosterior(
            values, weights=values.new_tensor([0.2, 0.3, 0.5])
        )
        sampler = IndexSampler(sample_shape=torch.Size([8]), seed=0)
        sampler(posterior)
        state = deepcopy(sampler.state_dict())
        for ensemble_size in (2, 3):
            with self.subTest(ensemble_size=ensemble_size):
                target = IndexSampler(sample_shape=sampler.sample_shape, seed=29)
                target(EnsemblePosterior(values[:ensemble_size].float()))
                indices = target.base_samples
                weights = target._base_sample_weights
                device = weights.device
                target.load_state_dict(state, strict=True)
                self.assertIs(target.base_samples, indices)
                self.assertEqual(target._base_sample_weights.dtype, torch.float32)
                self.assertEqual(target._base_sample_weights.device, device)
                if ensemble_size == 3:
                    self.assertIs(target._base_sample_weights, weights)
                self.assertAllClose(
                    target._base_sample_weights, state["_base_sample_weights"].float()
                )
                self.assertAllClose(target.base_samples, state["base_samples"])

    def test_index_sampler_load_rejects_incompatible_sample_shape(self):
        values = torch.arange(3, dtype=torch.double, device=self.device).view(3, 1, 1)
        target = IndexSampler(sample_shape=torch.Size([4]), seed=0)
        target(EnsemblePosterior(values))
        legacy = {"base_samples": torch.zeros(8, dtype=torch.long, device=self.device)}
        with self.assertRaisesRegex(RuntimeError, "size mismatch for base_samples"):
            target.load_state_dict(legacy, strict=True)
