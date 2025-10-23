#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.testing import BotorchTestCase


class TestEnsemblePosterior(BotorchTestCase):
    def test_EnsemblePosterior_invalid(self):
        for shape, dtype in itertools.product(
            ((5, 2), (5, 1)), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            values = torch.randn(*shape, **tkwargs)
            with self.assertRaisesRegex(
                ValueError,
                "Values has to be at least three-dimensional",
            ):
                EnsemblePosterior(values)

    def test_EnsemblePosterior_as_Deterministic(self):
        for shape, dtype in itertools.product(
            ((1, 3, 2), (2, 1, 3, 2)), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            values = torch.randn(*shape, **tkwargs)
            p = EnsemblePosterior(values)
            self.assertEqual(p.ensemble_size, 1)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(
                p._extended_shape(torch.Size((1,))),
                torch.Size((1, 3, 2)) if len(shape) == 3 else torch.Size((1, 2, 3, 2)),
            )
            self.assertEqual(p.weights, torch.ones(1, **tkwargs))
            with self.assertRaises(NotImplementedError):
                p.base_sample_shape
            self.assertTrue(torch.equal(p.mean, values.squeeze(-3)))
            self.assertTrue(
                torch.equal(p.variance, torch.zeros_like(values.squeeze(-3)))
            )
            # test sampling
            samples = p.rsample()
            self.assertTrue(torch.equal(samples, values.squeeze(-3).unsqueeze(0)))
            samples = p.rsample(torch.Size([2]))
            self.assertEqual(samples.shape, p._extended_shape(torch.Size([2])))

    def test_EnsemblePosterior(self):
        for shape, dtype in itertools.product(
            ((16, 5, 2), (2, 16, 5, 2)), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            values = torch.randn(*shape, **tkwargs)
            p = EnsemblePosterior(values)
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.ensemble_size, 16)
            self.assertEqual(
                p.batch_shape, torch.Size([]) if len(shape) == 3 else torch.Size([2])
            )
            self.assertAllClose(
                p.weights,
                torch.tensor([1.0 / p.ensemble_size] * p.ensemble_size).to(p.values),
            )
            # test mean and variance
            self.assertTrue(torch.equal(p.mean, values.mean(dim=-3)))
            self.assertTrue(torch.allclose(p.variance, values.var(dim=-3)))
            # test extended shape
            self.assertEqual(
                p._extended_shape(torch.Size((128,))),
                (
                    torch.Size((128, 5, 2))
                    if len(shape) == 3
                    else torch.Size((128, 2, 5, 2))
                ),
            )
            # test mixture_mean and mixture_variance
            expected_mixture_mean = values.mean(dim=list(range(values.ndim - 2)))
            expected_mixture_var = values.var(dim=list(range(values.ndim - 2)))
            self.assertAllClose(p.mixture_mean, expected_mixture_mean)
            self.assertAllClose(p.mixture_variance, expected_mixture_var)
            # test rsample
            samples = p.rsample(torch.Size((4096,)))
            self.assertEqual(samples.shape, p._extended_shape(torch.Size((4096,))))

            self.assertAllClose(p.mean, samples.mean(dim=0), rtol=1e-01, atol=1e-01)
            self.assertAllClose(p.variance, samples.var(dim=0), rtol=1e-01, atol=1e-01)
            # test error on base_samples, sample_shape mismatch
            with self.assertRaises(ValueError):
                p.rsample_from_base_samples(
                    sample_shape=torch.Size((17,)),
                    base_samples=torch.arange(
                        16,
                        **tkwargs,
                    ).expand(*shape[:-2]),
                )
            # test zero batch shape tensor - should return an empty second dimension
            empty_values = torch.empty((0, 1, 5, 2), **tkwargs)
            p_empty = EnsemblePosterior(empty_values)
            empty_samples = p_empty.rsample(torch.Size([4]))
            self.assertEqual(empty_samples.shape, torch.Size([4, 0, 5, 2]))

    def test_EnsemblePosterior_weighted(self):
        """Test that weighted mean and variance calculations work correctly."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # Test case 1: No batch dimensions, shape (3, 2, 1)
            values = torch.tensor(
                [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], **tkwargs
            )
            norm_weights = torch.tensor([0.5, 0.5, 0.0], **tkwargs)
            p = EnsemblePosterior(values, weights=norm_weights)

            expected_mean = torch.tensor([[2.0], [3.0]], **tkwargs)  # weighted means
            expected_variance = torch.tensor(
                [[2.0], [2.0]], **tkwargs
            )  # weighted variances
            self.assertAllClose(p.mean, expected_mean)
            self.assertAllClose(p.weights, norm_weights)

            # test for unnormalized weights
            weights = torch.tensor([3.0, 3.0, 0.0], **tkwargs)
            p = EnsemblePosterior(values, weights=weights)
            self.assertAllClose(p.weights, norm_weights)
            self.assertAllClose(p.mean, expected_mean)
            self.assertAllClose(p.variance, expected_variance)
            #  With batch dimensions, shape (2, 3, 2, 1)
            batch_values = torch.tensor(
                [
                    [[[1.0], [2.0]], [[3.0], [6.0]], [[5.0], [6.0]]],
                    [[[2.0], [4.0]], [[4.0], [8.0]], [[6.0], [7.0]]],
                ],
                **tkwargs,
            )
            p_batch = EnsemblePosterior(batch_values, weights=weights)
            # Weights should remain 1-dimensional regardless of batch dimensions

            self.assertAllClose(p_batch.weights, norm_weights)
            self.assertAllClose(p_batch.mixture_weights, norm_weights.repeat(2, 1) / 2)

            # batch size 2, so the mixture weights should be halved
            # Check that mean calculation works with batch dimensions
            # bottom row is 1 larger everywhere than the top row, so results
            # should be too.
            expected_batch_mean = torch.tensor(
                [[[2.0], [4.0]], [[3.0], [6.0]]], **tkwargs
            )
            expected_batch_variance = torch.tensor(
                [[[2.0], [8.0]], [[2.0], [8.0]]], **tkwargs
            )
            self.assertAllClose(p_batch.mean, expected_batch_mean)
            self.assertAllClose(p_batch.variance, expected_batch_variance)
            # Test mixture_mean and mixture_variance
            expected_mixture_mean = batch_values[:, :-1].mean(
                dim=(0, 1)
            )  # Mean across batch and ensemble dims
            expected_mixture_var = batch_values[:, :-1].var(
                dim=(0, 1)
            )  # Variance across batch and ensemble dims
            self.assertAllClose(p_batch.mixture_mean, expected_mixture_mean)
            self.assertAllClose(p_batch.mixture_variance, expected_mixture_var)

            # Test unnormalized 2-dimensional batch weights
            weights_2d = torch.tensor([[3.0, 3.0, 0.0], [6.0, 6.0, 0.0]], **tkwargs)
            p_2d = EnsemblePosterior(batch_values, weights=weights_2d)
            expected_norm_2d = torch.tensor(
                [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], **tkwargs
            )
            self.assertAllClose(p_2d.weights, expected_norm_2d)

    def test_EnsembleModel_weighted_rsample(self):
        """Test that weighted rsample works correctly. All negative `batch_values`
        have zero weight, so all samples should be positive."""
        batch_values = torch.tensor(
            [
                [[[1.0], [2.0]], [[3.0], [6.0]], [[-5.0], [-6.0]], [[-5.0], [-16.0]]],
                [[[1.0], [2.0]], [[-3.0], [-6.0]], [[5.0], [6.0]], [[-5.0], [-16.0]]],
                [[[2.0], [4.0]], [[-4.0], [-8.0]], [[-6.0], [-7.0]], [[6.0], [17.0]]],
            ]
        )
        weights_2d = torch.tensor(
            [[3.0, 3.0, 0.0, 0.0], [3.0, 0.0, 5.0, 0.0], [6.0, 0.0, 0.0, 60.0]]
        )
        p_2d = EnsemblePosterior(batch_values, weights=weights_2d)
        samples = p_2d.rsample(torch.Size([10]))
        self.assertEqual(samples.shape, torch.Size([10, 3, 2, 1]))
        self.assertTrue((samples >= 0).all())
