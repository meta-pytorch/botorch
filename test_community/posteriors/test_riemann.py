#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.posteriors.riemann import (
    BoundedRiemannPosterior,
    MultivariateRiemannPosterior,
)


class TestRiemannPosterior(BotorchTestCase):
    def test_properties(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            borders = torch.linspace(0, 1, 6, **tkwargs)
            probabilities = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.4], **tkwargs)
            posterior = BoundedRiemannPosterior(borders, probabilities)
            self.assertTrue(torch.equal(posterior.borders, borders))
            self.assertTrue(torch.equal(posterior.probabilities, probabilities))
            self.assertEqual(posterior.dtype, dtype)
            self.assertEqual(posterior.device, self.device)

    def test_integrate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            borders = torch.tensor([0.0, 0.5, 1.5], **tkwargs)
            probabilities = torch.tensor([0.2, 0.8], **tkwargs)
            posterior = BoundedRiemannPosterior(borders, probabilities)

            def ag_integrate(lower, upper):
                return (upper + lower) / 2

            result = posterior.integrate(ag_integrate)
            expected_result = (0.0 + 0.5) / 2 * 0.2 / 0.5 + (1.5 + 0.5) / 2 * 0.8 / 1.0
            self.assertLess((result - expected_result).abs(), 1e-5)

    def test_rsample(self):
        torch.manual_seed(13)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # Create Riemann Posterior based on a normal distribution
            dist_mean = torch.tensor(3.0, **tkwargs)
            dist_std = torch.tensor(2.0, **tkwargs)
            test_distribution = torch.distributions.Normal(dist_mean, dist_std)
            test_samples = test_distribution.rsample(torch.Size([10000]))

            n_buckets = 5000
            borders = torch.quantile(
                test_samples, torch.linspace(0, 1, n_buckets + 1, **tkwargs)
            )
            probabilities = torch.ones(n_buckets, **tkwargs) / n_buckets
            posterior = BoundedRiemannPosterior(borders, probabilities)

            # Check that the mean and variance of the samples are correct
            samples = posterior.rsample(torch.Size([5000000]))
            self.assertLess((samples.mean(dim=0) - dist_mean).abs().item(), 0.02)
            self.assertLess((samples.std(dim=0) - dist_std).abs().item(), 0.02)

            # Create Riemann Posterior based on a Gamma distribution
            dist_alpha = torch.tensor(2.0, **tkwargs)
            dist_beta = torch.tensor(4.5, **tkwargs)
            test_distribution = torch.distributions.Gamma(dist_alpha, dist_beta)
            test_samples = test_distribution.rsample(torch.Size([10000]))
            n_buckets = 1000
            borders = torch.quantile(
                test_samples, torch.linspace(0, 1, n_buckets + 1, **tkwargs)
            )
            probabilities = torch.ones(n_buckets, **tkwargs) / n_buckets
            posterior = BoundedRiemannPosterior(borders, probabilities)

            # Check that the mean and variance of the samples are correct
            samples = posterior.rsample(torch.Size([10000]))
            self.assertLess(
                (samples.mean(dim=0) - test_distribution.mean).abs().item(), 0.02
            )
            self.assertLess(
                (samples.std(dim=0).pow(2) - test_distribution.variance).abs().item(),
                0.02,
            )

    def test_rsample_from_base_samples(self):
        torch.manual_seed(13)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            borders = torch.linspace(-10, 10, 26, **tkwargs)
            probabilities = torch.rand(3, 25, **tkwargs)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            posterior = BoundedRiemannPosterior(
                borders,
                probabilities.expand(
                    12, 3, 25
                ),  # Two batch dimensions, with shapes 12 and 3
            )

            base_samples = torch.rand(30, 12, 3)  # 30 samples
            samples = posterior.rsample_from_base_samples(
                torch.Size([30, 12, 3]), base_samples
            )
            self.assertEqual(samples.shape, torch.Size([30, 12, 3, 1]))

            with self.assertRaises(ValueError):
                posterior.rsample_from_base_samples(torch.Size([10, 4]), base_samples)

    def test_mean(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            borders = torch.linspace(0, 1, 6, **tkwargs)
            probabilities = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.4], **tkwargs)
            posterior = BoundedRiemannPosterior(borders, probabilities)
            mean = posterior.mean
            expected_mean = ((borders[1:] + borders[:-1]) / 2 * probabilities).sum(-1)
            self.assertLess((mean - expected_mean).abs().sum().item(), 1e-5)

    def test_variance(self):
        torch.manual_seed(13)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # Create Riemann Posterior based on a normal distribution with non-zero mean
            dist_mean = torch.tensor(2.5, **tkwargs)
            dist_std = torch.tensor(1.5, **tkwargs)
            test_distribution = torch.distributions.Normal(dist_mean, dist_std)
            test_samples = test_distribution.rsample(torch.Size([100000]))
            n_buckets = 5000
            borders = torch.quantile(
                test_samples, torch.linspace(0, 1, n_buckets + 1, **tkwargs)
            )
            probabilities = torch.ones(n_buckets, **tkwargs) / n_buckets
            posterior = BoundedRiemannPosterior(borders, probabilities)

            # Check that the variance approximately matches the true variance
            true_variance = test_distribution.variance
            computed_variance = posterior.variance
            self.assertLess((computed_variance - true_variance).abs().item(), 0.05)

            # Test with a different distribution (non-zero mean, different variance)
            dist_mean = torch.tensor(-1.0, **tkwargs)
            dist_std = torch.tensor(0.8, **tkwargs)
            test_distribution = torch.distributions.Normal(dist_mean, dist_std)
            test_samples = test_distribution.rsample(torch.Size([100000]))
            n_buckets = 3000
            borders = torch.quantile(
                test_samples, torch.linspace(0, 1, n_buckets + 1, **tkwargs)
            )
            probabilities = torch.ones(n_buckets, **tkwargs) / n_buckets
            posterior = BoundedRiemannPosterior(borders, probabilities)

            # Check that the variance approximately matches the true variance
            true_variance = test_distribution.variance
            computed_variance = posterior.variance
            self.assertLess((computed_variance - true_variance).abs().item(), 0.05)

            # Check with batch dimension
            probabilities = torch.rand(2, n_buckets, **tkwargs)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            posterior = BoundedRiemannPosterior(borders, probabilities)
            self.assertEqual(posterior.variance.shape, torch.Size([2, 1]))
            self.assertEqual(posterior.mean.shape, torch.Size([2, 1]))

    def test_confidence_region(self):
        torch.manual_seed(13)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # Create Riemann Posterior based on a normal distribution
            dist_mean = torch.tensor(0.0, **tkwargs)
            dist_std = torch.tensor(1.0, **tkwargs)
            test_distribution = torch.distributions.Normal(dist_mean, dist_std)
            test_samples = test_distribution.rsample(torch.Size([100000]))
            n_buckets = 5000
            borders = torch.quantile(
                test_samples, torch.linspace(0, 1, n_buckets + 1, **tkwargs)
            )
            probabilities = torch.ones(n_buckets, **tkwargs) / n_buckets
            probabilities = probabilities.unsqueeze(0)
            posterior = BoundedRiemannPosterior(borders, probabilities)

            lower, upper = posterior.confidence_region(confidence_level=0.954)
            self.assertLess((lower - (dist_mean - 2 * dist_std)).abs().item(), 0.02)
            self.assertLess((upper - (dist_mean + 2 * dist_std)).abs().item(), 0.02)

    def test_icdf(self):
        torch.manual_seed(13)
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            probabilities = torch.tensor([0.5, 0.5], **tkwargs)
            probabilities = probabilities.unsqueeze(0).unsqueeze(0)
            posterior = BoundedRiemannPosterior(
                borders=torch.tensor([0.0, 1.0, 2.0], **tkwargs),
                probabilities=probabilities,
            )
            value = torch.tensor([0.6], **tkwargs)
            value = value.unsqueeze(0).unsqueeze(0)
            true_res = torch.tensor([1.2], **tkwargs)
            self.assertTrue(
                torch.allclose(posterior.icdf(value=value).squeeze(), true_res)
            )
            # And batch dimensions (5, 12)
            probabilities = probabilities.expand(torch.Size([5, 12, 2]))
            value = value.expand(torch.Size([30, 5, 12]))
            posterior = BoundedRiemannPosterior(
                borders=torch.tensor([0.0, 1.0, 2.0], **tkwargs),
                probabilities=probabilities,
            )
            res = posterior.icdf(value=value)
            self.assertEqual(res.shape, torch.Size([30, 5, 12, 1]))
            true_res = true_res.expand(30, 5, 12, 1)
            self.assertTrue(torch.allclose(res, true_res))
            # Test with float
            value = 0.6
            true_res = torch.tensor([1.2], **tkwargs)
            self.assertTrue(
                torch.allclose(posterior.icdf(value=value).squeeze(), true_res)
            )


class TestMultivariateRiemannPosterior(BotorchTestCase):
    def test_multivariate_rsample(self):
        borders = torch.linspace(0, 1, 101)
        probabilities = torch.ones(2, 3, 100)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        correlation_matrix = torch.cat(
            [
                torch.eye(3, 3).unsqueeze(0),
                torch.tensor(
                    [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ).unsqueeze(0),
            ],
            dim=0,
        )
        posterior = MultivariateRiemannPosterior(
            borders=borders,
            probabilities=probabilities,
            correlation_matrix=correlation_matrix,
        )
        base_samples = torch.rand(5000, 2, 3)
        samples = posterior.rsample_from_base_samples(
            torch.Size([5000, 2, 3]), base_samples
        )
        self.assertEqual(samples.shape, torch.Size([5000, 2, 3, 1]))
        # Uncorrelated samples
        self.assertTrue(
            torch.abs(torch.corrcoef(samples[:, 0, :].squeeze().T)[0, 1]).item() < 0.2
        )
        # Correlated samples
        self.assertTrue(
            torch.abs(torch.corrcoef(samples[:, 1, :].squeeze().T)[0, 1]).item() > 0.8
        )
