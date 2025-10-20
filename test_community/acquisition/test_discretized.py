#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from botorch.acquisition.objective import (
    ExpectationPosteriorTransform,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.discretized import (
    DiscretizedExpectedImprovement,
    DiscretizedProbabilityOfImprovement,
)
from botorch_community.posteriors.riemann import BoundedRiemannPosterior
from torch import Tensor


class MockDiscretizedModel:
    def __init__(self, borders, probabilities):
        """Mock model for testing discretized acquisition functions.

        This class simulates a model that returns a BoundedRiemannPosterior with
        predefined borders and probabilities, regardless of the input X.

        Attributes:
            borders: A tensor representing the boundaries of the discretized bins.
            probabilities: A tensor representing the probability mass for each bin.
        """

        self.borders = borders
        self.probabilities = probabilities

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> BoundedRiemannPosterior:
        return BoundedRiemannPosterior(self.borders, self.probabilities)


class TestDiscretizedExpectedImprovement(BotorchTestCase):
    def test_ag_integrate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            acqf = DiscretizedExpectedImprovement(None, best_f=2.0)

            # best_f <= lower
            lower_bound = torch.tensor(1.0, **tkwargs)
            upper_bound = torch.tensor(1.5, **tkwargs)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, 0.0)

            # lower < best_f <= upper
            lower_bound = torch.tensor(1.0)
            upper_bound = torch.tensor(3.0)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, (2.0 + 3.0) / 2 - 2.0)

            # upper <= best_f
            lower_bound = torch.tensor(3.0)
            upper_bound = torch.tensor(5.0)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, ((5.0 + 3.0) / 2 - 2.0) * 2.0)

    def test_ag_integrate_batch(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            acqf = DiscretizedExpectedImprovement(
                None, best_f=torch.tensor([[2.0], [3.0]], **tkwargs)
            )

            lower_bound = torch.tensor([[1.0], [2.0]], **tkwargs)
            upper_bound = torch.tensor([[1.5], [6.0]], **tkwargs)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement[0], 0.0)
            self.assertEqual(improvement[1], 3.0**2 / 2)

    def test_forward(self):
        for dtype in (torch.double, torch.float):
            tkwargs = {"device": self.device, "dtype": dtype}

            borders = torch.tensor([0.0, 1.0, 3.0, 6.0], **tkwargs)
            probs = torch.tensor([0.2, 0.3, 0.5], **tkwargs)
            mock_model = MockDiscretizedModel(borders, probs)

            acqf = DiscretizedExpectedImprovement(mock_model, best_f=2.0)
            # value will not be used since posterior is mocked
            X = torch.tensor([[1.0]])
            acqf_values = acqf(X)

            # EI is 0 for bucket 1, (1.0**2)/2 = 0.5 for bucket 2,
            # and 2.5*3. = 7.5 for bucket 3
            # so the expected value is
            # 0.2 / 1. * 0 + 0.3 / 2. * 0.5 + 0.5 / 3. * 7.5 = 1.325
            self.assertAlmostEqual(acqf_values.item(), 1.325)

    def test_forward_approximating_normal(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        borders = torch.linspace(-8, 8, 10_000, **tkwargs)

        mean = 1.1
        std = 1.3
        target_dist = torch.distributions.Normal(mean, std)
        probs = target_dist.cdf(borders[1:]) - target_dist.cdf(borders[:-1])

        mock_model = MockDiscretizedModel(borders, probs)
        # Test that the discretized EI approximates the analytical EI
        # for a normal distribution
        best_f = 0.5
        acqf = DiscretizedExpectedImprovement(mock_model, best_f=best_f)
        X = torch.tensor([[1.0]])  # value will not be used since posterior is mocked
        acqf_values = acqf(X)

        # Calculate analytical EI for normal distribution
        improvement = mean - best_f
        z = torch.tensor(improvement / std)
        normal_dist = torch.distributions.Normal(0, 1)
        analytical_ei = (
            improvement * normal_dist.cdf(z) + std * normal_dist.log_prob(z).exp()
        )

        # Check that the discretized EI is close to the analytical EI
        self.assertLess(torch.abs(acqf_values - analytical_ei) / analytical_ei, 0.01)

        # minimization task
        acqf = DiscretizedExpectedImprovement(
            mock_model,
            best_f=-best_f,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0])
            ),
        )
        acqf_values = acqf(X)

        # Calculate analytical EI for normal distribution
        improvement = -mean + best_f
        z = torch.tensor(improvement / std)
        normal_dist = torch.distributions.Normal(0, 1)
        analytical_ei = (
            improvement * normal_dist.cdf(z) + std * normal_dist.log_prob(z).exp()
        )

        # Check that the discretized EI is close to the analytical EI
        self.assertLess(torch.abs(acqf_values - analytical_ei) / analytical_ei, 0.01)

    def test_unsupported_posterior_transform(self):
        """Test that UnsupportedError is raised for non-ScalarizedPosteriorTransform."""
        mock_model = MockDiscretizedModel(
            torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.5, 0.5])
        )

        # Test with ExpectationPosteriorTransform
        expectation_transform = ExpectationPosteriorTransform(n_w=2)
        with self.assertRaises(UnsupportedError):
            DiscretizedExpectedImprovement(
                mock_model, best_f=1.0, posterior_transform=expectation_transform
            )


class TestDiscretizedProbabilityofImprovement(BotorchTestCase):
    def test_ag_integrate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            acqf = DiscretizedProbabilityOfImprovement(None, best_f=2.0)

            # best_f <= lower
            lower_bound = torch.tensor(1.0, **tkwargs)
            upper_bound = torch.tensor(1.5, **tkwargs)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, 0.0)

            # lower < best_f <= upper
            lower_bound = torch.tensor(1.0)
            upper_bound = torch.tensor(3.0)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, 1.0)

            # upper <= best_f
            lower_bound = torch.tensor(3.0)
            upper_bound = torch.tensor(5.0)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement, 2.0)

    def test_ag_integrate_batch(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            acqf = DiscretizedProbabilityOfImprovement(
                None, best_f=torch.tensor([[2.0], [3.0]], **tkwargs)
            )

            lower_bound = torch.tensor([[1.0], [2.0]], **tkwargs)
            upper_bound = torch.tensor([[1.5], [6.0]], **tkwargs)
            improvement = acqf.ag_integrate(lower_bound, upper_bound)
            self.assertEqual(improvement[0], 0.0)
            self.assertEqual(improvement[1], 3.0)

    def test_forward(self):
        borders = torch.tensor([0.0, 1.0, 3.0, 6.0])
        probs = torch.tensor([0.2, 0.3, 0.5])
        mock_model = MockDiscretizedModel(borders, probs)

        acqf = DiscretizedProbabilityOfImprovement(mock_model, best_f=2.0)
        X = torch.tensor([[1.0]])  # value will not be used since posterior is mocked
        acqf_values = acqf(X)

        # PI is 0 for bucket 1, 0.5 for bucket 2, and 1 for bucket 3
        # so the final probability is 0.2 * 0 + 0.3 * 0.5 + 0.5 * 1 = 1.4
        self.assertAlmostEqual(acqf_values.item(), 0.65)

    def test_forward_approximating_normal(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        borders = torch.linspace(-8, 8, 10_000, **tkwargs)

        mean = 1.1
        std = 1.3
        target_dist = torch.distributions.Normal(mean, std)
        probs = target_dist.cdf(borders[1:]) - target_dist.cdf(borders[:-1])

        mock_model = MockDiscretizedModel(borders, probs)

        # Compute PI
        best_f = 0.5
        acqf = DiscretizedProbabilityOfImprovement(mock_model, best_f=best_f)
        X = torch.tensor([[1.0]])  # value will not be used since posterior is mocked
        acqf_values = acqf(X)

        # Calculate analytical PI for normal distribution
        prob_improvement = 1.0 - target_dist.cdf(torch.tensor(best_f))

        # Check that the discretized EI is close to the analytical EI
        self.assertLess(
            torch.abs(acqf_values - prob_improvement) / prob_improvement, 0.01
        )

        # minimization task
        acqf = DiscretizedProbabilityOfImprovement(
            mock_model,
            best_f=-best_f,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=torch.tensor([-1.0])
            ),
        )
        acqf_values = acqf(X)

        # Calculate analytical PI for normal distribution
        prob_improvement = target_dist.cdf(torch.tensor(best_f))

        # Check that the discretized EI is close to the analytical EI
        self.assertLess(
            torch.abs(acqf_values - prob_improvement) / prob_improvement, 0.01
        )
