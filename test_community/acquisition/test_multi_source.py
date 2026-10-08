#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.augmented_multisource import (
    AugmentedUpperConfidenceBound,
)
from botorch_community.models.gp_regression_multisource import SingleTaskAugmentedGP


class TestAugmentedUpperConfidenceBound(BotorchTestCase):
    def _get_mock_agp(self, batch_shape, dtype):
        train_X = torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=self.device)
        train_Y = torch.tensor([[5.0], [0.5]], dtype=dtype, device=self.device)
        rep_shape = batch_shape + torch.Size([1, 1])
        train_X = train_X.repeat(rep_shape)
        train_Y = train_Y.repeat(rep_shape)
        covar_module = get_matern_kernel_with_gamma_prior(
            ard_num_dims=train_X.shape[-1] - 1,
        )
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "covar_module": covar_module,
            "likelihood": get_gaussian_likelihood_with_gamma_prior(),
        }
        model = SingleTaskAugmentedGP(**model_kwargs)

        return model

    def _get_expected_aucb(self, model, X, best_f, beta, cost, maximize=True):
        r"""Evaluate the AUCB formula point by point, for ``batch_shape x 1 x d`` X:
        ``(mu(x) + sqrt(beta) * sigma(x) - y^+) / (c(s) (1 + |mu(x) - mu_s(x)|))``,
        with numerator ``y^+ - (mu(x) - sqrt(beta) * sigma(x))`` for minimization.
        """
        values = []
        with torch.no_grad():
            for x in X.reshape(-1, X.shape[-1]):
                s = int(x[-1].round().item())
                x = x[:-1].unsqueeze(0)
                posterior = model.posterior(x)
                mean = posterior.mean.item()
                sigma = posterior.variance.sqrt().item()
                mean_s = model.models[s].posterior(x).mean.item()
                improvement = mean - best_f if maximize else best_f - mean
                values.append(
                    (improvement + beta**0.5 * sigma)
                    / (cost[s] * (1 + abs(mean - mean_s)))
                )
        return torch.tensor(values, device=X.device, dtype=X.dtype).view(X.shape[:-2])

    def test_upper_confidence_bound(self):
        for dtype in (torch.float, torch.double):
            mm = self._get_mock_agp(torch.Size([]), dtype)
            cost = {0: 0.5, 1: 1}
            module = AugmentedUpperConfidenceBound(
                model=mm,
                beta=1.0,
                best_f=torch.tensor(5.0, device=self.device, dtype=dtype),
                cost=cost,
            )
            X = torch.tensor([[0, 1]], device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = self._get_expected_aucb(
                mm, X.unsqueeze(0), best_f=5.0, beta=1.0, cost=cost
            )
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)

            module = AugmentedUpperConfidenceBound(
                model=mm,
                beta=1.0,
                maximize=False,
                best_f=torch.tensor(0.5, device=self.device, dtype=dtype),
                cost=cost,
            )
            X = torch.tensor([[0, 1]], device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = self._get_expected_aucb(
                mm, X.unsqueeze(0), best_f=0.5, beta=1.0, cost=cost, maximize=False
            )
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)

            # check for proper error if not multi-source model
            mean = torch.rand(1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(1, 1, device=self.device, dtype=dtype)
            mm1 = MockModel(MockPosterior(mean=mean, variance=variance))
            with self.assertRaises(UnsupportedError):
                AugmentedUpperConfidenceBound(
                    model=mm1,
                    beta=1.0,
                    best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                    cost={0: 0.5, 1: 1.0},
                )
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            mm2.models = []
            with self.assertRaises(UnsupportedError):
                AugmentedUpperConfidenceBound(
                    model=mm2,
                    beta=1.0,
                    best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                    cost={0: 0.5, 1: 1.0},
                )

    def test_upper_confidence_bound_batch(self):
        for dtype in (torch.float, torch.double):
            mm = self._get_mock_agp(torch.Size([]), dtype)
            cost = {0: 0.5, 1: 1.0}
            module = AugmentedUpperConfidenceBound(
                model=mm,
                beta=1.0,
                best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                cost=cost,
            )
            # ``b1 x b2 x 1 x d`` candidates from both sources
            X = torch.rand(2, 3, 1, 2, device=self.device, dtype=dtype)
            X[..., -1] = torch.tensor([0.0, 1.0, 1.0], device=self.device).view(3, 1)
            ucb = module(X)
            self.assertEqual(ucb.shape, torch.Size([2, 3]))
            ucb_expected = self._get_expected_aucb(
                mm, X, best_f=1.0, beta=1.0, cost=cost
            )
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)

            # check for proper error if not multi-source model
            mean = torch.rand(3, 1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(3, 1, 1, device=self.device, dtype=dtype)
            mm1 = MockModel(MockPosterior(mean=mean, variance=variance))
            with self.assertRaises(UnsupportedError):
                AugmentedUpperConfidenceBound(
                    model=mm1,
                    beta=1.0,
                    best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                    cost={0: 1, 1: 0.5},
                )
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            mm2.models = []
            with self.assertRaises(UnsupportedError):
                AugmentedUpperConfidenceBound(
                    model=mm2,
                    beta=1.0,
                    best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                    cost={0: 1, 1: 0.5},
                )

    def test_get_mean_and_sigma(self):
        for dtype in (torch.float, torch.double):
            # Test with overall model
            mean = torch.rand(1, 1, device=self.device, dtype=dtype)
            variance = torch.rand(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            mm.models = []
            module = AugmentedUpperConfidenceBound(
                model=mm,
                beta=1.0,
                best_f=torch.tensor(1.0, device=self.device, dtype=dtype),
                cost={0: 1, 1: 0.5},
            )
            X = torch.zeros(1, 2, device=self.device, dtype=dtype)
            mm_mean, mm_sigma = module._mean_and_sigma(X)
            self.assertAllClose(mm_mean, mean.squeeze(-1).squeeze(-1), atol=1e-4)
            self.assertAllClose(
                torch.pow(mm_sigma, 2), variance.squeeze(-1).squeeze(-1), atol=1e-4
            )
            _, mm_sigma = module._mean_and_sigma(X, compute_sigma=False)
            self.assertIsNone(mm_sigma)
            # Test with specific model
            mean2 = torch.rand(1, 1, device=self.device, dtype=dtype)
            variance2 = torch.rand(1, 1, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            X = torch.zeros(1, 2, device=self.device, dtype=dtype)
            mm_mean, mm_sigma = module._mean_and_sigma(X, mm2)
            self.assertAllClose(mm_mean, mean2.squeeze(-1).squeeze(-1), atol=1e-4)
            self.assertAllClose(
                torch.pow(mm_sigma, 2), variance2.squeeze(-1).squeeze(-1), atol=1e-4
            )
