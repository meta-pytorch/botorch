# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from math import log, pi

import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.test_utils.mock import mock_optimize
from botorch.utils.evaluation import AIC, BIC, compute_in_sample_model_fit_metric, MLL
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestEvaluation(BotorchTestCase):
    def test_multitask_metrics_use_task_specific_training_noise(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, dtype, labels in product(
            devices, (torch.float, torch.double), ((0, 1), (2, 5))
        ):
            with self.subTest(device=device, dtype=dtype, labels=labels):
                x = torch.linspace(0.1, 0.9, 6, dtype=dtype, device=device).unsqueeze(
                    -1
                )
                task_values = x.new_tensor([labels[i % 2] for i in range(6)]).unsqueeze(
                    -1
                )
                train_x = torch.cat((x, task_values), dim=-1)
                train_y = (x * 4).sin() + 0.2 * (task_values == labels[1])
                model = MultiTaskGP(train_x, train_y, task_feature=-1)
                model.likelihood.noise = x.new_tensor([0.025, 0.07])
                model.train()
                with torch.no_grad():
                    latent = model(*model.train_inputs)
                    # Explicit observed-index noise law: no calls to the utility
                    # or its likelihood argument construction in this oracle.
                    noise = model.likelihood.noise.repeat(3)
                    expected = (
                        torch.distributions.MultivariateNormal(
                            latent.mean,
                            covariance_matrix=latent.covariance_matrix
                            + torch.diag_embed(noise),
                        )
                        .log_prob(model.train_targets)
                        .item()
                    )
                num_params = sum(parameter.numel() for parameter in model.parameters())
                values = {
                    MLL: expected,
                    AIC: 2 * num_params - 2 * expected,
                    BIC: log(6) * num_params - 2 * expected,
                }
                tolerance = 2e-4 if dtype == torch.float else 1e-9
                for criterion, expected_value in values.items():
                    actual = compute_in_sample_model_fit_metric(model, criterion)
                    self.assertAlmostEqual(actual, expected_value, delta=tolerance)
                    self.assertFalse(model.training)

    @mock_optimize
    def test_compute_in_sample_model_fit_metric(self):
        torch.manual_seed(0)
        for dtype, model_cls in product(
            (torch.float, torch.double), (SingleTaskGP, SaasFullyBayesianSingleTaskGP)
        ):
            train_X = torch.linspace(
                0, 1, 10, dtype=dtype, device=self.device
            ).unsqueeze(-1)
            train_Y = torch.sin(2 * pi * train_X)
            model = model_cls(train_X=train_X, train_Y=train_Y)
            if model_cls is SingleTaskGP:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            else:
                fit_fully_bayesian_model_nuts(
                    model,
                    warmup_steps=8,
                    num_samples=6,
                    thinning=2,
                    disable_progbar=True,
                )
            num_params = sum(p.numel() for p in model.parameters())
            if model_cls is SaasFullyBayesianSingleTaskGP:
                num_params /= 3  # divide by number of MCMC samples
            mll = compute_in_sample_model_fit_metric(model=model, criterion=MLL)
            aic = compute_in_sample_model_fit_metric(model=model, criterion=AIC)
            bic = compute_in_sample_model_fit_metric(model=model, criterion=BIC)
            self.assertEqual(aic, 2 * num_params - 2 * mll)
            self.assertEqual(bic, log(10) * num_params - 2 * mll)
        # test invalid criterion
        with self.assertRaisesRegex(
            ValueError, "Invalid evaluation criterion invalid."
        ):
            compute_in_sample_model_fit_metric(model=model, criterion="invalid")
