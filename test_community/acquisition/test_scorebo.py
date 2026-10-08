#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.utils.test_helpers import _get_mcmc_samples, get_fully_bayesian_model
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization
from scipy.stats import truncnorm


class TestQSelfCorrectingBayesianOptimization(BotorchTestCase):
    def test_q_self_correcting_bayesian_optimization(self):
        torch.manual_seed(5)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            distance_metric,
            only_maxval,
            standardize_model,
        ) in [
            (torch.float, "hellinger", True, True),
            (torch.double, "hellinger", False, False),
            (torch.float, "kl_divergence", True, True),
            (torch.double, "kl_divergence", False, False),
            (torch.double, "kl_divergence", True, False),
        ]:
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(5, input_dim, **tkwargs)
            train_Y = torch.rand(5, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=True,
                **tkwargs,
            )

            num_optimal_samples = 7
            optimal_inputs = torch.rand(
                num_optimal_samples, num_models, input_dim, **tkwargs
            )

            # SCoreBO can work with only max-value, so we're testing that too
            if only_maxval:
                optimal_inputs = None
            optimal_outputs = torch.rand(
                num_optimal_samples, num_models, num_objectives, **tkwargs
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qSelfCorrectingBayesianOptimization(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    distance_metric=distance_metric,
                    X_pending=X_pending,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        acq = qSelfCorrectingBayesianOptimization(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=-torch.ones(1, **tkwargs)
            ),
        )
        self.assertTrue(torch.all(acq.optimal_output_values == -acq.optimal_outputs))
        acq_X = acq(test_Xs[j])
        self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qSelfCorrectingBayesianOptimization(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                distance_metric="NOT_A_DISTANCE",
                X_pending=X_pending,
            )

    def test_optimal_inputs_input_transform(self):
        # The optimal inputs are in the raw input space and must be transformed
        # (only) once when conditioning on them.
        tkwargs = {"device": self.device, "dtype": torch.double}
        num_models, d = 3, 2
        bounds = torch.tensor([[0.0] * d, [10.0] * d], **tkwargs)
        train_X = 10 * torch.rand(8, d, **tkwargs)
        model = SaasFullyBayesianSingleTaskGP(
            train_X,
            torch.sin(train_X.sum(dim=-1, keepdim=True)),
            input_transform=Normalize(d=d, bounds=bounds),
        )
        model.load_mcmc_samples(
            _get_mcmc_samples(num_models, d, infer_noise=True, **tkwargs)
        )
        model.eval()
        optimal_inputs = torch.full((1, num_models, d), 5.0, **tkwargs)
        optimal_outputs = torch.full((1, num_models, 1), 3.0, **tkwargs)
        acq = qSelfCorrectingBayesianOptimization(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
        )
        cond_train_X = acq.conditional_model.train_inputs[0]
        self.assertAllClose(
            cond_train_X[..., -1, :], torch.full_like(cond_train_X[..., -1, :], 0.5)
        )
        # The conditional models are (noiselessly) conditioned at the optimum.
        posterior = acq.conditional_model.posterior(
            optimal_inputs[0, 0].view(1, 1, d), observation_noise=False
        )
        self.assertTrue((posterior.variance < 1e-3).all())

    def test_truncated_moments(self):
        # The moments of the posterior truncated at the max-value must be accurate,
        # also far in the tail (where the normal cdf underflows).
        tkwargs = {"device": self.device, "dtype": torch.double}
        num_models, d = 3, 2
        train_X = torch.rand(8, d, **tkwargs)
        model = get_fully_bayesian_model(
            train_X=train_X,
            train_Y=torch.sin(6 * train_X.sum(dim=-1, keepdim=True)),
            num_models=num_models,
            **tkwargs,
        )
        X = torch.rand(1, 1, d, **tkwargs)
        with torch.no_grad():
            posterior = model.posterior(X, observation_noise=False)
            noise = model.posterior(X, observation_noise=True).variance.view(-1)
        mean = posterior.mean.view(-1)
        std = posterior.variance.sqrt().view(-1)
        noise = noise - std.square()
        for beta in (-2.0, -5.0, -10.0):
            acq = qSelfCorrectingBayesianOptimization(
                model=model, optimal_outputs=(mean + beta * std).view(1, -1, 1)
            )
            with mock.patch.object(acq, "distance", wraps=acq.distance) as distance:
                acq(X)
            trunc_mean, _, trunc_covar, _ = distance.call_args.args
            loc, scale = mean.cpu().numpy(), std.cpu().numpy()
            ref_mean = torch.as_tensor(
                truncnorm.mean(-np.inf, beta, loc=loc, scale=scale)
            )
            ref_var = torch.as_tensor(
                truncnorm.var(-np.inf, beta, loc=loc, scale=scale)
            )
            self.assertAllClose(trunc_mean.view(-1), ref_mean.to(X))
            self.assertAllClose(trunc_covar.view(-1), ref_var.to(X) + noise)
