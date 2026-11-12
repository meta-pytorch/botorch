#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.test_helpers import get_fully_bayesian_model
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qConditionalHyperparameterInformationGain,
    qExpectedPredictiveInformationGain,
    qHyperparameterInformedPredictiveExploration,
    qStatisticalDistanceActiveLearning,
)


class TestQStatisticalDistanceActiveLearning(BotorchTestCase):
    def test_invalid_distance_metric(self):
        """Test that invalid distance_metric raises ValueError."""
        torch.manual_seed(1)
        tkwargs = {"device": self.device, "dtype": torch.double}
        input_dim = 2

        model = get_fully_bayesian_model(
            train_X=torch.rand(4, input_dim, **tkwargs),
            train_Y=torch.rand(4, 1, **tkwargs),
            num_models=3,
            **tkwargs,
        )

        with self.assertRaises(ValueError):
            qStatisticalDistanceActiveLearning(
                model=model,
                distance_metric="invalid_metric",
            )

    def test_q_statistical_distance_active_learning(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        distance_metrics = ("hellinger", "kl_divergence")
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            distance_metric,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            distance_metrics,
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qStatisticalDistanceActiveLearning(
                    model=model,
                    X_pending=X_pending,
                    distance_metric=distance_metric,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])


class TestQConditionalHyperparameterInformationGain(BotorchTestCase):
    def test_q_conditional_hyperparameter_information_gain(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device, "dtype": torch.double}
        input_dim = 2
        num_models = 3

        model = get_fully_bayesian_model(
            train_X=torch.rand(4, input_dim, **tkwargs),
            train_Y=torch.rand(4, 1, **tkwargs),
            num_models=num_models,
            **tkwargs,
        )
        bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], **tkwargs)
        mc_points = draw_sobol_samples(bounds=bounds, n=11, q=1).squeeze(-2)

        acq = qConditionalHyperparameterInformationGain(
            model=model, mc_points=mc_points
        )
        test_X = torch.rand(7, 1, input_dim, **tkwargs)
        acq_X = acq(test_X)
        self.assertEqual(acq_X.shape, test_X.shape[:-2])
        self.assertTrue((acq_X >= 0).all())

        test_X = torch.rand(7, 3, input_dim, **tkwargs)
        acq_X = acq(test_X)
        self.assertEqual(acq_X.shape, test_X.shape[:-2])
        self.assertTrue((acq_X >= 0).all())


class TestQExpectedPredictiveInformationGain(BotorchTestCase):
    def test_q_expected_predictive_information_gain(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device, "dtype": torch.double}
        input_dim = 2

        model = get_fully_bayesian_model(
            train_X=torch.rand(0, input_dim, **tkwargs),
            train_Y=torch.rand(0, 1, **tkwargs),
            num_models=3,
            **tkwargs,
        )
        bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], **tkwargs)
        mc_points = draw_sobol_samples(bounds=bounds, n=16, q=1).squeeze(-2)

        acq = qExpectedPredictiveInformationGain(model=model, mc_points=mc_points)
        test_X = torch.rand(4, 2, input_dim, **tkwargs)
        acq_X = acq(test_X)
        self.assertEqual(acq_X.shape, test_X.shape[:-2])
        self.assertTrue((acq_X >= 0).all())

        # test that mc_points must be 2-dimensional
        with self.assertRaises(ValueError):
            qExpectedPredictiveInformationGain(
                model=model,
                mc_points=mc_points.unsqueeze(0),  # 3D tensor
            )


class TestQHyperparameterInformedPredictiveExploration(BotorchTestCase):
    def test_q_hyperparameter_informed_predictive_exploration(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            (True,),
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(0, input_dim, **tkwargs)
            train_Y = torch.rand(0, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=False,
                infer_noise=infer_noise,
                **tkwargs,
            )

            bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], **tkwargs)
            mc_points = draw_sobol_samples(bounds=bounds, n=16, q=1).squeeze(-2)

            # test with fixed beta
            acq = qHyperparameterInformedPredictiveExploration(
                model=model,
                mc_points=mc_points,
                bounds=bounds,
            )

            test_Xs = [
                torch.rand(4, 1, input_dim, **tkwargs),
                torch.rand(4, 3, input_dim, **tkwargs),
            ]

            for test_X in test_Xs:
                acq_X = acq(test_X)
                # assess shape
                self.assertTrue(acq_X.shape == test_X.shape[:-2])
                self.assertTrue((acq_X > 0).all())

            # test beta tuning (beta=None) and re-tuning when q changes
            acq = qHyperparameterInformedPredictiveExploration(
                model=model,
                mc_points=mc_points,
                bounds=bounds,
            )
            # first forward pass computes tuning factor for q=1
            acq(torch.rand(4, 1, input_dim, **tkwargs))
            tuning_factor_q1 = acq._tuning_factor
            # second forward pass with different q recomputes tuning factor
            acq(torch.rand(4, 3, input_dim, **tkwargs))
            tuning_factor_q3 = acq._tuning_factor
            self.assertNotEqual(tuning_factor_q1, tuning_factor_q3)

            # test that mc_points must be 2-dimensional
            with self.assertRaises(ValueError):
                qHyperparameterInformedPredictiveExploration(
                    model=model,
                    mc_points=mc_points.unsqueeze(0),  # 3D tensor
                    bounds=bounds,
                )

    def test_q_hyperparameter_informed_predictive_exploration_pending(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device, "dtype": torch.double}
        input_dim = 2
        model = get_fully_bayesian_model(
            train_X=torch.rand(6, input_dim, **tkwargs),
            train_Y=torch.randn(6, 1, **tkwargs),
            num_models=4,
            **tkwargs,
        )
        bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], **tkwargs)
        mc_points = draw_sobol_samples(bounds=bounds, n=16, q=1).squeeze(-2)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=0)
        X_pending = torch.rand(2, input_dim, **tkwargs)
        acq = qHyperparameterInformedPredictiveExploration(
            model=model,
            mc_points=mc_points,
            bounds=bounds,
            sampler=sampler,
            X_pending=X_pending,
            beta_tuning_samples=4,
        )
        X = torch.rand(3, 1, input_dim, **tkwargs)
        acq_X = acq(X)
        # The tuning factor is computed for the q of the candidates.
        self.assertEqual(acq._tuning_factor_q, 1)
        # The pending points are appended to X exactly once.
        X_full = torch.cat([X, X_pending.expand(3, 2, input_dim)], dim=-2)
        bald = qBayesianActiveLearningByDisagreement(model=model, sampler=sampler)
        epig = qExpectedPredictiveInformationGain(model=model, mc_points=mc_points)
        self.assertAllClose(acq_X, acq._tuning_factor * bald(X_full) + epig(X_full))


class TestQBayesianQueryByComittee(BotorchTestCase):
    def test_q_bayesian_query_by_comittee(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qBayesianQueryByComittee(
                    model=model,
                    X_pending=X_pending,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

    def test_q_bayesian_query_by_comittee_values(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device, "dtype": torch.double}
        model = get_fully_bayesian_model(
            train_X=torch.rand(6, 2, **tkwargs),
            train_Y=torch.randn(6, 1, **tkwargs),
            num_models=8,
            **tkwargs,
        )
        acq = qBayesianQueryByComittee(model=model)
        for q in (1, 3):
            X = torch.rand(5, q, 2, **tkwargs)
            # Determinant of the covariance of the posterior mean across models,
            # i.e. the variance of the posterior mean for q = 1.
            mean = model.posterior(X).mean.squeeze(-1)  # 5 x num_models x q
            mean_diff = mean - mean.mean(dim=-2, keepdim=True)
            covar_of_mean = (mean_diff.unsqueeze(-1) * mean_diff.unsqueeze(-2)).mean(
                dim=-3
            )
            expected = torch.det(covar_of_mean)
            self.assertTrue((expected > 0).all())
            self.assertAllClose(acq(X), expected, rtol=1e-6, atol=0.0)


class TestQBayesianVarianceReduction(BotorchTestCase):
    def test_q_bayesian_variance_reduction(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            standardize_model,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            (False, True),  # standardize_model
            (True,),  # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_fully_bayesian_model(
                train_X=train_X,
                train_Y=train_Y,
                num_models=num_models,
                standardize_model=standardize_model,
                infer_noise=infer_noise,
                **tkwargs,
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qBayesianVarianceReduction(
                    model=model,
                    X_pending=X_pending,
                )

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])
