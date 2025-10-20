#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.discretized import (
    DiscretizedExpectedImprovement,
    DiscretizedProbabilityOfImprovement,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization


class InputConstructorBaseTestCase(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.mock_model = MockModel(
            posterior=MockPosterior(mean=None, variance=None, base_shape=(1,))
        )

        X1 = torch.rand(3, 2)
        X2 = torch.rand(3, 2)
        Y1 = torch.rand(3, 1)
        Y2 = torch.rand(3, 1)
        feature_names = ["X1", "X2"]
        outcome_names = ["Y"]

        self.blockX_blockY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            )
        }
        self.blockX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X1, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.multiX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X2, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.bounds = 2 * [(0.0, 1.0)]


class TestAnalyticalAcquisitionFunctionInputConstructors(InputConstructorBaseTestCase):
    def test_construct_inputs_best_f(self) -> None:
        for acqf_cls in [
            DiscretizedProbabilityOfImprovement,
            DiscretizedExpectedImprovement,
        ]:
            with self.subTest(acqf_cls=acqf_cls):
                c = get_acqf_input_constructor(acqf_cls)
                mock_model = self.mock_model
                kwargs = c(model=mock_model, training_data=self.blockX_blockY)
                best_f_expected = self.blockX_blockY[0].Y.squeeze().max()
                self.assertIs(kwargs["model"], mock_model)
                self.assertIsNone(kwargs["posterior_transform"])
                self.assertEqual(kwargs["best_f"], best_f_expected)
                acqf = acqf_cls(**kwargs)
                self.assertIs(acqf.model, mock_model)

                kwargs = c(
                    model=mock_model, training_data=self.blockX_blockY, best_f=0.1
                )
                self.assertIs(kwargs["model"], mock_model)
                self.assertIsNone(kwargs["posterior_transform"])
                self.assertEqual(kwargs["best_f"], 0.1)
                acqf = acqf_cls(**kwargs)
                self.assertIs(acqf.model, mock_model)


class TestFullyBayesianAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase
):
    def test_construct_inputs_scorebo(self) -> None:
        func = get_acqf_input_constructor(qSelfCorrectingBayesianOptimization)
        # num_ensemble controls the ensemble size of the SAAS model
        # num_optima controls the number of Thompson samples used to infer the
        # distribution of optima
        num_ensemble, num_optima = 4, 7
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_ensemble,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_ensemble, dtype=torch.double),
                "mean": torch.randn(num_ensemble, dtype=torch.double),
                "noise": torch.rand(num_ensemble, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
            num_optima=num_optima,
            distance_metric="kl_divergence",
        )
        optimal_inputs = kwargs["optimal_inputs"]
        optimal_outputs = kwargs["optimal_outputs"]
        self.assertEqual(self.blockX_blockY[0].X.dtype, optimal_inputs.dtype)
        d = self.blockX_blockY[0].X.shape[-1]
        self.assertEqual(optimal_inputs.shape, (num_optima, num_ensemble, d))
        self.assertEqual(optimal_outputs.shape, (num_optima, num_ensemble, 1))

        # asserting that, for the non-batch case, the optimal inputs are
        # of shape num_models x N x D and outputs are num_models x N x 1
        self.assertEqual(kwargs["distance_metric"], "kl_divergence")
        qSelfCorrectingBayesianOptimization(**kwargs)

    def test_construct_inputs_sal(self) -> None:
        func = get_acqf_input_constructor(qStatisticalDistanceActiveLearning)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
            distance_metric="kl_divergence",
        )

        self.assertEqual(kwargs["distance_metric"], "kl_divergence")
        qStatisticalDistanceActiveLearning(**kwargs)

    def test_construct_inputs_bqbc(self) -> None:
        func = get_acqf_input_constructor(qBayesianQueryByComittee)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
        )

        qBayesianQueryByComittee(**kwargs)

    def test_construct_inputs_bayesian_variance_reduction(self) -> None:
        func = get_acqf_input_constructor(qBayesianVarianceReduction)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
        )

        qBayesianVarianceReduction(**kwargs)
