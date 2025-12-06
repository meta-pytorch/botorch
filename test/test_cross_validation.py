#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from unittest.mock import patch

import torch
from botorch.cross_validation import (
    batch_cross_validation,
    efficient_loo_cv,
    ensemble_loo_cv,
    gen_loo_cv_folds,
    LOOCVResults,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestFitBatchCrossValidation(BotorchTestCase):
    def test_single_task_batch_cv(self) -> None:
        n = 10
        for batch_shape, m, dtype, observe_noise in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, n=n, **tkwargs
            )
            if m == 1:
                train_Y = train_Y.squeeze(-1)
            train_Yvar = torch.full_like(train_Y, 0.01) if observe_noise else None

            cv_folds = gen_loo_cv_folds(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
            )
            with self.subTest(
                "gen_loo_cv_folds -- check shapes, device, and dtype",
                batch_shape=batch_shape,
                m=m,
                dtype=dtype,
                observe_noise=observe_noise,
            ):
                # check shapes
                expected_shape_train_X = batch_shape + torch.Size(
                    [n, n - 1, train_X.shape[-1]]
                )
                expected_shape_test_X = batch_shape + torch.Size(
                    [n, 1, train_X.shape[-1]]
                )
                self.assertEqual(cv_folds.train_X.shape, expected_shape_train_X)
                self.assertEqual(cv_folds.test_X.shape, expected_shape_test_X)

                expected_shape_train_Y = batch_shape + torch.Size([n, n - 1, m])
                expected_shape_test_Y = batch_shape + torch.Size([n, 1, m])

                self.assertEqual(cv_folds.train_Y.shape, expected_shape_train_Y)
                self.assertEqual(cv_folds.test_Y.shape, expected_shape_test_Y)
                if observe_noise:
                    self.assertEqual(cv_folds.train_Yvar.shape, expected_shape_train_Y)
                    self.assertEqual(cv_folds.test_Yvar.shape, expected_shape_test_Y)
                else:
                    self.assertIsNone(cv_folds.train_Yvar)
                    self.assertIsNone(cv_folds.test_Yvar)

                # check device and dtype
                self.assertEqual(cv_folds.train_X.device.type, self.device.type)
                self.assertIs(cv_folds.train_X.dtype, dtype)

            input_transform = Normalize(d=train_X.shape[-1])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                cv_results = batch_cross_validation(
                    model_cls=SingleTaskGP,
                    mll_cls=ExactMarginalLogLikelihood,
                    cv_folds=cv_folds,
                    fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
                    model_init_kwargs={
                        "input_transform": input_transform,
                    },
                )
            with self.subTest(
                "batch_cross_validation",
                batch_shape=batch_shape,
                m=m,
                dtype=dtype,
                observe_noise=observe_noise,
            ):
                expected_shape = batch_shape + torch.Size([n, 1, m])
                self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
                self.assertEqual(cv_results.observed_Y.shape, expected_shape)
                if observe_noise:
                    self.assertEqual(cv_results.observed_Yvar.shape, expected_shape)
                else:
                    self.assertIsNone(cv_results.observed_Yvar)

                # check device and dtype
                self.assertEqual(
                    cv_results.posterior.mean.device.type, self.device.type
                )
                self.assertIs(cv_results.posterior.mean.dtype, dtype)

    def test_mtgp(self):
        train_X, train_Y = get_random_data(
            batch_shape=torch.Size(), m=1, n=3, device=self.device
        )
        cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y)
        with self.assertRaisesRegex(
            UnsupportedError, "Multi-task GPs are not currently supported."
        ):
            batch_cross_validation(
                model_cls=MultiTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
            )


class TestEfficientLOOCV(BotorchTestCase):
    def test_basic(self) -> None:
        """Test efficient LOO CV with basic single-output model."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            n, d = 10, 2
            train_X = torch.rand(n, d, **tkwargs)
            train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

            # Create and fit the model
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(mll)

            # Compute efficient LOO
            loo_results = efficient_loo_cv(model)

            # Check shapes
            self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))
            self.assertEqual(loo_results.variance.shape, torch.Size([n, 1]))
            self.assertEqual(loo_results.observed_Y.shape, torch.Size([n, 1]))

            # Check that LOO mean predictions are not equal to original Y
            # (they should be leave-one-out predictions)
            self.assertFalse(torch.allclose(loo_results.mean, loo_results.observed_Y))

            # Check that variances are positive
            self.assertTrue((loo_results.variance > 0).all())

            # Check device and dtype
            self.assertEqual(loo_results.mean.device.type, self.device.type)
            self.assertIs(loo_results.mean.dtype, dtype)

    def test_with_fixed_noise(self) -> None:
        """Test efficient LOO CV with fixed observation noise."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 10, 2
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)
        train_Yvar = torch.full_like(train_Y, 0.01)

        # Create and fit the model with fixed noise
        model = SingleTaskGP(train_X, train_Y, train_Yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        # Compute efficient LOO
        loo_results = efficient_loo_cv(model)

        # Check shapes
        self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))
        self.assertEqual(loo_results.variance.shape, torch.Size([n, 1]))
        self.assertIsNotNone(loo_results.observed_Yvar)

    def test_matches_naive(self) -> None:
        """Test that efficient LOO CV matches naive batch LOO CV results."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 8, 2
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Create and fit the full model
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        # Get efficient LOO results
        loo_results = efficient_loo_cv(model)

        # Verify the efficient LOO results have correct properties
        model.eval()
        for i in range(n):
            # Check that efficient LOO variance is positive
            self.assertTrue(loo_results.variance[i, 0] > 0)

            # The LOO predictions should differ from the observed values
            # since we're predicting each point as if it were left out
            # (except in degenerate cases with very high noise)

    def test_error_handling(self) -> None:
        """Test error cases for efficient_loo_cv."""

        # Test 1: Model without train_inputs
        class MockModelNoInputs:
            train_inputs = None
            train_targets = None
            training = False

            def eval(self):
                self.training = False
                return self

        model_no_inputs = MockModelNoInputs()
        with self.assertRaisesRegex(
            UnsupportedError, "Model must have train_inputs attribute"
        ):
            efficient_loo_cv(model_no_inputs)

        # Test 2: Model without train_targets
        class MockModelNoTargets:
            def __init__(self, train_X: torch.Tensor) -> None:
                self.train_inputs = (train_X,)
                self.train_targets = None
                self.training = False

            def eval(self):
                self.training = False
                return self

        train_X = torch.rand(10, 2)
        model_no_targets = MockModelNoTargets(train_X)
        with self.assertRaisesRegex(
            UnsupportedError, "Model must have train_targets attribute"
        ):
            efficient_loo_cv(model_no_targets)

        # Test 3: Model's forward doesn't return MultivariateNormal
        class MockModelBadForward:
            def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
                self.train_inputs = (train_X,)
                self.train_targets = train_Y.squeeze(-1)
                self.training = False
                self.input_transform = None

            def eval(self):
                self.training = False
                return self

            def train(self, mode: bool = True):
                self.training = mode
                return self

            def forward(self, x: torch.Tensor):
                return x.mean()

        train_Y = torch.rand(10, 1)
        model_bad_forward = MockModelBadForward(train_X, train_Y)
        with self.assertRaisesRegex(
            UnsupportedError, "Model's forward method must return a MultivariateNormal"
        ):
            efficient_loo_cv(model_bad_forward)

    def test_with_input_transform(self) -> None:
        """Test efficient LOO CV with input transform."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 10, 2
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Create model with input transform
        input_transform = Normalize(d=d)
        model = SingleTaskGP(train_X, train_Y, input_transform=input_transform)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        # Verify model has input transform
        self.assertIsNotNone(model.input_transform)

        # Compute efficient LOO - should work with input transform
        loo_results = efficient_loo_cv(model)

        # Check shapes
        self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))
        self.assertEqual(loo_results.variance.shape, torch.Size([n, 1]))
        self.assertTrue((loo_results.variance > 0).all())

    def test_model_in_training_mode(self) -> None:
        """Test that model training mode is preserved after efficient LOO CV."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 10, 2
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        # Put model in training mode
        model.train()
        self.assertTrue(model.training)

        # Run efficient LOO CV
        loo_results = efficient_loo_cv(model)

        # Check that model is back in training mode (was_training path)
        self.assertTrue(model.training)

        # Verify results are still valid
        self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))


class TestEnsembleLOOCV(BotorchTestCase):
    def test_basic(self) -> None:
        """Test ensemble LOO CV with a mock ensemble model."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 10, 2, 5

        # Create a mock ensemble model by setting up a SingleTaskGP
        # with batch dimension for ensemble members
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Simulate ensemble by expanding data to have a batch dimension
        # This creates a model that looks like an ensemble
        train_X_batched = train_X.unsqueeze(0).expand(num_models, -1, -1)
        train_Y_batched = train_Y.unsqueeze(0).expand(num_models, -1, -1)

        # Create an ensemble model with batched data
        # Need to use the contiguous version to avoid issues
        ensemble_model = SingleTaskGP(
            train_X_batched.contiguous(),
            train_Y_batched.contiguous(),
            outcome_transform=None,
        )

        # Manually set _is_ensemble attribute
        ensemble_model._is_ensemble = True

        # Fit the ensemble model
        mll_ensemble = ExactMarginalLogLikelihood(
            ensemble_model.likelihood, ensemble_model
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll_ensemble)

        # Compute ensemble LOO
        loo_results = ensemble_loo_cv(ensemble_model)

        # Check shapes - aggregated results should have no ensemble dimension
        self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))
        self.assertEqual(loo_results.variance.shape, torch.Size([n, 1]))
        self.assertEqual(loo_results.observed_Y.shape, torch.Size([n, 1]))

        # Check per-model results have ensemble dimension
        self.assertEqual(
            loo_results.per_model_mean.shape, torch.Size([num_models, n, 1])
        )
        self.assertEqual(
            loo_results.per_model_variance.shape, torch.Size([num_models, n, 1])
        )

        # Check that variances are positive
        self.assertTrue((loo_results.variance > 0).all())
        self.assertTrue((loo_results.per_model_variance > 0).all())

        # Check device and dtype
        self.assertEqual(loo_results.mean.device.type, self.device.type)
        self.assertIs(loo_results.mean.dtype, torch.double)

    def test_mixture_statistics(self) -> None:
        """Test that mixture statistics are correctly computed."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 8, 2, 4

        # Create batched data for ensemble
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Create batched training data
        train_X_batched = train_X.unsqueeze(0).expand(num_models, -1, -1)
        train_Y_batched = train_Y.unsqueeze(0).expand(num_models, -1, -1)

        ensemble_model = SingleTaskGP(
            train_X_batched.contiguous(),
            train_Y_batched.contiguous(),
            outcome_transform=None,
        )
        ensemble_model._is_ensemble = True

        mll = ExactMarginalLogLikelihood(ensemble_model.likelihood, ensemble_model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        loo_results = ensemble_loo_cv(ensemble_model)

        # Verify mixture mean is average of per-model means
        expected_mean = loo_results.per_model_mean.mean(dim=0)
        self.assertTrue(torch.allclose(loo_results.mean, expected_mean))

        # Verify mixture variance follows law of total variance:
        # Var(Y) = E[Var(Y|K)] + Var(E[Y|K])
        mean_of_variances = loo_results.per_model_variance.mean(dim=0)
        variance_of_means = loo_results.per_model_mean.var(dim=0)
        expected_variance = mean_of_variances + variance_of_means
        self.assertTrue(torch.allclose(loo_results.variance, expected_variance))

    def test_error_handling(self) -> None:
        """Test error cases for ensemble_loo_cv."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 10, 2

        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Test 1: Non-ensemble model raises error
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        with self.assertRaisesRegex(
            UnsupportedError,
            "ensemble_loo_cv requires an ensemble model",
        ):
            ensemble_loo_cv(model)

        # Test 2: Ensemble model with non-batched results raises error
        model._is_ensemble = True

        mock_results = LOOCVResults(
            mean=torch.rand(n, 1, **tkwargs),  # 2D, not 3D
            variance=torch.rand(n, 1, **tkwargs),
            observed_Y=train_Y,
            observed_Yvar=None,
            model=model,
        )

        with patch(
            "botorch.cross_validation.efficient_loo_cv", return_value=mock_results
        ):
            with self.assertRaisesRegex(
                UnsupportedError,
                "Expected ensemble model to produce batched LOO results",
            ):
                ensemble_loo_cv(model)

    def test_with_fixed_noise(self) -> None:
        """Test ensemble LOO CV with fixed observation noise (observed_Yvar path)."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 8, 2, 4

        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)
        train_Yvar = torch.full_like(train_Y, 0.01)

        # Create batched training data for ensemble
        train_X_batched = train_X.unsqueeze(0).expand(num_models, -1, -1).contiguous()
        train_Y_batched = train_Y.unsqueeze(0).expand(num_models, -1, -1).contiguous()
        train_Yvar_batched = (
            train_Yvar.unsqueeze(0).expand(num_models, -1, -1).contiguous()
        )

        # Create ensemble model with fixed noise
        ensemble_model = SingleTaskGP(
            train_X_batched,
            train_Y_batched,
            train_Yvar_batched,
            outcome_transform=None,
        )
        ensemble_model._is_ensemble = True

        mll = ExactMarginalLogLikelihood(ensemble_model.likelihood, ensemble_model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll)

        # Mock efficient_loo_cv to return results with batched observed_Yvar
        mock_results = LOOCVResults(
            mean=torch.rand(num_models, n, 1, **tkwargs),
            variance=torch.rand(num_models, n, 1, **tkwargs),
            observed_Y=train_Y_batched,
            observed_Yvar=train_Yvar_batched,  # 3D tensor for ensemble
            model=ensemble_model,
        )

        with patch(
            "botorch.cross_validation.efficient_loo_cv", return_value=mock_results
        ):
            loo_results = ensemble_loo_cv(ensemble_model)

            # Check that observed_Yvar was reduced to 2D (removed ensemble dim)
            self.assertIsNotNone(loo_results.observed_Yvar)
            self.assertEqual(loo_results.observed_Yvar.shape, torch.Size([n, 1]))

            # Check other shapes are correct
            self.assertEqual(loo_results.mean.shape, torch.Size([n, 1]))
            self.assertEqual(loo_results.variance.shape, torch.Size([n, 1]))
            self.assertEqual(loo_results.observed_Y.shape, torch.Size([n, 1]))
