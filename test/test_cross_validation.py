#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from unittest.mock import MagicMock, patch

import torch
from botorch.cross_validation import (
    batch_cross_validation,
    CVResults,
    efficient_loo_cv,
    ensemble_loo_cv,
    gen_loo_cv_folds,
    loo_cv,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
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
        """Test efficient LOO CV with various data configurations.

        This test covers:
        - Single and multiple outputs (m=1 and m>1)
        - With and without batch dimensions
        """
        n = 10
        tkwargs = {"device": self.device, "dtype": torch.double}

        for m, batch_shape in itertools.product(
            (1, 3),  # single and multi-output
            (torch.Size(), torch.Size([2])),  # no batch and with batch
        ):
            with self.subTest(m=m, batch_shape=batch_shape):
                train_X, train_Y = get_random_data(
                    batch_shape=batch_shape, m=m, n=n, **tkwargs
                )

                model = SingleTaskGP(train_X, train_Y)
                model.eval()

                loo_results = efficient_loo_cv(model)

                # Output shape: batch_shape x n x 1 x m
                expected_shape = batch_shape + torch.Size([n, 1, m])
                self.assertEqual(loo_results.posterior.mean.shape, expected_shape)
                self.assertEqual(loo_results.posterior.variance.shape, expected_shape)
                self.assertEqual(loo_results.observed_Y.shape, expected_shape)
                self.assertTrue((loo_results.posterior.variance > 0).all())

    def test_matches_naive(self) -> None:
        """Test that efficient LOO CV matches naive LOO CV."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 6, 2

        for (
            m,
            batch_shape,
            use_transforms,
            use_fixed_noise,
            obs_noise,
        ) in itertools.product(
            (1, 3),  # single and multi-output
            (torch.Size(), torch.Size([2])),  # no batch and with batch
            (False, True),  # transforms
            (False, True),  # fixed noise
            (False, True),  # observation noise
        ):
            # Skip transforms with batch dimensions - Standardize requires
            # matching batch_shape argument which complicates the test setup.
            # The core functionality is tested without transforms.
            if batch_shape and use_transforms:
                continue

            with self.subTest(
                m=m,
                batch_shape=batch_shape,
                transforms=use_transforms,
                fixed_noise=use_fixed_noise,
                obs_noise=obs_noise,
            ):
                train_X, train_Y = get_random_data(
                    batch_shape=batch_shape, m=m, n=n, d=d, **tkwargs
                )

                # Build model kwargs with optional transforms
                model_kwargs = {}
                if use_transforms:
                    model_kwargs["input_transform"] = Normalize(d=d)
                    model_kwargs["outcome_transform"] = Standardize(m=m)
                else:
                    model_kwargs["outcome_transform"] = None

                train_Yvar = torch.full_like(train_Y, 5e-3) if use_fixed_noise else None
                if use_fixed_noise:
                    model = SingleTaskGP(train_X, train_Y, train_Yvar, **model_kwargs)
                else:
                    model = SingleTaskGP(train_X, train_Y, **model_kwargs)

                # Put into eval mode, simulating a post-fit model
                model.eval()

                # Compare efficient vs naive
                loo_results = efficient_loo_cv(model, observation_noise=obs_noise)
                naive_mean, naive_var = naive_loo_cv(
                    model, observation_noise=obs_noise, batch_shape=batch_shape
                )

                loo_mean = loo_results.posterior.mean.squeeze(-2)
                loo_var = loo_results.posterior.variance.squeeze(-2)
                self.assertAllClose(loo_mean, naive_mean, rtol=1e-6, atol=1e-6)
                self.assertAllClose(loo_var, naive_var, rtol=1e-6, atol=1e-6)

                # Verify observed_Y and observed_Yvar shapes
                expected_shape = batch_shape + torch.Size([n, 1, m])
                self.assertEqual(loo_results.observed_Y.shape, expected_shape)

                if use_fixed_noise:
                    self.assertIsNotNone(loo_results.observed_Yvar)
                    self.assertEqual(
                        loo_results.observed_Yvar.shape,
                        expected_shape,
                        f"observed_Yvar shape mismatch: got "
                        f"{loo_results.observed_Yvar.shape}, expected {expected_shape}",
                    )
                else:
                    self.assertIsNone(loo_results.observed_Yvar)

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

        train_X = torch.rand(10, 2, device=self.device)
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

        train_Y = torch.rand(10, 1, device=self.device)
        model_bad_forward = MockModelBadForward(train_X, train_Y)
        with self.assertRaisesRegex(
            UnsupportedError, "Model's forward method must return a MultivariateNormal"
        ):
            efficient_loo_cv(model_bad_forward)

        # Test 4: Model with auxiliary inputs (tuple train_inputs)
        model = SingleTaskGP(train_X, train_Y)
        model.eval()
        # Mock train_inputs[0] to be a tuple (simulating auxiliary inputs)
        model.train_inputs = ((train_X, train_X),)
        with self.assertRaisesRegex(
            UnsupportedError, "not supported for models with auxiliary inputs"
        ):
            efficient_loo_cv(model)


class TestLOOCV(BotorchTestCase):
    """Test the high-level loo_cv dispatch function."""

    def test_dispatches_to_correct_implementation(self) -> None:
        """Test that loo_cv dispatches based on _is_ensemble attribute."""
        test_cases = [
            ("efficient_loo_cv", False),  # Standard models -> efficient_loo_cv
            ("ensemble_loo_cv", True),  # Ensemble models -> ensemble_loo_cv
        ]

        for target_func, is_ensemble in test_cases:
            with self.subTest(target_func=target_func, is_ensemble=is_ensemble):
                # Create mock model
                mock_model = MagicMock(
                    spec=["train_inputs", "train_targets", "likelihood"]
                )
                mock_model._is_ensemble = is_ensemble

                # Create mock return value
                mock_results = CVResults(
                    model=mock_model,
                    posterior=MagicMock(),
                    observed_Y=torch.zeros(1, device=self.device),
                    observed_Yvar=None,
                )

                with patch(
                    f"botorch.cross_validation.{target_func}",
                    return_value=mock_results,
                ) as mock_func:
                    result = loo_cv(mock_model)

                    # Verify correct function was called with the model
                    mock_func.assert_called_once_with(
                        mock_model, observation_noise=True
                    )
                    self.assertIs(result, mock_results)


class TestEnsembleLOOCV(BotorchTestCase):
    def test_basic(self) -> None:
        """Test ensemble LOO CV with different configurations.

        This test covers:
        - Single-output (m=1) and multi-output (m>1) models
        - With and without observation noise
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 10, 2, 5

        for m in (1, 3):
            with self.subTest(m=m):
                train_X = torch.rand(n, d, **tkwargs)
                train_Y = torch.rand(n, m, **tkwargs)

                ensemble_model = _create_batched_single_task_gp(
                    train_X, train_Y, num_models
                )

                # Compute ensemble LOO
                loo_results = ensemble_loo_cv(ensemble_model)

                # Check that posterior is a GaussianMixturePosterior
                self.assertIsInstance(loo_results.posterior, GaussianMixturePosterior)

                # Check shapes - per-member results via posterior.mean
                # MCMC_DIM=-3 means shape is n x num_models x 1 x m
                per_model_mean = loo_results.posterior.mean
                per_model_var = loo_results.posterior.variance
                self.assertEqual(
                    per_model_mean.shape, torch.Size([n, num_models, 1, m])
                )
                self.assertEqual(per_model_var.shape, torch.Size([n, num_models, 1, m]))

                # Check mixture statistics via mixture_mean and mixture_variance
                mixture_mean = loo_results.posterior.mixture_mean
                mixture_var = loo_results.posterior.mixture_variance
                self.assertEqual(mixture_mean.shape, torch.Size([n, 1, m]))
                self.assertEqual(mixture_var.shape, torch.Size([n, 1, m]))

                # Check observed_Y shape (ensemble dim removed)
                self.assertEqual(loo_results.observed_Y.shape, torch.Size([n, 1, m]))

                # Check that variances are positive
                self.assertTrue((mixture_var > 0).all())
                self.assertTrue((per_model_var > 0).all())

                # Check device and dtype
                self.assertEqual(mixture_mean.device.type, self.device.type)
                self.assertIs(mixture_mean.dtype, torch.double)

                # Test observation_noise parameter
                # When observation_noise=False, variance should be smaller
                # (since observation noise is subtracted)
                loo_results_no_noise = ensemble_loo_cv(
                    ensemble_model, observation_noise=False
                )

                # Mean should be the same regardless of observation_noise
                self.assertTrue(
                    torch.allclose(
                        loo_results.posterior.mixture_mean,
                        loo_results_no_noise.posterior.mixture_mean,
                    )
                )
                # Mixture variance without noise should be < variance with noise
                self.assertTrue(
                    (
                        loo_results_no_noise.posterior.mixture_variance
                        < loo_results.posterior.mixture_variance
                    ).all()
                )
                # Per-model variance without noise should be < variance with noise
                self.assertTrue(
                    (
                        loo_results_no_noise.posterior.variance
                        < loo_results.posterior.variance
                    ).all()
                )
                # Variance without noise should still be non-negative
                self.assertTrue(
                    (loo_results_no_noise.posterior.mixture_variance >= 0).all()
                )
                self.assertTrue((loo_results_no_noise.posterior.variance >= 0).all())

    def test_matches_naive(self) -> None:
        """Test that ensemble_loo_cv matches naive per-model LOO CV."""
        from botorch.fit import fit_gpytorch_mll

        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 6, 2, 3

        for m, batch_shape, use_fixed_noise, obs_noise in itertools.product(
            (1, 3),  # single and multi-output
            (torch.Size(), torch.Size([2])),  # no batch and with batch
            (False, True),  # fixed noise
            (False, True),  # observation noise
        ):
            with self.subTest(
                m=m,
                batch_shape=batch_shape,
                fixed_noise=use_fixed_noise,
                obs_noise=obs_noise,
            ):
                train_X, train_Y = get_random_data(
                    batch_shape=batch_shape, m=m, n=n, d=d, **tkwargs
                )
                train_Yvar = torch.full_like(train_Y, 5e-3) if use_fixed_noise else None

                # Create ensemble model - num_models is inserted after batch_shape
                # For batch_shape=[], shape is: num_models x n x d
                # For batch_shape=[2], shape is: 2 x num_models x n x d
                ensemble_model = _create_batched_single_task_gp(
                    train_X,
                    train_Y,
                    num_models,
                    train_Yvar=train_Yvar,
                    batch_shape=batch_shape,
                )

                # Fit the ensemble model
                mll = ExactMarginalLogLikelihood(
                    ensemble_model.likelihood, ensemble_model
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizationWarning)
                    fit_gpytorch_mll(mll)
                ensemble_model.eval()

                # Get ensemble LOO CV results
                loo_results = ensemble_loo_cv(
                    ensemble_model, observation_noise=obs_noise
                )
                ensemble_mean = loo_results.posterior.mean
                ensemble_var = loo_results.posterior.variance

                # Compute naive LOO CV for each model in the ensemble
                naive_means = []
                naive_vars = []
                # Position of num_models dimension in ensemble params
                num_models_dim = len(batch_shape)

                for i in range(num_models):
                    # Create single model with same (non-batched num_models) data
                    if use_fixed_noise:
                        single_model = SingleTaskGP(
                            train_X, train_Y, train_Yvar, outcome_transform=None
                        )
                    else:
                        single_model = SingleTaskGP(
                            train_X, train_Y, outcome_transform=None
                        )

                    # Copy hyperparameters from the i-th model in the ensemble
                    # For batched ensemble, num_models is at position len(batch_shape)
                    ensemble_state = ensemble_model.state_dict()
                    single_state = single_model.state_dict()
                    for name, param in ensemble_state.items():
                        if name in single_state:
                            target_shape = single_state[name].shape
                            # Check if param has num_models dimension at the right
                            # position
                            if (
                                param.dim() > num_models_dim
                                and param.shape[num_models_dim] == num_models
                            ):
                                # Select the i-th ensemble member
                                selected = param.select(num_models_dim, i)
                                if target_shape == selected.shape:
                                    single_state[name] = selected
                            elif target_shape == param.shape:
                                single_state[name] = param
                    single_model.load_state_dict(single_state)
                    single_model.eval()

                    mean_i, var_i = naive_loo_cv(
                        single_model,
                        observation_noise=obs_noise,
                        batch_shape=batch_shape,
                    )
                    naive_means.append(mean_i)
                    naive_vars.append(var_i)

                # naive_means[i] has shape batch_shape x n x m
                # Stack to get batch_shape x n x num_models x m, then add q=1 dim
                # Final shape: batch_shape x n x num_models x 1 x m (MCMC_DIM=-3)
                # Stack along dim after batch_shape and n: len(batch_shape) + 1
                stack_dim = len(batch_shape) + 1
                naive_means = torch.stack(naive_means, dim=stack_dim).unsqueeze(-2)
                naive_vars = torch.stack(naive_vars, dim=stack_dim).unsqueeze(-2)

                # Compare per-model results
                self.assertAllClose(ensemble_mean, naive_means, rtol=1e-5, atol=1e-6)
                self.assertAllClose(ensemble_var, naive_vars, rtol=1e-5, atol=1e-6)

                # Verify mixture statistics (law of total expectation/variance)
                # num_models is at MCMC_DIM=-3 (works for any batch_shape)
                expected_mixture_mean = naive_means.mean(dim=-3)
                self.assertAllClose(
                    loo_results.posterior.mixture_mean,
                    expected_mixture_mean,
                    rtol=1e-5,
                    atol=1e-6,
                )

                # mixture_variance = E[Var] + Var[E] (law of total variance)
                mean_of_vars = naive_vars.mean(dim=-3)
                var_of_means = naive_means.var(dim=-3, correction=0)
                expected_mixture_var = mean_of_vars + var_of_means
                self.assertAllClose(
                    loo_results.posterior.mixture_variance,
                    expected_mixture_var,
                    rtol=1e-5,
                    atol=1e-6,
                )

                # Verify observed_Y and observed_Yvar shapes
                expected_shape = batch_shape + torch.Size([n, 1, m])
                self.assertEqual(loo_results.observed_Y.shape, expected_shape)

                if use_fixed_noise:
                    self.assertIsNotNone(loo_results.observed_Yvar)
                    self.assertEqual(
                        loo_results.observed_Yvar.shape,
                        expected_shape,
                        f"observed_Yvar shape mismatch: got "
                        f"{loo_results.observed_Yvar.shape}, expected {expected_shape}",
                    )
                else:
                    self.assertIsNone(loo_results.observed_Yvar)

    def test_fixed_noise_1d_edge_case(self) -> None:
        """Test observed_Yvar when noise has 1D shape (edge case).

        This tests the ``else`` branch in ensemble_loo_cv where noise.dim() < 2.
        Multi-output models always have 2D+ noise, so this only applies to m=1.
        The normal 2D noise case is covered by test_matches_naive.
        """
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d, num_models = 10, 2, 5

        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.rand(n, 1, **tkwargs)

        # Create model without fixed noise, then manually set 1D noise
        ensemble_model = _create_batched_single_task_gp(
            train_X,
            train_Y,
            num_models,
            train_Yvar=None,
        )
        # Replace likelihood with FixedNoiseGaussianLikelihood with 1D noise
        noise_1d = torch.full((n,), 0.01, **tkwargs)
        ensemble_model.likelihood = FixedNoiseGaussianLikelihood(noise=noise_1d)
        ensemble_model.eval()

        loo_results = ensemble_loo_cv(ensemble_model)

        # observed_Yvar should have shape n x 1 x 1
        self.assertIsNotNone(loo_results.observed_Yvar)
        self.assertEqual(loo_results.observed_Yvar.shape, torch.Size([n, 1, 1]))
        # All values should be 0.01
        self.assertTrue(
            torch.allclose(
                loo_results.observed_Yvar,
                torch.full((n, 1, 1), 0.01, **tkwargs),
            )
        )

    def test_error_handling(self) -> None:
        """Test error cases for ensemble_loo_cv."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        n, d = 10, 2

        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)

        # Test 1: Non-ensemble model raises error
        model = SingleTaskGP(train_X, train_Y)
        model.eval()

        with self.assertRaisesRegex(
            UnsupportedError,
            "ensemble_loo_cv requires an ensemble model",
        ):
            ensemble_loo_cv(model)

        # Test 2: Ensemble model with non-batched results raises error
        model._is_ensemble = True

        # Create a mock posterior with 2D results (not 4D as expected for ensembles)
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import DiagLinearOperator

        mock_mean = torch.rand(n, 1, **tkwargs)
        mock_var = torch.rand(n, 1, **tkwargs)
        mock_mvn = MultivariateNormal(
            mean=mock_mean, covariance_matrix=DiagLinearOperator(mock_var)
        )
        mock_posterior = GPyTorchPosterior(distribution=mock_mvn)
        mock_results = CVResults(
            model=model,
            posterior=mock_posterior,
            observed_Y=train_Y,
        )

        with patch(
            "botorch.cross_validation.efficient_loo_cv", return_value=mock_results
        ):
            with self.assertRaisesRegex(
                UnsupportedError,
                "Expected ensemble model to produce batched LOO results",
            ):
                ensemble_loo_cv(model)

        # Test 3: Inconsistent ensemble data should raise error
        from botorch.cross_validation import _verify_ensemble_data_consistency

        num_models, n, m = 3, 5, 2
        # Test different configurations: (shape, num_models_dim, tensor_name)
        test_configs = [
            # Single-output: num_models x n, num_models_dim = -2
            ((num_models, n), -2, "train_Y"),
            # Multi-output: num_models x m x n, num_models_dim = -3
            ((num_models, m, n), -3, "train_Y"),
            # Batched single-output: batch x num_models x n
            ((2, num_models, n), -2, "observation noise"),
        ]
        for shape, num_models_dim, tensor_name in test_configs:
            with self.subTest(shape=shape, num_models_dim=num_models_dim):
                # Consistent data should not raise
                consistent_data = torch.ones(shape, **tkwargs)
                _verify_ensemble_data_consistency(
                    consistent_data, num_models_dim, tensor_name
                )

                # Inconsistent data should raise
                inconsistent_data = torch.randn(shape, **tkwargs)
                with self.assertRaisesRegex(
                    UnsupportedError,
                    f"Ensemble members have different {tensor_name}",
                ):
                    _verify_ensemble_data_consistency(
                        inconsistent_data, num_models_dim, tensor_name
                    )

        # Edge case: single model (num_models = 1) should not raise
        single_model_data = torch.randn(1, n, **tkwargs)
        _verify_ensemble_data_consistency(single_model_data, -2, "train_Y")


_EMPTY_BATCH_SHAPE: torch.Size = torch.Size()


def naive_loo_cv(
    fitted_model: SingleTaskGP,
    observation_noise: bool = True,
    batch_shape: torch.Size = _EMPTY_BATCH_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute naive LOO CV by creating a model for each fold.

    This O(n^4) implementation creates a model for each left-out point,
    copying hyperparameters from fitted_model (no refitting).

    Args:
        fitted_model: A fitted GP model whose hyperparameters will be copied.
        observation_noise: If True, include observation noise in the posterior
            variance. For fixed noise models, uses the held-out point's noise.
        batch_shape: The batch shape of the data. For batched models, LOO is
            computed independently for each batch element.

    Returns:
        A tuple of (loo_means, loo_variances) with shape ``batch_shape x n x m``.
    """
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    fitted_model.eval()
    train_X = fitted_model.train_inputs[0]
    train_Y = fitted_model.train_targets
    n = train_X.shape[-2]
    m = fitted_model.num_outputs
    has_fixed_noise = isinstance(fitted_model.likelihood, FixedNoiseGaussianLikelihood)

    # Normalize train_X: for multi-output models, BoTorch internally stores X as
    # [batch x] m x n x d (replicated per output) because it uses batched GPs.
    # See BatchedMultiOutputGPyTorchModel.__init__ which calls:
    #   train_X = train_X.unsqueeze(-3).expand(..., self._num_outputs, ...)
    # We extract the canonical X by selecting the first output's X.
    if m > 1:
        train_X = train_X.select(-3, 0)  # Remove output dim: [batch x] n x d

    # Normalize train_Y from internal format to SingleTaskGP input format:
    # - Multi-output: [batch x] m x n -> [batch x] n x m
    # - Single-output: [batch x] n -> [batch x] n x 1
    if m > 1:
        train_Y = train_Y.movedim(-2, -1)
    else:
        train_Y = train_Y.unsqueeze(-1)

    # Normalize noise similarly if present
    if has_fixed_noise:
        noise = fitted_model.likelihood.noise
        if m > 1:
            noise = noise.movedim(-2, -1)  # [batch x] m x n -> [batch x] n x m
        else:
            noise = noise.unsqueeze(-1)  # [batch x] n -> [batch x] n x 1
    else:
        noise = None

    # Output shape: batch_shape x n x m
    output_shape = batch_shape + torch.Size([n, m])
    loo_means = torch.zeros(output_shape, dtype=train_X.dtype, device=train_X.device)
    loo_vars = torch.zeros(output_shape, dtype=train_X.dtype, device=train_X.device)

    for i in range(n):
        # Create mask excluding point i
        mask = torch.arange(n, device=train_X.device) != i

        # Extract fold data - ellipsis handles any batch dimensions
        # train_X: [batch x] n x d -> [batch x] (n-1) x d
        # train_Y: [batch x] n x m -> [batch x] (n-1) x m
        fold_X = train_X[..., mask, :]
        fold_Y = train_Y[..., mask, :]
        test_X = train_X[..., i : i + 1, :]

        # Create fold model
        kwargs = {"outcome_transform": None, "input_transform": None}
        if has_fixed_noise:
            fold_noise = noise[..., mask, :]
            model = SingleTaskGP(fold_X, fold_Y, fold_noise, **kwargs)
        else:
            model = SingleTaskGP(fold_X, fold_Y, **kwargs)

        # Copy matching hyperparameters
        fitted_state = fitted_model.state_dict()
        fold_state = model.state_dict()
        for name, param in fitted_state.items():
            if has_fixed_noise and "noise" in name.lower():
                continue
            if name in fold_state and fold_state[name].shape == param.shape:
                fold_state[name] = param
        model.load_state_dict(fold_state)
        model.eval()

        # Get posterior prediction
        with torch.no_grad():
            if has_fixed_noise and observation_noise:
                held_out_noise = noise[..., i : i + 1, :]
                posterior = model.posterior(test_X, observation_noise=held_out_noise)
            else:
                posterior = model.posterior(test_X, observation_noise=observation_noise)

            # posterior.mean/variance: [batch x] 1 x m -> [batch x] m
            loo_means[..., i, :] = posterior.mean.squeeze(-2)
            loo_vars[..., i, :] = posterior.variance.squeeze(-2)

    return loo_means, loo_vars


def _create_batched_single_task_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    num_models: int,
    train_Yvar: torch.Tensor | None = None,
    batch_shape: torch.Size | None = None,
) -> SingleTaskGP:
    """Create a batched SingleTaskGP that simulates an ensemble model.

    Args:
        train_X: Training inputs of shape batch_shape x n x d.
        train_Y: Training outputs of shape batch_shape x n x m.
        num_models: Number of models in the ensemble.
        train_Yvar: Optional observation noise of shape batch_shape x n x m.
        batch_shape: The batch shape of the input data. Defaults to empty.

    Returns:
        A batched SingleTaskGP with _is_ensemble=True. The num_models
        dimension is inserted at position len(batch_shape), which
        corresponds to MCMC_DIM=-3 in the output posterior (consistent
        with how fully Bayesian models handle MCMC samples).
    """
    if batch_shape is None:
        batch_shape = torch.Size()

    # Helper to insert and expand num_models dimension at the right position.
    # For batch_shape=[], this gives: num_models x n x d
    # For batch_shape=[2], this gives: 2 x num_models x n x d
    def expand_with_num_models(t: torch.Tensor) -> torch.Tensor:
        insert_dim = len(batch_shape)
        return (
            t.unsqueeze(insert_dim)
            .expand(*t.shape[:insert_dim], num_models, *t.shape[insert_dim:])
            .contiguous()
        )

    train_X_batched = expand_with_num_models(train_X)
    train_Y_batched = expand_with_num_models(train_Y)

    if train_Yvar is not None:
        train_Yvar_batched = expand_with_num_models(train_Yvar)
        model = SingleTaskGP(
            train_X_batched,
            train_Y_batched,
            train_Yvar_batched,
            outcome_transform=None,
        )
    else:
        model = SingleTaskGP(
            train_X_batched,
            train_Y_batched,
            outcome_transform=None,
        )

    model._is_ensemble = True
    return model
