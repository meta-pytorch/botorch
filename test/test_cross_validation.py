#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.cross_validation import (
    batch_cross_validation,
    efficient_loo_cv,
    gen_loo_cv_folds,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
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
        A tuple of (loo_means, loo_variances) with shape `batch_shape x n x m`.
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
