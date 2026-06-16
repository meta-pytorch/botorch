#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from unittest.mock import patch

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.empirical_gps import (
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch.models.empirical_gps.utils import (
    compute_sample_covariance,
    LinearInterpolation1D,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from torch import Tensor


class TestEmpiricalOneDimensionalGP(BotorchTestCase):
    """Tests for EmpiricalOneDimensionalGP and related modules."""

    # Use double precision for numerical stability in GP posterior computations
    dtype = torch.float64

    def _get_data(
        self,
        num_curves: int,
        num_progression: int,
        num_train: int,
        batch_shape: tuple[int, ...] = (),
        num_outputs: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate test data for empirical one-dimensional GP.

        Args:
            num_curves: Number of historical curves.
            num_progression: Number of progression values.
            num_train: Number of training progression values.
            batch_shape: Batch shape for outputs.
            num_outputs: Number of output dimensions (m).

        Returns:
            Tuple of (train_X, train_Y, all_Y, historical_X, historical_Y).
        """
        a, b = 1.0, 10.0
        historical_X = torch.linspace(
            a, b, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Generate historical_Y with shape: num_curves x num_progression x m
        historical_Y = torch.randn(
            num_curves,
            num_progression,
            num_outputs,
            device=self.device,
            dtype=self.dtype,
        )

        all_X = historical_X.expand((*batch_shape, *historical_X.shape))
        train_X = all_X[..., :num_train, :]

        # Generate targets as linear combinations of historical curves
        def _sample_from_curves(Y_2d: Tensor) -> Tensor:
            """Sample from the empirical distribution of historical curves.

            Computes mean + sqrt(cov) @ z where z ~ N(0, I) to generate samples
            that have the same mean and covariance as the historical curves.
            """
            mean_Y = Y_2d.mean(dim=0, keepdim=True)
            centered_Y = Y_2d - mean_Y
            root_cov = centered_Y / math.sqrt(num_curves - 1)
            activations = torch.randn(
                *batch_shape, num_curves, 1, device=self.device, dtype=self.dtype
            )
            return (mean_Y.T + root_cov.T @ activations).squeeze(-1)

        all_Y = torch.stack(
            [_sample_from_curves(historical_Y[..., i]) for i in range(num_outputs)],
            dim=-1,
        )

        train_Y = all_Y[..., :num_train, :]
        return train_X, train_Y, all_Y, historical_X, historical_Y

    # =========================================================================
    # Private test helpers
    # =========================================================================

    def _test_posterior_prediction(self) -> None:
        """Test basic posterior prediction with over-and under-determined cases."""
        num_progression = 128
        test_cases = [
            # (num_curves, num_train, expected_train_rmse, expected_pred_rmse)
            (4, 64, 1e-4, 1e-4),  # overdetermined: posterior contracts quickly
            (32, 28, 1e-2, None),  # underdetermined: check calibration instead
        ]

        for batch_shape in ((), (2, 3)):
            for num_curves, num_train, train_rmse_bound, pred_rmse_bound in test_cases:
                with self.subTest(
                    batch_shape=batch_shape,
                    num_curves=num_curves,
                    num_train=num_train,
                ):
                    torch.manual_seed(1234)
                    train_X, train_Y, all_Y, historical_X, historical_Y = (
                        self._get_data(
                            num_curves=num_curves,
                            num_progression=num_progression,
                            num_train=num_train,
                            batch_shape=batch_shape,
                        )
                    )
                    model = EmpiricalOneDimensionalGP(
                        train_X=train_X,
                        train_Y=train_Y,
                        train_Yvar=torch.full_like(train_Y, 1e-4),
                        historical_X=historical_X,
                        historical_Y=historical_Y,
                    )

                    # Verify module types
                    self.assertIsInstance(
                        model.mean_module, EmpiricalOneDimensionalMean
                    )
                    self.assertIsInstance(
                        model.covar_module, EmpiricalOneDimensionalKernel
                    )

                    # Compute posterior
                    post = model.posterior(historical_X, observation_noise=False)
                    Y_pred = post.mean
                    Ystd_pred = post.variance.sqrt()

                    self.assertEqual(
                        Ystd_pred.shape, (*batch_shape, num_progression, 1)
                    )

                    # Check residuals
                    R = Y_pred - all_Y
                    R_train = R[..., :num_train, :]
                    self.assertLess(R_train.square().mean().sqrt(), train_rmse_bound)

                    if pred_rmse_bound is not None:
                        R_pred = R[..., num_train:, :]
                        self.assertLess(R_pred.square().mean().sqrt(), pred_rmse_bound)
                        self.assertLess(Ystd_pred.max(), 1e-2)
                    else:
                        # Check calibration for underdetermined case
                        standardized = R / Ystd_pred
                        standardized = standardized[..., num_train:, :]
                        self.assertLess(standardized.square().mean().sqrt(), 2)
                        self.assertGreater(standardized.square().mean().sqrt(), 0.5)

    def _test_kernel_diagonal(self) -> None:
        """Test that kernel diag=True matches diagonal of full kernel matrix."""
        torch.manual_seed(1234)
        for batch_shape in ((), (2, 3)):
            with self.subTest(batch_shape=batch_shape):
                num_train = 28
                train_X, train_Y, _, historical_X, historical_Y = self._get_data(
                    num_curves=32,
                    num_progression=128,
                    num_train=num_train,
                    batch_shape=batch_shape,
                )
                model = EmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=torch.full_like(train_Y, 1e-4),
                    historical_X=historical_X,
                    historical_Y=historical_Y,
                )

                train_covar = model.covar_module(train_X)
                diag_train_covar = model.covar_module(train_X, diag=True)

                self.assertEqual(diag_train_covar.shape, (*batch_shape, num_train))
                expected_diag = train_covar.diagonal(dim1=-2, dim2=-1)
                self.assertAllClose(diag_train_covar, expected_diag)

    def _test_likelihood_handling(self) -> None:
        """Test that likelihood is correctly inferred or set."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_curves=5, num_progression=20, num_train=10
        )

        # Test inferred noise (train_Yvar=None)
        model_inferred = EmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertIsInstance(model_inferred.likelihood, GaussianLikelihood)

        # Test fixed noise (train_Yvar provided)
        train_Yvar = torch.full_like(train_Y, 0.01)
        model_fixed = EmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertIsInstance(model_fixed.likelihood, FixedNoiseGaussianLikelihood)

        # Test custom likelihood
        custom_likelihood = GaussianLikelihood()
        model_custom = EmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
            likelihood=custom_likelihood,
        )
        self.assertIs(model_custom.likelihood, custom_likelihood)

    def _test_input_validation(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_curves=7, num_progression=5, num_train=4
        )

        # Test invalid covar_module type
        with self.assertRaisesRegex(
            ValueError, "must be an instance of EmpiricalOneDimensionalKernel"
        ):
            EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                historical_X=historical_X,
                historical_Y=historical_Y,
                covar_module=RBFKernel(),
            )

        # Test ARD mismatch
        for ard in [True, False]:
            with self.subTest(ard=ard):
                with self.assertRaisesRegex(
                    ValueError, "`ard` argument must equal `covar_module.ard`"
                ):
                    EmpiricalOneDimensionalGP(
                        train_X=train_X,
                        train_Y=train_Y,
                        historical_X=historical_X,
                        historical_Y=historical_Y,
                        covar_module=EmpiricalOneDimensionalKernel(
                            X_full=historical_X,
                            Y_full=historical_Y,
                            ard=ard,
                        ),
                        ard=(not ard),
                    )

        # historical_Y / Y_full must be 3-dim (num_curves x num_progression x m)
        historical_Y_2d = historical_Y.squeeze(-1)
        with self.assertRaisesRegex(ValueError, "Expected historical_Y to be 3-dim"):
            EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                historical_X=historical_X,
                historical_Y=historical_Y_2d,
            )
        with self.assertRaisesRegex(ValueError, "Expected Y_full to be 3-dim"):
            EmpiricalOneDimensionalMean(X_full=historical_X, Y_full=historical_Y_2d)
        with self.assertRaisesRegex(ValueError, "Expected Y_full to be 3-dim"):
            EmpiricalOneDimensionalKernel(X_full=historical_X, Y_full=historical_Y_2d)

    def _test_edge_cases(self) -> None:
        """Cover the use_svd property, last_dim_is_batch, and the mean NaN guard."""
        _, _, _, historical_X, historical_Y = self._get_data(
            num_curves=5, num_progression=10, num_train=4
        )

        kernel = EmpiricalOneDimensionalKernel(X_full=historical_X, Y_full=historical_Y)
        # use_svd property getter
        self.assertIsInstance(kernel.use_svd, bool)
        self.assertEqual(kernel.use_svd, kernel._use_svd)

        # last_dim_is_batch is not supported
        with self.assertRaisesRegex(
            NotImplementedError, "last_dim_is_batch=True not supported"
        ):
            kernel.forward(historical_X, historical_X, last_dim_is_batch=True)

    def _test_unsupported_transforms(self) -> None:
        """Test that input_transform and outcome_transform raise UnsupportedError."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_curves=5, num_progression=10, num_train=6
        )

        # Test that input_transform raises UnsupportedError
        with self.assertRaisesRegex(
            UnsupportedError, "input_transform is not yet supported"
        ):
            EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                historical_X=historical_X,
                historical_Y=historical_Y,
                input_transform=Normalize(d=1),
            )

        # Test that outcome_transform raises UnsupportedError
        with self.assertRaisesRegex(
            UnsupportedError, "outcome_transform is not yet supported"
        ):
            EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                historical_X=historical_X,
                historical_Y=historical_Y,
                outcome_transform=Standardize(m=1),
            )

    def _test_ard_curve_selection(self) -> None:
        """Test that ARD correctly identifies target curves via sparsification."""
        torch.manual_seed(5678)
        num_curves = 6
        num_train = 4
        train_X, _, _, historical_X, historical_Y = self._get_data(
            num_curves=num_curves,
            num_progression=10,
            num_train=num_train,
        )

        for target_curve_idx in [0, 3]:
            with self.subTest(target_curve_idx=target_curve_idx):
                # Use replication of a single curve as target
                # historical_Y is now 3D: num_curves x num_progression x m (m=1)
                train_Y = historical_Y[target_curve_idx, :num_train, :]
                train_Yvar = torch.full_like(train_Y, 1e-10)  # very low noise

                ard_model = EmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    historical_X=historical_X,
                    historical_Y=historical_Y,
                    ard=True,
                )
                mll = ExactMarginalLogLikelihood(
                    model=ard_model, likelihood=ard_model.likelihood
                )
                fit_gpytorch_mll(mll)

                w = ard_model.covar_module.curve_weights
                # Target curve should have high weight
                self.assertGreater(w[target_curve_idx].item(), 0.9)
                # Other curves should be pruned
                for j in range(num_curves):
                    if j != target_curve_idx:
                        self.assertLess(w[j].item(), 1e-3)

                # ARD model should extrapolate accurately
                ard_post = ard_model.posterior(historical_X, observation_noise=False)
                full_Y = historical_Y[target_curve_idx, :, :]
                self.assertAllClose(ard_post.mean, full_Y, atol=1e-3, rtol=1e-3)

                # Non-ARD model should NOT extrapolate as accurately
                non_ard_model = EmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=torch.full_like(train_Y, 1e-10),
                    historical_X=historical_X,
                    historical_Y=historical_Y,
                    ard=False,
                )
                non_ard_post = non_ard_model.posterior(
                    historical_X, observation_noise=False
                )
                self.assertGreater((non_ard_post.mean - full_Y).abs().max(), 1e-3)

    def _test_mean_multi_output(self) -> None:
        """Test EmpiricalOneDimensionalMean with single and multi-output Y_full."""
        num_curves = 5
        num_progression = 10
        m = 3

        X_full = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Single-output (3D Y_full with m=1)
        Y_full_single = torch.randn(
            num_curves, num_progression, 1, device=self.device, dtype=self.dtype
        )
        mean_single = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_full_single)
        self.assertEqual(mean_single.mean_full.shape, (1, num_progression))

        # Multi-output (3D Y_full with m=3)
        Y_full_multi = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )
        mean_multi = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_full_multi)
        self.assertEqual(mean_multi.mean_full.shape, (m, num_progression))
        self.assertAllClose(mean_multi.mean_full, Y_full_multi.mean(dim=0).T)

        # Test forward shapes
        batch_shape = (2, 4)
        n = 7
        x = torch.rand(*batch_shape, n, 1, device=self.device, dtype=self.dtype)
        x = x * (X_full.max() - X_full.min()) + X_full.min()

        # Single-output forward
        y_single = mean_single(x)
        self.assertEqual(y_single.shape, (*batch_shape, n))

        # Multi-output forward (with SingleTaskGP-style batched input)
        x_batched = x.unsqueeze(-3).expand(*batch_shape, m, n, 1)
        y_multi = mean_multi(x_batched)
        self.assertEqual(y_multi.shape, (*batch_shape, m, n))

        # Verify each output matches single-output mean
        for i in range(m):
            Y_i = Y_full_multi[..., [i]]  # Keep 3D with m=1
            mean_i = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_i)
            y_i = mean_i(x)
            self.assertAllClose(y_multi[..., i, :], y_i, atol=1e-6, rtol=1e-4)

    def _test_kernel_multi_output(self) -> None:
        """Test EmpiricalOneDimensionalKernel produces independent covariances
        per output.
        """
        num_curves = 10
        num_progression = 20
        m = 3
        n = 5

        X_full = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        Y_full_3d = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )

        kernel = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full_3d)
        self.assertEqual(kernel.num_curves, num_curves)
        self.assertEqual(kernel.num_outputs, m)

        x = torch.linspace(0.1, 0.9, n, device=self.device, dtype=self.dtype)
        x = x.unsqueeze(-1)

        K = kernel.forward(x, x)
        self.assertEqual(K.shape, (m, n, n))

        # Each slice should be PSD and symmetric
        for i in range(m):
            K_i = K[i]
            self.assertAllClose(K_i, K_i.T, atol=1e-6)
            eigvals = torch.linalg.eigvalsh(K_i)
            self.assertTrue((eigvals >= -1e-6).all())

            # Should match single-output kernel on that output (use slicing to keep 3D)
            Y_i = Y_full_3d[..., [i]]  # Keep 3D with m=1
            kernel_i = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_i)
            K_i_expected = kernel_i.forward(x, x)
            self.assertAllClose(K[i], K_i_expected, atol=1e-7, rtol=1e-4)

        # Test diagonal
        K_diag = kernel.forward(x, x, diag=True)
        self.assertEqual(K_diag.shape, (m, n))
        for i in range(m):
            self.assertAllClose(K_diag[i], K[i].diag(), atol=1e-7)

        # Test batched multi-output input
        batch_shape = (2, 4)
        x_batch = torch.rand(*batch_shape, n, 1, device=self.device, dtype=self.dtype)
        x_batch = x_batch * 0.8 + 0.1
        x_batched_multi = x_batch.unsqueeze(-3).expand(*batch_shape, m, n, 1)

        K_batch = kernel.forward(x_batched_multi, x_batched_multi)
        self.assertEqual(K_batch.shape, (*batch_shape, m, n, n))

        K_diag_batch = kernel.forward(x_batched_multi, x_batched_multi, diag=True)
        self.assertEqual(K_diag_batch.shape, (*batch_shape, m, n))

        # Test ARD
        kernel_ard = EmpiricalOneDimensionalKernel(
            X_full=X_full, Y_full=Y_full_3d, ard=True
        )
        self.assertTrue(kernel_ard.ard)
        self.assertEqual(kernel_ard.curve_weights.shape, (num_curves,))
        self.assertEqual(kernel_ard.correction, 0)  # default

        # Test correction parameter is passed to compute_sample_covariance
        for correction in (0, 1, 2):
            kernel_corrected = EmpiricalOneDimensionalKernel(
                X_full=X_full, Y_full=Y_full_3d, correction=correction
            )
            self.assertEqual(kernel_corrected.correction, correction)
            with patch(
                "botorch.models.empirical_gps."
                "empirical_1d_gp.compute_sample_covariance",
                wraps=compute_sample_covariance,
            ) as mock_cov:
                kernel_corrected.forward(x, x)
            self.assertEqual(mock_cov.call_args.kwargs["correction"], correction)

    def _get_multi_output_data(
        self,
        num_curves: int = 8,
        num_progression: int = 20,
        num_train: int = 12,
        m: int = 3,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        """Generate multi-output test data for ModelListGP.

        Returns:
            Tuple of (historical_X, historical_Y, train_X, weights, train_Y_list).
        """
        historical_X = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        historical_Y = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )
        train_X = torch.linspace(
            0.05, 0.95, num_train, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Create training targets via interpolation
        weights = torch.randn(num_curves, device=self.device, dtype=self.dtype)
        weights = weights / weights.abs().sum()

        train_Y_list = []
        for i in range(m):
            curves_i = historical_Y[:, :, i]
            f_i = LinearInterpolation1D(historical_X.squeeze(-1), curves_i)
            curves_at_train = f_i(train_X.squeeze(-1))
            train_y_i = (weights.unsqueeze(-1) * curves_at_train).sum(dim=0)
            train_Y_list.append(train_y_i.unsqueeze(-1))

        return historical_X, historical_Y, train_X, weights, train_Y_list

    def _test_multi_output_gp(self) -> None:
        """Test multi-output GP with ModelListGP and direct 3D historical_Y."""
        torch.manual_seed(42)

        m = 3
        sigma = 1e-3
        historical_X, historical_Y, train_X, weights, train_Y_list = (
            self._get_multi_output_data(m=m)
        )

        # Create ModelListGP with individual models
        models = []
        for i in range(m):
            train_Y_i = train_Y_list[i]
            train_Yvar_i = torch.full_like(train_Y_i, sigma**2)
            model_i = EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y_i,
                train_Yvar=train_Yvar_i,
                historical_X=historical_X,
                historical_Y=historical_Y[..., [i]],  # Keep 3D with m=1
            )
            models.append(model_i)
        model_list = ModelListGP(*models)

        # Test posterior
        test_X = torch.linspace(
            0.1, 0.9, 5, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        # we'll use post to compare against the batched model
        post = model_list.posterior(test_X, observation_noise=False)
        self.assertEqual(post.mean.shape, (5, m))
        self.assertEqual(post.variance.shape, (5, m))
        self.assertTrue((post.variance >= 0).all())

        # Test direct multi-output GP with 3D historical_Y
        train_Y_multi = torch.stack([t.squeeze(-1) for t in train_Y_list], dim=-1)
        train_Yvar_multi = torch.full_like(train_Y_multi, sigma**2)

        model_direct = EmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y_multi,
            train_Yvar=train_Yvar_multi,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertEqual(model_direct.covar_module.num_outputs, m)
        self.assertEqual(model_direct.mean_module.num_outputs, m)

        # Predictions should match ModelListGP
        post_direct = model_direct.posterior(test_X, observation_noise=False)
        self.assertAllClose(post_direct.mean, post.mean, atol=1e-4, rtol=1e-3)

        # Test mismatched outputs error
        with self.assertRaisesRegex(
            ValueError, "Number of outputs in train_Y .* must match historical_Y"
        ):
            EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y_list[0],
                train_Yvar=torch.full_like(train_Y_list[0], sigma**2),
                historical_X=historical_X,
                historical_Y=historical_Y,
            )

    def _test_multi_output_ard(self) -> None:
        """Test ARD optimization with ModelListGP."""
        torch.manual_seed(42)

        m = 3
        sigma = 1e-3
        historical_X, historical_Y, train_X, weights, train_Y_list = (
            self._get_multi_output_data(m=m)
        )

        # Create ARD models
        models_ard = []
        for i in range(m):
            train_Y_i = train_Y_list[i]
            train_Yvar_i = torch.full_like(train_Y_i, sigma**2)
            model_i = EmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y_i,
                train_Yvar=train_Yvar_i,
                historical_X=historical_X,
                historical_Y=historical_Y[..., [i]],  # Keep 3D with m=1
                ard=True,
            )
            models_ard.append(model_i)
        model_list_ard = ModelListGP(*models_ard)

        # Fit with SumMarginalLogLikelihood
        mll = SumMarginalLogLikelihood(
            likelihood=model_list_ard.likelihood,
            model=model_list_ard,
        )
        fit_gpytorch_mll(mll)

        # Weights should have been updated
        for model in model_list_ard.models:
            self.assertTrue(model.covar_module.ard)
            curve_weights = model.covar_module.curve_weights
            self.assertFalse(
                torch.allclose(curve_weights, torch.ones_like(curve_weights)),
                "ARD weights should have been updated during fitting",
            )

    def _test_differentiability(self) -> None:
        """Test that GP mean and covariance are differentiable w.r.t. inputs."""
        torch.manual_seed(1234)
        num_curves = 5
        num_progression = 20

        X_full = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        Y_full = torch.randn(
            num_curves, num_progression, 1, device=self.device, dtype=self.dtype
        )

        mean_module = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_full)
        covar_module = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)

        # Test mean differentiability w.r.t. x
        x = torch.rand(5, 1, device=self.device, dtype=self.dtype) * 0.8 + 0.1
        x.requires_grad_(True)
        mean_module(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(x.grad.isnan().any())

        # Test covariance differentiability w.r.t. x1 and x2
        x1 = torch.rand(4, 1, device=self.device, dtype=self.dtype) * 0.8 + 0.1
        x2 = torch.rand(3, 1, device=self.device, dtype=self.dtype) * 0.8 + 0.1
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        covar_module(x1, x2).to_dense().sum().backward()
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)
        self.assertFalse(x1.grad.isnan().any())
        self.assertFalse(x2.grad.isnan().any())

        # Test full model posterior differentiability
        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_curves=5, num_progression=20, num_train=10
        )
        model = EmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, 1e-4),
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        test_X = torch.rand(3, 1, device=self.device, dtype=self.dtype) * 8 + 1
        test_X.requires_grad_(True)
        posterior = model.posterior(test_X, observation_noise=False)
        (posterior.mean.sum() + posterior.variance.sum()).backward()
        self.assertIsNotNone(test_X.grad)
        self.assertFalse(test_X.grad.isnan().any())

    def _test_svd_acceleration(self) -> None:
        """Test that the kernel produces identical values with and without SVD."""
        torch.manual_seed(42)

        for dtype in [torch.float32, torch.float64]:
            tol = 1e-4 if dtype == torch.float32 else 1e-10

            # Test case 1: num_curves > num_progression - default auto-enables SVD
            # Also verifies SVD produces equivalent results to non-SVD
            num_curves, num_progression = 300, 25
            historical_X = torch.linspace(
                1.0, 10.0, num_progression, dtype=dtype, device=self.device
            ).unsqueeze(-1)
            historical_Y = torch.randn(
                num_curves, num_progression, 1, dtype=dtype, device=self.device
            )

            kernel_default = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
            )
            kernel_explicit_svd = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                use_svd=True,
            )
            kernel_no_svd = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                use_svd=False,
            )

            n1, n2 = 10, 8
            x1 = torch.linspace(
                2.0, 9.0, n1, dtype=dtype, device=self.device
            ).unsqueeze(-1)
            x2 = torch.linspace(
                3.0, 8.0, n2, dtype=dtype, device=self.device
            ).unsqueeze(-1)

            K_default = kernel_default(x1, x2).to_dense()
            K_explicit_svd = kernel_explicit_svd(x1, x2).to_dense()
            K_no_svd = kernel_no_svd(x1, x2).to_dense()

            # Kernel squeezes m=1 dimension, so shape is (n1, n2) not (1, n1, n2)
            self.assertEqual(K_default.shape, (n1, n2))
            # Default should match explicit SVD=True (auto-enabled)
            self.assertAllClose(K_default, K_explicit_svd, atol=tol, rtol=tol)
            # All should produce equivalent results
            self.assertAllClose(K_default, K_no_svd, atol=tol, rtol=tol)

            # Also test diagonal computation
            K_default_diag = kernel_default(x1, diag=True)
            K_svd_diag = kernel_explicit_svd(x1, diag=True)
            K_no_svd_diag = kernel_no_svd(x1, diag=True)

            self.assertEqual(K_default_diag.shape, (n1,))
            self.assertAllClose(K_default_diag, K_svd_diag, atol=tol, rtol=tol)
            self.assertAllClose(K_svd_diag, K_no_svd_diag, atol=tol, rtol=tol)

            # Test case 2: num_curves <= num_progression - default does NOT use SVD
            num_curves, num_progression = 15, 50
            historical_Y = torch.randn(
                num_curves, num_progression, 1, dtype=dtype, device=self.device
            )
            historical_X = torch.linspace(
                1.0, 10.0, num_progression, dtype=dtype, device=self.device
            ).unsqueeze(-1)

            kernel_default = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
            )
            kernel_disabled = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                use_svd=False,
            )

            K_default = kernel_default(x1, x2).to_dense()
            K_disabled = kernel_disabled(x1, x2).to_dense()

            self.assertAllClose(K_default, K_disabled, atol=tol, rtol=tol)

            # Test case 3: with ard=True, default does NOT use SVD
            num_curves, num_progression = 300, 25
            historical_X = torch.linspace(
                1.0, 10.0, num_progression, dtype=dtype, device=self.device
            ).unsqueeze(-1)
            historical_Y = torch.randn(
                num_curves, num_progression, 1, dtype=dtype, device=self.device
            )

            kernel_ard_default = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                ard=True,
            )
            kernel_ard_no_svd = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                ard=True,
                use_svd=False,
            )

            K_ard_default = kernel_ard_default(x1, x2).to_dense()
            K_ard_no_svd = kernel_ard_no_svd(x1, x2).to_dense()

            self.assertAllClose(K_ard_default, K_ard_no_svd, atol=tol, rtol=tol)

            # Test case 4: explicit use_svd=True with ard=True is supported
            # (produces different prior but should still compute valid kernel)
            kernel_ard_with_svd = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                ard=True,
                use_svd=True,
            )
            K_ard_with_svd = kernel_ard_with_svd(x1, x2).to_dense()
            self.assertEqual(K_ard_with_svd.shape, (n1, n2))

            # Verify it's positive semi-definite
            K_self = kernel_ard_with_svd(x1, x1).to_dense()
            eigvals = torch.linalg.eigvalsh(K_self)
            self.assertTrue((eigvals >= -tol).all())

    # =========================================================================
    # Main public test method
    # =========================================================================

    def test_empirical_one_dimensional_gp(self) -> None:
        """Main test for EmpiricalOneDimensionalGP and related modules."""
        # Basic GP tests
        self._test_posterior_prediction()
        self._test_kernel_diagonal()

        # Likelihood tests
        self._test_likelihood_handling()

        # Input validation tests
        self._test_input_validation()
        self._test_unsupported_transforms()
        self._test_edge_cases()

        # ARD tests
        self._test_ard_curve_selection()

        # Multi-output tests
        self._test_mean_multi_output()
        self._test_kernel_multi_output()
        self._test_multi_output_gp()
        self._test_multi_output_ard()

        # Differentiability tests
        self._test_differentiability()

        # SVD acceleration tests
        self._test_svd_acceleration()
