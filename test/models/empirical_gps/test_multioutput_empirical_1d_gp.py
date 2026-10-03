#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MultiOutputEmpiricalOneDimensionalGP model."""

from __future__ import annotations

import math
import unittest

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.empirical_gps import (
    EmpiricalOneDimensionalMean,
    MultiOutputEmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalKernel,
    MultiOutputEmpiricalOneDimensionalMean,
)
from botorch.models.empirical_gps.empirical_1d_gp import BaseAugmentedEmpiricalKernel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from torch import Tensor


class TestMultiOutputEmpiricalOneDimensionalGP(BotorchTestCase):
    """Tests for MultiOutputEmpiricalOneDimensionalGP and related modules."""

    # Use double precision for numerical stability in GP posterior computations
    dtype = torch.float64

    def _get_data(
        self,
        num_curves: int = 10,
        num_progression: int = 20,
        num_train: int = 12,
        num_outputs: int = 2,
        batch_shape: tuple[int, ...] = (),
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate test data for multi-output empirical learning curve GP.

        Args:
            num_curves: Number of historical curves.
            num_progression: Number of progression values.
            num_train: Number of training progression values.
            num_outputs: Number of output dimensions (m).
            batch_shape: Batch shape for outputs.

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
    # Private test helpers - Mean Module
    # =========================================================================

    def _test_mean_module(self) -> None:
        """Test MultiOutputEmpiricalOneDimensionalMean shapes, values, and ordering."""
        num_curves = 8
        num_progression = 15
        m = 3

        historical_X = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        historical_Y = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )

        mean_module = MultiOutputEmpiricalOneDimensionalMean(
            X_full=historical_X,
            Y_full=historical_Y,
        )

        # Verify mean_full shape and values
        self.assertEqual(mean_module.mean_full.shape, (m, num_progression))
        self.assertEqual(mean_module.num_outputs, m)
        expected_mean = historical_Y.mean(dim=0).T  # m x num_progression
        self.assertAllClose(mean_module.mean_full, expected_mean)

        # Test forward shapes - flattened output
        n = 7
        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        x = x * (historical_X.max() - historical_X.min()) + historical_X.min()
        y = mean_module(x)
        self.assertEqual(y.shape, (n * m,))

        # Test consistency with EmpiricalOneDimensionalMean (interleaved format)
        single_output_mean = EmpiricalOneDimensionalMean(
            X_full=historical_X, Y_full=historical_Y
        )
        y_single = single_output_mean(x)  # m x n
        y_single_interleaved = y_single.T.reshape(-1)
        self.assertAllClose(y, y_single_interleaved, atol=1e-6)

        # Test ordering at exact historical points (no interpolation)
        y_exact = mean_module(historical_X)
        expected_mean_flat = historical_Y.mean(dim=0).reshape(-1)  # num_progression * m
        self.assertAllClose(y_exact, expected_mean_flat, atol=1e-6)

        # Test with batch shape
        batch_shape = (2, 4)
        x_batch = torch.rand(*batch_shape, n, 1, device=self.device, dtype=self.dtype)
        x_batch = (
            x_batch * (historical_X.max() - historical_X.min()) + historical_X.min()
        )
        y_batch = mean_module(x_batch)
        self.assertEqual(y_batch.shape, (*batch_shape, n * m))

        # Test invalid input raises error
        with self.assertRaisesRegex(ValueError, "Expected Y_full to be 3-dim"):
            MultiOutputEmpiricalOneDimensionalMean(
                X_full=historical_X, Y_full=historical_Y[..., 0]
            )

    def _test_mean_module_single_output(self) -> None:
        """Test MultiOutputEmpiricalOneDimensionalMean with m=1 (single output).

        This is a regression test for a bug where movedim(-2, -1) failed when
        the parent class squeezed the m=1 dimension.
        """
        num_curves = 8
        num_progression = 15
        m = 1  # Single output case

        historical_X = torch.linspace(
            0, 1, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        historical_Y = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )

        mean_module = MultiOutputEmpiricalOneDimensionalMean(
            X_full=historical_X,
            Y_full=historical_Y,
        )

        # Verify mean_full shape and values
        self.assertEqual(mean_module.mean_full.shape, (m, num_progression))
        self.assertEqual(mean_module.num_outputs, m)

        # Test forward shapes - for m=1, output should still be (n * m) = n
        n = 7
        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        x = x * (historical_X.max() - historical_X.min()) + historical_X.min()
        y = mean_module(x)
        self.assertEqual(y.shape, (n * m,))  # = (n,)

        # Test with batch shape
        batch_shape = (2, 4)
        x_batch = torch.rand(*batch_shape, n, 1, device=self.device, dtype=self.dtype)
        x_batch = (
            x_batch * (historical_X.max() - historical_X.min()) + historical_X.min()
        )
        y_batch = mean_module(x_batch)
        self.assertEqual(y_batch.shape, (*batch_shape, n * m))  # = (*batch_shape, n)

    # =========================================================================
    # Private test helpers - Kernel Module
    # =========================================================================

    def _test_kernel_shape(self) -> None:
        """Test MultiOutputEmpiricalOneDimensionalKernel output shapes."""
        num_curves = 10
        num_progression = 20
        m = 2

        historical_X = torch.linspace(
            1.0, 10.0, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)
        historical_Y = torch.randn(
            num_curves, num_progression, m, device=self.device, dtype=self.dtype
        )

        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X,
            Y_full=historical_Y,
        )

        self.assertEqual(kernel.num_outputs, m)
        self.assertEqual(kernel.num_curves, num_curves)

        # Test kernel output shape at a subset of points
        n = 5
        x = historical_X[:n]
        K = kernel.forward(x, x)

        # For multi-output, kernel should be (n*m) x (n*m)
        self.assertEqual(K.shape, (n * m, n * m))

        # Test with different x1 and x2
        n1, n2 = 5, 8
        x1 = historical_X[:n1]
        x2 = historical_X[:n2]
        K_rect = kernel.forward(x1, x2)
        self.assertEqual(K_rect.shape, (n1 * m, n2 * m))

        # Test diagonal
        K_diag = kernel.forward(x, x, diag=True)
        self.assertEqual(K_diag.shape, (n * m,))
        self.assertAllClose(K_diag, K.diag(), atol=1e-6)

    def _test_kernel_psd(self) -> None:
        """Test that kernel is positive semi-definite."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()

        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X,
            Y_full=historical_Y,
        )

        n = 8
        x = historical_X[:n]
        K = kernel.forward(x, x)

        # Verify kernel is positive semi-definite
        eigvals = torch.linalg.eigvalsh(K)
        self.assertTrue((eigvals >= -1e-6).all())

        # Verify symmetry
        self.assertAllClose(K, K.T, atol=1e-10)

    def _test_kernel_perfect_correlation(self) -> None:
        """Test that perfectly correlated outputs produce expected covariance."""
        num_curves = 10
        num_progression = 20
        m = 2

        historical_X = torch.linspace(
            1.0, 10.0, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Create Y_full where outputs are identical
        Y_single = torch.randn(
            num_curves, num_progression, device=self.device, dtype=self.dtype
        )
        historical_Y = torch.stack([Y_single, Y_single], dim=-1)

        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X,
            Y_full=historical_Y,
        )

        n = 5
        x = historical_X[:n]
        K = kernel.forward(x, x)

        # For each input, cross-output covariance should equal variance
        # The kernel ordering is: (x_0,t_0), (x_0,t_1), (x_1,t_0), (x_1,t_1), ...
        for i in range(n):
            idx_0 = i * m
            idx_1 = i * m + 1
            var_00 = K[idx_0, idx_0]
            var_11 = K[idx_1, idx_1]
            cov_01 = K[idx_0, idx_1]
            cov_10 = K[idx_1, idx_0]

            # For perfectly correlated outputs, these should all be equal
            self.assertAllClose(var_00, var_11, atol=1e-10)
            self.assertAllClose(var_00, cov_01, atol=1e-10)
            self.assertAllClose(var_00, cov_10, atol=1e-10)

    def _test_kernel_ard(self) -> None:
        """Test kernel with ARD enabled."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()

        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X,
            Y_full=historical_Y,
            ard=True,
        )

        self.assertTrue(kernel.ard)
        # num_curves should equal the number of historical curves (first dim of Y_full)
        self.assertEqual(kernel.num_curves, historical_Y.shape[0])
        self.assertEqual(kernel.curve_weights.shape, (kernel.num_curves,))

        # Verify kernel still computes valid output
        n = 5
        x = historical_X[:n]
        K = kernel.forward(x, x)
        self.assertEqual(K.shape, (n * kernel.num_outputs, n * kernel.num_outputs))

    def _test_kernel_svd(self) -> None:
        """Test kernel with SVD acceleration in various configurations."""
        num_progression = 15
        m = 2

        historical_X = torch.linspace(
            1.0, 10.0, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Test 1: Many curves (num_curves > num_progression * m) - default uses SVD
        num_curves_many = 100
        historical_Y_many = torch.randn(
            num_curves_many, num_progression, m, device=self.device, dtype=self.dtype
        )
        vectorized_dim = num_progression * m

        kernel_default = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_many
        )
        self.assertTrue(kernel_default.use_svd)
        self.assertEqual(kernel_default._effective_num_curves, vectorized_dim)

        # Test explicit use_svd=False
        kernel_no_svd = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_many, use_svd=False
        )
        self.assertFalse(kernel_no_svd.use_svd)
        self.assertEqual(kernel_no_svd._effective_num_curves, num_curves_many)

        # Verify both produce same covariance
        n = 5
        x = historical_X[:n]
        self.assertAllClose(
            kernel_default.forward(x, x), kernel_no_svd.forward(x, x), atol=1e-6
        )

        # Test 2: Few curves (num_curves < num_progression * m) - default no SVD
        num_curves_few = 10
        historical_Y_few = torch.randn(
            num_curves_few, num_progression, m, device=self.device, dtype=self.dtype
        )

        kernel_few_default = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_few
        )
        self.assertFalse(kernel_few_default.use_svd)

        kernel_few_svd = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_few, use_svd=True
        )
        self.assertTrue(kernel_few_svd.use_svd)
        self.assertEqual(kernel_few_svd._effective_num_curves, num_curves_few)
        self.assertAllClose(
            kernel_few_default.forward(x, x), kernel_few_svd.forward(x, x), atol=1e-6
        )

        # Test 3: SVD with ARD - default should NOT use SVD
        kernel_ard_default = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_many, ard=True
        )
        self.assertFalse(kernel_ard_default.use_svd)

        # But explicit use_svd=True should work with ARD
        kernel_ard_svd = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y_many, ard=True, use_svd=True
        )
        self.assertTrue(kernel_ard_svd.use_svd)
        self.assertTrue(kernel_ard_svd.ard)
        self.assertEqual(
            kernel_ard_svd.curve_weights.shape, (kernel_ard_svd._effective_num_curves,)
        )

    # =========================================================================
    # Private test helpers - GP Model
    # =========================================================================

    def _test_model_instantiation(self) -> None:
        """Test basic model instantiation."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()
        train_Yvar = torch.full_like(train_Y, 1e-6)

        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        self.assertEqual(model.num_outputs, historical_Y.shape[-1])
        # Check that base modules are accessible
        self.assertIsInstance(model._base_mean, MultiOutputEmpiricalOneDimensionalMean)
        self.assertIsInstance(
            model._base_kernel, MultiOutputEmpiricalOneDimensionalKernel
        )

    def _test_posterior_shape(self) -> None:
        """Test posterior prediction shapes with various batch shapes."""
        for batch_shape in ((), (2,), (2, 3)):
            with self.subTest(batch_shape=batch_shape):
                train_X, train_Y, _, historical_X, historical_Y = self._get_data(
                    batch_shape=batch_shape
                )
                m = historical_Y.shape[-1]
                train_Yvar = torch.full_like(train_Y, 1e-6)

                model = MultiOutputEmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    historical_X=historical_X,
                    historical_Y=historical_Y,
                )

                # Test posterior at new points
                q = 5
                test_X = historical_X[: train_X.shape[-2] + q][train_X.shape[-2] :]
                posterior = model.posterior(test_X)

                # Posterior mean should have shape [batch_shape x] q x m
                expected_mean_shape = (*batch_shape, q, m) if batch_shape else (q, m)
                self.assertEqual(posterior.mean.shape, expected_mean_shape)
                # Posterior variance should have shape [batch_shape x] q x m
                self.assertEqual(posterior.variance.shape, expected_mean_shape)
                # Variance should be non-negative
                self.assertTrue((posterior.variance >= 0).all())

    def _test_posterior_shape_single_output(self) -> None:
        """Test posterior prediction shapes with m=1 (single output).

        This is a regression test for a bug where the model failed with m=1
        because of dimension mismatch in the mean module.
        """
        # Test with m=1 to ensure single-output case works
        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_outputs=1,  # Single output
        )
        m = historical_Y.shape[-1]
        self.assertEqual(m, 1)  # Verify we're testing m=1

        train_Yvar = torch.full_like(train_Y, 1e-6)

        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        # Test posterior at new points
        q = 5
        test_X = historical_X[: train_X.shape[-2] + q][train_X.shape[-2] :]
        posterior = model.posterior(test_X)

        # Posterior mean should have shape q x m = q x 1
        expected_mean_shape = (q, m)
        self.assertEqual(posterior.mean.shape, expected_mean_shape)
        # Posterior variance should have shape q x m = q x 1
        self.assertEqual(posterior.variance.shape, expected_mean_shape)
        # Variance should be non-negative
        self.assertTrue((posterior.variance >= 0).all())

    def _test_posterior_prediction(self) -> None:
        """Test posterior predictions on training data with various batch shapes."""
        torch.manual_seed(1234)
        for batch_shape in ((), (2,)):
            with self.subTest(batch_shape=batch_shape):
                # Use more curves for better-conditioned covariance
                num_curves = 50
                num_progression = 20
                num_train = 15
                m = 2

                train_X, train_Y, _, historical_X, historical_Y = self._get_data(
                    num_curves=num_curves,
                    num_progression=num_progression,
                    num_train=num_train,
                    num_outputs=m,
                    batch_shape=batch_shape,
                )
                train_Yvar = torch.full_like(train_Y, 1e-6)

                model = MultiOutputEmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    historical_X=historical_X,
                    historical_Y=historical_Y,
                )

                # Posterior at training points should be close to training data
                posterior = model.posterior(
                    historical_X[:num_train], observation_noise=False
                )
                # For batched case, compare first batch element's posterior
                # to first batch element's training data
                if batch_shape:
                    expected_Y = train_Y[(0,) * len(batch_shape)]
                    posterior_mean_first = posterior.mean[(0,) * len(batch_shape)]
                    rmse = (posterior_mean_first - expected_Y).square().mean().sqrt()
                else:
                    expected_Y = train_Y
                    rmse = (posterior.mean - expected_Y).square().mean().sqrt()
                self.assertLess(rmse.item(), 0.1)

    def _test_posterior_covariance_structure(self) -> None:
        """Test multi-output GP covariance structure with perfectly correlated outputs.

        This test verifies:
        1. Samples are approximately equal for identical outputs
        2. Cross-output covariance equals diagonal (self) covariance
        3. Rank of covariance matrix is reduced (≈q, not q*m)

        This demonstrates the key difference from independent batched GPs.
        """
        torch.manual_seed(42)
        num_curves = 10
        num_progression = 20
        num_train = 12
        m = 2
        q = 4  # Number of test points

        historical_X = torch.linspace(
            1.0, 10.0, num_progression, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)

        # Create perfectly correlated outputs (identical)
        Y_single = torch.randn(
            num_curves, num_progression, device=self.device, dtype=self.dtype
        )
        historical_Y = torch.stack([Y_single, Y_single], dim=-1)

        train_X = historical_X[:num_train]
        train_Y = historical_Y.mean(dim=0)[:num_train]
        train_Yvar = torch.full_like(train_Y, 1e-6)

        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        # Get posterior at test points
        test_X = historical_X[num_train : num_train + q]
        posterior = model.posterior(test_X)

        # Test 1: Samples should be approximately equal across outputs
        num_samples = 100
        samples = posterior.rsample(torch.Size([num_samples]))
        self.assertEqual(samples.shape, (num_samples, q, m))
        self.assertAllClose(samples[..., 0], samples[..., 1], atol=1e-3)

        # Get the full posterior covariance matrix (q*m x q*m)
        cov = posterior.distribution.covariance_matrix
        self.assertEqual(cov.shape, (q * m, q * m))

        # Test 2: Cross-output covariance should equal diagonal blocks
        # For interleaved format: indices 0,2,4,... are output 0; 1,3,5,... are output 1
        idx_out0 = torch.arange(0, q * m, m)
        idx_out1 = torch.arange(1, q * m, m)
        cross_cov = cov[idx_out0][:, idx_out1]
        diag_cov_0 = cov[idx_out0][:, idx_out0]
        self.assertAllClose(cross_cov, diag_cov_0, rtol=1e-3)

        # Test 3: Rank should be approximately q (not q*m)
        eigvals = torch.linalg.eigvalsh(cov)
        threshold = eigvals.max() * 1e-6
        effective_rank = (eigvals > threshold).sum().item()
        self.assertLessEqual(effective_rank, q + 1)
        self.assertGreaterEqual(effective_rank, q - 1)

    def _test_posterior_independent_outputs(self) -> None:
        """Test samples from GP with independent outputs."""
        torch.manual_seed(42)
        num_curves = 10
        num_progression = 20
        num_train = 12
        m = 2

        train_X, train_Y, _, historical_X, historical_Y = self._get_data(
            num_curves=num_curves,
            num_progression=num_progression,
            num_train=num_train,
            num_outputs=m,
        )
        train_Yvar = torch.full_like(train_Y, 1e-6)

        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        test_X = historical_X[num_train : num_train + 3]
        posterior = model.posterior(test_X)

        num_samples = 100
        samples = posterior.rsample(torch.Size([num_samples]))

        # For independent outputs, samples should NOT be perfectly correlated
        for i in range(test_X.shape[0]):
            samples_0 = samples[:, i, 0]
            samples_1 = samples[:, i, 1]
            corr_matrix = torch.corrcoef(torch.stack([samples_0, samples_1]))
            correlation = corr_matrix[0, 1].abs()
            # Correlation should be much lower than 0.99
            self.assertLess(correlation.item(), 0.95)

    def _test_likelihood_handling(self) -> None:
        """Test likelihood inference and custom likelihood support."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()

        # Test 1: Fixed noise (train_Yvar provided) -> FixedNoiseGaussianLikelihood
        train_Yvar = torch.full_like(train_Y, 0.01)
        model_fixed = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertIsInstance(model_fixed.likelihood, FixedNoiseGaussianLikelihood)

        # Test 2: Inferred noise (train_Yvar=None) -> GaussianLikelihood
        model_inferred = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertIsInstance(model_inferred.likelihood, GaussianLikelihood)

        # Test 3: Custom likelihood passed to constructor
        custom_likelihood = GaussianLikelihood(noise_constraint=GreaterThan(0.01))
        model_custom = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
            likelihood=custom_likelihood,
        )
        self.assertIs(model_custom.likelihood, custom_likelihood)

        # Verify posterior can be computed with custom likelihood
        test_X = historical_X[: train_X.shape[0] + 3][train_X.shape[0] :]
        posterior = model_custom.posterior(test_X)
        self.assertEqual(posterior.mean.shape, (3, model_custom.num_outputs))

    def _test_input_validation(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()
        train_Yvar = torch.full_like(train_Y, 1e-6)

        # Test 1: Invalid historical_Y dimension for GP model
        with self.assertRaisesRegex(ValueError, "Expected historical_Y to be 3-dim"):
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                historical_X=historical_X,
                historical_Y=historical_Y[..., 0],  # 2D tensor
            )

        # Test 2: Invalid covar_module type
        with self.assertRaisesRegex(ValueError, "covar_module must be an instance of"):
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                historical_X=historical_X,
                historical_Y=historical_Y,
                covar_module=RBFKernel(),
            )

        # Test 3: Invalid Y_full dimension for kernel
        with self.assertRaisesRegex(ValueError, "Expected Y_full to be 3-dim"):
            MultiOutputEmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y[..., 0],  # 2D tensor
            )

    def _test_unsupported_transforms(self) -> None:
        """Test that input_transform and outcome_transform raise UnsupportedError."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()

        # Test that input_transform raises UnsupportedError
        with self.assertRaisesRegex(
            UnsupportedError, "input_transform is not yet supported"
        ):
            MultiOutputEmpiricalOneDimensionalGP(
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
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                historical_X=historical_X,
                historical_Y=historical_Y,
                outcome_transform=Standardize(m=train_Y.shape[-1]),
            )

    def _test_ard(self) -> None:
        """Test model with ARD enabled."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()
        train_Yvar = torch.full_like(train_Y, 1e-6)

        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
            ard=True,
        )

        self.assertTrue(model._base_kernel.ard)
        self.assertEqual(
            model._base_kernel.curve_weights.shape, (model._base_kernel.num_curves,)
        )

        # Verify posterior can still be computed
        test_X = historical_X[: train_X.shape[0] + 3][train_X.shape[0] :]
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, (3, model.num_outputs))

    def _test_differentiability(self) -> None:
        """Test that GP mean and covariance are differentiable w.r.t. inputs."""
        torch.manual_seed(1234)
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()

        # Test mean module differentiability
        mean_module = MultiOutputEmpiricalOneDimensionalMean(
            X_full=historical_X, Y_full=historical_Y
        )
        x = (
            torch.rand(5, 1, device=self.device, dtype=self.dtype)
            * (historical_X.max() - historical_X.min())
            + historical_X.min()
        )
        x.requires_grad_(True)
        mean_module(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(x.grad.isnan().any())

        # Test kernel differentiability
        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y
        )
        x1 = (
            torch.rand(4, 1, device=self.device, dtype=self.dtype)
            * (historical_X.max() - historical_X.min())
            + historical_X.min()
        )
        x2 = (
            torch.rand(3, 1, device=self.device, dtype=self.dtype)
            * (historical_X.max() - historical_X.min())
            + historical_X.min()
        )
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        kernel.forward(x1, x2).sum().backward()
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)
        self.assertFalse(x1.grad.isnan().any())
        self.assertFalse(x2.grad.isnan().any())

        # Test full model posterior differentiability
        train_Yvar = torch.full_like(train_Y, 1e-4)
        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )

        test_X = (
            torch.rand(3, 1, device=self.device, dtype=self.dtype)
            * (historical_X.max() - historical_X.min())
            + historical_X.min()
        )
        test_X.requires_grad_(True)
        posterior = model.posterior(test_X, observation_noise=False)
        (posterior.mean.sum() + posterior.variance.sum()).backward()
        self.assertIsNotNone(test_X.grad)
        self.assertFalse(test_X.grad.isnan().any())

    # =========================================================================
    # Main public test method
    # =========================================================================

    def _test_coverage_gaps(self) -> None:
        """Cover custom modules, ARD mismatch, last_dim_is_batch, observation
        noise branches, and posterior_transform."""
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()
        m = historical_Y.shape[-1]
        train_Yvar = torch.full_like(train_Y, 1e-6)

        # Kernel last_dim_is_batch is unsupported.
        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y
        )
        with self.assertRaisesRegex(UnsupportedError, "last_dim_is_batch"):
            kernel.forward(historical_X[:3], historical_X[:3], last_dim_is_batch=True)

        # Custom mean_module and covar_module are used as-is.
        custom_mean = MultiOutputEmpiricalOneDimensionalMean(
            X_full=historical_X, Y_full=historical_Y
        )
        custom_kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y
        )
        model_custom = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
            mean_module=custom_mean,
            covar_module=custom_kernel,
        )
        self.assertIs(model_custom._base_mean, custom_mean)
        self.assertIs(model_custom._base_kernel, custom_kernel)

        # ARD mismatch between `ard` and a provided covar_module raises.
        with self.assertRaisesRegex(
            ValueError, "`ard` argument must equal `covar_module.ard`"
        ):
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                historical_X=historical_X,
                historical_Y=historical_Y,
                covar_module=MultiOutputEmpiricalOneDimensionalKernel(
                    X_full=historical_X, Y_full=historical_Y, ard=False
                ),
                ard=True,
            )

        q = 4
        test_X = historical_X[:q]

        # observation_noise=True with fixed (train_Yvar) noise.
        model_fixed = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        post_noisy = model_fixed.posterior(test_X, observation_noise=True)
        post_clean = model_fixed.posterior(test_X, observation_noise=False)
        self.assertTrue((post_noisy.variance >= post_clean.variance - 1e-9).all())

        # observation_noise=True with inferred GaussianLikelihood.
        model_inferred = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertIsInstance(model_inferred.likelihood, GaussianLikelihood)
        post_inferred = model_inferred.posterior(test_X, observation_noise=True)
        self.assertEqual(post_inferred.mean.shape, (q, m))

        # observation_noise=True with a non-Gaussian likelihood and no train_Yvar
        # cannot be honored, so it must raise rather than silently return a
        # noiseless posterior. observation_noise=False on the same model is still
        # a valid request and must keep working.
        n_train = train_X.shape[-2]
        fixed_like = FixedNoiseGaussianLikelihood(
            noise=torch.full((n_train * m,), 1e-6, dtype=self.dtype, device=self.device)
        )
        model_zero = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=historical_X,
            historical_Y=historical_Y,
            likelihood=fixed_like,
        )
        self.assertNotIsInstance(model_zero.likelihood, GaussianLikelihood)
        with self.assertRaisesRegex(UnsupportedError, "observation_noise=True"):
            model_zero.posterior(test_X, observation_noise=True)
        model_zero.posterior(test_X, observation_noise=False)

        # Broadcast shorthands promised by the docstring: a per-output ``(m,)``
        # and a per-point ``(q, 1)`` must both broadcast onto the (q, m) grid.
        # Before the fix these raised, because the implementation reshaped
        # instead of broadcasting.
        clean_var = model_fixed.posterior(test_X, observation_noise=False).variance
        for shape in ((m,), (q, 1)):
            noise_b = torch.full(shape, 0.05, dtype=self.dtype, device=self.device)
            var_b = model_fixed.posterior(test_X, observation_noise=noise_b).variance
            self.assertEqual(var_b.shape, clean_var.shape)
            self.assertTrue((var_b > clean_var).all())

        # Tensor-valued observation_noise: a scalar tensor adds its value, and a
        # per-point (q x m) tensor adds per-point variance. Neither raises the
        # ambiguous-boolean error, and the requested values are not discarded.
        base_var = model_fixed.posterior(test_X, observation_noise=False).variance
        scalar_noise = torch.tensor(0.5, dtype=self.dtype, device=self.device)
        post_scalar = model_fixed.posterior(test_X, observation_noise=scalar_noise)
        self.assertAllClose(post_scalar.variance, base_var + 0.5)

        per_point_noise = (
            torch.arange(1, q * m + 1, dtype=self.dtype, device=self.device).reshape(
                q, m
            )
            * 0.01
        )
        post_pp = model_fixed.posterior(test_X, observation_noise=per_point_noise)
        self.assertAllClose(post_pp.variance, base_var + per_point_noise)

        # posterior_transform is applied (reduces to a single scalar output).
        pt = ScalarizedPosteriorTransform(
            weights=torch.ones(m, dtype=self.dtype, device=self.device)
        )
        post_transformed = model_fixed.posterior(test_X, posterior_transform=pt)
        self.assertEqual(post_transformed.mean.shape[-1], 1)

        # output_indices is not supported -> raises rather than silently ignoring.
        with self.assertRaisesRegex(UnsupportedError, "output_indices"):
            model_fixed.posterior(test_X, output_indices=[0])

        # The kernel `correction` param rescales the covariance by
        # num_curves / (num_curves - correction), matching the other kernels.
        k0 = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y, ard=False, correction=0
        )
        k1 = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=historical_X, Y_full=historical_Y, ard=False, correction=1
        )
        nc = historical_Y.shape[-3]
        K0 = k0.forward(test_X, test_X)
        K1 = k1.forward(test_X, test_X)
        self.assertAllClose(K1, K0 * (nc / (nc - 1)))

        # A correction that would make the denominator non-positive is rejected.
        with self.assertRaisesRegex(ValueError, "must be < num_curves"):
            MultiOutputEmpiricalOneDimensionalKernel(
                X_full=historical_X, Y_full=historical_Y, ard=False, correction=nc
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_observation_noise_follows_query_tensor(self) -> None:
        """`observation_noise=True` must read the noise onto the query tensor.

        The constructor pins the likelihood to the *training* device, but the bool
        branch of `posterior` multiplies the noise by an identity built on the query
        tensor. Reading the noise at its construction-time device therefore raises a
        device-mismatch `RuntimeError` for a model built on CPU and evaluated on CUDA.

        This needs two devices, so it is skipped on single-device hosts. A dtype-based
        stand-in was tried and rejected: the posterior takes its dtype from the model's
        parameters, so it reads float64 whether or not the fix is present and cannot
        discriminate. Better to skip honestly than to keep a test that always passes.
        """
        train_X, train_Y, _, historical_X, historical_Y = self._get_data()
        train_Yvar = torch.full_like(train_Y, 1e-6)

        for use_train_yvar in (True, False):
            with self.subTest(train_Yvar=use_train_yvar):
                # Built entirely on CPU, so the likelihood noise is pinned to CPU.
                model = MultiOutputEmpiricalOneDimensionalGP(
                    train_X=train_X.cpu(),
                    train_Y=train_Y.cpu(),
                    train_Yvar=train_Yvar.cpu() if use_train_yvar else None,
                    historical_X=historical_X.cpu(),
                    historical_Y=historical_Y.cpu(),
                )
                model.to("cuda")
                cuda_X = train_X[:4].to("cuda")
                posterior = model.posterior(cuda_X, observation_noise=True)
                self.assertEqual(posterior.variance.device.type, "cuda")

    def test_multioutput_empirical_learning_curve_gp(self) -> None:
        """Main test for MultiOutputEmpiricalOneDimensionalGP and related modules."""
        # Mean module tests
        self._test_mean_module()
        self._test_mean_module_single_output()

        # Kernel tests
        self._test_kernel_shape()
        self._test_kernel_psd()
        self._test_kernel_perfect_correlation()
        self._test_kernel_ard()
        self._test_kernel_svd()

        # GP model tests
        self._test_model_instantiation()
        self._test_posterior_shape()
        self._test_posterior_shape_single_output()
        self._test_posterior_prediction()
        self._test_posterior_covariance_structure()
        self._test_posterior_independent_outputs()

        # Likelihood and validation tests
        self._test_likelihood_handling()
        self._test_input_validation()
        self._test_unsupported_transforms()

        # ARD tests
        self._test_ard()

        # Differentiability tests
        self._test_differentiability()

        # Coverage gap tests
        self._test_coverage_gaps()


class TestMultiOutputShrinkage(BotorchTestCase):
    """Tests for base-kernel shrinkage wiring in the multi-output model."""

    def test_base_covar_module(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        torch.manual_seed(0)
        n_prog, n_curves, m = 8, 6, 2
        Xg = torch.linspace(0.0, 1.0, n_prog, **tkwargs).unsqueeze(-1)
        hist_Y = torch.stack(
            [
                torch.stack(
                    [
                        torch.sin(3.0 * Xg).squeeze(-1)
                        + 0.1 * torch.randn(n_prog, **tkwargs)
                        for _ in range(m)
                    ],
                    dim=-1,
                )
                for _ in range(n_curves)
            ]
        )  # num_curves x num_progression x m
        train_X = Xg[:4]
        train_Y = torch.cat(
            [torch.sin(3.0 * train_X), torch.cos(3.0 * train_X)], dim=-1
        )
        # A base kernel operating on the expanded (n*m, 1) inputs.
        base = ScaleKernel(RBFKernel()).to(**tkwargs)
        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=Xg,
            historical_Y=hist_Y,
            base_covar_module=base,
        )
        # Additive combination: empirical + base; base params fit iff requires_grad.
        self.assertIsInstance(model.covar_module, BaseAugmentedEmpiricalKernel)
        self.assertIs(model.covar_module.base_kernel, base)
        self.assertTrue(model.covar_module.base_kernel.raw_outputscale.requires_grad)
        model.eval()
        with torch.no_grad():
            post = model.posterior(Xg)
        self.assertEqual(post.mean.shape[-2], n_prog)


class TestCrossOutputShrinkage(BotorchTestCase):
    """Tests for the learnable cross-output shrinkage parameter ``rho``."""

    def _make_data(self, num_curves=8, num_progression=6, m=3, seed=0):
        torch.manual_seed(seed)
        X_full = torch.linspace(0, 1, num_progression, dtype=torch.double).unsqueeze(-1)
        Y_full = torch.randn(num_curves, num_progression, m, dtype=torch.double)
        return X_full, Y_full

    def _kernels(self, X_full, Y_full):
        plain = MultiOutputEmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)
        shrunk = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=X_full, Y_full=Y_full, learn_cross_output_shrinkage=True
        )
        return plain, shrunk

    def test_disabled_by_default(self) -> None:
        X_full, Y_full = self._make_data()
        kernel = MultiOutputEmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)
        self.assertIsNone(kernel.rho)
        # No new parameters, so existing state_dicts stay loadable.
        self.assertNotIn("rho", dict(kernel.named_parameters()))

    def test_rho_one_is_exact_noop(self) -> None:
        # The backward-compatibility guarantee: enabling the flag without moving
        # rho must not perturb any existing model.
        X_full, Y_full = self._make_data()
        plain, shrunk = self._kernels(X_full, Y_full)
        self.assertAlmostEqual(float(shrunk.rho), 1.0)
        K_plain = plain.forward(X_full, X_full)
        K_shrunk = shrunk.forward(X_full, X_full)
        self.assertTrue(torch.equal(K_plain, K_shrunk))

    def test_rho_zero_block_diagonalizes(self) -> None:
        X_full, Y_full = self._make_data()
        plain, shrunk = self._kernels(X_full, Y_full)
        with torch.no_grad():
            shrunk.rho.fill_(0.0)
        K_plain = plain.forward(X_full, X_full)
        K_shrunk = shrunk.forward(X_full, X_full)

        m = shrunk.num_outputs
        idx = torch.arange(K_plain.shape[-1])
        cross = idx.unsqueeze(-1) % m != idx.unsqueeze(-2) % m
        # Cross-output entries are exactly zero...
        self.assertTrue(torch.all(K_shrunk[cross] == 0.0))
        # ...and same-output entries are untouched.
        self.assertTrue(torch.equal(K_shrunk[~cross], K_plain[~cross]))

    def test_intermediate_rho_scales_only_cross_blocks(self) -> None:
        X_full, Y_full = self._make_data()
        plain, shrunk = self._kernels(X_full, Y_full)
        rho = 0.37
        with torch.no_grad():
            shrunk.rho.fill_(rho)
        K_plain = plain.forward(X_full, X_full)
        K_shrunk = shrunk.forward(X_full, X_full)

        m = shrunk.num_outputs
        idx = torch.arange(K_plain.shape[-1])
        cross = idx.unsqueeze(-1) % m != idx.unsqueeze(-2) % m
        self.assertAllClose(K_shrunk[cross], rho * K_plain[cross])
        self.assertTrue(torch.equal(K_shrunk[~cross], K_plain[~cross]))

    def test_psd_across_rho(self) -> None:
        # PSD is guaranteed by the Schur product theorem on [0, 1]; verify it.
        X_full, Y_full = self._make_data()
        _, shrunk = self._kernels(X_full, Y_full)
        for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
            with torch.no_grad():
                shrunk.rho.fill_(rho)
            K = shrunk.forward(X_full, X_full)
            eigvals = torch.linalg.eigvalsh(0.5 * (K + K.transpose(-1, -2)))
            self.assertGreater(float(eigvals.min()), -1e-8, msg=f"rho={rho}")

    def test_diag_is_rho_invariant(self) -> None:
        # diag only touches same-output entries, so rho must not change it.
        X_full, Y_full = self._make_data()
        _, shrunk = self._kernels(X_full, Y_full)
        diags = []
        for rho in (0.0, 0.5, 1.0):
            with torch.no_grad():
                shrunk.rho.fill_(rho)
            diags.append(shrunk.forward(X_full, X_full, diag=True))
            # diag must also still agree with the dense diagonal.
            dense_diag = shrunk.forward(X_full, X_full).diagonal(dim1=-2, dim2=-1)
            self.assertAllClose(diags[-1], dense_diag)
        self.assertTrue(torch.equal(diags[0], diags[1]))
        self.assertTrue(torch.equal(diags[1], diags[2]))

    def test_non_square_inputs(self) -> None:
        # x1 and x2 of different lengths exercise the n1 != n2 mask path.
        X_full, Y_full = self._make_data()
        _, shrunk = self._kernels(X_full, Y_full)
        with torch.no_grad():
            shrunk.rho.fill_(0.5)
        x2 = X_full[:3]
        K = shrunk.forward(X_full, x2)
        m = shrunk.num_outputs
        self.assertEqual(K.shape, torch.Size([X_full.shape[0] * m, x2.shape[0] * m]))

    def test_constraint_admits_endpoints(self) -> None:
        X_full, Y_full = self._make_data()
        _, shrunk = self._kernels(X_full, Y_full)
        constraint = shrunk.rho_constraint
        self.assertEqual(float(constraint.lower_bound), 0.0)
        self.assertEqual(float(constraint.upper_bound), 1.0)
        # A sigmoid Interval could not represent the endpoints exactly, which
        # would silently change rho=1 (the default) behavior.
        for value in (0.0, 1.0):
            t = torch.tensor(value, dtype=torch.double)
            self.assertEqual(float(constraint.transform(t)), value)

    def test_model_plumbing_and_posterior(self) -> None:
        X_full, Y_full = self._make_data()
        train_X = X_full[:4]
        train_Y = Y_full[0, :4]
        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            historical_X=X_full,
            historical_Y=Y_full,
            learn_cross_output_shrinkage=True,
        )
        self.assertIsNotNone(model._base_kernel.rho)
        self.assertIn("rho", {n.split(".")[-1] for n, _ in model.named_parameters()})
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_full)
        self.assertEqual(
            posterior.mean.shape, torch.Size([X_full.shape[0], Y_full.shape[-1]])
        )

    def test_model_rejects_mismatched_covar_module(self) -> None:
        X_full, Y_full = self._make_data()
        train_X, train_Y = X_full[:4], Y_full[0, :4]
        plain, shrunk = self._kernels(X_full, Y_full)
        for covar_module, flag in ((plain, True), (shrunk, False)):
            with self.assertRaisesRegex(ValueError, "learn_cross_output_shrinkage"):
                MultiOutputEmpiricalOneDimensionalGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    historical_X=X_full,
                    historical_Y=Y_full,
                    covar_module=covar_module,
                    learn_cross_output_shrinkage=flag,
                )


class TestCovarianceRidge(BotorchTestCase):
    """A nugget on the empirical covariance, to fix rank-deficiency.

    The empirical covariance is ``U^T U / N``, so its rank is capped by the
    number of historical curves. With fewer curves than index points, some
    directions carry *exactly zero* prior variance and the model is infinitely
    confident in them. A ridge shifts every eigenvalue up and lifts that null
    space. A scalar multiplier provably cannot: it leaves a zero at zero.
    """

    def _data(self, num_curves: int = 6, n: int = 10, m: int = 2):
        torch.manual_seed(0)
        epochs = torch.linspace(0, 1, n, dtype=torch.double).unsqueeze(-1)
        historical = torch.randn(num_curves, n, m, dtype=torch.double)
        return epochs, historical

    def _model(self, ridge: bool, num_curves: int = 6):
        epochs, historical = self._data(num_curves=num_curves)
        return MultiOutputEmpiricalOneDimensionalGP(
            train_X=epochs[:3],
            train_Y=torch.randn(3, 2, dtype=torch.double),
            historical_X=epochs,
            historical_Y=historical,
            learn_covariance_ridge=ridge,
        )

    def _set_ridge(self, model, value: float) -> None:
        kernel = model.covar_module.base_kernel
        with torch.no_grad():
            kernel.raw_ridge.fill_(
                kernel.raw_ridge_constraint.inverse_transform(
                    torch.tensor(value, dtype=torch.double)
                )
            )

    def test_off_by_default(self) -> None:
        model = self._model(ridge=False)
        self.assertIsNone(model.covar_module.base_kernel.raw_ridge)
        self.assertIsNone(model.covar_module.base_kernel.ridge)

    def test_lifts_the_null_space(self) -> None:
        # 6 historical curves over a 20-dim index set: rank <= 6, so 14 null
        # directions before the ridge and none after.
        epochs, _ = self._data()
        without = self._model(ridge=False)
        with torch.no_grad():
            K0 = without.covar_module(epochs).to_dense()
        eig0 = torch.linalg.eigvalsh(0.5 * (K0 + K0.transpose(-1, -2)))
        self.assertLess(float(eig0.min()), 1e-10)

        with_ridge = self._model(ridge=True)
        self._set_ridge(with_ridge, 1e-2)
        with torch.no_grad():
            K1 = with_ridge.covar_module(epochs).to_dense()
        eig1 = torch.linalg.eigvalsh(0.5 * (K1 + K1.transpose(-1, -2)))
        self.assertGreater(float(eig1.min()), 1e-3)

    def test_shifts_every_eigenvalue_by_the_ridge(self) -> None:
        # The defining property: K + lambda*I, so the whole spectrum moves up by
        # exactly lambda. This is what a scalar multiplier cannot do.
        epochs, _ = self._data()
        ridge_value = 5e-3
        without = self._model(ridge=False)
        with_ridge = self._model(ridge=True)
        self._set_ridge(with_ridge, ridge_value)
        with torch.no_grad():
            K0 = without.covar_module(epochs).to_dense()
            K1 = with_ridge.covar_module(epochs).to_dense()
        eig0 = torch.linalg.eigvalsh(0.5 * (K0 + K0.transpose(-1, -2)))
        eig1 = torch.linalg.eigvalsh(0.5 * (K1 + K1.transpose(-1, -2)))
        self.assertAllClose(eig1, eig0 + ridge_value, atol=1e-9)

    def test_only_added_where_inputs_coincide(self) -> None:
        # A genuine white-noise nugget, not a constant added to every entry
        # (which would be a rank-one bias term, not a ridge).
        epochs, historical = self._data()
        model = self._model(ridge=True)
        self._set_ridge(model, 1e-2)
        kernel = model.covar_module.base_kernel
        x1, x2 = epochs[:4], epochs[5:8]  # disjoint inputs
        with torch.no_grad():
            cross = kernel.forward(x1, x2)
            baseline = self._model(ridge=False).covar_module.base_kernel.forward(x1, x2)
        self.assertAllClose(cross, baseline)

    def test_diag_matches_the_full_diagonal(self) -> None:
        epochs, _ = self._data()
        model = self._model(ridge=True)
        self._set_ridge(model, 1e-2)
        kernel = model.covar_module.base_kernel
        with torch.no_grad():
            self.assertAllClose(
                kernel.forward(epochs, epochs, diag=True),
                kernel.forward(epochs, epochs).diagonal(dim1=-2, dim2=-1),
            )

    def test_increases_predictive_variance(self) -> None:
        epochs, _ = self._data()
        without = self._model(ridge=False)
        with_ridge = self._model(ridge=True)
        self._set_ridge(with_ridge, 1e-2)
        with torch.no_grad():
            v0 = without.posterior(epochs).variance
            v1 = with_ridge.posterior(epochs).variance
        self.assertTrue(torch.all(v1 >= v0 - 1e-9))
        self.assertGreater(float(v1.mean()), float(v0.mean()))

    def test_is_trainable(self) -> None:
        model = self._model(ridge=True)
        names = [
            n for n, p in model.named_parameters() if "ridge" in n and p.requires_grad
        ]
        self.assertEqual(len(names), 1)

    def test_flag_must_match_supplied_covar_module(self) -> None:
        epochs, historical = self._data()
        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=epochs, Y_full=historical, learn_covariance_ridge=False
        )
        with self.assertRaisesRegex(ValueError, "learn_covariance_ridge"):
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=epochs[:3],
                train_Y=torch.randn(3, 2, dtype=torch.double),
                historical_X=epochs,
                historical_Y=historical,
                covar_module=kernel,
                learn_covariance_ridge=True,
            )

    def test_composes_with_rho(self) -> None:
        epochs, historical = self._data()
        model = MultiOutputEmpiricalOneDimensionalGP(
            train_X=epochs[:3],
            train_Y=torch.randn(3, 2, dtype=torch.double),
            historical_X=epochs,
            historical_Y=historical,
            learn_cross_output_shrinkage=True,
            learn_covariance_ridge=True,
        )
        kernel = model.covar_module.base_kernel
        self.assertIsNotNone(kernel.rho)
        self.assertIsNotNone(kernel.ridge)
        with torch.no_grad():
            K = kernel.forward(epochs, epochs)
        self.assertTrue(torch.isfinite(K).all())


class TestKroneckerFactoredKernel(BotorchTestCase):
    """Separable empirical covariance: Sigma_progression (x) Sigma_output.

    The unconstrained estimate has rank <= num_curves, so below n*m curves it
    leaves directions with exactly zero prior variance. The Kronecker constraint
    needs only n(n+1)/2 + m(m+1)/2 parameters, so the same curves support a far
    better conditioned estimate -- and the benefit grows with m.
    """

    def _data(self, num_curves: int = 6, n: int = 10, m: int = 2):
        torch.manual_seed(0)
        epochs = torch.linspace(0, 1, n, dtype=torch.double).unsqueeze(-1)
        return epochs, torch.randn(num_curves, n, m, dtype=torch.double)

    def _model(self, factored: bool, num_curves: int = 6, m: int = 2):
        epochs, historical = self._data(num_curves=num_curves, m=m)
        return MultiOutputEmpiricalOneDimensionalGP(
            train_X=epochs[:3],
            train_Y=torch.randn(3, m, dtype=torch.double),
            historical_X=epochs,
            historical_Y=historical,
            kronecker_factored=factored,
        )

    def test_off_by_default(self) -> None:
        self.assertFalse(
            self._model(factored=False).covar_module.base_kernel.kronecker_factored
        )

    def test_raises_the_rank(self) -> None:
        epochs, _ = self._data()
        ranks = {}
        for factored in (False, True):
            with torch.no_grad():
                K = self._model(factored).covar_module(epochs).to_dense()
            K = 0.5 * (K + K.transpose(-1, -2))
            eig = torch.linalg.eigvalsh(K)
            tol = eig.max() * K.shape[-1] * torch.finfo(K.dtype).eps
            ranks[factored] = int((eig > tol).sum())
        self.assertGreater(ranks[True], ranks[False])

    def test_is_psd(self) -> None:
        epochs, _ = self._data()
        with torch.no_grad():
            K = self._model(True).covar_module(epochs).to_dense()
        eig = torch.linalg.eigvalsh(0.5 * (K + K.transpose(-1, -2)))
        self.assertGreater(float(eig.min()), -1e-8)

    def test_has_kronecker_structure(self) -> None:
        # The defining property: K[i*m+a, j*m+b] / K[i*m+c, j*m+d] must not depend
        # on (i, j). Equivalently every m x m block is a scalar multiple of S.
        epochs, _ = self._data(m=3)
        m = 3
        with torch.no_grad():
            K = self._model(True, m=m).covar_module.base_kernel.forward(epochs, epochs)
        block_00 = K[:m, :m]
        for i in (1, 2, 4):
            block = K[i * m : (i + 1) * m, :m]
            scale = block[0, 0] / block_00[0, 0]
            self.assertAllClose(block, block_00 * scale, atol=1e-8)

    def test_diag_matches_the_full_diagonal(self) -> None:
        epochs, _ = self._data()
        kernel = self._model(True).covar_module.base_kernel
        with torch.no_grad():
            self.assertAllClose(
                kernel.forward(epochs, epochs, diag=True),
                kernel.forward(epochs, epochs).diagonal(dim1=-2, dim2=-1),
            )

    def test_evaluates_off_the_historical_grid(self) -> None:
        # The whole reason the progression factor is rebuilt from the interpolated
        # basis rather than stored as a fixed grid matrix.
        _, _ = self._data()
        off_grid = torch.tensor([[0.05], [0.17], [0.93]], dtype=torch.double)
        with torch.no_grad():
            K = self._model(True).covar_module.base_kernel.forward(off_grid, off_grid)
        self.assertEqual(K.shape, torch.Size([6, 6]))
        self.assertTrue(torch.isfinite(K).all())

    def test_rectangular_inputs(self) -> None:
        epochs, _ = self._data()
        with torch.no_grad():
            K = self._model(True).covar_module.base_kernel.forward(
                epochs[:4], epochs[:2]
            )
        self.assertEqual(K.shape, torch.Size([8, 4]))

    def test_flag_must_match_supplied_covar_module(self) -> None:
        epochs, historical = self._data()
        kernel = MultiOutputEmpiricalOneDimensionalKernel(
            X_full=epochs, Y_full=historical, kronecker_factored=False
        )
        with self.assertRaisesRegex(ValueError, "kronecker_factored"):
            MultiOutputEmpiricalOneDimensionalGP(
                train_X=epochs[:3],
                train_Y=torch.randn(3, 2, dtype=torch.double),
                historical_X=epochs,
                historical_Y=historical,
                covar_module=kernel,
                kronecker_factored=True,
            )
