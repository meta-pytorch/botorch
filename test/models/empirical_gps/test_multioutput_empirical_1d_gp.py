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
