#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HyperBO model and loss functions.

Design: Minimize public test methods by using private helpers and loops.
Each public test method covers multiple configurations via subTest.
"""

from __future__ import annotations

import torch
from botorch.models.empirical_gps.hyperbo import (
    compute_gaussian_kl_divergence,
    compute_mle_estimates,
    HyperBODeepKernel,
    HyperBOEKL,
    HyperBOLinearMean,
    HyperBOModel,
    HyperBONLL,
    HyperBOPriorContainer,
    MLPFeatureExtractor,
    pretrain_hyperbo,
    validate_matching_inputs,
)
from botorch.models.empirical_gps.utils import ExperimentDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class TestHyperBO(BotorchTestCase):
    """Tests for HyperBO model and loss functions.

    Design: Minimize public test methods by using private helpers and loops.
    """

    # ===== Configuration Constants =====
    ARCHITECTURES = [
        {"hidden_dims": (8,), "name": "single_layer_8"},  # Original paper
        {"hidden_dims": (32, 32), "name": "two_layer_32_32"},  # Extended
    ]
    DTYPES = [torch.float32, torch.double]

    # ===== Private Test Helpers =====

    def _get_test_data(
        self,
        n_points: int = 10,
        input_dim: int = 2,
        n_tasks: int = 3,
        dtype: torch.dtype = torch.double,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, list[ExperimentDataset]]:
        """Generate synthetic test data for HyperBO tests.

        Args:
            n_points: Number of data points per task.
            input_dim: Input dimension.
            n_tasks: Number of tasks.
            dtype: Data type for tensors.
            device: Device for tensors.

        Returns:
            Tuple of (train_X, train_Y, datasets) where:
            - train_X: Shared input locations for all tasks.
            - train_Y: Target values (from first task).
            - datasets: List of ExperimentDataset instances.
        """
        if device is None:
            device = self.device

        # Shared input locations (matching inputs for EKL)
        train_X = torch.rand(n_points, input_dim, dtype=dtype, device=device)

        # Generate Y values for each task
        datasets = []
        for i in range(n_tasks):
            # Simple linear function with task-specific offset
            Y = train_X.sum(dim=-1, keepdim=True) + 0.1 * torch.randn(
                n_points, 1, dtype=dtype, device=device
            )
            Y = Y + i * 0.5  # Task-specific offset
            datasets.append(ExperimentDataset(X=train_X.clone(), Y=Y))

        # Use first task's Y for train_Y
        train_Y = datasets[0].Y

        return train_X, train_Y, datasets

    def _test_mlp_feature_extractor(
        self,
        hidden_dims: tuple[int, ...],
        input_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test MLPFeatureExtractor output shape and gradient flow."""
        extractor = MLPFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        X = torch.randn(5, input_dim, dtype=dtype, device=device)
        features = extractor(X)

        # Check output shape
        self.assertEqual(features.shape, (5, hidden_dims[-1]))

        # Check gradient flow
        loss = features.sum()
        loss.backward()
        for param in extractor.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())

    def _test_hyperbo_linear_mean(
        self,
        hidden_dims: tuple[int, ...],
        input_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBOLinearMean output shape and gradient flow."""
        extractor = MLPFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        mean_module = HyperBOLinearMean(
            feature_extractor=extractor,
            output_dim=1,
        ).to(dtype=dtype, device=device)

        X = torch.randn(5, input_dim, dtype=dtype, device=device)
        mean = mean_module(X)

        # Check output shape (should be squeezed for output_dim=1)
        self.assertEqual(mean.shape, (5,))

        # Check gradient flow
        loss = mean.sum()
        loss.backward()
        for param in mean_module.parameters():
            self.assertIsNotNone(param.grad)

    def _test_hyperbo_deep_kernel(
        self,
        hidden_dims: tuple[int, ...],
        input_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBODeepKernel output shape and gradient flow."""
        extractor = MLPFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        kernel = HyperBODeepKernel(
            feature_extractor=extractor,
        ).to(dtype=dtype, device=device)

        X1 = torch.randn(5, input_dim, dtype=dtype, device=device)
        X2 = torch.randn(3, input_dim, dtype=dtype, device=device)

        # Test full kernel matrix
        K = kernel(X1, X2).to_dense()
        self.assertEqual(K.shape, (5, 3))

        # Test diagonal
        K_diag = kernel(X1, X1, diag=True)
        self.assertEqual(K_diag.shape, (5,))

        # Check gradient flow
        loss = K.sum()
        loss.backward()
        for param in kernel.parameters():
            self.assertIsNotNone(param.grad)

    def _test_hyperbo_model_forward(
        self,
        hidden_dims: tuple[int, ...],
        use_linear_mean: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBOModel forward pass returns valid MultivariateNormal."""
        train_X, train_Y, _ = self._get_test_data(dtype=dtype, device=device)

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
            use_linear_mean=use_linear_mean,
        ).to(dtype=dtype, device=device)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(train_X)

        self.assertIsInstance(output, MultivariateNormal)
        self.assertEqual(output.mean.shape, train_Y.squeeze(-1).shape)

    def _test_hyperbo_model_freeze_unfreeze(
        self,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test freeze_pretrained_parameters and unfreeze_parameters methods."""
        train_X, train_Y, _ = self._get_test_data(dtype=dtype, device=device)

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        # Initially all parameters should require grad
        trainable_before = sum(p.requires_grad for p in model.parameters())
        self.assertGreater(trainable_before, 0)

        # After freezing, feature extractor and kernel params should not require grad
        model.freeze_pretrained_parameters()
        trainable_after_freeze = sum(
            p.requires_grad for p in model.feature_extractor.parameters()
        )
        self.assertEqual(trainable_after_freeze, 0)

        # Covariance module should also be frozen
        trainable_covar = sum(p.requires_grad for p in model.covar_module.parameters())
        self.assertEqual(trainable_covar, 0)

        # After unfreezing, all parameters should require grad again
        model.unfreeze_parameters()
        trainable_after_unfreeze = sum(p.requires_grad for p in model.parameters())
        self.assertEqual(trainable_after_unfreeze, trainable_before)

    def _test_nll_loss_computation(
        self,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBONLL loss computation and gradient flow."""
        train_X, train_Y, datasets = self._get_test_data(
            dtype=dtype, device=device, n_tasks=3
        )

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        mll = HyperBONLL(model.likelihood, model, datasets=datasets)

        # Compute loss
        model.train()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze(-1))

        # Check loss is finite
        self.assertTrue(torch.isfinite(loss))

        # Check gradient flow
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def _test_ekl_loss_computation(
        self,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBOEKL loss computation with matching inputs."""
        train_X, train_Y, datasets = self._get_test_data(
            dtype=dtype,
            device=device,
            n_tasks=5,  # Need enough tasks for EKL
        )

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        mll = HyperBOEKL(
            model.likelihood,
            model,
            datasets=datasets,
            matching_inputs=train_X,
        )

        # Compute loss
        model.train()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze(-1))

        # Check loss is finite (KL divergence, returned as negative for maximization)
        self.assertTrue(torch.isfinite(loss))

    def _test_end_to_end_optimization(
        self,
        hidden_dims: tuple[int, ...],
        loss_class: type,  # HyperBONLL or HyperBOEKL
        n_steps: int = 5,
        dtype: torch.dtype = torch.double,
        device: torch.device | None = None,
    ) -> None:
        """Test end-to-end optimization with a small number of gradient steps.

        Verifies that:
        1. Loss decreases (or stays stable) over optimization steps
        2. No NaN/Inf values appear during optimization
        3. Parameters are updated correctly
        """
        if device is None:
            device = self.device

        train_X, train_Y, datasets = self._get_test_data(
            n_points=8, n_tasks=4, dtype=dtype, device=device
        )

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        # Create loss function
        if loss_class == HyperBOEKL:
            mll = loss_class(
                model.likelihood,
                model,
                datasets=datasets,
                matching_inputs=train_X,
            )
        else:
            mll = loss_class(model.likelihood, model, datasets=datasets)

        # Store initial parameters
        initial_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        # Run optimization
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
        losses = []

        for _ in range(n_steps):
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, train_Y.squeeze(-1))
            losses.append(loss.item())

            # Check for NaN/Inf
            self.assertTrue(
                torch.isfinite(loss),
                f"Loss became non-finite: {loss.item()}",
            )

            loss.backward()
            optimizer.step()

        # Verify parameters were updated
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        self.assertTrue(
            params_changed, "No parameters were updated during optimization"
        )

        # Verify loss trend (should generally decrease or stay stable)
        # Allow for some noise in small-scale optimization
        self.assertLess(
            losses[-1],
            losses[0] * 1.5,  # Allow 50% increase due to noise, but not explode
            f"Loss increased significantly: {losses[0]} -> {losses[-1]}",
        )

    def _test_helper_functions(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test helper functions: compute_mle_estimates, validate_matching_inputs."""
        _, _, datasets = self._get_test_data(
            n_points=10, n_tasks=5, dtype=dtype, device=device
        )

        # Test validate_matching_inputs
        self.assertTrue(validate_matching_inputs(datasets))

        # Test with mismatched inputs
        mismatched_datasets = [
            ExperimentDataset(
                X=torch.rand(10, 2, dtype=dtype, device=device),
                Y=torch.rand(10, 1, dtype=dtype, device=device),
            ),
            ExperimentDataset(
                X=torch.rand(10, 2, dtype=dtype, device=device),
                Y=torch.rand(10, 1, dtype=dtype, device=device),
            ),
        ]
        self.assertFalse(validate_matching_inputs(mismatched_datasets))

        # Test compute_mle_estimates
        mu_tilde, Sigma_tilde = compute_mle_estimates(datasets)
        self.assertEqual(mu_tilde.shape, (10,))  # n_points
        self.assertEqual(Sigma_tilde.shape, (10, 10))

        # Sigma_tilde should be symmetric
        self.assertTrue(
            torch.allclose(Sigma_tilde, Sigma_tilde.T, atol=1e-6),
            "Sigma_tilde is not symmetric",
        )

    def _test_kl_divergence(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test compute_gaussian_kl_divergence with known cases."""
        dim = 5

        # Case 1: KL between identical distributions should be 0
        mu = torch.randn(dim, dtype=dtype, device=device)
        Sigma = torch.eye(dim, dtype=dtype, device=device)
        kl = compute_gaussian_kl_divergence(mu, Sigma, mu, Sigma)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)

        # Case 2: KL(N(0, I) || N(0, I)) = 0
        mu_zero = torch.zeros(dim, dtype=dtype, device=device)
        kl = compute_gaussian_kl_divergence(mu_zero, Sigma, mu_zero, Sigma)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)

        # Case 3: KL should be non-negative for different distributions
        mu_q = torch.randn(dim, dtype=dtype, device=device)
        Sigma_q = torch.eye(dim, dtype=dtype, device=device) * 2.0
        kl = compute_gaussian_kl_divergence(mu, Sigma, mu_q, Sigma_q)
        self.assertGreaterEqual(kl.item(), 0.0)

    def _test_nll_task_batch_size(
        self,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBONLL with task_batch_size for mini-batched task optimization.

        Verifies that:
        1. compute_loss works with task_batch_size parameter
        2. Loss is finite and gradients flow
        3. The rescaling produces an unbiased estimate (mean over many samples
           should approximate full-batch loss)
        4. task_batch_size >= n_tasks falls back to full batch (exact match)
        5. End-to-end optimization works with task mini-batching
        """
        n_tasks = 6
        train_X, train_Y, datasets = self._get_test_data(
            n_points=8, n_tasks=n_tasks, dtype=dtype, device=device
        )

        model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hidden_dims,
        ).to(dtype=dtype, device=device)

        mll = HyperBONLL(model.likelihood, model, datasets=datasets)

        model.train()

        # --- Test 1: Loss is finite with task_batch_size ---
        loss_batched = -mll.compute_loss(task_batch_size=2)
        self.assertTrue(
            torch.isfinite(loss_batched),
            f"Batched loss is not finite: {loss_batched.item()}",
        )

        # --- Test 2: Gradients flow ---
        loss_batched.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
        model.zero_grad()

        # --- Test 3: task_batch_size >= n_tasks gives exact full-batch result ---
        with torch.no_grad():
            loss_full = -mll.compute_loss()
            loss_oversized = -mll.compute_loss(task_batch_size=n_tasks + 10)
        self.assertAlmostEqual(
            loss_full.item(),
            loss_oversized.item(),
            places=6,
            msg="task_batch_size >= n_tasks should give same result as full batch",
        )

        # --- Test 4: Rescaling produces unbiased estimate ---
        # Average many batched estimates and check they approximate the full loss
        torch.manual_seed(42)
        n_samples = 200
        batched_losses = []
        with torch.no_grad():
            for _ in range(n_samples):
                bl = -mll.compute_loss(task_batch_size=2)
                batched_losses.append(bl.item())
        mean_batched = sum(batched_losses) / len(batched_losses)
        full_loss = loss_full.item()
        # Allow 15% relative tolerance for stochastic estimate
        self.assertAlmostEqual(
            mean_batched,
            full_loss,
            delta=abs(full_loss) * 0.15,
            msg=f"Mean batched loss ({mean_batched:.4f}) should approximate "
            f"full-batch loss ({full_loss:.4f})",
        )

        # --- Test 5: End-to-end optimization with task_batch_size ---
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = -mll.compute_loss(task_batch_size=3)
            losses.append(loss.item())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()

        # Loss shouldn't explode
        self.assertLess(
            losses[-1],
            losses[0] * 2.0,
            f"Loss exploded with task mini-batching: {losses[0]} -> {losses[-1]}",
        )

    # ===== Public Master Test Methods =====

    def test_mlp_feature_extractor(self) -> None:
        """Master test for MLPFeatureExtractor across architectures and dtypes."""
        for arch in self.ARCHITECTURES:
            for dtype in self.DTYPES:
                with self.subTest(arch=arch["name"], dtype=dtype):
                    self._test_mlp_feature_extractor(
                        hidden_dims=arch["hidden_dims"],
                        input_dim=3,
                        dtype=dtype,
                        device=self.device,
                    )

    def test_hyperbo_components(self) -> None:
        """Master test for HyperBO mean and kernel components."""
        for arch in self.ARCHITECTURES:
            for dtype in self.DTYPES:
                with self.subTest(arch=arch["name"], dtype=dtype, component="mean"):
                    self._test_hyperbo_linear_mean(
                        hidden_dims=arch["hidden_dims"],
                        input_dim=3,
                        dtype=dtype,
                        device=self.device,
                    )
                with self.subTest(arch=arch["name"], dtype=dtype, component="kernel"):
                    self._test_hyperbo_deep_kernel(
                        hidden_dims=arch["hidden_dims"],
                        input_dim=3,
                        dtype=dtype,
                        device=self.device,
                    )

    def test_hyperbo_model(self) -> None:
        """Master test for HyperBOModel forward pass and freezing."""
        for arch in self.ARCHITECTURES:
            for dtype in self.DTYPES:
                for use_linear_mean in [True, False]:
                    with self.subTest(
                        arch=arch["name"], dtype=dtype, linear_mean=use_linear_mean
                    ):
                        self._test_hyperbo_model_forward(
                            hidden_dims=arch["hidden_dims"],
                            use_linear_mean=use_linear_mean,
                            dtype=dtype,
                            device=self.device,
                        )
                # Test freeze/unfreeze (only need to test once per arch/dtype)
                with self.subTest(arch=arch["name"], dtype=dtype, test="freeze"):
                    self._test_hyperbo_model_freeze_unfreeze(
                        hidden_dims=arch["hidden_dims"],
                        dtype=dtype,
                        device=self.device,
                    )

    def test_loss_functions(self) -> None:
        """Master test for NLL and EKL loss computation."""
        for arch in self.ARCHITECTURES:
            for dtype in self.DTYPES:
                with self.subTest(arch=arch["name"], dtype=dtype, loss="NLL"):
                    self._test_nll_loss_computation(
                        hidden_dims=arch["hidden_dims"],
                        dtype=dtype,
                        device=self.device,
                    )
                with self.subTest(arch=arch["name"], dtype=dtype, loss="EKL"):
                    self._test_ekl_loss_computation(
                        hidden_dims=arch["hidden_dims"],
                        dtype=dtype,
                        device=self.device,
                    )

    def test_nll_task_batch_size(self) -> None:
        """Test task_batch_size parameter for mini-batched task optimization."""
        for arch in self.ARCHITECTURES:
            with self.subTest(arch=arch["name"]):
                self._test_nll_task_batch_size(
                    hidden_dims=arch["hidden_dims"],
                    dtype=torch.double,
                    device=self.device,
                )

    def test_end_to_end_optimization(self) -> None:
        """Master test for end-to-end optimization with small number of steps.

        Tests both architectures with both loss functions to ensure the full
        training pipeline works correctly.
        """
        for arch in self.ARCHITECTURES:
            for loss_class, loss_name in [
                (HyperBONLL, "NLL"),
                (HyperBOEKL, "EKL"),
            ]:
                with self.subTest(arch=arch["name"], loss=loss_name):
                    self._test_end_to_end_optimization(
                        hidden_dims=arch["hidden_dims"],
                        loss_class=loss_class,
                        n_steps=5,  # Small number for fast testing
                        dtype=torch.double,
                        device=self.device,
                    )

    def test_helper_functions(self) -> None:
        """Master test for helper functions."""
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype, test="helpers"):
                self._test_helper_functions(dtype=dtype, device=self.device)
            with self.subTest(dtype=dtype, test="kl_divergence"):
                self._test_kl_divergence(dtype=dtype, device=self.device)

    # ===== Pre-training Tests =====

    def _test_pretrain_hyperbo_and_container(
        self,
        hidden_dims: tuple[int, ...],
        loss_type: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test HyperBOPriorContainer creation and state dict shapes."""
        _, _, datasets = self._get_test_data(
            n_points=10, input_dim=3, n_tasks=3, dtype=dtype, device=device
        )

        # Pre-train with a small number of epochs
        prior = pretrain_hyperbo(
            datasets=datasets,
            input_dim=3,
            hidden_dims=hidden_dims,
            use_linear_mean=True,
            loss_type=loss_type,
            num_iterations=5,  # Small for fast testing
            device=device,
            dtype=dtype,
        )

        # Check container type and attributes
        self.assertIsInstance(prior, HyperBOPriorContainer)
        self.assertEqual(prior.input_dim, 3)
        self.assertEqual(prior.hidden_dims, hidden_dims)
        self.assertTrue(prior.use_linear_mean)
        self.assertEqual(prior.num_datasets, 3)
        self.assertEqual(prior.loss_type, loss_type)
        self.assertIsNotNone(prior.final_loss)

        # Check state dicts are present
        self.assertIsNotNone(prior.feature_extractor_state)
        self.assertIsNotNone(prior.mean_module_state)
        self.assertIsNotNone(prior.covar_module_state)
        self.assertIsNotNone(prior.likelihood_state)

        # Check that all tensors have requires_grad=False
        for name, tensor in prior.feature_extractor_state.items():
            self.assertFalse(
                tensor.requires_grad,
                f"feature_extractor_state[{name}] has requires_grad=True",
            )
        for name, tensor in prior.mean_module_state.items():
            self.assertFalse(
                tensor.requires_grad,
                f"mean_module_state[{name}] has requires_grad=True",
            )
        for name, tensor in prior.covar_module_state.items():
            self.assertFalse(
                tensor.requires_grad,
                f"covar_module_state[{name}] has requires_grad=True",
            )

    def _test_from_pretrained_factory_method(
        self,
        hidden_dims: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test model creation from pre-trained container."""
        train_X, train_Y, datasets = self._get_test_data(
            n_points=10, input_dim=3, n_tasks=3, dtype=dtype, device=device
        )

        # Pre-train
        prior = pretrain_hyperbo(
            datasets=datasets,
            input_dim=3,
            hidden_dims=hidden_dims,
            use_linear_mean=True,
            loss_type="NLL",
            num_iterations=5,
            device=device,
            dtype=dtype,
        )

        # Create model from pre-trained prior
        model = HyperBOModel.from_pretrained(
            hyperbo_prior=prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=True,
        )

        # Check model attributes
        self.assertEqual(model.hidden_dims, hidden_dims)
        self.assertTrue(model.use_linear_mean)

        # Check forward pass works
        model.eval()
        with torch.no_grad():
            output = model(train_X)
        self.assertIsInstance(output, MultivariateNormal)
        self.assertEqual(output.mean.shape, train_Y.squeeze(-1).shape)

    def _test_pretrained_model_has_frozen_parameters(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Verify all pre-trained parameters have requires_grad=False."""
        train_X, train_Y, datasets = self._get_test_data(
            n_points=10, input_dim=3, n_tasks=3, dtype=dtype, device=device
        )

        # Pre-train
        prior = pretrain_hyperbo(
            datasets=datasets,
            input_dim=3,
            hidden_dims=(8,),
            use_linear_mean=True,
            loss_type="NLL",
            num_iterations=5,
            device=device,
            dtype=dtype,
        )

        # Create model with freeze_pretrained=True
        model = HyperBOModel.from_pretrained(
            hyperbo_prior=prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=True,
        )

        # Check all parameters are frozen
        for name, param in model.feature_extractor.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"feature_extractor.{name} should be frozen",
            )
        for name, param in model.mean_module.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"mean_module.{name} should be frozen",
            )
        for name, param in model.covar_module.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"covar_module.{name} should be frozen",
            )

    def _test_pretrained_model_unfrozen_allows_finetuning(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Verify freeze_pretrained=False allows fine-tuning."""
        train_X, train_Y, datasets = self._get_test_data(
            n_points=10, input_dim=3, n_tasks=3, dtype=dtype, device=device
        )

        # Pre-train
        prior = pretrain_hyperbo(
            datasets=datasets,
            input_dim=3,
            hidden_dims=(8,),
            use_linear_mean=True,
            loss_type="NLL",
            num_iterations=5,
            device=device,
            dtype=dtype,
        )

        # Create model with freeze_pretrained=False (for fine-tuning)
        model = HyperBOModel.from_pretrained(
            hyperbo_prior=prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=False,
        )

        # Check parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)

    def _test_pretrained_model_output_matches_original(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test that from_pretrained model gives same output as original."""
        train_X, train_Y, datasets = self._get_test_data(
            n_points=10, input_dim=3, n_tasks=3, dtype=dtype, device=device
        )

        # Create and train original model
        original_model = HyperBOModel(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=(8,),
            use_linear_mean=True,
        ).to(device=device, dtype=dtype)

        # Extract state into container manually
        from botorch.models.empirical_gps.hyperbo import _extract_frozen_state

        prior = _extract_frozen_state(
            model=original_model,
            num_datasets=len(datasets),
            loss_type="manual",
        )

        # Create model from container
        loaded_model = HyperBOModel.from_pretrained(
            hyperbo_prior=prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=False,
        )

        # Compare outputs
        original_model.eval()
        loaded_model.eval()
        with torch.no_grad():
            original_output = original_model(train_X)
            loaded_output = loaded_model(train_X)

        self.assertTrue(
            torch.allclose(original_output.mean, loaded_output.mean, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                original_output.covariance_matrix,
                loaded_output.covariance_matrix,
                atol=1e-6,
            )
        )

    def test_pretrain_hyperbo(self) -> None:
        """Master test for HyperBO pre-training workflow."""
        for arch in self.ARCHITECTURES:
            for loss_type in ["NLL", "EKL"]:
                with self.subTest(
                    arch=arch["name"], loss_type=loss_type, test="container"
                ):
                    self._test_pretrain_hyperbo_and_container(
                        hidden_dims=arch["hidden_dims"],
                        loss_type=loss_type,
                        dtype=torch.double,
                        device=self.device,
                    )

        for arch in self.ARCHITECTURES:
            with self.subTest(arch=arch["name"], test="from_pretrained"):
                self._test_from_pretrained_factory_method(
                    hidden_dims=arch["hidden_dims"],
                    dtype=torch.double,
                    device=self.device,
                )

        with self.subTest(test="frozen_parameters"):
            self._test_pretrained_model_has_frozen_parameters(
                dtype=torch.double,
                device=self.device,
            )

        with self.subTest(test="unfrozen_finetuning"):
            self._test_pretrained_model_unfrozen_allows_finetuning(
                dtype=torch.double,
                device=self.device,
            )

        with self.subTest(test="output_matches"):
            self._test_pretrained_model_output_matches_original(
                dtype=torch.double,
                device=self.device,
            )
