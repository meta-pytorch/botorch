#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.additive_gp import OrthogonalAdditiveGP
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


# =============================================================================
# Test Configuration - Start strict, relax after correctness confirmed
# =============================================================================

# Ground truth recovery test
GROUND_TRUTH_N_TRAIN = 500  # Training points (increase for tighter tolerance)
GROUND_TRUTH_MEAN_ATOL = 1e-2  # Absolute tolerance for mean recovery
GROUND_TRUTH_MAX_VAR = 1e-2  # Maximum acceptable posterior variance
GROUND_TRUTH_GRID_SIZE = 50  # Evaluation grid size

# For partial observability test
PARTIAL_OBS_N_TRAIN = 200
PARTIAL_OBS_VARIANCE_RATIO = 2.0  # Min ratio of high/low uncertainty (relaxed)


class TestOrthogonalAdditiveGP(BotorchTestCase):
    def test_orthogonal_additive_gp(self) -> None:
        n, d = 10, 4
        tkwargs = {"dtype": torch.double, "device": self.device}
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.randn(n, 1, **tkwargs)

        def make_model(second_order: bool = False) -> OrthogonalAdditiveGP:
            oak = OrthogonalAdditiveKernel(
                RBFKernel(), dim=d, second_order=second_order, **tkwargs
            )
            return OrthogonalAdditiveGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=oak,
                outcome_transform=None,  # Disable for component-wise testing
            )

        test_X = torch.rand(5, d, **tkwargs)

        # Test init and forward
        for second_order in [False, True]:
            model = make_model(second_order=second_order)
            self.assertIsInstance(model, OrthogonalAdditiveGP)
            model.eval()
            self.assertEqual(model(test_X).mean.shape, torch.Size([5]))

        # Test standard posterior
        model = make_model()
        model.eval()
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        self.assertEqual(posterior.variance.shape, torch.Size([5, 1]))

        # Test individual component posterior and verify sum equals standard
        for second_order in [False, True]:
            model = make_model(second_order=second_order)
            model.eval()

            # Get standard posterior (summed over components)
            standard_posterior = model.posterior(test_X)

            # Get component-wise posterior
            component_posterior = model.posterior(test_X, infer_all_components=True)
            expected = 1 + d + (d * (d - 1) // 2 if second_order else 0)
            # Component posterior has shape (num_components, n_test, n_outputs)
            self.assertEqual(
                component_posterior.mean.shape, torch.Size([expected, 5, 1])
            )

            # Verify that sum of component means equals standard mean.
            # This holds because GP conditioning is linear in the mean:
            #   E[f(x)|y] = m(x) + K(x,X) @ K(X,X)^{-1} @ (y - m(X))
            # Since sum_i K_i(x,X) = K_sum(x,X), the component means sum correctly.
            component_mean_sum = component_posterior.mean.squeeze(-1).sum(dim=0)
            self.assertAllClose(
                component_mean_sum, standard_posterior.mean.squeeze(), atol=1e-4
            )

            # NOTE: The relationship between component posterior covariances and
            # standard posterior covariance is more complex, due to correlations
            # between the components in the posterior. We therefore do not include
            # a test for this here.

        # Test delegation to kernel for component_indices, num_components,
        # get_component_index (these are now simple wrappers)
        from unittest.mock import patch, PropertyMock

        model = make_model()

        # Verify component_indices delegates to kernel (patch at class level)
        with patch.object(
            type(model.covar_module),
            "component_indices",
            new_callable=PropertyMock,
            return_value={"mocked": True},
        ):
            self.assertEqual(model.component_indices, {"mocked": True})

        # Verify num_components delegates to kernel (patch at class level)
        with patch.object(
            type(model.covar_module),
            "num_components",
            new_callable=PropertyMock,
            return_value=-1,
        ):
            self.assertEqual(model.num_components, -1)

        # Verify get_component_index delegates to kernel
        with patch.object(
            type(model.covar_module), "get_component_index", return_value=42
        ) as mock_get_idx:
            result = model.get_component_index("first_order", 0)
            mock_get_idx.assert_called_once_with("first_order", 0)
            self.assertEqual(result, 42)

        # Test fitting
        model = make_model()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
        model.eval()
        self.assertEqual(model.posterior(test_X[:3]).mean.shape, torch.Size([3, 1]))

        # Test default kernel creation (no covar_module passed)
        # Also verifies default transforms are used from SingleTaskGP
        model_default = OrthogonalAdditiveGP(train_X=train_X, train_Y=train_Y)
        self.assertIsInstance(model_default.covar_module, OrthogonalAdditiveKernel)
        self.assertEqual(model_default.covar_module.dim, d)
        self.assertIsNone(model_default.covar_module.raw_coeffs_2)  # first-order only
        self.assertIsInstance(model_default.outcome_transform, Standardize)

        # Test default kernel with second_order=True
        model_2nd = OrthogonalAdditiveGP(
            train_X=train_X, train_Y=train_Y, second_order=True
        )
        self.assertIsNotNone(model_2nd.covar_module.raw_coeffs_2)

        # Test TypeError for non-OAK kernel
        with self.assertRaises(TypeError):
            OrthogonalAdditiveGP(
                train_X=train_X, train_Y=train_Y, covar_module=RBFKernel()
            )

    def test_evaluate_first_order_on_grid(self) -> None:
        """Test evaluate_first_order_on_grid shapes and convenience methods."""
        n, d = 20, 4
        tkwargs = {"dtype": torch.double, "device": self.device}
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = torch.randn(n, 1, **tkwargs)

        def make_model(second_order: bool = False) -> OrthogonalAdditiveGP:
            oak = OrthogonalAdditiveKernel(
                RBFKernel(), dim=d, second_order=second_order, **tkwargs
            )
            return OrthogonalAdditiveGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=oak,
                outcome_transform=None,  # Disable for component-wise testing
            )

        grid_size = 20
        grid = torch.linspace(0, 1, grid_size, **tkwargs)

        for second_order in [False, True]:
            model = make_model(second_order=second_order)
            model.eval()

            # Test evaluate_first_order_on_grid shapes
            (bias_mean, bias_var), (fo_mean, fo_var) = (
                model.evaluate_first_order_on_grid(grid)
            )

            # Check shapes
            self.assertEqual(fo_mean.shape, torch.Size([d, grid_size]))
            self.assertEqual(fo_var.shape, torch.Size([d, grid_size]))

            # Bias should be scalar
            self.assertEqual(bias_mean.shape, torch.Size([]))
            self.assertEqual(bias_var.shape, torch.Size([]))

            # Test infer_all_components with diagonal inputs
            X_diag = grid.unsqueeze(-1).expand(grid_size, d)
            posterior_all = model.posterior(X_diag, infer_all_components=True)
            self.assertEqual(posterior_all.mean.shape[0], model.num_components)

    def test_evaluate_first_order_ground_truth(self) -> None:
        """Test component recovery against known additive function.

        Ground truth: f(x) = x1 + x2^2 + sin(x3) + pi/2

        The OrthogonalAdditiveKernel uses Legendre basis which centers each
        component to have zero mean over [0, 1]. The recovered components
        should match the centered ground truth.
        """
        torch.manual_seed(0)
        n, d = GROUND_TRUTH_N_TRAIN, 3
        tkwargs = {"dtype": torch.double, "device": self.device}

        # Constants for centered ground truth
        # E[x] = 0.5, E[x²] = 1/3, E[sin(x)] = ∫₀¹ sin(x)dx = 1 - cos(1)
        INTEGRAL_SIN_0_1 = 1 - math.cos(1)  # ≈ 0.4597

        def centered_components(x1: torch.Tensor) -> tuple:
            """Return centered ground truth components evaluated at x1 values.

            Each component is centered (zero mean) over [0, 1]:
            - f1(t) = t - E[t] = t - 0.5
            - f2(t) = t² - E[t²] = t² - 1/3
            - f3(t) = sin(t) - E[sin(t)] = sin(t) - (1 - cos(1))
            - bias = π/2 + E[x] + E[x²] + E[sin(x)]

            Args:
                x1: 1D tensor of evaluation points in [0, 1]
            """
            f1 = x1 - 0.5
            f2 = x1**2 - 1 / 3
            f3 = torch.sin(x1) - INTEGRAL_SIN_0_1
            bias = torch.tensor(math.pi / 2 + 0.5 + 1 / 3 + INTEGRAL_SIN_0_1, **tkwargs)
            return bias, f1, f2, f3

        def test_function(x: torch.Tensor) -> torch.Tensor:
            """f(x) = x1 + x2^2 + sin(x3) + pi/2"""
            return x[..., 0] + x[..., 1] ** 2 + torch.sin(x[..., 2]) + math.pi / 2

        # Sample training data uniformly in [0, 1]^3
        train_X = torch.rand(n, d, **tkwargs)
        train_Y = test_function(train_X).unsqueeze(-1)

        # Fit model (disable outcome transform for ground truth comparison)
        oak = OrthogonalAdditiveKernel(RBFKernel(), dim=d, **tkwargs)
        model = OrthogonalAdditiveGP(
            train_X,
            train_Y,
            covar_module=oak,
            outcome_transform=None,  # Disable for ground truth comparison
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        # Evaluate on grid
        grid = torch.linspace(0, 1, GROUND_TRUTH_GRID_SIZE, **tkwargs)
        (bias_mean, _), (fo_means, fo_vars) = model.evaluate_first_order_on_grid(grid)

        # Compute centered ground truth on grid
        gt_bias, gt_f1, gt_f2, gt_f3 = centered_components(grid)

        # Verify recovered means are close to centered ground truth
        self.assertAllClose(fo_means[0], gt_f1, atol=GROUND_TRUTH_MEAN_ATOL)  # x - 0.5
        self.assertAllClose(fo_means[1], gt_f2, atol=GROUND_TRUTH_MEAN_ATOL)  # x² - 1/3
        self.assertAllClose(
            fo_means[2], gt_f3, atol=GROUND_TRUTH_MEAN_ATOL
        )  # sin(x) - E[sin]

        # Bias should be close to the sum of means + constant
        self.assertAllClose(bias_mean, gt_bias, atol=GROUND_TRUTH_MEAN_ATOL)

        # Variances should be small (we have good data coverage)
        self.assertTrue(
            fo_vars.max().item() < GROUND_TRUTH_MAX_VAR,
            f"Max variance {fo_vars.max().item():.4f} exceeds {GROUND_TRUTH_MAX_VAR}",
        )

    def test_evaluate_first_order_partial_observability(self) -> None:
        """Test that unobserved dimensions have high uncertainty.

        Setup: Sample only in (x1, x2) subspace with x3 fixed at a single value.

        Expected behavior:
        - Components for x1, x2: LOW variance (we have data covering [0,1])
        - Component for x3: HIGH variance except near the fixed value
        """
        torch.manual_seed(42)
        n, d = PARTIAL_OBS_N_TRAIN, 3
        tkwargs = {"dtype": torch.double, "device": self.device}

        # Sample only in (x1, x2) subspace, x3 fixed
        X3_FIXED = 0.3
        train_X = torch.rand(n, d, **tkwargs)
        train_X[:, 2] = X3_FIXED  # Fix x3

        # Use same test function
        train_Y = (
            train_X[:, 0] + train_X[:, 1] ** 2 + torch.sin(train_X[:, 2]) + math.pi / 2
        ).unsqueeze(-1)

        # Fit model (disable outcome transform for variance comparison)
        oak = OrthogonalAdditiveKernel(RBFKernel(), dim=d, **tkwargs)
        model = OrthogonalAdditiveGP(
            train_X,
            train_Y,
            covar_module=oak,
            outcome_transform=None,  # Disable for variance comparison
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        # Evaluate on grid
        grid = torch.linspace(0, 1, GROUND_TRUTH_GRID_SIZE, **tkwargs)
        (_, _), (fo_means, fo_vars) = model.evaluate_first_order_on_grid(grid)

        # Identify grid points near vs far from the fixed x3 value
        NEAR_THRESHOLD = 0.15
        near_fixed = (grid - X3_FIXED).abs() < NEAR_THRESHOLD
        far_from_fixed = ~near_fixed

        # x1, x2 components should have low variance (we have data covering [0,1])
        var_x1 = fo_vars[0].mean()
        var_x2 = fo_vars[1].mean()

        # x3 component should have HIGH variance FAR from the fixed point
        var_x3_far = fo_vars[2, far_from_fixed].mean()

        # x3 component should have LOW variance NEAR the fixed point
        var_x3_near = fo_vars[2, near_fixed].mean()

        # Assert uncertainty structure:
        # 1. x3 variance far from data >> x3 variance near data
        self.assertGreater(
            var_x3_far.item(),
            var_x3_near.item() * PARTIAL_OBS_VARIANCE_RATIO,
            f"x3 var far ({var_x3_far:.4f}) should be >> near ({var_x3_near:.4f})",
        )

        # 2. x3 variance far from data >> x1, x2 variance (which have full coverage)
        self.assertGreater(
            var_x3_far.item(),
            var_x1.item() * PARTIAL_OBS_VARIANCE_RATIO,
            f"x3 var far ({var_x3_far:.4f}) should be >> x1 var ({var_x1:.4f})",
        )
        self.assertGreater(
            var_x3_far.item(),
            var_x2.item() * PARTIAL_OBS_VARIANCE_RATIO,
            f"x3 var far ({var_x3_far:.4f}) should be >> x2 var ({var_x2:.4f})",
        )
