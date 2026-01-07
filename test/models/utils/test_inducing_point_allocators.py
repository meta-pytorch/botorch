#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.utils.inducing_point_allocators import (
    _pivoted_cholesky_init,
    ExpectedImprovementQualityFunction,
    GreedyImprovementReduction,
    GreedyVarianceReduction,
    UnitQualityFunction,
)
from botorch.utils.testing import BotorchTestCase

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO


class TestUnitQualityFunction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.quality_function = UnitQualityFunction()

    def test_returns_ones_and_correct_shape(self):
        train_X = torch.rand(15, 1, device=self.device)
        scores = self.quality_function(train_X)
        self.assertTrue(torch.equal(scores, torch.ones([15], device=self.device)))


class TestExpectedImprovementQualityFunction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        train_X = torch.rand(10, 1, device=self.device)
        train_y = torch.sin(train_X) + torch.randn_like(train_X) * 0.2

        self.previous_model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood()
        ).to(self.device)

        mll = VariationalELBO(
            self.previous_model.likelihood, self.previous_model.model, num_data=10
        )
        loss = -mll(
            self.previous_model.likelihood(self.previous_model(train_X)), train_y
        ).sum()
        loss.backward()

    def test_returns_correct_shape(self):
        train_X = torch.rand(15, 1, device=self.device)
        for maximize in [True, False]:
            quality_function = ExpectedImprovementQualityFunction(
                self.previous_model, maximize=maximize
            )
            scores = quality_function(train_X)
            self.assertEqual(scores.shape, torch.Size([15]))

    def test_raises_for_multi_output_model(self):
        train_X = torch.rand(15, 1, device=self.device)
        mo_model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood(), num_outputs=5
        ).to(self.device)
        with self.assertRaises(NotImplementedError):
            ExpectedImprovementQualityFunction(mo_model, maximize=True)

    def test_different_for_maximize_and_minimize(self):
        train_X = torch.rand(15, 1, device=self.device)

        quality_function_for_max = ExpectedImprovementQualityFunction(
            self.previous_model, maximize=True
        )
        scores_for_max = quality_function_for_max(train_X)

        quality_function_for_min = ExpectedImprovementQualityFunction(
            self.previous_model, maximize=False
        )
        scores_for_min = quality_function_for_min(train_X)

        self.assertFalse(torch.equal(scores_for_min, scores_for_max))

    def test_ei_calc_via_monte_carlo(self):
        for maximize in [True, False]:
            train_X = torch.rand(10, 1, device=self.device)
            posterior = self.previous_model.posterior(train_X)
            mean = posterior.mean.squeeze(-2).squeeze(-1)
            sigma = posterior.variance.sqrt().view(mean.shape)
            normal = torch.distributions.Normal(mean, sigma)
            samples = normal.sample([1_000_000])
            if maximize:
                baseline = torch.min(mean)
                ei = torch.clamp(samples - baseline, min=0.0).mean(axis=0)
            else:
                baseline = torch.max(mean)
                ei = torch.clamp(baseline - samples, min=0.0).mean(axis=0)

            quality_function = ExpectedImprovementQualityFunction(
                self.previous_model, maximize
            )

            self.assertAllClose(ei, quality_function(train_X), atol=0.01, rtol=0.01)


class TestGreedyVarianceReduction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.ipa = GreedyVarianceReduction()

    def test_initialization(self):
        self.assertIsInstance(self.ipa, GreedyVarianceReduction)

    def test_allocate_inducing_points_doesnt_leak(self) -> None:
        """
        Run 'allocate_inducing_points' and check that all tensors allocated
        in that function are garbabe-collected.
        """

        def _get_n_tensors_tracked_by_gc() -> int:
            gc.collect()
            return sum(1 for elt in gc.get_objects() if isinstance(elt, torch.Tensor))

        def f() -> None:
            """Construct and use a GreedyVarianceReduction allocator."""
            x = torch.rand(7, 3).to(self.device)
            kernel = ScaleKernel(MaternKernel())
            allocator = GreedyVarianceReduction()
            allocator.allocate_inducing_points(x, kernel, 4, x.shape[:-2])

        n_tensors_before = _get_n_tensors_tracked_by_gc()
        f()
        n_tensors_after = _get_n_tensors_tracked_by_gc()

        self.assertEqual(n_tensors_before, n_tensors_after)

    def test_inducing_points_shape_and_repeatability(self):
        for train_X in [
            torch.rand(15, 1, device=self.device),  # single task
            torch.rand(2, 15, 1, device=self.device),  # batched inputs
        ]:
            inducing_points_1 = self.ipa.allocate_inducing_points(
                inputs=train_X,
                covar_module=MaternKernel(),
                num_inducing=5,
                input_batch_shape=torch.Size([]),
            )

            inducing_points_2 = self.ipa.allocate_inducing_points(
                inputs=train_X,
                covar_module=MaternKernel(),
                num_inducing=5,
                input_batch_shape=torch.Size([]),
            )

            if len(train_X) == 3:  # batched inputs
                self.assertEqual(inducing_points_1.shape, (2, 5, 1))
                self.assertEqual(inducing_points_2.shape, (2, 5, 1))
            else:
                self.assertEqual(inducing_points_1.shape, (5, 1))
                self.assertEqual(inducing_points_2.shape, (5, 1))
            self.assertAllClose(inducing_points_1, inducing_points_2)

    def test_that_we_dont_get_redundant_inducing_points(self):
        train_X = torch.rand(15, 1, device=self.device)
        stacked_train_X = torch.cat((train_X, train_X), dim=0)
        num_inducing = 20
        inducing_points_1 = self.ipa.allocate_inducing_points(
            inputs=stacked_train_X,
            covar_module=MaternKernel(),
            num_inducing=num_inducing,
            input_batch_shape=torch.Size([]),
        )
        # should not have 20 inducing points when 15 singular dimensions
        # are passed
        self.assertLess(inducing_points_1.shape[-2], num_inducing)


class TestGreedyImprovementReduction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        train_X = torch.rand(10, 1, device=self.device)
        train_y = torch.sin(train_X) + torch.randn_like(train_X) * 0.2

        self.previous_model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood()
        ).to(self.device)

        mll = VariationalELBO(
            self.previous_model.likelihood, self.previous_model.model, num_data=10
        )
        loss = -mll(
            self.previous_model.likelihood(self.previous_model(train_X)), train_y
        ).sum()
        loss.backward()

        self.ipa = GreedyImprovementReduction(self.previous_model, maximize=True)

    def test_initialization(self):
        self.assertIsInstance(self.ipa, GreedyImprovementReduction)
        self.assertIsInstance(self.ipa._model, SingleTaskVariationalGP)
        self.assertEqual(self.ipa._maximize, True)

    def test_raises_for_multi_output_model(self):
        train_X = torch.rand(10, 1, device=self.device)
        model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood(), num_outputs=5
        ).to(self.device)
        ipa = GreedyImprovementReduction(model, maximize=True)
        with self.assertRaises(NotImplementedError):
            ipa.allocate_inducing_points(
                inputs=train_X,
                covar_module=MaternKernel(),
                num_inducing=5,
                input_batch_shape=torch.Size([]),
            )

    def test_inducing_points_shape_and_repeatability(self):
        train_X = torch.rand(15, 1, device=self.device)

        for train_X in [
            torch.rand(15, 1, device=self.device),  # single task
            torch.rand(2, 15, 1, device=self.device),  # batched inputs
        ]:
            inducing_points_1 = self.ipa.allocate_inducing_points(
                inputs=train_X,
                covar_module=MaternKernel(),
                num_inducing=5,
                input_batch_shape=torch.Size([]),
            )

            inducing_points_2 = self.ipa.allocate_inducing_points(
                inputs=train_X,
                covar_module=MaternKernel(),
                num_inducing=5,
                input_batch_shape=torch.Size([]),
            )

            if len(train_X) == 3:  # batched inputs
                self.assertEqual(inducing_points_1.shape, (2, 5, 1))
                self.assertEqual(inducing_points_2.shape, (2, 5, 1))
            else:
                self.assertEqual(inducing_points_1.shape, (5, 1))
                self.assertEqual(inducing_points_2.shape, (5, 1))
            self.assertAllClose(inducing_points_1, inducing_points_2)

    def test_that_we_dont_get_redundant_inducing_points(self):
        train_X = torch.rand(15, 1, device=self.device)
        stacked_train_X = torch.cat((train_X, train_X), dim=0)
        num_inducing = 20
        inducing_points_1 = self.ipa.allocate_inducing_points(
            inputs=stacked_train_X,
            covar_module=MaternKernel(),
            num_inducing=num_inducing,
            input_batch_shape=torch.Size([]),
        )
        # should not have 20 inducing points when 15 singular dimensions
        # are passed
        self.assertLess(inducing_points_1.shape[-2], num_inducing)

    def test_inducing_points_different_when_minimizing(self):
        ipa_for_max = GreedyImprovementReduction(self.previous_model, maximize=True)
        ipa_for_min = GreedyImprovementReduction(self.previous_model, maximize=False)

        train_X = torch.rand(15, 1, device=self.device)
        inducing_points_for_max = ipa_for_max.allocate_inducing_points(
            inputs=train_X,
            covar_module=MaternKernel(),
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )
        inducing_points_for_min = ipa_for_min.allocate_inducing_points(
            inputs=train_X,
            covar_module=MaternKernel(),
            num_inducing=10,
            input_batch_shape=torch.Size([]),
        )

        self.assertFalse(torch.equal(inducing_points_for_min, inducing_points_for_max))


class TestPivotedCholeskyInit(BotorchTestCase):
    def test_raises_for_quality_function_with_invalid_shape(self):
        inputs = torch.rand(15, 1, device=self.device)
        with torch.no_grad():
            train_train_kernel = (
                MaternKernel().to(self.device)(inputs).evaluate_kernel()
            )
        quality_scores = torch.ones([10, 1], device=self.device)
        with self.assertRaisesRegex(ValueError, ".*requires a quality score"):
            _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=train_train_kernel,
                max_length=10,
                quality_scores=quality_scores,
            )

    def test_raises_for_kernel_with_grad(self) -> None:
        inputs = torch.rand(15, 1, device=self.device)
        train_train_kernel = MaternKernel().to(self.device)(inputs).evaluate_kernel()
        quality_scores = torch.ones(15, device=self.device)
        with self.assertRaisesRegex(
            UnsupportedError,
            "`_pivoted_cholesky_init` does not support using a `kernel_matrix` "
            "with `requires_grad=True`.",
        ):
            _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=train_train_kernel,
                max_length=10,
                quality_scores=quality_scores,
            )
