#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

import torch
from botorch import models
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    GeneralizedLinearPath,
    PathList,
)
from botorch.sampling.pathwise.utils import is_finite_dimensional
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestDrawKernelFeaturePaths(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        config = TestCaseConfig(seed=0, device=self.device)
        batch_config = replace(config, batch_shape=Size([2]))

        self.base_models = [
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module("FixedNoiseGP", batch_config)),
            (batch_config, gen_module(models.MultiTaskGP, batch_config)),
            (config, gen_module(models.SingleTaskVariationalGP, config)),
        ]
        self.model_lists = [
            (batch_config, gen_module(models.ModelListGP, batch_config))
        ]

    def test_base_models(self, slack: float = 3.0):
        sample_shape = Size([32, 32])
        for config, model in self.base_models:
            kernel = (
                model.model.covar_module
                if isinstance(model, models.SingleTaskVariationalGP)
                else model.covar_module
            )
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                paths = draw_kernel_feature_paths(
                    model=model,
                    sample_shape=sample_shape,
                    num_random_features=config.num_random_features,
                )
                self.assertIsInstance(paths, GeneralizedLinearPath)
                n = 16
                X = gen_random_inputs(model, batch_shape=[n], transformed=False)

            prior = model.forward(X if model.training else model.input_transform(X))
            if isinstance(prior, MultitaskMultivariateNormal):
                num_tasks = kernel.batch_shape[0]
                exact_mean = prior.mean.view(num_tasks, n)
                exact_covar = prior.covariance_matrix.view(num_tasks, n, num_tasks, n)
                exact_covar = torch.stack(
                    [exact_covar[..., i, :, i, :] for i in range(num_tasks)], dim=-3
                )
            else:
                exact_mean = prior.loc
                exact_covar = prior.covariance_matrix

            istd = exact_covar.diagonal(dim1=-2, dim2=-1).rsqrt()
            exact_mean = istd * exact_mean
            exact_covar = istd.unsqueeze(-1) * exact_covar * istd.unsqueeze(-2)

            samples = paths(X)
            if hasattr(model, "outcome_transform"):
                model.outcome_transform.train(mode=False)
                if kernel.batch_shape:
                    samples, _ = model.outcome_transform(samples.transpose(-2, -1))
                    samples = samples.transpose(-2, -1)
                else:
                    samples, _ = model.outcome_transform(samples.unsqueeze(-1))
                    samples = samples.squeeze(-1)
                model.outcome_transform.train(mode=model.training)

            samples = istd * samples.view(-1, *samples.shape[len(sample_shape) :])
            sample_mean = samples.mean(dim=0)
            sample_covar = (samples - sample_mean).permute(*range(1, samples.ndim), 0)
            sample_covar = torch.divide(
                sample_covar @ sample_covar.transpose(-2, -1), sample_shape.numel()
            )

            allclose_kwargs = {"atol": slack * sample_shape.numel() ** -0.5}
            if not is_finite_dimensional(kernel):
                num_random_features_per_map = config.num_random_features / (
                    1
                    if not is_finite_dimensional(kernel, max_depth=0)
                    else sum(
                        not is_finite_dimensional(k)
                        for k in kernel.modules()
                        if k is not kernel
                    )
                )
                allclose_kwargs["atol"] += slack * num_random_features_per_map**-0.5
            self.assertTrue(exact_mean.allclose(sample_mean, **allclose_kwargs))
            self.assertTrue(exact_covar.allclose(sample_covar, **allclose_kwargs))

    def test_model_lists(self):
        sample_shape = Size([32, 32])
        for config, model_list in self.model_lists:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                path_list = draw_kernel_feature_paths(
                    model=model_list,
                    sample_shape=sample_shape,
                    num_random_features=config.num_random_features,
                )
                self.assertIsInstance(path_list, PathList)

                X = gen_random_inputs(model_list.models[0], batch_shape=[4])
                sample_list = path_list(X)
                self.assertIsInstance(sample_list, list)
                self.assertEqual(len(sample_list), len(model_list.models))
                for path, sample in zip(path_list, sample_list):
                    self.assertTrue(path(X).equal(sample))

    def test_weight_generator_custom(self):
        """Test custom weight generator in prior_samplers.py"""
        from botorch.sampling.pathwise.prior_samplers import (
            _draw_kernel_feature_paths_fallback,
        )
        from gpytorch.kernels import RBFKernel

        # Create kernel with ard_num_dims to avoid num_ambient_inputs issue
        kernel = RBFKernel(ard_num_dims=2)
        sample_shape = torch.Size([2, 3])

        # Custom weight generator
        def custom_weight_generator(weight_shape):
            return torch.ones(weight_shape)

        result = _draw_kernel_feature_paths_fallback(
            mean_module=None,
            covar_module=kernel,
            sample_shape=sample_shape,
            weight_generator=custom_weight_generator,
        )

        # Verify the result
        self.assertIsNotNone(result.weight)
        # Weight should be all ones (from our custom generator)
        self.assertTrue(torch.allclose(result.weight, torch.ones_like(result.weight)))

    def test_fallback_edge_cases(self):
        """Test edge cases in _draw_kernel_feature_paths_fallback."""
        from botorch.sampling.pathwise.prior_samplers import (
            _draw_kernel_feature_paths_fallback,
        )
        from gpytorch.kernels import RBFKernel
        from gpytorch.means import ZeroMean

        # Test with is_ensemble=True
        kernel = RBFKernel(ard_num_dims=2)
        result = _draw_kernel_feature_paths_fallback(
            mean_module=ZeroMean(),
            covar_module=kernel,
            sample_shape=Size([2]),
            is_ensemble=True,
        )
        self.assertTrue(result.is_ensemble)

        # Test with custom weight generator
        def custom_weight_generator(shape):
            return torch.ones(shape)

        result = _draw_kernel_feature_paths_fallback(
            mean_module=None,
            covar_module=kernel,
            sample_shape=Size([2]),
            weight_generator=custom_weight_generator,
        )
        self.assertTrue(torch.allclose(result.weight, torch.ones_like(result.weight)))

    def test_approximategp_dispatcher(self):
        """Test ApproximateGP dispatcher registration."""
        from botorch.sampling.pathwise.prior_samplers import DrawKernelFeaturePaths
        from gpytorch.models import ApproximateGP
        from gpytorch.variational import VariationalStrategy

        # Create a proper ApproximateGP with variational strategy
        inducing_points = torch.rand(5, 2)
        variational_strategy = VariationalStrategy(
            None, inducing_points, torch.rand(5, 2)
        )

        class MockApproximateGP(ApproximateGP):
            def __init__(self, variational_strategy):
                super().__init__(variational_strategy)
                from gpytorch.kernels import RBFKernel
                from gpytorch.means import ZeroMean

                self.mean_module = ZeroMean()
                self.covar_module = RBFKernel(ard_num_dims=2)

        model = MockApproximateGP(variational_strategy)

        # This should trigger the dispatcher registration for ApproximateGP
        result = DrawKernelFeaturePaths(model, sample_shape=Size([2]))
        self.assertIsNotNone(result)

    def test_task_specific_mean_fallback(self):
        """Test _TaskSpecificMean fallback when mean has 1D output."""
        from botorch.sampling.pathwise.prior_samplers import _TaskSpecificMean
        from gpytorch.means import ConstantMean

        # Test with a simple mean that returns 1D output
        simple_mean = ConstantMean()
        wrapped_mean = _TaskSpecificMean(simple_mean, task_feature=1)

        # Input with task indicators
        x = torch.rand(5, 2)
        output = wrapped_mean(x)

        # Should fall through to return all_means directly
        self.assertEqual(output.ndim, 1)

    def test_multitask_gp_kernel_handling(self):
        """Test MultiTaskGP kernel handling for various kernel configurations."""
        from botorch.models import MultiTaskGP
        from gpytorch.kernels import IndexKernel, ProductKernel, RBFKernel

        train_X = torch.rand(8, 3, device=self.device, dtype=torch.float64)
        train_Y = torch.rand(8, 1, device=self.device, dtype=torch.float64)

        # Test automatic IndexKernel creation when task kernel is missing
        model1 = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        k1 = RBFKernel()
        k1.active_dims = torch.tensor([0])
        k2 = RBFKernel()
        k2.active_dims = torch.tensor([1])
        model1.covar_module = ProductKernel(k1, k2)  # No task kernel

        paths1 = draw_kernel_feature_paths(model1, sample_shape=Size([1]))
        self.assertIsNotNone(paths1)

        # Test fallback to simple kernel structure
        model2 = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        simple_kernel = RBFKernel(ard_num_dims=3)
        model2.covar_module = simple_kernel  # Non-ProductKernel

        paths2 = draw_kernel_feature_paths(model2, sample_shape=Size([1]))
        self.assertIsNotNone(paths2)

        # Test kernel without active_dims to trigger active_dims assignment
        model3 = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        k3 = RBFKernel()  # No active_dims set
        k4 = IndexKernel(num_tasks=2, rank=1, active_dims=[2])  # Task kernel
        model3.covar_module = ProductKernel(k3, k4)

        paths3 = draw_kernel_feature_paths(model3, sample_shape=Size([1]))
        self.assertIsNotNone(paths3)

    def test_multitask_gp_missing_data_kernel_error(self):
        """Test error when ProductKernel has no identifiable data kernel."""
        from botorch.models import MultiTaskGP
        from gpytorch.kernels import IndexKernel, ProductKernel

        train_X = torch.rand(8, 3, device=self.device, dtype=torch.float64)
        train_Y = torch.rand(8, 1, device=self.device, dtype=torch.float64)

        # Create model with ProductKernel of only task kernels (both have task_index)
        model = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        # Both kernels have active_dims containing task_index (2)
        k1 = IndexKernel(num_tasks=2, rank=1, active_dims=[2])
        k2 = IndexKernel(num_tasks=2, rank=1, active_dims=[2])
        model.covar_module = ProductKernel(k1, k2)

        with self.assertRaises(ValueError) as cm:
            draw_kernel_feature_paths(model, sample_shape=Size([1]))
        self.assertIn("Could not identify data kernel", str(cm.exception))
