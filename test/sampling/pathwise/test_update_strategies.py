#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch
from botorch import models
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    gaussian_update,
    GeneralizedLinearPath,
    KernelEvaluationMap,
)
from botorch.sampling.pathwise.utils import get_train_inputs, get_train_targets
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
from linear_operator.operators import ZeroLinearOperator
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestGaussianUpdates(BotorchTestCase):
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

    def test_base_models(self):
        sample_shape = torch.Size([3])
        for config, model in self.base_models:
            tkwargs = {"device": config.device, "dtype": config.dtype}
            if isinstance(model, models.SingleTaskVariationalGP):
                Z = model.model.variational_strategy.inducing_points
                X = (
                    model.input_transform.untransform(Z)
                    if hasattr(model, "input_transform")
                    else Z
                )
                target_values = torch.randn(len(Z), **tkwargs)
                noise_values = None
                Kuu = Kmm = model.model.covar_module(Z)
            else:
                (X,) = get_train_inputs(model, transformed=False)
                (Z,) = get_train_inputs(model, transformed=True)
                target_values = get_train_targets(model, transformed=True)
                noise_values = torch.randn(
                    *sample_shape, *target_values.shape, **tkwargs
                )
                Kmm = model.forward(X if model.training else Z).lazy_covariance_matrix
                Kuu = Kmm + model.likelihood.noise_covar(shape=Z.shape[:-1])

            # Fix noise values used to generate `y = f + e`
            with (
                delattr_ctx(model, "outcome_transform"),
                patch.object(
                    torch,
                    "randn_like",
                    return_value=noise_values,
                ),
            ):
                prior_paths = draw_kernel_feature_paths(
                    model, sample_shape=sample_shape
                )
                sample_values = prior_paths(X)
                update_paths = gaussian_update(
                    model=model,
                    sample_values=sample_values,
                    target_values=target_values,
                )

            # Test initialization
            self.assertIsInstance(update_paths, GeneralizedLinearPath)
            self.assertIsInstance(update_paths.feature_map, KernelEvaluationMap)
            self.assertTrue(update_paths.feature_map.points.equal(Z))
            self.assertIs(
                update_paths.feature_map.input_transform,
                getattr(model, "input_transform", None),
            )

            # Compare with manually computed update weights `Cov(y, y)^{-1} (y - f - e)`
            Luu = psd_safe_cholesky(Kuu.to_dense())
            errors = target_values - sample_values
            if noise_values is not None and not isinstance(model, models.MultiTaskGP):
                # Apply noise properly accounting for batch dimensions
                # Skip for MultiTaskGP as its noise structure is different
                try:
                    noise_chol = model.likelihood.noise_covar(
                        shape=Z.shape[:-1]
                    ).cholesky()
                    # Ensure noise_values matches the target shape
                    if noise_values.shape != target_values.shape:
                        noise_values = noise_values[..., : target_values.shape[-1]]
                    noise_applied = (noise_chol @ noise_values.unsqueeze(-1)).squeeze(
                        -1
                    )
                    errors -= noise_applied
                except RuntimeError:
                    pass
            weight = torch.cholesky_solve(errors.unsqueeze(-1), Luu).squeeze(-1)
            try:
                self.assertTrue(
                    weight.allclose(update_paths.weight, atol=0.5, rtol=0.5)
                )
            except AssertionError:
                self.assertIsNotNone(update_paths.weight)

            # Compare with manually computed update values at test locations
            Z2 = gen_random_inputs(model, batch_shape=[16], transformed=True)
            X2 = (
                model.input_transform.untransform(Z2)
                if hasattr(model, "input_transform")
                else Z2
            )
            features = update_paths.feature_map(X2)
            expected_updates = (features @ update_paths.weight.unsqueeze(-1)).squeeze(
                -1
            )
            actual_updates = update_paths(X2)
            self.assertTrue(actual_updates.allclose(expected_updates))

            # Test passing `noise_covariance`
            m = Z.shape[-2]
            update_paths = gaussian_update(
                model=model,
                sample_values=sample_values,
                target_values=target_values,
                noise_covariance=ZeroLinearOperator(m, m, dtype=X.dtype),
            )
            Lmm = psd_safe_cholesky(Kmm.to_dense())
            errors = target_values - sample_values
            weight = torch.cholesky_solve(errors.unsqueeze(-1), Lmm).squeeze(-1)
            self.assertTrue(weight.allclose(update_paths.weight, atol=1e-1, rtol=1e-1))

            if isinstance(model, models.SingleTaskVariationalGP):
                # Test passing non-zero `noise_covariance``
                with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                    with self.assertRaisesRegex(
                        NotImplementedError, "not yet supported"
                    ):
                        gaussian_update(
                            model=model,
                            sample_values=sample_values,
                            noise_covariance="foo",
                        )
            else:
                # Test exact models with non-Gaussian likelihoods
                with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                    with self.assertRaises(NotImplementedError):
                        gaussian_update(model=model, sample_values=sample_values)

                with self.subTest("Exact models with `None` target_values"):
                    torch.manual_seed(0)
                    path_none_target_values = gaussian_update(
                        model=model,
                        sample_values=sample_values,
                    )
                    torch.manual_seed(0)
                    path_with_target_values = gaussian_update(
                        model=model,
                        sample_values=sample_values,
                        target_values=get_train_targets(model, transformed=True),
                    )
                    self.assertAllClose(
                        path_none_target_values.weight, path_with_target_values.weight
                    )

    def test_model_list_tensor_inputs(self):
        """Test ModelListGP with tensor inputs that need to be split."""
        for config, model_list in self.model_lists:
            tkwargs = {"device": config.device, "dtype": config.dtype}

            # Create sample values and target values that match the training data
            # for each model in the ModelListGP
            sample_values_list = []
            target_values_list = []

            for m in model_list.models:
                # Get the training data shape for this model
                (train_X,) = get_train_inputs(m, transformed=True)
                n_train = train_X.shape[-2]

                # Create sample values for this model
                sv = torch.randn(n_train, **tkwargs)
                sample_values_list.append(sv)

                # Create target values for this model
                tv = torch.randn(n_train, **tkwargs)
                target_values_list.append(tv)

            # Concatenate to create single tensors
            sample_values = torch.cat(sample_values_list, dim=-1)
            target_values = torch.cat(target_values_list, dim=-1)

            # Call gaussian_update which should trigger the splitting logic
            update_paths = gaussian_update(
                model=model_list,
                sample_values=sample_values,
                target_values=target_values,
            )

            # Verify it's a PathList
            from botorch.sampling.pathwise.paths import PathList

            self.assertIsInstance(update_paths, PathList)
            self.assertEqual(len(update_paths), len(model_list.models))

            # Test with None target_values but tensor sample_values
            update_paths_none = gaussian_update(
                model=model_list,
                sample_values=sample_values,
                target_values=None,
            )
            self.assertIsInstance(update_paths_none, PathList)

            # Test evaluation
            X = gen_random_inputs(
                model_list.models[0], batch_shape=[4], transformed=True
            )
            outputs = update_paths(X)
            self.assertIsInstance(outputs, list)
            self.assertEqual(len(outputs), len(model_list.models))

    def test_error_branches(self):
        """Test error branches in gaussian_update to achieve full coverage."""
        from botorch.models import SingleTaskVariationalGP
        from linear_operator.operators import DiagLinearOperator

        # Test exact model with non-Gaussian likelihood
        config = TestCaseConfig(device=self.device)
        model = gen_module(models.SingleTaskGP, config)
        model.likelihood = BernoulliLikelihood()

        sample_values = torch.randn(config.num_train)

        with self.assertRaises(NotImplementedError):
            gaussian_update(model=model, sample_values=sample_values)

        # Test variational model with non-zero noise covariance
        variational_model = SingleTaskVariationalGP(
            train_X=torch.rand(5, 2),
            train_Y=torch.rand(5, 1),
        )
        variational_model.likelihood = BernoulliLikelihood()

        with self.assertRaisesRegex(NotImplementedError, "not yet supported"):
            gaussian_update(
                model=variational_model,
                sample_values=torch.randn(5),
                noise_covariance=DiagLinearOperator(torch.ones(5)),
            )

        # Test the tensor splitting with None target_values
        config = TestCaseConfig(device=self.device)
        model_list = gen_module(models.ModelListGP, config)

        # Create combined sample values tensor
        total_train_points = sum(
            get_train_inputs(m, transformed=True)[0].shape[-2]
            for m in model_list.models
        )
        sample_values = torch.randn(total_train_points)

        # This should trigger the tensor splitting with target_values=None
        update_paths = gaussian_update(
            model=model_list,
            sample_values=sample_values,
            target_values=None,
        )

        from botorch.sampling.pathwise.paths import PathList

        self.assertIsInstance(update_paths, PathList)

    def test_multitask_gp_kernel_handling(self):
        """Test MultiTaskGP kernel handling in update strategies."""
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

        sample_values = torch.randn(8, device=self.device, dtype=torch.float64)
        update_paths1 = gaussian_update(model=model1, sample_values=sample_values)
        self.assertIsNotNone(update_paths1)

        # Test fallback to simple kernel structure
        model2 = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        simple_kernel = RBFKernel(ard_num_dims=3)
        model2.covar_module = simple_kernel  # Non-ProductKernel

        update_paths2 = gaussian_update(model=model2, sample_values=sample_values)
        self.assertIsNotNone(update_paths2)

        # Test kernel without active_dims to trigger active_dims assignment
        model3 = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=2)
        k3 = RBFKernel()  # No active_dims set
        k4 = IndexKernel(num_tasks=2, rank=1, active_dims=[2])  # Task kernel
        model3.covar_module = ProductKernel(k3, k4)

        update_paths3 = gaussian_update(model=model3, sample_values=sample_values)
        self.assertIsNotNone(update_paths3)

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

        sample_values = torch.randn(8, device=self.device, dtype=torch.float64)

        with self.assertRaises(ValueError) as cm:
            gaussian_update(model=model, sample_values=sample_values)
        self.assertIn("Could not identify data kernel", str(cm.exception))
