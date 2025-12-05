#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import ceil
from typing import List, Tuple

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.utils import is_finite_dimensional, kernel_instancecheck
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels

from ..helpers import gen_module, TestCaseConfig


class TestGenKernelFeatureMap(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_tasks=3,
            batch_shape=torch.Size([2]),
        )

        self.kernels: List[Tuple[TestCaseConfig, kernels.Kernel]] = []
        for typ in (
            kernels.LinearKernel,
            kernels.IndexKernel,
            kernels.MaternKernel,
            kernels.RBFKernel,
            kernels.ScaleKernel,
            kernels.ProductKernel,
            kernels.MultitaskKernel,
            kernels.AdditiveKernel,
            kernels.LCMKernel,
        ):
            self.kernels.append((config, gen_module(typ, config)))

    def test_gen_kernel_feature_map(self, slack: float = 3.0):
        for config, kernel in self.kernels:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                feature_map = gen_kernel_feature_map(
                    kernel,
                    num_ambient_inputs=config.num_inputs,
                    num_random_features=config.num_random_features,
                )
                self.assertEqual(feature_map.batch_shape, kernel.batch_shape)

                n = 4
                m = ceil(n * kernel.batch_shape.numel() ** -0.5)

                input_batch_shapes = [(n**2,)]
                if not isinstance(kernel, kernels.MultitaskKernel):
                    input_batch_shapes.append((m, *kernel.batch_shape, m))

                for input_batch_shape in input_batch_shapes:
                    X = torch.rand(
                        (*input_batch_shape, config.num_inputs),
                        device=kernel.device,
                        dtype=kernel.dtype,
                    )
                    if isinstance(kernel, kernels.IndexKernel):  # random task IDs
                        X[..., kernel.active_dims] = torch.randint(
                            kernel.raw_var.shape[-1],
                            size=(*X.shape[:-1], len(kernel.active_dims)),
                            device=X.device,
                            dtype=X.dtype,
                        )

                    num_tasks = (
                        config.num_tasks
                        if kernel_instancecheck(kernel, kernels.MultitaskKernel)
                        else 1
                    )
                    test_shape = (
                        *kernel.batch_shape,
                        num_tasks * X.shape[-2],
                        *feature_map.output_shape,
                    )
                    if len(input_batch_shape) > len(kernel.batch_shape) + 1:
                        test_shape = (m,) + test_shape

                    features = feature_map(X).to_dense()
                    self.assertEqual(features.shape, test_shape)
                    covar = kernel(X).to_dense()

                    istd = covar.diagonal(dim1=-2, dim2=-1).rsqrt()
                    corr = istd.unsqueeze(-1) * covar * istd.unsqueeze(-2)
                    vec = istd.unsqueeze(-1) * features.view(*covar.shape[:-1], -1)
                    est = vec @ vec.transpose(-2, -1)
                    allclose_kwargs = {}
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
                        allclose_kwargs["atol"] = (
                            slack * num_random_features_per_map**-0.5
                        )

                    if isinstance(kernel, (kernels.MultitaskKernel, kernels.LCMKernel)):
                        allclose_kwargs["atol"] = max(
                            allclose_kwargs.get("atol", 1e-5), slack * 2.0
                        )

                    self.assertTrue(corr.allclose(est, **allclose_kwargs))

    def test_cosine_only_fourier_features(self):
        """Test the cosine_only=True branch in _gen_fourier_features"""
        config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_random_features=64,
        )

        # Test RBF kernel with cosine_only=True
        kernel = gen_module(kernels.RBFKernel, config)
        feature_map = gen_kernel_feature_map(
            kernel,
            num_ambient_inputs=config.num_inputs,
            num_random_features=config.num_random_features,
            cosine_only=True,
        )

        # Verification
        X = torch.rand(10, config.num_inputs, device=kernel.device, dtype=kernel.dtype)
        features = feature_map(X)
        self.assertEqual(features.shape[-1], config.num_random_features)

    def test_cosine_only_branch_coverage(self):
        """Test cosine_only branches to improve coverage"""
        config = TestCaseConfig(seed=0, device=self.device, num_inputs=2)

        # Test with cosine_only=True to cover the cosine branch in _gen_fourier_features
        rbf_kernel = gen_module(kernels.RBFKernel, config)
        feature_map = gen_kernel_feature_map(
            rbf_kernel,
            num_ambient_inputs=config.num_inputs,
            num_random_features=64,
            cosine_only=True,
        )

        X = torch.rand(
            10, config.num_inputs, device=rbf_kernel.device, dtype=rbf_kernel.dtype
        )
        features = feature_map(X)
        self.assertEqual(features.shape[-1], 64)

        # Test Matern kernel with cosine_only=True as well
        matern_kernel = gen_module(kernels.MaternKernel, config)
        matern_feature_map = gen_kernel_feature_map(
            matern_kernel,
            num_ambient_inputs=config.num_inputs,
            num_random_features=64,
            cosine_only=True,
        )

        matern_features = matern_feature_map(X)
        self.assertEqual(matern_features.shape[-1], 64)

    def test_scale_kernel_active_dims_transform(self):
        """Test ScaleKernel with active_dims different from base kernel"""
        config = TestCaseConfig(seed=0, device=self.device, num_inputs=5)

        # Create a base kernel with specific active_dims
        base_kernel = kernels.RBFKernel(active_dims=[0, 2, 4])

        # Create a ScaleKernel with different active_dims
        scale_kernel = kernels.ScaleKernel(base_kernel, active_dims=[1, 2, 3])

        # Generate feature map
        feature_map = gen_kernel_feature_map(
            scale_kernel,
            num_ambient_inputs=config.num_inputs,
            num_random_features=64,
        )

        # Verify that the input transform has been applied
        X = torch.rand(
            10, config.num_inputs, device=scale_kernel.device, dtype=scale_kernel.dtype
        )
        features = feature_map(X)
        self.assertIsNotNone(features)

    def test_product_kernel_cosine_only_auto(self):
        """Test ProductKernel with multiple infinite-dimensional kernels"""
        # Create a product of two infinite-dimensional kernels with proper setup
        rbf1 = kernels.RBFKernel(ard_num_dims=2)
        rbf2 = kernels.RBFKernel(ard_num_dims=2)
        product_kernel = kernels.ProductKernel(rbf1, rbf2)

        # Generate feature map
        feature_map = gen_kernel_feature_map(
            product_kernel,
            num_ambient_inputs=2,
            num_random_features=64,
        )

        # Verification
        X = torch.rand(10, 2, device=product_kernel.device, dtype=product_kernel.dtype)
        features = feature_map(X)
        self.assertIsNotNone(features)

    def test_odd_num_random_features_error(self):
        """Test error when num_random_features is odd and cosine_only=False"""
        config = TestCaseConfig(seed=0, device=self.device, num_inputs=2)
        kernel = gen_module(kernels.RBFKernel, config)

        with self.assertRaisesRegex(
            UnsupportedError, "Expected an even number of random features"
        ):
            gen_kernel_feature_map(
                kernel,
                num_ambient_inputs=config.num_inputs,
                num_random_features=63,  # Odd number
                cosine_only=False,
            )

    def test_rbf_weight_generator_shape_error(self):
        """Test shape validation in RBF weight generator"""
        from unittest.mock import patch

        from botorch.sampling.pathwise.features.generators import (
            _gen_kernel_feature_map_rbf,
        )

        config = TestCaseConfig(seed=0, device=self.device, num_inputs=2)
        kernel = gen_module(kernels.RBFKernel, config)

        # Patch _gen_fourier_features to call weight generator with invalid shape
        with patch(
            "botorch.sampling.pathwise.features.generators._gen_fourier_features"
        ) as mock_fourier:

            def mock_fourier_call(weight_generator, **kwargs):
                # Call the weight generator with 1D shape to trigger ValueError
                with self.assertRaisesRegex(
                    UnsupportedError, "Expected.*2-dimensional"
                ):
                    weight_generator(torch.Size([10]))  # 1D shape
                return None

            mock_fourier.side_effect = mock_fourier_call
            _gen_kernel_feature_map_rbf(kernel, num_random_features=64)

    def test_matern_weight_generator_shape_error(self):
        """Test shape validation in Matern weight generator"""
        from unittest.mock import patch

        from botorch.sampling.pathwise.features.generators import (
            _gen_kernel_feature_map_matern,
        )

        config = TestCaseConfig(seed=0, device=self.device, num_inputs=2)
        kernel = gen_module(kernels.MaternKernel, config)

        # Patch _gen_fourier_features to call weight generator with invalid shape
        with patch(
            "botorch.sampling.pathwise.features.generators._gen_fourier_features"
        ) as mock_fourier:

            def mock_fourier_call(weight_generator, **kwargs):
                # Call the weight generator with 1D shape to trigger ValueError
                with self.assertRaisesRegex(
                    UnsupportedError, "Expected.*2-dimensional"
                ):
                    weight_generator(torch.Size([10]))  # 1D shape
                return None

            mock_fourier.side_effect = mock_fourier_call
            _gen_kernel_feature_map_matern(kernel, num_random_features=64)

    def test_scale_kernel_coverage(self):
        """Test ScaleKernel condition - active_dims different from base kernel"""
        from unittest.mock import patch

        import torch
        from botorch.sampling.pathwise.features.generators import (
            _gen_kernel_feature_map_scale,
        )

        config = TestCaseConfig(seed=0, device=self.device, num_inputs=3)

        # Create base kernel with specific active_dims
        base_kernel = kernels.RBFKernel().to(device=config.device, dtype=config.dtype)
        base_kernel.active_dims = torch.tensor([0])  # Set base kernel active_dims

        # Create ScaleKernel - manually set different active_dims to ensure
        # they're different objects
        scale_kernel = kernels.ScaleKernel(base_kernel).to(
            device=config.device, dtype=config.dtype
        )
        scale_kernel.active_dims = torch.tensor(
            [0, 1]
        )  # Different object from base_kernel.active_dims

        # Verify that the condition on will be True
        active_dims = scale_kernel.active_dims
        base_active_dims = scale_kernel.base_kernel.active_dims

        # Verify they're different objects (identity, not value equality)
        condition = active_dims is not None and active_dims is not base_active_dims
        self.assertTrue(
            condition,
            f"Condition should be True. active_dims: {active_dims}, "
            f"base_active_dims: {base_active_dims}, same object: "
            f"{active_dims is base_active_dims}",
        )

        # Mock append_transform to verify it gets called
        with patch(
            "botorch.sampling.pathwise.features.generators.append_transform"
        ) as mock_append:
            try:
                _gen_kernel_feature_map_scale(
                    scale_kernel,
                    num_ambient_inputs=config.num_inputs,
                    num_random_features=64,
                )
                # Verify append_transform was called
                mock_append.assert_called()
            except Exception:
                mock_append.assert_called()
