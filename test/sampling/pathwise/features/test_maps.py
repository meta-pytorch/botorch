#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import prod
from unittest.mock import patch

import torch
from botorch.sampling.pathwise.features import maps
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.utils.transforms import ChainedTransform, FeatureSelector
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Size
from torch.nn import Module, ModuleList

from ..helpers import gen_module, TestCaseConfig


class TestFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_tasks=3,
            batch_shape=Size([2]),
        )

        self.base_feature_maps = [
            gen_kernel_feature_map(gen_module(kernels.LinearKernel, self.config)),
            gen_kernel_feature_map(gen_module(kernels.IndexKernel, self.config)),
        ]

    def test_feature_map(self):
        feature_map = maps.FeatureMap()
        feature_map.raw_output_shape = Size([2, 3, 4])
        feature_map.output_transform = None
        feature_map.device = self.device
        feature_map.dtype = None
        self.assertEqual(feature_map.output_shape, (2, 3, 4))

        feature_map.output_transform = lambda x: torch.concat((x, x), dim=-1)
        self.assertEqual(feature_map.output_shape, (2, 3, 8))

    def test_feature_map_list(self):
        map_list = maps.FeatureMapList(feature_maps=self.base_feature_maps)
        self.assertEqual(map_list.device.type, self.config.device.type)
        self.assertEqual(map_list.dtype, self.config.dtype)

        X = torch.rand(
            16,
            self.config.num_inputs,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        output_list = map_list(X)
        self.assertIsInstance(output_list, list)
        self.assertEqual(len(output_list), len(map_list))
        for feature_map, output in zip(map_list, output_list):
            self.assertTrue(feature_map(X).to_dense().equal(output.to_dense()))

    def test_direct_sum_feature_map(self):
        feature_map = maps.DirectSumFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([sum(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(
            features.equal(torch.concat([f(X).to_dense() for f in feature_map], dim=-1))
        )

        # Test mixture of matrix-valued and vector-valued maps
        real_map = feature_map[0]

        # Create a proper feature map with 2D output
        class Mock2DFeatureMap(maps.FeatureMap):
            def __init__(self, d, batch_shape):
                super().__init__()
                self.raw_output_shape = Size([d, d])
                self.batch_shape = batch_shape
                self.input_transform = None
                self.output_transform = None
                self.device = real_map.device
                self.dtype = real_map.dtype
                self.d = d

            def forward(self, x):
                return x.unsqueeze(-1).expand(*self.batch_shape, *x.shape, self.d)

        mock_map = Mock2DFeatureMap(d, real_map.batch_shape)
        with patch.dict(
            feature_map._modules,
            {"_feature_maps_list": ModuleList([mock_map, real_map])},
        ):
            self.assertEqual(
                feature_map.output_shape, Size([d, d + real_map.output_shape[0]])
            )
            features = feature_map(X).to_dense()
            self.assertTrue(features[..., :d].equal(mock_map(X)))
            self.assertTrue(
                features[..., d:].eq((d**-0.5) * real_map(X).unsqueeze(-1)).all()
            )

    def test_hadamard_product_feature_map(self):
        feature_map = maps.HadamardProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            torch.broadcast_shapes(*(f.output_shape for f in feature_map)),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(features.equal(prod([f(X).to_dense() for f in feature_map])))

    def test_sparse_direct_sum_feature_map(self):
        feature_map = maps.SparseDirectSumFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([sum(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(
            features.equal(torch.concat([f(X).to_dense() for f in feature_map], dim=-1))
        )

        # Test mixture of matrix-valued and vector-valued maps
        real_map = feature_map[0]

        # Create a proper feature map with 2D output
        class Mock2DFeatureMap(maps.FeatureMap):
            def __init__(self, d, batch_shape):
                super().__init__()
                self.raw_output_shape = Size([d, d])
                self.batch_shape = batch_shape
                self.input_transform = None
                self.output_transform = None
                self.device = real_map.device
                self.dtype = real_map.dtype
                self.d = d

            def forward(self, x):
                return x.unsqueeze(-1).expand(*self.batch_shape, *x.shape, self.d)

        mock_map = Mock2DFeatureMap(d, real_map.batch_shape)
        with patch.dict(
            feature_map._modules,
            {"_feature_maps_list": ModuleList([mock_map, real_map])},
        ):
            self.assertEqual(
                feature_map.output_shape, Size([d, d + real_map.output_shape[0]])
            )
            features = feature_map(X).to_dense()
            self.assertTrue(features[..., :d, :d].equal(mock_map(X)))
            self.assertTrue(features[..., d:, d:].eq(real_map(X).unsqueeze(-2)).all())

    def test_outer_product_feature_map(self):
        feature_map = maps.OuterProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([prod(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )

        test_features = (
            feature_map[0](X).to_dense().unsqueeze(-1)
            * feature_map[1](X).to_dense().unsqueeze(-2)
        ).view(features.shape)
        self.assertTrue(features.equal(test_features))


class TestKernelFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.configs = [
            TestCaseConfig(
                seed=0,
                device=self.device,
                num_inputs=2,
                num_tasks=3,
                batch_shape=Size([2]),
            )
        ]

    def test_fourier_feature_map(self):
        for config in self.configs:
            tkwargs = {"device": config.device, "dtype": config.dtype}
            kernel = gen_module(kernels.RBFKernel, config)
            weight = torch.randn(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            bias = torch.rand(*kernel.batch_shape, 16, **tkwargs)
            feature_map = maps.FourierFeatureMap(
                kernel=kernel, weight=weight, bias=bias
            )
            self.assertEqual(feature_map.output_shape, (16,))

            X = torch.rand(32, config.num_inputs, **tkwargs)
            features = feature_map(X)
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(X @ weight.transpose(-2, -1) + bias.unsqueeze(-2))
            )

    def test_index_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.IndexKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            feature_map = maps.IndexKernelFeatureMap(kernel=kernel)
            self.assertEqual(feature_map.output_shape, kernel.raw_var.shape[-1:])

            X = torch.rand(*config.batch_shape, 16, config.num_inputs, **tkwargs)
            index_shape = (*config.batch_shape, 16, len(kernel.active_dims))
            indices = X[..., kernel.active_dims] = torch.randint(
                config.num_tasks, size=index_shape, **tkwargs
            )
            indices = indices.long().squeeze(-1)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )

            cholesky = kernel.covar_matrix.cholesky().to_dense()
            test_features = []
            for chol, idx in zip(
                cholesky.view(-1, *cholesky.shape[-2:]),
                indices.view(-1, *indices.shape[-1:]),
            ):
                test_features.append(chol.index_select(dim=-2, index=idx))
            test_features = torch.stack(test_features).view(features.shape)
            self.assertTrue(features.equal(test_features))

    def test_kernel_evaluation_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.RBFKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            points = torch.rand(4, config.num_inputs, **tkwargs)
            feature_map = maps.KernelEvaluationMap(kernel=kernel, points=points)
            self.assertEqual(
                feature_map.raw_output_shape, feature_map.points.shape[-2:-1]
            )

            X = torch.rand(16, config.num_inputs, **tkwargs)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(features.equal(kernel(X, points).to_dense()))

    def test_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.RBFKernel, config)
            kernel.active_dims = torch.tensor([0], device=config.device)

            feature_map = maps.KernelFeatureMap(kernel=kernel)
            self.assertEqual(feature_map.batch_shape, kernel.batch_shape)
            self.assertIsInstance(feature_map.input_transform, FeatureSelector)
            self.assertIsNone(
                maps.KernelFeatureMap(kernel, ignore_active_dims=True).input_transform
            )
            self.assertIsInstance(
                maps.KernelFeatureMap(kernel, input_transform=Module()).input_transform,
                ChainedTransform,
            )

    def test_linear_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.LinearKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            active_dims = (
                tuple(range(config.num_inputs))
                if kernel.active_dims is None
                else kernel.active_dims
            )
            feature_map = maps.LinearKernelFeatureMap(
                kernel=kernel, raw_output_shape=Size([len(active_dims)])
            )

            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(kernel.variance.sqrt() * X[..., active_dims])
            )

    def test_multitask_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.MultitaskKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            data_map = gen_kernel_feature_map(
                kernel=kernel.data_covar_module,
                num_ambient_inputs=config.num_inputs,
                num_random_features=config.num_random_features,
            )
            feature_map = maps.MultitaskKernelFeatureMap(
                kernel=kernel, data_feature_map=data_map
            )
            self.assertEqual(
                feature_map.output_shape,
                (feature_map.num_tasks * data_map.output_shape[0],)
                + data_map.output_shape[1:],
            )

            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)

            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            cholesky = kernel.task_covar_module.covar_matrix.cholesky()
            test_features = KroneckerProductLinearOperator(data_map(X), cholesky)
            self.assertTrue(features.equal(test_features.to_dense()))

    def test_feature_map_edge_cases(self):
        """Test edge cases for feature maps including empty maps and errors."""
        from botorch.exceptions.errors import UnsupportedError

        # Test empty FeatureMapList device/dtype
        empty_list = maps.FeatureMapList(feature_maps=[])
        self.assertIsNone(empty_list.device)
        self.assertIsNone(empty_list.dtype)

        # Test empty DirectSumFeatureMap
        empty_direct_sum = maps.DirectSumFeatureMap([])
        self.assertEqual(empty_direct_sum.raw_output_shape, Size([]))
        self.assertEqual(empty_direct_sum.batch_shape, Size([]))

        # Test DirectSumFeatureMap with only 0-dimensional feature maps
        class ZeroDimFeatureMap(maps.FeatureMap):
            def __init__(self):
                super().__init__()
                self.raw_output_shape = Size([])
                self.batch_shape = Size([])
                self.input_transform = None
                self.output_transform = None

            def forward(self, x):
                return torch.tensor(1.0)

        zero_dim_direct_sum = maps.DirectSumFeatureMap([ZeroDimFeatureMap()])
        self.assertEqual(zero_dim_direct_sum.raw_output_shape, Size([]))

        # Test DirectSumFeatureMap batch shape mismatch error
        class BatchMismatchFeatureMap1(maps.FeatureMap):
            def __init__(self):
                super().__init__()
                self.raw_output_shape = Size([3])
                self.batch_shape = Size([2])

            def forward(self, x):
                return torch.randn(2, x.shape[0], 3)

        class BatchMismatchFeatureMap2(maps.FeatureMap):
            def __init__(self):
                super().__init__()
                self.raw_output_shape = Size([3])
                self.batch_shape = Size([3])  # Different batch shape

            def forward(self, x):
                return torch.randn(3, x.shape[0], 3)

        mismatch_direct_sum = maps.DirectSumFeatureMap(
            [BatchMismatchFeatureMap1(), BatchMismatchFeatureMap2()]
        )
        with self.assertRaisesRegex(ValueError, "must have the same batch shapes"):
            _ = mismatch_direct_sum.batch_shape

        # Test empty HadamardProductFeatureMap device/dtype
        empty_hadamard = maps.HadamardProductFeatureMap([])
        self.assertIsNone(empty_hadamard.device)
        self.assertIsNone(empty_hadamard.dtype)

        # Test empty OuterProductFeatureMap device/dtype
        empty_outer = maps.OuterProductFeatureMap([])
        self.assertIsNone(empty_outer.device)
        self.assertIsNone(empty_outer.dtype)

        # Test KernelEvaluationMap dimension mismatch error
        kernel = gen_module(kernels.RBFKernel, self.configs[0])
        # Create points with wrong number of dimensions
        bad_points = torch.rand(
            self.configs[0].num_inputs, device=self.device
        )  # 1D instead of 2D

        with self.assertRaisesRegex(RuntimeError, "Dimension mismatch"):
            maps.KernelEvaluationMap(kernel=kernel, points=bad_points)

        # Test KernelEvaluationMap shape mismatch error
        kernel = gen_module(kernels.RBFKernel, self.configs[0])
        # Points with incompatible batch shape
        bad_points = torch.rand(3, 4, self.configs[0].num_inputs, device=self.device)
        kernel.batch_shape = Size([2])  # Incompatible with points shape

        with self.assertRaisesRegex(RuntimeError, "Shape mismatch"):
            maps.KernelEvaluationMap(kernel=kernel, points=bad_points)

        # Test IndexKernelFeatureMap with None input
        index_kernel = gen_module(kernels.IndexKernel, self.configs[0])
        index_feature_map = maps.IndexKernelFeatureMap(kernel=index_kernel)

        # Call with None input
        result = index_feature_map.forward(None)
        # Should return Cholesky of covar_matrix
        expected = index_kernel.covar_matrix.cholesky()
        self.assertTrue(result.to_dense().allclose(expected.to_dense()))

        # Test IndexKernelFeatureMap with wrong kernel type
        rbf_kernel = gen_module(kernels.RBFKernel, self.configs[0])
        with self.assertRaisesRegex(ValueError, "Expected.*IndexKernel"):
            maps.IndexKernelFeatureMap(kernel=rbf_kernel)

        # Test LinearKernelFeatureMap with wrong kernel type
        rbf_kernel = gen_module(kernels.RBFKernel, self.configs[0])
        with self.assertRaisesRegex(ValueError, "Expected.*LinearKernel"):
            maps.LinearKernelFeatureMap(kernel=rbf_kernel, raw_output_shape=Size([3]))

        # Test MultitaskKernelFeatureMap with wrong kernel type
        rbf_kernel = gen_module(kernels.RBFKernel, self.configs[0])
        data_map = gen_kernel_feature_map(rbf_kernel)
        with self.assertRaisesRegex(ValueError, "Expected.*MultitaskKernel"):
            maps.MultitaskKernelFeatureMap(kernel=rbf_kernel, data_feature_map=data_map)

        # Test FeatureMapList with device/dtype conflicts
        class DeviceFeatureMap(maps.FeatureMap):
            def __init__(self, device):
                super().__init__()
                self.raw_output_shape = Size([3])
                self.batch_shape = Size([])
                self.device = device
                self.dtype = torch.float32
                self.input_transform = None
                self.output_transform = None

            def forward(self, x):
                return torch.randn(x.shape[0], 3, device=self.device, dtype=self.dtype)

        # Force device mismatch for FeatureMapList
        device_map1 = DeviceFeatureMap(torch.device("cpu"))
        device_map2 = DeviceFeatureMap(torch.device("cpu"))
        # Create a fake device to force mismatch
        fake_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            # Force different device by creating a mock device
            device_map2.device = "fake_device"
        else:
            device_map2.device = fake_device

        if torch.cuda.is_available() or device_map2.device != device_map1.device:
            device_list = maps.FeatureMapList([device_map1, device_map2])
            with self.assertRaisesRegex(UnsupportedError, "must be colocated"):
                _ = device_list.device

        # Test multiple dtypes error
        dtype_map1 = DeviceFeatureMap(torch.device("cpu"))
        dtype_map2 = DeviceFeatureMap(torch.device("cpu"))
        dtype_map2.dtype = torch.float64

        dtype_list = maps.FeatureMapList([dtype_map1, dtype_map2])
        with self.assertRaisesRegex(UnsupportedError, "must have the same data type"):
            _ = dtype_list.dtype

        # Test DirectSumFeatureMap with mixed dimensions
        class MixedDimFeatureMap(maps.FeatureMap):
            def __init__(self, output_shape):
                super().__init__()
                self.raw_output_shape = output_shape
                self.batch_shape = Size([])
                self.input_transform = None
                self.output_transform = None
                self.device = torch.device("cpu")
                self.dtype = torch.float32

            def forward(self, x):
                return torch.randn(x.shape[0], *self.raw_output_shape)

        # Create maps with different dimensions to test the else branch
        # in raw_output_shape
        mixed_map1 = MixedDimFeatureMap(Size([2, 3]))  # 2D output
        mixed_map2 = MixedDimFeatureMap(Size([4]))  # 1D output

        mixed_direct_sum = maps.DirectSumFeatureMap([mixed_map1, mixed_map2])
        # This should trigger the else branch in raw_output_shape calculation
        shape = mixed_direct_sum.raw_output_shape
        self.assertEqual(len(shape), 2)  # Should have 2 dimensions
        self.assertEqual(shape[-1], 3 + 4)  # Concatenation dimension

        # Test specific case: mixed dimensions where lower-dim maps
        # have dimensions that need to be handled in the else branch of the inner if
        # Create a 3D map and a 2D map to force the condition: ndim < max_ndim
        # but with existing dimensions
        map_3d = MixedDimFeatureMap(Size([2, 3, 5]))  # 3D: max_ndim will be 3
        map_2d = MixedDimFeatureMap(Size([4, 6]))  # 2D: will be expanded to 3D

        # This should trigger code where ndim < max_ndim and we're in the else branch
        # for i in range(max_ndim - 1), specifically the else part where
        # idx = i - (max_ndim - ndim)
        mixed_direct_sum_2 = maps.DirectSumFeatureMap([map_3d, map_2d])
        shape_2 = mixed_direct_sum_2.raw_output_shape

        # For this case:
        # max_ndim = 3 (from map_3d)
        # map_2d has ndim = 2, so ndim < max_ndim
        # For i in range(2): i=0,1
        # For map_2d: when i >= max_ndim - ndim (i.e., i >= 3-2=1), we go to else branch
        # So when i=1, we execute: idx = 1 - (3-2) = 0,
        # result_shape[1] = max(result_shape[1], shape[0])
        self.assertEqual(len(shape_2), 3)  # Should have 3 dimensions
        self.assertEqual(shape_2[-1], 5 + 6)  # Concatenation: last dims added
        self.assertEqual(
            shape_2[0], max(2, 1)
        )  # max of first dimensions (with expansion)
        self.assertEqual(shape_2[1], max(3, 4))  # max of second dimensions

        # Force device mismatch for HadamardProductFeatureMap
        hadamard_map1 = DeviceFeatureMap(torch.device("cpu"))
        hadamard_map2 = DeviceFeatureMap(torch.device("cpu"))
        fake_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            hadamard_map2.device = "fake_device"
        else:
            hadamard_map2.device = fake_device

        if torch.cuda.is_available() or hadamard_map2.device != hadamard_map1.device:
            hadamard_list = maps.HadamardProductFeatureMap(
                [hadamard_map1, hadamard_map2]
            )
            with self.assertRaisesRegex(UnsupportedError, "must be colocated"):
                _ = hadamard_list.device

        hadamard_map2.device = torch.device("cpu")
        hadamard_map2.dtype = torch.float64
        hadamard_dtype_list = maps.HadamardProductFeatureMap(
            [hadamard_map1, hadamard_map2]
        )
        with self.assertRaisesRegex(UnsupportedError, "must have the same data type"):
            _ = hadamard_dtype_list.dtype

        # Force device mismatch for OuterProductFeatureMap
        outer_map1 = DeviceFeatureMap(torch.device("cpu"))
        outer_map2 = DeviceFeatureMap(torch.device("cpu"))
        fake_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            outer_map2.device = "fake_device"
        else:
            outer_map2.device = fake_device

        if torch.cuda.is_available() or outer_map2.device != outer_map1.device:
            outer_list = maps.OuterProductFeatureMap([outer_map1, outer_map2])
            with self.assertRaisesRegex(UnsupportedError, "must be colocated"):
                _ = outer_list.device

        outer_map2.device = torch.device("cpu")
        outer_map2.dtype = torch.float64
        outer_dtype_list = maps.OuterProductFeatureMap([outer_map1, outer_map2])
        with self.assertRaisesRegex(UnsupportedError, "must have the same data type"):
            _ = outer_dtype_list.dtype

    def test_feature_map_output_shape_none_transform(self):
        """Test FeatureMap output_shape when output_transform is None"""

        # Use a concrete subclass that can actually be instantiated
        class ConcreteFeatureMap(maps.FeatureMap):
            def __init__(self):
                super().__init__()
                self.raw_output_shape = Size([5])
                self.output_transform = None  # Explicitly set to None
                self.device = None
                self.dtype = None

            def forward(self, x, **kwargs):
                return torch.randn(x.shape[0], 5)

        feature_map = ConcreteFeatureMap()

        # return self.raw_output_shape
        output_shape = feature_map.output_shape
        self.assertEqual(output_shape, Size([5]))

    def test_fourier_feature_map_no_bias(self):
        """Test FourierFeatureMap with no bias"""
        config = TestCaseConfig(seed=0, device=self.device, num_inputs=2)
        kernel = gen_module(kernels.RBFKernel, config)
        weight = torch.randn(
            4, config.num_inputs, device=self.device, dtype=config.dtype
        )

        # Create FourierFeatureMap without bias (bias=None)
        fourier_map = maps.FourierFeatureMap(kernel=kernel, weight=weight, bias=None)

        X = torch.rand(5, config.num_inputs, device=self.device, dtype=config.dtype)
        output = fourier_map(X)

        # When bias is None, should just return out
        expected = X @ weight.transpose(-2, -1)
        self.assertTrue(output.allclose(expected))

    def test_direct_sum_feature_map_force_else_branch(self):
        """Test to force execution of else branch in DirectSumFeatureMap"""

        # Create custom feature maps that will definitely trigger the else branch
        class TestFeatureMap(maps.FeatureMap):
            def __init__(self, shape):
                super().__init__()
                self.raw_output_shape = Size(shape)
                self.batch_shape = Size([])
                self.input_transform = None
                self.output_transform = None
                self.device = torch.device("cpu")
                self.dtype = torch.float32

            def forward(self, x):
                return torch.randn(*([x.shape[0]] + list(self.raw_output_shape)))

        # Force the exact condition: ndim == max_ndim for all maps
        # Use 2D maps so max_ndim = 2, and both maps have ndim = 2
        map1 = TestFeatureMap([3, 4])  # 2D: [3, 4]
        map2 = TestFeatureMap([5, 6])  # 2D: [5, 6]
        map3 = TestFeatureMap([2, 7])  # 2D: [2, 7]

        # All maps have same ndim (2), so all will go to else branch
        feature_map = maps.DirectSumFeatureMap([map1, map2, map3])

        # Access raw_output_shape to trigger the computation
        shape = feature_map.raw_output_shape

        # result_shape[-1] += shape[-1] for each map: 0 + 4 + 6 + 7 = 17
        # result_shape[0] = max(3, 5, 2) = 5
        expected_shape = Size([5, 17])  # [max_first_dim, sum_last_dim]
        self.assertEqual(shape, expected_shape)
