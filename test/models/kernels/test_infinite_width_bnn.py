#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestInfiniteWidthBNNKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return InfiniteWidthBNNKernel(**kwargs)

    def test_properties(self):
        with self.subTest():
            kernel = InfiniteWidthBNNKernel(3)
            bias_var_init = torch.tensor(0.2)
            kernel.initialize(bias_var=bias_var_init)
            actual_value = bias_var_init.view_as(kernel.bias_var)
            self.assertLess(torch.linalg.norm(kernel.bias_var - actual_value), 1e-5)
        with self.subTest():
            kernel = InfiniteWidthBNNKernel(3)
            weight_var_init = torch.tensor(0.2)
            kernel.initialize(weight_var=weight_var_init)
            actual_value = weight_var_init.view_as(kernel.weight_var)
            self.assertLess(torch.linalg.norm(kernel.weight_var - actual_value), 1e-5)
        with self.subTest():
            kernel = InfiniteWidthBNNKernel(5, batch_shape=torch.Size([2]))
            bias_var_init = torch.tensor([0.2, 0.01])
            kernel.initialize(bias_var=bias_var_init)
            actual_value = bias_var_init.view_as(kernel.bias_var)
            self.assertLess(torch.linalg.norm(kernel.bias_var - actual_value), 1e-5)
        with self.subTest():
            kernel = InfiniteWidthBNNKernel(3, batch_shape=torch.Size([2]))
            weight_var_init = torch.tensor([1.0, 2.0])
            kernel.initialize(weight_var=weight_var_init)
            actual_value = weight_var_init.view_as(kernel.weight_var)
            self.assertLess(torch.linalg.norm(kernel.weight_var - actual_value), 1e-5)
        with self.subTest():
            kernel = InfiniteWidthBNNKernel(3, batch_shape=torch.Size([2]))
            x = torch.randn(3, 2)
            with self.assertRaises(RuntimeError):
                kernel(x, x, last_dim_is_batch=True).to_dense()

    def test_forward_0(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x1 = torch.tensor([[0.1, 0.2], [1.2, 0.4], [2.4, 0.3]]).to(**tkwargs)
            x2 = torch.tensor([[4.1, 2.3], [3.9, 0.0]]).to(**tkwargs)
            weight_var = 1.0
            bias_var = 0.1
            kernel = InfiniteWidthBNNKernel(0, device=self.device).initialize(
                weight_var=weight_var, bias_var=bias_var
            )
            kernel.eval()
            expected = (
                weight_var * (x1.matmul(x2.transpose(-2, -1)) / x1.shape[-1]) + bias_var
            ).to(**tkwargs)
            res = kernel(x1, x2).to_dense()
            self.assertAllClose(res, expected)

    def test_forward_0_batch(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x1 = torch.tensor(
                [
                    [
                        [0.4960, 0.7680, 0.0880],
                        [0.1320, 0.3070, 0.6340],
                        [0.4900, 0.8960, 0.4550],
                        [0.6320, 0.3480, 0.4010],
                        [0.0220, 0.1680, 0.2930],
                    ],
                    [
                        [0.5180, 0.6970, 0.8000],
                        [0.1610, 0.2820, 0.6810],
                        [0.9150, 0.3970, 0.8740],
                        [0.4190, 0.5520, 0.9520],
                        [0.0360, 0.1850, 0.3730],
                    ],
                ]
            ).to(**tkwargs)
            x2 = torch.tensor(
                [
                    [[0.3050, 0.9320, 0.1750], [0.2690, 0.1500, 0.0310]],
                    [[0.2080, 0.9290, 0.7230], [0.7420, 0.5260, 0.2430]],
                ]
            ).to(**tkwargs)
            weight_var = torch.tensor([1.0, 2.0]).to(**tkwargs)
            bias_var = torch.tensor([0.1, 0.5]).to(**tkwargs)
            kernel = InfiniteWidthBNNKernel(
                0, batch_shape=[2], device=self.device
            ).initialize(weight_var=weight_var, bias_var=bias_var)
            kernel.eval()
            expected = torch.tensor(
                [
                    [
                        [0.3942, 0.1838],
                        [0.2458, 0.1337],
                        [0.4547, 0.1934],
                        [0.2958, 0.1782],
                        [0.1715, 0.1134],
                    ],
                    [
                        [1.3891, 1.1303],
                        [1.0252, 0.7889],
                        [1.2940, 1.2334],
                        [1.3588, 1.0551],
                        [0.7994, 0.6431],
                    ],
                ]
            ).to(**tkwargs)
            res = kernel(x1, x2).to_dense()
            self.assertAllClose(res, expected, 0.0001, 0.0001)

    def test_forward_2(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x1 = torch.tensor(
                [
                    [
                        [0.4960, 0.7680, 0.0880],
                        [0.1320, 0.3070, 0.6340],
                        [0.4900, 0.8960, 0.4550],
                        [0.6320, 0.3480, 0.4010],
                        [0.0220, 0.1680, 0.2930],
                    ],
                    [
                        [0.5180, 0.6970, 0.8000],
                        [0.1610, 0.2820, 0.6810],
                        [0.9150, 0.3970, 0.8740],
                        [0.4190, 0.5520, 0.9520],
                        [0.0360, 0.1850, 0.3730],
                    ],
                ]
            ).to(**tkwargs)
            x2 = torch.tensor(
                [
                    [[0.3050, 0.9320, 0.1750], [0.2690, 0.1500, 0.0310]],
                    [[0.2080, 0.9290, 0.7230], [0.7420, 0.5260, 0.2430]],
                ]
            ).to(**tkwargs)
            weight_var = 1.0
            bias_var = 0.1
            kernel = InfiniteWidthBNNKernel(2, device=self.device).initialize(
                weight_var=weight_var, bias_var=bias_var
            )
            kernel.eval()
            expected = torch.tensor(
                [
                    [
                        [0.2488, 0.1985],
                        [0.2178, 0.1872],
                        [0.2641, 0.2036],
                        [0.2286, 0.1962],
                        [0.1983, 0.1793],
                    ],
                    [
                        [0.2869, 0.2564],
                        [0.2429, 0.2172],
                        [0.2820, 0.2691],
                        [0.2837, 0.2498],
                        [0.2160, 0.1986],
                    ],
                ]
            ).to(**tkwargs)
            res = kernel(x1, x2).to_dense()
            self.assertAllClose(res, expected, 0.0001, 0.0001)

    def test_forward_diag(self):
        # The diagonal must be computed for the pairs of rows of x1 and x2.
        for dtype, depth in ((torch.float, 0), (torch.double, 1), (torch.double, 3)):
            tkwargs = {"device": self.device, "dtype": dtype}
            kernel = InfiniteWidthBNNKernel(depth, device=self.device).to(**tkwargs)
            kernel.eval()
            x1 = torch.rand(2, 4, 3, **tkwargs)
            x2 = torch.rand(2, 4, 3, **tkwargs)
            with torch.no_grad():
                res = kernel(x1, x2, diag=True)
                expected = kernel(x1, x2).to_dense().diagonal(dim1=-2, dim2=-1)
                self.assertAllClose(res, expected)
                # identical inputs
                res = kernel(x1, x1, diag=True)
                expected = kernel(x1, x1).to_dense().diagonal(dim1=-2, dim2=-1)
                self.assertAllClose(res, expected, atol=1e-6, rtol=1e-6)
