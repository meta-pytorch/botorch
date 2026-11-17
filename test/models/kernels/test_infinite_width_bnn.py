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
    def test_coincident_values_and_full_derivatives(self):
        devices = [self.device]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for depth in (1, 3, 7):
                for eps in (0.0, 1e-7, 1e-2):
                    with self.subTest(device=device, depth=depth, eps=eps):
                        kernel = InfiniteWidthBNNKernel(
                            depth,
                            batch_shape=torch.Size([2]),
                            active_dims=(2, 0),
                            acos_eps=eps,
                        ).to(dtype=torch.double, device=device)
                        x = torch.tensor(
                            [[0.2, 2.0, 0.7], [0.4, 3.0, 0.5], [0.2, 2.0, 0.7]],
                            dtype=torch.double,
                            device=device,
                            requires_grad=True,
                        )
                        full = kernel(x).to_dense()
                        diagonal = kernel(x, diag=True)
                        self.assertAllClose(full.diagonal(dim1=-2, dim2=-1), diagonal)
                        # Duplicate off-diagonal rows have the same exact law.
                        self.assertAllClose(full[..., 0, 2], diagonal[..., 0])
                        inputs = (x, *kernel.parameters())
                        actual = torch.autograd.grad(
                            full.diagonal(dim1=-2, dim2=-1).sum(),
                            inputs,
                            create_graph=True,
                        )
                        expected = torch.autograd.grad(
                            diagonal.sum(), inputs, create_graph=True
                        )
                        for a, b in zip(actual, expected):
                            self.assertAllClose(a, b)
                        actual_second = torch.autograd.grad(
                            sum(g.square().sum() for g in actual), inputs
                        )
                        expected_second = torch.autograd.grad(
                            sum(g.square().sum() for g in expected), inputs
                        )
                        for a, b in zip(actual_second, expected_second):
                            self.assertTrue(torch.isfinite(a).all())
                            self.assertAllClose(a, b)

    def test_coincident_distinct_arguments_and_large_variances(self):
        devices = [self.device]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for depth in (1, 3):
                with self.subTest(device=device, depth=depth):
                    kernel = (
                        InfiniteWidthBNNKernel(depth, acos_eps=0).double().to(device)
                    )
                    left = torch.tensor(
                        [[0.3, 0.7]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    right = left.detach().clone().requires_grad_()
                    actual = kernel(left, right).to_dense().sum()
                    expected = (
                        kernel.weight_var * (left * right).sum() / 2 + kernel.bias_var
                    )
                    for _ in range(depth):
                        expected = kernel.weight_var * expected / 2 + kernel.bias_var
                    inputs = (left, right, *kernel.parameters())
                    a = torch.autograd.grad(actual, inputs, create_graph=True)
                    b = torch.autograd.grad(expected.sum(), inputs, create_graph=True)
                    for av, bv in zip(a, b):
                        self.assertAllClose(av, bv)
                    # Mixed partials must retain both independently supplied inputs.
                    self.assertAllClose(
                        torch.autograd.grad(a[0].sum(), right)[0],
                        torch.autograd.grad(b[0].sum(), right)[0],
                    )
                with self.subTest(device=device, depth=depth, large=True):
                    kernel = InfiniteWidthBNNKernel(depth).double().to(device)
                    kernel.initialize(weight_var=1.0, bias_var=1.0)
                    x = torch.tensor(
                        [[1e100]], dtype=torch.double, device=device, requires_grad=True
                    )
                    full = kernel(x).to_dense().sum()
                    diagonal = kernel(x, diag=True).sum()
                    self.assertTrue(torch.isfinite(full))
                    self.assertAllClose(full, diagonal)
                    self.assertAllClose(
                        torch.autograd.grad(full, x)[0],
                        torch.autograd.grad(diagonal, x)[0],
                    )

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
