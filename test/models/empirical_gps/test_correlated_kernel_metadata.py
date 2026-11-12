#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalKernel,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import ScaleKernel


class TestCorrelatedEmpiricalKernelMetadata(BotorchTestCase):
    def test_public_lazy_kernel_operations(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, outputs, svd, scaled in itertools.product(
            devices, (1, 2, 3), (False, True), (False, True)
        ):
            with self.subTest(device=device, outputs=outputs, svd=svd, scaled=scaled):
                grid = torch.linspace(
                    0, 1, 5, dtype=torch.double, device=device
                ).unsqueeze(-1)
                history = torch.rand(18, 5, outputs, dtype=torch.double, device=device)
                base = MultiOutputEmpiricalOneDimensionalKernel(
                    grid, history, use_svd=svd
                )
                kernel = ScaleKernel(base).to(grid) if scaled else base
                X = torch.tensor(
                    [[[0.12], [0.47], [0.83]], [[0.18], [0.57], [0.92]]],
                    dtype=torch.double,
                    device=device,
                    requires_grad=True,
                )
                Y = torch.tensor([[0.26], [0.74]], dtype=torch.double, device=device)
                lazy = kernel(X, Y)
                expected = kernel.forward(X, Y)
                if hasattr(expected, "to_dense"):
                    expected = expected.to_dense()
                self.assertEqual(lazy.shape, torch.Size([2, 3 * outputs, 2 * outputs]))
                self.assertAllClose(lazy.to_dense(), expected)
                rhs = torch.rand(2 * outputs, 2, dtype=torch.double, device=device)
                self.assertAllClose(lazy @ rhs, expected @ rhs)
                self.assertAllClose(
                    lazy[..., :outputs, :outputs].to_dense(),
                    expected[..., :outputs, :outputs],
                )
                symmetric = kernel(X, X)
                self.assertAllClose(
                    symmetric.diagonal(dim1=-2, dim2=-1), kernel(X, X, diag=True)
                )
                derivative = torch.autograd.grad(lazy.to_dense().sum(), X)[0]
                self.assertTrue(torch.isfinite(derivative).all())
