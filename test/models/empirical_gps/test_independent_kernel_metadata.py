#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import ScaleKernel


class TestIndependentEmpiricalKernelMetadata(BotorchTestCase):
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
                base = EmpiricalOneDimensionalKernel(grid, history, use_svd=svd)
                kernel = ScaleKernel(base).to(grid) if scaled else base
                X = grid.new_tensor([[0.12], [0.47], [0.83], [0.94]]).requires_grad_()
                Y = grid.new_tensor([[0.26], [0.74]])
                for batched in (False, True):
                    points = X
                    if batched:
                        points = (
                            X.expand(2, outputs, *X.shape)
                            if outputs > 1
                            else X.expand(2, *X.shape)
                        )
                    expected = kernel.forward(points, Y)
                    if hasattr(expected, "to_dense"):
                        expected = expected.to_dense()
                    lazy = kernel(points, Y)
                    batch = ((2,) if batched else ()) + (
                        (outputs,) if outputs > 1 else ()
                    )
                    self.assertEqual(lazy.shape, torch.Size((*batch, 4, 2)))
                    self.assertAllClose(lazy.to_dense(), expected)
                    if batched:
                        # The replicated progression queries must reproduce
                        # each independent unbatched historical-output block.
                        self.assertAllClose(
                            lazy.to_dense(),
                            kernel(X, Y).to_dense().expand(*batch, 4, 2),
                        )
                    rhs = grid.new_ones(2, 1)
                    self.assertAllClose(lazy @ rhs, expected @ rhs)
                    self.assertAllClose(
                        kernel(points).to_dense().diagonal(dim1=-2, dim2=-1),
                        kernel(points, diag=True),
                    )
                    self.assertTrue(
                        torch.isfinite(
                            torch.autograd.grad(lazy.to_dense().sum(), X)[0]
                        ).all()
                    )
