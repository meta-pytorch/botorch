#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase
from torch.utils._python_dispatch import TorchDispatchMode


class LargestTensorRecorder(TorchDispatchMode):
    """Records the element count of the largest tensor allocated in the block."""

    max_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.max_numel = max(self.max_numel, out.numel())
        return out


class TestCategoricalKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return CategoricalKernel(**kwargs)

    def create_data_no_batch(self):
        return torch.randint(3, size=(5, 10)).to(dtype=torch.float)

    def create_data_single_batch(self):
        return torch.randint(3, size=(2, 5, 3)).to(dtype=torch.float)

    def create_data_double_batch(self):
        return torch.randint(3, size=(3, 2, 5, 3)).to(dtype=torch.float)

    def test_initialize_lengthscale(self):
        kernel = CategoricalKernel()
        kernel.initialize(lengthscale=1)
        actual_value = torch.tensor(1.0).view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = CategoricalKernel(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([1.0, 2.0])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_forward(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5], [7, 6]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscale = 2
        kernel = CategoricalKernel().initialize(lengthscale=lengthscale)
        kernel.eval()
        sc_dists = (x1.unsqueeze(-2) != x2.unsqueeze(-3)) / lengthscale
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_active_dims(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5], [7, 6]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscale = 2
        kernel = CategoricalKernel(active_dims=[0]).initialize(lengthscale=lengthscale)
        kernel.eval()
        dists = x1[:, :1].unsqueeze(-2) != x2[:, :1].unsqueeze(-3)
        sc_dists = dists / lengthscale
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_ard(self):
        x1 = torch.tensor([[4, 2], [3, 1], [8, 5]], dtype=torch.float)
        x2 = torch.tensor([[4, 2], [3, 0], [4, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = CategoricalKernel(ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

        # diag
        res = kernel(x1, x2).diagonal()
        actual = torch.diagonal(actual, dim1=-1, dim2=-2)
        self.assertAllClose(res, actual)

        # batch_dims
        actual = torch.exp(-sc_dists).transpose(-1, -3)
        res = kernel(x1, x2, last_dim_is_batch=True).to_dense()
        self.assertAllClose(res, actual)

        # batch_dims + diag
        res = kernel(x1, x2, last_dim_is_batch=True).diagonal()
        self.assertAllClose(res, torch.diagonal(actual, dim1=-1, dim2=-2))

    def test_diag_matches_dense_diagonal(self):
        cases = {
            "batched": {"batch_shape": torch.Size([2])},
            "ard": {"ard_num_dims": 3},
            "batched ard": {"batch_shape": torch.Size([2]), "ard_num_dims": 3},
        }
        for name, kwargs in cases.items():
            for n1, n2 in [(4, 4), (5, 3), (1, 6)]:
                for last_dim_is_batch in [False, True]:
                    with self.subTest(name, n1=n1, n2=n2, ldib=last_dim_is_batch):
                        kernel = CategoricalKernel(**kwargs).to(dtype=torch.double)
                        # randomized so a misaligned lengthscale axis cannot cancel out
                        with torch.no_grad():
                            kernel.raw_lengthscale.copy_(
                                torch.rand_like(kernel.raw_lengthscale)
                            )
                        batch = kwargs.get("batch_shape", torch.Size([]))
                        x1 = torch.randint(3, size=(*batch, n1, 3)).to(torch.double)
                        x2 = torch.randint(3, size=(*batch, n2, 3)).to(torch.double)

                        dense = kernel.forward(
                            x1, x2, last_dim_is_batch=last_dim_is_batch
                        )
                        expected = torch.diagonal(dense, dim1=-1, dim2=-2)
                        res = kernel.forward(
                            x1, x2, diag=True, last_dim_is_batch=last_dim_is_batch
                        )
                        self.assertEqual(res.shape, expected.shape)
                        self.assertAllClose(res, expected)

    def test_diag_does_not_materialize_pairwise_matrix(self):
        # the diagonal should stay linear in n rather than allocating `n x n x d`
        n, d = 512, 3
        kernel = CategoricalKernel(ard_num_dims=d).to(dtype=torch.double)
        x1 = torch.randint(3, size=(n, d)).to(dtype=torch.double)
        x2 = torch.randint(3, size=(n, d)).to(dtype=torch.double)

        recorder = LargestTensorRecorder()
        with recorder:
            kernel.forward(x1, x2, diag=True)

        budget = 4 * n * d
        self.assertLess(budget, n * n)
        self.assertLessEqual(recorder.max_numel, budget)

    def test_ard_batch(self):
        x1 = torch.tensor(
            [
                [[4, 2, 1], [3, 1, 5]],
                [[3, 2, 3], [6, 1, 7]],
            ],
            dtype=torch.float,
        )
        x2 = torch.tensor([[[4, 2, 1], [6, 0, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = CategoricalKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales.unsqueeze(-2)
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

    def test_ard_separate_batch(self):
        x1 = torch.tensor(
            [
                [[4, 2, 1], [3, 1, 5]],
                [[3, 2, 3], [6, 1, 7]],
            ],
            dtype=torch.float,
        )
        x2 = torch.tensor([[[4, 2, 1], [6, 0, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = CategoricalKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        sc_dists = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        sc_dists = sc_dists / lengthscales.unsqueeze(-2)
        actual = torch.exp(-sc_dists.mean(-1))
        res = kernel(x1, x2).to_dense()
        self.assertAllClose(res, actual)

        # diag
        res = kernel(x1, x2).diagonal()
        actual = torch.diagonal(actual, dim1=-1, dim2=-2)
        self.assertAllClose(res, actual)

        # batch_dims
        actual = torch.exp(-sc_dists).transpose(-1, -3)
        res = kernel(x1, x2, last_dim_is_batch=True).to_dense()
        self.assertAllClose(res, actual)

        # batch_dims + diag
        res = kernel(x1, x2, last_dim_is_batch=True).diagonal()
        self.assertAllClose(res, torch.diagonal(actual, dim1=-1, dim2=-2))
