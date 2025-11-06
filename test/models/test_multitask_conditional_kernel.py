#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.heterogeneous_multitask import (
    DeltaKernel,
    find_subsets,
    map_subsets,
    MultiTaskConditionalKernel,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel


class TestMultiTaskConditionalKernel(BotorchTestCase):
    def test_find_subsets(self) -> None:
        # Test a bunch of cases against known answers.
        self.assertEqual(
            find_subsets([[0, 3, 1, 2], [0, 1, 2], [0, 2, 3, 4]]),
            [{0, 2}, {1}, {3}, {4}],
        )
        self.assertEqual(find_subsets([[1, 2]]), [{1, 2}])
        self.assertEqual(find_subsets([[1, 2], [1, 5]]), [{1}, {2}, {5}])
        self.assertEqual(find_subsets([[1, 2, 3, 4], [1, 2, 3]]), [{1, 2, 3}, {4}])
        self.assertEqual(
            find_subsets([[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 5, 6]]),
            [{1, 2, 3}, {4}, {5, 6}],
        )
        self.assertEqual(
            find_subsets([[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 5, 6], [5], [7], [1, 3]]),
            [{1, 3}, {2}, {4}, {5}, {6}, {7}],
        )

    def test_map_subsets(self) -> None:
        feature_index_map, binary_map = map_subsets(
            subsets=[{0, 2}, {1}, {3}, {4}],
            feature_indices=[[0, 3, 1, 2], [0, 1, 2], [0, 2, 3, 4]],
        )
        self.assertEqual(
            feature_index_map,
            {(0, 2): [0, 1, 2], (1,): [0, 1], (3,): [0, 2], (4,): [2]},
        )
        self.assertEqual(binary_map, [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1]])
        feature_index_map, binary_map = map_subsets(
            subsets=[{1, 2, 3}, {4}, {5, 6}],
            feature_indices=[[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 5, 6]],
        )
        self.assertEqual(
            feature_index_map, {(1, 2, 3): [0, 1, 2], (4,): [0], (5, 6): [2]}
        )
        self.assertEqual(binary_map, [[1, 1, 0], [1, 0, 0], [1, 0, 1]])

    def test_delta_kernel(self) -> None:
        kernel = DeltaKernel()
        x1 = torch.tensor([[[0], [1], [1]], [[1], [1], [0]]])
        x2 = torch.tensor([[1], [0]])
        k_k_eval = kernel(x1, x2).to_dense()
        self.assertEqual(k_k_eval.shape, torch.Size([2, 3, 2]))
        expected = torch.tensor([[[0, 0], [1, 0], [1, 0]], [[1, 0], [1, 0], [0, 0]]])
        self.assertTrue(torch.equal(k_k_eval, expected))

    def test_MultiTaskConditionalKernel(self) -> None:
        feature_indices = [[0, 1, 2], [0, 1], [0, 1, 3]]
        # expected_subsets = [{0, 1}, {2}, {3}]
        expected_active_index_map = {(0, 1): [0, 1, 2], (2,): [0], (3,): [2]}
        expected_binary_map = [[1, 1, 0], [1, 0, 0], [1, 0, 1]]

        # Test initialization.
        kernel = MultiTaskConditionalKernel(feature_indices=feature_indices)
        self.assertEqual(kernel.task_feature_index, -1)
        self.assertTrue(kernel.use_saas_prior)
        self.assertEqual(kernel.active_index_map, expected_active_index_map)
        self.assertEqual(kernel.binary_map, expected_binary_map)
        self.assertEqual(len(kernel.kernels), 3)
        self.assertEqual(kernel.combinatorial_kernel.base_kernel.ard_num_dims, 3)
        for k, active_indices in zip(kernel.kernels, expected_active_index_map):
            self.assertIsInstance(k, ScaleKernel)
            base_k = k.base_kernel
            self.assertIsInstance(base_k, MaternKernel)
            self.assertEqual(base_k.ard_num_dims, len(active_indices))
            self.assertEqual(base_k.active_dims.tolist(), list(active_indices))

        # Test map_task_to_binary.
        x_task = torch.tensor([[0, 1, 2], [0, 0, 2]])
        x_binary = kernel.map_task_to_binary(x_task)
        expected_binary = torch.tensor(
            [[[1, 1, 0], [1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 0], [1, 0, 1]]]
        )
        self.assertTrue(torch.equal(x_binary, expected_binary))

        # Test forward pass. Using single X since this is how it will be used.
        # X is 1 x 3 x 5 with the last column representing the task feature.
        X = torch.tensor(
            [
                [
                    [0.2, 0.3, 0.5, 0.0, 0],
                    [0.5, 0.2, 0.0, 0.0, 1],
                    [0.7, 0.5, 0.0, 0.3, 2],
                ]
            ]
        )
        k_eval = kernel(X).to_dense()
        self.assertEqual(k_eval.shape, torch.Size([1, 3, 3]))
        self.assertAllClose(k_eval, k_eval.transpose(-2, -1))
        # We will check some values against manual computation.
        combinatorial_k_eval = kernel.combinatorial_kernel(
            torch.tensor([expected_binary_map], dtype=torch.float)
        ).to_dense()
        kernel_k_evals = [k(X).to_dense() for k in kernel.kernels]
        self.assertEqual(combinatorial_k_eval.shape, torch.Size([1, 3, 3]))
        idx = (0, 0, 0)
        self.assertAllClose(
            k_eval[idx],
            combinatorial_k_eval[idx] + kernel_k_evals[0][idx] + kernel_k_evals[1][idx],
        )
        for idx in ((0, 1, 1), (0, 0, 1), (0, 0, 2)):
            self.assertAllClose(
                k_eval[idx], combinatorial_k_eval[idx] + kernel_k_evals[0][idx]
            )
        idx = (0, 2, 2)
        self.assertAllClose(
            k_eval[idx],
            combinatorial_k_eval[idx] + kernel_k_evals[0][idx] + kernel_k_evals[2][idx],
        )

        # Checking for correct ard_num_dims on the combinatorial kernel.
        feature_indices = [[0, 1, 2], [0, 1, 2], [0, 1]]
        kernel = MultiTaskConditionalKernel(feature_indices=feature_indices)
        self.assertEqual(kernel.combinatorial_kernel.base_kernel.ard_num_dims, 2)
        self.assertEqual(len(kernel.kernels), 2)
        self.assertEqual(len(kernel.binary_map), 3)
        k_eval = kernel(torch.rand(1, 4, 3)).to_dense()
        self.assertEqual(k_eval.shape, torch.Size([1, 4, 4]))

        # Test with use_combinatorial_kernel=False.
        kernel = MultiTaskConditionalKernel(
            feature_indices=feature_indices, use_combinatorial_kernel=False
        )
        self.assertFalse(kernel.use_combinatorial_kernel)
        k_eval = kernel(X).to_dense()
        self.assertEqual(k_eval.shape, torch.Size([1, 3, 3]))
        self.assertAllClose(k_eval, k_eval.transpose(-2, -1))
