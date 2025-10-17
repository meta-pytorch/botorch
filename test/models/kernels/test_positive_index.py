#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.positive_index import PositiveIndexKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.priors import NormalPrior


class TestPositiveIndexKernel(BotorchTestCase):
    def test_positive_index_kernel(self):
        # Test initialization
        with self.subTest("basic_initialization"):
            num_tasks = 4
            rank = 2
            kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=rank)

            self.assertEqual(kernel.num_tasks, num_tasks)
            self.assertEqual(kernel.raw_covar_factor.shape, (num_tasks, rank))
            self.assertEqual(kernel.normalize_covar_matrix, False)

        # Test initialization with batch shape
        with self.subTest("initialization_with_batch_shape"):
            num_tasks = 3
            rank = 2
            batch_shape = torch.Size([2])
            kernel = PositiveIndexKernel(
                num_tasks=num_tasks, rank=rank, batch_shape=batch_shape
            )

            self.assertEqual(kernel.raw_covar_factor.shape, (2, num_tasks, rank))

        # Test rank validation
        with self.subTest("rank_validation"):
            num_tasks = 3
            rank = 5
            with self.assertRaises(RuntimeError):
                PositiveIndexKernel(num_tasks=num_tasks, rank=rank)

        # Test target_task_index validation
        with self.subTest("target_task_index_validation"):
            num_tasks = 4
            # Test invalid negative index
            with self.assertRaises(ValueError):
                PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=-1)
            # Test invalid index >= num_tasks
            with self.assertRaises(ValueError):
                PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=4)
            # Test valid indices (should not raise)
            PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=0)
            PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=3)

        # Test covar_factor constraint
        with self.subTest("positive_correlations"):
            kernel = PositiveIndexKernel(num_tasks=5, rank=3)
            covar_factor = kernel.covar_factor

            # All elements should be positive
            self.assertTrue((covar_factor > 0).all())

            self.assertTrue((kernel.covar_matrix >= 0).all())

        # Test covariance matrix normalization (default target_task_index=0)
        with self.subTest("covar_matrix_normalization_default"):
            kernel = PositiveIndexKernel(num_tasks=4, rank=2)
            covar = kernel.covar_matrix

            # First diagonal element should be 1.0 (normalized by default)
            self.assertAllClose(covar[0, 0], torch.tensor(1.0), atol=1e-4)

        # Test covariance matrix normalization with custom target_task_index
        with self.subTest("covar_matrix_normalization_custom_target"):
            kernel = PositiveIndexKernel(num_tasks=4, rank=2, target_task_index=2)
            covar = kernel.covar_matrix

            # Third diagonal element should be 1.0 (target_task_index=2)
            self.assertAllClose(covar[2, 2], torch.tensor(1.0), atol=1e-4)

            # Other diagonal elements should not be 1.0
            self.assertNotEqual(covar[0, 0].item(), 1.0)

        # Test forward pass shape
        with self.subTest("forward"):
            num_tasks = 4
            kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=2)
            kernel.eval()

            i1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
            i2 = torch.tensor([[1, 2]], dtype=torch.long)

            result = kernel(i1, i2)
            self.assertEqual(result.shape, torch.Size([2, 1]))
            num_tasks = 3
            kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=1)
            kernel.eval()

            kernel.initialize(raw_covar_factor=torch.ones(num_tasks, 1))
            i1 = torch.tensor([[0]], dtype=torch.long)
            i2 = torch.tensor([[1]], dtype=torch.long)

            result = kernel(i1, i2).to_dense()
            covar_matrix = kernel.covar_matrix
            expected = covar_matrix[0, 1]

            self.assertAllClose(result.squeeze(), expected)

        # Test with priors
        with self.subTest("with_priors"):
            num_tasks = 4
            task_prior = NormalPrior(0, 1)
            diag_prior = NormalPrior(1, 0.1)

            kernel = PositiveIndexKernel(
                num_tasks=num_tasks,
                rank=2,
                task_prior=task_prior,
                diag_prior=diag_prior,
                initialize_to_mode=False,
            )
            prior_names = [p[0] for p in kernel.named_priors()]
            self.assertIn("IndexKernelPrior", prior_names)
            self.assertIn("ScalePrior", prior_names)

        # Test batch forward
        with self.subTest("batch_forward"):
            num_tasks = 3
            batch_shape = torch.Size([2])
            kernel = PositiveIndexKernel(
                num_tasks=num_tasks, rank=2, batch_shape=batch_shape
            )
            kernel.eval()

            i1 = torch.tensor([[[0], [1]]], dtype=torch.long)
            i2 = torch.tensor([[[1], [2]]], dtype=torch.long)

            result = kernel(i1, i2)

            # Check that batch dimensions are preserved
            self.assertEqual(result.shape[0], 2)

        # Test diagonal property (default target_task_index=0)
        with self.subTest("diagonal"):
            kernel = PositiveIndexKernel(num_tasks=4, rank=2)
            diag = kernel._diagonal

            self.assertEqual(diag.shape, torch.Size([4]))
            # First diagonal element should be 1.0 (default target_task_index=0)
            self.assertAllClose(diag[0], torch.tensor(1.0), atol=1e-4)

            # Test diagonal property with custom target_task_index
            kernel = PositiveIndexKernel(num_tasks=4, rank=2, target_task_index=1)
            diag = kernel._diagonal

            self.assertEqual(diag.shape, torch.Size([4]))
            # Second diagonal element should be 1.0 (target_task_index=1)
            self.assertAllClose(diag[1], torch.tensor(1.0), atol=1e-4)

        # Test lower triangle property
        with self.subTest("lower_triangle"):
            num_tasks = 5
            kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=2)
            lower_tri = kernel._lower_triangle

            # Number of lower triangular elements (excluding diagonal)
            expected_size = num_tasks * (num_tasks - 1) // 2
            self.assertEqual(lower_tri.shape[-1], expected_size)
            self.assertTrue((lower_tri >= 0).all())

        # Test invalid prior type
        with self.subTest("invalid_prior_type"):
            with self.assertRaises(TypeError):
                PositiveIndexKernel(num_tasks=4, rank=2, task_prior="not_a_prior")

        # Test covariance matrix properties
        with self.subTest("covar_matrix"):
            kernel = PositiveIndexKernel(num_tasks=5, rank=4)
            covar = kernel.covar_matrix

            # Should be square
            self.assertEqual(covar.shape[-2], covar.shape[-1])

            # Should be positive definite (all eigenvalues > 0)
            eigvals = torch.linalg.eigvalsh(covar)
            self.assertTrue((eigvals > 0).all())

            # Should be symmetric
            self.assertAllClose(covar, covar.T, atol=1e-5)

        # Test covar_factor setter and getter
        with self.subTest("covar_factor"):
            kernel = PositiveIndexKernel(num_tasks=3, rank=2)
            new_covar_factor = torch.ones(3, 2) * 2.0
            kernel.covar_factor = new_covar_factor
            self.assertAllClose(kernel.covar_factor, new_covar_factor, atol=1e-5)

            kernel = PositiveIndexKernel(num_tasks=3, rank=2)
            params = kernel._covar_factor_params(kernel)
            self.assertEqual(params.shape, torch.Size([3, 2]))
            self.assertTrue((params > 0).all())

            kernel = PositiveIndexKernel(num_tasks=3, rank=2)
            new_value = torch.ones(3, 2) * 3.0
            kernel._covar_factor_closure(kernel, new_value)
            self.assertAllClose(kernel.covar_factor, new_value, atol=1e-5)
