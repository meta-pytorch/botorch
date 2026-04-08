#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.positive_index import PositiveIndexKernel
from botorch.models.utils.priors import BetaPrior
from botorch.optim.utils import sample_all_priors
from botorch.utils.testing import BotorchTestCase
from gpytorch.priors import NormalPrior, UniformPrior


class TestPositiveIndexKernel(BotorchTestCase):
    def test_positive_index_kernel(self):
        for dtype in (torch.float32, torch.float64):
            # Test initialization
            with self.subTest("basic_initialization", dtype=dtype):
                num_tasks = 4
                rank = 2
                kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=rank).to(
                    dtype=dtype
                )

                self.assertEqual(kernel.num_tasks, num_tasks)
                self.assertEqual(kernel.raw_covar_factor.shape, (num_tasks, rank))
                self.assertEqual(kernel.normalize_covar_matrix, False)

            # Test initialization with batch shape
            with self.subTest("initialization_with_batch_shape", dtype=dtype):
                num_tasks = 3
                rank = 2
                batch_shape = torch.Size([2])
                kernel = PositiveIndexKernel(
                    num_tasks=num_tasks, rank=rank, batch_shape=batch_shape
                ).to(dtype=dtype)

                self.assertEqual(kernel.raw_covar_factor.shape, (2, num_tasks, rank))

            # Test rank validation
            with self.subTest("rank_validation", dtype=dtype):
                num_tasks = 3
                rank = 5
                with self.assertRaises(RuntimeError):
                    PositiveIndexKernel(num_tasks=num_tasks, rank=rank)

            # Test target_task_index validation
            with self.subTest("target_task_index_validation", dtype=dtype):
                num_tasks = 4
                # Test invalid negative index
                with self.assertRaises(ValueError):
                    PositiveIndexKernel(
                        num_tasks=num_tasks, rank=2, target_task_index=-1
                    )
                # Test invalid index >= num_tasks
                with self.assertRaises(ValueError):
                    PositiveIndexKernel(
                        num_tasks=num_tasks, rank=2, target_task_index=4
                    )
                # Test valid indices (should not raise)
                PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=0)
                PositiveIndexKernel(num_tasks=num_tasks, rank=2, target_task_index=3)

            # Test covar_factor constraint
            with self.subTest("positive_correlations", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=5, rank=3).to(dtype=dtype)
                covar_factor = kernel.covar_factor

                # All elements should be positive
                self.assertTrue((covar_factor > 0).all())

                self.assertTrue((kernel.covar_matrix >= 0).all())

            # Test covariance matrix normalization (default target_task_index=0)
            with self.subTest("covar_matrix_normalization_default", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=4, rank=2).to(dtype=dtype)
                covar = kernel.covar_matrix

                # First diagonal element should be 1.0 (normalized by default)
                self.assertAllClose(
                    covar[0, 0], torch.tensor(1.0, dtype=dtype), atol=1e-4
                )

            # Test covariance matrix normalization with custom target_task_index
            with self.subTest("covar_matrix_normalization_custom_target", dtype=dtype):
                kernel = PositiveIndexKernel(
                    num_tasks=4, rank=2, target_task_index=2
                ).to(dtype=dtype)
                covar = kernel.covar_matrix

                # Third diagonal element should be 1.0 (target_task_index=2)
                self.assertAllClose(
                    covar[2, 2], torch.tensor(1.0, dtype=dtype), atol=1e-4
                )

            # Test forward pass shape
            with self.subTest("forward", dtype=dtype):
                num_tasks = 4
                kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=2).to(
                    dtype=dtype
                )

                i1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
                i2 = torch.tensor([[1, 2]], dtype=torch.long)

                result = kernel(i1, i2)
                self.assertEqual(result.shape, torch.Size([2, 1]))
                num_tasks = 3
                kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=1).to(
                    dtype=dtype
                )

                kernel.initialize(
                    raw_covar_factor=torch.ones(num_tasks, 1, dtype=dtype)
                )
                i1 = torch.tensor([[0]], dtype=torch.long)
                i2 = torch.tensor([[1]], dtype=torch.long)

                result = kernel(i1, i2).to_dense()
                covar_matrix = kernel.covar_matrix
                expected = covar_matrix[0, 1]

                self.assertAllClose(result.squeeze(), expected)

            # Test with priors
            with self.subTest("with_priors", dtype=dtype):
                num_tasks = 4
                task_prior = NormalPrior(0, 1)

                kernel = PositiveIndexKernel(
                    num_tasks=num_tasks,
                    rank=2,
                    task_prior=task_prior,
                    initialize_to_mode=False,
                ).to(dtype=dtype)
                prior_names = [p[0] for p in kernel.named_priors()]
                self.assertIn("IndexKernelPrior", prior_names)

            # Test batch forward
            with self.subTest("batch_forward", dtype=dtype):
                num_tasks = 3
                batch_shape = torch.Size([2])
                kernel = PositiveIndexKernel(
                    num_tasks=num_tasks, rank=2, batch_shape=batch_shape
                ).to(dtype=dtype)

                i1 = torch.tensor([[[0], [1]]], dtype=torch.long)
                i2 = torch.tensor([[[1], [2]]], dtype=torch.long)

                result = kernel(i1, i2)

                # Check that batch dimensions are preserved
                self.assertEqual(result.shape[0], 2)

            # Test lower triangle property
            with self.subTest("lower_triangle", dtype=dtype):
                num_tasks = 5
                kernel = PositiveIndexKernel(num_tasks=num_tasks, rank=2).to(
                    dtype=dtype
                )
                lower_tri = kernel._lower_triangle_corr

                # Number of lower triangular elements (excluding diagonal)
                expected_size = num_tasks * (num_tasks - 1) // 2
                self.assertEqual(lower_tri.shape[-1], expected_size)
                self.assertTrue((lower_tri >= 0).all())

            # Test invalid prior type
            with self.subTest("invalid_prior_type", dtype=dtype):
                with self.assertRaises(TypeError):
                    PositiveIndexKernel(num_tasks=4, rank=2, task_prior="not_a_prior")

            # Test covariance matrix properties
            with self.subTest("covar_matrix", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=5, rank=4).to(dtype=dtype)
                covar = kernel.covar_matrix

                # Should be square
                self.assertEqual(covar.shape[-2], covar.shape[-1])

                # Should be positive definite (all eigenvalues > 0)
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())

                # Should be symmetric
                self.assertAllClose(covar, covar.T, atol=1e-5)

            # Test covar_factor setter and getter
            with self.subTest("covar_factor", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=3, rank=2).to(dtype=dtype)
                new_covar_factor = torch.ones(3, 2, dtype=dtype) * 2.0
                kernel.covar_factor = new_covar_factor
                self.assertAllClose(kernel.covar_factor, new_covar_factor, atol=1e-5)

                kernel = PositiveIndexKernel(num_tasks=3, rank=2).to(dtype=dtype)
                params = kernel._covar_factor_params(kernel)
                self.assertEqual(params.shape, torch.Size([3, 2]))
                self.assertTrue((params > 0).all())

                kernel = PositiveIndexKernel(num_tasks=3, rank=2).to(dtype=dtype)
                new_value = torch.ones(3, 2, dtype=dtype) * 3.0
                kernel._covar_factor_closure(kernel, new_value)
                self.assertAllClose(kernel.covar_factor, new_value, atol=1e-5)

            # Test _set_lower_triangle_corr produces valid covariance
            with self.subTest("set_lower_triangle_corr", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=3, rank=3).to(dtype=dtype)
                target_corr = torch.tensor([0.8, 0.5, 0.6], dtype=dtype)
                kernel._set_lower_triangle_corr(target_corr)

                # Covariance matrix should be PD and symmetric
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
                self.assertAllClose(covar, covar.T, atol=1e-5)

                # Recovered correlations should be positive
                recovered = kernel._lower_triangle_corr
                self.assertTrue((recovered >= 0).all())
                self.assertTrue((recovered <= 1).all())

            # Test _set_lower_triangle_corr with batch shape
            with self.subTest("set_lower_triangle_corr_batch", dtype=dtype):
                batch_shape = torch.Size([2])
                kernel = PositiveIndexKernel(
                    num_tasks=3, rank=3, batch_shape=batch_shape
                ).to(dtype=dtype)
                target_corr = torch.rand(*batch_shape, 3, dtype=dtype)
                kernel._set_lower_triangle_corr(target_corr)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
                self.assertEqual(covar.shape, torch.Size([2, 3, 3]))

            # Test sample_all_priors with batch shape
            with self.subTest("sample_all_priors_batch", dtype=dtype):
                batch_shape = torch.Size([2])
                task_prior = UniformPrior(0.0, 1.0)
                kernel = PositiveIndexKernel(
                    num_tasks=3,
                    rank=3,
                    task_prior=task_prior,
                    batch_shape=batch_shape,
                ).to(dtype=dtype)
                sample_all_priors(kernel)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
                self.assertEqual(covar.shape, torch.Size([2, 3, 3]))

            # Test _set_lower_triangle_corr with scalar input (under-batched)
            with self.subTest("set_lower_triangle_corr_scalar", dtype=dtype):
                batch_shape = torch.Size([2])
                kernel = PositiveIndexKernel(
                    num_tasks=3, rank=3, batch_shape=batch_shape
                ).to(dtype=dtype)
                # Scalar value — exercises dim()==0 branch
                kernel._set_lower_triangle_corr(torch.tensor(0.5, dtype=dtype))
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
                self.assertEqual(covar.shape, torch.Size([2, 3, 3]))

            # Test _set_lower_triangle_corr with unbatched input on batched kernel
            with self.subTest(
                "set_lower_triangle_corr_unbatched_on_batch", dtype=dtype
            ):
                batch_shape = torch.Size([2])
                kernel = PositiveIndexKernel(
                    num_tasks=3, rank=3, batch_shape=batch_shape
                ).to(dtype=dtype)
                # 1D input with correct n_lower but no batch — exercises expand branch
                target_corr = torch.rand(3, dtype=dtype)
                kernel._set_lower_triangle_corr(target_corr)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
                self.assertEqual(covar.shape, torch.Size([2, 3, 3]))

            # Test _set_lower_triangle_corr with boundary values
            with self.subTest("set_lower_triangle_corr_boundary", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=2, rank=2).to(dtype=dtype)
                kernel._set_lower_triangle_corr(torch.tensor([0.0], dtype=dtype))
                self.assertTrue(kernel._lower_triangle_corr.isfinite().all())
                kernel._set_lower_triangle_corr(torch.tensor([0.999], dtype=dtype))
                self.assertTrue(kernel._lower_triangle_corr.isfinite().all())

            # Test _set_lower_triangle_corr with non-PD input
            with self.subTest("set_lower_triangle_corr_non_pd", dtype=dtype):
                kernel = PositiveIndexKernel(num_tasks=3, rank=3).to(dtype=dtype)
                # [0.99, 0.01, 0.99] does not form a PD correlation matrix
                non_pd_corr = torch.tensor([0.99, 0.01, 0.99], dtype=dtype)
                kernel._set_lower_triangle_corr(non_pd_corr)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())

            # Test roundtrip accuracy for full-rank
            with self.subTest("set_lower_triangle_corr_roundtrip", dtype=dtype):
                kernel = PositiveIndexKernel(
                    num_tasks=3, rank=3, unit_scale_for_target=False
                ).to(dtype=dtype)
                # Set var to small known value to isolate correlation effect
                kernel.initialize(raw_var=torch.full((3,), -5.0, dtype=dtype))
                target_corr = torch.tensor([0.8, 0.5, 0.6], dtype=dtype)
                kernel._set_lower_triangle_corr(target_corr)
                recovered = kernel._lower_triangle_corr
                self.assertAllClose(recovered, target_corr, atol=0.05)

            # Test sample_all_priors with task_prior
            with self.subTest("sample_all_priors_unbatched", dtype=dtype):
                task_prior = UniformPrior(0.0, 1.0)
                kernel = PositiveIndexKernel(
                    num_tasks=3,
                    rank=3,
                    task_prior=task_prior,
                ).to(dtype=dtype)

                corr_before = kernel._lower_triangle_corr.clone()
                sample_all_priors(kernel)

                corr_after = kernel._lower_triangle_corr
                self.assertFalse(torch.allclose(corr_before, corr_after))

                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())

            # Test with BetaPrior
            with self.subTest("beta_prior", dtype=dtype):
                task_prior = BetaPrior(1.2, 0.9)
                kernel = PositiveIndexKernel(
                    num_tasks=4,
                    rank=4,
                    task_prior=task_prior,
                ).to(dtype=dtype)
                sample_all_priors(kernel)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())

            # Test sample_all_priors
            with self.subTest("sample_all_priors", dtype=dtype):
                task_prior = UniformPrior(0.0, 1.0)
                kernel = PositiveIndexKernel(
                    num_tasks=3,
                    rank=3,
                    task_prior=task_prior,
                ).to(dtype=dtype)
                sample_all_priors(kernel)
                covar = kernel.covar_matrix
                eigvals = torch.linalg.eigvalsh(covar)
                self.assertTrue((eigvals > 0).all())
