#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Contributor: @Mamunur02

import unittest
from unittest.mock import patch

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskVariationalGP
from botorch.models.utils.inducing_point_allocators import UnitQualityFunction
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.utils.inducing_point_allocators import (
    RandomizedPivotedCholesky,
)
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import VariationalELBO


class TestRandomizedPivotedCholesky(BotorchTestCase):
    def test_allocates_distinct_indices_and_is_reproducible(self):
        for dtype in (torch.float, torch.double):
            X = torch.rand(20, 3, device=self.device, dtype=dtype, requires_grad=True)
            kernel = ScaleKernel(RBFKernel(ard_num_dims=3)).to(
                device=self.device, dtype=dtype
            )
            kernel.base_kernel.lengthscale = torch.tensor([0.3, 0.7, 1.2])
            allocator = RandomizedPivotedCholesky()
            torch.manual_seed(123)
            points = allocator.allocate_inducing_points(X, kernel, 7, torch.Size())
            torch.manual_seed(123)
            repeated = allocator.allocate_inducing_points(X, kernel, 7, torch.Size())
            self.assertTrue(torch.equal(points, repeated))
            self.assertEqual(points.shape, torch.Size([7, 3]))
            self.assertEqual(points.dtype, dtype)
            self.assertEqual(points.device, self.device)
            self.assertFalse(points.requires_grad)
            self.assertEqual(torch.unique(points, dim=0).shape[0], 7)
            self.assertIsInstance(
                allocator._get_quality_function(), UnitQualityFunction
            )

    def test_multinomial_receives_residual_diagonal_probabilities(self):
        X = torch.arange(5, device=self.device, dtype=torch.double).unsqueeze(-1)
        kernel = ScaleKernel(RBFKernel()).to(device=self.device, dtype=torch.double)
        diagonal = kernel(X, diag=True).clone()
        seen = []

        def choose(weights, count):
            seen.append(weights.clone())
            return torch.tensor([len(seen) - 1], device=weights.device)

        with patch("torch.multinomial", side_effect=choose):
            RandomizedPivotedCholesky().allocate_inducing_points(
                X, kernel, 3, torch.Size()
            )
        self.assertTrue(torch.equal(seen[0], diagonal / diagonal.max()))
        matrix = kernel(X, X).to_dense()
        for j in range(3):
            residual = diagonal.clone()
            if j:
                cross = matrix[:, :j]
                residual -= (cross * torch.linalg.solve(matrix[:j, :j], cross.T).T).sum(
                    -1
                )
            residual[:j] = 0
            self.assertTrue(torch.allclose(seen[j], residual / diagonal.max()))

    def test_large_output_scale_preserves_sampling(self):
        X = torch.arange(5, device=self.device, dtype=torch.float).unsqueeze(-1)
        kernel = ScaleKernel(RBFKernel()).to(self.device)
        allocator = RandomizedPivotedCholesky()
        points = []
        for scale in (1.0, 1e38):
            kernel.outputscale = scale
            torch.manual_seed(123)
            points.append(
                allocator.allocate_inducing_points(X, kernel, 3, torch.Size())
            )
        self.assertTrue(torch.equal(*points))

    def test_duplicate_inputs_zero_kernel_and_rank_exhaustion_return_exact_count(self):
        X = torch.stack(
            [torch.zeros(6, device=self.device), torch.arange(6, device=self.device)],
            dim=-1,
        ).to(torch.double)
        zero_kernel = ScaleKernel(RBFKernel(active_dims=[0])).to(
            device=self.device, dtype=torch.double
        )
        with torch.no_grad():
            zero_kernel.outputscale = 0
        points = RandomizedPivotedCholesky().allocate_inducing_points(
            X, zero_kernel, 6, torch.Size()
        )
        self.assertEqual(points.shape, X.shape)
        self.assertEqual(torch.unique(points[:, 1]).numel(), 6)

        kernel = ScaleKernel(RBFKernel(active_dims=[0])).to(
            device=self.device, dtype=torch.double
        )
        duplicate_X = torch.zeros(6, 1, device=self.device, dtype=torch.double)
        duplicate_points = RandomizedPivotedCholesky().allocate_inducing_points(
            duplicate_X,
            ScaleKernel(RBFKernel()).to(device=self.device, dtype=torch.double),
            5,
            torch.Size(),
        )
        self.assertEqual(duplicate_points.shape, torch.Size([5, 1]))
        self.assertTrue(
            torch.equal(duplicate_points, torch.zeros_like(duplicate_points))
        )
        points = RandomizedPivotedCholesky().allocate_inducing_points(
            X, kernel, 5, torch.Size()
        )
        self.assertEqual(points.shape, torch.Size([5, 2]))
        self.assertEqual(torch.unique(points[:, 1]).numel(), 5)

    def test_only_evaluates_kernel_diagonal_and_single_columns(self):
        class InstrumentedKernel(ScaleKernel):
            def __init__(self):
                super().__init__(RBFKernel())
                self.full_calls = 0
                self.column_calls = 0
                self.shapes = []

            def forward(self, x1, x2, diag=False, **params):
                self.shapes.append((x1.shape[-2], x2.shape[-2], diag))
                if diag:
                    return super().forward(x1, x2, diag=True, **params)
                if x1.shape[-2] == x2.shape[-2]:
                    self.full_calls += 1
                else:
                    self.column_calls += 1
                return super().forward(x1, x2, **params)

        X = torch.rand(12, 2, device=self.device, dtype=torch.double)
        kernel = InstrumentedKernel().to(device=self.device, dtype=torch.double)
        RandomizedPivotedCholesky().allocate_inducing_points(X, kernel, 4, torch.Size())
        self.assertEqual(kernel.full_calls, 0)
        self.assertEqual(kernel.column_calls, 3)
        self.assertEqual(kernel.shapes, [(12, 12, True)] + [(12, 1, False)] * 3)

    def test_rejects_invalid_covariance_values(self):
        X = torch.rand(5, 2, device=self.device, dtype=torch.double)
        kernel = RBFKernel().to(X)
        allocator = RandomizedPivotedCholesky()
        for value in (-1.0, float("nan"), float("inf")):
            with patch.object(kernel, "forward", return_value=X.new_full((5,), value)):
                with self.assertRaisesRegex(UnsupportedError, "finite and nonnegative"):
                    allocator.allocate_inducing_points(X, kernel, 2, torch.Size())

        def invalid_column(x1, x2, diag=False, **kwargs):
            return X.new_ones(5) if diag else X.new_full((5, 1), float("nan"))

        with patch.object(kernel, "forward", side_effect=invalid_column):
            with self.assertRaisesRegex(UnsupportedError, "columns must be finite"):
                allocator.allocate_inducing_points(X, kernel, 2, torch.Size())

    def test_rejects_unsupported_arguments(self):
        X = torch.rand(5, 2, device=self.device, dtype=torch.double)
        kernel = RBFKernel().to(device=self.device, dtype=torch.double)
        allocator = RandomizedPivotedCholesky()
        cases = [
            (X.unsqueeze(0), 2, torch.Size(), "2-dimensional"),
            (X, 2, torch.Size([1]), "must be empty"),
            (X, True, torch.Size(), "integer"),
            (X, 0, torch.Size(), "1 <= m <= n"),
            (X, 6, torch.Size(), "1 <= m <= n"),
            (X.half(), 2, torch.Size(), "float32 or float64"),
        ]
        for inputs, m, batch_shape, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(UnsupportedError, message),
            ):
                allocator.allocate_inducing_points(inputs, kernel, m, batch_shape)

        with patch.object(kernel, "forward", return_value=torch.ones(1, 5)):
            with self.assertRaisesRegex(UnsupportedError, "diagonal must have shape"):
                allocator.allocate_inducing_points(X, kernel, 2, torch.Size())

    def test_variational_gp_training_and_reinitialization(self):
        train_X = torch.rand(18, 2, device=self.device, dtype=torch.double)
        train_Y = torch.sin(train_X[:, :1] * 5)
        allocator = RandomizedPivotedCholesky()
        covar_module = ScaleKernel(RBFKernel(active_dims=[0])).to(
            device=self.device, dtype=torch.double
        )
        covar_module.base_kernel.lengthscale = 0.35
        covar_module.outputscale = 1.7
        model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            covar_module=covar_module,
            inducing_points=6,
            learn_inducing_points=False,
            inducing_point_allocator=allocator,
        )
        inducing = model.model.variational_strategy.inducing_points.detach().clone()
        model.train()
        mll = VariationalELBO(model.likelihood, model.model, num_data=train_X.shape[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = -mll(model(train_X), train_Y.squeeze(-1))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_X[:3])
        self.assertTrue(torch.isfinite(posterior.mean).all())

        replacement_X = train_X.clone()
        replacement_X[:, 0] = 0.5
        next_inducing = model.init_inducing_points(replacement_X)
        self.assertEqual(next_inducing.shape, inducing.shape)
        self.assertEqual(torch.unique(next_inducing, dim=0).shape[0], 6)
        self.assertTrue(
            torch.equal(next_inducing, model.model.variational_strategy.inducing_points)
        )
        with torch.no_grad():
            posterior = model.posterior(train_X[:3])
        self.assertTrue(torch.isfinite(posterior.mean).all())
        self.assertTrue(torch.isfinite(posterior.variance).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        X = torch.rand(10, 2, device="cuda", dtype=torch.double)
        kernel = ScaleKernel(RBFKernel()).to(device="cuda", dtype=torch.double)
        points = RandomizedPivotedCholesky().allocate_inducing_points(
            X, kernel, 4, torch.Size()
        )
        self.assertEqual(points.device.type, "cuda")
