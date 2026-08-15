#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MultiTaskEmpiricalOneDimensionalGP model."""

from __future__ import annotations

import torch
from botorch.models.empirical_gps import BaseAugmentedEmpiricalKernel
from botorch.models.empirical_gps.multitask_empirical_1d_gp import (
    _extract_progression_and_task,
    _validate_heterogeneous_historical_data,
    _validate_task_indices,
    MultiTaskEmpiricalOneDimensionalGP,
)
from botorch.models.empirical_gps.utils import LinearInterpolation1D
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import IndexKernel, ProductKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from torch import Tensor


class TestMultiTaskEmpiricalOneDimensionalGP(BotorchTestCase):
    """Tests for MultiTaskEmpiricalOneDimensionalGP."""

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_data(
        self,
        num_curves: int = 20,
        num_progression_per_task: list[int] | None = None,
        num_train_per_task: int = 10,
        num_tasks: int = 2,
        different_domains: bool = False,
        correlation: str = "independent",
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
        """Generate test data with per-task historical data.

        Args:
            correlation: "independent" for random curves per task,
                "perfect" for identical curves across tasks (requires all
                tasks to share the same number of progression points).
        """
        if num_progression_per_task is None:
            num_progression_per_task = [50, 30] if num_tasks == 2 else [50] * num_tasks

        historical_Xs = []
        historical_Ys = []
        base_curves = None

        for t in range(num_tasks):
            num_prog = num_progression_per_task[t]
            if different_domains:
                start = 0.1 + t * 0.05
                end = 1.0 - t * 0.1
            else:
                start, end = 0.1, 1.0

            X_t = torch.linspace(start, end, num_prog, device=self.device)
            X_t = X_t.unsqueeze(-1)

            if correlation == "perfect":
                if base_curves is None:
                    base_curves = torch.randn(num_curves, num_prog, device=self.device)
                Y_t = base_curves.clone()
            else:
                Y_t = torch.randn(num_curves, num_prog, device=self.device)

            historical_Xs.append(X_t)
            historical_Ys.append(Y_t)

        train_X_list = []
        train_Y_list = []
        for task_idx in range(num_tasks):
            if different_domains:
                start = 0.1 + task_idx * 0.05
                end = 1.0 - task_idx * 0.1
            else:
                start, end = 0.1, 1.0

            prog = torch.rand(num_train_per_task, 1, device=self.device)
            prog = prog * (end - start) + start
            task_col = torch.full(
                (num_train_per_task, 1), task_idx, device=self.device, dtype=prog.dtype
            )
            train_X_list.append(torch.cat([prog, task_col], dim=-1))
            train_Y_list.append(torch.randn(num_train_per_task, 1, device=self.device))

        train_X = torch.cat(train_X_list, dim=0)
        train_Y = torch.cat(train_Y_list, dim=0)
        train_Yvar = torch.full_like(train_Y, 0.01)
        return train_X, train_Y, train_Yvar, historical_Xs, historical_Ys

    def _make_model(
        self, data: tuple | None = None, **kwargs
    ) -> MultiTaskEmpiricalOneDimensionalGP:
        """Build a model from data tuple, with overrides via kwargs."""
        if data is None:
            data = self._get_data()
        train_X, train_Y, train_Yvar, hist_Xs, hist_Ys = data
        defaults = dict(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=-1,
            historical_Xs=hist_Xs,
            historical_Ys=hist_Ys,
        )
        defaults.update(kwargs)
        return MultiTaskEmpiricalOneDimensionalGP(**defaults)

    # =========================================================================
    # Validation
    # =========================================================================

    def _test_validate_heterogeneous_historical_data(self) -> None:
        Xs = [torch.randn(50, 1), torch.randn(30, 1)]
        Ys = [torch.randn(20, 50), torch.randn(20, 30)]
        self.assertEqual(_validate_heterogeneous_historical_data(Xs, Ys), 20)

        for bad_args, regex in [
            ((Xs[:1], Ys), "historical_Xs has .* tasks"),
            (([], []), "cannot be empty"),
            ((Xs, [torch.randn(20, 50), torch.randn(15, 30)]), "same number of curves"),
            ((Xs, [torch.randn(20, 50, 2), torch.randn(20, 30)]), "must be 2-dim"),
            (([torch.randn(50), torch.randn(30, 1)], Ys), "must be .* x 1"),
            (([torch.randn(40, 1), torch.randn(30, 1)], Ys), "has .* points but"),
        ]:
            with self.assertRaisesRegex(ValueError, regex):
                _validate_heterogeneous_historical_data(*bad_args)

    def _test_model_validation_errors(self) -> None:
        torch.manual_seed(12345)

        # Task SUBSET is allowed — 2 tasks in train_X, 3 in historical
        data_2 = self._get_data(num_tasks=2)
        data_3 = self._get_data(num_tasks=3, num_progression_per_task=[50, 40, 30])
        model = self._make_model(
            data_2,
            historical_Xs=data_3[3],
            historical_Ys=data_3[4],
        )
        self.assertEqual(model.num_tasks, 3)

        # OUT OF RANGE task index
        train_X_bad = torch.tensor([[0.5, 0.0], [0.6, 5.0]], device=self.device)
        with self.assertRaisesRegex(ValueError, r"must be in \[0,"):
            self._make_model(
                data_2,
                train_X=train_X_bad,
                train_Y=torch.randn(2, 1, device=self.device),
            )

        # d != 1 error
        with self.assertRaisesRegex(ValueError, r"requires d=1.*got 3 columns"):
            self._make_model(
                data_3,
                train_X=torch.tensor(
                    [[0.5, 0.3, 0.0], [0.6, 0.4, 1.0]], device=self.device
                ),
                train_Y=torch.randn(2, 1, device=self.device),
            )

    def _test_module_consistency_validation(self) -> None:
        torch.manual_seed(12345)
        data_2 = self._get_data(num_tasks=2)
        data_3 = self._get_data(num_tasks=3, num_progression_per_task=[50, 40, 30])

        model_2 = self._make_model(data_2)
        model_3 = self._make_model(data_3)

        # mean_module.num_tasks != covar_module.num_tasks
        with self.assertRaisesRegex(ValueError, r"mean_module\.num_tasks.*!=.*covar"):
            self._make_model(
                data_2,
                mean_module=model_2.mean_module,
                covar_module=model_3.covar_module,
            )

        # Only one module provided (must be both or neither)
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            MultiTaskEmpiricalOneDimensionalGP(
                train_X=data_2[0],
                train_Y=data_2[1],
                task_feature=-1,
                mean_module=model_2.mean_module,
            )

        # No modules and no historical data
        with self.assertRaisesRegex(ValueError, "must be provided"):
            MultiTaskEmpiricalOneDimensionalGP(
                train_X=data_2[0],
                train_Y=data_2[1],
                task_feature=-1,
            )

        # correction conflict with pre-built covar_module
        with self.assertRaisesRegex(ValueError, "correction is ignored"):
            self._make_model(
                data_2,
                mean_module=model_2.mean_module,
                covar_module=model_2.covar_module,
                correction=1,
            )

    # =========================================================================
    # Basic model construction & forward/posterior
    # =========================================================================

    def _test_model_basic(self) -> None:
        """Test instantiation, forward, and posterior with both data types."""
        torch.manual_seed(12345)

        for data_kwargs in [
            dict(num_progression_per_task=[50, 50]),  # homogeneous
            dict(num_progression_per_task=[50, 30]),  # heterogeneous
        ]:
            data = self._get_data(**data_kwargs)
            train_X = data[0]
            model = self._make_model(data)
            self.assertEqual(model.num_outputs, 2)
            self.assertEqual(model.num_tasks, 2)

            # Forward in train mode
            model.train()
            prior = model(train_X)
            self.assertEqual(prior.mean.shape, (train_X.shape[0],))

            # Posterior in eval mode
            model.eval()
            test_X = torch.tensor([[0.3, 0.0], [0.5, 1.0]], device=self.device)
            self.assertEqual(model.posterior(test_X).mean.shape, (2, 1))

            # Multi-output posterior (no task column)
            self.assertEqual(
                model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
                (1, 2),
            )

    def _test_model_with_yvar(self) -> None:
        """Test model with fixed observation noise."""
        torch.manual_seed(12345)
        data = self._get_data()
        model = self._make_model(data, train_Yvar=data[2])

        self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
        model.eval()
        self.assertEqual(
            model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
            (1, 2),
        )

    def _test_kernel_psd(self) -> None:
        """Test that kernel is positive semi-definite."""
        torch.manual_seed(12345)
        data = self._get_data()
        model = self._make_model(data)
        K = model.covar_module(data[0], data[0]).to_dense()
        eigvals = torch.linalg.eigvalsh(K)
        self.assertTrue(
            (eigvals >= -1e-5).all(),
            f"Kernel has negative eigenvalues: {eigvals.min()}",
        )

    def _test_heterogeneous_domains(self) -> None:
        """Test that each task's interpolant uses its own domain."""
        torch.manual_seed(12345)
        historical_Xs = [
            torch.linspace(0.0, 0.5, 50, device=self.device).unsqueeze(-1),
            torch.linspace(0.5, 1.0, 50, device=self.device).unsqueeze(-1),
        ]
        historical_Ys = [
            torch.randn(20, 50, device=self.device),
            torch.randn(20, 50, device=self.device),
        ]
        train_X = torch.tensor([[0.25, 0.0], [0.75, 1.0]], device=self.device)
        train_Y = torch.randn(2, 1, device=self.device)
        model = MultiTaskEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=-1,
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
        )
        model.eval()
        self.assertFalse(model.posterior(train_X).mean.isnan().any())

        # Querying outside a task's domain should raise
        with self.assertRaisesRegex(ValueError, "interpolation range"):
            model.posterior(torch.tensor([[0.75, 0.0]], device=self.device))

    # =========================================================================
    # Convenience constructors
    # =========================================================================

    def _test_from_homogeneous_data(self) -> None:
        torch.manual_seed(12345)
        num_curves, num_progression, num_tasks = 20, 50, 2
        historical_X = torch.linspace(
            0.1, 1.0, num_progression, device=self.device
        ).unsqueeze(-1)
        historical_Y = torch.randn(
            num_curves, num_progression, num_tasks, device=self.device
        )
        data = self._get_data(num_progression_per_task=[num_progression] * num_tasks)

        model = MultiTaskEmpiricalOneDimensionalGP.from_homogeneous_data(
            train_X=data[0],
            train_Y=data[1],
            task_feature=-1,
            historical_X=historical_X,
            historical_Y=historical_Y,
        )
        self.assertEqual(model.num_outputs, num_tasks)
        model.eval()
        self.assertEqual(
            model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
            (1, num_tasks),
        )

        # Error for wrong dimension
        with self.assertRaisesRegex(ValueError, "must be 3-dim"):
            MultiTaskEmpiricalOneDimensionalGP.from_homogeneous_data(
                train_X=data[0],
                train_Y=data[1],
                task_feature=-1,
                historical_X=historical_X,
                historical_Y=historical_Y[..., 0],
            )

    def _test_from_wide_format(self) -> None:
        """Test from_wide_format: basic, with Yvar, with output_tasks, errors."""
        torch.manual_seed(54321)
        num_curves, num_tasks, num_train = 20, 2, 10

        historical_Xs = [
            torch.linspace(0.1, 1.0, 50, device=self.device).unsqueeze(-1),
            torch.linspace(0.1, 1.0, 30, device=self.device).unsqueeze(-1),
        ]
        historical_Ys = [
            torch.randn(num_curves, 50, device=self.device),
            torch.randn(num_curves, 30, device=self.device),
        ]
        train_X = torch.rand(num_train, 1, device=self.device) * 0.9 + 0.1
        train_Y = torch.randn(num_train, num_tasks, device=self.device)
        wf_kwargs = dict(
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
        )

        # Basic
        model = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
            train_X=train_X, train_Y=train_Y, **wf_kwargs
        )
        self.assertEqual(model.num_outputs, num_tasks)
        self.assertEqual(model.train_inputs[0].shape, (num_train * num_tasks, 2))
        model.eval()
        self.assertEqual(
            model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
            (1, num_tasks),
        )

        # With Yvar
        train_Yvar = torch.full((num_train, num_tasks), 0.01, device=self.device)
        model_yvar = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
            train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar, **wf_kwargs
        )
        self.assertIsInstance(model_yvar.likelihood, FixedNoiseGaussianLikelihood)

        # With output_tasks (3 tasks, subset to [0, 2])
        hist_Xs_3 = historical_Xs + [
            torch.linspace(0.1, 1.0, 25, device=self.device).unsqueeze(-1)
        ]
        hist_Ys_3 = historical_Ys + [torch.randn(num_curves, 25, device=self.device)]
        train_Y_3 = torch.randn(num_train, 3, device=self.device)
        model_out = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
            train_X=train_X,
            train_Y=train_Y_3,
            historical_Xs=hist_Xs_3,
            historical_Ys=hist_Ys_3,
            output_tasks=[0, 2],
        )
        self.assertEqual(model_out.num_outputs, 2)
        self.assertEqual(model_out.num_tasks, 3)

        # Validation errors
        with self.assertRaisesRegex(ValueError, "train_X has .* inputs but train_Y"):
            MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
                train_X=train_X,
                train_Y=torch.randn(11, num_tasks, device=self.device),
                **wf_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "train_Yvar shape .* must match"):
            MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=torch.randn(10, 3, device=self.device),
                **wf_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "train_Y has .* tasks but"):
            MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
                train_X=train_X,
                train_Y=torch.randn(10, 3, device=self.device),
                **wf_kwargs,
            )

    # =========================================================================
    # Cross-task correlation & correctness
    # =========================================================================

    def _test_cross_task_correlation(self) -> None:
        """Posterior samples show high correlation when historical curves match."""
        torch.manual_seed(12345)
        data = self._get_data(num_progression_per_task=[50, 50], correlation="perfect")
        model = self._make_model(data, train_Yvar=data[2])
        model.eval()

        test_X = torch.tensor([[0.5]], device=self.device)
        samples = model.posterior(test_X).rsample(torch.Size([1000])).squeeze(1)
        correlation = torch.corrcoef(samples.T)[0, 1]
        self.assertGreater(correlation.item(), 0.9)

    def _test_mean_interpolation_correctness(self) -> None:
        """Mean function correctly interpolates known linear historical data."""
        torch.manual_seed(12345)
        historical_Xs = [
            torch.tensor([[0.0], [0.5], [1.0]], device=self.device),
            torch.tensor([[0.2], [0.5], [0.8]], device=self.device),
        ]
        historical_Ys = [
            torch.tensor([[0.0, 0.5, 1.0]], device=self.device),
            torch.tensor([[1.0, 1.5, 2.0]], device=self.device),
        ]
        train_X = torch.tensor(
            [[0.25, 0.0], [0.75, 0.0], [0.35, 1.0], [0.65, 1.0]], device=self.device
        )
        model = MultiTaskEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=torch.randn(4, 1, device=self.device),
            task_feature=-1,
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
        )
        mean = model.mean_module(
            torch.tensor([[0.5, 0.0], [0.5, 1.0]], device=self.device)
        )
        self.assertAlmostEqual(mean[0].item(), 0.5, places=5)
        self.assertAlmostEqual(mean[1].item(), 1.5, places=5)

    def _test_kernel_cross_task_covariance(self) -> None:
        """Cross-task covariance ≈ 1 when curves are proportional."""
        torch.manual_seed(12345)
        num_curves = 10
        base = torch.randn(num_curves, device=self.device)
        X0 = torch.linspace(0, 1, 50, device=self.device).unsqueeze(-1)
        X1 = torch.linspace(0.2, 0.8, 30, device=self.device).unsqueeze(-1)
        historical_Xs = [X0, X1]
        historical_Ys = [
            base.unsqueeze(-1) * X0.squeeze(-1),
            base.unsqueeze(-1) * X1.squeeze(-1),
        ]

        train_X = torch.tensor([[0.5, 0.0], [0.5, 1.0]], device=self.device)
        model = MultiTaskEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=torch.randn(2, 1, device=self.device),
            task_feature=-1,
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
        )
        K = model.covar_module(train_X, train_X).to_dense()
        correlation = K[0, 1] / torch.sqrt(K[0, 0] * K[1, 1])
        self.assertGreater(correlation.item(), 0.99)

    # =========================================================================
    # Kernel edge cases: diag, batch, num_curves=1
    # =========================================================================

    def _test_kernel_diag_and_batch(self) -> None:
        """Test diag=True correctness and batched mean/kernel evaluation."""
        torch.manual_seed(12345)
        data = self._get_data(num_progression_per_task=[50, 50])
        model = self._make_model(data)
        train_X = data[0]

        # Unbatched diag
        K_full = model.covar_module(train_X, train_X).to_dense()
        K_diag = model.covar_module(train_X, train_X, diag=True)
        self.assertAllClose(K_diag, torch.diag(K_full), atol=1e-6)

        # Batched input: (batch=3, n=4, 2)
        test_X = torch.tensor(
            [
                [[0.3, 0.0], [0.5, 0.0], [0.3, 1.0], [0.5, 1.0]],
                [[0.4, 0.0], [0.6, 0.0], [0.4, 1.0], [0.6, 1.0]],
                [[0.2, 0.0], [0.7, 0.0], [0.2, 1.0], [0.7, 1.0]],
            ],
            device=self.device,
        )
        mean = model.mean_module(test_X)
        self.assertEqual(mean.shape, (3, 4))
        self.assertFalse(mean.isnan().any())

        K_batch = model.covar_module(test_X, test_X).to_dense()
        self.assertEqual(K_batch.shape, (3, 4, 4))
        for b in range(3):
            eigvals = torch.linalg.eigvalsh(K_batch[b])
            self.assertTrue((eigvals >= -1e-5).all())

        # Batched diag
        K_batch_diag = model.covar_module(test_X, test_X, diag=True)
        self.assertEqual(K_batch_diag.shape, (3, 4))
        for b in range(3):
            self.assertAllClose(K_batch_diag[b], torch.diag(K_batch[b]), atol=1e-6)

    def _test_num_curves_one(self) -> None:
        """With 1 curve, centering gives all zeros, so kernel should be zero."""
        torch.manual_seed(12345)
        historical_Xs = [
            torch.linspace(0.1, 1.0, 20, device=self.device).unsqueeze(-1),
            torch.linspace(0.1, 1.0, 15, device=self.device).unsqueeze(-1),
        ]
        historical_Ys = [
            torch.randn(1, 20, device=self.device),
            torch.randn(1, 15, device=self.device),
        ]
        train_X = torch.tensor(
            [[0.3, 0.0], [0.5, 0.0], [0.3, 1.0], [0.5, 1.0]], device=self.device
        )
        model = MultiTaskEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=torch.randn(4, 1, device=self.device),
            task_feature=-1,
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
        )
        K = model.covar_module(train_X, train_X).to_dense()
        self.assertAllClose(K, torch.zeros_like(K), atol=1e-10)

    # =========================================================================
    # Additional features: correction, task_feature, output_tasks, prebuilt, etc.
    # =========================================================================

    def _test_correction(self) -> None:
        """Bessel correction scales kernel by N/(N-1)."""
        torch.manual_seed(12345)
        data = self._get_data(num_curves=20, num_progression_per_task=[50, 50])
        model_ml = self._make_model(data, correction=0)
        model_bessel = self._make_model(data, correction=1)
        K_ml = model_ml.covar_module(data[0][:1], data[0][:1]).to_dense()
        K_bessel = model_bessel.covar_module(data[0][:1], data[0][:1]).to_dense()
        self.assertAlmostEqual((K_bessel[0, 0] / K_ml[0, 0]).item(), 20 / 19, places=4)

    def _test_task_feature_0(self) -> None:
        """task_feature=0 (first column is task index)."""
        torch.manual_seed(12345)
        data = self._get_data(num_progression_per_task=[50, 50])
        model = self._make_model(data, train_X=data[0].flip(-1), task_feature=0)
        model.eval()
        test_X = torch.tensor([[0.0, 0.5], [1.0, 0.5]], device=self.device)
        self.assertEqual(model.posterior(test_X).mean.shape, (2, 1))

    def _test_output_tasks(self) -> None:
        """output_tasks subsets outputs."""
        torch.manual_seed(12345)
        data = self._get_data(num_tasks=3, num_progression_per_task=[50, 40, 30])
        model = self._make_model(data, output_tasks=[0, 2])
        self.assertEqual(model.num_tasks, 3)
        self.assertEqual(model.num_outputs, 2)
        model.eval()
        self.assertEqual(
            model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
            (1, 2),
        )

    def _test_prebuilt_modules(self) -> None:
        """Pre-built mean+covar give same results as historical data."""
        torch.manual_seed(12345)
        data = self._get_data()
        model_ref = self._make_model(data)

        model_prebuilt = self._make_model(
            data,
            mean_module=model_ref.mean_module,
            covar_module=model_ref.covar_module,
        )

        test_X = torch.tensor([[0.5, 0.0], [0.5, 1.0]], device=self.device)
        model_ref.eval()
        model_prebuilt.eval()
        self.assertAllClose(
            model_prebuilt.posterior(test_X).mean,
            model_ref.posterior(test_X).mean,
        )

    def _test_partial_task_observation(self) -> None:
        """Observe only task 0, predict both tasks."""
        torch.manual_seed(12345)
        data = self._get_data()
        train_X = torch.tensor([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], device=self.device)
        model = self._make_model(
            data, train_X=train_X, train_Y=torch.randn(3, 1, device=self.device)
        )
        self.assertEqual(model.num_tasks, 2)
        model.eval()
        self.assertEqual(
            model.posterior(torch.tensor([[0.5]], device=self.device)).mean.shape,
            (1, 2),
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def _test_serialization(self) -> None:
        """Buffers registered + state_dict round-trip preserves predictions."""
        torch.manual_seed(12345)
        data = self._get_data()
        model = self._make_model(data)

        # Check buffers are in state_dict
        sd = model.state_dict()
        sd_keys = set(sd.keys())
        for prefix in ("mean_module", "covar_module"):
            for t in range(model.num_tasks):
                for suffix in ("_x", "_y"):
                    key = f"{prefix}.interpolants.{t}.{suffix}"
                    self.assertIn(key, sd_keys, f"Missing buffer {key}")

        # Round-trip: load state_dict into fresh model, compare predictions
        model.eval()
        test_X = torch.tensor([[0.5]], device=self.device)
        original_mean = model.posterior(test_X).mean.clone()
        original_var = model.posterior(test_X).variance.clone()

        model2 = self._make_model(data)
        model2.load_state_dict(sd)
        model2.eval()
        self.assertAllClose(model2.posterior(test_X).mean, original_mean)
        self.assertAllClose(model2.posterior(test_X).variance, original_var)

    def _test_coverage_gaps(self) -> None:
        """Cover the task-index guard, the mean NaN guard, and _split_inputs."""
        torch.manual_seed(12345)

        # _validate_task_indices raises on out-of-range indices.
        with self.assertRaisesRegex(ValueError, r"out of range \[0, 2\)"):
            _validate_task_indices(
                torch.tensor([0, 1, 5], device=self.device), num_tasks=2
            )

        # _extract_progression_and_task rejects non-integer task labels instead
        # of silently truncating them (0.9 -> 0, 1.5 -> 1).
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            _extract_progression_and_task(
                torch.tensor(
                    [[0.0, 0.9], [1.0, 1.5]], device=self.device, dtype=torch.double
                ),
                task_feature=1,
            )
        # NaN labels are surfaced in the message rather than yielding an empty
        # list, since NaN compares False against the non-integer threshold.
        with self.assertRaisesRegex(ValueError, "nan"):
            _extract_progression_and_task(
                torch.tensor(
                    [[0.0, float("nan")], [1.0, 1.0]],
                    device=self.device,
                    dtype=torch.double,
                ),
                task_feature=1,
            )
        # Integer-valued float labels are accepted and cast correctly.
        prog, task = _extract_progression_and_task(
            torch.tensor(
                [[0.5, 0.0], [1.5, 2.0]], device=self.device, dtype=torch.double
            ),
            task_feature=1,
        )
        self.assertTrue(torch.equal(task, torch.tensor([0, 2], device=self.device)))
        self.assertTrue(
            torch.allclose(
                prog, torch.tensor([0.5, 1.5], device=self.device, dtype=torch.double)
            )
        )
        # Near-integer floats (the float repr of an int) round to the nearest
        # label rather than truncating: 1.9999999 -> 2, not 1.
        _, task_near = _extract_progression_and_task(
            torch.tensor([[0.5, 1.9999999]], device=self.device, dtype=torch.double),
            task_feature=1,
        )
        self.assertTrue(torch.equal(task_near, torch.tensor([2], device=self.device)))
        # Integer-dtype task labels take the non-floating-point branch (no rounding).
        _, task_int = _extract_progression_and_task(
            torch.tensor([[0, 2], [1, 3]], device=self.device, dtype=torch.long),
            task_feature=1,
        )
        self.assertTrue(torch.equal(task_int, torch.tensor([2, 3], device=self.device)))

        data = self._get_data(num_progression_per_task=[50, 50])
        model = self._make_model(data)

        # The mean NaN guard fires when an interpolant yields NaN out of range.
        # Inject (via the public constructor) an interpolant that fills
        # out-of-bounds queries with NaN instead of raising, so the guard in
        # the mean module's forward is exercised.
        mean_module = model.mean_module
        knots = torch.linspace(0.1, 1.0, 5, device=self.device)
        mean_module.interpolants[0] = LinearInterpolation1D(
            knots,
            torch.zeros(5, device=self.device),
            bounds_error=False,
        )
        far_oob = torch.tensor([[1.0e6, 0.0]], device=self.device)
        with self.assertRaisesRegex(ValueError, "Mean contains NaN values"):
            mean_module(far_oob)

        # _split_inputs separates progression, task indices, and trailing features.
        x = torch.tensor([[0.5, 0.0], [0.6, 1.0]], device=self.device)
        before, task_idcs, after = model._split_inputs(x)
        self.assertEqual(task_idcs.shape, (2, 1))
        self.assertAllClose(
            task_idcs,
            torch.tensor([[0], [1]], device=self.device, dtype=task_idcs.dtype),
        )
        self.assertEqual(before.shape[-1], model._task_feature)

    # =========================================================================
    # Public test methods (entry points)
    # =========================================================================

    def test_validation(self) -> None:
        self._test_validate_heterogeneous_historical_data()
        self._test_model_validation_errors()
        self._test_module_consistency_validation()

    def test_model_basic(self) -> None:
        self._test_model_basic()
        self._test_model_with_yvar()
        self._test_kernel_psd()
        self._test_heterogeneous_domains()

    def test_convenience_constructors(self) -> None:
        self._test_from_homogeneous_data()
        self._test_from_wide_format()

    def test_cross_task_correlation_and_correctness(self) -> None:
        self._test_cross_task_correlation()
        self._test_mean_interpolation_correctness()
        self._test_kernel_cross_task_covariance()

    def test_kernel_edge_cases(self) -> None:
        self._test_kernel_diag_and_batch()
        self._test_num_curves_one()

    def test_additional_features(self) -> None:
        self._test_correction()
        self._test_task_feature_0()
        self._test_output_tasks()
        self._test_prebuilt_modules()
        self._test_partial_task_observation()

    def test_serialization(self) -> None:
        self._test_serialization()

    def test_coverage_gaps(self) -> None:
        self._test_coverage_gaps()


class TestMultiTaskShrinkage(BotorchTestCase):
    """Tests for base-kernel shrinkage wiring in the multi-task model."""

    def test_base_covar_module(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        torch.manual_seed(0)
        n_prog, n_curves, n_tasks = 8, 6, 2
        Xg = torch.linspace(0.0, 1.0, n_prog, **tkwargs).unsqueeze(-1)
        hist_Xs = [Xg for _ in range(n_tasks)]
        hist_Ys = [
            torch.sin(3.0 * Xg).squeeze(-1).expand(n_curves, n_prog)
            + 0.1 * torch.randn(n_curves, n_prog, **tkwargs)
            for _ in range(n_tasks)
        ]
        train_X = torch.cat(
            [
                torch.cat([Xg[:4], torch.zeros(4, 1, **tkwargs)], dim=-1),
                torch.cat([Xg[:2], torch.ones(2, 1, **tkwargs)], dim=-1),
            ],
            dim=0,
        )
        train_Y = torch.sin(3.0 * train_X[:, :1])
        base = ScaleKernel(
            ProductKernel(
                RBFKernel(active_dims=torch.tensor([0])),
                IndexKernel(num_tasks=n_tasks, active_dims=torch.tensor([1])),
            )
        ).to(**tkwargs)
        model = MultiTaskEmpiricalOneDimensionalGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=1,
            historical_Xs=hist_Xs,
            historical_Ys=hist_Ys,
            base_covar_module=base,
        )
        # Additive combination: empirical + base; base params fit iff requires_grad.
        self.assertIsInstance(model.covar_module, BaseAugmentedEmpiricalKernel)
        self.assertIs(model.covar_module.base_kernel, base)
        self.assertTrue(model.covar_module.base_kernel.raw_outputscale.requires_grad)
        model.eval()
        Xq = torch.cat([Xg, torch.ones(n_prog, 1, **tkwargs)], dim=-1)
        with torch.no_grad():
            post = model.posterior(Xq)
        self.assertEqual(post.mean.shape[-2], n_prog)
