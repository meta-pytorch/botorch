#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import botorch.acquisition.multi_objective.logei as logei_module
import torch
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    _try_load_fused_kernel,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models import SingleTaskGP
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior

# Trigger lazy loading so we can check availability for skipIf.
_try_load_fused_kernel()
_fused_C = logei_module._C


class DummyMultiObjectiveMCAcquisitionFunction(MultiObjectiveMCAcquisitionFunction):
    def forward(self, X):
        pass


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples, X=None):
        if X is not None:
            return samples[..., : X.shape[-2], :]
        else:
            return samples


class TestLogQExpectedHypervolumeImprovement(BotorchTestCase):
    def test_q_log_expected_hypervolume_improvement(self):
        for dtype, fat in itertools.product((torch.float, torch.double), (True, False)):
            with self.subTest(dtype=dtype, fat=fat):
                self._qLogEHVI_test(dtype, fat)

    def _qLogEHVI_test(self, dtype: torch.dtype, fat: bool):
        """NOTE: The purpose of this test is to test the numerical particularities
        of the qLogEHVI. For further tests including the non-numerical features of the
        acquisition function, please see the corresponding tests - unified with qEHVI -
        in ``multi_objective/test_monte_carlo.py``.
        """
        tkwargs = {"device": self.device, "dtype": dtype}
        ref_point = [0.0, 0.0]
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        pareto_Y = torch.tensor(
            [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
        )
        partitioning = NondominatedPartitioning(ref_point=t_ref_point)
        # the event shape is ``b x q x m`` = 1 x 1 x 2
        samples = torch.zeros(1, 1, 2, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        partitioning.update(Y=pareto_Y)

        X = torch.zeros(1, 1, **tkwargs)
        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = qLogExpectedHypervolumeImprovement(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            fat=fat,
        )
        res = acqf(X)
        exp_log_res = res.exp().item()

        # The log value is never -inf due to the smooth approximations.
        self.assertFalse(res.isinf().item())

        # Due to the smooth approximation, the value at zero should be close to, but
        # not exactly zero, and upper-bounded by the tau hyperparameter.
        if fat:
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= acqf.tau_relu)
        else:  # This is an interesting difference between the exp and the fat tail.
            # Even though the log value is never -inf, softmax's exponential tail gives
            # rise to a zero value upon the exponentiation of the log acquisition value.
            self.assertEqual(0, exp_log_res)

        # similar test for q=2
        X2 = torch.zeros(2, 1, **tkwargs)
        samples2 = torch.zeros(1, 2, 2, **tkwargs)
        mm2 = MockModel(MockPosterior(samples=samples2))
        acqf.model = mm2
        self.assertEqual(acqf.model, mm2)
        self.assertIn("model", acqf._modules)
        self.assertEqual(acqf._modules["model"], mm2)

        # see detailed comments for the tests around the first set of test above.
        res = acqf(X2)
        exp_log_res = res.exp().item()
        self.assertFalse(res.isinf().item())
        if fat:
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= acqf.tau_relu)
        else:  # This is an interesting difference between the exp and the fat tail.
            self.assertEqual(0, exp_log_res)

        X = torch.zeros(1, 1, **tkwargs)
        samples = torch.zeros(1, 1, 2, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = qLogExpectedHypervolumeImprovement(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            fat=fat,
        )
        res = acqf(X)
        # non-log EHVI is zero, but qLogEHVI is not -Inf.
        self.assertFalse(res.isinf().item())
        exp_log_res = res.exp().item()
        if fat:
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= 1e-10)  # should be *very* small
        else:  # This is an interesting difference between the exp and the fat tail.
            self.assertEqual(0, exp_log_res)

        # basic test, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = qLogExpectedHypervolumeImprovement(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            fat=fat,
        )
        res = acqf(X)
        exp_log_res = res.exp().item()
        # non-log EHVI is zero, but qLogEHVI is not -Inf.
        self.assertFalse(res.isinf().item())

        if fat:
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= 1e-10)  # should be *very* small
        else:  # This is an interesting difference between the exp and the fat tail.
            self.assertEqual(0, exp_log_res)


@unittest.skipIf(_fused_C is None, "C++ extension not available")
class TestFusedKernelCorrectness(BotorchTestCase):
    """Verify fused C++ kernel matches pure-Python path for values and gradients."""

    def test_fused_kernel_is_loaded(self) -> None:
        """Verify the pre-compiled C++ fused kernel extension is available."""
        self.assertIsNotNone(
            _fused_C,
            "Fused C++ kernel (_C) is None — the pre-compiled extension "
            "was not loaded. Check that the cpp_python_extension target "
            "'logei_fused_ext' is built and included as a dependency.",
        )
        # Verify the extension exposes the expected functions.
        self.assertTrue(hasattr(_fused_C, "forward"))
        self.assertTrue(hasattr(_fused_C, "backward"))

    def _compare_fused_vs_python(
        self,
        acqf: qLogExpectedHypervolumeImprovement,
        test_X: torch.Tensor,
        atol_val: float = 1e-6,
        atol_grad: float = 1e-6,
    ) -> None:
        """Run acqf with fused kernel and Python fallback, compare results."""
        test_X = test_X.clone().requires_grad_(True)

        # Fused C++ path
        torch.manual_seed(42)
        val_fused = acqf(test_X)
        val_fused.sum().backward()
        grad_fused = test_X.grad.clone()
        test_X.grad = None

        # Python fallback path
        saved = logei_module._C
        logei_module._C = None
        try:
            torch.manual_seed(42)
            val_python = acqf(test_X)
            val_python.sum().backward()
            grad_python = test_X.grad.clone()
            test_X.grad = None
        finally:
            logei_module._C = saved

        self.assertAllClose(val_fused, val_python, atol=atol_val, rtol=1e-6)
        self.assertAllClose(grad_fused, grad_python, atol=atol_grad, rtol=1e-6)

    def test_fused_kernel_qlogehvi(self) -> None:
        """Test fused kernel for qLogEHVI across a grid of shapes."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        d = 4

        configs = [
            # (m, q, num_pareto, mc_samples)
            (2, 1, 3, 8),
            (2, 2, 5, 16),
            (2, 4, 5, 16),
            (2, 6, 3, 8),
            (3, 2, 5, 8),
            (3, 4, 3, 8),
            (4, 2, 3, 8),
            (4, 4, 3, 8),
        ]

        for m, q, num_pareto, mc_samples in configs:
            with self.subTest(m=m, q=q, num_pareto=num_pareto, mc=mc_samples):
                torch.manual_seed(0)
                bounds = torch.stack(
                    [torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)]
                )
                train_X = (
                    draw_sobol_samples(bounds=bounds, n=num_pareto + 5, q=1)
                    .squeeze(1)
                    .to(**tkwargs)
                )
                train_Y = torch.randn(train_X.shape[0], m, **tkwargs)
                model = SingleTaskGP(train_X, train_Y)
                ref_point = train_Y.min(dim=0).values - 0.1
                partitioning = NondominatedPartitioning(
                    ref_point=ref_point, Y=train_Y[:num_pareto]
                )
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
                acqf = qLogExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point.tolist(),
                    partitioning=partitioning,
                    sampler=sampler,
                )
                test_X = torch.rand(q, d, **tkwargs)
                self._compare_fused_vs_python(acqf, test_X)

    def test_fused_kernel_qlognehvi(self) -> None:
        """Test fused kernel for qLogNEHVI (batched cell bounds)."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        d = 4

        configs = [
            # (m, q, n_train, mc_samples)
            (2, 2, 10, 8),
            (2, 4, 15, 8),
            (3, 2, 10, 8),
        ]

        for m, q, n_train, mc_samples in configs:
            with self.subTest(m=m, q=q, n_train=n_train, mc=mc_samples):
                torch.manual_seed(0)
                train_X = torch.rand(n_train, d, **tkwargs)
                train_Y = torch.randn(n_train, m, **tkwargs)
                model = SingleTaskGP(train_X, train_Y)
                ref_point = train_Y.min(dim=0).values - 0.1
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
                acqf = qLogNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point.tolist(),
                    X_baseline=train_X,
                    sampler=sampler,
                )
                test_X = torch.rand(q, d, **tkwargs)
                self._compare_fused_vs_python(acqf, test_X)

    def test_fused_kernel_tbatch_broadcasting(self) -> None:
        """Test fused kernel with t-batch dims (multi-restart optimization)."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        d = 4
        m = 2

        torch.manual_seed(0)
        train_X = torch.rand(3, 10, d, **tkwargs)  # model batch = 3
        train_Y = torch.randn(3, 10, m, **tkwargs)
        model = SingleTaskGP(train_X, train_Y)
        ref_point = train_Y.reshape(-1, m).min(dim=0).values - 0.1
        sampler = IIDNormalSampler(sample_shape=torch.Size([4]), seed=0)

        torch.manual_seed(0)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X[0],
            sampler=sampler,
            prune_baseline=False,
            cache_root=False,
        )

        for batch_shape in [
            torch.Size([]),
            torch.Size([3]),
            torch.Size([5, 3]),
        ]:
            with self.subTest(batch_shape=batch_shape):
                test_X = torch.rand(*batch_shape, 2, d, **tkwargs)
                self._compare_fused_vs_python(
                    acqf, test_X, atol_val=1e-3, atol_grad=1e-3
                )

    def test_fused_kernel_branch_coverage(self) -> None:
        """Test fused kernel with crafted inputs targeting specific C++ branches.

        Uses MockModel to inject exact objective values, bypassing the GP.
        With TAU_RELU=1e-6, safe_softplus branches at |obj - cell_lower| ≈ 2e-5:
          y > 20 (linear):   obj - cell_lower > 2e-5
          y < -20 (exp):     obj - cell_lower < -2e-5
          -20 <= y <= 20:    |obj - cell_lower| < 2e-5

        Case A (q=2, m=2): hits all three safe_softplus branches in one
          forward pass (each subset element × objective dim lands in a
          different regime) + compute_fatmin general case (n=2).
        Case B (q=1, m=2): hits compute_fatmin identity path (n=1).
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        ref_point = [0.0, 0.0]
        pareto_Y = torch.tensor([[3.0, 3.0]], **tkwargs)
        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(ref_point, **tkwargs),
            Y=pareto_Y,
        )
        # cell_lower ≈ [0, 0], cell_upper ≈ [3, 3] from the partitioning.

        # Case A: q=2, crafted to hit all safe_softplus branches.
        # elem[0]: dim0 diff=+5e-6 (y=5, middle), dim1 diff=+4.0 (y=4e6, linear)
        # elem[1]: dim0 diff=-6.0 (y=-6e6, exp), dim1 diff=+1e-5 (y=10, middle)
        # q=2 -> compute_fatmin with n=2 (general case).
        samples_a = torch.tensor(
            [[[[0.000005, 4.0], [-6.0, 0.00001]]]],
            **tkwargs,
        )  # (mc=1, batch=1, q=2, m=2)
        mm_a = MockModel(MockPosterior(samples=samples_a))
        acqf_a = qLogExpectedHypervolumeImprovement(
            model=mm_a,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=IIDNormalSampler(sample_shape=torch.Size([1])),
        )
        X_a = torch.zeros(2, 1, **tkwargs)
        # Compare forward values (no grad through MockModel).
        val_fused = acqf_a(X_a)
        saved = logei_module._C
        logei_module._C = None
        try:
            val_python = acqf_a(X_a)
        finally:
            logei_module._C = saved
        self.assertAllClose(val_fused, val_python, atol=1e-6, rtol=0)

        # Case B: q=1, objectives far above cell_lower (linear regime).
        # q=1 -> compute_fatmin with n=1 (identity path).
        samples_b = torch.tensor(
            [[[[5.0, 3.0]]]],
            **tkwargs,
        )  # (mc=1, batch=1, q=1, m=2)
        mm_b = MockModel(MockPosterior(samples=samples_b))
        acqf_b = qLogExpectedHypervolumeImprovement(
            model=mm_b,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=IIDNormalSampler(sample_shape=torch.Size([1])),
        )
        X_b = torch.zeros(1, 1, **tkwargs)
        val_fused = acqf_b(X_b)
        saved = logei_module._C
        logei_module._C = None
        try:
            val_python = acqf_b(X_b)
        finally:
            logei_module._C = saved
        self.assertAllClose(val_fused, val_python, atol=1e-6, rtol=0)
