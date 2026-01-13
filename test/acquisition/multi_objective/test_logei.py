#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


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
