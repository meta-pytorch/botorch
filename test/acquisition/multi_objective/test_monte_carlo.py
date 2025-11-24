#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from itertools import product
from math import pi
from typing import Any
from unittest import mock
from warnings import catch_warnings, simplefilter

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.cached_cholesky import _get_cache_root_not_supported_message
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputRiskMeasureMCObjective,
)
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning, NumericsWarning
from botorch.models import (
    GenericDeterministicModel,
    HigherOrderGP,
    KroneckerMultiTaskGP,
    MultiTaskGP,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputPerturbation
from botorch.posteriors.posterior_list import PosteriorList
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.low_rank import sample_cached_cholesky
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.safe_math import sigmoid
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import match_batch_shape, standardize
from torch import Tensor


def evaluate(acqf: MultiObjectiveMCAcquisitionFunction, X: Tensor) -> Tensor:
    """On a high level, this test file abstracts away the acqf_class and executes
    the respective tests for the LogEI, fat, and vanilla versions of the acquisition
    functions separately, by converting all values to the same space via `evaluate`.
    """
    return (
        acqf(X).exp()
        if isinstance(acqf, qLogExpectedHypervolumeImprovement)  # qLogNEHVI < qLogEHVI
        else acqf(X)
    )


class TestMultiObjectiveMCAcquisitionFunction(BotorchTestCase):
    def test_q_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_q_expected_hypervolume_improvement(
                        acqf_class=qExpectedHypervolumeImprovement, dtype=dtype
                    )

    # separating out LogEI test to avoid timeouts
    def test_q_log_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_q_expected_hypervolume_improvement(
                    acqf_class=qLogExpectedHypervolumeImprovement,
                    dtype=dtype,
                    # setting tau_max tighter than default to more easily unify tests
                    # with qEHVI.
                    acqf_kwargs={"tau_max": 1e-3, "fat": False},
                )

    def test_fat_q_log_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_q_expected_hypervolume_improvement(
                    acqf_class=qLogExpectedHypervolumeImprovement,
                    dtype=dtype,
                    # setting tau_max tighter than default to more easily unify tests
                    # with qEHVI.
                    acqf_kwargs={"tau_max": 1e-3, "fat": True},
                )

    def _test_q_expected_hypervolume_improvement(
        self,
        acqf_class: type[MultiObjectiveMCAcquisitionFunction],
        dtype: torch.dtype,
        acqf_kwargs: dict[str, Any] | None = None,
    ):
        if acqf_kwargs is None:
            acqf_kwargs = {}
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
        ref_point = [0.0, 0.0]
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        pareto_Y = torch.tensor(
            [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
        )
        partitioning = NondominatedPartitioning(ref_point=t_ref_point)
        # the event shape is `b x q x m` = 1 x 1 x 2
        samples = torch.zeros(1, 1, 2, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))

        # test error if there is not pareto_Y initialized in partitioning
        with self.assertRaises(BotorchError):
            acqf_class(model=mm, ref_point=ref_point, partitioning=partitioning)
        partitioning.update(Y=pareto_Y)
        # test error if ref point has wrong shape
        with self.assertRaises(ValueError):
            acqf_class(model=mm, ref_point=ref_point[:1], partitioning=partitioning)

        # test that legacy version raises a warning
        if acqf_class is qExpectedHypervolumeImprovement:
            with self.assertWarnsRegex(
                NumericsWarning, ".* qLogExpectedHypervolumeImprovement .*"
            ):
                acqf_class(model=mm, ref_point=ref_point, partitioning=partitioning)

        X = torch.zeros(1, 1, **tkwargs)
        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)
        # check ref point
        self.assertTrue(torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs)))
        # check cached indices
        self.assertTrue(hasattr(acqf, "q_subset_indices"))
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0]], device=self.device),
            )
        )

        # test q=2
        X2 = torch.zeros(2, 1, **tkwargs)
        samples2 = torch.zeros(1, 2, 2, **tkwargs)
        mm2 = MockModel(MockPosterior(samples=samples2))
        acqf.model = mm2
        self.assertEqual(acqf.model, mm2)
        self.assertIn("model", acqf._modules)
        self.assertEqual(acqf._modules["model"], mm2)
        res = evaluate(acqf, X2)
        self.assertAlmostEqual(res.item(), 0.0)
        # check cached indices
        self.assertTrue(hasattr(acqf, "q_subset_indices"))
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0], [1]], device=self.device),
            )
        )
        self.assertIn("q_choose_2", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_2"],
                torch.tensor([[0, 1]], device=self.device),
            )
        )
        self.assertNotIn("q_choose_3", acqf.q_subset_indices)
        # now back to 1 and sure all caches were cleared
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertNotIn("q_choose_2", acqf.q_subset_indices)
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0]], device=self.device),
            )
        )

        X = torch.zeros(1, 1, **tkwargs)
        samples = torch.zeros(1, 1, 2, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
        bs = acqf.sampler.base_samples.clone()
        res = evaluate(acqf, X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # basic test, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
        bs = acqf.sampler.base_samples.clone()
        evaluate(acqf, X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # basic test for X_pending and warning
        acqf.set_X_pending()
        self.assertIsNone(acqf.X_pending)
        acqf.set_X_pending(None)
        self.assertIsNone(acqf.X_pending)
        acqf.set_X_pending(X)
        self.assertEqual(acqf.X_pending, X)
        # get mm sample shape to match shape of X + X_pending
        acqf.model._posterior._samples = torch.zeros(1, 2, 2, **tkwargs)
        res = evaluate(acqf, X)
        X2 = torch.zeros(1, 1, 1, requires_grad=True, **tkwargs)
        with warnings.catch_warnings(record=True) as ws:
            acqf.set_X_pending(X2)
        self.assertEqual(acqf.X_pending, X2)
        self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)

        # test objective
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            **acqf_kwargs,
        )
        # get mm sample shape to match shape of X
        acqf.model._posterior._samples = torch.zeros(1, 1, 2, **tkwargs)
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)

        # Test that the hypervolume improvement is correct for given sample
        # test q = 1
        X = torch.zeros(1, 1, **tkwargs)
        # basic test
        samples = torch.tensor([[[6.5, 4.5]]], **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        tol = (
            1e-3  # due to smooth approximations
            if isinstance(acqf, qLogExpectedHypervolumeImprovement)
            else 1e-12
        )
        self.assertAlmostEqual(res.item(), 1.5, delta=tol)
        # test q = 1, does not contribute
        samples = torch.tensor([0.0, 1.0], **tkwargs).view(1, 1, 2)
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        mm = MockModel(MockPosterior(samples=samples))
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)

        # test q = 2, both points contribute
        X = torch.zeros(2, 1, **tkwargs)
        samples = torch.tensor([[6.5, 4.5], [7.0, 4.0]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 1.75, delta=tol)

        # test q = 2, only 1 point contributes
        samples = torch.tensor([[6.5, 4.5], [6.0, 4.0]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 1.5, delta=tol)

        # test q = 2, neither contributes
        samples = torch.tensor([[2.0, 2.0], [0.0, 0.1]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)

        # test q = 2, test point better than current best second objective
        samples = torch.tensor([[6.5, 4.5], [6.0, 6.0]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 8.0, delta=tol)

        # test q = 2, test point better than current-best first objective
        samples = torch.tensor([[6.5, 4.5], [9.0, 2.0]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        # Giving this a higher tolerance due to higher q
        self.assertAlmostEqual(res.item(), 2.0, delta=tol)
        # test q = 3, all contribute
        X = torch.zeros(3, 1, **tkwargs)
        samples = torch.tensor(
            [[6.5, 4.5], [9.0, 2.0], [7.0, 4.0]], **tkwargs
        ).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 2.25, delta=tol)
        # test q = 3, not all contribute
        samples = torch.tensor(
            [[6.5, 4.5], [9.0, 2.0], [7.0, 5.0]], **tkwargs
        ).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 3.5, delta=tol)
        # test q = 3, none contribute
        samples = torch.tensor(
            [[0.0, 4.5], [1.0, 2.0], [3.0, 0.0]], **tkwargs
        ).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)

        # test m = 3, q=1
        pareto_Y = torch.tensor(
            [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0], [1.0, 3.0, 4.0]],
            **tkwargs,
        )
        ref_point = [-1.0] * 3
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        samples = torch.tensor([[1.0, 2.0, 6.0]], **tkwargs).unsqueeze(0)
        mm = MockModel(MockPosterior(samples=samples))

        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        X = torch.zeros(1, 2, **tkwargs)
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 12.0, delta=tol)

        # change reference point
        ref_point = [0.0] * 3
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 4.0, delta=tol)

        # test m = 3, no contribution
        ref_point = [1.0] * 3
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0, delta=tol)

        # test m = 3, q = 2
        pareto_Y = torch.tensor(
            [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
        )
        samples = torch.tensor([[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs).unsqueeze(
            0
        )
        mm = MockModel(MockPosterior(samples=samples))
        ref_point = [-1.0] * 3
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        X = torch.zeros(2, 2, **tkwargs)
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 22.0, delta=22.0 * tol)

        # test batched model
        pareto_Y = torch.tensor(
            [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
        )
        samples = torch.tensor([[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs).unsqueeze(
            0
        )
        samples = torch.stack([samples, samples + 1], dim=1)
        mm = MockModel(MockPosterior(samples=samples))
        ref_point = [-1.0] * 3
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            **acqf_kwargs,
        )
        X = torch.zeros(2, 2, **tkwargs)
        res = evaluate(acqf, X)
        self.assertAllClose(
            res,
            # batch_shape x model_batch_shape
            torch.tensor([[22.0, 60.0]], **tkwargs),
            rtol=tol,
        )
        # test batched model with batched partitioning with multiple batch dims
        pareto_Y = torch.tensor(
            [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
        )
        pareto_Y = torch.stack(
            [
                pareto_Y,
                pareto_Y + 0.5,
            ],
            dim=0,
        )
        samples = torch.tensor([[6.5, 4.5], [7.0, 4.0]], **tkwargs).unsqueeze(0)
        samples = torch.stack([samples, samples + 1], dim=1)
        mm = MockModel(MockPosterior(samples=samples))
        ref_point = [-1.0] * 2
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        partitioning = FastNondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
        cell_bounds = partitioning.get_hypercell_bounds().unsqueeze(1)
        with mock.patch.object(
            partitioning, "get_hypercell_bounds", return_value=cell_bounds
        ):
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                **acqf_kwargs,
            )
            # test multiple batch dims
            self.assertEqual(acqf.cell_lower_bounds.shape, torch.Size([1, 2, 4, 2]))
            self.assertEqual(acqf.cell_upper_bounds.shape, torch.Size([1, 2, 4, 2]))
        X = torch.zeros(2, 2, **tkwargs)
        res = evaluate(acqf, X)
        self.assertAllClose(
            res,
            # batch_shape x model_batch_shape
            torch.tensor([[1.75, 3.5]], dtype=samples.dtype, device=samples.device),
            rtol=tol,
        )

    def test_constrained_q_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_constrained_q_expected_hypervolume_improvement(
                        acqf_class=qExpectedHypervolumeImprovement,
                        dtype=dtype,
                    )

    def test_constrained_q_log_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_constrained_q_expected_hypervolume_improvement(
                    acqf_class=qLogExpectedHypervolumeImprovement,
                    dtype=dtype,
                    # setting tau_max tighter than default to more easily unify tests
                    # with qEHVI.
                    acqf_kwargs={"tau_max": 1e-3, "fat": False},
                )

    def test_fat_constrained_q_log_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_constrained_q_expected_hypervolume_improvement(
                    acqf_class=qLogExpectedHypervolumeImprovement,
                    dtype=dtype,
                    # setting tau_max tighter than default to more easily unify tests
                    # with qEHVI.
                    acqf_kwargs={"tau_max": 1e-3, "fat": True},
                )

    def _test_constrained_q_expected_hypervolume_improvement(
        self,
        acqf_class: type[AcquisitionFunction],
        dtype: torch.dtype,
        acqf_kwargs: dict[str, Any] | None = None,
    ):
        if acqf_kwargs is None:
            acqf_kwargs = {}
        tkwargs = {"device": self.device, "dtype": dtype}
        ref_point = [0.0, 0.0]
        t_ref_point = torch.tensor(ref_point, **tkwargs)
        pareto_Y = torch.tensor(
            [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
        )
        partitioning = NondominatedPartitioning(ref_point=t_ref_point)
        partitioning.update(Y=pareto_Y)

        # test q=1
        # the event shape is `b x q x m` = 1 x 1 x 2
        samples = torch.tensor([[[6.5, 4.5]]], **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        X = torch.zeros(1, 1, **tkwargs)

        def get_zeros_like_last(Z: Tensor) -> Tensor:
            return torch.zeros_like(Z[..., -1])

        # test zero slack
        for eta in (1e-1, 1e-2):
            expected_values = [0.5 * 1.5, 0.5 * 0.5 * 1.5]
            for i, constraints in enumerate(
                [
                    [get_zeros_like_last],
                    [
                        get_zeros_like_last,
                        get_zeros_like_last,
                    ],
                ]
            ):
                acqf = acqf_class(
                    model=mm,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=sampler,
                    constraints=constraints,
                    eta=eta,
                    **acqf_kwargs,
                )
                res = evaluate(acqf, X)
                self.assertAlmostEqual(res.item(), expected_values[i], places=4)
        # test multiple constraints one and multiple etas
        constraints = [
            lambda Z: torch.ones_like(Z[..., -1]),
            lambda Z: torch.ones_like(Z[..., -1]),
        ]
        etas = [1, torch.tensor([1, 10])]
        fat = getattr(acqf, "fat", False)
        sigmoid_neg1 = sigmoid(torch.as_tensor(-1.0), fat=fat)
        expected_values = [
            (sigmoid_neg1**2 * 1.5).item(),
            (
                sigmoid_neg1 * sigmoid(torch.as_tensor(-1.0 / 10.0), fat=fat) * 1.5
            ).item(),
        ]
        for eta, expected_value in zip(etas, expected_values):
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                constraints=constraints,
                eta=eta,
                **acqf_kwargs,
            )
            res = evaluate(acqf, X)
            self.assertAlmostEqual(
                res.item(),
                expected_value,
                places=4,
            )
        # test feasible
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 1.5, places=4)
        # test infeasible
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            constraints=[lambda Z: 100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
            **acqf_kwargs,
        )
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0, places=4)

        # TODO: Test non-trivial constraint values, multiple constraints, and q > 1


class TestQNoisyExpectedHypervolumeImprovement(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.ref_point = [0.0, 0.0, 0.0]
        self.Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        self.pareto_Y_raw = torch.tensor(
            [
                [2.0, 0.5, 1.0],
                [1.0, 2.0, 1.0],
            ],
            device=self.device,
        )

    def test_q_noisy_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_q_noisy_expected_hypervolume_improvement_m1(
                    qNoisyExpectedHypervolumeImprovement, dtype
                )
                for m in (2, 3):
                    with self.subTest(dtype=dtype, m=m):
                        self._test_q_noisy_expected_hypervolume_improvement(
                            qNoisyExpectedHypervolumeImprovement, dtype, m
                        )

    def test_q_log_noisy_expected_hypervolume_improvement(self):
        for dtype in (torch.float, torch.double):
            self._test_q_noisy_expected_hypervolume_improvement_m1(
                qLogNoisyExpectedHypervolumeImprovement, dtype
            )
            for m in (2, 3):
                with self.subTest(dtype=dtype, m=m):
                    self._test_q_noisy_expected_hypervolume_improvement(
                        qLogNoisyExpectedHypervolumeImprovement, dtype, m
                    )

    def _test_q_noisy_expected_hypervolume_improvement_m1(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype
    ):
        # special case test for m = 1.
        (
            ref_point,
            X,
            X_baseline,
            mm,
            sampler,
            samples,
            baseline_samples,
            tkwargs,
        ) = self._setup_qnehvi_test(dtype=dtype, m=1)
        # test error is raised if m == 1
        with self.assertRaisesRegex(
            ValueError,
            "NoisyExpectedHypervolumeMixin supports m>=2 outcomes ",
        ):
            acqf_class(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                cache_root=False,
            )

    def _test_q_noisy_expected_hypervolume_improvement(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype, m: int
    ) -> None:
        self._test_qnehvi_base(acqf_class, dtype, m)
        # test with and without cached box decomposition (CBD)
        self._test_qnehvi_with_CBD(acqf_class, dtype, m)
        self._test_qnehvi_without_CBD(acqf_class, dtype, m)

    def _setup_qnehvi_test(self, dtype: torch.dtype, m: int) -> None:
        tkwargs = {"device": self.device}
        tkwargs["dtype"] = dtype
        ref_point = self.ref_point[:m]
        Y = self.Y_raw[:, :m].to(**tkwargs)
        X_baseline = torch.rand(Y.shape[0], 1, **tkwargs)
        # the event shape is `b x q + r x m` = 1 x 1 x 2
        baseline_samples = Y
        samples = torch.cat(
            [baseline_samples.unsqueeze(0), torch.zeros(1, 1, m, **tkwargs)],
            dim=1,
        )
        mm = MockModel(MockPosterior(samples=baseline_samples))
        X = torch.zeros(1, 1, **tkwargs)
        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        return ref_point, X, X_baseline, mm, sampler, samples, baseline_samples, tkwargs

    def _test_qnehvi_base(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype, m: int
    ) -> None:
        (
            ref_point,
            X,
            X_baseline,
            mm,
            sampler,
            samples,
            baseline_samples,
            tkwargs,
        ) = self._setup_qnehvi_test(dtype=dtype, m=m)

        if acqf_class is qNoisyExpectedHypervolumeImprovement:
            with self.assertWarnsRegex(
                NumericsWarning, ".* qLogNoisyExpectedHypervolumeImprovement .*"
            ):
                acqf_class(
                    model=mm,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                    cache_root=False,
                )

        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            cache_root=False,
        )
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        acqf.model._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)
        # check ref point
        self.assertTrue(torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs)))
        # check cached indices
        self.assertTrue(hasattr(acqf, "q_subset_indices"))
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0]], device=self.device),
            )
        )

        # test q=2
        X2 = torch.zeros(2, 1, **tkwargs)
        samples2 = torch.cat(
            [baseline_samples.unsqueeze(0), torch.zeros(1, 2, m, **tkwargs)],
            dim=1,
        )
        mm2 = MockModel(MockPosterior(samples=baseline_samples))
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm2,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            cache_root=False,
        )
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        acqf.model._posterior._samples = samples2
        res = evaluate(acqf, X2)
        self.assertAlmostEqual(res.item(), 0.0)
        # check cached indices
        self.assertTrue(hasattr(acqf, "q_subset_indices"))
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0], [1]], device=self.device),
            )
        )
        self.assertIn("q_choose_2", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_2"],
                torch.tensor([[0, 1]], device=self.device),
            )
        )
        self.assertNotIn("q_choose_3", acqf.q_subset_indices)
        # now back to 1 and sure all caches were cleared
        acqf.model = mm
        res = evaluate(acqf, X)
        self.assertNotIn("q_choose_2", acqf.q_subset_indices)
        self.assertIn("q_choose_1", acqf.q_subset_indices)
        self.assertTrue(
            torch.equal(
                acqf.q_subset_indices["q_choose_1"],
                torch.tensor([[0]], device=self.device),
            )
        )

        # test error is raised if X_baseline is batched
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        with self.assertRaises(UnsupportedError):
            acqf_class(
                model=mm2,
                ref_point=ref_point,
                X_baseline=X_baseline.unsqueeze(0),
                sampler=sampler,
                cache_root=False,
            )

        # test objective
        # set the MockPosterior to use samples over baseline points
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            cache_root=False,
        )
        # sample_shape x n x m
        original_base_samples = sampler.base_samples.detach().clone()
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0)
        # test that original base samples were retained
        self.assertTrue(
            torch.equal(
                # sample_shape x batch_shape x n x m
                sampler.base_samples[0, 0, : original_base_samples.shape[1], :],
                original_base_samples[0],
            )
        )

        # test that base_samples for X_baseline are fixed
        # set the MockPosterior to use samples over baseline points
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            cache_root=False,
        )
        orig_base_sampler = deepcopy(acqf.base_sampler)
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        mm._posterior._samples = samples
        with torch.no_grad():
            acqf(X)
        self.assertTrue(
            torch.equal(orig_base_sampler.base_samples, acqf.base_sampler.base_samples)
        )
        self.assertTrue(
            torch.allclose(
                acqf.base_sampler.base_samples,
                acqf.sampler.base_samples[..., : X_baseline.shape[0], :],
            )
        )
        mm._posterior._samples = baseline_samples
        # test empty pareto set
        ref_point2 = [15.0, 14.0, 16.0][:m]
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point2,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            cache_root=False,
        )
        self.assertTrue((acqf.cell_lower_bounds[..., 0] == 15).all())
        self.assertTrue((acqf.cell_lower_bounds[..., 1] == 14).all())
        if m == 3:
            self.assertTrue((acqf.cell_lower_bounds[..., 2] == 16).all())
        self.assertTrue(torch.isinf(acqf.cell_upper_bounds).all())
        for b in (acqf.cell_lower_bounds, acqf.cell_upper_bounds):
            self.assertEqual(list(b.shape), [1, 1, m])
            self.assertEqual(list(b.shape), [1, 1, m])

    def _test_qnehvi_with_CBD(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype, m: int
    ) -> None:
        (
            ref_point,
            X,
            X_baseline,
            mm,
            sampler,
            samples,
            baseline_samples,
            tkwargs,
        ) = self._setup_qnehvi_test(dtype=dtype, m=m)
        pareto_Y = self.pareto_Y_raw[:, :m].to(**tkwargs)

        # test no baseline points
        ref_point2 = [15.0, 14.0, 16.0][:m]
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point2,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            prune_baseline=True,
            cache_root=False,
        )
        self.assertTrue((acqf.cell_lower_bounds[..., 0] == 15).all())
        self.assertTrue((acqf.cell_lower_bounds[..., 1] == 14).all())
        if m == 3:
            self.assertTrue((acqf.cell_lower_bounds[..., 2] == 16).all())
        self.assertTrue(torch.isinf(acqf.cell_upper_bounds).all())
        for b in (acqf.cell_lower_bounds, acqf.cell_upper_bounds):
            self.assertEqual(list(b.shape), [1, 1, m])
            self.assertEqual(list(b.shape), [1, 1, m])

        # test X_pending with CBD
        for incremental_nehvi in (False, True):
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                incremental_nehvi=incremental_nehvi,
                cache_root=False,
            )
            original_base_samples = sampler.base_samples.detach().clone()
            # the box decomposition algorithm is faster on the CPU for m>2,
            # so NEHVI runs it on the CPU
            expected_pareto_Y = pareto_Y if m == 2 else pareto_Y.cpu()
            self.assertTrue(
                torch.equal(acqf.partitioning.pareto_Y[0], expected_pareto_Y)
            )
            self.assertIsNone(acqf.X_pending)
            new_Y = torch.tensor([[0.5, 3.0, 0.5][:m]], dtype=dtype, device=self.device)
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y,
                ]
            ).unsqueeze(0)
            bd = DominatedPartitioning(
                ref_point=torch.tensor(ref_point).to(**tkwargs), Y=pareto_Y
            )
            initial_hv = bd.compute_hypervolume()
            # test _initial_hvs
            if not incremental_nehvi:
                self.assertTrue(hasattr(acqf, "_initial_hvs"))
                # test that _initial_hvs has the correct shape
                self.assertEqual(acqf._initial_hvs.shape, acqf._batch_sample_shape)
                # test that _initial_hvs contains the correct hypervolume values
                self.assertTrue(torch.equal(acqf._initial_hvs, initial_hv.view(-1)))
            # test forward
            X_test = torch.rand(1, 1, dtype=dtype, device=self.device)
            with torch.no_grad():
                val = evaluate(acqf, X_test)
            bd.update(mm._posterior._samples[0, -1:])
            expected_val = bd.compute_hypervolume() - initial_hv
            self.assertAllClose(val, expected_val.view(-1))
            # test that original base_samples were retained
            self.assertTrue(
                torch.equal(
                    # sample_shape x batch_shape x n x m
                    sampler.base_samples[0, 0, : original_base_samples.shape[1], :],
                    original_base_samples[0],
                )
            )
            # test X_pending
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
                incremental_nehvi=incremental_nehvi,
                cache_root=False,
            )
            # sample_shape x n x m
            original_base_samples = sampler.base_samples.detach().clone()
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y,
                ],
                dim=0,
            )
            X_pending = torch.rand(1, 1, dtype=dtype, device=self.device)
            acqf.set_X_pending(X_pending)
            if not incremental_nehvi:
                self.assertAllClose(expected_val, acqf._prev_nehvi)
            self.assertTrue(torch.all(acqf.X_pending == X_pending))
            # check that X_baseline has been updated
            self.assertTrue(torch.equal(acqf.X_baseline[:-1], acqf._X_baseline))
            self.assertTrue(torch.equal(acqf.X_baseline[-1:], X_pending))
            # check that partitioning has been updated
            acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
            # the box decomposition algorithm is faster on the CPU for m>2,
            # so NEHVI runs it on the CPU
            self.assertTrue(torch.equal(acqf_pareto_Y[:-1], expected_pareto_Y))
            expected_new_Y = new_Y if m == 2 else new_Y.cpu()
            self.assertTrue(torch.equal(acqf_pareto_Y[-1:], expected_new_Y))
            # test that base samples were retained
            self.assertTrue(
                torch.equal(
                    # sample_shape x n x m
                    sampler.base_samples[0, : original_base_samples.shape[1], :],
                    original_base_samples[0],
                )
            )
            self.assertTrue(
                torch.equal(
                    acqf.sampler.base_samples,
                    acqf.base_sampler.base_samples,
                )
            )

            # test incremental nehvi in forward
            new_Y2 = torch.cat(
                [
                    new_Y,
                    torch.tensor(
                        [[0.25, 9.5, 1.5][:m]], dtype=dtype, device=self.device
                    ),
                ],
                dim=0,
            )
            mm._posterior._samples = torch.cat(
                [
                    baseline_samples,
                    new_Y2,
                ]
            ).unsqueeze(0)
            X_test = torch.rand(1, 1, dtype=dtype, device=self.device)
            with torch.no_grad():
                val = evaluate(acqf, X_test)
            if incremental_nehvi:
                # set initial hv to include X_pending
                initial_hv = bd.compute_hypervolume()
            bd.update(mm._posterior._samples[0, -1:])
            expected_val = bd.compute_hypervolume() - initial_hv
            self.assertAllClose(val, expected_val.view(-1))

        # add another point
        X_pending2 = torch.cat(
            [X_pending, torch.rand(1, 1, dtype=dtype, device=self.device)], dim=0
        )
        mm._posterior._samples = mm._posterior._samples.squeeze(0)
        acqf.set_X_pending(X_pending2)
        self.assertTrue(torch.all(acqf.X_pending == X_pending2))
        # check that X_baseline has been updated
        self.assertTrue(torch.equal(acqf.X_baseline[:-2], acqf._X_baseline))
        self.assertTrue(torch.equal(acqf.X_baseline[-2:], X_pending2))
        # check that partitioning has been updated
        acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
        self.assertTrue(torch.equal(acqf_pareto_Y[:-2], expected_pareto_Y))
        expected_new_Y2 = new_Y2 if m == 2 else new_Y2.cpu()
        self.assertTrue(torch.equal(acqf_pareto_Y[-2:], expected_new_Y2))

        # test set X_pending with grad
        # Get posterior samples to agree with X_pending
        mm._posterior._samples = torch.zeros(1, 7, m, **tkwargs)
        with warnings.catch_warnings(record=True) as ws:
            acqf.set_X_pending(
                torch.cat([X_pending2, X_pending2], dim=0).requires_grad_(True)
            )
        self.assertTrue(
            torch.all(acqf.X_pending == torch.cat([X_pending2, X_pending2], dim=0))
        )
        self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)

        # test max iep
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            incremental_nehvi=False,
            max_iep=1,
            cache_root=False,
        )
        mm._posterior._samples = torch.cat(
            [
                baseline_samples,
                new_Y,
            ]
        )
        acqf.set_X_pending(X_pending)
        self.assertTrue(torch.equal(acqf.X_pending, X_pending))
        acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
        self.assertTrue(torch.equal(acqf_pareto_Y, expected_pareto_Y))
        mm._posterior._samples = torch.cat(
            [
                baseline_samples,
                new_Y2,
            ]
        )
        # check that after second pending point is added, X_pending still includes
        # pending points, and the pending points are included in the box decompositions
        acqf.set_X_pending(X_pending2)
        self.assertTrue(torch.all(acqf.X_pending == X_pending2))
        acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
        self.assertTrue(torch.equal(acqf_pareto_Y[:-2], expected_pareto_Y))
        self.assertTrue(torch.equal(acqf_pareto_Y[-2:], expected_new_Y2))

    def _test_qnehvi_without_CBD(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype, m: int
    ) -> None:
        tkwargs = {"device": self.device}
        tkwargs["dtype"] = dtype
        ref_point = self.ref_point[:m]
        Y = self.Y_raw[:, :m].to(**tkwargs)
        pareto_Y = self.pareto_Y_raw[:, :m].to(**tkwargs)
        X_baseline = torch.rand(Y.shape[0], 1, **tkwargs)
        # the event shape is `b x q + r x m` = 1 x 1 x 2
        baseline_samples = Y
        mm = MockModel(MockPosterior(samples=baseline_samples))

        X_pending = torch.rand(1, 1, dtype=dtype, device=self.device)

        # test qNEHVI without CBD
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            cache_pending=False,
            cache_root=False,
        )
        new_Y = torch.tensor([[0.5, 3.0, 0.5][:m]], dtype=dtype, device=self.device)
        mm._posterior._samples = torch.cat(
            [
                baseline_samples,
                new_Y,
            ]
        ).unsqueeze(0)
        X_pending10 = X_pending.expand(10, 1)
        acqf.set_X_pending(X_pending10)
        self.assertTrue(torch.equal(acqf.X_pending, X_pending10))
        acqf_pareto_Y = acqf.partitioning.pareto_Y[0]
        expected_pareto_Y = pareto_Y if m == 2 else pareto_Y.cpu()
        self.assertTrue(torch.equal(acqf_pareto_Y, expected_pareto_Y))
        acqf.set_X_pending(X_pending)
        # test incremental nehvi in forward
        new_Y2 = torch.cat(
            [
                new_Y,
                torch.tensor([[0.25, 9.5, 1.5][:m]], dtype=dtype, device=self.device),
            ],
            dim=0,
        )
        mm._posterior._samples = torch.cat(
            [
                baseline_samples,
                new_Y2,
            ]
        ).unsqueeze(0)
        with torch.no_grad():
            X_test = torch.rand(1, 1, dtype=dtype, device=self.device)
            val = evaluate(acqf, X_test)
        bd = DominatedPartitioning(
            ref_point=torch.tensor(ref_point).to(**tkwargs), Y=pareto_Y
        )
        initial_hv = bd.compute_hypervolume()
        bd.update(mm._posterior._samples.squeeze(0))
        expected_val = bd.compute_hypervolume() - initial_hv
        self.assertAllClose(expected_val.view(1), val)
        # test alpha > 0
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            cache_pending=False,
            alpha=1e-3,
            cache_root=False,
        )
        if len(ref_point) == 2:
            partitioning = acqf.partitioning
        else:
            partitioning = acqf.partitioning.box_decompositions[0]
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, 1e-3)
        # test set_X_pending when X_pending = None
        acqf.set_X_pending(X_pending10)
        self.assertTrue(torch.equal(acqf.X_pending, X_pending10))
        acqf.set_X_pending(None)
        self.assertIsNone(acqf.X_pending)
        # test X_pending is not None on __init__
        mm._posterior._samples = torch.zeros(1, 5, m, **tkwargs)
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        # add another point
        X_pending2 = torch.cat(
            [X_pending, torch.rand(1, 1, dtype=dtype, device=self.device)], dim=0
        )
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
            alpha=1e-3,
            X_pending=X_pending2,
            cache_root=False,
        )
        self.assertTrue(torch.equal(X_baseline, acqf._X_baseline))
        self.assertTrue(torch.equal(acqf.X_baseline[:-2], acqf._X_baseline))
        self.assertTrue(torch.equal(acqf.X_baseline[-2:], X_pending2))

    def test_constrained_q_noisy_expected_hypervolume_improvement(self) -> None:
        for dtype, fat in product(
            (torch.float, torch.double),
            (True, False),
        ):
            with self.subTest(dtype=dtype, fat=fat):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_constrained_q_noisy_expected_hypervolume_improvement(
                        qNoisyExpectedHypervolumeImprovement, dtype, fat
                    )

    def test_constrained_q_log_noisy_expected_hypervolume_improvement(self) -> None:
        for dtype, fat in product(
            (torch.float, torch.double),
            (False, True),
        ):
            with self.subTest(dtype=dtype, fat=fat):
                self._test_constrained_q_noisy_expected_hypervolume_improvement(
                    qLogNoisyExpectedHypervolumeImprovement, dtype, fat
                )

    def _test_constrained_q_noisy_expected_hypervolume_improvement(
        self, acqf_class: type[AcquisitionFunction], dtype: torch.dtype, fat: bool
    ):
        # TODO: improve tests with constraints
        tkwargs = {"device": self.device, "dtype": dtype}
        ref_point = [0.0, 0.0]
        pareto_Y = torch.tensor(
            [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
        )
        X_baseline = torch.zeros(pareto_Y.shape[0], 1, **tkwargs)
        baseline_samples = pareto_Y

        # test q=1
        # the event shape is `b x q x m` = 1 x 1 x 2
        samples = torch.cat(
            [
                baseline_samples.unsqueeze(0),
                torch.tensor([[[6.5, 4.5]]], **tkwargs),
            ],
            dim=1,
        )
        mm = MockModel(MockPosterior(samples=baseline_samples))
        X = torch.zeros(1, 1, **tkwargs)
        # test zero slack multiple constraints, multiple etas
        for eta in [1e-1, 1e-2, torch.tensor([1.0, 10.0])]:
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                constraints=[
                    lambda Z: torch.zeros_like(Z[..., -1]),
                    lambda Z: torch.zeros_like(Z[..., -1]),
                ],
                eta=eta,
                cache_root=False,
                fat=fat,
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            mm._posterior._samples = samples
            res = evaluate(acqf, X)
            self.assertAlmostEqual(res.item(), 0.5 * 0.5 * 1.5, places=4)
        # test zero slack single constraint
        for eta in (1e-1, 1e-2):
            # set the MockPosterior to use samples over baseline points
            mm._posterior._samples = baseline_samples
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = acqf_class(
                model=mm,
                ref_point=ref_point,
                X_baseline=X_baseline,
                sampler=sampler,
                constraints=[lambda Z: torch.zeros_like(Z[..., -1])],
                eta=eta,
                cache_root=False,
                fat=fat,
            )
            # set the MockPosterior to use samples over baseline points and new
            # candidates
            mm._posterior._samples = samples
            res = evaluate(acqf, X)
            self.assertAlmostEqual(res.item(), 0.5 * 1.5, places=4)
        # set X_pending
        X_pending = torch.rand(1, 1, **tkwargs)
        acqf.set_X_pending(X_pending)
        samples = torch.cat(
            [
                samples,
                torch.tensor([[[10.0, 0.5]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.5 * 0.5, places=4)

        # test incremental nehvi=False
        mm._posterior._samples = baseline_samples
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=[lambda Z: torch.zeros_like(Z[..., -1])],
            eta=1e-3,
            incremental_nehvi=False,
            cache_root=False,
            fat=fat,
        )
        samples = torch.cat(
            [
                baseline_samples.unsqueeze(0),
                torch.tensor([[[6.5, 4.5]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.5 * 1.5, places=4)
        acqf.set_X_pending(X_pending)
        samples = torch.cat(
            [
                samples,
                torch.tensor([[[10.0, 0.5]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        # test that HVI is not incremental
        # Note that the cached pending point uses strict constraint evaluation
        # so the HVI from the cached pending point is 1.5.
        # The new X contributes an HVI of 0.5, but with a constraint slack of 0,
        # the sigmoid soft-evaluation yields a constrained HVI of 0.25
        self.assertAlmostEqual(res.item(), 1.75, places=4)

        # test feasible
        # set the MockPosterior to use samples over baseline points
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
            cache_root=False,
            fat=fat,
        )
        samples = torch.cat(
            [
                baseline_samples.unsqueeze(0),
                torch.tensor([[[6.5, 4.5]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 1.5, places=4)
        # test multiple constraints one eta with
        # this crashes for large etas, and I do not why
        # set the MockPosterior to use samples over baseline points
        etas = [torch.tensor([1.0]), torch.tensor([1.0, 10.0])]
        constraints = [
            [lambda Z: torch.ones_like(Z[..., -1])],
            [
                lambda Z: torch.ones_like(Z[..., -1]),
                lambda Z: torch.ones_like(Z[..., -1]),
            ],
        ]
        expected_values = [
            (sigmoid(torch.as_tensor(-1.0 / 1), fat=fat) * 1.5).item(),
            (
                sigmoid(torch.as_tensor(-1.0 / 1), fat=fat)
                * sigmoid(torch.as_tensor(-1.0 / 10), fat=fat)
                * 1.5
            ).item(),
        ]
        for eta, constraint, expected_value in zip(etas, constraints, expected_values):
            acqf.constraints = constraint
            acqf.eta = eta
            res = evaluate(acqf, X)

            self.assertAlmostEqual(
                res.item(),
                expected_value,
                places=4,
            )
        # test infeasible
        # set the MockPosterior to use samples over baseline points
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=[lambda Z: 100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
            cache_root=False,
            fat=fat,
        )
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 0.0, places=4)

        # test >2 objectives
        ref_point = [0.0, 0.0, 0.0]
        baseline_samples = torch.tensor(
            [
                [4.0, 5.0, 1.0],
                [5.0, 5.0, 1.0],
                [8.5, 3.5, 1.0],
                [8.5, 3.0, 1.0],
                [9.0, 1.0, 1.0],
            ],
            **tkwargs,
        )
        mm._posterior._samples = baseline_samples
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
        delta = 1e-4
        special_kwargs = (
            {}
            if acqf_class == qNoisyExpectedHypervolumeImprovement
            else {"tau_max": delta}
        )
        acqf = acqf_class(
            model=mm,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])],
            eta=1e-3,
            cache_root=False,
            fat=fat,
            **special_kwargs,
        )
        # set the MockPosterior to use samples over baseline points and new
        # candidates
        samples = torch.cat(
            [
                baseline_samples.unsqueeze(0),
                torch.tensor([[[6.5, 4.5, 1.0]]], **tkwargs),
            ],
            dim=1,
        )
        mm._posterior._samples = samples
        res = evaluate(acqf, X)
        self.assertAlmostEqual(res.item(), 1.5, delta=delta)

    def test_prune_baseline(self):
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_prune_baseline(acqf_class)

    def _test_prune_baseline(self, acqf_class: type[AcquisitionFunction]):
        # test prune_baseline
        no = "botorch.utils.testing.MockModel.num_outputs"
        prune = (
            "botorch.utils.multi_objective.hypervolume."
            "prune_inferior_points_multi_objective"
        )
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            ref_point = [0.0, 0.0]
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            X_baseline = torch.zeros(pareto_Y.shape[0], 1, **tkwargs)
            baseline_samples = pareto_Y
            X_pruned = torch.rand(1, 1, device=self.device, dtype=dtype)
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                # Reduce samples to same shape as X_pruned.
                mm = MockModel(MockPosterior(samples=baseline_samples[:1]))
                with mock.patch(prune, return_value=X_pruned) as mock_prune:
                    acqf = acqf_class(
                        model=mm,
                        ref_point=ref_point,
                        X_baseline=X_baseline,
                        sampler=sampler,
                        prune_baseline=True,
                        cache_root=False,
                    )
                mock_prune.assert_called_once()
                self.assertTrue(torch.equal(acqf.X_baseline, X_pruned))

    def test_cache_root(self):
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_cache_root(acqf_class)

    def _test_cache_root(self, acqf_class: type[AcquisitionFunction]):
        sample_cached_path = (
            "botorch.acquisition.cached_cholesky.sample_cached_cholesky"
        )
        state_dict = {
            "likelihood.noise_covar.raw_noise": torch.tensor(
                [[0.0895], [0.2594]], dtype=torch.float64
            ),
            "mean_module.raw_constant": torch.tensor(
                [-0.4545, -0.1285], dtype=torch.float64
            ),
            "covar_module.raw_outputscale": torch.tensor(
                [1.4876, 1.4897], dtype=torch.float64
            ),
            "covar_module.base_kernel.raw_lengthscale": torch.tensor(
                [[[-0.7202, -0.2868]], [[-0.8794, -1.2877]]], dtype=torch.float64
            ),
        }

        # test batched models (e.g. for MCMC)
        for train_batch_shape in (torch.Size([]), torch.Size([3])):
            if len(train_batch_shape) > 0:
                for k, v in state_dict.items():
                    state_dict[k] = v.unsqueeze(0).expand(*train_batch_shape, *v.shape)
            for dtype, ref_point in product(
                (torch.float, torch.double),
                ([-5.0, -5.0], [10.0, 10.0]),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                for k, v in state_dict.items():
                    state_dict[k] = v.to(**tkwargs)
                all_close_kwargs = (
                    {"atol": 1e-1, "rtol": 1e-2}
                    if dtype == torch.float
                    else {"atol": 1e-4, "rtol": 1e-6}
                )
                torch.manual_seed(1234)
                train_X = torch.rand(*train_batch_shape, 3, 2, **tkwargs)
                train_Y = torch.sin(train_X * 2 * pi) + torch.randn(
                    *train_batch_shape, 3, 2, **tkwargs
                )
                train_Y = standardize(train_Y)
                model = SingleTaskGP(train_X, train_Y)
                if len(train_batch_shape) > 0:
                    X_baseline = train_X[0]
                else:
                    X_baseline = train_X

                model.load_state_dict(state_dict, strict=False)
                sampler = IIDNormalSampler(sample_shape=torch.Size([5]), seed=0)
                torch.manual_seed(0)
                acqf = acqf_class(
                    model=model,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler,
                    prune_baseline=False,
                    cache_root=True,
                )

                sampler2 = IIDNormalSampler(sample_shape=torch.Size([5]), seed=0)
                torch.manual_seed(0)
                acqf_no_cache = acqf_class(
                    model=model,
                    ref_point=ref_point,
                    X_baseline=X_baseline,
                    sampler=sampler2,
                    prune_baseline=False,
                    cache_root=False,
                )
                # load CBD
                acqf_no_cache.cell_lower_bounds = acqf.cell_lower_bounds.clone()
                acqf_no_cache.cell_upper_bounds = acqf.cell_upper_bounds.clone()
                for q, batch_shape in product(
                    (1, 3), (torch.Size([]), torch.Size([3]), torch.Size([4, 3]))
                ):
                    torch.manual_seed(0)
                    acqf.q_in = -1
                    test_X = (
                        0.3 + 0.05 * torch.randn(*batch_shape, q, 2, **tkwargs)
                    ).requires_grad_(True)
                    with mock.patch(
                        sample_cached_path, wraps=sample_cached_cholesky
                    ) as mock_sample_cached:
                        torch.manual_seed(0)
                        val = evaluate(acqf, test_X)
                        mock_sample_cached.assert_called_once()
                    val.sum().backward()
                    base_samples = acqf.sampler.base_samples.detach().clone()
                    X_grad = test_X.grad.clone()
                    test_X2 = test_X.detach().clone().requires_grad_(True)
                    acqf_no_cache.sampler.base_samples = base_samples
                    with mock.patch(
                        sample_cached_path, wraps=sample_cached_cholesky
                    ) as mock_sample_cached:
                        torch.manual_seed(0)
                        val2 = evaluate(acqf_no_cache, test_X2)
                    mock_sample_cached.assert_not_called()
                    self.assertAllClose(val, val2, **all_close_kwargs)
                    val2.sum().backward()
                    if dtype == torch.double:
                        # The gradient computation is very unstable in single precision
                        # so we only check the gradient when using torch.double.
                        self.assertTrue(
                            torch.allclose(X_grad, test_X2.grad, **all_close_kwargs)
                        )
                    if ref_point == [-5.0, -5.0]:
                        self.assertTrue((X_grad != 0).any())
                # test we fall back to standard sampling for
                # ill-conditioned covariances
                acqf._baseline_L = torch.zeros_like(acqf._baseline_L)
                with warnings.catch_warnings(record=True) as ws, torch.no_grad():
                    evaluate(acqf, test_X)
                self.assertEqual(
                    sum(issubclass(w.category, BotorchWarning) for w in ws), 1
                )

    def test_cache_root_w_standardize(self):
        # Test caching with standardize transform.
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                train_x = torch.rand(3, 2, dtype=torch.float64)
                train_y = torch.randn(3, 2, dtype=torch.float64)
                model = SingleTaskGP(train_x, train_y)
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    acqf = acqf_class(
                        model=model,
                        X_baseline=train_x,
                        ref_point=torch.ones(2),
                        sampler=IIDNormalSampler(sample_shape=torch.Size([1])),
                        cache_root=True,
                    )
                self.assertIsNotNone(acqf._baseline_L)
                self.assertEqual(acqf(train_x[:1]).shape, torch.Size([1]))
                self.assertEqual(acqf(train_x.unsqueeze(-2)).shape, torch.Size([3]))

    def test_with_set_valued_objectives(self):
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_with_set_valued_objectives(acqf_class)

    def _test_with_set_valued_objectives(
        self, acqf_class: type[AcquisitionFunction]
    ) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            tx = torch.rand(5, 2, **tkwargs)
            ty = torch.randn(5, 2, **tkwargs)
            perturbation = InputPerturbation(
                perturbation_set=torch.randn(3, 2, **tkwargs)
            ).eval()
            baseline_samples = perturbation(ty)

            class DummyObjective(MultiOutputRiskMeasureMCObjective):
                r"""A dummy set valued objective."""

                _verify_output_shape = False

                def forward(self, samples, X=None):
                    samples = self._prepare_samples(samples)
                    return samples[..., :2, :].reshape(
                        *samples.shape[:-3], -1, samples.shape[-1]
                    )

            model = MockModel(MockPosterior(samples=baseline_samples))
            acqf = acqf_class(
                model=model,
                ref_point=torch.tensor([0.0, 0.0], **tkwargs),
                X_baseline=tx,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([2])),
                objective=DummyObjective(n_w=3),
                prune_baseline=False,
                cache_root=False,
            )
            test_x = torch.rand(3, 2, 2, **tkwargs)
            samples = torch.cat(
                [baseline_samples.expand(3, -1, -1), torch.zeros(3, 6, 2, **tkwargs)],
                dim=1,
            )
            acqf.model._posterior._samples = samples
            res = acqf(test_x)
            if acqf_class == qNoisyExpectedHypervolumeImprovement:
                self.assertTrue(torch.equal(res, torch.zeros(3, **tkwargs)))
            else:
                self.assertFalse(res.isinf().any())
                exp_log_res = res.exp()
                self.assertTrue((0 <= exp_log_res).all())
                self.assertTrue((exp_log_res < acqf.tau_relu).all())

            self.assertEqual(acqf.q_in, 6)
            self.assertEqual(acqf.q_out, 4)
            self.assertEqual(len(acqf.q_subset_indices.keys()), 4)

    def test_deterministic(self):
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_deterministic(acqf_class)

    def _test_deterministic(self, acqf_class: type[AcquisitionFunction]):
        for dtype, prune in ((torch.float, False), (torch.double, True)):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = GenericDeterministicModel(f=lambda x: x, num_outputs=2)
            with self.assertWarnsRegex(
                RuntimeWarning,
                _get_cache_root_not_supported_message(GenericDeterministicModel),
            ):
                acqf = acqf_class(
                    model=model,
                    ref_point=torch.tensor([0.0, 0.0], **tkwargs),
                    X_baseline=torch.rand(5, 2, **tkwargs),
                    prune_baseline=prune,
                    cache_root=True,
                )
            self.assertFalse(acqf._cache_root)
            self.assertEqual(
                evaluate(acqf, torch.rand(3, 2, 2, **tkwargs)).shape, torch.Size([3])
            )

    def test_with_multitask(self):
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    self._test_with_multitask(acqf_class)

    def _test_with_multitask(self, acqf_class: type[AcquisitionFunction]):
        # Verify that _set_sampler works with MTGP, KroneckerMTGP and HOGP.
        torch.manual_seed(1234)
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        train_x = torch.rand(6, 2, **tkwargs)
        train_y = torch.randn(6, 2, **tkwargs)
        mtgp_task = torch.cat(
            [torch.zeros(6, 1, **tkwargs), torch.ones(6, 1, **tkwargs)], dim=0
        )
        mtgp_x = torch.cat([train_x.repeat(2, 1), mtgp_task], dim=-1)
        mtgp = MultiTaskGP(mtgp_x, train_y.view(-1, 1), task_feature=2).eval()
        kmtgp = KroneckerMultiTaskGP(train_x, train_y).eval()
        hogp = HigherOrderGP(train_x, train_y.repeat(6, 1, 1)).eval()
        hogp_obj = GenericMCMultiOutputObjective(lambda Y, X: Y.mean(dim=-2))
        test_x = torch.rand(2, 3, 2, **tkwargs)

        def get_acqf(model):
            return acqf_class(
                model=model,
                ref_point=torch.tensor([-1.0, -1.0], **tkwargs),
                X_baseline=train_x,
                sampler=IIDNormalSampler(sample_shape=torch.Size([2])),
                objective=hogp_obj if isinstance(model, HigherOrderGP) else None,
                prune_baseline=True,
                cache_root=False,
            )

        compute_str = (
            "_compute_qehvi"
            if acqf_class == qNoisyExpectedHypervolumeImprovement
            else "_compute_log_qehvi"
        )
        for model in [mtgp, kmtgp, hogp]:
            acqf = get_acqf(model)
            posterior = model.posterior(acqf.X_baseline)
            base_evals = acqf.base_sampler(posterior)
            base_samples = acqf.base_sampler.base_samples
            with mock.patch.object(
                acqf_class,
                compute_str,
                wraps=getattr(acqf, compute_str),
            ) as wrapped_compute:
                acqf(test_x)
            wrapped_compute.assert_called_once()
            expected_shape = (
                torch.Size([2, 2, 3, 6, 2])
                if isinstance(model, HigherOrderGP)
                else torch.Size([2, 2, 3, 2])
            )
            self.assertEqual(
                wrapped_compute.call_args[-1]["samples"].shape, expected_shape
            )
            new_base_samples = acqf.sampler.base_samples
            # Check that the base samples are the same.
            if model is mtgp:
                expected = new_base_samples[..., :-3, :].squeeze(-3)
            else:
                n_train = base_samples.shape[-1] // 2
                expected = torch.cat(
                    [new_base_samples[..., :n_train], new_base_samples[..., -n_train:]],
                    dim=-1,
                ).squeeze(-2)
            self.assertTrue(torch.equal(base_samples, expected))
            # Check that they produce the same f_X for baseline points.
            X_full = torch.cat(
                [match_batch_shape(acqf.X_baseline, test_x), test_x], dim=-2
            )
            posterior = acqf.model.posterior(X_full)
            samples = acqf.sampler(posterior)
            expected = samples[:, :, :-3]
            repeat_shape = [1, 2, 1, 1]
            if model is hogp:
                repeat_shape.append(1)
            self.assertTrue(
                torch.allclose(
                    base_evals.unsqueeze(1).repeat(*repeat_shape),
                    expected,
                    atol=1e-2,
                    rtol=1e-4,
                )
            )

    def test_with_transformed(self):
        # Verify that _set_sampler works with transformed posteriors.
        mm = MockModel(
            posterior=PosteriorList(
                TransformedPosterior(
                    MockPosterior(samples=torch.rand(2, 3, 1)), lambda X: X
                ),
                TransformedPosterior(
                    MockPosterior(samples=torch.rand(2, 3, 1)), lambda X: X
                ),
            )
        )
        sampler = ListSampler(
            IIDNormalSampler(sample_shape=torch.Size([2])),
            IIDNormalSampler(sample_shape=torch.Size([2])),
        )
        for acqf_class in (
            qNoisyExpectedHypervolumeImprovement,
            qLogNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf_class.__name__):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    # This calls _set_sampler which used to error out in
                    # NormalMCSampler._update_base_samples with TransformedPosterior
                    # due to the missing batch_shape (fixed in #1625).
                    acqf_class(
                        model=mm,
                        ref_point=torch.tensor([0.0, 0.0]),
                        X_baseline=torch.rand(3, 2),
                        sampler=sampler,
                        cache_root=False,
                    )
