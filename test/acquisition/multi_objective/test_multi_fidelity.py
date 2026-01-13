#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import mock

import torch
from botorch.acquisition.multi_objective.multi_fidelity import MOMF
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.exceptions.errors import BotorchError
from botorch.exceptions.warnings import BotorchWarning
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestMOMF(BotorchTestCase):
    def test_momf(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            ref_point = [0.0, 0.0]
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            partitioning = NondominatedPartitioning(ref_point=t_ref_point)
            # the event shape is ``b x q x m`` = 1 x 1 x 2
            samples = torch.zeros(1, 1, 2, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # test error if there is not pareto_Y initialized in partitioning
            with self.assertRaises(BotorchError):
                MOMF(model=mm, ref_point=ref_point, partitioning=partitioning)
            partitioning.update(Y=pareto_Y)
            # test error if ref point has wrong shape
            with self.assertRaises(ValueError):
                MOMF(model=mm, ref_point=ref_point[:1], partitioning=partitioning)

            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # check ref point
            self.assertTrue(
                torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs))
            )
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
            res = acqf(X2)
            self.assertEqual(res.item(), 0.0)
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
            res = acqf(X)
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
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
            bs = acqf.sampler.base_samples.clone()
            res = acqf(X)
            self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

            # basic test, qmc
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2]))
            bs = acqf.sampler.base_samples.clone()
            acqf(X)
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
            res = acqf(X)
            X2 = torch.zeros(1, 1, 1, requires_grad=True, **tkwargs)
            with warnings.catch_warnings(record=True) as ws:
                acqf.set_X_pending(X2)
            self.assertEqual(acqf.X_pending, X2)
            self.assertEqual(len(ws), 1)
            self.assertTrue(issubclass(ws[-1].category, BotorchWarning))

            # test objective
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                objective=IdentityMCMultiOutputObjective(),
            )
            # get mm sample shape to match shape of X
            acqf.model._posterior._samples = torch.zeros(1, 1, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # Test that the hypervolume improvement is correct for given sample
            # test q = 1
            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            samples = torch.tensor([[[6.5, 4.5]]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 1.5)
            # test q = 1, does not contribute
            samples = torch.tensor([0.0, 1.0], **tkwargs).view(1, 1, 2)
            sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test q = 2, both points contribute
            X = torch.zeros(2, 1, **tkwargs)
            samples = torch.tensor([[6.5, 4.5], [7.0, 4.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            # since q = 2, fidelity cost is 0 but fixed cost is 1 for each
            # hence total cost is 2 MOMF defaults to an Affine Cost Model.
            self.assertEqual(res.item(), 1.75 / 2)

            # test q = 2, only 1 point contributes
            samples = torch.tensor([[6.5, 4.5], [6.0, 4.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 1.5 / 2)

            # test q = 2, neither contributes
            samples = torch.tensor([[2.0, 2.0], [0.0, 0.1]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test q = 2, test point better than current best second objective
            samples = torch.tensor([[6.5, 4.5], [6.0, 6.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf.model = mm
            res = acqf(X)
            self.assertEqual(res.item(), 8.0 / 2)

            # test q = 2, test point better than current-best first objective
            samples = torch.tensor([[6.5, 4.5], [9.0, 2.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 2.0 / 2)
            # test q = 3, all contribute
            X = torch.zeros(3, 1, **tkwargs)
            samples = torch.tensor(
                [[6.5, 4.5], [9.0, 2.0], [7.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            # since q = 3, fidelity cost is 0 but fixed cost is 1 for each
            # hence total cost is 3.
            self.assertEqual(res.item(), 2.25 / 3)
            # test q = 3, not all contribute
            samples = torch.tensor(
                [[6.5, 4.5], [9.0, 2.0], [7.0, 5.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertTrue(
                torch.allclose(res, torch.tensor(3.5 / 3, **tkwargs), atol=1e-15)
            )
            # test q = 3, none contribute
            samples = torch.tensor(
                [[0.0, 4.5], [1.0, 2.0], [3.0, 0.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

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

            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(1, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 12.0)

            # test m = 3, q=1, X is ones so fidelity + fixed_cost is 2
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0], [1.0, 3.0, 4.0]],
                **tkwargs,
            )
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            samples = torch.tensor([[1.0, 2.0, 6.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))

            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.ones(1, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 12.0 / 2)

            # test m = 3, q=1, with custom callable function
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0], [1.0, 3.0, 4.0]],
                **tkwargs,
            )
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            samples = torch.tensor([[1.0, 2.0, 6.0]], **tkwargs).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))

            def cost(x):
                return (6 * x[..., -1]).unsqueeze(-1)

            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
                cost_call=cost,
            )
            X = torch.ones(1, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 12.0 / 6)
            # change reference point
            ref_point = [0.0] * 3
            X = torch.zeros(1, 2, **tkwargs)
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 4.0)

            # test m = 3, no contribution
            ref_point = [1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)

            # test m = 3, q = 2
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
            )
            samples = torch.tensor(
                [[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            mm = MockModel(MockPosterior(samples=samples))
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertEqual(res.item(), 22.0 / 2)

            # test batched model
            pareto_Y = torch.tensor(
                [[4.0, 2.0, 3.0], [3.0, 5.0, 1.0], [2.0, 4.0, 2.0]], **tkwargs
            )
            samples = torch.tensor(
                [[1.0, 2.0, 6.0], [1.0, 3.0, 4.0]], **tkwargs
            ).unsqueeze(0)
            samples = torch.stack([samples, samples + 1], dim=1)
            mm = MockModel(MockPosterior(samples=samples))
            ref_point = [-1.0] * 3
            t_ref_point = torch.tensor(ref_point, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=t_ref_point, Y=pareto_Y)
            acqf = MOMF(
                model=mm,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertTrue(
                torch.equal(
                    res,
                    # batch_shape x model_batch_shape
                    torch.tensor([[22.0, 60.0]], **tkwargs) / 2,
                )
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
            partitioning = FastNondominatedPartitioning(
                ref_point=t_ref_point, Y=pareto_Y
            )
            cell_bounds = partitioning.get_hypercell_bounds().unsqueeze(1)
            with mock.patch.object(
                partitioning, "get_hypercell_bounds", return_value=cell_bounds
            ):
                acqf = MOMF(
                    model=mm,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=sampler,
                )
                # test multiple batch dims
                self.assertEqual(acqf.cell_lower_bounds.shape, torch.Size([1, 2, 4, 2]))
                self.assertEqual(acqf.cell_upper_bounds.shape, torch.Size([1, 2, 4, 2]))
            X = torch.zeros(2, 2, **tkwargs)
            res = acqf(X)
            self.assertTrue(
                torch.equal(
                    res,
                    # batch_shape x model_batch_shape
                    torch.tensor(
                        [[1.75, 3.5]], dtype=samples.dtype, device=samples.device
                    )
                    / 2,
                )
            )
