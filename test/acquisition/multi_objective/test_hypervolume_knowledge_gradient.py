#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from itertools import product
from unittest import mock

import numpy as np

import torch
from botorch.acquisition.cost_aware import (
    GenericCostAwareUtility,
    InverseCostWeightedUtility,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    _split_hvkg_fantasy_points,
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import NumericsWarning
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


NO = "botorch.models.model_list_gp_regression.ModelListGP.num_outputs"


def cost_fn(X):
    costs = torch.ones(X.shape[:-1] + torch.Size([2]), dtype=X.dtype, device=X.device)
    costs[..., 1] = 3.0
    return costs


class TestHypervolumeKnowledgeGradient(BotorchTestCase):
    def test_initialization(self):
        tkwargs = {"device": self.device}
        for dtype, acqf_class in product(
            (torch.float, torch.double),
            (qHypervolumeKnowledgeGradient, qMultiFidelityHypervolumeKnowledgeGradient),
        ):
            tkwargs["dtype"] = dtype
            X = torch.rand(4, 3, **tkwargs)
            Y1 = torch.rand(4, 1, **tkwargs)
            Y2 = torch.rand(4, 1, **tkwargs)
            m1 = SingleTaskGP(X, Y1)
            m2 = SingleTaskGP(X, Y2)
            model = ModelListGP(m1, m2)
            ref_point = torch.zeros(2, **tkwargs)
            # test sampler is None
            if acqf_class == qMultiFidelityHypervolumeKnowledgeGradient:
                mf_kwargs = {"target_fidelities": {-1: 1.0}}
            else:
                mf_kwargs = {}
            acqf = acqf_class(model=model, ref_point=ref_point, **mf_kwargs)

            self.assertIsInstance(acqf.sampler, ListSampler)
            self.assertEqual(acqf.sampler.samplers[0].sample_shape, torch.Size([8]))
            # test ref point
            self.assertTrue(torch.equal(acqf.ref_point, ref_point))
            # test sampler is not None
            sampler = ListSampler(
                SobolQMCNormalSampler(sample_shape=torch.Size([4])),
                SobolQMCNormalSampler(sample_shape=torch.Size([4])),
            )
            with self.assertRaisesRegex(
                ValueError, "The sampler shape must match num_fantasies=8."
            ):
                acqf_class(
                    model=model, ref_point=ref_point, sampler=sampler, **mf_kwargs
                )
            acqf = acqf_class(
                model=model,
                ref_point=ref_point,
                num_fantasies=4,
                num_pareto=8,
                sampler=sampler,
                use_posterior_mean=False,
                **mf_kwargs,
            )
            self.assertEqual(acqf.num_fantasies, 4)
            self.assertEqual(acqf.num_pareto, 8)
            self.assertEqual(acqf.num_pseudo_points, 32)
            self.assertFalse(acqf.use_posterior_mean)
            self.assertIsInstance(acqf.inner_sampler, SobolQMCNormalSampler)
            self.assertEqual(acqf.inner_sampler.sample_shape, torch.Size([32]))
            self.assertIsNone(acqf._cost_sampler)
            # test objective
            mc_objective = GenericMCMultiOutputObjective(lambda Y, X: 2 * Y)
            acqf = acqf_class(
                model=model, ref_point=ref_point, objective=mc_objective, **mf_kwargs
            )
            self.assertIs(acqf.objective, mc_objective)
            # test X_pending
            X_pending = torch.rand(2, 3, **tkwargs)
            acqf = acqf_class(
                model=model, ref_point=ref_point, X_pending=X_pending, **mf_kwargs
            )
            self.assertTrue(torch.equal(acqf.X_pending, X_pending))
            # test X_pending_evaluation_mask
            X_pending_evaluation_mask = torch.eye(2, device=self.device).bool()
            acqf = acqf_class(
                model=model,
                ref_point=ref_point,
                X_pending=X_pending,
                X_pending_evaluation_mask=X_pending_evaluation_mask,
                **mf_kwargs,
            )
            self.assertTrue(
                torch.equal(acqf.X_pending_evaluation_mask, X_pending_evaluation_mask)
            )
            # test cost aware utility
            cost_model = GenericDeterministicModel(
                lambda X: torch.ones(X.shape[:-1], 2, **tkwargs)
            )
            for use_mean in (True, False):
                cost_aware_utility = InverseCostWeightedUtility(
                    cost_model=cost_model, use_mean=use_mean
                )
                with self.assertRaisesRegex(
                    UnsupportedError,
                    "Cost-aware HVKG requires current_value to be specified.",
                ):
                    acqf_class(
                        model=model,
                        ref_point=ref_point,
                        cost_aware_utility=cost_aware_utility,
                        **mf_kwargs,
                    )
                acqf = acqf_class(
                    model=model,
                    ref_point=ref_point,
                    cost_aware_utility=cost_aware_utility,
                    current_value=0.0,
                    **mf_kwargs,
                )
                self.assertEqual(acqf.current_value, 0.0)
                self.assertIs(acqf.cost_aware_utility, cost_aware_utility)

            # test get_aug_q_batch_size
            self.assertEqual(acqf.get_augmented_q_batch_size(q=3), 83)

            if acqf_class is qMultiFidelityHypervolumeKnowledgeGradient:
                # test default
                x = torch.rand(5, 3, **tkwargs)
                self.assertTrue(torch.equal(acqf.project(x), x))
                # test expand raises exception
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "Trace observations are not currently supported.",
                ):
                    acqf_class(
                        model=model,
                        ref_point=ref_point,
                        expand=lambda X: X,
                        **mf_kwargs,
                    )

    def test_evaluate_q_hvkg(self):
        torch.manual_seed(torch.randint(low=0, high=10, size=(1,)).item())
        tkwargs = {"device": self.device}
        num_pareto = 3
        for dtype, acqf_class, log_ehvi in product(
            (torch.float, torch.double),
            (qHypervolumeKnowledgeGradient, qMultiFidelityHypervolumeKnowledgeGradient),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            # basic test
            n_f = 4
            mean = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
            variance = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            ref_point = torch.zeros(2, **tkwargs)
            models = [
                SingleTaskGP(torch.rand(2, 1, **tkwargs), torch.rand(2, 1, **tkwargs)),
                SingleTaskGP(torch.rand(4, 1, **tkwargs), torch.rand(4, 1, **tkwargs)),
            ]
            model = ModelListGP(*models)
            if acqf_class == qMultiFidelityHypervolumeKnowledgeGradient:
                mf_kwargs = {"target_fidelities": {-1: 1.0}}
            else:
                mf_kwargs = {}

            with mock.patch.object(
                ModelListGP, "fantasize", return_value=mfm
            ) as patch_f, mock.patch(
                NO, new_callable=mock.PropertyMock
            ) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                qHVKG = acqf_class(
                    model=model,
                    num_fantasies=n_f,
                    ref_point=ref_point,
                    num_pareto=num_pareto,
                    log=log_ehvi,
                    **mf_kwargs,
                )
                X = torch.rand(n_f * num_pareto + 1, 1, **tkwargs)
                val = qHVKG(X)
                patch_f.assert_called_once()
                cargs, ckwargs = patch_f.call_args
                self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
            expected_hv = (
                DominatedPartitioning(Y=mean.squeeze(1), ref_point=ref_point)
                .compute_hypervolume()
                .mean()
            )
            if log_ehvi:
                self.assertAllClose(val.exp().item(), expected_hv.item(), rtol=1e-1)
            else:
                self.assertAllClose(val.item(), expected_hv.item(), atol=1e-4)
            self.assertTrue(
                torch.equal(qHVKG.extract_candidates(X), X[..., : -n_f * num_pareto, :])
            )

            # batched evaluation
            b = 2
            mean = torch.rand(n_f, b, num_pareto, 2, **tkwargs)
            variance = torch.rand(n_f, b, num_pareto, 2, **tkwargs)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.rand(b, n_f * num_pareto + 1, 1, **tkwargs)
            with mock.patch.object(
                ModelListGP, "fantasize", return_value=mfm
            ) as patch_f, mock.patch(
                NO, new_callable=mock.PropertyMock
            ) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                qHVKG = acqf_class(
                    model=model,
                    num_fantasies=n_f,
                    ref_point=ref_point,
                    num_pareto=num_pareto,
                    log=log_ehvi,
                    **mf_kwargs,
                )
                val = qHVKG(X)
                patch_f.assert_called_once()
                cargs, ckwargs = patch_f.call_args
                self.assertEqual(ckwargs["X"].shape, torch.Size([b, 1, 1]))
            expected_hv = (
                DominatedPartitioning(
                    Y=mean.view(-1, num_pareto, 2), ref_point=ref_point
                )
                .compute_hypervolume()
                .view(n_f, b)
                .mean(dim=0)
            )
            if log_ehvi:
                self.assertAllClose(val.exp(), expected_hv, rtol=1e-1)
            else:
                self.assertAllClose(val, expected_hv, atol=1e-4)
            self.assertTrue(
                torch.equal(qHVKG.extract_candidates(X), X[..., : -n_f * num_pareto, :])
            )
            # pending points and current value
            X_pending = torch.rand(2, 1, **tkwargs)
            X_pending_evaluation_mask = torch.eye(2, device=self.device).bool()
            X_evaluation_mask = torch.tensor(
                [[False, True]], dtype=torch.bool, device=self.device
            )
            mean = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
            variance = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            X = torch.rand(n_f * num_pareto + 1, 1, **tkwargs)
            current_value = torch.tensor(0.01, **tkwargs)
            # if log flag is on, current_value is expected to be in log space
            input_current_value = (
                torch.log(current_value) if log_ehvi else current_value
            )
            # if log flag is on, the cost_aware_utility should be in log space
            cost_aware_utility = InverseCostWeightedUtility(
                cost_model=GenericDeterministicModel(cost_fn, num_outputs=2),
                log=log_ehvi,
            )
            with mock.patch.object(
                ModelListGP, "fantasize", return_value=mfm
            ) as patch_f, mock.patch(
                NO, new_callable=mock.PropertyMock
            ) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                qHVKG = acqf_class(
                    model=model,
                    num_fantasies=n_f,
                    X_pending=X_pending,
                    X_pending_evaluation_mask=X_pending_evaluation_mask,
                    X_evaluation_mask=X_evaluation_mask,
                    current_value=input_current_value,
                    ref_point=ref_point,
                    num_pareto=num_pareto,
                    cost_aware_utility=cost_aware_utility,
                    log=log_ehvi,
                    **mf_kwargs,
                )
                val = qHVKG(X)
                patch_f.assert_called_once()
                expected_eval_mask = torch.cat(
                    [X_evaluation_mask, X_pending_evaluation_mask], dim=0
                )
                cargs, ckwargs = patch_f.call_args
                self.assertEqual(ckwargs["X"].shape, torch.Size([1, 3, 1]))
                self.assertTrue(
                    torch.equal(ckwargs["evaluation_mask"], expected_eval_mask)
                )
            expected_hv = (
                DominatedPartitioning(Y=mean.squeeze(1), ref_point=ref_point)
                .compute_hypervolume()
                .mean(dim=0)
            )
            # divide by 3 because the cost is 3 per design
            expected = (expected_hv.mean() - current_value).item() / 3
            if log_ehvi:
                self.assertAllClose(
                    val.exp().item(),
                    expected,
                    rtol=1e-1,
                )
            else:
                self.assertAllClose(val.item(), expected, atol=1e-4)
            self.assertTrue(
                torch.equal(qHVKG.extract_candidates(X), X[..., : -n_f * num_pareto, :])
            )

            # test that it raises an error if cost_aware_utility is not Logged
            if log_ehvi:
                cost_aware_utility = InverseCostWeightedUtility(
                    cost_model=GenericDeterministicModel(cost_fn, num_outputs=2),
                    log=False,
                )
                with mock.patch.object(
                    ModelListGP, "fantasize", return_value=mfm
                ) as patch_f, mock.patch(
                    NO, new_callable=mock.PropertyMock
                ) as mock_num_outputs:
                    mock_num_outputs.return_value = 2
                    qHVKG = acqf_class(
                        model=model,
                        num_fantasies=n_f,
                        current_value=input_current_value,
                        ref_point=ref_point,
                        num_pareto=num_pareto,
                        cost_aware_utility=cost_aware_utility,
                        log=log_ehvi,
                        **mf_kwargs,
                    )
                    # raises error when log=False
                    with self.assertRaisesRegex(
                        BotorchError,
                        "Cost-aware HVKG has _log=True and requires cost_aware_utility "
                        "to output log utilities.",
                    ):
                        val = qHVKG(X)

                # check log is not a class attribute of cost_aware_utility
                # Generic cost aware utility does not have a log flag
                cost_aware_utility = GenericCostAwareUtility(
                    cost=GenericDeterministicModel(cost_fn, num_outputs=2),
                )
                with mock.patch.object(
                    ModelListGP, "fantasize", return_value=mfm
                ) as patch_f, mock.patch(
                    NO, new_callable=mock.PropertyMock
                ) as mock_num_outputs:
                    mock_num_outputs.return_value = 2
                    qHVKG = acqf_class(
                        model=model,
                        num_fantasies=n_f,
                        current_value=input_current_value,
                        ref_point=ref_point,
                        num_pareto=num_pareto,
                        cost_aware_utility=cost_aware_utility,
                        log=log_ehvi,
                        **mf_kwargs,
                    )
                    # raises error when log is not a class variable
                    with self.assertRaisesRegex(
                        BotorchError,
                        "Cost-aware HVKG has _log=True and requires cost_aware_utility "
                        "to output log utilities.",
                    ):
                        val = qHVKG(X)

            # test mfkg
            if acqf_class == qMultiFidelityHypervolumeKnowledgeGradient:
                mean = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
                variance = torch.rand(n_f, 1, num_pareto, 2, **tkwargs)
                mfm = MockModel(MockPosterior(mean=mean, variance=variance))
                current_value = torch.rand(1, **tkwargs) if not log_ehvi else None
                X = torch.rand(n_f * num_pareto + 1, 1, **tkwargs)
                with mock.patch(
                    "botorch.acquisition.multi_objective."
                    "hypervolume_knowledge_gradient._get_hv_value_function",
                    wraps=_get_hv_value_function,
                ) as mock_get_value_func:
                    with mock.patch.object(
                        ModelListGP, "fantasize", return_value=mfm
                    ) as patch_f, mock.patch(
                        NO, new_callable=mock.PropertyMock
                    ) as mock_num_outputs:
                        mock_num_outputs.return_value = 2
                        qHVKG = acqf_class(
                            model=model,
                            num_fantasies=n_f,
                            current_value=current_value,
                            ref_point=ref_point,
                            num_pareto=num_pareto,
                            log=log_ehvi,
                            **mf_kwargs,
                        )
                        val = qHVKG(X)
                        self.assertIsNotNone(
                            mock_get_value_func.call_args_list[0][1]["project"]
                        )

            # test objective (inner MC sampling)
            mean = torch.rand(n_f, 1, num_pareto, 3, **tkwargs)
            samples = mean + 1
            variance = torch.rand(n_f, 1, num_pareto, 3, **tkwargs)
            mfm = MockModel(
                MockPosterior(mean=mean, variance=variance, samples=samples)
            )
            models = [
                SingleTaskGP(torch.rand(2, 1, **tkwargs), torch.rand(2, 1, **tkwargs)),
                SingleTaskGP(torch.rand(4, 1, **tkwargs), torch.rand(4, 1, **tkwargs)),
                SingleTaskGP(torch.rand(5, 1, **tkwargs), torch.rand(5, 1, **tkwargs)),
            ]
            model = ModelListGP(*models)
            for num_objectives in (2, 3):
                # test using 1) a botorch objective that only uses 2 out of
                # 3 outcomes as objectives, 2) a botorch objective that uses
                # all 3 outcomes as objectives
                objective = (
                    IdentityMCMultiOutputObjective(outcomes=[0, 1])
                    if num_objectives == 2
                    else GenericMCMultiOutputObjective(lambda Y, X: 2 * Y)
                )

                ref_point = torch.zeros(num_objectives, **tkwargs)
                X = torch.rand(n_f * num_pareto + 1, 1, **tkwargs)

                for use_posterior_mean in (True, False):
                    with mock.patch.object(
                        ModelListGP, "fantasize", return_value=mfm
                    ) as patch_f, mock.patch(
                        NO, new_callable=mock.PropertyMock
                    ) as mock_num_outputs, warnings.catch_warnings(record=True) as ws:
                        mock_num_outputs.return_value = 3
                        qHVKG = acqf_class(
                            model=model,
                            num_fantasies=n_f,
                            objective=objective,
                            ref_point=ref_point,
                            num_pareto=num_pareto,
                            use_posterior_mean=use_posterior_mean,
                            log=log_ehvi,
                            **mf_kwargs,
                        )
                        val = qHVKG(X)
                    patch_f.assert_called_once()
                    cargs, ckwargs = patch_f.call_args
                    self.assertEqual(ckwargs["X"].shape, torch.Size([1, 1, 1]))
                    self.assertFalse(any(w.category is NumericsWarning for w in ws))
                    Ys = mean if use_posterior_mean else samples
                    objs = objective(Ys.squeeze(1)).view(-1, num_pareto, num_objectives)
                    if num_objectives == 2:
                        expected_hv = (
                            DominatedPartitioning(Y=objs, ref_point=ref_point)
                            .compute_hypervolume()
                            .mean()
                            .item()
                        )
                    else:
                        # batch box decomposition don't support > 2 objectives
                        objs = objective(Ys).view(-1, num_pareto, num_objectives)
                        expected_hv = np.mean(
                            [
                                DominatedPartitioning(Y=obj, ref_point=ref_point)
                                .compute_hypervolume()
                                .mean()
                                .item()
                                for obj in objs
                            ]
                        )
                    if log_ehvi:
                        self.assertAllClose(val.exp().item(), expected_hv, rtol=1e-1)
                    else:
                        self.assertAllClose(val.item(), expected_hv, atol=1e-4)
                    self.assertTrue(
                        torch.equal(
                            qHVKG.extract_candidates(X),
                            X[..., : -n_f * num_pareto, :],
                        )
                    )

    def test_split_hvkg_fantasy_points(self):
        d = 4
        for dtype, batch_shape, n_f, num_pareto, q in product(
            (torch.float, torch.double), ([], [2], [3, 2]), (1, 4), (1, 3), (1, 2)
        ):
            X = torch.rand(
                *batch_shape, q + num_pareto * n_f, d, dtype=dtype, device=self.device
            )
            X_actual, X_fant = _split_hvkg_fantasy_points(
                X=X, n_f=n_f, num_pareto=num_pareto
            )
            self.assertTrue(torch.equal(X_actual, X[..., :q, :]))
            self.assertTrue(
                torch.equal(
                    X_fant, X[..., q:, :].reshape(n_f, *batch_shape, num_pareto, d)
                )
            )
        # test two many fantasies
        X = torch.rand(10, d, device=self.device)
        n_f = 100
        num_pareto = 3
        msg = (
            rf".*\({n_f * num_pareto}\) must be less than"
            rf" the `q`-batch dimension of `X` \({X.size(-2)}\)\."
        )
        with self.assertRaisesRegex(ValueError, msg):
            _split_hvkg_fantasy_points(X=X, n_f=n_f, num_pareto=num_pareto)
