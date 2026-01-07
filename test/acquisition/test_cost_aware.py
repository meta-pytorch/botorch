#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition.cost_aware import (
    CostAwareUtility,
    GenericCostAwareUtility,
    InverseCostWeightedUtility,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling import IIDNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestCostAwareUtilities(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            CostAwareUtility()

    def test_GenericCostAwareUtility(self):
        def cost(X, deltas, **kwargs):
            return deltas.mean(dim=-1) / X[..., 1].sum(dim=-1)

        for dtype in (torch.float, torch.double):
            u = GenericCostAwareUtility(cost)
            X = torch.rand(3, 2, device=self.device, dtype=dtype)
            deltas = torch.rand(5, 3, device=self.device, dtype=dtype)
            self.assertIsInstance(u, GenericCostAwareUtility)
            self.assertTrue(torch.equal(u(X, deltas), cost(X, deltas)))
            X = torch.rand(4, 3, 2, device=self.device, dtype=dtype)
            deltas = torch.rand(5, 4, 3, device=self.device, dtype=dtype)
            self.assertIsInstance(u, GenericCostAwareUtility)
            self.assertTrue(torch.equal(u(X, deltas), cost(X, deltas)))

    def test_InverseCostWeightedUtility(self):
        for batch_shape in ([], [2]):
            for dtype in (torch.float, torch.double):
                # the event shape is `batch_shape x q x t`
                mean = 1 + torch.rand(
                    *batch_shape, 2, 1, device=self.device, dtype=dtype
                )
                mm = MockModel(MockPosterior(mean=mean))

                X = torch.randn(*batch_shape, 3, 2, device=self.device, dtype=dtype)
                deltas = torch.rand(4, *batch_shape, device=self.device, dtype=dtype)
                neg_deltas = -torch.rand(
                    4, *batch_shape, device=self.device, dtype=dtype
                )

                # test that sampler is required if use_mean=False
                icwu = InverseCostWeightedUtility(mm, use_mean=False)
                with self.assertRaises(RuntimeError):
                    icwu(X, deltas)

                # test ValueError if costs are negative
                mm = MockModel(MockPosterior(mean=mean.clamp_max(-1e-6)))
                icwu = InverseCostWeightedUtility(mm)
                with self.assertRaisesRegex(
                    ValueError,
                    (
                        "Costs must be strictly positive. "
                        "Consider clamping cost_objective."
                    ),
                ):
                    icwu(X, deltas)

                # test ValueError if costs are zero
                mm = MockModel(MockPosterior(mean=mean.clamp(min=0.0, max=0.0)))
                icwu = InverseCostWeightedUtility(mm)
                with self.assertRaisesRegex(
                    ValueError,
                    (
                        "Costs must be strictly positive. "
                        "Consider clamping cost_objective."
                    ),
                ):
                    icwu(X, deltas)

                # basic test for both positive and negative delta values
                mm = MockModel(MockPosterior(mean=mean))
                icwu = InverseCostWeightedUtility(mm)
                ratios = icwu(X, deltas)
                self.assertTrue(
                    torch.equal(ratios, deltas / mean.squeeze(-1).sum(dim=-1))
                )
                neg_ratios = icwu(X, neg_deltas)
                self.assertTrue(
                    torch.equal(neg_ratios, neg_deltas * mean.squeeze(-1).sum(dim=-1))
                )

                # test that ensures candidates of lower cost are preferred when they
                # have equal, negative delta values
                low_mean = 1 + torch.rand(
                    *batch_shape, 2, 1, device=self.device, dtype=dtype
                )
                high_mean = 2 + torch.rand(
                    *batch_shape, 2, 1, device=self.device, dtype=dtype
                )
                h_mm = MockModel(MockPosterior(mean=high_mean))
                h_icwu = InverseCostWeightedUtility(h_mm)
                # high cost ratios
                h_ratios = h_icwu(X, neg_deltas)
                l_mm = MockModel(MockPosterior(mean=low_mean))
                l_icwu = InverseCostWeightedUtility(l_mm)
                # low cost ratios
                l_ratios = l_icwu(X, neg_deltas)
                # assert that the low cost candidates are preferred
                self.assertTrue(torch.all(h_ratios < l_ratios))

                # sampling test
                samples = 1 + torch.rand(  # event shape is q x m
                    *batch_shape, 3, 1, device=self.device, dtype=dtype
                )
                mm = MockModel(MockPosterior(samples=samples))
                icwu = InverseCostWeightedUtility(mm, use_mean=False)
                ratios = icwu(
                    X, deltas, sampler=IIDNormalSampler(sample_shape=torch.Size([4]))
                )
                self.assertTrue(
                    torch.equal(ratios, deltas / samples.squeeze(-1).sum(dim=-1))
                )

                # test log cost
                mm = MockModel(MockPosterior(mean=mean))
                icwu = InverseCostWeightedUtility(mm, log=True)
                log_ratios = icwu(X, torch.log(deltas))

                self.assertAllClose(
                    log_ratios.exp(), deltas / mean.squeeze(-1).sum(dim=-1), atol=1e-4
                )

                # test evaluation_mask
                multi_output_mean = torch.cat([mean, 2 * mean], dim=-1)

                def cost_fn(X):
                    return multi_output_mean

                mm = GenericDeterministicModel(f=cost_fn, num_outputs=2)
                icwu = InverseCostWeightedUtility(mm)
                eval_mask = torch.zeros(3, 2, dtype=torch.bool, device=self.device)
                eval_mask[:, 1] = True  # 1 objective is evaluated
                ratios = icwu(X, deltas, X_evaluation_mask=eval_mask)

                self.assertTrue(
                    torch.equal(ratios, deltas / multi_output_mean[..., 1].sum(dim=-1))
                )
                eval_mask[:, 0] = True  # both objectives are evaluated
                ratios = icwu(X, deltas, X_evaluation_mask=eval_mask)
                self.assertAllClose(
                    ratios, deltas / multi_output_mean.sum(dim=(-1, -2))
                )
                # test eval_mask where not all rows are the same
                eval_mask[0, 1] = False
                msg = (
                    "Currently, all candidates must be evaluated "
                    "on the same outputs."
                )
                with self.assertRaisesRegex(NotImplementedError, msg):
                    icwu(X, deltas, X_evaluation_mask=eval_mask)
