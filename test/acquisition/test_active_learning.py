#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.acquisition.active_learning import (
    PairwiseMCPosteriorVariance,
    qNegIntegratedPosteriorVariance,
)
from botorch.acquisition.objective import (
    GenericMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.models.pairwise_gp import PairwiseGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal


class TestQNegIntegratedPosteriorVariance(BotorchTestCase):
    def test_init(self):
        mm = MockModel(MockPosterior(mean=torch.rand(2, 1)))
        mc_points = torch.rand(2, 2)
        qNIPV = qNegIntegratedPosteriorVariance(model=mm, mc_points=mc_points)
        sampler = qNIPV.sampler
        self.assertIsInstance(sampler, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, torch.Size([1]))
        self.assertTrue(torch.equal(mc_points, qNIPV.mc_points))
        self.assertIsNone(qNIPV.X_pending)
        self.assertIsNone(qNIPV.posterior_transform)
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        qNIPV = qNegIntegratedPosteriorVariance(
            model=mm, mc_points=mc_points, sampler=sampler
        )
        self.assertIsInstance(qNIPV.sampler, IIDNormalSampler)
        self.assertEqual(qNIPV.sampler.sample_shape, torch.Size([2]))

    def test_q_neg_int_post_variance(self):
        no = "botorch.utils.testing.MockModel.num_outputs"
        for dtype in (torch.float, torch.double):
            # basic test
            mean = torch.zeros(4, 1, device=self.device, dtype=dtype)
            variance = torch.rand(4, 1, device=self.device, dtype=dtype)
            mc_points = torch.rand(10, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with mock.patch.object(MockModel, "fantasize", return_value=mfm):
                with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    # TODO: Make this work with arbitrary models
                    mm = MockModel(None)
                    qNIPV = qNegIntegratedPosteriorVariance(
                        model=mm, mc_points=mc_points
                    )
                    X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
                    val = qNIPV(X)
                    self.assertAllClose(val, -(variance.mean()), atol=1e-4)
            # batched model
            mean = torch.zeros(2, 4, 1, device=self.device, dtype=dtype)
            variance = torch.rand(2, 4, 1, device=self.device, dtype=dtype)
            mc_points = torch.rand(2, 10, 1, device=self.device, dtype=dtype)
            mfm = MockModel(MockPosterior(mean=mean, variance=variance))
            with mock.patch.object(MockModel, "fantasize", return_value=mfm):
                with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    # TODO: Make this work with arbitrary models
                    mm = MockModel(None)
                    qNIPV = qNegIntegratedPosteriorVariance(
                        model=mm, mc_points=mc_points
                    )
                    # TODO: Allow broadcasting for batch evaluation
                    X = torch.empty(2, 1, 1, device=self.device, dtype=dtype)  # dummy
                    val = qNIPV(X)
                    val_exp = -variance.mean(dim=-2).squeeze(-1)
                    self.assertAllClose(val, val_exp, atol=1e-4)

            # multi-output model
            mean = torch.zeros(4, 2, device=self.device, dtype=dtype)
            variance = torch.rand(4, 2, device=self.device, dtype=dtype)
            cov = torch.diag_embed(variance.view(-1))
            f_posterior = GPyTorchPosterior(MultitaskMultivariateNormal(mean, cov))
            mc_points = torch.rand(10, 1, device=self.device, dtype=dtype)
            mfm = MockModel(f_posterior)
            with mock.patch.object(
                MockModel, "fantasize", return_value=mfm
            ), mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                mm = MockModel(None)

                weights = torch.tensor([0.5, 0.5], device=self.device, dtype=dtype)
                qNIPV = qNegIntegratedPosteriorVariance(
                    model=mm,
                    mc_points=mc_points,
                    posterior_transform=ScalarizedPosteriorTransform(weights=weights),
                )
                X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
                val = qNIPV(X)
                self.assertAllClose(val, -0.5 * variance.mean(), atol=1e-4)
                # without posterior_transform
                qNIPV = qNegIntegratedPosteriorVariance(
                    model=mm,
                    mc_points=mc_points,
                )
                val = qNIPV(X)
                self.assertAllClose(val, -variance.mean(0), atol=1e-4)

            # batched multi-output model
            mean = torch.zeros(4, 3, 1, 2, device=self.device, dtype=dtype)
            variance = torch.rand(4, 3, 1, 2, device=self.device, dtype=dtype)
            cov = torch.diag_embed(variance.view(4, 3, -1))
            f_posterior = GPyTorchPosterior(MultitaskMultivariateNormal(mean, cov))
            mc_points = torch.rand(4, 1, device=self.device, dtype=dtype)
            mfm = MockModel(f_posterior)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm):
                with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 2
                    mm = MockModel(None)
                    weights = torch.tensor([0.5, 0.5], device=self.device, dtype=dtype)
                    qNIPV = qNegIntegratedPosteriorVariance(
                        model=mm,
                        mc_points=mc_points,
                        posterior_transform=ScalarizedPosteriorTransform(
                            weights=weights
                        ),
                    )
                    X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)  # dummy
                    val = qNIPV(X)
                    val_exp = -0.5 * variance.mean(dim=0).view(3, -1).mean(dim=-1)
                    self.assertAllClose(val, val_exp, atol=1e-4)

            # single-output with posterior transform
            mean = torch.zeros(2, 10, 3, 1, device=self.device, dtype=dtype)
            variance = torch.ones(2, 10, 3, 1, device=self.device, dtype=dtype)
            cov = torch.diag_embed(variance.view(2, 10, -1))
            f_posterior = GPyTorchPosterior(MultitaskMultivariateNormal(mean, cov))
            mc_points = torch.rand(10, 1, device=self.device, dtype=dtype)
            mfm = MockModel(f_posterior)
            with mock.patch.object(MockModel, "fantasize", return_value=mfm):
                with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                    mock_num_outputs.return_value = 1
                    mm = MockModel(None)
                    qNIPV = qNegIntegratedPosteriorVariance(
                        model=mm,
                        mc_points=mc_points,
                        posterior_transform=ScalarizedPosteriorTransform(
                            weights=torch.tensor([1.0], device=self.device, dtype=dtype)
                        ),
                    )
                    X = torch.empty(2, 3, 1, 1, device=self.device, dtype=dtype)
                    val = qNIPV(X)
                    val_exp = -variance.mean(dim=-2).squeeze(-1).squeeze(0)
                    self.assertAllClose(val, val_exp, atol=1e-4)


class TestPairwiseMCPosteriorVariance(BotorchTestCase):
    def test_pairwise_mc_post_var(self):
        train_X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.double)
        train_comp = torch.tensor([[0, 1]], dtype=torch.long)
        model = PairwiseGP(train_X, train_comp)

        # example link function
        probit = torch.distributions.normal.Normal(0, 1).cdf
        probit_obj = GenericMCObjective(objective=lambda Y, X: probit(Y.squeeze(-1)))
        pv = PairwiseMCPosteriorVariance(model=model, objective=probit_obj)

        n_test_pair = 8
        good_X_2 = torch.rand((n_test_pair, 2, 3))
        good_X_4 = torch.rand((n_test_pair, 4, 3))
        bad_X = torch.rand((n_test_pair, 3, 3))

        # ensure q is a multiple of 2
        with self.assertRaises(RuntimeError):
            pv(bad_X)

        self.assertEqual(pv(good_X_2).shape, torch.Size([n_test_pair]))
        self.assertEqual(pv(good_X_4).shape, torch.Size([n_test_pair]))
