#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.objective import (
    IdentityMCObjective,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.generation.sampling import (
    BoltzmannSampling,
    ConstrainedMaxPosteriorSampling,
    MaxPosteriorSampling,
    SamplingStrategy,
)
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestSamplingStrategy(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            SamplingStrategy()


class TestMaxPosteriorSampling(BotorchTestCase):
    def test_init(self):
        mm = MockModel(MockPosterior(mean=None))
        MPS = MaxPosteriorSampling(mm)
        self.assertEqual(MPS.model, mm)
        self.assertTrue(MPS.replacement)
        self.assertIsInstance(MPS.objective, IdentityMCObjective)
        obj = LinearMCObjective(torch.rand(2))
        MPS = MaxPosteriorSampling(mm, objective=obj, replacement=False)
        self.assertEqual(MPS.objective, obj)
        self.assertFalse(MPS.replacement)

    def test_max_posterior_sampling(self):
        batch_shapes = (torch.Size(), torch.Size([3]), torch.Size([3, 2]))
        dtypes = (torch.float, torch.double)
        for batch_shape, dtype, N, num_samples, d in itertools.product(
            batch_shapes, dtypes, (5, 6), (1, 2), (1, 2)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            # X is ``batch_shape x N x d`` = batch_shape x N x 1.
            X = torch.randn(*batch_shape, N, d, **tkwargs)
            # the event shape is ``num_samples x batch_shape x N x m``
            psamples = torch.zeros(num_samples, *batch_shape, N, 1, **tkwargs)
            psamples[..., 0, :] = 1.0

            # IdentityMCObjective, with replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    MPS = MaxPosteriorSampling(mm)
                    s = MPS(X, num_samples=num_samples)
                    self.assertTrue(torch.equal(s, X[..., [0] * num_samples, :]))

            # ScalarizedPosteriorTransform w/ replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    with mock.patch.object(
                        ScalarizedPosteriorTransform, "forward", return_value=mp
                    ):
                        post_tf = ScalarizedPosteriorTransform(torch.rand(2, **tkwargs))
                        MPS = MaxPosteriorSampling(mm, posterior_transform=post_tf)
                        s = MPS(X, num_samples=num_samples)
                        self.assertTrue(torch.equal(s, X[..., [0] * num_samples, :]))

            # without replacement
            psamples[..., 1, 0] = 1e-6
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    MPS = MaxPosteriorSampling(mm, replacement=False)
                    if len(batch_shape) > 1:
                        with self.assertRaises(NotImplementedError):
                            MPS(X, num_samples=num_samples)
                    else:
                        s = MPS(X, num_samples=num_samples)
                        # order is not guaranteed, need to sort
                        self.assertTrue(
                            torch.equal(
                                torch.sort(s, dim=-2).values,
                                torch.sort(X[..., :num_samples, :], dim=-2).values,
                            )
                        )


class TestBoltzmannSampling(BotorchTestCase):
    def test_init(self):
        NO = "botorch.utils.testing.MockModel.num_outputs"
        with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 1
            mm = MockModel(None)
            acqf = PosteriorMean(mm)
            BS = BoltzmannSampling(acqf)
            self.assertEqual(BS.acq_func, acqf)
            self.assertEqual(BS.eta, 1.0)
            self.assertTrue(BS.replacement)
            BS = BoltzmannSampling(acqf, eta=0.5, replacement=False)
            self.assertEqual(BS.acq_func, acqf)
            self.assertEqual(BS.eta, 0.5)
            self.assertFalse(BS.replacement)

    def test_boltzmann_sampling(self):
        dtypes = (torch.float, torch.double)
        batch_shapes = (torch.Size(), torch.Size([3]))

        # test a bunch of combinations
        for batch_shape, N, d, num_samples, repl, dtype in itertools.product(
            batch_shapes, [6, 7], [1, 2], [4, 5], [True, False], dtypes
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.rand(*batch_shape, N, d, **tkwargs)
            acqval = torch.randn(N, *batch_shape, **tkwargs)
            acqf = mock.Mock(return_value=acqval)
            BS = BoltzmannSampling(acqf, replacement=repl, eta=2.0)
            samples = BS(X, num_samples=num_samples)
            self.assertEqual(samples.shape, batch_shape + torch.Size([num_samples, d]))
            self.assertEqual(samples.dtype, dtype)
            if not repl:
                # check that we don't repeat points
                self.assertEqual(torch.unique(samples, dim=-2).size(-2), num_samples)

        # check that we do indeed pick the maximum for large eta
        for N, d, dtype in itertools.product([6, 7], [1, 2], dtypes):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.rand(N, d, **tkwargs)
            acqval = torch.zeros(N, **tkwargs)
            max_idx = torch.randint(N, (1,))
            acqval[max_idx] = 10.0
            acqf = mock.Mock(return_value=acqval)
            BS = BoltzmannSampling(acqf, eta=10.0)
            samples = BS(X, num_samples=1)
            self.assertTrue(torch.equal(samples, X[max_idx, :]))


class TestConstrainedMaxPosteriorSampling(BotorchTestCase):
    def test_init(self):
        mm = MockModel(MockPosterior(mean=None))
        cmms = MockModel(MockPosterior(mean=None))
        for replacement in (True, False):
            MPS = ConstrainedMaxPosteriorSampling(mm, cmms, replacement=replacement)
            self.assertEqual(MPS.model, mm)
            self.assertEqual(MPS.replacement, replacement)
            self.assertIsInstance(MPS.objective, IdentityMCObjective)

        obj = LinearMCObjective(torch.rand(2))
        with self.assertRaisesRegex(
            NotImplementedError, "`objective` is not supported"
        ):
            ConstrainedMaxPosteriorSampling(mm, cmms, objective=obj, replacement=False)

    def test_constrained_max_posterior_sampling(self):
        for batch_shape, dtype, N, num_samples, d, observation_noise in [
            (torch.Size(), torch.float, 5, 1, 1, False),
            (torch.Size([3]), torch.float, 6, 3, 2, False),
            (torch.Size([3, 2]), torch.double, 6, 3, 2, True),
        ]:
            tkwargs = {"device": self.device, "dtype": dtype}
            expected_shape = torch.Size(list(batch_shape) + [num_samples] + [d])
            # X is ``batch_shape x N x d`` = batch_shape x N x 1.
            X = torch.randn(*batch_shape, N, d, **tkwargs)
            # the event shape is ``num_samples x batch_shape x N x m``
            psamples = torch.zeros(num_samples, *batch_shape, N, 1, **tkwargs)
            psamples[..., 0, :] = 1.0

            # IdentityMCObjective, with replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    c_model1 = SingleTaskGP(
                        X, torch.randn(X.shape[:-1], **tkwargs).unsqueeze(-1)
                    )
                    c_model2 = SingleTaskGP(
                        X, torch.randn(X.shape[:-1], **tkwargs).unsqueeze(-1)
                    )
                    c_model3 = SingleTaskGP(
                        X, torch.randn(X.shape[:-1], **tkwargs).unsqueeze(-1)
                    )
                    cmms1 = MockModel(MockPosterior(mean=None))
                    cmms2 = SingleTaskGP(  # Multi-output model as constraint.
                        X, torch.randn((X.shape[0:-1] + (4,)), **tkwargs)
                    )
                    # ModelListGP as constraint.
                    cmms3 = ModelListGP(c_model1, c_model2, c_model3)
                    for cmms in [cmms1, cmms2, cmms3]:
                        CPS = ConstrainedMaxPosteriorSampling(mm, cmms)
                        s1 = CPS(
                            X=X,
                            num_samples=num_samples,
                            observation_noise=observation_noise,
                        )
                        self.assertEqual(s1.shape, expected_shape)

            # Test selection (_convert_samples_to_scores is tested separately)
            m_model = SingleTaskGP(
                X, torch.randn(X.shape[0:-1], **tkwargs).unsqueeze(-1)
            )
            cmms = cmms2
            with torch.random.fork_rng():
                torch.manual_seed(123)
                Y = m_model.posterior(X=X, observation_noise=observation_noise).rsample(
                    sample_shape=torch.Size([num_samples])
                )
                C = cmms.posterior(X=X, observation_noise=observation_noise).rsample(
                    sample_shape=torch.Size([num_samples])
                )
                scores = CPS._convert_samples_to_scores(Y_samples=Y, C_samples=C)
                X_true = CPS.maximize_samples(
                    X=X, samples=scores, num_samples=num_samples
                )

                torch.manual_seed(123)
                CPS = ConstrainedMaxPosteriorSampling(m_model, cmms)
                X_cand = CPS(
                    X=X,
                    num_samples=num_samples,
                    observation_noise=observation_noise,
                )
                self.assertAllClose(X_true, X_cand)

        # Test ``_convert_samples_to_scores``
        N, num_constraints, batch_shape = 10, 3, torch.Size([2])
        X = torch.randn(*batch_shape, N, d, **tkwargs)
        Y_samples = torch.rand(num_samples, *batch_shape, N, 1, **tkwargs)
        C_samples = -torch.rand(
            num_samples, *batch_shape, N, num_constraints, **tkwargs
        )

        Y_samples[0, 0, 3] = 1.234
        C_samples[0, 1, 1:, 1] = 0.123 + torch.arange(N - 1, **tkwargs)
        C_samples[1, 0, :, :] = 1 + (torch.arange(N).unsqueeze(-1) - N // 2) ** 2
        Y_samples[1, 1, 7] = 10
        scores = ConstrainedMaxPosteriorSampling(
            m_model, cmms
        )._convert_samples_to_scores(Y_samples=Y_samples, C_samples=C_samples)
        self.assertEqual(scores[0, 0].argmax().item(), 3)
        self.assertEqual(scores[0, 1].argmax().item(), 0)
        self.assertEqual(scores[1, 0].argmax().item(), N // 2)
        self.assertEqual(scores[1, 1].argmax().item(), 7)
