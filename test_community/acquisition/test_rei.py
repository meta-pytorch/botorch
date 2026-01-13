#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.rei import (
    LogRegionalExpectedImprovement,
    qLogRegionalExpectedImprovement,
)
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


class TestLogRegionalExpectedImprovement(BotorchTestCase):
    def test_log_regional_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            self._test_log_regional_expected_improvement(dtype=dtype)

    def _test_log_regional_expected_improvement(self, dtype: torch.dtype) -> None:
        mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
        variance = torch.ones(1, 1, device=self.device, dtype=dtype)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))

        # basic test
        X_dev = torch.rand(1, 1, device=self.device, dtype=dtype)
        module = LogRegionalExpectedImprovement(model=mm, best_f=0.0, X_dev=X_dev)
        X = torch.empty(1, 1, device=self.device, dtype=dtype)
        log_rei = module(X)
        rei_expected = torch.tensor([0.19780], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei, rei_expected.log(), atol=1e-4)

        # test maximize
        X_dev = torch.rand(1, 1, device=self.device, dtype=dtype)
        module = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, maximize=False
        )
        X = torch.empty(1, 1, device=self.device, dtype=dtype)
        log_rei = module(X)
        rei_expected = torch.tensor([0.6978], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei, rei_expected.log(), atol=1e-4)

        with self.assertRaisesRegex(
            UnsupportedError,
            "Analytic acquisition functions do not account for X_pending yet.",
        ):
            module.set_X_pending(None)

        # test bounds argument
        X_dev = torch.rand(1, 1, device=self.device, dtype=dtype)
        bounds = torch.tensor([[0.0], [1.0]], device=self.device, dtype=dtype)
        module = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, bounds=bounds
        )
        X = torch.empty(1, 1, device=self.device, dtype=dtype)
        log_rei = module(X)
        rei_expected = torch.tensor([0.19780], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei, rei_expected.log(), atol=1e-4)

        # test posterior transform (single-output)
        mean = torch.tensor([0.5], device=self.device, dtype=dtype)
        covar = torch.tensor([[0.16]], device=self.device, dtype=dtype)
        mvn = MultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([0.5], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        X_dev = torch.rand(1, 2, device=self.device, dtype=dtype)
        log_rei = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        rei_expected = torch.tensor([0.2601], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei(X), rei_expected.log(), atol=1e-4)

        # test posterior transform (multi-output)
        mean = torch.tensor([[-0.25, 0.5]], device=self.device, dtype=dtype)
        covar = torch.tensor(
            [[[0.5, 0.125], [0.125, 0.5]]], device=self.device, dtype=dtype
        )
        mvn = MultitaskMultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([2.0, 1.0], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        X_dev = torch.rand(1, 2, device=self.device, dtype=dtype)
        log_rei = LogRegionalExpectedImprovement(
            model=mm, best_f=0.0, X_dev=X_dev, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        rei_expected = torch.tensor([0.6910], device=self.device, dtype=dtype)
        self.assertAllClose(log_rei(X), rei_expected.log(), atol=1e-4)


class TestQLogRegionalExpectedImprovement(BotorchTestCase):
    def test_q_log_regional_expected_improvement(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_q_log_regional_expected_improvement(dtype)

    def _test_q_log_regional_expected_improvement(self, dtype: torch.dtype) -> None:
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
        # ``mc_model_samples x mc_X_dev_samples x q x d`` = 1 x 1 x 1 x 1
        samples = torch.zeros(1, 1, 1, 1, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        # X is ``q x d`` = 1 x 1. X is a dummy and unused b/c of mocking
        X = torch.zeros(1, 1, **tkwargs)
        X_dev = torch.zeros(1, 1, **tkwargs)
        bounds = torch.tensor([[0.0], [1.0]], **tkwargs)

        # basic test
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        # test initialization
        for k in ["objective", "sampler"]:
            self.assertIn(k, acqf._modules)

        res = acqf(X)
        # self.assertEqual(res.item(), 0.0)
        self.assertAlmostEqual(res.item(), -14.0473, places=4)

        # test shifting best_f value
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=-1, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertEqual(res.item(), 0.0)

        # test bounds argument
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=-1, X_dev=X_dev, sampler=sampler, bounds=bounds
        )
        res = acqf(X)
        self.assertEqual(res.item(), 0.0)

        # basic test, no resample
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res.item(), -14.0473, places=4)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1, 1]))
        bs = acqf.sampler.base_samples.clone()
        res = acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # basic test, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res.item(), -14.0473, places=4)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 1, 1]))
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
        mm._posterior._samples = torch.zeros(1, 2, 1, **tkwargs)
        res = acqf(X)
        X2 = torch.zeros(1, 1, 1, **tkwargs, requires_grad=True)
        with warnings.catch_warnings(record=True) as ws:
            acqf.set_X_pending(X2)
        self.assertEqual(acqf.X_pending, X2)
        self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)

        # Test without mocking tensor dimensions
        n = 3
        d = 2
        X = torch.randn(n, d, dtype=torch.float64)
        Y = X.sin().sum(dim=-1, keepdim=True)

        model = SingleTaskGP(X, Y, input_transform=Normalize(d=X.shape[-1]))

        n_tr = 17
        X_dev = torch.rand(n_tr, d)
        acqf = qLogRegionalExpectedImprovement(model, best_f=0.0, X_dev=X_dev)
        batch_shape = ()
        X_test = torch.randn(*batch_shape, n, d)
        q_log_rei = acqf(X_test)
        self.assertEqual(q_log_rei.shape, torch.Size([1]))

    def test_q_log_regional_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_q_log_regional_expected_improvement_batch(dtype)

    def _test_q_log_regional_expected_improvement_batch(
        self, dtype: torch.dtype
    ) -> None:
        # ``mc_model_samples x mc_X_dev_samples x q x d`` = 1 x 1 x 1 x 1
        samples = torch.zeros(2, 2, 2, 1, device=self.device, dtype=dtype)
        samples[0, 0, 0] = 1.0
        mm = MockModel(MockPosterior(samples=samples))

        # X is a dummy and unused b/c of mocking
        X = torch.zeros(2, 2, 1, device=self.device, dtype=dtype)
        X_dev = torch.zeros(2, 1, device=self.device, dtype=dtype)

        # test batch mode
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res[0].item(), -0.6931, places=4)
        self.assertAlmostEqual(res[1].item(), -14.0403, places=4)

        # test batch model, batched best_f values
        sampler = IIDNormalSampler(sample_shape=torch.Size([3]))
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=torch.Tensor([0, 0]), X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res[0].item(), -0.6931, places=4)
        self.assertAlmostEqual(res[1].item(), -14.0403, places=4)

        # test shifting best_f value
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=-1, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res[0].item(), 0.4078, places=4)
        self.assertAlmostEqual(res[1].item(), 0.0069, places=4)

        # test batch mode
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]), seed=12345)
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)  # 1-dim batch
        self.assertAlmostEqual(res[0].item(), -0.6931, places=4)
        self.assertAlmostEqual(res[1].item(), -14.0403, places=4)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
        res = acqf(X.expand(2, 2, 1))  # 2-dim batch
        self.assertAlmostEqual(res[0].item(), -0.6931, places=4)
        self.assertAlmostEqual(res[1].item(), -14.0403, places=4)
        # the base samples should have the batch dim collapsed
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X.expand(2, 2, 1))
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # test batch mode, qmc
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        acqf = qLogRegionalExpectedImprovement(
            model=mm, best_f=0, X_dev=X_dev, sampler=sampler
        )
        res = acqf(X)
        self.assertAlmostEqual(res[0].item(), -0.6931, places=4)
        self.assertAlmostEqual(res[1].item(), -14.0403, places=4)
        self.assertEqual(acqf.sampler.base_samples.shape, torch.Size([2, 1, 1, 2, 1]))
        bs = acqf.sampler.base_samples.clone()
        acqf(X)
        self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))

        # Test without mocking
        n = 3
        d = 2
        X = torch.randn(n, d, dtype=torch.float64)
        Y = X.sin().sum(dim=-1, keepdim=True)

        model = SingleTaskGP(
            X,
            Y,
            input_transform=Normalize(d=X.shape[-1]),
        )

        n_tr = 17
        X_dev = torch.rand(n_tr, d)
        acqf = qLogRegionalExpectedImprovement(model, best_f=0.0, X_dev=X_dev)
        # Test for non-trivial batch shape
        batch_shape = (5,)
        X_test = torch.randn(*batch_shape, n, d)
        q_log_rei = acqf(X_test)
        self.assertEqual(q_log_rei.shape, torch.Size([5]))
