#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from unittest import mock
from warnings import catch_warnings, simplefilter

import torch
from botorch.acquisition import qAnalyticProbabilityOfImprovement
from botorch.acquisition.analytic import (
    _check_noisy_ei_model,
    _ei_helper,
    _log_ei_helper,
    AnalyticAcquisitionFunction,
    ConstrainedAnalyticAcquisitionFunctionMixin,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfFeasibility,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    ScalarizedPosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.objective import (
    IdentityMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions import UnsupportedError
from botorch.exceptions.warnings import NumericsWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import ChainedOutcomeTransform, Normalize, Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pathwise.utils import get_train_inputs
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.module import Module
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


NEI_NOISE = [
    [-0.099],
    [-0.004],
    [0.227],
    [-0.182],
    [0.018],
    [0.334],
    [-0.270],
    [0.156],
    [-0.237],
    [0.052],
]


class DummyAnalyticAcquisitionFunction(AnalyticAcquisitionFunction):
    def forward(self, X):
        pass


class TestAnalyticAcquisitionFunction(BotorchTestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            AnalyticAcquisitionFunction()
        # single-output models should always work
        mm = MockModel(MockPosterior(mean=torch.zeros(1, 1), variance=torch.ones(1, 1)))
        DummyAnalyticAcquisitionFunction(model=mm)
        # raise if model is multi-output, but no posterior transform is given
        mean = torch.zeros(1, 2)
        variance = torch.ones(1, 2)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        with self.assertRaises(UnsupportedError):
            DummyAnalyticAcquisitionFunction(model=mm)
        # do not raise if the allwo_multi_output flag is set
        DummyAnalyticAcquisitionFunction(model=mm, allow_multi_output=True)


class TestExpectedImprovement(BotorchTestCase):
    def test_expected_improvement(self):
        mean = torch.tensor([[-0.5]], device=self.device)
        variance = torch.ones(1, 1, device=self.device)
        model = MockModel(MockPosterior(mean=mean, variance=variance))
        with self.assertWarnsRegex(NumericsWarning, ".* LogExpectedImprovement .*"):
            ExpectedImprovement(model=model, best_f=0.0)

        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_expected_improvement(dtype=dtype)

        z = torch.tensor(-2.13, dtype=torch.float16, device=self.device)
        with self.assertRaises(TypeError):
            _log_ei_helper(z)

    def _test_expected_improvement(self, dtype: torch.dtype) -> None:
        mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
        variance = torch.ones(1, 1, device=self.device, dtype=dtype)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))

        # basic test
        module = ExpectedImprovement(model=mm, best_f=0.0)
        log_module = LogExpectedImprovement(model=mm, best_f=0.0)
        X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
        ei, log_ei = module(X), log_module(X)
        ei_expected = torch.tensor(0.19780, device=self.device, dtype=dtype)
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        self.assertAllClose(log_ei, ei_expected.log(), atol=1e-4)

        # test maximize
        module = ExpectedImprovement(model=mm, best_f=0.0, maximize=False)
        log_module = LogExpectedImprovement(model=mm, best_f=0.0, maximize=False)
        X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
        ei, log_ei = module(X), log_module(X)
        ei_expected = torch.tensor(0.6978, device=self.device, dtype=dtype)
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        self.assertAllClose(log_ei, ei_expected.log(), atol=1e-4)
        with self.assertRaises(UnsupportedError):
            module.set_X_pending(None)
        with self.assertRaises(UnsupportedError):
            log_module.set_X_pending(None)
        # test posterior transform (single-output)
        mean = torch.tensor([0.5], device=self.device, dtype=dtype)
        covar = torch.tensor([[0.16]], device=self.device, dtype=dtype)
        mvn = MultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([0.5], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        ei = ExpectedImprovement(model=mm, best_f=0.0, posterior_transform=transform)
        log_ei = LogExpectedImprovement(
            model=mm, best_f=0.0, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        ei_expected = torch.tensor(0.2601, device=self.device, dtype=dtype)
        self.assertAllClose(ei(X), ei_expected, atol=1e-4)
        self.assertAllClose(log_ei(X), ei_expected.log(), atol=1e-4)

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
        ei = ExpectedImprovement(model=mm, best_f=0.0, posterior_transform=transform)
        log_ei = LogExpectedImprovement(
            model=mm, best_f=0.0, posterior_transform=transform
        )
        X = torch.rand(1, 2, device=self.device, dtype=dtype)
        ei_expected = torch.tensor([0.6910], device=self.device, dtype=dtype)
        self.assertAllClose(ei(X), ei_expected, atol=1e-4)
        self.assertAllClose(log_ei(X), ei_expected.log(), atol=1e-4)

        # making sure we compare the lower branch of _log_ei_helper to _ei_helper
        z = torch.tensor(-2.13, dtype=dtype, device=self.device)
        self.assertAllClose(_log_ei_helper(z), _ei_helper(z).log(), atol=1e-6)

        # numerical stress test for log EI
        digits = 100 if dtype == torch.float64 else 20
        zero = torch.tensor([0], dtype=dtype, device=self.device)
        ten = torch.tensor(10, dtype=dtype, device=self.device)
        digits_tensor = torch.arange(0, digits, dtype=dtype, device=self.device)
        large_z = ten ** (digits_tensor)
        small_z = ten ** (-digits_tensor)
        # flipping the appropriate tensors so that elements are in increasing order
        test_z = [-large_z.flip(-1), -small_z, zero, small_z.flip(-1), large_z]
        for z in test_z:
            z.requires_grad = True
            y = _log_ei_helper(z)  # noqa
            # check that y isn't NaN of Inf
            self.assertFalse(y.isnan().any())
            self.assertFalse(y.isinf().any())
            # function values should increase with z
            self.assertTrue((y.diff() >= 0).all())
            # lets check the backward pass
            y.sum().backward()
            # check that gradients aren't NaN of Inf
            g = z.grad
            self.assertFalse(g.isnan().any())
            self.assertFalse(g.isinf().any())
            self.assertTrue((g >= 0).all())  # gradient is positive for all z

    def test_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_expected_improvement_batch(dtype=dtype)

    def _test_expected_improvement_batch(self, dtype):
        mean = torch.tensor([-0.5, 0.0, 0.5], device=self.device, dtype=dtype).view(
            3, 1, 1
        )
        variance = torch.ones(3, 1, 1, device=self.device, dtype=dtype)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        module = ExpectedImprovement(model=mm, best_f=0.0)
        log_module = LogExpectedImprovement(model=mm, best_f=0.0)
        X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)  # dummy
        ei, log_ei = module(X), log_module(X)
        ei_expected = torch.tensor(
            [0.19780, 0.39894, 0.69780], device=self.device, dtype=dtype
        )
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        self.assertAllClose(log_ei, ei_expected.log(), atol=1e-4)
        # check for proper error if multi-output model
        mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
        variance2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
        mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
        with self.assertRaises(UnsupportedError):
            ExpectedImprovement(model=mm2, best_f=0.0)
        with self.assertRaises(UnsupportedError):
            LogExpectedImprovement(model=mm2, best_f=0.0)
        # test posterior transform (single-output)
        mean = torch.tensor([[[0.5]], [[0.25]]], device=self.device, dtype=dtype)
        covar = torch.tensor([[[[0.16]]], [[[0.125]]]], device=self.device, dtype=dtype)
        mvn = MultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([0.5], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        ei = ExpectedImprovement(model=mm, best_f=0.0, posterior_transform=transform)
        log_ei = LogExpectedImprovement(
            model=mm, best_f=0.0, posterior_transform=transform
        )
        X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
        ei_expected = torch.tensor(
            [[0.2601], [0.1500]], device=self.device, dtype=dtype
        )
        self.assertAllClose(ei(X), ei_expected, atol=1e-4)
        self.assertAllClose(log_ei(X), ei(X).log(), atol=1e-4)

        # test posterior transform (multi-output)
        mean = torch.tensor(
            [[[-0.25, 0.5]], [[0.2, -0.1]]], device=self.device, dtype=dtype
        )
        covar = torch.tensor(
            [[[0.5, 0.125], [0.125, 0.5]], [[0.25, -0.1], [-0.1, 0.25]]],
            device=self.device,
            dtype=dtype,
        )
        mvn = MultitaskMultivariateNormal(mean, covar)
        p = GPyTorchPosterior(mvn)
        mm = MockModel(p)
        weights = torch.tensor([2.0, 1.0], device=self.device, dtype=dtype)
        transform = ScalarizedPosteriorTransform(weights)
        ei = ExpectedImprovement(model=mm, best_f=0.0, posterior_transform=transform)
        log_ei = LogExpectedImprovement(
            model=mm, best_f=0.0, posterior_transform=transform
        )
        X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
        ei_expected = torch.tensor([0.6910, 0.5371], device=self.device, dtype=dtype)
        self.assertAllClose(ei(X), ei_expected, atol=1e-4)
        self.assertAllClose(log_ei(X), ei_expected.log(), atol=1e-4)

        with self.assertRaises(UnsupportedError):
            ExpectedImprovement(
                model=mm, best_f=0.0, posterior_transform=IdentityMCObjective()
            )
        with self.assertRaises(UnsupportedError):
            LogExpectedImprovement(
                model=mm, best_f=0.0, posterior_transform=IdentityMCObjective()
            )


class TestPosteriorMean(BotorchTestCase):
    def test_posterior_mean(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))

            module = PosteriorMean(model=mm)
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))

            module = PosteriorMean(model=mm, maximize=False)
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, -mean.view(-1)))

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            with self.assertRaises(UnsupportedError):
                PosteriorMean(model=mm2)

    def test_posterior_mean_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([-0.5, 0.0, 0.5], device=self.device, dtype=dtype).view(
                3, 1, 1
            )
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            with self.assertRaises(UnsupportedError):
                PosteriorMean(model=mm2)


class TestPosteriorStandardDeviation(BotorchTestCase):
    def test_posterior_stddev(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 1, device=self.device, dtype=dtype)
            std = torch.rand_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=std.square()))

            acqf = PosteriorStandardDeviation(model=mm)
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            pm = acqf(X)
            self.assertTrue(torch.equal(pm, std.view(-1)))

            acqf = PosteriorStandardDeviation(model=mm, maximize=False)
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            pm = acqf(X)
            self.assertTrue(torch.equal(pm, -std.view(-1)))

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            std2 = torch.rand_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=std2.square()))
            with self.assertRaises(UnsupportedError):
                PosteriorStandardDeviation(model=mm2)

    def test_posterior_stddev_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 1, 1, device=self.device, dtype=dtype)
            std = torch.rand_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=std.square()))
            acqf = PosteriorStandardDeviation(model=mm)
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)
            pm = acqf(X)
            self.assertAllClose(pm, std.view(-1))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            std2 = torch.rand_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=std2.square()))
            msg = "Must specify a posterior transform when using a multi-output model."
            with self.assertRaisesRegex(UnsupportedError, msg):
                PosteriorStandardDeviation(model=mm2)


class TestProbabilityOfImprovement(BotorchTestCase):
    def test_probability_of_improvement(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            variance = torch.ones(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            kwargs = {"model": mm, "best_f": 1.96}
            module = ProbabilityOfImprovement(**kwargs)
            log_module = LogProbabilityOfImprovement(**kwargs)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            pi, log_pi = module(X), log_module(X)
            pi_expected = torch.tensor(0.0250, device=self.device, dtype=dtype)
            self.assertAllClose(pi, pi_expected, atol=1e-4)
            self.assertAllClose(log_pi.exp(), pi)
            kwargs = {"model": mm, "best_f": 1.96, "maximize": False}
            module = ProbabilityOfImprovement(**kwargs)
            log_module = LogProbabilityOfImprovement(**kwargs)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            pi, log_pi = module(X), log_module(X)
            pi_expected = torch.tensor(0.9750, device=self.device, dtype=dtype)
            self.assertAllClose(pi, pi_expected, atol=1e-4)
            self.assertAllClose(log_pi.exp(), pi)

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                ProbabilityOfImprovement(model=mm2, best_f=0.0)

            with self.assertRaises(UnsupportedError):
                LogProbabilityOfImprovement(model=mm2, best_f=0.0)

    def test_probability_of_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.67449], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            log_module = LogProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(2, 1, 1, device=self.device, dtype=dtype)
            pi, log_pi = module(X), log_module(X)
            pi_expected = torch.tensor([0.5, 0.75], device=self.device, dtype=dtype)
            self.assertAllClose(pi, pi_expected, atol=1e-4)
            self.assertAllClose(log_pi.exp(), pi)
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                ProbabilityOfImprovement(model=mm2, best_f=0.0)

            with self.assertRaises(UnsupportedError):
                LogProbabilityOfImprovement(model=mm2, best_f=0.0)


class TestqAnalyticProbabilityOfImprovement(BotorchTestCase):
    def test_q_analytic_probability_of_improvement(self):
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, device=self.device, dtype=dtype)
            cov = torch.eye(n=1, device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)

            # basic test
            module = qAnalyticProbabilityOfImprovement(model=mm, best_f=1.96)
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.0250, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # basic test, maximize
            module = qAnalyticProbabilityOfImprovement(
                model=mm, best_f=1.96, maximize=False
            )
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.9750, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # basic test, posterior transform (single-output)
            mean = torch.ones(1, device=self.device, dtype=dtype)
            cov = torch.eye(n=1, device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            weights = torch.tensor([0.5], device=self.device, dtype=dtype)
            transform = ScalarizedPosteriorTransform(weights)
            module = qAnalyticProbabilityOfImprovement(
                model=mm, best_f=0.0, posterior_transform=transform
            )
            X = torch.rand(1, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.8413, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # basic test, posterior transform (multi-output)
            mean = torch.ones(1, 2, device=self.device, dtype=dtype)
            cov = torch.eye(n=2, device=self.device, dtype=dtype).unsqueeze(0)
            mvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            weights = torch.ones(2, device=self.device, dtype=dtype)
            transform = ScalarizedPosteriorTransform(weights)
            module = qAnalyticProbabilityOfImprovement(
                model=mm, best_f=0.0, posterior_transform=transform
            )
            X = torch.rand(1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.9214, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # basic test, q = 2
            mean = torch.zeros(2, device=self.device, dtype=dtype)
            cov = torch.eye(n=2, device=self.device, dtype=dtype)
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            module = qAnalyticProbabilityOfImprovement(model=mm, best_f=1.96)
            X = torch.zeros(2, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.049375, device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

    def test_batch_q_analytic_probability_of_improvement(self):
        for dtype in (torch.float, torch.double):
            # test batch mode
            mean = torch.tensor([[0.0], [1.0]], device=self.device, dtype=dtype)
            cov = (
                torch.eye(n=1, device=self.device, dtype=dtype)
                .unsqueeze(0)
                .repeat(2, 1, 1)
            )
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            module = qAnalyticProbabilityOfImprovement(model=mm, best_f=0)
            X = torch.rand(2, 1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([0.5, 0.8413], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # test batched model and best_f values
            mean = torch.zeros(2, 1, device=self.device, dtype=dtype)
            cov = (
                torch.eye(n=1, device=self.device, dtype=dtype)
                .unsqueeze(0)
                .repeat(2, 1, 1)
            )
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            best_f = torch.tensor([0.0, -1.0], device=self.device, dtype=dtype)
            module = qAnalyticProbabilityOfImprovement(model=mm, best_f=best_f)
            X = torch.rand(2, 1, 1, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([[0.5, 0.8413]], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # test batched model, output transform (single output)
            mean = torch.tensor([[0.0], [1.0]], device=self.device, dtype=dtype)
            cov = (
                torch.eye(n=1, device=self.device, dtype=dtype)
                .unsqueeze(0)
                .repeat(2, 1, 1)
            )
            mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            weights = torch.tensor([0.5], device=self.device, dtype=dtype)
            transform = ScalarizedPosteriorTransform(weights)
            module = qAnalyticProbabilityOfImprovement(
                model=mm, best_f=0.0, posterior_transform=transform
            )
            X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([0.5, 0.8413], device=self.device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # test batched model, output transform (multiple output)
            mean = torch.tensor(
                [[[1.0, 1.0]], [[0.0, 1.0]]], device=self.device, dtype=dtype
            )
            cov = (
                torch.eye(n=2, device=self.device, dtype=dtype)
                .unsqueeze(0)
                .repeat(2, 1, 1)
            )
            mvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=cov)
            posterior = GPyTorchPosterior(mvn)
            mm = MockModel(posterior)
            weights = torch.ones(2, device=self.device, dtype=dtype)
            transform = ScalarizedPosteriorTransform(weights)
            module = qAnalyticProbabilityOfImprovement(
                model=mm, best_f=0.0, posterior_transform=transform
            )
            X = torch.rand(2, 1, 2, device=self.device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(
                [0.9214, 0.7602], device=self.device, dtype=dtype
            )
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))

            # test bad posterior transform class
            with self.assertRaises(UnsupportedError):
                qAnalyticProbabilityOfImprovement(
                    model=mm, best_f=0.0, posterior_transform=IdentityMCObjective()
                )


class TestUpperConfidenceBound(BotorchTestCase):
    def test_upper_confidence_bound(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.5]], device=self.device, dtype=dtype)
            variance = torch.tensor([[1.0]], device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor(1.5, device=self.device, dtype=dtype)
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)

            module = UpperConfidenceBound(model=mm, beta=1.0, maximize=False)
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor(0.5, device=self.device, dtype=dtype)
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)

            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                UpperConfidenceBound(model=mm2, beta=1.0)

    def test_upper_confidence_bound_batch(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0.0, 0.5], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            variance = torch.tensor([1.0, 4.0], device=self.device, dtype=dtype).view(
                2, 1, 1
            )
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(2, 1, 1, device=self.device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0, 2.5], device=self.device, dtype=dtype)
            self.assertAllClose(ucb, ucb_expected, atol=1e-4)
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            variance2 = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            with self.assertRaises(UnsupportedError):
                UpperConfidenceBound(model=mm2, beta=1.0)


class ConstrainedAnalyticAcquisitionFunctionTestHelper(
    ConstrainedAnalyticAcquisitionFunctionMixin
):
    """Derivative of the mixin class which implements register_buffer without inheriting
    from nn.Module, in order to test the methods of the mixin class in isolation.
    """

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        setattr(self, name, tensor)


class TestConstrainedExpectedImprovement(BotorchTestCase):
    def test_constrained_analytic_acquisition_function_mixin(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_constrained_analytic_acquisition_function_mixin(dtype=dtype)

    def _test_constrained_analytic_acquisition_function_mixin(
        self, dtype: torch.dtype
    ) -> None:
        # numerical stress test for _compute_log_prob_feas, which gets added to
        # log_ei in the forward pass, a quantity we already tested above
        # the limits here are determined by the largest power of ten x, such that
        #                          x - (b - a) < x
        # evaluates to true. In this test, the bounds are a, b = -digits, digits.
        means, sigmas, X_positive = self._get_numerical_stress_test_data(dtype=dtype)
        constraints = {0: [-5, 5]}

        mixin = ConstrainedAnalyticAcquisitionFunctionTestHelper(
            constraints=constraints
        )
        # test initialization
        for k in [
            "con_lower_inds",
            "con_upper_inds",
            "con_both_inds",
            "con_both",
            "con_lower",
            "con_upper",
        ]:
            self.assertTrue(hasattr(mixin, k))
            self.assertIsInstance(getattr(mixin, k), torch.Tensor)

        log_prob = mixin._compute_log_prob_feas(means=means, sigmas=sigmas)
        log_prob.sum().backward()

        self.assertFalse(log_prob.isnan().any())
        self.assertFalse(log_prob.isinf().any())
        self.assertFalse(means.grad.isnan().any())
        self.assertFalse(means.grad.isinf().any())
        # probability of feasibility increases until X = 0, decreases from there on
        prob_diff = log_prob.diff()
        k = len(X_positive)
        eps = 1e-6 if dtype == torch.float32 else 1e-15
        self.assertTrue((prob_diff[:k] > -eps).all())
        self.assertTrue((means.grad[:k] > -eps).all())
        # probability has stationary point at zero
        mean_grad_at_zero = means.grad[len(X_positive)]
        self.assertTrue(
            torch.allclose(mean_grad_at_zero, torch.zeros_like(mean_grad_at_zero))
        )
        # probability increases again
        self.assertTrue((prob_diff[-k:] < eps).all())
        self.assertTrue((means.grad[-k:] < eps).all())

    def _get_numerical_stress_test_data(self, dtype: torch.dtype) -> Tensor:
        digits = 10 if dtype == torch.float64 else 5
        zero = torch.tensor([0], dtype=dtype, device=self.device)
        ten = torch.tensor(10, dtype=dtype, device=self.device)
        digits_tensor = 1 + torch.arange(
            -digits, digits, dtype=dtype, device=self.device
        )
        X_positive = ten ** (digits_tensor)
        # flipping -X_positive so that elements are in increasing order
        means = torch.cat((-X_positive.flip(-1), zero, X_positive)).view(-1, 1, 1)
        means.requires_grad = True
        sigmas = torch.ones_like(means)
        return means, sigmas, X_positive

    def test_log_probability_of_feasibility(self) -> None:
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_log_probability_of_feasibility(dtype=dtype)
                self._test_log_probability_of_feasibility_e2e(dtype=dtype)

    def _test_log_probability_of_feasibility(self, dtype: torch.dtype) -> None:
        # Mock model for shape validation (model is not actually used due to mock below)
        mean = torch.tensor([[-0.5]], device=self.device)
        variance = torch.ones(1, 1, device=self.device)
        model = MockModel(MockPosterior(mean=mean, variance=variance))
        means, sigmas, _ = self._get_numerical_stress_test_data(dtype=dtype)
        constraints = {0: [-5, 5]}
        mixin = ConstrainedAnalyticAcquisitionFunctionTestHelper(
            constraints=constraints
        )
        acqf = LogProbabilityOfFeasibility(model=model, constraints=constraints)
        X = torch.randn_like(means).unsqueeze(-2)  # n x q=1 x d=1
        with mock.patch.object(acqf, "_mean_and_sigma", return_value=(means, sigmas)):
            acq_val = acqf(X)
        log_prob = mixin._compute_log_prob_feas(means=means, sigmas=sigmas)

        self.assertAllClose(acq_val, log_prob, atol=1e-4)

    def _test_log_probability_of_feasibility_e2e(self, dtype: torch.dtype) -> None:
        constraints = {0: [-5, 5]}
        expected_val = torch.tensor(
            [-3.41666852395495e-06], device=self.device, dtype=dtype
        )
        for batch_shape in (torch.Size(), torch.Size([2])):
            # batch shape of posterior mean/var is at least 1 due to the
            # @t_batch_mode_transform decorator
            mean = torch.tensor(-0.5, device=self.device, dtype=dtype).view(1, 1, 1)
            expected_acq_val = expected_val
            if len(batch_shape) > 0:
                mean = mean.expand(*batch_shape, 1, 1)
                expected_acq_val = expected_acq_val.expand(*batch_shape)
            variance = torch.ones_like(mean)
            model = MockModel(MockPosterior(mean=mean, variance=variance))
            acqf = LogProbabilityOfFeasibility(model=model, constraints=constraints)
            X = torch.rand(*batch_shape, 1, 1, device=self.device, dtype=dtype)
            acq_val = acqf(X)
            self.assertAllClose(acq_val, expected_acq_val, atol=1e-4)

    def test_constrained_expected_improvement(self) -> None:
        mean = torch.tensor([[-0.5]], device=self.device)
        variance = torch.ones(1, 1, device=self.device)
        model = MockModel(MockPosterior(mean=mean, variance=variance))
        with self.assertWarnsRegex(
            NumericsWarning, ".* LogConstrainedExpectedImprovement .*"
        ):
            ConstrainedExpectedImprovement(
                model=model,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0]},
            )

        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_constrained_expected_improvement(dtype=dtype)

    def _test_constrained_expected_improvement(self, dtype: torch.dtype) -> None:
        # one constraint
        mean = torch.tensor([[[-0.5, 0.0]]], device=self.device, dtype=dtype)
        variance = torch.ones_like(mean)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        kwargs = {
            "model": mm,
            "best_f": 0.0,
            "objective_index": 0,
            "constraints": {1: [None, 0]},
        }
        module = ConstrainedExpectedImprovement(**kwargs)
        log_module = LogConstrainedExpectedImprovement(**kwargs)
        # test initialization
        for k in [
            "con_lower_inds",
            "con_upper_inds",
            "con_both_inds",
            "con_both",
            "con_lower",
            "con_upper",
        ]:
            self.assertIn(k, module._buffers)
            self.assertIn(k, log_module._buffers)

        X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
        ei = module(X)
        ei_expected_unconstrained = torch.tensor(
            [0.19780], device=self.device, dtype=dtype
        )
        ei_expected = ei_expected_unconstrained * 0.5
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        log_ei = log_module(X)
        self.assertAllClose(log_ei, ei.log(), atol=1e-5)
        # testing LogCEI and CEI for lower, upper, and simultaneous bounds
        for bounds in [[None, 0], [0, None], [0, 1]]:
            kwargs["constraints"] = {1: bounds}
            module = ConstrainedExpectedImprovement(**kwargs)
            log_module = LogConstrainedExpectedImprovement(**kwargs)
            ei, log_ei = module(X), log_module(X)
            self.assertAllClose(log_ei, ei.log(), atol=1e-5)

        constructors = [
            ConstrainedExpectedImprovement,
            LogConstrainedExpectedImprovement,
        ]
        for constructor in constructors:
            # check that error raised if no constraints
            with self.assertRaises(ValueError):
                module = constructor(
                    model=mm, best_f=0.0, objective_index=0, constraints={}
                )

            # check that error raised if objective is a constraint
            with self.assertRaises(ValueError):
                module = constructor(
                    model=mm,
                    best_f=0.0,
                    objective_index=0,
                    constraints={0: [None, 0]},
                )

            # check that error raised if constraint lower > upper
            with self.assertRaises(ValueError):
                module = constructor(
                    model=mm, best_f=0.0, objective_index=0, constraints={0: [1, 0]}
                )

        # three constraints
        N = torch.distributions.Normal(loc=0.0, scale=1.0)
        a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
        mean = torch.tensor(
            [[-0.5, 0.0, 5.0, 0.0]], device=self.device, dtype=dtype
        ).unsqueeze(dim=-2)
        variance = torch.ones(1, 4, device=self.device, dtype=dtype).unsqueeze(dim=-2)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        kwargs = {
            "model": mm,
            "best_f": 0.0,
            "objective_index": 0,
            "constraints": {1: [None, 0], 2: [5.0, None], 3: [-a, a]},
        }
        module = ConstrainedExpectedImprovement(**kwargs)
        log_module = LogConstrainedExpectedImprovement(**kwargs)
        X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
        ei = module(X)
        ei_expected_unconstrained = torch.tensor(
            [0.19780], device=self.device, dtype=dtype
        )
        ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        # testing log module with regular implementation
        log_ei = log_module(X)
        self.assertAllClose(log_ei, ei_expected.log(), atol=1e-4)
        # test maximize
        kwargs = {
            "model": mm,
            "best_f": 0.0,
            "objective_index": 0,
            "constraints": {1: [None, 0]},
            "maximize": False,
        }
        module_min = ConstrainedExpectedImprovement(**kwargs)
        log_module_min = LogConstrainedExpectedImprovement(**kwargs)
        ei_min = module_min(X)
        ei_expected_unconstrained_min = torch.tensor(
            [0.6978], device=self.device, dtype=dtype
        )
        ei_expected_min = ei_expected_unconstrained_min * 0.5
        self.assertAllClose(ei_min, ei_expected_min, atol=1e-4)
        log_ei_min = log_module_min(X)
        self.assertAllClose(log_ei_min, ei_min.log(), atol=1e-4)

        # test invalid onstraints
        for constructor in constructors:
            with self.assertRaises(ValueError):
                constructor(
                    model=mm,
                    best_f=0.0,
                    objective_index=0,
                    constraints={1: [1.0, -1.0]},
                )

    def test_constrained_expected_improvement_batch(self):
        for dtype in (torch.float, torch.double):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_constrained_expected_improvement_batch(dtype=dtype)

    def _test_constrained_expected_improvement_batch(self, dtype: torch.dtype) -> None:
        # we test a batch of 3 points, each with one objective and three constraints
        obj_means = [-0.5, 0.0, 0.5]
        con1_means = [0.0, 0.0, 0.0]
        con2_means = [5.0, 5.0, 5.0]
        con3_means = [0.0, -0.25, 1.5]
        mean = (
            torch.tensor(
                [obj_means, con1_means, con2_means, con3_means],
                dtype=dtype,
                device=self.device,
            )  # outcome x batch_shape here, so we need to transpose
            .transpose(-1, -2)
            # and view this to be batch_shape x q=1 x d
            .view(3, 1, 4)
        )
        variance = torch.ones_like(mean)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))

        ei_expected_unconstrained = torch.tensor(
            [0.19780, 0.39894, 0.69780], device=self.device, dtype=dtype
        )

        N = torch.distributions.Normal(loc=0.0, scale=1.0)
        a = N.icdf(torch.tensor(0.7))  # get a so that P(-a <= N(0,1) <= a) = 0.4
        # Compute the respective probabilities of feasibility also for the
        # other means of the con3 outcome
        p_feas_con3 = torch.empty(3, device=self.device, dtype=dtype)
        for i, con3_mean in enumerate(con3_means):
            p_feas_con3[i] = N.cdf(a - con3_mean) - N.cdf(-a - con3_mean)

        # Make sure nothing got messed up in the test construction
        self.assertAlmostEqual(p_feas_con3[0].item(), 0.4)

        kwargs = {
            "model": mm,
            "best_f": 0.0,
            "objective_index": 0,
            "constraints": {1: [None, 0], 2: [5.0, None], 3: [-a, a]},
        }

        module = ConstrainedExpectedImprovement(**kwargs)
        log_module = LogConstrainedExpectedImprovement(**kwargs)
        X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)  # dummy
        ei, log_ei = module(X), log_module(X)
        self.assertTrue(ei.shape == torch.Size([3]))
        self.assertTrue(log_ei.shape == torch.Size([3]))
        ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * p_feas_con3
        self.assertAllClose(ei, ei_expected, atol=1e-4)
        self.assertAllClose(log_ei, ei.log(), atol=1e-4)


class TestNoisyExpectedImprovement(BotorchTestCase):
    def _get_model(
        self,
        dtype=torch.float,
        outcome_transform=None,
        input_transform=None,
        low_x=0.0,
        hi_x=1.0,
        covar_module=None,
    ) -> SingleTaskGP:
        state_dict = {
            "mean_module.raw_constant": torch.tensor([-0.0066]),
            "covar_module.raw_outputscale": torch.tensor(1.0143),
            "covar_module.base_kernel.raw_lengthscale": torch.tensor([[-0.99]]),
            "covar_module.base_kernel.lengthscale_prior.concentration": torch.tensor(
                3.0
            ),
            "covar_module.base_kernel.lengthscale_prior.rate": torch.tensor(6.0),
            "covar_module.outputscale_prior.concentration": torch.tensor(2.0),
            "covar_module.outputscale_prior.rate": torch.tensor(0.1500),
        }
        train_x = torch.linspace(
            0.0, 1.0, 10, device=self.device, dtype=dtype
        ).unsqueeze(-1)
        # Taking the sin of the *transformed* input to make the test equivalent
        # to when there are no input transforms
        train_y = torch.sin(train_x * (2 * math.pi))
        # Now transform the input to be passed into SingleTaskGP constructor
        train_x = train_x * (hi_x - low_x) + low_x
        noise = torch.tensor(NEI_NOISE, device=self.device, dtype=dtype)
        train_y += noise
        train_yvar = torch.full_like(train_y, 0.25**2)
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            covar_module=covar_module,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(train_x)
        model.eval()
        return model

    def test_noisy_expected_improvement(self) -> None:
        model = self._get_model(dtype=torch.float64)
        X_observed = model.train_inputs[0]
        nfan = 5
        with self.assertWarnsRegex(
            NumericsWarning, ".* LogNoisyExpectedImprovement .*"
        ):
            NoisyExpectedImprovement(model, X_observed, num_fantasies=nfan)

        # Same as the default Matern kernel
        # botorch.models.utils.gpytorch_modules.get_matern_kernel_with_gamma_prior,
        # except RBFKernel is used instead of MaternKernel.
        # For some reason, RBF gives numerical problems with torch.float but
        # Matern does not. Therefore, we'll skip the test for RBF when dtype is
        # torch.float.
        covar_module_2 = ScaleKernel(
            base_kernel=RBFKernel(
                ard_num_dims=1,
                batch_shape=torch.Size(),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=torch.Size(),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        for dtype, use_octf, use_intf, bounds, covar_module in itertools.product(
            (torch.float, torch.double),
            (False, True),
            (False, True),
            (torch.tensor([[-3.4], [0.8]]), torch.tensor([[0.0], [1.0]])),
            (None, covar_module_2),
        ):
            with catch_warnings():
                simplefilter("ignore", category=NumericsWarning)
                self._test_noisy_expected_improvement(
                    dtype=dtype,
                    use_octf=use_octf,
                    use_intf=use_intf,
                    bounds=bounds,
                    covar_module=covar_module,
                )

    def _test_noisy_expected_improvement(
        self,
        dtype: torch.dtype,
        use_octf: bool,
        use_intf: bool,
        bounds: torch.Tensor,
        covar_module: Module,
    ) -> None:
        if covar_module is not None and dtype == torch.float:
            # Skip this test because RBF runs into numerical problems with float
            # precision
            return
        octf = (
            ChainedOutcomeTransform(standardize=Standardize(m=1)) if use_octf else None
        )
        intf = (
            Normalize(
                d=1,
                bounds=bounds.to(device=self.device, dtype=dtype),
                transform_on_train=True,
            )
            if use_intf
            else None
        )
        low_x = bounds[0].item() if use_intf else 0.0
        hi_x = bounds[1].item() if use_intf else 1.0
        model = self._get_model(
            dtype=dtype,
            outcome_transform=octf,
            input_transform=intf,
            low_x=low_x,
            hi_x=hi_x,
            covar_module=covar_module,
        )
        # Make sure to get the non-transformed training inputs.
        X_observed = get_train_inputs(model, transformed=False)[0]

        nfan = 5
        torch.manual_seed(1)
        nEI = NoisyExpectedImprovement(model, X_observed, num_fantasies=nfan)
        LogNEI = LogNoisyExpectedImprovement(model, X_observed, num_fantasies=nfan)
        # before assigning, check that the attributes exist
        self.assertTrue(hasattr(LogNEI, "model"))
        self.assertTrue(hasattr(LogNEI, "best_f"))
        self.assertIsInstance(LogNEI.model, SingleTaskGP)
        self.assertIsInstance(LogNEI.model.likelihood, FixedNoiseGaussianLikelihood)
        # Make sure _get_noiseless_fantasy_model gives them
        # the same state_dict
        self.assertEqual(LogNEI.model.state_dict(), model.state_dict())

        LogNEI.model = nEI.model  # let the two share their values and fantasies
        LogNEI.best_f = nEI.best_f

        X_test = torch.tensor(
            [[[0.25]], [[0.75]]],
            device=X_observed.device,
            dtype=dtype,
        )
        X_test_log = X_test.clone()
        X_test.requires_grad = True
        X_test_log.requires_grad = True

        val = nEI(X_test * (hi_x - low_x) + low_x)
        # testing logNEI yields the same result (also checks dtype)
        log_val = LogNEI(X_test_log * (hi_x - low_x) + low_x)
        exp_log_val = log_val.exp()
        # notably, val[1] is usually zero in this test, which is precisely what
        # gives rise to problems during optimization, and what logNEI avoids
        # since it generally takes a large negative number (<-2000) and has
        # strong gradient signals in this regime.
        rtol = 1e-12 if dtype == torch.double else 1e-6
        atol = rtol
        self.assertAllClose(exp_log_val, val, atol=atol, rtol=rtol)
        # test basics
        self.assertEqual(val.dtype, dtype)
        self.assertEqual(val.device.type, X_observed.device.type)
        self.assertEqual(val.shape, torch.Size([2]))
        # test values
        self.assertGreater(val[0].item(), 8e-5)
        self.assertLess(val[1].item(), 1e-6)
        # test gradient
        val.sum().backward()
        self.assertGreater(X_test.grad[0].abs().item(), 8e-6)
        # testing gradient through exp of log computation
        exp_log_val.sum().backward()
        # testing that first gradient element coincides. The second is in the
        # regime where the naive implementation loses accuracy.
        atol = 2e-5 if dtype == torch.float32 else 1e-12
        rtol = atol
        self.assertAllClose(X_test.grad[0], X_test_log.grad[0], atol=atol, rtol=rtol)

        # test inferred noise model
        other_model = SingleTaskGP(X_observed, model.train_targets.unsqueeze(-1))
        for constructor in (
            NoisyExpectedImprovement,
            LogNoisyExpectedImprovement,
        ):
            with self.assertRaises(UnsupportedError):
                constructor(other_model, X_observed, num_fantasies=5)
            # Test constructor with minimize
            acqf = constructor(model, X_observed, num_fantasies=5, maximize=False)
            # test evaluation without gradients enabled
            with torch.no_grad():
                acqf(X_test)

            # testing gradients are only propagated if X_observed requires them
            # i.e. kernel hyper-parameters are not tracked through to best_f
            X_observed.requires_grad = False
            acqf = constructor(model, X_observed, num_fantasies=5)
            self.assertFalse(acqf.best_f.requires_grad)

            X_observed.requires_grad = True
            acqf = constructor(model, X_observed, num_fantasies=5)
            self.assertTrue(acqf.best_f.requires_grad)

    def test_check_noisy_ei_model(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        # Multi-output model.
        model = SingleTaskGP(
            train_X=torch.rand(5, 2, **tkwargs),
            train_Y=torch.rand(5, 2, **tkwargs),
            train_Yvar=torch.rand(5, 2, **tkwargs),
        )
        with self.assertRaisesRegex(UnsupportedError, "Model has 2 outputs"):
            _check_noisy_ei_model(model=model)
        # Not SingleTaskGP.
        with self.assertRaisesRegex(UnsupportedError, "Model is not"):
            _check_noisy_ei_model(model=ModelListGP(model))
        # Not fixed noise.
        model.likelihood = GaussianLikelihood()
        with self.assertRaisesRegex(UnsupportedError, "Model likelihood is not"):
            _check_noisy_ei_model(model=model)


class TestScalarizedPosteriorMean(BotorchTestCase):
    def test_scalarized_posterior_mean(self) -> None:
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.25], [0.5]], device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            weights = torch.tensor([0.5, 1.0], device=self.device, dtype=dtype)
            module = ScalarizedPosteriorMean(model=mm, weights=weights)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(
                torch.allclose(pm, (mean.squeeze(-1) * module.weights).sum(dim=-1))
            )

    def test_scalarized_posterior_mean_batch(self) -> None:
        for dtype in (torch.float, torch.double):
            mean = torch.tensor(
                [[-0.5, 1.0], [0.0, 1.0], [0.5, 1.0]], device=self.device, dtype=dtype
            ).view(3, 2, 1)
            mm = MockModel(MockPosterior(mean=mean))
            weights = torch.tensor([0.5, 1.0], device=self.device, dtype=dtype)

            module = ScalarizedPosteriorMean(model=mm, weights=weights)
            X = torch.empty(3, 1, 1, device=self.device, dtype=dtype)
            pm = module(X)
            self.assertTrue(
                torch.allclose(pm, (mean.squeeze(-1) * module.weights).sum(dim=-1))
            )
