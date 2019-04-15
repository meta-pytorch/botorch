#! /usr/bin/env python3

import math
import unittest

import torch
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.exceptions import UnsupportedError
from botorch.models import FixedNoiseGP

from ..mock import MockModel, MockPosterior


NEI_NOISE = [-0.099, -0.004, 0.227, -0.182, 0.018, 0.334, -0.270, 0.156, -0.237, 0.052]


class TestAnalyticAcquisitionFunction(unittest.TestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AnalyticAcquisitionFunction()


class TestExpectedImprovement(unittest.TestCase):
    def test_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5]], device=device, dtype=dtype)
            variance = torch.ones(1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(0.19780, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement(cuda=True)

    def test_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype)
            variance = torch.ones(3, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ExpectedImprovement(model=mm, best_f=0.0)
            X = torch.empty(3, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=device, dtype=dtype
            )
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ExpectedImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_expected_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_expected_improvement_batch(cuda=True)


class TestPosteriorMean(unittest.TestCase):
    def test_posterior_mean(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.25]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(1, device=device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean[0, 0]))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            module2 = PosteriorMean(model=mm2)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_posterior_mean_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean(cuda=True)

    def test_posterior_mean_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            module = PosteriorMean(model=mm)
            X = torch.empty(3, 1, device=device, dtype=dtype)
            pm = module(X)
            self.assertTrue(torch.equal(pm, mean.view(-1)))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2))
            module2 = PosteriorMean(model=mm2)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_posterior_mean_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_posterior_mean_batch(cuda=True)


class TestProbabilityOfImprovement(unittest.TestCase):
    def test_probability_of_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0]], device=device, dtype=dtype)
            variance = torch.ones(1, 1, device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(1, device=device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor(0.5, device=device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ProbabilityOfImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_probability_of_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement(cuda=True)

    def test_probability_of_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0], [0.67449]], device=device, dtype=dtype)
            variance = torch.ones_like(mean)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ProbabilityOfImprovement(model=mm, best_f=0.0)
            X = torch.zeros(2, 1, device=device, dtype=dtype)
            pi = module(X)
            pi_expected = torch.tensor([0.5, 0.75], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(pi, pi_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.ones_like(mean2)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = ProbabilityOfImprovement(model=mm2, best_f=0.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_probability_of_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_probability_of_improvement_batch(cuda=True)


class TestUpperConfidenceBound(unittest.TestCase):
    def test_upper_confidence_bound(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0]], device=device, dtype=dtype)
            variance = torch.tensor([[1.0]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(1, 1, device=device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(1, 2, device=device, dtype=dtype)
            variance2 = torch.rand(1, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = UpperConfidenceBound(model=mm2, beta=1.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_upper_confidence_bound_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound(cuda=True)

    def test_upper_confidence_bound_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[0.0], [0.5]], device=device, dtype=dtype)
            variance = torch.tensor([[1.0], [4.0]], device=device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = UpperConfidenceBound(model=mm, beta=1.0)
            X = torch.zeros(2, 1, device=device, dtype=dtype)
            ucb = module(X)
            ucb_expected = torch.tensor([1.0, 2.5], device=device, dtype=dtype)
            self.assertTrue(torch.allclose(ucb, ucb_expected, atol=1e-4))
            # check for proper error if multi-output model
            mean2 = torch.rand(3, 2, device=device, dtype=dtype)
            variance2 = torch.rand(3, 2, device=device, dtype=dtype)
            mm2 = MockModel(MockPosterior(mean=mean2, variance=variance2))
            module2 = UpperConfidenceBound(model=mm2, beta=1.0)
            with self.assertRaises(UnsupportedError):
                module2(X)

    def test_upper_confidence_bound_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_upper_confidence_bound_batch(cuda=True)


class TestConstrainedExpectedImprovement(unittest.TestCase):
    def test_constrained_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # one constraint
            mean = torch.tensor([[-0.5, 0.0]], device=device, dtype=dtype).unsqueeze(
                dim=-2
            )
            variance = torch.ones(1, 2, device=device, dtype=dtype).unsqueeze(dim=-2)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm, best_f=0.0, objective_index=0, constraints={1: [None, 0]}
            )
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

            # check that error raised if no constraints
            with self.assertRaises(ValueError):
                module = ConstrainedExpectedImprovement(
                    model=mm, best_f=0.0, objective_index=0, constraints={}
                )

            # check that error raised if objective is a constraint
            with self.assertRaises(ValueError):
                module = ConstrainedExpectedImprovement(
                    model=mm, best_f=0.0, objective_index=0, constraints={0: [None, 0]}
                )

            # three constraints
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0]], device=device, dtype=dtype
            ).unsqueeze(dim=-2)
            variance = torch.ones(1, 4, device=device, dtype=dtype).unsqueeze(dim=-2)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(1, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                0.19780, device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_constrained_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_constrained_expected_improvement(cuda=True)

    def test_constrained_expected_improvement_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor(
                [[-0.5, 0.0, 5.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.5, 0.0, 5.0, 0.0]],
                device=device,
                dtype=dtype,
            ).unsqueeze(dim=-2)
            variance = torch.ones(3, 4, device=device, dtype=dtype).unsqueeze(dim=-2)
            N = torch.distributions.Normal(loc=0.0, scale=1.0)
            a = N.icdf(torch.tensor(0.75))  # get a so that P(-a <= N <= a) = 0.5
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            module = ConstrainedExpectedImprovement(
                model=mm,
                best_f=0.0,
                objective_index=0,
                constraints={1: [None, 0], 2: [5.0, None], 3: [-a, a]},
            )
            X = torch.empty(3, 1, device=device, dtype=dtype)  # dummy
            ei = module(X)
            ei_expected_unconstrained = torch.tensor(
                [0.19780, 0.39894, 0.69780], device=device, dtype=dtype
            )
            ei_expected = ei_expected_unconstrained * 0.5 * 0.5 * 0.5
            self.assertTrue(torch.allclose(ei, ei_expected, atol=1e-4))

    def test_constrained_expected_improvement_batch_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_constrained_expected_improvement_batch(cuda=True)


class TestNosiyExpectedImprovement(unittest.TestCase):
    def _get_model(self, cuda=False, dtype=torch.float):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        state_dict = {
            "likelihood.noise_covar.noise": torch.full((10,), 0.0625),
            "mean_module.constant": torch.tensor([-0.0066]),
            "covar_module.raw_outputscale": torch.tensor(1.0143),
            "covar_module.base_kernel.raw_lengthscale": torch.tensor([[-0.99]]),
            "covar_module.base_kernel.lengthscale_prior.concentration": torch.tensor(
                3.0
            ),
            "covar_module.base_kernel.lengthscale_prior.rate": torch.tensor(6.0),
            "covar_module.outputscale_prior.concentration": torch.tensor(2.0),
            "covar_module.outputscale_prior.rate": torch.tensor(0.1500),
        }
        train_x = torch.linspace(0, 1, 10, device=device, dtype=dtype)
        train_y = torch.sin(train_x * (2 * math.pi))
        noise = torch.tensor(NEI_NOISE, device=device, dtype=dtype)
        train_y += noise
        train_yvar = torch.full_like(train_y, 0.25 ** 2)
        train_x = train_x.view(-1, 1)
        model = FixedNoiseGP(train_X=train_x, train_Y=train_y, train_Yvar=train_yvar)
        model.load_state_dict(state_dict)
        model.to(train_x)
        model.eval()
        return model

    def test_noisy_expected_improvement(self, cuda=False):
        for dtype in (torch.float, torch.double):
            model = self._get_model(cuda=cuda, dtype=dtype)
            X_observed = model.train_inputs[0]
            nEI = NoisyExpectedImprovement(model, X_observed, num_fantasies=5)
            X_test = torch.tensor(
                [[0.25], [0.75]],
                device=X_observed.device,
                dtype=dtype,
                requires_grad=True,
            )
            val = nEI(X_test)
            # test basics
            self.assertEqual(val.dtype, dtype)
            self.assertEqual(val.device.type, X_observed.device.type)
            self.assertEqual(val.shape, torch.Size([2]))
            # test values
            self.assertGreater(val[0].item(), 1e-4)
            self.assertLess(val[1].item(), 1e-6)
            # test gradient
            val.sum().backward()
            self.assertGreater(X_test.grad.norm().item(), 1e-3)
            # test without gradient
            with torch.no_grad():
                val = nEI(X_test)

    def test_noisy_expected_improvement_cuda(self, cuda=False):
        if torch.cuda.is_available():
            self.test_noisy_expected_improvement(cuda=True)
