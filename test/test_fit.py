#! /usr/bin/env python3

import math
import unittest

import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


NOISE = [0.127, -0.113, -0.345, -0.034, -0.069, -0.272, 0.013, 0.056, 0.087, -0.081]


class TestFitGPyTorchModel(unittest.TestCase):
    def setUp(self):
        self.train_x = torch.linspace(0, 1, 10)
        self.train_y = torch.sin(self.train_x * (2 * math.pi)) + torch.tensor(NOISE)

    def _getModel(self, cuda=False):
        train_x = self.train_x.cuda() if cuda else self.train_x
        train_y = self.train_y.cuda() if cuda else self.train_y
        model = SingleTaskGP(train_x.detach(), train_y.detach())
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll.cuda() if cuda else mll

    def test_fit_gpytorch_model_scipy(self, cuda=False):
        mll = self._getModel(cuda=cuda)
        mll = fit_gpytorch_model(mll, options={"maxiter": 1})
        model = mll.model
        # Make sure all of the parameters changed
        self.assertGreater(model.likelihood.raw_noise.abs().item(), 1e-3)
        self.assertGreater(model.mean_module.constant.abs().item(), 1e-3)
        self.assertGreater(
            model.covar_module.base_kernel.raw_lengthscale.abs().item(), 1e-3
        )
        self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

    def test_fit_gpytorch_model_torch(self, cuda=False):
        mll = self._getModel(cuda=cuda)
        mll = fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, maxiter=1)
        model = mll.model
        # Make sure all of the parameters changed
        self.assertGreater(model.likelihood.raw_noise.abs().item(), 1e-3)
        self.assertGreater(model.mean_module.constant.abs().item(), 1e-3)
        self.assertGreater(
            model.covar_module.base_kernel.raw_lengthscale.abs().item(), 1e-3
        )
        self.assertGreater(model.covar_module.raw_outputscale.abs().item(), 1e-3)

    def test_fit_gpytorch_model_scipy_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_gpytorch_model_scipy(cuda=True)

    def test_fit_gpytorch_model_torch_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_gpytorch_model_torch(cuda=True)
