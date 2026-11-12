#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless

import torch
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DenseLinearOperator


class ComponentModel(Model):
    def __init__(self, model, index):
        r"""Wrap one ensemble member of `model` as a single model.

        Args:
            model: The ensemble model.
            index: The index of the ensemble member.
        """
        super().__init__()
        self.model, self.index = model, index

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size()

    @property
    def train_inputs(self):
        return self.model.train_inputs

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        result = self.model.posterior(X, observation_noise=observation_noise)
        mean = result.mean[..., self.index, :, 0]
        covariance = result.distribution.covariance_matrix[..., self.index, :, :]
        posterior = GPyTorchPosterior(
            MultivariateNormal(mean, DenseLinearOperator(covariance))
        )
        return (
            posterior if posterior_transform is None else posterior_transform(posterior)
        )


def make_model(device):
    X = torch.tensor(
        [[0.1, 0.2], [0.3, 0.6], [0.6, 0.4], [0.9, 0.8]],
        dtype=torch.double,
        device=device,
    )
    Y = (4 * X[:, :1]).sin()
    model = SaasFullyBayesianSingleTaskGP(X, Y, outcome_transform=None)
    model.load_mcmc_samples(
        {
            "mean": X.new_tensor([-0.1, 0.05, 0.2]),
            "noise": X.new_tensor([0.025, 0.04, 0.055]),
            "lengthscale": X.new_tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            "outputscale": X.new_tensor([0.8, 1.0, 1.2]),
        }
    )
    return model.eval()


def acquisition(model, count, multi):
    candidate = next(model.parameters()).new_tensor(
        [[0.15, 0.7], [0.45, 0.3], [0.8, 0.9]]
    )
    options = (
        {"cost_aware_utility": lambda X, deltas, **kwargs: deltas} if multi else {}
    )
    result = (qMultiFidelityMaxValueEntropy if multi else qMaxValueEntropy)(
        model,
        candidate,
        num_mv_samples=count,
        num_y_samples=8,
        use_gumbel=False,
        **options,
    )
    result.sampler = SobolQMCNormalSampler(torch.Size([8]), seed=13)
    return result


class TestMaxValueEntropyEnsemble(BotorchTestCase):
    def test_implicit_model_batch_matches_independent_components(self):
        model = make_model(self.device)
        for multi in (False, True):
            # Unequal sample counts raised; equal counts silently coupled the
            # maximum-sample axis with the observation-sample axis.
            for count in (4, 8):
                result = acquisition(model, count, multi)
                maxima = torch.linspace(
                    0.9, 1.8, count, dtype=torch.double, device=self.device
                ).unsqueeze(-1) + model.train_inputs[0].new_tensor([0.0, 0.2, 0.4])
                result.posterior_max_values = maxima
                for batch in ((2,), (2, 3)):
                    with self.subTest(multi=multi, count=count, batch=batch):
                        X = (
                            torch.linspace(
                                0.2,
                                0.7,
                                2 * torch.Size(batch).numel(),
                                dtype=torch.double,
                                device=self.device,
                            )
                            .reshape(*batch, 1, 2)
                            .requires_grad_()
                        )
                        parts = []
                        for index in range(3):
                            component = acquisition(
                                ComponentModel(model, index), count, multi
                            )
                            component.posterior_max_values = maxima[
                                :, index : index + 1
                            ]
                            parts.append(component(X))
                        expected = torch.stack(parts).mean(0)
                        value = result(X)
                        self.assertEqual(value.shape, torch.Size(batch))
                        self.assertAllClose(value, expected, atol=1e-8, rtol=1e-8)
                        self.assertAllClose(
                            torch.autograd.grad(value.sum(), X)[0],
                            torch.autograd.grad(expected.sum(), X)[0],
                            atol=2e-7,
                            rtol=2e-7,
                        )


@skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestMaxValueEntropyEnsembleCUDA(TestMaxValueEntropyEnsemble):
    device = torch.device("cuda")
