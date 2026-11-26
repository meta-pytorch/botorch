#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from botorch.acquisition.joint_entropy_search import MC_ADD_TERM, qJointEntropySearch
from botorch.utils.testing import BotorchTestCase
from torch.distributions import Normal


class TestJointEntropyConditionalGaussian(BotorchTestCase):
    def test_conditional_normal_standardization(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                tkwargs = {"device": device, "dtype": torch.double}
                latent_var = torch.tensor(0.3, **tkwargs)
                noise_var = torch.tensor(0.7, **tkwargs)
                noisy_var = latent_var + noise_var
                draws = torch.tensor([-1.0, 0.2, 1.0], **tkwargs).requires_grad_()
                zero = torch.zeros(1, 1, 1, **tkwargs)
                noisy_normal = Normal(torch.zeros_like(noisy_var), noisy_var.sqrt())
                posterior = SimpleNamespace(
                    mean=zero, variance=noisy_var.expand_as(zero)
                )
                noiseless = SimpleNamespace(variance=latent_var.expand_as(zero))
                initial = SimpleNamespace(
                    mvn=SimpleNamespace(lazy_covariance_matrix=noisy_var.reshape(1, 1))
                )
                acq = SimpleNamespace(
                    initial_model=SimpleNamespace(posterior=lambda *a, **kw: initial),
                    conditional_model=SimpleNamespace(
                        posterior=lambda *a, **kw: (
                            posterior if kw["observation_noise"] else noiseless
                        )
                    ),
                    posterior_transform=None,
                    optimal_outputs=zero,
                    _compute_monte_carlo_variables=lambda p: (
                        draws.reshape(-1, 1, 1, 1),
                        noisy_normal.log_prob(draws).reshape(-1, 1, 1, 1),
                    ),
                )
                actual = qJointEntropySearch._compute_monte_carlo_information_gain(
                    acq, torch.zeros(1, 1, **tkwargs)
                )
                # Direct Gaussian conditioning: F|Y=y has this mean and variance.
                conditional = Normal(
                    latent_var / noisy_var * draws,
                    (latent_var * noise_var / noisy_var).sqrt(),
                )
                probability = conditional.cdf(torch.zeros_like(draws))
                self.assertAllClose(
                    probability[-1], torch.tensor(0.25634538013096175, **tkwargs)
                )
                ratio = probability / 0.5
                expected = (
                    MC_ADD_TERM
                    + 0.5 * noisy_var.log()
                    + (ratio * (ratio.log() + noisy_normal.log_prob(draws))).mean()
                )
                self.assertAllClose(actual, expected)
                actual_grad = torch.autograd.grad(actual, draws, create_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, draws, create_graph=True)[
                    0
                ]
                self.assertAllClose(actual_grad, expected_grad)
                self.assertAllClose(
                    torch.autograd.grad(actual_grad.sum(), draws)[0],
                    torch.autograd.grad(expected_grad.sum(), draws)[0],
                )
