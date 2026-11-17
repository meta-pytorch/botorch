#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestMESConstantControlVariate(BotorchTestCase):
    def test_zero_variance_control_has_finite_derivatives(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for sample_count in (1, 8):
                with self.subTest(device=device, sample_count=sample_count):
                    tkwargs = {"dtype": torch.double, "device": device}
                    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], **tkwargs)
                    Y = (5.3 * X).sin()
                    model = SingleTaskGP(X, Y, outcome_transform=None)
                    acq = qMaxValueEntropy(
                        model, X, use_gumbel=False, num_y_samples=sample_count
                    )
                    acq.sampler = SobolQMCNormalSampler(
                        torch.Size((sample_count,)), seed=13
                    )
                    acq.sampler.base_samples = X.new_zeros((sample_count, 1, 1, 1))
                    acq.posterior_max_values = X.new_tensor(
                        [[0.75], [1.0], [1.2], [1.4]]
                    )
                    query = X.new_tensor([[0.23]], requires_grad=True)
                    value = acq(query)
                    gradient = torch.autograd.grad(
                        value.sum(), query, create_graph=True
                    )[0]
                    hessian = torch.autograd.grad(gradient.sum(), query)[0]
                    for tensor in (value, gradient, hessian):
                        self.assertTrue(torch.isfinite(tensor).all())
