#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf
from botorch_community.acquisition.latent_information_gain import LatentInformationGain
from botorch_community.models.np_regression import NeuralProcessModel


class TestLatentInformationGain(unittest.TestCase):
    def setUp(self):
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 3
        self.r_hidden_dims = [16, 16]
        self.z_hidden_dims = [32, 32]
        self.decoder_hidden_dims = [16, 16]
        self.model = NeuralProcessModel(
            torch.rand(10, self.x_dim),
            torch.rand(10, self.y_dim),
            r_hidden_dims=self.r_hidden_dims,
            z_hidden_dims=self.z_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            r_dim=self.r_dim,
            z_dim=self.z_dim,
        )
        self.acquisition_function = LatentInformationGain(self.model)
        self.candidate_x = torch.rand(5, self.x_dim)

    def test_initialization(self):
        self.assertEqual(self.acquisition_function.num_samples, 10)
        self.assertEqual(self.acquisition_function.model, self.model)

    def test_acqf(self):
        bounds = torch.tensor([[0.0] * self.x_dim, [1.0] * self.x_dim])
        q = 3
        raw_samples = 8
        num_restarts = 2

        candidate = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=bounds,
            q=q,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
        )
        self.assertTrue(isinstance(candidate, tuple))
        self.assertEqual(candidate[0].shape, (q, self.x_dim))
        self.assertTrue(torch.all(candidate[1] >= 0))

    def test_model_state_unchanged(self):
        # Evaluating the acquisition function must not change the latent state
        # that the model's posterior uses.
        self.model(self.model.train_X, self.model.train_Y)
        state = [
            self.model.z_mu_all,
            self.model.z_logvar_all,
            self.model.z_mu_context,
            self.model.z_logvar_context,
        ]
        self.acquisition_function(torch.rand(4, 2, self.x_dim))
        self.assertTrue(torch.equal(self.model.z_mu_all, state[0]))
        self.assertTrue(torch.equal(self.model.z_logvar_all, state[1]))
        self.assertTrue(torch.equal(self.model.z_mu_context, state[2]))
        self.assertTrue(torch.equal(self.model.z_logvar_context, state[3]))

    def test_input_transform(self):
        # The encoder only sees input-transformed points.
        bounds = torch.tensor([[0.0] * self.x_dim, [10.0] * self.x_dim])
        model = NeuralProcessModel(
            10 * torch.rand(10, self.x_dim),
            torch.rand(10, self.y_dim),
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            input_transform=Normalize(d=self.x_dim, bounds=bounds),
        )
        with mock.patch.object(
            model.r_encoder, "forward", wraps=model.r_encoder.forward
        ) as mock_r_encoder:
            LatentInformationGain(model, num_samples=2)(
                10 * torch.rand(3, 2, self.x_dim)
            )
        for call in mock_r_encoder.call_args_list:
            self.assertLessEqual(call.args[0][:, : self.x_dim].max().item(), 1.0)

    def test_non_NPR(self):
        model = SingleTaskGP(
            torch.rand(10, self.x_dim, dtype=torch.float64),
            torch.rand(10, self.y_dim, dtype=torch.float64),
        )
        with self.assertRaisesRegex(UnsupportedError, "requires a NeuralProcessModel"):
            LatentInformationGain(model)


if __name__ == "__main__":
    unittest.main()
