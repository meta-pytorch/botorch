# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from botorch.models.transforms.input import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.test_helpers import DummyNonScalarizingPosteriorTransform
from botorch_community.models.np_regression import NeuralProcessModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestNeuralProcessModel(unittest.TestCase):
    def initialize(self):
        self.r_hidden_dims = [16, 16]
        self.z_hidden_dims = [32, 32]
        self.decoder_hidden_dims = [16, 16]
        self.x_dim = 2
        self.y_dim = 1
        self.r_dim = 8
        self.z_dim = 8
        self.n_context = 20
        self.model = NeuralProcessModel(
            torch.rand(100, self.x_dim),
            torch.rand(100, self.y_dim),
            self.r_hidden_dims,
            self.z_hidden_dims,
            self.decoder_hidden_dims,
            self.x_dim,
            self.y_dim,
            self.r_dim,
            self.z_dim,
            self.n_context,
        )

    def test_default_hidden_dims(self):
        """Test that default hidden dimensions are used when not provided."""
        x_dim = 2
        y_dim = 1
        r_dim = 8
        z_dim = 8
        n_context = 20

        # Create model without specifying hidden dimensions (use defaults)
        model = NeuralProcessModel(
            train_X=torch.rand(100, x_dim),
            train_Y=torch.rand(100, y_dim),
            r_hidden_dims=None,
            z_hidden_dims=None,
            decoder_hidden_dims=None,
            x_dim=x_dim,
            y_dim=y_dim,
            r_dim=r_dim,
            z_dim=z_dim,
            n_context=n_context,
        )

        # Test that the model works with default dimensions
        output = model(model.train_X, model.train_Y)
        self.assertEqual(output.loc.shape, (80,))

    def test_r_encoder(self):
        self.initialize()
        input = torch.rand(100, self.x_dim + self.y_dim)
        output = self.model.r_encoder(input)
        self.assertEqual(output.shape, (100, self.r_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_z_encoder(self):
        self.initialize()
        input = torch.rand(100, self.r_dim)
        mean, logvar = self.model.z_encoder(input)
        self.assertEqual(mean.shape, (100, self.z_dim))
        self.assertEqual(logvar.shape, (100, self.z_dim))
        self.assertTrue(torch.is_tensor(mean))
        self.assertTrue(torch.is_tensor(logvar))

    def test_decoder(self):
        self.initialize()
        x_pred = torch.rand(100, self.x_dim)
        z = torch.rand(self.z_dim)
        output = self.model.decoder(x_pred, z)
        self.assertEqual(output.shape, (100, self.y_dim))
        self.assertTrue(torch.is_tensor(output))

    def test_sample_z(self):
        self.initialize()
        mu = torch.rand(self.z_dim)
        logvar = torch.rand(self.z_dim)
        samples = self.model.sample_z(mu, logvar, n=5)
        self.assertEqual(samples.shape, (5, self.z_dim))
        self.assertTrue(torch.is_tensor(samples))
        with self.assertRaises(ValueError):
            self.model.sample_z(mu, logvar, n=5, scaler=-1)

    def test_KLD_gaussian(self):
        self.initialize()
        self.model.z_mu_all = torch.rand(self.z_dim)
        self.model.z_logvar_all = torch.rand(self.z_dim)
        self.model.z_mu_context = torch.rand(self.z_dim)
        self.model.z_logvar_context = torch.rand(self.z_dim)
        kld = self.model.KLD_gaussian()
        self.assertGreaterEqual(kld.item(), 0)
        self.assertTrue(torch.is_tensor(kld))
        with self.assertRaises(ValueError):
            self.model.KLD_gaussian(scaler=-1)

    def test_data_to_z_params(self):
        self.initialize()
        mu, logvar = self.model.data_to_z_params(self.model.train_X, self.model.train_Y)
        self.assertEqual(mu.shape, (self.z_dim,))
        self.assertEqual(logvar.shape, (self.z_dim,))
        self.assertTrue(torch.is_tensor(mu))
        self.assertTrue(torch.is_tensor(logvar))

    def test_forward(self):
        self.initialize()
        output = self.model(self.model.train_X, self.model.train_Y)
        # a distribution over the 80 target points
        self.assertEqual(output.loc.shape, (80,))
        self.assertEqual(output.covariance_matrix.shape, (80, 80))
        # no target points if all points are used as context
        model = NeuralProcessModel(torch.rand(10, 2), torch.rand(10, 1), n_context=20)
        self.assertEqual(model(model.train_X, model.train_Y).loc.shape, (0,))

    def test_forward_input_transform(self):
        bounds = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        train_X = 10 * torch.rand(30, 2)
        train_Y = torch.rand(30, 1)
        model = NeuralProcessModel(
            train_X,
            train_Y,
            n_context=10,
            input_transform=Normalize(d=2, bounds=bounds),
        )
        with mock.patch.object(
            model.decoder, "forward", wraps=model.decoder.forward
        ) as mock_decoder:
            model(train_X, train_Y)
        # The target inputs passed to the decoder are transformed exactly once.
        x_pred = mock_decoder.call_args.args[0]
        self.assertEqual(x_pred.shape, (20, 2))
        diff = (x_pred.unsqueeze(-2) - train_X.to(x_pred) / 10).abs().amax(dim=-1)
        self.assertTrue((diff.min(dim=-1).values < 1e-6).all())
        # The latent parameters of all points use the transformed inputs, too.
        z_mu_all, z_logvar_all = model.data_to_z_params(train_X / 10, train_Y)
        self.assertTrue(torch.allclose(model.z_mu_all, z_mu_all))
        self.assertTrue(torch.allclose(model.z_logvar_all, z_logvar_all))

    def test_random_split_context_target(self):
        self.initialize()
        x_c, y_c, x_t, y_t = self.model.random_split_context_target(
            self.model.train_X[:, 0], self.model.train_Y, self.model.n_context
        )
        self.assertEqual(x_c.shape[0], 20)
        self.assertEqual(y_c.shape[0], 20)
        self.assertEqual(x_t.shape[0], 80)
        self.assertEqual(y_t.shape[0], 80)

    def test_posterior(self):
        self.initialize()
        self.model(self.model.train_X, self.model.train_Y)
        identity_posterior = self.model.posterior(
            self.model.train_X,
            observation_noise=True,
            posterior_transform=DummyNonScalarizingPosteriorTransform(),
        )
        posterior = self.model.posterior(self.model.train_X, observation_noise=True)
        self.assertIsInstance(identity_posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        mvn = posterior.mvn
        self.assertEqual(mvn.covariance_matrix.size(), (100, 100))
        self.assertEqual(posterior.mean.shape, (100, 1))
        self.assertEqual(posterior.variance.shape, (100, 1))
        # batched inputs, as used by acquisition functions
        posterior = self.model.posterior(torch.rand(5, 3, self.x_dim))
        self.assertEqual(posterior.mean.shape, (5, 3, 1))
        self.assertEqual(posterior.variance.shape, (5, 3, 1))
        self.assertEqual(posterior.rsample(torch.Size([4])).shape, (4, 5, 3, 1))
        # multiple outputs
        model = NeuralProcessModel(
            torch.rand(30, 2), torch.rand(30, 2), x_dim=2, y_dim=2, n_context=10
        )
        model(model.train_X, model.train_Y)
        posterior = model.posterior(torch.rand(5, 3, 2))
        self.assertEqual(posterior.mean.shape, (5, 3, 2))
        self.assertEqual(posterior.variance.shape, (5, 3, 2))

    def test_transform_inputs(self):
        self.initialize()
        X = torch.rand(5, 3)
        self.assertTrue(torch.equal(self.model.transform_inputs(X), X.to(device)))
        self.assertFalse(
            torch.equal(
                self.model.transform_inputs(X, input_transform=Normalize(d=3)),
                X.to(device),
            )
        )


if __name__ == "__main__":
    unittest.main()
