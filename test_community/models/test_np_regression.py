import unittest

import torch
from botorch.models.transforms.input import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch_community.models.np_regression import NeuralProcessModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity:
    def __call__(self, posterior):
        return posterior


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
        self.assertEqual(output.loc.shape, (80, self.y_dim))

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
            self.model.train_X, observation_noise=True, posterior_transform=Identity()
        )
        posterior = self.model.posterior(self.model.train_X, observation_noise=True)
        self.assertIsInstance(identity_posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        mvn = posterior.mvn
        self.assertEqual(mvn.covariance_matrix.size(), (100, 100, 100))

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
