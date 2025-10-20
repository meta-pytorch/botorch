import unittest

import torch
from botorch.models import SingleTaskGP
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

    def test_non_NPR(self):
        self.model = SingleTaskGP(
            torch.rand(10, self.x_dim, dtype=torch.float64),
            torch.rand(10, self.y_dim, dtype=torch.float64),
        )
        self.acquisition_function = LatentInformationGain(self.model)
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


if __name__ == "__main__":
    unittest.main()
