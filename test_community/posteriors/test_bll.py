#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from botorch.utils.testing import BotorchTestCase
from botorch_community.models.vblls import VBLLModel
from botorch_community.posteriors.bll_posterior import BLLPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import to_linear_operator
from torch.distributions import Normal


def _get_vbll_model(num_inputs: int = 2, num_hidden: int = 3, **tkwargs) -> VBLLModel:
    test_backbone = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, num_hidden),
    ).to(**tkwargs)
    model = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)
    return model


class TestBLLPosterior(BotorchTestCase):
    def setUp(self):
        """Set up common test components."""
        super().setUp()
        self.tkwargs = {"dtype": torch.float64, "device": self.device}
        self.X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], **self.tkwargs)
        self.num_X = self.X.shape[0]
        self.output_dim = 1
        self.model = _get_vbll_model(num_inputs=2, **self.tkwargs)

        # Create a proper MultivariateNormal distribution for GPyTorchPosterior
        mean = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)

        # Create a positive-definite covariance matrix
        cov_diag = torch.tensor([0.1, 0.2, 0.3], **self.tkwargs)
        cov = torch.diag(cov_diag)

        self.distribution = MultivariateNormal(mean, cov)

        self.bll_posterior = BLLPosterior(
            distribution=self.distribution,
            model=self.model,
            X=self.X,
            output_dim=self.model.num_outputs,
        )

    def test_initialization(self):
        """Test that BLLPosterior initializes correctly."""
        self.assertEqual(self.bll_posterior.distribution, self.distribution)
        self.assertEqual(self.bll_posterior.model, self.model)
        self.assertEqual(self.bll_posterior.output_dim, self.output_dim)
        self.assertTrue(torch.equal(self.bll_posterior.X, self.X))

    def test_rsample_default(self):
        """Test rsample method with default sample_shape."""
        samples = self.bll_posterior.rsample()
        expected_shape = torch.Size([1, self.num_X, 1])
        self.assertEqual(samples.shape, expected_shape)

    def test_rsample_with_explicit_shape(self):
        """Test rsample method with custom sample_shape."""
        for sample_shape in ([3], [2, 3]):
            samples = self.bll_posterior.rsample(sample_shape=torch.Size(sample_shape))
            expected_shape = torch.Size(sample_shape + [self.num_X, 1])
            self.assertEqual(samples.shape, expected_shape)

    def test_mean(self):
        """Test mean property."""
        mean = self.bll_posterior.mean
        expected_shape = torch.Size([self.num_X, 1])
        self.assertEqual(mean.shape, expected_shape)

    def test_variance(self):
        """Test variance property."""
        variance = self.bll_posterior.variance
        expected_shape = torch.Size([self.num_X, 1])
        self.assertEqual(variance.shape, expected_shape)

    def test_device(self):
        """Test device property."""
        self.assertEqual(self.bll_posterior.device, self.device)

    def test_dtype(self):
        """Test dtype property."""
        self.assertEqual(self.bll_posterior.dtype, self.tkwargs["dtype"])

    def test_mvn_property(self):
        """Test the inherited mvn property."""
        self.assertIs(self.bll_posterior.mvn, self.distribution)

    def test_base_sample_shape(self):
        """Test the inherited base_sample_shape property."""
        expected_shape = (
            self.distribution.batch_shape + self.distribution.base_sample_shape
        )
        self.assertEqual(self.bll_posterior.base_sample_shape, expected_shape)

    def test_batch_range(self):
        """Test the inherited batch_range property."""
        self.assertEqual(self.bll_posterior.batch_range, (0, -1))

    def test_extended_shape(self):
        """Test the inherited _extended_shape method."""
        sample_shape = torch.Size([4, 2])
        extended_shape = self.bll_posterior._extended_shape(sample_shape)
        base_shape = (
            self.bll_posterior.batch_shape
            + self.bll_posterior.event_shape
            + torch.Size([self.output_dim])
        )
        expected_shape = sample_shape + base_shape
        self.assertEqual(extended_shape, expected_shape)

    def test_rsample_from_base_samples(self):
        """Test the inherited rsample_from_base_samples method."""
        sample_shapes = [torch.Size([4]), torch.Size([2, 3])]
        for sample_shape in sample_shapes:
            base_sample_shape = self.bll_posterior.base_sample_shape
            base_samples = torch.randn(sample_shape + base_sample_shape, **self.tkwargs)

            samples = self.bll_posterior.rsample_from_base_samples(
                sample_shape=sample_shape, base_samples=base_samples
            )
            expected_shape = self.bll_posterior._extended_shape(sample_shape)
            self.assertEqual(samples.shape, expected_shape)

    def test_quantile(self):
        """Test the inherited quantile method."""
        value_single = torch.tensor(0.5, **self.tkwargs)
        quantiles_single = self.bll_posterior.quantile(value_single)
        self.assertTrue(torch.allclose(quantiles_single, self.bll_posterior.mean))

        value_multi = torch.tensor([0.25, 0.5, 0.75], **self.tkwargs)
        quantiles_multi = self.bll_posterior.quantile(value_multi)
        marginal = Normal(
            loc=self.bll_posterior.mean, scale=self.bll_posterior.variance.sqrt()
        )
        expected_quantiles = marginal.icdf(
            value_multi.view(-1, 1, 1)
        )  # [3] -> [3, 1, 1]
        self.assertEqual(
            quantiles_multi.shape, torch.Size([3, self.num_X, self.output_dim])
        )
        self.assertTrue(torch.allclose(quantiles_multi, expected_quantiles))

    def test_density(self):
        """Test the inherited density method."""
        marginal = Normal(
            loc=self.bll_posterior.mean, scale=self.bll_posterior.variance.sqrt()
        )
        value_single = torch.tensor(1.0, **self.tkwargs)
        density_single = self.bll_posterior.density(value_single)
        expected_density_single = marginal.log_prob(value_single).exp()
        self.assertTrue(torch.allclose(density_single, expected_density_single))

        value_multi = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        density_multi = self.bll_posterior.density(value_multi)
        expected_density_multi = marginal.log_prob(
            value_multi.view(-1, 1, 1)
        ).exp()  # [3] -> [3, 1, 1]

        self.assertTrue(torch.allclose(density_multi, expected_density_multi))
        self.assertEqual(
            density_multi.shape, torch.Size([3, self.num_X, self.output_dim])
        )

    def test_multi_output(self):
        """Test with multiple output dimensions."""
        output_dim = 2

        # Create a model with multiple outputs
        backbone = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 3),
        ).to(**self.tkwargs)

        model = VBLLModel(backbone=backbone, hidden_features=3, out_features=output_dim)

        # Create mean tensor with shape [batch_size, output_dim]
        mean = torch.rand(3, 2, **self.tkwargs)
        variance = 1 + torch.rand(3, 2, **self.tkwargs)
        covar = variance.view(-1).diag()
        distribution = MultitaskMultivariateNormal(mean, to_linear_operator(covar))
        bll_posterior = BLLPosterior(
            distribution=distribution, model=model, X=self.X, output_dim=output_dim
        )

        # Test rsample
        samples = bll_posterior.rsample()
        expected_shape = torch.Size(
            [1, self.num_X, output_dim]
        )  # [samples, batch, output_dim]
        self.assertEqual(samples.shape, expected_shape)

        # Test rsample with explicit shape
        for sample_shape in ([3], [2, 3]):
            samples = bll_posterior.rsample(sample_shape=torch.Size(sample_shape))

            # Check shape: should be [sample_shape, num_X, output_dim]
            expected_shape = torch.Size(sample_shape + [self.num_X, output_dim])
            self.assertEqual(samples.shape, expected_shape)

        # Test mean with multi-output
        multi_mean = bll_posterior.mean
        expected_mean_shape = torch.Size(
            [self.num_X, output_dim]
        )  # [batch, output_dim]
        self.assertEqual(multi_mean.shape, expected_mean_shape)

        # check mean values
        self.assertTrue(torch.allclose(multi_mean, mean))

        # Test variance with multi-output
        multi_variance = bll_posterior.variance
        expected_var_shape = torch.Size([self.num_X, output_dim])  # [batch, output_dim]
        self.assertEqual(multi_variance.shape, expected_var_shape)

        # check variance values
        expected_variance = variance
        self.assertTrue(torch.allclose(multi_variance, expected_variance))

        self.assertEqual(bll_posterior.batch_range, (0, -2))
        value = torch.tensor(0.5, **self.tkwargs)
        quantiles = bll_posterior.quantile(value)
        self.assertTrue(torch.allclose(quantiles, bll_posterior.mean))
