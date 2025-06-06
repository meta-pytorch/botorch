#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior,
    MIN_INFERRED_NOISE_LEVEL,
    SQRT2,
    SQRT3,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior


class TestGPyTorchModules(BotorchTestCase):
    def test_get_matern_kernel_with_gamma_prior(self):
        for batch_shape in (None, torch.Size([2])):
            kernel = get_matern_kernel_with_gamma_prior(
                ard_num_dims=2, batch_shape=batch_shape
            )
            self.assertIsInstance(kernel, ScaleKernel)
            self.assertEqual(kernel.batch_shape, batch_shape or torch.Size([]))
            prior = kernel.outputscale_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 2.0)
            self.assertAllClose(prior.rate.item(), 0.15)
            base_kernel = kernel.base_kernel
            self.assertIsInstance(base_kernel, MaternKernel)
            self.assertEqual(base_kernel.batch_shape, batch_shape or torch.Size([]))
            self.assertEqual(base_kernel.ard_num_dims, 2)
            prior = base_kernel.lengthscale_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 3.0)
            self.assertAllClose(prior.rate.item(), 6.0)

    def test_get_gaussian_likelihood_with_gamma_prior(self):
        for batch_shape in (None, torch.Size([2])):
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=batch_shape
            )
            self.assertIsInstance(likelihood, GaussianLikelihood)
            expected_shape = (batch_shape or torch.Size([])) + (1,)
            self.assertEqual(likelihood.raw_noise.shape, expected_shape)
            prior = likelihood.noise_covar.noise_prior
            self.assertIsInstance(prior, GammaPrior)
            self.assertAllClose(prior.concentration.item(), 1.1)
            self.assertAllClose(prior.rate.item(), 0.05)
            constraint = likelihood.noise_covar.raw_noise_constraint
            self.assertIsInstance(constraint, GreaterThan)
            self.assertAllClose(constraint.lower_bound.item(), MIN_INFERRED_NOISE_LEVEL)
            self.assertIsNone(constraint._transform)
            self.assertAllClose(constraint.initial_value.item(), 2.0)

    def test_get_covar_module_with_dim_scaled_prior(self):
        for batch_shape in (None, torch.Size([2])):
            kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=2, batch_shape=batch_shape
            )
            self.assertIsInstance(kernel, RBFKernel)
            self.assertEqual(kernel.batch_shape, batch_shape or torch.Size([]))
            prior = kernel.lengthscale_prior
            self.assertIsInstance(prior, LogNormalPrior)
            self.assertAllClose(prior.loc.item(), SQRT2 + 0.5 * math.log(2))
            self.assertAllClose(prior.scale.item(), SQRT3)
            self.assertIsInstance(kernel, RBFKernel)
            matern_kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=2,
                batch_shape=batch_shape,
                use_rbf_kernel=False,
            )
            self.assertIsInstance(matern_kernel, MaternKernel)

            self.assertEqual(matern_kernel.batch_shape, batch_shape or torch.Size([]))
            self.assertEqual(matern_kernel.ard_num_dims, 2)

    def test_get_gaussian_likelihood_with_log_normal_prior(self):
        for batch_shape in (None, torch.Size([2])):
            likelihood = get_gaussian_likelihood_with_lognormal_prior(
                batch_shape=batch_shape
            )
            self.assertIsInstance(likelihood, GaussianLikelihood)
            expected_shape = (batch_shape or torch.Size([])) + (1,)
            self.assertEqual(likelihood.raw_noise.shape, expected_shape)
            prior = likelihood.noise_covar.noise_prior
            self.assertIsInstance(prior, LogNormalPrior)
            self.assertAllClose(prior.loc.item(), -4.0)
            self.assertAllClose(prior.scale.item(), 1.0)
            constraint = likelihood.noise_covar.raw_noise_constraint
            self.assertIsInstance(constraint, GreaterThan)
            self.assertAllClose(constraint.lower_bound.item(), MIN_INFERRED_NOISE_LEVEL)
            self.assertIsNone(constraint._transform)
