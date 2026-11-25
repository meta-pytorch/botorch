#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator import to_dense


class TestGaussianMixturePosterior(BotorchTestCase):
    def test_mixture_covariance_matrix(self) -> None:
        # Compare against the covariance of the mixture computed explicitly, where
        # the means are flattened in the order of the rows of the covariances.
        tkwargs = {"device": self.device, "dtype": torch.double}
        num_mcmc_samples, q = 4, 3
        for m, interleaved in ((1, True), (2, True), (2, False)):
            mean = torch.randn(num_mcmc_samples, q, m, **tkwargs)
            root = torch.randn(num_mcmc_samples, q * m, q * m, **tkwargs)
            covar = root @ root.mT + torch.eye(q * m, **tkwargs)
            if m == 1:
                mvn = MultivariateNormal(mean.squeeze(-1), covar)
                flat_mean = mean.squeeze(-1)
            else:
                mvn = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)
                flat_mean = (mean if interleaved else mean.mT).reshape(
                    num_mcmc_samples, q * m
                )
            posterior = GaussianMixturePosterior(distribution=mvn)
            mean_diff = flat_mean - flat_mean.mean(dim=0)
            expected_covar = covar.mean(dim=0) + (
                mean_diff.unsqueeze(-1) * mean_diff.unsqueeze(-2)
            ).mean(dim=0)
            mixture_covar = to_dense(posterior.mixture_covariance_matrix)
            self.assertAllClose(mixture_covar, expected_covar)
            expected_variance = expected_covar.diagonal()
            if interleaved:
                expected_variance = expected_variance.view(q, m)
            else:
                expected_variance = expected_variance.view(m, q).mT
            self.assertAllClose(posterior.mixture_variance, expected_variance)
