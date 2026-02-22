#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for Self-Correcting Bayesian Optimization [hvarfner2023scorebo]_.

References

.. [hvarfner2023scorebo]
    C. Hvarfner, E. Hellsten, F. Hutter, L. Nardi.
    Self-Correcting Bayesian Optimization thorugh Bayesian Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2023.

Contributor: hvarfner
"""

from __future__ import annotations

import warnings

import torch
from botorch import settings
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.utils import fantasize as fantasize_flag
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch_community.acquisition.bayesian_active_learning import DISTANCE_METRICS
from torch import Tensor

# The lower bound on the CDF value of the max-values
CLAMP_LB = 1e-6


class qSelfCorrectingBayesianOptimization(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        optimal_outputs: Tensor,
        optimal_inputs: Tensor | None = None,
        X_pending: Tensor | None = None,
        distance_metric: str | None = "hellinger",
        posterior_transform: ScalarizedPosteriorTransform | None = None,
    ) -> None:
        r"""Self-correcting Bayesian optimization [hvarfner2023scorebo]_ acquisition
        function. SCoreBO seeks to find accurate hyperparameters during the course
        of optimization by incorporating an active learning-like objective into
        optimization.

        Args:
            model: A fully bayesian model single-outcome model.
            optimal_inputs: A ``num_samples x num_models x d``-dim tensor containing
                the sampled optimal inputs of dimension ``d``.
            optimal_outputs: A ``num_samples x num_models x 1``-dim Tensor containing
                the optimal objective values.
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
            distance_metric: The distance metric used. Defaults to
                "hellinger".
        """

        super().__init__(model=model)
        # To enable fully bayesian GP conditioning, we need to unsqueeze
        # to get num_optima x num_gps unique GPs
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        self.optimal_output_values = (
            posterior_transform.evaluate(Y=self.optimal_outputs, X=None).unsqueeze(-1)
            if posterior_transform
            else self.optimal_outputs
        )
        self.posterior_transform = posterior_transform

        # JES-like version of SCoreBO if optimal inputs are provided
        if optimal_inputs is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with fantasize_flag():
                    with settings.propagate_grads(False):
                        # We must do a forward pass one before conditioning.
                        self.model.posterior(
                            self.model.train_inputs[0], observation_noise=False
                        )

            self.optimal_inputs = optimal_inputs.unsqueeze(-2)
            self.conditional_model = self.model.condition_on_observations(
                X=self.model.transform_inputs(self.optimal_inputs),
                Y=self.optimal_outputs,
                noise=torch.full_like(self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL),
            )
        # otherwise, we do a MES-like variant (which places vastly more emphasis on
        # HP learning as supposed to optimization)
        else:
            self.conditional_model = self.model

        self.normal = torch.distributions.Normal(
            torch.zeros(
                1, device=self.optimal_outputs.device, dtype=self.optimal_outputs.dtype
            ),
            torch.ones(
                1, device=self.optimal_outputs.device, dtype=self.optimal_outputs.dtype
            ),
        )
        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(  # pragma: no cover
                f"Distance metric need to be one of {list(DISTANCE_METRICS.keys())}"
            )
        self.distance = DISTANCE_METRICS[distance_metric]
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        # since we have two MC dims (over models and optima), we need to
        # unsqueeze a second dim to accommodate the posterior pass
        prev_posterior = self.model.posterior(
            X.unsqueeze(MCMC_DIM),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        noiseless_posterior = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        posterior = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )

        marg_mean = prev_posterior.mean.mean(dim=MCMC_DIM, keepdim=True)
        cond_means = posterior.mean
        cond_variances = posterior.variance
        cond_covar = posterior.covariance_matrix

        # the mixture variance is squeezed, need it unsqueezed
        marg_covar = prev_posterior.mixture_covariance_matrix.unsqueeze(MCMC_DIM)
        noiseless_var = noiseless_posterior.variance
        normalized_mvs = (
            self.optimal_output_values - cond_means
        ) / noiseless_var.sqrt()
        cdf_mvs = self.normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(self.normal.log_prob(normalized_mvs))
        mean_truncated = cond_means - noiseless_var.sqrt() * pdf_mvs / cdf_mvs

        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = noiseless_var * (
            1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2)
        )
        # and add the (possibly heteroskedastic) noise
        var_truncated = var_truncated + (cond_variances - noiseless_var)

        # truncating the entire covariance matrix is not trivial, so we assume the
        # truncation is proportional on the off-diags as on the diagonals and scale
        # the covariance accordingly for all elements in the q-batch (if there is one)
        # for q=1, this is equivalent to simply truncating the posterior
        covar_scaling = (var_truncated / cond_variances).sqrt()
        trunc_covar = covar_scaling.transpose(-1, -2) * covar_scaling * cond_covar
        dist = self.distance(mean_truncated, marg_mean, trunc_covar, marg_covar)

        # squeeze output dim and average over optimal samples dim (MCMC_DIM).
        # Model dim is averaged later
        return dist.mean(MCMC_DIM).sum(-1)
