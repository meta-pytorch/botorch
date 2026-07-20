#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for alpha entropy search (AES).

.. [Fernandez2025alpha]
    D. Fernández-Sánchez, E. C. Garrido-Merchán, D. Hernández-Lobato,
    Alpha entropy search for new information-based Bayesian optimization.
    Knowledge-Based Systems, 322, 113612, 2025.
"""

from __future__ import annotations

import warnings

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.models.utils import check_no_nans, fantasize as fantasize_flag
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Normal

MCMC_DIM = -3  # Only relevant if you do Fully Bayesian GPs.
ESTIMATION_TYPES = ["LB"]

# The CDF query cannot be strictly zero in the division
# and this clamping helps assure that it is always positive.
CLAMP_LB = torch.finfo(torch.float32).eps
FULLY_BAYESIAN_ERROR_MSG = (
    "AES is not yet available with Fully Bayesian GPs. Track the issue, "
    "which regards conditioning on a number of optima on a collection "
    "of models, in detail at https://github.com/meta-pytorch/botorch/issues/1680"
)


class qAlphaEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""The batch mode for the acquisition function Alpha Entropy Search,
    it is not supported.

    Alpha entropy search (AES) acquisition function is a generalization of
    joint entropy search (JES). Instead of computing the Kullback-Leibler
    divergence between the joint distribution between the joint distribution
    p({x*, y*}, y) and the marginals p({x*, y*}) and p(y), it computes the
    Amari's alpha divergence between the joint distribution p({x*, y*}, y)
    and the marginals p({x*, y*}) and p(y).

    See [Tu2022joint]_ for a discussion on the estimation procedure.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        alpha: int = 0.5,
        eps: float = 1e-6,
    ) -> None:
        r"""Alpha entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A ``num_samples x d``-dim tensor containing the sampled
                optimal inputs of dimension ``d``. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            optimal_outputs: A ``num_samples x 1``-dim Tensor containing the optimal
                set of objectives of dimension ``1``.
            condition_noiseless: Whether to condition on noiseless optimal observations
                ``f*`` [Hvarfner2022joint]_ or noisy optimal observations ``y*``
                [Tu2022joint]_. These are sampled identically, so this only controls
                the fashion in which the GP is reshaped as a result of conditioning
                on the optimum.
            posterior_transform: PosteriorTransform to negate or scalarize the output.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
                Since AES only supports ``q=1``, any non-empty ``X_pending`` will
                raise an error once the acquisition function is evaluated, as it is
                concatenated onto the ``q`` dimension of the evaluation points.
            estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB").
                Monte Carlo ("MC") estimation is not currently supported.
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate (if supported).
            alpha: Hyper-parameter of the acquisition function. Is the alpha
                parameter of the Amari's alpha divergence. In the limit of alpha=1
                gives direct KL-divergence.
            eps: Epsilon value to prevent alpha evaluated strictly at 1.0 or 0.0.
        """
        super().__init__(model=model)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        # To enable fully bayesian GP conditioning, we need to unsqueeze
        # to get num_optima x num_gps unique GPs

        # inputs come as num_optima_per_model x (num_models) x d
        # but we want it four-dimensional in the Fully bayesian case,
        # and three-dimensional otherwise.
        self.optimal_inputs = optimal_inputs.unsqueeze(-2)
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        self.optimal_output_values = (
            posterior_transform.evaluate(
                Y=self.optimal_outputs, X=self.optimal_inputs
            ).unsqueeze(-1)
            if posterior_transform
            else self.optimal_outputs
        )
        self.posterior_transform = posterior_transform

        self.num_samples = optimal_inputs.shape[0]
        self.condition_noiseless = condition_noiseless
        self.initial_model = model

        # Here, the optimal inputs have shapes num_optima x [num_models if FB] x 1 x D
        # and the optimal outputs have shapes num_optima x [num_models if FB] x 1 x 1
        # The third dimension equaling 1 is required to get one optimum per model,
        # which raises a BotorchTensorDimensionWarning.
        if isinstance(model, FullyBayesianSingleTaskGP):
            raise NotImplementedError(FULLY_BAYESIAN_ERROR_MSG)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with fantasize_flag():
                with settings.propagate_grads(False):
                    # We must do a forward pass one before conditioning.
                    self.initial_model.posterior(
                        self.optimal_inputs[:1], observation_noise=False
                    )

                # This equates to the JES version proposed by Hvarfner et. al.
                if self.condition_noiseless:
                    opt_noise = torch.full_like(
                        self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL
                    )
                    # conditional (batch) model of shape (num_models)
                    # x num_optima_per_model
                    self.conditional_model = (
                        self.initial_model.condition_on_observations(
                            X=self.initial_model.transform_inputs(self.optimal_inputs),
                            Y=self.optimal_outputs,
                            noise=opt_noise,
                        )
                    )
                else:
                    self.conditional_model = (
                        self.initial_model.condition_on_observations(
                            X=self.initial_model.transform_inputs(self.optimal_inputs),
                            Y=self.optimal_outputs,
                        )
                    )

        if estimation_type not in ESTIMATION_TYPES:
            raise ValueError(
                f"Estimation type {estimation_type} is not valid. "
                f"Please specify any of {ESTIMATION_TYPES}"
            )
        self.estimation_type = estimation_type
        self.set_X_pending(X_pending)

        self.eps = eps
        # Since Amari's alpha divergence is not defined when alpha=1.0 or alpha=0.0
        # we need to avoid computing the divergence for those values. So, we increase
        # or reduce the alpha by an epsilon in those cases
        self.alpha = (alpha - self.eps) if alpha == 1.0 else alpha
        self.alpha = self.eps if self.alpha == 0.0 else self.alpha

    @concatenate_pending_points
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qAlphaEntropySearch at the design points ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape``
                t-batches with ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        if self.estimation_type == "LB":
            res = self._compute_lower_bound_information_gain(X)
        else:
            raise ValueError(
                f"Estimation type {self.estimation_type} is not valid. "
                f"Please specify any of {ESTIMATION_TYPES}"
            )
        return res

    def g_eta_factor(self, mean: Tensor, var: Tensor) -> Tensor:
        return 0.5 * torch.log(2 * torch.pi * var) + 0.5 * mean**2 / var

    def _compute_lower_bound_information_gain(
        self, X: Tensor, return_parts: bool = False
    ) -> Tensor:
        r"""Evaluates the lower bound information gain at the design points ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape``
                t-batches with ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        initial_posterior = self.initial_model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )

        # we store the predicted mean and variance of the current predictive
        # distribution
        mean_pred = initial_posterior.mean.unsqueeze(MCMC_DIM)
        var_pred = initial_posterior.variance.unsqueeze(MCMC_DIM)

        # need to check if there is a two-dimensional batch shape -
        # the sampled optima appear in the dimension right after
        batch_shape = X.shape[:-2]
        sample_dim = len(batch_shape)

        # Compute the mixture mean and variance
        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        # we store the noisy predicted mean and variance of the conditional model
        mean_cond = posterior_m.mean
        var_cond = posterior_m.variance

        check_no_nans(var_cond)

        noiseless_posterior_m = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )

        # we store the noiseless predicted variance of the conditional model
        noiseless_var = noiseless_posterior_m.variance

        check_no_nans(noiseless_var)

        # get stdv of noiseless variance
        noiseless_stdv = noiseless_var.sqrt()
        # batch_shape x 1
        normal = Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        noiseless_normalized_mvs = (self.optimal_output_values - mean_cond) / noiseless_stdv
        noiseless_cdf_mvs = normal.cdf(noiseless_normalized_mvs).clamp_min(CLAMP_LB)
        noiseless_pdf_mvs = torch.exp(normal.log_prob(noiseless_normalized_mvs))

        noiseless_ratio = noiseless_pdf_mvs / noiseless_cdf_mvs

        noiseless_var_cond_trunc = noiseless_var * (
            1 - (noiseless_normalized_mvs + noiseless_ratio) * noiseless_ratio
        ).clamp_min(CLAMP_LB)

        var_cond_trunc = noiseless_var_cond_trunc + (var_cond - noiseless_var) - 1e-8
        mean_cond_trunc = mean_cond - noiseless_stdv * noiseless_ratio

        # We compute the natural parameters of the distributions to compute the
        # integral analytically

        v3 = 1.0 / (self.alpha * (1.0 / var_cond_trunc - 1.0 / var_pred)).clamp_min(
            CLAMP_LB
        )
        m3 = v3 * (
            self.alpha * (mean_cond_trunc / var_cond_trunc - mean_pred / var_pred)
        )

        g_m1v1 = self.g_eta_factor(mean_pred, var_pred)
        g_m2v2 = self.g_eta_factor(mean_cond_trunc, var_cond_trunc)

        v_prod_pred_dist_3 = 1.0 / (1.0 / v3 + 1.0 / var_pred).clamp_min(CLAMP_LB)
        m_prod_pred_dist_3 = v_prod_pred_dist_3 * (m3 / v3 + mean_pred / var_pred)

        # We compute the integral: int (p(y|x*) / p(y))^alpha p(y) d y

        integral_val = torch.exp(
            -self.alpha * g_m2v2
            + self.alpha * g_m1v1
            - g_m1v1
            + self.g_eta_factor(m_prod_pred_dist_3, v_prod_pred_dist_3)
        )

        return (1.0 / (self.alpha * (1.0 - self.alpha))) - (
            1.0 / (self.alpha * (1.0 - self.alpha))
        ) * integral_val.mean(dim=sample_dim).squeeze(-1).squeeze(-1)
