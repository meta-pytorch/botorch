#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Bayesian active learning, including entropy reduction,
 (batch)-BALD, Bayesian Query-By-Comittee and Statistical distance-based
 Active Learning. See [mackay1992alm]_, [houlsby2011bald]_ [kirsch2011batchbald]_,
 [riis2022fbgp]_ and [Hvarfner2023scorebo]_.

References

.. [mackay1992alm]
    D. MacKay.
    Information-Based Objective Functions for Active Data Selection.
    Neural Computation, 1992.
.. [kirsch2011batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.
.. [riis2022fbgp]
    C. Riis, F. Antunes, F. Hüttel, C. Azevedo, F. Pereira.
    Bayesian Active Learning with Fully Bayesian Gaussian Processes.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.

Contributor: hvarfner
"""

from __future__ import annotations

import math

from typing import Literal

import torch
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
    qBayesianActiveLearningByDisagreement,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)

from botorch_community.utils.stat_dist import mvn_hellinger_distance, mvn_kl_divergence
from torch import Tensor


SAMPLE_DIM = -4
TWO_PI_E = 2 * math.pi * math.e
DISTANCE_METRICS = {
    "hellinger": mvn_hellinger_distance,
    "kl_divergence": mvn_kl_divergence,
}


class qBayesianVarianceReduction(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Tensor | None = None,
    ) -> None:
        """Global variance reduction with fully Bayesian hyperparameter treatment by
        [mackay1992alm]_.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        res = torch.logdet(posterior.mixture_covariance_matrix).exp()

        # the MCMC dim is averaged out in the mixture postrior,
        # so the result needs to be unsqueezed for the averaging
        # in the decorator
        return res.unsqueeze(-1)


class qBayesianQueryByComittee(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Tensor | None = None,
    ) -> None:
        """
        Bayesian Query-By-Comittee [riis2022fbgp]_, which minimizes the variance
        of the mean in the posterior.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        posterior_mean = posterior.mean
        marg_mean = posterior.mixture_mean.unsqueeze(MCMC_DIM)
        mean_diff = posterior_mean - marg_mean
        covar_of_mean = torch.matmul(mean_diff, mean_diff.transpose(-1, -2))

        res = torch.logdet(covar_of_mean).exp()
        return torch.nan_to_num(res, 0)


class qStatisticalDistanceActiveLearning(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Tensor | None = None,
        distance_metric: str = "hellinger",
    ) -> None:
        """Batch implementation of SAL [hvarfner2023scorebo]_, which minimizes
        discrepancy in the posterior predictive as measured by a statistical
        distance (or semi-metric). Computed by an (approx.) lower bound estimate.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            distance_metric: The distance metric used. Defaults to
                "hellinger".
        """
        super().__init__(model)
        self.set_X_pending(X_pending)
        # the default number of MC samples (512) are too many when doing FB modeling.
        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(
                f"Distance metric need to be one of " f"{list(DISTANCE_METRICS.keys())}"
            )
        self.distance = DISTANCE_METRICS[distance_metric]

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        cond_means = posterior.mean
        marg_mean = posterior.mixture_mean.unsqueeze(MCMC_DIM)
        cond_covar = posterior.covariance_matrix

        # the mixture variance is squeezed, need it unsqueezed
        marg_covar = posterior.mixture_covariance_matrix.unsqueeze(MCMC_DIM)
        dist = self.distance(cond_means, marg_mean, cond_covar, marg_covar)

        # squeeze output dim - batch dim computed and reduced inside of dist
        # MCMC dim is averaged in decorator
        return dist.squeeze(-1)


class qExpectedPredictiveInformationGain(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        mc_points: Tensor,
        X_pending: Tensor | None = None,
    ) -> None:
        """Expected predictive information gain for active learning.

        Computes the mutual information between candidate queries and a test set
        (typically MC samples over the design space).

        Args:
            model: A fully bayesian model (SaasFullyBayesianSingleTaskGP).
            mc_points: A `N x d` tensor of points to use for MC-integrating the
                posterior entropy (test set).
            X_pending: A `m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be a 2-dimensional tensor, but got shape "
                f"{mc_points.shape}"
            )
        self.register_buffer("mc_points", mc_points)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate test set information gain.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of input points.

        Returns:
            A Tensor of information gain values.
        """
        # Get the posterior for the candidate points
        posterior = self.model.posterior(X, observation_noise=True)
        noise = (
            posterior.variance
            - self.model.posterior(X, observation_noise=False).variance
        )
        cond_Y = posterior.mean

        # Condition the model on the candidate observations
        cond_X = X.unsqueeze(-3).expand(*cond_Y.shape[:-1], *X.shape[-1:])
        conditional_model = self.model.condition_on_observations(
            X=cond_X,
            Y=cond_Y,
            noise=noise,
        )

        # Evaluate posterior variance at test set with and without conditioning
        uncond_var = self.model.posterior(
            self.mc_points, observation_noise=True
        ).variance
        cond_var = conditional_model.posterior(
            self.mc_points, observation_noise=True
        ).variance

        # Compute information gain as reduction in entropy
        prev_entropy = torch.log(uncond_var * TWO_PI_E).sum(-1) / 2
        post_entropy = torch.log(cond_var * TWO_PI_E).sum(-1) / 2
        return (prev_entropy - post_entropy).mean(-1)


class qHyperparameterInformedPredictiveExploration(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        mc_points: Tensor,
        bounds: Tensor,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        num_samples: int = 512,
        beta: float | None = None,
        beta_tuning_method: Literal["sobol", "optimize"] = "sobol",
    ) -> None:
        """Hyperparameter-informed Predictive Exploration acquisition function.

        This acquisition function combines the mutual information between the
        subsequent queries and a test set (predictive information gain) with the
        mutual information between observations and hyperparameters (BALD), weighted
        by a tuning factor. This balances exploration of the design space with
        reduction of hyperparameter uncertainty.

        The acquisition function is computed as:
            beta * BALD + TSIG
        where beta is either provided or automatically tuned.

        Args:
            model: A fully bayesian model (SaasFullyBayesianSingleTaskGP).
            mc_points: A `N x d` tensor of points to use for MC-integrating the
                posterior entropy (test set). Usually, these are qMC samples on
                the whole design space.
            bounds: A `2 x d` tensor of bounds for the design space, used for
                beta tuning.
            sampler: The sampler used for drawing samples to approximate the entropy
                of the Gaussian Mixture posterior. If None, uses default sampler.
            posterior_transform: A PosteriorTransform. If provided, the posterior
                obtained from the model will be transformed before computing the
                acquisition value.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for evaluation but have not yet been observed.
            num_samples: Number of samples to use for MC estimation of entropy.
            beta: Fixed tuning factor. If None, it will be automatically computed
                on the first forward pass based on the batch size q.
            beta_tuning_method: Method for tuning beta. Options are "optimize"
                (optimize acquisition function to find beta) or "sobol" (use sobol
                samples). Only used when beta is None.
        """
        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be a 2-dimensional tensor, but got shape "
                f"{mc_points.shape}"
            )
        super().__init__(model=model)
        MCSamplerMixin.__init__(self)
        self.set_X_pending(X_pending)
        self.num_samples = num_samples
        self.beta_tuning_method = beta_tuning_method
        self.register_buffer("mc_points", mc_points)
        self.register_buffer("bounds", bounds)
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        self._tuning_factor: float | None = beta
        self._tuning_factor_q: int | None = None

    def _compute_tuning_factor(self, q: int) -> None:
        """Compute the tuning factor beta for weighting BALD vs TSIG."""
        if self.beta_tuning_method == "sobol":
            draws = draw_sobol_samples(
                bounds=self.bounds,
                q=q,
                n=1,
            ).squeeze(0)
            # Compute the ratio at sobol samples
            tsig_val = qExpectedPredictiveInformationGain.forward(
                self,
                draws,
            )
            bald_val = qBayesianActiveLearningByDisagreement.forward(self, draws)
            self._tuning_factor = (tsig_val / (bald_val + 1e-8)).mean().item()
        elif self.beta_tuning_method == "optimize":
            # Optimize to find the best tuning factor
            bald_acqf = qBayesianActiveLearningByDisagreement(
                model=self.model,
                sampler=self.sampler,
            )
            _, bald_val = optimize_acqf(
                bald_acqf,
                bounds=self.bounds,
                q=q,
                num_restarts=1,
                raw_samples=128,
                options={"batch_limit": 16},
            )
            self._tuning_factor = bald_val.mean().item()
        self._tuning_factor_q = q

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function at X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of input points.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values.
        """
        q = X.shape[-2]
        # Compute tuning factor if not set or if q has changed
        if self._tuning_factor is None or self._tuning_factor_q != q:
            self._compute_tuning_factor(q)

        tsig = qExpectedPredictiveInformationGain.forward(self, X)
        bald = qBayesianActiveLearningByDisagreement.forward(self, X)
        # Since both acquisition functions are averaged over the ensemble,
        # we do not average over the ensemble again here.
        return self._tuning_factor * bald + tsig
