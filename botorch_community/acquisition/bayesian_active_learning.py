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

import torch
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
    qBayesianActiveLearningByDisagreement,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import (
    AbstractFullyBayesianSingleTaskGP,
    MCMC_DIM,
    SaasFullyBayesianSingleTaskGP,
)
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
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


class qConditionalHyperparameterInformationGain(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: AbstractFullyBayesianSingleTaskGP,
        mc_points: Tensor,
        X_pending: Tensor | None = None,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        """Conditional hyperparameter information gain for active learning.

        Computes the mutual information between mc_points and the
        hyperparameters of the model, conditioned on the candidates currently under
        evaluation. Used as a helper to compute beta in
        qHyperparameterInformedPredictiveExploration, but does not have empirically
        demonstrated stand-alone utility. The calculation of the acquisition function
        is very similar to qBayesianActiveLearningByDisagreement, but is computed on
        the mc_points, with the model conditioned on the points under consideration.

        Args:
            model: A fully bayesian model (FullyBayesianSingleTaskGP).
            mc_points: A `N x d` tensor of points to use for MC-integrating the
                posterior entropy (test set). Usually, these are qMC samples on
                the whole design space.
            X_pending: A `m x d`-dim Tensor of `m` design points.
            sampler: The sampler used for drawing samples to approximate the entropy
                of the Gaussian Mixture posterior.
            posterior_transform: A PosteriorTransform. If provided, the posterior
                obtained from the model will be transformed before computing the
                acquisition value.
        """
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        self.mc_points = mc_points
        self.set_X_pending(X_pending)
        self.posterior_transform = posterior_transform

        # Calling once to be able to condition on observations later
        model.posterior(model.train_inputs[0])

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        # Start by Conditioning on the candidate point X.
        # NOTE: Multi-task GPs are not supported. This is a limitation
        # in the current implementation of `condition_on_observations`.
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        noise = (
            posterior.variance
            - self.model.posterior(
                X,
                observation_noise=False,
                posterior_transform=self.posterior_transform,
            ).variance
        )

        # The posterior entropy is independent of the actual outcome Y, so we can
        # simply condition on a dummy observation (e.g. the posterior mean).
        cond_Y = posterior.mean
        cond_X = X.unsqueeze(-3).expand(*[cond_Y.shape[:-1] + X.shape[-1:]])
        conditional_model = self.model.condition_on_observations(
            X=cond_X,
            Y=cond_Y,
            noise=noise,
        )

        # This should not be computed as an entropy over one multivariate normal,
        # but the expectation of entropies over individual gaussians. Thus, we
        # unsqueeze to ensure that this happens.
        conditional_posterior = conditional_model.posterior(
            self.mc_points.unsqueeze(-2).unsqueeze(-2), observation_noise=True
        )
        samples = self.get_posterior_samples(conditional_posterior)
        prev_samples = samples.unsqueeze(0).transpose(0, MCMC_DIM).squeeze(-1)
        component_sample_probs = conditional_posterior.mvn.log_prob(prev_samples).exp()

        # average over mixture components
        mixture_sample_probs = component_sample_probs.mean(dim=-1, keepdim=True)

        # this is the average over the model and sample dim
        prev_entropy = -mixture_sample_probs.log().mean(dim=[0, 1])

        # the posterior entropy is an average entropy over gaussians, so no mixture
        post_entropy = -conditional_posterior.mvn.log_prob(samples.squeeze(-1)).mean(0)
        hyperparameter_info_gain = (prev_entropy - post_entropy).mean(0)
        return hyperparameter_info_gain


class qHyperparameterInformedPredictiveExploration(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: AbstractFullyBayesianSingleTaskGP,
        mc_points: Tensor,
        bounds: Tensor,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        beta_tuning_samples: int = 32,
    ) -> None:
        """Hyperparameter-informed Predictive Exploration acquisition function.

        This acquisition function combines the mutual information between the
        subsequent queries and a test set (Expected Predictive Information Gain) with
        mutual information between observations and hyperparameters (BALD), weighted
        by a tuning factor. This balances exploration of the design space with
        reduction of hyperparameter uncertainty.

        The acquisition function is computed as:
            beta * BALD + TSIG
        where beta is automatically tuned.

        Args:
            model: A fully bayesian model (FullyBayesianSingleTaskGP).
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
            beta_tuning_samples: The number of sobol samples to use for tuning
                beta. Does not use proper acquisition function optimization due to
                a memory leak in settings.propagate_grads, outlined here:
                (https://github.com/meta-pytorch/botorch/issues/2728)
        """
        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be a 2-dimensional tensor, but got shape "
                f"{mc_points.shape}"
            )
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        self.set_X_pending(X_pending)
        self.register_buffer("mc_points", mc_points)
        self.register_buffer("bounds", bounds)
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        self.num_beta_tuning_samples = beta_tuning_samples

        self._tuning_factor = None
        self._tuning_factor_q = 0

    def _compute_tuning_factor(self, q: int) -> None:
        """Compute the tuning factor beta for weighting BALD vs TSIG."""
        tuning_acq = qConditionalHyperparameterInformationGain(
            model=self.model,
            mc_points=self.mc_points,
            sampler=self.sampler,
            X_pending=self.X_pending,
            posterior_transform=self.posterior_transform,
        )
        tuning_samples = draw_sobol_samples(
            n=self.num_beta_tuning_samples, bounds=self.bounds, q=q
        )
        # Optimize the acquisition function over the tuning samples
        # NOTE: This is not a proper acquisition function optimization,
        # but gradient-based optimization of the tuning factor has not been
        #  found to be important (and is not possible due to a memory leak).
        acq_evals = torch.cat(
            [
                tuning_acq(sample_split)
                for sample_split in tuning_samples.split(split_size=16, dim=0)
            ]
        )
        self._tuning_factor = acq_evals.max().item()
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
