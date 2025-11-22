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

from typing import Optional

import torch
from botorch.acquisition.bayesian_active_learning import (
    FullyBayesianAcquisitionFunction,
)
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch_community.utils.stat_dist import mvn_hellinger_distance, mvn_kl_divergence
from torch import Tensor


SAMPLE_DIM = -4
DISTANCE_METRICS = {
    "hellinger": mvn_hellinger_distance,
    "kl_divergence": mvn_kl_divergence,
}


class qBayesianVarianceReduction(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
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
        X_pending: Optional[Tensor] = None,
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
        X_pending: Optional[Tensor] = None,
        distance_metric: Optional[str] = "hellinger",
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
                f"Distance metric need to be one of {list(DISTANCE_METRICS.keys())}"
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
