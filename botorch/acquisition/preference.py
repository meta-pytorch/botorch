#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Preference acquisition functions. This includes:
Analytical EUBO acquisition function as introduced in [Lin2022preference]_
and its MC-based generalization qEUBO as proposed in [Astudillo2023qeubo]_.

.. [Astudillo2023qeubo]
    Astudillo, R., Lin, Z.J., Bakshy, E. and Frazier, P.I. qEUBO: A Decision-Theoretic
    Acquisition Function for Preferential Bayesian Optimization. International
    Conference on Artificial Intelligence and Statistics (AISTATS), 2023.

.. [Lin2022preference]
    Lin, Z.J., Astudillo, R., Frazier, P.I. and Bakshy, E. Preference Exploration
    for Efficient Bayesian Optimization with Multiple Outcomes. International
    Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

.. [Houlsby2011bald]
    Houlsby, N., Huszár, F., Ghahramani, Z. and Lengyel, M.
    Bayesian Active Learning for Gaussian Process Classification.
    NIPS Workshop on Bayesian optimization, experimental design and bandits:
    Theory and applications, 2011.
"""

from __future__ import annotations

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Bernoulli, Normal

SIGMA_JITTER = 1e-8


class AnalyticExpectedUtilityOfBestOption(AnalyticAcquisitionFunction):
    r"""Analytic Preferential Expected Utility of Best Options, i.e., Analytical EUBO"""

    def __init__(
        self,
        pref_model: Model,
        outcome_model: DeterministicModel | None = None,
        previous_winner: Tensor | None = None,
    ) -> None:
        r"""Analytic implementation of Expected Utility of the Best Option under the
        Laplace model (assumes a PairwiseGP is used as the preference model) as
        proposed in [Lin2022preference]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating EUBO on
                the parameter space. When used with ``OneSamplePosteriorDrawModel``,
                we are obtaining EUBO-zeta as described in [Lin2022preference]_.
            previous_winner: Tensor representing the previous winner in the Y space.
        """
        super().__init__(model=pref_model)
        # ensure the model is in eval mode
        self.add_module("outcome_model", outcome_model)
        self.register_buffer("previous_winner", previous_winner)

        tkwargs = {
            "dtype": pref_model.datapoints.dtype,
            "device": pref_model.datapoints.device,
        }
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, **tkwargs),
            torch.ones(1, **tkwargs),
        )
        self.std_norm = std_norm

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate analytical EUBO on the candidate set X.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor, where ``q = 2`` if
                ``previous_winner`` is ``None``, and ``q = 1`` otherwise.

        Returns:
            The acquisition value for each batch as a tensor of shape ``batch_shape``.
        """
        if not (
            ((X.shape[-2] == 2) and (self.previous_winner is None))
            or ((X.shape[-2] == 1) and (self.previous_winner is not None))
        ):
            raise UnsupportedError(
                f"{self.__class__.__name__} only supports q=2 (no previous "
                "winner specified) or q=1 (previous winner specified)."
            )

        Y = X if self.outcome_model is None else self.outcome_model(X)

        if self.previous_winner is not None:
            Y = torch.cat([Y, match_batch_shape(self.previous_winner, Y)], dim=-2)

        pref_posterior = self.model.posterior(Y)
        pref_mean = pref_posterior.mean.squeeze(-1)
        pref_cov = pref_posterior.covariance_matrix
        delta = pref_mean[..., 0] - pref_mean[..., 1]

        w = torch.tensor([1.0, -1.0], dtype=pref_cov.dtype, device=pref_cov.device)
        var = w @ pref_cov @ w
        sigma = torch.sqrt(var.clamp(min=SIGMA_JITTER))

        u = delta / sigma

        ucdf = self.std_norm.cdf(u)
        updf = torch.exp(self.std_norm.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)
        if self.previous_winner is None:
            acqf_val = acqf_val + pref_mean[..., 1]
        return acqf_val


class qExpectedUtilityOfBestOption(MCAcquisitionFunction):
    r"""MC-based Expected Utility of Best Option (qEUBO)

    This computes qEUBO by
    (1) sampling the joint posterior over q points
    (2) evaluating the maximum objective value across the q points
    (3) averaging over the samples

    ``qEUBO(X) = E[max Y], Y ~ f(X), where X = (x_1,...,x_q)``
    """

    def __init__(
        self,
        pref_model: Model,
        outcome_model: DeterministicModel | None = None,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""MC-based Expected Utility of Best Option (qEUBO) as proposed
        in [Astudillo2023qeubo]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating qEUBO on
                the parameter space.
             sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set
                to have no gradient.
        """
        super().__init__(
            model=pref_model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        # ensure the model is in eval mode
        self.add_module("outcome_model", outcome_model)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qEUBO on the candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of t-batches with ``q``
                ``d``-dim design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of qEUBO values at the given design
            points ``X``, where ``batch_shape'`` is the broadcasted batch shape
            of model and input ``X``.
        """
        Y = X if self.outcome_model is None else self.outcome_model(X)

        _, obj = self._get_samples_and_objectives(Y)
        obj_best = obj.max(dim=-1).values
        return obj_best.mean(dim=0)


class PairwiseBayesianActiveLearningByDisagreement(MCAcquisitionFunction):
    r"""MC Bayesian Active Learning by Disagreement"""

    def __init__(
        self,
        pref_model: Model,
        outcome_model: DeterministicModel | None = None,
        num_samples: int | None = 1024,
        std_noise: float | None = 0.0,
    ) -> None:
        """
        Monte Carlo implementation of Bayesian Active Learning by Disagreement (BALD)
        proposed in [Houlsby2011bald]_.

        Args:
            pref_model: The preference model that maps the outcomes (i.e., Y) to
                scalar-valued utility.
            outcome_model: A deterministic model that maps parameters (i.e., X) to
                outcomes (i.e., Y). The outcome model f defines the search space of
                Y = f(X). If model is None, we are directly calculating BALD on
                the parameter space.
            num_samples: number of samples to approximate the conditional_entropy.
            std_noise: Additional observational noise to include. Defaults to 0.
        """
        super().__init__(model=pref_model)
        # ensure the model is in eval mode
        self.add_module("outcome_model", outcome_model)

        self.num_samples = num_samples
        # assuming the relative observation noise is fixed at 1.0 (e.g., in PairwiseGP)
        self.std_noise = std_noise
        self.std_normal = Normal(0, 1)

    @t_batch_mode_transform(expected_q=2)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate MC BALD on the candidate set ``X``.

        Args:
            X: A ``batch_shape x 2 x d``-dim Tensor of t-batches with ``q=2``
                ``d``-dim design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of MC BALD values at the given
            design points pair ``X``, where ``batch_shape'`` is the broadcasted
            batch shape of model and input ``X``.
        """
        Y = X if self.outcome_model is None else self.outcome_model(X)

        pref_posterior = self.model.posterior(Y)
        pref_mean = pref_posterior.mean.squeeze(-1)
        pref_cov = pref_posterior.covariance_matrix

        mu = pref_mean[..., 0] - pref_mean[..., 1]
        w = torch.tensor([1.0, -1.0], dtype=pref_cov.dtype, device=pref_cov.device)
        var = 2 * self.std_noise + w @ pref_cov @ w
        sigma = torch.sqrt(var.clamp(min=SIGMA_JITTER))

        # eq (3) in Houlsby, et al. (2011)
        posterior_entropies = Bernoulli(
            self.std_normal.cdf(mu / torch.sqrt(var + 1))
        ).entropy()

        # Sample-based approx to eq (4) in Houlsby, et al. (2011)
        obj_samples = self.std_normal.cdf(
            Normal(loc=mu, scale=sigma).rsample(torch.Size([self.num_samples]))
        )
        sample_entropies = Bernoulli(obj_samples).entropy()
        conditional_entropies = sample_entropies.mean(dim=0)

        return posterior_entropies - conditional_entropies
