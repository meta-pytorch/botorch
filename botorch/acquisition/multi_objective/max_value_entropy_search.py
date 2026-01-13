#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for max-value entropy search for multi-objective
Bayesian optimization (MESMO).
"""

from __future__ import annotations

from math import pi

import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.joint_entropy_search import (
    LowerBoundMultiObjectiveEntropySearch,
)
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


# Can be removed in version 0.15.0, or potentially sooner because the code has
# already been raising deprecation warnings for a long time
class qMultiObjectiveMaxValueEntropy(
    qMaxValueEntropy, MultiObjectiveMCAcquisitionFunction
):
    r"""The acquisition function for MESMO.

    This is no longer available. We recommend
    ``qLowerBoundMultiObjectiveMaxValueEntropySearch`` as a replacement.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Multi-objective max-value entropy search acquisition function."""
        raise NotImplementedError(
            "qMultiObjectiveMaxValueEntropy is no longer available. We suggest "
            "qLowerBoundMultiObjectiveMaxValueEntropySearch as a replacement."
        )


class qLowerBoundMultiObjectiveMaxValueEntropySearch(
    LowerBoundMultiObjectiveEntropySearch
):
    r"""The acquisition function for the multi-objective Max-value Entropy Search,
    where the batches ``q > 1`` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point ``X`` and the Pareto optimal outputs.

    See [Tu2022]_ for a discussion on the estimation procedure.

    NOTES:
    (i) The estimated acquisition value could be negative.

    (ii) The lower bound batch acquisition function might not be monotone in the
    sense that adding more elements to the batch does not necessarily increase the
    acquisition value. Specifically, the acquisition value can become smaller when
    more inputs are added.
    """

    def __init__(
        self,
        model: Model,
        hypercell_bounds: Tensor,
        X_pending: Tensor | None = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
    ) -> None:
        r"""Lower bound multi-objective max-value entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M' number of outputs.
            hypercell_bounds:  A ``num_pareto_samples x 2 x J x M``-dim Tensor
                containing the hyper-rectangle bounds for integration, where ``J`` is
                the number of hyper-rectangles. In the unconstrained case, this gives
                the partition of the dominated space. In the constrained case, this
                gives the partition of the feasible dominated space union the
                infeasible space.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "0", "LB", "LB2", or "MC".
            num_samples: The number of Monte Carlo samples for the Monte Carlo
                estimate.
        """
        super().__init__(
            model=model,
            pareto_sets=None,
            pareto_fronts=None,
            hypercell_bounds=hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            num_samples=num_samples,
        )

    def _compute_posterior_statistics(
        self, X: Tensor
    ) -> dict[str, GPyTorchPosterior | Tensor]:
        r"""Compute the posterior statistics.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of inputs.

        Returns:
            A dictionary containing the posterior variables used to estimate the
            entropy.

            - "initial_entropy": A ``batch_shape``-dim Tensor containing the entropy of
                the Gaussian random variable ``p(Y| X, D_n)``.
            - "posterior_mean": A ``batch_shape x num_pareto_samples x q x 1 x M``-dim
                Tensor containing the posterior mean at the input ``X``.
            - "posterior_variance": A `batch_shape x num_pareto_samples x q x 1 x
                M``-dim Tensor containing the posterior variance at the input ``X`
                excluding the observation noise.
            - "observation_noise": A ``batch_shape x num_pareto_samples x q x 1 x M``
                -dim Tensor containing the observation noise at the input ``X``.
            - "posterior_with_noise": The posterior distribution at ``X`` which
                includes the observation noise. This is used to compute the marginal
                log-probabilities with respect to ``p(y| x, D_n)`` for ``x`` in ``X``.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

        # Compute the initial entropy term depending on ``X``.
        # TODO: Below we compute posterior_plus_noise twice:
        #  (1) Firstly, we compute p(Y| X, D_n) when computing the initial entropy
        #  (2) Secondly, we compute p(y| x, D_n) for x in X in order to compute
        #  log(p(y|x, D_n)) for x in X in the Monte Carlo estimate..
        #  This could be simplified if we could evaluate log(p(y|x, D_n)) using the
        #  the posterior p(Y| X, D_n)
        posterior_plus_noise = self.initial_model.posterior(X, observation_noise=True)

        # Additional constant term.
        add_term = (
            0.5
            * self.model.num_outputs
            * (1 + torch.log(2 * pi * torch.ones(1, **tkwargs)))
        )
        # The variance initially has shape ``batch_shape x (q*M) x (q*M)``
        # prior_entropy has shape ``batch_shape x num_fantasies``
        initial_entropy = add_term + 0.5 * torch.logdet(
            posterior_plus_noise.mvn.covariance_matrix
        )
        posterior_statistics = {"initial_entropy": initial_entropy}

        # Compute the posterior entropy term.
        posterior_plus_noise = self.model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # ``batch_shape x q x 1 x M``
        mean = posterior_plus_noise.mean
        var_plus_noise = posterior_plus_noise.variance.clamp_min(CLAMP_LB)
        # Expand shapes to ``batch_shape x num_pareto_samples x q x 1 x M``
        new_shape = (
            mean.shape[:-3] + torch.Size([self.num_pareto_samples]) + mean.shape[-3:]
        )
        mean = mean.unsqueeze(-4).expand(new_shape)
        var_plus_noise = var_plus_noise.unsqueeze(-4).expand(new_shape)

        # TODO: This computes the observation noise via a second evaluation of the
        #   posterior. This step could be done better.
        posterior = self.model.posterior(X.unsqueeze(-2), observation_noise=False)
        var = posterior.variance.clamp_min(CLAMP_LB)
        var = var.unsqueeze(-4).expand(new_shape)
        obs_noise = var_plus_noise - var

        posterior_statistics["posterior_mean"] = mean
        posterior_statistics["posterior_variance"] = var
        posterior_statistics["observation_noise"] = obs_noise
        posterior_statistics["posterior_with_noise"] = posterior_plus_noise

        return posterior_statistics

    def _compute_monte_carlo_variables(
        self, posterior: GPyTorchPosterior
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the samples and log-probability associated with a posterior
        distribution.

        Args:
            posterior: The posterior distribution, which includes the observation
                noise.

        Returns:
            A two-element tuple containing

            - samples: A `num_mc_samples x batch_shape x num_pareto_samples x q x 1
                x M`-dim Tensor containing the Monte Carlo samples.
            - samples_log_prob: A `num_mc_samples x batch_shape x num_pareto_samples
                x q`-dim Tensor containing the log-probabilities of the Monte Carlo
                samples.
        """

        # ``num_mc_samples x batch_shape x q x 1 x M``
        samples = self.get_posterior_samples(posterior)

        # ``num_mc_samples x batch_shape x q``
        if self.model.num_outputs == 1:
            samples_log_prob = posterior.mvn.log_prob(samples.squeeze(-1))
        else:
            samples_log_prob = posterior.mvn.log_prob(samples)

        # Expand shape to `num_mc_samples x batch_shape x num_pareto_samples x
        # q x 1 x M`
        new_shape = (
            samples.shape[:-3]
            + torch.Size([self.num_pareto_samples])
            + samples.shape[-3:]
        )
        samples = samples.unsqueeze(-4).expand(new_shape)

        # Expand shape to ``num_mc_samples x batch_shape x num_pareto_samples x q``
        new_shape = (
            samples_log_prob.shape[:-1]
            + torch.Size([self.num_pareto_samples])
            + samples_log_prob.shape[-1:]
        )
        samples_log_prob = samples_log_prob.unsqueeze(-2).expand(new_shape)

        return samples, samples_log_prob

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundMultiObjectiveMaxValueEntropySearch at the design
        points ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape`` t-batches with
            ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        return self._compute_lower_bound_information_gain(X)
