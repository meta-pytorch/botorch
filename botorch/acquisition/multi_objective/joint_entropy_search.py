#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for joint entropy search for Bayesian optimization (JES).

References:

.. [Tu2022]
    B. Tu, A. Gandy, N. Kantas and B.Shafei. Joint Entropy Search for Multi-Objective
    Bayesian Optimization. Advances in Neural Information Processing Systems, 35.
    2022.

"""

from __future__ import annotations

from abc import abstractmethod
from math import pi

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.utils import fantasize as fantasize_flag
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Normal


class LowerBoundMultiObjectiveEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""Abstract base class for the lower bound multi-objective entropy search
    acquisition functions.
    """

    def __init__(
        self,
        model: Model,
        pareto_sets: Tensor,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        X_pending: Tensor | None = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
    ) -> None:
        r"""Lower bound multi-objective entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M' number of outputs.
            pareto_sets: A ``num_pareto_samples x num_pareto_points x d``-dim Tensor
                containing the sampled Pareto optimal sets of inputs.
            pareto_fronts: A ``num_pareto_samples x num_pareto_points x M``-dim Tensor
                containing the sampled Pareto optimal sets of outputs.
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
        super().__init__(model=model)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        # Batch GP models (e.g. fantasized models) are not currently supported
        if isinstance(model, ModelListGP):
            train_X = model.models[0].train_inputs[0]
        else:
            train_X = model.train_inputs[0]
        if (model.num_outputs > 1 and train_X.ndim > 3) or (
            model.num_outputs == 1 and train_X.ndim > 2
        ):
            raise NotImplementedError(
                "Batch GP models (e.g. fantasized models) are not supported."
            )

        self.initial_model = model
        if (pareto_sets is not None and pareto_sets.ndim != 3) or (
            pareto_fronts is not None and pareto_fronts.ndim != 3
        ):
            raise UnsupportedError(
                "The Pareto set and front should have a shape of "
                "`num_pareto_samples x num_pareto_points x input_dim` and "
                "`num_pareto_samples x num_pareto_points x num_objectives`, "
                "respectively"
            )
        else:
            self.pareto_sets = pareto_sets
            self.pareto_fronts = pareto_fronts

        if hypercell_bounds.ndim != 4:
            raise UnsupportedError(
                "The hypercell_bounds should have a shape of "
                "`num_pareto_samples x 2 x num_boxes x num_objectives`."
            )
        else:
            self.hypercell_bounds = hypercell_bounds
            self.num_pareto_samples = hypercell_bounds.shape[0]

        self.estimation_type = estimation_type
        estimation_types = ["0", "LB", "LB2", "MC"]

        if estimation_type not in estimation_types:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                + ", ".join(f'"{h}"' for h in estimation_types)
                + "."
            )

        self.set_X_pending(X_pending)

    @abstractmethod
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
            - "posterior_variance": A ``batch_shape x num_pareto_samples x q x 1 x M``
                -dim Tensor containing the posterior variance at the input ``X``
                excluding the observation noise.
            - "observation_noise": A ``batch_shape x num_pareto_samples x q x 1 x M``
                -dim Tensor containing the observation noise at the input ``X``.
            - "posterior_with_noise": The posterior distribution at ``X`` which
                includes the observation noise. This is used to compute the marginal
                log-probabilities with respect to ``p(y| x, D_n)`` for ``x`` in ``X``.
        """

        pass  # pragma: no cover

    @abstractmethod
    def _compute_monte_carlo_variables(
        self, posterior: GPyTorchPosterior
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the samples and log-probability associated with a posterior
        distribution.

        Args:
            posterior: A posterior distribution.

        Returns:
            A two-element tuple containing:

            - samples: A `num_mc_samples x batch_shape x num_pareto_samples x q x 1
                x M`-dim Tensor containing the Monte Carlo samples.
            - samples_log_prob: A `num_mc_samples x batch_shape x num_pareto_samples
                x q`-dim Tensor containing the log-probabilities of the Monte Carlo
                samples.
        """

        pass  # pragma: no cover

    def _compute_lower_bound_information_gain(self, X: Tensor) -> Tensor:
        r"""Evaluates the lower bound information gain at the design points ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape``
                t-batches with ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        posterior_statistics = self._compute_posterior_statistics(X)
        initial_entropy = posterior_statistics["initial_entropy"]
        post_mean = posterior_statistics["posterior_mean"]
        post_var = posterior_statistics["posterior_variance"]
        obs_noise = posterior_statistics["observation_noise"]

        # Estimate the expected conditional entropy.
        # ``batch_shape x q`` dim Tensor of entropy estimates
        if self.estimation_type == "0":
            conditional_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
            )

        elif self.estimation_type == "LB":
            conditional_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
                only_diagonal=False,
            )

        elif self.estimation_type == "LB2":
            conditional_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
                only_diagonal=True,
            )

        elif self.estimation_type == "MC":
            posterior_with_noise = posterior_statistics["posterior_with_noise"]
            samples, samples_log_prob = self._compute_monte_carlo_variables(
                posterior_with_noise
            )

            conditional_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
                samples=samples,
                samples_log_prob=samples_log_prob,
            )

        # Sum over the batch.
        return initial_entropy - conditional_entropy.sum(dim=-1)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute lower bound multi-objective entropy search at the design points
        ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape``
                t-batches with ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """

        pass  # pragma: no cover


class qLowerBoundMultiObjectiveJointEntropySearch(
    LowerBoundMultiObjectiveEntropySearch
):
    r"""The acquisition function for the multi-objective joint entropy search, where
    the batches ``q > 1`` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point ``X`` and the Pareto optimal input-output pairs.

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
        pareto_sets: Tensor,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        X_pending: Tensor | None = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
    ) -> None:
        r"""Lower bound multi-objective joint entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M' number of outputs.
            pareto_sets: A ``num_pareto_samples x num_pareto_points x d``-dim Tensor
                containing the sampled Pareto optimal sets of inputs.
            pareto_fronts: A ``num_pareto_samples x num_pareto_points x M``-dim Tensor
                containing the sampled Pareto optimal sets of outputs.
            hypercell_bounds:  A ``num_pareto_samples x 2 x J x M``-dim Tensor
                containing the hyper-rectangle bounds for integration. In the
                unconstrained case, this gives the partition of the dominated space.
                In the constrained case, this gives the partition of the feasible
                dominated space union the infeasible space.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "0", "LB", "LB2", or "MC".
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
        """
        super().__init__(
            model=model,
            pareto_sets=pareto_sets,
            pareto_fronts=pareto_fronts,
            hypercell_bounds=hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            num_samples=num_samples,
        )

        # Condition the model on the sampled pareto optimal points.
        # TODO: Apparently, we need to make a call to the posterior otherwise
        #  we run into a gpytorch runtime error:
        #  "Fantasy observations can only be added after making predictions with a
        #  model so that all test independent caches exist."

        with fantasize_flag():
            with settings.propagate_grads(False):
                _ = self.initial_model.posterior(
                    self.pareto_sets, observation_noise=False
                )
            # Condition with observation noise.
            self.conditional_model = self.initial_model.condition_on_observations(
                X=self.initial_model.transform_inputs(self.pareto_sets),
                Y=self.pareto_fronts,
            )

    def _compute_posterior_statistics(
        self, X: Tensor
    ) -> dict[str, Tensor | GPyTorchPosterior]:
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
            - "posterior_variance": A ``batch_shape x num_pareto_samples x q x 1 x M``
                -dim Tensor containing the posterior variance at the input ``X``
                excluding the observation noise.
            - "observation_noise": A ``batch_shape x num_pareto_samples x q x 1 x M``
                -dim Tensor containing the observation noise at the input ``X``.
            - "posterior_with_noise": The posterior distribution at ``X`` which
                includes the observation noise. This is used to compute the marginal
                log-probabilities with respect to ``p(y| x, D_n)`` for ``x`` in ``X``.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

        # Compute the prior entropy term depending on ``X``.
        initial_posterior_plus_noise = self.initial_model.posterior(
            X, observation_noise=True
        )

        # Additional constant term.
        add_term = (
            0.5
            * self.model.num_outputs
            * (1 + torch.log(2 * pi * torch.ones(1, **tkwargs)))
        )
        # The variance initially has shape ``batch_shape x (q*M) x (q*M)``
        # prior_entropy has shape ``batch_shape``.
        initial_entropy = add_term + 0.5 * torch.logdet(
            initial_posterior_plus_noise.mvn.covariance_matrix
        )

        posterior_statistics = {"initial_entropy": initial_entropy}

        # Compute the posterior entropy term.
        conditional_posterior_with_noise = self.conditional_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=True
        )

        # ``batch_shape x num_pareto_samples x q x 1 x M``
        post_mean = conditional_posterior_with_noise.mean.swapaxes(-4, -3)
        post_var_with_noise = conditional_posterior_with_noise.variance.clamp_min(
            CLAMP_LB
        ).swapaxes(-4, -3)

        # TODO: This computes the observation noise via a second evaluation of the
        #   posterior. This step could be done better.
        conditional_posterior = self.conditional_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=False
        )

        # ``batch_shape x num_pareto_samples x q x 1 x M``
        post_var = conditional_posterior.variance.clamp_min(CLAMP_LB).swapaxes(-4, -3)
        obs_noise = (post_var_with_noise - post_var).clamp_min(CLAMP_LB)

        posterior_statistics["posterior_mean"] = post_mean
        posterior_statistics["posterior_variance"] = post_var
        posterior_statistics["observation_noise"] = obs_noise
        posterior_statistics["posterior_with_noise"] = conditional_posterior_with_noise

        return posterior_statistics

    def _compute_monte_carlo_variables(
        self, posterior: GPyTorchPosterior
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the samples and log-probability associated with the posterior
        distribution that conditions on the Pareto optimal points.

        Args:
            posterior: The conditional posterior distribution at an input ``X``, where
                we have also conditioned over the ``num_pareto_samples`` of optimal
                points. Note that this posterior includes the observation noise.

        Returns:
            A two-element tuple containing

            - samples: A `num_mc_samples x batch_shape x num_pareto_samples x q x 1
                x M`-dim Tensor containing the Monte Carlo samples.
            - samples_log_probs: A `num_mc_samples x batch_shape x num_pareto_samples
                x q`-dim Tensor containing the log-probabilities of the Monte Carlo
                samples.
        """
        # ``num_mc_samples x batch_shape x q x num_pareto_samples x 1 x M``
        samples = self.get_posterior_samples(posterior)

        # ``num_mc_samples x batch_shape x q x num_pareto_samples``
        if self.model.num_outputs == 1:
            samples_log_prob = posterior.mvn.log_prob(samples.squeeze(-1))
        else:
            samples_log_prob = posterior.mvn.log_prob(samples)

        # Swap axes to get the correct shape:
        # samples:``num_mc_samples x batch_shape x num_pareto_samples x q x 1 x M``
        # log prob:``num_mc_samples x batch_shape x num_pareto_samples x q``

        return samples.swapaxes(-4, -3), samples_log_prob.swapaxes(-2, -1)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundMultiObjectiveJointEntropySearch at the design
        points ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of ``batch_shape``
                t-batches with ``q`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        return self._compute_lower_bound_information_gain(X)


def _compute_entropy_noiseless(
    hypercell_bounds: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
) -> Tensor:
    r"""Computes the entropy estimate at the design points ``X`` assuming noiseless
    observations. This is used for the JES-0 and MES-0 estimate.

    Args:
        hypercell_bounds: A ``num_pareto_samples x 2 x J x M`` -dim Tensor containing
            the box decomposition bounds, where ``J = max(num_boxes)``.
        mean: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor containing
            the posterior mean at X.
        variance: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor
            containing the posterior variance at X excluding observation noise.
        observation_noise: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim
            Tensor containing the observation noise at X.

    Returns:
        A ``batch_shape x q``-dim Tensor of entropy estimate at the given design points
        ``X``.
    """
    tkwargs = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

    variance_plus_noise = variance + observation_noise

    # Standardize the box decomposition bounds and compute normal quantities.
    # ``batch_shape x num_pareto_samples x q x 2 x J x M``
    g = (hypercell_bounds.unsqueeze(-4) - mean.unsqueeze(-2)) / torch.sqrt(
        variance.unsqueeze(-2)
    )
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    # Compute the sum of ratios.
    ratios = 0.5 * (Wj * (Vjm / Wjm)) / W
    # ``batch_shape x num_pareto_samples x q x 1 x 1``
    ratio_term = torch.sum(ratios, dim=(-2, -1), keepdims=True)

    # Compute the logarithm of the variance.
    log_term = 0.5 * torch.log(variance_plus_noise).sum(-1, keepdims=True)

    # ``batch_shape x num_pareto_samples x q x 1 x 1``
    log_term = log_term + torch.log(W)

    # Additional constant term.
    M_plus_K = mean.shape[-1]
    add_term = 0.5 * M_plus_K * (1 + torch.log(torch.ones(1, **tkwargs) * 2 * pi))

    # ``batch_shape x num_pareto_samples x q``
    entropy = add_term + (log_term - ratio_term).squeeze(-1).squeeze(-1)

    return entropy.mean(-2)


def _compute_entropy_upper_bound(
    hypercell_bounds: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
    only_diagonal: bool = False,
) -> Tensor:
    r"""Computes the entropy upper bound at the design points ``X``. This is used for
    the JES-LB and MES-LB estimate. If ``only_diagonal`` is True, then this computes
    the entropy estimate for the JES-LB2 and MES-LB2.

    Args:
        hypercell_bounds: A ``num_pareto_samples x 2 x J x M`` -dim Tensor containing
            the box decomposition bounds, where ``J`` = max(num_boxes).
        mean: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor containing
            the posterior mean at X.
        variance: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor
            containing the posterior variance at X excluding observation noise.
        observation_noise: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim
            Tensor containing the observation noise at X.
        only_diagonal: If true, we only compute the diagonal elements of the variance.

    Returns:
        A ``batch_shape x q``-dim Tensor of entropy estimate at the given design points
        ``X``.
    """
    tkwargs = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

    variance_plus_noise = variance + observation_noise

    # Standardize the box decomposition bounds and compute normal quantities.
    # ``batch_shape x num_pareto_samples x q x 2 x J x M``
    g = (hypercell_bounds.unsqueeze(-4) - mean.unsqueeze(-2)) / torch.sqrt(
        variance.unsqueeze(-2)
    )
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]
    Gjm = gpdf[..., 1, :, :] - gpdf[..., 0, :, :]

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    Cjm = Gjm / Wjm

    # First moment:
    Rjm = Cjm * Wj / W
    # `batch_shape x num_pareto_samples x q x 1 x M
    mom1 = mean - torch.sqrt(variance) * Rjm.sum(-2, keepdims=True)
    # diagonal weighted sum
    # `batch_shape x num_pareto_samples x q x 1 x M
    diag_weighted_sum = (Wj * variance * Vjm / Wjm / W).sum(-2, keepdims=True)

    if only_diagonal:
        # ``batch_shape x num_pareto_samples x q x 1 x M``
        mean_squared = mean.pow(2)
        cross_sum = -2 * (mean * torch.sqrt(variance) * Rjm).sum(-2, keepdims=True)
        # ``batch_shape x num_pareto_samples x q x 1 x M``
        mom2 = variance_plus_noise - diag_weighted_sum + cross_sum + mean_squared
        var = (mom2 - mom1.pow(2)).clamp_min(CLAMP_LB)

        # `batch_shape x num_pareto_samples x q
        log_det_term = 0.5 * torch.log(var).sum(dim=-1).squeeze(-1)
    else:
        # First moment x First moment
        # `batch_shape x num_pareto_samples x q x 1 x M x M
        cross_mom1 = torch.einsum("...i,...j->...ij", mom1, mom1)

        # Second moment:
        # `batch_shape x num_pareto_samples x q x 1 x M x M
        # firstly compute the general terms
        mom2_cross1 = -torch.einsum(
            "...i,...j->...ij", mean, torch.sqrt(variance) * Cjm
        )
        mom2_cross2 = -torch.einsum(
            "...i,...j->...ji", mean, torch.sqrt(variance) * Cjm
        )
        mom2_mean_squared = torch.einsum("...i,...j->...ij", mean, mean)

        mom2_weighted_sum = (
            (mom2_cross1 + mom2_cross2) * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)
        mom2_weighted_sum = mom2_weighted_sum + mom2_mean_squared

        # Compute the additional off-diagonal terms.
        mom2_off_diag = torch.einsum(
            "...i,...j->...ij", torch.sqrt(variance) * Cjm, torch.sqrt(variance) * Cjm
        )
        mom2_off_diag_sum = (mom2_off_diag * Wj.unsqueeze(-1) / W.unsqueeze(-1)).sum(
            -3, keepdims=True
        )

        # Compute the diagonal terms and subtract the diagonal computed before.
        init_diag = torch.diagonal(mom2_off_diag_sum, dim1=-2, dim2=-1)
        diag_weighted_sum = torch.diag_embed(
            variance_plus_noise - diag_weighted_sum - init_diag
        )
        mom2 = mom2_weighted_sum + mom2_off_diag_sum + diag_weighted_sum
        # Compute the variance
        var = (mom2 - cross_mom1).squeeze(-3)

        # Jitter the diagonal.
        # The jitter is probably not needed here at all.
        jitter_diag = 1e-6 * torch.diag_embed(torch.ones(var.shape[:-1], **tkwargs))
        log_det_term = 0.5 * torch.logdet(var + jitter_diag)

    # Additional terms.
    M_plus_K = mean.shape[-1]
    add_term = 0.5 * M_plus_K * (1 + torch.log(torch.ones(1, **tkwargs) * 2 * pi))

    # `batch_shape x num_pareto_samples x q
    entropy = add_term + log_det_term
    return entropy.mean(-2)


def _compute_entropy_monte_carlo(
    hypercell_bounds: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
    samples: Tensor,
    samples_log_prob: Tensor,
) -> Tensor:
    r"""Computes the Monte Carlo entropy at the design points ``X``. This is used for
    the JES-MC and MES-MC estimate.

    Args:
        hypercell_bounds: A ``num_pareto_samples x 2 x J x M``-dim Tensor containing
            the box decomposition bounds, where ``J`` = max(num_boxes).
        mean: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor containing
            the posterior mean at X.
        variance: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim Tensor
            containing the posterior variance at X excluding observation noise.
        observation_noise: A ``batch_shape x num_pareto_samples x q x 1 x M``-dim
            Tensor containing the observation noise at X.
        samples: A ``num_mc_samples x batch_shape x num_pareto_samples x q x 1 x M``-dim
            Tensor containing the noisy samples at ``X`` from the posterior conditioned
            on the Pareto optimal points.
        samples_log_prob:  A `num_mc_samples x batch_shape x num_pareto_samples
            x q`-dim  Tensor containing the log probability densities of the samples.

    Returns:
        A ``batch_shape x q``-dim Tensor of entropy estimate at the given design points
        ``X``.
    """
    tkwargs = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

    variance_plus_noise = variance + observation_noise

    ####################################################################
    # Standardize the box decomposition bounds and compute normal quantities.
    # ``batch_shape x num_pareto_samples x q x 2 x J x M``
    g = (hypercell_bounds.unsqueeze(-4) - mean.unsqueeze(-2)) / torch.sqrt(
        variance.unsqueeze(-2)
    )
    # ``batch_shape x num_pareto_samples x q x 1 x M``
    rho = torch.sqrt(variance / variance_plus_noise)

    # Compute the initial normal quantities.
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)

    # Compute the differences between the upper and lower terms.
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    # ``batch_shape x num_pareto_samples x q x 1 x 1``
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    g = g.unsqueeze(0)
    rho = rho.unsqueeze(0).unsqueeze(-2)
    # ``num_mc_samples x batch_shape x num_pareto_samples x q x 1 x 1 x M``
    z = ((samples - mean) / torch.sqrt(variance_plus_noise)).unsqueeze(-2)
    # ``num_mc_samples x batch_shape x num_pareto_samples x q x 2 x J x M``
    # Clamping here is important because ``1 - rho^2 = 0`` at an input where
    # observation noise is zero.
    g_new = (g - rho * z) / torch.sqrt((1 - rho * rho).clamp_min(CLAMP_LB))

    # Compute the initial normal quantities.
    normal_new = Normal(torch.zeros_like(g_new), torch.ones_like(g_new))
    gcdf_new = normal_new.cdf(g_new)

    # Compute the differences between the upper and lower terms.
    Wjm_new = (gcdf_new[..., 1, :, :] - gcdf_new[..., 0, :, :]).clamp_min(CLAMP_LB)

    # Compute W+.
    Wj_new = torch.exp(torch.sum(torch.log(Wjm_new), dim=-1, keepdims=True))
    # ``num_mc_samples x batch_shape x num_pareto_samples x q x 1 x 1``
    W_new = torch.sum(Wj_new, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    # W_ratio = W+ / W
    W_ratio = torch.exp(torch.log(W_new) - torch.log(W).unsqueeze(0))
    samples_log_prob = samples_log_prob.unsqueeze(-1).unsqueeze(-1)

    # Compute the Monte Carlo average: - E[W_ratio * log(W+ p(y))] + log(W)
    log_term = torch.log(W_new) + samples_log_prob
    mc_estimate = -(W_ratio * log_term).mean(0)
    # `batch_shape x num_pareto_samples x q
    entropy = (mc_estimate + torch.log(W)).squeeze(-1).squeeze(-1)

    # An alternative Monte Carlo estimate: - E[W_ratio * log(W_ratio p(y))]
    # log_term = torch.log(W_ratio) + samples_log_prob
    # mc_estimate = - (W_ratio * log_term).mean(0)
    # # `batch_shape x num_pareto_samples x q
    # entropy = mc_estimate.squeeze(-1).squeeze(-1)

    return entropy.mean(-2)
