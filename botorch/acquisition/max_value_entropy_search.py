#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Max-value Entropy Search (MES), General
Information-Based Bayesian Optimization (GIBBON), and
multi-fidelity MES with noisy observations and trace observations.

References

.. [Moss2021gibbon]
    Moss, H. B., et al.,
    GIBBON: General-purpose Information-Based Bayesian OptimisatioN.
    Journal of Machine Learning Research, 2021.

.. [Takeno2020mfmves]
    S. Takeno, H. Fukuoka, Y. Tsukada, T. Koyama, M. Shiga, I. Takeuchi,
    M. Karasuyama. Multi-fidelity Bayesian Optimization with Max-value Entropy
    Search and its Parallelization. Proceedings of the 37th International
    Conference on Machine Learning, 2020.

.. [Wang2017mves]
    Z. Wang, S. Jegelka, Max-value Entropy Search for Efficient
    Bayesian Optimization. Proceedings of the 37th International
    Conference on Machine Learning, 2017.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from math import log

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.cost_aware import CostAwareUtility, InverseCostWeightedUtility
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    match_batch_shape,
    t_batch_mode_transform,
)
from linear_operator.functions import inv_quad
from linear_operator.utils.cholesky import psd_safe_cholesky
from scipy.optimize import brentq
from scipy.stats import norm
from torch import Tensor


CLAMP_LB = 1.0e-8


class MaxValueBase(AcquisitionFunction, ABC):
    r"""Abstract base class for acquisition functions based on Max-value Entropy Search,
    using discrete max posterior sampling.

    This class provides the basic building blocks for constructing max-value
    entropy-based acquisition functions along the lines of [Wang2017mves]_.
    It provides basic functionality for sampling posterior maximum values from
    a surrogate Gaussian process model using a discrete set of candidates. It supports
    either exact (w.r.t. the candidate set) sampling, or using a Gumbel approximation.

    Subclasses must implement ``_compute_information_gain``.
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_mv_samples: int = 10,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        train_inputs: Tensor | None = None,
    ) -> None:
        r"""Single-outcome max-value entropy search-based acquisition functions
        based on discrete MV sampling.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A ``n x d`` Tensor including ``n`` candidate points to
                discretize the design space. Max values are sampled from the
                (joint) model posterior over these points.
            num_mv_samples: Number of max value samples.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            train_inputs: A ``n_train x d`` Tensor that the model has been fitted on.
                Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        super().__init__(model=model)

        # NOTE The _init_model is used to create the fantasy model and draw max-value
        # samples. The self.model is overwritten in qMaxValueEntropy.set_X_pending
        # and its multi-fidelity version by a fantasy model with batch_size fantasies.
        # For these acqfs, the fantasy model is used throughout the forward pass.
        # qLowerBound methods do not do this.
        self._init_model = model
        if model.num_outputs > 1:
            raise UnsupportedError(
                f"Multi-output models are not supported by {self.__class__.__name__}."
            )
        if train_inputs is None and hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs[0]
        if train_inputs is not None:
            if train_inputs.ndim > 2:
                raise NotImplementedError(
                    "Batched GP models (e.g., fantasized models) are not yet "
                    f"supported by `{self.__class__.__name__}`."
                )
            train_inputs = match_batch_shape(train_inputs, candidate_set)
            candidate_set = torch.cat([candidate_set, train_inputs], dim=0)

        self.candidate_set = candidate_set
        self.num_mv_samples = num_mv_samples
        self.posterior_transform = posterior_transform
        self.use_gumbel = use_gumbel
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0

        # NOTE X_pending is not really needed to make the parallelism work here,
        # it just makes the approximation of the max values slightly more accurate.
        self._sample_max_values(num_samples=num_mv_samples, X_pending=X_pending)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points ``X``.

        Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design points each.

        Returns:
            A ``batch_shape``-dim Tensor of MVE values at the given design points ``X``.
        """
        # Compute the posterior, posterior mean, variance and std.
        posterior = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        # batch_shape x num_fantasies x (m)
        mean = self.weight * posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        ig = self._compute_information_gain(
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1)
        )
        # Average over fantasies, ig is of shape ``num_fantasies x batch_shape x (m)``.
        return ig.mean(dim=0)

    def _sample_max_values(
        self, num_samples: int, X_pending: Tensor | None = None
    ) -> None:
        r"""Draw samples from the posterior over maximum values on a discrete set.

        These samples are used to compute Monte Carlo approximations of expectations
        over the posterior over the function maximum. This function sets
        ``self.posterior_max_values``.

        Args:
            num_samples: The number of samples to draw.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.

        Returns:
            A ``num_samples x num_fantasies`` Tensor of posterior max value samples
            (``num_fantasies=1`` for non-fantasized models).
        """
        if self.use_gumbel:
            sample_max_values = _sample_max_value_Gumbel
        else:
            sample_max_values = _sample_max_value_Thompson
        candidate_set = self.candidate_set

        with torch.no_grad():
            if X_pending is not None:
                # Append X_pending to candidate set - heuristic since X_pending is
                # presumably a good point to sample from
                X_pending = match_batch_shape(X_pending, self.candidate_set)
                candidate_set = torch.cat([self.candidate_set, X_pending], dim=0)

            # project the candidate_set to the highest fidelity,
            # which is needed for the multi-fidelity MES
            try:
                candidate_set = self.project(candidate_set)
            except AttributeError:
                pass

            self.posterior_max_values = sample_max_values(
                model=self._init_model,
                candidate_set=candidate_set,
                num_samples=num_samples,
                posterior_transform=self.posterior_transform,
                maximize=self.maximize,
            )

    # ------- Abstract methods that need to be implemented by subclasses ------- #

    @abstractmethod
    def _compute_information_gain(self, X: Tensor) -> Tensor:
        r"""Compute the information gain at the design points ``X``.

        ``num_fantasies = 1`` for non-fantasized models.

         Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design point each.

        Returns:
            A ``num_fantasies x batch_shape``-dim Tensor of information gains at the
            given design points ``X`` (``num_fantasies=1`` for non-fantasized models).
        """


class qMaxValueEntropy(MaxValueBase, MCSamplerMixin):
    r"""The acquisition function for Max-value Entropy Search.

    This acquisition function computes the mutual information of max values and
    a candidate point X. See [Wang2017mves]_ for a detailed discussion.

    The model must be single-outcome. The batch case ``q > 1`` is supported
    through cyclic optimization and fantasies.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MES = qMaxValueEntropy(model, candidate_set)
        >>> mes = MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        train_inputs: Tensor | None = None,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A ``n x d`` Tensor including ``n`` candidate points to
                discretize the design space. Max values are sampled from the
                (joint) model posterior over these points.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity, wall time and memory). Ignored if ``X_pending`` is ``None``.
            num_mv_samples: Number of max value samples.
            num_y_samples: Number of posterior samples at specific design point ``X``.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            train_inputs: A ``n_train x d`` Tensor that the model has been fitted on.
                Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
        )
        MCSamplerMixin.__init__(
            self,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_y_samples])),
        )
        self._init_model = model  # used for ``fantasize()`` when setting ``X_pending``
        self.fantasies_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([num_fantasies])
        )
        self.num_fantasies = num_fantasies
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        Args:
            X_pending: ``m x d`` Tensor with ``m`` ``d``-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            self.X_pending = X_pending.detach()
            # fantasize the model and use this as the new model
            self.model = self._init_model.fantasize(
                X=X_pending.detach(),
                sampler=self.fantasies_sampler,
            )
        else:
            self.X_pending = None
            self.model = self._init_model

    # NOTE: This may not work with m > 1, and currently the only supported use
    # cases are with m=1.
    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Computes the information gain at the design points ``X``.

        Approximately computes the information gain at the design points ``X``,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the papers on multi-fidelity MES by
        [Takeno2020mfmves]_. The notation in the comments in this function follows
        the Appendix C of [Takeno2020mfmves]_.

        ``num_fantasies = 1`` for non-fantasized models.

        Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design point each.
            mean_M: A ``batch_shape x num_fantasies x (m)``-dim Tensor of means.
            variance_M: A ``batch_shape x num_fantasies x (m)``-dim Tensor of variances.
            covar_mM: A
                ``batch_shape x num_fantasies x (m) x (1 + num_trace_observations)``-dim
                Tensor of covariances.

        Returns:
            A ``num_fantasies x batch_shape``-dim Tensor of information gains at the
            given design points ``X`` (``num_fantasies=1`` for non-fantasized models).
        """
        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x num_fantasies x (m)
        # x (1 + num_trace_observations) x (1 + num_trace_observations)
        variance_m = posterior_m.distribution.covariance_matrix
        check_no_nans(variance_m)

        # compute mean and std for fM|ym, x, Dt ~ N(u, s^2)
        samples_m = self.weight * self.get_posterior_samples(posterior_m).squeeze(-1)
        # s_m x batch_shape x num_fantasies x (m) x (1 + num_trace) x (1 + num_trace)
        L = psd_safe_cholesky(variance_m)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)
        # equivalent to torch.matmul(covar_mM.unsqueeze(-2), torch.inverse(variance_m))
        # batch_shape x num_fantasies (m) x 1 x (1 + num_trace_observations)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        # s_m x batch_shape x num_fantasies x (m)
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies x (m)
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()
        # batch_shape x num_fantasies x (m)

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # Compute p(fM <= f* | ym, x, Dt)
        view_shape = torch.Size(
            [
                self.posterior_max_values.shape[0],
                # add 1s to broadcast across the batch_shape of X
                *[1 for _ in range(X.ndim - self.posterior_max_values.ndim)],
                *self.posterior_max_values.shape[1:],
            ]
        )  # s_M x batch_shape x num_fantasies x (m)
        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1)
        # s_M x 1 x batch_shape x num_fantasies x (m)
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        # s_M x s_m x batch_shape x num_fantasies x (m)  =
        #   s_M x 1 x batch_shape x num_fantasies x (m)
        #   - s_m x batch_shape x num_fantasies x (m)
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        # Compute p(fM <= f* | x, Dt)
        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        # s_M x 1 x batch_shape x num_fantasies  x (m) =
        # s_M x 1 x 1 x num_fantasies x (m) - batch_shape x num_fantasies x (m)
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        # s_M x 1 x batch_shape x num_fantasies x (m)

        # Compute log(p(ym | x, Dt))
        log_pdf_fm = posterior_m.distribution.log_prob(
            self.weight * samples_m
        ).unsqueeze(0)
        # 1 x s_m x batch_shape x num_fantasies x (m)

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.distribution.entropy()  # batch_shape x num_fantasies x (m)

        # regression adjusted H1 estimation, H1_hat = H1_bar - beta * (H0_bar - H0)
        # H1 = E_{f*|x, Dt}[H(ym|f*, x, Dt)]
        Z = cdf_mvs_new / cdf_mvs  # s_M x s_m x batch_shape x num_fantasies x (m)
        # s_M x s_m x batch_shape x num_fantasies x (m)
        h1 = -Z * Z.log() - Z * log_pdf_fm
        check_no_nans(h1)
        dim = [0, 1]  # dimension of fm samples, fM samples
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat  # batch_shape x num_fantasies x (m)
        permute_idcs = [-1, *range(ig.ndim - 1)]
        ig = ig.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        return ig


class qLowerBoundMaxValueEntropy(MaxValueBase):
    r"""The acquisition function for General-purpose Information-Based
    Bayesian Optimisation (GIBBON).

    This acquisition function provides a computationally cheap approximation of
    the mutual information between max values and a batch of candidate points ``X``.
    See [Moss2021gibbon]_ for a detailed discussion.

    The model must be single-outcome, unless using a PosteriorTransform.
    q > 1 is supported through greedy batch filling.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
        >>> candidates, _ = optimize_acqf(qGIBBON, bounds, q=5)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_mv_samples: int = 10,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        train_inputs: Tensor | None = None,
    ) -> None:
        r"""Lower bound max-value entropy search acquisition function (GIBBON).

        Args:
            model: A fitted single-outcome model.
            candidate_set: A ``n x d`` Tensor including ``n`` candidate points to
                discretize the design space. Max values are sampled from the
                (joint) model posterior over these points.
            num_mv_samples: Number of max value samples.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            train_inputs: A ``n_train x d`` Tensor that the model has been fitted on.
                Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
        )
        self.set_X_pending(X_pending)

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Compute GIBBON's approximation of information gain at the
        design points ``X``.

        When using GIBBON for batch optimization (i.e ``q > 1``), we calculate the
        additional information provided by adding a new candidate point to the current
        batch of design points (``X_pending``), rather than calculating the information
        provided by the whole batch. This allows a modest computational saving.

        Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design point each.
            mean_M: A ``batch_shape x 1``-dim Tensor of means.
            variance_M: A ``batch_shape x 1``-dim Tensor of variances consisting of
                ``batch_shape`` t-batches with ``num_fantasies`` fantasies.
            covar_mM: A ``batch_shape x num_fantasies x (1 + num_trace_observations)``
                -dim Tensor of covariances.

        Returns:
            A ``num_fantasies x batch_shape``-dim Tensor of information gains at the
            given design points ``X``.
        """
        # TODO: give the Posterior API an add_observation_noise function to avoid
        # doing posterior computations twice

        # compute the mean_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB).squeeze(-1)
        # batch_shape x 1
        check_no_nans(variance_m)

        # get stdv of noiseless variance
        stdv = variance_M.sqrt()
        # batch_shape x 1

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        # 1 x s_M
        normalized_mvs = (mvs - mean_M) / stdv
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rhos_squared = torch.pow(covar_mM.squeeze(-1), 2) / (variance_m * variance_M)
        # batch_shape x 1
        check_no_nans(rhos_squared)

        # calculate quality contribution to the GIBBON acquisition function
        inner_term = 1 - rhos_squared * ratio * (normalized_mvs + ratio)
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        acq = acq.mean(dim=1).unsqueeze(0)

        if self.X_pending is None:
            # for q=1, no repulsion term required
            return acq

        # for q>1 GIBBON requires repulsion terms r_i, where
        # r_i = log |C_i| for the predictive
        # correlation matricies C_i between each candidate point in X and
        # the m current batch elements in X_pending.

        # Each predictive covariance matrix can be expressed as
        # V_i = [[v_i, A_i], [A_i,B]] for a shared m x m tensor B.
        # So we can efficiently calculate |V_i| using the formula for
        # determinant of block matricies, i.e.
        # |V_i| = (v_i - A_i^T * B^{-1} * A_i) * |B|
        # As the |B| term does not depend on X and we later take its log,
        # it provides only a translation of the acquisition function surface
        # and can thus be ignored.

        if self.posterior_transform is not None:
            raise UnsupportedError(
                "qLowerBoundMaxValueEntropy does not support PosteriorTransforms"
                "when X_pending is not None."
            )

        X_batches = torch.cat(
            [X, self.X_pending.unsqueeze(0).repeat(X.shape[0], 1, 1)], 1
        )
        # batch_shape x (1 + m) x d
        # NOTE: This is the blocker for supporting posterior transforms.
        # We would have to process this MVN, applying whatever operations
        # are typically applied for the corresponding posterior, then applying
        # the posterior transform onto the resulting object.
        V = self.model(X_batches)
        # Evaluate terms required for A
        A = V.lazy_covariance_matrix[:, 0, 1:].unsqueeze(1)
        # batch_shape x 1 x m
        # Evaluate terms required for B
        B = self.model.posterior(
            self.X_pending,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        ).distribution.covariance_matrix.unsqueeze(0)
        # 1 x m x m

        # use determinant of block matrix formula
        inv_quad_term = inv_quad(B, A.transpose(1, 2)).unsqueeze(1)
        # NOTE: Even when using Cholesky to compute inv_quad, ``V_determinant`` can be
        # negative due to numerical issues. To avoid this, we clamp the variance
        # so that ``V_determinant`` > 0, while still allowing gradients to be
        # propagated through ``inv_quad_term``, as well as through ``variance_m``
        # in the expression for ``r`` below.
        # choosing eps to be small while avoiding numerical underflow
        eps = 1e-6 if inv_quad_term.dtype == torch.float32 else 1e-12
        V_determinant = variance_m.clamp(inv_quad_term * (1 + eps)) - inv_quad_term
        # batch_shape x 1

        # Take logs and convert covariances to correlations.
        r = V_determinant.log() - variance_m.log()  # = log(1 - inv_quad / var)
        r = 0.5 * r.transpose(0, 1)
        return acq + r


class qMultiFidelityMaxValueEntropy(qMaxValueEntropy):
    r"""Multi-fidelity max-value entropy.

    The acquisition function for multi-fidelity max-value entropy search
    with support for trace observations. See [Takeno2020mfmves]_
    for a detailed discussion of the basic ideas on multi-fidelity MES
    (note that this implementation is somewhat different).

    The model must be single-outcome.
    The batch case ``q > 1`` is supported through cyclic optimization and fantasies.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MF_MES = qMultiFidelityMaxValueEntropy(model, candidate_set)
        >>> mf_mes = MF_MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] = lambda X: X,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A ``n x d`` or ``n x (d + s)`` Tensor including ``n``
                candidate points to discretize the design space, which will be
                used to sample the max values from their posteriors. ``s`` is the
                number of fidelity dimensions. The ``project`` callable is applied
                to the candidate set before use, so if it handles inserting fidelity
                dimensions (e.g., ``project_to_target_fidelity`` with ``d``
                specified), ``candidate_set`` can omit them.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance) and it's only used when ``X_pending``
                is not ``None``.
            num_mv_samples: Number of max value samples.
            num_y_samples: Number of posterior samples at specific design point ``X``.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            project: A callable mapping a ``batch_shape x q x d`` tensor of design
                points to a tensor of the same shape projected to the desired
                target set (e.g. the target fidelities in case of multi-fidelity
                optimization). This is also applied to the candidate set during
                initialization.
            expand: A callable mapping a ``batch_shape x q x d`` input tensor to
                a ``batch_shape x (q + q_e)' x d``-dim output tensor, where the
                ``q_e`` additional points in each q-batch correspond to
                additional ("trace") observations.
        """
        # Project candidate_set to target fidelity before passing to super.
        # This ensures candidate_set has the right dimensions for concatenation
        # with train_inputs in MaxValueBase.__init__, and also allows
        # candidate_set to omit fidelity dimensions if project handles
        # inserting them (e.g., project_to_target_fidelity with d specified).
        candidate_set = project(candidate_set)
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
        )

        if cost_aware_utility is None:
            cost_model = AffineFidelityCostModel(fidelity_weights={-1: 1.0})
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        self.cost_aware_utility = cost_aware_utility
        self.expand = expand
        self.project = project
        self._cost_sampler = None
        # @TODO make sure fidelity_dims align in project, expand & cost_aware_utility
        # It seems very difficult due to the current way of handling project/expand

        # resample max values after initializing self.project
        # so that the max value samples are at the highest fidelity
        self._sample_max_values(num_samples=self.num_mv_samples, X_pending=X_pending)

    @property
    def cost_sampler(self):
        if self._cost_sampler is None:
            # Note: Using the deepcopy here is essential. Removing this poses a
            # problem if the base model and the cost model have a different number
            # of outputs or test points (this would be caused by expand), as this
            # would trigger re-sampling the base samples in the fantasy sampler.
            # By cloning the sampler here, the right thing will happen if the
            # the sizes are compatible, if they are not this will result in
            # samples being drawn using different base samples, but it will at
            # least avoid changing state of the fantasy sampler.
            self._cost_sampler = deepcopy(self.fantasies_sampler)
        return self._cost_sampler

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates ``qMultifidelityMaxValueEntropy`` at the design points ``X``

        Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design point each.

        Returns:
            A ``batch_shape``-dim Tensor of MF-MVES values at the design points ``X``.
        """
        X_expand = self.expand(X)  # batch_shape x (1 + num_trace_observations) x d
        X_max_fidelity = self.project(X)  # batch_shape x 1 x d
        X_all = torch.cat((X_expand, X_max_fidelity), dim=-2).unsqueeze(-3)
        # batch_shape x num_fantasies x (2 + num_trace_observations) x d

        # Compute the posterior, posterior mean, variance without noise
        # ``_m`` and ``_M`` in the var names means the current and the max fidelity.
        posterior = self.model.posterior(
            X_all, observation_noise=False, posterior_transform=self.posterior_transform
        )
        mean_M = self.weight * posterior.mean[..., -1, 0]  # batch_shape x num_fantasies
        variance_M = posterior.variance[..., -1, 0].clamp_min(CLAMP_LB)
        # get the covariance between the low fidelities and max fidelity
        covar_mM = posterior.distribution.covariance_matrix[..., :-1, -1]
        # batch_shape x num_fantasies x (1 + num_trace_observations)

        check_no_nans(mean_M)
        check_no_nans(variance_M)
        check_no_nans(covar_mM)

        # compute the information gain (IG)
        ig = self._compute_information_gain(
            X=X_expand, mean_M=mean_M, variance_M=variance_M, covar_mM=covar_mM
        )
        ig = self.cost_aware_utility(X=X, deltas=ig, sampler=self.cost_sampler)
        return ig.mean(dim=0)  # average over the fantasies


class qMultiFidelityLowerBoundMaxValueEntropy(qMultiFidelityMaxValueEntropy):
    r"""Multi-fidelity acquisition function for General-purpose Information-Based
    Bayesian optimization (GIBBON).

    The acquisition function for multi-fidelity max-value entropy search
    with support for trace observations. See [Takeno2020mfmves]_
    for a detailed discussion of the basic ideas on multi-fidelity MES
    (note that this implementation is somewhat different). This acquisition function
    is similar to ``qMultiFidelityMaxValueEntropy`` but computes the information gain
    from the lower bound described in [Moss2021gibbon].

    The model must be single-outcome, unless using a PosteriorTransform.
    The batch case ``q > 1`` is supported through cyclic optimization and fantasies.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MF_qGIBBON = qMultiFidelityLowerBoundMaxValueEntropy(model, candidate_set)
        >>> mf_gibbon = MF_qGIBBON(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        X_pending: Tensor | None = None,
        posterior_transform: PosteriorTransform | None = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A ``n x d`` or ``n x (d + s)`` Tensor including ``n``
                candidate points to discretize the design space, which will be
                used to sample the max values from their posteriors. ``s`` is the
                number of fidelity dimensions. The ``project`` callable is applied
                to the candidate set before use, so if it handles inserting fidelity
                dimensions (e.g., ``project_to_target_fidelity`` with ``d``
                specified), ``candidate_set`` can omit them.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance) and it's only used when ``X_pending``
                is not ``None``.
            num_mv_samples: Number of max value samples.
            num_y_samples: Number of posterior samples at specific design point ``X``.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            maximize: If True, consider the problem a maximization problem.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            project: A callable mapping a ``batch_shape x q x d`` tensor of design
                points to a tensor of the same shape projected to the desired
                target set (e.g. the target fidelities in case of multi-fidelity
                optimization).
            expand: A callable mapping a ``batch_shape x q x d`` input tensor to
                a ``batch_shape x (q + q_e)' x d``-dim output tensor, where the
                ``q_e`` additional points in each q-batch correspond to
                additional ("trace") observations.
                NOTE: This is currently not supported. It leads to wrong outputs.
        """
        if expand is not None:
            raise UnsupportedError(
                f"{self.__class__.__name__} does not support trace observations. "
            )
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            cost_aware_utility=cost_aware_utility,
            project=project,
        )

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """
        For the non-lower bound methods, X_pending creates a new fantasy model
        with a batch shape of (num_fantasies, batch_shape, m). Lower bound methods
        don't operate with the same logic, so we don't need to create a new fantasy
        model. Moreover, this causes shape issues in the forward pass due to
        tensor broadcasting inconsistencies.
        """
        AcquisitionFunction.set_X_pending(self, X_pending)

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Compute GIBBON's approximation of information gain at the
        design points ``X``.

        When using GIBBON for batch optimization (i.e ``q > 1``), we
        calculate the additional information provided by adding a new
        candidate point to the current batch of design points
        (``X_pending``), rather than calculating the information provided
        by the whole batch. This allows a modest computational saving.

        Args:
            X: A ``batch_shape x 1 x d``-dim Tensor of ``batch_shape`` t-batches
                with ``1`` ``d``-dim design point each.
            mean_M: A ``batch_shape x 1``-dim Tensor of means.
            variance_M: A ``batch_shape x 1``-dim Tensor of variances
                consisting of ``batch_shape`` t-batches with
                ``num_fantasies`` fantasies.
            covar_mM: A ``batch_shape x num_fantasies x (1 + num_trace_observations)``
                -dim Tensor of covariances.

        Returns:
            A ``num_fantasies x batch_shape``-dim Tensor of information gains at the
            given design points ``X``.
        """
        return qLowerBoundMaxValueEntropy._compute_information_gain(
            self, X=X, mean_M=mean_M, variance_M=variance_M, covar_mM=covar_mM
        )


def _sample_max_value_Thompson(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: PosteriorTransform | None = None,
    maximize: bool = True,
) -> Tensor:
    """Samples the max values by discrete Thompson sampling.

    Should generally be called within a ``with torch.no_grad()`` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A ``n x d`` Tensor including ``n`` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A ``num_samples x 1`` Tensor of posterior max value samples.
    """
    posterior = model.posterior(candidate_set, posterior_transform=posterior_transform)
    weight = 1.0 if maximize else -1.0
    samples = weight * posterior.rsample(torch.Size([num_samples])).squeeze(-1)
    # samples is num_samples x (num_fantasies) x n
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x 1

    return max_values


def _sample_max_value_Gumbel(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: PosteriorTransform | None = None,
    maximize: bool = True,
) -> Tensor:
    """Samples the max values by Gumbel approximation.

    Should generally be called within a ``with torch.no_grad()`` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A ``n x d`` Tensor including ``n`` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A ``num_samples x num_fantasies`` Tensor of posterior max value samples.
    """
    # define the approximate CDF for the max value under the independence assumption
    posterior = model.posterior(candidate_set, posterior_transform=posterior_transform)
    weight = 1.0 if maximize else -1.0
    mu = weight * posterior.mean
    sigma = posterior.variance.clamp_min(1e-8).sqrt()
    # mu, sigma is (num_fantasies) X n X 1
    if len(mu.shape) == 3 and mu.shape[-1] == 1:
        mu = mu.squeeze(-1).T
        sigma = sigma.squeeze(-1).T
    # mu, sigma is now n X num_fantasies or n X 1

    # bisect search to find the quantiles 25, 50, 75
    lo = (mu - 3 * sigma).min(dim=0).values
    hi = (mu + 5 * sigma).max(dim=0).values
    num_fantasies = mu.shape[1]
    device = candidate_set.device
    dtype = candidate_set.dtype
    quantiles = torch.zeros(num_fantasies, 3, device=device, dtype=dtype)
    for i in range(num_fantasies):
        lo_, hi_ = lo[i], hi[i]
        N = norm(mu[:, i].numpy(force=True), sigma[:, i].numpy(force=True))
        quantiles[i, :] = torch.tensor(
            [
                brentq(lambda y: np.exp(np.sum(N.logcdf(y))) - p, lo_, hi_)
                for p in [0.25, 0.50, 0.75]
            ]
        )
    q25, q50, q75 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    # q25, q50, q75 are 1 dimensional tensor with size of either 1 or num_fantasies

    # parameter fitting based on matching percentiles for the Gumbel distribution
    b = (q25 - q75) / (log(log(4.0 / 3.0)) - log(log(4.0)))
    a = q50 + b * log(log(2.0))

    # inverse sampling from the fitted Gumbel CDF distribution
    sample_shape = (num_samples, 1)
    eps = torch.rand(*sample_shape, device=device, dtype=dtype)
    max_values = a - b * eps.log().mul(-1.0).log()

    return max_values  # num_samples x 1
