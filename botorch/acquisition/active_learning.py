#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Active learning acquisition functions.

.. [Seo2014activedata]
    S. Seo, M. Wallat, T. Graepel, and K. Obermayer. `Gaussian process regression:
    Active data selection and test point rejection.
    <https://ieeexplore.ieee.org/document/861310>`_
    IJCNN 2000.

.. [Chen2014seqexpdesign]
    X. Chen and Q. Zhou. `Sequential experimental designs for stochastic kriging.
    <https://ieeexplore.ieee.org/document/7020209>`_
    Winter Simulation Conference 2014.

.. [Binois2017repexp]
    M. Binois, J. Huang, R. B. Gramacy, and M. Ludkovski. `Replication or
    exploration? Sequential design for stochastic simulation experiments.
    <https://arxiv.org/abs/1710.03206>`_
    ArXiv 2017.
"""

from __future__ import annotations

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class qNegIntegratedPosteriorVariance(AcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior variance
    (excluding observation noise, computed using MC integration) of the model.
    In that, it is a proxy for global model uncertainty, and thus purely focused on
    "exploration", rather the "exploitation" of many of the classic Bayesian
    Optimization acquisition functions.

    See [Seo2014activedata]_, [Chen2014seqexpdesign]_, and [Binois2017repexp]_.
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        sampler: MCSampler | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""q-Integrated Negative Posterior Variance.

        Args:
            model: A fitted model.
            mc_points: A ``batch_shape x N x d`` tensor of points to use for
                MC-integrating the posterior variance. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration of the posterior variance.
            sampler: The sampler used for drawing fantasy samples. In the basic setting
                of a standard GP (default) this is a dummy, since the variance of the
                model after conditioning does not actually depend on the sampled values.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A ``n' x d``-dim Tensor of ``n'`` design points that have
                points that have been submitted for function evaluation but
                have not yet been evaluated.
        """
        super().__init__(model=model)
        self.posterior_transform = posterior_transform
        if sampler is None:
            # If no sampler is provided, we use the following dummy sampler for the
            # fantasize() method in forward. IMPORTANT: This assumes that the posterior
            # variance does not depend on the samples y (only on x), which is true for
            # standard GP models, but not in general (e.g. for other likelihoods or
            # heteroskedastic GPs using a separate noise model fit on data).
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        # Construct the fantasy model (we actually do not use the full model,
        # this is just a convenient way of computing fast posterior covariances
        fantasy_model = self.model.fantasize(
            X=X,
            sampler=self.sampler,
        )

        bdims = tuple(1 for _ in X.shape[:-2])
        if self.model.num_outputs > 1:
            # We use q=1 here b/c ScalarizedObjective currently does not fully exploit
            # LinearOperator operations and thus may be slow / overly memory-hungry.
            # TODO (T52818288): Properly use LinearOperators in scalarize_posterior
            mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
        else:
            # While we only need marginal variances, we can evaluate for q>1
            # b/c for GPyTorch models lazy evaluation can make this quite a bit
            # faster than evaluating in t-batch mode with q-batch size of 1
            mc_points = self.mc_points.view(*bdims, -1, X.size(-1))

        # evaluate the posterior at the grid points
        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(
                mc_points, posterior_transform=self.posterior_transform
            )

        neg_variance = posterior.variance.mul(-1.0)
        if self.model.num_outputs > 1:
            # if multi-output + obj, shape is num_grid_points x batch_shape x 1 x 1
            return neg_variance.mean(dim=0).squeeze(-1).squeeze(-1)
        else:
            # if single-output, shape is 1 x batch_shape x num_grid_points x 1
            return neg_variance.mean(dim=-2).squeeze(-1).squeeze(0)


class PairwiseMCPosteriorVariance(MCAcquisitionFunction):
    r"""Variance of difference for Active Learning

    Given a model and an objective, calculate the posterior sample variance
    of the objective on the difference of pairs of points. See more implementation
    details in ``forward``. This acquisition function is typically used with a
    pairwise model (e.g., PairwiseGP) and a likelihood/link function
    on the pair difference (e.g., logistic or probit) for pure exploration
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective,
        sampler: MCSampler | None = None,
    ) -> None:
        r"""Pairwise Monte Carlo Posterior Variance

        Args:
            model: A fitted model.
            objective: An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.) applied on the difference of (usually 1-d)
                two samples. Can be implemented via GenericMCObjective.
            sampler: The sampler used for drawing MC samples.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=None
        )

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PairwiseMCPosteriorVariance on the candidate set ``X``.

        Args:
            X: A ``batch_size x q x d``-dim Tensor. q should be a multiple of 2.

        Returns:
            Tensor of shape ``batch_size x q`` representing the posterior variance
            of link function at X that active learning hopes to maximize
        """
        if X.shape[-2] == 0 or X.shape[-2] % 2 != 0:
            raise RuntimeError(
                "q must be a multiple of 2 for PairwiseMCPosteriorVariance"
            )

        # The output is of shape batch_shape x 2 x d
        # For PairwiseGP, d = 1
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # num_samples x batch_shape x 2 x d

        # The output is of shape num_samples x batch_shape x q/2 x d
        # assuming the comparison is made between the 2 * i and 2 * i + 1 elements
        samples_diff = samples[..., ::2, :] - samples[..., 1::2, :]
        mc_var = self.objective(samples_diff).var(dim=0)
        mean_mc_var = mc_var.mean(dim=-1)

        return mean_mc_var
