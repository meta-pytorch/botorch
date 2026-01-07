#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Source Upper Confidence Bound.

References:

.. [Ca2021ms]
    Candelieri, A., & Archetti, F. (2021).
    Sparsifying to optimize over multiple information sources:
    an augmented Gaussian process based algorithm.
    Structural and Multidisciplinary Optimization, 64, 239-255.

Contributor: andreaponti5
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from gpytorch.models import ExactGP
from torch import Tensor


class AugmentedUpperConfidenceBound(UpperConfidenceBound):
    r"""Single-outcome Multi-Source Upper Confidence Bound (UCB).

    A modified version of the UCB for Multi Information Source, that consider
    the most optimistic improvement with respect to the best value observed so far.
    The improvement is then penalized depending on source’s cost, and
    the discrepancy between the GP associated to the source and the AGP.

    `AUCB(x, s, y^+) = ((mu(x) + sqrt(beta) * sigma(x)) - y^+)
    / (c(s) (1 + abs(mu(x) - mu_s(x))))`,
    where `mu` and `sigma` are the posterior mean and standard deviation of the AGP,
    `mu_s` is the posterior mean of the GP modelling the s-th source and
    c(s) is the cost of the source s.
    """

    def __init__(
        self,
        model: Model,
        cost: Dict,
        best_f: Union[float, Tensor],
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Augmented Upper Confidence Bound.

        Args:
            model: A fitted single-outcome Augmented GP model.
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            cost: A dictionary containing the cost of querying each source.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        if not hasattr(model, "models"):
            raise UnsupportedError("Model have to be multi-source.")
        super().__init__(
            model=model,
            beta=beta,
            maximize=maximize,
            posterior_transform=posterior_transform,
        )
        self.cost = cost
        self.best_f = best_f

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Augmented Upper Confidence Bound values at
            the given design points `X`.
        """
        alpha = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        agp_mean, agp_sigma = self._mean_and_sigma(X[..., :-1])
        cb = (self.best_f if self.maximize else -self.best_f) + (
            (agp_mean if self.maximize else -agp_mean) + self.beta.sqrt() * agp_sigma
        )
        source_idxs = {
            s.item(): torch.where(torch.round(X[..., -1], decimals=0) == s)[0]
            for s in torch.round(X[..., -1], decimals=0).unique().int()
        }
        for s in source_idxs:
            mean, sigma = self._mean_and_sigma(
                X[source_idxs[s], :, :-1], self.model.models[s]
            )
            alpha[source_idxs[s]] = (
                cb[source_idxs[s]]
                / self.cost[s]
                * (1 + torch.abs(agp_mean[source_idxs[s]] - mean))
            )
        return alpha

    def _mean_and_sigma(
        self,
        X: Tensor,
        model: ExactGP = None,
        compute_sigma: bool = True,
        min_var: float = 1e-12,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            model: the model to use. If None, self is used.
            compute_sigma: Boolean indicating whether to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)
        if model is None:
            posterior = self.model.posterior(
                X=X, posterior_transform=self.posterior_transform
            )
        else:
            posterior = model.posterior(
                X=X, posterior_transform=self.posterior_transform
            )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma
