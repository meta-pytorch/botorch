#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Source Gaussian Process Regression models based on GPyTorch models.

References:

.. [Ca2021ms]
    Candelieri, A., & Archetti, F. (2021).
    Sparsifying to optimize over multiple information sources:
    an augmented Gaussian process based algorithm.
    Structural and Multidisciplinary Optimization, 64, 239-255.

Contributor: andreaponti5
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch

from botorch import fit_gpytorch_mll
from botorch.exceptions import InputDataError
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils import draw_sobol_samples
from gpytorch import ExactMarginalLogLikelihood, Module
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.mean import Mean
from gpytorch.models import ExactGP
from torch import Tensor


def get_random_x_for_agp(
    n: int,
    bounds: Tensor,
    q: int,
    seed: Optional[int] = None,
):
    r"""Draw qMC samples from the box defined by bounds.
    The function assures that at least one point belong to
    the highest fidelity source (the source associated to bounds[1, -1]).

    Args:
        n: The number of samples.
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively. The last dimension represents
            number of sources.
        q: The size of each q-batch.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x q x d`-dim tensor of qMC samples from the box
        defined by bounds.
    """
    if n < 1:
        raise InputDataError(
            f"The number of points n should be greater than 0 (given n={n})."
        )
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed).squeeze(1)
    train_x[..., -1] = torch.round(train_x[..., -1], decimals=0)
    if bounds[1, -1] not in train_x[..., -1]:
        true_idxs = torch.randint(0, n, [max(1, int(n * 0.2))])
        train_x[true_idxs, -1] = bounds[1, -1]
    return train_x


class SingleTaskAugmentedGP(SingleTaskGP):
    r"""A single-task multi-source GP model.

    The Augmented Gaussian Process (AGP) is described in [Ca2021ms]_.
    The basic idea is to select a subset of function evaluations (namely,
    augmenting observations), among those performed so far over all the
    cheap sources, to augment the set of observations from the ground
    truth source. The resulting augmented set of observations is used to
    fit the AGP.
    The set of "augmenting observations" is selected depending on a
    discrepancy measure between GP's predictive mean and GP’s predictive
    uncertainty.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        m: int = 1,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x (d + 1)` tensor of training features,
                where the additional dimension is for the source parameter.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            m: The multiplication factor of the model standard deviation used to select
                points from other sources to add to the Augmented GP.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        if m <= 0:
            raise InputDataError(f"The value of m must be greater than 0, given m={m}.")
        # Divide train_X and train_Y based on the source
        train_S = train_X[..., -1]
        sources = torch.unique(train_S).int()
        if sources.shape[0] == 1:
            raise InputDataError("AGP is meant to be used with more than one source.")
        train_X = [train_X[torch.where(train_S == s)] for s in sources]
        train_Y = [train_Y[torch.where(train_S == s)] for s in sources]
        if train_Yvar is not None:
            train_Yvar = [train_Yvar[torch.where(train_S == s)] for s in sources]
        self.n_true_points = len(train_X[-1])
        self.max_n_cheap_points = max([len(points) for points in train_X[:-1]])

        # Init and fit a SingleTaskGP for each source
        self.models = [
            self._init_fit_gp(
                train_X[s][..., :-1],
                train_Y[s],
                None if train_Yvar is None else train_Yvar[s],
                likelihood,
                covar_module,
                mean_module,
                outcome_transform,
                input_transform,
            )
            for s in sources
        ]

        # Create the training set for the AGP selecting all
        # the observations from the high fidelity source
        # and the reliable observations from the other sources
        reliable_idxs = [
            _get_reliable_observations(
                self.models[-1], self.models[s], train_X[s][..., :-1], m
            )
            for s in sources[:-1]
        ]
        train_X = torch.cat(
            [
                train_X[s] if s == len(sources) - 1 else train_X[s][reliable_idxs[s]]
                for s in sources
            ]
        )[:, :-1]
        train_Y = torch.cat(
            [
                train_Y[s] if s == len(sources) - 1 else train_Y[s][reliable_idxs[s]]
                for s in sources
            ]
        )
        if train_Yvar is not None:
            train_Yvar = torch.cat(
                [
                    (
                        train_Yvar[s]
                        if s == len(sources) - 1
                        else train_Yvar[s][reliable_idxs[s]]
                    )
                    for s in sources
                ]
            )

        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
        )

    def _init_fit_gp(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> SingleTaskGP:
        r"""Initialize and fit a Single Task GP model.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.

        Returns:
            The fitted Single Task GP and its Marginal Log Likelihood.
        """
        gp = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar,
            likelihood=deepcopy(likelihood),
            covar_module=deepcopy(covar_module),
            mean_module=deepcopy(mean_module),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp


def _get_reliable_observations(
    trusty_model: ExactGP,
    other_model: ExactGP,
    x: Tensor,
    m: int = 1,
) -> Tensor:
    r"""Get the points whose posterior mean computed with other_model
    is inside m * trusty_model standard deviation.

    Args:
        trusty_model: The GP model of the trust source.
        other_model: The GP model of a lower fidelity source.
        x: A `batch_shape x n x d` tensor of training features.
        m: The moltiplicator factor of the model standard deviation used to select
            points from other sources to add to the Augmented GP.
    Returns:
        A `batch_shape x N x d` tensor of reliable points
    """
    m0_posterior = trusty_model.posterior(x)
    m0_mu = torch.flatten(m0_posterior.mean)
    m0_sigma = torch.sqrt(torch.flatten(m0_posterior.variance))
    m1_posterior = other_model.posterior(x)
    m1_mu = torch.flatten(m1_posterior.mean)

    return torch.where(torch.abs(m0_mu - m1_mu) < m * m0_sigma)[0]
