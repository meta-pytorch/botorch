#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Target-aware GP models.

References

.. [Feng2025experimenting]
    Q. Feng, S. Daulton, B. Letham, M. Balandat, and E. Bakshy.
    Experimenting, Fast and Slow: Bayesian Optimization of Long-term Outcomes
    with Online Experiments. Proceedings of the 31st ACM SIGKDD Conference on
    Knowledge Discovery and Data Mining, 2025.
"""

from __future__ import annotations

from typing import Any, Self

import torch
from botorch.models.fully_bayesian import MCMC_DIM
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.utils.datasets import SupervisedDataset
from gpytorch.constraints import Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.means.mean import Mean
from gpytorch.priors import HalfCauchyPrior, Prior
from linear_operator.operators import PsdSumLinearOperator
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.modules import ModuleDict

WEIGHT_THRESHOLD = 0.01


class _FixedModelDict(ModuleDict):
    r"""A module dictionary that keeps its contained models in evaluation mode."""

    def train(self, mode: bool = True) -> Self:
        r"""Keep the fixed models in evaluation mode regardless of parent mode."""
        return super().train(mode=False)


class TargetAwareEnsembleGP(SingleTaskGP):
    r"""A target-aware ensemble GP that takes a set of GPs pre-trained on the
    auxiliary data to improve prediction accuracy of the target task.

    The target outputs are modeled as a weighted sum of an offset function and
    functions of each auxiliary source. The offset function is unknown.
    The ensemble model jointly optimizes the kernel hyperparameters of the
    target task together with the weights.

    Each signed ensemble weight is constrained to ``[-10, 10]`` by transforming
    an unconstrained ``raw_weight`` parameter. The prior is evaluated on the
    squared weights, so it shrinks positive and negative weights toward zero
    symmetrically. When sampling from the prior, the positive square root is
    used because a squared weight does not encode its original sign.
    For example, assigning weights ``[-0.5, 1.5]`` preserves those signed
    coefficients; the predictive mean uses them directly, while the predictive
    covariance uses their squares.

    This model is described in [Feng2025experimenting]_.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        base_model_dict: dict[str, GPyTorchModel],
        train_Yvar: Tensor | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        ensemble_weight_prior: Prior | None = None,
    ) -> None:
        r"""Initialize a target-aware ensemble GP.

        Args:
            train_X: (n x d) X training data of target task.
            train_Y: (n x 1) Y training data of target task.
            train_Yvar: (n x 1) Noise variances of each training Y.
            base_model_dict: Dict of GP models that each corresponds to a model trained
                on an auxiliary dataset. Keys are the name of auxiliary dataset.
            covar_module: The module computing the covariance (Kernel) matrix for the
                target data. If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used for the target data. If omitted,
                use a `ConstantMean`.
            ensemble_weight_prior: The prior over the weights of the ensemble model.
                The prior must have nonnegative support because it is registered on
                the squared weights. If omitted, use a `HalfCauchyPrior` with scale
                1.0.
        """
        if ensemble_weight_prior is None:
            ensemble_weight_prior = HalfCauchyPrior(scale=1.0)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.base_model_dict = _FixedModelDict(base_model_dict)
        self.base_model_dict.eval()
        self.base_model_dict.requires_grad_(False)

        # The constraint's initial value is expressed in transformed space. GPyTorch
        # inverse-transforms it and replaces the zero-initialized raw parameter.
        self.register_parameter(
            name="raw_weight",
            parameter=Parameter(
                torch.zeros(len(self.base_model_dict), device=train_X.device)
            ),
        )
        self.register_constraint(
            param_name="raw_weight",
            constraint=Interval(
                lower_bound=-10,
                upper_bound=10,
                initial_value=0.1,
            ),
            replace=True,
        )
        # set prior on weights so that the unimportant auxiliary
        # sources can be shrunk to 0.
        self.register_prior(
            name="weight_prior",
            prior=ensemble_weight_prior.to(train_X),
            param_or_closure=lambda m: m.weight**2,
            setting_closure=lambda m, v: m._set_weight_from_prior(v),
        )
        self.to(train_X)

    @property
    def weight(self) -> Tensor:
        r"""The signed ensemble coefficients mapped from raw space to ``[-10, 10]``."""
        return self.raw_weight_constraint.transform(self.raw_weight)

    @weight.setter
    def weight(self, value: Tensor) -> None:
        self._set_weight(value=value)

    def _set_weight(self, value: Tensor) -> None:
        value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(raw_weight=self.raw_weight_constraint.inverse_transform(value))

    def _set_weight_from_prior(self, value: Tensor) -> None:
        # The prior is registered on weight ** 2, so its samples are squared weights.
        self._set_weight(value=torch.as_tensor(value).to(self.raw_weight).sqrt())

    def train(self, mode: bool = True) -> Self:
        r"""Set the model's mode while leaving the fixed base models in eval mode."""
        return super().train(mode=mode)

    def forward(self, x: Tensor) -> MultivariateNormal:
        r"""Compute the ensemble's predictive distribution.

        The predictive mean is the offset GP mean plus each base-model mean
        scaled by its signed weight. The covariance is the offset GP covariance
        plus each base-model covariance scaled by the squared weight. Base models
        whose absolute weight is below `WEIGHT_THRESHOLD` are not evaluated.

        Args:
            x: The points at which to evaluate the model.

        Returns:
            The joint target-aware predictive distribution at `x`.
        """
        if self.training:
            x = self.transform_inputs(x)
        weighted_means = []
        weighted_covars = []
        weights = self.weight
        for i, m in enumerate(self.base_model_dict.values()):
            weight = weights[i]
            if weight.abs() < WEIGHT_THRESHOLD:
                continue
            posterior = m.posterior(x)
            if isinstance(posterior, GaussianMixturePosterior):
                mean = posterior.mixture_mean
                covar = posterior.mvn.covariance_matrix.mean(dim=MCMC_DIM)
            else:
                mean = posterior.mean
                covar = posterior.mvn.covariance_matrix
            weighted_means.append(weight * mean)
            weighted_covars.append(covar * weight.square())
        # obtain mean and covar from the offset function
        weighted_means.append(self.mean_module(x).unsqueeze(-1))
        weighted_covars.append(self.covar_module(x))
        # average across a list of posteriors
        mean_x = torch.stack(weighted_means).sum(dim=0).squeeze(-1)
        covar_x = PsdSumLinearOperator(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        base_model_dict: dict[str, GPyTorchModel],
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        ensemble_weight_prior: Prior | None = None,
    ) -> dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`.

        Args:
            training_data: A `SupervisedDataset` containing the training data for the
                target task only.
            base_model_dict: Dict of GP models that each corresponds to a model trained
                on an auxiliary dataset. Keys are the name of auxiliary dataset.
            covar_module: The module computing the covariance (Kernel) matrix for the
                target data. If omitted, use a `MaternKernel`.
            mean_module: The mean function to be used for the target data. If omitted,
                use a `ConstantMean`.
            ensemble_weight_prior: The prior over the weights of the ensemble model.
                The prior must have nonnegative support because it is registered on
                the squared weights. If omitted, use a `HalfCauchyPrior` with scale
                1.0.
        """
        base_inputs = super().construct_inputs(training_data=training_data)
        return {
            **base_inputs,
            "base_model_dict": base_model_dict,
            "covar_module": covar_module,
            "mean_module": mean_module,
            "ensemble_weight_prior": ensemble_weight_prior,
        }
