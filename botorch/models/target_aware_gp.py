#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Any

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


class TargetAwareEnsembleGP(SingleTaskGP):
    r"""A target-aware ensemble GP that takes a set of GPs pre-trained on the
    auxiliary data to improve prediction accuracy of the target task.

    The target outputs are modeled as weighted sum of an offset function and
    functions of each auxiliary sources. The offset function is unknown.
    The ensemble model jointly optimizes the kernel hyperparameters of the
    target task together with the weights.
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
        r"""A dictionary kernel with a scale kernel.

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
                If omitted, use default HalfCauchyPrior with scale = 1.0.
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

        self.base_model_dict = ModuleDict(base_model_dict)
        self.base_model_dict.eval()

        # register ensemble weights
        self.register_parameter(
            name="raw_weight",
            parameter=Parameter(
                torch.zeros(len(self.base_model_dict), device=train_X.device)
            ),
        )
        self.register_constraint(
            param_name="raw_weight",
            constraint=Interval(-10, 10, initial_value=0.1),
            replace=True,
        )
        # set prior on weights so that the unimportant auxiliary
        # sources can be shrunk to 0.
        self.register_prior(
            "weight_prior",
            ensemble_weight_prior.to(train_X),
            lambda m: m.weight**2,
            lambda m, v: m._set_weight(v),
        )
        self.to(train_X)

    @property
    def weight(self) -> Tensor:
        return self.raw_weight_constraint.transform(self.raw_weight)

    @weight.setter
    def weight(self, value: Tensor) -> None:
        self._set_weight(value)

    def _set_weight(self, value: Tensor) -> None:
        value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(
            raw_weight=self.raw_weight_constraint.inverse_transform(value.sqrt())
        )

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        self.training = mode
        for module in self.children():
            if module is self.base_model_dict:
                # set base_model module training always be False
                module.training = False
                module.requires_grad_(False)
            else:
                module.train(mode)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        weighted_means = []
        weighted_covars = []
        for i, m in enumerate(self.base_model_dict.values()):
            posterior = m.posterior(x)
            if abs(self.weight[i]) < WEIGHT_THRESHOLD:  # Or some appropriate threshold
                continue
            if isinstance(posterior, GaussianMixturePosterior):
                mean = posterior.mixture_mean
                covar = posterior.mvn.covariance_matrix.mean(dim=MCMC_DIM)
            else:
                mean = posterior.mean
                covar = posterior.mvn.covariance_matrix
            weighted_means.append(self.weight[i] * mean)
            weighted_covars.append(covar * (self.weight[i] ** 2))
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
                If omitted, use default HalfCauchyPrior with scale = 1.0.
        """
        base_inputs = super().construct_inputs(training_data=training_data)
        return {
            **base_inputs,
            "base_model_dict": base_model_dict,
            "covar_module": covar_module,
            "mean_module": mean_module,
            "ensemble_weight_prior": ensemble_weight_prior,
        }
