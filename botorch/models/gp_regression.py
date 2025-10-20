#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

These models are often a good starting point and are further documented in the
tutorials.

`SingleTaskGP` is a single-task exact GP model that uses relatively strong priors on
the Kernel hyperparameters, which work best when covariates are normalized to the unit
cube and outcomes are standardized (zero mean, unit variance). By default, this model
uses a `Standardize` outcome transform, which applies this standardization. However,
it does not (yet) use an input transform by default.

`SingleTaskGP` model works in batch mode (each batch having its own hyperparameters).
When the training observations include multiple outputs, `SingleTaskGP` uses
batching to model outputs independently.

`SingleTaskGP` supports multiple outputs. However, as a single-task model,
`SingleTaskGP` should be used only when the outputs are independent and all
use the same training inputs. If outputs are independent but they have different
training inputs, use the `ModelListGP`. When modeling correlations between outputs,
use a multi-task model like `MultiTaskGP`.
"""

from __future__ import annotations

import warnings

import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from torch import Tensor


class SingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""A single-task exact GP model, supporting both known and inferred noise levels.

    A single-task exact GP which, by default, utilizes hyperparameter priors
    from [Hvarfner2024vanilla]_. These priors designed to perform well independently of
    the dimensionality of the problem. Moreover, they suggest a moderately low level of
    noise. Importantly, The model works best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance). For a detailed
    discussion on the hyperparameter priors, see
    https://github.com/meta-pytorch/botorch/discussions/2451.

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the
    same training data. If outputs are independent and outputs have different
    training data, use the ModelListGP. When modeling correlations between
    outputs, use the MultiTaskGP.

    An example of a case in which noise levels are known is online
    experimentation, where noise can be measured using the variability of
    different observations from the same arm, or provided by outside software.
    Another use case is simulation optimization, where the evaluation can
    provide variance estimates, perhaps from bootstrapping. In any case, these
    noise levels can be provided to `SingleTaskGP` as `train_Yvar`.

    `SingleTaskGP` can also be used when the observations are known to be
    noise-free. Noise-free observations can be modeled using arbitrarily small
    noise values, such as `train_Yvar=torch.full_like(train_Y, 1e-6)`.

    Example:
        Model with inferred noise levels:

        >>> import torch
        >>> from botorch.models.gp_regression import SingleTaskGP
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> inferred_noise_model = SingleTaskGP(train_X, train_Y)

        Model with a known observation variance of 0.2:

        >>> train_Yvar = torch.full_like(train_Y, 0.2)
        >>> observed_noise_model = SingleTaskGP(train_X, train_Y, train_Yvar)

        With noise-free observations:

        >>> train_Yvar = torch.full_like(train_Y, 1e-6)
        >>> noise_free_model = SingleTaskGP(train_X, train_Y, train_Yvar)
    """

    train_targets: Tensor
    train_inputs: tuple[Tensor]

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred noise level if `train_Yvar`
                is None, and a `FixedNoiseGaussianLikelihood` with the given
                noise observations if `train_Yvar` is not None.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, uses an `RBFKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform. Note that `.train()` will
                be called on the outcome transform during instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(
                m=train_Y.shape[-1], batch_shape=train_X.shape[:-2]
            )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, train_Yvar = outcome_transform(
                Y=train_Y, Yvar=train_Yvar, X=transformed_X
            )
        # Validate again after applying the transforms
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            ignore_X_dims=ignore_X_dims,
            check_nans_only=covar_module is not None,
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        if likelihood is None:
            if train_Yvar is None:
                likelihood = get_gaussian_likelihood_with_lognormal_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=train_Yvar, batch_shape=self._aug_batch_shape
                )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=transformed_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            )
            # Used for subsetting along the output dimension. See Model.subset_output.
            self._subset_batch_dict = {
                "mean_module.raw_constant": -1,
                "covar_module.raw_lengthscale": -3,
            }
            if train_Yvar is None:
                self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.covar_module: Module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    @classmethod
    def construct_inputs(
        cls, training_data: SupervisedDataset, *, task_feature: int | None = None
    ) -> dict[str, BotorchContainer | Tensor]:
        r"""Construct `SingleTaskGP` keyword arguments from a `SupervisedDataset`.

        Args:
            training_data: A `SupervisedDataset`, with attributes `train_X`,
                `train_Y`, and, optionally, `train_Yvar`.
            task_feature: Deprecated and allowed only for backward
                compatibility; ignored.

        Returns:
            A dict of keyword arguments that can be used to initialize a `SingleTaskGP`,
            with keys `train_X`, `train_Y`, and, optionally, `train_Yvar`.
        """
        if task_feature is not None:
            warnings.warn(
                "`task_feature` is deprecated and will be ignored. In the "
                "future, this will be an error.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().construct_inputs(training_data=training_data)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# Note: There used to be `HeteroskedasticSingleTaskGP` here,
# but due to persistent bugs, it was removed in #2616.
