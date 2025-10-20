#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dummy classes and other helpers that are used in multiple test files
should be defined here to avoid relative imports.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model import FantasizeMixin, Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils import add_output_dim
from botorch.models.utils.assorted import fantasize
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from torch import Size, Tensor
from torch.nn.functional import pad


def get_model(
    train_X: Tensor,
    train_Y: Tensor,
    standardize_model: bool = False,
    use_model_list: bool = False,
) -> SingleTaskGP | ModelListGP:
    num_objectives = train_Y.shape[-1]

    if standardize_model:
        if use_model_list:
            outcome_transform = Standardize(m=1)
        else:
            outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None

    if use_model_list:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],
                    outcome_transform=outcome_transform,
                )
                for i in range(num_objectives)
            ]
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )

    return model


def get_fully_bayesian_model(
    train_X: Tensor,
    train_Y: Tensor,
    num_models: int,
    standardize_model: bool = False,
    infer_noise: bool = True,
    **tkwargs: Any,
) -> SaasFullyBayesianSingleTaskGP:
    num_objectives = train_Y.shape[-1]

    if standardize_model:
        outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None
    mcmc_samples = _get_mcmc_samples(
        num_samples=num_models,
        dim=train_X.shape[-1],
        infer_noise=infer_noise,
        **tkwargs,
    )
    train_Yvar = None if infer_noise else torch.full_like(train_Y, 0.01)

    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=outcome_transform,
    )
    model.load_mcmc_samples(mcmc_samples)

    return model


def _get_mcmc_samples(
    num_samples: int, dim: int, infer_noise: bool, **tkwargs
) -> dict[str, Tensor]:
    mcmc_samples = {
        "lengthscale": 1 + torch.rand(num_samples, 1, dim, **tkwargs),
        "outputscale": 1 + torch.rand(num_samples, **tkwargs),
        "mean": torch.randn(num_samples, **tkwargs),
    }
    if infer_noise:
        mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)

    return mcmc_samples


def get_fully_bayesian_model_list(
    train_X: Tensor,
    train_Y: Tensor,
    num_models: int,
    standardize_model: bool,
    infer_noise: bool,
    **tkwargs: Any,
) -> ModelListGP:
    model = ModelListGP(
        *[
            get_fully_bayesian_model(
                train_X, train_Y, num_models, standardize_model, infer_noise, **tkwargs
            )
            for _ in range(2)
        ]
    )
    return model


def get_sample_moments(samples: Tensor, sample_shape: Size) -> tuple[Tensor, Tensor]:
    """Computes the mean and covariance of a set of samples.

    Args:
        samples: A tensor of shape `sample_shape x batch_shape x q`.
        sample_shape: The sample_shape input used while generating the samples using
            the pathwise sampling API.
    """
    sample_dim = len(sample_shape)
    samples = samples.view(-1, *samples.shape[sample_dim:])
    loc = samples.mean(dim=0)
    residuals = (samples - loc).permute(*range(1, samples.ndim), 0)
    return loc, (residuals @ residuals.transpose(-2, -1)) / sample_shape.numel()


def standardize_moments(
    transform: Standardize,
    loc: Tensor,
    covariance_matrix: Tensor,
) -> tuple[Tensor, Tensor]:
    """Standardizes the loc and covariance_matrix using the mean and standard
    deviations from a Standardize transform.
    """
    m = transform.means.squeeze().unsqueeze(-1)
    s = transform.stdvs.squeeze().reciprocal().unsqueeze(-1)
    loc = s * (loc - m)
    correlation_matrix = s.unsqueeze(-1) * covariance_matrix * s.unsqueeze(-2)
    return loc, correlation_matrix


def gen_multi_task_dataset(
    yvar: float | None = None,
    task_values: list[int] | None = None,
    skip_task_features_in_datasets: bool = False,
    **tkwargs,
) -> tuple[MultiTaskDataset, tuple[Tensor, Tensor, Tensor | None]]:
    """Constructs a multi-task dataset with two tasks, each with 10 data points.

    Args:
        yvar: The noise level to use for `train_Yvar`. If None, uses `train_Yvar=None`.
        task_values: The values of the task features. If None, uses [0, 1].
        skip_task_features_in_datasets: If True, the task features are not included in
            Xs of the datasets used to construct the datasets. This is useful for
            testing `MultiTaskDataset`.
    """
    if task_values is not None and skip_task_features_in_datasets:
        raise UnsupportedError(  # pragma: no cover
            "`task_values` and `skip_task_features_in_datasets` can't be used together."
        )
    X = torch.linspace(0, 0.95, 10, **tkwargs) + 0.05 * torch.rand(10, **tkwargs)
    X = X.unsqueeze(dim=-1)
    Y1 = torch.sin(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    Y2 = torch.cos(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    if task_values is None:
        task_values = [0, 1]
    train_X = torch.cat([pad(X, (1, 0), value=i) for i in task_values])
    train_Y = torch.cat([Y1, Y2])

    Yvar1 = None if yvar is None else torch.full_like(Y1, yvar)
    Yvar2 = None if yvar is None else torch.full_like(Y2, 2 * yvar)
    train_Yvar = None if yvar is None else torch.cat([Yvar1, Yvar2])
    Y3 = torch.tan(X * (2 * math.pi)) + torch.randn_like(X) * 0.2
    Yvar3 = None if yvar is None else torch.full_like(Y3, yvar)
    if len(task_values) == 3:
        train_Y = torch.cat([train_Y, Y3])
        if train_Yvar is not None:
            train_Yvar = torch.cat([train_Yvar, Yvar3])
    feature_slice = slice(1, None) if skip_task_features_in_datasets else slice(None)
    datasets = [
        SupervisedDataset(
            X=train_X[:10, feature_slice],
            Y=Y1,
            Yvar=Yvar1,
            feature_names=["task", "X"][feature_slice],
            outcome_names=["y"],
        ),
        SupervisedDataset(
            X=train_X[10:20, feature_slice],
            Y=Y2,
            Yvar=Yvar2,
            feature_names=["task", "X"][feature_slice],
            outcome_names=["y1"],
        ),
    ]
    if len(task_values) == 3:
        datasets.append(
            SupervisedDataset(
                X=train_X[20:, feature_slice],
                Y=Y3,
                Yvar=Yvar3,
                feature_names=["task", "X"][feature_slice],
                outcome_names=["y2"],
            )
        )
    dataset = MultiTaskDataset(
        datasets=datasets,
        target_outcome_name="y",
        task_feature_index=None if skip_task_features_in_datasets else 0,
    )
    return dataset, (train_X, train_Y, train_Yvar)


def get_pvar_expected(
    posterior: TorchPosterior, model: Model, X: Tensor, m: int
) -> Tensor:
    """Computes the expected variance of a posterior after adding the
    predictive noise from the likelihood.

    Args:
        posterior: The posterior to compute the variance of. Must be a
            `TorchPosterior` object.
        model: The model that generated the posterior. If `m > 1`, this must be
            a `BatchedMultiOutputGPyTorchModel`.
        X: The test inputs.
        m: The number of outputs.

    Returns:
        The expected variance of the posterior after adding the observation
        noise from the likelihood.
    """
    X = model.transform_inputs(X)
    lh_kwargs = {}
    odim = -1  # this is the output dimension index

    if m > 1:
        if not isinstance(model, BatchedMultiOutputGPyTorchModel):
            raise UnsupportedError(
                "`get_pvar_expected` only supports `BatchedMultiOutputGPyTorchModel`s."
            )
        # We need to add a batch dimension to the input to be compatible with the
        # augmented batch shape of the model. This also changes the output dimension
        # index.
        X, odim = add_output_dim(X=X, original_batch_shape=model._input_batch_shape)

    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        noise = model.likelihood.noise.mean(dim=-1, keepdim=True)
        broadcasted_shape = torch.broadcast_shapes(noise.shape, X.shape[:-1])
        lh_kwargs["noise"] = noise.expand(broadcasted_shape)

    pvar_exp = model.likelihood(model(X), X, **lh_kwargs).variance
    if m == 1:
        pvar_exp = pvar_exp.unsqueeze(-1)
    pvar_exp = torch.stack(
        [pvar_exp.select(dim=odim, index=i) for i in range(m)], dim=-1
    )

    # If the model has an outcome transform, we need to untransform the
    # variance according to that transform.
    if hasattr(model, "outcome_transform"):
        _, pvar_exp = model.outcome_transform.untransform(
            Y=torch.zeros_like(pvar_exp), Yvar=pvar_exp, X=X
        )

    return pvar_exp


class DummyNonScalarizingPosteriorTransform(PosteriorTransform):
    scalarize = False

    def evaluate(self, Y):
        pass  # pragma: no cover

    def forward(self, posterior):
        pass  # pragma: no cover


class SimpleGPyTorchModel(GPyTorchModel, ExactGP, FantasizeMixin):
    last_fantasize_flag: bool = False

    def __init__(self, train_X, train_Y, outcome_transform=None, input_transform=None):
        r"""
        Args:
            train_X: A tensor of inputs, passed to self.transform_inputs.
            train_Y: Passed to outcome_transform.
            outcome_transform: Transform applied to train_Y.
            input_transform: A Module that performs the input transformation, passed to
                self.transform_inputs.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(transformed_X, train_Y)
        train_Y = train_Y.squeeze(-1)
        likelihood = GaussianLikelihood()
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self._num_outputs = 1
        self.to(train_X)
        self.transformed_call_args = []

    def forward(self, x):
        self.last_fantasize_flag = fantasize.on()
        if self.training:
            x = self.transform_inputs(x)
        self.transformed_call_args.append(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
