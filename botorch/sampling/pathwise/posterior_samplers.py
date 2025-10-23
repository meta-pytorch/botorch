#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
.. [wilson2020sampling]
    J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. Efficiently
    sampling functions from Gaussian process posteriors. International Conference on
    Machine Learning (2020).

.. [wilson2021pathwise]
    J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. Pathwise
    Conditioning of Gaussian Processes. Journal of Machine Learning Research (2021).
"""

from __future__ import annotations

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.deterministic import GenericDeterministicModel, MatheronPathModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.pathwise.paths import PathDict, PathList, SamplePath
from botorch.sampling.pathwise.prior_samplers import (
    draw_kernel_feature_paths,
    TPathwisePriorSampler,
)
from botorch.sampling.pathwise.update_strategies import gaussian_update, TPathwiseUpdate
from botorch.sampling.pathwise.utils import (
    get_output_transform,
    get_train_inputs,
    get_train_targets,
    TInputTransform,
    TOutputTransform,
)
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.dispatcher import Dispatcher
from gpytorch.models import ApproximateGP, ExactGP, GP
from torch import Size

DrawMatheronPaths = Dispatcher("draw_matheron_paths")


class MatheronPath(PathDict):
    r"""Represents function draws from a GP posterior via Matheron's rule:

    .. code-block:: text

                  "Prior path"
                       v
        (f | y)(·) = f(·) + Cov(f(·), y) Cov(y, y)^{-1} (y - f(X) - ε),
                            \_______________________________________/
                                                v
                                          "Update path"

    where `=` denotes equality in distribution, :math:`f \sim GP(0, k)`,
    :math:`y \sim N(f(X), \Sigma)`, and :math:`\epsilon \sim N(0, \Sigma)`.
    For more information, see [wilson2020sampling]_ and [wilson2021pathwise]_.
    """

    def __init__(
        self,
        prior_paths: SamplePath,
        update_paths: SamplePath,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a MatheronPath instance.

        Args:
            prior_paths: Sample paths used to represent the prior.
            update_paths: Sample paths used to represent the data.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """

        super().__init__(
            join=sum,
            paths={"prior_paths": prior_paths, "update_paths": update_paths},
            input_transform=input_transform,
            output_transform=output_transform,
        )


def get_matheron_path_model(
    model: GP, sample_shape: Size | None = None, ensemble_as_batch: bool = False
) -> GenericDeterministicModel:
    r"""Generates a deterministic model using a single Matheron path drawn
    from the model's posterior.

    The deterministic model evalutes the output of `draw_matheron_paths`,
    and reshapes it to mimic the output behavior of the model's posterior.

    Args:
        model: The model whose posterior is to be sampled.
        sample_shape: The shape of the sample paths to be drawn, if an ensemble
            of sample paths is desired. If this is specified, the resulting
            deterministic model will behave as if the `sample_shape` is prepended
            to the `batch_shape` of the model. The inputs used to evaluate the model
            must be adjusted to match.
        ensemble_as_batch: If True, and model is an ensemble model, the resulting path
            model will treat the ensemble dimension as a batch dimension, which means
            that its inputs have to contain the ensemble dimension in the -3 position,
            i.e. `batch_shape x ensemble_size x q x d`. This is used when optimizing the
            paths of all members of an ensemble jointly, with distinct optima for each
            member of the ensemble.

    Returns:
        A deterministic model that evaluates the Matheron path.
    """
    # Delegate to the MatheronPathModel class
    return MatheronPathModel(
        model=model,
        sample_shape=sample_shape,
        ensemble_as_batch=ensemble_as_batch,
    )


def draw_matheron_paths(
    model: GP,
    sample_shape: Size,
    prior_sampler: TPathwisePriorSampler = draw_kernel_feature_paths,
    update_strategy: TPathwiseUpdate = gaussian_update,
) -> MatheronPath:
    r"""Generates function draws from (an approximate) Gaussian process posterior.

    When evaluted, sample paths produced by this method return Tensors with dimensions
    `sample_dims x batch_dims x [joint_dim]`, where `joint_dim` denotes the penultimate
    dimension of the input tensor. For multioutput models, outputs are returned as the
    final batch dimension.

    Args:
        model: Gaussian process whose posterior is to be sampled.
        sample_shape: Sizes of sample dimensions.
        prior_sample: A callable that takes a model and a sample shape and returns
            a set of sample paths representing the prior.
        update_strategy: A callable that takes a model and a tensor of prior process
            values and returns a set of sample paths representing the data.
    """

    return DrawMatheronPaths(
        model,
        sample_shape=sample_shape,
        prior_sampler=prior_sampler,
        update_strategy=update_strategy,
    )


@DrawMatheronPaths.register(ModelListGP)
def _draw_matheron_paths_ModelListGP(
    model: ModelListGP,
    sample_shape: Size,
    *,
    prior_sampler: TPathwisePriorSampler = draw_kernel_feature_paths,
    update_strategy: TPathwiseUpdate = gaussian_update,
):
    return PathList(
        [
            draw_matheron_paths(
                model=m,
                sample_shape=sample_shape,
                prior_sampler=prior_sampler,
                update_strategy=update_strategy,
            )
            for m in model.models
        ]
    )


@DrawMatheronPaths.register(ExactGP)
def _draw_matheron_paths_ExactGP(
    model: ExactGP,
    *,
    sample_shape: Size,
    prior_sampler: TPathwisePriorSampler,
    update_strategy: TPathwiseUpdate,
) -> MatheronPath:
    (train_X,) = get_train_inputs(model, transformed=True)
    train_Y = get_train_targets(model, transformed=True)
    with delattr_ctx(model, "outcome_transform"):
        # Generate draws from the prior
        prior_paths = prior_sampler(model=model, sample_shape=sample_shape)
        sample_values = prior_paths.forward(train_X)

        # Compute pathwise updates
        update_paths = update_strategy(
            model=model,
            sample_values=sample_values,
            target_values=train_Y,
        )

    return MatheronPath(
        prior_paths=prior_paths,
        update_paths=update_paths,
        output_transform=get_output_transform(model),
    )


@DrawMatheronPaths.register((ApproximateGP, ApproximateGPyTorchModel))
def _draw_matheron_paths_ApproximateGP(
    model: ApproximateGP | ApproximateGPyTorchModel,
    *,
    sample_shape: Size,
    prior_sampler: TPathwisePriorSampler,
    update_strategy: TPathwiseUpdate,
) -> MatheronPath:
    # Note: Inducing points are assumed to be pre-transformed
    Z = (
        model.model.variational_strategy.inducing_points
        if isinstance(model, ApproximateGPyTorchModel)
        else model.variational_strategy.inducing_points
    )
    with delattr_ctx(model, "outcome_transform"):
        # Generate draws from the prior
        prior_paths = prior_sampler(model=model, sample_shape=sample_shape)
        sample_values = prior_paths.forward(Z)  # `forward` bypasses transforms

        # Compute pathwise updates
        update_paths = update_strategy(model=model, sample_values=sample_values)

    return MatheronPath(
        prior_paths=prior_paths,
        update_paths=update_paths,
        output_transform=get_output_transform(model),
    )
