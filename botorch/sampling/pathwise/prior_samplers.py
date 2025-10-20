#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

from typing import Any

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.pathwise.features import gen_kernel_features
from botorch.sampling.pathwise.features.generators import TKernelFeatureMapGenerator
from botorch.sampling.pathwise.paths import GeneralizedLinearPath, PathList, SamplePath
from botorch.sampling.pathwise.utils import (
    get_input_transform,
    get_output_transform,
    get_train_inputs,
    TInputTransform,
    TOutputTransform,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.utils.transforms import is_ensemble
from gpytorch.kernels import Kernel
from gpytorch.models import ApproximateGP, ExactGP, GP
from gpytorch.variational import _VariationalStrategy
from torch import Size, Tensor
from torch.nn import Module

TPathwisePriorSampler = Callable[[GP, Size], SamplePath]
DrawKernelFeaturePaths = Dispatcher("draw_kernel_feature_paths")


def draw_kernel_feature_paths(
    model: GP, sample_shape: Size, **kwargs: Any
) -> GeneralizedLinearPath:
    r"""Draws functions from a Bayesian-linear-model-based approximation to a GP prior.

    When evaluted, sample paths produced by this method return Tensors with dimensions
    `sample_dims x batch_dims x [joint_dim]`, where `joint_dim` denotes the penultimate
    dimension of the input tensor. For multioutput models, outputs are returned as the
    final batch dimension.

    Args:
        model: The prior over functions.
        sample_shape: The shape of the sample paths to be drawn.
    """
    return DrawKernelFeaturePaths(model, sample_shape=sample_shape, **kwargs)


def _draw_kernel_feature_paths_fallback(
    num_inputs: int,
    mean_module: Module | None,
    covar_module: Kernel,
    sample_shape: Size,
    num_features: int = 1024,
    map_generator: TKernelFeatureMapGenerator = gen_kernel_features,
    input_transform: TInputTransform | None = None,
    output_transform: TOutputTransform | None = None,
    weight_generator: Callable[[Size], Tensor] | None = None,
    is_ensemble: bool = False,
) -> GeneralizedLinearPath:
    # Generate a kernel feature map
    feature_map = map_generator(
        kernel=covar_module,
        num_inputs=num_inputs,
        num_outputs=num_features,
    )

    # Sample random weights with which to combine kernel features
    if weight_generator is None:
        # weight is sample_shape x batch_shape x num_outputs
        weight = draw_sobol_normal_samples(
            n=sample_shape.numel() * covar_module.batch_shape.numel(),
            d=feature_map.num_outputs,
            device=covar_module.device,
            dtype=covar_module.dtype,
        ).reshape(sample_shape + covar_module.batch_shape + (feature_map.num_outputs,))
    else:
        weight = weight_generator(
            sample_shape + covar_module.batch_shape + (feature_map.num_outputs,)
        ).to(device=covar_module.device, dtype=covar_module.dtype)

    # Return the sample paths
    return GeneralizedLinearPath(
        feature_map=feature_map,
        weight=weight,
        bias_module=mean_module,
        input_transform=input_transform,
        output_transform=output_transform,
        is_ensemble=is_ensemble,
    )


@DrawKernelFeaturePaths.register(ExactGP)
def _draw_kernel_feature_paths_ExactGP(
    model: ExactGP, **kwargs: Any
) -> GeneralizedLinearPath:
    (train_X,) = get_train_inputs(model, transformed=False)
    return _draw_kernel_feature_paths_fallback(
        num_inputs=train_X.shape[-1],
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        input_transform=get_input_transform(model),
        output_transform=get_output_transform(model),
        is_ensemble=is_ensemble(model),
        **kwargs,
    )


@DrawKernelFeaturePaths.register(ModelListGP)
def _draw_kernel_feature_paths_list(
    model: ModelListGP,
    join: Callable[[list[Tensor]], Tensor] | None = None,
    **kwargs: Any,
) -> PathList:
    paths = [draw_kernel_feature_paths(m, **kwargs) for m in model.models]
    return PathList(paths=paths, join=join)


@DrawKernelFeaturePaths.register(ApproximateGPyTorchModel)
def _draw_kernel_feature_paths_ApproximateGPyTorchModel(
    model: ApproximateGPyTorchModel, **kwargs: Any
) -> GeneralizedLinearPath:
    (train_X,) = get_train_inputs(model, transformed=False)
    return DrawKernelFeaturePaths(
        model.model,
        num_inputs=train_X.shape[-1],
        input_transform=get_input_transform(model),
        output_transform=get_output_transform(model),
        **kwargs,
    )


@DrawKernelFeaturePaths.register(ApproximateGP)
def _draw_kernel_feature_paths_ApproximateGP(
    model: ApproximateGP, **kwargs: Any
) -> GeneralizedLinearPath:
    return DrawKernelFeaturePaths(model, model.variational_strategy, **kwargs)


@DrawKernelFeaturePaths.register(ApproximateGP, _VariationalStrategy)
def _draw_kernel_feature_paths_ApproximateGP_fallback(
    model: ApproximateGP,
    _: _VariationalStrategy,
    *,
    num_inputs: int,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    return _draw_kernel_feature_paths_fallback(
        num_inputs=num_inputs,
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        is_ensemble=is_ensemble(model),
        **kwargs,
    )
