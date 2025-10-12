#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, List

import torch
from botorch import models
from botorch.sampling.pathwise.features import gen_kernel_feature_map
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
        **kwargs: Additional keyword arguments are passed to subroutines.
    """
    return DrawKernelFeaturePaths(model, sample_shape=sample_shape, **kwargs)


def _draw_kernel_feature_paths_fallback(
    mean_module: Module | None,
    covar_module: Kernel,
    sample_shape: Size,
    map_generator: TKernelFeatureMapGenerator = gen_kernel_feature_map,
    input_transform: TInputTransform | None = None,
    output_transform: TOutputTransform | None = None,
    weight_generator: Callable[[Size], Tensor] | None = None,
    is_ensemble: bool = False,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    r"""Generate sample paths from a kernel-based prior using feature maps.

    Generates a feature map for the kernel and combines it with random weights to
    create sample paths. The weights are either generated using Sobol sequences or
    provided by a custom weight generator.

    Args:
        mean_module: Optional mean function to add to the sample paths.
        covar_module: The kernel to generate features for.
        sample_shape: The shape of the sample paths to be drawn.
        map_generator: A callable that generates feature maps from kernels.
            Defaults to :func:`gen_kernel_feature_map`.
        input_transform: Optional transform applied to input before feature generation.
        output_transform: Optional transform applied to output after feature generation.
        weight_generator: Optional callable to generate random weights. If None,
            uses Sobol sequences to generate normally distributed weights.
        **kwargs: Additional arguments passed to :func:`map_generator`.
    """
    feature_map = map_generator(kernel=covar_module, **kwargs)

    weight_shape = (
        *sample_shape,
        *covar_module.batch_shape,
        *feature_map.output_shape,
    )
    if weight_generator is None:
        # weight is sample_shape x batch_shape x num_outputs
        weight = draw_sobol_normal_samples(
            n=sample_shape.numel() * covar_module.batch_shape.numel(),
            d=feature_map.output_shape.numel(),
            device=covar_module.device,
            dtype=covar_module.dtype,
        ).reshape(weight_shape)
    else:
        weight = weight_generator(weight_shape).to(
            device=covar_module.device, dtype=covar_module.dtype
        )

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
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        input_transform=get_input_transform(model),
        output_transform=get_output_transform(model),
        is_ensemble=is_ensemble(model),
        num_ambient_inputs=train_X.shape[-1],
        **kwargs,
    )


@DrawKernelFeaturePaths.register(models.ModelListGP)
def _draw_kernel_feature_paths_ModelListGP(
    model: models.ModelListGP,
    reducer: Callable[[List[Tensor]], Tensor] | None = None,
    **kwargs: Any,
) -> PathList:
    paths = [draw_kernel_feature_paths(m, **kwargs) for m in model.models]
    return PathList(paths=paths, reducer=reducer)


@DrawKernelFeaturePaths.register(models.MultiTaskGP)
def _draw_kernel_feature_paths_MultiTaskGP(
    model: models.MultiTaskGP, **kwargs: Any
) -> GeneralizedLinearPath:
    (train_X,) = get_train_inputs(model, transformed=False)
    num_ambient_inputs = train_X.shape[-1]
    task_index = (
        num_ambient_inputs + model._task_feature
        if model._task_feature < 0
        else model._task_feature
    )

    # MultiTaskGP *always* wraps data_covar_module and task_covar_module in a
    # ProductKernel (see MTGP implementation).  If that invariant is violated we
    # raise an error rather than silently guessing how to proceed.

    from gpytorch.kernels import ProductKernel

    if not isinstance(model.covar_module, ProductKernel):
        raise RuntimeError(
            "Expected `model.covar_module` to be a ProductKernel (data × task), "
            "but found {type(model.covar_module).__name__}. If you are wrapping "
            "kernels manually please combine them with gpytorch.kernels.ProductKernel "
            "so the path-wise utilities can reason about the structure."
        )

    # The product already represents data_kernel * task_kernel; we can pass it
    # straight through to downstream routines.
    combined_kernel = model.covar_module

    # Ensure the data kernel inside the product has `active_dims` set; this is
    # required downstream by `get_kernel_num_inputs`.  MTGPs created via the
    # public constructor already do this, but if a user manually overwrote the
    # `covar_module` we may need to patch it up here.
    kernels = combined_kernel.kernels  # type: ignore[attr-defined]
    for k in kernels:
        if getattr(k, "active_dims", None) is None:
            k.active_dims = torch.LongTensor(
                [idx for idx in range(num_ambient_inputs) if idx != task_index],
                device=k.device,
            )

    return _draw_kernel_feature_paths_fallback(
        mean_module=model.mean_module,
        covar_module=combined_kernel,
        input_transform=get_input_transform(model),
        output_transform=get_output_transform(model),
        num_ambient_inputs=num_ambient_inputs,
        **kwargs,
    )


@DrawKernelFeaturePaths.register(models.ApproximateGPyTorchModel)
def _draw_kernel_feature_paths_ApproximateGPyTorchModel(
    model: models.ApproximateGPyTorchModel, **kwargs: Any
) -> GeneralizedLinearPath:
    (train_X,) = get_train_inputs(model, transformed=False)
    return DrawKernelFeaturePaths(
        model.model,
        input_transform=get_input_transform(model),
        output_transform=get_output_transform(model),
        num_ambient_inputs=train_X.shape[-1],
        **kwargs,
    )


@DrawKernelFeaturePaths.register(ApproximateGP)
def _draw_kernel_feature_paths_ApproximateGP(
    model: ApproximateGP, **kwargs: Any
) -> GeneralizedLinearPath:
    return DrawKernelFeaturePaths(model, model.variational_strategy, **kwargs)


@DrawKernelFeaturePaths.register(ApproximateGP, _VariationalStrategy)
def _draw_kernel_feature_paths_ApproximateGP_fallback(
    model: ApproximateGP, _: _VariationalStrategy, **kwargs: Any
) -> GeneralizedLinearPath:
    return _draw_kernel_feature_paths_fallback(
        mean_module=model.mean_module,
        covar_module=model.covar_module,
        is_ensemble=is_ensemble(model),
        **kwargs,
    )
