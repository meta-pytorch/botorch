#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
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


class _TaskSpecificMean(Module):
    """Wrapper that extracts task-specific means from a MultitaskMean.

    For MultiTaskGP with stacked inputs (where each point includes a task indicator),
    the MultitaskMean outputs [n_points, n_tasks]. This wrapper extracts the
    appropriate mean for each point based on its task indicator, returning [n_points].
    """

    def __init__(self, mean_module: Module, task_feature: int):
        super().__init__()
        self.mean_module = mean_module
        self.task_feature = task_feature

    def forward(self, x: Tensor) -> Tensor:
        # Get all task means: [n_points, n_tasks]
        all_means = self.mean_module(x)

        # Extract task indicator from inputs
        task_indices = x[..., self.task_feature].long()

        # Select the mean for each point's specific task
        # all_means: [..., n_points, n_tasks]
        # task_indices: [..., n_points]
        # result: [..., n_points]
        if all_means.ndim > 1 and all_means.shape[-1] > 1:
            # Use indexing to select task-specific means
            batch_indices = torch.arange(all_means.shape[-2], device=x.device)
            return all_means[..., batch_indices, task_indices]
        return all_means


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

    # Extract kernels from the product kernel structure
    # model.covar_module is a ProductKernel by definition for MTGPs
    # containing data_covar_module * task_covar_module
    from gpytorch.kernels import ProductKernel

    if not isinstance(model.covar_module, ProductKernel):
        # Fallback for non-ProductKernel cases (legacy support)
        import warnings

        warnings.warn(
            f"MultiTaskGP with non-ProductKernel detected "
            f"({type(model.covar_module)}). Consider using "
            "ProductKernel(SomeKernel, IndexKernel) for better compatibility.",
            UserWarning,
            stacklevel=2,
        )
        combined_kernel = model.covar_module
    else:
        # Get the individual kernels from the product kernel
        kernels = model.covar_module.kernels

        # Find data and task kernels based on their active_dims
        data_kernel = None
        task_kernel = None

        for kernel in kernels:
            if hasattr(kernel, "active_dims") and kernel.active_dims is not None:
                if task_index in kernel.active_dims:
                    task_kernel = deepcopy(kernel)
                else:
                    data_kernel = deepcopy(kernel)
            else:
                # If no active_dims on data kernel, add them so downstream
                # helpers don't error
                data_kernel = deepcopy(kernel)
                data_kernel.active_dims = torch.LongTensor(
                    [
                        index
                        for index in range(train_X.shape[-1])
                        if index != task_index
                    ],
                    device=data_kernel.device,
                )

        # If the task kernel can't be found, create it based on the structure
        if task_kernel is None:
            from gpytorch.kernels import IndexKernel

            task_kernel = IndexKernel(
                num_tasks=model.num_tasks,
                rank=model._rank,
                active_dims=[task_index],
            ).to(device=model.covar_module.device, dtype=model.covar_module.dtype)

        # Ensure the data kernel was found
        if data_kernel is None:
            raise ValueError(
                "Could not identify data kernel from ProductKernel. "
                "MTGPs should follow the standard "
                "ProductKernel(IndexKernel, SomeOtherKernel) pattern."
            )

        # Use the existing product kernel structure
        combined_kernel = data_kernel * task_kernel

    # Wrap the MultitaskMean to extract task-specific means for each point
    # MultitaskMean outputs [n_points, n_tasks], but we need [n_points]
    mean_module = _TaskSpecificMean(model.mean_module, task_feature=task_index)

    return _draw_kernel_feature_paths_fallback(
        mean_module=mean_module,
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
