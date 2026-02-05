#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from types import NoneType
from typing import Any

import torch
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.sampling.pathwise.features import KernelEvaluationMap
from botorch.sampling.pathwise.paths import GeneralizedLinearPath, PathList, SamplePath
from botorch.sampling.pathwise.utils import (
    get_input_transform,
    get_train_inputs,
    get_train_targets,
    TInputTransform,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.transforms import is_ensemble
from botorch.utils.types import DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood, LikelihoodList
from gpytorch.models import ApproximateGP, ExactGP, GP
from gpytorch.variational import VariationalStrategy
from linear_operator.operators import (
    LinearOperator,
    SumLinearOperator,
    ZeroLinearOperator,
)
from torch import Tensor

TPathwiseUpdate = Callable[[GP, Tensor], SamplePath]
GaussianUpdate = Dispatcher("gaussian_update")


def gaussian_update(
    model: GP,
    sample_values: Tensor,
    likelihood: Likelihood | None = DEFAULT,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    r"""Computes a Gaussian pathwise update in exact arithmetic:

     .. code-block:: text

        (f | y)(·) = f(·) + Cov(f(·), y) Cov(y, y)^{-1} (y - f(X) - ε),
                            \_______________________________________/
                                                V
                                    "Gaussian pathwise update"

    where `=` denotes equality in distribution, :math:`f \sim GP(0, k)`,
    :math:`y \sim N(f(X), \Sigma)`, and :math:`\epsilon \sim N(0, \Sigma)`.
    For more information, see [wilson2020sampling]_ and [wilson2021pathwise]_.
    """
    if likelihood is DEFAULT:
        likelihood = getattr(model, "likelihood", None)

    return GaussianUpdate(model, likelihood, sample_values=sample_values, **kwargs)


def _gaussian_update_exact(
    kernel: Kernel,
    points: Tensor,
    target_values: Tensor,
    sample_values: Tensor,
    noise_covariance: Tensor | LinearOperator | None = None,
    scale_tril: Tensor | LinearOperator | None = None,
    input_transform: TInputTransform | None = None,
    is_ensemble: bool = False,
) -> GeneralizedLinearPath:
    # Prepare Cholesky factor of `Cov(y, y)` and noise sample values as needed
    if isinstance(noise_covariance, (NoneType, ZeroLinearOperator)):
        scale_tril = kernel(points).cholesky() if scale_tril is None else scale_tril
    else:
        # Generate noise values with correct shape
        noise_shape = sample_values.shape[-len(target_values.shape) :]
        noise_values = torch.randn(
            noise_shape, device=sample_values.device, dtype=sample_values.dtype
        )
        noise_values = (
            noise_covariance.cholesky() @ noise_values.unsqueeze(-1)
        ).squeeze(-1)
        sample_values = sample_values + noise_values
        scale_tril = (
            SumLinearOperator(kernel(points), noise_covariance).cholesky()
            if scale_tril is None
            else scale_tril
        )

    # Solve for `Cov(y, y)^{-1}(y - f(X) - ε)`
    errors = target_values - sample_values
    weight = torch.cholesky_solve(errors.unsqueeze(-1), scale_tril.to_dense())

    # Define update feature map and paths
    feature_map = KernelEvaluationMap(
        kernel=kernel,
        points=points,
        input_transform=input_transform,
    )
    return GeneralizedLinearPath(
        feature_map=feature_map, weight=weight.squeeze(-1), is_ensemble=is_ensemble
    )


@GaussianUpdate.register(ExactGP, _GaussianLikelihoodBase)
def _gaussian_update_ExactGP(
    model: ExactGP,
    likelihood: _GaussianLikelihoodBase,
    *,
    sample_values: Tensor,
    target_values: Tensor | None = None,
    points: Tensor | None = None,
    noise_covariance: Tensor | LinearOperator | None = None,
    scale_tril: Tensor | LinearOperator | None = None,
) -> GeneralizedLinearPath:
    if points is None:
        (points,) = get_train_inputs(model, transformed=True)

    if target_values is None:
        target_values = get_train_targets(model, transformed=True)

    if noise_covariance is None:
        noise_covariance = likelihood.noise_covar(shape=points.shape[:-1])

    return _gaussian_update_exact(
        kernel=model.covar_module,
        points=points,
        target_values=target_values,
        sample_values=sample_values,
        noise_covariance=noise_covariance,
        scale_tril=scale_tril,
        input_transform=get_input_transform(model),
        is_ensemble=is_ensemble(model),
    )


@GaussianUpdate.register(MultiTaskGP, _GaussianLikelihoodBase)
def _draw_kernel_feature_paths_MultiTaskGP(
    model: MultiTaskGP,
    likelihood: _GaussianLikelihoodBase,
    *,
    sample_values: Tensor,
    target_values: Tensor | None = None,
    points: Tensor | None = None,
    noise_covariance: Tensor | LinearOperator | None = None,
    **ignore: Any,
) -> GeneralizedLinearPath:
    from linear_operator.operators import DiagLinearOperator

    if points is None:
        (points,) = get_train_inputs(model, transformed=True)

    if target_values is None:
        target_values = get_train_targets(model, transformed=True)

    # Build proper noise covariance for stacked format
    # The default noise_covar returns [batch, n_tasks, n, n] which doesn't match
    # our stacked format. Instead, we need a diagonal [n, n] matrix where each
    # point's noise is selected based on its task indicator.
    if noise_covariance is None:
        num_inputs = points.shape[-1]
        task_index = (
            num_inputs + model._task_feature
            if model._task_feature < 0
            else model._task_feature
        )
        task_indices = points[..., task_index].long()

        # Get per-task noise and select for each point
        if hasattr(likelihood, "noise") and likelihood.noise.numel() > 1:
            # MultitaskGaussianLikelihood with per-task noise
            stacked_noise = likelihood.noise[task_indices]
            noise_covariance = DiagLinearOperator(stacked_noise)
        else:
            # Scalar noise case
            noise_covariance = likelihood.noise_covar(shape=points.shape[:-1])

    # Prepare product kernel
    num_inputs = points.shape[-1]
    # TODO: Changed `MultiTaskGP` to normalize the task feature in its constructor.
    task_index = (
        num_inputs + model._task_feature
        if model._task_feature < 0
        else model._task_feature
    )

    # Extract kernels from the product kernel structure
    # model.covar_module is a ProductKernel by definition for MTGPs
    # containing data_covar_module * task_covar_module
    from gpytorch.kernels import ProductKernel

    if not isinstance(model.covar_module, ProductKernel):
        # Fallback for non-ProductKernel cases (legacy support)
        # This should be rare as MTGPs typically use ProductKernels by definition
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
                    [index for index in range(num_inputs) if index != task_index],
                    device=data_kernel.device,
                )

        # If we couldn't find the task kernel, create it based on the structure
        if task_kernel is None:
            from gpytorch.kernels import IndexKernel

            task_kernel = IndexKernel(
                num_tasks=model.num_tasks,
                rank=model._rank,
                active_dims=[task_index],
            ).to(device=model.covar_module.device, dtype=model.covar_module.dtype)

        # Ensure data kernel was found
        if data_kernel is None:
            raise ValueError(
                "Could not identify data kernel from ProductKernel. "
                "MTGPs should follow the standard "
                "ProductKernel(SomeKernel, IndexKernel) pattern."
            )

        # Use the existing product kernel structure
        combined_kernel = data_kernel * task_kernel

    # Return exact update using product kernel
    return _gaussian_update_exact(
        kernel=combined_kernel,
        points=points,
        target_values=target_values,
        sample_values=sample_values,
        noise_covariance=noise_covariance,
        input_transform=get_input_transform(model),
    )


@GaussianUpdate.register(ModelListGP, LikelihoodList)
def _gaussian_update_ModelListGP(
    model: ModelListGP,
    likelihood: LikelihoodList,
    *,
    sample_values: list[Tensor] | Tensor,
    target_values: list[Tensor] | Tensor | None = None,
    **kwargs: Any,
) -> PathList:
    """Computes a Gaussian pathwise update for a list of models.

    Args:
        model: A list of Gaussian process models.
        likelihood: A list of likelihoods.
        sample_values: A list of sample values or a tensor that can be split.
        target_values: A list of target values or a tensor that can be split.
        **kwargs: Additional keyword arguments are passed to subroutines.

    Returns:
        A list of Gaussian pathwise updates.
    """
    if not isinstance(sample_values, list):
        # Handle tensor input by splitting based on number of training points
        # Each model may have different number of training points
        sample_values_list = []
        start_idx = 0
        for submodel in model.models:
            # Get the number of training points for this submodel
            (train_inputs,) = get_train_inputs(submodel, transformed=True)
            n_train = train_inputs.shape[-2]
            # Split the tensor for this submodel
            end_idx = start_idx + n_train
            sample_values_list.append(sample_values[..., start_idx:end_idx])
            start_idx = end_idx
        sample_values = sample_values_list

    if target_values is not None and not isinstance(target_values, list):
        # Similar splitting logic for target values based on training points
        # This ensures each submodel gets its corresponding targets
        target_values_list = []
        start_idx = 0
        for submodel in model.models:
            (train_inputs,) = get_train_inputs(submodel, transformed=True)
            n_train = train_inputs.shape[-2]
            end_idx = start_idx + n_train
            target_values_list.append(target_values[..., start_idx:end_idx])
            start_idx = end_idx
        target_values = target_values_list

    # Create individual paths for each submodel
    paths = []
    for i, submodel in enumerate(model.models):
        # Apply gaussian update to each submodel with its corresponding values
        paths.append(
            gaussian_update(
                model=submodel,
                likelihood=likelihood.likelihoods[i],
                sample_values=sample_values[i],
                target_values=None if target_values is None else target_values[i],
                **kwargs,
            )
        )
    # Return a PathList containing all individual paths
    return PathList(paths=paths)


@GaussianUpdate.register(ApproximateGPyTorchModel, (Likelihood, NoneType))
def _gaussian_update_ApproximateGPyTorchModel(
    model: ApproximateGPyTorchModel,
    likelihood: Likelihood | None,
    **kwargs: Any,
) -> GeneralizedLinearPath:
    return GaussianUpdate(
        model.model, likelihood, input_transform=get_input_transform(model), **kwargs
    )


@GaussianUpdate.register(ApproximateGP, (Likelihood, NoneType))
def _gaussian_update_ApproximateGP(
    model: ApproximateGP, likelihood: Likelihood | None, **kwargs: Any
) -> GeneralizedLinearPath:
    return GaussianUpdate(model, model.variational_strategy, **kwargs)


@GaussianUpdate.register(ApproximateGP, VariationalStrategy)
def _gaussian_update_ApproximateGP_VariationalStrategy(
    model: ApproximateGP,
    variational_strategy: VariationalStrategy,
    *,
    sample_values: Tensor,
    target_values: Tensor | None = None,
    noise_covariance: Tensor | LinearOperator | None = None,
    input_transform: InputTransform | None = None,
    **ignore: Any,
) -> GeneralizedLinearPath:
    # TODO: Account for jitter added by `psd_safe_cholesky`
    if not isinstance(noise_covariance, (NoneType, ZeroLinearOperator)):
        raise NotImplementedError(
            f"`noise_covariance` argument not yet supported for {type(model)}."
        )

    # Inducing points `Z` are assumed to live in transformed space
    batch_shape = model.covar_module.batch_shape
    Z = variational_strategy.inducing_points
    L = variational_strategy._cholesky_factor(
        variational_strategy(Z, prior=True).lazy_covariance_matrix
    ).to(dtype=sample_values.dtype)

    # Generate whitened inducing variables `u`, then location-scale transform
    if target_values is None:
        base_values = variational_strategy.variational_distribution.rsample(
            sample_values.shape[: sample_values.ndim - len(batch_shape) - 1],
        )
        target_values = model.mean_module(Z) + (L @ base_values.unsqueeze(-1)).squeeze(
            -1
        )

    return _gaussian_update_exact(
        kernel=model.covar_module,
        points=Z,
        target_values=target_values,
        sample_values=sample_values,
        scale_tril=L,
        input_transform=input_transform,
        is_ensemble=is_ensemble(model),
    )
