#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Fidelity Gaussian Process Regression models based on GPyTorch models.

For more on Multi-Fidelity BO, see the
`tutorial <https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo>`__.

A common use case of multi-fidelity regression modeling is optimizing a
"high-fidelity" function that is expensive to simulate when you have access to
one or more cheaper "lower-fidelity" versions that are not fully accurate but
are correlated with the high-fidelity function. The multi-fidelity model models
both the low- and high-fidelity functions together, including the correlation
between them, which can help you predict and optimize the high-fidelity function
without having to do too many expensive high-fidelity evaluations.

.. [Wu2019mf]
    J. Wu, S. Toscano-Palmerin, P. I. Frazier, and A. G. Wilson. Practical
    multi-fidelity bayesian optimization for hyperparameter tuning. ArXiv 2019.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing import Any

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch.models.kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[3])
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x (d + s)` tensor of training features,
                where `s` is the dimension of the fidelity parameters (either one
                or two).
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            iteration_fidelity: The column index for the training iteration fidelity
                parameter (optional).
            data_fidelities: The column indices for the downsampling fidelity parameter.
                If a list/tuple of indices is provided, a kernel will be constructed for
                each index (optional).
            linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
                of the default kernel.
            nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
                5/2. Only used when `linear_truncated=True`.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                    forward pass.
        """
        self._init_args = {
            "iteration_fidelity": iteration_fidelity,
            "data_fidelities": data_fidelities,
            "linear_truncated": linear_truncated,
            "nu": nu,
            "outcome_transform": outcome_transform,
        }
        if iteration_fidelity is None and (
            data_fidelities is None or len(data_fidelities) == 0
        ):
            raise UnsupportedError(
                f"{self.__class__.__name__} requires at least one fidelity parameter."
            )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self._set_dimensions(train_X=transformed_X, train_Y=train_Y)
        covar_module, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=transformed_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        # Used for subsetting along the output dimension. See Model.subset_output.
        self._subset_batch_dict = {
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }
        if train_Yvar is None:
            self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.to(train_X)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        fidelity_features: list[int],
    ) -> dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`.

        Args:
            training_data: Dictionary of `SupervisedDataset`.
            fidelity_features: Index of fidelity parameter as input columns.
        """
        inputs = super().construct_inputs(training_data=training_data)
        inputs["data_fidelities"] = fidelity_features
        return inputs


def _setup_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: torch.Size,
    iteration_fidelity: int | None,
    data_fidelities: Sequence[int] | None,
    linear_truncated: bool,
    nu: float,
) -> tuple[ScaleKernel, dict]:
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.

    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            `BatchedMultiOutputGPyTorchModel`.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelities: The column indices for the downsampling fidelity parameters
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.

    Returns:
        The covariance module and subset_batch_dict.
    """

    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity = dim + iteration_fidelity
    if data_fidelities is not None:
        data_fidelities = list(data_fidelities)
        for i in range(len(data_fidelities)):
            if data_fidelities[i] < 0:
                data_fidelities[i] = dim + data_fidelities[i]

    kernels = []

    if linear_truncated:
        leading_dims = [iteration_fidelity] if iteration_fidelity is not None else []
        trailing_dims = (
            [[i] for i in data_fidelities] if data_fidelities is not None else [[]]
        )
        for tdims in trailing_dims:
            kernels.append(
                LinearTruncatedFidelityKernel(
                    fidelity_dims=leading_dims + tdims,
                    dimension=dim,
                    nu=nu,
                    batch_shape=aug_batch_shape,
                    power_prior=GammaPrior(3.0, 3.0),
                )
            )
    else:
        non_active_dims = set(data_fidelities or [])
        if iteration_fidelity is not None:
            non_active_dims.add(iteration_fidelity)
        active_dimsX = sorted(set(range(dim)) - non_active_dims)
        kernels.append(
            get_covar_module_with_dim_scaled_prior(
                ard_num_dims=len(active_dimsX),
                batch_shape=aug_batch_shape,
                active_dims=active_dimsX,
            )
        )
        if iteration_fidelity is not None:
            kernels.append(
                ExponentialDecayKernel(
                    batch_shape=aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                    offset_prior=GammaPrior(3.0, 6.0),
                    power_prior=GammaPrior(3.0, 6.0),
                    active_dims=[iteration_fidelity],
                )
            )
        if data_fidelities is not None:
            for data_fidelity in data_fidelities:
                kernels.append(
                    DownsamplingKernel(
                        batch_shape=aug_batch_shape,
                        offset_prior=GammaPrior(3.0, 6.0),
                        power_prior=GammaPrior(3.0, 6.0),
                        active_dims=[data_fidelity],
                    )
                )

    kernel = ProductKernel(*kernels)

    covar_module = ScaleKernel(
        kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15)
    )

    key_prefix = "covar_module.base_kernel.kernels"
    if linear_truncated:
        subset_batch_dict = {}
        for i in range(len(kernels)):
            subset_batch_dict.update(
                {
                    f"{key_prefix}.{i}.raw_power": -2,
                    f"{key_prefix}.{i}.covar_module_unbiased.raw_lengthscale": -3,
                    f"{key_prefix}.{i}.covar_module_biased.raw_lengthscale": -3,
                }
            )
    else:
        subset_batch_dict = {
            f"{key_prefix}.0.raw_lengthscale": -3,
        }

        if iteration_fidelity is not None:
            subset_batch_dict.update(
                {
                    f"{key_prefix}.1.raw_power": -2,
                    f"{key_prefix}.1.raw_offset": -2,
                    f"{key_prefix}.1.raw_lengthscale": -3,
                }
            )
        if data_fidelities is not None:
            start_idx = 2 if iteration_fidelity is not None else 1
            for i in range(start_idx, len(data_fidelities) + start_idx):
                subset_batch_dict.update(
                    {
                        f"{key_prefix}.{i}.raw_power": -2,
                        f"{key_prefix}.{i}.raw_offset": -2,
                    }
                )

    return covar_module, subset_batch_dict
