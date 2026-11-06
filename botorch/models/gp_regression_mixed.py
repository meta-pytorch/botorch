#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import normalize_indices
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from torch import Tensor


def _default_cat_kernel_factory(
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: list[int] | None,
) -> Kernel:
    r"""Construct the default categorical kernel.

    Args:
        batch_shape: The batch shape of the kernel.
        ard_num_dims: The number of categorical dimensions for ARD.
        active_dims: The categorical dimensions to operate on, or None.

    Returns:
        The instantiated CategoricalKernel.
    """
    return CategoricalKernel(
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_constraint=GreaterThan(1e-06),
    )


def _get_mixed_covar_module(
    aug_batch_shape: torch.Size,
    cat_dims: list[int],
    ord_dims: list[int],
    cont_kernel_factory: Callable[..., Kernel],
    cat_kernel_factory: Callable[..., Kernel],
) -> Kernel:
    r"""Assemble the covariance module for a mixed single-task GP.

    Args:
        aug_batch_shape: The augmented batch shape.
        cat_dims: List of categorical feature dimensions.
        ord_dims: List of ordinal (continuous) feature dimensions.
        cont_kernel_factory: Factory function returning continuous kernel.
        cat_kernel_factory: Factory function returning categorical kernel.

    Returns:
        The assembled covariance module.
    """
    if len(ord_dims) == 0:
        return ScaleKernel(
            cat_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=None,
            )
        )
    sum_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(ord_dims),
            active_dims=ord_dims,
        )
        + ScaleKernel(
            cat_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )
        )
    )
    prod_kernel = ScaleKernel(
        cont_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(ord_dims),
            active_dims=ord_dims,
        )
        * cat_kernel_factory(
            batch_shape=aug_batch_shape,
            ard_num_dims=len(cat_dims),
            active_dims=cat_dims,
        )
    )
    return sum_kernel + prod_kernel


class MixedSingleTaskGP(SingleTaskGP):
    r"""A single-task exact GP model for mixed search spaces.

    This model is similar to ``SingleTaskGP``, but supports mixed search spaces,
    which combine discrete and continuous features, as well as solely discrete
    spaces. It uses a kernel that combines a CategoricalKernel (based on
    Hamming distances) and a regular kernel into a kernel of the form

        K((x1, c1), (x2, c2)) =
            K_cont_1(x1, x2) + K_cat_1(c1, c2) +
            K_cont_2(x1, x2) * K_cat_2(c1, c2)

    where ``xi`` and ``ci`` are the continuous and categorical features of the
    input, respectively. The suffix ``_i`` indicates that we fit different
    lengthscales for the kernels in the sum and product terms. The categorical
    kernel `K_cat` can be customized via ``cat_kernel_factory``.

    Since this model does not provide gradients for the categorical features,
    optimization of the acquisition function will need to be performed in
    a mixed fashion, i.e., treating the categorical features properly as
    discrete optimization variables. We recommend using ``optimize_acqf_mixed.``

    Example:
        >>> train_X = torch.cat(
                [torch.rand(20, 2), torch.randint(3, (20, 1))], dim=-1)
            )
        >>> train_Y = (
                torch.sin(train_X[..., :-1]).sum(dim=1, keepdim=True)
                + train_X[..., -1:]
            )
        >>> model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[-1])
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: None
        | (Callable[[torch.Size, int, list[int]], Kernel]) = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,  # TODO
        cat_kernel_factory: None
        | (Callable[[torch.Size, int, list[int] | None], Kernel]) = None,
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A ``batch_shape x n x d`` tensor of training features.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input ``X`` that should be considered categorical features.
            train_Yvar: An optional ``batch_shape x n x m`` tensor of observed
                measurement noise.
            cont_kernel_factory: A method that accepts  ``batch_shape``,
                ``ard_num_dims``, and ``active_dims`` arguments and returns an
                instantiated GPyTorch ``Kernel`` object to be used as the base
                kernel for the continuous dimensions. If omitted, this model uses
                an ``RBFKernel`` as the kernel for the ordinal parameters.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale). We use a
                ``Standardize`` transform if no ``outcome_transform`` is specified.
                Pass down ``None`` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                forward pass. Only input transforms are allowed which do not
                transform the categorical dimensions. If you want to use it
                for example in combination with a ``OneHotToNumeric`` input
                transform one has to instantiate the transform with
                ``transform_on_train`` == False and pass in the already
                transformed input.
            cat_kernel_factory: A method that accepts ``batch_shape``,
                ``ard_num_dims``, and ``active_dims`` arguments and returns an
                instantiated GPyTorch ``Kernel`` object to be used as the base
                kernel for the categorical dimensions. active_dims is None if all
                inputs are categorical. If omitted, this model uses a
                ``CategoricalKernel`` with ARD.
        """
        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        self._ignore_X_dims_scaling_check = cat_dims
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)

        if cont_kernel_factory is None:
            cont_kernel_factory = get_covar_module_with_dim_scaled_prior

        if cat_kernel_factory is None:
            cat_kernel_factory = _default_cat_kernel_factory

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))
        covar_module = _get_mixed_covar_module(
            aug_batch_shape=aug_batch_shape,
            cat_dims=cat_dims,
            ord_dims=ord_dims,
            cont_kernel_factory=cont_kernel_factory,
            cat_kernel_factory=cat_kernel_factory,
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

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        categorical_features: list[int],
        likelihood: Likelihood | None = None,
    ) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a dict of ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset`` containing the training data.
            categorical_features: Column indices of categorical features.
            likelihood: Optional likelihood used to constuct the model.
        """
        base_inputs = super().construct_inputs(training_data=training_data)
        return {
            **base_inputs,
            "cat_dims": categorical_features,
            "likelihood": likelihood,
        }
