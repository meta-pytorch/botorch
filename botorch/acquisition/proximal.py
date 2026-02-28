#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AcquisitionFunctions to add proximal weighting of the
acquisition function.
"""

from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.wrapper import AbstractAcquisitionFunctionWrapper
from botorch.exceptions.errors import UnsupportedError
from botorch.models import ModelListGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.utils import t_batch_mode_transform
from botorch.utils.transforms import average_over_ensemble_models
from torch import Tensor


class ProximalAcquisitionFunction(AbstractAcquisitionFunctionWrapper):
    """A wrapper around AcquisitionFunctions to add proximal weighting of the
    acquisition function. The acquisition function is
    weighted via a squared exponential centered at the last training point,
    with varying lengthscales corresponding to ``proximal_weights``. Can only be used
    with acquisition functions based on single batch models. Acquisition functions
    must be positive or ``beta`` must be specified to apply a SoftPlus transform before
    proximal weighting.

    Small values of ``proximal_weights`` corresponds to strong biasing towards recently
    observed points, which smoothes optimization with a small potential decrese in
    convergence rate.



    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> EI_proximal = ProximalAcquisitionFunction(EI, proximal_weights)
        >>> eip = EI_proximal(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        proximal_weights: Tensor,
        transformed_weighting: bool | None = True,
        beta: float | None = None,
    ) -> None:
        r"""Derived Acquisition Function weighted by proximity to recently
        observed point.

        Args:
            acq_function: The base acquisition function, operating on input tensors
                of feature dimension ``d``.
            proximal_weights: A ``d`` dim tensor used to bias locality
                along each axis.
            transformed_weighting: If True, the proximal weights are applied in
                the transformed input space given by
                ``acq_function.model.input_transform`` (if available), otherwise
                proximal weights are applied in real input space.
            beta: If not None, apply a softplus transform to the base acquisition
                function, allows negative base acquisition function values.
        """
        AbstractAcquisitionFunctionWrapper.__init__(self, acq_function=acq_function)
        model = self.acq_func.model

        if hasattr(acq_function, "X_pending"):
            if acq_function.X_pending is not None:
                raise UnsupportedError(
                    "Proximal acquisition function requires `X_pending` to be None."
                )

        self.register_buffer("proximal_weights", proximal_weights)
        self.register_buffer(
            "transformed_weighting", torch.tensor(transformed_weighting)
        )

        self.register_buffer("beta", None if beta is None else torch.tensor(beta))

        _validate_model(model, proximal_weights)

    def set_X_pending(self, X_pending: Tensor | None) -> None:
        r"""Sets the `X_pending` of the base acquisition function."""
        raise UnsupportedError(
            "Proximal acquisition function does not support `X_pending`."
        )

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate base acquisition function with proximal weighting.

        Args:
            X: Input tensor of feature dimension ``d`` .

        Returns:
            Base acquisition function evaluated on tensor ``X`` multiplied by proximal
            weighting.
        """
        model = self.acq_func.model

        train_inputs = model.train_inputs[0]

        # if the model is ModelListGP then get the first model
        if isinstance(model, ModelListGP):
            train_inputs = train_inputs[0]
            model = model.models[0]

        # if the model has more than one output get the first copy of training inputs
        if isinstance(model, BatchedMultiOutputGPyTorchModel) and model.num_outputs > 1:
            train_inputs = train_inputs[0]

        input_transform = _get_input_transform(model)

        last_X = train_inputs[-1].reshape(1, 1, -1)

        # if transformed_weighting, transform X to calculate diff
        # (proximal weighting in transformed space)
        # otherwise,un-transform the last observed point to real space
        # (proximal weighting in real space)
        if input_transform is not None:
            if self.transformed_weighting:
                # transformed space weighting
                diff = input_transform.transform(X) - last_X
            else:
                # real space weighting
                diff = X - input_transform.untransform(last_X)

        else:
            # no transformation
            diff = X - last_X

        M = torch.linalg.norm(diff / self.proximal_weights, dim=-1) ** 2
        proximal_acq_weight = torch.exp(-0.5 * M)

        base_acqf = self.acq_func(X)
        if self.beta is None:
            if torch.any(base_acqf < 0):
                raise RuntimeError(
                    "Cannot use proximal biasing for negative "
                    "acquisition function values, set a value for beta to "
                    "fix this with a softplus transform"
                )

        else:
            base_acqf = torch.nn.functional.softplus(base_acqf, beta=self.beta)

        return base_acqf * proximal_acq_weight.flatten()


def _validate_model(model: Model, proximal_weights: Tensor) -> None:
    r"""Validate model

    Perform vaidation checks on model used in base acquisition function to make sure
    it is compatible with proximal weighting.

    Args:
        model: Model associated with base acquisition function to be validated.
        proximal_weights: A ``d`` dim tensor used to bias locality
                along each axis.
    """

    # check model for train_inputs and single batch
    if not hasattr(model, "train_inputs"):
        raise UnsupportedError("Acquisition function model must have `train_inputs`.")

    # get train inputs for each type of possible model
    if isinstance(model, ModelListGP):
        # ModelListGP models
        # check to make sure that the training inputs and input transformers for each
        # model match and are reversible
        train_inputs = model.train_inputs[0][0]
        input_transform = _get_input_transform(model.models[0])

        for i in range(len(model.train_inputs)):
            if not torch.equal(train_inputs, model.train_inputs[i][0]):
                raise UnsupportedError(
                    "Proximal acquisition function does not support unequal "
                    "training inputs"
                )

            if not input_transform == _get_input_transform(model.models[i]):
                raise UnsupportedError(
                    "Proximal acquisition function does not support non-identical "
                    "input transforms"
                )

    else:
        # any non-ModelListGP model
        train_inputs = model.train_inputs[0]

    # check to make sure that the model is single t-batch (q-batches are allowed)
    if model.batch_shape != torch.Size([]) and train_inputs.shape[1] != 1:
        raise UnsupportedError(
            "Proximal acquisition function requires a single batch model"
        )

    # check to make sure that weights match the training data shape
    if (
        len(proximal_weights.shape) != 1
        or proximal_weights.shape[0] != train_inputs.shape[-1]
    ):
        raise ValueError(
            "`proximal_weights` must be a one dimensional tensor with "
            "same feature dimension as model."
        )


def _get_input_transform(model: Model) -> InputTransform | None:
    """get input transform if defined"""
    try:
        return model.input_transform
    except AttributeError:
        return None
