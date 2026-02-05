#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, List, overload, Tuple, TYPE_CHECKING

import torch
from botorch.utils.dispatcher import Dispatcher
from torch import Tensor

if TYPE_CHECKING:
    from botorch.models.model import Model, ModelList

GetTrainInputs = Dispatcher("get_train_inputs")
GetTrainTargets = Dispatcher("get_train_targets")


@overload
def get_train_inputs(model: Model, transformed: bool = False) -> Tuple[Tensor, ...]:
    pass  # pragma: no cover


@overload
def get_train_inputs(model: ModelList, transformed: bool = False) -> List[...]:
    pass  # pragma: no cover


def get_train_inputs(model: Any, transformed: bool = False):
    """Get training inputs from a model, with optional transformation handling.

    Args:
        model: A BoTorch Model or ModelList.
        transformed: If True, return the transformed inputs. If False, return the
            original (untransformed) inputs.

    Returns:
        A tuple of training input tensors for Model, or a list of tuples for ModelList.
    """
    # Lazy import to avoid circular dependencies
    _register_get_train_inputs()
    return GetTrainInputs(model, transformed=transformed)


def _register_get_train_inputs():
    """Register dispatcher implementations for get_train_inputs (lazy)."""
    # Only register once
    if hasattr(_register_get_train_inputs, "_registered"):
        return
    _register_get_train_inputs._registered = True

    from botorch.models.approximate_gp import SingleTaskVariationalGP
    from botorch.models.model import Model, ModelList

    @GetTrainInputs.register(Model)
    def _get_train_inputs_Model(
        model: Model, transformed: bool = False
    ) -> Tuple[Tensor]:
        if not transformed:
            original_train_input = getattr(model, "_original_train_inputs", None)
            if torch.is_tensor(original_train_input):
                return (original_train_input,)

        (X,) = model.train_inputs
        transform = getattr(model, "input_transform", None)
        if transform is None:
            return (X,)

        if model.training:
            return (transform.forward(X) if transformed else X,)
        return (X if transformed else transform.untransform(X),)

    @GetTrainInputs.register(SingleTaskVariationalGP)
    def _get_train_inputs_SingleTaskVariationalGP(
        model: SingleTaskVariationalGP, transformed: bool = False
    ) -> Tuple[Tensor]:
        (X,) = model.model.train_inputs
        if model.training != transformed:
            return (X,)

        transform = getattr(model, "input_transform", None)
        if transform is None:
            return (X,)

        return (transform.forward(X) if model.training else transform.untransform(X),)

    @GetTrainInputs.register(ModelList)
    def _get_train_inputs_ModelList(
        model: ModelList, transformed: bool = False
    ) -> List[...]:
        return [get_train_inputs(m, transformed=transformed) for m in model.models]


@overload
def get_train_targets(model: Model, transformed: bool = False) -> Tensor:
    pass  # pragma: no cover


@overload
def get_train_targets(model: ModelList, transformed: bool = False) -> List[...]:
    pass  # pragma: no cover


def get_train_targets(model: Any, transformed: bool = False):
    """Get training targets from a model, with optional transformation handling.

    Args:
        model: A BoTorch Model or ModelList.
        transformed: If True, return the transformed targets. If False, return the
            original (untransformed) targets.

    Returns:
        Training target tensors for Model, or a list of tensors for ModelList.
    """
    # Lazy import to avoid circular dependencies
    _register_get_train_targets()
    return GetTrainTargets(model, transformed=transformed)


def _register_get_train_targets():
    """Register dispatcher implementations for get_train_targets (lazy)."""
    # Only register once
    if hasattr(_register_get_train_targets, "_registered"):
        return
    _register_get_train_targets._registered = True

    from botorch.models.approximate_gp import SingleTaskVariationalGP
    from botorch.models.model import Model, ModelList

    @GetTrainTargets.register(Model)
    def _get_train_targets_Model(model: Model, transformed: bool = False) -> Tensor:
        Y = model.train_targets

        # Note: Avoid using `get_output_transform` here since it creates a Module
        transform = getattr(model, "outcome_transform", None)
        if transformed or transform is None:
            return Y

        if model.num_outputs == 1:
            return transform.untransform(Y.unsqueeze(-1))[0].squeeze(-1)
        return transform.untransform(Y.transpose(-2, -1))[0].transpose(-2, -1)

    @GetTrainTargets.register(SingleTaskVariationalGP)
    def _get_train_targets_SingleTaskVariationalGP(
        model: Model, transformed: bool = False
    ) -> Tensor:
        Y = model.model.train_targets
        transform = getattr(model, "outcome_transform", None)
        if transformed or transform is None:
            return Y

        if model.num_outputs == 1:
            return transform.untransform(Y.unsqueeze(-1))[0].squeeze(-1)

        # SingleTaskVariationalGP.__init__ doesn't bring the
        # multioutput dimension inside
        return transform.untransform(Y)[0]

    @GetTrainTargets.register(ModelList)
    def _get_train_targets_ModelList(
        model: ModelList, transformed: bool = False
    ) -> List[...]:
        return [get_train_targets(m, transformed=transformed) for m in model.models]
