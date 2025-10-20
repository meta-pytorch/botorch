#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for building model-based closures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import chain, repeat
from types import NoneType
from typing import Any

from botorch.optim.closures.core import ForwardBackwardClosure
from botorch.utils.dispatcher import Dispatcher, type_bypassing_encoder
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    MarginalLogLikelihood,
    SumMarginalLogLikelihood,
)
from torch import Tensor
from torch.utils.data import DataLoader

GetLossClosure = Dispatcher("get_loss_closure", encoder=type_bypassing_encoder)


def get_loss_closure(
    mll: MarginalLogLikelihood,
    data_loader: DataLoader | None = None,
    **kwargs: Any,
) -> Callable[[], Tensor]:
    r"""Public API for GetLossClosure dispatcher.

    This method, and the dispatcher that powers it, acts as a clearing house
    for factory functions that define how `mll` is evaluated.

    Users may specify custom evaluation routines by registering a factory function
    with GetLossClosure. These factories should be registered using the type signature

        `Type[MarginalLogLikeLihood], Type[Likelihood], Type[Model], Type[DataLoader]`.

    The final argument, Type[DataLoader], is optional. Evaluation routines that obtain
    training data from, e.g., `mll.model` should register this argument as `type(None)`.

    Args:
        mll: A MarginalLogLikelihood instance whose negative defines the loss.
        data_loader: An optional DataLoader instance for cases where training
            data is passed in rather than obtained from `mll.model`.

    Returns:
        A closure that takes zero positional arguments and returns the negated
        value of `mll`.
    """
    return GetLossClosure(
        mll, type(mll.likelihood), type(mll.model), data_loader, **kwargs
    )


def get_loss_closure_with_grads(
    mll: MarginalLogLikelihood,
    parameters: dict[str, Tensor],
    data_loader: DataLoader | None = None,
    **kwargs: Any,
) -> ForwardBackwardClosure:
    """
    Add a backward pass to a loss closure obtained by calling
    `get_loss_closure`, wrapping it in a ``ForwardBackwardClosure``.

    For further details, see `get_loss_closure`.

    Args:
        mll: A MarginalLogLikelihood instance whose negative defines the loss.
        parameters: A dictionary of tensors whose `grad` fields are to be returned.
        data_loader: An optional DataLoader instance for cases where training
            data is passed in rather than obtained from `mll.model`.
        kwargs: Keyword arguments passed to `get_loss_closure`.

    Returns:
        A closure that takes zero positional arguments and returns the reduced and
        negated value of `mll` along with the gradients of `parameters`.
    """
    loss_closure = get_loss_closure(mll, data_loader=data_loader, **kwargs)
    return ForwardBackwardClosure(forward=loss_closure, parameters=parameters)


@GetLossClosure.register(MarginalLogLikelihood, object, object, DataLoader)
def _get_loss_closure_fallback_external(
    mll: MarginalLogLikelihood,
    _likelihood_type: object,
    _model_type: object,
    data_loader: DataLoader,
    **ignore: Any,
) -> Callable[[], Tensor]:
    r"""Fallback loss closure with externally provided data."""
    batch_generator = chain.from_iterable(iter(data_loader) for _ in repeat(None))

    def closure(**kwargs: Any) -> Tensor:
        batch = next(batch_generator)
        if not isinstance(batch, Sequence):
            raise TypeError(
                "Expected `data_loader` to generate a batch of tensors, "
                f"but found {type(batch)}."
            )

        num_inputs = len(mll.model.train_inputs)
        model_output = mll.model(*batch[:num_inputs])
        log_likelihood = mll(model_output, *batch[num_inputs:], **kwargs)
        return -log_likelihood

    return closure


@GetLossClosure.register(MarginalLogLikelihood, object, object, NoneType)
def _get_loss_closure_fallback_internal(
    mll: MarginalLogLikelihood, _: object, __: object, ___: None, **ignore: Any
) -> Callable[[], Tensor]:
    r"""Fallback loss closure with internally managed data."""

    def closure(**kwargs: Any) -> Tensor:
        model_output = mll.model(*mll.model.train_inputs)
        log_likelihood = mll(model_output, mll.model.train_targets, **kwargs)
        return -log_likelihood

    return closure


@GetLossClosure.register(ExactMarginalLogLikelihood, object, object, NoneType)
def _get_loss_closure_exact_internal(
    mll: ExactMarginalLogLikelihood, _: object, __: object, ___: None, **ignore: Any
) -> Callable[[], Tensor]:
    r"""ExactMarginalLogLikelihood loss closure with internally managed data."""

    def closure(**kwargs: Any) -> Tensor:
        model = mll.model
        # The inputs will get transformed in forward here.
        model_output = model(*model.train_inputs)
        log_likelihood = mll(
            model_output,
            model.train_targets,
            # During model training, the model inputs get transformed in the forward
            # pass. The train_inputs property is not transformed yet, so we need to
            # transform it before passing it to the likelihood for consistency.
            *(model.transform_inputs(X=t_in) for t_in in model.train_inputs),
            **kwargs,
        )
        return -log_likelihood

    return closure


@GetLossClosure.register(SumMarginalLogLikelihood, object, object, NoneType)
def _get_loss_closure_sum_internal(
    mll: SumMarginalLogLikelihood, _: object, __: object, ___: None, **ignore: Any
) -> Callable[[], Tensor]:
    r"""SumMarginalLogLikelihood loss closure with internally managed data."""

    def closure(**kwargs: Any) -> Tensor:
        model = mll.model
        # The inputs will get transformed in forward here.
        model_output = model(*model.train_inputs)
        log_likelihood = mll(
            model_output,
            model.train_targets,
            # During model training, the model inputs get transformed in the forward
            # pass. The train_inputs property is not transformed yet, so we need to
            # transform it before passing it to the likelihood for consistency.
            *(
                (model.transform_inputs(X=t_in) for t_in in sub_t_in)
                for sub_t_in in model.train_inputs
            ),
            **kwargs,
        )
        return -log_likelihood

    return closure
