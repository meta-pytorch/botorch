#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, overload, Union

import torch
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.dispatcher import Dispatcher
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels.kernel import Kernel
from torch import LongTensor, Tensor
from torch.nn import Module, ModuleList

TInputTransform = Union[InputTransform, Callable[[Tensor], Tensor]]
TOutputTransform = Union[OutcomeTransform, Callable[[Tensor], Tensor]]
GetTrainInputs = Dispatcher("get_train_inputs")
GetTrainTargets = Dispatcher("get_train_targets")


class TransformedModuleMixin:
    r"""Mixin that wraps a module's __call__ method with optional transforms."""

    input_transform: TInputTransform | None
    output_transform: TOutputTransform | None

    def __call__(self, values: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        input_transform = getattr(self, "input_transform", None)
        if input_transform is not None:
            values = (
                input_transform.forward(values)
                if isinstance(input_transform, InputTransform)
                else input_transform(values)
            )

        output = super().__call__(values, *args, **kwargs)
        output_transform = getattr(self, "output_transform", None)
        if output_transform is None:
            return output

        return (
            output_transform.untransform(output)[0]
            if isinstance(output_transform, OutcomeTransform)
            else output_transform(output)
        )


class TensorTransform(ABC, Module):
    r"""Abstract base class for transforms that map tensor to tensor."""

    @abstractmethod
    def forward(self, values: Tensor, **kwargs: Any) -> Tensor:
        pass  # pragma: no cover


class ChainedTransform(TensorTransform):
    r"""A composition of TensorTransforms."""

    def __init__(self, *transforms: TensorTransform):
        r"""Initializes a ChainedTransform instance.

        Args:
            transforms: A set of transforms to be applied from right to left.
        """
        super().__init__()
        self.transforms = ModuleList(transforms)

    def forward(self, values: Tensor) -> Tensor:
        for transform in reversed(self.transforms):
            values = transform(values)
        return values


class SineCosineTransform(TensorTransform):
    r"""A transform that returns concatenated sine and cosine features."""

    def __init__(self, scale: Tensor | None = None):
        r"""Initializes a SineCosineTransform instance.

        Args:
            scale: An optional tensor used to rescale the module's outputs.
        """
        super().__init__()
        self.scale = scale

    def forward(self, values: Tensor) -> Tensor:
        sincos = torch.concat([values.sin(), values.cos()], dim=-1)
        return sincos if self.scale is None else self.scale * sincos


class InverseLengthscaleTransform(TensorTransform):
    r"""A transform that divides its inputs by a kernels lengthscales."""

    def __init__(self, kernel: Kernel):
        r"""Initializes an InverseLengthscaleTransform instance.

        Args:
            kernel: The kernel whose lengthscales are to be used.
        """
        if not kernel.has_lengthscale:
            raise RuntimeError(f"{type(kernel)} does not implement `lengthscale`.")

        super().__init__()
        self.kernel = kernel

    def forward(self, values: Tensor) -> Tensor:
        return self.kernel.lengthscale.reciprocal() * values


class OutputscaleTransform(TensorTransform):
    r"""A transform that multiplies its inputs by the square root of a
    kernel's outputscale."""

    def __init__(self, kernel: ScaleKernel):
        r"""Initializes an OutputscaleTransform instance.

        Args:
            kernel: A ScaleKernel whose ``outputscale`` is to be used.
        """
        super().__init__()
        self.kernel = kernel

    def forward(self, values: Tensor) -> Tensor:
        outputscale = (
            self.kernel.outputscale[..., None, None]
            if self.kernel.batch_shape
            else self.kernel.outputscale
        )
        return outputscale.sqrt() * values


class FeatureSelector(TensorTransform):
    r"""A transform that returns a subset of its input's features.
    along a given tensor dimension."""

    def __init__(self, indices: Iterable[int], dim: int | LongTensor = -1):
        r"""Initializes a FeatureSelector instance.

        Args:
            indices: A LongTensor of feature indices.
            dim: The dimensional along which to index features.
        """
        super().__init__()
        self.register_buffer("dim", dim if torch.is_tensor(dim) else torch.tensor(dim))
        self.register_buffer(
            "indices", indices if torch.is_tensor(indices) else torch.tensor(indices)
        )

    def forward(self, values: Tensor) -> Tensor:
        return values.index_select(dim=self.dim, index=self.indices)


class OutcomeUntransformer(TensorTransform):
    r"""Module acting as a bridge for ``OutcomeTransform.untransform``."""

    def __init__(
        self,
        transform: OutcomeTransform,
        num_outputs: int | LongTensor,
    ):
        r"""Initializes an OutcomeUntransformer instance.

        Args:
            transform: The wrapped OutcomeTransform instance.
            num_outputs: The number of outcome features that the
                OutcomeTransform transforms.
        """
        super().__init__()
        self.transform = transform
        self.register_buffer(
            "num_outputs",
            num_outputs if torch.is_tensor(num_outputs) else torch.tensor(num_outputs),
        )

    def forward(self, values: Tensor) -> Tensor:
        # OutcomeTransforms expect an explicit output dimension in the final position.
        if self.num_outputs == 1:  # BoTorch has suppressed the output dimension
            output_values, _ = self.transform.untransform(values.unsqueeze(-1))
            return output_values.squeeze(-1)

        # BoTorch has moved the output dimension inside as the final batch dimension.
        output_values, _ = self.transform.untransform(values.transpose(-2, -1))
        return output_values.transpose(-2, -1)


def get_input_transform(model: GPyTorchModel) -> InputTransform | None:
    r"""Returns a model's input_transform or None."""
    return getattr(model, "input_transform", None)


def get_output_transform(model: GPyTorchModel) -> OutcomeUntransformer | None:
    r"""Returns a wrapped version of a model's outcome_transform or None."""
    transform = getattr(model, "outcome_transform", None)
    if transform is None:
        return None

    return OutcomeUntransformer(transform=transform, num_outputs=model.num_outputs)


@overload
def get_train_inputs(model: Model, transformed: bool = False) -> tuple[Tensor, ...]:
    pass  # pragma: no cover


@overload
def get_train_inputs(model: ModelList, transformed: bool = False) -> list[...]:
    pass  # pragma: no cover


def get_train_inputs(model: Model, transformed: bool = False):
    return GetTrainInputs(model, transformed=transformed)


@GetTrainInputs.register(Model)
def _get_train_inputs_Model(model: Model, transformed: bool = False) -> tuple[Tensor]:
    if not transformed:
        original_train_input = getattr(model, "_original_train_inputs", None)
        if torch.is_tensor(original_train_input):
            return (original_train_input,)

    (X,) = model.train_inputs
    transform = get_input_transform(model)
    if transform is None:
        return (X,)

    if model.training:
        return (transform.forward(X) if transformed else X,)
    return (X if transformed else transform.untransform(X),)


@GetTrainInputs.register(SingleTaskVariationalGP)
def _get_train_inputs_SingleTaskVariationalGP(
    model: SingleTaskVariationalGP, transformed: bool = False
) -> tuple[Tensor]:
    (X,) = model.model.train_inputs
    if model.training != transformed:
        return (X,)

    transform = get_input_transform(model)
    if transform is None:
        return (X,)

    return (transform.forward(X) if model.training else transform.untransform(X),)


@GetTrainInputs.register(ModelList)
def _get_train_inputs_ModelList(
    model: ModelList, transformed: bool = False
) -> list[...]:
    return [get_train_inputs(m, transformed=transformed) for m in model.models]


@overload
def get_train_targets(model: Model, transformed: bool = False) -> Tensor:
    pass  # pragma: no cover


@overload
def get_train_targets(model: ModelList, transformed: bool = False) -> list[...]:
    pass  # pragma: no cover


def get_train_targets(model: Model, transformed: bool = False):
    return GetTrainTargets(model, transformed=transformed)


@GetTrainTargets.register(Model)
def _get_train_targets_Model(model: Model, transformed: bool = False) -> Tensor:
    Y = model.train_targets

    # Note: Avoid using ``get_output_transform`` here since it creates a Module
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

    # SingleTaskVariationalGP.__init__ doesn't bring the multioutput dimension inside
    return transform.untransform(Y)[0]


@GetTrainTargets.register(ModelList)
def _get_train_targets_ModelList(
    model: ModelList, transformed: bool = False
) -> list[...]:
    return [get_train_targets(m, transformed=transformed) for m in model.models]
