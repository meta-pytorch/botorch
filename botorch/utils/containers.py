#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Representations for different kinds of data."""

from __future__ import annotations

import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

import torch

from torch import device as Device, dtype as Dtype, LongTensor, Size, Tensor


class BotorchContainer(ABC):
    r"""Abstract base class for BoTorch's data containers.

    A BotorchContainer represents a tensor, which should be the sole object
    returned by its `__call__` method. Said tensor is expected to consist of
    one or more "events" (e.g. data points or feature vectors), whose shape is
    given by the required `event_shape` field.

    Notice: Once version 3.10 becomes standard, this class should
    be reworked to take advantage of dataclasses' `kw_only` flag.
    """

    event_shape: Size

    def __post_init__(self, validate_init: bool = True) -> None:
        if validate_init:
            self._validate()

    @abstractmethod
    def __call__(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> Device:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> Dtype:
        raise NotImplementedError

    def _validate(self) -> None:
        for field in fields(self):
            if field.name == "event_shape":
                return
        raise AttributeError("Missing required field `event_shape`.")


@dataclass(eq=False)
class DenseContainer(BotorchContainer):
    r"""Basic representation of data stored as a dense Tensor."""

    values: Tensor
    event_shape: Size

    def __call__(self) -> Tensor:
        """Returns a dense tensor representation of the container's contents."""
        return self.values

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.shape == other.shape
            and self.values.equal(other.values)
        )

    @property
    def shape(self) -> Size:
        return self.values.shape

    @property
    def device(self) -> Device:
        return self.values.device

    @property
    def dtype(self) -> Dtype:
        return self.values.dtype

    def _validate(self) -> None:
        super()._validate()
        for a, b in zip(reversed(self.event_shape), reversed(self.values.shape)):
            if a != b:
                raise ValueError(
                    f"Shape of `values` {self.values.shape} incompatible with "
                    f"`event shape` {self.event_shape}."
                )

    def clone(self) -> DenseContainer:
        return dataclasses.replace(self)


@dataclass(eq=False)
class SliceContainer(BotorchContainer):
    r"""Represent data points formed by concatenating (n-1)-dimensional slices
    taken from the leading dimension of an n-dimensional source tensor."""

    values: Tensor
    indices: LongTensor
    event_shape: Size

    def __call__(self) -> Tensor:
        flat = self.values.index_select(dim=0, index=self.indices.view(-1))
        return flat.view(*self.indices.shape[:-1], -1, *self.values.shape[2:])

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and self.values.equal(other.values)
            and self.indices.equal(other.indices)
        )

    @property
    def shape(self) -> Size:
        return self.indices.shape[:-1] + self.event_shape

    @property
    def device(self) -> Device:
        return self.values.device

    @property
    def dtype(self) -> Dtype:
        return self.values.dtype

    def _validate(self) -> None:
        super()._validate()
        values = self.values
        indices = self.indices
        assert indices.ndim > 1
        assert (-1 < indices.min()) & (indices.max() < len(values))

        event_shape = self.event_shape
        _event_shape = (indices.shape[-1] * values.shape[1],) + values.shape[2:]
        if event_shape != _event_shape:
            raise ValueError(
                f"Shapes of `values` {values.shape} and `indices` "
                f"{indices.shape} incompatible with `event_shape` {event_shape}."
            )

    def clone(self) -> SliceContainer:
        return type(self)(
            values=self.values.clone(),
            indices=self.indices.clone(),
            event_shape=torch.Size(self.event_shape),
        )
