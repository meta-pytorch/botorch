#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Core methods for building closures in torch and interfacing with numpy."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from botorch.optim.utils import _handle_numerical_errors
from botorch.optim.utils.numpy_utils import as_ndarray
from botorch.utils.context_managers import zero_grad_ctx
from numpy import float64 as np_float64, zeros as np_zeros
from torch import Tensor

FILL_VALUE = 0.0


class ForwardBackwardClosure:
    r"""Wrapper for fused forward and backward closures."""

    def __init__(
        self, forward: Callable[[], Tensor], parameters: dict[str, Tensor]
    ) -> None:
        r"""Initializes a ForwardBackwardClosure instance.

        Args:
            forward: Callable that returns a tensor.
            parameters: A dictionary of tensors whose `grad` fields are to be returned.
        """
        self.forward = forward
        self.parameters = parameters

    def __call__(self, **kwargs: Any) -> tuple[Tensor, tuple[Tensor | None, ...]]:
        with zero_grad_ctx(parameters=self.parameters):
            value = self.forward(**kwargs).sum()
            value.backward()

            grads = tuple(param.grad for param in self.parameters.values())
            return value, grads


class NdarrayOptimizationClosure:
    r"""Adds stateful behavior and a numpy.ndarray-typed API to a closure with an
    expected return type Tuple[Tensor, Union[Tensor, Sequence[Optional[Tensor]]]].

    NaN values will be replaced with 0.0 in the returned ndarray."""

    def __init__(
        self,
        closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]],
        parameters: dict[str, Tensor],
    ) -> None:
        r"""Initializes a NdarrayOptimizationClosure instance.

        Args:
            closure: A ForwardBackwardClosure instance.
            parameters: A dictionary of tensors representing the closure's state.
                Expected to correspond with the first `len(parameters)` optional
                gradient tensors returned by `closure`.
        """

        self.closure = closure
        self.parameters = parameters
        self._gradient_ndarray: npt.NDArray | None = None

    def __call__(
        self, state: npt.NDArray | None = None, **kwargs: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if state is not None:
            self.state = state

        try:
            value_tensor, grad_tensors = self.closure(**kwargs)
            value = as_ndarray(values=value_tensor, dtype=np_float64)
            grads = self._get_gradient_ndarray()
            index = 0
            for param, grad in zip(self.parameters.values(), grad_tensors):
                size = param.numel()
                if grad is not None:
                    grads[index : index + size] = as_ndarray(
                        values=grad.view(-1), dtype=np_float64
                    )
                index += size
        except RuntimeError as e:
            value, grads = _handle_numerical_errors(e, x=self.state, dtype=np_float64)

        return value, grads

    @property
    def state(self) -> npt.NDArray:
        if len(self.parameters) == 0:
            raise RuntimeError("No parameters to get state from.")

        size = sum(tnsr.numel() for tnsr in self.parameters.values())
        tnsr = next(iter(self.parameters.values()))
        dtype = np_float64

        out = np.empty([size], dtype=dtype)

        index = 0
        for tnsr in self.parameters.values():
            size = tnsr.numel()
            out[index : index + size] = as_ndarray(tnsr.view(-1))
            index += size

        return out

    @state.setter
    def state(self, state: npt.NDArray) -> None:
        with torch.no_grad():
            index = 0
            for tnsr in self.parameters.values():
                size = tnsr.numel()
                vals = state[index : index + size] if tnsr.ndim else state[index]
                tnsr.copy_(
                    torch.as_tensor(vals, device=tnsr.device, dtype=tnsr.dtype).view(
                        tnsr.shape
                    )
                )
                index += size

    def _get_gradient_ndarray(self) -> npt.NDArray:
        if self._gradient_ndarray is not None:
            self._gradient_ndarray.fill(FILL_VALUE)
            return self._gradient_ndarray

        size = sum(param.numel() for param in self.parameters.values())
        self._gradient_ndarray = np_zeros(size, dtype=np_float64)
        return self._gradient_ndarray
