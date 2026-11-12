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
            parameters: A dictionary of tensors whose ``grad`` fields are to
                be returned.
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
                Expected to correspond with the first ``len(parameters)`` optional
                gradient tensors returned by ``closure``.
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


class BatchedNDarrayOptimizationClosure:
    r"""Wraps a forward closure and batched parameters for use with
    ``fmin_l_bfgs_b_batched``.

    Unlike ``NdarrayOptimizationClosure`` which flattens all parameters into a
    single 1D vector, this class manages parameters as a 2D array of shape
    ``(batch_size, per_element_size)`` where each row corresponds to one batch
    element's independent parameter vector.

    This enables independent optimization of each batch element (e.g., each
    output of a ``BatchedMultiOutputGPyTorchModel``) using batched L-BFGS-B.

    If the batched evaluation fails with a numerical error (e.g., a ``NanError``
    caused by the NaN covariance of a single batch element), the requested batch
    elements are evaluated one at a time, with the other elements set to values
    at which they were evaluated successfully. Only the elements that fail on
    their own get NaN values and gradients, as in ``NdarrayOptimizationClosure``.
    """

    def __init__(
        self,
        forward: Callable[[], Tensor],
        parameters: dict[str, Tensor],
        batch_shape: torch.Size,
    ) -> None:
        r"""Initializes a BatchedNDarrayOptimizationClosure instance.

        Args:
            forward: Callable that returns a tensor of shape ``batch_shape``
                (per-batch-element loss values, e.g., negated per-output MLL).
            parameters: A dictionary of parameter tensors, each with shape
                ``(*batch_shape, *trailing_shape)``.
            batch_shape: The batch shape shared by all parameters (typically
                ``model._aug_batch_shape``).
        """
        self.forward = forward
        self.parameters = parameters
        self.batch_shape = batch_shape
        self.batch_size = max(int(torch.Size(batch_shape).numel()), 1)
        n_batch_dims = len(batch_shape)

        self._trailing_sizes: dict[str, int] = {}
        self._per_element_size = 0
        for name, param in parameters.items():
            trailing = param.shape[n_batch_dims:]
            trailing_numel = max(int(torch.Size(trailing).numel()), 1)
            self._trailing_sizes[name] = trailing_numel
            self._per_element_size += trailing_numel

        # The last state at which all batch elements were evaluated without a
        # numerical error. Used to evaluate the elements one at a time.
        self._finite_state: npt.NDArray | None = None

    @property
    def state(self) -> npt.NDArray:
        """Returns the current parameter state as a 2D ndarray of shape
        ``(batch_size, per_element_size)``."""
        out = np.empty((self.batch_size, self._per_element_size), dtype=np_float64)
        index = 0
        for name, param in self.parameters.items():
            size = self._trailing_sizes[name]
            out[:, index : index + size] = as_ndarray(
                param.detach().reshape(self.batch_size, size), dtype=np_float64
            )
            index += size
        return out

    @state.setter
    def state(self, state: npt.NDArray) -> None:
        """Sets parameter values from a 2D ndarray of shape
        ``(batch_size, per_element_size)``."""
        with torch.no_grad():
            index = 0
            for name, param in self.parameters.items():
                size = self._trailing_sizes[name]
                vals = state[:, index : index + size]
                param.copy_(
                    torch.as_tensor(
                        vals, device=param.device, dtype=param.dtype
                    ).reshape(param.shape)
                )
                index += size

    def __call__(
        self,
        state: npt.NDArray | None = None,
        batch_indices: npt.NDArray | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Evaluate the closure and return per-batch values and gradients.

        Args:
            state: Optional 2D ndarray to set as the current state before
                evaluation. Shape ``(active_batch_size, per_element_size)``
                if ``batch_indices`` is provided, else
                ``(batch_size, per_element_size)``.
            batch_indices: Optional 1D ndarray of indices into the original
                batch, indicating which elements are being evaluated.
                Used with ``fmin_l_bfgs_b_batched(pass_batch_indices=True)``.
            **kwargs: Keyword arguments passed to ``self.forward``.

        Returns:
            A tuple ``(values, grads)`` where ``values`` has shape
            ``(active_batch_size,)`` and ``grads`` has shape
            ``(active_batch_size, per_element_size)``. Elements that fail with
            a numerical error get NaN values and gradients. After such a
            failure, the parameters of the elements not in ``batch_indices``
            are set to the last values (if any) at which they were evaluated
            successfully.
        """
        if state is None:
            full_state = self.state
        else:
            if batch_indices is None:
                full_state = np.array(state, dtype=np_float64)
            else:
                # Update only active batch elements
                full_state = self.state
                full_state[batch_indices] = state
            self.state = full_state

        try:
            values, grads = self._evaluate(**kwargs)
        except RuntimeError as e:
            # Re-raises errors that are not numerical, e.g. ``NotPSDError``.
            _handle_numerical_errors(e, x=full_state, dtype=np_float64)
            indices = (
                np.arange(self.batch_size) if batch_indices is None else batch_indices
            )
            values, grads = self._evaluate_elementwise(
                full_state=full_state, indices=indices, **kwargs
            )
        else:
            self._finite_state = full_state

        if batch_indices is not None:
            return values[batch_indices], grads[batch_indices]
        return values, grads

    def _evaluate(self, **kwargs: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """Evaluate all batch elements at the current state."""
        with zero_grad_ctx(parameters=self.parameters):
            per_batch_values = self.forward(**kwargs)
            scalar = per_batch_values.sum()
            scalar.backward()

            values = as_ndarray(
                per_batch_values.detach().reshape(self.batch_size),
                dtype=np_float64,
            )

            grads = np.zeros(
                (self.batch_size, self._per_element_size), dtype=np_float64
            )
            index = 0
            for name, param in self.parameters.items():
                size = self._trailing_sizes[name]
                if param.grad is not None:
                    grads[:, index : index + size] = as_ndarray(
                        param.grad.reshape(self.batch_size, size),
                        dtype=np_float64,
                    )
                index += size
        return values, grads

    def _evaluate_elementwise(
        self, full_state: npt.NDArray, indices: npt.NDArray, **kwargs: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Evaluate the batch elements in ``indices`` one at a time.

        Batch element ``i`` is evaluated at ``full_state[i]``, with the other
        elements set to values at which they were evaluated successfully (or,
        before the first successful evaluation, to ``full_state[i]``). Since the
        batch elements are independent, this gives the values and gradients of
        element ``i`` even if other elements fail at ``full_state``.
        """
        values = np.full(self.batch_size, np.nan, dtype=np_float64)
        grads = np.full(
            (self.batch_size, self._per_element_size), np.nan, dtype=np_float64
        )
        finite_state = self._finite_state
        for i in indices:
            if finite_state is None:
                probe = np.repeat(full_state[i : i + 1], self.batch_size, axis=0)
            else:
                probe = finite_state.copy()
                probe[i] = full_state[i]
            self.state = probe
            try:
                probe_values, probe_grads = self._evaluate(**kwargs)
            except RuntimeError as e:
                values[i], grads[i] = _handle_numerical_errors(
                    e, x=full_state[i], dtype=np_float64
                )
            else:
                values[i], grads[i] = probe_values[i], probe_grads[i]
                finite_state = probe
        self._finite_state = finite_state

        # Restore the evaluated elements. Set the others to finite values, so that
        # an element that stopped at a failing point does not make the following
        # batched evaluations fail.
        state = full_state if finite_state is None else finite_state.copy()
        state[indices] = full_state[indices]
        self.state = state
        return values, grads
