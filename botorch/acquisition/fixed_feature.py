#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AcquisitionFunctions to fix certain features for optimization.
This is useful e.g. for performing contextual optimization.
"""

from __future__ import annotations

from collections.abc import Sequence

from numbers import Number

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.nn import Module


def get_dtype_of_sequence(values: Sequence[Tensor | float]) -> torch.dtype:
    """
    Return torch.float32 if everything is single-precision and torch.float64
    otherwise.

    Numbers (non-tensors) are double-precision.
    """

    def _is_single(value: Tensor | float) -> bool:
        return isinstance(value, Tensor) and value.dtype == torch.float32

    all_single_precision = all(_is_single(value) for value in values)
    return torch.float32 if all_single_precision else torch.float64


def get_device_of_sequence(values: Sequence[Tensor | float]) -> torch.dtype:
    """
    CPU if everything is on the CPU; Cuda otherwise.

    Numbers (non-tensors) are considered to be on the CPU.
    """

    def _is_cuda(value: Tensor | float) -> bool:
        return hasattr(value, "device") and value.device == torch.device("cuda")

    any_cuda = any(_is_cuda(value) for value in values)
    return torch.device("cuda") if any_cuda else torch.device("cpu")


class FixedFeatureAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AcquisitionFunctions to fix a subset of features.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)  # d = 5
        >>> qEI = qExpectedImprovement(model, best_f=0.0)
        >>> columns = [2, 4]
        >>> values = X[..., columns]
        >>> qEI_FF = FixedFeatureAcquisitionFunction(qEI, 5, columns, values)
        >>> qei = qEI_FF(test_X)  # d' = 3
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        d: int,
        columns: list[int],
        values: Tensor | Sequence[Tensor | float],
    ) -> None:
        r"""Derived Acquisition Function by fixing a subset of input features.

        Args:
            acq_function: The base acquisition function, operating on input
                tensors `X_full` of feature dimension `d`.
            d: The feature dimension expected by `acq_function`.
            columns: `d_f < d` indices of columns in `X_full` that are to be
                fixed to the provided values.
            values: The values to which to fix the columns in `columns`. Either
                a full `batch_shape x q x d_f` tensor of values (if values are
                different for each of the `q` input points), or an array-like of
                values that is broadcastable to the input across `t`-batch and
                `q`-batch dimensions, e.g. a list of length `d_f` if values
                are the same across all `t` and `q`-batch dimensions, or a
                combination of `Tensor`s and numbers which can be broadcasted
                to form a tensor with trailing dimension size of `d_f`.
        """
        Module.__init__(self)
        self.acq_func = acq_function
        self.d = d

        if isinstance(values, Tensor):
            new_values = values.detach().clone()
        else:
            dtype = get_dtype_of_sequence(values)
            device = get_device_of_sequence(values)

            new_values = []
            for value in values:
                if isinstance(value, Number):
                    value = torch.tensor([value], dtype=dtype)
                else:
                    if value.ndim == 0:  # since we can't broadcast with zero-d tensors
                        value = value.unsqueeze(0)
                    value = value.detach().clone()

                new_values.append(value.to(dtype=dtype, device=device))

            # There are 3 cases for when `values` is a `Sequence`.
            # 1) `values` == list of floats as earlier.
            # 2) `values` == combination of floats and `Tensor`s.
            # 3) `values` == a list of `Tensor`s.
            # For 1), the below step creates a vector of length `len(values)`
            # For 2), the below step creates a `Tensor` of shape `batch_shape x q x d_f`
            # with the broadcasting functionality.
            # For 3), this is simply a concatenation, yielding a `Tensor` with the
            # same shape as in 2).
            # The key difference arises when `_construct_X_full` is invoked.
            # In 1), the expansion (`self.values.expand`) will expand the `Tensor` to
            # size `batch_shape x q x d_f`.
            # In 2) and 3), this expansion is a no-op because they are already of the
            # required size. However, 2) and 3) _cannot_ support varying `batch_shape`,
            # which means that all calls to `FixedFeatureAcquisitionFunction` have
            # to have the same size throughout when `values` contains a `Tensor`.
            # This is consistent with the scenario when a singular `Tensor` is passed
            # as the `values` argument.
            new_values = torch.cat(torch.broadcast_tensors(*new_values), dim=-1)

        self.register_buffer("values", new_values)
        # build selector for _construct_X_full
        self._selector = []
        idx_X, idx_f = 0, d - new_values.shape[-1]
        for i in range(self.d):
            if i in columns:
                self._selector.append(idx_f)
                idx_f += 1
            else:
                self._selector.append(idx_X)
                idx_X += 1

    def forward(self, X: Tensor):
        r"""Evaluate base acquisition function under the fixed features.

        Args:
            X: Input tensor of feature dimension `d' < d` such that `d' + d_f = d`.

        Returns:
            Base acquisition function evaluated on tensor `X_full` constructed
            by adding `values` in the appropriate places (see
            `_construct_X_full`).
        """
        X_full = self._construct_X_full(X)
        return self.acq_func(X_full)

    @property
    def X_pending(self):
        r"""Return the `X_pending` of the base acquisition function."""
        try:
            return self.acq_func.X_pending
        except (ValueError, AttributeError):
            raise ValueError(
                f"Base acquisition function {type(self.acq_func).__name__} "
                "does not have an `X_pending` attribute."
            )

    @X_pending.setter
    def X_pending(self, X_pending: Tensor | None):
        r"""Sets the `X_pending` of the base acquisition function."""
        if X_pending is not None:
            self.acq_func.X_pending = self._construct_X_full(X_pending)
        else:
            self.acq_func.X_pending = X_pending

    def _construct_X_full(self, X: Tensor) -> Tensor:
        r"""Constructs the full input for the base acquisition function.

        Args:
            X: Input tensor with shape `batch_shape x q x d'` such that
                `d' + d_f = d`.

        Returns:
            Tensor `X_full` of shape `batch_shape x q x d`, where
            `X_full[..., i] = values[..., i]` if `i in columns`,
            and `X_full[..., i] = X[..., j]`, with
            `j = i - sum_{l<=i} 1_{l in fixed_colunns}`.
        """
        d_prime, d_f = X.shape[-1], self.values.shape[-1]
        if d_prime + d_f != self.d:
            raise ValueError(
                f"Feature dimension d' ({d_prime}) of input must be "
                f"d - d_f ({self.d - d_f})."
            )
        # concatenate values to the end
        values = self.values.to(X).expand(*X.shape[:-1], d_f)
        X_perm = torch.cat([X, values], dim=-1)
        # now select the appropriate column order
        return X_perm[..., self._selector]
