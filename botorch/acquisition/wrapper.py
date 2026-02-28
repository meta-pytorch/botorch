#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper classes around AcquisitionFunctions to modify inputs and outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.nn import Module


class AbstractAcquisitionFunctionWrapper(AcquisitionFunction, ABC):
    r"""Abstract acquisition wrapper."""

    def __init__(self, acq_function: AcquisitionFunction) -> None:
        r"""Initialize the acquisition function wrapper.

        Args:
            acq_function: The inner acquisition function to wrap.
        """
        Module.__init__(self)
        self.acq_func = acq_function

    @property
    def X_pending(self) -> Tensor | None:
        r"""Return the `X_pending` of the base acquisition function."""
        try:
            return self.acq_func.X_pending
        except (ValueError, AttributeError):
            raise ValueError(
                f"Base acquisition function {type(self.acq_func).__name__} "
                "does not have an `X_pending` attribute."
            )

    def set_X_pending(self, X_pending: Tensor | None) -> None:
        r"""Sets the `X_pending` of the base acquisition function."""
        self.acq_func.set_X_pending(X_pending)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the wrapped acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        pass  # pragma: no cover
