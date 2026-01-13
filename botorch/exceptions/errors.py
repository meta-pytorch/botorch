#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Botorch Errors.
"""

from typing import Any

import numpy.typing as npt


class BotorchError(Exception):
    r"""Base botorch exception."""

    pass


class CandidateGenerationError(BotorchError):
    r"""Exception raised during generating candidates."""

    pass


class DeprecationError(BotorchError):
    r"""Exception raised due to deprecations"""

    pass


class InputDataError(BotorchError):
    r"""Exception raised when input data does not comply with conventions."""

    pass


class UnsupportedError(BotorchError):
    r"""Currently unsupported feature."""

    pass


class BotorchTensorDimensionError(BotorchError):
    r"""Exception raised when a tensor violates a botorch convention."""

    pass


class ModelFittingError(Exception):
    r"""Exception raised when attempts to fit a model terminate unsuccessfully."""

    pass


class OptimizationTimeoutError(BotorchError):
    r"""Exception raised when optimization times out."""

    def __init__(
        self, /, *args: Any, current_x: npt.NDArray, runtime: float, **kwargs: Any
    ) -> None:
        r"""
        Args:
            *args: Standard args to ``BotorchError``.
            current_x: A numpy array representing the current iterate.
            runtime: The total runtime in seconds after which the optimization
                timed out.
            **kwargs: Standard kwargs to ``BotorchError``.
        """
        super().__init__(*args, **kwargs)
        self.current_x = current_x
        self.runtime = runtime


class OptimizationGradientError(BotorchError, RuntimeError):
    r"""Exception raised when gradient array ``gradf`` contains NaNs."""

    def __init__(self, /, *args: Any, current_x: npt.NDArray, **kwargs: Any) -> None:
        r"""
        Args:
            *args: Standard args to ``BotorchError``.
            current_x: A numpy array representing the current iterate.
            **kwargs: Standard kwargs to ``BotorchError``.
        """
        super().__init__(*args, **kwargs)
        self.current_x = current_x


class InfeasibilityError(BotorchError, ValueError):
    r"""Exception raised when infeasibility occurs."""

    pass
