#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from botorch.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    CandidateGenerationError,
    DeprecationError,
    InfeasibilityError,
    InputDataError,
    OptimizationGradientError,
    OptimizationTimeoutError,
    UnsupportedError,
)
from botorch.utils.testing import BotorchTestCase


class TestBotorchExceptions(BotorchTestCase):
    def test_botorch_exception_hierarchy(self):
        self.assertIsInstance(BotorchError(), Exception)
        for ErrorClass in [
            CandidateGenerationError,
            DeprecationError,
            InputDataError,
            UnsupportedError,
            BotorchTensorDimensionError,
            InfeasibilityError,
        ]:
            self.assertIsInstance(ErrorClass(), BotorchError)

    def test_raise_botorch_exceptions(self):
        for ErrorClass in (
            BotorchError,
            BotorchTensorDimensionError,
            CandidateGenerationError,
            InputDataError,
            UnsupportedError,
            InfeasibilityError,
        ):
            with self.assertRaises(ErrorClass):
                raise ErrorClass("message")

    def test_OptimizationTimeoutError(self):
        error = OptimizationTimeoutError(
            "message", current_x=np.array([1.0]), runtime=0.123
        )
        self.assertEqual(error.runtime, 0.123)
        self.assertTrue(np.array_equal(error.current_x, np.array([1.0])))
        with self.assertRaises(OptimizationTimeoutError):
            raise error

    def test_OptimizationGradientError(self):
        error = OptimizationGradientError("message", current_x=np.array([1.0]))
        self.assertTrue(np.array_equal(error.current_x, np.array([1.0])))
        with self.assertRaisesRegex(OptimizationGradientError, "message"):
            raise error
