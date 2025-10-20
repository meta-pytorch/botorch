#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from botorch.optim.closures.core import as_ndarray
from botorch.optim.utils import get_bounds_as_ndarray
from botorch.optim.utils.numpy_utils import torch_to_numpy_dtype_dict
from botorch.utils.testing import BotorchTestCase
from torch.nn import Parameter


class TestNumpyUtils(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.parameters = {"foo": torch.rand(2), "bar": Parameter(torch.rand(3))}

    def test_as_ndarray(self):
        base = np.random.randn(3)
        tnsr = torch.from_numpy(base)

        # Test inplace conversion
        result = as_ndarray(tnsr)
        self.assertTrue(np.shares_memory(base, result))

        # Test conversion with memory allocation
        result = as_ndarray(tnsr, inplace=False)
        self.assertTrue(np.allclose(base, result))
        self.assertFalse(np.shares_memory(base, result))

        result = as_ndarray(tnsr, dtype=np.float32)
        self.assertTrue(np.allclose(base, result))
        self.assertFalse(np.shares_memory(base, result))
        self.assertEqual(result.dtype, np.float32)

        # Test that `clone` does not get called on non-CPU tensors
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.device.return_value = "foo"
        mock_tensor.clone.return_value = mock_tensor

        as_ndarray(mock_tensor)
        mock_tensor.cpu.assert_called_once()
        mock_tensor.clone.assert_not_called()
        mock_tensor.numpy.assert_called_once()

    def test_as_ndarray_dtypes(self) -> None:
        for torch_dtype, np_dtype in torch_to_numpy_dtype_dict.items():
            tens = torch.tensor(0, dtype=torch_dtype, device="cpu")
            self.assertEqual(torch_dtype, tens.dtype)
            self.assertEqual(tens.numpy().dtype, np_dtype)
            self.assertEqual(as_ndarray(tens, np_dtype).dtype, np_dtype)

    def test_get_bounds_as_ndarray(self):
        params = {"a": torch.rand(1), "b": torch.rand(1), "c": torch.rand(2)}
        bounds = {"a": (None, 1), "c": (0, None)}

        test = np.full((4, 2), (-float("inf"), float("inf")))
        test[0, 1] = 1
        test[2, 0] = 0
        test[3, 0] = 0

        array = get_bounds_as_ndarray(parameters=params, bounds=bounds)
        self.assertTrue(np.array_equal(test, array))

        # Test with tensor bounds.
        bounds = {
            "a": (None, torch.tensor(1, device=self.device)),
            "c": (torch.tensor(0, device=self.device), None),
        }
        array = get_bounds_as_ndarray(parameters=params, bounds=bounds)
        self.assertTrue(np.array_equal(test, array))

        # Test with n-dim tensor bounds.
        bounds = {
            "a": (None, torch.tensor(1, device=self.device)),
            "c": (
                torch.tensor([0, 0], device=self.device),
                torch.tensor([1, 1], device=self.device),
            ),
        }
        test[2:, 1] = 1
        array = get_bounds_as_ndarray(parameters=params, bounds=bounds)
        self.assertTrue(np.array_equal(test, array))
