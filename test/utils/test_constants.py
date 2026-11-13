#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import patch

import torch
from botorch.utils import constants
from botorch.utils.testing import BotorchTestCase


class TestConstants(BotorchTestCase):
    def test_get_constants(self):
        tkwargs = {"device": self.device, "dtype": torch.float16}
        const = constants.get_constants(0.123, **tkwargs)
        self.assertEqual(const, 0.123)
        self.assertEqual(const.device.type, tkwargs["device"].type)
        self.assertEqual(const.dtype, tkwargs["dtype"])

        # ``get_constants`` caches its outputs, so ``const`` must be restored exactly
        # after the in-place modification below. Subtracting 1 again would not
        # restore it in half precision.
        const_value = const.clone()
        try:  # test in-place modification
            const.add_(1)
            const2 = constants.get_constants(0.123, **tkwargs)
            self.assertEqual(const2, 1.123)
        finally:
            const.copy_(const_value)
        self.assertEqual(constants.get_constants(0.123, **tkwargs), 0.123)

        # Test fetching of multiple constants
        const_tuple = constants.get_constants(values=(0, 1, 2), **tkwargs)
        self.assertIsInstance(const_tuple, tuple)
        self.assertEqual(len(const_tuple), 3)
        for i, const in enumerate(const_tuple):
            self.assertEqual(const, i)

    def test_get_constants_like(self):
        def mock_get_constants(values: torch.Tensor, **kwargs):
            return kwargs

        tkwargs = {"device": self.device, "dtype": torch.float16}
        with patch.object(constants, "get_constants", new=mock_get_constants):
            ref = torch.tensor([123], **tkwargs)
            other = constants.get_constants_like(0.123, ref=ref)
            self.assertEqual(other["device"].type, tkwargs["device"].type)
            self.assertEqual(other["dtype"], tkwargs["dtype"])
