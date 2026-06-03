#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
import torch
from botorch.utils.jax_utils import jax_to_torch, torch_to_jax
from botorch.utils.testing import BotorchTestCase


class TestJaxUtils(BotorchTestCase):
    def test_conversions(self) -> None:
        with self.subTest("torch_to_jax"):
            t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)
            result = torch_to_jax(t)
            self.assertEqual(result.shape, (2, 2))
            expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)
            self.assertAllClose(
                jax_to_torch(result, device=t.device, dtype=t.dtype), expected
            )

        with self.subTest("torch_to_jax_with_grad"):
            t = torch.tensor([1.0, 2.0], requires_grad=True)
            result = torch_to_jax(t)
            self.assertEqual(result.shape, (1, 2) if result.ndim == 2 else (2,))
            expected = torch.tensor([1.0, 2.0])
            self.assertAllClose(
                jax_to_torch(result, device=t.device, dtype=t.dtype), expected
            )

        with self.subTest("jax_to_torch_double"):
            a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            result = jax_to_torch(a, device=torch.device("cpu"), dtype=torch.double)
            expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)
            self.assertAllClose(result, expected)
            self.assertEqual(result.dtype, torch.double)
            self.assertEqual(result.device.type, "cpu")

        with self.subTest("jax_to_torch_float32"):
            a = jnp.array([1.0, 2.0, 3.0])
            result = jax_to_torch(a, device=torch.device("cpu"), dtype=torch.float32)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.shape, (3,))

        with self.subTest("roundtrip"):
            original = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], dtype=torch.double, device=self.device
            )
            jax_array = torch_to_jax(original)
            recovered = jax_to_torch(
                jax_array, device=torch.device("cpu"), dtype=torch.double
            )
            self.assertAllClose(original.cpu(), recovered)
