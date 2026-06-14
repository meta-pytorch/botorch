# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tests for encode/decode round-trip in :mod:`botorch_community.models.addtree`."""

import numpy as np
import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.addtree.encoding import decode, encode, param_key
from botorch_community.models.addtree.space import AddTreeSpace


def _jenatton_small_spec() -> dict:
    leaf = lambda name: {  # noqa: E731
        "name": name,
        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
    }
    return {
        "name": "root",
        "choices": [
            {
                "name": "L1",
                "options": {
                    "left": {
                        "name": "L1_left",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_ll"),
                                    "right": leaf("L2_lr"),
                                },
                            }
                        ],
                    },
                    "right": {
                        "name": "L1_right",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_rl"),
                                    "right": leaf("L2_rr"),
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }


class TestEncodeDecode(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = AddTreeSpace.from_dict(_jenatton_small_spec())

    def test_param_key(self) -> None:
        self.assertEqual(param_key("L1_left", "v"), "L1_left.v")

    def test_encode_left_left_path(self) -> None:
        p = {"L1": "left", "L1_left.v": 0.3, "L2": "left", "L2_ll.v": 0.8}
        x = encode(self.space, p)
        self.assertEqual(x.shape, (self.space.dim,))
        # Off-path slots are sentinels.
        # For path left/left: L1_right slots (3, 4) and L2_lr slots (7, 8) and
        # L2_rl slots (9, 10) and L2_rr slots (11, 12) are sentinels.
        for off in (3, 4, 7, 8, 9, 10, 11, 12):
            self.assertAlmostEqual(x[off].item(), -1.0)
        # On-path values.
        self.assertAlmostEqual(x[2].item(), 0.3)  # L1_left.v
        self.assertAlmostEqual(x[6].item(), 0.8)  # L2_ll.v

    def test_round_trip_random(self) -> None:
        rng = np.random.default_rng(42)
        choices = ["left", "right"]
        for _ in range(50):
            L1 = rng.choice(choices)
            L2 = rng.choice(choices)
            p = {
                "L1": L1,
                f"L1_{L1}.v": float(rng.random()),
                "L2": L2,
                f"L2_{L1[0]}{L2[0]}.v": float(rng.random()),
            }
            x = encode(self.space, p)
            back = decode(self.space, x)
            self.assertEqual(back, p)

    def test_decode_2d_batch_of_one(self) -> None:
        p = {"L1": "right", "L1_right.v": 0.5, "L2": "right", "L2_rr.v": 0.1}
        x = encode(self.space, p).unsqueeze(0)  # (1, dim)
        self.assertEqual(decode(self.space, x), p)

    def test_encode_missing_choice(self) -> None:
        with self.assertRaises(KeyError):
            encode(self.space, {"L1_left.v": 0.5})  # missing L1

    def test_encode_unknown_option(self) -> None:
        with self.assertRaises(ValueError):
            encode(self.space, {"L1": "middle"})  # not a valid option

    def test_encode_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            encode(
                self.space,
                {"L1": "left", "L1_left.v": 99.0, "L2": "left", "L2_ll.v": 0.5},
            )

    def test_encode_dtype(self) -> None:
        p = {"L1": "left", "L1_left.v": 0.3, "L2": "left", "L2_ll.v": 0.8}
        for dtype in (torch.float32, torch.float64):
            x = encode(self.space, p, dtype=dtype)
            self.assertEqual(x.dtype, dtype)

    def test_decode_wrong_dim(self) -> None:
        bad = torch.zeros(self.space.dim - 1)
        with self.assertRaises(ValueError):
            decode(self.space, bad)
