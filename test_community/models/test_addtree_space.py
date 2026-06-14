# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tests for :class:`AddTreeSpace`."""

import pytest
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.addtree.space import (
    AddTreeSpace,
    Choice,
    Continuous,
    make_node,
    Node,
)


# ---------------------------------------------------------------------------
# Fixture specs
# ---------------------------------------------------------------------------


def _binary_spec() -> dict:
    """A 1-level tree: root with 2 children, each holding one Continuous."""
    return {
        "name": "root",
        "choices": [
            {
                "name": "L1",
                "options": {
                    "left": {
                        "name": "L1_left",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                    },
                    "right": {
                        "name": "L1_right",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                    },
                },
            }
        ],
    }


def _jenatton_small_spec() -> dict:
    """The Jenatton-small two-level binary tree."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAddTreeSpace(BotorchTestCase):
    def test_binary_dim_and_paths(self) -> None:
        space = AddTreeSpace.from_dict(_binary_spec())
        # 1 (root flag) + 1 (L1_left flag) + 1 (L1_left.v) + 1 (L1_right flag)
        # + 1 (L1_right.v) = 5
        self.assertEqual(space.dim, 5)
        self.assertEqual(space.num_paths, 2)
        self.assertEqual(space.path_ids, ("left", "right"))

    def test_jenatton_dim_and_paths(self) -> None:
        space = AddTreeSpace.from_dict(_jenatton_small_spec())
        # 7 nodes, each with 1 flag + (0 or 1) cont -> 1 + 6*(1+1) = 13.
        self.assertEqual(space.dim, 13)
        self.assertEqual(space.num_paths, 4)
        self.assertEqual(
            space.path_ids,
            ("left/left", "left/right", "right/left", "right/right"),
        )

    def test_bounds_shape_and_continuous_values(self) -> None:
        space = AddTreeSpace.from_dict(_binary_spec())
        self.assertEqual(tuple(space.bounds.shape), (2, space.dim))
        # Every cont_dim has bounds [0, 1].
        for d in space.cont_dims:
            self.assertAlmostEqual(space.bounds[0, d].item(), 0.0)
            self.assertAlmostEqual(space.bounds[1, d].item(), 1.0)

    def test_fixed_features_list_pins_off_path_to_sentinel(self) -> None:
        space = AddTreeSpace.from_dict(_binary_spec())
        # path "left": L1_right flag (slot 3) and L1_right.v (slot 4) pinned to -1.
        # path "right": L1_left flag (slot 1) and L1_left.v (slot 2) pinned to -1.
        ff_left = dict(space.fixed_features_list[0])
        ff_right = dict(space.fixed_features_list[1])
        # Root flag is always pinned to 0 (root local_id == 0).
        self.assertEqual(ff_left[0], 0.0)
        self.assertEqual(ff_right[0], 0.0)
        # On-path L1_left flag = 0 (first option), continuous slot is *not* fixed.
        self.assertEqual(ff_left[1], 0.0)
        self.assertNotIn(2, ff_left)  # L1_left.v is free to optimise
        # Off-path L1_right slots pinned to sentinel.
        self.assertEqual(ff_left[3], -1.0)
        self.assertEqual(ff_left[4], -1.0)
        # On-path right-side path: L1_right flag = 1 (second option).
        self.assertEqual(ff_right[3], 1.0)
        self.assertNotIn(4, ff_right)

    def test_yaml_load(self) -> None:
        pytest.importorskip("yaml")
        import os
        import tempfile

        import yaml

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "spec.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(_binary_spec(), f, sort_keys=False)
            space = AddTreeSpace.from_yaml(path)
            self.assertEqual(space.dim, 5)
            self.assertEqual(space.path_ids, ("left", "right"))

    def test_python_builder_equivalent_to_dict(self) -> None:
        builder = Node(
            name="root",
            choice=Choice(
                name="L1",
                options=(
                    (
                        "left",
                        make_node(
                            "L1_left",
                            continuous=[Continuous("v", 0.0, 1.0)],
                        ),
                    ),
                    (
                        "right",
                        make_node(
                            "L1_right",
                            continuous=[Continuous("v", 0.0, 1.0)],
                        ),
                    ),
                ),
            ),
        )
        s_py = AddTreeSpace(builder)
        s_d = AddTreeSpace.from_dict(_binary_spec())
        self.assertEqual(s_py.dim, s_d.dim)
        self.assertEqual(s_py.path_ids, s_d.path_ids)
        self.assertEqual(s_py.cat_dims, s_d.cat_dims)
        self.assertEqual(s_py.cont_dims, s_d.cont_dims)
        # fixed_features_list dicts are equal element-wise.
        for a, b in zip(s_py.fixed_features_list, s_d.fixed_features_list):
            self.assertEqual(a, b)

    def test_no_globals(self) -> None:
        # Building two specs in the same process must not interfere.
        s1 = AddTreeSpace.from_dict(_binary_spec())
        s2 = AddTreeSpace.from_dict(_jenatton_small_spec())
        self.assertEqual(s1.dim, 5)
        self.assertEqual(s2.dim, 13)

    def test_duplicate_node_name_rejected(self) -> None:
        bad = {
            "name": "root",
            "choices": [
                {
                    "name": "L1",
                    "options": {
                        "left": {"name": "dup"},
                        "right": {"name": "dup"},
                    },
                }
            ],
        }
        with self.assertRaises(ValueError):
            AddTreeSpace.from_dict(bad)

    def test_invalid_continuous_bounds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Continuous(name="v", lo=1.0, hi=0.0)

    def test_choice_must_have_options(self) -> None:
        with self.assertRaises(ValueError):
            Choice(name="empty", options=())

    def test_multi_choice_per_node_rejected(self) -> None:
        bad = {
            "name": "root",
            "choices": [
                {"name": "A", "options": {"x": {"name": "a"}}},
                {"name": "B", "options": {"y": {"name": "b"}}},
            ],
        }
        with self.assertRaises(ValueError):
            AddTreeSpace.from_dict(bad)

    def test_per_continuous_bounds(self) -> None:
        space = AddTreeSpace.from_dict(
            {
                "name": "root",
                "continuous": [
                    {"name": "x", "lo": -2.0, "hi": 5.0},
                    {"name": "y", "lo": 0.5, "hi": 0.7},
                ],
            }
        )
        # cont_dims = (1, 2) (slot 0 is the root flag).
        self.assertEqual(space.cont_dims, (1, 2))
        self.assertAlmostEqual(space.bounds[0, 1].item(), -2.0)
        self.assertAlmostEqual(space.bounds[1, 1].item(), 5.0)
        self.assertAlmostEqual(space.bounds[0, 2].item(), 0.5)
        self.assertAlmostEqual(space.bounds[1, 2].item(), 0.7)
