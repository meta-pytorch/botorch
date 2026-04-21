#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.models.hierarchical.utils import get_blocks_with_paths
from botorch.utils.testing import BotorchTestCase


class TestHierarchicalUtils(BotorchTestCase):
    test_cases = []

    # ```
    # ROOT
    # ├── C0, C1
    # ├── P2
    # │   ├── (0) EMPTY
    # │   └── (1) C4, C5
    # └── P3
    #     ├── (0) EMPTY
    #     └── (1) C6, C7
    # ```
    # The features are ordered in the vector as `(C0, C1, P2, P3, C4, C5, C6, C7)`.
    test_cases.append(
        {
            "dim": 8,
            "hierarchical_dependencies": {
                # C0 does not have any dependents. It's okay to include it as a key
                # with an empty dict, or exclude it altogether.
                2: {0: [], 1: [4, 5]},  # dependents of P2
                3: {0: [], 1: [6, 7]},  # dependents of P3
            },
            "separate_hierarchical_features": True,
            "blocks": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "paths": [
                [],
                [],
                [(2, 1)],  # The path (P2 == 1) makes C4 and C5 active.
                [(3, 1)],  # The path (P3 == 1) makes C6 and C7 active.
            ],
        }
    )

    # The ultimate test.
    # ```
    # ROOT
    # ├── C0
    # ├── P1
    # │   ├── (0) C3, C4
    # │   └── (1) P5
    # │           ├── (0) C7
    # │           ├── (1) C8
    # │           └── (2) C9
    # └── P2
    #     ├── (0) EMPTY
    #     └── (1) C6
    # ```
    # The features are ordered as `(C0, P1, P2, C3, C4, P5, C6, C7, C8, C9)`.
    test_cases.append(
        {
            "dim": 10,
            "hierarchical_dependencies": {
                1: {0: [3, 4], 1: [5]},  # dependents of P1
                2: {0: [], 1: [6]},  # dependents of P2
                5: {0: [7], 1: [8], 2: [9]},  # dependents of P5
                6: {},  # C6 does not have any dependents.
            },
            "separate_hierarchical_features": True,
            "blocks": [[0], [1, 2], [3, 4], [5], [6], [7], [8], [9]],
            "paths": [
                [],  # C0 is always active.
                [],  # So do P1 and P2.
                [(1, 0)],  # The path (P1 == 0) makes C3 and C4 active.
                [(1, 1)],  # The path (P1 == 1) makes P5 active.
                [(2, 1)],  # The path (P2 == 1) makes C6 active.
                [(1, 1), (5, 0)],  # The path (P1 == 1, P5 == 0) makes C7 active.
                [(1, 1), (5, 1)],  # The path (P1 == 1, P5 == 1) makes C8 active.
                [(1, 1), (5, 2)],  # The path (P1 == 1, P5 == 2) makes C9 active.
            ],
        }
    )

    # This is obtained from the Jenatton test function in
    # `ax.benchmark.problems.synthetic.hss.jenatton`.
    test_cases.append(
        {
            "dim": 9,
            "hierarchical_dependencies": {
                0: {0: [1, 7], 1: [2, 8]},
                1: {0: [3], 1: [4]},
                2: {0: [5], 1: [6]},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
            },
            "separate_hierarchical_features": False,
            "blocks": [[0], [1, 7], [2, 8], [3], [4], [5], [6]],
            "paths": [
                [],
                [(0, 0)],
                [(0, 1)],
                [(0, 0), (1, 0)],
                [(0, 0), (1, 1)],
                [(0, 1), (2, 0)],
                [(0, 1), (2, 1)],
            ],
        }
    )

    def test_blocks_and_paths(self):
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(case=i, dim=test_case["dim"]):
                blocks, paths = get_blocks_with_paths(
                    dim=test_case["dim"],
                    hierarchical_dependencies=test_case["hierarchical_dependencies"],
                    separate_hierarchical_features=test_case[
                        "separate_hierarchical_features"
                    ],
                )

                # There should be no empty block.
                self.assertTrue(all(len(block) > 0 for block in blocks))

                # The returned `blocks` has been sorted in ascending order.
                # So it is safe to directly assert equality. No permutation needed.
                self.assertEqual(blocks, test_case["blocks"])
                self.assertEqual(paths, test_case["paths"])

    def test_duplicate_child_indices_rejected(self):
        """A child index appearing under two different parents is a malformed
        hierarchy and should raise."""
        # Index 3 is listed as a child of both parent 0 and parent 1.
        hierarchical_dependencies = {
            0: {0: [3], 1: [4]},
            1: {0: [3], 1: [5]},
        }
        with self.assertRaisesRegex(ValueError, "contains duplicate indices"):
            get_blocks_with_paths(
                dim=6, hierarchical_dependencies=hierarchical_dependencies
            )
