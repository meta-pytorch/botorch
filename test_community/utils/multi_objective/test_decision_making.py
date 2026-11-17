# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.utils.multi_objective.decision_making import (
    _pareto_tiled,
    knee_region,
    select_preferred,
)


class TestDecisionMaking(BotorchTestCase):
    def test_tiled_dominance_preserves_duplicates_and_batch_order(self) -> None:
        costs = torch.tensor(
            [[0.0, 1.0], [0.2, 0.4], [0.2, 0.4], [1.0, 0.0], [2.0, 2.0], [0.4, 0.2]],
            dtype=torch.double,
            device=self.device,
        )
        batch = torch.stack((costs, costs.flip(0))).unsqueeze(0)
        expected = torch.tensor(
            [
                [
                    [True, True, False, True, False, True],
                    [True, False, True, True, False, True],
                ]
            ],
            device=self.device,
        )
        for budget in (1, 32, 128, 1024):
            self.assertTrue(
                torch.equal(_pareto_tiled(batch, max_bytes=budget), expected)
            )

    def test_large_front_matches_cpu_reference(self) -> None:
        generator = torch.Generator().manual_seed(18)
        costs = torch.rand(2, 1025, 3, generator=generator, dtype=torch.double)
        costs[:, 100] = costs[:, 0]
        reference = knee_region(costs, maximize=False)
        self.assertTrue(
            torch.equal(
                knee_region(costs.to(self.device), maximize=False).cpu(), reference
            )
        )

    def front(self, dtype: torch.dtype = torch.double) -> torch.Tensor:
        return torch.tensor(
            [[0.0, 1.0], [0.2, 0.4], [0.4, 0.2], [1.0, 0.0]],
            dtype=dtype,
            device=self.device,
        )

    def test_source_equations(self) -> None:
        for dtype in (torch.float, torch.double):
            Y = self.front(dtype)
            mask = knee_region(Y, alpha=0.8, maximize=False)
            self.assertEqual(mask.tolist(), [False, True, True, False])
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(mask.device, Y.device)
            index = select_preferred(Y, [80.0, 20.0], alpha=0.8, maximize=False)
            self.assertEqual(index.item(), 1)
            self.assertEqual(index.dtype, torch.long)
            self.assertEqual(index.device, Y.device)
            self.assertEqual(
                select_preferred(Y, [20.0, 80.0], alpha=0.8, maximize=False).item(),
                2,
            )

    def test_signed_distance_and_zero_threshold(self) -> None:
        # The largest-norm row need not have the largest coordinate sum.
        Y = torch.tensor(
            [[1, 1, 0], [0.8, 0.8, 0.8], [0, 1, 1], [1, 0, 1]],
            dtype=torch.double,
            device=self.device,
        )
        for alpha in (0.0, 0.5, 1.0):
            self.assertEqual(
                knee_region(Y, alpha=alpha, maximize=False).tolist(),
                [True, False, True, True],
            )

    def test_filtering_and_original_indices(self) -> None:
        Y = self.front()
        Y = torch.cat([Y.new_full((1, 2), 1000), Y, Y[1:2]])
        self.assertEqual(
            knee_region(Y, alpha=0.8, maximize=False).tolist(),
            [False, False, True, True, False, False],
        )
        self.assertEqual(
            select_preferred(Y, [4, 1], alpha=0.8, maximize=False).item(), 2
        )

    def test_flat_axes_and_ties(self) -> None:
        Y = self.front()
        flat = torch.cat([Y, Y.new_full((4, 1), 7)], dim=-1)
        self.assertEqual(select_preferred(flat, [4, 1, 0], maximize=False).item(), 1)
        repeated = Y.new_full((3, 2), 5)
        self.assertEqual(knee_region(repeated).tolist(), [True, False, False])
        self.assertEqual(select_preferred(repeated, [1, 1]).item(), 0)
        # Equal weighted costs keep input order within the knee region.
        self.assertEqual(select_preferred(Y, [1, 1], maximize=False).item(), 1)

    def test_batched_preferences_and_noncontiguous_input(self) -> None:
        Y = self.front().expand(2, 3, 4, 2).transpose(0, 1)
        preferences = Y.new_tensor([[4, 1], [1, 4]])
        mask = knee_region(Y, maximize=False)
        indices = select_preferred(Y, preferences, maximize=False)
        self.assertEqual(mask.shape, torch.Size([3, 2, 4]))
        self.assertEqual(indices.tolist(), [[1, 2], [1, 2], [1, 2]])
        with self.assertRaisesRegex(ValueError, "broadcast"):
            select_preferred(Y, preferences.expand(4, 2, 2), maximize=False)

    def test_scale_and_orientation_invariance(self) -> None:
        Y = self.front()
        # Use distinct knee distances: exact-threshold ties can move by a few
        # ulps under finite-precision changes of objective units.
        Y[1, 1] = 0.3
        Y[2] = Y.new_tensor([0.6, 0.15])
        changed = Y * Y.new_tensor([1000, 0.01]) + Y.new_tensor([30, -2])
        for alpha in (0.0, 0.5, 1.0):
            expected = select_preferred(Y, [4, 1], alpha=alpha, maximize=False)
            self.assertEqual(
                select_preferred(
                    changed, [400, 100], alpha=alpha, maximize=False
                ).item(),
                expected.item(),
            )
            self.assertEqual(
                select_preferred(-Y, [4, 1], alpha=alpha).item(), expected.item()
            )

    def test_extreme_preferences_can_be_excluded(self) -> None:
        Y = self.front()
        # The unrestricted optimum for these preferences is row zero. Knee
        # filtering deliberately trades away some preferred objective value.
        self.assertEqual(Y[:, 0].argmin().item(), 0)
        self.assertEqual(select_preferred(Y, [1, 0], maximize=False).item(), 1)

    def test_extreme_finite_units(self) -> None:
        Y = torch.tensor(
            [[-1e308, 1e308], [-5e307, -5e307], [1e308, -1e308]],
            dtype=torch.double,
            device=self.device,
        )
        self.assertEqual(knee_region(Y, maximize=False).tolist(), [False, True, False])
        self.assertEqual(select_preferred(Y, [1e308, 1e308], maximize=False).item(), 1)

    def test_validation(self) -> None:
        Y = self.front()
        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                knee_region(Y, alpha=invalid)
        with self.assertRaises(TypeError):
            knee_region(Y, alpha=True)
        with self.assertRaises(TypeError):
            knee_region(Y, maximize=1)
        with self.assertRaises(TypeError):
            knee_region(Y.long())
        for shape in ((0, 2), (0, 3, 2), (3, 0), (2,)):
            with self.assertRaises(ValueError):
                knee_region(Y.new_empty(shape))
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                knee_region(Y.new_full((1, 2), invalid))
        for p in ([0, 0], [-1, 1], [float("nan"), 1], [float("inf"), 1], [1]):
            with self.assertRaises(ValueError):
                select_preferred(Y, p)


@skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestDecisionMakingCUDA(TestDecisionMaking):
    device = torch.device("cuda")
