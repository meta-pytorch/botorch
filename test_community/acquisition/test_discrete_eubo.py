#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.exceptions import UnsupportedError
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.discrete_eubo import select_discrete_eubo_pair
from botorch_community.models.dual_pairwise_gp import DualPairwiseGP


class TestDiscreteEUBO(BotorchTestCase):
    def test_matches_analytic_eubo_and_recovers_optimum(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.float64}
        candidates = torch.linspace(0.0, 1.0, 17, **tkwargs).unsqueeze(-1)
        true_u = -((candidates.squeeze(-1) - 0.75) ** 2)

        model = DualPairwiseGP(candidates)
        best_pair, best_val, eubo_mat = select_discrete_eubo_pair(model, candidates)
        self.assertEqual(best_pair.shape, torch.Size([2]))
        self.assertEqual(eubo_mat.shape, torch.Size([17, 17]))

        # Cross-check scalar value against core AnalyticExpectedUtilityOfBestOption
        analytic_eubo = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        pair_X = candidates[best_pair].unsqueeze(0)
        ref_val = analytic_eubo(pair_X).squeeze(0)
        self.assertTrue(torch.allclose(best_val, ref_val, atol=1e-6))

        # Run 6 closed-loop discrete EUBO steps and verify convergence near 0.75
        queried: list[list[int]] = []
        winners_losers: list[list[int]] = []
        for _ in range(6):
            comp_tensor = (
                torch.tensor(queried, dtype=torch.long, device=self.device)
                if queried
                else None
            )
            pair, _, _ = select_discrete_eubo_pair(
                model, candidates, compared_pairs=comp_tensor
            )
            i_idx, j_idx = int(pair[0].item()), int(pair[1].item())
            queried.append([i_idx, j_idx])
            if true_u[i_idx] >= true_u[j_idx]:
                winners_losers.append([i_idx, j_idx])
            else:
                winners_losers.append([j_idx, i_idx])
            model.set_preferences(
                comparisons=torch.tensor(
                    winners_losers, dtype=torch.long, device=self.device
                )
            )

        post_mean = model.posterior(candidates).mean.squeeze(-1)
        best_found_x = float(candidates[int(torch.argmax(post_mean).item()), 0].item())
        self.assertLessEqual(abs(best_found_x - 0.75), 0.13)

    def test_exhausted_and_invalid_inputs(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.float64}
        two_pts = torch.tensor([[0.0], [1.0]], **tkwargs)
        model = DualPairwiseGP(two_pts)
        with self.assertRaises(UnsupportedError):
            select_discrete_eubo_pair(model, two_pts[:1])
        with self.assertRaises(UnsupportedError):
            select_discrete_eubo_pair(
                model, two_pts, compared_pairs=torch.tensor([[0, 1]])
            )
