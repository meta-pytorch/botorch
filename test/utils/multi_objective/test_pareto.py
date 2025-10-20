#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product
from unittest.mock import patch

import torch
from botorch.utils.multi_objective.pareto import (
    _is_non_dominated_loop,
    is_non_dominated,
)
from botorch.utils.testing import BotorchTestCase


class TestPareto(BotorchTestCase):
    def test_is_non_dominated(self) -> None:
        tkwargs = {"device": self.device}
        Y = torch.tensor(
            [
                [1.0, 5.0],
                [10.0, 3.0],
                [4.0, 5.0],
                [4.0, 5.0],
                [5.0, 5.0],
                [8.5, 3.5],
                [8.5, 3.5],
                [8.5, 3.0],
                [9.0, 1.0],
            ]
        )
        expected_nondom_Y = torch.tensor([[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]])
        Yb = Y.clone()
        Yb[1] = 0
        expected_nondom_Yb = torch.tensor([[5.0, 5.0], [8.5, 3.5], [9.0, 1.0]])
        Y3 = torch.tensor(
            [
                [4.0, 2.0, 3.0],
                [2.0, 4.0, 1.0],
                [3.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [2.0, 4.0, 2.0],
                [1.0, 3.0, 4.0],
                [1.0, 2.0, 4.0],
                [1.0, 2.0, 6.0],
            ]
        )
        Y3b = Y3.clone()
        Y3b[0] = 0
        expected_nondom_Y3 = torch.tensor(
            [
                [4.0, 2.0, 3.0],
                [3.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 3.0, 4.0],
                [1.0, 2.0, 6.0],
            ]
        )
        expected_nondom_Y3b = expected_nondom_Y3[1:]
        for dtype, maximize in product((torch.float, torch.double), (True, False)):
            tkwargs["dtype"] = dtype
            Y = Y.to(**tkwargs)
            expected_nondom_Y = expected_nondom_Y.to(**tkwargs)
            Yb = Yb.to(**tkwargs)
            expected_nondom_Yb = expected_nondom_Yb.to(**tkwargs)
            Y3 = Y3.to(**tkwargs)
            expected_nondom_Y3 = expected_nondom_Y3.to(**tkwargs)
            Y3b = Y3b.to(**tkwargs)
            expected_nondom_Y3b = expected_nondom_Y3b.to(**tkwargs)

            test_Y = Y if maximize else -Y

            # test 2d
            nondom_Y = Y[is_non_dominated(test_Y, maximize=maximize)]
            self.assertTrue(torch.equal(expected_nondom_Y, nondom_Y))
            # test deduplicate=False
            expected_nondom_Y_no_dedup = torch.cat(
                [expected_nondom_Y, expected_nondom_Y[-1:]], dim=0
            )
            nondom_Y = Y[is_non_dominated(test_Y, maximize=maximize, deduplicate=False)]
            self.assertTrue(torch.equal(expected_nondom_Y_no_dedup, nondom_Y))

            # test batch
            batch_Y = torch.stack([Y, Yb], dim=0)
            test_batch_Y = batch_Y if maximize else -batch_Y
            nondom_mask = is_non_dominated(test_batch_Y, maximize=maximize)
            self.assertTrue(torch.equal(batch_Y[0][nondom_mask[0]], expected_nondom_Y))
            self.assertTrue(torch.equal(batch_Y[1][nondom_mask[1]], expected_nondom_Yb))
            # test deduplicate=False
            expected_nondom_Yb_no_dedup = torch.cat(
                [expected_nondom_Yb[:-1], expected_nondom_Yb[-2:]], dim=0
            )
            nondom_mask = is_non_dominated(
                test_batch_Y, maximize=maximize, deduplicate=False
            )
            self.assertTrue(
                torch.equal(batch_Y[0][nondom_mask[0]], expected_nondom_Y_no_dedup)
            )
            self.assertTrue(
                torch.equal(batch_Y[1][nondom_mask[1]], expected_nondom_Yb_no_dedup)
            )

            # test 3d
            test_Y3 = Y3 if maximize else -Y3
            nondom_Y3 = Y3[is_non_dominated(test_Y3, maximize=maximize)]
            self.assertTrue(torch.equal(expected_nondom_Y3, nondom_Y3))
            # test deduplicate=False
            expected_nondom_Y3_no_dedup = torch.cat(
                [expected_nondom_Y3[:3], expected_nondom_Y3[2:]], dim=0
            )
            nondom_Y3 = Y3[
                is_non_dominated(test_Y3, maximize=maximize, deduplicate=False)
            ]
            self.assertTrue(torch.equal(expected_nondom_Y3_no_dedup, nondom_Y3))
            # test batch
            batch_Y3 = torch.stack([Y3, Y3b], dim=0)
            test_batch_Y3 = batch_Y3 if maximize else -batch_Y3
            nondom_mask3 = is_non_dominated(test_batch_Y3, maximize=maximize)
            self.assertTrue(
                torch.equal(batch_Y3[0][nondom_mask3[0]], expected_nondom_Y3)
            )
            self.assertTrue(
                torch.equal(batch_Y3[1][nondom_mask3[1]], expected_nondom_Y3b)
            )
            # test deduplicate=False
            nondom_mask3 = is_non_dominated(
                test_batch_Y3, maximize=maximize, deduplicate=False
            )
            self.assertTrue(
                torch.equal(batch_Y3[0][nondom_mask3[0]], expected_nondom_Y3_no_dedup)
            )
            expected_nondom_Y3b_no_dedup = torch.cat(
                [expected_nondom_Y3b[:2], expected_nondom_Y3b[1:]], dim=0
            )
            self.assertTrue(
                torch.equal(batch_Y3[1][nondom_mask3[1]], expected_nondom_Y3b_no_dedup)
            )

            # test empty pareto
            mask = is_non_dominated(Y3[:0])
            expected_mask = torch.zeros(0, dtype=torch.bool, device=Y3.device)
            self.assertTrue(torch.equal(expected_mask, mask))
            mask = is_non_dominated(batch_Y3[:, :0])
            expected_mask = torch.zeros(
                *batch_Y3.shape[:-2], 0, dtype=torch.bool, device=Y3.device
            )
            self.assertTrue(torch.equal(expected_mask, mask))
            with patch(
                "botorch.utils.multi_objective.pareto._is_non_dominated_loop"
            ) as mock_is_non_dominated_loop:
                y = torch.rand(1001, 2, dtype=dtype, device=Y3.device)
                is_non_dominated(y)
                mock_is_non_dominated_loop.assert_called_once()
                cargs = mock_is_non_dominated_loop.call_args[0]
                self.assertTrue(torch.equal(cargs[0], y))

    def test_is_non_dominated_with_nan(self) -> None:
        # NaN should always evaluate to False.
        Y = torch.rand(10, 2)
        Y[3, 1] = float("nan")
        Y[7, 0] = float("nan")
        self.assertFalse(is_non_dominated(Y)[[3, 7]].any())
        # Check edge case if all elements are NaN
        nans = torch.full((2, 2, 3), torch.nan)
        rands = torch.rand((2, 2, 3))
        Y = torch.hstack([nans, rands])
        non_dominated = is_non_dominated(Y)
        self.assertFalse(non_dominated[..., :, :2].any().item())

    def test_is_non_dominated_loop(self):
        n = 20
        tkwargs = {"device": self.device}
        for dtype, batch_shape, m, maximize in product(
            (torch.float, torch.double),
            (torch.Size([]), torch.Size([2])),
            (1, 2, 3),
            (True, False),
        ):
            tkwargs["dtype"] = dtype
            Y = torch.rand(batch_shape + torch.Size([n, m]), **tkwargs)
            pareto_mask = _is_non_dominated_loop(
                # this is so that we can assume maximization in the test code
                Y=Y if maximize else -Y,
                maximize=maximize,
            )
            self.assertEqual(pareto_mask.shape, Y.shape[:-1])
            self.assertEqual(pareto_mask.dtype, torch.bool)
            self.assertEqual(pareto_mask.device.type, self.device.type)
            if len(batch_shape) > 0:
                pareto_masks = [pareto_mask[i] for i in range(pareto_mask.shape[0])]
            else:
                pareto_masks = [pareto_mask]
                Y = Y.unsqueeze(0)
            for i, mask in enumerate(pareto_masks):
                pareto_Y = Y[i][mask]
                pareto_indices = mask.nonzero().view(-1)
                if pareto_Y.shape[0] > 1:
                    # compare against other pareto points
                    point_mask = torch.zeros(
                        pareto_Y.shape[0], dtype=torch.bool, device=self.device
                    )
                    Y_not_j_mask = torch.ones(
                        Y[i].shape[0], dtype=torch.bool, device=self.device
                    )
                    for j in range(pareto_Y.shape[0]):
                        point_mask[j] = True
                        # check each pareto point is non-dominated
                        Y_idx = pareto_indices[j].item()
                        Y_not_j_mask[Y_idx] = False
                        self.assertFalse(
                            (pareto_Y[point_mask] <= Y[i][Y_not_j_mask])
                            .all(dim=-1)
                            .any()
                        )
                        Y_not_j_mask[Y_idx] = True
                        if pareto_Y.shape[0] > 1:
                            # check that each point is better than
                            # pareto_Y[j] in some objective
                            j_better_than_Y = (
                                pareto_Y[point_mask] > pareto_Y[~point_mask]
                            )
                            best_obj_mask = torch.zeros(
                                m, dtype=torch.bool, device=self.device
                            )
                            for k in range(m):
                                best_obj_mask[k] = True
                                j_k_better_than_Y = j_better_than_Y[:, k]
                                if j_k_better_than_Y.any():
                                    self.assertTrue(
                                        (
                                            pareto_Y[point_mask, ~best_obj_mask]
                                            < pareto_Y[~point_mask][j_k_better_than_Y][
                                                :, ~best_obj_mask
                                            ]
                                        )
                                        .any(dim=-1)
                                        .all()
                                    )
                                best_obj_mask[k] = False
                        point_mask[j] = False

    def test_is_non_dominated_loop_no_dedup(self):
        not_dom_expected_dedup = torch.tensor(
            [
                [True, False, False, False, True, False],
                [False, True, True, False, False, False],
            ],
            device=self.device,
        )
        not_dom_expected_no_dedup = torch.tensor(
            [
                [True, False, True, False, True, False],
                [False, True, True, False, False, True],
            ],
            device=self.device,
        )

        for dtype, maximize, deduplicate in product(
            (torch.float, torch.double), (True, False), (True, False)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            y1 = torch.tensor([1.0, 0.75], **tkwargs)
            y2 = torch.tensor([0.75, 1.0], **tkwargs)
            y3 = torch.tensor([1.0, 0.3], **tkwargs)
            # 0.5 ensures all points are dominated by y1/y2
            Y = 0.5 * torch.rand(2, 6, 2, **tkwargs)

            Y[0, 0] = y1
            Y[0, 2] = y1
            Y[0, 4] = y2
            Y[0, 3] = y3  # weakly Pareto optimal, dominated by y1

            Y[1, 1] = y1
            Y[1, 2] = y2
            Y[1, 5] = y2

            Y = Y if maximize else -Y
            not_dom = _is_non_dominated_loop(
                Y, maximize=maximize, deduplicate=deduplicate
            )

            if deduplicate:
                self.assertTrue(torch.equal(not_dom, not_dom_expected_dedup))
            else:
                self.assertTrue(torch.equal(not_dom, not_dom_expected_no_dedup))
