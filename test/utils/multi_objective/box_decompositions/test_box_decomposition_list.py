#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase


class TestBoxDecompositionList(BotorchTestCase):
    def test_box_decomposition_list(self):
        ref_point_raw = torch.zeros(3, device=self.device)
        pareto_Y_raw = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 0.5, 1.0],
            ],
            device=self.device,
        )
        for m, dtype in product((2, 3), (torch.float, torch.double)):
            ref_point = ref_point_raw[:m].to(dtype=dtype)
            pareto_Y = pareto_Y_raw[:, :m].to(dtype=dtype)
            pareto_Y_list = [pareto_Y[:0, :m], pareto_Y[:, :m]]
            bds = [
                FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
                for Y in pareto_Y_list
            ]
            bd = BoxDecompositionList(*bds)
            # test pareto Y
            bd_pareto_Y_list = bd.pareto_Y
            pareto_Y1 = pareto_Y_list[1]
            expected_pareto_Y1 = (
                pareto_Y1[torch.argsort(-pareto_Y1[:, 0])] if m == 2 else pareto_Y1
            )
            self.assertTrue(torch.equal(bd_pareto_Y_list[0], pareto_Y_list[0]))
            self.assertTrue(torch.equal(bd_pareto_Y_list[1], expected_pareto_Y1))
            # test ref_point
            self.assertTrue(
                torch.equal(bd.ref_point, ref_point.unsqueeze(0).expand(2, -1))
            )
            # test get_hypercell_bounds
            cell_bounds = bd.get_hypercell_bounds()
            expected_cell_bounds1 = bds[1].get_hypercell_bounds()
            self.assertTrue(torch.equal(cell_bounds[:, 1], expected_cell_bounds1))
            # the first pareto set in the list is empty so the cell bounds
            # should contain one cell that spans the entire area (bounded by the
            # ref_point) and then empty cells, bounded from above and below by the
            # ref point.
            expected_cell_bounds0 = torch.zeros_like(expected_cell_bounds1)
            # set the upper bound for the first cell to be inf
            expected_cell_bounds0[1, 0, :] = float("inf")
            self.assertTrue(torch.equal(cell_bounds[:, 0], expected_cell_bounds0))
            # test compute_hypervolume
            expected_hv = torch.stack([b.compute_hypervolume() for b in bds], dim=0)
            hv = bd.compute_hypervolume()
            self.assertTrue(torch.equal(expected_hv, hv))

            # test update with batched tensor
            new_Y = torch.empty(2, 1, m, dtype=dtype, device=self.device)
            new_Y[0] = 1
            new_Y[1] = 3
            bd.update(new_Y)
            bd_pareto_Y_list = bd.pareto_Y
            self.assertTrue(torch.equal(bd_pareto_Y_list[0], new_Y[0]))
            self.assertTrue(torch.equal(bd_pareto_Y_list[1], new_Y[1]))

            # test update with list
            bd = BoxDecompositionList(*bds)
            bd.update([new_Y[0], new_Y[1]])
            bd_pareto_Y_list = bd.pareto_Y
            self.assertTrue(torch.equal(bd_pareto_Y_list[0], new_Y[0]))
            self.assertTrue(torch.equal(bd_pareto_Y_list[1], new_Y[1]))

            # test update with wrong shape
            bd = BoxDecompositionList(*bds)
            with self.assertRaises(BotorchTensorDimensionError):
                bd.update(new_Y.unsqueeze(0))
            # more batches than box decompositions
            with self.assertRaises(BotorchTensorDimensionError):
                bd.update(torch.cat([new_Y, new_Y[:1]], dim=0))
            # a 2-dim tensor with one point per box decomposition
            with self.assertRaises(BotorchTensorDimensionError):
                bd.update(new_Y.squeeze(-2))
