#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase


class TestDominatedPartitioning(BotorchTestCase):
    def test_dominated_partitioning(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            ref_point = torch.zeros(2, **tkwargs)
            partitioning = DominatedPartitioning(ref_point=ref_point)
            # assert error is raised if pareto_Y has not been computed
            with self.assertRaises(BotorchError):
                partitioning.pareto_Y
            partitioning = DominatedPartitioning(ref_point=ref_point)
            # test _reset_pareto_Y
            Y = torch.ones(1, 2, **tkwargs)
            partitioning.update(Y=Y)

            partitioning._neg_Y = -Y
            partitioning.batch_shape = torch.Size([])
            self.assertFalse(partitioning._reset_pareto_Y())

            # test m=2
            arange = torch.arange(3, 9, **tkwargs)
            pareto_Y = torch.stack([arange, 11 - arange], dim=-1)
            Y = torch.cat(
                [
                    pareto_Y,
                    torch.tensor(
                        [[8.0, 2.0], [7.0, 1.0]], **tkwargs
                    ),  # add some non-pareto elements
                ],
                dim=0,
            )
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y)
            sorting = torch.argsort(pareto_Y[:, 0], descending=True)
            self.assertTrue(torch.equal(pareto_Y[sorting], partitioning.pareto_Y))
            expected_cell_bounds = torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [3.0, 0.0],
                        [4.0, 0.0],
                        [5.0, 0.0],
                        [6.0, 0.0],
                        [7.0, 0.0],
                    ],
                    [
                        [3.0, 8.0],
                        [4.0, 7.0],
                        [5.0, 6.0],
                        [6.0, 5.0],
                        [7.0, 4.0],
                        [8.0, 3.0],
                    ],
                ],
                **tkwargs,
            )
            cell_bounds = partitioning.get_hypercell_bounds()
            self.assertTrue(torch.equal(cell_bounds, expected_cell_bounds))
            # test compute hypervolume
            hv = partitioning.compute_hypervolume()
            self.assertEqual(hv.item(), 49.0)
            # test no pareto points better than the reference point
            partitioning = DominatedPartitioning(
                ref_point=pareto_Y.max(dim=-2).values + 1, Y=Y
            )
            self.assertTrue(torch.equal(partitioning.pareto_Y, Y[:0]))
            self.assertEqual(partitioning.compute_hypervolume().item(), 0)

            Y = torch.rand(3, 10, 2, **tkwargs)

            # test batched m=2
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y)
            cell_bounds = partitioning.get_hypercell_bounds()
            partitionings = []
            for i in range(Y.shape[0]):
                partitioning_i = DominatedPartitioning(ref_point=ref_point, Y=Y[i])
                partitionings.append(partitioning_i)
                # check pareto_Y
                pareto_set1 = {tuple(x) for x in partitioning_i.pareto_Y.tolist()}
                pareto_set2 = {tuple(x) for x in partitioning.pareto_Y[i].tolist()}
                self.assertEqual(pareto_set1, pareto_set2)
                expected_cell_bounds_i = partitioning_i.get_hypercell_bounds()
                # remove padding
                no_padding_cell_bounds_i = cell_bounds[:, i][
                    :, ((cell_bounds[1, i] - cell_bounds[0, i]) != 0).all(dim=-1)
                ]
                self.assertTrue(
                    torch.equal(expected_cell_bounds_i, no_padding_cell_bounds_i)
                )

            # test batch ref point
            partitioning = DominatedPartitioning(
                ref_point=ref_point.unsqueeze(0).expand(3, *ref_point.shape), Y=Y
            )
            cell_bounds2 = partitioning.get_hypercell_bounds()
            self.assertTrue(torch.equal(cell_bounds, cell_bounds2))

            # test batch ref point, no pareto points better than the reference point
            batch_ref_point = Y.max(dim=-2).values + 1
            partitioning = DominatedPartitioning(ref_point=batch_ref_point, Y=Y)
            self.assertTrue(torch.equal(partitioning.pareto_Y, Y[:, :0]))
            # one empty cell per batch element, bounded by its reference point
            self.assertTrue(
                torch.equal(
                    partitioning.get_hypercell_bounds(),
                    batch_ref_point.unsqueeze(-2).expand(2, 3, 1, 2),
                )
            )
            self.assertTrue(
                torch.equal(
                    partitioning.compute_hypervolume(), torch.zeros(3, **tkwargs)
                )
            )

            # test batched where batches have different numbers of pareto points
            partitioning = DominatedPartitioning(
                ref_point=pareto_Y.max(dim=-2).values,
                Y=torch.stack(
                    [pareto_Y, pareto_Y + pareto_Y.max(dim=-2).values], dim=0
                ),
            )
            hv = partitioning.compute_hypervolume()
            self.assertEqual(hv[0].item(), 0.0)
            self.assertEqual(hv[1].item(), 49.0)
            cell_bounds = partitioning.get_hypercell_bounds()
            self.assertEqual(cell_bounds.shape, torch.Size([2, 2, 6, 2]))

            # test batched m>2
            ref_point = torch.zeros(3, **tkwargs)
            with self.assertRaises(NotImplementedError):
                DominatedPartitioning(
                    ref_point=ref_point, Y=torch.cat([Y, Y[..., :1]], dim=-1)
                )

            # test m=3
            pareto_Y = torch.tensor(
                [[1.0, 6.0, 8.0], [2.0, 4.0, 10.0], [3.0, 5.0, 7.0]], **tkwargs
            )
            ref_point = torch.tensor([-1.0, -2.0, -3.0], **tkwargs)
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
            self.assertTrue(torch.equal(pareto_Y, partitioning.pareto_Y))
            # test compute hypervolume
            hv = partitioning.compute_hypervolume()
            self.assertEqual(hv.item(), 358.0)

            # test no pareto points better than the reference point, non-batched
            partitioning = DominatedPartitioning(
                ref_point=pareto_Y.max(dim=-2).values + 1, Y=pareto_Y
            )
            self.assertTrue(torch.equal(partitioning.pareto_Y, pareto_Y[:0]))
            self.assertEqual(
                partitioning.get_hypercell_bounds().shape,
                torch.Size([2, 1, pareto_Y.shape[-1]]),
            )
            self.assertEqual(partitioning.compute_hypervolume().item(), 0)

        # Test that updating the partitioning does not lead to a buffer error.
        partitioning = DominatedPartitioning(
            ref_point=torch.zeros(3), Y=-torch.ones(1, 3)
        )
        self.assertTrue(
            torch.equal(partitioning.hypercell_bounds, torch.zeros(2, 1, 3))
        )
        partitioning.update(Y=torch.ones(1, 3))
        self.assertEqual(partitioning.compute_hypervolume().item(), 1)
