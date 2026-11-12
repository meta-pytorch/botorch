#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    _expand_ref_point,
    _pad_batch_pareto_frontier,
    compute_dominated_hypercell_bounds_2d,
    compute_local_upper_bounds,
    compute_non_dominated_hypercell_bounds_2d,
    get_partition_bounds,
    update_local_upper_bounds_incremental,
)
from botorch.utils.testing import BotorchTestCase


class TestUtils(BotorchTestCase):
    def test_expand_ref_point(self):
        ref_point = torch.tensor([1.0, 2.0], device=self.device)
        for dtype in (torch.float, torch.double):
            ref_point = ref_point.to(dtype=dtype)
            # test non-batch
            self.assertTrue(
                torch.equal(
                    _expand_ref_point(ref_point, batch_shape=torch.Size([])),
                    ref_point,
                )
            )
            self.assertTrue(
                torch.equal(
                    _expand_ref_point(ref_point, batch_shape=torch.Size([3])),
                    ref_point.unsqueeze(0).expand(3, -1),
                )
            )
            # test ref point with wrong shape batch_shape
            with self.assertRaises(BotorchTensorDimensionError):
                _expand_ref_point(ref_point.unsqueeze(0), batch_shape=torch.Size([]))
            with self.assertRaises(BotorchTensorDimensionError):
                _expand_ref_point(ref_point.unsqueeze(0).expand(3, -1), torch.Size([2]))

    def test_pad_batch_pareto_frontier(self):
        for dtype in (torch.float, torch.double):
            Y1 = torch.tensor(
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
                    [8.0, 1.0],
                ],
                dtype=dtype,
                device=self.device,
            )

            Y2 = torch.tensor(
                [
                    [1.0, 9.0],
                    [10.0, 3.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                    [5.0, 5.0],
                    [8.5, 3.5],
                    [8.5, 3.5],
                    [8.5, 3.0],
                    [9.0, 5.0],
                    [9.0, 4.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            Y = torch.stack([Y1, Y2], dim=0)
            ref_point = torch.full((2, 2), 2.0, dtype=dtype, device=self.device)
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, is_pareto=False
            )
            expected_nondom_Y1 = torch.tensor(
                [[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_nondom_Y2 = torch.tensor(
                [
                    [10.0, 3.0],
                    [9.0, 5.0],
                    [9.0, 5.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_pareto = torch.stack(
                [expected_nondom_Y1, expected_padded_nondom_Y2], dim=0
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

            # test feasibility mask
            feas = (Y >= 9.0).any(dim=-1)
            expected_nondom_Y1 = torch.tensor(
                [[10.0, 3.0], [10.0, 3.0]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_nondom_Y2 = torch.tensor(
                [[10.0, 3.0], [9.0, 5.0]],
                dtype=dtype,
                device=self.device,
            )
            expected_padded_pareto = torch.stack(
                [expected_nondom_Y1, expected_padded_nondom_Y2], dim=0
            )
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, feasibility_mask=feas, is_pareto=False
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

            # test is_pareto=True
            # one row of Y2 should be dropped because it is not better than the
            # reference point
            Y1 = torch.tensor(
                [[10.0, 3.0], [5.0, 5.0], [8.5, 3.5]],
                dtype=dtype,
                device=self.device,
            )
            Y2 = torch.tensor(
                [
                    [1.0, 9.0],
                    [10.0, 3.0],
                    [9.0, 5.0],
                ],
                dtype=dtype,
                device=self.device,
            )
            Y = torch.stack([Y1, Y2], dim=0)
            expected_padded_pareto = torch.stack(
                [
                    Y1,
                    torch.cat([Y2[1:], Y2[-1:]], dim=0),
                ],
                dim=0,
            )
            padded_pareto = _pad_batch_pareto_frontier(
                Y=Y, ref_point=ref_point, is_pareto=True
            )
            self.assertTrue(torch.equal(padded_pareto, expected_padded_pareto))

        # test multiple batch dims
        with self.assertRaises(UnsupportedError):
            _pad_batch_pareto_frontier(
                Y=Y.unsqueeze(0), ref_point=ref_point, is_pareto=False
            )

    def test_compute_hypercell_bounds_2d(self):
        ref_point_raw = torch.zeros(2, device=self.device)
        arange = torch.arange(3, 9, device=self.device)
        pareto_Y_raw = torch.stack([arange, 11 - arange], dim=-1)
        inf = float("inf")
        for method in (
            compute_non_dominated_hypercell_bounds_2d,
            compute_dominated_hypercell_bounds_2d,
        ):
            if method == compute_non_dominated_hypercell_bounds_2d:
                expected_cell_bounds_raw = torch.tensor(
                    [
                        [
                            [0.0, 8.0],
                            [3.0, 7.0],
                            [4.0, 6.0],
                            [5.0, 5.0],
                            [6.0, 4.0],
                            [7.0, 3.0],
                            [8.0, 0.0],
                        ],
                        [
                            [3.0, inf],
                            [4.0, inf],
                            [5.0, inf],
                            [6.0, inf],
                            [7.0, inf],
                            [8.0, inf],
                            [inf, inf],
                        ],
                    ],
                    device=self.device,
                )
            else:
                expected_cell_bounds_raw = torch.tensor(
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
                    device=self.device,
                )
            for dtype in (torch.float, torch.double):
                pareto_Y = pareto_Y_raw.to(dtype=dtype)
                ref_point = ref_point_raw.to(dtype=dtype)
                expected_cell_bounds = expected_cell_bounds_raw.to(dtype=dtype)
                # test non-batch
                cell_bounds = method(
                    pareto_Y_sorted=pareto_Y,
                    ref_point=ref_point,
                )
                self.assertTrue(torch.equal(cell_bounds, expected_cell_bounds))
                # test batch
                pareto_Y_batch = torch.stack(
                    [pareto_Y, pareto_Y + pareto_Y.max(dim=-2).values], dim=0
                )
                # filter out points that are not better than ref_point
                ref_point = pareto_Y.max(dim=-2).values
                pareto_Y_batch = _pad_batch_pareto_frontier(
                    Y=pareto_Y_batch, ref_point=ref_point, is_pareto=True
                )
                # sort pareto_Y_batch
                pareto_Y_batch = pareto_Y_batch.gather(
                    index=torch.argsort(pareto_Y_batch[..., :1], dim=-2).expand(
                        pareto_Y_batch.shape
                    ),
                    dim=-2,
                )
                cell_bounds = method(
                    ref_point=ref_point,
                    pareto_Y_sorted=pareto_Y_batch,
                )
                # check hypervolume
                max_vals = (pareto_Y + pareto_Y).max(dim=-2).values
                if method == compute_non_dominated_hypercell_bounds_2d:
                    clamped_cell_bounds = torch.min(cell_bounds, max_vals)
                    total_hv = (max_vals - ref_point).prod()
                    nondom_hv = (
                        (clamped_cell_bounds[1] - clamped_cell_bounds[0])
                        .prod(dim=-1)
                        .sum(dim=-1)
                    )
                    hv = total_hv - nondom_hv
                else:
                    hv = (cell_bounds[1] - cell_bounds[0]).prod(dim=-1).sum(dim=-1)
                self.assertEqual(hv[0].item(), 0.0)
                self.assertEqual(hv[1].item(), 49.0)


class TestFastPartitioningUtils(BotorchTestCase):
    """
    Test on the problem (with the simplying assumption on general position)
    from Table 1 in:
    https://www.sciencedirect.com/science/article/pii/S0305054816301538
    """

    def setUp(self):
        super().setUp()
        self.ref_point = -torch.tensor([10.0, 10.0, 10.0], device=self.device)
        self.U = -self.ref_point.clone().view(1, -1)
        self.Z = torch.empty(1, 3, 3, device=self.device)
        ideal_value = 0.0
        for j in range(self.U.shape[-1]):
            self.Z[0, j] = torch.full(
                (1, self.U.shape[-1]),
                ideal_value,
                dtype=self.Z.dtype,
                device=self.device,
            )
            self.Z[0, j, j] = self.U[0][j]
        self.pareto_Y = -torch.tensor(
            [
                [3.0, 5.0, 7.0],
                [6.0, 2.0, 4.0],
                [4.0, 7.0, 3.0],
            ],
            device=self.device,
        )
        self.expected_U_after_update = torch.tensor(
            [
                [3.0, 10.0, 10.0],
                [6.0, 5.0, 10.0],
                [10.0, 2.0, 10.0],
                [4.0, 10.0, 7.0],
                [6.0, 7.0, 7.0],
                [10.0, 7.0, 4.0],
                [10.0, 10.0, 3.0],
            ],
            device=self.device,
        )
        self.expected_Z_after_update = torch.tensor(
            [
                [[3.0, 5.0, 7.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[6.0, 2.0, 4.0], [3.0, 5.0, 7.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [6.0, 2.0, 4.0], [0.0, 0.0, 10.0]],
                [[4.0, 7.0, 3.0], [0.0, 10.0, 0.0], [3.0, 5.0, 7.0]],
                [[6.0, 2.0, 4.0], [4.0, 7.0, 3.0], [3.0, 5.0, 7.0]],
                [[10.0, 0.0, 0.0], [4.0, 7.0, 3.0], [6.0, 2.0, 4.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [4.0, 7.0, 3.0]],
            ],
            device=self.device,
        )

    def test_local_upper_bounds_utils(self):
        for dtype in (torch.float, torch.double):
            U = self.U.to(dtype=dtype)
            Z = self.Z.to(dtype=dtype)
            pareto_Y = self.pareto_Y.to(dtype=dtype)
            expected_U = self.expected_U_after_update.to(dtype=dtype)
            expected_Z = self.expected_Z_after_update.to(dtype=dtype)

            # test z dominates U
            U_new, Z_new = compute_local_upper_bounds(U=U, Z=Z, z=-self.ref_point + 1)
            self.assertTrue(torch.equal(U_new, U))
            self.assertTrue(torch.equal(Z_new, Z))

            # test compute_local_upper_bounds
            for i in range(pareto_Y.shape[0]):
                U, Z = compute_local_upper_bounds(U=U, Z=Z, z=-pareto_Y[i])
            self.assertTrue(torch.equal(U, expected_U))
            self.assertTrue(torch.equal(Z, expected_Z))

            # test update_local_upper_bounds_incremental
            # test that calling update_local_upper_bounds_incremental once with
            # the entire Pareto set yields the same result
            U2, Z2 = update_local_upper_bounds_incremental(
                new_pareto_Y=-pareto_Y,
                U=self.U.to(dtype=dtype),
                Z=self.Z.to(dtype=dtype),
            )
            self.assertTrue(torch.equal(U2, expected_U))
            self.assertTrue(torch.equal(Z2, expected_Z))

    def test_get_partition_bounds(self):
        expected_bounds_raw = torch.tensor(
            [
                [[3.0, 5.0, 7.0], [6.0, 2.0, 7.0], [4.0, 7.0, 3.0], [6.0, 2.0, 4.0]],
                [
                    [10.0, 10.0, 10.0],
                    [10.0, 5.0, 10.0],
                    [10.0, 10.0, 7.0],
                    [10.0, 7.0, 7.0],
                ],
            ],
            device=self.device,
        )
        for dtype in (torch.float, torch.double):
            final_U = self.expected_U_after_update.to(dtype=dtype)
            final_Z = self.expected_Z_after_update.to(dtype=dtype)
            bounds = get_partition_bounds(
                Z=final_Z, U=final_U, ref_point=-self.ref_point
            )
            expected_bounds = expected_bounds_raw.to(dtype=dtype)
            self.assertTrue(torch.equal(bounds, expected_bounds))

    def test_get_partition_bounds_matches_loop(self):
        # Compare against a direct implementation of Equation 2 in [Lacour17]_.
        def get_partition_bounds_loop(Z, U, ref_point):
            bounds = torch.empty(2, *U.shape, dtype=U.dtype, device=U.device)
            for u_idx in range(U.shape[0]):
                bounds[0, u_idx, 0] = Z[u_idx, 0, 0]
                bounds[1, u_idx, 0] = ref_point[0]
                for j in range(1, U.shape[-1]):
                    bounds[0, u_idx, j] = Z[u_idx, :j, j].max()
                    bounds[1, u_idx, j] = U[u_idx, j]
            empty = (bounds[1] <= bounds[0]).any(dim=-1)
            return bounds[:, ~empty]

        for dtype, m in product((torch.float, torch.double), (3, 4)):
            tkwargs = {"device": self.device, "dtype": dtype}
            # Use the local upper bounds and defining points of both steps.
            bd = FastNondominatedPartitioning(
                ref_point=torch.zeros(m, **tkwargs), Y=torch.rand(10, m, **tkwargs)
            )
            for Z, U, ref_point in (
                (bd._Z, bd._U, bd._neg_ref_point.view(-1)),
                (bd._Z2, bd._U2, torch.full((m,), float("inf"), **tkwargs)),
            ):
                self.assertTrue(
                    torch.equal(
                        get_partition_bounds(Z=Z, U=U, ref_point=ref_point),
                        get_partition_bounds_loop(Z=Z, U=U, ref_point=ref_point),
                    )
                )
