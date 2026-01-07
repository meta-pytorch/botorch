#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product
from unittest import mock

import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
    FastPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    update_local_upper_bounds_incremental,
)
from botorch.utils.testing import BotorchTestCase


class DummyBoxDecomposition(BoxDecomposition):
    def _partition_space(self):
        pass

    def _compute_hypervolume_if_y_has_data(self):
        pass

    def get_hypercell_bounds(self):
        pass


class DummyFastPartitioning(FastPartitioning, DummyBoxDecomposition):
    def _get_partitioning(self):
        pass

    def _get_single_cell(self):
        pass


class TestBoxDecomposition(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.ref_point_raw = torch.zeros(3, device=self.device)
        self.Y_raw = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.5, 1.0],
            ],
            device=self.device,
        )
        self.pareto_Y_raw = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 0.5, 1.0],
            ],
            device=self.device,
        )

    def test_box_decomposition(self) -> None:
        with self.assertRaises(TypeError):
            BoxDecomposition()
        for dtype, m, sort in product(
            (torch.float, torch.double), (2, 3), (True, False)
        ):
            with mock.patch.object(
                DummyBoxDecomposition,
                "_partition_space_2d" if m == 2 else "_partition_space",
            ) as mock_partition_space:
                ref_point = self.ref_point_raw[:m].to(dtype=dtype)
                Y = self.Y_raw[:, :m].to(dtype=dtype)
                pareto_Y = self.pareto_Y_raw[:, :m].to(dtype=dtype)
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort)

                # test pareto_Y before it is initialized
                with self.assertRaises(BotorchError):
                    bd.pareto_Y
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort, Y=Y)

                mock_partition_space.assert_called_once()
                # test attributes
                expected_pareto_Y = (
                    pareto_Y[torch.argsort(-pareto_Y[:, 0])] if sort else pareto_Y
                )
                self.assertTrue(torch.equal(bd.pareto_Y, expected_pareto_Y))
                self.assertTrue(torch.equal(bd.Y, Y))
                self.assertTrue(torch.equal(bd._neg_Y, -Y))
                self.assertTrue(torch.equal(bd._neg_pareto_Y, -expected_pareto_Y))
                self.assertTrue(torch.equal(bd.ref_point, ref_point))
                self.assertTrue(torch.equal(bd._neg_ref_point, -ref_point))
                self.assertEqual(bd.num_outcomes, m)

                # test empty Y
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort, Y=Y[:0])
                self.assertTrue(torch.equal(bd.pareto_Y, expected_pareto_Y[:0]))

                # test _update_neg_Y
                bd = DummyBoxDecomposition(ref_point=ref_point, sort=sort)
                bd._update_neg_Y(Y[:2])
                self.assertTrue(torch.equal(bd._neg_Y, -Y[:2]))
                bd._update_neg_Y(Y[2:])
                self.assertTrue(torch.equal(bd._neg_Y, -Y))

                # test batch mode
                if m == 2:
                    batch_Y = torch.stack([Y, Y + 1], dim=0)
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y
                    )
                    batch_expected_pareto_Y = torch.stack(
                        [expected_pareto_Y, expected_pareto_Y + 1], dim=0
                    )
                    self.assertTrue(torch.equal(bd.pareto_Y, batch_expected_pareto_Y))
                    self.assertTrue(torch.equal(bd.Y, batch_Y))
                    self.assertTrue(torch.equal(bd.ref_point, ref_point))
                    # test batch ref point
                    batch_ref_point = torch.stack([ref_point, ref_point + 1], dim=0)
                    bd = DummyBoxDecomposition(
                        ref_point=batch_ref_point, sort=sort, Y=batch_Y
                    )
                    self.assertTrue(torch.equal(bd.ref_point, batch_ref_point))
                    # test multiple batch dims
                    with self.assertRaises(NotImplementedError):
                        DummyBoxDecomposition(
                            ref_point=ref_point,
                            sort=sort,
                            Y=batch_Y.unsqueeze(0),
                        )
                    # test empty Y
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y[:, :0]
                    )
                    self.assertTrue(
                        torch.equal(bd.pareto_Y, batch_expected_pareto_Y[:, :0])
                    )

                    # test padded pareto frontiers with different numbers of
                    # points
                    batch_Y[1, 1] = batch_Y[1, 0] - 1
                    batch_Y[1, 2] = batch_Y[1, 0] - 2
                    bd = DummyBoxDecomposition(
                        ref_point=ref_point, sort=sort, Y=batch_Y
                    )
                    batch_expected_pareto_Y = torch.stack(
                        [
                            expected_pareto_Y,
                            batch_Y[1, :1].expand(expected_pareto_Y.shape),
                        ],
                        dim=0,
                    )
                    self.assertTrue(torch.equal(bd.pareto_Y, batch_expected_pareto_Y))
                    self.assertTrue(torch.equal(bd.Y, batch_Y))

                else:
                    with self.assertRaises(NotImplementedError):
                        DummyBoxDecomposition(
                            ref_point=ref_point, sort=sort, Y=Y.unsqueeze(0)
                        )

    def test_fast_partitioning(self):
        with self.assertRaises(TypeError):
            FastPartitioning()
        for dtype, m in product(
            (torch.float, torch.double),
            (2, 3),
        ):
            ref_point = self.ref_point_raw[:m].to(dtype=dtype)
            Y = self.Y_raw[:, :m].to(dtype=dtype)
            pareto_Y = self.pareto_Y_raw[:, :m].to(dtype=dtype)
            sort = m == 2
            expected_pareto_Y = (
                pareto_Y[torch.argsort(-pareto_Y[:, 0])] if sort else pareto_Y
            )
            bd = DummyFastPartitioning(ref_point=ref_point, Y=Y)
            self.assertTrue(torch.equal(bd.pareto_Y, expected_pareto_Y))
            self.assertTrue(torch.equal(bd.Y, Y))
            self.assertTrue(torch.equal(bd._neg_Y, -Y))
            self.assertTrue(torch.equal(bd._neg_pareto_Y, -expected_pareto_Y))
            self.assertTrue(torch.equal(bd.ref_point, ref_point))
            self.assertTrue(torch.equal(bd._neg_ref_point, -ref_point))
            self.assertEqual(bd.num_outcomes, m)
            # test update
            bd = DummyFastPartitioning(ref_point=ref_point)
            with mock.patch.object(
                DummyFastPartitioning,
                "reset",
                wraps=bd.reset,
            ) as mock_reset:
                # with no existing neg_Y
                bd.update(Y=Y[:2])
                mock_reset.assert_called_once()
                # test with existing Y
                bd.update(Y=Y[2:])
                # check that reset is only called when m=2
                if m == 2:
                    mock_reset.assert_has_calls([mock.call(), mock.call()])
                else:
                    mock_reset.assert_called_once()

            # with existing neg_Y, and empty pareto_Y
            bd = DummyFastPartitioning(ref_point=ref_point, Y=Y[:0])
            with mock.patch.object(
                DummyFastPartitioning,
                "reset",
                wraps=bd.reset,
            ) as mock_reset:
                bd.update(Y=Y[0:])
                mock_reset.assert_called_once()

            # test empty pareto Y
            bd = DummyFastPartitioning(ref_point=ref_point)
            with mock.patch.object(
                DummyFastPartitioning,
                "_get_single_cell",
                wraps=bd._get_single_cell,
            ) as mock_get_single_cell:
                bd.update(Y=Y[:0])
                mock_get_single_cell.assert_called_once()
            # test batched empty pareto Y
            if m == 2:
                bd = DummyFastPartitioning(ref_point=ref_point)
                with mock.patch.object(
                    DummyFastPartitioning,
                    "_get_single_cell",
                    wraps=bd._get_single_cell,
                ) as mock_get_single_cell:
                    bd.update(Y=Y.unsqueeze(0)[:, :0])
                    mock_get_single_cell.assert_called_once()

            # test that update_local_upper_bounds_incremental is called when m>2
            bd = DummyFastPartitioning(ref_point=ref_point)
            with mock.patch(
                "botorch.utils.multi_objective.box_decompositions.box_decomposition."
                "update_local_upper_bounds_incremental",
                wraps=update_local_upper_bounds_incremental,
            ) as mock_update_local_upper_bounds_incremental, mock.patch.object(
                DummyFastPartitioning,
                "_get_partitioning",
                wraps=bd._get_partitioning,
            ) as mock_get_partitioning, mock.patch.object(
                DummyFastPartitioning,
                "_partition_space_2d",
            ):
                bd.update(Y=Y)
                if m > 2:
                    mock_update_local_upper_bounds_incremental.assert_called_once()
                    # check that it is not called if the pareto set does not change
                    bd.update(Y=Y)
                    mock_update_local_upper_bounds_incremental.assert_called_once()
                    mock_get_partitioning.assert_called_once()
                else:
                    self.assertEqual(
                        len(mock_update_local_upper_bounds_incremental.call_args_list),
                        0,
                    )

            # test exception is raised for m=2, batched box decomposition using
            # _partition_space
            if m == 2:
                with self.assertRaises(NotImplementedError):
                    DummyFastPartitioning(ref_point=ref_point, Y=Y.unsqueeze(0))

    def test_nan_values(self) -> None:
        Y = torch.rand(10, 2)
        Y[8:, 1] = float("nan")
        ref_pt = torch.rand(2)
        # On init.
        with self.assertRaisesRegex(ValueError, "with 2 NaN values"):
            DummyBoxDecomposition(ref_point=ref_pt, sort=True, Y=Y)
        # On update.
        bd = DummyBoxDecomposition(ref_point=ref_pt, sort=True)
        with self.assertRaisesRegex(ValueError, "with 2 NaN values"):
            bd.update(Y=Y)


class TestBoxDecomposition_no_set_up(BotorchTestCase):
    def helper_hypervolume(self, Box_Decomp_cls: type) -> None:
        """
        This test should be run for each non-abstract subclass of `BoxDecomposition`.
        """
        # batching
        n_outcomes, batch_dim, n = 2, 3, 4

        ref_point = torch.zeros(n_outcomes)
        Y = torch.ones(batch_dim, n, n_outcomes)

        box_decomp = Box_Decomp_cls(ref_point=ref_point, Y=Y)
        hv = box_decomp.compute_hypervolume()
        self.assertEqual(hv.shape, (batch_dim,))
        self.assertAllClose(hv, torch.ones(batch_dim))

        # no batching
        Y = torch.ones(n, n_outcomes)

        box_decomp = Box_Decomp_cls(ref_point=ref_point, Y=Y)
        hv = box_decomp.compute_hypervolume()
        self.assertEqual(hv.shape, ())
        self.assertAllClose(hv, torch.tensor(1.0))

        # cases where there is nothing in Y, either because n=0 or Y is None
        n = 0
        Y_and_expected_shape = [
            (torch.ones(batch_dim, n, n_outcomes), (batch_dim,)),
            (torch.ones(n, n_outcomes), ()),
            (None, ()),
        ]
        for Y, expected_shape in Y_and_expected_shape:
            box_decomp = Box_Decomp_cls(ref_point=ref_point, Y=Y)
            hv = box_decomp.compute_hypervolume()
            self.assertEqual(hv.shape, expected_shape)
            self.assertAllClose(hv, torch.zeros(expected_shape))

    def test_hypervolume(self) -> None:
        for cl in [
            NondominatedPartitioning,
            DominatedPartitioning,
            FastNondominatedPartitioning,
        ]:
            self.helper_hypervolume(cl)

    def test_uninitialized_y(self) -> None:
        ref_point = torch.zeros(2)
        box_decomp = NondominatedPartitioning(ref_point=ref_point)
        with self.assertRaises(BotorchError):
            box_decomp.Y
        with self.assertRaises(BotorchError):
            box_decomp._compute_pareto_Y()
