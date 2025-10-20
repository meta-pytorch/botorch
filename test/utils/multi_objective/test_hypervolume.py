#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.utils.multi_objective.hypervolume import (
    get_hypervolume_maximizing_subset,
    Hypervolume,
    infer_reference_point,
)
from botorch.utils.testing import BotorchTestCase

EPS = 1e-4

pareto_Y_5d = [
    [
        -0.42890000759972685,
        -0.1446377658556118,
        -0.10335085850913295,
        -0.49502106785623134,
        -0.7344368200145969,
    ],
    [
        -0.5124511265981003,
        -0.5332028064973291,
        -0.36775794432917486,
        -0.5261970836251835,
        -0.20238412378158688,
    ],
    [
        -0.5960106882406603,
        -0.32491865590163566,
        -0.5815435820797972,
        -0.08375675085018466,
        -0.44044408882261904,
    ],
    [
        -0.6135323874039154,
        -0.5658986040644925,
        -0.39684098121151284,
        -0.3798488823307603,
        -0.03960860698719982,
    ],
    [
        -0.3957157311550265,
        -0.4045394517331393,
        -0.07282417302694655,
        -0.5699496614967537,
        -0.5912790502720109,
    ],
    [
        -0.06392539039575441,
        -0.17204800894814581,
        -0.6620860391018546,
        -0.7241037454151875,
        -0.06024010111083461,
    ],
]


class TestHypervolume(BotorchTestCase):
    def test_hypervolume(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype

            ref_point = torch.tensor([0.0, 0.0], **tkwargs)
            hv = Hypervolume(ref_point)

            # test ref point
            self.assertTrue(torch.equal(ref_point, hv.ref_point))
            self.assertTrue(torch.equal(-ref_point, hv._ref_point))

            # test dimension errors
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(2, **tkwargs))
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(1, 1, 2, **tkwargs))
            with self.assertRaises(BotorchTensorDimensionError):
                hv.compute(pareto_Y=torch.empty(1, 3, **tkwargs))

            # test no pareto points
            pareto_Y = (ref_point - 1).view(1, 2)
            volume = hv.compute(pareto_Y)
            self.assertEqual(volume, 0.0)

            # test 1-d
            hv = Hypervolume(ref_point[:1])
            volume = hv.compute(pareto_Y=torch.ones(1, 1, **tkwargs))
            self.assertEqual(volume, 1.0)

            # test m=2
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]], **tkwargs
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 37.75) < EPS)

            # test nonzero reference point
            ref_point = torch.tensor([1.0, 0.5], **tkwargs)
            hv = Hypervolume(ref_point)
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 28.75) < EPS)

            # test m=3
            # ref_point = torch.tensor([-1.1, -1.1, -1.1], **tkwargs)
            ref_point = torch.tensor([-2.1, -2.5, -2.3], **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], **tkwargs
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 11.075) < EPS)
            # self.assertTrue(abs(volume - 0.45980908291719647) < EPS)

            # test m=4
            ref_point = torch.tensor([-2.1, -2.5, -2.3, -2.0], **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ],
                **tkwargs,
            )
            volume = hv.compute(pareto_Y=pareto_Y)
            self.assertTrue(abs(volume - 23.15) < EPS)

            # test m=5
            # this pareto front is from DTLZ2 and covers several edge cases
            ref_point = torch.full(torch.Size([5]), -1.1, **tkwargs)
            hv = Hypervolume(ref_point)
            pareto_Y = torch.tensor(pareto_Y_5d, **tkwargs)
            volume = hv.compute(pareto_Y)
            self.assertTrue(abs(volume - 0.42127855991587) < EPS)


class TestGetReferencePoint(BotorchTestCase):
    def test_infer_reference_point(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            Y = torch.tensor(
                [
                    [-13.9599, -24.0326],
                    [-19.6755, -11.4721],
                    [-18.7742, -11.9193],
                    [-16.6614, -12.3283],
                    [-17.7663, -11.9941],
                    [-17.4367, -12.2948],
                    [-19.4244, -11.9158],
                    [-14.0806, -22.0004],
                ],
                **tkwargs,
            )

            # test empty pareto_Y and no max_ref_point
            with self.assertRaises(BotorchError):
                infer_reference_point(pareto_Y=Y[:0])

            # test max_ref_point does not change when there exists a better Y point
            max_ref_point = Y.min(dim=0).values
            ref_point = infer_reference_point(max_ref_point=max_ref_point, pareto_Y=Y)
            self.assertTrue(torch.equal(max_ref_point, ref_point))
            # test scale_max_ref_point
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y, scale_max_ref_point=True
            )
            better_than_ref = (Y > max_ref_point).all(dim=-1)
            Y_better_than_ref = Y[better_than_ref]
            ideal_better_than_ref = Y_better_than_ref.max(dim=0).values
            self.assertTrue(
                torch.equal(
                    max_ref_point - 0.1 * (ideal_better_than_ref - max_ref_point),
                    ref_point,
                )
            )
            # test case when there does not exist a better Y point
            max_ref_point = torch.tensor([-2.2, -2.3], **tkwargs)
            ref_point = infer_reference_point(max_ref_point=max_ref_point, pareto_Y=Y)
            self.assertTrue((ref_point < Y).all(dim=-1).any())
            nadir = Y.min(dim=0).values
            ideal = Y.max(dim=0).values
            expected_ref_point = nadir - 0.1 * (ideal - nadir)
            self.assertAllClose(ref_point, expected_ref_point)
            # test with scale
            expected_ref_point = nadir - 0.2 * (ideal - nadir)
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y, scale=0.2
            )
            self.assertAllClose(ref_point, expected_ref_point)

            # test case when one objective is better than max_ref_point, and
            # one objective is worse
            max_ref_point = torch.tensor([-2.2, -12.1], **tkwargs)
            expected_ref_point = nadir - 0.1 * (ideal - nadir)
            expected_ref_point = torch.min(expected_ref_point, max_ref_point)
            ref_point = infer_reference_point(max_ref_point=max_ref_point, pareto_Y=Y)
            self.assertTrue(torch.equal(expected_ref_point, ref_point))
            # test case when one objective is better than max_ref_point, and
            # one objective is worse with scale_max_ref_point
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y, scale_max_ref_point=True
            )
            nadir2 = torch.min(nadir, max_ref_point)
            expected_ref_point = nadir2 - 0.1 * (ideal - nadir2)
            self.assertTrue(torch.equal(expected_ref_point, ref_point))

            # test case when size of pareto_Y is 0
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y[:0]
            )
            self.assertTrue(torch.equal(max_ref_point, ref_point))
            # test case when size of pareto_Y is 0 with scale_max_ref_point
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point,
                pareto_Y=Y[:0],
                scale_max_ref_point=True,
                scale=0.2,
            )
            self.assertTrue(
                torch.equal(max_ref_point - 0.2 * max_ref_point.abs(), ref_point)
            )
            # test case when size of pareto_Y is 1
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y[:1]
            )
            expected_ref_point = Y[0] - 0.1 * Y[0].abs()
            self.assertTrue(torch.equal(expected_ref_point, ref_point))
            # test case when size of pareto_Y is 1 with scale parameter
            ref_point = infer_reference_point(
                max_ref_point=max_ref_point, pareto_Y=Y[:1], scale=0.2
            )
            expected_ref_point = Y[0] - 0.2 * Y[0].abs()
            self.assertTrue(torch.equal(expected_ref_point, ref_point))

            # test no max_ref_point specified
            expected_ref_point = nadir - 0.2 * (ideal - nadir)
            ref_point = infer_reference_point(pareto_Y=Y, scale=0.2)
            self.assertAllClose(ref_point, expected_ref_point)
            ref_point = infer_reference_point(pareto_Y=Y)
            expected_ref_point = nadir - 0.1 * (ideal - nadir)
            self.assertAllClose(ref_point, expected_ref_point)

            # Test all NaN max_ref_point.
            ref_point = infer_reference_point(
                pareto_Y=Y,
                max_ref_point=torch.tensor([float("nan"), float("nan")], **tkwargs),
            )
            self.assertAllClose(ref_point, expected_ref_point)
            # Test partial NaN, partial worse than nadir.
            expected_ref_point = nadir.clone()
            expected_ref_point[1] = -1e5
            ref_point = infer_reference_point(
                pareto_Y=Y,
                max_ref_point=torch.tensor([float("nan"), -1e5], **tkwargs),
                scale=0.0,
            )
            self.assertAllClose(ref_point, expected_ref_point)
            # Test partial NaN, partial better than nadir.
            expected_ref_point = nadir
            ref_point = infer_reference_point(
                pareto_Y=Y,
                max_ref_point=torch.tensor([float("nan"), 1e5], **tkwargs),
                scale=0.0,
            )
            self.assertAllClose(ref_point, expected_ref_point)
            # Test partial NaN, partial worse than nadir with scale_max_ref_point.
            expected_ref_point[1] = -1e5
            expected_ref_point = expected_ref_point - 0.2 * (ideal - expected_ref_point)
            ref_point = infer_reference_point(
                pareto_Y=Y,
                max_ref_point=torch.tensor([float("nan"), -1e5], **tkwargs),
                scale=0.2,
                scale_max_ref_point=True,
            )
            self.assertAllClose(ref_point, expected_ref_point)
            # Test with single point in Pareto_Y, worse than ref point.
            ref_point = infer_reference_point(
                pareto_Y=Y[:1],
                max_ref_point=torch.tensor([float("nan"), 1e5], **tkwargs),
            )
            expected_ref_point = Y[0] - 0.1 * Y[0].abs()
            self.assertTrue(torch.equal(expected_ref_point, ref_point))
            # Test with single point in Pareto_Y, better than ref point.
            ref_point = infer_reference_point(
                pareto_Y=Y[:1],
                max_ref_point=torch.tensor([float("nan"), -1e5], **tkwargs),
                scale_max_ref_point=True,
            )
            expected_ref_point[1] = -1e5 - 0.1 * Y[0, 1].abs()
            self.assertTrue(torch.equal(expected_ref_point, ref_point))
            # Empty pareto_Y with nan ref point.
            with self.assertRaisesRegex(BotorchError, "ref point includes NaN"):
                ref_point = infer_reference_point(
                    pareto_Y=Y[:0],
                    max_ref_point=torch.tensor([float("nan"), -1e5], **tkwargs),
                )


class TestGetHypervolumeMaximizingSubset(BotorchTestCase):
    def test_get_hypervolume_maximizing_subset(self) -> None:
        # test invalid shapes
        ref_point = torch.torch.zeros(2)
        for invalid_y in (torch.zeros(2), torch.zeros(1, 1, 2)):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"Only two dimensions are supported \(no additional\) batch dims.",
            ):
                get_hypervolume_maximizing_subset(n=1, Y=invalid_y, ref_point=ref_point)
            # test n > Y.shape[0]
            with self.assertRaisesRegex(
                ValueError,
                r"Y has fewer points \(1\) than the requested subset size \(2\).",
            ):
                get_hypervolume_maximizing_subset(
                    n=2, Y=torch.zeros(1, 2), ref_point=ref_point
                )
        for dtype in (torch.float, torch.double):
            Y = torch.tensor(
                [
                    [-13.9599, -24.0326],
                    [-19.6755, -11.4721],
                    [-18.7742, -11.9193],
                    [-16.6614, -12.3283],
                    [-17.7663, -11.9941],
                    [-17.4367, -12.2948],
                    [-19.4244, -11.9158],
                    [-14.0806, -22.0004],
                ],
                dtype=dtype,
                device=self.device,
            )
            ref_point = torch.tensor([-20.0, -20.0], dtype=dtype, device=self.device)

            Y_subset, idcs = get_hypervolume_maximizing_subset(
                n=3, Y=Y, ref_point=ref_point
            )
            self.assertTrue(torch.equal(Y_subset, Y[idcs]))
            self.assertTrue(
                torch.equal(
                    idcs, torch.tensor([3, 4, 1], dtype=torch.long, device=self.device)
                )
            )
            # test without `n` pareto optimal points
            Y = torch.tensor(
                [[-5.0, -5.0], [-10.0, -10.0]],
                dtype=dtype,
                device=self.device,
            )
            Y_subset, idcs = get_hypervolume_maximizing_subset(
                n=2, Y=Y, ref_point=ref_point
            )
            self.assertTrue(torch.equal(Y_subset, Y))
            self.assertTrue(
                torch.equal(
                    idcs, torch.tensor([0, 1], dtype=torch.long, device=self.device)
                )
            )
