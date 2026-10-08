#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.exceptions.errors import BotorchError
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestExpectedHypervolumeImprovement(BotorchTestCase):
    def test_expected_hypervolume_improvement(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            ref_point = [0.0, 0.0]
            tkwargs["dtype"] = dtype
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            partitioning = NondominatedPartitioning(
                ref_point=torch.tensor(ref_point, **tkwargs)
            )
            # the event shape is ``b x q x m`` = 1 x 1 x 1
            mean = torch.zeros(1, 1, 2, **tkwargs)
            variance = torch.zeros(1, 1, 2, **tkwargs)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            # test error if there is not pareto_Y initialized in partitioning
            with self.assertRaises(BotorchError):
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point, partitioning=partitioning
                )
            partitioning.update(Y=pareto_Y)
            # test error if ref point has wrong shape
            with self.assertRaises(ValueError):
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point[:1], partitioning=partitioning
                )

            with self.assertRaises(ValueError):
                # test error if no pareto_Y point is better than ref_point
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=[10.0, 10.0], partitioning=partitioning
                )
            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            acqf = ExpectedHypervolumeImprovement(
                model=mm, ref_point=ref_point, partitioning=partitioning
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # check ref point
            self.assertTrue(
                torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs))
            )
            # check bounds
            self.assertTrue(hasattr(acqf, "cell_lower_bounds"))
            self.assertTrue(hasattr(acqf, "cell_upper_bounds"))

    def test_expected_hypervolume_improvement_matches_expansion(self):
        # The EHVI of a hypercell is the product over the outcomes of
        # ``psi_diff + nu``. Compare against the explicit expansion of this product
        # into the sum over all 2^m products of ``psi_diff`` or ``nu`` per outcome.
        torch.manual_seed(0)
        for dtype, m in product((torch.float, torch.double), (2, 3, 4)):
            tkwargs = {"device": self.device, "dtype": dtype}
            # The two forms differ only by floating point rounding.
            tols = (
                {"rtol": 1e-10, "atol": 1e-12}
                if dtype == torch.double
                else {"rtol": 1e-3, "atol": 1e-5}
            )
            ref_point = torch.zeros(m, **tkwargs)
            partitioning = NondominatedPartitioning(
                ref_point=ref_point, Y=torch.rand(5, m, **tkwargs)
            )
            mean = torch.rand(3, 1, m, **tkwargs).requires_grad_(True)
            variance = (0.1 * torch.rand(3, 1, m, **tkwargs)).requires_grad_(True)
            acqf = ExpectedHypervolumeImprovement(
                model=MockModel(MockPosterior(mean=mean, variance=variance)),
                ref_point=ref_point.tolist(),
                partitioning=partitioning,
            )
            res = acqf(torch.zeros(3, 1, 1, **tkwargs))

            sigma = variance.clamp_min(1e-9).sqrt()
            lower = acqf.cell_lower_bounds
            upper = acqf.cell_upper_bounds.clamp_max(
                1e10 if dtype == torch.double else 1e8
            )
            psi_diff = acqf.psi(lower, lower, mean, sigma) - acqf.psi(
                lower, upper, mean, sigma
            )
            factors = torch.stack([psi_diff, acqf.nu(lower, upper, mean, sigma)])
            expected = sum(
                torch.stack([factors[s_k, ..., k] for k, s_k in enumerate(s)])
                .prod(dim=0)
                .sum(dim=-1)
                for s in product((0, 1), repeat=m)
            )
            self.assertGreater(expected.min().item(), 0.0)
            self.assertAllClose(res, expected, **tols)
            for r, e in zip(
                torch.autograd.grad(res.sum(), (mean, variance)),
                torch.autograd.grad(expected.sum(), (mean, variance)),
            ):
                self.assertAllClose(r, e, **tols)
