#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean
from botorch.models.deterministic import GenericDeterministicModel
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.testing import BotorchTestCase, skip_if_import_error


class TestOptimizeWithNSGAII(BotorchTestCase):
    @skip_if_import_error
    def test_optimize_with_nsgaii(self) -> None:
        from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            dim = 6
            num_objectives = 2
            prob = DTLZ2(dim=dim, num_objectives=num_objectives, negate=True).to(
                **tkwargs
            )

            model = GenericDeterministicModel(f=prob, num_outputs=2)
            acqf = MultiOutputPosteriorMean(model=model)
            bounds = torch.zeros(2, dim, **tkwargs)
            bounds[1] = 1
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                q=5,
                num_objectives=num_objectives,
                max_gen=4,
            )
            # Since duplicates are eliminated and only pareto optimal points
            # are returned, the pareto set should be <= 5.
            self.assertLessEqual(pareto_X.shape[0], 5)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertLessEqual(pareto_Y.shape[0], 5)
            self.assertEqual(pareto_Y.shape[1], num_objectives)
            self.assertTrue(torch.equal(prob(pareto_X), pareto_Y))
            # test with ref_point
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                q=5,
                num_objectives=num_objectives,
                ref_point=prob.ref_point,
                max_gen=2,
            )
            self.assertLessEqual(pareto_X.shape[0], 5)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertTrue(torch.equal(prob(pareto_X), pareto_Y))
            self.assertLessEqual(pareto_Y.shape[0], 5)
            self.assertEqual(pareto_Y.shape[1], num_objectives)
            self.assertTrue((pareto_Y >= prob.ref_point).all())
            # test with objective
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                q=5,
                num_objectives=num_objectives,
                objective=WeightedMCMultiOutputObjective(
                    weights=-torch.ones(num_objectives, **tkwargs)
                ),
                max_gen=2,
            )
            self.assertLessEqual(pareto_X.shape[0], 5)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertTrue(torch.equal(prob(pareto_X), -pareto_Y))
            self.assertLessEqual(pareto_Y.shape[0], 5)
            self.assertEqual(pareto_Y.shape[1], num_objectives)
            self.assertTrue((pareto_Y >= 0.0).all())

            # test with constraints
            def constraint(Y):
                # first objective should be >= -0.5
                return -0.5 - Y[..., 0]

            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                q=5,
                num_objectives=num_objectives,
                constraints=[constraint],
                max_gen=2,
            )
            self.assertLessEqual(pareto_X.shape[0], 5)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertTrue(torch.equal(prob(pareto_X), pareto_Y))
            self.assertLessEqual(pareto_Y.shape[0], 5)
            self.assertEqual(pareto_Y.shape[1], num_objectives)
            self.assertTrue((pareto_Y[:, 0] >= -0.5).all())

            # test with ref point and constraints
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                q=5,
                num_objectives=num_objectives,
                constraints=[constraint],
                max_gen=2,
                ref_point=prob.ref_point,
            )
            self.assertLessEqual(pareto_X.shape[0], 5)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertTrue(torch.equal(prob(pareto_X), pareto_Y))
            self.assertLessEqual(pareto_Y.shape[0], 5)
            self.assertEqual(pareto_Y.shape[1], num_objectives)
            # the constraint is tighter than the ref point
            # on objective 0
            self.assertTrue((pareto_Y[:, 0] >= -0.5).all())
            self.assertTrue((pareto_Y[:, 1] >= prob.ref_point[1]).all())

            # test without q
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                num_objectives=num_objectives,
                constraints=[constraint],
                max_gen=2,
                ref_point=prob.ref_point,
            )
            self.assertLessEqual(pareto_X.shape[0], 250)
            self.assertEqual(pareto_X.shape[1], dim)
            self.assertTrue(torch.equal(prob(pareto_X), pareto_Y))
            self.assertLessEqual(pareto_Y.shape[0], 250)
            self.assertEqual(pareto_Y.shape[1], num_objectives)

            # test with fixed features
            pareto_X, pareto_Y = optimize_with_nsgaii(
                acq_function=acqf,
                bounds=bounds,
                num_objectives=num_objectives,
                constraints=[constraint],
                max_gen=2,
                ref_point=prob.ref_point,
                fixed_features={5: 0.5},
            )
            self.assertTrue(torch.all(pareto_X[:, 5] == 0.5))

            # test where minimize returns fewer than q pareto optimal points
            # but at least q points overall
            X = torch.rand(3, 6, dtype=dtype, device=self.device)
            F = torch.tensor(
                [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]], dtype=dtype, device=self.device
            )
            with patch(
                "botorch.utils.multi_objective.optimize.minimize",
                return_value=Mock(X=X.cpu().numpy(), F=F.cpu().numpy()),
            ):
                pareto_X, pareto_Y = optimize_with_nsgaii(
                    acq_function=acqf,
                    bounds=bounds,
                    num_objectives=num_objectives,
                    max_gen=2,
                    q=3,
                )
                self.assertTrue(torch.equal(pareto_X, X))
                self.assertTrue(torch.equal(pareto_Y, -F))
