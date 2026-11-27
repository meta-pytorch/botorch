#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputExpectation,
)
from botorch.acquisition.multi_objective.objective import (
    FeasibilityWeightedMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestMCMultiOutputObjective(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCMultiOutputObjective()


class TestIdentityMCMultiOutputObjective(BotorchTestCase):
    def test_identity_mc_multi_output_objective(self):
        objective = IdentityMCMultiOutputObjective()
        with self.assertRaises(BotorchTensorDimensionError):
            IdentityMCMultiOutputObjective(outcomes=[0])
        # Test negative outcome without specifying num_outcomes.
        with self.assertRaises(BotorchError):
            IdentityMCMultiOutputObjective(outcomes=[0, -1])
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        ):
            samples = torch.rand(*batch_shape, 2, m, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(objective(samples), samples))
        # Test negative outcome with num_outcomes.
        objective = IdentityMCMultiOutputObjective(outcomes=[0, -1], num_outcomes=3)
        self.assertEqual(objective.outcomes.tolist(), [0, 2])


class TestWeightedMCMultiOutputObjective(BotorchTestCase):
    def test_weighted_mc_multi_output_objective(self):
        with self.assertRaises(BotorchTensorDimensionError):
            WeightedMCMultiOutputObjective(weights=torch.rand(3, 1))
        with self.assertRaises(BotorchTensorDimensionError):
            WeightedMCMultiOutputObjective(
                weights=torch.rand(3), outcomes=[0, 1], num_outcomes=3
            )
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (2, 3), (torch.float, torch.double)
        ):
            weights = torch.rand(m, device=self.device, dtype=dtype)
            objective = WeightedMCMultiOutputObjective(weights=weights)
            samples = torch.rand(*batch_shape, 2, m, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(objective(samples), samples * weights))


class TestFeasibilityWeightedMCMultiOutputObjective(BotorchTestCase):
    def test_feasibility_weighted_mc_multi_output_objective(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            X = torch.zeros(5, 1, **tkwargs)
            # The infeasible cost will be 0.0.
            means = torch.tensor(
                [
                    [1.0, 0.5],
                    [2.0, -1.0],
                    [3.0, -0.5],
                    [4.0, 1.0],
                    [5.0, 1.0],
                ],
                **tkwargs,
            )
            variances = torch.zeros(5, 2, **tkwargs)
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[-1],
                objective=None,
            )
            feas_samples = feas_obj(means)
            expected = torch.tensor([[1.0], [0.0], [0.0], [4.0], [5.0]], **tkwargs)
            self.assertTrue(torch.allclose(feas_samples, expected))
            self.assertTrue(feas_obj._verify_output_shape)

            # With an objective.
            preprocessing_function = WeightedMCMultiOutputObjective(
                weights=torch.tensor([2.0])
            )
            dummy_obj = MultiOutputExpectation(
                n_w=1, preprocessing_function=preprocessing_function
            )
            dummy_obj._verify_output_shape = False  # for testing
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[1],
                objective=dummy_obj,
            )
            feas_samples = feas_obj(means)
            self.assertTrue(torch.allclose(feas_samples, expected * 2.0))
            self.assertFalse(feas_obj._verify_output_shape)

            # No constraints.
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[],
                objective=None,
            )
            feas_samples = feas_obj(means)
            self.assertIs(feas_samples, means)

            # With a single-output objective.
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=X,
                constraint_idcs=[1],
                objective=IdentityMCObjective(),
            )
            feas_samples = feas_obj(means)
            self.assertTrue(torch.allclose(feas_samples, expected.squeeze(-1)))

            # Error with duplicate idcs.
            with self.assertRaisesRegex(ValueError, "duplicate"):
                FeasibilityWeightedMCMultiOutputObjective(
                    model=mm,
                    X_baseline=X,
                    constraint_idcs=[1, -1],
                )

    def test_feasibility_weighted_multiple_constraints(self):
        # Each constraint must check its own outcome. Regression test for all
        # constraints evaluating the last constraint index.
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            # Outcomes: two objectives, followed by two constraints.
            # The infeasible cost will be 0.0.
            means = torch.tensor([[1.0, 2.0, 1.0, 1.0]], **tkwargs)
            mm = MockModel(MockPosterior(mean=means, variance=torch.zeros_like(means)))
            feas_obj = FeasibilityWeightedMCMultiOutputObjective(
                model=mm,
                X_baseline=torch.zeros(1, 1, **tkwargs),
                constraint_idcs=[2, 3],
            )
            samples = torch.tensor(
                [
                    [1.0, 2.0, 1.0, 1.0],  # Both constraints satisfied.
                    [1.0, 2.0, -1.0, 1.0],  # First constraint violated.
                    [1.0, 2.0, 1.0, -1.0],  # Second constraint violated.
                ],
                **tkwargs,
            )
            expected = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]], **tkwargs)
            self.assertAllClose(feas_obj(samples), expected)
