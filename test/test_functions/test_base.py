#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.test_functions.base import (
    BaseTestProblem,
    ConstrainedBaseTestProblem,
    validate_parameter_indices,
)
from botorch.utils.testing import BotorchTestCase, TestCorruptedProblemsMixin
from botorch.utils.transforms import unnormalize
from torch import Tensor


class DummyTestProblem(BaseTestProblem):
    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 1.0), (2.0, 3.0)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -X.pow(2).sum(dim=-1)


class DummyMixedTestProblem(BaseTestProblem):
    dim = 4
    continuous_inds = [1, 3]
    discrete_inds = [0]
    categorical_inds = [2]
    _bounds = [(0.0, 1.0), (2.0, 3.0), (0.0, 2.0), (3.0, 4.0)]
    _is_minimization_by_default = False

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -X.pow(2).sum(dim=-1)


class DummyConstrainedTestProblem(DummyTestProblem, ConstrainedBaseTestProblem):
    num_constraints = 1

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        return 0.25 - X.sum(dim=-1, keepdim=True)


class TestValidation(BotorchTestCase):
    def test_validation(self):
        for dtype in (torch.float, torch.double):
            lb = torch.zeros(3, device=self.device, dtype=dtype)
            ub = torch.arange(1, 4, device=self.device, dtype=dtype)
            bounds = torch.stack([lb, ub])
            for continuous_inds, categorical_inds, discrete_inds in [
                ([0, 1, 0], [], [2]),
                ([0], [2, 2], [1]),
                ([1], [0], [3]),
            ]:
                with self.assertRaisesRegex(
                    ValueError,
                    "All parameter indices must be a list with unique integers "
                    "between 0 and dim - 1",
                ):
                    validate_parameter_indices(
                        dim=3,
                        bounds=bounds,
                        continuous_inds=continuous_inds,
                        categorical_inds=categorical_inds,
                        discrete_inds=discrete_inds,
                    )
            for continuous_inds, categorical_inds, discrete_inds in [
                ([0, 1], [0], []),
                ([1], [], [2]),
                ([0], [1], [0]),
            ]:
                with self.assertRaisesRegex(
                    ValueError, "All parameter indices must be present"
                ):
                    validate_parameter_indices(
                        dim=3,
                        bounds=bounds,
                        continuous_inds=continuous_inds,
                        categorical_inds=categorical_inds,
                        discrete_inds=discrete_inds,
                    )
            for i in [1, 2]:
                new_bounds = bounds.clone()
                new_bounds[0, i] = 1.1
                with self.assertRaisesRegex(
                    ValueError,
                    "Expected the lower and upper bounds of the discrete and "
                    "categorical parameters to be integer-valued.",
                ):
                    validate_parameter_indices(
                        dim=3,
                        bounds=new_bounds,
                        continuous_inds=[0],
                        categorical_inds=[1],
                        discrete_inds=[2],
                    )
            with self.assertRaisesRegex(
                ValueError, "Expected `bounds` to have shape `2 x d`"
            ):
                validate_parameter_indices(
                    dim=2,
                    bounds=bounds,
                    continuous_inds=[0],
                    categorical_inds=[1],
                    discrete_inds=[],
                )


class TestBaseTestProblems(BotorchTestCase):
    def test_base_test_problem(self):
        for dtype in (torch.float, torch.double):
            problem = DummyTestProblem()
            self.assertIsNone(problem.noise_std)
            self.assertFalse(problem.negate)
            self.assertTrue(problem.is_minimization_problem)
            bnds_expected = torch.tensor([(0, 2), (1, 3)], dtype=torch.float)
            self.assertTrue(torch.equal(problem.bounds, bnds_expected))
            problem = problem.to(device=self.device, dtype=dtype)
            bnds_expected = bnds_expected.to(device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(problem.bounds, bnds_expected))
            X = torch.rand(2, 2, device=self.device, dtype=dtype)
            X = unnormalize(X, bounds=problem.bounds)
            Y = problem(X)
            self.assertAllClose(Y, -X.pow(2).sum(dim=-1))
            problem = DummyTestProblem(negate=True, noise_std=0.1)
            self.assertEqual(problem.noise_std, 0.1)
            self.assertTrue(problem.negate)
            self.assertFalse(problem.is_minimization_problem)

    def test_mixed_base_test_problem(self):
        for dtype in (torch.float, torch.double):
            problem = DummyMixedTestProblem()
            self.assertFalse(problem.is_minimization_problem)
            problem = problem.to(device=self.device, dtype=dtype)
            X = torch.rand(2, 4, device=self.device, dtype=dtype)
            with self.assertRaisesRegex(
                ValueError, "Expected `X` to be within the bounds of the test problem."
            ):
                problem(X)
            X = unnormalize(X, bounds=problem.bounds)
            with self.assertRaisesRegex(
                ValueError,
                "Expected `X` to have integer values for the discrete and "
                "categorical parameters.",
            ):
                problem(X)
            X[:, problem.discrete_inds] = X[:, problem.discrete_inds].round()
            X[:, problem.categorical_inds] = X[:, problem.categorical_inds].round()
            self.assertEqual(problem(X).shape, torch.Size([2]))
            with self.assertRaisesRegex(ValueError, "Expected `X` to have shape"):
                problem(torch.rand(2, 5, device=self.device, dtype=dtype))

    def test_constrained_base_test_problem(self):
        for dtype in (torch.float, torch.double):
            problem = DummyConstrainedTestProblem().to(device=self.device, dtype=dtype)
            problem.bounds = torch.tensor(
                [[0, 0], [1, 1]], device=self.device, dtype=dtype
            )
            X = torch.tensor([[0.4, 0.6], [0.1, 0.1]], device=self.device, dtype=dtype)
            feas = problem.is_feasible(X=X)
            self.assertFalse(feas[0].item())
            self.assertTrue(feas[1].item())
            problem = DummyConstrainedTestProblem(noise_std=0.0).to(
                device=self.device, dtype=dtype
            )
            problem.bounds = torch.tensor(
                [[0, 0], [1, 1]], device=self.device, dtype=dtype
            )
            feas = problem.is_feasible(X=X)
            self.assertFalse(feas[0].item())
            self.assertTrue(feas[1].item())


class TestSeedingMixin(TestCorruptedProblemsMixin):
    def test_seed_iteration(self) -> None:
        problem = self.rosenbrock_problem

        self.assertTrue(problem.has_seeds)
        self.assertIsNone(problem.seed)  # increment_seed needs to be called first
        problem.increment_seed()
        self.assertEqual(problem.seed, 1)
        problem.increment_seed()
        self.assertEqual(problem.seed, 2)
        with self.assertRaises(StopIteration):
            problem.increment_seed()


class TestCorruptedTestProblem(TestCorruptedProblemsMixin):
    def test_basic_rosenbrock(self) -> None:
        problem = self.rosenbrock_problem
        x = torch.rand(5, 2)
        result = problem(x)
        # the outlier_generator sets corruptions to 1
        self.assertTrue((result == 1).all())
