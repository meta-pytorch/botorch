#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils import apply_constraints, get_objective_weights_transform
from botorch.utils.objective import (
    compute_feasibility_indicator,
    compute_smoothed_feasibility_indicator,
)
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


def ones_f(samples: Tensor) -> Tensor:
    return torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def zeros_f(samples: Tensor) -> Tensor:
    return torch.zeros(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def nonzeros_f(samples: Tensor) -> Tensor:
    t = torch.zeros(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    t[:] = 0.1
    return t


def minus_one_f(samples: Tensor) -> Tensor:
    return -(
        torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    )


class TestApplyConstraints(BotorchTestCase):
    def test_apply_constraints(self):
        # nonnegative objective, one constraint
        samples = torch.randn(1)
        obj = ones_f(samples)
        obj = apply_constraints(
            obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=0.0
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5))
        # nonnegative objective, two constraint
        samples = torch.randn(1)
        obj = ones_f(samples)
        obj = apply_constraints(
            obj=obj,
            constraints=[zeros_f, zeros_f],
            samples=samples,
            infeasible_cost=0.0,
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5 * 0.5))
        # negative objective, one constraint, infeasible_cost
        samples = torch.randn(1)
        obj = minus_one_f(samples)
        obj = apply_constraints(
            obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=2.0
        )
        self.assertTrue(torch.equal(obj, ones_f(samples) * 0.5 - 2.0))

        # nonnegative objective, one constraint, eta = 0
        samples = torch.randn(1)
        obj = ones_f(samples)
        with self.assertRaisesRegex(ValueError, "eta must be positive."):
            apply_constraints(
                obj=obj,
                constraints=[zeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=0.0,
            )

    def test_apply_constraints_multi_output(self):
        # nonnegative objective, one constraint
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            samples = torch.rand(3, 2, **tkwargs)
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=0.0
            )
            self.assertTrue(torch.equal(obj, samples * 0.5))
            # nonnegative objective, two constraint
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f, zeros_f],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj, samples * 0.5 * 0.5))
            # nonnegative objective, two constraint explicit eta
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f, zeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([10e-3, 10e-3]).to(**tkwargs),
            )
            self.assertTrue(torch.equal(obj, samples * 0.5 * 0.5))
            # nonnegative objective, two constraint explicit different eta
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[nonzeros_f, nonzeros_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([10e-1, 10e-2]).to(**tkwargs),
            )
            self.assertTrue(
                torch.allclose(
                    obj,
                    samples
                    * torch.sigmoid(torch.as_tensor(-0.1) / 10e-1)
                    * torch.sigmoid(torch.as_tensor(-0.1) / 10e-2),
                )
            )
            # nonnegative objective, two constraint explicit different eta
            # use ones_f
            obj = samples.clone()
            obj = apply_constraints(
                obj=obj,
                constraints=[ones_f, ones_f],
                samples=samples,
                infeasible_cost=0.0,
                eta=torch.tensor([1, 10]).to(**tkwargs),
            )
            self.assertTrue(
                torch.allclose(
                    obj,
                    samples
                    * torch.sigmoid(torch.as_tensor(-1.0) / 1.0)
                    * torch.sigmoid(torch.as_tensor(-1.0) / 10.0),
                )
            )
            # negative objective, one constraint, infeasible_cost
            obj = samples.clone().clamp_min(-1.0)
            obj = apply_constraints(
                obj=obj, constraints=[zeros_f], samples=samples, infeasible_cost=2.0
            )
            self.assertAllClose(obj, samples.clamp_min(-1.0) * 0.5 - 1.0)
            # negative objective, one constraint, infeasible_cost, explicit eta
            obj = samples.clone().clamp_min(-1.0)
            obj = apply_constraints(
                obj=obj,
                constraints=[zeros_f],
                samples=samples,
                infeasible_cost=2.0,
                eta=torch.tensor([10e-3]).to(**tkwargs),
            )
            self.assertAllClose(obj, samples.clamp_min(-1.0) * 0.5 - 1.0)
            # nonnegative objective, one constraint, eta = 0
            obj = samples
            with self.assertRaisesRegex(ValueError, "eta must be positive"):
                apply_constraints(
                    obj=obj,
                    constraints=[zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=0.0,
                )

    def test_apply_constraints_wrong_eta_dim(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            samples = torch.rand(3, 2, **tkwargs)
            obj = samples.clone()
            with self.assertRaisesRegex(ValueError, "Number of provided constraints"):
                obj = apply_constraints(
                    obj=obj,
                    constraints=[zeros_f, zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=torch.tensor([0.1]).to(**tkwargs),
                )
            with self.assertRaisesRegex(ValueError, "Number of provided constraints"):
                obj = apply_constraints(
                    obj=obj,
                    constraints=[zeros_f, zeros_f],
                    samples=samples,
                    infeasible_cost=0.0,
                    eta=torch.tensor([0.1, 0.1, 0.3]).to(**tkwargs),
                )

    def test_constraint_indicators(self):
        # nonnegative objective, one constraint
        samples = torch.randn(1)
        ind = compute_feasibility_indicator(constraints=[zeros_f], samples=samples)
        self.assertAllClose(ind, torch.ones_like(ind))
        self.assertEqual(ind.dtype, torch.bool)

        smoothed_ind = compute_smoothed_feasibility_indicator(
            constraints=[zeros_f], samples=samples, eta=1e-3
        )
        self.assertAllClose(smoothed_ind, ones_f(samples) / 2)

        # two constraints
        samples = torch.randn(1)
        smoothed_ind = compute_smoothed_feasibility_indicator(
            constraints=[zeros_f, zeros_f],
            samples=samples,
            eta=1e-3,
        )
        self.assertAllClose(smoothed_ind, ones_f(samples) * 0.5 * 0.5)

        # test it without fatmoid/sigmoid transformation
        smoothed_ind = compute_smoothed_feasibility_indicator(
            constraints=[zeros_f, zeros_f],
            samples=samples,
            eta=1e-3,
            fat=[False, None],
        )
        self.assertAllClose(smoothed_ind, ones_f(samples) * 0.5 * 0)

        # feasible
        samples = torch.randn(1)
        ind = compute_feasibility_indicator(
            constraints=[minus_one_f],
            samples=samples,
        )
        self.assertAllClose(ind, torch.ones_like(ind))

        smoothed_ind = compute_smoothed_feasibility_indicator(
            constraints=[minus_one_f], samples=samples, eta=1e-3
        )
        self.assertTrue((smoothed_ind > 3 / 4).all())

        with self.assertRaisesRegex(ValueError, "Number of provided constraints"):
            compute_smoothed_feasibility_indicator(
                constraints=[zeros_f, zeros_f],
                samples=samples,
                eta=torch.tensor([0.1], device=self.device),
            )
        with self.assertRaisesRegex(ValueError, "Number of provided constraints"):
            compute_smoothed_feasibility_indicator(
                constraints=[zeros_f, zeros_f],
                samples=samples,
                eta=torch.tensor([0.1, 0.1], device=self.device),
                fat=[True],
            )

        # test marginalize_dim
        samples = torch.randn(1, 2, 1, 1)
        ind = compute_feasibility_indicator(
            constraints=[zeros_f], samples=samples, marginalize_dim=-3
        )
        self.assertAllClose(ind, torch.ones_like(ind))
        self.assertTrue(ind.shape == torch.Size([1, 1]))
        self.assertEqual(ind.dtype, torch.bool)

    def test_feasibility_indicator_probability_constraints(self):
        # the last output is a probability of feasibility
        samples = torch.tensor(
            [[1.0, 0.2], [-1.0, 0.5], [1.0, 0.9]], device=self.device
        )

        def prob_f(Y):
            return Y[..., 1]

        def neg_f(Y):
            return Y[..., 0]

        # probabilities are positive, so without ``fat=None`` nothing is feasible
        ind = compute_feasibility_indicator(constraints=[prob_f], samples=samples)
        self.assertFalse(ind.any())
        # with ``fat=None``, a probability of at least 0.5 is feasible
        ind = compute_feasibility_indicator(
            constraints=[prob_f], samples=samples, fat=[None]
        )
        self.assertEqual(ind.tolist(), [False, True, True])
        # mixed with a regular constraint
        ind = compute_feasibility_indicator(
            constraints=[prob_f, neg_f], samples=samples, fat=[None, False]
        )
        self.assertEqual(ind.tolist(), [False, True, False])
        # a boolean ``fat`` applies to all constraints and changes nothing here
        ind = compute_feasibility_indicator(
            constraints=[neg_f], samples=samples, fat=True
        )
        self.assertEqual(ind.tolist(), [False, True, False])
        with self.assertRaisesRegex(ValueError, "Number of provided constraints"):
            compute_feasibility_indicator(
                constraints=[prob_f, neg_f], samples=samples, fat=[None]
            )


class TestGetObjectiveWeightsTransform(BotorchTestCase):
    def test_NoWeights(self) -> None:
        Y = torch.ones(5, 2, 4, 1)
        objective_transform = get_objective_weights_transform(None)
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(Y.squeeze(-1), Y_transformed))
        Y_transformed_X_None = objective_transform(Y, X=None)
        self.assertTrue(torch.equal(Y.squeeze(-1), Y_transformed_X_None))

    def test_OneWeightBroadcasting(self) -> None:
        Y = torch.ones(5, 2, 4, 1)
        objective_transform = get_objective_weights_transform(torch.tensor([0.5]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(0.5 * Y.sum(dim=-1), Y_transformed))
        Y_transformed_X_None = objective_transform(Y, X=None)
        self.assertTrue(torch.equal(0.5 * Y.sum(dim=-1), Y_transformed_X_None))

    def test_IncompatibleNumberOfWeights(self) -> None:
        Y = torch.ones(5, 2, 4, 3)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 2.0]))
        with self.assertRaises(RuntimeError):
            objective_transform(Y)

    def test_MultiTaskWeights(self):
        Y = torch.ones(5, 2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))

    def test_NoMCSamples(self):
        Y = torch.ones(2, 4, 2)
        objective_transform = get_objective_weights_transform(torch.tensor([1.0, 1.0]))
        Y_transformed = objective_transform(Y)
        self.assertTrue(torch.equal(torch.sum(Y, dim=-1), Y_transformed))
