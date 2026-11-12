#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.objective import ConstrainedMCObjective, LearnedObjective
from botorch.models import PairwiseGP
from botorch.utils.objective import apply_constraints
from botorch.utils.testing import BotorchTestCase


class TestConstrainedObjectiveComposition(BotorchTestCase):
    def test_input_dependent_objective(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, dtype in itertools.product(devices, (torch.float, torch.double)):
            with self.subTest(device=device, dtype=dtype):
                samples = torch.rand(3, 2, 2, 2, device=device, dtype=dtype)
                X = torch.rand(2, 2, 3, device=device, dtype=dtype)

                def objective(Y, X):
                    return Y.sum(-1) + X.square().sum(-1)

                constraints = [lambda Y: Y[..., 0] - 0.5]
                constrained = ConstrainedMCObjective(
                    objective, constraints, eta=0.2, infeasible_cost=2.0
                ).to(samples)
                expected = apply_constraints(
                    objective(samples, X),
                    constraints,
                    samples,
                    infeasible_cost=constrained.infeasible_cost,
                    eta=constrained.eta,
                )
                self.assertAllClose(constrained(samples, X=X), expected)

    def test_augmented_samples_and_gradients(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, repeats in itertools.product(devices, (2, 5)):
            with self.subTest(device=device, repeats=repeats):
                samples = torch.rand(
                    3, 2, 2, 2, device=device, dtype=torch.double, requires_grad=True
                )

                def objective(Y, X):
                    return (Y.square().sum(-1) + 1).repeat(repeats, 1, 1)

                constraints = [lambda Y: Y[..., 0] - 0.5]
                constrained = ConstrainedMCObjective(
                    objective, constraints, eta=0.2, infeasible_cost=2.0
                ).to(samples)
                actual = constrained(samples)
                expected = apply_constraints(
                    objective(samples, None),
                    constraints,
                    samples.repeat(repeats, 1, 1, 1),
                    infeasible_cost=constrained.infeasible_cost,
                    eta=constrained.eta,
                )
                self.assertEqual(actual.shape, (repeats * 3, 2, 2))
                self.assertAllClose(actual, expected)
                gradients = [
                    torch.autograd.grad(value.sum(), samples, create_graph=True)[0]
                    for value in (actual, expected)
                ]
                self.assertAllClose(*gradients)
                self.assertAllClose(
                    *[
                        torch.autograd.grad(gradient.square().sum(), samples)[0]
                        for gradient in gradients
                    ]
                )

    def test_pairwise_learned_objective(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                points = torch.rand(6, 2, device=device, dtype=torch.double)
                preference = PairwiseGP(
                    points, torch.tensor([[0, 1], [2, 3], [4, 5]], device=device)
                )
                learned = LearnedObjective(
                    preference, sample_shape=torch.Size([4]), seed=4
                )
                samples = torch.rand(3, 2, 2, 2, device=device, dtype=torch.double)
                constraints = [lambda Y: Y[..., 0] - 0.5]
                constrained = ConstrainedMCObjective(
                    learned, constraints, eta=0.2, infeasible_cost=2.0
                ).to(samples)
                expected = apply_constraints(
                    learned(samples),
                    constraints,
                    samples.repeat(4, 1, 1, 1),
                    infeasible_cost=constrained.infeasible_cost,
                    eta=constrained.eta,
                )
                self.assertAllClose(constrained(samples), expected)
