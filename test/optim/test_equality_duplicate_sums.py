# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.optim.parameter_constraints import project_to_equality_constraints
from botorch.utils.testing import BotorchTestCase


class TestEqualityDuplicateSums(BotorchTestCase):
    def test_repeated_indices_follow_declared_sum_and_have_complete_derivatives(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for scalar_coefficient in (False, True):
                with self.subTest(device=device, scalar_coefficient=scalar_coefficient):
                    X = torch.tensor(
                        [[0.2, 0.3, 0.8], [0.6, 0.4, 0.1]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    coefficients = X.new_tensor(
                        0.7 if scalar_coefficient else [1.2, -0.7, 0.8],
                        requires_grad=True,
                    )
                    rhs = X.new_tensor(0.4, requires_grad=True)
                    # Constraint metadata may remain on CPU for CUDA points.
                    indices = torch.tensor([0, 0, 2])

                    def project(points, coefficient, right):
                        return project_to_equality_constraints(
                            points, [(indices, coefficient, right)]
                        )

                    # The tuple declares c0*x0 + c1*x0 + c2*x2 = rhs.
                    # This independent one-hyperplane formula does not use the
                    # implementation's constraint assembly or matrix solve.
                    row = (
                        torch.stack((2 * coefficients, X.new_zeros(()), coefficients))
                        if scalar_coefficient
                        else torch.stack(
                            (
                                coefficients[0] + coefficients[1],
                                X.new_zeros(()),
                                coefficients[2],
                            )
                        )
                    )
                    expected = (
                        X
                        + (rhs - (X * row).sum(-1, keepdim=True))
                        * row
                        / row.square().sum()
                    )
                    actual = project(X, coefficients, rhs)
                    self.assertAllClose(actual, expected, atol=1e-13, rtol=1e-13)
                    declared_lhs = (actual[..., indices] * coefficients).sum(-1)
                    self.assertAllClose(declared_lhs, rhs.expand_as(declared_lhs))
                    expected_gradients = torch.autograd.grad(
                        expected.square().sum(), (X, coefficients, rhs)
                    )
                    gradients = torch.autograd.grad(
                        actual.square().sum(), (X, coefficients, rhs)
                    )
                    for actual_g, expected_g in zip(gradients, expected_gradients):
                        self.assertAllClose(
                            actual_g, expected_g, atol=1e-12, rtol=1e-12
                        )
                    # Check every input and coefficient/RHS partial, followed
                    # by mixed second derivatives of the complete operation.
                    self.assertTrue(
                        torch.autograd.gradcheck(project, (X, coefficients, rhs))
                    )
                    self.assertTrue(
                        torch.autograd.gradgradcheck(
                            project, (X, coefficients, rhs), fast_mode=True
                        )
                    )
