#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.test_functions.sensitivity_analysis import Gsobol, Ishigami, Morris
from botorch.utils.testing import BotorchTestCase


class TestIshigami(BotorchTestCase):
    def testFunction(self):
        with self.assertRaises(ValueError):
            Ishigami(b=0.33)
        f = Ishigami(b=0.1)
        self.assertEqual(f.b, 0.1)
        f = Ishigami(b=0.05)
        self.assertEqual(f.b, 0.05)
        X = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        m1, m2, m3 = f.compute_dgsm(X)
        for m in [m1, m2, m3]:
            self.assertEqual(len(m), 3)
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([5.8401, 7.4245])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value

    def test_dgsm(self):
        for b in (0.1, 0.05):
            f = Ishigami(b=b)
            X = torch.rand(10, 3, dtype=torch.double) * 2 * math.pi - math.pi
            X.requires_grad_(True)
            (grad,) = torch.autograd.grad(f.evaluate_true(X).sum(), X)
            mean, abs_mean, sq_mean = f.compute_dgsm(X.detach())
            self.assertAllClose(torch.tensor(mean).to(grad), grad.mean(dim=0))
            self.assertAllClose(torch.tensor(abs_mean).to(grad), grad.abs().mean(dim=0))
            self.assertAllClose(torch.tensor(sq_mean).to(grad), grad.pow(2).mean(dim=0))
            # Exact values for x_3: E|df/dx_3| = 2 b pi^2 and
            # E[(df/dx_3)^2] = 8 b^2 pi^6 / 7 for x ~ U[-pi, pi]^3.
            self.assertAlmostEqual(
                f.dgsm_gradient_abs[2], 2 * b * math.pi**2, delta=5e-3
            )
            self.assertAlmostEqual(
                f.dgsm_gradient_square[2], 8 * b**2 * math.pi**6 / 7, delta=2e-2
            )


class TestGsobol(BotorchTestCase):
    def testFunction(self):
        for dim in [6, 8, 15]:
            f = Gsobol(dim=dim)
            self.assertIsNotNone(f.a)
            self.assertEqual(len(f.a), dim)
        f = Gsobol(dim=3, a=[1, 2, 3])
        self.assertEqual(f.a, [1, 2, 3])
        X = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) * 0.5
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([0.25, 2.5])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value


class TestMorris(BotorchTestCase):
    def testFunction(self):
        f = Morris()
        X = torch.stack((torch.zeros(20), torch.ones(20)))
        Z = f.evaluate_true(X)
        Ztrue = torch.tensor([-327.0, -127.0])
        self.assertAllClose(Z, Ztrue, atol=1e-3)
        # Inputs for which w_i = 0, except for w_i = 1 at the given indices.
        X = torch.full((3, 20), 0.5)
        X[:, [2, 4, 6]] = 1 / 12
        X[1, [0, 6]] = 1.0  # 20 + 20 + beta_{1,7} with beta_{1,7} = +1
        X[2, [0, 1, 5]] = 1.0  # 3 * 20 - 3 * 15 with beta_{1,2,6} = 0
        self.assertAllClose(f.evaluate_true(X), torch.tensor([0.0, 41.0, 15.0]))
        self.assertIsNone(f._optimizers)
        with self.assertRaises(NotImplementedError):
            f.optimal_value
