#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.utils.testing import BotorchTestCase


class TestStoppingCriterion(BotorchTestCase):
    def test_exponential_moving_average(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}

            # test max iter
            sc = ExpMAStoppingCriterion(maxiter=2)
            self.assertEqual(sc.maxiter, 2)
            self.assertEqual(sc.n_window, 10)
            self.assertEqual(sc.rel_tol, 1e-5)
            self.assertFalse(sc(fvals=torch.ones(1, **tkwargs)))
            self.assertTrue(sc(fvals=torch.zeros(1, **tkwargs)))

            # test convergence
            n_window = 4
            for minimize in (True, False):
                # test basic
                sc = ExpMAStoppingCriterion(
                    minimize=minimize, n_window=n_window, rel_tol=0.0375
                )
                self.assertEqual(sc.rel_tol, 0.0375)
                self.assertIsNone(sc._prev_fvals)
                weights_exp = torch.tensor([0.1416, 0.1976, 0.2758, 0.3849])
                self.assertAllClose(sc.weights, weights_exp, atol=1e-4)
                f_vals = 1 + torch.linspace(1, 0, 25, **tkwargs) ** 2
                if not minimize:
                    f_vals = -f_vals
                for i, fval in enumerate(f_vals):
                    if sc(fval):
                        self.assertEqual(i, 10)
                        break
                # test multiple components
                sc = ExpMAStoppingCriterion(
                    minimize=minimize, n_window=n_window, rel_tol=0.0375
                )
                df = torch.linspace(0, 0.1, 25, **tkwargs)
                if not minimize:
                    df = -df
                f_vals = torch.stack([f_vals, f_vals + df], dim=-1)
                for i, fval in enumerate(f_vals):
                    if sc(fval):
                        self.assertEqual(i, 10)
                        break

                # Test reset functionality - verify state after use, reset, and reuse
                self.assertGreater(sc.iter, 0)
                self.assertIsNotNone(sc._prev_fvals)
                sc.reset()
                self.assertEqual(sc.iter, 0)
                self.assertIsNone(sc._prev_fvals)
                # Verify criterion works after reset
                self.assertFalse(sc(f_vals[0]))
                self.assertEqual(sc.iter, 1)
