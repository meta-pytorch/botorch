#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.wrapper import AbstractAcquisitionFunctionWrapper
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyWrapper(AbstractAcquisitionFunctionWrapper):
    def forward(self, X):
        return self.acq_func(X)


class TestAbstractAcquisitionFunctionWrapper(BotorchTestCase):
    def test_abstract_acquisition_function_wrapper(self):
        for dtype in (torch.float, torch.double):
            mm = MockModel(
                MockPosterior(
                    mean=torch.rand(1, 1, dtype=dtype, device=self.device),
                    variance=torch.ones(1, 1, dtype=dtype, device=self.device),
                )
            )
            acq_func = ExpectedImprovement(model=mm, best_f=-1.0)
            wrapped_af = DummyWrapper(acq_function=acq_func)
            self.assertIs(wrapped_af.acq_func, acq_func)
            # test forward
            X = torch.rand(1, 1, dtype=dtype, device=self.device)
            with torch.no_grad():
                wrapped_val = wrapped_af(X)
                af_val = acq_func(X)
            self.assertEqual(wrapped_val.item(), af_val.item())

            # test X_pending
            with self.assertRaises(ValueError):
                self.assertIsNone(wrapped_af.X_pending)
            with self.assertRaises(UnsupportedError):
                wrapped_af.set_X_pending(X)
            acq_func = qExpectedImprovement(model=mm, best_f=-1.0)
            wrapped_af = DummyWrapper(acq_function=acq_func)
            self.assertIsNone(wrapped_af.X_pending)
            wrapped_af.set_X_pending(X)
            self.assertTrue(torch.equal(X, wrapped_af.X_pending))
            self.assertTrue(torch.equal(X, acq_func.X_pending))
            wrapped_af.set_X_pending(None)
            self.assertIsNone(wrapped_af.X_pending)
            self.assertIsNone(acq_func.X_pending)
