#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.multioutput_acquisition import (
    MultiOutputAcquisitionFunction,
    MultiOutputAcquisitionFunctionWrapper,
    MultiOutputPosteriorMean,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMultiOutputAcqf(MultiOutputAcquisitionFunction):
    def forward(self, X):
        pass


class TestMultiOutputAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MultiOutputAcquisitionFunction()

    def test_set_X_pending(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedError,
            "X_pending is not supported for multi-output acquisition functions.",
        ):
            DummyMultiOutputAcqf(
                model=MockModel(posterior=MockPosterior())
            ).set_X_pending(torch.ones(1, 1))

    def test_multioutput_posterior_mean(self) -> None:
        # test single output model
        with self.assertRaisesRegex(
            NotImplementedError, "MultiPosteriorMean only supports multi-output models."
        ):
            MultiOutputPosteriorMean(
                model=MockModel(posterior=MockPosterior(mean=torch.tensor([[1.0]])))
            )
        # test invalid weights
        with self.assertRaisesRegex(
            ValueError, "weights must have 2 elements, but got 1."
        ):
            MultiOutputPosteriorMean(
                model=MockModel(
                    posterior=MockPosterior(mean=torch.tensor([[1.0, 2.0]]))
                ),
                weights=torch.tensor([1.0]),
            )
        for dtype in (torch.float, torch.double):
            # basic test
            mean = torch.tensor([[1.0, 2.0]], dtype=dtype, device=self.device)
            acqf = MultiOutputPosteriorMean(
                model=MockModel(posterior=MockPosterior(mean=mean))
            )
            self.assertTrue(
                torch.equal(
                    acqf(torch.ones(1, 1, 1, dtype=dtype, device=self.device)),
                    mean.squeeze(-2),
                )
            )
            # test weights
            weights = torch.tensor([-1.0, 1.0], dtype=dtype, device=self.device)
            acqf = MultiOutputPosteriorMean(
                model=MockModel(posterior=MockPosterior(mean=mean)), weights=weights
            )
            self.assertTrue(
                torch.equal(
                    acqf(torch.ones(1, 1, 1, dtype=dtype, device=self.device)),
                    mean.squeeze(-2) * weights,
                )
            )

    def test_multioutput_wrapper(self) -> None:
        for dtype in (torch.float, torch.double):
            model = MockModel(
                posterior=MockPosterior(
                    mean=torch.tensor([[1.0]], dtype=dtype, device=self.device),
                    variance=torch.tensor([[0.1]], dtype=dtype, device=self.device),
                )
            )
            ei = LogExpectedImprovement(model=model, best_f=0.0)
            ucb = UpperConfidenceBound(model=model, beta=2.0)
            acqf = MultiOutputAcquisitionFunctionWrapper(acqfs=[ei, ucb])
            X = torch.ones(1, 1, 1, dtype=dtype, device=self.device)
            expected_af_vals = torch.stack([ei(X=X), ucb(X=X)], dim=-1)
            self.assertTrue(torch.equal(acqf(X), expected_af_vals))
