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
from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
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

    def test_multioutput_posterior_mean_ensemble(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(10, 3, **tkwargs)
        train_Ys = [torch.sin(6 * train_X[:, :1]), 5 + train_X[:, 1:2] ** 2]
        test_X = torch.rand(4, 1, 3, **tkwargs)
        # Check both an ensemble size that differs from the number of outputs
        # and one that matches it.
        for num_taus in (2, 3):
            models = [
                EnsembleMapSaasSingleTaskGP(
                    train_X=train_X, train_Y=train_Y, num_taus=num_taus
                )
                for train_Y in train_Ys
            ]
            for m in models:
                # Give the members different means, so that the result depends
                # on averaging over all of them.
                m.mean_module.constant = torch.arange(num_taus, **tkwargs)
            model = ModelListGP(*models)
            with torch.no_grad():
                expected = torch.cat(
                    [m.posterior(test_X).mixture_mean for m in models], dim=-1
                ).squeeze(-2)
            for weights in (None, torch.tensor([-1.0, 2.0], **tkwargs)):
                acqf = MultiOutputPosteriorMean(model=model, weights=weights)
                with torch.no_grad():
                    acqf_vals = acqf(test_X)
                self.assertEqual(acqf_vals.shape, torch.Size([4, 2]))
                self.assertAllClose(
                    acqf_vals, expected if weights is None else expected * weights
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
