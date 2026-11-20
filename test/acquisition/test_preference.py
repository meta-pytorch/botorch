#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    PairwiseBayesianActiveLearningByDisagreement,
    qExpectedUtilityOfBestOption,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.testing import BotorchTestCase


class TestPreferenceAcquisitionFunctions(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.twargs = {"dtype": torch.double}
        self.X_dim = 3
        self.Y_dim = 2
        X = torch.rand(2, self.X_dim, **self.twargs)
        Y = torch.rand(2, self.Y_dim, **self.twargs)
        comps = torch.tensor([[1, 0]], dtype=torch.long)

        self.model = SingleTaskGP(X, Y)
        self.pref_model_on_X = PairwiseGP(X, comps)
        self.pref_model_on_Y = PairwiseGP(Y, comps)
        self.deterministic_model = FixedSingleSampleModel(model=self.model)

        self.X1 = torch.rand(1, self.X_dim, **self.twargs)
        self.X2 = torch.rand(2, self.X_dim, **self.twargs)
        self.X3 = torch.rand(3, self.X_dim, **self.twargs)


class TestAnalyticEUBOAndBald(TestPreferenceAcquisitionFunctions):
    def test_only_pairwise_allowed(self) -> None:
        outcome_models = [self.deterministic_model, None]
        acqf_classes = [
            AnalyticExpectedUtilityOfBestOption,
            PairwiseBayesianActiveLearningByDisagreement,
        ]
        for outcome_model, acqf_class in product(outcome_models, acqf_classes):
            with self.subTest(outcome_model=outcome_model, acqf_cls=acqf_class):
                pref_model = (
                    self.pref_model_on_X
                    if outcome_model is None
                    else self.pref_model_on_Y
                )
                # Test with an outcome model and a preference model
                acqf = acqf_class(pref_model=pref_model, outcome_model=outcome_model)

                # test forward with different number of points
                # q = 1
                with self.assertRaises((UnsupportedError, AssertionError)):
                    acqf(self.X1)
                # q = 2
                acqf(self.X2)
                # q > 2
                with self.assertRaises((UnsupportedError, AssertionError)):
                    acqf(self.X3)

    def test_analytic_eubo_previous_winner(self) -> None:
        for outcome_model in [self.deterministic_model, None]:
            pref_model = (
                self.pref_model_on_X if outcome_model is None else self.pref_model_on_Y
            )
            # Test with an outcome model and a preference model
            acqf = AnalyticExpectedUtilityOfBestOption(
                pref_model=pref_model, outcome_model=outcome_model
            )
            previous_winner = (
                torch.rand(1, self.X_dim, **self.twargs)
                if outcome_model is None
                else torch.rand(1, self.Y_dim, **self.twargs)
            )
            acqf = AnalyticExpectedUtilityOfBestOption(
                pref_model=pref_model,
                outcome_model=outcome_model,
                previous_winner=previous_winner,
            )
            # q = 1
            acqf(self.X1)
            # q = 2
            with self.assertRaises((UnsupportedError, AssertionError)):
                acqf(self.X2)
            # q > 2
            with self.assertRaises((UnsupportedError, AssertionError)):
                acqf(self.X3)

    def test_analytic_eubo_prior_only_model(self) -> None:
        prior_pref_model = PairwiseGP(None, None).to(**self.twargs)
        acqf = AnalyticExpectedUtilityOfBestOption(pref_model=prior_pref_model)
        val = acqf(self.X2)
        self.assertEqual(val.shape, torch.Size([1]))
        self.assertTrue(torch.all(torch.isfinite(val)))


class TestQExpectedUtilityOfBestOption(TestPreferenceAcquisitionFunctions):
    def test_qeubo(self) -> None:
        for outcome_model in [self.deterministic_model, None]:
            pref_model = (
                self.pref_model_on_X if outcome_model is None else self.pref_model_on_Y
            )
            # Test with an outcome model and a preference model
            acqf = qExpectedUtilityOfBestOption(
                pref_model=pref_model, outcome_model=outcome_model
            )

            # test forward with different number of points
            acq_val = acqf(self.X1)
            self.assertEqual(acq_val.shape, torch.Size([1]))
            acq_val = acqf(self.X2)
            self.assertEqual(acq_val.shape, torch.Size([1]))
            acq_val = acqf(self.X3)
            self.assertEqual(acq_val.shape, torch.Size([1]))
