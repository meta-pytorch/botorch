#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for analytic acquisition functions using the test harness."""

import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    qAnalyticProbabilityOfImprovement,
    ScalarizedPosteriorMean,
    UpperConfidenceBound,
)
from botorch.utils.testing import BotorchTestCase

from .harness.mixins import AcquisitionSpec, AnalyticAcquisitionTestMixin


class TestAnalyticAcquisitionHarness(AnalyticAcquisitionTestMixin, BotorchTestCase):
    """Test analytic acquisition functions using the test harness."""

    @property
    def acquisition_specs(self) -> list[AcquisitionSpec]:
        """Return the list of AcquisitionSpec instances to test."""
        return [
            AcquisitionSpec(
                acqf_class=LogExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                acqf_class=ExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                acqf_class=LogProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                acqf_class=ProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                acqf_class=qAnalyticProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
                q=3,
                bypass_tests=["test_maximize"],
            ),
            AcquisitionSpec(
                acqf_class=UpperConfidenceBound,
                required_kwargs={"beta": 0.2},
            ),
            AcquisitionSpec(
                acqf_class=PosteriorMean,
                required_kwargs={},
            ),
            AcquisitionSpec(
                acqf_class=PosteriorStandardDeviation,
                required_kwargs={},
                bypass_tests=["test_maximize"],
            ),
            AcquisitionSpec(
                acqf_class=ScalarizedPosteriorMean,
                required_kwargs={"weights": torch.tensor([1.0, 0.5, 0.25])},
                bypass_tests=["test_maximize"],
                q=3,
            ),
            AcquisitionSpec(
                acqf_class=LogNoisyExpectedImprovement,
                required_kwargs={},
                requires_X_observed=True,
                requires_fixed_noise=True,
                bypass_tests=["test_maximize"],
            ),
            AcquisitionSpec(
                acqf_class=NoisyExpectedImprovement,
                required_kwargs={"num_fantasies": 8},
                requires_X_observed=True,
                requires_fixed_noise=True,
                bypass_tests=["test_maximize"],
            ),
        ]
