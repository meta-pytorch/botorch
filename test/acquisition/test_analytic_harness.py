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

from .harness import AcquisitionSpec, AnalyticAcquisitionTestMixin


class TestAnalyticAcquisitionHarness(AnalyticAcquisitionTestMixin, BotorchTestCase):
    """Test analytic acquisition functions using the test harness."""

    @property
    def acquisition_specs(self) -> list[AcquisitionSpec]:
        """Return the list of AcquisitionSpec instances to test."""
        return [
            AcquisitionSpec(
                cls=LogExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                cls=ExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                cls=LogProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                cls=ProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                cls=qAnalyticProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            AcquisitionSpec(
                cls=UpperConfidenceBound,
                required_kwargs={"beta": 0.2},
            ),
            AcquisitionSpec(
                cls=PosteriorMean,
                required_kwargs={},
            ),
            AcquisitionSpec(
                cls=PosteriorStandardDeviation,
                required_kwargs={},
            ),
            AcquisitionSpec(
                cls=ScalarizedPosteriorMean,
                required_kwargs={"weights": torch.tensor([1.0, 1.0, 1.0])},
                q_dim=3,
                bypass_tests=["test_maximize"],
            ),
            AcquisitionSpec(
                cls=LogNoisyExpectedImprovement,
                required_kwargs={},
                requires_X_observed=True,
                requires_fixed_noise=True,
            ),
            AcquisitionSpec(
                cls=NoisyExpectedImprovement,
                required_kwargs={"num_fantasies": 8},
                requires_X_observed=True,
                requires_fixed_noise=True,
            ),
        ]
