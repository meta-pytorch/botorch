#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MC acquisition functions using the test harness."""

from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qPosteriorStandardDeviation,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.utils.testing import BotorchTestCase

from .harness import MCAcquisitionSpec, MCAcquisitionTestMixin


class TestMCAcquisitionHarness(MCAcquisitionTestMixin, BotorchTestCase):
    """Test MC acquisition functions using the test harness."""

    @property
    def acquisition_specs(self) -> list[MCAcquisitionSpec]:
        return [
            MCAcquisitionSpec(
                acqf_class=qExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            MCAcquisitionSpec(
                acqf_class=qLogExpectedImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            MCAcquisitionSpec(
                acqf_class=qProbabilityOfImprovement,
                required_kwargs={"best_f": 0.0},
            ),
            MCAcquisitionSpec(
                acqf_class=qUpperConfidenceBound,
                required_kwargs={"beta": 0.2},
            ),
            MCAcquisitionSpec(
                acqf_class=qSimpleRegret,
            ),
            MCAcquisitionSpec(
                acqf_class=qPosteriorStandardDeviation,
            ),
            # cache_root=False: allows forward passes with varying q without
            # needing to reset q_in=-1 before each call (see test_cache_root).
            MCAcquisitionSpec(
                acqf_class=qNoisyExpectedImprovement,
                requires_X_baseline=True,
                required_kwargs={"cache_root": False},
            ),
            MCAcquisitionSpec(
                acqf_class=qLogNoisyExpectedImprovement,
                requires_X_baseline=True,
                required_kwargs={"cache_root": False, "incremental": False},
            ),
        ]
