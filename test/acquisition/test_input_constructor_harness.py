#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for acquisition functions via input constructors.

This test harness mirrors Ax MBM usage patterns by using
`get_acqf_input_constructor(acqf_cls)(**kwargs)` → `acqf_cls(**result)`.
"""

from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    TAU_MAX,
    TAU_RELU,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.utils.testing import BotorchTestCase

from .harness import InputConstructorSpec, InputConstructorTestMixin


class TestInputConstructorHarness(InputConstructorTestMixin, BotorchTestCase):
    """Systematic tests for acquisition input constructors.

    Tests that acquisition functions can be constructed via their input constructors
    and that optional inputs (posterior_transform, objective) are properly propagated.
    """

    @property
    def input_constructor_specs(self) -> list[InputConstructorSpec]:
        return [
            # ========== Analytic (single-output) ==========
            InputConstructorSpec(
                cls=PosteriorMean,
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=ExpectedImprovement,
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=LogExpectedImprovement,
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=ProbabilityOfImprovement,
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=LogProbabilityOfImprovement,
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=UpperConfidenceBound,
                constructor_kwargs={"beta": 0.2},
                supports_posterior_transform=True,
                bypass_tests=["test_defaults"],
            ),
            # ========== MC (single-objective) ==========
            InputConstructorSpec(
                cls=qSimpleRegret,
                supports_posterior_transform=True,
                supports_objective=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=qExpectedImprovement,
                supports_posterior_transform=True,
                supports_objective=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=qLogExpectedImprovement,
                supports_posterior_transform=True,
                supports_objective=True,
                expected_defaults={
                    "tau_max": TAU_MAX,
                    "tau_relu": TAU_RELU,
                },
            ),
            InputConstructorSpec(
                cls=qProbabilityOfImprovement,
                supports_posterior_transform=True,
                supports_objective=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=qUpperConfidenceBound,
                constructor_kwargs={"beta": 0.2},
                supports_posterior_transform=True,
                supports_objective=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=qNoisyExpectedImprovement,
                supports_posterior_transform=True,
                supports_objective=True,
                bypass_tests=["test_defaults"],
            ),
            InputConstructorSpec(
                cls=qLogNoisyExpectedImprovement,
                supports_posterior_transform=True,
                supports_objective=True,
                expected_defaults={
                    "tau_max": TAU_MAX,
                    "tau_relu": TAU_RELU,
                    "prune_baseline": True,
                },
            ),
        ]
