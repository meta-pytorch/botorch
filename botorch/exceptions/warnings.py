#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Botorch Warnings.
"""

import warnings


class BotorchWarning(Warning):
    r"""Base botorch warning."""

    pass


class BadInitialCandidatesWarning(BotorchWarning):
    r"""Warning issued if set of initial candidates for optimziation is bad."""

    pass


class InputDataWarning(BotorchWarning):
    r"""Warning raised when input data does not comply with conventions."""

    pass


class CostAwareWarning(BotorchWarning):
    r"""Warning raised in the context of cost-aware acquisition strategies."""

    pass


class OptimizationWarning(BotorchWarning):
    r"""Optimization-related warnings."""

    pass


class SamplingWarning(BotorchWarning):
    r"""Sampling related warnings."""

    pass


class BotorchTensorDimensionWarning(BotorchWarning):
    r"""Warning raised when a tensor possibly violates a botorch convention."""

    pass


class UserInputWarning(BotorchWarning):
    r"""Warning raised when a potential issue is detected with user provided inputs."""

    pass


class NumericsWarning(BotorchWarning):
    r"""Warning raised when numerical issues are detected."""

    pass


def legacy_ei_numerics_warning(legacy_name: str) -> None:
    """Raises a warning for legacy EI acquisition functions that are known to have
    numerical issues and should be replaced with the LogEI version for virtually all
    use-cases except for explicit benchmarking of the numerical issues of legacy EI.

    Args:
        legacy_name: The name of the legacy EI acquisition function.
        logei_name: The name of the associated LogEI acquisition function.
    """
    legacy_to_logei = {
        "ExpectedImprovement": "LogExpectedImprovement",
        "ConstrainedExpectedImprovement": "LogConstrainedExpectedImprovement",
        "NoisyExpectedImprovement": "LogNoisyExpectedImprovement",
        "qExpectedImprovement": "qLogExpectedImprovement",
        "qNoisyExpectedImprovement": "qLogNoisyExpectedImprovement",
        "qExpectedHypervolumeImprovement": "qLogExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement": (
            "qLogNoisyExpectedHypervolumeImprovement"
        ),
    }
    # Only raise the warning if the legacy name is in the mapping. It can fail to be in
    # the mapping if the legacy acquisition function derives from a legacy EI class,
    # e.g. MOMF, which derives from qEHVI, but there is not corresponding LogMOMF yet.
    if legacy_name in legacy_to_logei:
        logei_name = legacy_to_logei[legacy_name]
        msg = (
            f"{legacy_name} has known numerical issues that lead to suboptimal "
            "optimization performance. It is strongly recommended to simply replace"
            f"\n\n\t {legacy_name} \t --> \t {logei_name} \n\n"
            "instead, which fixes the issues and has the same "
            "API. See https://arxiv.org/abs/2310.20708 for details."
        )
        warnings.warn(msg, NumericsWarning, stacklevel=2)


def _get_single_precision_warning(dtype_str: str) -> str:
    msg = (
        f"The model inputs are of type {dtype_str}. It is strongly recommended "
        "to use double precision in BoTorch, as this improves both "
        "precision and stability and can help avoid numerical errors. "
        "See https://github.com/meta-pytorch/botorch/discussions/1444"
    )
    return msg
