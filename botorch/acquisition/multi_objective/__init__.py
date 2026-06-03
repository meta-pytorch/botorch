#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.acquisition.multi_objective.analytic import (  # noqa: F401
        ExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.base import (  # noqa: F401
        MultiObjectiveAnalyticAcquisitionFunction,
        MultiObjectiveMCAcquisitionFunction,
    )
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (  # noqa: F401, E501
        qHypervolumeKnowledgeGradient,
        qMultiFidelityHypervolumeKnowledgeGradient,
    )
    from botorch.acquisition.multi_objective.logei import (  # noqa: F401
        qLogExpectedHypervolumeImprovement,
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.monte_carlo import (  # noqa: F401
        qExpectedHypervolumeImprovement,
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.multi_fidelity import MOMF  # noqa: F401
    from botorch.acquisition.multi_objective.objective import (  # noqa: F401
        IdentityMCMultiOutputObjective,
        MCMultiOutputObjective,
        WeightedMCMultiOutputObjective,
    )
    from botorch.acquisition.multi_objective.utils import (  # noqa: F401
        get_default_partitioning_alpha,
        prune_inferior_points_multi_objective,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ExpectedHypervolumeImprovement": (".analytic", "ExpectedHypervolumeImprovement"),
    "MultiObjectiveAnalyticAcquisitionFunction": (
        ".base",
        "MultiObjectiveAnalyticAcquisitionFunction",
    ),
    "MultiObjectiveMCAcquisitionFunction": (
        ".base",
        "MultiObjectiveMCAcquisitionFunction",
    ),
    "qHypervolumeKnowledgeGradient": (
        ".hypervolume_knowledge_gradient",
        "qHypervolumeKnowledgeGradient",
    ),
    "qMultiFidelityHypervolumeKnowledgeGradient": (
        ".hypervolume_knowledge_gradient",
        "qMultiFidelityHypervolumeKnowledgeGradient",
    ),
    "qLogExpectedHypervolumeImprovement": (
        ".logei",
        "qLogExpectedHypervolumeImprovement",
    ),
    "qLogNoisyExpectedHypervolumeImprovement": (
        ".logei",
        "qLogNoisyExpectedHypervolumeImprovement",
    ),
    "qExpectedHypervolumeImprovement": (
        ".monte_carlo",
        "qExpectedHypervolumeImprovement",
    ),
    "qNoisyExpectedHypervolumeImprovement": (
        ".monte_carlo",
        "qNoisyExpectedHypervolumeImprovement",
    ),
    "MOMF": (".multi_fidelity", "MOMF"),
    "IdentityMCMultiOutputObjective": (".objective", "IdentityMCMultiOutputObjective"),
    "MCMultiOutputObjective": (".objective", "MCMultiOutputObjective"),
    "WeightedMCMultiOutputObjective": (".objective", "WeightedMCMultiOutputObjective"),
    "get_default_partitioning_alpha": (".utils", "get_default_partitioning_alpha"),
    "prune_inferior_points_multi_objective": (
        ".utils",
        "prune_inferior_points_multi_objective",
    ),
}

__all__ = list(_LAZY_IMPORTS.keys())


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:  # pragma: no cover
        if name in _LAZY_IMPORTS:
            rel_module, attr = _LAZY_IMPORTS[name]
            module = importlib.import_module(rel_module, __name__)
            value = getattr(module, attr)
            globals()[name] = value
            return value
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__() -> list[str]:  # pragma: no cover
        return __all__
