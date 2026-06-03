#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (  # noqa: F401
        AcquisitionFunction,
        OneShotAcquisitionFunction,
    )
    from botorch.acquisition.active_learning import (  # noqa: F401
        PairwiseMCPosteriorVariance,
        qNegIntegratedPosteriorVariance,
    )
    from botorch.acquisition.analytic import (  # noqa: F401
        AnalyticAcquisitionFunction,
        ConstrainedExpectedImprovement,
        ExpectedImprovement,
        LogExpectedImprovement,
        LogNoisyExpectedImprovement,
        NoisyExpectedImprovement,
        PosteriorMean,
        PosteriorStandardDeviation,
        ProbabilityOfImprovement,
        qAnalyticProbabilityOfImprovement,
        UpperConfidenceBound,
    )
    from botorch.acquisition.bayesian_active_learning import (  # noqa: F401
        qBayesianActiveLearningByDisagreement,
    )
    from botorch.acquisition.cost_aware import (  # noqa: F401
        GenericCostAwareUtility,
        InverseCostWeightedUtility,
    )
    from botorch.acquisition.decoupled import DecoupledAcquisitionFunction  # noqa: F401
    from botorch.acquisition.factory import get_acquisition_function  # noqa: F401
    from botorch.acquisition.fixed_feature import (  # noqa: F401
        FixedFeatureAcquisitionFunction,
    )
    from botorch.acquisition.input_constructors import (  # noqa: F401
        get_acqf_input_constructor,
    )
    from botorch.acquisition.knowledge_gradient import (  # noqa: F401
        qKnowledgeGradient,
        qMultiFidelityKnowledgeGradient,
    )
    from botorch.acquisition.logei import (  # noqa: F401
        LogImprovementMCAcquisitionFunction,
        qLogExpectedImprovement,
        qLogNoisyExpectedImprovement,
    )
    from botorch.acquisition.max_value_entropy_search import (  # noqa: F401
        MaxValueBase,
        qLowerBoundMaxValueEntropy,
        qMaxValueEntropy,
        qMultiFidelityLowerBoundMaxValueEntropy,
        qMultiFidelityMaxValueEntropy,
    )
    from botorch.acquisition.monte_carlo import (  # noqa: F401
        MCAcquisitionFunction,
        qExpectedImprovement,
        qLowerConfidenceBound,
        qNoisyExpectedImprovement,
        qPosteriorStandardDeviation,
        qProbabilityOfImprovement,
        qSimpleRegret,
        qUpperConfidenceBound,
        SampleReducingMCAcquisitionFunction,
    )
    from botorch.acquisition.multi_step_lookahead import (  # noqa: F401
        qMultiStepLookahead,
    )
    from botorch.acquisition.multioutput_acquisition import (  # noqa: F401
        MultiOutputAcquisitionFunction,
    )
    from botorch.acquisition.objective import (  # noqa: F401
        ConstrainedMCObjective,
        GenericMCObjective,
        IdentityMCObjective,
        LearnedObjective,
        LinearMCObjective,
        MCAcquisitionObjective,
        ScalarizedPosteriorTransform,
    )
    from botorch.acquisition.preference import (  # noqa: F401
        AnalyticExpectedUtilityOfBestOption,
        PairwiseBayesianActiveLearningByDisagreement,
        qExpectedUtilityOfBestOption,
    )
    from botorch.acquisition.prior_guided import (  # noqa: F401
        PriorGuidedAcquisitionFunction,
    )
    from botorch.acquisition.proximal import ProximalAcquisitionFunction  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AcquisitionFunction": (".acquisition", "AcquisitionFunction"),
    "OneShotAcquisitionFunction": (".acquisition", "OneShotAcquisitionFunction"),
    "PairwiseMCPosteriorVariance": (".active_learning", "PairwiseMCPosteriorVariance"),
    "qNegIntegratedPosteriorVariance": (
        ".active_learning",
        "qNegIntegratedPosteriorVariance",
    ),
    "AnalyticAcquisitionFunction": (".analytic", "AnalyticAcquisitionFunction"),
    "ConstrainedExpectedImprovement": (".analytic", "ConstrainedExpectedImprovement"),
    "ExpectedImprovement": (".analytic", "ExpectedImprovement"),
    "LogExpectedImprovement": (".analytic", "LogExpectedImprovement"),
    "LogNoisyExpectedImprovement": (".analytic", "LogNoisyExpectedImprovement"),
    "NoisyExpectedImprovement": (".analytic", "NoisyExpectedImprovement"),
    "PosteriorMean": (".analytic", "PosteriorMean"),
    "PosteriorStandardDeviation": (".analytic", "PosteriorStandardDeviation"),
    "ProbabilityOfImprovement": (".analytic", "ProbabilityOfImprovement"),
    "qAnalyticProbabilityOfImprovement": (
        ".analytic",
        "qAnalyticProbabilityOfImprovement",
    ),
    "UpperConfidenceBound": (".analytic", "UpperConfidenceBound"),
    "qBayesianActiveLearningByDisagreement": (
        ".bayesian_active_learning",
        "qBayesianActiveLearningByDisagreement",
    ),
    "GenericCostAwareUtility": (".cost_aware", "GenericCostAwareUtility"),
    "InverseCostWeightedUtility": (".cost_aware", "InverseCostWeightedUtility"),
    "DecoupledAcquisitionFunction": (".decoupled", "DecoupledAcquisitionFunction"),
    "get_acquisition_function": (".factory", "get_acquisition_function"),
    "FixedFeatureAcquisitionFunction": (
        ".fixed_feature",
        "FixedFeatureAcquisitionFunction",
    ),
    "get_acqf_input_constructor": (".input_constructors", "get_acqf_input_constructor"),
    "qKnowledgeGradient": (".knowledge_gradient", "qKnowledgeGradient"),
    "qMultiFidelityKnowledgeGradient": (
        ".knowledge_gradient",
        "qMultiFidelityKnowledgeGradient",
    ),
    "LogImprovementMCAcquisitionFunction": (
        ".logei",
        "LogImprovementMCAcquisitionFunction",
    ),
    "qLogExpectedImprovement": (".logei", "qLogExpectedImprovement"),
    "qLogNoisyExpectedImprovement": (".logei", "qLogNoisyExpectedImprovement"),
    "MaxValueBase": (".max_value_entropy_search", "MaxValueBase"),
    "qLowerBoundMaxValueEntropy": (
        ".max_value_entropy_search",
        "qLowerBoundMaxValueEntropy",
    ),
    "qMaxValueEntropy": (".max_value_entropy_search", "qMaxValueEntropy"),
    "qMultiFidelityLowerBoundMaxValueEntropy": (
        ".max_value_entropy_search",
        "qMultiFidelityLowerBoundMaxValueEntropy",
    ),
    "qMultiFidelityMaxValueEntropy": (
        ".max_value_entropy_search",
        "qMultiFidelityMaxValueEntropy",
    ),
    "MCAcquisitionFunction": (".monte_carlo", "MCAcquisitionFunction"),
    "qExpectedImprovement": (".monte_carlo", "qExpectedImprovement"),
    "qLowerConfidenceBound": (".monte_carlo", "qLowerConfidenceBound"),
    "qNoisyExpectedImprovement": (".monte_carlo", "qNoisyExpectedImprovement"),
    "qPosteriorStandardDeviation": (".monte_carlo", "qPosteriorStandardDeviation"),
    "qProbabilityOfImprovement": (".monte_carlo", "qProbabilityOfImprovement"),
    "qSimpleRegret": (".monte_carlo", "qSimpleRegret"),
    "qUpperConfidenceBound": (".monte_carlo", "qUpperConfidenceBound"),
    "SampleReducingMCAcquisitionFunction": (
        ".monte_carlo",
        "SampleReducingMCAcquisitionFunction",
    ),
    "qMultiStepLookahead": (".multi_step_lookahead", "qMultiStepLookahead"),
    "MultiOutputAcquisitionFunction": (
        ".multioutput_acquisition",
        "MultiOutputAcquisitionFunction",
    ),
    "ConstrainedMCObjective": (".objective", "ConstrainedMCObjective"),
    "GenericMCObjective": (".objective", "GenericMCObjective"),
    "IdentityMCObjective": (".objective", "IdentityMCObjective"),
    "LearnedObjective": (".objective", "LearnedObjective"),
    "LinearMCObjective": (".objective", "LinearMCObjective"),
    "MCAcquisitionObjective": (".objective", "MCAcquisitionObjective"),
    "ScalarizedPosteriorTransform": (".objective", "ScalarizedPosteriorTransform"),
    "AnalyticExpectedUtilityOfBestOption": (
        ".preference",
        "AnalyticExpectedUtilityOfBestOption",
    ),
    "PairwiseBayesianActiveLearningByDisagreement": (
        ".preference",
        "PairwiseBayesianActiveLearningByDisagreement",
    ),
    "qExpectedUtilityOfBestOption": (".preference", "qExpectedUtilityOfBestOption"),
    "PriorGuidedAcquisitionFunction": (
        ".prior_guided",
        "PriorGuidedAcquisitionFunction",
    ),
    "ProximalAcquisitionFunction": (".proximal", "ProximalAcquisitionFunction"),
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
