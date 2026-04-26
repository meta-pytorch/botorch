#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy-loading module for botorch.models.

Submodules are imported on first access via ``__getattr__`` (PEP 562),
so heavy transitive dependencies (e.g. JAX via ``fully_bayesian``) are
never loaded unless explicitly requested.
"""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.models.approximate_gp import (  # noqa: F401
        ApproximateGPyTorchModel,
        SingleTaskVariationalGP,
    )
    from botorch.models.cost import AffineFidelityCostModel  # noqa: F401
    from botorch.models.deterministic import (  # noqa: F401
        AffineDeterministicModel,
        GenericDeterministicModel,
        PosteriorMeanModel,
    )
    from botorch.models.gp_regression import SingleTaskGP as SingleTaskGP  # noqa: F401
    from botorch.models.gp_regression_fidelity import (  # noqa: F401
        SingleTaskMultiFidelityGP,
    )
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    from botorch.models.higher_order_gp import HigherOrderGP  # noqa: F401
    from botorch.models.map_saas import (  # noqa: F401
        add_saas_prior,
        AdditiveMapSaasSingleTaskGP,
        EnsembleMapSaasSingleTaskGP,
    )
    from botorch.models.model import ModelList as ModelList  # noqa: F401
    from botorch.models.model_list_gp_regression import ModelListGP  # noqa: F401
    from botorch.models.multitask import KroneckerMultiTaskGP, MultiTaskGP  # noqa: F401
    from botorch.models.pairwise_gp import (  # noqa: F401
        PairwiseGP,
        PairwiseLaplaceMarginalLogLikelihood,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ApproximateGPyTorchModel": (".approximate_gp", "ApproximateGPyTorchModel"),
    "SingleTaskVariationalGP": (".approximate_gp", "SingleTaskVariationalGP"),
    "AffineFidelityCostModel": (".cost", "AffineFidelityCostModel"),
    "AffineDeterministicModel": (".deterministic", "AffineDeterministicModel"),
    "GenericDeterministicModel": (".deterministic", "GenericDeterministicModel"),
    "PosteriorMeanModel": (".deterministic", "PosteriorMeanModel"),
    "SingleTaskGP": (".gp_regression", "SingleTaskGP"),
    "SingleTaskMultiFidelityGP": (
        ".gp_regression_fidelity",
        "SingleTaskMultiFidelityGP",
    ),
    "MixedSingleTaskGP": (".gp_regression_mixed", "MixedSingleTaskGP"),
    "HigherOrderGP": (".higher_order_gp", "HigherOrderGP"),
    "add_saas_prior": (".map_saas", "add_saas_prior"),
    "AdditiveMapSaasSingleTaskGP": (".map_saas", "AdditiveMapSaasSingleTaskGP"),
    "EnsembleMapSaasSingleTaskGP": (".map_saas", "EnsembleMapSaasSingleTaskGP"),
    "ModelList": (".model", "ModelList"),
    "ModelListGP": (".model_list_gp_regression", "ModelListGP"),
    "KroneckerMultiTaskGP": (".multitask", "KroneckerMultiTaskGP"),
    "MultiTaskGP": (".multitask", "MultiTaskGP"),
    "PairwiseGP": (".pairwise_gp", "PairwiseGP"),
    "PairwiseLaplaceMarginalLogLikelihood": (
        ".pairwise_gp",
        "PairwiseLaplaceMarginalLogLikelihood",
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
