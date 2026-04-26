#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.sampling.base import MCSampler  # noqa: F401
    from botorch.sampling.get_sampler import get_sampler  # noqa: F401
    from botorch.sampling.list_sampler import ListSampler  # noqa: F401
    from botorch.sampling.normal import (  # noqa: F401
        IIDNormalSampler,
        SobolQMCNormalSampler,
    )
    from botorch.sampling.pairwise_samplers import (  # noqa: F401
        PairwiseIIDNormalSampler,
        PairwiseMCSampler,
        PairwiseSobolQMCNormalSampler,
    )
    from botorch.sampling.qmc import (  # noqa: F401
        MultivariateNormalQMCEngine,
        NormalQMCEngine,
    )
    from botorch.sampling.stochastic_samplers import (  # noqa: F401
        ForkedRNGSampler,
        StochasticSampler,
    )
    from torch.quasirandom import SobolEngine  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MCSampler": (".base", "MCSampler"),
    "get_sampler": (".get_sampler", "get_sampler"),
    "ListSampler": (".list_sampler", "ListSampler"),
    "IIDNormalSampler": (".normal", "IIDNormalSampler"),
    "SobolQMCNormalSampler": (".normal", "SobolQMCNormalSampler"),
    "PairwiseIIDNormalSampler": (".pairwise_samplers", "PairwiseIIDNormalSampler"),
    "PairwiseMCSampler": (".pairwise_samplers", "PairwiseMCSampler"),
    "PairwiseSobolQMCNormalSampler": (
        ".pairwise_samplers",
        "PairwiseSobolQMCNormalSampler",
    ),
    "MultivariateNormalQMCEngine": (".qmc", "MultivariateNormalQMCEngine"),
    "NormalQMCEngine": (".qmc", "NormalQMCEngine"),
    "ForkedRNGSampler": (".stochastic_samplers", "ForkedRNGSampler"),
    "StochasticSampler": (".stochastic_samplers", "StochasticSampler"),
    "SobolEngine": ("torch.quasirandom", "SobolEngine"),
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
