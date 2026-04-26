#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.generation.gen import (  # noqa: F401
        gen_candidates_scipy,
        gen_candidates_torch,
        get_best_candidates,
    )
    from botorch.generation.sampling import (  # noqa: F401
        BoltzmannSampling,
        MaxPosteriorSampling,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "gen_candidates_scipy": (".gen", "gen_candidates_scipy"),
    "gen_candidates_torch": (".gen", "gen_candidates_torch"),
    "get_best_candidates": (".gen", "get_best_candidates"),
    "BoltzmannSampling": (".sampling", "BoltzmannSampling"),
    "MaxPosteriorSampling": (".sampling", "MaxPosteriorSampling"),
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
