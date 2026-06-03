#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.utils.multi_objective.hypervolume import (  # noqa: F401
        Hypervolume,
        infer_reference_point,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated  # noqa: F401
    from botorch.utils.multi_objective.scalarization import (  # noqa: F401
        get_chebyshev_scalarization,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Hypervolume": (".hypervolume", "Hypervolume"),
    "infer_reference_point": (".hypervolume", "infer_reference_point"),
    "is_non_dominated": (".pareto", "is_non_dominated"),
    "get_chebyshev_scalarization": (".scalarization", "get_chebyshev_scalarization"),
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
