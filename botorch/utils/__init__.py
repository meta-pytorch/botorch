#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.utils.constraints import (  # noqa: F401
        get_outcome_constraint_transforms,
    )
    from botorch.utils.feasible_volume import estimate_feasible_volume  # noqa: F401
    from botorch.utils.objective import (  # noqa: F401
        apply_constraints,
        get_objective_weights_transform,
    )
    from botorch.utils.rounding import approximate_round  # noqa: F401
    from botorch.utils.sampling import (  # noqa: F401
        batched_multinomial,
        draw_sobol_normal_samples,
        draw_sobol_samples,
        manual_seed,
    )
    from botorch.utils.transforms import (  # noqa: F401
        average_over_ensemble_models,
        standardize,
        t_batch_mode_transform,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_outcome_constraint_transforms": (
        ".constraints",
        "get_outcome_constraint_transforms",
    ),
    "estimate_feasible_volume": (".feasible_volume", "estimate_feasible_volume"),
    "apply_constraints": (".objective", "apply_constraints"),
    "get_objective_weights_transform": (
        ".objective",
        "get_objective_weights_transform",
    ),
    "approximate_round": (".rounding", "approximate_round"),
    "batched_multinomial": (".sampling", "batched_multinomial"),
    "draw_sobol_normal_samples": (".sampling", "draw_sobol_normal_samples"),
    "draw_sobol_samples": (".sampling", "draw_sobol_samples"),
    "manual_seed": (".sampling", "manual_seed"),
    "average_over_ensemble_models": (".transforms", "average_over_ensemble_models"),
    "standardize": (".transforms", "standardize"),
    "t_batch_mode_transform": (".transforms", "t_batch_mode_transform"),
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
