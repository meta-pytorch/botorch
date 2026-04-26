#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch import (  # noqa: F401
        acquisition,
        exceptions,
        models,
        optim,
        posteriors,
        settings,
        test_functions,
    )
    from botorch.cross_validation import batch_cross_validation  # noqa: F401
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_mll  # noqa: F401
    from botorch.generation.gen import (  # noqa: F401
        gen_candidates_scipy,
        gen_candidates_torch,
        get_best_candidates,
    )
    from botorch.utils import manual_seed as manual_seed  # noqa: F401

import gpytorch.settings as gp_settings
import linear_operator.settings as linop_settings
from botorch.logging import logger

try:
    # Marking this as a manual import to avoid autodeps complaints
    # due to imports from non-existent file.
    # lint-ignore: UnusedImportsRule
    from botorch.version import version as __version__  # @manual
except Exception:  # pragma: no cover
    __version__ = "Unknown"

logger.info(
    "Turning off `fast_computations` in linear operator and increasing "
    "`max_cholesky_size` and `max_eager_kernel_size` to 4096, and "
    "`cholesky_max_tries` to 6. The approximate computations available in "
    "GPyTorch aim to speed up GP training and inference in large data "
    "regime but they are generally not robust enough to be used in a BO-loop. "
    "See gpytorch.settings & linear_operator.settings for more details."
)
linop_settings._fast_covar_root_decomposition._default = False
linop_settings._fast_log_prob._default = False
linop_settings._fast_solves._default = False
linop_settings.cholesky_max_tries._global_value = 6
linop_settings.max_cholesky_size._global_value = 4096
gp_settings.max_eager_kernel_size._global_value = 4096

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "acquisition": ("botorch.acquisition", None),
    "exceptions": ("botorch.exceptions", None),
    "models": ("botorch.models", None),
    "optim": ("botorch.optim", None),
    "posteriors": ("botorch.posteriors", None),
    "settings": ("botorch.settings", None),
    "test_functions": ("botorch.test_functions", None),
    "batch_cross_validation": ("botorch.cross_validation", "batch_cross_validation"),
    "fit_gpytorch_mll": ("botorch.fit", "fit_gpytorch_mll"),
    "gen_candidates_scipy": ("botorch.generation.gen", "gen_candidates_scipy"),
    "gen_candidates_torch": ("botorch.generation.gen", "gen_candidates_torch"),
    "get_best_candidates": ("botorch.generation.gen", "get_best_candidates"),
    "manual_seed": ("botorch.utils", "manual_seed"),
}

__all__ = list(_LAZY_IMPORTS.keys())


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:  # pragma: no cover
        if name in _LAZY_IMPORTS:
            abs_module, attr = _LAZY_IMPORTS[name]
            module = importlib.import_module(abs_module)
            if attr is None:
                value = module
            else:
                value = getattr(module, attr)
            globals()[name] = value
            return value
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__() -> list[str]:  # pragma: no cover
        return __all__
