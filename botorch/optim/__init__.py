#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.optim.closures import (  # noqa: F401
        ForwardBackwardClosure,
        get_loss_closure,
        get_loss_closure_with_grads,
    )
    from botorch.optim.core import (  # noqa: F401
        OptimizationResult,
        OptimizationStatus,
        scipy_minimize,
        torch_minimize,
    )
    from botorch.optim.homotopy import (  # noqa: F401
        FixedHomotopySchedule,
        Homotopy,
        HomotopyParameter,
        LinearHomotopySchedule,
        LogLinearHomotopySchedule,
    )
    from botorch.optim.initializers import (  # noqa: F401
        initialize_q_batch,
        initialize_q_batch_nonneg,
        initialize_q_batch_topn,
    )
    from botorch.optim.optimize import (  # noqa: F401
        gen_batch_initial_conditions,
        optimize_acqf,
        optimize_acqf_cyclic,
        optimize_acqf_discrete,
        optimize_acqf_discrete_local_search,
        optimize_acqf_mixed,
    )
    from botorch.optim.optimize_homotopy import optimize_acqf_homotopy  # noqa: F401
    from botorch.optim.optimize_mixed import (  # noqa: F401
        optimize_acqf_mixed_alternating,
    )
    from botorch.optim.stopping import ExpMAStoppingCriterion  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ForwardBackwardClosure": (".closures", "ForwardBackwardClosure"),
    "get_loss_closure": (".closures", "get_loss_closure"),
    "get_loss_closure_with_grads": (".closures", "get_loss_closure_with_grads"),
    "OptimizationResult": (".core", "OptimizationResult"),
    "OptimizationStatus": (".core", "OptimizationStatus"),
    "scipy_minimize": (".core", "scipy_minimize"),
    "torch_minimize": (".core", "torch_minimize"),
    "FixedHomotopySchedule": (".homotopy", "FixedHomotopySchedule"),
    "Homotopy": (".homotopy", "Homotopy"),
    "HomotopyParameter": (".homotopy", "HomotopyParameter"),
    "LinearHomotopySchedule": (".homotopy", "LinearHomotopySchedule"),
    "LogLinearHomotopySchedule": (".homotopy", "LogLinearHomotopySchedule"),
    "initialize_q_batch": (".initializers", "initialize_q_batch"),
    "initialize_q_batch_nonneg": (".initializers", "initialize_q_batch_nonneg"),
    "initialize_q_batch_topn": (".initializers", "initialize_q_batch_topn"),
    "gen_batch_initial_conditions": (".optimize", "gen_batch_initial_conditions"),
    "optimize_acqf": (".optimize", "optimize_acqf"),
    "optimize_acqf_cyclic": (".optimize", "optimize_acqf_cyclic"),
    "optimize_acqf_discrete": (".optimize", "optimize_acqf_discrete"),
    "optimize_acqf_discrete_local_search": (
        ".optimize",
        "optimize_acqf_discrete_local_search",
    ),
    "optimize_acqf_mixed": (".optimize", "optimize_acqf_mixed"),
    "optimize_acqf_homotopy": (".optimize_homotopy", "optimize_acqf_homotopy"),
    "optimize_acqf_mixed_alternating": (
        ".optimize_mixed",
        "optimize_acqf_mixed_alternating",
    ),
    "ExpMAStoppingCriterion": (".stopping", "ExpMAStoppingCriterion"),
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
