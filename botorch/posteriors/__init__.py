#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.posteriors.gpytorch import GPyTorchPosterior  # noqa: F401
    from botorch.posteriors.higher_order import HigherOrderGPPosterior  # noqa: F401
    from botorch.posteriors.multitask import MultitaskGPPosterior  # noqa: F401
    from botorch.posteriors.posterior import Posterior  # noqa: F401
    from botorch.posteriors.posterior_list import PosteriorList  # noqa: F401
    from botorch.posteriors.torch import TorchPosterior  # noqa: F401
    from botorch.posteriors.transformed import TransformedPosterior  # noqa: F401

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "GPyTorchPosterior": (".gpytorch", "GPyTorchPosterior"),
    "HigherOrderGPPosterior": (".higher_order", "HigherOrderGPPosterior"),
    "MultitaskGPPosterior": (".multitask", "MultitaskGPPosterior"),
    "Posterior": (".posterior", "Posterior"),
    "PosteriorList": (".posterior_list", "PosteriorList"),
    "TorchPosterior": (".torch", "TorchPosterior"),
    "TransformedPosterior": (".transformed", "TransformedPosterior"),
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
