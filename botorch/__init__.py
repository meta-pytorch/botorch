#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch.settings as gp_settings
import linear_operator.settings as linop_settings
from botorch import (
    acquisition,
    exceptions,
    models,
    optim,
    posteriors,
    settings,
    test_functions,
)
from botorch.cross_validation import batch_cross_validation
from botorch.fit import fit_gpytorch_mll
from botorch.generation.gen import (
    gen_candidates_scipy,
    gen_candidates_torch,
    get_best_candidates,
)
from botorch.logging import logger
from botorch.utils import manual_seed

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


__all__ = [
    "acquisition",
    "batch_cross_validation",
    "exceptions",
    "fit_gpytorch_mll",
    "gen_candidates_scipy",
    "gen_candidates_torch",
    "get_best_candidates",
    "manual_seed",
    "models",
    "optim",
    "posteriors",
    "settings",
    "test_functions",
]
