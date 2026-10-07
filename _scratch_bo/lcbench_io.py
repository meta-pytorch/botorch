#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""LCBench access for the scratch BO experiments, routed through internal object storage.

The scripts here used to import ``load_lcbench_data`` from
``ax.fb.benchmark.problems.surrogate.lcbench.data``. That internal loader was
deliberately narrowed to the single Parquet read seam
(``read_lcbench_parquet_from_internal_store``) so internal CI could not silently diverge from
the open-source data path -- which left every scratch script importing a symbol that no
longer exists. This module is the replacement: it exposes the **open-source**
``botorch.utils.lcbench`` loader and patches only the download seam, so column
selection, the log transform and the epoch truncation all stay on the OSS code path.

Devservers and CI hosts have no external network access, so the GitHub download in
``botorch.utils.lcbench._read_lcbench_parquet`` cannot be used; this is the same patch
that ``botorch_fb/test/test_tutorials.py`` applies for the tutorial job.

``LCBENCH_DATASET_NAMES`` is byte-identical (same order, same 35 entries) to the
``DATASET_NAMES`` the scripts previously took from ``ax.benchmark``, so dataset indices
-- and therefore the pretrain/eval splits drawn from a seeded ``randperm`` -- are
unchanged.
"""

from __future__ import annotations

import torch
from botorch.utils import lcbench as _lcbench
from botorch.utils.lcbench import (
    DEFAULT_LCBENCH_METRIC_NAME,
    LCBENCH_DATASET_NAMES,
    LCBenchData,
)

__all__ = [
    "DEFAULT_LCBENCH_METRIC_NAME",
    "LCBENCH_DATASET_NAMES",
    "LCBenchData",
    "load_lcbench_data",
    "use_internal_store_reader",
]

_PATCHED = False


def use_internal_store_reader() -> None:
    """Point botorch's LCBench Parquet reader at the internal object-storage mirror."""
    global _PATCHED
    if _PATCHED:
        return
    # Local import: pulls the internal object storage client, which is only needed internally.
    from ax.fb.benchmark.problems.surrogate.lcbench.data import (
        read_lcbench_parquet_from_internal_store,
    )

    _lcbench._read_lcbench_parquet = read_lcbench_parquet_from_internal_store
    _PATCHED = True


def load_lcbench_data(
    dataset_name: str,
    metric_name: str = DEFAULT_LCBENCH_METRIC_NAME,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> LCBenchData:
    """``botorch.utils.lcbench.load_lcbench_data``, internal object storage reader installed."""
    use_internal_store_reader()
    return _lcbench.load_lcbench_data(
        dataset_name, metric_name, dtype=dtype, device=device
    )
