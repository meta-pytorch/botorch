#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Loader for the LCBench learning-curve benchmark.

LCBench [Zimmer2021]_ records the learning curves of a neural network trained
under 2000 hyperparameter configurations on each of 35 OpenML datasets. The
data is distributed as GZIP-compressed Parquet files, one ``config`` and one
``metrics`` file per dataset, which this module downloads on demand and caches
locally.

.. [Zimmer2021]
    L. Zimmer, M. Lindauer, and F. Hutter. Auto-PyTorch: Multi-Fidelity
    MetaLearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern
    Analysis and Machine Intelligence, 2021.
"""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

try:
    # pyarrow is an implicit runtime dependency: it is only referenced through
    # `engine="pyarrow"` below, so autodeps cannot infer it from an import.
    # @dep=fbsource//third-party/pypi/pyarrow:pyarrow
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False


LCBENCH_DATASET_NAMES: tuple[str, ...] = (
    "APSFailure",
    "Amazon_employee_access",
    "Australian",
    "Fashion-MNIST",
    "KDDCup09_appetency",
    "MiniBooNE",
    "adult",
    "airlines",
    "albert",
    "bank-marketing",
    "blood-transfusion-service-center",
    "car",
    "christine",
    "cnae-9",
    "connect-4",
    "covertype",
    "credit-g",
    "dionis",
    "fabert",
    "helena",
    "higgs",
    "jannis",
    "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "nomao",
    "numerai28.6",
    "phoneme",
    "segment",
    "shuttle",
    "sylvine",
    "vehicle",
    "volkert",
)

# Canonical order of the 7 LCBench hyperparameters. The ``config`` frame holds
# additional columns (e.g. the OpenML task id), which are dropped.
LCBENCH_PARAMETER_NAMES: tuple[str, ...] = (
    "batch_size",
    "max_dropout",
    "max_units",
    "num_layers",
    "learning_rate",
    "momentum",
    "weight_decay",
)

# Parameters whose LCBench search space is log-scaled, and which are therefore
# log-transformed before being returned.
LCBENCH_LOG_SCALE_PARAMETER_NAMES: tuple[str, ...] = (
    "batch_size",
    "max_units",
    "learning_rate",
    "momentum",
)

DEFAULT_LCBENCH_METRIC_NAME = "Train/val_accuracy"

LCBENCH_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ltiao/LCBenchLite/main/"
    "{dataset_name}/{stem}.parquet.gzip"
)

# Resolved lazily via `_default_cache_dir()`; `expanduser()` is deliberately not
# called at module scope so that `$HOME` is read when the loader runs.
DEFAULT_LCBENCH_CACHE_DIR = "~/.cache/botorch/lcbench"

DEFAULT_LCBENCH_DOWNLOAD_TIMEOUT = 60.0


def _default_cache_dir() -> Path:
    return Path(DEFAULT_LCBENCH_CACHE_DIR).expanduser()


def _check_pandas_available() -> None:
    if not _HAS_PANDAS:
        raise ImportError(
            "BoTorch's LCBench loader requires pandas and pyarrow, which are "
            "optional dependencies. Please install them with "
            '`pip install "botorch[lcbench]"`.'
        )


@dataclass(frozen=True, kw_only=True)
class LCBenchData:
    r"""The learning curves of a single LCBench dataset.

    Args:
        parameters: A `n x 7`-dim tensor of hyperparameter configurations, with
            columns ordered as in `LCBENCH_PARAMETER_NAMES` and the columns in
            `LCBENCH_LOG_SCALE_PARAMETER_NAMES` log-transformed.
        metrics: A `n x 50`-dim tensor of metric values, one row per
            configuration and one column per epoch.
        epochs: A `50`-dim tensor holding the epoch indices `1, ..., 50` that
            the columns of `metrics` correspond to.
    """

    parameters: Tensor
    metrics: Tensor
    epochs: Tensor


def _read_lcbench_parquet(
    dataset_name: str, stem: str, cache_dir: Path
) -> pd.DataFrame:
    r"""Read one LCBench Parquet file, downloading and caching it if needed.

    Args:
        dataset_name: The name of the LCBench dataset, e.g. "Fashion-MNIST".
        stem: The file to read, either "config" or "metrics".
        cache_dir: The directory that cached files are read from and written to.
            Files are laid out as `<cache_dir>/<dataset_name>/<stem>.parquet.gzip`.

    Returns:
        The contents of the Parquet file as a DataFrame.
    """
    cache_path = cache_dir / dataset_name / f"{stem}.parquet.gzip"
    if cache_path.exists():
        return pd.read_parquet(cache_path, engine="pyarrow")
    url = LCBENCH_URL_TEMPLATE.format(dataset_name=dataset_name, stem=stem)
    with urllib.request.urlopen(
        url, timeout=DEFAULT_LCBENCH_DOWNLOAD_TIMEOUT
    ) as response:
        raw = response.read()
    # Parse before caching so that a truncated download or an HTML error body is
    # never persisted, and write via a temporary file in the destination
    # directory so a concurrent reader cannot observe a half-written cache entry.
    df = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent, suffix=".tmp", delete=False
    ) as tmp_file:
        tmp_file.write(raw)
        tmp_path = Path(tmp_file.name)
    os.replace(tmp_path, cache_path)
    return df


def load_lcbench_data(
    dataset_name: str,
    metric_name: str = DEFAULT_LCBENCH_METRIC_NAME,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
    cache_dir: Path | None = None,
) -> LCBenchData:
    r"""Load the learning curves of a single LCBench dataset.

    The Parquet files backing the requested dataset are downloaded from GitHub
    on first use and cached locally, so that subsequent calls require no
    network access.

    Args:
        dataset_name: The name of the LCBench dataset, which must be one of
            `LCBENCH_DATASET_NAMES`, e.g. "Fashion-MNIST".
        metric_name: The name of the recorded metric to extract, e.g.
            "Train/val_accuracy".
        dtype: The dtype of the returned tensors.
        device: The device of the returned tensors.
        cache_dir: The directory that downloaded files are cached in. Defaults
            to `DEFAULT_LCBENCH_CACHE_DIR`, expanded at call time.

    Returns:
        An `LCBenchData` holding the configurations and their learning curves.
    """
    _check_pandas_available()
    if dataset_name not in LCBENCH_DATASET_NAMES:
        raise ValueError(
            f"Invalid dataset '{dataset_name}'. Valid datasets: "
            f"{list(LCBENCH_DATASET_NAMES)}."
        )
    if cache_dir is None:
        cache_dir = _default_cache_dir()

    parameter_df = _read_lcbench_parquet(
        dataset_name=dataset_name, stem="config", cache_dir=cache_dir
    )
    metrics_df = _read_lcbench_parquet(
        dataset_name=dataset_name, stem="metrics", cache_dir=cache_dir
    )

    log_scale_names = list(LCBENCH_LOG_SCALE_PARAMETER_NAMES)
    parameter_df[log_scale_names] = parameter_df[log_scale_names].transform("log")
    parameter_df = parameter_df[list(LCBENCH_PARAMETER_NAMES)]

    # LCBench records 52 epochs [0, ..., 51]; drop the first and last to obtain
    # the 50 epochs [1, ..., 50] that the benchmark is defined over.
    metric_df = metrics_df[metric_name].unstack(level="epoch").iloc[:, 1:-1]

    metrics = torch.from_numpy(metric_df.values).to(dtype=dtype, device=device)
    return LCBenchData(
        parameters=torch.from_numpy(parameter_df.values).to(dtype=dtype, device=device),
        metrics=metrics,
        epochs=torch.arange(1, metrics.shape[-1] + 1, dtype=dtype, device=device),
    )
