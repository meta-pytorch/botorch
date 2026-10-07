#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for the FULL PD1 Neural Net Tuning Dataset (Wang et al. 2021).

The internal ``pd1_data.py`` reads **PD1Lite**, a reduced subset mirrored to internal object storage as
Parquet. PD1Lite yields only ~50 hyperparameter configurations shared across all tasks
(~73 within the batch-size-256 subgroup), and §12.4/§12.5 of EXPERIMENTAL_RESULTS.md
localized our failure to reproduce the paper's PD1 result to exactly that: too few
matched configs, not the method. This module reads the **full public release** instead
(https://arxiv.org/abs/2109.08215, CC-BY 4.0), which shares an identical 400-point
Halton grid across all 23 evaluable tasks -- an 8x larger matched pool.

The release is JSON Lines, one row per trial, split over matched/unmatched x phase0/1.
Only the *matched* files are used here, since the empirical prior needs a shared pool.

Conventions, all of which change the numbers and so are stated explicitly:

- **Objective** ``best_valid/error_rate``: the best validation error over the training
  curve, which is what the paper optimizes. Returned negated so that higher is better,
  matching the harness (and LCBench accuracy).
- **Diverged trials with no evaluation** (1348 of ~11.5k matched rows, all of them
  ``status == "diverged"``) are assigned error 1.0 rather than dropped. A configuration
  that blows up before producing a single eval is maximally bad, and dropping them would
  silently shrink the matched pool and delete precisely the divergence tail that makes
  PD1 hard (§9.3).
- **Phase de-duplication**: 2277 (task, config) pairs appear in both collection phases.
  Phase 1 is preferred as the scaled-up primary run, falling back to phase 0.
- **Inputs** follow the documented PD1 search space: ``initial_value`` and
  ``one_minus_momentum`` are log-transformed, then all four are min-max scaled to [0,1]
  using the *declared* bounds rather than the observed min/max, so that a subgroup and
  the full task set produce an identical embedding.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

# Where the extracted PD1 release lives. The four ``*.jsonl.gz`` files are
# expected directly inside this directory.
#
# The release is not redistributed with this code: it is a separate public
# download (CC-BY 4.0) accompanying Wang et al. 2021, arXiv:2109.08215. Fetch it
# from the release linked in that paper and extract it here, or point
# ``PD1_DATA_ROOT`` at wherever you extracted it.
DEFAULT_PD1_ROOT: str = os.environ.get(
    "PD1_DATA_ROOT",
    str(Path.home() / ".cache" / "botorch" / "pd1"),
)

#: Files this loader reads. Only the *matched* files are required; the empirical
#: prior needs a configuration pool shared across tasks.
REQUIRED_PD1_FILES: tuple[str, ...] = (
    "pd1_matched_phase0_results.jsonl.gz",
    "pd1_matched_phase1_results.jsonl.gz",
)

# (column, lower, upper, log_scale) -- the PD1 search space, matching
# pd1_data.get_pd1_search_space() including parameter ORDER.
PARAM_SPECS: tuple[tuple[str, float, float, bool], ...] = (
    ("hps.lr_hparams.initial_value", 1e-5, 1e1, True),
    ("hps.lr_hparams.power", 0.1, 2.0, False),
    ("hps.opt_hparams.one_minus_momentum", 1e-3, 1e0, True),
    ("hps.lr_hparams.decay_steps_factor", 0.01, 0.99, False),
)
DIVERGED_ERROR = 1.0  # assigned to diverged trials with no recorded evaluation
_ROUND = 10  # decimals used to key a configuration


def _raw_params(row: dict) -> tuple[float, ...] | None:
    """The four search-space coordinates of a row, or None if any is absent."""
    mom = row.get("hps.opt_hparams.momentum")
    vals = []
    for name, _, _, _ in PARAM_SPECS:
        v = (1.0 - mom) if name.endswith("one_minus_momentum") else row.get(name)
        if v is None:
            return None
        vals.append(float(v))
    return tuple(round(v, _ROUND) for v in vals)


def _read_matched(root: str, phases: tuple[int, ...] = (0, 1)) -> dict:
    """``{study_group: {config_tuple: error_rate}}`` from the matched JSONL files."""
    return _read(root, ("matched",), phases)


def _read(root: str, kinds: tuple[str, ...], phases: tuple[int, ...] = (0, 1)) -> dict:
    """``{study_group: {config_tuple: error_rate}}`` over the requested files."""
    per_phase: dict[tuple[str, int], dict] = {}
    for kind in kinds:
        for ph in phases:
            path = os.path.join(root, f"pd1_{kind}_phase{ph}_results.jsonl.gz")
            if not os.path.exists(path):
                if kind == "unmatched":
                    continue  # unmatched phases are optional
                raise FileNotFoundError(
                    f"{path} not found.\n\n"
                    "The PD1 Neural Net Tuning Dataset is a separate public download "
                    "(CC-BY 4.0) and is not redistributed with this code. Obtain it "
                    "from the release accompanying Wang et al. 2021, "
                    "arXiv:2109.08215, extract it, and then either:\n"
                    f"  - extract into {DEFAULT_PD1_ROOT}, or\n"
                    "  - set PD1_DATA_ROOT to the extracted directory, or\n"
                    "  - pass --pd1-root explicitly.\n"
                    "The directory must contain "
                    "pd1_matched_phase{0,1}_results.jsonl.gz."
                )
            table: dict = defaultdict(dict)
            with gzip.open(path, "rt") as f:
                for line in f:
                    row = json.loads(line)
                    key = _raw_params(row)
                    if key is None:
                        continue
                    err = row.get("best_valid/error_rate")
                    if err is None:
                        # Only ever true for status == "diverged"; see docstring.
                        err = DIVERGED_ERROR
                    table[row["study_group"]][key] = float(err)
            per_phase[(kind, ph)] = table
    merged: dict = defaultdict(dict)
    for k in per_phase:  # later entries win, so phase 1 overrides phase 0
        for group, cfgs in per_phase[k].items():
            merged[group].update(cfgs)
    return merged


def dataset_names(root: str = DEFAULT_PD1_ROOT) -> list[str]:
    """Sorted ``study_group`` identifiers present in the matched data."""
    return sorted(_read_matched(root).keys())


def load_pd1_full_pool(
    root: str = DEFAULT_PD1_ROOT,
    dataset_names_filter: list[str] | None = None,
    dtype: torch.dtype = torch.double,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Matched-config pool over the full PD1 release.

    Args:
        root: Directory holding the extracted release.
        dataset_names_filter: Restrict to these ``study_group``s; default is all.
        dtype: dtype of the returned tensors.

    Returns:
        ``(Xn, Y_all, names)`` with ``Xn`` an ``(N, 4)`` tensor normalized to [0,1],
        ``Y_all[i]`` an ``(N, 1)`` tensor of ``-error_rate`` (higher is better) for task
        ``names[i]``, and rows aligned across tasks (the shared matched grid).
    """
    table = _read_matched(root)
    names = (
        sorted(table)
        if dataset_names_filter is None
        else [n for n in sorted(table) if n in set(dataset_names_filter)]
    )
    if len(names) < 2:
        raise ValueError(f"need >=2 PD1 tasks, got {len(names)}: {names}")
    shared = set.intersection(*(set(table[n].keys()) for n in names))
    if not shared:
        raise RuntimeError("matched-config intersection is empty across the tasks")
    keys = sorted(shared)

    X = torch.tensor(keys, dtype=dtype)  # (N, 4) in raw units, PARAM_SPECS order
    for j, (_, lo, hi, log_scale) in enumerate(PARAM_SPECS):
        col = X[:, j]
        if log_scale:
            col, lo, hi = (
                col.clamp_min(1e-300).log(),
                torch.tensor(lo).log().item(),
                torch.tensor(hi).log().item(),
            )
        X[:, j] = ((col - lo) / (hi - lo)).clamp(0.0, 1.0)
    Y_all = [
        -torch.tensor([table[n][k] for k in keys], dtype=dtype).unsqueeze(-1)
        for n in names
    ]
    return X, Y_all, names


def summarize(root: str = DEFAULT_PD1_ROOT) -> str:
    """One-line-per-task summary, for logging what a run actually loaded."""
    X, Y_all, names = load_pd1_full_pool(root)
    out = [
        f"PD1 (full): {len(names)} tasks, {X.shape[0]} matched configs, d={X.shape[1]}"
    ]
    for n, Y in zip(names, Y_all):
        err = -Y.squeeze(-1)
        out.append(
            f"  {n.split(',')[0][:26]:<28} bs={n.split(',')[-1]:<5} "
            f"err min={err.min():.4f} med={err.median():.4f} "
            f"diverged={100.0 * (err >= 0.8).double().mean():.1f}%"
        )
    return "\n".join(out)


def _normalize(X: torch.Tensor) -> torch.Tensor:
    """Map raw PARAM_SPECS-ordered coordinates into [0,1] using the DECLARED bounds.

    Declared rather than observed bounds, so a per-task pool and the matched grid land
    in the same coordinate system -- required for pre-training on one and searching the
    other.
    """
    X = X.clone()
    for j, (_, lo, hi, log_scale) in enumerate(PARAM_SPECS):
        col = X[:, j]
        if log_scale:
            col = col.clamp_min(1e-300).log()
            lo, hi = math.log(lo), math.log(hi)
        X[:, j] = ((col - lo) / (hi - lo)).clamp(0.0, 1.0)
    return X


def load_pd1_task_pools(
    root: str = DEFAULT_PD1_ROOT,
    dataset_names_filter: list[str] | None = None,
    include_unmatched: bool = True,
    dtype: torch.dtype = torch.double,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Per-task pools over ALL of a task's configurations, not the shared intersection.

    Wang et al. search each task's own design space -- "roughly 500 matched datapoints
    and 1500 unmatched datapoints per tuning task" -- while the matched intersection is
    only ~400. Restricting BO to the intersection makes the problem strictly easier: it
    saturates within 50 iterations and every method returns the same configuration, so
    the ordering carries no signal (§21.2). This returns the real per-task pools.

    Returns ``(X_list, Y_list, names)`` with ragged ``X_list[i]`` of shape ``(n_i, 4)``
    normalized to [0,1] and ``Y_list[i]`` of shape ``(n_i, 1)`` holding ``-error_rate``.
    """
    kinds = ("matched", "unmatched") if include_unmatched else ("matched",)
    table = _read(root, kinds)
    names = (
        sorted(table)
        if dataset_names_filter is None
        else [n for n in sorted(table) if n in set(dataset_names_filter)]
    )
    X_list, Y_list = [], []
    for n in names:
        keys = sorted(table[n])
        X_list.append(_normalize(torch.tensor(keys, dtype=dtype)))
        Y_list.append(
            -torch.tensor([table[n][k] for k in keys], dtype=dtype).unsqueeze(-1)
        )
    return X_list, Y_list, names
