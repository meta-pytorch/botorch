# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Summarise one stage of results/raw/v2/<stage>/ into SUMMARY.json.

Runs after every stage AND is safe to run mid-stage: it summarises
whatever shards exist,
records how many are missing, and never fails on a partial directory.
That is the point --
if the machine dies at hour 30 of 40, everything completed so far is already reduced to
numbers and nothing has to be re-run to find out what was learned.

Scored per .llms/rules/metrics.md: budget-to-target first, best-so-far regret second.
Regret-AUC is deliberately NOT emitted as a headline field.

Downstream, stage 2 and stage 3 read these files to choose their own
cells, which is what
lets the pipeline run unattended.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)


def _atomic_json(path: str, payload) -> None:
    """tmp + fsync + os.replace, so no reader can observe a partial file."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".partial")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def budget_to_target(
    traj: list[list[float]], pool_max: list[float], tol: float
) -> float:
    """Mean evaluations to reach absolute regret <= tol, censored at the budget."""
    tot, n = 0.0, 0
    for r, row in enumerate(traj):
        horizon = len(row)
        hit = horizon  # censored: one past the last index
        for t, v in enumerate(row):
            if pool_max[r] - v <= tol:
                hit = t
                break
        tot += hit
        n += 1
    return tot / n if n else float("nan")


def summarise_bo(files: list[str]) -> dict:
    """Pool shard files into per-method metrics.

    Rows are accumulated PER METHOD alongside that method's own pool_max slice. An
    earlier version appended pool_max once per file but traj rows per method, so a shard
    missing one method left that method's rows misaligned against pool_max -- every
    regret silently computed against the wrong run's optimum (review, metric #2).
    """
    traj: dict[str, list[list[float]]] = defaultdict(list)
    pmax: dict[str, list[float]] = defaultdict(list)
    ds_idx: list[int] = []
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            # One truncated shard must not cost the other eleven (review N/B3).
            continue
        pr = d.get("per_run", {})
        if "traj_raw" not in pr:
            continue
        file_pmax = pr.get("pool_max_raw", [])
        ds_idx += pr.get("eval_dataset_idx", [])
        for m, rows in pr["traj_raw"].items():
            if len(rows) != len(file_pmax):
                # Cannot be aligned; skipping is the only safe action.
                continue
            traj[m] += rows
            pmax[m] += file_pmax
    out: dict[str, dict] = {}
    for m, rows in traj.items():
        pool_max = pmax[m]
        assert len(pool_max) == len(rows), f"{m}: pool_max/traj misaligned"
        final = [pool_max[r] - rows[r][-1] for r in range(len(rows))]
        out[m] = {
            "n_runs": len(rows),
            "budget_to_0.01": budget_to_target(rows, pool_max, 0.01),
            "budget_to_0.001": budget_to_target(rows, pool_max, 0.001),
            "final_regret_mean": sum(final) / len(final) if final else float("nan"),
            "solve_rate_0.01": sum(1 for v in final if v <= 0.01) / len(final)
            if final
            else float("nan"),
        }
    return {"kind": "bo", "n_tasks": len(set(ds_idx)), "methods": out}


def summarise_regression(files: list[str]) -> dict:
    """bo_diagnose output.

    Reads the keys bo_diagnose ACTUALLY writes. An earlier version looked under
    ``summary`` / ``results`` / ``metrics``, none of which exist in that schema, so
    every regression cell reduced to ``{"methods": {}}`` while the .DONE gate accepted
    it as complete. The whole regression half of the matrix -- including stage 4.1,
    "does regression predict BO" -- would have run for hours and produced nothing
    (round-6 P1). Verified against bo_diagnose's writers AND against a real artifact
    (``results/raw/diagnose.json``): ``prior_mean`` is ``{method: {metric: value}}`` and
    ``by_n_obs`` is ``{n_obs: {method: {metric: value}}}`` -- both dicts keyed by method.
    An earlier version claimed to have verified this and had not; it read ``by_n_obs`` as
    a list and dropped all of it.
    """
    methods: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_read = 0
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        n_read += 1
        for m, vals in (d.get("prior_mean") or {}).items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    if isinstance(v, (int, float)):
                        methods[m][f"prior_{k}"].append(float(v))
        for n_obs, rows in (d.get("by_n_obs") or {}).items():
            # bo_diagnose writes {n_obs: {method: {metric: value}}} -- a DICT keyed by
            # method, not a list of rows carrying a "method" field. Reading it as a list
            # silently dropped EVERY by_n_obs metric, i.e. the entire
            # ranking-quality-vs-#observations analysis, while prior_mean still parsed
            # so the emptiness guard never fired (round-7 R7-2). Verified against
            # results/raw/diagnose.json.
            if not isinstance(rows, dict):
                continue
            for m, row in rows.items():
                if not isinstance(row, dict):
                    continue
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        methods[m][f"n{n_obs}_{k}"].append(float(v))
    out = {
        m: {k: sum(v) / len(v) for k, v in ks.items() if v} for m, ks in methods.items()
    }
    if n_read and not out:
        # Never look complete while carrying nothing -- the .DONE gate keys on `kind`.
        return {
            "kind": "error",
            "error": "bo_diagnose output contained no prior_mean/by_n_obs entries",
        }
    return {"kind": "regression", "methods": out}


def resolve_stage_dir(stage: str) -> str:
    """Locate a stage's directory, tolerating either naming convention.

    ``run_cell_queue.sh`` writes to ``results/raw/$STUDY`` with STUDY passed verbatim,
    while this module used to hardcode ``results/raw/v2/<stage>``. The orchestrator makes
    them agree only by convention (``STUDY="v2/$tag"`` alongside ``--stage "$tag"``), so a
    caller using either form alone silently summarised nothing -- and because the
    orchestrator tolerates a failed summarise with ``|| true``, the .DONE gate would then
    fail forever on the missing SUMMARY.json.

    Found by the end-to-end smoke; seven review rounds and 83 unit tests missed it,
    because both halves are individually correct and only their coupling is wrong.
    """
    candidates = [
        os.path.join(ROOT, "results/raw/v2", stage),
        os.path.join(ROOT, "results/raw", stage),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "no stage directory found; tried:\n  " + "\n  ".join(candidates)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--expect-shards", type=int, default=0)
    args = ap.parse_args()

    try:
        stage_dir = resolve_stage_dir(args.stage)
    except FileNotFoundError as exc:
        print(exc)
        return

    prov_path = os.path.join(stage_dir, "PROVENANCE.json")
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            prov = json.load(f)

    cells: dict[str, list[str]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(stage_dir, "*_s*.json"))):
        base = os.path.basename(path).rsplit("_s", 1)[0]
        cells[base].append(path)

    # Cells are named "<config>_p<k>", one per pre-training prior (gen_stage_queue
    # --priors). Grouping by cell alone therefore leaves every prior in its own bucket
    # and NOTHING pools them -- which would make the between-prior variance identically
    # zero again, the very thing --priors was added to fix (round-7 self-check).
    configs: dict[str, list[str]] = defaultdict(list)
    for cell, paths in cells.items():
        configs[re.sub(r"_p\d+$", "", cell)].extend(paths)

    summary = {
        "stage": args.stage,
        "commit": prov.get("commit"),
        "regime": prov.get("regime"),
        "cells_expected": len(prov.get("cells", [])) or None,
        "cells_present": len(cells),
        "cells": {},
    }
    for name, files in sorted(cells.items()):
        try:
            with open(files[0]) as f:
                probe = json.load(f)
            is_bo = "traj_raw" in probe.get("per_run", {})
            body = summarise_bo(files) if is_bo else summarise_regression(files)
        except Exception as exc:  # partial/corrupt shard must not kill the summary
            body = {"kind": "error", "error": repr(exc)}
        body["shards_present"] = len(files)
        if args.expect_shards:
            body["shards_missing"] = max(0, args.expect_shards - len(files))
        summary["cells"][name] = body

    # Pooled across priors. This is the level every claim should be read at: a single
    # prior cannot rank these methods (S27.3), and the between-prior spread is the
    # dominant variance component (S27.21).
    summary["configs"] = {}
    for cfg, files in sorted(configs.items()):
        n_priors = len(
            {
                re.sub(r".*_p(\d+)_s\d+\.json$", r"\1", os.path.basename(f))
                for f in files
            }
        )
        try:
            with open(files[0]) as f:
                probe = json.load(f)
            is_bo = "traj_raw" in probe.get("per_run", {})
            body = summarise_bo(files) if is_bo else summarise_regression(files)
        except Exception as exc:
            body = {"kind": "error", "error": repr(exc)}
        body["n_priors"] = n_priors
        body["shards_present"] = len(files)
        # R7-8: the docstring says this view exists BECAUSE the between-prior spread is
        # the dominant variance component, then emitted only point estimates. Compute it
        # here so --priors actually buys something an artifact can carry.
        # Between-prior spread, for BOTH kinds. Regression cells replicate across priors
        # too, and stage 4.1 correlates regression metrics against BO metrics -- doing
        # that with an uncertainty estimate on one side and none on the other would make
        # the correlation uninterpretable. An earlier version computed this for BO only,
        # which the end-to-end smoke exposed: smoke_reg carried no `between_prior`.
        if n_priors > 1 and body.get("kind") in ("bo", "regression"):
            is_bo = body.get("kind") == "bo"
            # The metric each kind is summarised on: budget-to-target is primary for BO
            # (.llms/rules/metrics.md); regression has no single primary, so every
            # numeric key gets its own spread.
            spread = {}
            for m in body.get("methods", {}):
                per_prior: dict[str, list[float]] = defaultdict(list)
                for k in range(n_priors):
                    kf = [f for f in files if f"_p{k}_s" in os.path.basename(f)]
                    if not kf:
                        continue
                    sub = (
                        (summarise_bo(kf) if is_bo else summarise_regression(kf))
                        .get("methods", {})
                        .get(m)
                    )
                    if not sub:
                        continue
                    keys = ["budget_to_0.01"] if is_bo else list(sub)
                    for key in keys:
                        v = sub.get(key)
                        if isinstance(v, (int, float)):
                            per_prior[key].append(float(v))
                entry = {}
                for key, vals in per_prior.items():
                    if len(vals) > 1:
                        mu = sum(vals) / len(vals)
                        var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
                        entry[key] = {
                            "per_prior": vals,
                            "between_prior_var_of_mean": var / len(vals),
                        }
                if entry:
                    spread[m] = entry
            if spread:
                body["between_prior"] = spread
        if n_priors < 2:
            body["warning"] = (
                "single prior: between-prior variance is unestimable, so any SE from "
                "this config understates the true uncertainty"
            )
        summary["configs"][cfg] = body

    out_path = os.path.join(stage_dir, "SUMMARY.json")
    # Rewritten after every cell, by concurrent writers, and read by the .DONE gate.
    # A non-atomic write here let the gate see a truncated file and re-run a
    # completed stage (review N16).
    _atomic_json(out_path, summary)

    done = sum(
        1
        for c in summary["cells"].values()
        if not c.get("shards_missing") and c.get("kind") != "error"
    )
    print(f"{args.stage}: {len(cells)} cells present, {done} complete -> {out_path}")
    for name, c in sorted(summary["cells"].items()):
        if c.get("kind") == "bo":
            best = min(
                (
                    (v.get("budget_to_0.01", math.inf), m)
                    for m, v in c["methods"].items()
                    if m.startswith("em")
                ),
                default=(math.inf, "-"),
            )
            print(f"  {name:26s} best EM budget@0.01 = {best[0]:6.2f} ({best[1]})")


if __name__ == "__main__":
    main()
