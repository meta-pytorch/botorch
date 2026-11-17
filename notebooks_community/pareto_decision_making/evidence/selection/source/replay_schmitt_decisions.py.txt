"""Replay final decisions on fixed MOBO archives; never fit or evaluate a model.

The exact-feedback counterfactual uses the *same designs* as each source log.
Linear and quadratic-distance utilities are declared separately, each with its
own matching ordinary reference. Preference corruption is independent of the
source logs' objective-observation corruption. This is descriptive simulation
evidence about selection, not a human study or a new optimization benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyse_boba_adaptations as aba
from replay_mo_front import parse_run_name

PROBLEMS = ("branincurrin", "zdt1", "dtlz2", "vehiclesafety")
SEEDS = (12, 13, 14, 15, 16)
RULES = ("weighted", "nearest_ideal", "knee_weighted")
FAMILIES = ("linear", "quadratic_distance")
REFERENCE = {"linear": "weighted", "quadratic_distance": "nearest_ideal"}
WEIGHT_NOISE = (0.0, 0.25, 1.0)
BASE_RNG_SEED = 20261021


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_costs(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paper range normalization, using only perceived nondominated outcomes."""
    from botorch.utils.multi_objective import is_non_dominated

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or not Y.size or not np.isfinite(Y).all():
        raise ValueError("Outcomes must be a nonempty finite n-by-m matrix.")
    mask = is_non_dominated(torch.as_tensor(Y, dtype=torch.double)).numpy()
    C = -Y
    lo, hi = C[mask].min(0), C[mask].max(0)
    scale = np.maximum(np.maximum(abs(lo), abs(hi)), 1.0)
    span = hi / scale - lo / scale
    Z = (C / scale - lo / scale) / np.where(span > 0, span, 1.0)
    Z[:, span == 0] = 0
    return Z, mask


def preference_bank(
    m: int, seed: int, n_random: int = 24
) -> tuple[np.ndarray, list[str]]:
    """Shared streams for equal-dimensional problems; all preferences predeclared."""
    rng = np.random.default_rng(np.random.SeedSequence([BASE_RNG_SEED, seed, m, 1]))
    P = np.concatenate(
        [np.full((1, m), 1 / m), np.eye(m), rng.dirichlet(np.ones(m), size=n_random)]
    )
    return P, ["equal"] + ["axis"] * m + ["dirichlet"] * n_random


def corrupt_preferences(P: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Multiplicative log-weight error; exact zero weights remain zero."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Preference log-noise SD must be finite and nonnegative.")
    if sigma == 0:
        return P.copy()
    rng = np.random.default_rng(
        np.random.SeedSequence([BASE_RNG_SEED, seed, P.shape[1], 2])
    )
    logits = np.where(P > 0, np.log(np.maximum(P, np.finfo(float).tiny)), -np.inf)
    logits = logits + sigma * rng.normal(size=P.shape)
    logits -= logits.max(1, keepdims=True)
    Q = np.exp(logits)
    return Q / Q.sum(1, keepdims=True)


def choices(
    Y: np.ndarray, P: np.ndarray, alpha: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from openbo.decision_making import knee_region, select_preferred

    Z, front = normalized_costs(Y)
    weighted = np.where(front[None], P @ Z.T, np.inf).argmin(1)
    ideal = np.where(front[None], P @ np.square(Z).T, np.inf).argmin(1)
    tensor_Y = torch.as_tensor(Y, dtype=torch.double)
    knee = knee_region(tensor_Y, alpha=alpha).numpy()
    # Call the implemented public selector, retaining original row indices.
    selected = select_preferred(
        tensor_Y.expand(len(P), *tensor_Y.shape),
        torch.as_tensor(P, dtype=torch.double),
        alpha=alpha,
    ).numpy()
    return {
        "weighted": weighted,
        "nearest_ideal": ideal,
        "knee_weighted": selected,
    }, knee


def archive_decisions(
    Y_true: np.ndarray,
    Y_observed: np.ndarray,
    *,
    seed: int,
    alpha: float = 0.5,
    n_random: int = 24,
) -> pd.DataFrame:
    """All selectors share candidates; truth is used only for scoring/counterfactuals."""
    Z_true, true_front = normalized_costs(Y_true)
    # Duplicate rows are not truly dominated. Perceived front deduplication
    # affects normalization/choices but must not inflate this diagnostic.
    from botorch.utils.multi_objective import is_non_dominated

    true_front = is_non_dominated(
        torch.as_tensor(Y_true, dtype=torch.double), deduplicate=False
    ).numpy()
    if Y_observed.shape != Y_true.shape or not np.isfinite(Y_observed).all():
        raise ValueError("Observed and true outcomes must have the same finite shape.")
    P, labels = preference_bank(Y_true.shape[1], seed, n_random)
    clean_choices, _ = choices(Y_true, P, alpha)
    true_scores = {
        "linear": P @ Z_true.T,
        "quadratic_distance": P @ np.square(Z_true).T,
    }
    optimum = {family: scores.min(1) for family, scores in true_scores.items()}
    rows: list[dict[str, Any]] = []
    for observation_mode, Y in (("exact", Y_true), ("recorded", Y_observed)):
        for sigma in WEIGHT_NOISE:
            Q = corrupt_preferences(P, sigma, seed)
            selected, knee = choices(Y, Q, alpha)
            for family, scores in true_scores.items():
                for rule, indices in selected.items():
                    for j, index in enumerate(indices):
                        clean_index = int(clean_choices[rule][j])
                        regret = float(scores[j, index] - optimum[family][j])
                        clean_regret = float(
                            scores[j, clean_index] - optimum[family][j]
                        )
                        if regret < -1e-10 or clean_regret < -1e-10:
                            raise AssertionError("Selection regret cannot be negative.")
                        rows.append(
                            {
                                "observation_mode": observation_mode,
                                "preference_noise": sigma,
                                "utility_family": family,
                                "rule": rule,
                                "preference_id": j,
                                "preference_type": labels[j],
                                "chosen_index": int(index),
                                "clean_index": clean_index,
                                "regret": max(regret, 0.0),
                                "clean_regret": max(clean_regret, 0.0),
                                "choice_changed": int(index != clean_index),
                                "chosen_true_dominated": int(not true_front[index]),
                                "knee_fraction": float(knee.mean()),
                                "oracle_best_excluded": int(
                                    not np.any(
                                        knee
                                        & np.isclose(
                                            scores[j],
                                            optimum[family][j],
                                            atol=1e-10,
                                            rtol=0,
                                        )
                                    )
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def discover(input_dir: Path) -> list[tuple[Path, Any]]:
    selected = []
    for path in sorted(input_dir.glob("*/bo_sensor_error_*_multi_objective_*.csv")):
        run = parse_run_name(path.name)
        if run is None or run.dataset not in PROBLEMS or run.seed not in SEEDS:
            continue
        if run.acquisition != "qlognehvi":
            continue
        if (
            run.kind == "clean"
            and run.suffix == ""
            or (
                run.kind == "noisy"
                and run.channel == "gaussian"
                and run.jitter_std == 1
                and run.jitter_iteration in (0, 20)
                and run.suffix in ("", "_xc0.85")
            )
        ):
            selected.append((path, run))
    expected = len(PROBLEMS) * len(SEEDS) * 5
    if len(selected) != expected:
        raise ValueError(f"Fixed plan requires {expected} logs, found {len(selected)}.")
    keys = {
        (r.dataset, r.seed, r.kind, r.jitter_iteration, r.suffix) for _, r in selected
    }
    if len(keys) != expected:
        raise ValueError("Duplicate archives in the fixed plan.")
    return selected


def read_archive(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    frame = pd.read_csv(path)
    if len(frame) != 50 or frame["iteration"].tolist() != list(range(1, 51)):
        raise ValueError(f"{path.name}: expected 50 ordered trials.")
    true_cols = [c for c in frame if c.startswith("objective_true_f")]
    true_cols.sort(key=lambda c: int(c.rsplit("f", 1)[-1]))
    obs_cols = [c.replace("objective_true", "objective_observed") for c in true_cols]
    if len(true_cols) not in (2, 3) or any(c not in frame for c in obs_cols):
        raise ValueError(f"{path.name}: missing vector outcomes.")
    failed = frame["acq_opt_failed"].astype(str).str.lower().eq("true").sum()
    if failed:
        raise ValueError(
            f"{path.name}: {failed} acquisition failures; do not silently include."
        )
    return frame[true_cols].to_numpy(float), frame[obs_cols].to_numpy(float), len(frame)


def comparisons(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, landscapes = [], []
    grouping = [
        "measurement_setting",
        "observation_mode",
        "preference_noise",
        "utility_family",
    ]
    metrics = metrics.copy()
    noisy_pool = metrics.loc[metrics.source_kind == "noisy"].assign(
        measurement_setting="pooled_noisy_archives"
    )
    metrics = pd.concat([metrics, noisy_pool], ignore_index=True)
    rng = np.random.default_rng(BASE_RNG_SEED + 4)
    for key, block in metrics.groupby(grouping, sort=True):
        tags = dict(zip(grouping, key, strict=True))
        family = tags["utility_family"]
        reference = block.loc[block.rule == REFERENCE[family]]
        for rule in RULES:
            treatment = block.loc[block.rule == rule]
            pairs = aba.paired_frame(
                reference,
                treatment,
                "utility_regret",
                {name: 1.0 for name in PROBLEMS},
                pool=False,
            )
            # The response is already dimensionless, in a fixed per-archive
            # normalized utility. No hypervolume or analytic-optimum scale enters.
            summary = aba.summarise(pairs, rng)
            summary.pop("wilcoxon_p", None)  # Four correlated problems: no stars/tests.
            per = aba.per_dataset(pairs)
            nearzero = summary["cost"] < 0.01 or (per["cost"] < 0).any()
            if nearzero:
                for col in ("recovered", "recovered_lo", "recovered_hi"):
                    summary[col] = float("nan")
            summary["recovery_suppressed"] = bool(nearzero)
            for col in ("ref_noisy", "ref_clean", "trt_noisy", "trt_clean"):
                summary[col] = float(pairs.groupby("dataset")[col].mean().mean())
            draws = rng.integers(0, len(per), size=(2000, len(per)))
            for col in ("gain", "price"):
                means = per[col].to_numpy()[draws].mean(1)
                summary[f"{col}_lo"], summary[f"{col}_hi"] = np.percentile(
                    means, [2.5, 97.5]
                )
            summaries.append(
                {**tags, "reference_rule": REFERENCE[family], "rule": rule, **summary}
            )
            for row in per.to_dict("records"):
                landscapes.append(
                    {**tags, "reference_rule": REFERENCE[family], "rule": rule, **row}
                )
    return pd.DataFrame(summaries), pd.DataFrame(landscapes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output-boba-mo-halo"))
    parser.add_argument("--openbo-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    paths = discover(args.input_dir.resolve())
    plan = {
        "status": "fixed before outcome scoring; exploratory, not confirmatory",
        "problems": PROBLEMS,
        "seeds": SEEDS,
        "acquisition": "qlognehvi",
        "source_logs": len(paths),
        "trials_per_archive": 50,
        "objective_error": "logged Gaussian SD1, onsets0/20, cross-correlation0/.85; plus clean archives",
        "rules": RULES,
        "alpha": 0.5,
        "utility_families": FAMILIES,
        "reference_rules": REFERENCE,
        "random_preferences_per_seed": 24,
        "additional_preferences": "equal and each objective axis",
        "preference_log_noise_sd": WEIGHT_NOISE,
        "zero_weight_policy": "multiplicative noise preserves exact zero weights",
        "normalization": "decision inputs use their perceived PF ranges; evaluator freezes true PF ranges within each archive",
        "clean_comparison": "same archive, exact objective outcomes and true preferences, for each rule",
        "scope": "final selection only; no new designs, evaluations, GP fits, or acquisition comparisons",
        "inference": "four correlated problem clusters; descriptive problem bootstrap only; no significance tests",
        "command": [sys.executable, *sys.argv],
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty output directory; results are immutable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    sys.path.insert(0, str(args.openbo_src.resolve()))
    import openbo.decision_making as dm

    selector_dir = Path(dm.__file__).parent
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "botorch", "numpy", "pandas", "scipy")
    }
    manifest = {
        "python": sys.version,
        "versions": versions,
        "script_sha256": sha256(Path(__file__)),
        "selector_source_sha256": {
            p.name: sha256(p) for p in sorted(selector_dir.glob("*.py"))
        },
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path, _ in paths],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    torch.set_num_threads(1)
    frames = []
    for number, (path, run) in enumerate(paths, 1):
        truth, observed, n_trials = read_archive(path)
        if run.kind == "clean" and not np.array_equal(truth, observed):
            raise ValueError(f"{path.name}: clean archive is not exact.")
        result = archive_decisions(truth, observed, seed=run.seed)
        corr = 0.85 if run.suffix == "_xc0.85" else 0.0
        setting = (
            "clean_optimization_archive"
            if run.kind == "clean"
            else f"gaussian1_onset{run.jitter_iteration}_corr{corr:g}"
        )
        for name, value in {
            "dataset": run.dataset,
            "acquisition": run.acquisition,
            "seed": run.seed,
            "source_kind": run.kind,
            "source_file": path.name,
            "n_trials": n_trials,
            "error_model": "none" if run.kind == "clean" else f"gaussian_corr{corr:g}",
            "jitter_std": run.jitter_std or 0.0,
            "jitter_iteration": run.jitter_iteration or 0,
            "measurement_setting": setting,
        }.items():
            result[name] = value
        frames.append(result)
        if number % 10 == 0:
            print(f"Scored {number}/{len(paths)} fixed archives", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(
        args.output_dir / "per_preference_decisions.csv.gz",
        index=False,
        compression="gzip",
    )
    keys = [
        "dataset",
        "acquisition",
        "seed",
        "source_kind",
        "source_file",
        "measurement_setting",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "observation_mode",
        "preference_noise",
        "utility_family",
        "rule",
    ]
    metrics = raw.groupby(keys, as_index=False).agg(
        utility_regret_jitter=("regret", "mean"),
        utility_regret_baseline=("clean_regret", "mean"),
        choice_changed_rate=("choice_changed", "mean"),
        true_dominated_rate=("chosen_true_dominated", "mean"),
        knee_fraction=("knee_fraction", "mean"),
        oracle_best_excluded_rate=("oracle_best_excluded", "mean"),
        n_preferences=("preference_id", "size"),
    )
    metrics.to_csv(args.output_dir / "per_archive_metrics.csv", index=False)
    summary, per_problem = comparisons(metrics)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    per_problem.to_csv(args.output_dir / "comparison_per_problem.csv", index=False)
    preference_groups = raw.groupby(
        [
            "dataset",
            "measurement_setting",
            "observation_mode",
            "preference_noise",
            "utility_family",
            "preference_type",
            "rule",
        ],
        as_index=False,
    ).agg(
        regret=("regret", "mean"),
        clean_regret=("clean_regret", "mean"),
        change_rate=("choice_changed", "mean"),
    )
    preference_groups.to_csv(args.output_dir / "preference_groups.csv", index=False)
    head = summary.loc[
        (summary.measurement_setting == "pooled_noisy_archives")
        & (summary.observation_mode == "recorded")
        & (summary.preference_noise == 0),
        ["utility_family", "rule", "trt_noisy", "gain", "price", "gain_lo", "gain_hi"],
    ]
    report = (
        f"Fixed-archive selection replay: {len(paths)} logs, {len(PROBLEMS)} problems, seeds {SEEDS}.\n"
        f"{len(raw)} preference/family/condition/rule rows; no new objective evaluations or fits.\n\n"
        "Headline: recorded measurement error, exact preference weights, all noisy archives pooled.\n"
        "Units: final selected design's regret in within-archive true-PF normalized utility.\n"
        "gain = matching ordinary reference noisy regret - treatment noisy regret.\n"
        "price = treatment exact-feedback regret - matching ordinary reference exact-feedback regret.\n\n"
        + head.to_string(index=False, float_format=lambda x: f"{x:.6f}")
        + "\n\nSee comparison_per_problem.csv: only four problem clusters, correlated source streams.\n"
        "Intervals are descriptive problem bootstraps treating these clusters as independent; no p-values/stars.\n"
        "Truth families are declared separately. Schmitt uses linear preferences even under quadratic truth;\n"
        "its resulting clean loss is model/decision-rule mismatch, not mislabeled observation error.\n"
        "Preference noise is a separate illustrative multiplicative log-weight process; axis weights remain axes.\n"
        "Objective errors are the archive's Gaussian1 process; no bias/drift/AR1 extrapolation is supported.\n"
        "Same candidates across selectors/counterfactuals; different source logs can have different candidates.\n"
        "Archive normalization is oracle-only for scoring and is not a deployable source of true ranges.\n"
        "These endpoints concern selection among visited designs, not global regret, Pareto coverage or human benefit.\n"
    )
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
