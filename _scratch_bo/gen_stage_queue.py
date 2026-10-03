# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate cell queues for the EXPERIMENT_PLAN.md stages, and stamp provenance.

Every stage writes into ``results/raw/v2/<stage>/`` -- the ``v2/`` prefix is the hard
boundary between results produced by the current binary and everything from before
2026-08-17, which the S27.15 RNG-order change made incomparable. Nothing outside ``v2/``
may be compared with anything inside it.

Each stage also gets a ``PROVENANCE.json`` recording the commit, the date and the exact
cell list, so a future agent can tell what produced a directory without
this conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)

# The five regime cells of the plan. `harness` selects the binary; `flags` are the
# regime-defining arguments that every cell in that regime shares.
REGIMES: dict[str, dict[str, str]] = {
    "lcb_bo": {
        "harness": "bo_experiment",
        "flags": "--benchmark lcbench --n-configs 200 --n-iters 40 --n-seeds 1 "
        "--n-eval 6 --n-pretrain 25 --meta-iters 2000 --hyperbo-iters 25000 "
        "--ablr-iters 25000",
        "shards": "1",
    },
    "pd1_bo_armC": {
        "harness": "bo_experiment",
        "flags": "--benchmark pd1_loo --pd1-source full --pd1-group all "
        "--pd1-candidate-pool full --pd1-pretrain-pool full --n-iters 50 --n-seeds 1 "
        "--n-init 3 --per-task-standardize --output-warp neglog --meta-subsample 64 "
        "--meta-task-batch 5 --hyperbo-iters 25000",
        "shards": "12",
    },
    "pd1_bo_armA": {
        "harness": "bo_experiment",
        "flags": "--benchmark pd1_loo --pd1-source full --pd1-group all "
        "--pd1-candidate-pool matched --pd1-pretrain-pool matched --n-iters 50 "
        "--n-seeds 1 --n-init 3 --per-task-standardize --output-warp neglog "
        "--meta-subsample 64 --meta-task-batch 5 --hyperbo-iters 25000",
        "shards": "12",
    },
    # Arm B completes the 2x2. A and C differ in TWO flags at once, so they
    # confound "off-grid candidates" with "off-grid pre-training". B holds
    # pre-training matched (as in A) while searching the full pool (as in C):
    #   A vs B  -> isolates the INTERPOLATION effect (pre-training fixed)
    #   B vs C  -> isolates the PRE-TRAINING POOL effect (candidates fixed)
    # S27.34 attributes EM's off-grid deficit to Nystrom extrapolation; that is
    # currently INFERRED from a two-flag contrast, and this measures it.
    "pd1_bo_armB": {
        "harness": "bo_experiment",
        "flags": "--benchmark pd1_loo --pd1-source full --pd1-group all "
        "--pd1-candidate-pool full --pd1-pretrain-pool matched --n-iters 50 "
        "--n-seeds 1 --n-init 3 --per-task-standardize --output-warp neglog "
        "--meta-subsample 64 --meta-task-batch 5 --hyperbo-iters 25000",
        "shards": "12",
    },
    "lcb_reg": {
        "harness": "bo_diagnose",
        "flags": "--n-configs 300 --n-pretrain 25 --n-eval 10 --n-obs-grid 5,20,50 "
        "--meta-iters 25000",
        "shards": "1",
    },
    "pd1_reg": {
        "harness": "bo_diagnose",
        "flags": "--benchmark pd1_loo --pd1-source full --n-obs-grid 5,20,50 "
        "--meta-iters 25000",
        "shards": "1",
    },
}

BASELINE_METHODS = (
    "em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,ablr,vanilla_gp,random"
)
EM_METHODS = "em_frozen,em_additive_hyperbo,em_additive_hyperbo_frozen,hyperbo_frozen"

# A dedicated MEAN sweep, because OFAT structurally cannot answer the question. OFAT
# varies one factor at a time, so it only ever compares each mean variant against `base`
# -- it never crosses the two mean axes, and it never checks whether a mean that is inert
# on-grid matters off-grid. S27.31 found every mean transfer null on LCBench (|t| <= 0.76),
# but LCBench is on-grid and S27.29 showed on-grid/off-grid is the axis that organises
# these results, so a null there says little about arm C.
#
# The two axes:
#   --em-base-mean {constant,linear}   the parametric base m(.) fitted by the shared MLL
#   --em-mean {empirical,hyperbo,blend} what m(.) is replaced or blended with
#
# --em-mean != empirical needs a HyperBO prior to transfer FROM, hence --em-canonical
# hyperbo on those rows (round-6 R6: omitting it makes the cell raise at runtime).
#
# THAT REQUIREMENT IS ALSO A CONFOUND, and it invalidated the first reading of this
# sweep.
# Because --em-canonical hyperbo appears on exactly the rows whose --em-mean is not
# empirical, the kernel and the mean transfer are perfectly confounded: on arm C the
# ranking is *precisely* the canonical split (all 6 hyperbo-canonical rows 35.4-36.3,
# all 4 sumll rows 39.5-39.9, between-group t=-3.57) while every within-mean contrast is
# null (blend weight |t| <= 0.92, base mean t=-1.13, mean prior t=+0.37). The data
# attributes the whole off-grid effect to the kernel at least as well as to the mean
# (S28.3). `mean_const_hbk` is the missing cell that breaks it: canonical=hyperbo with
# the mean left empirical, so the kernel moves alone. Do not remove it.
MEANS = [
    ("mean_const_emp", "--em-base-mean constant"),
    ("mean_lin_emp", "--em-base-mean linear"),
    (
        "mean_const_hbk",
        "--em-base-mean constant --em-canonical hyperbo",
    ),
    (
        "mean_const_hb",
        "--em-base-mean constant --em-canonical hyperbo --em-mean hyperbo",
    ),
    ("mean_lin_hb", "--em-base-mean linear --em-canonical hyperbo --em-mean hyperbo"),
    (
        "mean_const_b25",
        "--em-base-mean constant --em-canonical hyperbo --em-mean blend "
        "--em-mean-weight 0.25",
    ),
    (
        "mean_const_b50",
        "--em-base-mean constant --em-canonical hyperbo --em-mean blend "
        "--em-mean-weight 0.5",
    ),
    (
        "mean_const_b75",
        "--em-base-mean constant --em-canonical hyperbo --em-mean blend "
        "--em-mean-weight 0.75",
    ),
    (
        "mean_lin_b50",
        "--em-base-mean linear --em-canonical hyperbo --em-mean blend "
        "--em-mean-weight 0.5",
    ),
    ("mean_const_prior", "--em-base-mean constant --use-mean-prior"),
    ("mean_lin_prior", "--em-base-mean linear --use-mean-prior"),
]


# EM pre-training noise sweep (S27.35 / S27.36). `likelihood_noise` was hardcoded at 1e-2
# with no flag and never fitted; a scan put the optimum near sigma^2 = 1-3, worth -28% RMSE,
# and S27.36 confirmed that is genuine regularisation rather than EM degenerating into its
# base kernel (which only happens near sigma^2 = 100). It is a VARIANCE, so on
# unit-variance standardized data 1e-2 means sigma = 0.1.
#
# The grid brackets the optimum on both sides so the inverted U is measurable rather than
# assumed, and includes 1e-2 so every cell has an in-sweep comparison against the old
# default. S27.34 showed these effects are regime-dependent, so this runs on all regimes.
NOISE = [
    ("emnoise_1e-3", "--em-likelihood-noise 0.001"),
    ("emnoise_1e-2", "--em-likelihood-noise 0.01"),
    ("emnoise_1e-1", "--em-likelihood-noise 0.1"),
    ("emnoise_0.3", "--em-likelihood-noise 0.3"),
    ("emnoise_1", "--em-likelihood-noise 1.0"),
    ("emnoise_3", "--em-likelihood-noise 3.0"),
    ("emnoise_10", "--em-likelihood-noise 10.0"),
]


# The Inverse-Wishart nu axis: is nu better ESTIMATED from the pre-training tasks than
# fixed by hand? --iw-nu-mode offers OAS and Ledoit-Wolf empirical-Bayes estimators.
#
# TWO TRAPS, both of which this list exists to avoid:
#
# 1. SILENT NO-OP. resolve_iw_nu() returns args.iw_nu unchanged unless --use-covar-prior
#    is set, and NEITHER pd1_bo_armA NOR pd1_bo_armC sets it. Passing --iw-nu-mode oas
#    on those regimes alone would run the estimator never, produce cells numerically
#    identical to the control, and read as "OAS makes no difference" -- a measurement
#    with an instrument that cannot register the effect. Every entry below therefore
#    carries --use-covar-prior explicitly.
# 2. CONFOUNDED CONTROL. If the control were the bare regime (covar prior OFF) and the
#    treatments had it ON, the contrast would be "covar prior on/off" and not "how nu is
#    chosen". The control below holds --use-covar-prior ON and varies only the mode, so
#    the nu axis is the single thing that moves.
# S31 killed the iw-nu route: Psi is scaled BY nu, so the realised intensity is floored at
# 0.970 and a 9x range of intended alpha collapsed to 0.0012. covariance_shrinkage takes
# alpha directly in [0,1], decoupled from N_inducing, so it is the knob that can actually
# express what the estimators recommend.
#
# NO --use-covar-prior anywhere here, deliberately. S31.3 measured the IW prior at
# +7.60 +/- 1.88 evals WORSE on armA, and leaving it on would pin alpha at 0.97 underneath
# whatever this sweep sets. shrink000 is therefore the true control and should reproduce
# stage1 base.
#
# The grid is concentrated LOW because 31.3 showed heavy shrinkage to Sigma_init is
# harmful here; the two estimator cells are included to test whether the analytic alpha
# (~0.13 spherical, ~0.65 whitened) lands anywhere near the empirical optimum.
SHRINK = [
    ("shrink000", "--em-shrinkage 0.0"),
    ("shrink005", "--em-shrinkage 0.05"),
    ("shrink010", "--em-shrinkage 0.10"),
    ("shrink020", "--em-shrinkage 0.20"),
    ("shrink030", "--em-shrinkage 0.30"),
    ("shrink_oas", "--em-shrinkage-mode oas"),
    ("shrink_whiten", "--em-shrinkage-mode oas_whitened --iw-target-ridge 0.05"),
]

# S32 follow-ups. Both reuse existing flags, so neither needs a harness change.
#
# RIDGE: S32 measured whitened alpha at 0.633 against an empirical optimum near 0.10 and
# it HURT (+2.13 +/- 1.00). tau=1 recovers plain OAS exactly, and plain OAS (alpha=0.133)
# WON (-1.73 +/- 1.33), so the ridge is the dial between a setting that hurts and one that
# helps. If the whitened target has any value it must show up between them; if the curve
# is monotone toward tau=1 then whitening adds nothing here and should be dropped.
#
# GRID: S32 bracketed the em_frozen optimum at alpha=0.10 (-2.19 +/- 1.04 at tol 0.01,
# -3.14 +/- 1.64 at 0.001) but did not resolve it -- 0.05 and 0.20 are both worse and the
# spacing is coarse. These fill in around the minimum. alpha=0 is repeated as the control
# so this stage is self-contained rather than relying on a cross-stage comparison.
SHRINK2 = [
    ("s2_ctrl", "--em-shrinkage 0.0"),
    ("s2_a008", "--em-shrinkage 0.08"),
    ("s2_a012", "--em-shrinkage 0.12"),
    ("s2_a015", "--em-shrinkage 0.15"),
    ("s2_tau020", "--em-shrinkage-mode oas_whitened --iw-target-ridge 0.20"),
    ("s2_tau050", "--em-shrinkage-mode oas_whitened --iw-target-ridge 0.50"),
    ("s2_tau080", "--em-shrinkage-mode oas_whitened --iw-target-ridge 0.80"),
    ("s2_oas", "--em-shrinkage-mode oas"),
]

IWNU = [
    ("iwnu_manual", "--use-covar-prior"),
    ("iwnu_oas", "--use-covar-prior --iw-nu-mode oas"),
    ("iwnu_ledoit", "--use-covar-prior --iw-nu-mode ledoit_wolf"),
    # S30.7. The three above all estimate alpha against a SPHERICAL target, but
    # Psi = (nu+N+1)*K(Z,Z) under the default init_mode="kernel" -- so the intensity is
    # chosen for one target and applied to another. These two whiten by K so the
    # estimator sees the target actually in force. Two ridges because the simulation
    # optimum (0.05) and the point where a BAD target stops hurting (0.20) differ, and
    # the real kernel's quality on PD1 is unknown.
    (
        "iwnu_whiten005",
        "--use-covar-prior --iw-nu-mode oas_whitened --iw-target-ridge 0.05",
    ),
    (
        "iwnu_whiten020",
        "--use-covar-prior --iw-nu-mode oas_whitened --iw-target-ridge 0.20",
    ),
]

# Stage 1 is one-factor-at-a-time off a common baseline: every entry changes exactly one
# thing. Anything that cannot be shown to change the output is dropped before it runs.
OFAT: list[tuple[str, str]] = [
    ("base", ""),
    # A -- canonical / interpolation kernel
    ("canon_hyperbo", "--em-canonical hyperbo"),
    ("canon_deep", "--deep-kernel 32,32"),
    # B -- base mean (never varied before 2026-08-17)
    ("mean_linear", "--em-base-mean linear"),
    # C -- mean transfer from HyperBO
    # --em-mean needs a HyperBO prior. Without --em-canonical hyperbo these raise at
    # runtime on the bo_diagnose regimes, and a cell that never writes shards blocks
    # its stage's .DONE forever (round-4 R6).
    ("meanxfer_full", "--em-canonical hyperbo --em-mean hyperbo"),
    ("meanxfer_blend", "--em-canonical hyperbo --em-mean blend --em-mean-weight 0.5"),
    # D -- ad-hoc covariance shrinkage. S27.9 found it unhelpful on PD1;
    # re-tested here post-fix.
    ("shrink_0.1", "--em-shrinkage 0.1"),
    ("shrink_0.3", "--em-shrinkage 0.3"),
    # E -- principled covariance prior (never run; largest single-shard NLL move)
    ("covprior", "--use-covar-prior"),
    ("covprior_nu", "--use-covar-prior --iw-nu 500"),
    # F -- mean prior
    ("meanprior", "--use-mean-prior"),
    # G -- EM initialisation
    ("init_naive", "--em-init-mode naive"),
    # Promising combinations, in the screen because S27.16 flagged B and E
    # as the
    # two largest single-shard movers and they are expected to interact.
    ("mean_linear+covprior", "--em-base-mean linear --use-covar-prior"),
    (
        "mean_linear+meanxfer",
        "--em-base-mean linear --em-canonical hyperbo --em-mean hyperbo",
    ),
]


def _atomic_json(path: str, payload) -> None:
    """tmp + fsync + os.replace."""
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


def sl_commit() -> str:
    try:
        return subprocess.run(
            ["sl", "log", "-r", ".", "-T", "{node}"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=60,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_provenance(stage: str, regime: str, cells: list[tuple[str, str]]) -> str:
    out_dir = os.path.join(ROOT, "results/raw/v2", stage)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "PROVENANCE.json")
    prov = {
        "stage": stage,
        "regime": regime,
        "commit": sl_commit(),
        "generated_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
        ).stdout.strip(),
        "regime_flags": REGIMES[regime]["flags"],
        "harness": REGIMES[regime]["harness"],
        "cells": [{"name": n, "flags": f} for n, f in cells],
        "note": (
            "Everything under results/raw/v2/ was produced by the post-2026-08-17 "
            "binary. The S27.15 RNG-order change makes it incomparable with any result "
            "outside v2/. Do not mix them."
        ),
    }
    existing = []
    if os.path.exists(path):
        # A half-written or older-schema provenance file must not wedge the stage
        # forever: this used to raise and make queue generation fail on every
        # subsequent run (review N18).
        try:
            with open(path) as f:
                prior = json.load(f)
            existing = list(prior.get("history", []))
            existing.append({k: prior.get(k) for k in ("commit", "generated_utc")})
        except (OSError, ValueError):
            existing = [{"commit": "unreadable", "generated_utc": "unknown"}]
    prov["history"] = existing
    _atomic_json(path, prov)
    return path


def select_top_configs(stage: str, k: int = 3) -> list[tuple[str, str]]:
    """Read a completed stage's SUMMARY.json and return its best k configs.

    This is what makes stages 2 and 3 autonomous: they choose their own cells from the
    previous stage's artifact instead of waiting for a human. Selection is on the
    POOLED `configs` view, never on individual `cells`, because a single prior cannot
    rank these methods (S27.3).

    Ranked on the primary metric per .llms/rules/metrics.md -- the best EM variant's
    budget-to-target at 0.01, lower being better. Configs whose summary is an error, or
    which carry only one prior, are skipped rather than ranked: a single-prior
    number has
    no estimable between-prior variance (S27.21) and would win by looking artificially
    precise.
    """
    path = os.path.join(ROOT, "results/raw/v2", stage, "SUMMARY.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"cannot select stage-2/3 cells: {path} does not exist. Run the earlier "
            "stage first; refusing to silently fall back to a default cell list."
        )
    with open(path) as f:
        summary = json.load(f)

    scored: list[tuple[float, str]] = []
    skipped: list[str] = []
    for name, body in (summary.get("configs") or {}).items():
        if body.get("kind") == "error" or body.get("n_priors", 0) < 2:
            skipped.append(name)
            continue
        # Regression summaries have no budget_to_target -- they carry rank_corr / nll /
        # rmse per n_obs. Ranking them on the BO metric found nothing and (correctly)
        # raised, which is why stage3_lcb_reg and stage3_pd1_reg failed on the first
        # autonomous run. Pick the metric from the summary's own kind, and keep the
        # direction explicit: budget is lower-better, rank correlation is higher-better.
        methods = (body.get("methods") or {}).items()
        if body.get("kind") == "regression":
            # Sort by the PARSED n_obs, not lexicographically. sorted() on the raw keys
            # returns 'n20_rank_corr' before 'n5_rank_corr' ('2' < '5'), so the previous
            # keys[0] silently selected n_obs=20 while this comment claimed n_obs=5
            # (§28.6). Inert until a regression stage 3 runs, which is exactly when it
            # would have mis-selected without any visible symptom.
            def _n_obs(k: str) -> int:
                return int(k.split("_", 1)[0][1:])

            keys = sorted(
                {k for _m, v in methods for k in v if k.endswith("_rank_corr")},
                key=_n_obs,
            )
            if not keys:
                skipped.append(name)
                continue
            key = keys[0]  # smallest n_obs: the regime BO actually operates in
            em = [
                -v[key]  # negate so that, as for budget, smaller is better
                for m, v in methods
                if m.startswith("em") and isinstance(v.get(key), (int, float))
            ]
        else:
            em = [
                v.get("budget_to_0.01")
                for m, v in methods
                if m.startswith("em")
                and isinstance(v.get("budget_to_0.01"), (int, float))
            ]
        if em:
            scored.append((min(em), name))

    if not scored:
        raise ValueError(
            f"{stage}: no rankable configs (skipped {len(skipped)}: {skipped[:5]}). "
            "Refusing to proceed with an empty or arbitrary selection."
        )
    scored.sort()
    chosen = [n for _, n in scored[:k]]

    # Recover each winner's flags from the stage's own provenance, so stage 2/3 rerun
    # exactly what stage 1 ran rather than a reconstruction that could drift.
    prov_path = os.path.join(ROOT, "results/raw/v2", stage, "PROVENANCE.json")
    flags_by_config: dict[str, str] = {}
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            prov = json.load(f)
        for cell in prov.get("cells", []):
            base = re.sub(r"_p\d+$", "", cell.get("name", ""))
            # Strip the prior seed; the caller re-expands across priors.
            flags_by_config.setdefault(
                base, re.sub(r"\s*--pretrain-seed \d+", "", cell.get("flags", ""))
            )
    missing = [c for c in chosen if c not in flags_by_config]
    if missing:
        raise ValueError(
            f"{stage}: winners {missing} have no provenance entry, so their flags "
            "cannot be reproduced. Refusing to guess."
        )
    return [(c, flags_by_config[c]) for c in chosen]


def source_prior_seeds(stage: str) -> set[int]:
    """The --pretrain-seed values a completed stage actually ran.

    Read from the stage's own PROVENANCE.json rather than assumed from --priors, so a
    stage that was launched with a different prior count still reports the truth.
    """
    path = os.path.join(ROOT, "results/raw/v2", stage, "PROVENANCE.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        prov = json.load(f)
    seeds = set()
    for cell in prov.get("cells", []):
        m = re.search(r"--pretrain-seed (\d+)", cell.get("flags", ""))
        if m:
            seeds.add(int(m.group(1)))
            continue
        m = re.search(r"_p(\d+)$", cell.get("name", ""))
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--regime", required=True, choices=sorted(REGIMES))
    ap.add_argument(
        "--kind",
        default="ofat",
        choices=[
            "baseline",
            "ofat",
            "replicate",
            "means",
            "noise",
            "iwnu",
            "shrink",
            "shrink2",
        ],
    )
    ap.add_argument(
        "--select-from",
        type=str,
        default=None,
        help="For --kind replicate: the completed stage whose SUMMARY.json chooses "
        "the cells (e.g. stage1_pd1_bo_armC). This is what makes stages 2/3 "
        "autonomous.",
    )
    ap.add_argument(
        "--top-k", type=int, default=3, help="How many winners to replicate."
    )
    ap.add_argument(
        "--priors",
        type=int,
        default=9,
        help="Number of pre-training priors per cell. Each becomes its own cell "
        "with --pretrain-seed k, and the analysis pools them. Nothing in the "
        "matrix varied this before, so every result was a SINGLE prior and the "
        "between-prior variance term was identically zero (round-6 P3). S27.3 "
        "showed one prior cannot rank these methods, so 1 is only for smoke runs. "
        "Default 9 with --n-seeds 1: at a fixed budget B = P*n, "
        "Var = sB^2/P + sW^2/B, so the second term depends only on B and the "
        "first only on P -- more priors is free variance reduction, and it "
        "lifts the critical value from t(2)=4.30 to t(8)=2.31 (S27.24).",
    )
    ap.add_argument("--out", required=True, help="queue TSV to write")
    ap.add_argument(
        "--prior-offset",
        type=int,
        default=None,
        help="First --pretrain-seed to use. Priors run offset..offset+priors-1. For "
        "--kind replicate this defaults to one past the source stage's highest seed, "
        "because a replicate that reuses the source's seeds is not a replicate: "
        "stage 3 re-ran stage 1's flags at --pretrain-seed 0..8 against deterministic "
        "code and reproduced all 675 shards byte-for-byte, certifying nothing (S28.1).",
    )
    args = ap.parse_args()

    if args.kind == "baseline":
        base = [("baseline", f"--methods {BASELINE_METHODS}")]
    elif args.kind == "noise":
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in NOISE]
    elif args.kind == "iwnu":
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in IWNU]
    elif args.kind == "shrink":
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in SHRINK]
    elif args.kind == "shrink2":
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in SHRINK2]
    elif args.kind == "means":
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in MEANS]
    elif args.kind == "replicate":
        if not args.select_from:
            raise ValueError("--kind replicate requires --select-from <stage>")
        base = select_top_configs(args.select_from, k=args.top_k)
    else:
        base = [(n, f"{f} --methods {EM_METHODS}".strip()) for n, f in OFAT]

    # Expand each configuration across priors. One prior per cell keeps the shard
    # layout unchanged and lets PooledStudy pool them afterwards.
    offset = args.prior_offset
    source_seeds: set[int] = set()
    if args.kind == "replicate":
        source_seeds = source_prior_seeds(args.select_from)
        if offset is None:
            offset = (max(source_seeds) + 1) if source_seeds else args.priors
    offset = offset or 0

    if args.priors <= 1:
        cells = base
        used = {offset} if offset else set()
    else:
        used = set(range(offset, offset + args.priors))
        cells = [
            (f"{n}_p{k}", f"{f} --pretrain-seed {k}".strip())
            for n, f in base
            for k in sorted(used)
        ]

    # A replicate whose seeds overlap its source re-runs deterministic code on identical
    # inputs and reproduces it exactly. That is not a measurement, and it silently looks
    # like perfect stability (S28.1). Refuse rather than warn.
    clash = used & source_seeds
    if args.kind == "replicate" and clash:
        raise ValueError(
            f"{args.stage}: prior seeds {sorted(clash)} were already used by "
            f"{args.select_from}. A replicate must draw DISJOINT priors or it "
            "reproduces the source byte-for-byte and certifies nothing (S28.1). "
            "Pass --prior-offset to choose an unused range."
        )

    with open(args.out, "w") as f:
        f.write(f"# stage={args.stage} regime={args.regime} kind={args.kind}\n")
        for name, flags in cells:
            f.write(f"{name}\t{flags}\n")

    prov = write_provenance(args.stage, args.regime, cells)
    print(f"wrote {len(cells)} cells -> {args.out}")
    print(f"provenance -> {prov}")
    print(f"shards={REGIMES[args.regime]['shards']}")


if __name__ == "__main__":
    sys.exit(main())
