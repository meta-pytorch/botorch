#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregation + figures for the empirical-GP BO baseline study.

Reads the result JSONs written by ``bo_experiment.py`` and produces every summary
statistic and figure quoted in ``EXPERIMENTAL_RESULTS.md``. Previously this analysis
lived in ad-hoc chat-side scratch code and was lost with ``/tmp``; it is committed here
so every number and plot in the write-up is reproducible from the stored JSONs.

Statistics
----------
- Simple regret at arbitrary horizons, with **dataset-clustered SEM** (SEM over
  per-dataset means, so correlated seeds within a dataset are not counted as
  independent evidence).
- **Regret-AUC**: the mean regret over the whole trajectory. This is the quantity BO
  sample-efficiency actually lives in, and unlike final regret it does not saturate
  once a finite pool is exhausted.
- **Average ranks and win-rate.** Per run, methods are ranked by the metric (rank 1 =
  lowest regret) with **ties averaged**. ``avg_rank`` is the mean over runs;
  ``avg_rank_clustered`` first averages within each eval dataset and then over datasets
  (equal weight per dataset). ``win%`` gives a run's winning method 1 credit, split
  evenly on ties, so the column sums to 100.
- **Friedman test + Nemenyi critical difference** over runs (N = #runs, k = #methods),
  the standard Demsar (2006) protocol. Two methods differ significantly at alpha iff
  their average ranks differ by more than CD.

Figures
-------
- ``*_trajectories``  : full mean-regret trajectories with shaded clustered SEM.
- ``*_cd_<metric>``   : Demsar critical-difference diagram (cliques = not significant).
- ``*_perfprofile``   : Dolan-More performance profile on final regret.
- ``*_dataprofile``   : fraction of runs solved to a tolerance vs evaluation budget.
- ``*_cost``          : regret vs cumulative wall-clock (pre-training + online).

Usage
-----
    python3 -m _scratch_bo.analyze \
      --result results/lcbench_highpower.json --tag lcbench_highpower \
      --out-dir results/figures --horizons 1,5,10,20,40
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Studentized-range critical values q_alpha / sqrt(2) for the Nemenyi post-hoc test,
# indexed by the number of compared methods k (Demsar 2006, Table 5).
NEMENYI_Q = {
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
        16: 3.426,
        17: 3.458,
        18: 3.489,
        19: 3.517,
        20: 3.544,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
        11: 2.978,
        12: 3.030,
        13: 3.077,
        14: 3.120,
        15: 3.159,
        16: 3.196,
        17: 3.230,
        18: 3.261,
        19: 3.291,
        20: 3.319,
    },
}

# Stable colors/markers so a method looks the same in every figure.
STYLE: dict[str, tuple[str, str]] = {
    "em_frozen": ("#1f77b4", "o"),
    "em_finetuned": ("#4c9ed9", "s"),
    "em_noisefit": ("#7fc7ef", "^"),
    "em_additive_freshbase": ("#a6d8f0", "v"),
    "em_additive_3way": ("#c9e7f7", "<"),
    "hyperbo_frozen": ("#d62728", "D"),
    "hyperbo_adapt": ("#ff7f6b", "d"),
    "pacoh_frozen": ("#9467bd", "P"),
    "pacoh_adapt": ("#c5a5db", "X"),
    "ablr": ("#2ca02c", "*"),
    "vanilla_gp": ("#8c564b", "h"),
    "pretrained_gp_frozen": ("#bcbd22", "p"),
    "pretrained_gp_tuned": ("#dbdc6b", "8"),
    "warmstart_gp": ("#e377c2", "H"),
    "random": ("#7f7f7f", "x"),
}


def _style(method: str) -> tuple[str, str]:
    return STYLE.get(method, ("#333333", "o"))


# --------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------


class Study:
    """One or more ``bo_experiment.py`` result files, reshaped for analysis.

    Passing several comma-separated paths concatenates them, which is how leave-one-out
    shards (``--loo-shard i/n``) are recombined: the folds are disjoint, so the per-run
    records simply append and the timings sum.
    """

    def __init__(self, path: str) -> None:
        paths = [p.strip() for p in path.split(",") if p.strip()]
        raws = []
        for pth in paths:
            with open(pth) as f:
                raws.append(json.load(f))
        raw = raws[0]
        self.path = path
        self.config: dict = raw.get("config", {})
        self.eval_datasets: list[str] = raw.get("eval_datasets", [])
        self.calibration: dict = raw.get("calibration", {})
        self.methods: list[str] = list(raw["per_run"]["traj_raw"].keys())

        pool_max: list[float] = []
        ds_idx: list[int] = []
        traj: dict[str, list] = {m: [] for m in self.methods}
        self.timings: dict = {}
        for r in raws:
            pr = r["per_run"]
            pool_max += pr["pool_max_raw"]
            ds_idx += pr["eval_dataset_idx"]
            for m in self.methods:
                traj[m] += pr["traj_raw"][m]
            for k, v in r.get("timings", {}).items():
                if isinstance(v, (int, float)):
                    self.timings[k] = self.timings.get(k, 0) + v
                elif isinstance(v, dict):
                    acc = self.timings.setdefault(k, {})
                    for mk, mv in v.items():
                        acc[mk] = acc.get(mk, 0) + mv
            for ename in r.get("eval_datasets", []):
                if ename not in self.eval_datasets:
                    self.eval_datasets.append(ename)
        if len(raws) > 1:
            # Calibration is a mean over acquired points; recombine weighted by count.
            comb: dict = {}
            for r in raws:
                for m, c in r.get("calibration", {}).items():
                    a = comb.setdefault(
                        m, {"mean_nll": 0.0, "coverage95": 0.0, "n_points": 0}
                    )
                    n = c.get("n_points", 0)
                    a["mean_nll"] += c.get("mean_nll", 0.0) * n
                    a["coverage95"] += c.get("coverage95", 0.0) * n
                    a["n_points"] += n
            for a in comb.values():
                if a["n_points"]:
                    a["mean_nll"] /= a["n_points"]
                    a["coverage95"] /= a["n_points"]
            self.calibration = comb

        self.pool_max = np.asarray(pool_max, dtype=float)  # (R,)
        self.ds_idx = np.asarray(ds_idx, dtype=int)  # (R,)
        # regret[m] : (R, T) simple regret, raw objective units
        self.regret: dict[str, np.ndarray] = {}
        for m in self.methods:
            T = np.asarray(traj[m], dtype=float)  # (R, T) best-so-far
            self.regret[m] = self.pool_max[:, None] - T
        self.n_runs = int(self.pool_max.shape[0])
        self.n_pts = int(next(iter(self.regret.values())).shape[1])
        # Leave-one-out re-trains a prior per fold, so pre-training cost is per
        # task rather than one-time, and cannot be amortized across tasks.
        self.is_loo = str(self.config.get("benchmark", "")).endswith("_loo")
        self.n_priors = int(len(np.unique(self.ds_idx))) if self.is_loo else 1

    def at(self, method: str, horizon: int | None = None) -> np.ndarray:
        """Per-run regret at a horizon (``None`` = final)."""
        R = self.regret[method]
        h = R.shape[1] - 1 if horizon is None else min(horizon, R.shape[1] - 1)
        return R[:, h]

    def auc(self, method: str) -> np.ndarray:
        """Per-run regret-AUC = mean regret over the trajectory."""
        return self.regret[method].mean(axis=1)

    def clustered_sem(self, values: np.ndarray) -> float:
        """SEM over per-dataset means (seeds within a dataset are correlated)."""
        groups = [values[self.ds_idx == d] for d in np.unique(self.ds_idx)]
        means = np.array([g.mean() for g in groups if g.size])
        if means.size < 2:
            return float("nan")
        return float(means.std(ddof=1) / math.sqrt(means.size))


# --------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------


def rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Average-tie ranks along axis 1 (rank 1 = smallest value)."""
    order = matrix.argsort(axis=1, kind="stable")
    ranks = np.empty_like(matrix, dtype=float)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, matrix.shape[1] + 1)[None, :]
    # average ties
    for i in range(matrix.shape[0]):
        vals = matrix[i]
        for v in np.unique(vals):
            tie = vals == v
            if tie.sum() > 1:
                ranks[i, tie] = ranks[i, tie].mean()
    return ranks


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-squared distribution (regularized incomplete gamma)."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:  # series expansion for the lower regularized gamma
        term = 1.0 / a
        total = term
        n = 1
        while n < 10000:
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            n += 1
        lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    # Lentz continued fraction for the upper regularized gamma
    tiny = 1e-300
    b, c, d = z + 1.0 - a, 1.0 / tiny, 1.0 / (z + 1.0 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, h * math.exp(-z + a * math.log(z) - math.lgamma(a))))


def rank_stats(
    study: Study, methods: Sequence[str], metric: str = "final", horizon: int = 0
) -> dict:
    """Ranks, win-rate, Friedman chi-squared and the Nemenyi critical difference."""
    if metric == "auc":
        cols = [study.auc(m) for m in methods]
    elif metric == "final":
        cols = [study.at(m, None) for m in methods]
    else:  # "@k"
        cols = [study.at(m, horizon) for m in methods]
    M = np.column_stack(cols)  # (R, k)
    ranks = rank_rows(M)  # (R, k)
    k, N = len(methods), M.shape[0]

    # win-rate with ties splitting the credit, so the column sums to 100%
    wins = np.zeros(k)
    for i in range(N):
        best = M[i].min()
        tied = np.flatnonzero(M[i] == best)
        wins[tied] += 1.0 / tied.size
    win_pct = 100.0 * wins / N

    avg_rank = ranks.mean(axis=0)
    clustered = []
    for d in np.unique(study.ds_idx):
        clustered.append(ranks[study.ds_idx == d].mean(axis=0))
    avg_rank_clustered = np.mean(np.array(clustered), axis=0)

    # Friedman statistic on the average ranks
    chi2 = (12.0 * N / (k * (k + 1))) * (
        float((avg_rank**2).sum()) - k * (k + 1) ** 2 / 4.0
    )
    p = _chi2_sf(chi2, k - 1)
    cds = {}
    for alpha, table in NEMENYI_Q.items():
        q = table.get(k)
        cds[alpha] = None if q is None else q * math.sqrt(k * (k + 1) / (6.0 * N))
    return {
        "methods": list(methods),
        "metric": metric if metric != "at" else f"@{horizon}",
        "n_runs": N,
        "n_methods": k,
        "mean": {m: float(M[:, j].mean()) for j, m in enumerate(methods)},
        "clustered_sem": {
            m: study.clustered_sem(M[:, j]) for j, m in enumerate(methods)
        },
        "avg_rank": {m: float(avg_rank[j]) for j, m in enumerate(methods)},
        "avg_rank_clustered": {
            m: float(avg_rank_clustered[j]) for j, m in enumerate(methods)
        },
        "win_pct": {m: float(win_pct[j]) for j, m in enumerate(methods)},
        "friedman_chi2": float(chi2),
        "friedman_df": k - 1,
        "friedman_p": float(p),
        "nemenyi_cd": {str(a): cds[a] for a in cds},
    }


def cliques(ordered: list[tuple[str, float]], cd: float) -> list[tuple[int, int]]:
    """Maximal runs of rank-adjacent methods within CD of each other."""
    out: list[tuple[int, int]] = []
    n = len(ordered)
    for i in range(n):
        j = i
        while j + 1 < n and ordered[j + 1][1] - ordered[i][1] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


# --------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------


def fig_trajectories(
    study: Study, methods: Sequence[str], out: str, title: str, logy: bool = True
) -> None:
    """Mean simple regret vs evaluations, shaded with dataset-clustered SEM."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(study.n_pts)
    for m in methods:
        R = study.regret[m]  # (R, T)
        mean = R.mean(axis=0)
        sem = np.array([study.clustered_sem(R[:, t]) for t in range(R.shape[1])])
        color, marker = _style(m)
        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markersize=3.5,
            markevery=max(1, study.n_pts // 12),
            linewidth=1.6,
            label=m,
        )
        lower = mean - sem
        if logy:
            # A log axis cannot show a non-positive lower band; floor it just under
            # the smallest positive mean so the ribbon stays readable.
            floor = (
                max(np.min(mean[mean > 0]) * 0.5, 1e-12) if (mean > 0).any() else 1e-12
            )
            lower = np.maximum(lower, floor)
        ax.fill_between(x, lower, mean + sem, color=color, alpha=0.16, linewidth=0)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("evaluations" if study.config.get("batch_q", 1) == 1 else "rounds")
    ax.set_ylabel("simple regret (val-accuracy points)")
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out)


def fig_cd_diagram(stats: dict, out: str, title: str, alpha: float = 0.05) -> None:
    """Demsar critical-difference diagram: average ranks + non-significance cliques.

    Best rank fans out top-left, worst top-right; a thick bar under the axis joins
    methods whose average ranks differ by less than the critical difference (i.e. are
    statistically indistinguishable).
    """
    cd = stats["nemenyi_cd"].get(str(alpha))
    ordered = sorted(stats["avg_rank"].items(), key=lambda kv: kv[1])
    n = len(ordered)
    lo = math.floor(min(r for _, r in ordered) - 0.4)
    hi = math.ceil(max(r for _, r in ordered) + 0.4)
    span = max(hi - lo, 1)
    grps = cliques(ordered, cd) if cd else []

    row_gap, clique_gap = 0.62, 0.30
    n_side = (n + 1) // 2
    first_row = -(0.45 + clique_gap * max(len(grps), 1))
    bottom = first_row - row_gap * (n_side - 1) - 0.45
    top = 1.35 if cd else 0.85

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * n_side + 2.1))
    ax.set_xlim(lo - span * 0.62, hi + span * 0.62)
    ax.set_ylim(bottom, top)
    ax.axis("off")

    ax.plot([lo, hi], [0, 0], color="black", lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.10], color="black", lw=1.2)
        ax.text(t, 0.16, str(t), ha="center", va="bottom", fontsize=9)
    if cd:
        ax.plot([lo, lo + cd], [0.62, 0.62], color="black", lw=2.0)
        for xe in (lo, lo + cd):
            ax.plot([xe, xe], [0.56, 0.68], color="black", lw=1.2)
        ax.text(
            lo + cd / 2,
            0.72,
            f"CD = {cd:.2f}  (alpha={alpha})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text((lo + hi) / 2, top, title, ha="center", va="top", fontsize=10)

    for d, (a, b) in enumerate(grps):
        y = -0.22 - clique_gap * d
        ax.plot(
            [ordered[a][1] - 0.03, ordered[b][1] + 0.03],
            [y, y],
            color="black",
            lw=3.4,
            solid_capstyle="round",
        )

    for i, (m, r) in enumerate(ordered):
        left = i < n_side
        row = i if left else n - 1 - i
        y = first_row - row_gap * row
        x_end = (lo - span * 0.05) if left else (hi + span * 0.05)
        color, _ = _style(m)
        ax.plot([r, r], [0, y], color=color, lw=1.3)
        ax.plot([r, x_end], [y, y], color=color, lw=1.3)
        ax.text(
            x_end - span * 0.02 if left else x_end + span * 0.02,
            y,
            f"{m}  ({r:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, out)


def fig_perf_profile(
    study: Study, methods: Sequence[str], out: str, title: str, metric: str = "final"
) -> None:
    """Dolan-More performance profile.

    ``rho_{r,m} = c_{r,m} / min_m' c_{r,m'}`` with ``c`` the regret regularized by
    ``eps_r`` = 0.1% of that run's initial gap. The epsilon is what makes the ratio
    finite when a method reaches exactly zero regret (common once a finite pool is
    largely exhausted); it also encodes "closer than 0.1% of the starting gap counts
    as solved", which is the honest reading of an exhausted pool.
    """
    cols = (
        [study.auc(m) for m in methods]
        if metric == "auc"
        else [study.at(m, None) for m in methods]
    )
    C = np.column_stack(cols)
    eps = 1e-3 * np.maximum(study.regret[methods[0]][:, 0], 1e-12)
    C = C + eps[:, None]
    ratio = C / C.min(axis=1, keepdims=True)
    taus = np.unique(np.concatenate([[1.0], np.sort(ratio.ravel())]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for j, m in enumerate(methods):
        frac = [(ratio[:, j] <= t).mean() for t in taus]
        color, marker = _style(m)
        ax.step(taus, frac, where="post", color=color, linewidth=1.7, label=m)
    ax.set_xscale("log")
    ax.set_xlabel(r"performance ratio $\tau$ (within $\tau\times$ the best method)")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, out)


def fig_data_profile(
    study: Study, methods: Sequence[str], out: str, title: str, tol: float = 0.05
) -> None:
    """Fraction of runs solved vs evaluation budget -- a performance profile.

    This, not regret-AUC, is the metric Wang et al. (2021) headline: "the fraction of all
    test tasks that each method is able to solve by reaching at most 0.05, 0.01 and 0.001
    regrets" at each BO iteration. AUC appears nowhere in their paper. ``tol`` > 0 is a
    fraction of each run's initial gap; ``tol`` < 0 is an absolute regret threshold
    ``|tol|``, matching their C values.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(study.n_pts)
    for m in methods:
        R = study.regret[m]
        thresh = (
            np.full(R.shape[0], -tol, dtype=float)
            if tol < 0
            else tol * np.maximum(R[:, 0], 1e-12)
        )
        solved = (R <= thresh[:, None]).mean(axis=0)
        color, marker = _style(m)
        ax.plot(
            x,
            solved,
            color=color,
            marker=marker,
            markersize=3.5,
            markevery=max(1, study.n_pts // 12),
            linewidth=1.6,
            label=m,
        )
    ax.set_xlabel("evaluations")
    ax.set_ylabel(
        f"fraction solved (regret <= {-tol:g})"
        if tol < 0
        else f"fraction of runs within {tol:.0%} of the initial gap"
    )
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, out)


PRETRAIN_KEY = {
    "em_frozen": "em_pretrain_s",
    "em_finetuned": "em_pretrain_s",
    "em_noisefit": "em_pretrain_s",
    "em_additive_freshbase": "em_pretrain_s",
    "em_additive_3way": "em_pretrain_s",
    "hyperbo_frozen": "hyperbo_pretrain_s",
    "hyperbo_adapt": "hyperbo_pretrain_s",
    "pacoh_frozen": "pacoh_pretrain_s",
    "pacoh_adapt": "pacoh_pretrain_s",
    "ablr": "ablr_pretrain_s",
}


def _pretrain_per_task(study: Study, method: str) -> float:
    """One prior's pre-training cost charged to a single target task.

    Leave-one-out trains a prior per fold and that prior is task-specific, so the
    per-task cost is ``total / n_folds`` and there is nothing to amortize. A normal
    sweep pre-trains once for the whole eval set, so the total is the cold-start cost
    and amortization is meaningful.
    """
    total = float(study.timings.get(PRETRAIN_KEY.get(method, ""), 0.0) or 0.0)
    return total / max(1, study.n_priors) if study.is_loo else total


def fig_cost(
    study: Study,
    methods: Sequence[str],
    out: str,
    title: str,
    amortize: int = 50,
) -> None:
    """Regret vs cumulative wall-clock (pre-training + online inference).

    A normal sweep gets two panels, because the answer depends on how many target tasks
    the one-time pre-training is spread over: cold start charges the whole bill to one
    task, amortized divides it by ``amortize`` reused tasks. A leave-one-out sweep gets
    one panel -- its priors are task-specific and cannot be reused.
    """
    n_runs = max(1, int(study.timings.get("n_runs", study.n_runs)))
    online = study.timings.get("method_bo_s", {})
    panels = (
        [(1, "per task (leave-one-out: every task trains its own prior)")]
        if study.is_loo
        else [
            (1, "cold start (pre-training charged to one task)"),
            (amortize, f"amortized over {amortize} tasks"),
        ]
    )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.9 * len(panels), 4.6), sharey=True, squeeze=False
    )
    for ax, (amort, label) in zip(axes[0], panels):
        for m in methods:
            mean = study.regret[m].mean(axis=0)
            pre = _pretrain_per_task(study, m) / max(1, amort)
            per_step = (float(online.get(m, 0.0)) / n_runs) / max(1, study.n_pts - 1)
            cost = pre + per_step * np.arange(study.n_pts)
            color, marker = _style(m)
            ax.plot(
                np.maximum(cost, 1e-4),
                mean,
                color=color,
                marker=marker,
                markersize=3.5,
                markevery=max(1, study.n_pts // 10),
                linewidth=1.6,
                label=m,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cumulative wall-clock per task (s)")
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3, linewidth=0.5, which="both")
    axes[0][0].set_ylabel("simple regret")
    axes[0][-1].legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    _save(fig, out)


def fig_cost_quality(
    study: Study, methods: Sequence[str], out: str, title: str
) -> None:
    """Quality vs cost: regret-AUC against pre-training and against inference cost.

    The cost of a meta-learned prior splits into a one-off meta-training bill and a
    per-evaluation inference bill, and methods sit very differently on the two: an
    analytic Bayesian-linear head is expensive to meta-train but nearly free to query,
    while a from-scratch GP inverts that. A single "total cost" number hides this, so
    both axes are plotted against the same quality metric; bottom-left is best.
    """
    n_runs = max(1, int(study.timings.get("n_runs", study.n_runs)))
    online = study.timings.get("method_bo_s", {})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, mode in zip(axes, ("pretrain", "online")):
        for m in methods:
            auc = float(study.auc(m).mean())
            x = (
                _pretrain_per_task(study, m)
                if mode == "pretrain"
                else 1000.0 * float(online.get(m, 0.0)) / n_runs
            )
            if x <= 0:
                continue  # random / untrained arms have no cost on this axis
            color, marker = _style(m)
            ax.scatter(
                x,
                auc,
                color=color,
                marker=marker,
                s=80,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
            ax.annotate(
                m, (x, auc), textcoords="offset points", xytext=(6, 4), fontsize=7.5
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, linewidth=0.5, which="both")
        ax.set_xlabel(
            "pre-training per task (s)"
            if mode == "pretrain"
            else "online inference (ms / BO run)"
        )
    axes[0].set_ylabel("regret-AUC (lower = better)")
    fig.suptitle(f"{title} - quality vs cost (bottom-left is best)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, out)


def fig_budget_pareto(points: list[tuple[float, str]], out: str, title: str) -> None:
    """Quality vs pre-training budget: regret-AUC against pre-training seconds.

    ``points`` pairs a pre-training budget (steps) with a result file. Each file is a
    separate pre-training run of the *same* sweep, so the meta-learner curves move while
    the reference methods stay fixed -- which is what makes this a clean Pareto rather
    than a confounded comparison.

    Caveat worth carrying: one pre-trained prior per budget means the curve mixes the
    budget effect with pre-training-seed variance. Large monotone moves are trustworthy;
    small non-monotone wiggles are not.
    """
    studies = [(steps, Study(path)) for steps, path in points]
    varying = ("hyperbo_frozen", "hyperbo_adapt", "pacoh_frozen", "ablr")
    fixed = ("em_frozen", "random")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for m in varying:
        xs, ys = [], []
        for _steps, s in studies:
            if m not in s.regret:
                continue
            pre = _pretrain_per_task(s, m)
            if pre <= 0:
                continue
            xs.append(pre)
            ys.append(float(s.auc(m).mean()))
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        color, marker = _style(m)
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.6,
            label=m,
            zorder=3,
        )
        for (steps, _), x, y in zip(studies, xs, ys):
            ax.annotate(
                f"{int(steps / 1000)}k",
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=color,
            )
    for m in fixed:
        s = studies[0][1]
        if m not in s.regret:
            continue
        color, _ = _style(m)
        ax.axhline(
            float(s.auc(m).mean()),
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"{m} (budget-independent)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pre-training wall-clock per task (s)")
    ax.set_ylabel("regret-AUC (lower = better)")
    ax.set_title(title)
    ax.grid(alpha=0.3, linewidth=0.5, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=180)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print(f"  wrote {out}.png / .pdf", flush=True)


# --------------------------------------------------------------------------------
# markdown emission
# --------------------------------------------------------------------------------


def budget_to_target(
    study: Study, method: str, tol: float
) -> tuple[float, float, float]:
    """Evaluations needed to reach a target, and the fraction of runs that reach it.

    Regret-AUC integrates the whole trajectory, which answers "how good was the search
    on average" -- but the question a practitioner actually asks is "how many
    evaluations until I hit a good-enough configuration". Those differ: a method that
    explores for ten rounds and then jumps to the optimum can lose on AUC while
    reaching every target sooner than a steadily-improving competitor.

    The target is ``tol`` times each run's own initial regret, which makes it scale-free
    across tasks. Runs that never reach it are right-censored, so the median is taken
    over the runs that did solve and the solve rate is returned alongside -- a median
    without its solve rate is misleading, since a method can look fast by solving only
    the easy runs.

    Returns ``(median_evals_among_solved, mean_evals_among_solved, solve_rate)``.
    """
    R = study.regret[method]
    # tol > 0 is a fraction of each run's own initial gap (scale-free across tasks);
    # tol < 0 encodes an ABSOLUTE regret threshold |tol|, which is what Wang et al.
    # report performance profiles at (C = 0.05, 0.01, 0.001).
    thresh = np.full(R.shape[0], -tol, dtype=float) if tol < 0 else tol * R[:, 0]
    hit = R <= thresh[:, None]
    solved = hit.any(axis=1)
    if not solved.any():
        return float("nan"), float("nan"), 0.0
    first = np.argmax(hit, axis=1).astype(float)[solved]
    return float(np.median(first)), float(first.mean()), float(solved.mean())


def budget_table(study: Study, methods: Sequence[str], tols: Sequence[float]) -> str:
    """Markdown table of budget-to-target at several target levels."""

    def _lbl(v: float) -> str:
        return f"to regret {-v:g}" if v < 0 else f"to {v:.0%} of gap"

    head = "| method | " + " | ".join(_lbl(t) for t in tols)
    rows = [head + " |", "|" + "---|" * (len(tols) + 1)]

    def _key(m: str) -> float:
        med, _, rate = budget_to_target(study, m, tols[0])
        return float("inf") if rate == 0 else med

    for m in sorted(methods, key=_key):
        cells = []
        for tol in tols:
            med, _, rate = budget_to_target(study, m, tol)
            cells.append("never" if rate == 0 else f"{med:.0f} ({rate:.0%} solved)")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def markdown_table(study: Study, methods: Sequence[str], horizons: list[int]) -> str:
    """The main results table, sorted by dataset-clustered average rank."""
    final = rank_stats(study, methods, "final")
    auc = rank_stats(study, methods, "auc")
    order = sorted(methods, key=lambda m: final["avg_rank_clustered"][m])
    head = (
        "| method | "
        + " | ".join(f"@{h}" for h in horizons)
        + " | final | AUC | avg_rank | win% |"
    )
    sep = "|---" * (len(horizons) + 5) + "|"
    rows = [head, sep]
    for m in order:
        cells = []
        for h in horizons:
            v = study.at(m, h)
            cells.append(f"{v.mean():.3f}±{study.clustered_sem(v):.3f}")
        rows.append(
            f"| {m} | "
            + " | ".join(cells)
            + f" | {final['mean'][m]:.3f}±{final['clustered_sem'][m]:.3f}"
            + f" | {auc['mean'][m]:.3f}"
            + f" | {final['avg_rank_clustered'][m]:.2f}"
            + f" | {final['win_pct'][m]:.0f} |"
        )
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=str, required=True, help="bo_experiment JSON")
    p.add_argument("--tag", type=str, required=True, help="output filename prefix")
    p.add_argument("--out-dir", type=str, default="results/figures")
    p.add_argument("--title", type=str, default="")
    p.add_argument("--horizons", type=str, default="1,5,10,20,40")
    p.add_argument(
        "--methods", type=str, default="", help="comma-separated subset; default all"
    )
    p.add_argument("--tol", type=float, default=0.05, help="data-profile tolerance")
    p.add_argument(
        "--amortize", type=int, default=50, help="tasks to amortize pretrain"
    )
    p.add_argument(
        "--budget-pareto",
        type=str,
        default="",
        help="Comma-separated steps=path pairs; emits the pre-training budget/quality "
        "Pareto instead of the per-study figures.",
    )
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    matplotlib.use("Agg")
    if args.budget_pareto:
        pts = [
            (float(kv.split("=")[0]), kv.split("=", 1)[1])
            for kv in args.budget_pareto.split(",")
            if kv.strip()
        ]
        os.makedirs(args.out_dir, exist_ok=True)
        fig_budget_pareto(
            pts,
            os.path.join(args.out_dir, f"{args.tag}_budget_pareto"),
            args.title or args.tag,
        )
        return
    study = Study(args.result)
    methods = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods
        else study.methods
    )
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    horizons = [h for h in horizons if h < study.n_pts]
    title = args.title or args.tag

    print(
        f"=== {args.tag}: {study.n_runs} runs x {len(methods)} methods "
        f"({study.n_pts} trajectory points) ===\n",
        flush=True,
    )
    print(markdown_table(study, methods, horizons), flush=True)
    # Budget-to-target: AUC integrates the whole trajectory, but the practical
    # question is how many evaluations until a good-enough configuration.
    tols = (0.5, 0.1, 0.05)
    print("\nBudget to target (median evals, scale-free per run):", flush=True)
    print(budget_table(study, methods, tols), flush=True)

    summary = {
        "result_file": os.path.basename(args.result),
        "config": study.config,
        "budget_to_target": {
            f"{tol}": {
                m: dict(
                    zip(
                        ("median", "mean", "solve_rate"),
                        budget_to_target(study, m, tol),
                    )
                )
                for m in methods
            }
            for tol in (0.5, 0.1, 0.05)
        },
        "eval_datasets": study.eval_datasets,
        "n_runs": study.n_runs,
        "calibration": study.calibration,
        "timings": study.timings,
        "stats": {},
    }
    for label, metric, hz in [("final", "final", 0), ("auc", "auc", 0)] + [
        (f"@{h}", "at", h) for h in horizons
    ]:
        st = rank_stats(study, methods, metric, hz)
        summary["stats"][label] = st
        cd = st["nemenyi_cd"]["0.05"]
        print(
            f"\n[{label}] Friedman chi2({st['friedman_df']}) = "
            f"{st['friedman_chi2']:.1f}, p = {st['friedman_p']:.3g}; "
            f"Nemenyi CD(0.05) = {cd:.2f}"
            if cd
            else "",
            flush=True,
        )
        ordered = sorted(st["avg_rank"].items(), key=lambda kv: kv[1])
        print("  ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in ordered), flush=True)
        if cd:
            grps = cliques(ordered, cd)
            print(
                "  cliques (not significantly different): "
                + "; ".join(
                    "{" + ", ".join(ordered[i][0] for i in range(a, b + 1)) + "}"
                    for a, b in grps
                ),
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    spath = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {spath}", flush=True)

    if args.no_figures:
        return
    base = os.path.join(args.out_dir, args.tag)
    fig_trajectories(study, methods, f"{base}_trajectories", title)
    fig_perf_profile(
        study,
        methods,
        f"{base}_perfprofile",
        f"{title}: performance profile (final regret)",
    )
    fig_data_profile(
        study,
        methods,
        f"{base}_dataprofile",
        f"{title}: data profile (tol={args.tol:.0%})",
        tol=args.tol,
    )
    fig_cost(
        study,
        methods,
        f"{base}_cost",
        f"{title}: regret vs wall-clock",
        amortize=args.amortize,
    )
    fig_cost_quality(study, methods, f"{base}_cost_quality", title)
    for label in ["final", "auc"] + [f"@{h}" for h in horizons]:
        st = summary["stats"][label]
        safe = label.replace("@", "at")
        fig_cd_diagram(
            st, f"{base}_cd_{safe}", f"{title}: critical difference ({label})"
        )


if __name__ == "__main__":
    main()
