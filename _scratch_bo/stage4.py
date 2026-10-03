# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage 4 synthesis: the four cross-stage analyses of EXPERIMENT_PLAN.md S4 and S10.

None of these had ever been implemented; a grep for
`p_div|stage4|regression_predicts_bo` returned nothing across the whole scratch
directory. All four run on data already on disk, so nothing here launches compute.

    4.1  Does regression quality predict BO performance?
    4.2  On-grid vs off-grid factor effects.
    4.3  The cost Pareto.
    4.4  The pre-registered divergence moderator (S10).

Kept OUT of the shared `lib` srcs and given its own buck target on purpose: everything
in `lib` rebuilds `bo_experiment` and `bo_diagnose`, and editing an analysis script
must never be able to change the binary that a running sweep executes.

Conventions, all inherited rather than reinvented:

* Primary metric is budget-to-target (`.llms/rules/metrics.md`); regret-AUC is not
  reported at all.
* The replication unit is the PRE-TRAINING PRIOR (S27.26), so contrasts are paired per
  prior and tested with t(P-1).
* Every claim states its estimand. Fixed-task ("wins on this suite") and task-clustered
  ("wins on a new task") are reported side by side, because S28 found the two disagree
  on every headline in this project.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)
RAW = os.path.join(ROOT, "results/raw/v2")

# Module-level so it is not re-evaluated per call and does not trip B008.
DEFAULT_PRIORS = range(9)

# S28.6: on both regression regimes `hyperbo_frozen` -- a control that cannot depend on
# any EM flag -- takes three distinct values, partitioned exactly by the two OFAT levels
# that allocate extra parameterised modules. Only this block is internally comparable.
# `mean_linear`, `mean_linear+covprior` and `canon_deep` are excluded until the RNG
# guard lands and the 252 regression cells are re-run.
CLEAN_REGRESSION_CONFIGS = frozenset(
    {
        "base",
        "canon_hyperbo",
        "covprior",
        "covprior_nu",
        "init_naive",
        "meanprior",
        "meanxfer_full",
        "meanxfer_blend",
        "mean_linear+meanxfer",
        "shrink_0.1",
        "shrink_0.3",
    }
)

EM_METHODS = ("em_frozen", "em_additive_hyperbo", "em_additive_hyperbo_frozen")

# Below this, a recorded pre-training time is a prior-cache `torch.load`, not a fit.
# Measured hits are 0.03-0.05 s; the cheapest real fit anywhere in this project is 35 s
# (HyperBO at default iters). Nothing legitimate lands in between.
CACHE_HIT_SECONDS = 1.0


def t_crit(df: int) -> float:
    """Two-sided 5% critical value. Small-df values matter here and 1.96 does not."""
    table = {
        1: 12.71,
        2: 4.30,
        3: 3.18,
        4: 2.78,
        5: 2.57,
        6: 2.45,
        7: 2.36,
        8: 2.31,
        9: 2.26,
        10: 2.23,
        12: 2.18,
        15: 2.13,
        20: 2.09,
        22: 2.07,
        25: 2.06,
        30: 2.04,
    }
    if df in table:
        return table[df]
    keys = sorted(table)
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= df)
    return table[lo]


class Contrast:
    """A paired contrast with its own sample size, kept together with its estimand."""

    def __init__(self, diffs: list[float], estimand: str) -> None:
        self.d = list(diffs)
        self.estimand = estimand

    @property
    def n(self) -> int:
        return len(self.d)

    @property
    def mean(self) -> float:
        return st.mean(self.d) if self.d else float("nan")

    @property
    def se(self) -> float:
        if len(self.d) < 2:
            return float("nan")
        return st.stdev(self.d) / math.sqrt(len(self.d))

    @property
    def t(self) -> float:
        se = self.se
        # A zero SE is not infinite significance, it is a dead replication axis. S28.6
        # found 7/14 regression configs bit-identical across all 9 priors, which printed
        # t = 2.5e16 rather than announcing that nothing varied.
        if not se or math.isnan(se):
            return float("nan")
        return self.mean / se

    @property
    def significant(self) -> bool:
        return abs(self.t) > t_crit(self.n - 1) if self.n > 1 else False

    def line(self) -> str:
        if self.n < 2:
            return f"n={self.n} (cannot estimate a SE)"
        if math.isnan(self.t):
            return (
                f"d={self.mean:+.4f}  SE=0  t=undefined  n={self.n} "
                "[DEAD AXIS: every replicate identical]"
            )
        mark = " *" if self.significant else ""
        return (
            f"d={self.mean:+7.3f} +-{self.se:6.3f}  t({self.n - 1})={self.t:+6.2f}"
            f"  crit={t_crit(self.n - 1):.2f}{mark}"
        )


def summary(stage: str) -> dict:
    path = os.path.join(RAW, stage, "SUMMARY.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{stage}: no SUMMARY.json at {path}")
    with open(path) as f:
        return json.load(f)


def cell_metric(stage: str, config: str, prior: int, method: str, key: str):
    cells = summary(stage).get("cells", {})
    body = cells.get(f"{config}_p{prior}", {})
    return (body.get("methods") or {}).get(method, {}).get(key)


def per_prior(stage: str, config: str, method: str, key: str, priors=DEFAULT_PRIORS):
    cells = summary(stage).get("cells", {})
    out = {}
    for p in priors:
        body = cells.get(f"{config}_p{p}", {})
        v = (body.get("methods") or {}).get(method, {}).get(key)
        if isinstance(v, (int, float)):
            out[p] = float(v)
    return out


def paired(stage: str, a: str, b: str, method: str, key: str) -> Contrast:
    va, vb = per_prior(stage, a, method, key), per_prior(stage, b, method, key)
    shared = sorted(set(va) & set(vb))
    return Contrast([va[p] - vb[p] for p in shared], "fixed-task (unit = prior)")


def configs_in(stage: str) -> list[str]:
    cells = summary(stage).get("cells", {})
    return sorted({c.rsplit("_p", 1)[0] for c in cells})


# --------------------------------------------------------------------------------------
# per-task machinery, used by 4.2's clustered estimand and by 4.4


def per_task_budget(
    stage: str, config: str, method: str, tol: float, priors=DEFAULT_PRIORS
) -> dict[int, float]:
    """Mean budget-to-target per EVAL TASK, pooled over priors and shards.

    The task index is the unit the clustered estimand needs; SUMMARY.json only carries
    the already-averaged value, so this reads the shards.
    """
    acc: dict[int, list[float]] = defaultdict(list)
    for p in priors:
        for path in sorted(
            glob.glob(os.path.join(RAW, stage, f"{config}_p{p}_s*.json"))
        ):
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            pr = d.get("per_run", {})
            traj = (pr.get("traj_raw") or {}).get(method)
            pmax = pr.get("pool_max_raw") or []
            idx = pr.get("eval_dataset_idx") or []
            if not traj or len(traj) != len(idx) or len(pmax) < len(traj):
                continue
            for r, row in enumerate(traj):
                hit = len(row)
                for t, v in enumerate(row):
                    if pmax[r] - v <= tol:
                        hit = t
                        break
                acc[idx[r]].append(float(hit))
    return {k: st.mean(v) for k, v in acc.items() if v}


def clustered(stage: str, a: str, b: str, method: str, tol: float) -> Contrast:
    """Same contrast, but the unit is the TASK -- 'wins on a new task'."""
    va = per_task_budget(stage, a, method, tol)
    vb = per_task_budget(stage, b, method, tol)
    shared = sorted(set(va) & set(vb))
    return Contrast([va[k] - vb[k] for k in shared], "task-clustered (unit = task)")


# --------------------------------------------------------------------------------------
# 4.1  Does regression quality predict BO performance?


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def analysis_41(reg_stage: str, bo_stage: str, label: str) -> None:
    """Correlate each config's regression quality against its BO budget-to-target.

    The pairing is by CONFIG, within a benchmark: the same flag string was run on both
    harnesses, so a config is the natural join key. If regression quality predicted BO
    we would expect better rank_corr / lower RMSE to go with a lower budget.
    """
    print(f"\n### 4.1  Does regression predict BO?  [{label}]")
    print(
        "    unit = config; regression restricted to the 11-config clean block "
        "(S28.6)\n"
    )
    shared = [
        c
        for c in configs_in(reg_stage)
        if c in set(configs_in(bo_stage)) and c in CLEAN_REGRESSION_CONFIGS
    ]
    dropped = [
        c
        for c in configs_in(reg_stage)
        if c in set(configs_in(bo_stage)) and c not in CLEAN_REGRESSION_CONFIGS
    ]
    if dropped:
        print(f"    excluded as RNG-contaminated: {', '.join(sorted(dropped))}")
    if len(shared) < 3:
        print("    NOT MEASURABLE: fewer than 3 shared configs.")
        return

    method = "em_frozen"
    rows = []
    for c in shared:
        rc = per_prior(reg_stage, c, method, "n5_rank_corr")
        rmse = per_prior(reg_stage, c, method, "n5_rmse")
        nll = per_prior(reg_stage, c, method, "n5_nll")
        bo = per_prior(bo_stage, c, method, "budget_to_0.01")
        if rc and bo:
            rows.append(
                (
                    c,
                    st.mean(rc.values()),
                    st.mean(rmse.values()) if rmse else float("nan"),
                    st.mean(nll.values()) if nll else float("nan"),
                    st.mean(bo.values()),
                )
            )
    if len(rows) < 3:
        print("    NOT MEASURABLE: metrics missing on one side.")
        return

    print(
        f"    {'config':<22}{'n5_rank_corr':>13}{'n5_rmse':>10}"
        f"{'n5_nll':>10}{'budget':>10}"
    )
    for c, rc, rmse, nll, bo in sorted(rows, key=lambda r: r[4]):
        print(f"    {c:<22}{rc:>13.4f}{rmse:>10.4f}{nll:>10.4f}{bo:>10.3f}")

    rcs = [r[1] for r in rows]
    rmses = [r[2] for r in rows]
    nlls = [r[3] for r in rows]
    bos = [r[4] for r in rows]
    print()
    print(
        f"    Spearman(rank_corr, budget) = {spearman(rcs, bos):+.3f}"
        "   (expect NEGATIVE if regression predicts BO)"
    )
    print(
        f"    Spearman(rmse,      budget) = {spearman(rmses, bos):+.3f}"
        "   (expect POSITIVE)"
    )
    print(
        f"    Spearman(nll,       budget) = {spearman(nlls, bos):+.3f}"
        "   (expect POSITIVE)"
    )
    print(
        "\n    Instrument check: budget spread "
        f"{max(bos) - min(bos):.3f} over {len(bos)} configs; "
        f"rank_corr spread {max(rcs) - min(rcs):.4f}."
    )
    if max(bos) - min(bos) < 1e-9 or max(rcs) - min(rcs) < 1e-9:
        print("    ONE AXIS IS CONSTANT -- the correlation is undefined, not zero.")


# --------------------------------------------------------------------------------------
# 4.2  On-grid vs off-grid factor effects


def analysis_42(base: str = "base") -> None:
    """The same factor, the same priors, on-grid vs off-grid -- and the interaction.

    S28.5 found the only large consistent win (the canonical kernel) is off-grid-only,
    and S28.2 found the noise knob actually INVERTS. Both are interactions, so the
    difference-in-differences is the estimate that matters, not either arm alone.
    """
    print("\n### 4.2  On-grid vs off-grid factor effects")
    print("    paired per prior; DiD = (off-grid effect) - (on-grid effect)\n")
    on, off = "stage1_pd1_bo_armA", "stage1_pd1_bo_armC"
    method = "em_frozen"
    common = sorted(set(configs_in(on)) & set(configs_in(off)) - {base})
    print(f"    {'factor':<24}{'on-grid (armA)':>22}{'off-grid (armC)':>22}{'DiD':>24}")
    for c in common:
        a = paired(on, c, base, method, "budget_to_0.01")
        b = paired(off, c, base, method, "budget_to_0.01")
        va = per_prior(on, c, method, "budget_to_0.01")
        vb = per_prior(on, base, method, "budget_to_0.01")
        wa = per_prior(off, c, method, "budget_to_0.01")
        wb = per_prior(off, base, method, "budget_to_0.01")
        shared = sorted(set(va) & set(vb) & set(wa) & set(wb))
        did = Contrast(
            [(wa[p] - wb[p]) - (va[p] - vb[p]) for p in shared], "fixed-task"
        )
        star = "*" if did.significant else " "
        print(
            f"    {c:<24}{a.mean:>+12.3f} (t={a.t:+5.2f}){b.mean:>+12.3f} "
            f"(t={b.t:+5.2f}){did.mean:>+12.3f} (t={did.t:+5.2f}){star}"
        )
    print(
        "\n    A significant DiD means the factor is REGIME-dependent and must "
        "never be "
        "\n    reported as a single pooled effect (S28.3, S28.5)."
    )


# --------------------------------------------------------------------------------------
# 4.3  The cost Pareto


def pretrain_cost(
    stage: str, config: str, method: str, priors=DEFAULT_PRIORS
) -> tuple[float, int, int]:
    """Real pre-training seconds for a method, plus (n_usable, n_cache_served).

    EM methods pay `em_pretrain_s`; the additive hybrids also pay for the HyperBO kernel
    they borrow; `hyperbo_frozen` pays only HyperBO. Charging every method the whole
    cell would make the Pareto plot meaningless, which is the point of S26.2.

    CACHE HITS ARE EXCLUDED, NOT COUNTED AS ZERO. The prior cache serves HyperBO from
    disk for every config after the first, and those cells record
    `hyperbo_pretrain_s = 0.0`. Measured on `stage1_pd1_bo_armC`: `base` has a median
    882 s over 108 shards, while `canon_hyperbo`, `shrink_0.1`, `init_naive`,
    `meanprior`, `covprior` and `mean_linear` each record 0.0 in 108/108. Averaging
    those zeros concludes pre-training is free, which is how a cost Pareto quietly
    inverts. A hit is MISSING DATA for the cost question, not a measurement of zero
    (S30.1).
    """
    tot: list[float] = []
    served = 0
    for p_ in priors:
        for path in sorted(
            glob.glob(os.path.join(RAW, stage, f"{config}_p{p_}_s*.json"))
        ):
            try:
                with open(path) as f:
                    tm = (json.load(f) or {}).get("timings") or {}
            except (OSError, ValueError):
                continue
            em = float(tm.get("em_pretrain_s") or 0.0)
            hb = float(tm.get("hyperbo_pretrain_s") or 0.0)
            pac = float(tm.get("pacoh_pretrain_s") or 0.0)
            ablr = float(tm.get("ablr_pretrain_s") or 0.0)
            hits = tm.get("prior_cache_hits") or {}

            # (value, kind) per component. kind=None means "always really computed"
            # (EM pre-training is never cached), so it is exempt from the hit check.
            if method.startswith("em_additive"):
                parts = [(em, None), (hb, "hyperbo")]
            elif method.startswith("em"):
                parts = [(em, None)]
            elif method.startswith("hyperbo"):
                parts = [(hb, "hyperbo")]
            elif method.startswith("pacoh"):
                parts = [(pac, "pacoh")]
            elif method.startswith("ablr"):
                parts = [(ablr, "ablr")]
            else:
                continue

            # Test EVERY cacheable COMPONENT, not the total. An additive hybrid pays
            # em + hyperbo; when only the hyperbo half is served from cache the total
            # is still ~400 s and a total-based check passes it through while silently
            # understating the true cost by ~845 s. Two earlier versions of this guard
            # were wrong -- one compared against 0.0 when hits actually record 0.03 s,
            # one compared the total -- so it is now written per component.
            hit = False
            for val_i, kind_i in parts:
                if kind_i is None:
                    continue
                if int(hits.get(kind_i) or 0) > 0 or val_i < CACHE_HIT_SECONDS:
                    hit = True
                    break
            if hit:
                served += 1
                continue
            tot.append(sum(v for v, _k in parts))
    return (st.mean(tot) if tot else float("nan")), len(tot), served


def analysis_43(stage: str, config: str = "base") -> None:
    print(f"\n### 4.3  Cost Pareto  [{stage}, config={config}]")
    print("    pre-training seconds vs budget-to-target; lower-left dominates")
    print("    cache-served cells are EXCLUDED, not counted as zero cost (S30.1)\n")
    s = summary(stage).get("cells", {})
    methods = sorted(
        {
            m
            for k, v in s.items()
            if k.startswith(f"{config}_p")
            for m in (v.get("methods") or {})
        }
    )
    rows = []
    excluded = []
    for m in methods:
        bo = per_prior(stage, config, m, "budget_to_0.01")
        if not bo:
            continue
        cost, n_used, n_served = pretrain_cost(stage, config, m)
        if math.isnan(cost):
            excluded.append((m, n_served))
            continue
        rows.append((m, cost, st.mean(bo.values()), n_used, n_served))
    for m, n_served in excluded:
        print(
            f"    {m:<30} NO COST DATA -- all {n_served} cells cache-served; "
            "cost unmeasurable here"
        )
    if excluded:
        print()
    if not rows:
        print(
            "    NOT MEASURABLE: every method on this config was served from the prior"
            "\n    cache. Re-run one cell with --prior-cache-mode off to measure cost."
        )
        return
    print(
        f"    {'method':<30}{'pretrain_s':>12}{'budget':>10}{'n':>5}"
        f"{'cached':>8}   {'dominated by':<28}"
    )
    for m, cost, bud, n_used, n_served in sorted(rows, key=lambda r: r[2]):
        dom = [
            o
            for o, c2, b2, _n, _s in rows
            if o != m and c2 <= cost and b2 <= bud and (c2 < cost or b2 < bud)
        ]
        print(
            f"    {m:<30}{cost:>12.1f}{bud:>10.3f}{n_used:>5}{n_served:>8}   "
            f"{(', '.join(dom) if dom else 'PARETO-OPTIMAL'):<28}"
        )


# --------------------------------------------------------------------------------------
# 4.4  The pre-registered divergence moderator (EXPERIMENT_PLAN.md S10)
#
# Pre-registered before the data existed, specifically so it could not become a post-hoc
# slice: the covariate, the model, and BOTH interpretations were fixed in advance.
#
#     d_t = a + b * p_div_t + e_t
#
#   b ~ 0  => the EM deficit is uniform, and the tail explanation in S23/S27 is WRONG.
#   b > 0  => the deficit concentrates on degenerate-heavy tasks, which localises the
#             mechanism and is a much stronger claim than any average effect.


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float, int]:
    """Slope, its SE, intercept, n.

    Plain OLS; n is small and the model is univariate.
    """
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        # The covariate is constant: the slope is UNDEFINED, not zero. Returning 0 here
        # would silently answer the pre-registered question with the null.
        return float("nan"), float("nan"), float("nan"), n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(rss / (n - 2) / sxx) if n > 2 else float("nan")
    return b, se, a, n


def p_div_lcbench() -> dict[int, float]:
    """Fraction of each dataset's pool that is flat: max(curve) - min(curve) <= 0."""
    from _scratch_bo.bo_experiment import METRIC
    from _scratch_bo.lcbench_io import LCBENCH_DATASET_NAMES, load_lcbench_data

    out = {}
    for i, nm in enumerate(LCBENCH_DATASET_NAMES):
        data = load_lcbench_data(nm, METRIC, dtype=__import__("torch").float64)
        m = data.metrics  # (n_configs, n_epochs)
        flat = (m.max(dim=1).values - m.min(dim=1).values) <= 0
        out[i] = float(flat.double().mean())
    return out


def p_div_pd1() -> dict[int, float]:
    """Fraction of each task's pool with error >= 0.8.

    The harness stores -error_rate (higher is better), so the threshold is y <= -0.8.
    0.8 is the value pd1_full_data.py already uses for reporting, not a new choice.
    """
    from _scratch_bo.bo_experiment import _pd1_dataset_names, load_pd1_pool

    names = _pd1_dataset_names()
    _X, Ys, used = load_pd1_pool(names)
    pos = {nm: i for i, nm in enumerate(names)}
    out = {}
    for nm, Y in zip(used, Ys):
        y = Y.reshape(-1)
        out[pos.get(nm, len(out))] = float((y <= -0.8).double().mean())
    return out


def analysis_44(
    stage: str,
    benchmark: str,
    config: str = "base",
    em: str = "em_frozen",
    control: str = "hyperbo_frozen",
    tol: float = 0.01,
) -> None:
    print(f"\n### 4.4  Divergence moderator  [{stage}]   d_t = a + b * p_div_t")
    print(f"    d_t = budget({em}) - budget({control}) per task; positive = EM worse\n")

    try:
        p_div = p_div_lcbench() if benchmark == "lcbench" else p_div_pd1()
    except Exception as exc:  # noqa: BLE001 - report, never silently skip
        print(
            f"    NOT MEASURABLE: could not compute p_div ({type(exc).__name__}: {exc})"
        )
        return

    a_t = per_task_budget(stage, config, em, tol)
    b_t = per_task_budget(stage, config, control, tol)
    tasks = sorted(set(a_t) & set(b_t) & set(p_div))
    if len(tasks) < 3:
        print(
            f"    NOT MEASURABLE: only {len(tasks)} tasks with both a d_t and a p_div."
        )
        return

    xs = [p_div[t] for t in tasks]
    ys = [a_t[t] - b_t[t] for t in tasks]

    # Instrument check FIRST: a moderator cannot be estimated off a constant covariate,
    # and reporting b=0 in that case would answer the question with an artifact.
    print(
        f"    p_div over {len(tasks)} tasks: min={min(xs):.4f} median="
        f"{st.median(xs):.4f} max={max(xs):.4f} sd={st.stdev(xs):.4f}"
        if len(xs) > 1
        else "    p_div: single task"
    )
    if max(xs) - min(xs) < 1e-12:
        print(
            "    COVARIATE IS CONSTANT -- the moderator is UNDEFINED here, not null. "
            "This benchmark has no degeneracy to moderate on."
        )
        return

    b, se, a, n = ols(xs, ys)
    if math.isnan(b):
        print("    slope undefined")
        return
    tstat = b / se if se else float("nan")
    crit = t_crit(n - 2)
    print(f"    intercept a = {a:+.3f}")
    print(
        f"    slope     b = {b:+.3f} +-{se:.3f}   t({n - 2})={tstat:+.2f}"
        f"  crit={crit:.2f}"
    )
    print(f"    95% CI      = [{b - crit * se:+.3f}, {b + crit * se:+.3f}]")

    med = st.median(xs)
    lo = [y for x, y in zip(xs, ys) if x <= med]
    hi = [y for x, y in zip(xs, ys) if x > med]
    if lo and hi:
        print(
            f"    split at median p_div: low  n={len(lo)} mean d_t={st.mean(lo):+.3f}"
            f"\n                           high n={len(hi)} mean d_t={st.mean(hi):+.3f}"
        )

    # Leave-one-task-out. With few tasks and a narrow covariate a single point can
    # manufacture the slope, and a moderator that flips sign when any one task is
    # dropped is not a finding. Cheap, so always run it.
    loo = []
    stable = 0
    for i in range(n):
        bx = xs[:i] + xs[i + 1 :]
        by = ys[:i] + ys[i + 1 :]
        bb, bse, _a, bn = ols(bx, by)
        if not math.isnan(bb):
            loo.append((bb, abs(bb / bse) > t_crit(bn - 2) if bse else False))
    if loo:
        slopes = [s for s, _ in loo]
        stable = sum(1 for _s, sig in loo if sig)
        full_sig = abs(tstat) > crit
        print(
            f"    leave-one-task-out: slope in "
            f"[{min(slopes):+.3f}, {max(slopes):+.3f}], "
            f"significant in {stable}/{len(loo)} folds"
        )
        if min(slopes) * max(slopes) < 0:
            print(
                "    FRAGILE: the slope CHANGES SIGN when a single task is dropped. "
                "Do not report this moderator."
            )
        elif full_sig and stable < len(loo):
            # Only a claim of significance can be undermined by this check. A null that
            # is null in every fold is CONSISTENT, not fragile -- saying otherwise would
            # invent a caveat and obscure a clean result.
            print(
                f"    FRAGILE: the full-sample result is significant but survives only "
                f"{stable}/{len(loo)} folds. Report the DIRECTION only, not the test."
            )
        elif not full_sig and stable == 0:
            print(
                "    The null is consistent: not significant in the full sample nor in "
                "any fold."
            )

    print()
    robust = abs(tstat) > crit and bool(loo) and stable == len(loo)
    if robust:
        print(
            "    => b is SIGNIFICANT and survives every leave-one-task-out fold: the "
            "deficit\n       concentrates on degenerate-heavy tasks. This "
            "localises the "
            "mechanism\n       (the pre-registered 'b > 0' reading)."
        )
    elif abs(tstat) > crit:
        print(
            "    => b is significant in the full sample ONLY. Under the pre-registered "
            "reading\n       this is not enough to claim 'b > 0': the covariate "
            "range is "
            "narrow and the\n       test does not survive resampling. Report the "
            "direction, make no claim."
        )
    else:
        print(
            "    => b is NOT distinguishable from 0: the deficit is UNIFORM across "
            "tasks,\n       and the tail explanation in S23/S27 is not supported "
            "(the pre-registered 'b ~ 0' reading)."
        )


# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--which",
        default="all",
        choices=["all", "4.1", "4.2", "4.3", "4.4"],
        help="Which stage-4 analysis to run.",
    )
    args = ap.parse_args()
    w = args.which

    print("=" * 86)
    print("STAGE 4 SYNTHESIS -- all analyses run on data already on disk")
    print(
        "Primary metric: budget-to-target (.llms/rules/metrics.md). AUC not reported."
    )
    print("=" * 86)

    if w in ("all", "4.1"):
        for reg, bo, label in [
            ("stage1_lcb_reg", "stage1_lcb_bo", "LCBench (on-grid)"),
            ("stage1_pd1_reg", "stage1_pd1_bo_armC", "PD1 (off-grid)"),
        ]:
            try:
                analysis_41(reg, bo, label)
            except FileNotFoundError as exc:
                print(f"\n### 4.1 [{label}] SKIPPED: {exc}")
        print(
            "\n    NOTE: every regression number above is a POINT ESTIMATE only. Until "
            "\n    the two bo_diagnose defects are fixed and the 252 cells re-run, the "
            "\n    prior axis is dead for 7/14 configs and no SE is defensible (S28.6)."
        )

    if w in ("all", "4.2"):
        try:
            analysis_42()
        except FileNotFoundError as exc:
            print(f"\n### 4.2 SKIPPED: {exc}")

    if w in ("all", "4.3"):
        for stage in ("stage1_lcb_bo", "stage1_pd1_bo_armC"):
            try:
                analysis_43(stage)
            except FileNotFoundError as exc:
                print(f"\n### 4.3 [{stage}] SKIPPED: {exc}")

    if w in ("all", "4.4"):
        for stage, bench in [
            ("stage1_lcb_bo", "lcbench"),
            ("stage1_pd1_bo_armC", "pd1"),
        ]:
            try:
                analysis_44(stage, bench)
            except FileNotFoundError as exc:
                print(f"\n### 4.4 [{stage}] SKIPPED: {exc}")


if __name__ == "__main__":
    main()
