# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analysis of the PD1 pool sweep -- the two-factor protocol ablation of S21.2.

Arms (all leave-one-out over 23 PD1 tasks, --pd1-source full, neglog warp):

    A  matched candidates, matched pre-training  (raw/pd1full/pd1_loo_neglog_s*)
    B  FULL candidates,    matched pre-training  (raw/pd1pool/fullcand_s*)
    C  FULL candidates,    FULL pre-training     (raw/pd1pool/fullboth[_p1,_p2]_s*)

A->B isolates the search space, B->C adds the baselines' data advantage. Arm C is
replicated over three pre-training priors (p0/p1/p2) because S19.4 established that a
single prior cannot rank the neural meta-learners.

Ranks are compared WITHIN an arm only: each arm has a different candidate pool, so the
absolute regret scales are not commensurable across arms.
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import numpy as np
from _scratch_bo.analyze import budget_to_target, Study

# the build-tool runner does not preserve the caller's cwd and __file__ resolves into the build
# link-tree, so the results root is addressed absolutely, as pd1_full_data already does.
ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)

ARMS: dict[str, str] = {
    "A matched/matched": "results/raw/pd1full/pd1_loo_neglog_s*.json",
    "B full/matched": "results/raw/pd1pool/fullcand_s*.json",
    "C full/full p0": "results/raw/pd1pool/fullboth_s*.json",
    "C full/full p1": "results/raw/pd1pool/fullboth_p1_s*.json",
    "C full/full p2": "results/raw/pd1pool/fullboth_p2_s*.json",
}

TRANSFER = ["em_frozen", "em_noisefit", "em_finetuned", "hyperbo_frozen", "ablr"]


def load(pattern: str) -> Study:
    paths = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not paths:
        raise FileNotFoundError(pattern)
    return Study(",".join(paths))


def clustered_paired_z(
    study: Study, a: str, b: str, metric: str = "final"
) -> tuple[float, float, int]:
    """Paired a-b difference, clustered by evaluation task.

    Seeds within a task share the same held-out problem and are strongly correlated, so
    the test is over per-task mean differences, not over runs (S25.3).
    Returns (mean_difference, z, n_clusters). Negative favours ``a``.
    """
    da = study.at(a, None) if metric == "final" else study.auc(a)
    db = study.at(b, None) if metric == "final" else study.auc(b)
    d = da - db
    means = np.array(
        [d[study.ds_idx == t].mean() for t in np.unique(study.ds_idx)], dtype=float
    )
    if means.size < 2:
        return float(d.mean()), float("nan"), int(means.size)
    sem = means.std(ddof=1) / math.sqrt(means.size)
    z = means.mean() / sem if sem > 0 else float("nan")
    return float(means.mean()), float(z), int(means.size)


def solved_indicator(study: Study, method: str, tol: float) -> np.ndarray:
    """Per-run 1/0 for reaching absolute regret ``|tol|`` within the budget.

    Budget-to-target medians are taken over solved runs only, so a method can look fast
    by solving only the easy runs. The solve rate is the part that carries the signal
    off-grid, and unlike the median it is defined for every run, so it can be paired.
    """
    R = study.regret[method]
    return (R <= -tol).any(axis=1).astype(float)


def clustered_paired_z_values(
    ds_idx: np.ndarray, d: np.ndarray
) -> tuple[float, float, int]:
    means = np.array([d[ds_idx == t].mean() for t in np.unique(ds_idx)], dtype=float)
    if means.size < 2:
        return float(d.mean()), float("nan"), int(means.size)
    sem = means.std(ddof=1) / math.sqrt(means.size)
    z = means.mean() / sem if sem > 0 else float("nan")
    return float(means.mean()), float(z), int(means.size)


class PooledStudy:
    """Several priors of the same arm stacked into one sample.

    Each pre-training prior is an independent replicate (S19.4): one prior cannot rank
    the neural meta-learners, so arm C's three priors are pooled and clustered by task.
    Only methods present in every prior are kept -- the p1/p2 replicates dropped PACOH.
    """

    def __init__(self, patterns: list[str]) -> None:
        subs = [load(p) for p in patterns]
        common = set(subs[0].methods)
        for s in subs[1:]:
            common &= set(s.methods)
        self.methods: list[str] = [m for m in subs[0].methods if m in common]
        self.ds_idx = np.concatenate([s.ds_idx for s in subs])
        self.regret = {
            m: np.concatenate([s.regret[m] for s in subs], axis=0) for m in self.methods
        }
        self.n_runs = int(self.ds_idx.shape[0])
        self.n_priors = len(subs)
        # Which prior each run came from. Priors do NOT contribute equal run counts --
        # arm C's p0 has 5 seeds against p1/p2's 3 -- so a plain mean over the pooled
        # runs weights p0 at 45.5% instead of 33.3%. S27.3 showed priors disagree on
        # ordering, so that is a real bias in every pooled headline (review, metric #1).
        self.prior_idx = np.concatenate(
            [np.full(sub.ds_idx.shape[0], k) for k, sub in enumerate(subs)]
        )
        # Tasks that are not present under every prior; reported by report_gaps().
        self.coverage_gaps: set[int] = set()
        self.run_w = np.ones(self.n_runs, dtype=float)
        for k in range(self.n_priors):
            sel = self.prior_idx == k
            n_k = int(sel.sum())
            if n_k:
                self.run_w[sel] = 1.0 / (self.n_priors * n_k)

    def report_gaps(self) -> None:
        """Warn if any task was not observed under every prior."""
        if self.coverage_gaps:
            print(
                f"    WARNING: {len(self.coverage_gaps)} task(s) missing from at least "
                f"one prior; their per-task means average over fewer priors: "
                f"{sorted(self.coverage_gaps)}"
            )

    def wmean(self, values: np.ndarray) -> float:
        """Mean with every PRIOR weighted equally, regardless of its seed count."""
        w = self.run_w
        return float((values * w).sum() / w.sum())

    def task_wvar(self, values: np.ndarray, task: int) -> float:
        """Sampling variance OF ``task_wmean`` for one task.

        With ``ybar_p = theta_t + b_p + e_p`` we have ``Var(ybar_p) = s2_B + s2_W/n_p``,
        and for ``theta_hat_t = (1/P) sum_p ybar_p``::

            E[ s^2(ybar_1..ybar_P) ] = (1/P) sum_p Var(ybar_p)
            => E[ s^2 / P ]          = (1/P^2) sum_p Var(ybar_p) = Var(theta_hat_t)

        So the **spread of the per-prior means already IS the whole variance** -- it
        contains the within component, because each ``ybar_p`` is itself a noisy mean.
        An earlier version returned that plus a separate within term, which counted the
        within component twice and inflated the SE by up to sqrt(2) at n=(5,3,3). That
        inflation is what turned two arm-C contrasts into ties in S27.20 (round-6 S1).

        The within-only form is kept for ``P = 1``, where there is no between-prior
        spread to estimate and it is the only thing available.
        """
        sel = self.ds_idx == task
        v, p = values[sel], self.prior_idx[sel]
        present = np.unique(p)
        if present.size == 0:
            return 0.0
        if present.size < self.n_priors:
            # A task missing from a prior is averaged over the priors that remain, which
            # re-introduces exactly the imbalance this method exists to remove -- and
            # S27.3 showed priors disagree on ordering, so it is not benign. Surfaced
            # rather than silently tolerated (review N11).
            self.coverage_gaps.add(int(task))

        prior_means = [float(v[p == k].mean()) for k in present]
        if len(prior_means) > 1:
            s2 = float(np.var(prior_means, ddof=1))
            return s2 / len(prior_means)

        # P = 1: fall back to the within-prior variance of the single available mean.
        vk = v[p == present[0]]
        return float(vk.var(ddof=1) / vk.size) if vk.size > 1 else 0.0

    def task_wmean(self, values: np.ndarray, task: int) -> float:
        """Per-task mean, equalising priors WITHIN the task.

        A task observed more often under one prior would otherwise tilt that task's
        contribution to the fixed-task estimand.
        """
        sel = self.ds_idx == task
        v, p = values[sel], self.prior_idx[sel]
        present = np.unique(p)
        if present.size < self.n_priors:
            # A task missing from a prior is averaged over the priors that remain, which
            # re-introduces exactly the imbalance this method exists to remove -- and
            # S27.3 showed priors disagree on ordering, so it is not benign. Surfaced
            # rather than silently tolerated (review N11).
            self.coverage_gaps.add(int(task))
        return float(np.mean([v[p == k].mean() for k in present]))

    def at(self, method: str, horizon: int | None = None) -> np.ndarray:
        R = self.regret[method]
        h = R.shape[1] - 1 if horizon is None else min(horizon, R.shape[1] - 1)
        return R[:, h]

    def auc(self, method: str) -> np.ndarray:
        """Per-run regret-AUC = mean regret over the trajectory, as in S26.1."""
        return self.regret[method].mean(axis=1)

    def clustered_sem(self, values: np.ndarray) -> float:
        means = np.array(
            [values[self.ds_idx == t].mean() for t in np.unique(self.ds_idx)]
        )
        return float(means.std(ddof=1) / math.sqrt(means.size))


def budget_censored(study, method: str, tol: float) -> np.ndarray:
    """Evaluations to reach absolute regret ``|tol|``, RIGHT-CENSORED at the budget.

    This is the **restricted mean time to target (RMTT@T)**, not the mean time to target,
    and the distinction is not pedantic. A non-solver is charged exactly one evaluation
    more than a run that solved at the last index, so the estimator is bounded above by
    ``T+1`` no matter how badly a method fails. Worked example at a 50-evaluation budget:

        method A: solves 50% of runs at eval 5, NEVER solves the rest -> (5 + 51)/2 = 28.0
        method B: solves 100% of runs at eval 30                      ->            30.0

    **A "wins" while failing half its runs.** The bias favours a method that solves fewer
    runs provided it solves them early -- i.e. exactly the high-variance explore-heavy
    profile the EM-vs-HyperBO contrast is adjudicating. An earlier docstring here claimed
    censoring means "a method cannot look fast by solving only the easy runs"; that claim
    was false and has been removed.

    Consequences to respect when reporting:

    * ALWAYS print the solve rate beside it (``.llms/rules/metrics.md`` rank 3). RMTT
      alone is not interpretable.
    * The censoring constant is the horizon, so values are NOT comparable across suites
      with different ``--n-iters`` (PD1 arm C uses 50, LCBench 40).
    * If a headline flips between censoring at ``T`` and at ``2T``, it is an artefact of
      the constant rather than a property of the method. ``budget_censored_at`` exists
      for exactly that sensitivity check.
    """
    return budget_censored_at(study, method, tol, penalty_mult=1.0)


def budget_censored_at(
    study, method: str, tol: float, penalty_mult: float = 1.0
) -> np.ndarray:
    """RMTT with a tunable censoring penalty, for sensitivity analysis.

    ``penalty_mult=1`` censors at the horizon ``T`` (the default estimator);
    ``penalty_mult=2`` charges non-solvers ``2T``, which is punitive enough that a method
    cannot win by failing often. Comparing the two is the cheapest available check on
    whether a conclusion is real.
    """
    R = study.regret[method]
    hit = R <= -tol
    first = np.argmax(hit, axis=1).astype(float)
    horizon = R.shape[1]
    first[~hit.any(axis=1)] = horizon * penalty_mult
    return first
    """Evaluations to reach absolute regret ``|tol|``, censored at the budget.

    THE primary BO metric for this project (S22, and the metric policy in
    .llms/rules/metrics.md): "how many evaluations until I have a very good
    configuration". Runs that never reach the target are right-censored at one past
    the budget rather than dropped, so a method cannot look fast by solving only the
    easy runs -- the failure mode that makes a median-over-solved-runs misleading.
    """
    R = study.regret[method]
    hit = R <= -tol
    first = np.argmax(hit, axis=1).astype(float)
    first[~hit.any(axis=1)] = R.shape[1]  # censored: one past the last index
    return first


def critical_value(df: int) -> float:
    """Two-sided 95% critical value for ``df`` degrees of freedom.

    The fixed-task SE is now dominated by the between-prior spread, which is estimated
    from P prior draws and therefore carries **P-1 degrees of freedom** -- 2 at the
    project's usual P=3, not the ~46 a naive per-task count suggests, because all 23
    per-task terms come from the same three draws. Comparing against the normal value of
    1.96 is anti-conservative by more than a factor of two at that df (round-6 S3).

    Small table rather than scipy: this module runs under python3 where scipy is not
    guaranteed, and the values are standard.
    """
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
        30: 2.04,
        60: 2.00,
    }
    if df <= 0:
        return float("inf")
    if df in table:
        return table[df]
    if df > 60:
        return 1.96
    below = max(k for k in table if k < df)
    above = min(k for k in table if k > df)
    w = (df - below) / (above - below)
    return table[below] * (1 - w) + table[above] * w


def fixed_effects_test(
    pooled, a: str, b: str, tol: float | None = None, metric: str = "final"
) -> None:
    """The fixed-task estimand: which method wins ON THIS SUITE of 23 tasks?

    ``clustered_paired_z`` answers a different and much more demanding question -- which
    method wins on a NEW, UNSEEN task -- by treating the 23 tasks as a random sample. For
    that estimand the between-task spread is irreducible noise and T caps the precision,
    which is what S27.6's "unreachable" verdicts describe.

    Benchmark comparisons normally report the fixed-task estimand instead: the tasks ARE
    the population, so between-task spread is structure, not noise, and

        theta = (1/T) sum_t theta_t,   Var(theta_hat) = (1/T^2) sum_t s_t^2 / n_t

    which shrinks as 1/n with **no floor**. Priors and seeds do buy resolution here. Both
    are legitimate; they answer different questions, and a claim must say which one it is.
    """
    if tol is not None and metric == "budget":
        d = budget_censored(pooled, a, tol) - budget_censored(pooled, b, tol)
    elif tol is not None:
        d = solved_indicator(pooled, a, tol) - solved_indicator(pooled, b, tol)
    elif metric == "auc":
        d = pooled.auc(a) - pooled.auc(b)
    else:
        d = pooled.at(a) - pooled.at(b)

    tasks = np.unique(pooled.ds_idx)
    groups = [d[pooled.ds_idx == t] for t in tasks]
    # Equalise priors WITHIN each task when several are pooled: arm C's p0 contributes
    # 5 seeds against p1/p2's 3, so a plain per-task mean tilts toward p0 (review #1).
    if hasattr(pooled, "task_wmean"):
        per_task = np.array([pooled.task_wmean(d, t) for t in tasks])
        # The SE must describe the SAME estimator as the effect, otherwise z divides a
        # weighted numerator by an unweighted denominator.
        var_terms = [v for v in (pooled.task_wvar(d, t) for t in tasks) if v > 0.0]
    else:
        per_task = np.array([g.mean() for g in groups])
        var_terms = [g.var(ddof=1) / g.size for g in groups if g.size > 1]
    effect = float(per_task.mean())
    se = math.sqrt(sum(var_terms)) / len(tasks)
    z = effect / se if se > 0 else float("nan")
    # LOO folds share T-2 of T-1 pre-training tasks, so the per-task differences are
    # positively equicorrelated and sqrt(sum Var_t)/T is ANTI-CONSERVATIVE. Report the
    # inflation a modest correlation would imply rather than pretending rho = 0
    # (round-4 S2). rho is not estimable from one run; 0.05 and 0.15 bracket it.
    _T = len(tasks)
    _vif = {r: 1.0 + (_T - 1) * r for r in (0.05, 0.15)}

    n_per = float(np.mean([g.size for g in groups]))
    within = float(np.mean([g.var(ddof=1) for g in groups if g.size > 1]))
    # Runs per task needed for |z| = 2 under this estimand.
    need_se = abs(effect) / 2.0
    n_needed = within / (need_se**2 * len(tasks)) if need_se > 0 else float("inf")

    wins = int((per_task > 0).sum())
    label = "" if tol is None else f"   [solved to {-tol:g}]"
    if tol is not None and metric == "budget":
        label = f"   [evals to {-tol:g}, censored]"
    if tol is None and metric == "auc":
        label = "   [regret-AUC]"
    # Direction depends on the metric: final regret, AUC and evals-to-target are all
    # better LOWER; only solve RATE is better higher. One convention for all inverts
    # half the verdicts.
    higher_is_better = tol is not None and metric != "budget"
    # Ties were being folded into `a`'s win count in the lower-is-better branch only,
    # so the same tie counted differently depending on the metric (review, metric #3).
    # Count them explicitly and report them rather than silently attributing them.
    ties = int(sum(1 for v in per_task if float(v) == 0.0))
    if not higher_is_better:
        wins = len(tasks) - wins - ties
    # The SE is dominated by the between-prior spread, estimated from P draws, so the
    # threshold must come from t(P-1), not the normal 1.96. At P=3 that is 4.30 -- every
    # earlier |z|>2 verdict in this report was anti-conservative (round-6 S3).
    df = max(1, getattr(pooled, "n_priors", 1) - 1)
    crit = critical_value(df)
    favours_a = (z > crit) if higher_is_better else (z < -crit)
    favours_b = (z < -crit) if higher_is_better else (z > crit)
    verdict = f"{a} better" if favours_a else (f"{b} better" if favours_b else "tied")
    if se <= 0:
        # Every per-task difference identical across priors -- common on the binary
        # solve-rate metric. Reporting 'tied' beside a nonzero effect would be a
        # silent wrong answer, so name it (round-7 R7-4ii).
        verdict = (
            "UNDETERMINED (zero between-prior spread; SE unestimable)"
            if abs(effect) > 0
            else "tied (exactly zero effect)"
        )
    print(f"\n  {a} - {b}{label}")
    print(f"    effect (mean over tasks)  {effect:+.5f}")
    print(
        f"    fixed-task SE             {se:.5f}   t = {z:+.2f}   {verdict}"
        f"   [crit t({df})={crit:.2f}]"
    )
    if hasattr(pooled, "report_gaps"):
        pooled.report_gaps()
    if math.isfinite(z):
        print(
            f"    LOO-dependence caveat     z would be "
            f"{z / math.sqrt(_vif[0.05]):+.2f} at rho=0.05, "
            f"{z / math.sqrt(_vif[0.15]):+.2f} at rho=0.15"
        )
    else:
        print("    LOO-dependence caveat     n/a (z undefined; SE is zero)")
    print(f"    tasks where {a[:14]:14s} wins  {wins}/{len(tasks)}  (ties {ties})")
    print(
        f"    runs/task for |z|=2       {n_needed:.0f} (have {n_per:.0f}, "
        f"{n_needed / n_per:.1f}x)"
    )


def variance_decomposition(
    pooled, a: str, b: str, tol: float | None = None, metric: str = "final"
) -> None:
    """Can more pre-training priors buy power, or is 23 tasks a hard ceiling?

    The SEM of a task-clustered paired test is ``sd(cluster means)/sqrt(T)``, and

        Var(cluster mean) = sigma_between^2 + sigma_within^2 / n_per_cluster

    Only the second term shrinks when priors or seeds are added. If the between-task
    component dominates, no amount of extra compute moves any z in S27 and the honest
    move is to stop running BO sweeps on PD1. If the within component dominates, priors
    are a cheap lever. This decides whether S5 items 2 and 3 deserve an overnight slot.

    ``tol`` selects the metric: ``None`` uses final regret, a negative value uses the
    binary "solved to absolute regret |tol|" indicator that S27.4's trend is stated on.
    """
    if tol is None:
        d = pooled.at(a) - pooled.at(b)
    else:
        d = solved_indicator(pooled, a, tol) - solved_indicator(pooled, b, tol)
    tasks = np.unique(pooled.ds_idx)
    groups = [d[pooled.ds_idx == t] for t in tasks]
    n_per = float(np.mean([g.size for g in groups]))
    means = np.array([g.mean() for g in groups])

    # np.mean([]) is NaN plus a warning nobody reads, and the NaN then propagates
    # into the printed verdict (review N15).
    within_terms = [g.var(ddof=1) for g in groups if g.size > 1]
    if not within_terms:
        print(f"\n  {a} - {b}: every task has a single run; cannot decompose")
        return
    within = float(np.mean(within_terms))
    observed = float(means.var(ddof=1))
    # The observed spread of cluster means already contains within/n; remove it.
    between = max(observed - within / n_per, 0.0)
    # Keep the sign: an abs() here prints a positive z whichever method won, which
    # is the exact direction hazard metrics.md says has already caused two bugs.
    effect_signed = float(means.mean())
    effect = abs(effect_signed)
    sem = math.sqrt(observed / len(tasks))
    floor = math.sqrt(between / len(tasks))

    print(f"\n  {a} - {b}" + ("" if tol is None else f"   [solved to {-tol:g}]"))
    print(f"    |effect|                    {effect:.5f}")
    print(
        f"    current SEM                 {sem:.5f}   (z = {effect_signed / sem:+.2f})"
        if sem > 0
        else f"    current SEM                 {sem:.5f}   (z undefined)"
    )
    print(f"    between-task sd             {math.sqrt(between):.5f}")
    print(
        f"    within-task sd              {math.sqrt(within):.5f}  n/task={n_per:.0f}"
    )
    print(f"    SEM floor, infinite priors  {floor:.5f}")

    need = effect / 2.0  # SEM required for |z| = 2
    if need <= floor:
        print("    VERDICT: UNREACHABLE - between-task variance alone exceeds the")
        print("             SEM a z=2 result needs. More compute cannot deliver it.")
        return
    target = need**2 * len(tasks) - between
    n_needed = within / target if target > 0 else float("inf")
    print(
        f"    VERDICT: reachable - needs ~{n_needed:.0f} runs/task "
        f"({n_needed / n_per:.1f}x current)"
    )


def shrinkage_screen() -> None:
    """Does EM's M-step shrinkage help on PD1 BO? Existing data, never analysed.

    Every headline BO run -- LCBench `bo_multisplit` AND PD1 `pd1pool` -- used the
    DEFAULT `--em-shrinkage 0.0`, i.e. shrinkage OFF. The evidence that shrinkage is
    EM's largest single improvement (S23.1, "worth up to 14 nats") comes from 7D
    REGRESSION NLL, and was never carried into any BO comparison.

    `results/raw/followup/pd1_shrink_full_a*.json` sweeps alpha on PD1 with the full
    candidate pool. Same shard, seeds and tasks across the three levels, so runs pair
    index-wise. Reported on the primary metric per .llms/rules/metrics.md.
    """
    alphas = ("0.0", "0.1", "0.3")
    studies = {}
    for a in alphas:
        try:
            studies[a] = Study(
                os.path.join(ROOT, f"results/raw/followup/pd1_shrink_full_a{a}.json")
            )
        except FileNotFoundError:
            print(f"MISSING alpha={a}")
            return

    base = studies["0.0"]
    for a in alphas[1:]:
        if not np.array_equal(base.ds_idx, studies[a].ds_idx):
            print("run alignment differs across alpha; cannot pair")
            return

    print("\n" + "=" * 78)
    print("SHRINKAGE SCREEN ON PD1 (existing data, previously unanalysed)")
    print(f"arm B, {base.n_runs} runs, {len(np.unique(base.ds_idx))} tasks")
    print("=" * 78)

    for tol in (-0.01, -0.001):
        print(f"\n  mean evaluations to reach |regret| <= {-tol:g} (lower is better)")
        header = "    %-16s" % "method" + "".join(f"a={a:<8}" for a in alphas)
        print(header)
        for m in base.methods:
            cells = []
            for a in alphas:
                cells.append(f"{budget_censored(studies[a], m, tol).mean():<10.2f}")
            print("    %-16s" % m + "".join(cells))

        # Paired: does turning shrinkage on beat leaving it off, on the same runs?
        for m in [x for x in base.methods if x.startswith("em")]:
            for a in alphas[1:]:
                d = budget_censored(studies[a], m, tol) - budget_censored(base, m, tol)
                diff, z, n = clustered_paired_z_values(base.ds_idx, d)
                tag = "better" if z < -2 else ("worse" if z > 2 else "no change")
                print(
                    f"      {m:16s} a={a} vs a=0.0: {diff:+.2f} evals "
                    f"z={z:+.2f} ({tag})"
                )


def drift_comparison() -> None:
    """Does the code drift (S27.10) change S27's CONCLUSIONS, or only the bits?

    Stage A re-ran the arm-C configuration on current code. The bits are already known
    not to reproduce. What matters is whether the method ORDERING and the significance
    verdicts move -- if they do, S27 must be re-run before anything is published.

    Scored on budget-to-target first per .llms/rules/metrics.md, with final regret as
    the companion. PACOH is absent from the re-run and is excluded from both sides.
    """
    old_paths = sorted(
        glob.glob(os.path.join(ROOT, "results/raw/pd1pool/fullboth_s*.json"))
    )
    new_paths = sorted(
        glob.glob(os.path.join(ROOT, "results/raw/drift_check/rerun_s*.json"))
    )
    if not old_paths or not new_paths:
        print("\nDRIFT COMPARISON: inputs missing")
        return

    old, new = Study(",".join(old_paths)), Study(",".join(new_paths))
    shared = [m for m in old.methods if m in new.methods]

    print("\n" + "=" * 78)
    print("S27 DRIFT: does the ordering survive? (2026-08-08 vs current code)")
    print(f"old {old.n_runs} runs, new {new.n_runs} runs, {len(shared)} shared methods")
    print("=" * 78)

    for tol in (-0.01, -0.001):
        print(f"\n  mean evaluations to |regret| <= {-tol:g}  (PRIMARY metric)")
        rows = []
        for m in shared:
            rows.append(
                (
                    m,
                    budget_censored(old, m, tol).mean(),
                    budget_censored(new, m, tol).mean(),
                )
            )
        old_rank = [m for m, _, _ in sorted(rows, key=lambda r: r[1])]
        new_rank = [m for m, _, _ in sorted(rows, key=lambda r: r[2])]
        print("    %-18s %10s %10s %8s" % ("method", "old", "new", "delta"))
        for m, o, n in sorted(rows, key=lambda r: r[1]):
            print("    %-18s %10.2f %10.2f %+8.2f" % (m, o, n, n - o))
        print(f"    old order: {' < '.join(old_rank)}")
        print(f"    new order: {' < '.join(new_rank)}")
        print(f"    ORDERING {'PRESERVED' if old_rank == new_rank else 'CHANGED'}")

    print("\n  final regret, mean")
    rows = [(m, old.at(m).mean(), new.at(m).mean()) for m in shared]
    old_rank = [m for m, _, _ in sorted(rows, key=lambda r: r[1])]
    new_rank = [m for m, _, _ in sorted(rows, key=lambda r: r[2])]
    for m, o, n in sorted(rows, key=lambda r: r[1]):
        print("    %-18s %10.5f %10.5f %+9.5f" % (m, o, n, n - o))
    print(f"    ORDERING {'PRESERVED' if old_rank == new_rank else 'CHANGED'}")

    # The decisive question is not the ranking but whether the SIGNIFICANCE VERDICTS
    # survive. If they do, S27's conclusions stand and only its numbers moved.
    print("\n  do the fixed-task verdicts survive? (primary metric, |z| > 2)")
    for tol in (-0.01, -0.001):
        print(f"\n    target {-tol:g}")
        for a, b in (
            ("hyperbo_frozen", "em_finetuned"),
            ("hyperbo_adapt", "em_finetuned"),
            ("ablr", "em_finetuned"),
            ("em_finetuned", "vanilla_gp"),
        ):
            cells = []
            for study in (old, new):
                d = budget_censored(study, a, tol) - budget_censored(study, b, tol)
                tasks = np.unique(study.ds_idx)
                groups = [d[study.ds_idx == t] for t in tasks]
                per_task = np.array([g.mean() for g in groups])
                var_terms = [g.var(ddof=1) / g.size for g in groups if g.size > 1]
                eff = float(per_task.mean())
                se = math.sqrt(sum(var_terms)) / len(tasks)
                z = eff / se if se > 0 else float("nan")
                # Fewer evaluations is better, so z < -2 favours `a`.
                cells.append((a if z < -2 else (b if z > 2 else "tied"), z))
            (vo, zo), (vn, zn) = cells
            flag = "" if vo == vn else "   <-- VERDICT CHANGED"
            print(
                f"      {a:15s} vs {b:15s} old: {vo:14s} z={zo:+.2f} | "
                f"new: {vn:14s} z={zn:+.2f}{flag}"
            )


def hybrid_analysis() -> None:
    """Does the additive HyperBO kernel close EM's PD1 gap? (S27.5's open question)

    Uses ONLY the hybrid sweep's own runs. S27.10 established these do not reproduce
    `fullboth*.json`, so the sweep carries `em_frozen` and `hyperbo_frozen` as internal
    reference arms and every comparison here is within-sweep.

    Scored on budget-to-target, fixed-task estimand, per .llms/rules/metrics.md.
    """
    pats = [
        "results/raw/pd1pool/hybrid_p0_s*.json",
        "results/raw/pd1pool/hybrid_p1_s*.json",
        "results/raw/pd1pool/hybrid_p2_s*.json",
    ]
    if not all(glob.glob(os.path.join(ROOT, p)) for p in pats):
        print("\nHYBRID ANALYSIS: inputs missing")
        return
    pooled = PooledStudy(pats)

    print("\n" + "=" * 78)
    print("THE ADDITIVE HYBRID ON PD1 ARM C (within-sweep only; see S27.10)")
    print(f"{pooled.n_priors} priors, {pooled.n_runs} runs, methods: {pooled.methods}")
    print("=" * 78)

    for tol in (-0.01, -0.001):
        print(f"\n  mean evaluations to |regret| <= {-tol:g} (lower is better)")
        for m in sorted(
            pooled.methods,
            key=lambda x: pooled.wmean(budget_censored(pooled, x, tol)),
        ):
            print(f"    {m:28s} {pooled.wmean(budget_censored(pooled, m, tol)):6.2f}")
        print("\n  paired, fixed-task:")
        for a, b in (
            ("em_additive_hyperbo", "em_frozen"),
            ("em_additive_hyperbo_frozen", "em_frozen"),
            ("em_additive_hyperbo", "hyperbo_frozen"),
            ("em_frozen", "hyperbo_frozen"),
        ):
            if a in pooled.methods and b in pooled.methods:
                fixed_effects_test(pooled, a, b, tol=tol, metric="budget")


def lcbench_hybrid_analysis() -> None:
    """Does the additive hybrid beat standalone HyperBO on LCBench *BO*?

    S26.3 found the hybrid tops all four LCBench cells and beats standalone HyperBO on
    both metrics -- but those are REGRESSION cells scored on NLL/RMSE. Whether the win
    transfers to BO has never been checked, and `hb_hybrid/bo_lcb_*` (4 cells, 2 canonical
    settings x 2 priors, 36 runs each) was sitting unanalysed.

    This matters for interpreting S27.14: on PD1 BO the hybrid improves EM but does not
    reach standalone HyperBO. If the same holds on LCBench BO, the regression win simply
    does not transfer to BO on either benchmark, and the mean function -- which the hybrid
    does NOT inherit -- becomes the leading explanation.

    Scored on budget-to-target, fixed-task estimand, per .llms/rules/metrics.md.
    """
    for canon in ("sumll", "hyperbo"):
        pats = [f"results/raw/hb_hybrid/bo_lcb_{canon}_p{p}.json" for p in (0, 1)]
        if not all(os.path.exists(os.path.join(ROOT, p)) for p in pats):
            print(f"\nLCBENCH BO HYBRID [{canon}]: inputs missing")
            continue
        pooled = PooledStudy(pats)
        print("\n" + "=" * 78)
        print(f"LCBENCH BO x ADDITIVE HYBRID -- canonical={canon}")
        print(f"{pooled.n_priors} priors, {pooled.n_runs} runs")
        print("=" * 78)
        for tol in (-0.01, -0.001):
            print(f"\n  mean evaluations to |regret| <= {-tol:g} (lower is better)")
            ranked = sorted(
                pooled.methods,
                key=lambda m: pooled.wmean(budget_censored(pooled, m, tol)),
            )
            for m in ranked:
                print(
                    f"    {m:30s} {pooled.wmean(budget_censored(pooled, m, tol)):6.2f}"
                )
            for a, b in (
                ("em_additive_hyperbo", "hyperbo_frozen"),
                ("em_additive_hyperbo_frozen", "hyperbo_frozen"),
                ("em_additive_hyperbo", "em_frozen"),
            ):
                if a in pooled.methods and b in pooled.methods:
                    fixed_effects_test(pooled, a, b, tol=tol, metric="budget")


def main() -> None:
    studies: dict[str, Study] = {}
    for name, pat in ARMS.items():
        try:
            studies[name] = load(pat)
        except FileNotFoundError:
            print(f"MISSING: {name} ({pat})")

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    for name, s in studies.items():
        c = s.config
        print(
            f"{name:20s} runs={s.n_runs:4d} tasks={len(np.unique(s.ds_idx)):3d} "
            f"pts={s.n_pts:3d} bench={c.get('benchmark')} "
            f"cand={c.get('pd1_candidate_pool')} pre={c.get('pd1_pretrain_pool')} "
            f"seeds={c.get('n_seeds')} hb_iters={c.get('hyperbo_iters')} "
            f"pretrain_seed={c.get('pretrain_seed')}"
        )

    # Sanity: the arms must NOT be identical. A silent no-op would show up here as
    # equal final regret across arms (gotcha #1).
    print("\nControl -- mean final regret per arm (must differ across arms):")
    for name, s in studies.items():
        vals = {m: float(s.at(m, None).mean()) for m in s.methods if m in TRANSFER}
        print(f"  {name:20s} " + "  ".join(f"{m}={v:.5f}" for m, v in vals.items()))

    print("\n" + "=" * 78)
    print("WITHIN-ARM ORDERING -- final regret, mean +- clustered SEM")
    print("=" * 78)
    for name, s in studies.items():
        rows = []
        for m in s.methods:
            v = s.at(m, None)
            rows.append((m, float(v.mean()), s.clustered_sem(v)))
        rows.sort(key=lambda r: r[1])
        print(f"\n{name}")
        for i, (m, mu, sem) in enumerate(rows, 1):
            print(f"  {i}. {m:16s} {mu:.5f} +- {sem:.5f}")

    print("\n" + "=" * 78)
    print("BUDGET TO TARGET -- median evals (solve rate), absolute regret thresholds")
    print("=" * 78)
    for name, s in studies.items():
        print(f"\n{name}")
        for m in s.methods:
            cells = []
            for tol in (-0.05, -0.01, -0.001):
                med, _, rate = budget_to_target(s, m, tol)
                cells.append("never" if rate == 0 else f"{med:.0f} ({rate:.0%})")
            print(f"  {m:16s} " + "  ".join(f"{c:>12s}" for c in cells))

    print("\n" + "=" * 78)
    print("THE S21.3 CLAIM -- does HyperBO beat EM once the pool goes off-grid?")
    print("negative difference favours the FIRST method; |z|>2 is significant")
    print("=" * 78)
    for name, s in studies.items():
        print(f"\n{name}")
        for a, b in (
            ("hyperbo_frozen", "em_finetuned"),
            ("hyperbo_frozen", "em_frozen"),
            ("ablr", "em_finetuned"),
        ):
            if a in s.methods and b in s.methods:
                diff, z, n = clustered_paired_z(s, a, b)
                verdict = (
                    f"{a} better" if z < -2 else (f"{b} better" if z > 2 else "tied")
                )
                print(f"  {a:15s} - {b:15s} {diff:+.5f}  z={z:+.2f} (n={n})  {verdict}")

    # ------------------------------------------------------------------ pooled arm C
    # Three priors, clustered by task: the properly-powered version of the S21.3 test.
    pooled = PooledStudy(
        [ARMS["C full/full p0"], ARMS["C full/full p1"], ARMS["C full/full p2"]]
    )
    print("\n" + "=" * 78)
    print(f"POOLED ARM C -- {pooled.n_priors} priors, {pooled.n_runs} runs")
    print("=" * 78)
    rows = [
        (m, pooled.wmean(pooled.at(m)), pooled.clustered_sem(pooled.at(m)))
        for m in pooled.methods
    ]
    rows.sort(key=lambda r: r[1])
    print("\nfinal regret, mean +- clustered SEM")
    for i, (m, mu, sem) in enumerate(rows, 1):
        print(f"  {i}. {m:16s} {mu:.5f} +- {sem:.5f}")

    print("\npaired final-regret differences (clustered by task)")
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_frozen", "em_frozen"),
        ("ablr", "em_finetuned"),
        ("em_finetuned", "vanilla_gp"),
    ):
        diff, z, n = clustered_paired_z_values(
            pooled.ds_idx, pooled.at(a) - pooled.at(b)
        )
        verdict = f"{a} better" if z < -2 else (f"{b} better" if z > 2 else "tied")
        print(f"  {a:15s} - {b:15s} {diff:+.5f}  z={z:+.2f} (n={n})  {verdict}")

    print("\nSOLVE RATE to absolute regret, and the paired EM-vs-HyperBO gap")
    for tol in (-0.05, -0.01, -0.001):
        print(f"\n  target |regret| <= {-tol:g}")
        for m in pooled.methods:
            print(f"    {m:16s} {pooled.wmean(solved_indicator(pooled, m, tol)):.0%}")
        for a, b in (
            ("hyperbo_frozen", "em_finetuned"),
            ("hyperbo_adapt", "em_finetuned"),
            ("ablr", "em_finetuned"),
        ):
            d = solved_indicator(pooled, a, tol) - solved_indicator(pooled, b, tol)
            diff, z, n = clustered_paired_z_values(pooled.ds_idx, d)
            verdict = f"{a} better" if z > 2 else (f"{b} better" if z < -2 else "tied")
            print(
                f"    paired {a:15s} - {b:15s} {diff:+.3f}  z={z:+.2f} "
                f"(n={n})  {verdict}"
            )

    print("\n" + "=" * 78)
    print("POWER: is 23 task clusters a hard ceiling, or do more priors help?")
    print("=" * 78)
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
        ("ablr", "em_finetuned"),
        ("em_finetuned", "vanilla_gp"),
    ):
        variance_decomposition(pooled, a, b)

    # S27.4's trend is stated on solve rate, not final regret, so it needs its own
    # power check before S5 item 2 is worth any compute.
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
    ):
        variance_decomposition(pooled, a, b, tol=-0.01)
        variance_decomposition(pooled, a, b, tol=-0.001)

    print("\n" + "=" * 78)
    print("FIXED-TASK ESTIMAND: which method wins ON THIS SUITE?")
    print("(the benchmark question; no variance floor, priors/seeds do buy resolution)")
    print("=" * 78)
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
        ("ablr", "em_finetuned"),
        ("em_finetuned", "vanilla_gp"),
    ):
        fixed_effects_test(pooled, a, b)
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
    ):
        fixed_effects_test(pooled, a, b, tol=-0.01)
        fixed_effects_test(pooled, a, b, tol=-0.001)

    print("\n" + "=" * 78)
    print("PRIMARY METRIC: evaluations to reach a very good configuration")
    print("(budget-to-target, censored at the budget; fixed-task estimand)")
    print("=" * 78)
    for tol in (-0.01, -0.001):
        for a, b in (
            ("hyperbo_frozen", "em_finetuned"),
            ("hyperbo_adapt", "em_finetuned"),
            ("ablr", "em_finetuned"),
            ("em_finetuned", "vanilla_gp"),
        ):
            fixed_effects_test(pooled, a, b, tol=tol, metric="budget")

    # S26.1's metric, kept only as a cross-check. Per S22 and the metric policy in
    # .llms/rules/metrics.md, AUC is NOT a headline metric for this project.
    print("\n" + "=" * 78)
    print("SECONDARY CROSS-CHECK ONLY: regret-AUC (not a headline metric, see S22)")
    print("=" * 78)
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
        ("ablr", "em_finetuned"),
        ("em_finetuned", "vanilla_gp"),
    ):
        fixed_effects_test(pooled, a, b, metric="auc")

    print("\n  loose target, for contrast with the tight ones above")
    for a, b in (
        ("hyperbo_frozen", "em_finetuned"),
        ("hyperbo_adapt", "em_finetuned"),
    ):
        fixed_effects_test(pooled, a, b, tol=-0.05)

    shrinkage_screen()
    drift_comparison()
    hybrid_analysis()
    lcbench_hybrid_analysis()


if __name__ == "__main__":
    main()
