# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent / "results" / "raw" / "v2")
BO = ["stage2means_lcb_bo", "stage2means_pd1_bo_armA", "stage2means_pd1_bo_armC"]
REG = ["stage2means_lcb_reg"]
ALL = BO + REG + ["stage2means_pd1_reg"]
BASE = "mean_const_emp"
CONFIGS = [
    "mean_const_emp",
    "mean_lin_emp",
    "mean_const_hb",
    "mean_lin_hb",
    "mean_const_b25",
    "mean_const_b50",
    "mean_const_b75",
    "mean_lin_b50",
    "mean_const_prior",
    "mean_lin_prior",
]
METHODS = [
    "em_frozen",
    "em_additive_hyperbo",
    "em_additive_hyperbo_frozen",
    "hyperbo_frozen",
]
CELLRE = re.compile(r"^(.*)_p(\d+)$")


def load(stage):
    with open(os.path.join(ROOT, stage, "SUMMARY.json")) as f:
        return json.load(f)


def per_prior(d, metric, method):
    """-> {config: {prior: value}}"""
    out = defaultdict(dict)
    for k, c in d["cells"].items():
        m = CELLRE.match(k)
        cfg, p = m.group(1), int(m.group(2))
        md = c["methods"].get(method)
        if md is None or metric not in md:
            continue
        out[cfg][p] = md[metric]
    return out


def mean_se(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan")
    v = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(v / n)


def paired(a, b, priors):
    """a - b paired over priors -> (mean, se, t, n)"""
    d = [a[p] - b[p] for p in priors]
    mu, se = mean_se(d)
    t = mu / se if se and se > 0 else (0.0 if mu == 0 else float("inf"))
    return mu, se, t, len(d)


def fmt(x, n=3):
    if x is None:
        return "  n/a"
    if isinstance(x, float) and (math.isnan(x)):
        return "  nan"
    return f"{x:.{n}f}"


# ---------------------------------------------------------------- no-op check
print("=" * 78)
print("Q3a  NO-OP CHECK: do the 10 mean variants produce different numbers?")
print("=" * 78)
NOOP_METRIC = {
    "stage2means_lcb_bo": "budget_to_0.01",
    "stage2means_pd1_bo_armA": "budget_to_0.01",
    "stage2means_pd1_bo_armC": "budget_to_0.01",
    "stage2means_lcb_reg": "n5_rmse",
    "stage2means_pd1_reg": "n5_rmse",
}
noop_summary = {}
for st in ALL:
    d = load(st)
    metric = NOOP_METRIC[st]
    print(
        f"\n-- {st}  (metric {metric}, cells {d['cells_present']}/{d['cells_expected']})"
    )
    for meth in METHODS:
        pp = per_prior(d, metric, meth)
        cfgs = [c for c in CONFIGS if c in pp]
        if len(cfgs) < 2:
            print(f"   {meth:28s}  only {len(cfgs)} config(s) present -- cannot test")
            continue
        priors = sorted(set.intersection(*[set(pp[c]) for c in cfgs]))
        n_ident, n_diff, distinct_tot = 0, 0, []
        for p in priors:
            vals = [pp[c][p] for c in cfgs]
            k = len(set(vals))
            distinct_tot.append(k)
            if k == 1:
                n_ident += 1
            else:
                n_diff += 1
        # how many configs are bit-identical to baseline across ALL priors
        ident_cfgs = [
            c
            for c in cfgs
            if c != BASE and all(pp[c][p] == pp[BASE][p] for p in priors)
        ]
        avg_distinct = sum(distinct_tot) / len(distinct_tot)
        verdict = (
            "NO-OP" if avg_distinct == 1.0 else ("partial" if ident_cfgs else "LIVE")
        )
        print(
            f"   {meth:28s}  priors={len(priors)}  distinct values/prior "
            f"= {avg_distinct:.2f} of {len(cfgs)}   -> {verdict}"
        )
        if ident_cfgs:
            print(f"        bit-identical to {BASE} on every prior: {ident_cfgs}")
        noop_summary[(st, meth)] = (avg_distinct, len(cfgs), ident_cfgs)

# ------------------------------------------------------------------- ranking
print()
print("=" * 78)
print("Q1  RANKING per regime  (method em_frozen; priors = replication unit, P=9)")
print("=" * 78)

RANK_SPEC = [
    ("stage2means_lcb_bo", "budget_to_0.01", "lower", "final_regret_mean"),
    ("stage2means_pd1_bo_armA", "budget_to_0.01", "lower", "final_regret_mean"),
    ("stage2means_pd1_bo_armC", "budget_to_0.01", "lower", "final_regret_mean"),
    ("stage2means_lcb_reg", "n5_rank_corr", "higher", "n5_rmse"),
]
rank_tables = {}
for meth in ["em_frozen", "em_additive_hyperbo"]:
    print(f"\n################ METHOD = {meth} ################")
    for st, metric, direction, companion in RANK_SPEC:
        d = load(st)
        pp = per_prior(d, metric, meth)
        cp = per_prior(d, companion, meth)
        cfgs = [c for c in CONFIGS if c in pp]
        priors = sorted(set.intersection(*[set(pp[c]) for c in cfgs]))
        rows = []
        for c in cfgs:
            mu, se = mean_se([pp[c][p] for p in priors])
            cmu, cse = mean_se([cp[c][p] for p in priors])
            if c == BASE:
                dm, ds, t, _ = 0.0, 0.0, 0.0, len(priors)
            else:
                dm, ds, t, _ = paired(pp[c], pp[BASE], priors)
            rows.append((c, mu, se, cmu, cse, dm, ds, t))
        rows.sort(key=lambda r: r[1], reverse=(direction == "higher"))
        rank_tables[(meth, st)] = [r[0] for r in rows]
        print(f"\n  == {st}   primary {metric} ({direction} better), P={len(priors)}")
        print(
            f"     {'rank':>4} {'config':18s} {'mean':>8} {'±SE':>7} | "
            f"{companion:>18} {'±SE':>8} | {'Δ vs base':>9} {'±SE':>7} {'t(8)':>7}  sig"
        )
        for i, (c, mu, se, cmu, cse, dm, ds, t) in enumerate(rows, 1):
            sig = "" if c == BASE else ("*" if abs(t) > 2.31 else "")
            tag = " (base)" if c == BASE else ""
            print(
                f"     {i:>4} {c:18s} {fmt(mu):>8} {fmt(se):>7} | "
                f"{fmt(cmu, 4):>18} {fmt(cse, 4):>8} | {fmt(dm):>9} {fmt(ds):>7} "
                f"{fmt(t, 2):>7}  {sig}{tag}"
            )

# ------------------------------------------------- spread vs between-prior SE
print()
print("=" * 78)
print("Q3b  EFFECT SIZE: spread across mean variants vs between-prior SE")
print("=" * 78)
for meth in ["em_frozen", "em_additive_hyperbo"]:
    print(f"\n-- {meth}")
    for st, metric, direction, _ in RANK_SPEC:
        d = load(st)
        pp = per_prior(d, metric, meth)
        cfgs = [c for c in CONFIGS if c in pp]
        priors = sorted(set.intersection(*[set(pp[c]) for c in cfgs]))
        means = [mean_se([pp[c][p] for p in priors])[0] for c in cfgs]
        spread = max(means) - min(means)
        ses = [mean_se([pp[c][p] for p in priors])[1] for c in cfgs]
        avg_se = sum(ses) / len(ses)
        # largest |paired t| against base
        ts = []
        for c in cfgs:
            if c == BASE:
                continue
            _, _, t, _ = paired(pp[c], pp[BASE], priors)
            ts.append((abs(t), c))
        ts.sort(reverse=True)
        print(
            f"   {st:26s} {metric:14s} spread(max-min of config means) = {spread:.4f}"
            f"   mean between-prior SE = {avg_se:.4f}"
            f"   ratio = {spread / avg_se if avg_se else float('nan'):.2f}"
        )
        print(
            f"        largest |paired t(8)| vs base: {ts[0][1]} t={ts[0][0]:.2f}"
            f"   (# of 9 with |t|>2.31: {sum(1 for a, _ in ts if a > 2.31)})"
        )

# --------------------------------------------------------------- consistency
print()
print("=" * 78)
print("Q2  CONSISTENCY of the winner across regimes")
print("=" * 78)


def spearman(r1, r2):
    n = len(r1)
    p1 = {c: i for i, c in enumerate(r1)}
    p2 = {c: i for i, c in enumerate(r2)}
    dsq = sum((p1[c] - p2[c]) ** 2 for c in r1)
    return 1 - 6 * dsq / (n * (n * n - 1))


for meth in ["em_frozen", "em_additive_hyperbo"]:
    print(f"\n-- {meth}")
    sts = [s for s, _, _, _ in RANK_SPEC]
    for st in sts:
        print(
            f"   {st:26s} winner = {rank_tables[(meth, st)][0]:18s} "
            f"worst = {rank_tables[(meth, st)][-1]}"
        )
    print("   pairwise Spearman of the 10-config ordering:")
    for i in range(len(sts)):
        for j in range(i + 1, len(sts)):
            print(
                f"      {sts[i][11:]:14s} vs {sts[j][11:]:14s}  rho = "
                f"{spearman(rank_tables[(meth, sts[i])], rank_tables[(meth, sts[j])]):+.3f}"
            )

# ------------------------------------------------------ arm A x C interaction
print()
print("=" * 78)
print("Q5  ARM x MEAN-VARIANT INTERACTION (PD1 arm A vs arm C, paired over priors)")
print("=" * 78)
for meth in ["em_frozen", "em_additive_hyperbo"]:
    dA = load("stage2means_pd1_bo_armA")
    dC = load("stage2means_pd1_bo_armC")
    ppA = per_prior(dA, "budget_to_0.01", meth)
    ppC = per_prior(dC, "budget_to_0.01", meth)
    cfgs = [c for c in CONFIGS if c in ppA and c in ppC]
    priors = sorted(
        set.intersection(*([set(ppA[c]) for c in cfgs] + [set(ppC[c]) for c in cfgs]))
    )
    print(f"\n-- {meth}   (Δ = config − {BASE}, within arm; DiD = Δ_C − Δ_A)")
    print(
        f"   {'config':18s} {'Δ_A':>8} {'t_A':>6} | {'Δ_C':>8} {'t_C':>6} | "
        f"{'DiD':>8} {'±SE':>7} {'t(8)':>6}  sig"
    )
    for c in cfgs:
        if c == BASE:
            continue
        dAm, dAs, tA, _ = paired(ppA[c], ppA[BASE], priors)
        dCm, dCs, tC, _ = paired(ppC[c], ppC[BASE], priors)
        did = [(ppC[c][p] - ppC[BASE][p]) - (ppA[c][p] - ppA[BASE][p]) for p in priors]
        m, s = mean_se(did)
        t = m / s if s else float("nan")
        sig = "*" if (s and abs(t) > 2.31) else ""
        print(
            f"   {c:18s} {fmt(dAm, 2):>8} {fmt(tA, 2):>6} | {fmt(dCm, 2):>8} {fmt(tC, 2):>6} | "
            f"{fmt(m, 2):>8} {fmt(s, 2):>7} {fmt(t, 2):>6}  {sig}"
        )

# ------------------------------------------------------------- pd1_reg partial
print()
print("=" * 78)
print("PARTIAL: stage2means_pd1_reg (9/90 cells) -- indicative only")
print("=" * 78)
d = load("stage2means_pd1_reg")
print("  configs present:", list(d["configs"].keys()))
for meth in METHODS:
    pp5 = per_prior(d, "n5_rank_corr", meth)
    pp5r = per_prior(d, "n5_rmse", meth)
    for c in pp5:
        priors = sorted(pp5[c])
        m1, s1 = mean_se([pp5[c][p] for p in priors])
        m2, s2 = mean_se([pp5r[c][p] for p in priors])
        print(
            f"   {meth:28s} {c:18s} P={len(priors)}  "
            f"n5_rank_corr {m1:.4f}±{s1:.4f}   n5_rmse {m2:.4f}±{s2:.4f}"
        )
