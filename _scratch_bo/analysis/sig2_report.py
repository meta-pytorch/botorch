# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import statistics as st
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent / "results" / "raw" / "v2")
SIG = ["1e-3", "1e-2", "1e-1", "0.3", "1", "3", "10"]
BASE = "1e-2"
P = 9
TC = {4: 2.776, 8: 2.306, 9: 2.262, 22: 2.074}


def load(r):
    return json.load(open(os.path.join(ROOT, "stage2noise_" + r, "SUMMARY.json")))


def mat(S, meth, metric):
    return {
        s: [
            S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"][meth][metric]
            for p in range(P)
        ]
        for s in SIG
    }


def tstat(d):
    m = sum(d) / len(d)
    if len(d) < 2:
        return m, float("nan"), float("nan")
    sdv = st.stdev(d)
    se = sdv / math.sqrt(len(d))
    t = m / se if se > 0 else (0.0 if m == 0 else float("inf"))
    return m, se, t


def profile(name, M, direction, note=""):
    means = {s: sum(M[s]) / len(M[s]) for s in SIG}
    best = (min if direction == "lower" else max)(SIG, key=lambda s: means[s])
    worst = (max if direction == "lower" else min)(SIG, key=lambda s: means[s])
    sd_base = st.stdev(M[BASE])
    se_base = sd_base / math.sqrt(len(M[BASE]))
    spread = max(means.values()) - min(means.values())
    print("\n  %s  [%s better]%s" % (name, direction, note))
    print("    sigma2 " + "".join("%9s" % s for s in SIG))
    print("    mean   " + "".join("%9.4f" % means[s] for s in SIG))
    rows = {"d_mean": [], "d_se": [], "t": []}
    for s in SIG:
        if s == BASE:
            rows["d_mean"].append("    ---")
            rows["d_se"].append("    ---")
            rows["t"].append("    ---")
            continue
        d = [M[s][i] - M[BASE][i] for i in range(len(M[s]))]
        m, se, t = tstat(d)
        rows["d_mean"].append("%9.4f" % m)
        rows["d_se"].append("%9.4f" % se)
        rows["t"].append("%9.2f" % t if math.isfinite(t) else "      inf")
    print("    d_mean " + "".join("%9s" % x.strip() for x in rows["d_mean"]))
    print("    d_se   " + "".join("%9s" % x.strip() for x in rows["d_se"]))
    print("    t      " + "".join("%9s" % x.strip() for x in rows["t"]))
    # per-replicate argmin
    cnt = {}
    for i in range(len(M[BASE])):
        b = (min if direction == "lower" else max)(SIG, key=lambda s: M[s][i])
        cnt[b] = cnt.get(b, 0) + 1
    print(
        "    argbest per replicate: "
        + ", ".join("%s:%d" % (s, cnt.get(s, 0)) for s in SIG if cnt.get(s))
    )
    print(
        "    argbest(pooled)=%s  argworst=%s  spread=%.5g  betweenRepSE@base=%.5g  spread/SE=%s"
        % (
            best,
            worst,
            spread,
            se_base,
            ("%.2f" % (spread / se_base)) if se_base > 0 else "INF (SE=0)",
        )
    )
    # flatness: neighbourhood of best
    ordered = [(s, means[s]) for s in SIG]
    return means, best, spread, se_base


print("#" * 96)
print("# PART 1 -- BO REGIMES: budget-to-target (primary), final regret (companion)")
print(
    "# replication unit = 9 pretrain_seed 'priors'; paired vs sigma2=1e-2; df=8, t_crit=2.306"
)
print("#" * 96)

for reg in ["lcb_bo", "pd1_bo_armA", "pd1_bo_armC"]:
    S = load(reg)
    nt = S["cells"]["emnoise_1e-2_p0"]["n_tasks"]
    print("\n" + "=" * 96)
    print(
        "REGIME %s  (n_tasks=%d, cells %d/%d)"
        % (reg, nt, S["cells_present"], S["cells_expected"])
    )
    print("=" * 96)
    for meth in [
        "em_frozen",
        "em_additive_hyperbo",
        "em_additive_hyperbo_frozen",
        "hyperbo_frozen",
    ]:
        print("\n [%s]" % meth)
        profile("budget_to_0.01", mat(S, meth, "budget_to_0.01"), "lower")
        profile("budget_to_0.001", mat(S, meth, "budget_to_0.001"), "lower")
        profile("final_regret_mean", mat(S, meth, "final_regret_mean"), "lower")
        profile("solve_rate_0.01", mat(S, meth, "solve_rate_0.01"), "higher")

print("\n\n" + "#" * 96)
print("# PART 2 -- REGRESSION REGIMES: rank_corr / rmse at n=5,20,50")
print(
    "# WARNING: for em_frozen the 9 'priors' are BIT-IDENTICAL (pretrain_seed is a no-op)."
)
print("#" * 96)

for reg in ["lcb_reg", "pd1_reg"]:
    S = load(reg)
    print("\n" + "=" * 96)
    print("REGIME %s  (cells %d/%d)" % (reg, S["cells_present"], S["cells_expected"]))
    print("=" * 96)
    for meth in [
        "em_frozen",
        "em_additive_hyperbo",
        "em_additive_hyperbo_frozen",
        "hyperbo_frozen",
    ]:
        print("\n [%s]" % meth)
        for n in [5, 20, 50]:
            profile("n%d_rank_corr" % n, mat(S, meth, "n%d_rank_corr" % n), "higher")
            profile("n%d_rmse" % n, mat(S, meth, "n%d_rmse" % n), "lower")
