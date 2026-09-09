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


def shard(reg, s, p):
    return json.load(
        open(
            os.path.join(ROOT, "stage2noise_" + reg, "emnoise_%s_p%d_s0.json" % (s, p))
        )
    )


for reg, tcrit in [("lcb_reg", None), ("pd1_reg", None)]:
    print("\n" + "=" * 92)
    print(
        "PER-TASK (eval dataset) PAIRED ANALYSIS -- %s   [valid replication unit]" % reg
    )
    print("=" * 92)
    # verify priors identical for em_frozen at task level
    a = shard(reg, BASE, 0)
    b = shard(reg, BASE, 8)
    for n in ["5", "20", "50"]:
        ident = (
            a["by_n_obs"][n]["em_frozen"]["per_dataset_rmse"]
            == b["by_n_obs"][n]["em_frozen"]["per_dataset_rmse"]
        )
        print("  prior p0 vs p8 per-task RMSE identical @ n=%s : %s" % (n, ident))
    ds = a["eval_datasets"]
    T = len(ds)
    df = T - 1
    tc = {4: 2.776, 9: 2.262}[df]
    print("  n_eval_tasks = %d  -> df = %d, t_crit(.05,2-sided) = %.3f" % (T, df, tc))
    for meth in ["em_frozen", "em_additive_hyperbo"]:
        for n in ["5", "20", "50"]:
            per = {}
            for s in SIG:
                # average across the 9 priors first (no-op for em_frozen), keeping task identity
                cols = [
                    shard(reg, s, p)["by_n_obs"][n][meth]["per_dataset_rmse"]
                    for p in range(9)
                ]
                per[s] = [sum(c[i] for c in cols) / 9.0 for i in range(T)]
            means = {s: sum(per[s]) / T for s in SIG}
            best = min(SIG, key=lambda s: means[s])
            se_task = st.stdev(per[BASE]) / math.sqrt(T)
            print(
                "\n  [%s] n%s_rmse  (per-task pooling, lower better)  argmin=%s"
                % (meth, n, best)
            )
            print("    sigma2 " + "".join("%9s" % s for s in SIG))
            print("    mean   " + "".join("%9.4f" % means[s] for s in SIG))
            dm, dse, tt, nw = [], [], [], []
            for s in SIG:
                if s == BASE:
                    dm.append("---")
                    dse.append("---")
                    tt.append("---")
                    nw.append("---")
                    continue
                d = [per[s][i] - per[BASE][i] for i in range(T)]
                m = sum(d) / T
                se = st.stdev(d) / math.sqrt(T)
                dm.append("%.4f" % m)
                dse.append("%.4f" % se)
                tt.append("%.2f" % (m / se) if se > 0 else "inf")
                nw.append("%d/%d" % (sum(1 for x in d if x < 0), T))
            print("    d_mean " + "".join("%9s" % x for x in dm))
            print("    d_se   " + "".join("%9s" % x for x in dse))
            print("    t(%d)   " % df + "".join("%9s" % x for x in tt))
            print("    better " + "".join("%9s" % x for x in nw))
            print(
                "    betweenTaskSE@base=%.4f   spread=%.4f   spread/SE=%.2f"
                % (
                    se_task,
                    max(means.values()) - min(means.values()),
                    (max(means.values()) - min(means.values())) / se_task,
                )
            )

# Threshold-collapse check on lcb_bo
print("\n" + "=" * 92)
print("INSTRUMENT CHECK: does budget_to_0.01 differ from budget_to_0.001 ?")
print("=" * 92)
for reg in ["lcb_bo", "pd1_bo_armA", "pd1_bo_armC"]:
    S = json.load(open(os.path.join(ROOT, "stage2noise_" + reg, "SUMMARY.json")))
    same = tot = 0
    for s in SIG:
        for p in range(9):
            for m in S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"]:
                v = S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"][m]
                tot += 1
                if v["budget_to_0.01"] == v["budget_to_0.001"]:
                    same += 1
    print("  %-14s identical in %d/%d cell-methods" % (reg, same, tot))

# censoring / saturation check
print("\n" + "=" * 92)
print("CENSORING CHECK: budget-to-target vs budget cap")
print("=" * 92)
for reg in ["lcb_bo", "pd1_bo_armA", "pd1_bo_armC"]:
    a = json.load(
        open(os.path.join(ROOT, "stage2noise_" + reg, "emnoise_1e-2_p0_s0.json"))
    )
    tr = a["summary"]["em_frozen"]["mean_best"]
    print(
        "  %-14s BO budget (traj len) = %d   init_evals=%s"
        % (reg, len(tr), a.get("init_evals"))
    )
    S = json.load(open(os.path.join(ROOT, "stage2noise_" + reg, "SUMMARY.json")))
    mx = max(
        S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"][m]["budget_to_0.001"]
        for s in SIG
        for p in range(9)
        for m in S["cells"]["emnoise_%s_p0" % s]["methods"]
    )
    print("                 max observed budget_to_0.001 across all cells = %.2f" % mx)
