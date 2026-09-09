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


def M(reg, meth, met):
    S = json.load(open(os.path.join(ROOT, "stage2noise_" + reg, "SUMMARY.json")))
    return {
        s: [
            S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"][meth][met]
            for p in range(9)
        ]
        for s in SIG
    }


def tt(d):
    m = sum(d) / len(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return m, se, (m / se if se else float("nan"))


print("=" * 92)
print(
    "ARM A x SIGMA^2 INTERACTION  (paired by prior; same 23 tasks, only pool flags differ)"
)
print(
    "contrast = [armC(s) - armC(1e-2)] - [armA(s) - armA(1e-2)]   negative => sigma^2 helps MORE off-grid"
)
print("=" * 92)
for meth in ["em_frozen", "em_additive_hyperbo"]:
    A = M("pd1_bo_armA", meth, "budget_to_0.01")
    C = M("pd1_bo_armC", meth, "budget_to_0.01")
    print("\n [%s] budget_to_0.01" % meth)
    print("   sigma2 " + "".join("%10s" % s for s in SIG))
    row_m, row_t = [], []
    for s in SIG:
        if s == BASE:
            row_m.append("---")
            row_t.append("---")
            continue
        d = [(C[s][p] - C[BASE][p]) - (A[s][p] - A[BASE][p]) for p in range(9)]
        m, se, t = tt(d)
        row_m.append("%.3f" % m)
        row_t.append("%.2f" % t)
    print("   d_int  " + "".join("%10s" % x for x in row_m))
    print("   t(8)   " + "".join("%10s" % x for x in row_t))

print("\n" + "=" * 92)
print(
    "CENSORING FRACTION (runs never reaching the target -> budget_to_target pinned at cap)"
)
print("=" * 92)
CAP = {"lcb_bo": 41, "pd1_bo_armA": 51, "pd1_bo_armC": 51}
for reg in CAP:
    S = json.load(open(os.path.join(ROOT, "stage2noise_" + reg, "SUMMARY.json")))
    print("  %-13s cap=%d" % (reg, CAP[reg]))
    for s in SIG:
        sr = (
            sum(
                S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"]["em_frozen"][
                    "solve_rate_0.01"
                ]
                for p in range(9)
            )
            / 9
        )
        bt = (
            sum(
                S["cells"]["emnoise_%s_p%d" % (s, p)]["methods"]["em_frozen"][
                    "budget_to_0.01"
                ]
                for p in range(9)
            )
            / 9
        )
        uncens = (bt - (1 - sr) * CAP[reg]) / sr if sr > 0 else float("nan")
        print(
            "     sigma2=%-5s solve_rate=%.3f  censored=%.1f%%  budget=%.2f  implied E[t|solved]=%.2f"
            % (s, sr, 100 * (1 - sr), bt, uncens)
        )

print("\n" + "=" * 92)
print(
    "REGRESSION: is the 9-prior SE exactly zero for em_frozen?  (pre-registered pooling validity)"
)
print("=" * 92)
for reg in ["lcb_reg", "pd1_reg"]:
    for meth in [
        "em_frozen",
        "em_additive_hyperbo",
        "em_additive_hyperbo_frozen",
        "hyperbo_frozen",
    ]:
        m = M(reg, meth, "n5_rmse")
        sds = [st.stdev(m[s]) for s in SIG]
        print(
            "  %-9s %-28s max sd over 9 priors (n5_rmse) = %.3e" % (reg, meth, max(sds))
        )
