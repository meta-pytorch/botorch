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
REGIMES = ["lcb_bo", "lcb_reg", "pd1_bo_armA", "pd1_bo_armC", "pd1_reg"]
SIGMAS = ["1e-3", "1e-2", "1e-1", "0.3", "1", "3", "10"]
SVAL = {
    "1e-3": 1e-3,
    "1e-2": 1e-2,
    "1e-1": 1e-1,
    "0.3": 0.3,
    "1": 1.0,
    "3": 3.0,
    "10": 10.0,
}
BASE = "1e-2"
NP_ = 9
TCRIT = 2.306

BO_METRICS = [
    ("budget_to_0.01", "lower"),
    ("budget_to_0.001", "lower"),
    ("final_regret_mean", "lower"),
    ("solve_rate_0.01", "higher"),
]
REG_METRICS = [
    ("n5_rank_corr", "higher"),
    ("n5_rmse", "lower"),
    ("n20_rank_corr", "higher"),
    ("n20_rmse", "lower"),
    ("n50_rank_corr", "higher"),
    ("n50_rmse", "lower"),
    ("n5_nll", "lower"),
    ("n5_calib_ratio", "n/a"),
    ("n5_coverage95", "n/a"),
    ("n5_rmse_obs_proxy", "lower"),
]


def load(reg):
    with open(os.path.join(ROOT, "stage2noise_" + reg, "SUMMARY.json")) as f:
        return json.load(f)


def cellmat(S, method, metric):
    """returns dict sigma -> list over 9 priors (None if absent)"""
    out = {}
    for s in SIGMAS:
        row = []
        for p in range(NP_):
            k = "emnoise_%s_p%d" % (s, p)
            c = S["cells"].get(k)
            v = None
            if c is not None:
                m = c["methods"].get(method)
                if m is not None:
                    v = m.get(metric)
            row.append(v)
        out[s] = row
    return out


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    return st.stdev(xs)


def paired(mat, s):
    d = []
    for p in range(NP_):
        a, b = mat[s][p], mat[BASE][p]
        if a is None or b is None:
            return None
        d.append(a - b)
    m = mean(d)
    s_d = sd(d)
    se = s_d / math.sqrt(len(d))
    t = m / se if se > 0 else (0.0 if abs(m) < 1e-15 else float("inf"))
    nwin = sum(1 for x in d if x != 0)
    return dict(mean=m, se=se, t=t, d=d, nnonzero=nwin)


def fmt(x, n=4):
    if x is None:
        return "  None "
    if isinstance(x, float) and (math.isnan(x)):
        return "  nan  "
    return ("%%.%df" % n) % x


print("=" * 100)
print("SIGMA^2 GRID:", SIGMAS, " baseline =", BASE, " priors P=9, df=8, t_crit=2.306")
print("NOTE: sigma^2=0.6 IS NOT ON THIS GRID. Nearest points are 0.3 and 1.0.")
print("=" * 100)

summary_lines = []
degen = []

for reg in REGIMES:
    S = load(reg)
    kind = S["cells"]["emnoise_1e-2_p0"]["kind"]
    methods = list(S["cells"]["emnoise_1e-2_p0"]["methods"].keys())
    ntasks = S["cells"]["emnoise_1e-2_p0"].get("n_tasks")
    print("\n" + "#" * 100)
    print(
        "### REGIME %s   kind=%s  cells %d/%d  n_tasks=%s  methods=%s"
        % (reg, kind, S["cells_present"], S["cells_expected"], ntasks, methods)
    )
    print("#" * 100)

    metrics = BO_METRICS if kind == "bo" else REG_METRICS
    metrics = [(m, d) for m, d in metrics if not m.endswith("_obs_proxy")]

    for method in methods:
        print("\n--- method: %s" % method)
        for metric, direction in metrics:
            mat = cellmat(S, method, metric)
            if any(v is None for s in SIGMAS for v in mat[s]):
                print("   %-20s  MISSING VALUES" % metric)
                continue
            means = {s: mean(mat[s]) for s in SIGMAS}
            # between-prior SE at baseline
            se_base = sd(mat[BASE]) / math.sqrt(NP_)
            spread = max(means.values()) - min(means.values())
            if direction == "lower":
                best = min(SIGMAS, key=lambda s: means[s])
            elif direction == "higher":
                best = max(SIGMAS, key=lambda s: means[s])
            else:
                best = "-"
            print(
                "   %-20s (%s better)  best=%s   spread=%.5g  betweenPriorSE(base)=%.5g  spread/SE=%.2f"
                % (
                    metric,
                    direction,
                    best,
                    spread,
                    se_base,
                    (spread / se_base) if se_base > 0 else float("inf"),
                )
            )
            hdr = "      sigma2 :" + "".join("%10s" % s for s in SIGMAS)
            print(hdr)
            print("      mean   :" + "".join("%10.4f" % means[s] for s in SIGMAS))
            # paired contrasts
            trow, mrow, serow, nz = [], [], [], []
            for s in SIGMAS:
                if s == BASE:
                    trow.append("     --")
                    mrow.append("      0.0000")
                    serow.append("        --")
                    nz.append("   -")
                    continue
                r = paired(mat, s)
                trow.append(
                    "%10.2f" % r["t"] if math.isfinite(r["t"]) else "       inf"
                )
                mrow.append("%10.4f" % r["mean"])
                serow.append("%10.4f" % r["se"])
                nz.append("%10d" % r["nnonzero"])
            print(
                "      d_mean :"
                + "".join(m if len(m) == 10 else "%10s" % m.strip() for m in mrow)
            )
            print("      d_se   :" + "".join("%10s" % x.strip() for x in serow))
            print("      t(8)   :" + "".join("%10s" % x.strip() for x in trow))
            print("      n!=0/9 :" + "".join("%10s" % x.strip() for x in nz))

            # degeneracy scan: are all sigma columns identical?
            sig_of = {}
            allsame = True
            for s in SIGMAS[1:]:
                if any(abs(mat[s][p] - mat[SIGMAS[0]][p]) > 1e-12 for p in range(NP_)):
                    allsame = False
            if allsame:
                degen.append(
                    "%s / %s / %s : IDENTICAL across all 7 sigma^2 values"
                    % (reg, method, metric)
                )
            # constant across priors?
            for s in SIGMAS:
                if len(set(round(v, 12) for v in mat[s])) == 1:
                    degen.append(
                        "%s / %s / %s @ sigma2=%s : identical across all 9 priors (val=%.6g)"
                        % (reg, method, metric, s, mat[s][0])
                    )

print("\n\n" + "=" * 100)
print("DEGENERACY / NO-OP SCAN")
print("=" * 100)
for line in degen:
    print(" *", line)
if not degen:
    print(" (none)")
