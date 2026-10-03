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
BASE = "mean_const_emp"
# factor inventory from gen_stage_queue.py MEANS
FACTORS = {
    # config: (base_mean, mean_transfer, canonical, mean_prior)
    "mean_const_emp": ("constant", "empirical", "sumll", False),
    "mean_lin_emp": ("linear", "empirical", "sumll", False),
    "mean_const_hb": ("constant", "hyperbo", "hyperbo", False),
    "mean_lin_hb": ("linear", "hyperbo", "hyperbo", False),
    "mean_const_b25": ("constant", "blend0.25", "hyperbo", False),
    "mean_const_b50": ("constant", "blend0.50", "hyperbo", False),
    "mean_const_b75": ("constant", "blend0.75", "hyperbo", False),
    "mean_lin_b50": ("linear", "blend0.50", "hyperbo", False),
    "mean_const_prior": ("constant", "empirical", "sumll", True),
    "mean_lin_prior": ("linear", "empirical", "sumll", True),
}
CONFIGS = list(FACTORS)
CELLRE = re.compile(r"^(.*)_p(\d+)$")
STAGES = {
    "lcb_bo": ("stage2means_lcb_bo", "budget_to_0.01", "lower"),
    "pd1_armA": ("stage2means_pd1_bo_armA", "budget_to_0.01", "lower"),
    "pd1_armC": ("stage2means_pd1_bo_armC", "budget_to_0.01", "lower"),
    "lcb_reg": ("stage2means_lcb_reg", "n5_rank_corr", "higher"),
}


def load(s):
    with open(os.path.join(ROOT, s, "SUMMARY.json")) as f:
        return json.load(f)


def per_prior(d, metric, method):
    out = defaultdict(dict)
    for k, c in d["cells"].items():
        m = CELLRE.match(k)
        md = c["methods"].get(method)
        if md and metric in md:
            out[m.group(1)][int(m.group(2))] = md[metric]
    return out


def mean_se(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    v = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(v / n)


def paired(a, b, pr):
    d = [a[p] - b[p] for p in pr]
    mu, se = mean_se(d)
    t = (mu / se) if se > 0 else (0.0 if mu == 0 else float("inf"))
    return mu, se, t


print("=" * 80)
print(
    "A. BIT-IDENTITY MATRIX (em_frozen) -- which config PAIRS are literally the same run?"
)
print("=" * 80)
for name, (st, metric, _) in STAGES.items():
    d = load(st)
    pp = per_prior(d, metric, "em_frozen")
    pr = sorted(set.intersection(*[set(pp[c]) for c in CONFIGS]))
    dup = []
    for i in range(len(CONFIGS)):
        for j in range(i + 1, len(CONFIGS)):
            a, b = CONFIGS[i], CONFIGS[j]
            if all(pp[a][p] == pp[b][p] for p in pr):
                dup.append((a, b))
    print(
        f"  {name:10s}  identical pairs over all {len(pr)} priors: "
        f"{dup if dup else 'none'}"
    )

print()
print("=" * 80)
print("B. CONTROL CHECK: hyperbo_frozen must be invariant to every mean flag")
print("=" * 80)
for name, (st, metric, _) in STAGES.items():
    d = load(st)
    pp = per_prior(d, metric, "hyperbo_frozen")
    pr = sorted(set.intersection(*[set(pp[c]) for c in CONFIGS]))
    bad = []
    for c in CONFIGS:
        if c == BASE:
            continue
        diffs = [(p, pp[c][p] - pp[BASE][p]) for p in pr if pp[c][p] != pp[BASE][p]]
        if diffs:
            mu, se, t = paired(pp[c], pp[BASE], pr)
            bad.append((c, len(diffs), mu, t))
    if not bad:
        print(f"  {name:10s}  CLEAN -- all 9 non-base configs bit-identical to base")
    else:
        print(f"  {name:10s}  *** CONTROL DRIFT on {len(bad)} config(s):")
        for c, nd, mu, t in bad:
            print(
                f"       {c:18s} differs on {nd}/{len(pr)} priors  mean Δ={mu:+.3f} t={t:+.2f}"
            )

print()
print("=" * 80)
print(
    "C. FACTOR DECOMPOSITION (em_frozen) -- is the effect the MEAN or the CANONICAL KERNEL?"
)
print("=" * 80)
HB = [c for c in CONFIGS if FACTORS[c][2] == "hyperbo"]  # 6 configs
SUM = [c for c in CONFIGS if FACTORS[c][2] == "sumll"]  # 4 configs
print(f"  canonical=hyperbo group ({len(HB)}): {HB}")
print(f"  canonical=sumll  group ({len(SUM)}): {SUM}")
for name, (st, metric, direction) in STAGES.items():
    d = load(st)
    pp = per_prior(d, metric, "em_frozen")
    pr = sorted(set.intersection(*[set(pp[c]) for c in CONFIGS]))
    ghb = [sum(pp[c][p] for c in HB) / len(HB) for p in pr]
    gsu = [sum(pp[c][p] for c in SUM) / len(SUM) for p in pr]
    dd = [a - b for a, b in zip(ghb, gsu)]
    mu, se = mean_se(dd)
    t = mu / se if se else float("nan")
    mhb, shb = mean_se(ghb)
    msu, ssu = mean_se(gsu)

    # within-group spread of config means
    def spread(g):
        ms = [mean_se([pp[c][p] for p in pr])[0] for c in g]
        return max(ms) - min(ms)

    print(f"\n  -- {name} ({metric}, {direction} better)")
    print(
        f"     canonical=hyperbo group mean {mhb:.4f} ± {shb:.4f}   "
        f"(within-group spread {spread(HB):.4f})"
    )
    print(
        f"     canonical=sumll   group mean {msu:.4f} ± {ssu:.4f}   "
        f"(within-group spread {spread(SUM):.4f})"
    )
    print(
        f"     BETWEEN-GROUP paired Δ (hb − sumll) = {mu:+.4f} ± {se:.4f}  "
        f"t(8) = {t:+.2f}  {'SIG' if abs(t) > 2.31 else 'ns'}"
    )

print()
print("=" * 80)
print("D. PURE MEAN CONTRASTS, holding canonical fixed (em_frozen)")
print("=" * 80)
CONTRASTS = [
    (
        "base mean linear vs constant | canonical=sumll, emp",
        "mean_lin_emp",
        "mean_const_emp",
    ),
    (
        "base mean linear vs constant | canonical=hyperbo, hb",
        "mean_lin_hb",
        "mean_const_hb",
    ),
    (
        "base mean linear vs constant | canonical=hyperbo, b50",
        "mean_lin_b50",
        "mean_const_b50",
    ),
    (
        "base mean linear vs constant | mean-prior on",
        "mean_lin_prior",
        "mean_const_prior",
    ),
    ("mean-prior on vs off | constant, sumll", "mean_const_prior", "mean_const_emp"),
    ("mean-prior on vs off | linear, sumll", "mean_lin_prior", "mean_lin_emp"),
    (
        "transfer hyperbo vs blend0.50 | const, canon=hb",
        "mean_const_hb",
        "mean_const_b50",
    ),
    ("blend 0.25 vs 0.50 | const, canon=hb", "mean_const_b25", "mean_const_b50"),
    ("blend 0.75 vs 0.50 | const, canon=hb", "mean_const_b75", "mean_const_b50"),
    ("blend 0.75 vs 0.25 | const, canon=hb", "mean_const_b75", "mean_const_b25"),
]
hdr = f"  {'contrast':52s}" + "".join(f"{n:>19s}" for n in STAGES)
print(hdr)
for label, a, b in CONTRASTS:
    cells = []
    for name, (st, metric, _) in STAGES.items():
        d = load(st)
        pp = per_prior(d, metric, "em_frozen")
        pr = sorted(set(pp[a]) & set(pp[b]))
        mu, se, t = paired(pp[a], pp[b], pr)
        star = "*" if abs(t) > 2.31 else " "
        cells.append(f"{mu:+8.3f} t={t:+6.2f}{star}")
    print(f"  {label:52s}" + "".join(f"{c:>19s}" for c in cells))

print()
print("=" * 80)
print(
    "E. ON-GRID vs OFF-GRID: canonical-kernel effect size by regime geometry (em_frozen)"
)
print("=" * 80)
print("   regime      geometry   Δ(hb−sumll)   ±SE    t(8)   as % of base mean")
for name, (st, metric, _) in STAGES.items():
    geo = "off-grid" if name == "pd1_armC" else "on-grid"
    d = load(st)
    pp = per_prior(d, metric, "em_frozen")
    pr = sorted(set.intersection(*[set(pp[c]) for c in CONFIGS]))
    ghb = [sum(pp[c][p] for c in HB) / len(HB) for p in pr]
    gsu = [sum(pp[c][p] for c in SUM) / len(SUM) for p in pr]
    mu, se = mean_se([a - b for a, b in zip(ghb, gsu)])
    t = mu / se if se else float("nan")
    bm = mean_se([pp[BASE][p] for p in pr])[0]
    print(
        f"   {name:11s} {geo:9s} {mu:+11.4f} {se:7.4f} {t:+7.2f}   {100 * mu / bm:+7.2f}%"
    )
