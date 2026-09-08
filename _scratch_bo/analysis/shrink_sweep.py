import glob
import json
import math
import statistics as st
import sys

sys.path.insert(0, ".")
from summarize_stage import summarise_bo

CELLS = [
    "shrink000",
    "shrink005",
    "shrink010",
    "shrink020",
    "shrink030",
    "shrink_oas",
    "shrink_whiten",
]
T8 = 2.306
arm = sys.argv[1] if len(sys.argv) > 1 else "pd1_bo_armA"
root = f"results/raw/v2/stage2shrink_{arm}"
allf = [
    f
    for f in glob.glob(f"{root}/*.json")
    if "SUMMARY" not in f and "PROVENANCE" not in f
]
uns = 0
commits = set()
res = {c: [] for c in CELLS}
for f in allf:
    d = json.load(open(f))
    pv = d.get("provenance") or {}
    if not pv.get("code_commit"):
        uns += 1
    else:
        commits.add(pv["code_commit"][:12])
    c = next((c for c in CELLS if f"/{c}_p" in f), None)
    v = d.get("config", {}).get("resolved_em_shrinkage")
    if c and v is not None:
        res[c].append(v)
print(f"{arm}: {len(allf)} shards, unstamped={uns}, commits={sorted(commits)}")
print("  estimated alpha (resolved_em_shrinkage):")
for c in CELLS:
    if res[c]:
        print(
            f"    {c:15s} n={len(res[c]):4d}  mean={st.mean(res[c]):.4f}  sd={st.stdev(res[c]) if len(res[c]) > 1 else 0:.4f}"
            f"  min={min(res[c]):.3f} max={max(res[c]):.3f}"
        )
per = {}
for c in CELLS:
    per[c] = {}
    for p in range(9):
        fs = sorted(glob.glob(f"{root}/{c}_p{p}_s*.json"))
        if fs:
            per[c][p] = summarise_bo(fs)["methods"]
methods = sorted(set(m for p in per["shrink000"].values() for m in p))
for tol in ("budget_to_0.01", "budget_to_0.001"):
    print(f"\n  === {tol} : paired by prior vs shrink000, t(8)={T8} ===")
    for m in methods:
        base = {
            p: per["shrink000"][p][m][tol]
            for p in per["shrink000"]
            if m in per["shrink000"][p]
        }
        print(f"\n  {m}")
        print(
            f"    {'shrink000':<15}{st.mean(base.values()):>8.2f}   between-prior sd {st.stdev(base.values()):>5.2f}  (control)"
        )
        for c in CELLS[1:]:
            d = [
                per[c][p][m][tol] - base[p]
                for p in sorted(base)
                if p in per[c] and m in per[c][p]
            ]
            if len(d) < 2:
                continue
            md = st.mean(d)
            hw = T8 * st.stdev(d) / math.sqrt(len(d))
            mean_c = st.mean([per[c][p][m][tol] for p in sorted(base) if p in per[c]])
            print(
                f"    {c:<15}{mean_c:>8.2f}   delta {md:+7.3f} +/- {hw:5.3f}  {'SIG' if abs(md) > hw else ''}"
            )
