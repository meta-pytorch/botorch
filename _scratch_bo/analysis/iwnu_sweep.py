import glob
import json
import math
import re
import statistics as st
import sys

sys.path.insert(0, ".")
from summarize_stage import summarise_bo

CELLS = ["iwnu_manual", "iwnu_oas", "iwnu_ledoit", "iwnu_whiten005", "iwnu_whiten020"]
ARMS = ["pd1_bo_armA", "pd1_bo_armC"]
T8 = 2.306  # t(8), .95


def prior_of(p):
    m = re.search(r"_p(\d+)_s\d+\.json$", p)
    return int(m.group(1)) if m else None


for arm in ARMS:
    root = f"results/raw/v2/stage2iwnu_{arm}"
    print(f"\n{'=' * 78}\n{arm}\n{'=' * 78}")
    # provenance + alpha audit
    allf = [
        f
        for f in glob.glob(f"{root}/*.json")
        if "SUMMARY" not in f and "PROVENANCE" not in f
    ]
    commits, unstamped, dirty = set(), 0, 0
    alpha = {c: [] for c in CELLS}
    nu = {c: [] for c in CELLS}
    for f in allf:
        d = json.load(open(f))
        pv = d.get("provenance")
        if not pv or not pv.get("code_commit"):
            unstamped += 1
        else:
            commits.add(pv["code_commit"][:12])
            if pv.get("code_dirty"):
                dirty += 1
        cfg = d.get("config", {})
        cell = next((c for c in CELLS if f"/{c}_p" in f), None)
        if cell and cfg.get("resolved_iw_alpha") is not None:
            alpha[cell].append(cfg["resolved_iw_alpha"])
            nu[cell].append(cfg["resolved_iw_nu"])
    print(
        f"shards={len(allf)}  unstamped={unstamped}  commits={sorted(commits)}  dirty_shards={dirty}"
    )
    print(
        f"\n  {'cell':<16}{'n_est':>6}{'alpha mean':>12}{'alpha sd':>10}{'nu mean':>10}"
    )
    for c in CELLS:
        if alpha[c]:
            a = alpha[c]
            sd = st.stdev(a) if len(a) > 1 else 0.0
            print(
                f"  {c:<16}{len(a):>6}{st.mean(a):>12.4f}{sd:>10.4f}{st.mean(nu[c]):>10.1f}"
            )
        else:
            print(
                f"  {c:<16}{0:>6}{'--':>12}{'--':>10}{'--':>10}  (manual: nu not estimated)"
            )

    # per-prior budget-to-target, per cell, per method
    methods = set()
    per = {c: {} for c in CELLS}
    for c in CELLS:
        for p in range(9):
            fs = sorted(glob.glob(f"{root}/{c}_p{p}_s*.json"))
            if not fs:
                continue
            s = summarise_bo(fs)["methods"]
            per[c][p] = s
            methods |= set(s)
    methods = sorted(methods)

    for tol in ("budget_to_0.01", "budget_to_0.001"):
        print(f"\n  --- {tol} : paired vs iwnu_manual, over 9 priors, t(8)={T8} ---")
        for m in methods:
            print(f"\n  method {m}")
            base = {
                p: per["iwnu_manual"][p][m][tol]
                for p in per["iwnu_manual"]
                if m in per["iwnu_manual"][p]
            }
            bm = st.mean(base.values()) if base else float("nan")
            bsd = st.stdev(base.values()) if len(base) > 1 else 0.0
            print(
                f"    {'iwnu_manual':<16}{bm:>9.2f}  between-prior sd {bsd:>6.2f}  (control)"
            )
            for c in CELLS[1:]:
                d = [
                    per[c][p][m][tol] - base[p]
                    for p in sorted(base)
                    if p in per[c] and m in per[c][p]
                ]
                if len(d) < 2:
                    continue
                md, sd = st.mean(d), st.stdev(d)
                sem = sd / math.sqrt(len(d))
                hw = T8 * sem
                sig = "*" if abs(md) > hw else " "
                print(
                    f"    {c:<16}{st.mean([per[c][p][m][tol] for p in sorted(base) if p in per[c]]):>9.2f}"
                    f"  delta {md:+7.2f} +/- {hw:5.2f} (n={len(d)}) {sig}"
                )
