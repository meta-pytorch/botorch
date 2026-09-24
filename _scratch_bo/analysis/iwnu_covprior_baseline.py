import glob
import math
import statistics as st
import sys

sys.path.insert(0, ".")
from summarize_stage import summarise_bo

T8 = 2.306
for arm in ("pd1_bo_armA", "pd1_bo_armC"):
    print(
        f"\n=== {arm} : covar-prior ON (iwnu_manual) vs OFF (stage1 base), NON-PAIRED ==="
    )
    for tol in ("budget_to_0.01", "budget_to_0.001"):
        rows = {}
        for tag, pat in (
            ("OFF stage1 base", f"results/raw/v2/stage1_{arm}/base_p{{}}_s*.json"),
            (
                "ON  iwnu_manual",
                f"results/raw/v2/stage2iwnu_{arm}/iwnu_manual_p{{}}_s*.json",
            ),
        ):
            per = {}
            for p in range(9):
                fs = sorted(glob.glob(pat.format(p)))
                if fs:
                    per[p] = summarise_bo(fs)["methods"]
            rows[tag] = per
        methods = sorted(
            set.intersection(
                *[set(m for p in v.values() for m in p) for v in rows.values()]
            )
        )
        print(f"  -- {tol}")
        for m in methods:
            out = []
            for tag, per in rows.items():
                vals = [per[p][m][tol] for p in sorted(per) if m in per[p]]
                if len(vals) < 2:
                    continue
                out.append((tag, st.mean(vals), st.stdev(vals), len(vals)))
            if len(out) == 2:
                (t1, m1, s1, n1), (t2, m2, s2, n2) = out
                se = math.sqrt(s1**2 / n1 + s2**2 / n2)
                d = m2 - m1
                print(
                    f"     {m:30s} OFF {m1:6.2f}(sd {s1:4.2f})  ON {m2:6.2f}(sd {s2:4.2f})  "
                    f"delta {d:+6.2f} +/- {T8 * se:5.2f}{'  *' if abs(d) > T8 * se else ''}"
                )
