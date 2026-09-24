# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Merge-validation for the arm-C hybrid sweep.

The hybrid sweep (`run_hybrid_armc.sh`) was run with a REDUCED method list so its
results could be paired with the existing arm-C data instead of re-running the shared
arms. That merge is only legitimate if the shared arms come out bit-identical.

Two changes were made to the method list relative to `fullboth*.json`:

1. `em_additive_hyperbo*` was ADDED. Verified safe before launching -- `em_frozen` was
   bit-identical across three separate processes.
2. `pacoh_frozen` and `ablr` were DROPPED. **Not** covered by that check. Their
   pre-training consumes global RNG before the BO loop, so in principle it can shift
   anything downstream that reads the global generator.

The BO loop itself seeds explicit local generators from ``(held, seed)``
(``bo_experiment.py``, ``gm = torch.Generator().manual_seed(...)``), so trajectories
should be immune -- but "should be" is exactly what this project keeps getting wrong, so
this checks rather than assumes.

Exit status is non-zero if the merge is invalid, so it can gate the factorial launch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)
POOL = os.path.join(ROOT, "results/raw/pd1pool")

# The two arms carried purely as merge controls. Both are frozen variants, so they are
# the least likely to read global RNG during BO -- which is what makes them a clean test
# of the pre-training RNG question rather than a test of fitting noise.
CONTROLS = ("em_frozen", "hyperbo_frozen")


def existing(path: str) -> str | None:
    return path if os.path.exists(path) else None


def main() -> None:
    failures: list[str] = []
    checked = 0
    missing = 0

    for prior in (0, 1, 2):
        suffix = "" if prior == 0 else f"_p{prior}"
        for shard in range(12):
            new = existing(os.path.join(POOL, f"hybrid_p{prior}_s{shard}.json"))
            old = existing(os.path.join(POOL, f"fullboth{suffix}_s{shard}.json"))
            if new is None or old is None:
                missing += 1
                continue
            dn, do = json.load(open(new)), json.load(open(old))
            pn, po = dn["per_run"], do["per_run"]

            tag = f"prior {prior} shard {shard}"
            if pn["eval_dataset_idx"] != po["eval_dataset_idx"]:
                failures.append(f"{tag}: eval_dataset_idx differs -- runs do not align")
                continue
            if pn["pool_max_raw"] != po["pool_max_raw"]:
                failures.append(f"{tag}: pool_max_raw differs -- different candidates")
                continue

            for m in CONTROLS:
                if m not in pn["traj_raw"] or m not in po["traj_raw"]:
                    failures.append(f"{tag}: {m} absent from one side")
                    continue
                a, b = pn["traj_raw"][m], po["traj_raw"][m]
                checked += 1
                if a != b:
                    where = next(
                        (
                            f"run {r} step {s}"
                            for r, (ra, rb) in enumerate(zip(a, b))
                            for s, (x, y) in enumerate(zip(ra, rb))
                            if x != y
                        ),
                        "unknown",
                    )
                    failures.append(f"{tag}: {m} DIFFERS (first at {where})")

    print("=" * 72)
    print("MERGE VALIDATION: hybrid_p*.json vs fullboth*.json")
    print("=" * 72)
    print(f"control comparisons run : {checked}")
    print(f"shard pairs unavailable : {missing}")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures[:20]:
            print("  " + f)
        print(
            "\nThe shared arms are NOT reproducible, so the two sets of runs did not "
            "come from equivalent code. Do NOT merge them; compare only within a "
            "single sweep.\n"
            "Diagnose before assuming a cause (S27.10 did, and the obvious suspect "
            "was wrong):\n"
            "  1. Same binary, with vs without the dropped methods -- isolates the "
            "--methods list.\n"
            "  2. Re-run a saved config on the current binary -- isolates code drift.\n"
            "In S27.10 the method list was innocent and code drift was the cause."
        )
        sys.exit(1)

    if checked == 0:
        print("\nINCONCLUSIVE: no comparable shard pairs found.")
        sys.exit(1)

    print(
        "\nPASSED: every shared arm is bit-identical. Dropping PACOH/ABLR did not "
        "perturb them, so the hybrid results pair with the existing arm-C data by "
        "(task, seed, prior)."
    )


if __name__ == "__main__":
    main()
