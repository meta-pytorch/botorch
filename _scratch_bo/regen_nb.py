#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Full end-to-end (non-smoke) execution of the empirical-GP tutorial.

Runs the notebook with the internal LCBench loader (mirrors the CI patch) in FULL
mode, with fail_on_error=True (so any cell error aborts). `run_notebook` returns the
kernel's user namespace, from which we print the key result variables (Section E
multi-task RMSE/NLL, Section F 7D RMSE/NLL, Section G BO regret) so the additive
results can be compared against the previous status quo.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent / "tutorials")
NB = f"{ROOT}/empirical_gaussian_processes/empirical_gaussian_processes.ipynb"


def _mean(x) -> float:
    import torch

    t = torch.as_tensor(x, dtype=torch.double).reshape(-1)
    return float(t.mean()) if t.numel() else float("nan")


def main() -> None:
    os.environ.pop("SMOKE_TEST", None)  # FULL mode
    os.environ["DATA_LOCATION"] = f"{ROOT}/data"
    from _scratch_bo.lcbench_io import use_internal_store_reader
    from bento.testutil import run_notebook

    # The notebook loads LCBench through botorch's OSS loader; route its Parquet
    # reads at the internal object storage mirror (no external network on devservers/CI).
    use_internal_store_reader()
    ns = run_notebook(path=NB, fail_on_error=True)

    print("\n######## FULL-MODE NOTEBOOK RESULTS (fail_on_error passed) ########")

    # ---- Section E: multi-task RMSE / NLL ----
    try:
        methods, n_targets = ns["methods"], ns["n_targets"]
        for label, tbl in [("RMSE", ns["RM"]), ("NLL", ns["NLd"])]:
            print(f"\n[Section E multi-task] {label} (lower=better):")
            print(f"  {'method':<44}" + "".join(f"@{n:<8}" for n in n_targets))
            for m in methods:
                print(
                    f"  {m:<44}"
                    + "".join(f"{_mean(tbl[m][n]):<9.2f}" for n in n_targets)
                )
        print("\n[Section E] MLL-fit additive base outputscale by budget:")
        print("  " + "".join(f"@{n}={_mean(ns['os_fit'][n]):.2f}  " for n in n_targets))
    except Exception as e:
        print(f"[Section E extraction error] {e!r}")

    # ---- Section F: 7D EM transfer RMSE / NLL ----
    try:
        sizes = ns["n_train_sizes_7d"]
        for label, tbl in [("RMSE(std)", ns["rmse_7d"]), ("NLL", ns["nll_7d"])]:
            print(f"\n[Section F 7D] {label} (lower=better):")
            print(f"  {'model':<40}" + "".join(f"n={n:<7}" for n in sizes))
            for m in tbl:
                print(
                    f"  {m:<40}" + "".join(f"{_mean(tbl[m][n]):<9.3f}" for n in sizes)
                )
    except Exception as e:
        print(f"[Section F extraction error] {e!r}")

    # ---- Section G: BO simple regret ----
    try:
        mean_regret, methods_bo, cps = ns["mean_regret_bo"], ns["methods_bo"], ns["cps"]
        print("\n[Section G BO] mean simple regret (lower=better):")
        print(f"  {'method':<26}" + "".join(f"@{c:<6}" for c in cps))
        for m in methods_bo:
            r = mean_regret[m]
            print(f"  {m:<26}" + "".join(f"{float(r[c]):<7.2f}" for c in cps))
    except Exception as e:
        print(f"[Section G extraction error] {e!r}")


if __name__ == "__main__":
    main()
