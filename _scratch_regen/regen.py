#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate committed cell outputs for the empirical GP tutorial (in-process),
or time each cell under SMOKE_TEST.

Usage:
  buck2 run :regen -- <nb_path>                 # full regen, write outputs
  buck2 run :regen -- <nb_path> --smoke --time  # SMOKE mode, time cells, no write
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from unittest import mock

from IPython.utils.capture import capture_output


def _outputs_from_capture(cap, result, shell, exec_count):
    outs = []
    if cap.stdout:
        outs.append({"output_type": "stream", "name": "stdout", "text": cap.stdout})
    if cap.stderr:
        outs.append({"output_type": "stream", "name": "stderr", "text": cap.stderr})
    for ro in cap.outputs:
        outs.append(
            {
                "output_type": "display_data",
                "data": dict(ro.data),
                "metadata": dict(ro.metadata or {}),
            }
        )
    if result is not None:
        data, md = shell.display_formatter.format(result)
        outs.append(
            {
                "output_type": "execute_result",
                "data": data,
                "metadata": md,
                "execution_count": exec_count,
            }
        )
    return outs


def main() -> None:
    args = sys.argv[1:]
    smoke = "--smoke" in args
    time_only = "--time" in args
    stop_after = next(
        (int(a.split("=")[1]) for a in args if a.startswith("--stop-after=")), None
    )
    nb_path = [a for a in args if not a.startswith("--")][0]

    if smoke:
        os.environ["SMOKE_TEST"] = "1"

    with open(nb_path) as f:
        nb = json.load(f)

    from ipykernel.inprocess.manager import InProcessKernelManager

    km = InProcessKernelManager()
    km.start_kernel()
    shell = km.kernel.shell
    shell.reset(new_session=True)
    shell.run_cell("%matplotlib inline")

    from ax.fb.benchmark.problems.surrogate.lcbench.data import (
        read_lcbench_parquet_from_manifold,
    )

    failed = False
    timings = []
    with mock.patch(
        "botorch.utils.lcbench._read_lcbench_parquet",
        read_lcbench_parquet_from_manifold,
    ):
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            t0 = time.perf_counter()
            with capture_output() as cap:
                res = shell.run_cell(source, store_history=True)
            dt = time.perf_counter() - t0
            timings.append((i, dt))
            ec = shell.execution_count - 1
            if res.error_in_exec is not None:
                print(f"[cell {i}] ERROR after {dt:.2f}s: {res.error_in_exec!r}")
                traceback.print_exception(
                    type(res.error_in_exec),
                    res.error_in_exec,
                    res.error_in_exec.__traceback__,
                )
                failed = True
                break
            if not time_only:
                cell["outputs"] = _outputs_from_capture(cap, res.result, shell, ec)
                cell["execution_count"] = ec
            print(f"[cell {i}] {dt:6.2f}s", flush=True)
            if stop_after is not None and i >= stop_after:
                break

    km.shutdown_kernel()
    if failed:
        sys.exit(2)

    total = sum(t for _, t in timings)
    print(
        f"\n=== {'SMOKE' if smoke else 'FULL'} total: {total:.1f}s "
        f"over {len(timings)} code cells ==="
    )
    print("slowest cells:")
    for i, dt in sorted(timings, key=lambda x: -x[1])[:8]:
        print(f"  cell {i}: {dt:.2f}s")

    if not time_only:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote outputs to {nb_path}")


if __name__ == "__main__":
    main()
