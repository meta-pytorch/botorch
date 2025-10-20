#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from subprocess import CalledProcessError
from typing import Dict, Optional, Tuple

from memory_profiler import memory_usage


IGNORE_ALWAYS = set()  # ignored in smoke tests and full runs
RUN_IF_SMOKE_TEST_IGNORE_IF_STANDARD = {"robot.ipynb"}  # only used in smoke tests


def run_script(
    tutorial: Path, timeout_minutes: int, env: Optional[Dict[str, str]] = None
) -> None:
    if env is not None:
        env = {**os.environ, **env}
    run_out = subprocess.run(
        ["papermill", tutorial, "|"],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_minutes * 60,
    )
    return run_out


def run_tutorial(
    tutorial: Path, smoke_test: bool = False
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Runs the tutorial in a subprocess, catches any raised errors and returns
    them as a string, and returns runtime and memory information as a dict.
    """
    timeout_minutes = 5 if smoke_test else 30
    tic = time.monotonic()
    print(f"Running tutorial {tutorial.name}.")
    env = {"SMOKE_TEST": "True"} if smoke_test else None
    try:
        mem_usage, run_out = memory_usage(
            (run_script, (tutorial, timeout_minutes), {"env": env}),
            retval=True,
            include_children=True,
        )
    except subprocess.TimeoutExpired:
        error = (
            f"Tutorial {tutorial.name} exceeded the maximum runtime of "
            f"{timeout_minutes} minutes."
        )
        return error, {}

    try:
        run_out.check_returncode()
    except CalledProcessError:
        error = "\n".join(
            [
                f"Encountered error running tutorial {tutorial.name}:",
                "stdout:",
                run_out.stdout,
                "stderr:",
                run_out.stderr,
            ]
        )
        return error, {}
    runtime = time.monotonic() - tic
    performance_info = {
        "runtime": runtime,
        "start_mem": mem_usage[0],
        "max_mem": max(mem_usage),
    }

    return None, performance_info


def run_tutorials(
    repo_dir: str,
    include_ignored: bool = False,
    smoke_test: bool = False,
    name: Optional[str] = None,
) -> None:
    """Run each tutorial and print statements on its runtime and memory usage."""
    mode = "smoke test" if smoke_test else "standard"
    print(f"Running tutorial(s) in {mode} mode.")
    if not smoke_test:
        print("This may take a long time...")
    tutorial_dir = Path(repo_dir).joinpath("tutorials")
    num_runs = 0
    num_errors = 0
    ignored_tutorials = (
        IGNORE_ALWAYS
        if smoke_test
        else IGNORE_ALWAYS | RUN_IF_SMOKE_TEST_IGNORE_IF_STANDARD
    )

    tutorials = sorted(t for t in tutorial_dir.rglob("*.ipynb") if t.is_file())
    if name is not None:
        tutorials = [t for t in tutorials if t.name == name]
        if len(tutorials) == 0:
            raise RuntimeError(f"Specified tutorial {name} not found in directory.")

    for tutorial in tutorials:
        if not include_ignored and tutorial.name in ignored_tutorials:
            print(f"Ignoring tutorial {tutorial.name}.")
            continue
        num_runs += 1
        error, performance_info = run_tutorial(tutorial, smoke_test=smoke_test)
        if error:
            num_errors += 1
            print(error)
        else:
            print(
                f"Running tutorial {tutorial.name} took "
                f"{performance_info['runtime']:.2f} seconds. Memory usage "
                f"started at {performance_info['start_mem']} MB and the maximum"
                f" was {performance_info['max_mem']} MB."
            )

    if num_errors > 0:
        raise RuntimeError(
            f"Running {num_runs} tutorials resulted in {num_errors} errors."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tutorials.")
    parser.add_argument(
        "-p", "--path", metavar="path", required=True, help="botorch repo directory."
    )
    parser.add_argument(
        "-s", "--smoke", action="store_true", help="Run in smoke test mode."
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Run all tutorials (incl. ignored).",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Run a specific tutorial by name. The name should include the "
        ".ipynb extension. If the tutorial is on the ignore list, you still need "
        "to specify --include-ignored.",
    )
    args = parser.parse_args()
    run_tutorials(
        repo_dir=args.path,
        include_ignored=args.include_ignored,
        smoke_test=args.smoke,
        name=args.name,
    )
