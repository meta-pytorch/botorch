#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Apply the §27.38 RNG re-seed guard to bo_diagnose.py. Idempotent; refuses if busy.

WHY. `hyperbo_frozen` never touches the EM prior, so it must be identical across configs
within a regime. On the BO regimes it is (1 distinct value). On `lcb_reg` it took THREE
values -- 0.7118 for eleven configs, 0.7087 for `mean_linear` and
`mean_linear+covprior`,
0.7082 for `canon_deep`. Exactly the configs that build extra parameterised modules
(a LinearMean, an MLP): they consume RNG draws, shift the stream, and silently
re-initialise every neural baseline downstream.

`bo_experiment` has guarded this since round 5 with a re-seed before each baseline.
The fix
was never propagated to `bo_diagnose` -- the same failure mode as the §27.23 units
correction.

WHY THIS IS NOT INERT, unlike the other recent changes. It deliberately CHANGES RNG
consumption; that is the entire point. So it cannot be applied while cells are in
flight,
and this script refuses to run if any bo_diagnose process is alive.

Run with --check to see what it would do.
"""

import argparse
import subprocess
import sys

TARGET = "bo_diagnose.py"
GUARD = "torch.manual_seed(args.pretrain_seed * 1009 + 11)"
COMMENT = (
    "# Re-seed per baseline so its init does not depend on how many RNG draws\n"
    "        # upstream code consumed. Without this, configs that build extra\n"
    "        # parameterised modules (--em-base-mean linear, --deep-kernel) shift the\n"
    "        # stream and silently move `hyperbo_frozen`, which is supposed to be a\n"
    "        # bit-identical control across configs (§27.38).\n"
)

# The four neural-baseline pre-training calls. Anchored on the print() that precedes
# each,
# which is stable, rather than on line numbers, which are not.
ANCHORS = [
    'print("Pre-training HyperBO first (its kernel seeds EM) ...", flush=True)',
    'print("Pre-training HyperBO ...", flush=True)',
    'print("Pre-training PACOH-GP ...", flush=True)',
    'print("Pre-training ABLR ...", flush=True)',
]


def busy() -> int:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return sum(1 for ln in out.splitlines() if "bo_diagnose-inplace" in ln)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report, do not modify")
    args = ap.parse_args()

    src = open(TARGET).read()
    n_before = src.count(GUARD)
    print(f"guard occurrences before: {n_before}")

    missing = [a for a in ANCHORS if a not in src]
    if missing:
        print("ERROR: anchors not found, the file has drifted:")
        for m in missing:
            print("   ", m[:78])
        return 1

    todo = []
    for a in ANCHORS:
        i = src.index(a)
        after = src[i + len(a) : i + len(a) + 400]
        todo.append((a, GUARD not in after))
    need = sum(1 for _a, t in todo if t)
    print(f"sites needing the guard: {need} / {len(ANCHORS)}")
    for a, t in todo:
        print(f"   [{'ADD' if t else 'ok '}] {a[7:60]}")

    if args.check:
        return 0
    if need == 0:
        print("nothing to do; already applied")
        return 0

    live = busy()
    if live:
        print(
            f"REFUSING: {live} live bo_diagnose process(es). This change is NOT inert."
        )
        return 2

    out = src
    for a, t in todo:
        if not t:
            continue
        indent = " " * 8
        out = out.replace(a, f"{a}\n{indent}{COMMENT}{indent}{GUARD}", 1)
    open(TARGET, "w").write(out)
    print(f"applied to {need} site(s); guard occurrences now: {out.count(GUARD)}")

    import ast

    ast.parse(out)
    print("parses OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
