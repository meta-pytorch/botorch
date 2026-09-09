#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Overnight driver: run the two remaining PD1 experiments back to back.
#
# STAGE A -- drift check on S27 (~45 min). S27.10 established that the current binary
# does not reproduce the 2026-08-08 arm-C data. S27's conclusions rest on that data, so
# the open question is whether the drift changes the method ORDERING or only the bits.
# This re-runs the arm-C configuration on current code so orderings can be compared.
# PACOH is dropped: it costs ~12616 s/cell, and S27.10 proved the --methods list does
# not perturb the other arms, so dropping it is free here.
#
# STAGE B -- alpha x target factorial (~6 h). The main event; rationale in
# run_armc_factorial.sh. Self-contained: carries its own reference arms and runs on one
# binary, so the S27.10 drift does not affect it.
#
# Ordering matters: A is short and its result decides what B gets compared against, so
# it goes first. B does not depend on A completing successfully.
#
# Launch with an ABSOLUTE path -- systemd-run does not inherit the caller's cwd, and a
# relative path exits in milliseconds with no logs (learned the hard way).

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
R="$S/results"
SHARDS=12
mkdir -p "$R/raw/drift_check" "$R/logs"

say() { echo "[overnight $(date -u +%H:%M:%SZ)] $*"; }

say "STAGE A: S27 drift check"
DRIFT="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --pd1-candidate-pool full --pd1-pretrain-pool full \
  --n-iters 50 --n-seeds 5 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-iters 25000 --ablr-iters 10000 --ablr-per-task-precision --threads 3 \
  --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,ablr,vanilla_gp,random"

for i in $(seq 0 $((SHARDS - 1))); do
  out="$R/raw/drift_check/rerun_s${i}.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_experiment $DRIFT \
    --pretrain-seed 0 --loo-shard "${i}/${SHARDS}" --out "$out" \
    > "$R/logs/drift_rerun_s${i}.txt" 2>&1 &
done
wait
done=$(find "$R/raw/drift_check" -name 'rerun_s*.json' 2>/dev/null | wc -l)
say "STAGE A done: $done/$SHARDS shards"

say "STAGE B: alpha x target factorial"
bash "$S/results/run_armc_factorial.sh"
say "STAGE B done"

say "ALL DONE"
