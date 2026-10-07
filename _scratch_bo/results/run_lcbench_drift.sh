#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# STAGE C -- drift check on the LCBench headline sweep (S26.1 / S26.2).
#
# WHY. S27.10 found the current binary does not reproduce `pd1pool` (run 2026-08-08).
# `bo_multisplit` -- which backs S26.1 ("EM is top-tier") and S26.2 (the Pareto claim),
# the project's surviving headline results -- ran 2026-08-11, i.e. AFTER pd1pool but
# BEFORE several further changes to bo_experiment.py, one of them titled "RNG fix".
# So it sits inside the drift window and its reproducibility is untested.
#
# If two cells reproduce bit-identically, S26 is safe and the drift is bounded to the
# Aug-8 era. If they do not, the two headline claims rest on unreproducible data and
# must be re-run before publication. Either answer is worth an hour.
#
# PACOH is dropped: ~12616 s/cell, and main() re-seeds before each baseline's
# pre-training (gotcha #3), so removing one does not perturb the others. Two cells are
# enough to detect drift; this is a reproducibility check, not a re-measurement.
#
# WAITS for pd1-overnight-batch so the box is not oversubscribed.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
R="$S/results"
mkdir -p "$R/raw/drift_lcbench" "$R/logs"

say() { echo "[stageC $(date -u +%H:%M:%SZ)] $*"; }

say "waiting for pd1-overnight-batch"
DEADLINE=$((SECONDS + 43200))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  systemctl --user is-active --quiet pd1-overnight-batch.service || break
  sleep 60
done
say "proceeding (batch state: $(systemctl --user is-active pd1-overnight-batch.service 2>&1))"

# Config copied from bo_multisplit's recorded config, minus pacoh_frozen.
# NOTE: ABLR's flags must match bo_multisplit exactly (ablr_iters=25000, and NO
# --ablr-per-task-precision). The first version of this script copied ABLR's flags from
# the PD1 runner instead (10000 + per-task precision), which made ablr/ablr_adapt differ
# for a reason that had nothing to do with drift -- see S27.13.
LCB="--benchmark lcbench --n-configs 200 --n-iters 40 --n-seeds 6 --n-eval 6 \
  --n-pretrain 25 --meta-iters 2000 --hyperbo-iters 25000 --ablr-iters 25000 \
  --threads 5 \
  --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,ablr,ablr_adapt,vanilla_gp,random"

for cell in "0 0" "1 0"; do
  # shellcheck disable=SC2086
  set -- $cell
  sp=$1 pp=$2
  out="$R/raw/drift_lcbench/sp${sp}_p${pp}.json"
  [ -f "$out" ] && continue
  say "cell sp${sp}_p${pp}"
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_experiment $LCB \
    --split-seed "$sp" --pretrain-seed "$pp" --out "$out" \
    > "$R/logs/drift_lcbench_sp${sp}_p${pp}.txt" 2>&1 &
done
wait

done_n=$(find "$R/raw/drift_lcbench" -name 'sp*.json' 2>/dev/null | wc -l)
say "STAGE C done: $done_n/2 cells"
say "compare against results/raw/bo_multisplit/ -- bit-identity first, then ordering"
