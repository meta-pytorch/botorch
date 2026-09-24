#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Stage 3 for the two REGRESSION regimes, which failed on the first autonomous run.
#
# select_top_configs ranked on budget_to_0.01, which regression summaries do not carry, so
# it raised "no rankable configs" -- the guard behaving correctly rather than picking
# arbitrary cells. Fixed to be metric-aware (regression ranks on smallest-n_obs rank_corr).
#
# ORDERING MATTERS. This deliberately waits for the noise sweep, because §27.38's RNG-guard
# fix must land in bo_diagnose FIRST -- otherwise these cells inherit the same contaminated
# control (`hyperbo_frozen` taking 3 distinct values instead of 1). The script refuses to
# run until it can see the guard in the source.
set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
PARALLEL="${CELLS_PARALLEL:-3}"

say() { echo "[s3reg $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for_unit() {
  local unit="$1" poll="${2:-300}" st misses=0
  while :; do
    st=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
    case "$st" in
      active|activating|deactivating|reloading) misses=0 ;;
      inactive|failed)
        sleep 5
        st=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
        case "$st" in inactive|failed) return 0 ;; esac ;;
      *)
        misses=$((misses + 1))
        [ $((misses % 12)) -eq 1 ] && say "cannot read $unit state ('$st'); still waiting" ;;
    esac
    sleep "$poll"
  done
}

for u in em-noise-sweep.service em-chain-stage2.service; do
  say "waiting for $u"
  wait_for_unit "$u" 300
  say "$u finished"
done

# Gate: do not generate regression results with the known-contaminated control.
#
# This WAITS rather than aborts. The guard cannot be applied while bo_diagnose is busy
# (editing it rebuilds the binary mid-sweep), so the natural order is: sweeps finish ->
# guard applied -> this proceeds. Aborting would mean the guard landing five minutes too
# late silently costs the whole stage, which is how the em-chain-stage2 chain was lost on
# 2026-08-20.
guard='manual_seed(args.pretrain_seed \* 1009 + 11)'
waited=0
while ! grep -q "$guard" "$ABS/bo_diagnose.py"; do
  [ $((waited % 12)) -eq 0 ] && say "waiting for the §27.38 RNG guard in bo_diagnose.py (apply it to proceed)"
  waited=$((waited + 1))
  sleep 300
  if [ "$waited" -gt 576 ]; then     # 48 h
    say "GIVING UP: guard still absent after 48 h"
    exit 1
  fi
done
say "RNG guard present in bo_diagnose; proceeding"

FAILED=0
for rg in lcb_reg pd1_reg; do
  stage="stage3_$rg"
  queue="/tmp/queue_${stage}.tsv"
  info=$(python3 "$ABS/gen_stage_queue.py" --stage "$stage" --regime "$rg" \
    --kind replicate --select-from "stage1_$rg" --out "$queue" 2>&1) \
    || { say "$stage queue generation FAILED: $info"; FAILED=$((FAILED + 1)); continue; }
  shards=$(echo "$info" | sed -n 's/^shards=//p')
  case "$shards" in "" | *[!0-9]*) say "$stage: no usable shards="; FAILED=$((FAILED + 1)); continue ;; esac

  common=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['$rg']['flags'])")
  [ -n "$common" ] || { say "$stage: empty COMMON"; FAILED=$((FAILED + 1)); continue; }

  say "$stage: $info"
  STUDY="v2/$stage" QUEUE="$queue" COMMON="$common" SHARDS="$shards" \
    HARNESS=bo_diagnose THREADS=3 CELLS_PARALLEL="$PARALLEL" \
    bash "$ABS/results/run_cell_queue.sh" || FAILED=$((FAILED + 1))
  python3 "$ABS/summarize_stage.py" --stage "$stage" 2>&1 | tail -2
done

say "=== done, $FAILED failures ==="
exit $([ "$FAILED" -eq 0 ] && echo 0 || echo 1)
