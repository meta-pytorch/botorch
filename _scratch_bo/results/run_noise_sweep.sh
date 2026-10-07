#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# EM pre-training noise sweep (§27.35 / §27.36), chained behind Arm B.
#
# WHY IT IS QUEUED RATHER THAN RUN NOW
#
# Measured at launch time: load1 87.9-94.5 on 96 cores with the run queue spiking to 146,
# i.e. already saturated by Arm B (12 shards x 3 cells) plus the stage-1 regression
# regimes. Memory was not the constraint (75 GB free). Adding 63 more cells would have
# slowed everything without finishing sooner, so this waits.
#
# It also waits for a second reason: the sweep VARIES a parameter the other arms hold
# fixed, so overlapping it with Arm B would put two different EM configurations in flight
# against a shared prior cache at the same time. The cache keys on the parameter, so that
# is safe -- but serialising removes the question entirely.
#
# Runs on all five regimes because §27.34 showed these effects are regime-dependent.
set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
PARALLEL="${CELLS_PARALLEL:-3}"
REGIMES="${REGIMES:-pd1_bo_armC pd1_bo_armA lcb_bo lcb_reg pd1_reg}"
WAIT_FOR="${WAIT_FOR:-em-armB.service em-stage13.service}"

# Wait for a unit, distinguishing "finished" from "systemctl itself failed".
#
# `systemctl is-active --quiet` exits non-zero for BOTH cases, and a transient
# "Failed to connect to user scope bus" is enough to look like completion. That is
# exactly what killed em-chain-stage2 on 2026-08-20: a dbus blip made it conclude
# em-stage13 had finished while it was still running. Its later guard caught the
# mistake and aborted, but the chain was lost. Read the state explicitly and treat an
# unreadable state as "keep waiting", never as "done".
wait_for_unit() {
  local unit="$1" poll="${2:-300}" st misses=0
  while :; do
    st=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
    case "$st" in
      active|activating|deactivating|reloading)
        misses=0 ;;
      inactive|failed)
        # Confirm with a second read; one clean sample is not proof.
        sleep 5
        st=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
        case "$st" in inactive|failed) return 0 ;; esac ;;
      *)
        # Empty or unrecognised => systemctl failed. Keep waiting, do not advance.
        misses=$((misses + 1))
        [ $((misses % 12)) -eq 1 ] && \
          echo "[wait] cannot read $unit state ('"'"'$st'"'"'); still waiting" ;;
    esac
    sleep "$poll"
  done
}

say() { echo "[noise $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

for u in $WAIT_FOR; do
  say "waiting for $u"
  wait_for_unit "$u" 300
  say "$u finished"
done

# Do not start into a box someone else has filled in the meantime.
for _ in $(seq 1 60); do
  l=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  [ "$l" -lt 48 ] && break
  say "load $l still high; waiting"
  sleep 300
done

FAILED=0
for rg in $REGIMES; do
  stage="stage2noise_$rg"
  queue="/tmp/queue_${stage}.tsv"

  info=$(python3 "$ABS/gen_stage_queue.py" --stage "$stage" --regime "$rg" \
    --kind noise --out "$queue" 2>&1) \
    || { say "$stage queue generation FAILED: $info"; FAILED=$((FAILED + 1)); continue; }
  shards=$(echo "$info" | sed -n 's/^shards=//p')
  case "$shards" in "" | *[!0-9]*) say "$stage: no usable shards="; FAILED=$((FAILED + 1)); continue ;; esac

  harness=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['$rg']['harness'])")
  common=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['$rg']['flags'])")
  [ -n "$common" ] && [ -n "$harness" ] || { say "$stage: empty flags/harness"; FAILED=$((FAILED + 1)); continue; }

  # Both harnesses accept the prior cache since S30.2; see run_autonomous.sh for why
  # the RNG re-seed guard is the precondition that makes a cache hit safe.
  common="$common --prior-cache $ABS/results/prior_cache --prior-cache-mode readwrite"

  say "$stage: $info, harness=$harness, parallel=$PARALLEL"
  STUDY="v2/$stage" QUEUE="$queue" COMMON="$common" SHARDS="$shards" \
    HARNESS="$harness" THREADS=3 CELLS_PARALLEL="$PARALLEL" \
    bash "$ABS/results/run_cell_queue.sh" || FAILED=$((FAILED + 1))

  python3 "$ABS/summarize_stage.py" --stage "$stage" 2>&1 | tail -2
done

say "=== noise sweep done, $FAILED regime failures ==="
exit $([ "$FAILED" -eq 0 ] && echo 0 || echo 1)
