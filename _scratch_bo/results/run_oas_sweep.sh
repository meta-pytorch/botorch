#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# OAS / Ledoit-Wolf sweep for the Inverse-Wishart nu on PD1 arms A and C.
#
# Asks whether nu is better ESTIMATED from the pre-training tasks than fixed by hand.
# The cell list lives in gen_stage_queue.IWNU; read the comment there before changing
# it, because two traps are already designed around:
#
#   * --iw-nu-mode does NOTHING without --use-covar-prior, and neither PD1 arm sets it,
#     so a sweep varying only the mode would be a silent no-op reading as "no effect".
#   * The control must ALSO hold --use-covar-prior on, or the contrast becomes
#     "covar prior on/off" rather than "how nu is chosen".
#
# Serialised behind em-autonomous rather than run alongside it: the sweep varies an EM
# parameter the other stages hold fixed, and one box running two EM configurations at
# once just makes both slower. The prior cache keys on the parameter, so overlapping
# would be SAFE -- serialising simply removes the question (same reasoning as
# run_noise_sweep.sh).
#
# LAUNCH:
#   systemd-run --user --slice=user.slice --unit=em-oas --collect \
#     bash -c 'exec > /tmp/em_oas.log 2>&1; exec bash <ABS>/results/run_oas_sweep.sh'
set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
PARALLEL="${CELLS_PARALLEL:-3}"
REGIMES="${REGIMES:-pd1_bo_armA pd1_bo_armC}"
WAIT_FOR="${WAIT_FOR:-em-autonomous.service}"

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

say() { echo "[oas $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

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
  stage="stage2iwnu_$rg"
  queue="/tmp/queue_${stage}.tsv"

  info=$(python3 "$ABS/gen_stage_queue.py" --stage "$stage" --regime "$rg" \
    --kind iwnu --out "$queue" 2>&1) \
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

say "=== oas sweep done, $FAILED regime failures ==="
exit $([ "$FAILED" -eq 0 ] && echo 0 || echo 1)
