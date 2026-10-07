#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Chain stage 2 (the crossed MEAN sweep) onto the end of the running stage 1/3 job.
#
# WHY THIS EXISTS
#
# The driver runs the stages it was launched with and then exits; nothing picks up idle
# capacity afterwards. Stage 2 was added to gen_stage_queue AFTER the current run started,
# and the matching driver edit had to be pulled back out because editing a bash script
# while bash is executing it corrupts the read offset (see HANDOFF "PENDING"). So stage 2
# needs (a) the driver swap and (b) a launch, both strictly AFTER the current unit exits.
#
# This waits for that, does both, and refuses to act if anything looks wrong.
#
# Usage:
#   systemd-run --user --unit=em-chain-stage2 --collect bash <ABS>/results/chain_stage2.sh

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ABS="$ROOT/_scratch_bo"
PENDING="$ABS/results/run_autonomous.stage2.sh.pending"
DRIVER="$ABS/results/run_autonomous.sh"
WATCH_UNIT="${WATCH_UNIT:-em-stage13.service}"
POLL="${POLL:-120}"

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

say() { echo "[chain $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

say "waiting for $WATCH_UNIT to finish before touching the driver"
wait_for_unit "$WATCH_UNIT" "$POLL"
say "$WATCH_UNIT is done"

# Belt and braces: the flock is what actually prevents two drivers, but if some other
# run_autonomous is alive we must not swap the script out from under it.
if pgrep -f "bash .*run_autonomous.sh" >/dev/null 2>&1; then
  say "ABORT: a run_autonomous.sh process is still alive; refusing to swap the driver"
  exit 1
fi

[ -f "$PENDING" ] || { say "ABORT: $PENDING missing"; exit 1; }
bash -n "$PENDING" || { say "ABORT: pending driver does not parse"; exit 1; }
grep -q "stage2means" "$PENDING" || { say "ABORT: pending driver lacks stage 2"; exit 1; }

cp "$DRIVER" "$ABS/results/.run_autonomous.prechain.bak"
mv "$PENDING" "$DRIVER"
bash -n "$DRIVER" || { say "ABORT: driver broken after swap"; exit 1; }
say "driver swapped in ($(wc -c < "$DRIVER") bytes); stage 2 is now available"

# Only stage 2. Stages 1 and 3 already ran in the job we just waited for, and re-running
# them would be a no-op anyway (.DONE gating) but would waste hours re-summarising.
say "launching stage 2"
STAGES=2 CELLS_PARALLEL=3 bash "$DRIVER"
rc=$?
say "stage 2 exited $rc"
exit $rc
