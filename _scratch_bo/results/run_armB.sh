#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Arm B: the missing cell of the 2x2, run standalone.
#
# Arms A and C differ in TWO flags at once, so §27.34's attribution of EM's off-grid
# deficit to Nystrom extrapolation is INFERRED rather than measured. Arm B holds
# pre-training matched (as in A) while searching the full pool (as in C):
#
#   A vs B  -> isolates the INTERPOLATION effect     (pre-training held fixed)
#   B vs C  -> isolates the PRE-TRAINING POOL effect (candidates held fixed)
#
# Run standalone rather than through run_autonomous.sh, because that driver's regime list
# is fixed at launch and it is currently executing (editing it mid-run corrupts bash's
# read offset -- see HANDOFF).
#
# Comparability: this uses the same binary as arms A and C. The --em-likelihood-noise flag
# added alongside is provably inert at its default -- an identical reference cell hashes
# bit-identically before and after (6d04729e...), so arm B remains comparable.
set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
STAGE="${STAGE:-stage1_pd1_bo_armB}"
KIND="${KIND:-ofat}"
PARALLEL="${CELLS_PARALLEL:-3}"

say() { echo "[armB $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Do not fight the main driver for cores while it is on a 12-shard BO regime.
while systemctl --user is-active --quiet em-stage13.service; do
  cur=$(tail -1 /tmp/em_autonomous.log 2>/dev/null | grep -oE "stage1_[a-z0-9_]+" | head -1)
  case "$cur" in
    *bo_arm*) say "main driver is on $cur (12 shards, saturating); waiting"; sleep 300 ;;
    *) break ;;
  esac
done
say "proceeding (main driver is on a 1-shard regime or finished)"

queue="/tmp/queue_${STAGE}.tsv"
info=$(python3 "$ABS/gen_stage_queue.py" --stage "$STAGE" --regime pd1_bo_armB \
  --kind "$KIND" --out "$queue" 2>&1) || { say "queue generation FAILED: $info"; exit 1; }
say "$info"

shards=$(echo "$info" | sed -n 's/^shards=//p')
case "$shards" in "" | *[!0-9]*) say "no usable shards= from generator"; exit 1 ;; esac

common=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['pd1_bo_armB']['flags'])")
[ -n "$common" ] || { say "empty COMMON"; exit 1; }
common="$common --prior-cache $ABS/results/prior_cache --prior-cache-mode readwrite"

say "launching $KIND on pd1_bo_armB, shards=$shards, parallel=$PARALLEL"
STUDY="v2/$STAGE" QUEUE="$queue" COMMON="$common" SHARDS="$shards" \
  HARNESS=bo_experiment THREADS=3 CELLS_PARALLEL="$PARALLEL" \
  bash "$ABS/results/run_cell_queue.sh"
rc=$?
say "queue exited $rc"

python3 "$ABS/summarize_stage.py" --stage "$STAGE" 2>&1 | tail -3
say "done"
exit $rc
