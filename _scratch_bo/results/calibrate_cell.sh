#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# STAGE-LAUNCH STEP 1: one FULL-SIZE cell, to calibrate before committing days of compute.
#
# The toy smoke (smoke_pipeline.sh) proved the plumbing. This proves the timing and
# anything scale-dependent, on the exact REGIMES config the matrix will use -- 50 BO
# iterations, 25k HyperBO steps, full/full pools, 12 shards. One prior, one cell.
#
# What it establishes before stage 0 is released:
#   * real seconds per cell at CELLS_PARALLEL=1, to re-derive the ~25 min/cell estimate
#     that the plan's sizing rests on (that number came from PRE-fix runs)
#   * that --n-seeds 1 produces the expected 23 runs per shard-set, not 3x that
#   * peak memory and load, to choose CELLS_PARALLEL for the real run
set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
OUT="/tmp/calib_$(date -u +%s)"
mkdir -p "$OUT"

say() { echo "[calib $(date -u +%H:%M:%SZ)] $*"; }

# Pull the exact flags the matrix will use, rather than restating them here where they
# could drift from REGIMES.
COMMON=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['pd1_bo_armC']['flags'])")
SHARDS=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['pd1_bo_armC']['shards'])")
say "config: $COMMON"
say "shards: $SHARDS"

printf 'calib_p0\t--methods em_frozen,em_additive_hyperbo,hyperbo_frozen --pretrain-seed 0\n' \
  > "$OUT/q.tsv"

START=$(date +%s)
( while true; do
    sleep 60
    printf '[calib mon %s] load=%s mem_avail=%sG shards_done=%s/%s\n' \
      "$(date -u +%H:%M:%SZ)" \
      "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)" \
      "$(find "$OUT/results/raw/calib" -name '*.json' 2>/dev/null | wc -l)" "$SHARDS"
  done ) &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

STUDY="calib" QUEUE="$OUT/q.tsv" COMMON="$COMMON" SHARDS="$SHARDS" THREADS=3 \
  CELLS_PARALLEL=1 HARNESS=bo_experiment RESULTS_DIR="$OUT/results" \
  bash "$ABS/results/run_cell_queue.sh"
RC=$?
ELAPSED=$(( $(date +%s) - START ))
kill $MON 2>/dev/null

say "=== cell exit=$RC in ${ELAPSED}s ($((ELAPSED / 60)) min) ==="
n=$(find "$OUT/results/raw/calib" -name '*.json' 2>/dev/null | wc -l)
say "shards written: $n/$SHARDS"

OUT="$OUT" SHARDS="$SHARDS" ELAPSED="$ELAPSED" python3 - <<'PYEOF'
import glob, json, os

out, shards, elapsed = os.environ["OUT"], int(os.environ["SHARDS"]), int(os.environ["ELAPSED"])
files = sorted(glob.glob(os.path.join(out, "results/raw/calib/*.json")))
if not files:
    print("  NO SHARDS -- calibration failed")
    raise SystemExit(1)

runs = tasks = 0
methods = set()
for f in files:
    d = json.load(open(f))
    pr = d["per_run"]
    runs += len(pr["pool_max_raw"])
    tasks += len(set(pr.get("eval_dataset_idx", [])))
    methods |= set(pr["traj_raw"])

print(f"  runs total          : {runs}")
print(f"  methods             : {sorted(methods)}")
print(f"  n_seeds in config   : {json.load(open(files[0]))['config'].get('n_seeds')}")
print()
print("  PROJECTION for the real matrix:")
print(f"    1 cell (12 shards, 1 prior)     : {elapsed / 60:.1f} min")
for par in (2, 3):
    per = elapsed / 60 / par
    print(f"    at CELLS_PARALLEL={par}            : {per:.1f} min/cell effective")
    for label, cells in (("stage0 (5 regimes x 9 priors)", 45),
                         ("stage1 pd1_bo_armC (126 cells)", 126)):
        print(f"      {label:32s} {per * cells / 60:6.1f} h")
PYEOF

say "artifacts: $OUT"
exit $RC
