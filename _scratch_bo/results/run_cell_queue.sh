#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Generic parallel CELL QUEUE for the EXPERIMENT_PLAN.md matrix.
#
# WHY THIS EXISTS
#
# Every runner so far ran cells STRICTLY SEQUENTIALLY, each cell spawning 12 shards at
# 3 torch threads = 36 threads on a 96-core box. That is 37% utilisation, and it is why
# the plan's first sizing came out at ~9 days.
#
# Measured reality (from the mtimes of the hybrid sweep and the sumll factorial cells):
#   cell-to-cell wall clock  ~25 min   (12 shards, 3 threads, PD1 arm C, 3 seeds)
#   shard span within a cell ~12 min   (the rest is buck startup + stragglers)
# The plan had assumed ~60 min/cell. So the original estimate was wrong twice: per-cell
# cost overestimated ~2.4x, AND cross-cell parallelism ignored entirely.
#
# This driver runs CELLS_PARALLEL cells at once. At the default 3 x 12 x 3 = 108 threads
# it slightly oversubscribes 96 cores, which is usually a net win because shards idle
# during buck startup and I/O; drop to 2 if load climbs past ~110 or runtimes regress.
#
# INPUT: a queue file, one cell per line:
#     <cell_name><TAB><extra flags for bo_experiment>
# Blank lines and #-comments ignored. Per-cell output goes to
# $R/raw/$STUDY/<cell_name>_s<i>.json, so the whole queue is resumable -- rerunning skips
# any shard whose json already exists.
#
# USAGE
#   STUDY=stage1_ofat QUEUE=/path/to/cells.tsv COMMON="--benchmark pd1_loo ..." \
#     bash run_cell_queue.sh
#
# Launch it with an ABSOLUTE path under systemd-run; a relative path exits silently.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
R="${RESULTS_DIR:-$S/results}"
STUDY="${STUDY:?set STUDY (output subdir under results/raw)}"
QUEUE="${QUEUE:?set QUEUE (tsv of cell_name<TAB>flags)}"
COMMON="${COMMON:?set COMMON (shared bo_experiment flags)}"
# No default: an empty SHARDS silently became 12, which on the single-shard regimes
# produced 12 byte-identical runs and inflated n_runs 12x (review B14).
SHARDS="${SHARDS:?set SHARDS}"
HARNESS="${HARNESS:-bo_experiment}"   # bo_experiment (BO) or bo_diagnose (regression)
THREADS="${THREADS:-3}"
CELLS_PARALLEL="${CELLS_PARALLEL:-3}"
# Non-numeric values make the throttle test error out, which skips the throttle
# entirely and launches every cell at once (review B15).
case "$SHARDS" in "" | *[!0-9]*) echo "SHARDS must be numeric" >&2; exit 2 ;; esac
case "$CELLS_PARALLEL" in "" | *[!0-9]*) echo "CELLS_PARALLEL must be numeric" >&2; exit 2 ;; esac

mkdir -p "$R/raw/$STUDY" "$R/logs/$STUDY"
# Stale from a previous run: without this the standalone entry point reports
# failure forever even after a clean resume (review N3).
rm -f "$R/raw/$STUDY/.incomplete"

# Kill the whole process group on TERM/INT so a stopped queue does not leave
# reparented build-tool grandchildren still writing shard json (review N19).
cleanup() {
  say "signal received, terminating shards"
  kill -- -$$ 2>/dev/null
  exit 130
}
trap cleanup TERM INT
say() { echo "[queue $(date -u +%H:%M:%SZ)] $*"; }

run_cell() {
  local name="$1" flags="$2" i out shard_flag rc=0 p have
  local pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    out="$R/raw/$STUDY/${name}_s${i}.json"
    # A shard json that exists but is not valid JSON is a truncated write from an
    # earlier crash. `[ -f ]` alone would skip it forever and summarise_stage would
    # then turn the whole cell into {"kind": "error"} (review, Q2).
    if [ -f "$out" ]; then
      if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$out" \
        > /dev/null 2>&1; then
        continue
      fi
      say "CORRUPT shard, quarantining and re-running: $out"
      mv -f "$out" "$out.corrupt.$(date -u +%s)"
    fi
    # --loo-shard is a bo_experiment concept. Single-shard regimes, and
    # bo_diagnose, must not receive it.
    shard_flag=""
    [ "$SHARDS" -gt 1 ] && shard_flag="--loo-shard ${i}/${SHARDS}"
    printf '%s\t%s\t%s\n' "${STUDY}_${name}_s${i}" "$(date -u +%FT%TZ)" \
      "$HARNESS $COMMON $flags" >> "$R/MANIFEST.tsv"
    # stdin from /dev/null: the queue file is this loop's stdin, and a child that
    # reads it advances the SHARED offset, silently skipping queue lines (review B9).
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.$HARNESS $COMMON $flags \
      --threads "$THREADS" $shard_flag --out "$out" \
      < /dev/null >> "$R/logs/$STUDY/${name}_s${i}.txt" 2>&1 &
    # Record the PID at spawn: `jobs -p` omits jobs already reaped, and waiting on
    # a missing job returns 127, which falsely marked the cell INCOMPLETE (N2).
    pids+=($!)
  done
  # Bare `wait` returns 0 regardless of child exit codes, so shard failures were
  # invisible (review B16). Collect them explicitly.
  for p in "${pids[@]:-}"; do
    [ -n "$p" ] || continue
    wait "$p" || rc=1
  done
  have=$(find "$R/raw/$STUDY" -maxdepth 1 -name "${name}_s*.json" | wc -l)
  # Stale shards from a different SHARDS value would otherwise be pooled with the
  # new ones, duplicating tasks in every mean (review N8).
  if [ "$have" -gt "$SHARDS" ]; then
    say "cell $name has $have shard files but SHARDS=$SHARDS -- stale shards from a"
    say "  different sharding are present; refusing to summarise a mixed cell"
    return 1
  fi
  if [ "$have" -eq "$SHARDS" ] && [ "$rc" -eq 0 ]; then
    say "cell $name done ($have/$SHARDS)"
    return 0
  fi
  say "cell $name INCOMPLETE ($have/$SHARDS, rc=$rc)"
  return 1
}

n=0
incomplete=0
while IFS=$'\t' read -r -u 3 name flags; do
  case "$name" in ''|\#*) continue ;; esac
  # Throttle to CELLS_PARALLEL concurrent cells.
  while [ "$(jobs -rp | wc -l)" -ge "$CELLS_PARALLEL" ]; do wait -n; done
  say "starting cell $name"
  { run_cell "$name" "$flags" || echo "$name" >> "$R/raw/$STUDY/.incomplete"; } &
  n=$((n + 1))
done 3< "$QUEUE"
wait

if [ -s "$R/raw/$STUDY/.incomplete" ]; then
  incomplete=$(wc -l < "$R/raw/$STUDY/.incomplete")
  say "queue finished with $incomplete INCOMPLETE cell(s) of $n"
  exit 1
fi
say "queue complete: $n cells -> $R/raw/$STUDY/"
