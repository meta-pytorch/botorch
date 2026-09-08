#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# AUTONOMOUS driver for EXPERIMENT_PLAN.md. Runs unattended for ~2 days; no check-ins.
#
# DESIGN FOR UNATTENDED OPERATION
#
# 1. RESUMABLE AT EVERY LEVEL. Shards skip if their json exists, cells skip if complete,
#    stages skip if their DONE marker exists. Re-running this script after any failure
#    picks up exactly where it stopped and re-does nothing.
# 2. NOTHING IS LOST ON CRASH. Each shard writes its own json the moment it finishes, and
#    the stage is re-summarised into SUMMARY.json after EVERY cell -- not at the end. If
#    the box dies at hour 30 of 40, every completed cell is already reduced to numbers.
# 3. NO HUMAN DECISIONS MID-RUN. Stage 2 and 3 pick their own cells from the previous
#    stage's SUMMARY.json, so the pipeline never blocks waiting for a judgement call.
# 4. FAILURE IS ISOLATED. A cell that errors leaves its shards missing and is recorded as
#    incomplete; the queue continues. One bad cell cannot take down the run.
# 5. STATUS IS ON DISK. STATUS.txt is rewritten continuously so progress can be read at
#    any time without attaching to anything.
#
# OUTPUT LAYOUT -- the hard old/new boundary
#
#   results/raw/v2/<stage>/         <- EVERYTHING new. Post-2026-08-17 binary only.
#   results/raw/<anything else>/    <- old. RNG-incomparable (S27.15). Never mix.
#
# Each stage dir carries PROVENANCE.json (commit, date, cell list) and SUMMARY.json.
#
# PREREQUISITE: Stage 0.2 (the PD1 regression harness) is CODE and cannot be written by
# this script. If bo_diagnose does not accept --benchmark pd1_loo, the pd1_reg regime is
# skipped and recorded as skipped; every other regime still runs.
#
# LAUNCH (absolute path required -- systemd-run does not inherit cwd):
#   systemd-run --user --slice=user.slice --unit=em-autonomous --collect \
#     bash -c 'exec > /tmp/em_autonomous.log 2>&1; exec bash <ABS>/run_autonomous.sh'

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
R="$S/results"
V2="$R/raw/v2"
CELLS_PARALLEL="${CELLS_PARALLEL:-3}"
THREADS="${THREADS:-3}"
# Total concurrency target, in threads: CELLS_PARALLEL is DERIVED per stage as
# THREAD_BUDGET/(shards*THREADS) so single-shard regimes stop idling the box. 96 cores,
# slight oversubscription is a net win because shards idle during buck startup and I/O.
THREAD_BUDGET="${THREAD_BUDGET:-108}"
# Memory ceiling on concurrent cells: ~2 GB resident each, 235 GB box shared with two
# build-tool daemons at ~45 GB combined.
CELL_CAP="${CELL_CAP:-32}"
mkdir -p "$V2" "$R/logs"

# One driver at a time: two would write the same shard paths (review N19).
exec 8> "/tmp/em_autonomous.lock"
if ! flock -n 8; then
  echo "another run_autonomous.sh holds the lock; exiting" >&2
  exit 3
fi

STATUS="$ABS/results/raw/v2/STATUS.txt"
say() {
  echo "[auto $(date -u +%FT%TZ)] $*"
  { echo "[auto $(date -u +%FT%TZ)] $*"; } >> "$STATUS"
}

# ---------------------------------------------------------------- regime availability
PD1_REG_OK=0
# Build first so a BUILD FAILURE is not misreported as "flag absent", and anchor the
# grep on the option column -- the bare substring also matches --pd1-source's help
# text, so the probe could pass with the flag removed (review B12).
if ! python3 -c \"import _scratch_bo.bo_diagnose\" > /dev/null 2>&1; then
  say "FATAL: bo_diagnose does not build; cannot judge pd1_reg availability"
  exit 2
fi
# NB: do NOT pipe into `grep -q` here. `set -o pipefail` is global, and grep -q exits at
# the first match, which SIGPIPEs the build tool and makes the pipeline non-zero -- so the probe
# would report FALSE precisely when the flag IS present, silently deleting the pd1_reg
# regime from every run. Capture first, then match on the variable.
_diag_help=$(python3 -m _scratch_bo.bo_diagnose --help 2>/dev/null)
if printf '%s\n' "$_diag_help" | grep -qE '^[[:space:]]*--benchmark[[:space:], ]'; then
  PD1_REG_OK=1
fi
unset _diag_help
[ "$PD1_REG_OK" -eq 1 ] || say "pd1_reg SKIPPED: bo_diagnose has no --benchmark (stage 0.2 not built)"

BO_REGIMES="lcb_bo pd1_bo_armC pd1_bo_armA"
REG_REGIMES="lcb_reg"
[ "$PD1_REG_OK" -eq 1 ] && REG_REGIMES="lcb_reg pd1_reg"

# ---------------------------------------------------------------- one stage x regime
run_block() { # run_block <stage> <regime> <kind> [select_from]
  local stage="$1" regime="$2" kind="$3" select_from="${4:-}"
  local tag="${stage}_${regime}"
  local done_marker="$ABS/results/raw/v2/$tag/.DONE"

  if [ -f "$done_marker" ]; then
    say "$tag already complete, skipping"
    return 0
  fi

  local queue="/tmp/queue_${tag}.tsv"
  local info shards common harness qrc
  local sel_args=()
  [ -n "$select_from" ] && sel_args=(--select-from "$select_from")
  # NB: the array expansion below must stay quoted-and-subscripted. An earlier edit lost
  # it and passed a bare "" instead, so --select-from never reached the generator and
  # every stage-3 run failed on "requires --select-from".
  info=$(python3 "$ABS/gen_stage_queue.py" --stage "$tag" --regime "$regime" \
    --kind "$kind" --out "$queue" ${sel_args[@]+"${sel_args[@]}"} 2>&1) \
    || { say "$tag queue generation FAILED: $info"; return 1; }
  shards=$(echo "$info" | sed -n 's/^shards=//p')
  # An empty shards silently became 12 downstream, producing duplicate runs on the
  # single-shard regimes and inflating every SEM (review B14).
  case "$shards" in "" | *[!0-9]*) say "$tag: no usable shards= from generator"; return 1 ;; esac

  # Hold the THREAD budget constant, not the cell count. Total concurrency is
  # CELLS_PARALLEL * shards * THREADS, and run_cell_queue.sh's "3 x 12 x 3 = 108 threads"
  # note only holds for the 12-shard PD1 BO arms. lcb_bo, lcb_reg, pd1_reg and the
  # stage-3 lcb blocks all report shards=1, so a fixed CELLS_PARALLEL=3 ran 3 x 1 x 3 = 9
  # threads on a 96-core box -- measured at ~355% CPU, about 4% utilisation, with 142 GB
  # RAM idle. Scaling by the shard count leaves the 12-shard arms at 3 (unchanged, and
  # already correct) and lifts the 1-shard regimes to CELL_CAP.
  #
  # Numerically neutral: every shard is an independent process with deterministic
  # per-cell seeding, so how many run concurrently cannot change any shard's value. This
  # is why it is safe to change mid-corpus when rebuilding the harness is not.
  cells_par=$(( THREAD_BUDGET / (shards * THREADS) ))
  [ "$cells_par" -lt 1 ] && cells_par=1
  # Cap on memory, not cores: each cell is ~2 GB resident.
  [ "$cells_par" -gt "$CELL_CAP" ] && cells_par=$CELL_CAP
  say "$tag: $(grep -vc '^#' "$queue") cells, shards=$shards, parallel=$cells_par (budget ${THREAD_BUDGET}t)"

  common=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['$regime']['flags'])") \
    || { say "$tag: cannot read regime flags"; return 1; }
  [ -n "$common" ] || { say "$tag: empty COMMON, aborting stage"; return 1; }

  harness=$(python3 -c "
import sys; sys.path.insert(0,'$ABS')
from gen_stage_queue import REGIMES; print(REGIMES['$regime']['harness'])") \
    || { say "$tag: cannot read harness"; return 1; }
  [ -n "$harness" ] || { say "$tag: empty HARNESS, aborting stage"; return 1; }

  # BOTH harnesses now accept the prior cache (S30.2). bo_diagnose gained it together
  # with the RNG re-seed guard, which is the precondition: without the guard, skipping a
  # pre-training fit on a cache hit would leave the global RNG in a different state than
  # on a miss. With it, every baseline re-seeds deterministically and all post-pretrain
  # randomness uses local torch.Generator(), so hit and miss are observationally equal.
  # Do not withhold the flag from one harness again -- that is what left 252 regression
  # cells re-fitting a 25k-step HyperBO prior they could have loaded.
  common="$common --prior-cache $PRIOR_CACHE --prior-cache-mode readwrite"

  rm -f "$ABS/results/raw/v2/$tag/.incomplete"
  set -o pipefail
  STUDY="v2/$tag" QUEUE="$queue" COMMON="$common" SHARDS="$shards" \
    HARNESS="$harness" THREADS="$THREADS" CELLS_PARALLEL="$cells_par" \
    bash "$ABS/results/run_cell_queue.sh" 2>&1 | while read -r line; do
      echo "$line"
      # Detached and locked: run inline this stalls the pipe, which blocks the queue
      # from launching further cells (review B13).
      case "$line" in *"cell "*" done"*)
        ( flock -n 9 && python3 "$ABS/summarize_stage.py" --stage "$tag" \
            --expect-shards "$shards" < /dev/null > /dev/null 2>&1
        ) 9> "/tmp/sum_${tag}.lock" & ;;
      esac
    done
  qrc=${PIPESTATUS[0]}
  wait

  python3 "$ABS/summarize_stage.py" --stage "$tag" --expect-shards "$shards" \
    < /dev/null || true

  # .DONE only when the queue succeeded AND the summary certifies every cell is
  # complete. It used to be unconditional, so a stage in which every shard crashed was
  # marked complete and permanently skipped (review B1).
  if [ "${qrc:-1}" -eq 0 ] && python3 - "$ABS/results/raw/v2/$tag/SUMMARY.json" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
cells = s.get("cells", {})
expected = s.get("cells_expected") or 0
ok = len(cells) >= expected and expected > 0 and all(
    c.get("kind") != "error" and not c.get("shards_missing") for c in cells.values()
)
sys.exit(0 if ok else 1)
PYEOF
  then
    touch "$done_marker"
    say "$tag COMPLETE"
  else
    say "$tag INCOMPLETE (qrc=${qrc:-?}) -- NOT marking DONE; a rerun resumes it"
    return 1
  fi
}

# ---------------------------------------------------------------- stages
say "=== AUTONOMOUS RUN START (commit $(sl log -r . -T '{node|short}' 2>/dev/null)) ==="

FAILED_STAGES=0

# STAGES selects which stages run, so a staged launch can release one at a time and
# inspect the artifacts in between. Default is all of them; the calibration/inspection
# workflow uses STAGES="0" then STAGES="1" then STAGES="3".
STAGES="${STAGES:-0 1 2 3}"
# One shared prior store across every stage and regime. Pre-training dominates cost
# (~61 min of HyperBO per LCBench cell at 25000 iters) and most OFAT factors change only
# the EM side or the BO loop, so cells that share a pre-training config pay once. The key
# includes --pretrain-seed, so the 9 priors stay 9 distinct fits rather than collapsing.
PRIOR_CACHE="${PRIOR_CACHE:-$ABS/results/prior_cache}"
mkdir -p "$PRIOR_CACHE"
say "prior cache: $PRIOR_CACHE"
say "stages to run: $STAGES"
runs_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

say "--- STAGE 0: baselines on every available regime ---"
if runs_stage 0; then
  for rg in $BO_REGIMES $REG_REGIMES; do
    run_block stage0 "$rg" baseline || FAILED_STAGES=$((FAILED_STAGES + 1))
  done
else
  say "stage 0 skipped (not in STAGES)"
fi

say "--- STAGE 1: OFAT screen ---"
if runs_stage 1; then
  for rg in $BO_REGIMES $REG_REGIMES; do
    run_block stage1 "$rg" ofat || FAILED_STAGES=$((FAILED_STAGES + 1))
  done
else
  say "stage 1 skipped (not in STAGES)"
fi

# Stages 2 and 3 select their own cells from stage 1's summaries. Implemented as a
# generator call so no human decision is needed mid-run; see gen_stage_queue.py.
say "--- STAGE 2: crossed MEAN sweep on every regime ---"
# OFAT compares each mean variant only against `base` and never crosses the two mean
# axes, so it cannot answer "which mean is right" -- see S27.31, where every transfer was
# null on LCBench. LCBench is on-grid though, and S27.29 showed on-grid/off-grid is the
# organising axis, so this runs on ALL regimes.
if runs_stage 2; then
  for rg in $BO_REGIMES $REG_REGIMES; do
    run_block stage2means "$rg" means || FAILED_STAGES=$((FAILED_STAGES + 1))
  done
else
  say "stage 2 skipped (not in STAGES)"
fi

say "--- STAGE 3: replication of stage-1 winners, selected automatically ---"
# gen_stage_queue --kind replicate reads the previous stage's SUMMARY.json and picks its
# top configs, so no human decision is needed mid-run. It RAISES rather than falling back
# to a default list if the summary is missing or unrankable, which surfaces as a failed
# stage rather than a silently arbitrary experiment.
if runs_stage 3; then
  for rg in $BO_REGIMES $REG_REGIMES; do
    run_block stage3 "$rg" replicate "stage1_$rg" || FAILED_STAGES=$((FAILED_STAGES + 1))
  done
else
  say "stage 3 skipped (not in STAGES)"
fi

say "=== AUTONOMOUS RUN END ==="
say "summaries: $(find "$ABS/results/raw/v2" -name SUMMARY.json | wc -l)"
# systemd recorded success even when every stage failed (review N7).
if [ "$FAILED_STAGES" -gt 0 ]; then
  say "EXIT NON-ZERO: $FAILED_STAGES stage(s) incomplete; rerun to resume them"
  exit 1
fi
say "all stages complete"
