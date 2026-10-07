#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Unattended overnight driver: watch the in-flight PD1 pool ablation, relaunch any shard
# that dies, then run the follow-up wave.
#
# Launched via `systemd-run --user --slice=user.slice` so it lives OUTSIDE <internal-agent-tool>.slice.
# The jobs it supervises were started from a Devmate tool call and therefore sit in a
# per-call transient scope; if that slice is torn down with the agent they would die, and
# nothing else would notice. This script does.
#
# Everything is idempotent: work is keyed on the existence of its output JSON, so a
# relaunch only ever redoes shards that are actually missing.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
FBCODE="$ROOT"
cd "$FBCODE" || exit 1
SCRATCH="pytorch/botorch/_scratch_bo"
R="$SCRATCH/results"
BO="python3 -m _scratch_bo.bo_experiment"
LOG="$R/logs/overnight_driver.txt"
mkdir -p "$R/raw/pd1pool" "$R/raw/followup" "$R/logs"
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "driver start (pid $$)"

# ---------------------------------------------------------------- 1. supervise pd1pool
PD1_COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --n-iters 50 --n-seeds 5 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-iters 25000 --pacoh-iters 10000 --ablr-iters 10000 --ablr-per-task-precision \
  --pacoh-particles 10 --threads 3 \
  --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,vanilla_gp,random"

for _ in $(seq 1 240); do          # up to ~8 h of supervision
  missing=0
  for arm_spec in "fullcand full matched" "fullboth full full"; do
    # shellcheck disable=SC2086
    set -- $arm_spec
    arm=$1 cand=$2 pre=$3
    for i in $(seq 0 11); do
      out="$R/raw/pd1pool/${arm}_s${i}.json"
      [ -f "$out" ] && continue
      missing=$((missing + 1))
      # Relaunch only if no live process is already working this shard.
      if ! pgrep -f "pd1pool/${arm}_s${i}.json" > /dev/null 2>&1; then
        say "relaunching ${arm}_s${i}"
        # shellcheck disable=SC2086
        $BO $PD1_COMMON --pd1-candidate-pool "$cand" --pd1-pretrain-pool "$pre" \
          --loo-shard "${i}/12" --out "$out" \
          > "$R/logs/pd1pool_${arm}_s${i}.txt" 2>&1 &
      fi
    done
  done
  [ "$missing" -eq 0 ] && { say "pd1pool complete"; break; }
  sleep 120
done
wait

# ------------------------------------------------------- 2. follow-up wave (LCBench)
say "follow-up wave start"
LC="--n-configs 300 --n-iters 40 --n-eval 12 --n-seeds 16 --n-pretrain 25 \
  --split-seed 42 --meta-subsample 64 --meta-task-batch 5 --meta-iters 2000 \
  --hyperbo-iters 25000 --pacoh-iters 10000 --ablr-iters 10000 --ablr-per-task-precision \
  --threads 5"

run() { # run <name> <args...>
  local n="$1"; shift
  [ -f "$R/raw/followup/$n.json" ] && return
  say "  $n"
  # shellcheck disable=SC2086
  $BO "$@" --out "$R/raw/followup/$n.json" > "$R/logs/followup_$n.txt" 2>&1 &
}

# (a) em_additive_warmbase: the one variant that warm-starts its additive base from the
#     PRE-TRAINED canonical kernel rather than a fresh one. Never run before, and only
#     meaningful now that the additive hook actually takes effect (§21.1).
for s in 0 1 2; do
  # shellcheck disable=SC2086
  run "warmbase_p$s" $LC --pretrain-seed "$s" \
    --methods em_frozen,em_finetuned,em_additive_freshbase,em_additive_warmbase,em_additive_3way,ablr,random
done
wait

# (b) More pre-training priors for the headline. PACOH's prior-draw sd is 0.248 -- larger
#     than most between-method gaps -- so 3 priors cannot rank it. Extend 3 -> 8.
for s in 3 4 5 6 7; do
  # shellcheck disable=SC2086
  run "headline_p$s" $LC --pretrain-seed "$s" \
    --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,ablr_adapt,vanilla_gp,random
done
wait

# (c) Shrinkage with an OFF-GRID candidate pool. §16 found shrinkage hurts in-distribution
#     and helps out-of-distribution, but that was measured on-grid where the Nystrom
#     residual Lambda(X) vanishes. Off-grid it does not, so the trade-off may move.
for a in 0.0 0.1 0.3; do
  # shellcheck disable=SC2086
  run "pd1_shrink_full_a$a" --benchmark pd1_loo --pd1-source full --pd1-group all \
    --pd1-candidate-pool full --pd1-pretrain-pool matched --em-shrinkage "$a" \
    --n-iters 50 --n-seeds 3 --n-init 3 --per-task-standardize --output-warp neglog \
    --meta-subsample 64 --meta-task-batch 5 --hyperbo-iters 25000 --threads 6 \
    --loo-shard 0/4 \
    --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp,random
done
wait

say "ALL DONE"
