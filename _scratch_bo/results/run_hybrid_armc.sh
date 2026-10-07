#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Arm C x the HyperBO additive hybrid -- the one PD1 configuration never run (S27.5).
#
# WHY THIS EXISTS, AND WHAT IT CAN AND CANNOT SHOW
#
# S27.6 established that PD1 BO cannot resolve any transfer-method contrast on ACCURACY
# at any budget: the SEM floor at infinite priors already exceeds what z=2 needs, because
# floor = sigma_between/sqrt(T) and PD1 has T=23 tasks. So this sweep is deliberately NOT
# run to produce a significance claim, and any "tied" it yields describes PD1's task
# count rather than the methods.
#
# It is run for the two things that ARE well determined here:
#
#   1. COST. Pre-training and BO wall-clock carry no sampling noise, so the
#      performance-vs-cost Pareto (S26.2, the project's most durable claim) extends to
#      the hybrid with no power caveat at all.
#   2. PER-TASK STRUCTURE. S27.6 found the between-task sd of the paired difference is
#      0.040 against a mean effect of 0.002 -- per-task effects are ~20x the average, in
#      both directions. Point estimates with honest, wide CIs plus the per-task
#      breakdown are the useful output.
#
# MERGEABILITY. Verified before launching: em_frozen is bit-identical across three
# separate processes with different --methods lists, so these results pair with the
# existing arm-C data by (task, seed, prior) and the shared arms do not need re-running.
# em_frozen and hyperbo_frozen are carried here purely as merge-validation controls --
# they MUST reproduce results/raw/pd1pool/fullboth*.json exactly. If they do not, the
# merge is invalid and the hybrid numbers must not be compared against S27's.
#
# PACOH and ABLR are dropped: they are not needed for the hybrid and PACOH alone costs
# ~12616 s/cell.
#
# Config matches the existing arm C exactly so the merge is valid: 50 iters, n_init 3,
# per-task standardization, neglog warp, 25000 HyperBO steps, full/full pools, and
# n_seeds 5 for prior 0 but 3 for priors 1 and 2 (that asymmetry is in the existing data).

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
R="${RESULTS_DIR:-$S/results}"
SHARDS="${SHARDS:-12}"
THREADS="${THREADS:-3}"
mkdir -p "$R/raw/pd1pool" "$R/logs"

METHODS="em_additive_hyperbo,em_additive_hyperbo_frozen,em_frozen,hyperbo_frozen"
COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --pd1-candidate-pool full --pd1-pretrain-pool full \
  --n-iters 50 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-iters 25000 --threads $THREADS --methods $METHODS"

for ps in 0 1 2; do
  # Prior 0 of the existing arm C used 5 seeds, priors 1 and 2 used 3. Match it.
  if [ "$ps" = "0" ]; then NSEEDS=5; else NSEEDS=3; fi
  echo "[hybrid_armc] prior $ps, $NSEEDS seeds, $SHARDS shards"
  for i in $(seq 0 $((SHARDS - 1))); do
    out="$R/raw/pd1pool/hybrid_p${ps}_s${i}.json"
    [ -f "$out" ] && continue
    printf '%s\t%s\t%s\n' "hybrid_p${ps}_s${i}" "$(date -u +%FT%TZ)" \
      "bo_experiment $COMMON --n-seeds $NSEEDS --pretrain-seed $ps" \
      >> "$R/MANIFEST.tsv"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment $COMMON \
      --n-seeds "$NSEEDS" --pretrain-seed "$ps" \
      --loo-shard "${i}/${SHARDS}" --out "$out" \
      > "$R/logs/hybrid_p${ps}_s${i}.txt" 2>&1 &
  done
  # Priors run sequentially: 12 shards x 3 threads already saturates the box.
  wait
  echo "[hybrid_armc] prior $ps done"
done

echo "[hybrid_armc] all done -> $R/raw/pd1pool/hybrid_p*.json"
