#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Faithful PD1 reproduction on the FULL public release (Wang et al. 2021).
#
# Addresses every HIGH-impact deviation catalogued in EXPERIMENTAL_RESULTS.md 12.4:
#   * full PD1 instead of PD1Lite -> 400 matched configs shared by all 23 tasks (was 50)
#   * leave-one-out over all 23 tasks, no batch-size subgroup needed
#   * the paper's -log(error) output warp, plus a no-warp control (12.2 found no-warp
#     was HyperBO's best on the tiny PD1Lite pool -- retested here at proper scale)
#   * 50000 HyperBO pre-training steps, per the paper, with PACOH and ABLR given their
#     own generous budgets rather than being forced onto a shared --meta-iters
#
# Leave-one-out re-pre-trains every prior per fold, which is ~10 h sequential, so folds
# are sharded round-robin across processes and recombined by analyze.py.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/

SCRATCH="pytorch/botorch/_scratch_bo"
R="${RESULTS_DIR:-$SCRATCH/results}"
SHARDS="${SHARDS:-12}"
THREADS="${THREADS:-3}"
mkdir -p "$R/raw/pd1full" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,vanilla_gp,random"
# HYPERBO_LOSS=EKL reproduces the paper's strongest variant, H-EKL. The LOO pool is
# matched (400 shared configs), which is exactly the setting EKL requires and the one
# Wang et al. report it winning in. EKL is optimized with L-BFGS per their setup, not
# Adam -- optimizing EKL with Adam under-trains it.
HYPERBO_LOSS="${HYPERBO_LOSS:-NLL}"
SUFFIX=""
[ "$HYPERBO_LOSS" = "EKL" ] && SUFFIX="_ekl"
COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --n-iters 50 --n-seeds 5 --n-init 3 --per-task-standardize \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-loss $HYPERBO_LOSS --ekl-optimizer lbfgs --ekl-lbfgs-iters 100 \
  --hyperbo-iters 50000 --pacoh-iters 10000 --ablr-iters 10000 --pacoh-particles 10 \
  --threads $THREADS --methods $METHODS"

last=$((SHARDS - 1))
for warp in "$@"; do
  echo "[pd1full] warp=$warp, loss=$HYPERBO_LOSS, $SHARDS shards"
  for i in $(seq 0 $last); do
    out="$R/raw/pd1full/pd1_loo_${warp}${SUFFIX}_s${i}.json"
    log="$R/logs/pd1_loo_${warp}${SUFFIX}_s${i}.txt"
    printf '%s\t%s\t%s\n' "pd1_loo_${warp}${SUFFIX}_s${i}" "$(date -u +%FT%TZ)" \
      "bo_experiment $COMMON --output-warp ${warp} --loo-shard ${i}/${SHARDS}" \
      >> "$R/MANIFEST.tsv"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment $COMMON \
      --output-warp "$warp" --loo-shard "${i}/${SHARDS}" --out "$out" > "$log" 2>&1 &
  done
done
wait
echo "[pd1full] done; merge shards with:"
echo "  analyze --result \"\$(ls $R/raw/pd1full/pd1_loo_neglog_s*.json | paste -sd,)\""
