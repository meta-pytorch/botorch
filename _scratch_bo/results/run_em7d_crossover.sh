#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# WHERE exactly does the Empirical GP stop beating HyperBO?
#
# This is the experiment that decides whether the paper has a story. Meta-learned priors
# earn their keep where evaluations are expensive, so the low-sample regime is the one
# that matters. So far we know only that EM wins at n=5 and loses at n=20 -- a 4x range,
# far too coarse to claim anything.
#
# Design choices that matter:
#   * Dense low-n grid (1..30) and a sparse tail (50, 100) for context. The interesting
#     structure is all below 30.
#   * n_eval raised 5 -> 10 (n_pretrain 30 -> 25) to halve the eval-side standard error.
#     Locating a crossover needs precision in the SCORE, not in the prior.
#   * bo_diagnose now emits per_dataset_nll / per_dataset_rmse, so the crossover can be
#     reported with a dataset-clustered SEM and a paired EM-vs-HyperBO test rather than a
#     bare mean. Runs on one dataset share a historical corpus and are not independent.
#   * Both canonical kernels are run. The deep kernel wins at n=100 but LOSES at n=20, so
#     which one is better in the low-sample regime is an open question, not an assumption.
#   * alpha in {0, 0.1}: the calibration sweep found the optimum is 0 at n=5 and 0.1 at
#     n=20, so the low-n optimum lies in that range. alpha=0.3 only wins at n>=50.
#
# HyperBO runs at its converged 25k steps in every arm; under-training it was the original
# S19 error and would manufacture a crossover that does not exist.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_crossover" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp"
GRID="1,2,3,4,5,6,8,10,12,15,20,25,30,50,100"

for a in 0.0 0.1; do
  for kern in plain deep; do
    out="$R/raw/em7d_crossover/${kern}_a${a}.json"
    [ -f "$out" ] && continue
    extra=""
    [ "$kern" = "deep" ] && extra="--deep-kernel 32,32"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-pretrain 25 --n-eval 10 \
      --n-obs-grid "$GRID" --em-shrinkage "$a" $extra \
      --meta-iters 25000 --threads 6 --methods "$METHODS" \
      --out "$out" > "$R/logs/em7d_crossover_${kern}_a${a}.txt" 2>&1 &
  done
done
wait
echo "[em7d_crossover] done -> $R/raw/em7d_crossover/"
