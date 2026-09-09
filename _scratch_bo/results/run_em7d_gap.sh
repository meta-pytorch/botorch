#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Why can't the Empirical GP close the 7D NLL gap to HyperBO, and can it?
#
# At n_train=100 the tuned EM model (em_finetuned, alpha=0.1) reaches NLL -0.417 against
# HyperBO's -0.995, and loses on every axis: RMSE 0.198 vs 0.176, coverage 0.90 vs 0.96,
# sigma/RMSE 0.64 vs 0.89. The residual defect is OVERCONFIDENCE, not inaccuracy -- the
# RMSE gap is small, the variance gap is not. Two structural causes are testable:
#
#   H1  RANK vs DIMENSION. Sigma has rank <= K-1 = 29 in an M-dimensional space
#       (measured effective rank 2.9 at M=200). The prior puts almost no variance in the
#       null space, but a new task has components there. Trace-matched shrinkage
#       REDISTRIBUTES variance rather than adding it, so it can only partly repair this.
#       If H1 binds, reducing M toward K should sharply improve NLL and coverage, and
#       increasing K at fixed M should do the same. HyperBO has no analogue of this
#       limit: its deep kernel is full-rank by construction.
#
#   H2  CONDITIONING NOISE. COND_NOISE defaults to 1e-3 in standardized units, i.e.
#       essentially noiseless. Conditioning near-noiselessly on genuinely noisy targets
#       collapses the posterior variance -- exactly the observed signature. Newly exposed
#       as --cond-noise.
#
# Wave 1 sweeps M x cond_noise; wave 2 sweeps K at fixed M to separate "few tasks" from
# "too many inducing points", since both change the rank-to-dimension ratio.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_gap" "$R/logs"

# hyperbo_frozen is carried in every cell as the reference we are chasing.
METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp"

# ---- wave 1: inducing-point count x conditioning noise (alpha fixed at its 7D best)
for M in 50 100 200 300; do
  for cn in 0.001 0.01 0.05 0.2; do
    out="$R/raw/em7d_gap/M${M}_cn${cn}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs "$M" --n-pretrain 30 --n-eval 5 \
      --n-obs-grid 5,20,40 --em-shrinkage 0.1 --cond-noise "$cn" \
      --meta-iters 25000 --threads 4 --methods "$METHODS" \
      --out "$out" > "$R/logs/em7d_gap_M${M}_cn${cn}.txt" 2>&1 &
  done
  wait   # bound concurrency to one M-block at a time
done

# ---- wave 2: number of pre-training tasks K, at the paper's M=100
for K in 10 20 30; do
  out="$R/raw/em7d_gap/K${K}_M100.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_diagnose \
    --n-configs 100 --n-pretrain "$K" --n-eval 5 \
    --n-obs-grid 5,20,40 --em-shrinkage 0.1 --cond-noise 0.01 \
    --meta-iters 25000 --threads 5 --methods "$METHODS" \
    --out "$out" > "$R/logs/em7d_gap_K${K}.txt" 2>&1 &
done
wait
echo "[em7d_gap] done -> $R/raw/em7d_gap/"
