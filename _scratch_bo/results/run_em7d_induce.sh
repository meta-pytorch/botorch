#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Is the Empirical GP actually a universal approximator here? Decisive test.
#
# The continuous-domain EGP's interpolated covariance is
#     Sigma(X) = Lambda(X) + W Sigma_Z W^T,   Lambda(X) = K(X,X) - W K(Z,X),
# where Lambda is the canonical kernel's Nystrom residual and is what gives the model
# full-rank, universal capacity away from the reference set. Every previous 7D run
# anchored Z on the ENTIRE candidate pool, so every test point satisfied X in Z and
# Lambda was identically zero -- reducing the model to the rank-(K-1) = rank-29 empirical
# covariance in a 300-dimensional test space. HyperBO's Matern kernel has no such ceiling,
# which is a complete explanation for why EM's advantage evaporates as n_train grows.
#
# The earlier M sweep could not detect this: --n-configs moved the pool and the reference
# set together, so it never left the on-grid regime.
#
# Here the pool is FIXED at 300 and only the reference set varies. n_inducing=300 is the
# old on-grid behaviour (Lambda == 0) and is the control; everything below it is off-grid
# (Lambda > 0). If the rank ceiling is the binding constraint, the off-grid arms should
# improve markedly at large n_train, where the ceiling bites hardest -- and should give up
# a little at small n, where a coarser reference set costs resolution.
#
# alpha is held at its per-budget optimum from the calibration sweep rather than fixed,
# so the comparison is not confounded by shrinkage.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_induce" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen"

for M in 30 60 100 150 300; do
  for a in 0.0 0.1 0.3; do
    out="$R/raw/em7d_induce/M${M}_a${a}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-inducing "$M" --n-pretrain 30 --n-eval 5 \
      --n-obs-grid 5,20,50,100 --em-shrinkage "$a" \
      --meta-iters 25000 --threads 4 --methods "$METHODS" \
      --out "$out" > "$R/logs/em7d_induce_M${M}_a${a}.txt" 2>&1 &
  done
  wait   # one M-block at a time, to bound concurrency
done
echo "[em7d_induce] done -> $R/raw/em7d_induce/"
