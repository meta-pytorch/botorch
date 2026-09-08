#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Does giving the EM prior a learned kernel metric close the gap to HyperBO?
#
# Diagnosis chain that led here. Three explanations for the residual 7D NLL gap were
# tested and refuted: (a) a rank ceiling -- shrinkage adds a full-rank base gram and the
# additive base is full rank at the query points, and the M=30 arm gives a nearly
# full-rank Sigma yet performs worst; (b) variance SCALE -- giving both models an optimal
# global variance rescale WIDENS the gap (+0.60 -> +0.74 nats), so no post-hoc
# calibration can fix it; (c) off-grid capacity -- the first test of this was confounded
# and is re-run separately. What survived is variance SHAPE: with optimal rescaling the
# residual lives in mean(log sigma_i), and RMSE accounts for only ~0.12 of 0.74 nats.
# HyperBO's Matern acts on a learned 32-d feature map; the EM canonical kernel had 7 ARD
# lengthscales. The canonical kernel drives W, Lambda and the additive base, so it is
# exactly the object that sets variance shape.
#
# The deep kernel uses a skip connection, phi(x) = [x, MLP(x)], so it strictly
# GENERALIZES the plain ARD Matern -- if the learned features are useless their ARD
# lengthscales grow and it reduces to the baseline. It should not be able to lose at the
# optimum.
#
# A 2-dataset smoke test gave em_noisefit NLL -0.744 / RMSE 0.143 against the plain
# kernel's -0.233 / 0.195, with an RMSE better than HyperBO's 0.173 -- but at n_eval=2
# and an under-trained HyperBO. This is the honest comparison: 5 eval datasets, HyperBO
# at its converged 25k steps, deep vs plain across the alpha grid.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_deep" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen"

for a in 0.0 0.1 0.3; do
  # deep canonical kernel
  out="$R/raw/em7d_deep/deep_a${a}.json"
  if [ ! -f "$out" ]; then
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-pretrain 30 --n-eval 5 --n-obs-grid 5,20,50,100 \
      --em-shrinkage "$a" --deep-kernel 32,32 \
      --meta-iters 25000 --threads 5 --methods "$METHODS" \
      --out "$out" > "$R/logs/em7d_deep_a${a}.txt" 2>&1 &
  fi
  # matched plain-kernel control, same seed and settings
  out2="$R/raw/em7d_deep/plain_a${a}.json"
  if [ ! -f "$out2" ]; then
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-pretrain 30 --n-eval 5 --n-obs-grid 5,20,50,100 \
      --em-shrinkage "$a" \
      --meta-iters 25000 --threads 5 --methods "$METHODS" \
      --out "$out2" > "$R/logs/em7d_plain_a${a}.txt" 2>&1 &
  fi
done
wait
echo "[em7d_deep] done -> $R/raw/em7d_deep/"
