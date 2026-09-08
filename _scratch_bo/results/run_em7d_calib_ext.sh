#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Extend the shrinkage grid past 0.2 at the data-rich end.
#
# The calibration sweep found the optimal alpha rising monotonically with the target
# budget (0.0 at n=5, 0.1 at n=20, 0.2 at n>=50) and hitting the grid ceiling: at
# alpha=0.2 the model is still overconfident at n=50-100 (sigma/RMSE 0.51-0.64 against
# HyperBO's 0.89), so alpha=0.2 is a boundary solution, not an optimum. A coarser earlier
# ablation saw alpha=0.3 reach -0.398 at n=100, better than anything in that grid.
#
# This extends alpha to {0.3, 0.5, 0.7} on the same M=300 setup so the optimum is
# interior and the "alpha grows with n" rule can be stated with an actual argmin rather
# than a grid edge.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_calib" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen"

for a in 0.3 0.5 0.7; do
  out="$R/raw/em7d_calib/alpha_${a}.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_diagnose \
    --n-configs 300 --n-pretrain 30 --n-eval 5 --n-obs-grid 5,20,50,100 \
    --em-shrinkage "$a" --meta-iters 25000 --threads 6 --methods "$METHODS" \
    --out "$out" > "$R/logs/em7d_calib_alpha_${a}.txt" 2>&1 &
done
wait
echo "[em7d_calib_ext] done -> $R/raw/em7d_calib/"
