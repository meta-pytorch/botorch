#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tune EM's shrinkage to CALIBRATION rather than to a fixed constant.
#
# The M x cond-noise sweep refuted both structural hypotheses and relocated the defect:
# with alpha=0.1 at M=200 the model is no longer overconfident but OVER-dispersed
# (sigma/RMSE = 1.20, coverage 0.98) while HyperBO sits at sigma/RMSE = 0.89 with
# coverage 0.91 and a much better NLL. RMSE is essentially tied (0.197 vs 0.191), so the
# residual NLL gap is entirely a predictive-variance calibration gap, and alpha is the
# knob that sets it.
#
# alpha=0.1 was selected at M=300/n=100, where it produced sigma/RMSE = 0.64
# (under-dispersed). The same alpha at M=200/n=40 gives 1.20 (over-dispersed). So the
# optimum is not a constant: it depends on M and on the target-task budget n. This sweep
# takes a fine alpha grid at the best M and reads off, per n, the alpha that drives
# sigma/RMSE toward 1 -- and whether that alpha also minimizes NLL, which is the actual
# claim worth making.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_calib" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen"

for a in 0.0 0.01 0.02 0.05 0.1 0.2; do
  out="$R/raw/em7d_calib/alpha_${a}.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_diagnose \
    --n-configs 300 --n-pretrain 30 --n-eval 5 --n-obs-grid 5,20,50,100 \
    --em-shrinkage "$a" --meta-iters 25000 --threads 5 --methods "$METHODS" \
    --out "$out" > "$R/logs/em7d_calib_alpha_${a}.txt" 2>&1 &
done
wait
echo "[em7d_calib] done -> $R/raw/em7d_calib/"
