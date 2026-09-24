#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Can the EM model's 7D regression NLL be improved? Ablate the two levers we have --
# M-step shrinkage and the conditioning-time additive base -- across the full EM
# variant set.
#
# Motivating observation from the refreshed 7D run: em_frozen's NLL EXPLODES with data
# (1.5 -> 6.9 -> 14.1 at n_train 20/50/100) while its RMSE stays fine (0.33 -> 0.20).
# That is pure overconfidence, and it is exactly what shrinkage repairs (S16: coverage
# 0.65 -> 0.98, NLL 8.30 -> 0.10). em_finetuned, which already carries a fitted additive
# base, stays bounded at -0.145. So both levers target the same defect by different
# routes, and the question is which wins and whether they compose.
#
# HyperBO and PACOH are excluded: they are unaffected by these knobs and their 25k-step
# pre-training would dominate the runtime. vanilla_gp is kept as a cheap reference.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_ablate" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,em_additive_freshbase,em_additive_warmbase,em_additive_3way,vanilla_gp"

for a in 0.0 0.05 0.1 0.3 0.5; do
  out="$R/raw/em7d_ablate/em7d_a${a}.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_diagnose \
    --n-configs 300 --n-pretrain 30 --n-eval 5 \
    --n-obs-grid 5,10,20,50,100 \
    --em-shrinkage "$a" --meta-iters 2000 --threads 6 --methods "$METHODS" \
    --out "$out" > "$R/logs/em7d_a${a}.txt" 2>&1 &
done
wait
echo "[em7d] done -> $R/raw/em7d_ablate/"
