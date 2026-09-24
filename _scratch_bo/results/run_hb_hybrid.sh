#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Full HyperBO-kernel hybrid cross-product, on both benchmarks and both metrics.
#
# The pre-trained HyperBO kernel reaches the EM posterior through three INDEPENDENT
# doors, and the PD1 on-grid falsification showed we cannot assume any of them is inert:
#
#   canonical  (--em-canonical hyperbo)   drives W, the Nystrom residual Lambda, the
#                                         shrinkage target B = K(Z,Z), and EM init.
#   additive, weight fitted               k_emp + s * k_hb, with s fitted on the target.
#                                         Exactly ONE trainable scalar, because
#                                         hyperbo_kernel_from_prior freezes the internals
#                                         -- unlike em_additive_freshbase, which fits a
#                                         whole ARD Matern and hits NLL 54.8 at n=5.
#   additive, frozen                      k_emp + k_hb with s pinned at 1. Separates "is
#                                         the representation useful" from "does adapting
#                                         its weight help".
#
# The frozen arm is the control that makes the fitted arm interpretable; a smoke test hinted
# the fitted weight helps (0.76 vs 0.85 baseline) while the frozen addition hurts (0.94),
# which is the signature of a useful-but-mis-scaled prior. Worth confirming at scale.
#
# Sized for the idle box: 96 cores, load ~8, 133 GB free. 12 jobs x 6 threads ~= 72 cores,
# leaving headroom for the three sweeps already running.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/hb_hybrid" "$R/logs"

EM_ADD="em_frozen,em_noisefit,em_additive_hyperbo,em_additive_hyperbo_frozen"
REG_METHODS="$EM_ADD,em_additive_warmbase,hyperbo_frozen,vanilla_gp"
BO_METHODS="$EM_ADD,hyperbo_frozen,ablr,vanilla_gp,random"

# ---- 1. LCBench regression, on- and off-grid x canonical ---------------------------
for canon in sumll hyperbo; do
  for grid in ongrid offgrid; do
    out="$R/raw/hb_hybrid/reg_${grid}_${canon}.json"
    [ -f "$out" ] && continue
    GRIDARG=""
    [ "$grid" = "offgrid" ] && GRIDARG="--n-inducing 100 --hetero-frac 0.25"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-pretrain 25 --n-eval 10 --n-obs-grid 5,20,50 \
      --em-shrinkage 0.1 --em-canonical "$canon" $GRIDARG \
      --meta-iters 25000 --threads 6 --methods "$REG_METHODS" \
      --out "$out" > "$R/logs/hb_reg_${grid}_${canon}.txt" 2>&1 &
  done
done
wait

# ---- 2. LCBench BO x canonical x prior ---------------------------------------------
for canon in sumll hyperbo; do
  for ps in 0 1; do
    out="$R/raw/hb_hybrid/bo_lcb_${canon}_p${ps}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment \
      --n-configs 200 --n-iters 40 --n-seeds 6 --pretrain-seed "$ps" \
      --em-shrinkage 0.1 --em-canonical "$canon" \
      --hyperbo-iters 25000 --ablr-iters 25000 --threads 6 \
      --methods "$BO_METHODS" \
      --out "$out" > "$R/logs/hb_bo_lcb_${canon}_p${ps}.txt" 2>&1 &
  done
done
wait

# ---- 3. PD1 BO x canonical x pool (full pool is NATURALLY off-grid) ----------------
for canon in sumll hyperbo; do
  for pool in matched full; do
    out="$R/raw/hb_hybrid/bo_pd1_${pool}_${canon}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment \
      --benchmark pd1_loo \
      --pd1-source full --pd1-candidate-pool "$pool" --pd1-pretrain-pool "$pool" \
      --n-iters 30 --n-seeds 4 --em-shrinkage 0.1 --em-canonical "$canon" \
      --hyperbo-iters 25000 --ablr-iters 25000 --threads 6 \
      --methods "$BO_METHODS" \
      --out "$out" > "$R/logs/hb_bo_pd1_${pool}_${canon}.txt" 2>&1 &
  done
done
wait
echo "[hb_hybrid] done -> $R/raw/hb_hybrid/"
