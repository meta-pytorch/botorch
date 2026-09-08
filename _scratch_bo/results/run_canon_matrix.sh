#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The complete canonical-kernel matrix: {sumll, hyperbo} x {on-grid, off-grid} x
# {LCBench, PD1}, plus PD1's NATURAL heterogeneity.
#
# The hybrid under test reuses HyperBO's PRE-TRAINED deep kernel (25k steps against its
# own NLL/EKL objective) as the EM model's canonical/interpolation kernel -- HyperBO's
# learned metric driving W and the Nystrom residual Lambda, with EM's closed-form
# empirical mean and covariance on top.
#
# PREDICTION, stated up front so the run can falsify it: the ON-GRID arms should be
# EXACTLY no-ops. Sigma(X) = Lambda(X) + W Sigma_Z W^T, and when evaluation points lie in
# the reference set Lambda == 0 and W == I, so the canonical kernel cannot touch the EM
# posterior. The on-grid arms are included deliberately as an empirical check of that
# proof -- a non-zero difference there would mean the argument, or the implementation, is
# wrong, and that is worth catching.
#
# The off-grid arms are where the hybrid can actually pay. PD1 is the most interesting
# case because its heterogeneity is REAL rather than synthetic: each task carries ~2040
# configs of which only ~400 are shared, so --pd1-candidate-pool full is genuinely
# off-grid without any subsetting on our part.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/canon_matrix" "$R/logs"

MET_REG="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp"

# ---- LCBench regression: on-grid (control, expect exact no-op) and off-grid ----------
for canon in sumll hyperbo; do
  # ON-grid: inducing set == pool
  out="$R/raw/canon_matrix/lcb_ongrid_${canon}.json"
  if [ ! -f "$out" ]; then
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-pretrain 25 --n-eval 10 --n-obs-grid 5,20,50 \
      --em-shrinkage 0.1 --em-canonical "$canon" \
      --meta-iters 25000 --threads 4 --methods "$MET_REG" \
      --out "$out" > "$R/logs/canon_lcb_ongrid_${canon}.txt" 2>&1 &
  fi
  # OFF-grid, heterogeneous: the regime the construction is built for
  out2="$R/raw/canon_matrix/lcb_offgrid_${canon}.json"
  if [ ! -f "$out2" ]; then
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-inducing 100 --hetero-frac 0.25 \
      --n-pretrain 25 --n-eval 10 --n-obs-grid 5,20,50 \
      --em-shrinkage 0.1 --em-canonical "$canon" \
      --meta-iters 25000 --threads 4 --methods "$MET_REG" \
      --out "$out2" > "$R/logs/canon_lcb_offgrid_${canon}.txt" 2>&1 &
  fi
done
wait

# ---- PD1 BO: matched pool (on-grid) vs full pool (naturally off-grid) ----------------
MET_BO="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,ablr,vanilla_gp,random"
for canon in sumll hyperbo; do
  for pool in matched full; do
    out="$R/raw/canon_matrix/pd1_${pool}_${canon}.json"
    [ -f "$out" ] && continue
    python3 -m _scratch_bo.bo_experiment \
      --benchmark pd1_loo \
      --pd1-source full --pd1-candidate-pool "$pool" --pd1-pretrain-pool "$pool" \
      --n-iters 30 --n-seeds 4 --em-shrinkage 0.1 --em-canonical "$canon" \
      --hyperbo-iters 25000 --ablr-iters 25000 --threads 5 --methods "$MET_BO" \
      --out "$out" > "$R/logs/canon_pd1_${pool}_${canon}.txt" 2>&1 &
  done
  wait
done
echo "[canon_matrix] done -> $R/raw/canon_matrix/"
