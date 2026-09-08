#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Decompose HOW the canonical kernel actually reaches the EM posterior.
#
# The on-grid PD1 result falsified the claim that the canonical kernel is a no-op on a
# shared grid: em_* moved by up to 0.73 regret while vanilla_gp, random, ablr and
# hyperbo_frozen were all bit-identical (0.000e+00), so it was not an RNG shift. The
# Lambda == 0 algebra only covers the INTERPOLATION role. The kernel has at least two
# other doors into the prior:
#
#   (1) SHRINKAGE TARGET. Sigma_alpha = (1-alpha) Sigma_ML + alpha * s * B with
#       B = K(Z,Z), the canonical kernel's own gram. Every run above used alpha=0.1, so
#       the kernel was injected into Sigma regardless of grid geometry.
#   (2) EM INITIALIZATION. init_mode="kernel" seeds Sigma^0 from the kernel, and EM at
#       finite iterations is only locally convergent, so a different init can settle on a
#       different fixed point.
#
# This matters beyond bookkeeping: the deep-kernel result (38% of the 7D NLL gap closed)
# was measured at alpha = 0.1-0.3, so part of that gain may come from a better SHRINKAGE
# TARGET rather than from better interpolation. Those are different claims and the paper
# should not conflate them.
#
# The 2 x 2 x 2 design isolates each door:
#   alpha=0, on-grid   -> interpolation OFF (Lambda==0), shrinkage OFF  => init only.
#                         PREDICTION: bit-identical across canonical kernels, or the
#                         difference is pure EM-init sensitivity.
#   alpha=0, off-grid  -> interpolation ON, shrinkage OFF => pure interpolation effect.
#   alpha=0.1, on-grid -> interpolation OFF, shrinkage ON => pure shrinkage-target effect.
#   alpha=0.1, off-grid-> both, i.e. the configuration all prior results used.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/canon_decomp" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp"

for a in 0.0 0.1; do
  for grid in ongrid offgrid; do
    for canon in sumll hyperbo deep; do
      out="$R/raw/canon_decomp/${grid}_a${a}_${canon}.json"
      [ -f "$out" ] && continue
      GRIDARG=""
      [ "$grid" = "offgrid" ] && GRIDARG="--n-inducing 100 --hetero-frac 0.25"
      KARG="--em-canonical sumll"
      [ "$canon" = "hyperbo" ] && KARG="--em-canonical hyperbo"
      [ "$canon" = "deep" ] && KARG="--deep-kernel 32,32"
      # shellcheck disable=SC2086
      python3 -m _scratch_bo.bo_diagnose \
        --n-configs 300 --n-pretrain 25 --n-eval 10 --n-obs-grid 5,20,50 \
        --em-shrinkage "$a" $GRIDARG $KARG \
        --meta-iters 25000 --threads 3 --methods "$METHODS" \
        --out "$out" > "$R/logs/canon_decomp_${grid}_a${a}_${canon}.txt" 2>&1 &
    done
    wait
  done
done
echo "[canon_decomp] done -> $R/raw/canon_decomp/"
