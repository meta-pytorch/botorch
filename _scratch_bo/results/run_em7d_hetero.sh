#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Does the Empirical GP win where it is actually designed to?
#
# Every benchmark in this stack so far -- LCBench and PD1 alike -- gives every historical
# task the SAME dense grid of configurations. That is the setting where a parametric deep
# kernel is strongest, and where the continuous-domain EM construction's distinctive
# machinery (per-task sparse observations resolved onto a shared reference set by the
# E-step, with a non-zero Nystrom residual Lambda) is idle. On a shared grid the canonical
# kernel is provably a no-op: Sigma(X) = Lambda(X) + W Sigma_Z W^T with Lambda == 0 and
# W == I, which is exactly why swapping in a deep canonical kernel changed on-grid BO
# results by precisely zero once the RNG confound was fixed.
#
# So we have been benchmarking EM on the baselines' home turf. This sweep moves to EM's:
# each task observes its OWN random subset of the pool (--hetero-frac), no two tasks share
# a grid, and evaluation is off-grid so Lambda > 0.
#
# The claim under test is NOT that EM wins at any single sparsity. It is that EM's
# position RELATIVE to HyperBO improves monotonically as observations get sparser and more
# irregular. hetero_frac=1.0 is the dense-grid control and should reproduce the standing
# result (HyperBO ahead); if the gap narrows or reverses as frac falls, that is the
# structural advantage the method claims. If it does not, the claim is not supported by
# this benchmark and we should say so.
#
# Both canonical kernels are run: off-grid is the only regime where the deep kernel can
# matter for EM at all, so this is also its fair test.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/em7d_hetero" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp"

for frac in 1.0 0.5 0.25 0.1; do
  for kern in plain deep; do
    out="$R/raw/em7d_hetero/f${frac}_${kern}.json"
    [ -f "$out" ] && continue
    extra=""
    [ "$kern" = "deep" ] && extra="--deep-kernel 32,32"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_diagnose \
      --n-configs 300 --n-inducing 100 --hetero-frac "$frac" \
      --n-pretrain 25 --n-eval 10 --n-obs-grid 5,10,20,50 \
      --em-shrinkage 0.1 --meta-iters 25000 --threads 5 \
      --methods "$METHODS" $extra \
      --out "$out" > "$R/logs/em7d_hetero_f${frac}_${kern}.txt" 2>&1 &
  done
  wait
done
echo "[em7d_hetero] done -> $R/raw/em7d_hetero/"
