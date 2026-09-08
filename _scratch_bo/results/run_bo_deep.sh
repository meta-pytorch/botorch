#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Does the learned kernel metric that helped 7D regression also help BO?
#
# Standing BO result (LCBench, 8 independent pre-training priors, regret-AUC):
#     ablr 0.537 +- 0.026 | hyperbo_adapt 0.601 | ablr_adapt 0.602 |
#     hyperbo_frozen 0.610 | em_frozen 0.639 | em_noisefit 0.671 |
#     em_finetuned 0.786 | vanilla_gp 1.250 | pacoh 1.559 | random 1.926
# EM is 5th of 10 -- clearly in the transfer tier and far ahead of vanilla/PACOH/random,
# but behind ABLR and both HyperBO variants. ABLR's lead over em_frozen is ~4x ABLR's own
# prior-draw sd, so it is real and not a seeding artifact.
#
# Two observations motivate this run:
#   * ABLR is a shared MLP feature map plus a Bayesian linear head -- essentially "learned
#     metric + analytic posterior". EM with a deep canonical kernel is the closest
#     structural analogue we can build, so if metric learning is what ABLR's lead buys,
#     this should capture most of it.
#   * In BO the SIMPLEST EM variant wins (em_frozen 0.639 < em_noisefit 0.671 <
#     em_finetuned 0.786), the reverse of the regression ordering. That is consistent with
#     the shrinkage sweep: BO spends most of its budget at low n, where alpha=0 and a
#     frozen prior are optimal. So the deep kernel is tested against em_frozen first.
#
# Caveat worth stating: the deep kernel helped regression at LARGE n and was neutral or
# harmful at n<=20. BO lives at low n, so this may not transfer. That is the question.
#
# EM pre-training is deterministic given the data, so EM's prior-draw sd is exactly 0 --
# a single prior suffices for the EM arms. The neural baselines are re-seeded across the
# 8 headline priors already committed, so no re-run of those is needed here.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/bo_deep" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,ablr,vanilla_gp"

for kern in plain deep; do
  for ps in 0 1; do
    out="$R/raw/bo_deep/${kern}_p${ps}.json"
    [ -f "$out" ] && continue
    extra=""
    [ "$kern" = "deep" ] && extra="--deep-kernel 32,32"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment \
      --n-configs 200 --n-iters 40 --n-seeds 6 --pretrain-seed "$ps" \
      --hyperbo-iters 25000 --ablr-iters 25000 --threads 6 \
      --methods "$METHODS" $extra \
      --out "$out" > "$R/logs/bo_deep_${kern}_p${ps}.txt" 2>&1 &
  done
done
wait
echo "[bo_deep] done -> $R/raw/bo_deep/"
