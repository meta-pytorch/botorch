#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The definitive LCBench BO measurement: 5 dataset splits x 2 pre-training priors.
#
# Why this run exists. Every method-level conclusion in this stack has turned out to sit
# at or below the measurement noise:
#   * the "EM wins at n=5" regression result reversed when eval datasets went 5 -> 10;
#   * HyperBO's n=100 NLL swung 0.8 nats from an n_pretrain 30 -> 25 change plus a
#     different split;
#   * the BO deep-vs-plain arms moved control methods (which do not use the canonical
#     kernel at all) by ~0.1 regret-AUC purely through a shifted global RNG stream.
# The last of those is now fixed -- each neural baseline re-seeds immediately before its
# own pre-training, verified by control arms being bit-identical across --deep-kernel.
#
# Varying --split-seed varies BOTH the config subsample and the pretrain/eval dataset
# split, which is the dominant noise source. Varying --pretrain-seed varies the neural
# baselines' prior draw; EM is unaffected because its pre-training is deterministic, so
# EM's prior-draw sd is exactly zero by construction.
#
# 10 cells x 6 seeds x 4 eval datasets gives enough replicates to report BUDGET-TO-TARGET
# -- the metric we actually care about, and the one the paper uses -- with a split-clustered
# standard error and a paired per-split comparison against each baseline, rather than a
# bare mean over one split.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/bo_multisplit" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,ablr,ablr_adapt,pacoh_frozen,vanilla_gp,random"

for sp in 0 1 2 3 4; do
  for ps in 0 1; do
    out="$R/raw/bo_multisplit/sp${sp}_p${ps}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment \
      --n-configs 200 --n-iters 40 --n-seeds 6 \
      --split-seed "$sp" --pretrain-seed "$ps" \
      --hyperbo-iters 25000 --ablr-iters 25000 --pacoh-iters 25000 \
      --threads 5 --methods "$METHODS" \
      --out "$out" > "$R/logs/bo_multisplit_sp${sp}_p${ps}.txt" 2>&1 &
  done
  wait   # one split-block at a time to bound concurrency
done
echo "[bo_multisplit] done -> $R/raw/bo_multisplit/"
