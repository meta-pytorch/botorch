#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Arm C (full candidate pool + full pre-training data) replicated across additional
# pre-training priors. The §21.3 finding -- that the full search space flips HyperBO from
# last to first among the transfer methods, reproducing Wang et al.'s ordering -- rests
# on a single prior per arm, which §19.4 established is not enough to rank neural
# meta-learners. These replicates test whether the flip survives.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
M="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,ablr,vanilla_gp,random"
COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --pd1-candidate-pool full --pd1-pretrain-pool full \
  --n-iters 50 --n-seeds 3 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 --hyperbo-iters 25000 --pacoh-iters 10000 \
  --ablr-iters 10000 --ablr-per-task-precision --threads 3 --methods $M"

for ps in 1 2; do
  for i in $(seq 0 11); do
    out="$R/raw/pd1pool/fullboth_p${ps}_s${i}.json"
    [ -f "$out" ] && continue
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment $COMMON \
      --pretrain-seed "$ps" --loo-shard "${i}/12" --out "$out" \
      > "$R/logs/pd1pool_fullboth_p${ps}_s${i}.txt" 2>&1 &
  done
done
wait
echo "arm-C replicates done"
