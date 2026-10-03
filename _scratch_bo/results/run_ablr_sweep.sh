#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ABLR capacity/faithfulness sweep on a DEV split.
#
# ABLR (Perrone et al. 2018) is the strongest baseline we have -- rank 1 on PD1 AUC and
# top-clique on LCBench -- and it is the only one that has never been tuned. It is also
# the only meta-learner whose implementation deviated from its paper: a single shared
# (alpha, beta) across all pre-training tasks instead of per-task precisions, and no
# task sub-sampling. This sweep gives it its best honest shot.
#
# Selection is on --split-seed 7, whose eval datasets are disjoint from the 12 reported
# test datasets AND, by construction, from the pre-training pool -- so a capacity knob is
# never chosen on data the features were fit to. That matters: capacity and
# regularization knobs look best at their most permissive setting when scored in-sample.
#
# Grid: feat_dim x hidden x per-task-precision, with both the frozen (ablr) and
# per-target-adapted (ablr_adapt) heads scored in every run.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/

SCRATCH="pytorch/botorch/_scratch_bo"
R="${RESULTS_DIR:-$SCRATCH/results}"
THREADS="${THREADS:-5}"
mkdir -p "$R/raw/ablr" "$R/logs"

COMMON="--n-configs 300 --n-iters 40 --n-eval 8 --n-seeds 8 --n-pretrain 25 \
  --split-seed 7 --meta-subsample 64 --meta-task-batch 5 --meta-iters 2000 \
  --ablr-iters 10000 --threads $THREADS \
  --methods ablr,ablr_adapt,em_frozen,hyperbo_frozen,random"

for feat in 25 50 100 200; do
  for hid in 32,32 64,64; do
    for prec in shared pertask; do
      flag=""
      [ "$prec" = "pertask" ] && flag="--ablr-per-task-precision"
      tag="ablr_f${feat}_h${hid//,/x}_${prec}"
      # shellcheck disable=SC2086
      python3 -m _scratch_bo.bo_experiment $COMMON \
        --ablr-feat-dim "$feat" --ablr-hidden "$hid" $flag \
        --out "$R/raw/ablr/$tag.json" > "$R/logs/$tag.txt" 2>&1 &
    done
  done
done
wait
echo "[ablr-sweep] done -> $R/raw/ablr/"
