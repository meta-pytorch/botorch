#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Refreshed 7D regression numbers, the analogue of the paper's Table 2 (tab:hd_nll).
#
# The original table's generating code is not in this stack, so this is NOT an update
# of those numbers -- it is a new, self-consistent table produced by the committed
# pipeline, with all methods regenerated together. Two known deviations from the
# paper's stated protocol, both recorded here so the caption can be honest:
#
#   * anchors: bo_diagnose ties the EM inducing set to the candidate pool, so M = 300
#     rather than the paper's M = 100. The pool must exceed max(n_train) + n_test, and
#     the grid runs to n_train = 100.
#   * splits: drawn by this harness's own seeding, so they differ from the original run.
#
# Methods match the four columns of tab:hd_nll. em_finetuned is the variant that
# actually produced the published numbers (EM prior + fitted additive base kernel,
# S21.1), which the manuscript never named -- em_frozen is included alongside so the
# cost of omitting the base kernel is visible.
#
# Three pre-training seeds, because S19.4 showed a single prior cannot rank the
# neural baselines.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
S=_scratch_bo
R=$S/results
mkdir -p "$R/raw/reg7d" "$R/logs"

METHODS="em_finetuned,em_frozen,em_noisefit,hyperbo_frozen,pretrained_gp_frozen,vanilla_gp"

for ps in 0 1 2; do
  out="$R/raw/reg7d/reg7d_p${ps}.json"
  [ -f "$out" ] && continue
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_diagnose \
    --n-configs 300 --n-pretrain 30 --n-eval 5 \
    --n-obs-grid 5,10,20,50,100 \
    --meta-iters 25000 --threads 6 --methods "$METHODS" \
    --out "$out" > "$R/logs/reg7d_p${ps}.txt" 2>&1 &
done
wait
echo "[reg7d] done -> $R/raw/reg7d/"
