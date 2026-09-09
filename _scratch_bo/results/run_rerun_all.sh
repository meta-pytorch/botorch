#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Overnight re-run of every study contaminated by the bugs found in S19.
#
# What was wrong, and what it touches:
#
#  1. HyperBO trained for 2000 steps, far below the ~5000-step knee (S19.2). Costs it
#     33% of its regret-AUC. EVERY section using HyperBO at 2000 steps is affected:
#     S3, S5B, S5D, S5E, S5F, S5G, S5H, S5I, S5J, S5K, S5L, S14, S16.
#  2. ABLR was also left at --meta-iters (2000) in the old headline, and deviated from
#     Perrone et al. (single shared precisions, no task sub-sampling).
#  3. EKL was optimized with Adam rather than L-BFGS, so S5C's "NLL beats EKL in 7D"
#     conclusion was drawn against an under-optimized EKL.
#  4. One pre-trained prior is shared by every BO run, so per-run error bars on neural
#     meta-learners are not evidence about the method (S19.4). Studies where the
#     meta-learner comparison is the POINT get multiple --pretrain-seed draws.
#
# Corrected budgets: HyperBO 25000 (best point on the S19.2 Pareto, and cheaper than
# 50000), PACOH 10000, ABLR 10000, ABLR faithful per-task precisions.
#
# Outputs go to results/raw/rerun/ so they never overwrite the contaminated originals --
# the old numbers stay auditable next to the corrected ones.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/

SCRATCH="pytorch/botorch/_scratch_bo"
R="${RESULTS_DIR:-$SCRATCH/results}"
OUT="$R/raw/rerun"
THREADS="${THREADS:-5}"
MAXJOBS="${MAXJOBS:-14}"
mkdir -p "$OUT" "$R/logs/rerun"

BO="python3 -m _scratch_bo.bo_experiment"
CURVE="python3 -m _scratch_bo.curve_experiment"

# Corrected, per-method converged budgets + the faithful ABLR.
BUDGETS="--hyperbo-iters 25000 --pacoh-iters 10000 --ablr-iters 10000 \
  --ablr-per-task-precision --meta-subsample 64 --meta-task-batch 5 --meta-iters 2000"
FULL="em_frozen,em_finetuned,em_noisefit,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,ablr_adapt,vanilla_gp,random"
CHEAP="em_frozen,em_noisefit,hyperbo_frozen,hyperbo_adapt,ablr,pacoh_frozen,vanilla_gp,random"

throttle() {  # keep at most MAXJOBS background jobs alive
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
}

job() {  # job <name> <args...>
  local name="$1"; shift
  throttle
  printf '%s\t%s\t%s\n' "rerun/$name" "$(date -u +%FT%TZ)" "$BO $*" >> "$R/MANIFEST.tsv"
  echo "  -> $name"
  # shellcheck disable=SC2086
  ( $BO "$@" --threads "$THREADS" --out "$OUT/$name.json" \
      > "$R/logs/rerun/$name.txt" 2>&1 ) &
}

echo "=== A. headline, 3 pre-training seeds (S5J) ==="
for s in 0 1 2; do
  job "headline_p$s" --n-configs 300 --n-iters 40 --n-eval 12 --n-seeds 16 \
    --n-pretrain 25 --split-seed 42 --pretrain-seed "$s" $BUDGETS --methods "$FULL"
done

echo "=== B. meta-data scaling (S5F) ==="
for n in 2 5 10 15 25; do
  job "scaling_n$n" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 \
    --n-pretrain "$n" $BUDGETS --methods "$CHEAP"
done

echo "=== C. initial-design sweep (S5G) ==="
for k in 0 1 3 5 10; do
  job "ninit_$k" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 \
    --n-pretrain 25 --n-init "$k" $BUDGETS --methods "$CHEAP"
done

echo "=== D. OOD split + EM variants (S5H, S5I) ==="
job ood_split --n-configs 200 --n-iters 30 --n-eval 6 --n-seeds 10 --n-pretrain 25 \
  --ood-split $BUDGETS --methods "$FULL"
job em_variants --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 --n-pretrain 25 \
  $BUDGETS --methods em_frozen,em_finetuned,em_noisefit,em_additive_freshbase,em_additive_3way,hyperbo_frozen,ablr,random

echo "=== E. observation-noise robustness (S5K) ==="
for s in 0 0.1 0.25 0.5; do
  job "noise_$s" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 --n-pretrain 25 \
    --obs-noise "$s" $BUDGETS --methods "$CHEAP"
done

echo "=== F. acquisition ablation (S5B) ==="
for a in logei greedy ucb; do
  job "acq_$a" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 --n-pretrain 25 \
    --acquisition "$a" $BUDGETS --methods "$CHEAP"
done

echo "=== G. novel-config split (S5D) ==="
job novel_config --n-configs 200 --n-iters 30 --n-eval 6 --n-seeds 6 --n-pretrain 25 \
  --novel-config-split $BUDGETS --methods "$CHEAP"

echo "=== H. batch BO (S5L) ==="
for q in 2 4; do
  for mode in topq kb fantasy; do
    job "batch_${mode}_q$q" --n-configs 200 --n-iters 20 --n-eval 4 --n-seeds 6 \
      --n-pretrain 25 --batch-q "$q" --batch-mode "$mode" --n-fantasies 4 \
      $BUDGETS --methods em_frozen,hyperbo_frozen,ablr,vanilla_gp,random
  done
done
job batch_q1 --n-configs 200 --n-iters 20 --n-eval 4 --n-seeds 6 --n-pretrain 25 \
  $BUDGETS --methods em_frozen,hyperbo_frozen,ablr,vanilla_gp,random

echo "=== I. EM shrinkage, in-distribution + OOD (S16) ==="
for a in 0.0 0.1 0.3; do
  job "shrink_a$a" --n-configs 300 --n-iters 40 --n-eval 8 --n-seeds 8 --n-pretrain 25 \
    --split-seed 7 --em-shrinkage "$a" $BUDGETS \
    --methods em_frozen,em_noisefit,hyperbo_frozen,vanilla_gp,random
  job "ood_shrink_a$a" --n-configs 200 --n-iters 30 --n-eval 6 --n-seeds 10 \
    --n-pretrain 25 --ood-split --em-shrinkage "$a" $BUDGETS \
    --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp,random
done

echo "=== J. 1D learning-curve extrapolation (S3, S5E) ==="
for c in 0.1 0.2 0.3 0.4 0.5; do
  throttle
  printf '%s\t%s\t%s\n' "rerun/curve_ctx$c" "$(date -u +%FT%TZ)" "$CURVE ctx=$c" \
    >> "$R/MANIFEST.tsv"
  echo "  -> curve_ctx$c"
  # curve_experiment has no per-method budget flag; --meta-iters is the HyperBO budget.
  ( $CURVE --n-curves 100 --n-pretrain 20 --n-eval 5 --n-eval-curves 30 \
      --context-frac "$c" --meta-iters 25000 --meta-subsample 64 --meta-task-batch 5 \
      --hyperbo-loss NLL --out "$OUT/curve_ctx$c.json" \
      > "$R/logs/rerun/curve_ctx$c.txt" 2>&1 ) &
done

echo "=== K. all-method diagnostics (S14) ==="
throttle
( python3 -m _scratch_bo.bo_diagnose --n-eval 6 --n-pretrain 25 \
    --n-obs-grid 5,10,20,50 --meta-iters 25000 --threads 8 \
    --out "$OUT/diagnose.json" > "$R/logs/rerun/diagnose.txt" 2>&1 ) &

wait
echo "=== ALL RERUNS COMPLETE: $(ls "$OUT" | wc -l) outputs in $OUT ==="
