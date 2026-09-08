#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Debugging why our HyperBO underperforms ABLR, when Wang et al. (2021) report the
# opposite. Paired per-run evidence puts HyperBO significantly behind ABLR on regret-AUC
# (z=+2.81 on LCBench, z=+2.20 on PD1), and the entire LCBench deficit comes from 2 of
# 12 eval datasets -- sylvine and kr-vs-kp, the two hardest. Since the LCBench prior is
# trained once and shared across all runs, that is not an unlucky fit; the prior itself
# generalizes poorly on hard tasks.
#
# Two hypotheses, both tested here:
#
#   H1  Under-training. We use 2000 Adam steps; the paper uses 50000. Section 4 already
#       showed @10 regret improving 7x (0.56 -> 0.08) going from 2k to 5k, so the budget
#       was chosen on a criterion (final regret) that Section 5J later showed is
#       saturated. The sweep below traces the whole budget/quality curve and doubles as
#       the performance-vs-pre-training-time Pareto.
#
#   H2  The learned mean hurts instead of helping. HyperBO uses mu(x)=W.phi(x); ABLR has
#       a zero prior mean and so cannot be misled. Section 14.1 found our HyperBO mean is
#       well-ordered globally (Spearman 0.771) but poor at the extremes (top-10 precision
#       0.13 vs EM's 0.37) -- exactly what would bite on hard tasks. Wang et al. report
#       the OPPOSITE ordering: H-NLL beats FSBO (= HyperBO with zero mean) "especially in
#       early stages of BO". Reproducing their ordering is therefore a soundness check:
#       if our zero-mean variant WINS, our mean is mis-trained.
#
# All arms are paired -- identical datasets, seeds and split-seed as the headline -- and
# use a deliberately cheap method set so the online BO loop does not dominate: the
# expensive per-step-refit arms (vanilla_gp, em_finetuned) are excluded because they are
# irrelevant to the question and cost 6-14 s/run.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/

SCRATCH="pytorch/botorch/_scratch_bo"
R="${RESULTS_DIR:-$SCRATCH/results}"
THREADS="${THREADS:-6}"
mkdir -p "$R/raw/hyperbo_debug" "$R/logs"

# em_frozen / ablr / random are fixed references; only the HyperBO arms vary.
METHODS="hyperbo_frozen,hyperbo_adapt,ablr,em_frozen,random"
COMMON="--n-configs 300 --n-iters 40 --n-eval 12 --n-seeds 16 --n-pretrain 25 \
  --split-seed 42 --meta-subsample 64 --meta-task-batch 5 --meta-iters 2000 \
  --ablr-iters 10000 --threads $THREADS --methods $METHODS"

launch() {  # launch <tag> <extra-args...>
  local tag="$1"; shift
  printf '%s\t%s\t%s\n' "$tag" "$(date -u +%FT%TZ)" "bo_experiment $COMMON $*" \
    >> "$R/MANIFEST.tsv"
  # shellcheck disable=SC2086
  python3 -m _scratch_bo.bo_experiment $COMMON "$@" \
    --out "$R/raw/hyperbo_debug/$tag.json" > "$R/logs/$tag.txt" 2>&1 &
}

# H1: budget/quality Pareto. 2000 and 50000 also exist in the headline runs, but are
# repeated here with the cheap method set so every Pareto point is directly comparable.
for it in 2000 5000 10000 25000 50000; do
  launch "hbdebug_nll_$it" --hyperbo-iters "$it"
done

# H2: zero-mean (FSBO) at the low and high ends of the budget curve.
for it in 2000 50000; do
  launch "hbdebug_zeromean_$it" --hyperbo-iters "$it" --hyperbo-zero-mean
done

wait
echo "[hyperbo-debug] done -> $R/raw/hyperbo_debug/"
