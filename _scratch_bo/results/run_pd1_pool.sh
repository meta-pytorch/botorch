#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# PD1 at the prior work's actual protocol, as a two-factor ablation.
#
# §21.2 found we were solving a different, strictly easier problem than Wang et al.:
#
#   (a) SEARCH SPACE. We ran BO over the 400-config matched intersection; they search
#       each task's own ~2040 configurations. Ours saturates within 50 iterations --
#       every method returns the identical configuration -- and our pool's oracle
#       (20.86% on cifar100_wrn) is WORSE than their reported result (20.48%), so their
#       number was literally unreachable in our candidate set. On the full pool the
#       oracle is 20.40%, i.e. their result is now inside the search space.
#
#   (b) PRE-TRAINING DATA. "We used data from all but the test task ... ranging from 4 to
#       about 1600 [points per task]" -- H-NLL / FSBO / ABLR / MIMO train on matched +
#       unmatched, and only H-EKL is restricted to matching inputs. We had been giving
#       every baseline the 400 matched points, a 4x data disadvantage in exactly the
#       regime their scaling study says HyperBO and ABLR gain most.
#
# The two factors are separated so we can attribute the change:
#   arm B = full candidates, matched pre-training  -> isolates the search space
#   arm C = full candidates, full pre-training     -> adds the data advantage
#   (arm A = matched/matched is §17, already committed as pd1_loo_neglog_*.)
#
# EM always pre-trains on the matched grid -- its inducing points require it -- and
# reaches the off-grid candidates through shift interpolation, which is the
# continuous-domain mechanism the paper itself contributes.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/

SCRATCH="pytorch/botorch/_scratch_bo"
R="${RESULTS_DIR:-$SCRATCH/results}"
SHARDS="${SHARDS:-12}"
THREADS="${THREADS:-3}"
mkdir -p "$R/raw/pd1pool" "$R/logs"

METHODS="em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,vanilla_gp,random"
COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --n-iters 50 --n-seeds 5 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-iters 25000 --pacoh-iters 10000 --ablr-iters 10000 --ablr-per-task-precision \
  --pacoh-particles 10 --threads $THREADS --methods $METHODS"

launch() {  # launch <arm> <candidate-pool> <pretrain-pool>
  local arm="$1" cand="$2" pre="$3" i
  echo "[pd1pool] arm=$arm candidates=$cand pretrain=$pre, $SHARDS shards"
  for i in $(seq 0 $((SHARDS - 1))); do
    printf '%s\t%s\t%s\n' "pd1pool_${arm}_s${i}" "$(date -u +%FT%TZ)" \
      "bo_experiment $COMMON --pd1-candidate-pool $cand --pd1-pretrain-pool $pre" \
      >> "$R/MANIFEST.tsv"
    # shellcheck disable=SC2086
    python3 -m _scratch_bo.bo_experiment $COMMON \
      --pd1-candidate-pool "$cand" --pd1-pretrain-pool "$pre" \
      --loo-shard "${i}/${SHARDS}" --out "$R/raw/pd1pool/${arm}_s${i}.json" \
      > "$R/logs/pd1pool_${arm}_s${i}.txt" 2>&1 &
  done
}

launch fullcand full matched
launch fullboth full full
wait
echo "[pd1pool] done -> $R/raw/pd1pool/"
