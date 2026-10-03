#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# EM tuning factorial on PD1 arm C: shrinkage alpha x shrinkage TARGET.
#
# WHY THESE TWO FACTORS, TOGETHER
#
# The M-step is  Sigma <- (1 - a) * Sigma_ML + a * s * B.  At a = 0 the target B drops
# out of the equation entirely, so alpha and the target are only meaningful JOINTLY.
# Every PD1 run to date fixed a = 0 (S27.9), which means:
#   - the shrinkage target has never been exercised on PD1 at all; and
#   - the one alpha screen that exists (followup/pd1_shrink_full_a*) ran in ARM B, where
#     build_shared_kernel sees only the 400-config matched grid rather than the ~2040
#     full pools -- starving the very kernel that drives the Nystrom map
#     W = K(X,Z)K(Z,Z)^-1 that carries the empirical covariance off-grid
#     (bo_experiment.py:1463). It found alpha HURTING, which that handicap may explain.
#
# So this runs in ARM C (full candidate AND full pre-training pools) and crosses:
#
#     alpha  in {0, 0.1, 0.3}        x     target in {sumll, hyperbo}
#
# canon_decomp found the shrinkage-target door the largest of the three (4.10 nats), so
# (alpha > 0, target = hyperbo) is the predicted winner and has never been run.
#
# WHY alpha=0 IS STILL INCLUDED FOR BOTH TARGETS
#
# At alpha = 0 the target is a no-op for SHRINKAGE, but --em-canonical hyperbo also
# replaces the interpolation kernel and the EM initialisation -- two of the three doors.
# So (0, sumll) and (0, hyperbo) are NOT the same run, and the pair isolates the
# shrinkage door from the other two. (0, sumll) doubles as the internal control.
#
# SCORING: budget-to-target, per .llms/rules/metrics.md. NOT regret-AUC.
#
# BUILT-IN CONTROLS, check these before trusting any cell:
#   - hyperbo_frozen / vanilla_gp / random must be INVARIANT across alpha within a
#     target, since alpha touches only EM. If they move, the run is contaminated.
#   - (0, sumll) and (0, hyperbo) must DIFFER for the EM methods, or --em-canonical is a
#     no-op on this path and the whole target axis is void.
#
# KNOWN INEFFICIENCY: with --em-canonical hyperbo, _pd1_canonical trains its own HyperBO
# prior, and the main gate trains another for hyperbo_frozen / the additive base. That is
# up to 2x the most expensive component per fold. Correct but wasteful; left alone rather
# than risk a code change mid-study.
#
# STAGE 1 of 2. This is a SCREEN: one pre-training prior, 3 seeds. S27.3 is emphatic that
# one prior cannot rank neural meta-learners -- so the screen ranks CELLS, and the winning
# cell must then be replicated at 3 priors before any claim is made.

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
R="${RESULTS_DIR:-$S/results}"
SHARDS="${SHARDS:-12}"
THREADS="${THREADS:-3}"
PRETRAIN_SEED="${PRETRAIN_SEED:-0}"
NSEEDS="${NSEEDS:-3}"
ALPHAS="${ALPHAS:-0.0 0.1 0.3}"
TARGETS="${TARGETS:-sumll hyperbo}"
mkdir -p "$R/raw/armc_factorial" "$R/logs"

# Refuse to oversubscribe the box: 12 shards x 3 threads already saturates it.
if systemctl --user is-active --quiet hybrid-armc-sweep.service; then
  echo "[factorial] hybrid-armc-sweep is still running; refusing to start." >&2
  echo "[factorial] wait for it, or run with a smaller SHARDS." >&2
  exit 1
fi

METHODS="em_frozen,em_noisefit,em_finetuned,em_additive_hyperbo,\
em_additive_hyperbo_frozen,hyperbo_frozen,vanilla_gp,random"

COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --pd1-candidate-pool full --pd1-pretrain-pool full \
  --n-iters 50 --n-init 3 --per-task-standardize --output-warp neglog \
  --meta-subsample 64 --meta-task-batch 5 \
  --hyperbo-iters 25000 --threads $THREADS --methods $METHODS"

for target in $TARGETS; do
  for alpha in $ALPHAS; do
    cell="a${alpha}_${target}"
    echo "[factorial] cell $cell (prior $PRETRAIN_SEED, $NSEEDS seeds, $SHARDS shards)"
    for i in $(seq 0 $((SHARDS - 1))); do
      out="$R/raw/armc_factorial/${cell}_s${i}.json"
      [ -f "$out" ] && continue
      printf '%s\t%s\t%s\n' "armc_factorial_${cell}_s${i}" "$(date -u +%FT%TZ)" \
        "bo_experiment $COMMON --em-shrinkage $alpha --em-canonical $target" \
        >> "$R/MANIFEST.tsv"
      # shellcheck disable=SC2086
      python3 -m _scratch_bo.bo_experiment $COMMON \
        --em-shrinkage "$alpha" --em-canonical "$target" \
        --n-seeds "$NSEEDS" --pretrain-seed "$PRETRAIN_SEED" \
        --loo-shard "${i}/${SHARDS}" --out "$out" \
        > "$R/logs/armc_factorial_${cell}_s${i}.txt" 2>&1 &
    done
    # Cells run sequentially; shards within a cell run in parallel.
    wait
    echo "[factorial] cell $cell done"
  done
done

echo "[factorial] all cells done -> $R/raw/armc_factorial/"
