#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Reproduction driver for the empirical-GP BO baseline study.
#
# Every sweep quoted in EXPERIMENTAL_RESULTS.md is defined here, so the exact
# invocations are version-controlled. (An earlier iteration kept them in /tmp/wave1.sh,
# which was lost when /tmp was cleaned -- hence this file.)
#
# Usage:
#   ./run_all.sh <wave> [<wave> ...]     # e.g. ./run_all.sh headline batch diagnose
#   ./run_all.sh all
#   RESULTS_DIR=/somewhere ./run_all.sh headline
#
# Jobs inside a wave run concurrently; each is capped at THREADS intra-op torch threads
# so a wave does not oversubscribe the box. Logs land next to the JSONs (as .txt, since
# *.log is gitignored repo-wide).
#
# Long sweeps must be launched detached -- `setsid ... </dev/null &` -- because a plain
# `nohup ... &` from a short-lived shell gets reaped along with its parent.
#
# shellcheck disable=SC2086
# $META and $COMMON hold multiple flags and are deliberately left unquoted so they
# word-split into separate arguments.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1  # -> <repo>/
SCRATCH="pytorch/botorch/_scratch_bo"
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/results}"
RAW="$RESULTS_DIR/raw"
LOGS="$RESULTS_DIR/logs"
THREADS="${THREADS:-4}"
BO="python3 -m _scratch_bo.bo_experiment"
CURVE="python3 -m _scratch_bo.curve_experiment"
DIAG="python3 -m _scratch_bo.bo_diagnose"

mkdir -p "$RAW" "$LOGS"
MANIFEST="$RESULTS_DIR/MANIFEST.tsv"
[ -f "$MANIFEST" ] || printf 'name\tstarted_utc\tcommand\n' > "$MANIFEST"

PIDS=()
job() {  # job <name> <args...>
  local name="$1"; shift
  local out="$RAW/$name.json"
  local log="$LOGS/$name.txt"
  printf '%s\t%s\t%s\n' "$name" "$(date -u +%FT%TZ)" "$BO $* --out $out" >> "$MANIFEST"
  echo "  -> $name (log: $log)"
  ( $BO "$@" --threads "$THREADS" --out "$out" > "$log" 2>&1 ) &
  PIDS+=($!)
}

wait_wave() {
  local rc=0
  for p in "${PIDS[@]:-}"; do wait "$p" || rc=1; done
  PIDS=()
  [ "$rc" -eq 0 ] && echo "  wave OK" || echo "  WAVE HAD FAILURES (see logs)"
  return 0
}

# Shared meta-learner settings: the best-faithful operating point from the sweeps
# (32,32)-tanh deep kernel, NLL pre-training, subsample 64, task-minibatch 5.
META="--meta-subsample 64 --meta-task-batch 5 --meta-iters 2000 --pacoh-particles 10"
ALL_METHODS="em_frozen,em_finetuned,em_noisefit,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,vanilla_gp,random"
BATCH_METHODS="em_frozen,hyperbo_frozen,vanilla_gp,random"

# ---------------------------------------------------------------- headline (S5J)
# High-power LCBench table + the main figure. Wider eval set than the original
# 8-dataset run: 12 eval datasets x 16 seeds = 192 runs.
#
# NOTE: LCBench has 35 datasets, so --n-eval 12 + --n-pretrain 25 = 37 overruns and the
# pretrain slice silently truncates to 23 datasets. That is the number actually used.
wave_headline() {
  echo "[headline] 12 eval datasets x 16 seeds, 300 configs, 40 iters"
  job lcbench_highpower \
    --n-configs 300 --n-iters 40 --n-eval 12 --n-seeds 16 --n-pretrain 25 \
    $META --methods "$ALL_METHODS"
  wait_wave
}

# Same headline with each meta-learner at a converged pre-training budget rather than a
# shared 2000 steps. S4's "knee at 2000, 5000 is wasted compute" was read off FINAL
# regret, which S5J later showed is saturated; at @10 the 5000-step HyperBO was 7x
# better. Paired with wave_headline (identical datasets/seeds/methods, only the budgets
# differ), so the EM/vanilla/random arms must come out bit-identical.
wave_headline_50k() {
  echo "[headline_50k] HyperBO 50k / PACOH 10k / ABLR 10k steps"
  job lcbench_highpower_50k \
    --n-configs 300 --n-iters 40 --n-eval 12 --n-seeds 16 --n-pretrain 25 \
    --split-seed 42 $META \
    --hyperbo-iters 50000 --pacoh-iters 10000 --ablr-iters 10000 \
    --methods "$ALL_METHODS"
  wait_wave
}

# ------------------------------------------------------------------- batch (S5L)
# Batch BO: greedy top-q vs kriging-believer vs MC-fantasy qLogEI, at q in {2,4},
# with the q=1 sequential control on the same seeds.
wave_batch() {
  echo "[batch] q in {1,2,4} x {topq, kb, fantasy}"
  local COMMON="--n-configs 200 --n-iters 20 --n-eval 4 --n-seeds 6 --n-pretrain 25 \
    --meta-subsample 64 --meta-task-batch 5 --meta-iters 2000 --methods $BATCH_METHODS"
  # shellcheck disable=SC2086
  job batch_q1            $COMMON --batch-q 1
  # shellcheck disable=SC2086
  job batch_topq_q2       $COMMON --batch-q 2 --batch-mode topq
  # shellcheck disable=SC2086
  job batch_topq_q4       $COMMON --batch-q 4 --batch-mode topq
  # shellcheck disable=SC2086
  job batch_kb_q2         $COMMON --batch-q 2 --batch-mode kb
  # shellcheck disable=SC2086
  job batch_kb_q4         $COMMON --batch-q 4 --batch-mode kb
  # shellcheck disable=SC2086
  job batch_fantasy_q2    $COMMON --batch-q 2 --batch-mode fantasy --n-fantasies 4
  # shellcheck disable=SC2086
  job batch_fantasy_q4    $COMMON --batch-q 4 --batch-mode fantasy --n-fantasies 4
  wait_wave
}

# ---------------------------------------------------------------- diagnose (S7)
wave_diagnose() {
  echo "[diagnose] prior-mean informativeness + calibration, all methods"
  local log="$LOGS/diagnose.txt"
  printf '%s\t%s\t%s\n' diagnose "$(date -u +%FT%TZ)" "$DIAG (all methods)" >> "$MANIFEST"
  $DIAG --n-eval 6 --n-pretrain 25 --n-obs-grid 5,10,20,50 --meta-iters 2000 \
    --threads 8 --out "$RAW/diagnose.json" > "$log" 2>&1
  echo "  diagnose done (log: $log)"
}

# ------------------------------------------------- legacy sweeps (S5F-S5K, S9-S12)
# These reproduce the older sections. They are not re-run by `all` because they are
# long; invoke explicitly. Committed so the invocations are never lost again.
wave_scaling() {  # S5F meta-data scaling
  echo "[scaling] regret vs #pretrain datasets"
  for n in 2 5 10 15 25; do
    job "scaling_n$n" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 4 \
      --n-pretrain "$n" $META \
      --methods em_frozen,hyperbo_frozen,pacoh_frozen,random
  done
  wait_wave
}

wave_ninit() {  # S5G initial-design sweep
  echo "[ninit] zero-shot -> warm start"
  for k in 0 1 3 5 10; do
    job "ninit_$k" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 4 \
      --n-pretrain 25 --n-init "$k" $META \
      --methods em_frozen,hyperbo_frozen,pacoh_frozen,vanilla_gp,random
  done
  wait_wave
}

wave_ood() {  # S5H OOD split + S5I EM variants
  echo "[ood] cross-distribution transfer + EM variants"
  job ood_split --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 4 \
    --n-pretrain 25 --ood-split $META \
    --methods em_frozen,em_finetuned,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,vanilla_gp,random
  job em_variants --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 4 \
    --n-pretrain 25 $META \
    --methods em_frozen,em_finetuned,em_additive_freshbase,em_additive_3way,hyperbo_frozen,random
  wait_wave
}

# --------------------------------------------------- EM shrinkage (S16)
# Does the EM M-step covariance shrinkage -- which S13 established as the default for
# 1D incomplete-curve imputation -- help or hurt 7D BO? Run in-distribution and under
# the OOD split, since the two regimes are expected to disagree. Paired: the same seeds
# and datasets across alpha, and alpha is an EM-only knob so the non-EM arms must come
# out bit-identical (a built-in sanity check).
wave_shrinkage() {
  echo "[shrinkage] in-distribution, alpha in {0, 0.1, 0.3}"
  for a in 0.0 0.1 0.3; do
    job "shrinkage_a$a" --n-configs 300 --n-iters 40 --n-eval 8 --n-seeds 8 \
      --n-pretrain 25 --split-seed 7 --em-shrinkage "$a" $META \
      --methods em_frozen,em_noisefit,hyperbo_frozen,vanilla_gp,random
  done
  wait_wave
}

wave_ood_shrinkage() {
  echo "[ood_shrinkage] OOD split, alpha in {0, 0.1, 0.3}"
  for a in 0.0 0.1 0.3; do
    job "ood_shrinkage_a$a" --n-configs 200 --n-iters 30 --n-eval 6 --n-seeds 10 \
      --n-pretrain 25 --ood-split --em-shrinkage "$a" $META \
      --methods em_frozen,em_noisefit,em_finetuned,hyperbo_frozen,vanilla_gp,random
  done
  wait_wave
}

# EM prior covariance eigenspectrum vs alpha (effective rank, trace concentration).
wave_spectrum() {
  echo "[spectrum] EM covariance spectrum vs alpha"
  for a in 0.0 0.1 0.3; do
    printf '%s\t%s\t%s\n' "spectrum_a$a" "$(date -u +%FT%TZ)" "$DIAG --em-shrinkage $a" \
      >> "$MANIFEST"
    ( $DIAG --n-eval 6 --n-pretrain 25 --n-obs-grid 5,50 --meta-iters 500 \
        --em-shrinkage "$a" --threads 6 --methods em_frozen,em_noisefit \
        --out "$RAW/spectrum_a$a.json" > "$LOGS/spectrum_a$a.txt" 2>&1 ) &
    PIDS+=($!)
  done
  wait_wave
}

wave_noise() {  # S5K observation-noise robustness  echo "[noise] obs-noise sweep"
  for s in 0 0.1 0.25 0.5; do
    job "noise_$s" --n-configs 200 --n-iters 30 --n-eval 4 --n-seeds 6 \
      --n-pretrain 25 --obs-noise "$s" $META \
      --methods em_frozen,em_noisefit,hyperbo_frozen,pacoh_frozen,vanilla_gp,random
  done
  wait_wave
}

wave_pd1() {  # S9-S12 PD1
  echo "[pd1] matched, full, and leave-one-out"
  job pd1_matched --benchmark pd1 --n-configs 50 --n-iters 30 --n-eval 4 --n-seeds 6 \
    --n-pretrain 15 $META --methods "$ALL_METHODS"
  job pd1_full --benchmark pd1_full --n-iters 30 --n-eval 4 --n-seeds 6 \
    --n-pretrain 15 --per-task-standardize $META \
    --methods hyperbo_frozen,hyperbo_adapt,pacoh_frozen,ablr,vanilla_gp,random
  for w in none gaussian neglog; do
    job "pd1_loo_$w" --benchmark pd1_loo --pd1-group 256 --n-iters 30 --n-seeds 5 \
      --n-pretrain 9 --per-task-standardize --output-warp "$w" $META \
      --methods "$ALL_METHODS"
  done
  wait_wave
}

wave_curve() {  # S3/S5E 1D learning-curve extrapolation
  echo "[curve] 1D extrapolation"
  for c in 0.1 0.2 0.3 0.4 0.5; do
    local out="$RAW/curve_ctx$c.json"
    printf '%s\t%s\t%s\n' "curve_ctx$c" "$(date -u +%FT%TZ)" "$CURVE ctx=$c" >> "$MANIFEST"
    ( $CURVE --n-curves 100 --n-pretrain 20 --n-eval 5 --n-eval-curves 30 \
        --context-frac "$c" --meta-iters 2000 --meta-subsample 64 \
        --meta-task-batch 5 --hyperbo-loss NLL --out "$out" \
        > "$LOGS/curve_ctx$c.txt" 2>&1 ) &
    PIDS+=($!)
  done
  wait_wave
}

# ------------------------------------------------------------------------ figures
wave_figures() {
  echo "[figures] aggregation + plots"
  local AN="python3 -m _scratch_bo.analyze"
  $AN --result "$RAW/lcbench_highpower.json" --tag lcbench_highpower \
    --out-dir "$RESULTS_DIR/figures" --horizons 1,5,10,20,40 \
    --title "LCBench 7D finite-pool BO (12 datasets x 16 seeds)" \
    | tee "$RESULTS_DIR/lcbench_highpower.md"
  for f in "$RAW"/batch_*.json; do
    [ -e "$f" ] || continue
    local tag; tag="$(basename "$f" .json)"
    $AN --result "$f" --tag "$tag" --out-dir "$RESULTS_DIR/figures" \
      --horizons 1,5,10,20 --title "$tag" --no-figures \
      | tee "$RESULTS_DIR/$tag.md"
  done
}

[ $# -eq 0 ] && { echo "usage: $0 <headline|headline_50k|batch|diagnose|figures|shrinkage|ood_shrinkage|spectrum|scaling|ninit|ood|noise|pd1|curve|all>"; exit 1; }
for w in "$@"; do
  case "$w" in
    all) wave_headline; wave_batch; wave_diagnose; wave_figures ;;
    headline) wave_headline ;;
    headline_50k) wave_headline_50k ;;
    batch) wave_batch ;;
    diagnose) wave_diagnose ;;
    figures) wave_figures ;;
    scaling) wave_scaling ;;
    shrinkage) wave_shrinkage ;;
    ood_shrinkage) wave_ood_shrinkage ;;
    spectrum) wave_spectrum ;;
    ninit) wave_ninit ;;
    ood) wave_ood ;;
    noise) wave_noise ;;
    pd1) wave_pd1 ;;
    curve) wave_curve ;;
    *) echo "unknown wave: $w"; exit 1 ;;
  esac
done
echo "done. results under $RESULTS_DIR"
