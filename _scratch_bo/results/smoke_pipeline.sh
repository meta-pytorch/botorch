#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every TXT2
# The literal tabs in this file are NOT indentation -- they are field separators
# in the .tsv cell queues written by the heredocs below, consumed by
# `while IFS=$'\t' read -r name flags` in run_cell_queue.sh. Replacing them with
# spaces merges the name and flags fields and silently breaks the pipeline.

# END-TO-END PIPELINE SMOKE.
#
# WHY THIS EXISTS
#
# Seven adversarial review rounds found 125 defects by READING the code. The two
# guaranteed criticals in rounds 6 and 7 -- a NameError on every batch run, and
# --pretrain-seed passed to a harness that does not have it -- would each have been
# caught in ten minutes by RUNNING the pipeline once, and neither was caught by reading
# it or by 83 unit tests. Three of those tests passed on the very bug they were written
# to prevent.
#
# So this runs the real thing: gen_stage_queue -> run_cell_queue -> summarize_stage, on
# one BO regime and one REGRESSION regime, at toy settings. It asserts on artifacts, not
# on source text.
#
# It is deliberately NOT a unit test: it needs Buck, real data and minutes, and its job
# is to catch exactly the class of defect that unit tests kept certifying.
#
# Usage:  bash <ABS>/results/smoke_pipeline.sh          (~15 min)

set -uo pipefail

# Repository root, derived from this script's location so the sweep runs from
# any checkout. Override with BOTORCH_ROOT if the tree is laid out differently.
ROOT="${BOTORCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1

S=_scratch_bo
ABS="$ROOT/$S"
SMOKE="/tmp/pipeline_smoke_$(date -u +%s)"
mkdir -p "$SMOKE"
FAILED=0

say() { echo "[smoke $(date -u +%H:%M:%SZ)] $*"; }
fail() { say "FAIL: $*"; FAILED=$((FAILED + 1)); }

# Toy overrides: enough to exercise every code path, small enough to finish.
BO_COMMON="--benchmark pd1_loo --pd1-source full --pd1-group all \
  --pd1-candidate-pool full --pd1-pretrain-pool full --n-iters 4 --n-seeds 1 --n-init 3 \
  --per-task-standardize --output-warp neglog --meta-subsample 32 --meta-task-batch 3 \
  --hyperbo-iters 200 --meta-iters 200 --n-em 5"
REG_COMMON="--benchmark pd1_loo --pd1-source full --n-configs 200 --n-eval 4 \
  --n-pretrain 10 --n-obs-grid 5 --meta-iters 200 --n-em 5"

# ---------------------------------------------------------------- 1. BO regime
say "=== BO regime (pd1_bo_armC), 2 priors, 2 shards ==="
cat > "$SMOKE/bo.tsv" <<'EOF'
base_p0	--methods em_frozen,hyperbo_frozen --pretrain-seed 0
base_p1	--methods em_frozen,hyperbo_frozen --pretrain-seed 1
EOF

STUDY="smoke_bo" QUEUE="$SMOKE/bo.tsv" COMMON="$BO_COMMON" SHARDS=2 THREADS=4 \
  CELLS_PARALLEL=2 HARNESS=bo_experiment RESULTS_DIR="$SMOKE/results" \
  bash "$ABS/results/run_cell_queue.sh" 2>&1 | tail -4
BO_RC=$?
[ "$BO_RC" -eq 0 ] || fail "BO queue exited $BO_RC"

n=$(find "$SMOKE/results/raw/smoke_bo" -name '*.json' 2>/dev/null | wc -l)
[ "$n" -eq 4 ] || fail "expected 4 BO shards, found $n"

# ---------------------------------------------------------------- 2. Regression regime
say "=== REGRESSION regime (pd1_reg), 2 priors, 1 shard -- the R7-1 path ==="
cat > "$SMOKE/reg.tsv" <<'EOF'
base_p0	--methods em_frozen,hyperbo_frozen,vanilla_gp --pretrain-seed 0
base_p1	--methods em_frozen,hyperbo_frozen,vanilla_gp --pretrain-seed 1
EOF

STUDY="smoke_reg" QUEUE="$SMOKE/reg.tsv" COMMON="$REG_COMMON" SHARDS=1 THREADS=4 \
  CELLS_PARALLEL=2 HARNESS=bo_diagnose RESULTS_DIR="$SMOKE/results" \
  bash "$ABS/results/run_cell_queue.sh" 2>&1 | tail -4
REG_RC=$?
[ "$REG_RC" -eq 0 ] || fail "regression queue exited $REG_RC"

n=$(find "$SMOKE/results/raw/smoke_reg" -name '*.json' 2>/dev/null | wc -l)
[ "$n" -eq 2 ] || fail "expected 2 regression shards, found $n"

# ---------------------------------------------------------------- 3. Summarise both
say "=== summarise, and assert the artifacts carry real numbers ==="
for study in smoke_bo smoke_reg; do
  SCRATCH_BO_ROOT="$SMOKE" python3 "$ABS/summarize_stage.py" --stage "$study" \
    2>&1 | tail -3 || fail "summarise failed for $study"
done

SMOKE_DIR="$SMOKE" python3 - <<'PYEOF'
import json, os, sys

smoke = os.environ["SMOKE_DIR"]
bad = 0


def check(cond, msg):
    global bad
    if not cond:
        print(f"  FAIL: {msg}")
        bad += 1
    else:
        print(f"  ok  : {msg}")


for study, kind in (("smoke_bo", "bo"), ("smoke_reg", "regression")):
    path = os.path.join(smoke, "results/raw/v2", study, "SUMMARY.json")
    if not os.path.exists(path):
        print(f"  FAIL: no SUMMARY.json for {study}")
        bad += 1
        continue
    s = json.load(open(path))
    print(f"\n{study}:")
    check(len(s.get("cells", {})) == 2, f"{study}: two per-prior cells")
    check("configs" in s, f"{study}: pooled configs view exists")
    cfg = s.get("configs", {}).get("base", {})
    check(cfg.get("kind") == kind, f"{study}: pooled kind == {kind} (not error/empty)")
    check(cfg.get("n_priors") == 2, f"{study}: pooled n_priors == 2")
    methods = cfg.get("methods", {})
    check(bool(methods), f"{study}: pooled view carries methods, not {{}}")
    if kind == "bo":
        for m, v in methods.items():
            check(
                v.get("n_runs", 0) > 0 and v.get("budget_to_0.01") is not None,
                f"{study}: {m} has real BO metrics",
            )
        check(
            "between_prior" in cfg,
            f"{study}: between-prior spread present (the reason --priors exists)",
        )
    else:
        # R7-2: by_n_obs must survive, not just prior_mean.
        any_nobs = any(
            any(k.startswith("n5_") for k in v) for v in methods.values()
        )
        check(any_nobs, f"{study}: by_n_obs metrics survived the summariser")
        any_prior = any(
            any(k.startswith("prior_") for k in v) for v in methods.values()
        )
        check(any_prior, f"{study}: prior_mean metrics survived")

print()
print("SMOKE ASSERTION FAILURES:", bad)
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || FAILED=$((FAILED + 1))

say "=== artifacts in $SMOKE ==="
say "TOTAL FAILURES: $FAILED"
exit $([ "$FAILED" -eq 0 ] && echo 0 || echo 1)
