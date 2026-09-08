# Adversarial review of the experiment setup — 2026-08-17

Three independent adversarial probes (orchestration/crash-robustness, metric
correctness, flag-to-model integration) run before launching the EXPERIMENT_PLAN
matrix. Verbatim output, preserved because several findings invalidate numbers
already recorded in EXPERIMENTAL_RESULTS.md.


==============================================================================
Adversarial review: orchestration and crash robustness
==============================================================================
## Direct Answer

I traced all five. **Q1's premise is false** â `run_cell`'s bare `wait` is actually correct. But the file is riddled with other real defects, and **the single worst bug is `run_autonomous.sh:108`: `.DONE` is written unconditionally**, which falsifies the entire resumability contract stated in the header comment. A stage in which *every shard crashed* is marked complete and permanently skipped.

16 concrete bugs below, ordered by severity.

---

## Key Locations

| file:line | bug |
|---|---|
| `run_autonomous.sh:108` | `touch "$done_marker"` unconditional â failed stage marked DONE |
| `run_autonomous.sh:87-98` | empty `common`/`harness` â queue script aborts instantly â B1 seals it |
| `run_cell_queue.sh:57` + `bo_experiment.py:1806` | truncated JSON skipped forever; `open(out,"w")` is non-atomic |
| `gen_stage_queue.py:140-148` | non-atomic PROVENANCE read-modify-write â permanent restart wedge |
| `summarize_stage.py:150-152` | non-atomic SUMMARY.json, rewritten ~15Ã/stage |
| `run_cell_queue.sh:65-69` | the build tool exit status never checked; bare `wait` always returns 0 |
| `summarize_stage.py:124-127` | cells with zero shards vanish from the summary entirely |
| everywhere | no lockfile, no signal trap |

---

## Q1 â `jobs -rp` throttle vs. `run_cell`'s bare `wait`

**Verdict: NOT a bug. The premise is wrong.** Precise trace:

1. `run_cell_queue.sh:79` â `run_cell "$name" "$flags" &`. Backgrounding a shell *function* makes bash `fork()` a subshell and run the function body there. Bash's `execute_in_subshell()` calls `without_job_control()` and clears the inherited jobs list, so the child starts with an **empty job table**.
2. `run_cell_queue.sh:65-67` â the 12 `the build tool run ... &` are children *of that subshell*, and are the only entries in its job table.
3. `run_cell_queue.sh:69` â bare `wait` waits for "all currently active child processes **of the invoking shell**". Sibling cells' shards are children of *different* subshells (different PIDs), so they are not waitable here â `wait` would return immediately with 127 if you tried. **No cross-cell contamination.**
4. `run_cell_queue.sh:82` â the parent's bare `wait` waits for the run_cell subshells, which transitively covers the shards. Correct.
5. `run_cell_queue.sh:77` â `$(jobs -rp | wc -l)` works in bash: `command_substitute()` deliberately preserves the jobs list (unlike `execute_in_subshell`), and the pipeline fork copies it. Concurrency is correctly capped at `CELLS_PARALLEL`.

Verify yourself: `bash -c 'f(){ sleep 5 & wait; }; sleep 99 & f & wait -n; echo $?'` returns after 5s, not 99s.

**But the code path around it has three real defects:**

**B16 (HIGH) â `wait` swallows every shard failure.** `run_cell_queue.sh:69`: bare `wait` returns 0 unless it was interrupted, *regardless of child exit codes*. Line 70 then prints `cell X done (7/12)` and nothing anywhere reads that `7`. Combined with B1, a cell that lost 5 shards is indistinguishable from a complete one.

```bash
# run_cell_queue.sh:69 â replace bare wait
  local rc=0 p
  for p in $(jobs -p); do wait "$p" || rc=1; done
  local have; have=$(find "$R/raw/$STUDY" -maxdepth 1 -name "${name}_s*.json" | wc -l)
  say "cell $name done ($have/$SHARDS)${rc:+ rc=$rc}"
  [ "$have" -eq "$SHARDS" ] || say "cell $name INCOMPLETE"
```

**B9 (MEDIUM) â stdin is the queue file.** `run_cell_queue.sh:81` attaches `$QUEUE` to the loop's stdin. That fd is inherited by the `run_cell` subshell (line 79) and by every `the build tool run` (line 65). `the build tool run` connects stdin to the executed binary. Any read advances the *shared* file offset and the outer `read` at line 74 **silently skips queue lines** â cells never run, and B1 then marks the stage DONE. Classic `ssh`-in-a-while-loop bug. Fix is two tokens:

```bash
# line 65-67: add </dev/null
    the build tool run @//mode/opt //$S:$HARNESS -- $COMMON $flags \
      --threads "$THREADS" $shard_flag --out "$out" \
      < /dev/null > "$R/logs/$STUDY/${name}_s${i}.txt" 2>&1 &
# line 74/81: read the queue on fd 3 instead of stdin
while IFS=$'\t' read -r -u 3 name flags; do ... done 3< "$QUEUE"
```
Same for `run_autonomous.sh:102` â `python3 summarize_stage.py` inherits the pipe from `run_cell_queue.sh` as stdin; add `< /dev/null`.

**B15 (LOW) â throttle degrades to nothing on bad input.** `run_cell_queue.sh:77`: if `CELLS_PARALLEL` is non-numeric, `[ -ge ]` errors â non-zero â loop never entered â **all 15 cells launch at once = 180 concurrent the build tool processes**. And on bash <4.3, `wait -n` errors immediately â infinite busy-loop. Add `case "$CELLS_PARALLEL" in ''|*[!0-9]*) exit 1;; esac` after line 48.

---

## Q2 â truncated JSON is a **real** data-integrity hole. Confirmed at the source.

`bo_experiment.py:1806-1807` (and `:2093`, `:2942`, `bo_diagnose.py:725`):

```python
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
```

`open(..., "w")` **truncates the target immediately**, then `json.dump(..., indent=2)` streams hundreds of KB through a buffered writer. So the failure window is not a narrow race â from the first instant of serialization until the last flush, `$out` exists on disk and is invalid JSON. A zero-byte `$out` is also a legal outcome.

`run_cell_queue.sh:57` â `[ -f "$out" ] && continue` â **nothing validates well-formedness, anywhere.** Downstream consequences:

- The corrupt shard is skipped on every future restart. Permanently lost.
- `summarize_stage.py:145` counts it: `shards_present = len(files)` â `shards_missing = 0` â the cell reports **complete**.
- `summarize_stage.py:138-144` â one bad file raises inside `summarise_bo`, so the `except` turns the **entire 12-shard cell** into `{"kind": "error"}`. Eleven good shards produce zero numbers, silently, forever.
- `run_autonomous.sh:108` then writes `.DONE`.

Note `summarize_stage.py:139` only probes `files[0]` for `traj_raw`; if *that* file is the corrupt one you get the same cell-wide error even though shards 1-11 are fine.

**Fix â two halves.** Atomic write in the harness (all four `json.dump` sites):

```python
import os, tempfile
tmp = args.out + ".partial"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, args.out)          # atomic within a filesystem
```

Plus a validating skip in `run_cell_queue.sh:57`, so the ~200 already-written legacy shards are also protected:

```bash
    if [ -f "$out" ]; then
      if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$out" 2>/dev/null; then
        continue
      fi
      say "CORRUPT shard, re-running: $out"
      mv -f "$out" "$out.corrupt.$(date -u +%s)"
    fi
```
(`--out "$out.partial"` in the shell + `mv` in the shell is *not* sufficient, because the shell can't know the harness finished cleanly without checking its exit status â do B16 too.)

---

## Q3 â the `| while read` subshell: yes, it breaks things, and **yes, failed stages get marked DONE**

`run_autonomous.sh:96-108`.

**Variable scope:** benign *today* â nothing is assigned inside the loop. But it's a loaded gun: the natural fix for B1 (setting `local ok=0` when a cell reports incomplete) would silently not work.

**Exit status: lost, and that is the critical bug.** The pipeline's status is the `while` loop's (pipefail would surface a non-zero left side, but *nobody reads `$?`*). Then line 108 runs unconditionally:

```bash
  python3 "$ABS/summarize_stage.py" ... || true     # 107: failure explicitly discarded
  touch "$done_marker"                               # 108: UNCONDITIONAL
  say "$tag COMPLETE"                                # 109: lies
```

**B2 (CRITICAL) â the realistic trigger.** Lines 87-94:

```bash
  common=$(python3 -c "... print(REGIMES['$regime']['flags'])")
```
No status check. If `python3` fails (import error, `SCRATCH_BO_ROOT` drift, transient), `common` is the **empty string**. `run_cell_queue.sh:44` is `COMMON="${COMMON:?set COMMON}"` â `:?` fires on *null as well as unset* â the queue script exits 1 in under a second â the `while read` loop sees a couple of lines on stderr and exits â **`touch .DONE`**. The stage is now permanently skipped with **zero shard JSONs on disk**, and STATUS.txt says `COMPLETE`. Re-running `run_autonomous.sh` will never recover it. Same for `harness` at 92-94, and for `cd` failure at `run_cell_queue.sh:38`.

**Fix (B1 + B2):**

```bash
  local common harness
  common=$(python3 -c "...") || { say "$tag: cannot read regime flags"; return 1; }
  [ -n "$common" ] || { say "$tag: empty COMMON, aborting stage"; return 1; }
  harness=$(python3 -c "...") || { say "$tag: cannot read harness"; return 1; }
  [ -n "$harness" ] || { say "$tag: empty HARNESS, aborting stage"; return 1; }

  set -o pipefail
  STUDY="v2/$tag" QUEUE="$queue" COMMON="$common" SHARDS="$shards" \
    HARNESS="$harness" THREADS="$THREADS" CELLS_PARALLEL="$CELLS_PARALLEL" \
    bash "$ABS/results/run_cell_queue.sh" 2>&1 | while read -r line; do ... done
  local qrc=${PIPESTATUS[0]}

  python3 "$ABS/summarize_stage.py" --stage "$tag" --expect-shards "$shards" || true

  # DONE only when the summary itself certifies completeness.
  if [ "$qrc" -eq 0 ] && python3 - "$ABS/results/raw/v2/$tag/SUMMARY.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
cells = s.get("cells", {})
exp = s.get("cells_expected") or 0
ok = len(cells) >= exp and all(
    c.get("kind") != "error" and not c.get("shards_missing") for c in cells.values())
sys.exit(0 if ok else 1)
PY
  then
    touch "$done_marker"; say "$tag COMPLETE"
  else
    say "$tag INCOMPLETE (qrc=$qrc) -- NOT marking DONE; next run resumes it"
    return 1
  fi
```
`${PIPESTATUS[0]}` is the only way to recover the left-hand status; note it must be captured on the very next line.

**B13 (LOW) â the read loop throttles the queue.** `run_autonomous.sh:101-104` runs `summarize_stage.py` *synchronously inside* the pipe reader. With ~180 shard JSONs of a few hundred KB each, that's seconds of JSON parsing per cell. While it runs, nothing drains the pipe; once 64 KB accumulates, `say` at `run_cell_queue.sh:78` blocks, and **the main queue loop stops launching new cells**. Run it detached: `( python3 ... >/dev/null 2>&1 & )` with a simple "already running" guard, or debounce to every Nth cell.

---

## Q4 â `set -uo pipefail` without `-e`

Defensible as a *choice* (you explicitly want a bad cell not to abort the queue â design goal #4), but it is not paired with the manual status checks that choice obligates. Concretely swallowed:

| site | swallowed |
|---|---|
| `run_autonomous.sh:107` | `\|\| true` â summarize failure discarded, then `.DONE` written anyway |
| `run_autonomous.sh:96-105` | queue-script exit status never read (B2) |
| `run_autonomous.sh:87-94` | `common`/`harness` capture status never read (B2) |
| `run_autonomous.sh:83` | `sed -n 's/^shards=//p'` yielding empty is not detected (B14) |
| `run_cell_queue.sh:69` | every `the build tool` exit code (B16) |
| `run_cell_queue.sh:65-67` | the build tool build failure is indistinguishable from a run failure; only the log file knows |

`pipefail` is essentially inert here: the only pipelines are `run_autonomous.sh:58` (inside `if`, so it just reads as false) and `:98` (status discarded). `set -u` is doing useful work at `run_cell_queue.sh:42-44`, but B2 shows those `:?` aborts are currently *silent*.

**B14 (MEDIUM) â a fabricated-statistics path.** `run_autonomous.sh:83`: if the sed yields empty, `SHARDS=""` is exported; `run_cell_queue.sh:45`'s `${SHARDS:-12}` treats null as unset and **defaults to 12**. For `lcb_bo`/`lcb_reg`/`pd1_reg` (declared `shards: "1"` at `gen_stage_queue.py:39,61,67`) that launches 12 shards. `--loo-shard` is a no-op on the lcbench path, so you get **12 byte-identical runs written to 12 filenames**, which `summarise_bo` (line 65-68) blindly concatenates â `n_runs` inflated 12Ã â SEMs shrunk by â12 on duplicated data. Meanwhile `--expect-shards ""` makes argparse `SystemExit(2)`, hidden by `> /dev/null 2>&1 || true` at line 103. Fix: `[ -n "$shards" ] || { say "$tag: no shards= in generator output"; return 1; }` after line 83, and drop the `:-12` default in favour of `SHARDS="${SHARDS:?}"`.

**B12 (MEDIUM) â the capability probe cannot detect what it claims.** `run_autonomous.sh:58` greps `--help` output for the *substring* `--benchmark`. But `bo_diagnose.py:263` contains the help text `"PD1 release to read when --benchmark pd1_loo"` on the `--pd1-source` option. If `--benchmark` were ever removed, the probe would still pass and every `pd1_reg` shard would die on an unknown-flag argparse error. Conversely, a **build failure** (stderr â `/dev/null`) reports as "stage 0.2 not built" and silently drops an entire regime from the plan. Fix: `grep -qE '^[[:space:]]+--benchmark[[:space:], ]'` and separate the build step (`the build tool build`) from the probe so a build error is reported as an error.

---

## Q5 â resumability traced end-to-end

### Shard level â partially works
- â Missing file â re-run. Correct.
- â Crash before any write â no file â re-run. Correct.
- â **B3**: crash *during* write â corrupt file â skipped forever.
- â **B8 (MEDIUM): the cell name is the only cache key.** `run_cell_queue.sh:56` keys on `${name}_s${i}.json`. Nothing hashes `$COMMON`, `$flags`, or the commit. Edit `REGIMES['pd1_bo_armC']['flags']` or an `OFAT` entry (`gen_stage_queue.py:33-104`) and restart: existing shards are reused under the new name and **silently merged with new shards run under different settings**. Worse, `gen_stage_queue.py:120-149` *overwrites* `PROVENANCE.json` with the new flags while keeping only `commit`/`generated_utc` in `history` (line 145) â so the provenance file actively misrepresents what produced the directory. Fix: put a short hash of `COMMON+flags` in the filename (`${name}_${cfghash}_s${i}.json`), or at minimum store `regime_flags` + `cells` in each history entry and refuse to reuse shards whose recorded flags differ.
- â ï¸ `MANIFEST.tsv` (`run_cell_queue.sh:62-63`) records intent *before* the run and omits `--threads`/`--loo-shard`/`--out`; skipped shards write nothing, so the manifest under-reports on resume.

### Cell level â **does not exist**
Header claim `run_autonomous.sh:11` says "cells skip if complete". There is no cell-level completion check anywhere; `run_cell` always spins its 12-iteration loop and relies on the shard check (cheap, but the doc is wrong). More importantly:

- â **B11 (MEDIUM): a cell whose shards all failed is invisible.** `summarize_stage.py:124-127` builds the cell set from `glob("*_s*.json")` â from **what is on disk**, never from `prov["cells"]`. A cell with zero outputs is absent from `summary["cells"]` entirely, so `shards_missing` never fires for it. Only `cells_present` vs `cells_expected` (line 133-134) would reveal it, and **nothing reads those fields**. Fix:

```python
expected = [c["name"] for c in prov.get("cells", [])]
for name in sorted(set(cells) | set(expected)):
    files = cells.get(name, [])
    if not files:
        summary["cells"][name] = {"kind": "missing", "shards_present": 0,
                                  "shards_missing": args.expect_shards}
        continue
    ...
```

- â **B10 (MEDIUM): silent numeric corruption on heterogeneous shards.** `summarize_stage.py:60-71` accumulates `pool_max` once per file but `traj[m]` **per method key found in that file** (`for m, rows in pr["traj_raw"].items()`). `analyze.py:157-164` gets this right by iterating a *fixed* method list. If any shard's `traj_raw` lacks a method (partial run, or shards from a run with a different `--methods`), the per-method row lists desynchronize from `pool_max`, and line 71 / `budget_to_target:46` index `pool_max[r]` against the **wrong run**. That is a wrong number, not a crash â the worst kind. Fix: validate per file and reject the shard.

```python
        rows_by_m = pr["traj_raw"]
        n = len(pr.get("pool_max_raw", []))
        if not rows_by_m or any(len(v) != n for v in rows_by_m.values()):
            skipped.append(path); continue
        if traj and set(rows_by_m) != set(traj):
            skipped.append(path); continue
```
and surface `skipped` in the emitted dict so it is visible in `SUMMARY.json`.

### Stage level â the resumability claim is **false**
- â **B1/B2**: `.DONE` is unconditional â *skips incomplete work*, permanently, with no recovery path short of `rm`. This is the headline bug and it directly negates `run_autonomous.sh:11-13`.
- â **B4 (HIGH): permanent wedge via corrupt PROVENANCE.** `gen_stage_queue.py:140-148` does a read-modify-write with no atomicity: `json.load` (141-143) â `json.dump` (147-148) straight onto the live path. A crash mid-dump leaves truncated JSON. On the *next* run, line 143's `json.load` raises â uncaught â non-zero exit â `run_autonomous.sh:81` says "queue generation FAILED" and returns 1 â forever, on every restart, for that stage. `existing.append({k: prior[k] ...})` at line 145 raises `KeyError` on any older-schema file with the same effect. Fix:

```python
    prior = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                prior = json.load(f)
        except (json.JSONDecodeError, OSError):
            os.replace(path, path + ".corrupt")
            prior = {}
    existing = prior.get("history", [])
    if prior:
        existing.append({k: prior.get(k) for k in
                         ("commit", "generated_utc", "regime_flags")})
    prov["history"] = existing
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prov, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
```

- â **B5 (HIGH): non-atomic SUMMARY.json, written ~15Ã per stage.** `summarize_stage.py:150-152`. The whole point of the mid-stage re-summarize (`run_autonomous.sh:100-104`) is crash safety, but the crash-safety artifact is itself written non-atomically, and it is rewritten after *every cell*, so the window is hit repeatedly. Design goal #3 has stage 2/3 selecting cells from these files â they'd crash on a truncated one. Same `tmp` + `os.fsync` + `os.replace` fix.
- â **B6 (HIGH): no lock.** Nothing prevents two live `run_autonomous.sh` (systemd restart while the old unit lingers, or a manual run alongside the unit). Both regenerate `/tmp/queue_${tag}.tsv` and both launch the *same* `--out` paths, since `[ -f "$out" ]` is false for both until the first write. Two processes `open(out,"w")` on the same path â interleaved garbage that is nonetheless a plausible-looking file. Add at the top of `run_autonomous.sh`:

```bash
exec 9> /tmp/em_autonomous.lock
flock -n 9 || { echo "another run_autonomous.sh holds the lock; exiting"; exit 1; }
```

- â **B7 (HIGH): no signal trap â orphan writers.** `run_cell_queue.sh` installs no `trap`. If the script or a `run_cell` subshell dies (OOM killer, Ctrl-C on a manual run), the `the build tool run` grandchildren are reparented and **keep running and keep writing `$out`**. The parent's `wait` (line 82) returns, `run_autonomous` summarizes files that are mid-write, and stamps `.DONE`. Under `systemd-run --collect` the cgroup teardown covers this; for any manual invocation it does not. Add to `run_cell_queue.sh` after line 51:

```bash
trap 'say "signal received, killing children"; trap - TERM INT; kill 0' TERM INT
```

---

## Context

The design intent in `run_autonomous.sh:9-22` is sound; the gap is that **every completeness signal the pipeline computes is discarded before it reaches a decision.** `run_cell` computes `N/$SHARDS` (line 70) and prints it. `summarize_stage.py` computes `shards_missing` (line 147) and `cells_expected` (line 133) and writes them. `run_block` computes a pipeline exit status. Not one of those three is read by the code that decides whether to write `.DONE`. Wiring `summarize_stage.py`'s own verdict into line 108 (the B1 fix) closes B1, B2, B11 and most of B16 in one change, and is the single highest-value edit.

Fix order: **B1+B2 â B3 â B4+B5 â B6+B7 â B16+B9 â B10+B11 â the rest.**

## Next Steps

1. `find results/raw/v2 -name '*_s*.json' -size -1k` and run a `json.load` sweep over all existing shards â B3 may already have poisoned cells in the current output tree, and the `SUMMARY.json` entries with `"kind": "error"` will tell you which.
2. `ls results/raw/v2/*/.DONE` cross-referenced against each stage's `SUMMARY.json` `cells_present`/`shards_missing` â any `.DONE` beside a short cell list is a stage B1 already lost.
3. `results/run_overnight_driver.sh` and `run_rerun_all.sh` are the predecessors of this driver; worth checking whether they share the unconditional-marker and `[ -f ]`-skip patterns before you reuse them.",
  "subagentUsed": "code_search",
  "agent_id": "subagent-c2ef604a-a813-45a3-ba2d-f7b344f20a35

==============================================================================
Adversarial review: metric and analysis correctness
==============================================================================
## Direct Answer

I found **6 concrete defects**, one of which corrupts every headline number in the "POOLED ARM C" section (the primary result). Your suspicion in Q1 was aimed at the wrong function â the censoring values *do* agree; the silent inconsistency is elsewhere (two different estimators sharing the name `budget_to_target`). Q2/Q3 are real latent bugs. Q4 is a **confirmed, currently-active** bug with data on disk to prove it. Q5's direction logic is correct on all four paths, but the *win-count* line right underneath it is direction-asymmetric on ties.

---

## Key Locations

| # | Severity | File:line | Defect |
|---|---|---|---|
| 1 | **CRITICAL** | `analyze_pd1pool.py:114-118` (used at `:571-573`, `:433`) | `PooledStudy` pools p0 (5 seeds) with p1/p2 (3 seeds) â p0 gets 45.5% weight, not 33% |
| 2 | **HIGH** | `summarize_stage.py:65-71` | `pool_max`/`traj` misalign silently if any shard lacks a method |
| 3 | **MEDIUM** | `analyze_pd1pool.py:194-205` | Ties counted as wins for `a` in the lower-is-better branch only |
| 4 | **MEDIUM** | `summarize_stage.py:38` vs `analyze.py:781` vs `analyze_pd1pool.py:137` | Three different estimators, two sharing the name `budget_to_target` |
| 5 | **MEDIUM** | `analyze_pd1pool.py:219-237`, `:626-635` | Power analysis never run on the primary metric |
| 6 | **LOW-MED** | `analyze_pd1pool.py:183-185`, `:402-404` | Singleton clusters contribute to the effect but not to the SE â inflated \|z\| |

---

## Q1 â Censoring values: they DO agree. The real inconsistency is the estimator.

`_scratch_bo/summarize_stage.py:42-43`
```python
horizon = len(row)      # T
hit = horizon           # censored: one past the last index
```
`_scratch_bo/analyze_pd1pool.py:149`
```python
first[~hit.any(axis=1)] = R.shape[1]   # also T
```

`Study.regret[m]` is built as `pool_max[:, None] - T` from the very same `traj_raw` rows (`analyze.py:198-199`), so `R.shape[1] == len(row)`. Both censor at **T = n_iters + 1 = 51**. Verified against `bo_experiment.py:1301-1306` ("Length is `n_iters+1`"). **No bug here.** The sign conventions also compose correctly (`tol=+0.01` with `pool_max - v <= tol` vs `tol=-0.01` with `R <= -tol`).

**But the numbers still won't match**, because there are three different statistics in play:

| function | statistic | non-solvers |
|---|---|---|
| `summarize_stage.budget_to_target` (`summarize_stage.py:38`) | **mean** over all runs | censored at T |
| `analyze.budget_to_target` (`analyze.py:781`) | **median** over solved runs | **dropped** |
| `analyze_pd1pool.budget_censored` (`analyze_pd1pool.py:137`) | per-run vector, `.mean()` by callers | censored at T |

`analyze_pd1pool.py:547` calls the **imported** `analyze.budget_to_target` for the per-arm "BUDGET TO TARGET" table â the solved-only median. That is precisely the estimator `.llms/rules/metrics.md` warns about ("a method can look fast by solving only the easy runs"), and it is printed under the heading of the primary metric, one screen above the `budget_censored`-based tests that use a different estimator. `SUMMARY.json`'s `budget_to_0.01` will not equal either.

**Fix:** rename to make the estimator explicit and stop the name collision.
```python
# analyze.py:781
def budget_to_target_solved_median(study, method, tol) -> tuple[float, float, float]:
# summarize_stage.py:38
def censored_mean_budget(traj, pool_max, tol) -> float:
# analyze_pd1pool.py:547 â use the censored mean, the policy's metric:
mean_evals = budget_censored(s, m, tol).mean()
rate = solved_indicator(s, m, tol).mean()
```
Also note both report **acquisition rounds**, excluding the `n_init=3` initial evaluations (`bo_experiment.py:1731-1734`). Consistent between the two, but "evaluations to target" is understated by 3. For the `batch_q>1` result files the trajectory is indexed by *round*, so `budget_censored` would be off by a factor of `q` if ever pointed at them (`bo_experiment.py:1311-1312`).

---

## Q2 â Ordering IS consistent; **method-set heterogeneity is not**. Real bug.

`_scratch_bo/summarize_stage.py:59-68`
```python
for path in files:                          # files from sorted(glob(...)) â deterministic
    ...
    pool_max += pr.get("pool_max_raw", [])  # grows for EVERY file
    ds_idx   += pr.get("eval_dataset_idx", [])
    for m, rows in pr["traj_raw"].items():
        traj[m] += rows                     # grows only for files that HAVE m
```

File order is fine (`summarize_stage.py:125`, `sorted(glob(...))`; the same `files` list drives every method). **The alignment invariant that actually breaks is per-method presence.**

`pool_max` advances once per *file*; `traj[m]` advances once per *file that contains `m`*. If shard `s0` lacks method `m` but `s1..s11` have it, then `traj[m][0]` is `s1`'s first run while `pool_max[0]` is `s0`'s pool max â **every regret and every budget for that method is computed against the wrong task's optimum**, and it fails silently (`defaultdict(list)` never raises).

This is not hypothetical in this codebase:
- `analyze_pd1pool.py:105-106` documents exactly this: *"the p1/p2 replicates dropped PACOH."*
- `bo_experiment.py:1830-1838` actively filters the method list per harness (`methods = [m for m in methods if m in allowed]`) and only *prints* a warning.

Contrast with `analyze.Study` (`analyze.py:153`, `:164`), which pins `self.methods` from `raws[0]` and does `pr["traj_raw"][m]` â **KeyError, loud crash**. The summarize path is the silent one.

**Fix** â carry `pool_max` per method, or hard-fail on a heterogeneous method set:
```python
# summarize_stage.py:56-68
traj: dict[str, list[list[float]]] = defaultdict(list)
pmax: dict[str, list[float]] = defaultdict(list)   # per-method, so it cannot skew
ds_idx: list[int] = []
for path in files:
    ...
    pm = pr["pool_max_raw"]                        # NOT .get(..., []) â must be present
    if len(pm) != len(pr["eval_dataset_idx"]):
        raise ValueError(f"{path}: pool_max_raw/eval_dataset_idx length mismatch")
    ds_idx += pr["eval_dataset_idx"]
    for m, rows in pr["traj_raw"].items():
        if len(rows) != len(pm):
            raise ValueError(f"{path}: {m} has {len(rows)} rows, {len(pm)} pool maxima")
        traj[m] += rows
        pmax[m] += pm
```
Then use `pmax[m]` instead of the shared `pool_max` at lines 71, 74, 75.

The `.get("pool_max_raw", [])` / `.get("eval_dataset_idx", [])` defaults at `:65-66` are the wrong choice for a correctness-critical alignment key â a missing key silently shifts every subsequent index rather than failing.

---

## Q3 â `pool_max` is **not** guaranteed the same length as `rows`. Confirmed.

`_scratch_bo/summarize_stage.py:71`
```python
final = [pool_max[r] - rows[r][-1] for r in range(len(rows))]
```
Two failure modes:
- `len(rows) < len(pool_max)` (a shard lacks method `m`): loops over a **prefix** of `pool_max` â wrong pool maxima, **no error**. This is the dangerous case.
- `len(rows) > len(pool_max)`: `IndexError`, caught by the blanket `except Exception` at `summarize_stage.py:143` â the whole cell degrades to `{"kind": "error"}`. Loud, but it silently discards a cell that could have been partially summarized.

Same indexing pattern inside `budget_to_target` (`summarize_stage.py:46`, `pool_max[r]`), reached via `:74-75`. Fixed by the per-method `pmax` change above; additionally add an explicit guard:
```python
if len(rows) != len(pmax[m]):
    raise ValueError(f"{m}: {len(rows)} rows vs {len(pmax[m])} pool maxima")
```

---

## Q4 â CRITICAL, and it is firing right now on data in the repo.

`_scratch_bo/analyze_pd1pool.py:114-118`
```python
self.ds_idx = np.concatenate([s.ds_idx for s in subs])
self.regret = {m: np.concatenate([s.regret[m] for s in subs], axis=0) for m in self.methods}
```

Verified on disk:

| pattern | `n_seeds` | runs |
|---|---|---|
| `results/raw/pd1pool/fullboth_s0.json` (p0) | **5** | 23Ã5 = 115 |
| `results/raw/pd1pool/fullboth_p1_s0.json` | **3** | 23Ã3 = 69 |
| `results/raw/pd1pool/fullboth_p2_s0.json` | **3** | 23Ã3 = 69 |

The docstring at `:101-105` asserts *"Each pre-training prior is an independent replicateâ¦ arm C's three priors are pooled"* â clearly intending equal weight per prior. In fact **p0 receives 115/253 = 45.5%** of the weight and p1/p2 receive 27.3% each. p0 is over-weighted by **1.67Ã**.

This is not fixed by task-clustering, because the imbalance is *within* every task group: `fixed_effects_test`'s `per_task = g.mean()` (`:182`) and `clustered_sem` (`:130-134`) both average over the 11 runs of a task, of which 5 are p0's. Every number in the `POOLED ARM C` block (`:574-615`), every `variance_decomposition` verdict (`:620-635`), and â worst â every **primary-metric** `fixed_effects_test` (`:659-666`) inherits the bias. `hybrid_analysis()` at `:433` has the identical 5/3/3 split (`hybrid_p0_s0.json` n_seeds=5, `hybrid_p1_s0.json` / `hybrid_p2_s0.json` n_seeds=3).

Additional consequence: `within = np.mean([g.var(ddof=1) ...])` (`:189`, `:243`) is a *pooled within-task* variance that now mixes three priors â so it contains **between-prior** variance, inflating `within`, deflating `between`, and biasing the "UNREACHABLE vs reachable" verdicts toward *reachable* (i.e. "spend more compute").

**Fix** â average within prior first, then across priors (equal weight per replicate), and record a prior id:
```python
class PooledStudy:
    def __init__(self, patterns):
        subs = [load(p) for p in patterns]
        ...
        self.prior_idx = np.concatenate(
            [np.full(s.ds_idx.shape[0], i) for i, s in enumerate(subs)]
        )
        n = {int(i): int((self.prior_idx == i).sum()) for i in np.unique(self.prior_idx)}
        if len(set(n.values())) > 1:
            print(f"WARNING: unbalanced priors {n}; using prior-balanced weights")

    def task_means(self, values):
        """Per-task mean, equal weight per prior (not per run)."""
        out = []
        for t in np.unique(self.ds_idx):
            per_prior = [
                values[(self.ds_idx == t) & (self.prior_idx == p)].mean()
                for p in np.unique(self.prior_idx)
                if ((self.ds_idx == t) & (self.prior_idx == p)).any()
            ]
            out.append(float(np.mean(per_prior)))
        return np.array(out)
```
Then route `fixed_effects_test`, `variance_decomposition`, `clustered_sem` and the ranking table at `:577-580` through `task_means`. Until this lands, **no pooled arm-C number should go in the paper.**

`PooledStudy` also silently assumes equal trajectory length `T` across priors (it happens to hold: `n_iters=50` everywhere) â add an explicit assert so a future prior with a different budget crashes instead of raising an opaque numpy error.

---

## Q5 â The `higher_is_better` flag is correct on all four paths. The **win count** underneath it is not.

`_scratch_bo/analyze_pd1pool.py:171-206`

| path | condition | `higher_is_better` | correct? |
|---|---|---|---|
| budget-to-target | `tol is not None and metric=="budget"` (`:171`) | `False` | â lower is better |
| solve-rate | `tol is not None`, other metric (`:173`) | `True` | â higher is better |
| regret-AUC | `tol is None and metric=="auc"` (`:175`) | `False` | â lower is better |
| final regret | else (`:177`) | `False` | â lower is better |

`favours_a`/`favours_b` at `:206-207` are also correct. **Q5 passes.** Two latent traps though:
- `fixed_effects_test(pooled, a, b, tol=-0.01, metric="auc")` would silently compute the *solve-rate*, since `:173` tests only `tol is not None`. No caller does this today. Guard it: `raise ValueError` on `tol is not None and metric not in ("budget", "solve")`.
- `metric="budget"` with `tol=None` silently falls through to final regret at `:177`.

**The actual bug is at `:194` + `:204-205`:**
```python
wins = int((per_task > 0).sum())
...
if not higher_is_better:
    wins = len(tasks) - wins        # == count(per_task <= 0), i.e. ties count for `a`
```
For `higher_is_better=True` (solve-rate), ties (`per_task == 0`) count for **neither**. For `higher_is_better=False` (budget, regret, AUC), `len(tasks) - count(>0)` = `count(<= 0)`, so ties count for **`a`**. Exact ties are common on the primary metric â any task where both methods are censored on all 11 runs gives `per_task == 0` exactly, which is the normal case for the tight `-0.001` target on hard tasks. The printed line `tasks where <a> wins n/23` (`:212`) is therefore inflated for the metric you lead with, and it is a *direction-dependent* inversion â exactly the failure class `metrics.md` says has already bitten twice.

**Fix:**
```python
# analyze_pd1pool.py:194, 204-205
a_wins = int((per_task > 0).sum()) if higher_is_better else int((per_task < 0).sum())
ties   = int((per_task == 0).sum())
...
print(f"    tasks where {a[:14]:14s} wins  {a_wins}/{len(tasks)}  ({ties} tied)")
```

---

## Q6 â Raw-vs-regret: no inversions found. Two related issues instead.

Every metric is computed on regret, correctly:
- `summarize_stage.py:47` `pool_max[r] - v <= tol` â; `:71` `pool_max[r] - rows[r][-1]` â (matches `bo_experiment.py:1771`).
- `analyze.py:199` `self.regret[m] = self.pool_max[:, None] - T` â.
- `analyze.py:804` `tol * R[:, 0]` and `analyze.py:494` `eps = 1e-3 * regret[methods[0]][:, 0]` use one method's initial regret for all â **benign**, because `init_idx` is drawn once per `(task, seed)` and shared across methods (`bo_experiment.py:1730-1733`), so `R[:,0]` is identical across methods.
- `summarise_bo`'s `solve_rate_0.01` (`summarize_stage.py:77`) uses only *final* regret while `solved_indicator` (`analyze_pd1pool.py:86`) uses `.any(axis=1)` â **equivalent**, since `traj_raw` is best-so-far monotone (verified in `results/raw/pd1pool/hybrid_p0_s7.json`).

Two genuine problems in the same family:

**(5) The power analysis never touches the primary metric.** `variance_decomposition` (`:219-237`) only supports `final regret` (`tol=None`) and `solve-rate` (`tol` given) â there is no budget path. Callers at `:626` and `:634-635` use exactly those two. So the `VERDICT: UNREACHABLE` lines that decide *whether to spend an overnight compute slot* are computed on metrics ranked #2 and #3, never on the #1 metric. Add a `metric` parameter mirroring `fixed_effects_test`:
```python
def variance_decomposition(pooled, a, b, tol=None, metric="final"):
    if tol is not None and metric == "budget":
        d = budget_censored(pooled, a, tol) - budget_censored(pooled, b, tol)
    elif tol is not None:
        d = solved_indicator(pooled, a, tol) - solved_indicator(pooled, b, tol)
    ...
```

**(6) Singleton clusters inflate \|z\|.** `analyze_pd1pool.py:183-185`:
```python
var_terms = [g.var(ddof=1) / g.size for g in groups if g.size > 1]
effect = float(per_task.mean())      # includes size-1 groups
se = math.sqrt(sum(var_terms)) / len(tasks)
```
A size-1 task contributes to the numerator but zero to the SE, so `se` is understated and `z` overstated. Currently masked in pooled arm C (11 runs/task), but `drift_comparison` replicates the identical arithmetic at `:402-404` on `results/raw/drift_check/rerun_s*` â any task with one surviving run there produces a spuriously significant `<-- VERDICT CHANGED`. Either drop singleton tasks from `per_task` too, or impute the pooled within-task variance for them.

---

## Context

- `results/raw/v2/` is **empty**, so the `summarize_stage.py` bugs (Q2/Q3) are latent â they will fire the first time a stage runs with a heterogeneous method set. `tests/test_experiment_setup.py:192-223` covers censoring and direction well but has **zero** multi-shard / missing-method cases; that is exactly the gap.
- `eval_dataset_idx` is **globally** indexed (shard `s3` writes tasks `3` and `15`, `s7` writes `7` and `19`), so cross-shard concatenation of `ds_idx` is safe. Good.
- Shards are unequal in size (23 tasks / 12 shards â `s11` has one task). `summarise_regression` (`summarize_stage.py:101-103`) equal-weights *shards*, not tasks â currently harmless because both regression regimes use `shards: 1` (`gen_stage_queue.py:61`, `:66`), but it will silently mis-weight the moment a regression regime is sharded.
- `summarize_stage.py:126` `rsplit("_s", 1)` survives all current OFAT cell names (including `shrink_0.1`), but is fragile to any future cell name containing `_s`.

## Next Steps

1. **Block on Q4.** Re-run `analyze_pd1pool.py` with prior-balanced weighting before any arm-C number is quoted; the direction of the shift is unknown a priori because p0 differs from p1/p2 in more than seed count.
2. Add a regression test for Q2/Q3: two synthetic shards where the second omits one method, assert that `summarise_bo` either raises or produces per-method-correct regret.
3. Add the `metrics.md` cross-check the policy itself mandates: assert `budget_censored(...).mean()` from `analyze_pd1pool` equals `SUMMARY.json`'s `budget_to_*` on a shared fixture â that single assertion catches Q1, Q2 and Q3 at once.
4. Grep for other `PooledStudy` / `Study(",".join(...))` sites that may mix unequal `n_seeds`; `results/raw/hb_hybrid/bo_lcb_*` (`analyze_pd1pool.py:473`) is the next one to check.",
  "subagentUsed": "code_search",
  "agent_id": "subagent-08c500ee-efcc-4470-a575-221e199d6743

==============================================================================
Adversarial review: flag-to-model integration
==============================================================================
# Adversarial Integration Review: `bo_experiment.py` / `bo_diagnose.py` â `pretrain_em_prior`

**â  Note on file state:** `_scratch_bo/bo_diagnose.py` was **modified externally mid-review** (it grew from 668 â 732 lines). Everything below reflects the **current on-disk state**, which I re-read. The pre-edit version had *none* of the new EM knobs threaded; that gap is now largely closed, but **three defects remain**, one of which is a hard crash.

---

## Direct Answer

There are **four live silent no-ops and two hard-crash mis-wirings** remaining:

| # | Defect | Severity |
|---|---|---|
| 1 | `--em-mean` / `--em-mean-weight` silently ignored on the **lcbench** path of `bo_experiment.main()` | **SILENT NO-OP** |
| 2 | All six EM knobs silently ignored in **`run_pd1_full`** | **SILENT NO-OP** |
| 3 | `--em-canonical-pool` passes the PD1 guard for `--benchmark pd1_full` but is never read | **SILENT NO-OP** |
| 4 | `--em-mean` ignored in `_pd1_probe` | SILENT NO-OP (diagnostic only) |
| 5 | `bo_diagnose` line 579 `float(mean_s.constant)` â `AttributeError` under `--em-base-mean linear` or any `--em-mean` | **RAISES (blocks 4 OFAT cells)** |
| 6 | `make_surrogate`'s `pretrained_gp_frozen` reads `mean_s.constant` â `AttributeError` under `--em-base-mean linear` / `--em-mean` | **RAISES / silently drops a baseline in `bo_diagnose`** |

---

## Q1 â Per-flag trace, `bo_experiment.py`, pd1_loo vs lcbench

Call sites: pd1_loo = `run_pd1_matched_loo` L1659â1678. lcbench = `main()` body L2693â2710.

| Flag | pd1_loo | lcbench | Evidence |
|---|---|---|---|
| `--em-base-mean` | **REACHES** | **REACHES** | `bo_experiment.py:1659-1661` / `:2693-2695` â both pass `base_mean=args.em_base_mean` into `build_shared_kernel` (`:511-517`, `:541`) |
| `--em-mean` | **REACHES** | **SILENTLY IGNORED** | `_apply_em_mean` called only at `:1665`. Nothing between `:2693` and `:2710`. `args.em_mean` is read at exactly 2 places (`:1619` want_hyperbo gate, `:367` inside `_apply_em_mean`) â both pd1_loo-only |
| `--em-mean-weight` | **REACHES** | **SILENTLY IGNORED** | Consumed only inside `_apply_em_mean` (`:375`) |
| `--iw-nu` | **REACHES** | **REACHES** | `:1674` / `:2706`. Guarded at `:2528-2534` (raises without `--use-covar-prior`) |
| `--use-mean-prior` | **REACHES** | **REACHES** | `:1675` / `:2707` |
| `--use-covar-prior` | **REACHES** | **REACHES** | `:1676` / `:2708` |
| `--em-init-mode` | **REACHES** | **REACHES** | `:1677` / `:2709` |
| `--em-canonical-pool` | **REACHES** | **RAISES (correctly)** | Read only at `:1586`. Listed in the `pd1_only` argv guard at `:2548`, so LCBench + this flag â `ValueError` at `:2554` |

### Defect 1 â `--em-mean` on lcbench is a silent no-op

`bo_experiment.py:2693-2710` builds `mean_s` and hands it straight to `pretrain_em_prior` with **no `_apply_em_mean` call**, and `--em-mean` is **not** in the `pd1_only` guard tuple (`:2542-2549`). A HyperBO prior *is* available on this path (`hyperbo_prior_early`, `:2679`, when `--em-canonical hyperbo`) â the wiring simply isn't there.

Impact via `gen_stage_queue.py`: regime `lcb_bo` (`:34-40`) crosses the full `OFAT` list, which contains `("meanxfer_full", "--em-mean hyperbo")` (`:86`), `("meanxfer_blend", ...)` (`:87`) and `("mean_linear+meanxfer", ...)` (`:103`). **Three LCBench-BO cells would run bit-identical to `base` while being filed as mean-transfer results.** This is exactly the Â§26.4 failure mode the `pd1_only` guard was written to stop.

**Fix:** add `mean_s = _apply_em_mean(mean_s, args, hyperbo_prior_early)` after `bo_experiment.py:2697`, and extend the `want_hyperbo`-style logic so `--em-mean` forces the early HyperBO pre-train â or, minimally, add `--em-mean`/`--em-mean-weight` to a "pd1_loo-only" guard.

---

## Q2 â Does `bo_diagnose` thread the new EM knobs? **YES, now.** (It did not before this session's external edit.)

`bo_diagnose.py:409-422`:
```
409| em_prior = pretrain_em_prior(
...
418|     iw_nu=args.iw_nu,
419|     use_mean_prior=args.use_mean_prior,
420|     use_covar_prior=args.use_covar_prior,
421|     init_mode=args.em_init_mode,
422| )
```
Flags declared at `:224-247`. `_apply_em_mean` called at `:408`. The `--iw-nu`-without-`--use-covar-prior` guard is mirrored at `:288-292`.

**Verdict per flag in `bo_diagnose` (both `lcbench` and `pd1_loo`, which share one code path via `_load_benchmark` at `:97-123`):**

| Flag | Status |
|---|---|
| `--em-base-mean` | REACHES `build_shared_kernel` (`:400-402`) â **but see Defect 5, the run then crashes** |
| `--em-mean` / `--em-mean-weight` | REACHES `_apply_em_mean` (`:408`) â **but RAISES unless `--em-canonical hyperbo`; see below** |
| `--iw-nu` | REACHES (`:418`), guarded (`:288`) |
| `--use-mean-prior` | REACHES (`:419`) |
| `--use-covar-prior` | REACHES (`:420`) |
| `--em-init-mode` | REACHES (`:421`) |
| `--em-canonical-pool` | **Not defined** â argparse `unrecognized arguments` (hard fail, acceptable) |

### Residual behavioural divergence â `--em-mean` needs different flags in the two harnesses

`bo_diagnose.py:408` passes `hyperbo_prior_early`, which is `None` unless `--em-canonical hyperbo` (`:385`). The `hyperbo_prior` trained for the `hyperbo_*` **methods** is built *later*, at `:482-499`, and is never fed back.

By contrast `bo_experiment.py:1614-1621` has `want_hyperbo` include `args.em_mean != "empirical"`, so on pd1_loo `--em-mean hyperbo` works **standalone**.

Consequence: OFAT cells `meanxfer_full`, `meanxfer_blend`, `mean_linear+meanxfer` (`gen_stage_queue.py:86,87,103`) **raise** on the `lcb_reg` / `pd1_reg` regimes. Loud, not silent â but it means the regression half of the matrix cannot execute those cells, and the only workaround (`--em-canonical hyperbo`) confounds the mean-transfer factor with a canonical-kernel change, destroying OFAT.

**Fix:** hoist HyperBO pre-training in `bo_diagnose.main()` above `:400` whenever `args.em_mean != "empirical"`, mirroring `bo_experiment.py:1614-1621`.

---

## Q3 â Does `bo_diagnose`'s `build_shared_kernel` pass `base_mean`? **YES, now** â but the run then dies

`bo_diagnose.py:400-402`:
```
400| mean_s, covar_s = build_shared_kernel(
401|     pretrain_ds, deep_hidden=args.deep_kernel, base_mean=args.em_base_mean
402| )
```

### Defect 5 â `--em-base-mean linear` crashes `bo_diagnose` at reporting time

`bo_diagnose.py:579`:
```
579|         "mean_constant": float(mean_s.constant),
```
Unguarded. `build_shared_kernel` returns `LinearMean(input_size=d)` when `base_mean == "linear"` (`bo_experiment.py:541`), and `gpytorch/means/linear_mean.py:13-17` registers only `weights` and `bias` â **no `constant`**. `_BlendedMean` (`bo_experiment.py:303-334`) likewise has no `constant`.

So `--em-base-mean linear`, `--em-mean hyperbo`, `--em-mean blend`, and `mean_linear+covprior` / `mean_linear+meanxfer` all abort with `AttributeError` at `:579` â **after** paying for HyperBO/EM pre-training, and before any table is emitted. That is 5 of the 14 OFAT cells Ã 2 regression regimes.

Note `:569-578` already guards `lengthscales` and `outputscale` with `getattr(..., None)` â the mean field was simply not given the same treatment.

**Fix:** `"mean_constant": (float(mean_s.constant) if getattr(mean_s, "constant", None) is not None else None)`, and make the `print` at `:585` tolerate `None`.

### Defect 6 â `pretrained_gp_frozen` also reads `mean_s.constant`

`bo_experiment.py:838-843`:
```
841|             mm = ConstantMean()
842|             with torch.no_grad():
843|                 mm.constant.data.copy_(mean_s.constant.data)
```
Same `AttributeError` under `--em-base-mean linear` / `--em-mean`. Behaviour differs by harness:
- **`bo_experiment`** run loops call `surrogate_fn` unguarded (`:1225`, `:1283`, `:1376`) â loud abort. Acceptable.
- **`bo_diagnose`** wraps `build()` in `except Exception` at the prior-mean stage (prints `unavailable`) and, worse, at the `by_n_obs` stage uses a bare `except Exception: continue` with **no message** â `pretrained_gp_frozen` **silently vanishes from the results table** rather than erroring. A missing baseline row is precisely the kind of thing that gets read as "the method wasn't requested".

**Fix:** in `make_surrogate`, deep-copy `mean_s` (or fall back to a fitted `ConstantMean`) instead of reaching for `.constant`; and make `bo_diagnose`'s `by_n_obs` `except` log the method + exception type.

---

## Q4 â `_apply_em_mean` on the lcbench path: neither raises nor no-ops-at-the-guard â it is **never called**

`_apply_em_mean` raises when `hyperbo_prior is None` (`bo_experiment.py:369-374`). On the lcbench path that check is **unreachable**: `main()` at `:2693-2710` never invokes it.

So the answer to "is a HyperBO prior guaranteed available?" is moot in the worst way â the flag is a **silent no-op**, not a raise. Had `_apply_em_mean` been called there, the prior *would* be available only when `--em-canonical hyperbo` is also set (`:2673-2691`), since the method-driven `hyperbo_prior` at `:2725-2753` is built **after** `pretrain_em_prior`. The correct fix therefore has two parts: call `_apply_em_mean`, **and** hoist/gate the HyperBO pre-train on `args.em_mean != "empirical"` as pd1_loo does at `:1618-1620`.

---

## Q5 â `run_pd1_full` (`bo_experiment.py:1813-2096`): **none** of the new knobs reach it

`bo_experiment.py:1967-1976`:
```
1967|         mean_s, covar_s = build_shared_kernel(em_pre_ds)
1968|         em_prior = pretrain_em_prior(
1969|             datasets=em_pre_ds,
1970|             mean_module=mean_s,
1971|             covar_module=covar_s,
1972|             likelihood_noise=torch.tensor(1e-2, dtype=torch.double),
1973|             covariance_shrinkage=EM_SHRINKAGE,
1974|             num_em_iterations=args.n_em,
1975|             enable_interpolation=True,
1976|         )
```

| Flag | `run_pd1_full` |
|---|---|
| `--em-base-mean` | **SILENTLY IGNORED** â no `base_mean=` at `:1967` |
| `--em-mean` / `--em-mean-weight` | **SILENTLY IGNORED** â no `_apply_em_mean` |
| `--iw-nu` | **SILENTLY IGNORED** â passes the `main()` guard (`:2528`, since `--use-covar-prior` is also given) then does nothing |
| `--use-mean-prior` | **SILENTLY IGNORED** |
| `--use-covar-prior` | **SILENTLY IGNORED** |
| `--em-init-mode` | **SILENTLY IGNORED** |
| `--em-canonical-pool` | **SILENTLY IGNORED** â `args.benchmark == "pd1_full"` satisfies `startswith("pd1")` at `:2541`, so the guard waves it through, but `args.em_canonical_pool` is read at exactly one place, `:1586`, inside `run_pd1_matched_loo` |

Two pre-existing bonus omissions on the same path: `--deep-kernel` (`DEEP_KERNEL` not passed to `build_shared_kernel` at `:1967`) and `--em-canonical` (no `_pd1_canonical` call). Both are silently ignored on `pd1_full` too.

The docstring at `:1814-1818` claims "EM / pretrained-GP / warm-start REQUIRE a shared pool and are skipped here", but `:1953` runs EM unconditionally when an `em*` method is requested â so these are live, mis-configured EM runs, not skipped ones.

**Fix:** either thread all six knobs through `:1967-1976` (preferred â it is a five-line change), or extend the `main()` guard so anything other than `pd1_loo` rejects them.

---

## Q6 (bonus) â `_pd1_probe` (`--benchmark pd1_probe`)

`:229-244` passes `base_mean`, `iw_nu`, `use_mean_prior`, `use_covar_prior`, `init_mode` and calls `_pd1_canonical` â but **not** `_apply_em_mean`. `--em-mean` is a silent no-op there. Diagnostic-only, low severity, but it is the same omission as Defects 1 and 2 and should be fixed in the same pass.

---

## Key Locations

- `_scratch_bo/bo_experiment.py:2693-2710` â lcbench EM build; missing `_apply_em_mean`
- `â¦/bo_experiment.py:1659-1678` â pd1_loo EM build; the **reference implementation**, all knobs wired
- `â¦/bo_experiment.py:1967-1976` â `run_pd1_full` EM build; zero knobs wired
- `â¦/bo_experiment.py:229-245` â `_pd1_probe` EM build; missing `_apply_em_mean`
- `â¦/bo_experiment.py:838-843` â `pretrained_gp_frozen` reads `mean_s.constant`
- `â¦/bo_experiment.py:2528-2558` â `main()` guards (`--iw-nu`, `pd1_only` argv check)
- `â¦/bo_diagnose.py:400-422` â kernel + EM build; now fully wired
- `â¦/bo_diagnose.py:579` â unguarded `float(mean_s.constant)`
- `â¦/gen_stage_queue.py:33-104` â `REGIMES` Ã `OFAT`, the matrix these defects corrupt
- `â¦/botorch/models/empirical_gps/em_empirical_gp.py:555-570` â `pretrain_em_prior` signature (the contract)

## Recommended Order of Fixes

1. **`bo_diagnose.py:579`** â one-line `getattr` guard. Unblocks 5 OFAT cells Ã 2 regimes today.
2. **`bo_experiment.py:2697`** â add `_apply_em_mean` + HyperBO gating on the lcbench path. Stops 3 LCBench-BO cells from being mislabelled.
3. **`bo_experiment.py:1967`** â thread all six knobs into `run_pd1_full`.
4. **`bo_experiment.py:843`** â stop `pretrained_gp_frozen` reaching for `.constant`.
5. **`bo_diagnose.py`** â hoist HyperBO pre-train when `em_mean != "empirical"` so `--em-mean` does not require `--em-canonical hyperbo`.
6. **`bo_diagnose.py` `by_n_obs` `except Exception: continue`** â log the method and exception; a silently dropped baseline row is indistinguishable from an unrequested one.

Finally, `tests/test_experiment_setup.py:73-78` documents the "seven silent no-ops" pattern and asserts guards at the *parser* level only. Consider adding a reach-test that monkeypatches `pretrain_em_prior` and asserts the received kwargs, parameterised over `{pd1_loo, pd1_full, lcbench} Ã {bo_experiment, bo_diagnose}` â that single test would have caught every defect above.",
  "subagentUsed": "code_search",
  "agent_id": "subagent-ecf09ab0-2d18-4178-8047-622b4c6d8b1c


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
