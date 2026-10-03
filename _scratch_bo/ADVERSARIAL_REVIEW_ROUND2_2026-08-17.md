# Adversarial review, ROUND TWO — 2026-08-17

Run after fixing round one's 28 defects, specifically to find (a) what round one missed
and (b) bugs **introduced by round one's fixes**. It found both.

**Status key:** ✅ fixed in `fc20ad4f30fa`+ · ⬜ open

---

## Regressions created by the round-one fixes

| id | sev | file | finding | status |
|---|---|---|---|---|
| **N1** | **CRITICAL** | `run_autonomous.sh` | Global `set -o pipefail` + `grep -q` in the capability probe: grep exits at the first match, SIGPIPEs `the build tool`, pipeline goes non-zero → **the probe reports FALSE precisely when the flag IS present**, so `pd1_reg` was silently dropped from *every* run. Introduced by round one's B12 fix. Confirmed empirically. | ✅ capture then match a variable |
| **N10** | **CRITICAL** | `analyze_pd1pool.task_wvar` | Single-run priors are skipped from the numerator but still counted in `present.size**2` → at n=(5,1,1) the variance is **11× too small and \|z\| inflated 3.3×**, on the primary metric. Introduced by the weighted-SE fix. | ✅ divide by `len(terms)**2` |

## Round-one defects that were only half-fixed

| id | sev | finding | status |
|---|---|---|---|
| N12 | **HIGH** | `wmean` was wired into **1 of 4** pooled call sites. The POOLED ARM C section (final-regret ranking, solve-rate table, paired z tests) still uses raw means, so §27.3–§27.8 remain biased. | ⬜ |
| N16 | HIGH | `SUMMARY.json` still written non-atomically, ~15×/stage, by concurrent writers, and read by the `.DONE` gate → a truncated read marks a complete stage INCOMPLETE. | ⬜ |
| N17 | HIGH | Producer-side atomic write never done: `bo_experiment.py:1819/2106/2955`, `bo_diagnose.py:732` still `open(out,"w")` + stream. Only the shell-side quarantine shipped. | ⬜ |
| N18 | MED | `PROVENANCE.json` read-modify-write non-atomic and `KeyError`-prone → wedges a stage permanently. | ⬜ |
| N19 | MED | No lockfile on the driver, no signal trap → two drivers can write the same shard paths; killed cells leave reparented `the build tool` grandchildren. | ⬜ |
| N13/N14/N15 | MED | `variance_decomposition` has no budget-metric path (decides compute spend on the *wrong* metric), prints `abs()` z (loses direction), unguarded `effect/sem` division, and `np.mean([])` → silent NaN verdicts. | ⬜ |

## New orchestration findings

| id | sev | finding | status |
|---|---|---|---|
| N5 | HIGH | Detached summarisers are children of the *pipeline subshell*, so `wait` in `run_block` waits for nothing; an orphan can overwrite the final summary with a mid-stage one containing `"kind": "error"` cells → complete stage marked INCOMPLETE and re-run. | ⬜ |
| N3 | MED | `.incomplete` is never cleared by `run_cell_queue.sh` itself → the documented standalone entry point has a permanently sticky failure. | ⬜ |
| N7 | MED | `run_block`'s `return 1` is discarded by the stage loops; the driver exits 0 even if every stage failed. | ⬜ |
| N8 | MED | `SHARDS` is not part of the resume key: changing it leaves stale shards that are silently pooled with the new ones. | ⬜ |
| N2 | LOW | `jobs -p` can miss a shard reaped before the loop → false INCOMPLETE. Capture PIDs at spawn instead. | ⬜ |
| N9/N20/N21 | LOW | Shard re-run truncates the forensic log; `MANIFEST.tsv` records intent not outcome and omits shard/commit; `STATUS.txt`, ~1800 buck logs and `.corrupt.*` files are unbounded. | ⬜ |

**Confirmed fine:** `jobs -p` scoping inside the backgrounded `run_cell` (no cross-cell
contamination), `.incomplete` append atomicity, the direction logic on all four metric
paths, the tie accounting, `se = sqrt(ΣVar)/T`'s form, the per-method `pmax` fix, and
`.corrupt.*` exclusion from every glob.

---

## The tests do not bind the contract

**All six `OrchestrationContractTest` methods are string greps on shell source.** They
would pass against a driver that runs nothing:

- `test_corrupt_shards_are_quarantined` asserts the log *message*; delete the `mv` and it stays green.
- `test_done_marker_is_conditional` asserts **indentation**, not conditionality — rewriting the guard as `else\n    touch ...` inverts the guarantee and still passes.
- `test_shards_has_no_silent_default` forbids one literal; `:-6` passes.
- `test_shard_exit_codes_are_collected` never checks that `rc` is *read*.

Others assert the wrong thing:

- `GuardTest.test_iw_nu_without_covar_prior_raises` — **nothing raises**; the final assert restates the two above it. The guard lives in `main()` and is never invoked.
- `test_pd1_flags_are_recognised_as_pd1_only` — **inverted polarity**: it locks in the §27.1 no-op state and would *fail a fix*.
- `SummariserAlignmentTest.test_corrupt_shard_does_not_kill_the_whole_cell` — **there is no corrupt shard in the test**, and the behaviour is still broken (`json.load` has no per-file `try`).
- `PoolingWeightTest` — `_FakePooled` **re-implements** the weight loop instead of constructing a real `PooledStudy`, so breaking production changes nothing.

**Required:** execute the drivers in a tempdir with a stub harness and assert on exit
code, `.incomplete` contents and `.DONE` presence; construct real `PooledStudy` objects.

---

## Scientific validity: the censoring estimator is biased, and its docstring is false

`budget_censored` censors non-solvers at `T = len(row)` — one evaluation past the last
index. The docstring claims this means "a method cannot look fast by solving only the
easy runs". **That claim is false.** With a 50-evaluation budget:

| method | behaviour | score |
|---|---|---|
| A | solves 50% at eval 5, **never** solves the rest | (5+51)/2 = **28.0** |
| B | solves **100%** at eval 30 | **30.0** |

**A "wins" the primary metric while failing half its runs**, and the distortion favours
exactly the explore-heavy profile the EM-vs-HyperBO contrast is adjudicating. It is the
restricted mean (RMTT@T), not the mean time to target.

Two further consequences: the censoring constant is the horizon, so `pd1_bo_armC`
(`--n-iters 50`) and `lcb_bo` (`--n-iters 40`) are **on different scales**, and
`summarize_stage.budget_to_target` takes `horizon = len(row)` *per row* with no check
that rows share a length.

**Required:** report it as RMTT@T, always beside the solve rate (already `metrics.md`
rank 3), and add a sensitivity row at `2T`. **Any headline that flips between `T` and
`2T` is an artefact of the censoring constant, not a property of the method.**

---

## Priority order

1. ~~N1~~, ~~N10~~ — done; without N1 the regression half of the matrix silently did not exist.
2. **N12** — route the remaining pooled sites through `wmean`, or arm C stays unusable.
3. **N5 + N16 + N17** — atomic writes and one lock for all summarise calls.
4. **N18 + N19** — the "wedges the multi-day run permanently" class.
5. **Tests** — replace the vacuous ones with executing tests.
6. **Censoring** — rename, pair with solve rate, add the `2T` sensitivity row.
