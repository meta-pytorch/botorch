# Adversarial review, ROUND SEVEN — 2026-08-17

Run after round 6. **Three of the five round-6 fixes were broken, two of them
guaranteed**, and one item marked ✅ in the round-6 doc was never fixed at all.

**Status key:** ✅ fixed in this pass · ⬜ open

---

## Round-6 fixes that were broken

| id | sev | finding | status |
|---|---|---|---|
| **R7-1** | **CRITICAL** | `--pretrain-seed` **does not exist on `bo_diagnose`**, but the round-6 `--priors` expansion appends it to *every* regime. `argparse` exits 2, no shard is written, the cell is INCOMPLETE, the stage never reaches `.DONE`, and it re-runs identically forever. **96 cells, 100% guaranteed failure** — the whole regression half. | ✅ flag added to `bo_diagnose`, and wired to actually re-seed the prior draw |
| **R7-2** | **CRITICAL** | `by_n_obs` is `{n_obs: {method: {metric: value}}}` — a **dict keyed by method**, not a list of rows. The round-6 reader took the `isinstance(rows, list)` branch on every real file and **silently dropped every by_n_obs metric**, i.e. the entire ranking-quality-vs-observations analysis. The emptiness guard could not fire because `prior_mean` parsed. Verified against `results/raw/diagnose.json`. | ✅ |
| **R7-3** | HIGH | **`report_gaps()` has zero call sites.** Round 6 marked R6-P6 fixed; it was not. `coverage_gaps` is populated, but the warning can never print. | ✅ now called from `fixed_effects_test` |
| **R7-4** | HIGH | The `task_wvar` rewrite is algebraically right but strips the SE to a **2-df** estimate (1 df on gap tasks), aggravating R6-S3 — and can now yield `se == 0 → z = nan → "tied"` **printed beside a nonzero effect**, which is common on the binary solve-rate metric. | ✅ (ii) fixed: reports UNDETERMINED. ⬜ (i) df remains |
| R7-5 | MEDIUM | `_prior_mean_pick`'s guard has correct arg order but passes the literal `"prior_mean"` instead of the actual method, and applies acquisition-space `+inf` reasoning to a mean. | ⬜ |
| R7-6 | MEDIUM | The `+inf` raise is numerically safe (no legitimate `+inf` found) but **operationally unsafe**: one `+inf` at iteration 37 kills a shard, and the stage then crash-loops. Asymmetric with the NaN path, which is tolerated. | ⬜ |

## New: the prior expansion did not actually pool

| id | sev | finding | status |
|---|---|---|---|
| self-check | HIGH | Cells are named `<config>_p<k>`, and `summarize_stage` grouped by `rsplit("_s",1)` — so **each prior became its own cell and nothing pooled them**. The between-prior variance would have been identically zero again, i.e. the round-6 fix was inert in a new way. | ✅ `configs` pooled view added |
| **R7-7** | HIGH | That pooled view uses a **plain unweighted mean**, reintroducing exactly the prior-weight bias whose removal reversed §27.14. Mid-run summaries have unequal shards per prior by construction. | ⬜ |
| **R7-8** | HIGH | The pooled view emitted **no variance at all**, despite a docstring saying it exists *because* the between-prior spread is the dominant component. | ✅ `between_prior` block added |
| R7-9 | MEDIUM | `n_priors` uses `re.sub` as a matcher; a non-matching filename contributes its own basename as a distinct "prior id". Correct today only by accident. | ⬜ |
| R7-10 | MEDIUM | `--priors 3` triples the matrix (540 shards for one stage×regime) while the driver still advertises "~2 days". | ⬜ |
| R7-11/12 | LOW | Config-level errors are invisible to the `.DONE` gate; `rows[r][-1]` can `IndexError` on an empty trajectory at `--n-init 0`. | ⬜ |

## Verified correct

`rsplit("_s", 1)` on names containing `_p0` and dots (`shrink_0.1_p0_s0.json`); the
`${name}_s*.json` glob; `.corrupt.<ts>` exclusion; the `task_wvar` P=1 fallback,
`present[0]` validity, and `coverage_gaps` population; and no legitimate acquisition path
produces `+inf`.

## Tests

`RegressionSummarySchemaTest` **certified R7-2** — its fixture used the fictional
list schema. Same failure as round 6's `test_finite_guard_is_wired_into_both_selection_points`.
Fixed to the real schema plus a negative test that the fictional shape yields `kind=error`.

`BehaviouralGuardTest` is 1-of-4 behavioural despite its docstring condemning source
greps; `PriorReplicationTest` re-implemented the expansion inline instead of calling
`main()`, **which is why R7-1 shipped**.

Added `GeneratedCellsParseTest`: generates the cells `main()` actually produces, for every
regime × kind × `priors ∈ {1,3}`, and parses each against that regime's real harness.
That single test catches R7-1 and would have caught round-4 R6.

---

## Convergence

| round | findings |
|---|---|
| 1 | 28 |
| 2 | 21 |
| 3 | 11 |
| 4 | 23 |
| 5 | (fix round) |
| 6 | 30 |
| 7 | 12 |

**Not converging.** The dominant source is no longer the original code but the *fixes*:
rounds 4, 6 and 7 each found that the previous round's repairs introduced new defects, and
three separate tests have now certified the very bugs they were written to prevent. The
pattern is specific and repeatable — **a test that greps source or encodes an assumed
schema will pass on the bug**. Every test added from here must execute the path or be
checked against a real artifact.
