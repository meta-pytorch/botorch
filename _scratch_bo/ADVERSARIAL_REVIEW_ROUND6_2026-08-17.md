# Adversarial review, ROUND SIX — 2026-08-17

Run after round 5. **Three of round 5's five fixes were broken, and two composed into a
guaranteed crash.** 30 findings.

**Status key:** ✅ fixed in this pass · ⬜ open

---

## Round-5 fixes that were broken

| id | sev | finding | status |
|---|---|---|---|
| **R6-B1** | **CRITICAL** | `bo_experiment.py:1405` passed `method` inside `_fantasy_batch_pick`, whose parameter is `eff_method` — a **`NameError` on every `--batch-mode kb\|fantasy` run**. The line did not exist before round 5, and round 5's *other* fix (N1 forwarding) newly routed both PD1 paths into it. A crashing cell writes no shard, so its stage can never reach `.DONE` and re-runs forever. | ✅ |
| **R6-S1** | **HIGH** | The between-prior variance **double-counted the within term**. `E[s²(ȳ_p)/P] = Var(θ̂_t)` exactly — the spread of per-prior means already contains the within component. Inflated the SE by up to √2 and caused §27.20's wrong retraction. | ✅ |
| R6-B2 | MEDIUM | `posinf=-inf` silently ranked a model blow-up **last** instead of raising. | ✅ |
| R6-B3 | HIGH | The guard reached 2 of 3 selection sites; the missing one is `_prior_mean_pick`, i.e. **the zero-shot pick** behind the "EM owns @1" claim. | ✅ |
| R6-C1 | MEDIUM | The warp guard's *condition* was widened; its *message* still names only `--pd1-candidate-pool full`, so a user tripping it via another route follows advice that cannot clear the error. | ⬜ |
| R6-P6 | MEDIUM | `report_gaps()` is never called, so the round-4 coverage warning can never print. | ✅ |

## New, not previously examined

| id | sev | finding | status |
|---|---|---|---|
| **R6-P1** | **CRITICAL** | `summarise_regression` read `summary`/`results`/`metrics` — **none of which `bo_diagnose` writes** (it writes `prior_mean`, `by_n_obs`, …). Every regression cell would reduce to `{}` and the `.DONE` gate would accept it. **The entire regression half of the plan, including stage 4.1 "does regression predict BO", would have run for hours and produced nothing.** Silent no-op #10. | ✅ |
| **R6-P3** | **HIGH** | **Nothing in the matrix varied `--pretrain-seed`.** Every cell was a single prior, so `n_priors == 1` and the between-prior term was **identically zero on the entire forthcoming corpus** — the round-5 statistical fix was inert on everything the pipeline would produce. | ✅ (`--priors`, default 3) |
| R6-P10 | MEDIUM | `budget_censored_at` carried an **orphaned second docstring** stating the *retracted* claim that censoring stops a method "looking fast by solving only the easy runs". | ✅ |
| R6-S3 | HIGH | The 23 per-task between terms come from **3 prior draws**, so that component carries ≈2 df, not 46. The right critical value is **t(2)=4.30**, not 2. Every `\|z\|>2` verdict is anti-conservative on that component. | ⬜ |
| R6-P2 | HIGH | `summarise_bo`'s probe (`json.load(files[0])`) sits **outside** the per-file `try`, so a truncated shard 0 still poisons the whole cell. | ⬜ |
| R6-P4 | MEDIUM | The driver's headline promise — "stage 2/3 pick their own cells" — is two `say` lines. **Stage 3, the ≥3-prior replication, does not exist in code.** | ⬜ |
| R6-P5 | MEDIUM | Three `budget_to_target`-family functions, three estimators, two sign conventions; all omit the `n_init` initial evaluations. | ⬜ |
| R6-E7…E11 | MED | EM internals: dead `_mu_init_inducing`; the same module registered under two names (priors possibly double-counted); `learnable_inducing_points` + `from_pretrained` leaves a stale cache — **and "fixing" E3 naively would trigger it**; IW-MAP and trace shrinkage stack silently; the convergence check never inspects a likelihood. | ⬜ |
| R6-P7…P9 | LOW | `kill -- -$$` assumes group leadership; detached summarisers unreaped; `random` is in `BASELINE_METHODS` but `bo_diagnose` cannot build it. | ⬜ |

## Eight tests deleted as worthless

The worst: **`test_finite_guard_is_wired_into_both_selection_points`** asserted the
literal string `'_assert_finite_scores(score, method, "fantasy'` — so it **passed on the
`NameError` and certified the bug it was written to prevent.**

Also removed: `test_warp_guard_covers_every_multi_pool_path`,
`test_batch_settings_forwarded_on_every_path`, `test_cond_noise_is_documented_as_variance`,
`test_shrinkage_comment_does_not_claim_to_fill_the_null_space` (all grep source text or
comments), `test_task_variance_exceeds_the_within_only_form` (algebraically guaranteed),
`test_between_prior_variance_is_included_in_the_se` (tests a fixture),
`test_no_unused_module_level_helpers_in_analysis` (substring-based; `load` matched
`json.load`, and it scanned module level only — so the newest dead code, a *method*, was
structurally invisible).

Replaced with `BehaviouralGuardTest`, which walks the AST and asserts every name passed
to `_assert_finite_scores` is actually bound in its enclosing function — the check that
would have caught R6-B1.

**Lesson: grepping source proves a line exists; only running it proves the line works.**
