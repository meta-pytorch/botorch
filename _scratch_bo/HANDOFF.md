# Handoff — Empirical GPs for Bayesian Optimization

**Last updated:** 2026-08-18 · **Working commit:** see `sl ssl`, top of the
`[EmpiricalGP][internal, do not land]` stack.

> ## ⚠️ Read `EMPIRICAL_GP_INDEX.md` first (added 2026-09-08)
>
> This file is still the best operational handoff for workstream A, but **two parts of it
> are now historical**:
>
> 1. **The "RESUME HERE" snapshot in §0 predates §32–§33.** It was written 2026-08-22; the
>    OAS and direct-shrinkage stages ran afterwards and completed. The on-disk summaries,
>    not this section, are authoritative for what has run.
> 2. **The OAS relaunch instructions near the end are historical** for the same reason —
>    that work is done. See `EXPERIMENTAL_RESULTS.md` §32–§33 for the outcome.
>
> `EMPIRICAL_GP_INDEX.md` also carries the consolidated retraction list across *both*
> research stacks, and explains the vocabulary collision with the multi-output workstream
> (their `rho` is not this stack's `alpha`).

Read this first, then `EXPERIMENT_PLAN.md` (the design — now executed, see its header),
then `EXPERIMENTAL_RESULTS.md` (§32–§33 are the most current; several earlier sections
are **retracted or superseded** — see "Retracted" below).

---

## 0. RESUME HERE — live state as of 2026-08-22 08:00 UTC

**NOTHING IS RUNNING. The machine rebooted and every `em-*` systemd unit is gone**
(`systemctl --user list-units 'em-*'` returns 0 units; no `bo_experiment` /
`bo_diagnose` / `run_autonomous` processes). The user systemd instance was reset, so the
transient `systemd-run --user` units did not survive. **The pipeline will not restart
itself.** Uptime at the time of writing: 13 h.

### What actually finished — this supersedes every "waiting on" table below

The tables written on 2026-08-21 at 14:31 were stale within hours. Re-derived from
`cells_present` / `.DONE` on disk:

| stage | state |
|---|---|
| stage 0 (5 regimes), stage 1 (**6** regimes incl. arm B), stage 3 BO (3 regimes) | **complete** |
| `stage2noise` | **63/63 in all five regimes** — including `pd1_reg`, which the old table said was 12/63 |
| `stage2means` | `lcb_bo`, `lcb_reg`, `pd1_bo_armA` complete; `pd1_bo_armC` 90/90 cells but **11 of 1080 shards missing**; `pd1_reg` **9/90** (baseline config only) |
| stage 3 regression | **never ran** |

**Missing `.DONE` markers are not evidence of incompleteness here** — several stages
finished and were killed before the marker was written. Check `cells_present` against
`cells_expected`, not the marker.

Remaining compute from the original plan is therefore small: **81 cells + 11 shards.**

### THE HEADLINE: everything on disk has now been analysed, and it changed the picture

See **§28** of `EXPERIMENTAL_RESULTS.md`. Four findings and three defects, in priority
order:

1. 🔴 **Stage 3 replicated nothing.** All **675/675** stage-3 shards are byte-identical to
   their stage-1 namesakes (verified, not inferred). Stage 3 re-ran the same flags with the
   same `--pretrain-seed 0..8` against deterministic code. The plan's rule "only Stage 3
   numbers may carry significance claims" therefore **certifies nothing**, and the
   Δ = 0.000 must never be reported as "the winners replicated" (§28.1).
2. **The σ² optimum is off-grid-specific and INVERTS on-grid** — interior optimum [0.3, 1]
   on arm C (t=−3.50), monotone penalty on arm A (t=+6.24 at σ²=10), interaction t=−4.10.
   The §27.35 n=5 anchor **does not reproduce** and is now unresolved (§28.2).
3. ⚠️ **The mean sweep is confounded.** `--em-canonical hyperbo` is set on exactly the 6
   configs with non-empirical mean transfer, so kernel and mean are perfectly confounded and
   the arm-C ranking is *precisely* that split. **One added config breaks it** (§28.3).
4. **The arm decomposition reproduces and strengthens** — off-grid alone costs 5.12 evals
   (t=+5.92), B→C returns 52.8% (§28.4). But like every other headline here it is
   **fixed-task only**; task-clustered it is tied (t(22)=+1.78).

### 🔴 The RNG-guard instruction in older versions of this file was WRONG — do not follow it

`bo_diagnose` has **two** defects, and the documented one is the smaller (§28.6):

* **Known (§27.38):** no RNG re-seed guard before the neural baselines, so `hyperbo_frozen`
  takes 3 values instead of 1. Affects **3 of 14** configs. Magnitude 0.51% (`lcb_reg`) /
  2.74% (`pd1_reg`).
* **Larger, previously unreported:** `--pretrain-seed` is a **no-op for EM**.
  `bo_diagnose.py:702` seeds the observation subset on `1000*ei` with no prior term, and EM
  pre-training is deterministic, so `em_frozen` is bit-identical across all 9 "priors" for
  **7 of 14** configs. Between-prior variance is exactly 0 and every df=8 t is 0/0 —
  visible in raw output as `t = 2.5e16`.

**Applying `apply_rng_guard.py` alone is now the wrong move.** It would re-seed
`hyperbo_frozen` and make the currently-clean 11-config block incomparable while fixing
none of the dead-prior problem. **Land both fixes together, then re-run the 252 regression
cells** (`stage1_lcb_reg` + `stage1_pd1_reg`). Until that lands, **no regression number may
carry an SE, a t, or a significance claim** — point estimates from the 11-config clean
block only. **BO regimes are provably clean** and need no re-run.

There is also a latent third defect: `select_top_configs` picks `n20_rank_corr`, not `n5`
as its own comment claims (`gen_stage_queue.py:301-307`). Inert until a regression stage 3
runs.

### Resuming after a crash or a restart

Everything is resumable at three levels — shard (json exists), cell (all shards present),
stage (`.DONE` marker). **Re-running the same command picks up exactly where it stopped
and redoes nothing:**

```
STAGES=0 CELLS_PARALLEL=3 bash results/run_autonomous.sh
```

`.DONE` is written only when the queue succeeded AND `SUMMARY.json` certifies every cell
complete, so a stage that half-failed will correctly re-run its missing cells.

### Cost model, measured

Per-cell cost is **regime-dependent**, and any single-regime extrapolation will be wrong.
PD1 arm C runs 16.7 min/cell across 12 shards; LCBench runs **1 shard** and pre-trains
HyperBO, ABLR and the EM prior serially first. The dominant term is measured:

| `hyperbo_iters` | HyperBO pre-train |
|---|---|
| default (`None`) | **35 s** |
| **25000** (what every LCBench cell uses) | **3662–3672 s ≈ 61 min** |

So ~1 h of every LCBench cell is HyperBO pre-training alone. Measured reference points:
stage 0 = 7 h 34 m for 45 cells; `lcb_reg` 1.91 min/cell; `pd1_reg` 4.00 min/cell.

**A long silence in a 1-shard log is normal** — pre-training prints nothing for ~1 h.
Before concluding a hang, check CPU time is advancing
(`ps -eo pid,etimes,time,pcpu,args | grep bo_experiment-inplace`; healthy is +80 s CPU per
60 s wall at ~150%). This project lost 62 hours to a real hang that looked identical from
the log alone, so make the distinction with CPU time, not patience.

### Do not edit `run_autonomous.sh` or `run_cell_queue.sh` while they execute

**Bash reads a script by byte offset as it goes.** An edit on 2026-08-19 inserted 2 bytes
before the currently-executing region and grew the file 10432 → 10998 bytes; bash would
have resumed at a shifted offset and executed a fragment. The corruption is silent and
arbitrary, not merely inconsistent.

`gen_stage_queue.py` and the Python harnesses are safer (a fresh process reads them per
stage) but still trigger a **Buck rebuild**, so cells launched before and after an edit run
different binaries — a build-hash discontinuity inside an experiment.

**Rule: land every source edit while the box is idle, then launch. Never the reverse.**

### Running other work on this machine at the same time

Concurrency depends on the regime's shard count:

| regime | shards | threads at `CELLS_PARALLEL=3` |
|---|---|---|
| `lcb_bo`, `lcb_reg`, `pd1_reg` | 1 | **9** of 96 — box is ~90% idle |
| `pd1_bo_armC`, `pd1_bo_armA` | 12 | **108** of 96 — saturated |

Memory has never been the binding constraint (~130 GB free throughout).

**The real hazard is NOT CPU — it is switching the working copy.** `run_cell_queue.sh`
invokes `the build tool run` **per shard**, so an `sl goto` deletes `_scratch_bo/` from the checkout
and every subsequent shard fails to resolve its target. This already happened once here.
**Use a separate checkout for parallel work.**

### Artifacts — where the σ² / shrinkage work lives

| path | what it is |
|---|---|
| `.llms/rules/em_noise.md` | **Auto-loaded rule.** Read before touching the noise parameter. Now leads with the regime-dependence finding (§28.2). |
| `em_noise_selection.py` | Ledoit–Wolf, OAS, effective-rank matching, Marchenko–Pastur edge, `estimate_iw_nu`. **Its own buck target**, so editing it cannot rebuild the running harnesses. |
| `--iw-nu-mode {manual,oas,ledoit_wolf}` | In `bo_experiment`, default `manual`. Empirical-Bayes IW `ν`. Verified firing and bit-identical at the default. **Never swept.** |
| `--em-likelihood-noise` | Exposes the formerly hardcoded σ², default 1e-2 so every committed result reproduces. |
| `apply_rng_guard.py` | Stages the §27.38 patch. **Do not apply alone — see above.** Note it writes offset `+11` at all four sites; `bo_experiment` uses 11/11/13/17, so PACOH and ABLR must be corrected by hand after running it. |
| `stage4.py` / `the build tool run :stage4` | **Stage 4 synthesis (§29).** The four cross-stage analyses of EXPERIMENT_PLAN §4/§10, none of which existed before 2026-08-22. `--which {all,4.1,4.2,4.3,4.4}`. Runs on data already on disk; launches no compute. **Its own buck target**, so editing it cannot rebuild the running harnesses. |
| `prior_stability.py` / `the build tool run :prior_stability` | **§30.3.** Reads the prior cache, groups fits by everything except the seed, and reports how far independent pre-training runs land apart in parameter space; then decomposes result variance into prior / task-shard / residual. The only analysis that separates pre-training randomness from evaluation randomness. **Its own buck target.** |
| `analysis/` + `analysis/README.md` | Seven one-off scripts behind §28.3–§28.5, recovered from `/tmp`. Indexed by which § each produced. Not in `:lib`. |
| `--prior-cache` on **both** harnesses | `bo_experiment` and, since §30.2, `bo_diagnose`. Content-addressed, `pretrain_seed` in the key so repeated fits are retained separately. Verified cold == warm == cache-off. Serving `hyperbo_frozen` from cache also makes it byte-identical across configs by construction. |
| `results/run_armB.sh`, `run_noise_sweep.sh`, `run_stage3_reg.sh`, `chain_stage2.sh` | Per-sweep runners, each documenting why it exists. |

### The EM noise is an EFFECTIVE RANK knob, not a noise level (§27.40)

Diagnosed why the optimum is so large. It is **not** a scale bug — per-task variance 1.000,
kernel outputscale 1.099, and `Σ_ind` diagonal 0.37-0.41 against a measured cross-task
variance of 0.4461, i.e. correctly scaled.

σ² acts as a **spectral truncation**: the E-step retains eigendirection λ with weight
λ/(λ+σ²), so σ²=1e-2 keeps 60/60 directions while σ²=3 keeps 5. But `Σ_ind` from T
pre-training tasks has **rank ≤ T−1**, so the hardcoded 1e-2 was asserting rank 60 for a
rank-5 object and fitting 55 directions of pure estimation noise.

**That prediction was TESTED AND REFUTED (§27.41).** The optimum is **0.6 at T = 6, 10 and
17 alike** — flat, not tracking `n_pretrain`. The effective-rank matcher moves exactly as
predicted (17.7 → 4.54 → 1.78), so it faithfully implements the hypothesis; the hypothesis
is wrong. Do not resurrect it.

**Why σ²≈0.6–1 is not alarming (§27.42).** `K(K+σ²I)⁻¹` is ridge regression, and the
optimum sits at the 50/50 point: weight on the typical direction is
`1.099/(1.099+1) = 0.523`, and Ledoit–Wolf independently keeps 0.49–0.55 of the sample
covariance. Two caveats that are easy to get wrong:

* "50/50" describes the **mean**. Shrinking deviations by `w` scales the covariance by
  `w²` (verified exactly), so it is a **4× covariance shrink**.
* The shrinkage target is the **current EM mean μ** (`residual = y - mu_S`), a *vector*.
  That is what makes it structurally different from `--em-shrinkage`, which blends the
  finished *covariance* toward a base-kernel gram and is **trace-preserving** — and so
  cannot reduce an overfitted magnitude, which is why §27.9 found it useless.

**And the cause is finite-sample severity, not misspecification (§27.45).** Data drawn from
*exactly* the model EM assumes, observed noiselessly, still wants σ²=1–2. `Σ_ind` has
**147% relative error at T=6**, matching `√((r_eff+1)/T)` to within a few percent. The
implementation is fine; the estimator is simply that bad at realistic T.

**The real fix is not to hand-pick σ² but to stop using it for this job.** Ledoit–Wolf/OAS
applied to `Σ_ind` directly **beats the best tuned σ² at ~zero noise** (§27.44), needs no
tuning, and is equivalent to empirical-Bayes selection of the IW `ν` we already expose
(§27.46). Implemented as `--iw-nu-mode {manual,oas,ledoit_wolf}`, default `manual`.

### ~~NEXT CODE FIX: BOTH `bo_diagnose` defects~~ — DONE 2026-08-22 (§28.6)

**Both defects are fixed in the working copy and 136/136 tests pass.** Kept here only to
record what was wrong and how it was verified.

1. **RNG re-seed guard** — `bo_experiment.py:3121/3160/3188` re-seeds with
   `torch.manual_seed(pretrain_seed * 1009 + {11,13,17})` before each neural baseline;
   `bo_diagnose.py` did not. Affected **3 of 14** configs (`mean_linear`,
   `mean_linear+covprior`, `canon_deep`). Now guarded at all four sites.
   ⚠️ `apply_rng_guard.py` writes `+11` at every site; the correct offsets are
   **11/11/13/17**, so PACOH and ABLR had to be corrected by hand after running it.
2. **Dead `pretrain_seed` axis** — the observation subset was seeded on `1000 * ei`,
   omitting the prior term. Since EM pre-training is deterministic this made `em_frozen`
   bit-identical across all 9 priors for **7 of 14** configs, so between-prior variance
   was exactly 0 and every df=8 t was 0/0. Now `1000*ei + 100003*pretrain_seed`.

Both were landed together, because applying (1) alone re-seeds `hyperbo_frozen` and makes
the currently-clean 11-config block incomparable while fixing none of (2).

**On the tests.** Four were added, and each was checked to **fail against the pre-fix
file** — recover it with `sl cat -r <pre-fix-rev> bo_diagnose.py`, not from the
`/tmp/bo_diagnose.py.bak` originally used here, which no longer exists. The project has
been bitten repeatedly by tests that
certify the bug, and there was already one here:
`test_pretrain_seed_exists_on_both_harnesses` asserted only that `--pretrain-seed` is
*accepted* by both parsers, and passed happily throughout the entire period the flag was
a no-op. Two of the new tests were themselves wrong on the first attempt — one was
vacuous (`{True} == {True}`), and one would have demanded that the train/eval split vary
per prior, which would have silently unpaired every contrast in the study. Both are
corrected, and a companion test now **pins the split and the inducing set as FIXED**
across priors so a future "make everything depend on the prior" fix cannot pass.

**Still required:** re-run the 252 regression cells. BO regimes are provably clean and
must not be re-run.

`select_top_configs` (`gen_stage_queue.py`) is also fixed — it took `n20_rank_corr` from a
lexicographically sorted set while its comment claimed `n5`; it now sorts numerically.

### `em_noise_selection.py` cleanup + decoupling — now UNBLOCKED

The box is idle and no sweep is running, so the reason for deferring these is gone. Do
them before launching anything, since all three touch `lib` or `BUCK` and would otherwise
have to wait for the next idle window:

1. **Missing license header** (LICENSELINT warning) — a policy item, not style. Every
   other file in the directory has the Meta copyright block.
2. 7 E501 lines, all docstring prose (lines 4, 5, 12, 14, 20, 21, 51).
3. **Move it OUT of `lib` into its own target.** Nothing in the harness imports it; it is
   a research utility. Having it in `lib` means every future edit rebuilds the running
   harnesses for no reason. Fixing this once removes the whole class of deferral.

Also one trailing blank line in `BUCK` that AUTODEPS2 flags as an error — cosmetic, and
`arc lint -a` fixes it, but that touches BUCK and so also rebuilds.

Verified the current state is safe: after adding the file to `lib`, all three units stayed
active with 0 build failures and cells continued completing.

### Deferred cosmetic cleanup (do AFTER stage 1/3 finish)

Three E501 lines were introduced with the prior cache and left unfixed **on purpose**:

* `bo_experiment.py` — 2 docstring lines in `_hyperbo_cache_params` (89, 90 chars)
* `results/run_autonomous.sh` — 1 comment line (89 chars)
* `bo_experiment.py` — 5 more lines in the `--em-likelihood-noise` help string, and
  `bo_diagnose.py` — 2, both added 2026-08-20. Deferred because BOTH files were live at
  the time: `bo_experiment` was running arm B's 126 cells and `bo_diagnose` the regression
  regimes. Arm B is the run that most needs to stay byte-comparable with arms A and C, so
  a rebuild for comment formatting was not worth it.
* `gen_stage_queue.py` — 2 comment lines in the `MEANS` block (lines 80, 81; 89 and 91
  chars). Safe to edit in principle (a fresh `python3` reads it per stage, so there is no
  byte-offset hazard), but the running driver still invokes it for three more regimes and
  stage 3, and two Advice-level comment lines do not justify touching a live dependency.

All three are Advice-severity and all are prose, so they change no behaviour. They were
not fixed because the stage was already running: editing a source file mid-stage triggers
a Buck rebuild, so cells launched before and after the edit would execute different
binaries. That is a build-hash discontinuity inside an experiment for zero benefit, and it
is the same hazard as the `sl goto` warning above. **The box is now idle, so these are
unblocked — fix them in the same window as the `bo_diagnose` defects, before relaunching.**

---

## 0b. How this codebase fails — read before writing any code or test

Seven adversarial review rounds found **125 defects**. They are not random; they cluster
into five shapes that keep recurring. Every one of these cost real time, and several
produced published numbers that had to be retracted.

**1. Silent no-ops (10 occurrences).** A flag is accepted, recorded in the config, and
does nothing: `--em-canonical hyperbo` never exercised, `--iw-nu` read only when another
flag is set, `--batch-mode` not forwarded on the PD1 paths, eight EM flags ignored in
`run_pd1_full`, `summarise_regression` reading keys that do not exist.
→ **Every new flag needs a non-no-op check before it enters a matrix.** Compare posterior
NLL, not just the BO trajectory — a 5-iteration trajectory is too coarse to move.

**2. Tests that certify the bug (4 occurrences).** A test that greps source, walks the
AST, or encodes an *assumed* schema will pass on the very defect it targets:
`test_finite_guard_is_wired…` passed on a `NameError`; `RegressionSummarySchemaTest`
encoded a `by_n_obs` shape that never existed; `PriorReplicationTest` re-implemented
production inline; `test_driver_actually_runs_stage3` checked the call and missed a
dropped argument.
→ **Execute the path, or check against a real artifact on disk.** "The call exists" is
not "the call works".

**3. Fixes that introduce new defects (rounds 4, 6, 7).** Round 5's two fixes *composed*
into a guaranteed crash. Round 3's `mean_baseline` and `_safe_copy_*` were dead code.
Round 6's between-prior term double-counted. A hardening fix turned a loud crash into a
silent wrong baseline.
→ **A fix is a change like any other. Review it as adversarially as the code it repairs.**

**4. Measuring with an instrument that cannot register the effect (3 occurrences).**
§27.24 answered "is there divergence?" on a metric bounded in [0,100]. §27.13 called a
config mismatch "drift". §27.19 propagated a severity claim without checking configs.
→ **Before quoting a zero, confirm the instrument could have shown a non-zero.**

**5. Estimator errors that move headline conclusions (3 retractions).** Pooling weighted
priors by run count (§27.17). The between-prior term was first omitted, then
double-counted (§27.21). The critical value was 1.96 when the df justified 4.30 (§27.22).
→ **State the estimand, state the estimator, and re-derive before citing a number.**

**Two mechanical traps specific to this environment:**

* The tool layer silently strips shell variable and array expansions (dollar-brace forms)
  from inline bash. It has broken a patch script and, later, the `--select-from`
  forwarding — and it stripped the expansion from *this very sentence* on the first
  attempt to write it. **Write patch scripts to a file and run them; verify the effect,
  never the exit code.**
* `arc f` reformats on commit, so a file written and then committed differs from what you
  wrote. Re-read before patching again, or you will splice two versions together.

---

## 0c. Documentation map — read in this order

| doc | what it is | when to read |
|---|---|---|
| **`METHODS.md`** | **What is actually computed** — the EM algorithm, the Nyström map, the arms, the metrics, the regularisation story. Written so a newcomer can read a result and know what it means. | **first** |
| `HANDOFF.md` (this file) | Live state, current claims, what not to cite, next steps | second |
| `EXPERIMENT_PLAN.md` | The staged design and factor inventory | for design questions |
| `EXPERIMENTAL_RESULTS.md` | 5000-line chronological log, now with a §-index at the top | only for specific § numbers |
| `.llms/rules/em_noise.md` | Auto-loaded rule on the σ² / shrinkage parameter | automatic |
| `.llms/rules/metrics.md` | Auto-loaded rule on metric conventions | automatic |
| `ADVERSARIAL_REVIEW_ROUND*.md` | Seven rounds of self-review, 125+ defects | for "has this been checked" |
| `PAPER_INTEGRATION_PLAN.md` | How results map into the paper | writing time |

**Do not read `EXPERIMENTAL_RESULTS.md` front to back.** It is chronological and includes
retracted work; §3 below lists what not to cite.

## 1. What this project is

Two deliverables sharing one codebase:

1. **An open-source BoTorch stack** (12 diffs, ending `<diff-11>`) — the Empirical GP
   model family plus a tutorial notebook. **Accepted, four diffs amended, awaiting
   re-review.**
2. **Internal research** benchmarking EM-Empirical GPs against meta-learning baselines
   (HyperBO, PACOH, ABLR) on LCBench and PD1, for a **follow-up paper**. The already
   published paper (`paper/`) is frozen apart from a budget caveat.

---

## 2. The current headline claims (use these)

| claim | evidence | strength |
|---|---|---|
| **EM is top-tier in BO** — tied with HyperBO, significantly beats ABLR/PACOH/vanilla/random | §26.1, paired over 5 splits | strong |
| **EM is Pareto-dominant** — best budget-to-target at 44–629× lower pre-training cost | §26.2 | **strongest; immune to split noise** |
| **EM ties every baseline on PD1 FINAL REGRET**, while handed 5× less pre-training data | §27.3–§27.5 | strong, but **narrow — see the two rows below** |
| **On PD1, EM reaches a very good configuration 1–4 evals LATER than HyperBO/ABLR** | §27.8 + §27.11, primary metric, fixed-task | **strong — direction survives the code drift** |
| **On PD1, EM is NOT distinguishable from a from-scratch GP on budget-to-target** | §27.8, §27.11 (tied under both codebases) | strong; state PD1 Pareto as **cost only** |
| ⚠️ §27.3–§27.8 pooled numbers are SUSPECT | §27.17 — same biased estimator (5/3/3 seed imbalance across priors) | **re-derive before citing** |
| ⚠️ §26.1 / §26.2 / §26.3 (LCBench) are affected by the hidden per-task Standardize | §27.19 — LCBench runs globally standardised, so D2 bites there. PD1 runs set --per-task-standardize and are INERT to it | **re-derive the LCBench numbers** |
| ⚠️ §27's *numbers* are not reproducible on current code — only its *conclusions* are | §27.10, §27.11 | **cite conclusions, never the effect sizes** |
| ⚠️ ~~The additive HyperBO kernel significantly improves EM on PD1~~ | **RETRACTED §27.17** — pooled estimator over-weighted prior 0 (45.5% vs 33.3%) and counted ties as wins; corrected z=−1.83/−1.71, **tied** | **do not cite** |
| ⚠️ ~~EM's pre-training noise was hardcoded 2-3 orders of magnitude too small~~ | §27.35 — **PARTLY SUPERSEDED by §28.2.** The gain is real but **off-grid only**: interior optimum [0.3, 1] on PD1 arm C (t=−3.50), a *monotone penalty* on arm A (t=+6.24 at σ²=10), interaction t=−4.10. The n=5 anchor (+23.5% / −78%) **does not reproduce** — it comes out sign-flipped (−8.6% / +18.5%) | **condition on geometry; the small-n claim is unresolved** |
| 🔴 **STAGE 3 CERTIFIES NOTHING** — all 675/675 stage-3 shards are byte-identical to stage 1 (same flags, same `--pretrain-seed 0..8`, deterministic code) | §28.1, verified directly | **no result in this project currently has an independent replicate** |
| ⚠️ **The arm-C off-grid win is the KERNEL, not the mean transfer** — at fixed kernel, `meanxfer_full − canon_hyperbo` = **+0.18 ± 0.48 (t=+0.39)** on arm C and null in all four regimes, against −3.58 ± 1.07 for `--em-canonical hyperbo` alone | §28.3, stage-1 OFAT, zero new compute | **write it as kernel transfer; the means sweep alone cannot say (it confounds the two)** |
| ⚠️ **Factor E (covariance prior) is the best NLL mover and the WORST factor on the primary metric** — +3.8 evals `lcb_bo`, +7.6 arm A; DiD −6.10 (t=−8.68), so it is also strongly regime-dependent | §28.5 vs §27.16, DiD in §29.2 | **sharpest case yet of regression quality dissociating from BO** |
| 🔴 **Screening EM variants by regression quality is UNSOUND off-grid** — Spearman(`n5_rank_corr`, budget) = **−0.689 on LCBench but +0.635 on PD1 arm C**; `n5_nll` = +0.04 vs **−0.62**. The best-regression configs (`shrink_*`) are mid-tier at BO; the best-BO configs have ~3× the NLL | §29.1 (stage 4.1) | **never promote a config on regression metrics in the off-grid regime** |
| ✅ **Cost Pareto measured, not asserted** — LCBench: `em_frozen` beats `hyperbo_frozen` on budget (9.87 vs 12.00) at **128× less** pre-training (26.7s vs 3428s). PD1 arm C: `hyperbo_frozen` wins on budget; EM is **2.1× cheaper and 2.3 evals worse** | §29.3 (stage 4.3) | **on-grid = performance + cost; off-grid = COST ONLY (confirms §27.8)** |
| 🔴 **Cache hits were being recorded as zero-cost pre-training** — the timing bracket wrapped `prior_cache.cached()`, not the fit. On `pd1_bo_armC`, `base` records 882s (0/108 hits) while `canon_hyperbo`, `shrink_0.1`, `init_naive`, `meanprior`, `covprior`, `mean_linear` each record ~0.03s in **108/108**. §29.3 survives only because `analysis_43` defaults to `config="base"` | §30.1 | **any cost number off a non-`base` config is cache-hit rate, not cost** |
| ✅ **HyperBO's weights are unstable; its behaviour is not** — 47 models × 9 independent fits: median relative L2 between fits **1.371** (√2 ≈ 1.414 is mutual orthogonality), yet on armC sd_prior is 0.8–2.1 budget units against sd_task 12.5–14.3 | §30.3 (`prior_stability.py`) | **the 25k-step fit is massively non-identifiable, but the induced function is stable** |
| ⚠️ **The prior axis is NOT a pure pre-training axis** — `_bo_seed = 1000·ei + seed + 100003·pretrain_seed`, so varying the prior also moves the initial design. EM pre-training is deterministic, so `em_frozen`'s sd_prior (1.428) is a pure evaluation-noise floor; HyperBO's genuine pre-training contribution is the excess, ≈**1.56** budget units | §30.3 | **§27.26's "prior variance dominates" is not what armC shows — it is the smallest of the three components** |
| 🔴 **The tail/degeneracy explanation of the EM deficit is NOT supported** — pre-registered moderator `d_t = a + b·p_div_t` on PD1 arm C (23 tasks, `p_div` 0.02–0.96): `b = +11.9 ± 10.2`, t(21)=+1.17, CI [−9.4, +33.2], significant in **0/23** LOO folds. LCBench is significant in the full sample but survives only **2/6** folds over a 0.02-wide covariate | §29.4 (stage 4.4), pre-registered in EXPERIMENT_PLAN §10 | **the deficit is UNIFORM; retire the tail explanation from §23/§27** |
| **The HyperBO kernel transfer works OFF-GRID** — `canon_hyperbo` erases EM's arm-C deficit (t=−5.21 vs base) and turns arm A into a decisive EM win (t=−4.28 vs HyperBO) | §27.34, 9 priors, both PD1 arms | **strong; the headline result** |
| It is INERT on-grid (LCBench t=+0.11) — the mechanism is regime-dependent | §27.31 vs §27.34 | strong |
| **Arm B decomposes the off-grid deficit** — going off-grid ALONE costs EM **5.12 evals (t=+5.92)** with pre-training held fixed; richer pre-training returns **52.8%** | §27.39, **reproduced at 126 cells in §28.4** (was 4.97/t=+4.41) | **strong — but FIXED-TASK ONLY; task-clustered t(22)=+1.78, tied** |
| **Ledoit–Wolf on `Σ_ind` beats the best hand-tuned σ², at ~zero noise** — RMSE 0.5296 vs 0.5390 (T=6), 0.4568 vs 0.4726 (T=17); they are SUBSTITUTES, stacking them over-shrinks | §27.44 | strong on LCBench; **unconfirmed on PD1** |
| **Use OAS, not LW** — OAS wins at small T (0.607 vs 0.545 PRIAL at K=20/T=6), which is our regime. LW captures only 0–37% of the oracle at T≥200, so it is NOT a general-purpose estimator | §27.46, controlled study over K/T/spectrum | strong |
| **Choosing α by OAS *is* empirical Bayes for the IW prior** — `E[Σ\|data]=(T·S+Ψ)/(T+ν−K−1)` is linear shrinkage with `α=(ν−K−1)/(T+ν−K−1)`, so `ν=(K+1)+αT/(1−α)`. Implemented as `--iw-nu-mode` | §27.46 + `em_noise_selection.estimate_iw_nu` | **unifies 3 overlapping regularisers** |
| **The heavy regularisation is finite-sample severity, NOT misspecification** — `Σ_ind` has 147% relative error at T=6, matching `√((r_eff+1)/T)` to within a few percent | §27.45, synthetic ground truth | **strong; bounds what the method can deliver** |
| **`bo_diagnose` records BOTH latent and noise-inclusive calibration** — `nll_obs`, `mean_sigma_obs`, `coverage95_obs`, `calib_ratio_obs` beside the latent versions | §27.33, implemented | **every v2 regression result carries both; readable either way without a re-run** |
| **The noise-inclusive calibration correction is INERT in practice** — ranking identical, EM-vs-control differential ~4.5pp not the 23pp a small probe suggested | §27.37 (lcb_reg, 14 configs × 9 priors) | strong |
| 🔴 **`bo_diagnose` has TWO defects, and the known one is smaller** — (a) missing RNG guard, 3/14 configs perturb the control; (b) **`--pretrain-seed` is a no-op for EM**, so 7/14 configs are bit-identical across all 9 priors and every df=8 t is 0/0 | §27.38 + §28.6 | **no regression number may carry an SE or a t until both are fixed and 252 cells re-run** |
| **LCBench divergence lives in `Train/loss`, not accuracy** | §27.25 — 89 curves >10⁶, one `inf`; 0 in the accuracy we optimise | our objective maps it to a finite bad score; **do NOT filter** |
| **Replicate over PRIORS, not seeds** | §27.26 — at fixed budget only P reduces σ²_B; 19–48% lower variance, t(8)=2.31 vs t(2)=4.30 | matrix now `--priors 9 --n-seeds 1` |
| ⚠️ **NO PD1 arm-C contrast is significant** | §27.22 — correct SE against correct threshold t(2)=4.30 at P=3 priors | **directional only; do not write any PD1 arm-C claim as significant** |
| HyperBO is ahead of every EM variant on PD1 budget-to-target | §27.22, consistent direction across all three estimators tried | **direction only** — needs ≥10 priors to certify |
| **HyperBO's kernel as an ADDITIVE component is the best variant** | §26.3 (LCBench regression) + §27.14 (PD1 BO) | **strong — now corroborated on a second benchmark** |
| EM degrades gracefully under sparse observations; HyperBO collapses | §24/§25 hetero sweep, z=+3.3 → −3.5 | **LCBench + SYNTHETIC sparsity only — does NOT replicate on PD1 (§27.4)** |
| EM pre-training is deterministic (prior-draw sd exactly 0.000) | §25.4 | strong — **and consequential: `pretrain_seed` never resamples the EM prior anywhere. In BO it varies the initial design, so for EM "9 priors" means 9 design replicates and its error bars contain no prior-draw component (§28.6)** |

**METRIC POLICY — see `.llms/rules/metrics.md`, which auto-loads for agents working here.**
Primary metric is **budget-to-target** (evaluations to a very good configuration), with
best-so-far regret as its companion. **Regret-AUC is NOT a headline metric** (§22) —
cross-check only. On PD1 arm C, AUC claims EM crushes `vanilla_gp` (z=−9.93) where
budget-to-target says **tied**; leading with AUC produces claims the primary metric does
not support.

**Always state which estimand a claim uses (§27.7).** Task-clustered ("wins on a new,
unseen task") is floored by T=23 and returns "tied" for almost everything. Fixed-task
("wins on this suite", the usual benchmark question) has no floor and resolves several
of the same contrasts. §27 originally conflated them and overstated what was unknowable.

---

## 3. Retracted / superseded — do not cite these

**From the σ² investigation (2026-08-21) — three of my own explanations, all refuted by
tests I wrote to check them:**

| claim | status |
|---|---|
| ~~The optimal σ² tracks the sample rank, so it falls as `n_pretrain` grows~~ | **REFUTED §27.41.** Optimum is 0.6 at T=6, 10 and 17 alike. The effective-rank matcher moves as predicted (17.7→4.54→1.78) — the hypothesis was simply wrong. |
| ~~More tasks add directions but do not sharpen any single direction~~ | **REFUTED §27.43.** Relative error falls as 1/√T (0.545→0.367→0.282→0.090). The flat optimum needed a different explanation. |
| ~~σ² is large because it absorbs model discrepancy (Kennedy–O'Hagan)~~ | **REFUTED §27.45.** Data drawn from *exactly* the model EM assumes, observed noiselessly, still wants σ²=1–2. |
| ~~The σ² optimum is 1.0~~ | **Superseded §27.44.** A 1.4×-spaced grid puts it at **0.6**; the "1.0" was an artefact of 3.3× grid spacing. |

What survives: σ² is a **ridge coefficient acting as spectral truncation** (§27.40,
measured), the LCBench optimum is **flat at 0.6** (§27.44), and **Ledoit–Wolf/OAS does the
job better with no tuning** (§27.44, §27.46).


- **§27.8's `hyperbo_frozen` − `em_finetuned` row at the 0.01 target** (z=−6.32) — does not
  reproduce on current code (z=−1.31, tied). §27.11.
- **All §27 effect SIZES** — the pre-training path drifted (§27.10); magnitudes shrank
  across the board. Directions survive, numbers do not.
- **"ABLR leads, EM 5th"** (§25.4) — single-split artifact. Superseded by §26.1.
- **"Off-grid, HyperBO flips from last to first"** (insight 35, the §21.3 that was never
  written up) — one prior. Two of three replicates put EM above HyperBO; pooled and
  paired, everything is tied (§27.3).
- **"EM wins under heterogeneity"** — true only for LCBench with *synthetic* sparsity. On
  PD1's natural heterogeneity every method ties (§27.4).
- **The 8 PD1 cells in `canon_matrix` and `hb_hybrid`** — they ran LCBench, not PD1
  (§27.1). Renamed to `RETRACTED_ranLCBench_*`; their LCBench content is redundant with
  cells we already hold under correct names.
- **"EM wins at n=5 on 7D regression"** (§24.4) — did not replicate at 10 eval datasets
  (§25.2). EM never significantly wins on 7D regression.
- **"The canonical kernel is a provable no-op on-grid"** — false. It reaches the posterior
  through **three** doors: interpolation, the shrinkage target `B=K(Z,Z)`, and EM init
  (§26 predecessor, canon_decomp). Shrinkage target is the **largest** (4.10 nats).
- **"The deep kernel closes 38% of the gap via better interpolation"** — the mechanism is
  probably the shrinkage target, not interpolation. Reword before using.
- **"Deep kernel slightly hurts EM in BO"** — 100% RNG confound; true on-grid effect is
  exactly zero.

---

## 4. Known bugs / blockers

**All five defects found by §28 are now FIXED IN THE WORKING COPY (uncommitted,
2026-08-22). 136/136 tests pass. Each fix was verified to FAIL on the pre-fix code, so
none of them is a test that merely certifies the bug.**

| # | defect | fix | still needed |
|---|---|---|---|
| 1 | 🔴 `bo_diagnose`: `--pretrain-seed` was a no-op for EM — `em_frozen` bit-identical across all 9 priors for 7/14 configs, every df=8 t was 0/0 (§28.6) | prior index now enters the observation-subset generator (`1000*ei + 100003*pretrain_seed`) | **re-run the 252 regression cells** — until then no regression number may carry an SE |
| 2 | 🔴 `bo_diagnose`: missing RNG re-seed guard (§27.38/§28.6), 3/14 configs perturbed the `hyperbo_frozen` control | guard at all 4 sites, offsets **11/11/13/17** matching `bo_experiment` | same re-run |
| 3 | `select_top_configs` took `n20_rank_corr` while its comment claimed `n5` | numeric sort key on the `n` prefix | nothing; inert until a regression stage 3 |
| 4 | Stage 3 reused stage 1's `--pretrain-seed 0..8` against deterministic code → 675/675 byte-identical, certified nothing (§28.1) | `--prior-offset`, defaulting to a disjoint range read from the source stage's PROVENANCE, and a **hard refusal** on overlap | **run a real stage 3** |
| 5 | `--em-canonical hyperbo` perfectly confounded with mean transfer in the MEANS sweep (§28.3) | added `mean_const_hbk` (`canonical=hyperbo × mean=empirical`) | nothing scientifically — §28.3 was **resolved from stage-1 OFAT**: the mean transfer adds nothing (+0.18 ± 0.48, t=+0.39 off-grid) and the kernel carries the whole win. The cell is only for the sweep's internal coherence |

**Resolved:** the former "PD1 pool flag is a silent no-op" blocker was a misdiagnosis — the
loader always worked, and `run_canon_matrix.sh` / `run_hb_hybrid.sh` simply omitted
`--benchmark pd1_loo`, so eight "PD1" cells ran LCBench (§27.1). Both scripts are fixed and
`bo_experiment.py` now **raises** if any `--pd1-*` flag is passed with a non-PD1 benchmark.

**Pre-existing:** `B023` warnings in `bo_experiment.py`'s LOO fold-loop closure (benign).

---

## 5. Next steps, in priority order

The box is **idle**. Land every source edit first, then launch — never the reverse (§0).

**Steps 1–4 of the previous list are DONE (2026-08-22, uncommitted).** All five §28
defects are fixed, 136/136 tests pass, each new test verified to fail on the pre-fix
code, and Stage 4 is implemented and run (§29). What remains is compute plus the
long-standing science items.

1. **[COMPUTE, do first] Re-run the 252 regression cells** with the fixed `bo_diagnose`.
   This is the single highest-value run available: it unblocks **every** regression
   significance claim in the project, all of which are currently point estimates from an
   11-config clean block. Nothing else should launch before it.

2. **[COMPUTE] Run a real stage 3.** The generator now defaults to a disjoint prior range
   and refuses overlap, so `--kind replicate` finally produces an independent replicate.
   The existing stage-3 directories are byte-identical duplicates and should be deleted or
   clearly marked as having **no evidentiary value** (§28.1). No result in this project
   currently has an independent replicate.

3. **[COMPUTE] Fill the genuine gaps:** 81 `stage2means_pd1_reg` cells (9/90 present) and
   11 missing `stage1_pd1_bo_armC` shards out of 1080. Note `stage2noise` is **63/63 in
   all five regimes** — the HANDOFF previously claimed `pd1_reg` was 12/63, which was
   wrong; re-derive completeness from disk, never from this file.

4. **Confirm Ledoit–Wolf/OAS on PD1, especially arm C (§27.44, §27.46).** The whole
   shrinkage result is LCBench-only. `--iw-nu-mode oas` is implemented and verified firing
   but **has never been swept**. §28.2 sharpens this: since σ² only helps off-grid, an
   estimator that adapts to geometry is the natural successor — and note per §28.2 that
   *no* spectrum-based estimator can know the geometry on its own.

5. **Resolve the §27.35 n=5 anchor discrepancy (§28.2).** The result that motivated the
   entire σ² line does not reproduce at n_obs=5 and is sign-flipped. It reproduces at n20
   and n50. Until explained, the small-n claim cannot be cited.

6. **Re-test the kernel-gram shrinkage target on real data (§27.46).** The synthetic test
   used a *random rotation*, which is pessimistic — the real canonical kernel shares inputs
   with `Σ_true` and should share eigenvectors. That test was unfair and its negative
   result must not be cited.

7. **Nonlinear shrinkage is OPEN, not dismissed (§27.46).** Eigenvalue-clipping scored ≈0
   because the Marchenko–Pastur edge was estimated from the median eigenvalue, meaningless
   for a fast-decaying spectrum. Needs a spectral-density estimator (QuEST). Do not record
   clipping as a negative result.

8. **Consider making σ² an estimated quantity rather than a flag.** §27.45 showed the
   inferentially correct value on noiseless data is 0, everything above being deliberate
   bias. If OAS-as-ν carries the regularisation, σ² should go to ~0.

9. **Get more tasks if a mean effect is ever needed.** T is the only lever on the floor of
   `Σ_ind`'s error: `√((r_eff+1)/T)` with `r_eff ≈ 10`. More data *per task* does not help
    (§27.45). This is also the binding constraint on the task-clustered estimand, under
    which **every §28 headline is tied**.

**Do not bother with:** more σ² grid search (the optimum is bracketed in [0.3, 1] off-grid
and is a *penalty* on-grid — §28.2); `--em-init naive` (inert on both methods in all four
BO regimes, |Δ| ≤ 0.35 — §28.5); `--use-mean-prior` (null on all three BO regimes, §28.3);
more α tuning for `--em-shrinkage` (trace-preserving, §27.43); and citing `canon_deep` on
`pd1_reg` (its RNG artifact exceeds its effect, §28.6).

## 6. Hard-won gotchas — read before running anything

1. **Silent no-ops are the recurring failure mode. Five occurrences so far**: additive
   variants overriding an unused hook; `DEEP_KERNEL` assigned without a `global`
   declaration; `--n-inducing` subsetting datasets instead of passing `inducing_points`;
   the diagnostic-print crash; and eight "PD1" cells that were LCBench. **Always verify a
   flag CHANGES the output before building a comparison on it.** A control arm that must
   be bit-identical is the cheapest way to check — §27.2 uses `vanilla_gp`/`random`, which
   ignore pre-training data, to prove arms B and C differ only where they should.
2. **Read the run's recorded `config` before theorising about the code.** The §26.4
   misdiagnosis cost more than the bug did: it declared the highest-value experiment
   blocked while its results sat finished on disk, and every one of the eight suspect
   JSONs said `benchmark: lcbench` in its own metadata (§27.1).
3. **Do not trust "everything has been analyzed".** Twice now the largest correction
   available came from analysing data already collected — most recently a 48-file sweep
   whose replicate priors existed specifically to test the project's shakiest claim, and
   which had never been read (§27).
4. **The dataset split dominates every effect.** HyperBO's 7D NLL moved 0.8 nats from a
   split change. Pair and cluster, or the result is noise.
5. **One pre-training prior cannot rank the neural meta-learners** (§19.4), and this has
   now produced two retracted headlines (§26.1, §27.3). Replicate over priors.
6. **RNG contamination across arms.** Building an MLP consumes global RNG draws and
   reinitialises every neural baseline downstream (moved controls by 0.1 AUC). Each
   baseline now re-seeds before its own pre-training — keep it that way, and assert
   controls are bit-identical.
7. **`${var}` in inline bash gets mangled** by the tool layer and silently becomes empty,
   causing filename collisions. **Always write runner scripts to disk** with
   `write_to_file` and execute those.
8. **Long jobs**: launch with `systemd-run --user --slice=user.slice` so they survive
   disconnection. `<agent>.slice` does not.
9. **Notebook edits**: re-serialize with `ensure_ascii=True, indent=1` or you rewrite
   every em-dash in the file and bury the real change.

---

## 7. Layout

```
_scratch_bo/
  EXPERIMENTAL_RESULTS.md   # the research log; §27 current, insights 1-68
  EXPERIMENT_PLAN.md        # the comprehensive EM evaluation plan (all settings x
                            # BO+regression x LCBench+PD1). Read before launching work.
  HANDOFF.md                # this file
  bo_experiment.py          # BO harness (LCBench + PD1)
  bo_diagnose.py            # regression/calibration harness (LCBench)
  mo_mt_experiment.py       # multi-output / multi-task benchmark
  analyze.py                # budget_table, fig_cost, fig_budget_pareto
  analyze_pd1pool.py        # §27 PD1 arm A/B/C ablation, pooled over priors
  paper/                    # published manuscript + PAPER_CAVEAT_BUDGET.tex
  PAPER_ADDITIONS.tex       # appendix-only additions for the published paper
  results/
    run_*.sh                # every sweep, each documenting WHY it exists
    raw/<study>/*.json      # results; logs/ has the trimmed stdout
```

Analysis entry points: `analyze.py` (`budget_table`, `fig_cost`, `fig_budget_pareto`) and
`analyze_pd1pool.py` (`the build tool run @//mode/opt //pytorch/botorch/_scratch_bo:analyze_pd1pool`).
The §26 numbers were produced by ad-hoc scripts over `raw/bo_multisplit/` and
`raw/hb_hybrid/` — worth folding into `analyze.py`.

---

## 8. Key flags

```bash
# BO (LCBench + PD1)
--split-seed N            # varies config subsample AND dataset split (dominant noise)
--pretrain-seed N         # varies the neural baselines' prior; EM is deterministic
--em-shrinkage A          # M-step shrinkage; optimum grows with target budget
--em-canonical {sumll,hyperbo}
--deep-kernel 32,32
--methods em_additive_hyperbo,em_additive_hyperbo_frozen,...
--hyperbo-iters 25000     # 2000 UNDER-TRAINS HyperBO by 33% regret-AUC

# regression / calibration (LCBench)
--n-inducing M            # < --n-configs => off-grid (Lambda > 0)
--hetero-frac F           # per-task observation subsets (synthetic heterogeneity)
--n-obs-grid 5,20,50
```

Budgets: HyperBO 25k ≈ 3431 s, PACOH ≈ 12616 s, ABLR ≈ 874 s, EM ≈ 20 s per cell.
A 10-cell BO sweep is an overnight job.

---

## 9. Provenance index — every result directory, and its status

`results/raw/` holds 260+ JSON files across 20 studies. **Without this table a new agent
cannot tell which directory backs which claim, or which are superseded.** Statuses:
CURRENT (cite it), SUPERSEDED (a better-designed run replaced it), CONFOUNDED /
RETRACTED (**do not cite**).

| directory | files | analyzed in | status |
|---|---|---|---|
| `pd1pool/hybrid_*` | 36 | **§27.14** | **CURRENT** — arm C × additive hybrid, 3 priors. **Compare only WITHIN this sweep** (§27.10); carries `em_frozen`/`hyperbo_frozen` as internal reference arms |
| `drift_check` | 12 | **§27.11** | **CURRENT** — arm C re-run on current code (prior 0). Shows the PD1 drift changes numbers but not most verdicts |
| `drift_lcbench` | 2 | **§27.13** | **CURRENT** — LCBench drift check. 7/9 methods bit-identical; `ablr`/`ablr_adapt` differ only because the re-run misconfigured them |
| `armc_factorial` | 36 | **UNANALYSED** | **PARTIAL — 3 of 6 cells** (`a0.0/0.1/0.3_sumll`). The three `hyperbo`-target cells are MISSING: they hung 62 h on the §27.12 bug, now fixed. Re-run them before drawing any α × target conclusion |
| `bo_multisplit` | 10 | **§26.1, §26.2** | **CURRENT — the definitive BO measurement**; bit-reproducible on current code (§27.13) |
| `hb_hybrid` | 12 | **§26.3** | **CURRENT** (8 regression cells, `bo_diagnose` LCBench); 4 cells **RETRACTED — ran LCBench**, now `RETRACTED_ranLCBench_*` (§27.1) |
| `canon_decomp` | 12 | §26 predecessor / three-door decomposition | **CURRENT** |
| `em7d_hetero` | 8 | §24–§25 heterogeneity | CURRENT (LCBench-only, *synthetic* sparsity) |
| `em7d_crossover` | 4 | §25.2 | CURRENT — showed the n=5 win does NOT replicate |
| `em7d_calib` | 9 | §23.3 | CURRENT — α optimum grows with budget (0→0.1→0.2→0.3) |
| `em7d_ablate` | 5 | §23.1 | CURRENT — shrinkage worth up to 14 nats |
| `em7d_deep` | 6 | §24.4 | CURRENT numbers, but the **mechanism attribution is retracted** |
| `em7d_gap` | 19 | §24.1–§24.2 | CURRENT — rank + variance-scale hypotheses refuted |
| `momt` | 2 | §23.4 | CURRENT — multi-output/multi-task vs baselines |
| `rerun` | 43 | §20 | CURRENT — corrected-budget re-run |
| `followup` | 11 | §20 | CURRENT — 8-prior headline extension |
| `pd1full` | 36 | §17 | CURRENT — PD1 full-release loader results |
| `hyperbo_debug` | 17 | §19 | CURRENT — HyperBO under-training diagnosis |
| `pd1pool` (`fullboth*`, `fullcand*`) | 48 | **§27.2–§27.4** | **CURRENT** for its own analyses, but **NOT reproducible on current code** (§27.10). Cite §27's conclusions, never its numbers |
| `reg7d` | 3 | §20 7D refresh | SUPERSEDED by `em7d_*`; also only 1 effective prior |
| `canon_matrix` | 8 | LCBench cells; retracted cells → §27.1 | 4 LCBench cells SUPERSEDED by `canon_decomp` (4.0988 on-grid matches its 4.10); 4 cells **RETRACTED — ran LCBench**, now `RETRACTED_ranLCBench_*` |
| `em7d_induce` | 15 | §24.3 | **CONFOUNDED — do not cite.** Subsetted datasets, starving every method |
| `bo_deep` | 4 | §26 predecessor | **RETRACTED — do not cite.** 100% RNG confound |
| top-level `spectrum_a*.json`, `shrinkage_a*.json` | — | §16 / `PAPER_ADDITIONS.tex` | CURRENT — back the published shrinkage table |

**Everything on disk has now been analyzed** — EXCEPT `pd1pool/hybrid_*` (36 files,
complete, unanalysed) and `armc_factorial` (partial, 3/6 cells). Both are flagged above.
The previous version of this table asserted full coverage while `pd1pool`'s 48 files were
unread, so **re-derive this claim rather than inheriting it**: `ls results/raw/*/` against
the section each directory is cited in.

**Off-repo backup.** The 84 result JSONs produced on 2026-08-14–17 were copied to
`~/pd1_results_backup_20260815` while they were still untracked and the
working copy was checked out to an unrelated commit. Once they are committed that copy is
redundant and can be deleted.

---

## 10. Open-source stack state

**UPDATE 2026-08-14: the whole stack has been RESUBMITTED.** `(local changes)` is gone
from every OSS diff; the two `[internal, do not land]` diffs (`<diff-14>`,
`<diff-13>`) correctly remain Unpublished. `<diff-01>` landed back in June.

**Three blockers stand between the stack and landing; two are now fixed.**

**BLOCKER A — GitHub export required (NON_BYPASSABLE).** `github-export-checks` fails on
the amended diffs: *"This diff is not in sync with the Pull Request(s) it is linked to on
meta-pytorch/botorch."* These are OSS diffs mirrored to GitHub PRs, so amending them
requires re-exporting. Use the export link on each diff
(`the per-diff open-source export preview page`) or the PR's import button.
This is not bypassable and blocks landing regardless of review state.

**Confirmed failing on all of** `<diff-05>`, `<diff-04>`, `<diff-12>`, `<diff-08>`,
`<diff-11>`. **It must be the LAST step**: the check compares the diff against the PR, so
any later `jf submit` re-invalidates it and the export has to be redone. For the same
reason the reviewer note must go through `the internal code-review CLI's diff-update command --summary`, which
edits fields without creating a new version, **not** `jf submit`.

**BLOCKER B — device-mismatch bug — FIXED 2026-08-14, amended into `<diff-05>`.**
Arctic flagged it on `<diff-05>` and it was confirmed present in
`multioutput_empirical_1d_gp.py`, `posterior()`, bool-`observation_noise` branch. Both
`avg_noise` assignments now use `.to(X)`.

**The test for it is CUDA-gated and therefore does NOT run on a CPU host.** A
dtype-based stand-in was written first and **rejected after it failed with the fix
applied**: the posterior takes its dtype from the model's float64 parameters, so it reads
`float64` whether or not the fix is present and cannot discriminate. Skipping honestly
beats a test that always passes. Suite is **281 pass / 0 fail** with both fixes in.

**BLOCKER C — Windows temp-file cleanup — FIXED 2026-08-14, amended into `<diff-12>`.**
`test_read_parquet_cleans_up_temp_file_on_write_failure` failed on `windows-latest` for
Python 3.11 and 3.14: `lcbench.py` called `tmp_path.unlink()` *inside* the
`NamedTemporaryFile` `with` block, and Windows refuses to unlink an open file. The write
and replace paths are now one `try` whose cleanup runs after the handle closes. **Only
Windows CI can confirm this** — it passes on Linux either way.

**NOT OURS, BUT FIXED — the lint failure.** `check-requirements-versions` was failing on
every BoTorch diff (land-blocking): `.pre-commit-config.yaml` pinned `ruff-api` 0.2.0 and
`ufmt` v2.9.0 against a `requirements-fmt.txt` holding 0.2.1 and 2.9.1.

**Cause: `<diff-15>`** ("[MSDK] Update pyfmt component on FBS:master", landed 2026-08-12,
`ca3823eca614`), which bumped exactly those two components in
`tools/lint/pyfmt/reqs/requirements-fmt.txt` — the hook's source of truth — without
updating BoTorch's config. The other three pins (`black`, `usort`, `stdlibs`) still matched
exactly, which is what confirms the config tracks that file and only these two drifted.

Fixed in **`ecb7b93d425e`**, deliberately committed **standalone on
`remote/<repo>/stable`**, not stacked on the empirical-GP work: it is repo hygiene that
unblocks every BoTorch diff and should land on its own. It still needs submitting.

**FALSE ALARM — Arctic's tutorial `[error]`.** The claim that `<diff-09>` uses
`EmpiricalOneDimensionalKernel` without importing it is **wrong**: cell 2 imports it and
cell 32 uses it, in that order. A cumulative undefined-name scan over the notebook at
`<diff-09>`, `<diff-08>` and `<diff-11>` reports **no undefined names at any of the
three**. The companion claim that the tutorial needs a `tutorials.json` entry is also
wrong — it is registered at `website/tutorials.json:138`, added by an already-landed diff.
Do not "fix" either.

**Also worth pre-empting with the reviewer:** Arctic scored *drift from intent* 26/100 and
listed our amendments as unrequested scope expansion — tensor `observation_noise`
broadcasting, the new `UnsupportedError`s, device/dtype sync, and a kernel fast-path fix.
They are all defensible, but the reviewer accepted a narrower diff, so call them out
explicitly rather than letting them be discovered.


12 diffs ending `<diff-11>`, all **Accepted**. Four were **amended after acceptance** and
need re-review before landing:

| diff | what was amended |
|---|---|
| `<diff-05>` MultiOutput | raise instead of silent zero-noise; real broadcasting for `(m,)`/`(q,1)`; imports hoisted; tests updated; qualitative caveat |
| `<diff-04>` MultiTask | qualitative caveat (cross-task transfer is where it wins) |
| `<diff-12>` LCBench loader | pyarrow probe; temp-file cleanup on write **and** replace failure; 2 new regression tests |
| `<diff-10>` CI integration | `LCBENCH_TUTORIAL_NAME` + collection-time guard against notebook rename |
| `<diff-08>` tutorial D-F | notebook §D reframed as qualitative (no aggregate claim) |

Full suite green at stack top: **281 pass, 0 fail** (`//pytorch/botorch:test_empirical_gp`
+ `:test_utils`).

Two reviewer comments were **deliberately not actioned**, with reasons:
- **MultiTask GPU-sync** (`_extract_progression_and_task`): the suggested fix skips
  validation on the forward paths, which silently rounds non-integer task labels at
  *query* time. The sync is unavoidable if we want to raise. Reviewer marked it
  informational; trading a correctness check for an unprofiled perf win is the wrong call.
- **LCBench `momentum` log-scaling**: verified correct as written. Momentum ∈ [0.100,
  0.982] (no non-finite risk) and the median 0.3033 matches the log-uniform prediction
  0.3134, not uniform 0.5411 — LCBench sampled it log-uniformly.

## Pending: per-shard provenance stamp (agreed 2026-08-22, NOT yet applied)

**Why.** `PROVENANCE.json` records `commit` at QUEUE-GENERATION time, but cells run hours
to days later and the binary can be rebuilt in between (§27.15). So the stage commit is
not a statement about what actually produced a given shard. Shard JSONs carry 64 config
keys and **no code version at all**. Separately, `iw_nu_mode` is not in the recorded
config, and the estimated ν is printed to stdout only — it survives in
`results/logs/$STUDY/*.txt`, which are untracked.

**What to add**, written by each harness at result-write time (`bo_experiment.py` and
`bo_diagnose.py`, both call `atomic_write_json`):

* `code_commit` — `sl whereami` at EXECUTION time, not queue-gen time
* `code_dirty` — whether the working copy had uncommitted changes
* `harness` + `run_utc`
* the RESOLVED `iw_nu_mode` and the estimated `iw_nu` returned by `resolve_iw_nu()`,
  so the sweep's headline number stops living only in an uncommitted log

**Scope:** going forward only. Existing ~9,900 shards keep stage-level PROVENANCE; they
are NOT backfilled, because the only commit available to backfill is the queue-gen one,
which is precisely the value this change exists to stop trusting.

**Timing:** must NOT be applied while `em-autonomous` is live. Editing either harness
triggers a Buck rebuild, so cells launched before and after the edit would run different
binaries — splitting a single stage across two, the exact hazard being fixed. `em-oas`
was stopped on 2026-08-22 before doing any work (0 shards) so the first OAS data is
stamped from the start. Apply once `em-autonomous` is inactive, then relaunch `em-oas`.


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
