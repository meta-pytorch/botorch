# Empirical GP — master index

**If you are picking up this work cold, read this file and nothing else first.**
It is a router, not a corpus. Every section points at the document that owns the detail.

Last updated 2026-09-08.

There are two active research workstreams and one open-source line. They live in
**two separate Sapling stacks that cannot be checked out at the same time**, which is the
single most confusing thing about this project. Section 1 explains how to get to each.

---

## 1. Stack map

`sl log -r 'draft() and head()'` returns 210 heads across all of the author's work. Six of
them are empirical-GP related and are bookmarked (`sl bookmark | grep egp`). All are backed
up in commit cloud (`sl cloud status` → 210 heads, 6 bookmarks, Success).

| Bookmark | Head | Commits | Status |
|---|---|---:|---|
| `egp/bo-live` | `dfe906239496` | 114 | **LIVE — stack A, BO research.** |
| `egp/mo-live` | `fec6047c3d71` | 42 | **LIVE — stack B, multi-output shrinkage.** |
| `egp/em-convergence` | `a5f393975301` | 16 | Older OSS model line; its diffs have landed. Triage. |
| `egp/stale-bo-106` | `63a4f039fb60` | 106 | **STALE.** Exact prefix of stack A. |
| `egp/stale-mo-12` | `403487e52c7c` | 12 | **STALE.** Exact prefix of stack B. |
| `egp/stale-bo-6` | `8f7b0650be50` | 6 | **STALE.** Exact prefix of stack A. |

The three stale heads were verified obsolete by comparing ordered commit-message sequences
against the live stacks; each matched its live counterpart's prefix with zero differences.
They are rebased copies onto different master revisions, nothing more. They are kept only
until stack B is folded into stack A.

**Both live stacks share the same four tutorial commits as their base** (`<diff-10>`,
`<diff-09>`, `<diff-08>`, `<diff-11>`), then diverge. That shared base is why the merge
in §7 is expected to be cheap.

### How to read a document that is not in your checkout

Stack B's files do not exist on disk while stack A is checked out, and vice versa. Do not
conclude that a file is missing — read it out of the revision:

```bash
sl cat -r fec6047c3d71 <repo>/<internal-workflow-system>/flow/projects/botorch/empirical_gp/multioutput_shrinkage/README.md
sl manifest -r fec6047c3d71 | grep multioutput_shrinkage
```

---

## 1b. What is NOT in this appendix

A **third** workstream exists and was deliberately left out of the open-source
appendix: the **EM-convergence / early-benchmark line**, on the local bookmark
`egp/em-convergence`. It holds roughly 50 files not present here, including the
EM convergence study, a hierarchical GP prior over sample paths, an EM-vs-
hierarchical comparison, curve-completion analysis, and the FBLearner `bo_benchmark`,
`gift_eval` and `lcbench` benchmark integrations.

It is not lost -- it is backed up in commit cloud under that bookmark -- but it has
**not** been scrubbed of internal references and is **not** covered by anything in
this document. Do not assume the appendix is the whole project.

## 2. Workstream map

| Workstream | Lives in | Authoritative doc | Rules |
|---|---|---|---|
| **A. BO research** (HyperBO/PACOH baselines, EM levers, covariance shrinkage) | `_scratch_bo/` | `EXPERIMENTAL_RESULTS.md` (S1–S33), routed by `HANDOFF.md` | `_scratch_bo/.llms/rules/metrics.md`, `.../em_noise.md` |
| **B. Multi-output shrinkage** (ρ, `PerOutputBaseKernel`, Kronecker) | `<repo>/<internal-workflow-system>/flow/projects/botorch/empirical_gp/multioutput_shrinkage/` | `README.md`, routed by `HANDOFF.md`; plus `LCBENCH_DATA_QUALITY.md` | `.llms/rules/gp_evaluation.md` (3 identical copies) |
| **C. Open source + tutorial** | `<botorch>/botorch/models/empirical_gps/`, `.../tutorials/empirical_gaussian_processes/` | §5 below | `botorch/.llms/rules/botorch.md` |

### ⚠️ Vocabulary collision between A and B

The two workstreams use the same English words for different mechanisms. Conflating them
will produce wrong conclusions.

| Term | In workstream A | In workstream B |
|---|---|---|
| "shrinkage" / α | `trace_matched_shrinkage` applied in the **EM M-step at pre-training**. A plain float. | — |
| ρ | — | A learnable **Hadamard mask on cross-output covariance blocks at conditioning time**. |
| "additive" | A HyperBO base kernel on the EM model over **progression** inputs. | `PerOutputBaseKernel`, block-diagonal across **outputs**. |

Different mechanisms, different stages of the pipeline.

---

## 3. What is settled

Two headline claims survive. **Each is stated with its estimand, because the two estimands
disagree here and conflating them has already required a published correction.**

- **fixed-task** — "wins on *this* suite". Datasets/tasks are the population.
- **random-effects / task-clustered** — "wins on a *new, unseen* task". With P=5 replicates
  the critical value is t(4) = 2.78, not 1.96.

### A. BO — direct covariance shrinkage works on-grid, and only on-grid

Spherical OAS shrinkage of the empirical covariance toward the base kernel gives a **broad
plateau at α ≈ 0.08–0.20 on PD1 arm A (on-grid)**, worth roughly **10 % budget-to-target**
(α = 0.10: −2.19 ± 1.04 evals @0.01, −3.14 ± 1.64 @0.001, against a control of 21.66). The
parameter-free OAS estimator picks **α = 0.1326 ± 0.0021**, essentially on the empirical
optimum, with no tuning. The same intervention is a **plain null on PD1 arm C (off-grid)**.
Whitening was tested and **rejected** — it estimates α ≈ 0.633, a ~5× overshoot, is 25×
more variable, and significantly *hurts* (+2.13 ± 1.00).

Estimand: **fixed-task** throughout (the same 23 PD1 tasks, replication over 9 priors,
t(8) = 2.31). Detail: `EXPERIMENTAL_RESULTS.md` §32–§33.

> #### ⚠️ Do not merge this with the EM-noise result — they run in OPPOSITE directions
>
> There are two different shrinkage mechanisms in this stack and their geometry
> dependence is **inverted**. Conflating them will produce a wrong prescription.
>
> | Mechanism | Where it acts | arm A (on-grid) | arm C (off-grid) |
> |---|---|---|---|
> | EM likelihood noise σ² (`--em-likelihood-noise`) | E-step eigendirection retention `λ/(λ+σ²)` at pre-training | **monotone penalty** (+1.83 evals at σ²=1) | **interior optimum** in [0.3, 1] (−3.00 evals) |
> | Direct covariance shrinkage α (`--em-shrinkage`) | blends the finished covariance toward the base-kernel gram at conditioning | **helps**, plateau 0.08–0.20 | **null** |
>
> The σ² direction is what `METHODS.md` §2 predicts from the Nyström geometry (on-grid
> `Λ(X) = 0`, so `Σ_ind` is used directly and shrinking discards signal). **The α result is
> not explained by that argument and currently has no accepted mechanism.** §32.4 offers a
> partial one — arm C's budgets sit near the censoring ceiling (39–47 of 50), which
> compresses any effect — but that is a power explanation, not a mechanistic one. Treat
> "why does α invert relative to σ²" as **open**.
>
> The two knobs are also not interchangeable: σ² *reduces* the covariance trace, `--em-shrinkage`
> is trace-preserving. That is why §27.9 found the latter useless in the role σ² was playing.

> #### ⚠️ arm C: within-stage comparisons only
>
> arm A's `shrink000` control reproduces `stage1 base` exactly to 4 dp. **arm C's does
> not** (39.7585 vs 39.7343), with 27/36 method-runs diverging by up to 2e-1. This is not
> nondeterminism — it is a genuine harness change somewhere in the §30–§32 interval. Do not
> compare arm C numbers across stages.

Supporting, durable (no sampling noise, so the most robust claim available): on LCBench, EM
reached better budget-to-target using roughly **128× less pre-training time** than HyperBO
on valid cache-miss timings.

### B. Multi-output — full-covariance joint models lose; two narrow interventions win

From the final 2026-08-20 campaign: six sweeps, 3.44 M scored rows, 35 datasets, 0 failed
replications, observation-predictive scoring, per-replicate splits.

- **Full-covariance joint models (`multi`, `kron`, `canonical_single`, `canonical_kron`) are
  significantly *worse* than independent per-output GPs on RMSE.** For `multi` specifically,
  fixed-task on `Train/loss` at cutoff 0.3, Holm-corrected: it wins **1/35, 3/35, 6/35**
  datasets at n_hist 25/50/100 (all three significantly *worse*, p ≤ 0.006), and is never
  significantly better at any budget. "Wins" here counts datasets on which that arm beats
  independent, not statistical victories.
- **`shrunk` (learned ρ) is a small but genuinely significant win** — 6 replicate-level RMSE
  cells (effects −0.0007 to −0.0041) plus 4 NLL cells. It is never significantly worse than
  independent anywhere. Its advantage *grows* with history (fitted ρ rises 0.43 → 0.82
  across n_hist 25 → 400).
  **⚠️ This number is meaningless without the noise regime.** ρ and the observation noise
  are substitutable: median fitted ρ moves 0.058 → 0.990 on *unchanged data*, purely by
  raising the noise floor. The claim above is at the study's default fixed noise. ρ is still
  not redundant — its NLL edge *grows* with noise (2 → 6 of 6 cells) — but whether it
  survives a *fitted* noise is open question 4 in §9.
- **`additive` (`PerOutputBaseKernel`) is a tail-risk fix, not a general improvement** —
  best arm on the *mean* at n ≤ 100 (0.2105 vs `single`'s 0.2614 at n=25) but worse on the
  *median* (0.0959 vs 0.0762), winning only 21/35 datasets. It rescues rank-deficiency
  collapses and costs a little on the typical dataset.
- **The failure mechanism is a rank ceiling.** The empirical covariance is `UᵀU/N`, so its
  rank is capped by the number of historical curves. Rank and coverage track exactly:
  24/50 → coverage 0.452, 49/50 → 0.707, 50/50 → 0.850. Nominal is **0.9545**, not 0.95.
- **Per-output outputscale is established.** `PerOutputBaseKernel(..., per_output_outputscale=True)`
  gives **6 wins / 0 losses / 6 ties across the 12 comparison cells**, on paired Wilcoxon
  with Holm, computed from a 525-cell evaluation (35 datasets × 5 seeds × 3 history sizes,
  0 failures). Every win is on `Train/loss`, every tie on `val_accuracy`, never a loss. The
  mechanism is visible in sharpness — it reallocates predictive width between outputs.
  Caveat: the fitted scales are *more* unstable with m free parameters (sd 23.9 → 1105.9)
  and accuracy improves anyway.
- **The empirical prior is worth 3.5–4.1× median RMSE** over a *properly fitted*
  `SingleTaskGP`, and the advantage grows with history. The gain comes from the **mean**,
  not the covariance.

Detail: `multioutput_shrinkage/README.md` §1–§8h.

---

## 4. What is RETRACTED — do not re-cite

This is the highest-value section in this file. Each of these was published in an earlier
revision of a document that still exists on disk.

### Workstream A

1. Early HyperBO rankings were affected by **undertraining**.
2. **All PD1Lite conclusions (§9–§12) are superseded** by the full-release PD1 in §17.
   PD1Lite has only ~50 shared configs; the failure to reproduce the paper was the pool
   size, not the method.
3. **Regret-AUC is demoted from headline status.** It measurably misleads on this data: on
   PD1 arm C, AUC says `em_finetuned` crushes `vanilla_gp` (z = −9.93) while
   budget-to-target on the same runs says **tied**.
4. An additive-kernel flag/path was a **silent no-op**.
5. Single-prior PD1 rankings were insufficient — retracted.
6. The pooling interpretation in §27.14 — retracted.
7. §27.20 corrected a **double-counted between-prior standard error**.
8. §27.24 corrected a divergence interpretation that read validation accuracy without
   inspecting training loss.
9. §27.31 withdrew an off-grid generalization drawn from on-grid mean-transfer nulls.
10. **§27.35's `n_obs=5` anchor does NOT reproduce** — the 63-cell sweep comes out
    sign-flipped. Treat as **unresolved**; do not cite.
11. Sample-rank matching, per-direction SNR, and model-discrepancy explanations of the
    regularization optimum were all **refuted**.
12. **Original Stage 3 is not an independent replication** — 675/675 shards were
    byte-identical to Stage 1. Computationally complete, evidentially worthless.
13. The apparent mean-transfer gain was **confounded with canonical-kernel transfer**; OFAT
    attributes ~all of the effect to the kernel.
14. **The IW-ν sweep was a saturated instrument.** `α = (ν+N+1)/(K+ν+N+1)` with N ≈ 400 mapped
    intended α of 0.075–0.652 onto 0.9698–0.9710. Its "tight null" is not evidence of no
    effect — it is evidence the knob was inert.
15. Whitened shrinkage — **rejected** on PD1.

### Workstream B

16. **"The joint model wins NLL" does not survive the random-effects estimand** — significant
    in 8/10 fixed-task cells, **0/10** replicate-level.
17. **"No multi-output variant beats independent on RMSE at any setting" is false** on the
    full P=5 grid — that was a power artifact of the P=3 partial grid.
18. **The `additive` negative result (0/35 wins, 0.99 coverage) was a bug**, not a finding.
    `_set_noise` only fitted hyperparameters on the `--fit-noise` path, so every arm with
    trainable hyperparameters ran at *initialization*. Its `ScaleKernel` sat at outputscale
    0.693 against a prior diagonal of 0.903.
19. **The ~11× advantage over `KroneckerMultiTaskGP` was the same bug.** Against fitted
    baselines it is 3.5–4.1× on the median.
20. **§8d's "`additive` loses `Train/loss` RMSE at n ≥ 200" was wrong.** The penalty is flat
    at ≈ +0.015 across a 16× range; only the between-split variance shrinks.
21. **§8g's `p_diverged` moderator conclusion was drawn from one cell.** Across all ten,
    five are significant after Holm — contamination was *masking* the penalty.
22. The `ridge` arm was **removed as a duplicate of the noise axis** — a kernel ridge and
    observation noise are the same operator for any scored point (agreement 3e-9 on
    posterior means).
23. `kron_shrunk` ≡ `kron`, bitwise, in every cell. Once the covariance is
    Kronecker-constrained, ρ has nothing left to do.

---

## 5. Open-source state

Landed in OSS BoTorch: `<diff-01>` (EmpiricalOneDimensionalGP), `<diff-03>` (shared
helpers), `<diff-05>` (MultiOutput), `<diff-04>` (MultiTask), `<diff-02>` (EM-based GP),
`<diff-07>`, `<diff-06>`, `<diff-12>` (`botorch/utils/lcbench.py`).

**Accepted but NOT landed — the four tutorial diffs**, all blocked on one thing:

| Diff | Content | Blocker |
|---|---|---|
| `<diff-09>` | Tutorial Part 2 A–C | `github-export-checks` NON_BYPASSABLE — diff out of sync with its GitHub PR |
| `<diff-08>` | Tutorial Part 2 D–F | same |
| `<diff-11>` | Tutorial Section G (BO) | same |
| `<diff-10>` | Internal tutorial CI integration (never exported) | rides along |

Re-export via `https://www.the per-diff open-source export preview page` — there is
no CLI. **Re-export must be the LAST action**, because any subsequent `jf submit`
re-invalidates the check (recorded in commit `401accbef5db`).

Other blockers are already resolved: the multi-output `posterior()` device mismatch was
fixed into `<diff-05>`; the Windows temp-file unlink into `<diff-12>`. The
pre-commit / requirements-fmt version mismatch is **not ours** —
`.pre-commit-config.yaml` is byte-identical to master and `requirements-fmt.txt` is not
mirrored into <repo>. Needs a decision, not a fix.

Internal, do-not-land: `<diff-13>` (BO research scaffolding), `<diff-14>` (HyperBO &
PACOH-GP baselines).

**Not yet open-sourced, implemented and tested in stack B:** `learn_cross_output_shrinkage`
(ρ), `PerOutputBaseKernel` incl. `per_output_outputscale`, `filter_diverged_curves`,
`build_sliding_window_curves`, `kronecker_factored_covariance`.

---

## 6. Data provenance — none of it is Meta-internal

| Data | Public? | Where it comes from | Status |
|---|---|---|---|
| **LCBench** | Yes | `botorch/utils/lcbench.py` downloads from `raw.githubusercontent.com/ltiao/LCBenchLite`, caches in `~/.cache/botorch/lcbench`. Internal runs patch only the `_read_lcbench_parquet` seam to internal object storage, so OSS and internal cannot diverge. | **Solved** (`<diff-12>`, landed) |
| **PD1** | Yes, CC-BY 4.0 (Wang et al. 2021, arXiv 2109.08215). **See `PD1_DATA.md`** for the two sources, the three convention differences that change the numbers, and how to rebuild the internal PD1Lite pool from the public release. | Hand-downloaded to `~/pd1_paper/pd1`; read by `_scratch_bo/pd1_full_data.py`. Four release files = 234 MB; only the two *matched* files (~50 MB) are used. **The canonical download URL is not recorded anywhere in this repo or in the release's own `README.txt` — recover it from the paper before writing the loader; do not guess.** | **No OSS loader yet.** Needs a `botorch/utils/pd1.py` mirroring `lcbench.py`. |
| **Generated results (A)** | n/a | `_scratch_bo/results/` = **701 MB**: raw 576 MB, logs 88 MB, prior_cache 17 MB, figures 9 MB, `MANIFEST.tsv` 8.2 MB. ~30 k files, committed into the stack. | Cannot go in a PR. |
| **Generated results (B)** | n/a | Only the headline `d7_all_arms_fitted_curves.parquet` (10 MB) + CSVs are committed; the rest lives at `<internal-object-store>/empirical_gp/multioutput_shrinkage/`. | The disciplined model to copy. |

**PD1 loader conventions that must ship with the loader** (they change the numbers; see the
docstring of `pd1_full_data.py`): objective `best_valid/error_rate`, negated so higher is
better; diverged trials with no evaluation assigned error 1.0 rather than dropped;
phase-1-preferred de-duplication of the 2277 pairs present in both phases; inputs min-max
scaled using the *declared* bounds, not observed extrema.

**internal object storage trap (workstream B).** `final/d2_noise_frontier.parquet` was briefly a
*complete-looking* file from the pre-scoring-fix run — right row count, right dataset count,
but only 4 arms and latent-posterior scoring. It now sits in
`INVALID_prefilter_do_not_use/`. The tell was that the "complete" file was *smaller* than
the 70 %-done corrected one. **Audit the arm list, not just row counts and filenames.**

---

## 7. Traps — every one of these actually happened here

Merged from workstream A's five recurring failure modes (`HANDOFF.md`) and workstream B's
trap catalogue (`HANDOFF.md` §6).

### Instrumentation

1. **Silent no-ops.** A flag is accepted and logged but changes nothing. Prove a flag moves a
   posterior or a metric *before* launching a matrix behind it.
2. **Instruments incapable of observing the effect.** A saturated, bounded or structurally
   inert control cannot support a "no effect" conclusion. The IW-ν sweep is the canonical
   example (retraction 14).
3. **Tests that certify the bug.** Source greps, AST checks, or tests duplicating the faulty
   logic pass while production stays wrong. Exercise the real path; compare real artifacts.
4. **Fixes that introduce new defects.** Seven rounds of adversarial review here, and rounds
   5, 6 and 7 each fixed defects *introduced by the previous round's fixes*.

### Statistics

5. **Name the estimand.** Fixed-task and random-effects disagree, and "tied" means different
   things under each. §27 conflated them and had to be corrected.
6. **A median and a mean can disagree in sign.** RMSE across datasets is right-skewed
   (mean 0.26, median 0.076). `additive` is the best arm on the mean and worse than
   independent on the median at the same budget. Say which you mean.
7. **A cell becoming significant is not an effect appearing.** Read the effect-size table
   beside the significance table, never instead of it.
8. **Testing one cell and reporting it as the check.** Run the whole grid, then Holm-correct.
9. **A marginal p on a cell you chose is not marginal evidence.** p = 0.060 out of ten cells
   is the multiplicity it is.
10. **Never summarize a skewed or near-binary quantity with a median.** Coverage medians read
    1.000 where the mean was 0.78, hiding an 8-point calibration gap.
11. **Bare win-rates mean nothing without a test.** Report Holm-corrected sign/Wilcoxon *and*
    the correlation-adjusted z. The adjustment can only withdraw a finding, never create one.
12. **A missing model arm is silently scored as a loss**, because `NaN < x` is `False`.
13. **Estimator errors move headline claims.** State the estimand and re-derive the estimator
    before interpreting.

### Modelling

14. **Check that a "new" mechanism is not an existing one in disguise.** Construct both and
    diff the posteriors — reasoning about where terms sit in the equations is what got the
    kernel-ridge/observation-noise question wrong.
15. **Two shrinkage knobs, opposite geometry dependence.** EM likelihood noise σ² helps
    *off*-grid and is a monotone penalty *on*-grid; direct covariance shrinkage α helps
    *on*-grid and is null *off*-grid. See the boxed warning in §3A. Do not generalize a
    prescription from one to the other, and do not reuse the `METHODS.md` §2 Nyström
    argument to explain the α result — it predicts the wrong sign there.
15. **Score the observation-predictive distribution, not the latent one.** `observation_noise=True`.
    Scoring `posterior().variance` silently hands kernel-side arms extra predictive variance.
16. **GPyTorch's `likelihood.noise` is a VARIANCE.** "noise = 1e-4" is σ = 0.01.
    `GaussianLikelihood` defaults to `GreaterThan(1e-4)` on it, so smaller values *raise*
    rather than apply.
17. **The EM pre-training noise is an effective-rank knob, not a noise level** — see
    `.llms/rules/em_noise.md`. Its optimum **inverts** between on-grid and off-grid.
18. **Fit noise first with ρ frozen, then ρ.** Fitting them jointly moves ρ before its
    dedicated fit and makes the fixed- and fitted-noise arms different estimators.
19. **Regression quality does not universally predict BO.** Off-grid, some regression metrics
    *anti*-predict. The covariance prior improves NLL while being the worst primary-BO factor.

### Data

20. **LCBench `Train/loss` diverges** (to 1e13 and `inf`) on 5 of 35 datasets. One such curve
    destroys any corpus statistic. It is a *loss* phenomenon — inspecting `val_accuracy`
    alone wrongly concludes no filtering is needed. Fix per-curve on the **input** with
    `filter_diverged_curves`. Do not drop whole datasets (74 % of the benchmark to excise
    1.8 % of curves) and do not filter on an **outcome**.

### Infrastructure

21. `the build tool run ... &` does not survive its parent shell — use `setsid` on the built `.par`.
22. **Verify a backup's *effect*, not that a process exists.** `pgrep -f archive-loop` matched
    the shell commands checking on it; a dead archiver reported "alive" for hours.
23. **A sweep that suddenly gets fast is a sweep that has stopped computing.** The `$HOME`
    LCBench cache does not survive machine reassignment; run `seed_lcbench_cache.sh` first.
24. Never edit a bash script while it is running — bash reads by byte offset.
25. Use absolute paths under `systemd-run`; relative script paths have failed silently here.
26. `pgrep -f` misses processes whose identity is in an env var (`PAR_MAIN_OVERRIDE`);
    check `/proc/<pid>/environ` and prefer `ps`.
27. **`.DONE` files are not a reliable completion test.** Use `cells_present == cells_expected`
    in each `SUMMARY.json`, and `shards_missing == 0` where available.
28. **`MANIFEST.tsv` is a launch ledger, not a success ledger.** A row proves a command was
    launched, nothing more.

---

## 8. Current computational state

Workstream A: **complete at the summarized v2-cell level** — 32/32 stage directories,
**2,079/2,079 cells**, no failures or pending cells in the current `SUMMARY.json` files.
The failures recorded in `results/raw/v2/STATUS.txt` are resolved history. Latest launch was
the Stage 3 arm-C OAS replication, 2026-08-23.

Workstream B: **complete** — all six sweeps finished 2026-08-20, 0 failures.

Nothing is known to be running. Before relaunching anything, check live processes with `ps`
(not `pgrep`) and check user systemd units.

Resume command for workstream A, if shard-level verification finds a discrepancy:

```bash
STAGES=0 CELLS_PARALLEL=3 \
  bash _scratch_bo/results/run_autonomous.sh
```

Resume for workstream B: `./seed_lcbench_cache.sh` **first**, then `./run_sweeps.sh`.

---

## 9. How to continue — ranked

1. **Wire the Kronecker-factored covariance into a model arm.** It is implemented and tested
   (`kronecker_factored_covariance`) and it fixes the diagnosed mechanism — rank 24/50 → 50/50
   at n_hist=25, coverage 0.700 → 0.805 — but it has never been evaluated as a model arm.
   This is the top open task in workstream B. Note it does *not* currently buy accuracy
   (median RMSE 0.0824 vs `single`'s 0.0762), so the honest framing is a calibration fix.
2. **Replicate direct OAS shrinkage on a second benchmark** before presenting it as general.
   Right now it is one benchmark, one arm.
3. **Explain the `n_obs=5` non-reproduction** (retraction 10) or stop citing the region.
4. **Does ρ survive a fitted noise?** ρ and the observation noise are substitutable — median
   ρ moves 0.058 → 0.990 on unchanged data purely by raising the noise floor. Every ρ claim
   must state the noise regime. D2 was designed to answer this.
5. **ρ for the multi-task model.** The same mask applies to
   `MultiTaskEmpiricalOneDimensionalGP.forward`, condition
   `task_idcs_1[:, None] != task_idcs_2[None, :]`.
6. **Compute the SEM floor at infinite seeds before spending more compute.** LCBench has 35
   datasets sharing 2000 configs; PD1 has 23 tasks. Workstream A hit exactly this power
   ceiling on PD1 and formalized it — PD1 BO cannot resolve these methods at any budget.
7. **Stale tutorial text.** Cell 29 still hedges Section D as illustrative; the evidence now
   supports a stronger and *different* claim. Cell 35's assertion that the additive ICM
   "delivers the calibration gain" still has no ablation behind it.

---

## 10. Document inventory

Workstream A — `_scratch_bo/`:

| File | Role |
|---|---|
| `EMPIRICAL_GP_INDEX.md` | this file — the router |
| `HANDOFF.md` (57 KB) | operational handoff. **Its opening snapshot and OAS relaunch instructions predate S32–S33.** |
| `EXPERIMENTAL_RESULTS.md` (377 KB) | S1–S33, the full record |
| `METHODS.md` | the model; §2 has the on-grid/off-grid geometry that explains most results |
| `EXPERIMENT_PLAN.md` | the Stage 0–4 design. **Header says "not yet started"; it is done.** |
| `PAPER_INTEGRATION_PLAN.md` | what goes into the paper |
| `ADVERSARIAL_REVIEW_*.md` (6 files) | rounds 1–7 of adversarial review |
| `PD1_DATA.md` | the two PD1 sources, their differing conventions, and the PD1Lite rebuild recipe |
| `.llms/rules/metrics.md` | metric policy — budget-to-target leads, AUC is a cross-check |
| `.llms/rules/em_noise.md` | the effective-rank knob |

Workstream B — `.../empirical_gp/multioutput_shrinkage/` (read via `sl cat -r fec6047c3d71`):

| File | Role |
|---|---|
| `HANDOFF.md` | the map. **Dated 2026-08-19; README was revised 2026-08-21.** |
| `README.md` | protocol and results, §1–§8h |
| `LCBENCH_DATA_QUALITY.md` | the divergence defect — affects any LCBench or PD1 study |
| `.llms/rules/gp_evaluation.md` | evaluation rules (3 identical copies across botorch/botorch_fb/<internal-workflow-system>) |

Key drivers, workstream A: `bo_experiment.py` (main BO harness), `bo_diagnose.py`
(regression/calibration), `analyze.py` + `analyze_pd1pool.py` (aggregation and inference),
`prior_stability.py`, `stage4.py` (synthesis), `gen_stage_queue.py`, `summarize_stage.py`,
`em_noise_selection.py` (LW/OAS/RBLW selectors), `prior_cache.py`, `pd1_full_data.py`,
`lcbench_io.py`, `regen_nb.py` (notebook re-render).

Key drivers, workstream B: `study.py` (arms and scoring), `analysis.py` (tests, t(P−1)),
`main.py` (CLI/sweep/resume), `p5_headline.py` (every headline table from the committed
grid), `vendor_results.py`, `tests/` (55 tests, statistical ones mutation-checked).


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
