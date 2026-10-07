# Empirical-GP BO: Baseline Study (EM vs HyperBO vs PACOH-GP)

> **Router:** `EMPIRICAL_GP_INDEX.md` is the entry point for this project. It carries the
> consolidated retraction list across both research stacks — read it before citing anything
> from this file. This log is chronological and contains superseded work.

<!-- INDEX:BEGIN -->
## Index of §26–§27 sections

Generated 2026-08-21. This log is chronological and **contains retracted work** —
see `HANDOFF.md` §3 and `EMPIRICAL_GP_INDEX.md` §4 for what not to cite.
Start from `METHODS.md`.

| § | topic |
|---|---|
| **20.1** | The headline, corrected — and it changes the ranking |
| **20.2** | Which earlier conclusions survived |
| **20.3** | Provenance |
| **21.1** | A silent no-op: the additive-kernel variants were never additive |
| **21.2** | Why the published ordering is unreproducible: we are solving a different problem |
| **22.1** | The prior work never used AUC |
| **23.1** | Shrinkage repairs the 7D NLL, and the fix is large |
| **23.2** | Two structural hypotheses for the residual gap — both refuted |
| **23.3** | The gap is regime-dependent, and EM wins where the paper claims it does |
| **23.4** | Multi-output and multi-task: the additive base is what mattered |
| **24.1** | Refuted: the rank ceiling |
| **24.2** | Refuted: variance scale |
| **24.3** | Confounded, still open: off-grid capacity |
| **24.4** | What worked: a learned kernel metric |
| **24.5** | Where this leaves the story |
| **25.1** | Design |
| **25.2** | EM never significantly wins |
| **25.3** | The split matters more than the effects we were chasing |
| **25.4** | BO: EM is competitive but mid-tier |
| **25.5** | What the evidence now supports |
| **26.1** | Paired, multi-split BO: EM is top-tier |
| **26.2** | EM is Pareto-dominant on performance vs cost |
| **26.3** | HyperBO's pre-trained kernel as an ADDITIVE component is the best variant |
| **26.4** | The PD1 "pool no-op" was a mislabelling, not a code bug |
| **27.1** | The real cause: two sweep scripts omitted `--benchmark` |
| **27.2** | Design: a two-factor protocol ablation, replicated over priors |
| **27.3** | The §21.3 "HyperBO flips to first" claim does not survive replication |
| **27.4** | The honest result: EM ties on real heterogeneity, and does not win |
| **27.5** | What this does to the story |
| **27.6** | Power ceiling: PD1 BO cannot resolve these methods, at any budget |
| **27.7** | Correction: §27.6 answered only one of two questions, and the other is resolvable |
| **27.8** | §26.1 vs §27.7: it is the BENCHMARK, not the metric — and EM does worse than §27.7 said |
| **27.9** | EM has been run UNTUNED in every BO comparison, and α alone does not fix it |
| **27.10** | The hybrid sweep cannot be merged with the 2026-08-08 arm-C data: code drift |
| **27.11** | The drift changes numbers, not most conclusions — but one §27.8 row is retracted |
| **27.12** | BLOCKER FIXED: `--em-canonical hyperbo` hung the factorial for 62 hours |
| **27.13** | LCBench shows NO code drift — and my drift check was misconfigured |
| **27.14** | The additive hybrid is the first thing to significantly improve EM on PD1 |
| **27.15** | The hybrid IS state-of-the-art on LCBench BO — the PD1 shortfall is off-grid-specific |
| **27.16** | The EM model has four never-exposed knobs; two look immediately promising |
| **27.17** | RETRACTION: §27.14's hybrid result does not survive correct pooling |
| **27.18** | §27.9's shrinkage conclusion is SUSPECT: the target was on the wrong scale |
| **27.19** | CORRECTION to §27.18: D2 hits LCBench, not PD1 |
| **27.20** ⚠️ | With between-prior variance included, almost nothing on PD1 is significant |
| **27.21** | CORRECTION to §27.20: the between-prior fix double-counted, inflating every SE |
| **27.22** | With correct degrees of freedom, NOTHING on PD1 arm C is significant |
| **27.23** | The end-to-end smoke found what seven review rounds and 83 tests did not |
| **27.24** | LCBench divergence: our stack never filtered, and it should not start |
| **27.25** | CORRECTION to §27.24: LCBench divergence is real — it lives in the LOSS metric |
| **27.26** | Variance allocation: σ²_within often DOMINATES σ²_between, and priors are still the better buy |
| **27.27** | FUTURE STUDY: the diverged loss metrics are auxiliary signal, not noise |
| **27.28** | FIRST v2 RESULT: stage-0 LCBench baselines, and the 9-prior design earns its keep |
| **27.29** | STAGE 0 COMPLETE: all five regimes, and regression quality dissociates from BO |
| **27.30** | The EM+HyperBO hybrids: they close the arm-C gap, but the improvement is not certified |
| **27.31** | STAGE 1 / LCBench: only the ADDITIVE hybrid helps — kernel and mean transfer do nothing |
| **27.32** ⚠️ | AUDIT: we do NOT have the missing-observation-noise bug — verified empirically |
| **27.33** | CORRECTION to §27.32: the model's own noise term belongs in the predictive |
| **27.34** | THE HYPERBO KERNEL TRANSFER WORKS — off-grid, which is exactly where EM was losing |
| **27.35** | THE EM PRE-TRAINING NOISE IS HARDCODED, UNEXPLORED, AND BADLY WRONG |
| **27.36** | DEGENERACY CHECK: §27.35's gain is REAL — EM does not collapse into the GP |
| **27.37** | STAGE 1 / LCBench regression: the noise-inclusive correction changes nothing here |
| **27.38** | BUG: the RNG re-seed guard was never ported to `bo_diagnose` |
| **27.39** | ARM B SETTLES IT: the deficit is interpolation, and richer pre-training half-repairs it |
| **27.40** ⚠️ | WHY the optimal EM noise is so large: it is a spectral truncation, not a bug |
| **27.41** | SELECTORS: my rank-matching prediction is REFUTED; the optimum is scale-matched |
| **27.42** ⚠️ | WHY σ²≈1 is principled: it is a ridge coefficient at the 50/50 point, not a noise level |
| **27.43** | CORRECTIONS to §27.42: what the E-step shrinks toward, and why the flatness claim was wrong |
| **27.44** | LEDOIT–WOLF REPLACES THE σ² HACK — and beats it, with no tuning |
| **27.45** | SYNTHETIC GROUND TRUTH: §27.42's "model discrepancy" explanation is REFUTED |
| **27.46** | IS LEDOIT–WOLF THE RIGHT ESTIMATOR? Controlled study across K, T and spectrum |
| **27.47** | PRIMARY REFERENCES for the covariance-shrinkage work |
| **28.1** 🔴 | STAGE 3 REPLICATED NOTHING — it is a byte-for-byte copy of stage 1 |
| **28.2** | The σ² optimum is OFF-GRID-SPECIFIC, and it INVERTS on-grid |
| **28.3** ⚠️ | The mean sweep is CONFOUNDED: canonical kernel ≡ mean transfer |
| **28.4** | The arm decomposition reproduces at 126 cells — and it is fixed-task only |
| **28.5** | Stage-1 screen: what is live, and factor E is the screen's worst |
| **28.6** 🔴 | TWO defects in `bo_diagnose`, and the known one is the smaller |
| **28.7** | Process findings |

⚠️ = superseded in part; read the correction before citing.
<!-- INDEX:END -->

**Status:** internal research notes; basis for refreshing the `[lin2026empirical]` paper's
baseline comparisons. Not for open source.

**Code:** `_scratch_bo/{bo_experiment,curve_experiment,bo_diagnose,analyze}.py`
(surrogates, benchmarks, diagnostics, aggregation+figures),
`botorch_fb/models/empirical_gaussian_processes/{hyperbo,pacoh,svgd}.py` (vendored baselines).

**Provenance — read this before quoting a number.** Sections are tagged:

- **[committed]** — raw per-run JSON, logs, the exact invocation, and the figures are
  checked in under `_scratch_bo/results/` and regenerate end-to-end via
  `results/run_all.sh`. Statistics come from the committed `analyze.py`, whose
  conventions are documented in `results/README.md`.
- **[legacy]** — produced by an earlier iteration whose raw outputs lived in `/tmp` and
  were lost when `/tmp` was cleaned. The numbers are reported as originally recorded;
  the *invocations* have been reconstructed into `run_all.sh` (waves `scaling`, `ninit`,
  `ood`, `noise`, `pd1`, `curve`) so any of them can be re-run, but they have not been
  re-run since and the aggregation that produced them was ad-hoc.

> **CORRECTION (§19).** Every HyperBO result in §2–§18 was produced with **2 000**
> pre-training steps, which §19 shows is far below the knee (~5 000) and costs
> HyperBO 33 % of its regret-AUC. **All HyperBO numbers below under-sell it.** At
> converged budgets EM, HyperBO and ABLR are statistically tied on LCBench.
> §19 also shows that per-run error bars on neural meta-learners are not valid
> evidence about the method, because one prior is shared by every run.
> **§20 supersedes §5B–§5L and §14 with a corrected 42-job re-run.** The corrected
> LCBench ranking is **ABLR ≥ HyperBO ≥ EM**, not EM first: properly trained
> baselines gain 24–27 % while EM is unchanged. EM's LCBench claim is now cost,
> determinism and calibration — not raw BO quality.

The headline (§5J), batch BO (§5L), diagnostics (§14), the shrinkage study (§16), the
faithful PD1 reproduction (§17) and the HyperBO debugging study (§19) were **recomputed from scratch** with the committed
pipeline; everything else is legacy pending a re-run. **§17 supersedes the PD1
conclusions of §9–§12**, which were drawn on the reduced PD1Lite subset.

> **Note on a bug this uncovered.** Every script here was broken at import: commit
> `0464a979cecb` narrowed the internal LCBench loader to a Parquet read seam
> (`read_lcbench_parquet_from_internal_store`) but the scripts still imported a
> `load_lcbench_data` that no longer existed. `lcbench_io.py` now exposes the
> open-source `botorch.utils.lcbench` loader and patches only the download seam (the
> same pattern `test_tutorials.py` uses). `LCBENCH_DATASET_NAMES` is identical in
> content *and order* to the `DATASET_NAMES` previously imported from `ax.benchmark`,
> so seeded dataset splits are unchanged and legacy numbers stay comparable.

---

## 1. Setup

**Data.** LCBench (35 datasets). Each dataset gives the *same* pool of hyperparameter
configs (7D) evaluated to a validation-accuracy learning curve. Two tasks:

- **7D final-performance BO** (`bo_experiment.py`): maximize final val-accuracy over a
  finite pool of configs; the surrogate only *ranks* pool candidates (objective is a
  lookup). Meta-train the prior on a set of pretrain datasets, evaluate BO on held-out
  datasets. Acquisition: analytic **LogEI** over the finite pool.
- **1D learning-curve extrapolation** (`curve_experiment.py`): meta-train on complete
  historical curves; for each held-out curve, condition on the first 30% of epochs and
  extrapolate the tail. Metrics: tail **RMSE / NLL / 95% coverage**.

**Fairness controls (identical across all methods).**
- Same pretrain/eval split, same standardized targets (global mean/std from the pretrain
  pool), same initial design per (dataset, seed), same LogEI acquisition, same finite pool.
- **Consistent outcome handling — no per-model output warp** (so we compare the *priors*,
  not preprocessing; every model stays Gaussian in the same target space).
- Meta-learners (HyperBO, PACOH) pretrained on the *same* `ExperimentDataset`s as EM.

**Methods.**
| method | description |
|---|---|
| `em_frozen` | EM empirical-GP prior (meta-learned per-config empirical mean + covariance), frozen. |
| `em_finetuned` | EM prior + a fresh additive base kernel fit on the incoming task. |
| `hyperbo_frozen` | HyperBO deep-kernel GP (MLP features + Matérn), prior frozen (paper-faithful). |
| `hyperbo_adapt` | HyperBO, per-task Gaussian-noise re-fit only (features/kernel frozen). |
| `pacoh_frozen` | PACOH-GP: PAC-Bayes hyper-posterior over the HyperBO prior via K SVGD particles; mixture-of-GPs. |
| `pacoh_adapt` | PACOH, per-task noise re-fit. |
| `vanilla_gp` | `SingleTaskGP` fit from scratch on the incoming task (no transfer). |
| `random` | random pool pick. |

Best-faithful meta-learner config (from the sweep, §5): (32,32)-tanh MLP deep kernel, NLL
pretraining, point-subsample 64, task-minibatch 5; HyperBO ≥2000 iters; PACOH K=10,
hyper-prior σ=1.0, median-heuristic SVGD length-scale.

---

## 2. 7D final-performance BO (headline) **[legacy]**

8 held-out datasets × 8 seeds = 64 runs, 300-config pool, 40 BO iters, 3 random init.
**Simple regret** (val-accuracy points below the pool optimum; lower = better) ± clustered SEM,
plus one-time pretraining cost and amortized online cost.

| method | @5 | @10 | @20 | @40 | pretrain | online ms/run |
|---|---|---|---|---|---|---|
| **em_frozen** | **0.62±0.15** | **0.38±0.13** | 0.25±0.09 | **0.086±0.031** | 42 s | 429 |
| em_finetuned | 0.59±0.15 | 0.38±0.12 | 0.33±0.12 | 0.110±0.035 | 42 s | 14 813 |
| hyperbo_frozen | 1.21±0.21 | 0.45±0.15 | **0.17±0.04** | 0.098±0.027 | 165 s | 439 |
| hyperbo_adapt | 1.16±0.21 | 0.36±0.10 | 0.20±0.05 | 0.124±0.036 | 165 s | 2 342 |
| vanilla_gp | 2.60±0.29 | 1.70±0.22 | 0.72±0.15 | 0.173±0.037 | 0 | 21 359 |
| pacoh_frozen | 2.81±0.25 | 2.09±0.21 | 1.78±0.19 | 0.621±0.173 | 161 s | 731 |
| pacoh_adapt | 2.66±0.24 | 2.06±0.21 | 1.59±0.19 | 0.588±0.174 | 161 s | 10 551 |
| random | 3.27±0.35 | 2.83±0.32 | 1.98±0.28 | 1.156±0.180 | 0 | 0.3 |

**Takeaways.**
- **EM wins early and is the most compute-efficient** (cheapest pretraining, ~0.4 ms-scale
  online, best @5/@10). Its per-config empirical mean directly exploits the shared pool.
- **HyperBO ties EM at the frontier** (@40: 0.098 vs 0.086, overlapping SEM; @20 it is
  nominally *best* at 0.17). It lags early because the deep kernel must *learn* what EM's
  empirical mean encodes directly. Noise adaptation helps at low pretraining budgets (§4).
- **`em_finetuned` and `vanilla_gp` pay 15–21 s/run** for per-step MLL fits with **no
  accuracy gain** over frozen priors — the per-step refit is wasted compute here.
- **PACOH-GP is genuinely weak**: clearly above random but far behind EM/HyperBO/vanilla,
  and `pacoh_adapt` additionally costs 10.5 s/run. This is *not* an implementation artifact
  (§5).

> **High-power confirmation (§5J).** At **128 runs** (8 datasets × 16 seeds) the frontier is
> a *four-way tie* — `hyperbo_adapt`, `em_finetuned`, `em_frozen`, `hyperbo_frozen` all reach
> @40 regret 0.09–0.13 with overlapping SEM (~18–19% win each); EM still owns the early
> regime (@5 ≈ 0.56 vs HyperBO 1.16). The em_frozen ≈ hyperbo_frozen tie is robust.

### 2.1 Zero-shot start (`n_init=0`)

With **no** initial design, the first candidate is chosen by the surrogate's **prior
mean** (greedy) — the model's single best guess with zero task data. This isolates the
value of the meta-learned *mean*. 8×8 runs, simple regret ± SEM.

| method | @0 (no-data pick) | @1 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| **em_frozen** | **1.71±0.23** | 1.62±0.22 | 0.91±0.22 | 0.22±0.04 | 0.17±0.04 |
| pacoh_frozen | 3.43±0.22 | 3.43±0.22 | 2.51±0.24 | 1.54±0.22 | 0.77±0.10 |
| hyperbo_frozen | 4.70±0.33 | 2.53±0.27 | 1.07±0.21 | 0.85±0.22 | 0.17±0.04 |
| pretrained_gp_frozen | 21.53±3.01 | 6.68±0.88 | 3.16±0.29 | 2.09±0.24 | 1.09±0.19 |
| vanilla_gp | 21.53±3.01 | 6.72±0.89 | 2.94±0.28 | 1.80±0.22 | 0.81±0.14 |
| random | 21.53±3.01 | 11.61±2.11 | 4.35±0.64 | 3.11±0.31 | 1.84±0.24 |

**Takeaways.**
- **The meta-learned prior gives an excellent no-data guess.** EM's *single* zero-shot
  pick (1.71) already beats what random/vanilla reach after **10–20** evaluations.
- **Flat-mean models degrade to random at @0** (`vanilla_gp`, `pretrained_gp_frozen` =
  `random` = 21.53), confirming `n_init=0` isolates the value of the meta-learned mean.
- **Mean vs. uncertainty disaggregation:** PACOH's zero-shot pick (3.43) *beats*
  HyperBO's (4.70) — yet PACOH is the worst GP method in the full loop (§2). So **PACOH's
  prior mean is fine; its failure is the SVGD-mixture *uncertainty/exploration*, not the
  mean.** HyperBO has the weakest zero-shot mean but recovers fastest with data (ties EM
  by @20). This motivates the acquisition sweep (Recommendations) to formalize the
  mean-vs-uncertainty attribution.

*(`bo_experiment.py --n-init 0`; the first pick uses `model.forward`'s prior mean —
EM's shift-interpolated empirical mean, HyperBO's linear-mean-on-features, PACOH's
particle-averaged prior mean.)*

---

## 3. 1D learning-curve extrapolation **[legacy]**

5 held-out datasets × 30 curves, 30% context (predict the final 35/50 epochs).

**Default (NLL) objective:**
| method | RMSE | NLL | 95% cov | pretrain |
|---|---|---|---|---|
| **em_1d** | **0.106±0.018** | **−0.54±0.15** | 0.93±0.03 | ~0 |
| hyperbo_adapt | 0.115±0.011 | −0.59±0.12 | 0.88±0.02 | 55 s |
| hyperbo_frozen | 0.177±0.034 | −0.12±0.10 | 0.98±0.01 | 55 s |
| vanilla_gp | 0.431±0.081 | 0.36±0.39 | 0.92±0.01 | 0 |
| pacoh_frozen | 0.608±0.172 | 1.11±0.13 | 1.00±0.00 | 96 s |
| pacoh_adapt | 0.590±0.165 | 1.09±0.13 | 1.00±0.00 | 96 s |

**HyperBO EKL ablation** (empirical-KL objective; valid because curves share the epoch grid):
| method | RMSE | NLL | 95% cov | pretrain |
|---|---|---|---|---|
| em_1d | 0.106±0.018 | −0.54±0.15 | 0.93±0.03 | ~0 |
| **hyperbo_adapt (EKL)** | **0.093±0.011** | 0.86±0.64 | 0.71±0.04 | 16 s |
| hyperbo_frozen (EKL) | 0.142±0.029 | −0.15±0.26 | 0.91±0.04 | 16 s |

**Takeaways.**
- **EM is strong and well-calibrated** (RMSE 0.106, coverage 0.93) — its empirical curve
  basis is an excellent inductive bias.
- **HyperBO is competitive**; **EKL trains ~3× faster (16 s vs 55 s) and gives the best point
  accuracy** (RMSE 0.093, beating EM) — but EKL+adapt is **overconfident** (coverage 0.71,
  high NLL). EKL+frozen is a good accuracy/calibration compromise (0.142, cov 0.91).
- **PACOH is weak on 1D too — worse than a from-scratch GP** (0.59–0.61 vs 0.43), with
  degenerate uncertainty (coverage 1.00 / high NLL: intervals wide *and* mean poor).

---

## 4. Performance-vs-cost Pareto **[legacy]**

HyperBO trades optimization performance against pretraining budget; PACOH does not.

**HyperBO (3×3, 7D):**
| meta_iters | pretrain | frozen @10 | frozen @25 | adapt @10 | adapt @25 |
|---|---|---|---|---|---|
| 250 | 9 s | 3.19 | 1.69 | 2.62 | 0.33 |
| 500 | 18 s | 2.53 | 1.15 | 1.80 | 0.39 |
| 1000 | 34 s | 1.77 | 0.63 | 0.66 | 0.35 |
| **2000** | **67 s** | 0.56 | **0.00** | **0.04** | **0.00** |
| 5000 | 165 s | 0.08 | 0.00 | 0.08 | 0.00 |

- **Knee at ~2000 iters (67 s):** HyperBO reaches EM-level final regret there; **5000 iters
  is wasted compute.** Recommend 2000 as the default operating point.
- **Noise adaptation is a large low-budget efficiency win:** at 250 iters, `adapt` @25 = 0.33
  vs `frozen` 1.69. It buys the equivalent of far more pretraining for ~2 s/run online.

**PACOH (3×3, 7D):** flat and weak — 1000 iters (67 s) → @25 = 1.00 ≈ random (0.97); 2500
iters (162 s) → no improvement. No budget makes PACOH competitive.

---

## 5. PACOH faithful single-knob sweep (why it's weak) **[legacy]**

3×3, 7D, one knob changed from the baseline at a time (final @25 regret; em anchor = 0.028):
| config | pacoh_frozen | pacoh_adapt |
|---|---|---|
| **base** (K=10, σ=1.0, median-ls, 2500) | **1.00** | 1.09 |
| σ_hyperprior = 0.3 | 3.60 | 1.14 |
| K = 30 particles | 2.18 | 2.18 |
| meta_iters = 5000 | 1.17 | 1.09 |
| SVGD length-scale = 0.3 (fixed) | 1.16 | 1.16 |

**No faithful knob improves on the baseline (~1.0 ≈ random).** More particles hurt, a
stronger hyper-prior hurts, a fixed SVGD length-scale hurts, 2× training is neutral.
Interpretation: in the ~1,150-dim deep-kernel parameter space, K=10–30 SVGD particles
behave like independent MAP estimates (kernel collapses toward identity), so the mixture
gives no meaningful Bayesian benefit, and the deep kernel underfits the structured pool.
**PACOH's weakness reproduces the paper's finding and is a real property of the method on
this problem, not an implementation bug.**

> Not yet swept (candidate to give PACOH its strongest honest shot): a **smaller MLP**
> (e.g. `(8,)` or `(16,)`), which lowers the SVGD particle dimension and may reduce the
> collapse. See *Recommendations*.

---

## 5A. PACOH capacity & MAP ablation **[legacy]**

*Setup: 3×3 (3 eval datasets × 3 seeds, 150 configs, 25 iters, 20 pretrain, meta_iters=2500),
plus a 6×6 confirmation (6×6, 200 configs, 30 iters, meta_iters=2000). Own seeds — do not
compare absolute values across scales.*

Goal: give PACOH its strongest honest configuration by sweeping MLP width and collapsing
the SVGD hyper-posterior to a single MAP particle (K=1). @-final simple regret.

| config (3×3) | pacoh_frozen | pacoh_adapt | pretrain |
|---|---|---|---|
| h=(8,) | 1.17 | 0.78 | 132 s |
| h=(16,) | 1.57 | 0.78 | 138 s |
| h=(16,16) | 0.03\* | 1.25 | 156 s |
| h=(32,32) (base) | 1.00 | 1.09 | 164 s |
| **MAP K=1, h=(16,)** | 1.26 | **0.20** | 62 s |

\*3×3 lucky-optimum outlier (matches EM's 0.028 exactly — not reproducible).

The 3×3 MAP number looked dramatic, so we **confirmed at 6×6** (paired, same seeds):

| config (6×6) | pacoh_frozen | pacoh_adapt | pretrain | (anchors) |
|---|---|---|---|---|
| MAP K=1, h=(16,) | 0.41 | 0.25 | 60 s | em=0.03 |
| SVGD K=10, h=(32,32) | 0.36 | 0.28 | 130 s | hyperbo_adapt=0.07 |

**Conclusion.** No faithful PACOH configuration closes the gap to EM/HyperBO. At 6×6,
**MAP ≈ SVGD** (0.25 vs 0.28; the 3×3 MAP=0.20 was small-sample noise) — MAP is ~2× cheaper
to pre-train but not more accurate. Combined with the earlier hyper-prior/SVGD-ls/iters
sweep (§5), PACOH's weakness is now verified across *every* faithful axis
(width, particles, prior scale, SVGD bandwidth, iterations): it is a genuine property, not
a mis-configuration.

---

## 5B. Acquisition-function ablation **[legacy]**

*Setup: 4×4 (4 eval datasets × 4 seeds, 200 configs, 30 iters, meta_iters=2000). Same
seeds across the three acquisitions.* Simple regret @5 / @30.

| method | greedy @5/@30 | LogEI @5/@30 | UCB @5/@30 |
|---|---|---|---|
| em_frozen | 1.38 / 0.35 | 1.47 / 0.07 | 0.99 / 0.07 |
| hyperbo_frozen | 1.71 / **1.17** | 1.39 / **0.00** | 0.39 / **0.00** |
| pacoh_frozen | 3.21 / 0.87 | 3.21 / 0.55 | 4.26 / 0.30 |
| vanilla_gp | 3.67 / 0.68 | 3.07 / 0.24 | 3.04 / 0.42 |
| random | 3.68 / 1.09 | 3.68 / 1.09 | 3.68 / 1.09 |

**Conclusion.** The method ranking (EM/HyperBO ≫ vanilla > PACOH > random at the frontier)
is **robust across acquisitions** — not a LogEI artifact. The disaggregation is clean:
- **HyperBO's value is uncertainty-driven.** It is the *worst* GP method under greedy
  (mean-only, @30=1.17) but ties for best under LogEI/UCB (0.00) — its weak prior mean is
  rescued by exploration. Under UCB it is already best at @5 (0.39).
- **EM's value is its mean** (best/robust under greedy, @30=0.35) and it stays top under
  LogEI/UCB.
- **PACOH** has a mediocre mean and benefits from exploration (UCB @30=0.30) but still lags.

This confirms the zero-shot §2.1 story with an in-loop acquisition axis.

---

## 5C. HyperBO EKL vs NLL objective (7D) **[legacy]**

*Setup: 4×4, 200 configs, 30 iters, meta_iters=2000, same seeds. The config pool is shared
across datasets, so EKL's matched-input requirement holds in 7D.* Simple regret @5 / @30.

| objective | hyperbo_frozen | hyperbo_adapt | pretrain |
|---|---|---|---|
| **NLL** | 1.39 / **0.00** | 0.87 / **0.18** | 68 s |
| EKL | 1.63 / 1.12 | 1.63 / 0.37 | 96 s |

**Conclusion.** In 7D, **NLL clearly beats EKL** (frozen 0.00 vs 1.12) and pre-trains faster.
The EKL point-accuracy advantage seen on 1D curves (§3) does **not** transfer to 7D BO —
use NLL as the 7D HyperBO objective.

---

## 5D. Novel-config split (transfer to unseen configs) **[legacy]**

*Setup: 6×6, 200 configs, 30 iters, meta_iters=2000. Novel split pre-trains the prior on
one config subset and runs BO on a **disjoint** held-out subset; the standard control uses
the same scale/seeds on the shared pool. Absolute regrets are not comparable across the two
pools (different difficulty); compare method *ordering* and *degradation*.* Regret @5 / @30.

| method | STANDARD (shared) @5/@30 | NOVEL (held-out) @5/@30 |
|---|---|---|
| em_frozen | 0.98 / 0.03 | 0.55 / 0.00 |
| hyperbo_frozen | 1.67 / 0.00 | 0.72 / 0.03 |
| pacoh_frozen | 2.57 / 0.36 | 1.56 / 0.00 |
| vanilla_gp | 3.04 / 0.26 | 2.46 / 0.05 |
| random | 4.17 / 1.55 | 3.76 / 1.90 |

**Conclusion.** The meta-learned prior **transfers cleanly to configs never seen in
pre-training** — EM/HyperBO/PACOH do *not* degrade on the held-out pool (they reach ~0
regret), and the ranking is preserved. So EM's advantage is **not** an artifact of the
shared-pool setup: its shift-interpolated empirical mean generalizes to unseen configs from
the same distribution. (Random is nominally *worse* on the novel pool — 1.90 vs 1.55 — so
the novel pool is not simply easier.)

---

## 5E. 1D context-fraction sweep **[legacy]**

*Setup: 5 eval datasets × 30 curves, meta_iters=2000, NLL. Condition on the first
`context` fraction of each curve, extrapolate the rest.* Held-out-tail **RMSE** (± SEM).

| context | em_1d | hyperbo_frozen | hyperbo_adapt | pacoh_frozen | vanilla_gp |
|---|---|---|---|---|---|
| 0.1 | 0.298 | 0.366 | 0.311 | 0.881 | 0.510 |
| 0.2 | 0.164 | 0.242 | 0.209 | 0.803 | 0.444 |
| 0.3 | 0.106 | 0.177 | 0.115 | 0.608 | 0.431 |
| 0.4 | 0.074 | 0.121 | **0.071** | 0.403 | 0.352 |
| 0.5 | 0.048 | 0.085 | **0.045** | 0.297 | 0.272 |

**Conclusion.** All methods improve monotonically with context. **EM and HyperBO-adapt are
tied for best** (HyperBO-adapt edges EM at ≥40% context). The meta-learners' advantage over
`vanilla_gp` **persists and widens** with context (vanilla plateaus ~0.27–0.51 — a
stationary GP cannot extrapolate the curve trend, while the empirical/deep-kernel priors
lock onto it). PACOH is worst at every context length, consistent with 7D.

---

## 5F. Meta-data scaling (regret vs #pre-training datasets) **[legacy]**

*Setup: 4 eval × 4 seeds = 16 runs, 200 configs, 30 iters, meta_iters=2000. Vary the
number of pre-training datasets `n_pretrain ∈ {2,5,10,15,25}` on a fixed eval set
(albert, bank-marketing, jannis, sylvine).* Simple regret @5 / @30 (± SEM in text).

| n_pretrain | em_frozen @5/@30 | hyperbo_frozen @5/@30 | pacoh_frozen @5/@30 | random @5/@30 |
|---|---|---|---|---|
| 2  | 1.62 / 1.32 | 3.18 / 1.32 | 3.84 / 0.69 | 3.68 / 1.09 |
| 5  | 2.34 / 0.52 | 2.45 / 0.14 | 3.20 / 1.40 | 3.68 / 1.09 |
| 10 | 2.00 / **0.05** | 3.14 / 1.39 | 3.86 / 0.79 | 3.89 / 1.43 |
| 15 | **0.28** / 0.06 | 2.96 / 0.02 | 4.07 / 1.68 | 3.89 / 1.43 |
| 25 | 1.47 / 0.07 | 1.39 / **0.00** | 3.21 / 0.55 | 3.89 / 1.43 |

Dataset-clustered average rank at @30 (lower = better; win-rate %) — `per_run` was
recorded only from `n_pretrain≥10`:

| n_pretrain | em_frozen | hyperbo_frozen | pacoh_frozen | random |
|---|---|---|---|---|
| 10 | **1.72** (41%) | 2.53 (30%) | 2.62 (23%) | 3.12 (6%) |
| 15 | 1.69 (41%) | **1.56** (53%) | 3.69 (0%) | 3.06 (6%) |
| 25 | 2.09 (23%) | **1.84** (48%) | 2.53 (23%) | 3.53 (5%) |

**Takeaways.**
- **EM stabilizes with the fewest tasks.** Its converged regret drops to ~0.05 by
  `n_pretrain=10` and its early-BO regret (@5) is best at moderate budgets; it is rank-1
  at `n_pretrain=10` and within 1σ of the best at 15/25.
- **HyperBO is data-hungry.** Its deep kernel is erratic at low budgets (the @30 spike to
  1.39 at `n_pretrain=10` is a bad-fit run) and only overtakes EM once ~15–25 tasks are
  available — exactly the regime where it wins the headline (§2).
- **PACOH never catches up:** best rank is 2.53 and it is last at `n_pretrain=15`,
  consistent with its full-loop weakness (§5/§5A).
- *Caveat:* only 16 runs, so SEMs are wide and single bad fits move the means; the
  `random` reference for `n_pretrain≤5` used an earlier pick-RNG (harmless — the
  meta-learner curves are byte-identical across that code change). The high-seed run
  (§5J) tightens these.

---

## 5G. Initial-design sweep (`n_init` 0→10): zero-shot → warm-start **[legacy]**

*Setup: same 4×4 scale, `n_pretrain=25`, sweep `n_init ∈ {0,1,3,5,10}`.* Because the
trajectory x-axis shifts with `n_init`, we compare on the directly-comparable **final
regret** and **dataset-clustered average rank** (win-rate %), plus the first-pick regret
`@1` that isolates the no-data prior mean.

| n_init | em_frozen @1 · final(rank) | hyperbo_frozen @1 · final(rank) | pacoh_frozen final(rank) | vanilla_gp final(rank) | random final(rank) |
|---|---|---|---|---|---|
| 0  | **1.72** · 0.06 (2.41) | 3.47 · **0.00** (**2.09**) | 0.55 (3.03) | 0.19 (3.34) | 1.63 (4.12) |
| 1  | **2.10** · 0.06 (2.38) | 4.33 · **0.00** (**2.06**) | 0.55 (3.03) | 0.17 (3.09) | 1.46 (4.44) |
| 3  | **1.75** · 0.07 (2.59) | 3.93 · **0.00** (**2.19**) | 0.55 (3.03) | 0.24 (2.91) | 1.43 (4.28) |
| 5  | **2.03** · 0.08 (2.47) | 3.88 · 0.02 (**2.19**) | 0.69 (2.97) | 0.14 (2.94) | 2.06 (4.44) |
| 10 | **1.63** · 0.06 (2.53) | 3.24 · 0.02 (**2.41**) | 0.43 (3.09) | 0.21 (3.28) | 0.68 (3.69) |

**Takeaways.**
- **EM owns the no-data pick at every `n_init`** (@1 ≈ 1.6–2.1 vs HyperBO 3.2–4.3,
  vanilla 3.4–6.5) — the meta-learned *mean* is the differentiator (confirms §2.1).
- **HyperBO wins on converged rank** (best avg-rank at every `n_init`) because its deep
  kernel closes the gap once it has a few observations; EM is a close 2nd everywhere.
- **The meta-prior advantage is concentrated at small `n_init`.** As the random-init
  budget grows to 10, the flat-mean baselines (vanilla, random) catch up markedly
  (random final regret 1.63→0.68), shrinking the gap — i.e. the prior matters most when
  evaluations are scarce, which is the entire point of BO. This cleanly connects the
  zero-shot start (§2.1) to a warm-started regime.

---

## 5H. OOD dataset split (cross-distribution transfer) **[legacy]**

*Setup: 4×4, 200 configs, 30 iters, `n_pretrain=25`, `--ood-split` — datasets are
grouped by the SHAPE of their config→accuracy response and the prior is pre-trained on
one cluster then evaluated on the **most dissimilar** cluster (connect-4,
Amazon_employee_access, KDDCup09_appetency, car).* Simple regret @5 / @30 and avg-rank.

| method | @5 | @10 | @30 | avg_rank | win% |
|---|---|---|---|---|---|
| em_finetuned | 0.33±0.29 | 0.04±0.02 | **0.00±0.00** | **3.12** | 25 |
| pacoh_frozen | 6.10±1.48 | 0.50±0.33 | **0.00±0.00** | 3.34 | 19 |
| vanilla_gp | 2.44±1.21 | 0.88±0.44 | 0.12±0.11 | 3.53 | 18 |
| hyperbo_frozen | 3.45±1.05 | 1.42±0.50 | 0.65±0.44 | 3.97 | 13 |
| hyperbo_adapt | 3.51±1.19 | 1.74±0.54 | 1.28±0.57 | 4.44 | 11 |
| em_frozen | 2.86±1.83 | 0.90±0.44 | 0.58±0.34 | 4.53 | 9 |
| random | 4.18±1.23 | 3.22±1.18 | 1.42±0.64 | 5.06 | 6 |

**Takeaways (honest, this is the hard test).**
- **Under a deliberate distribution shift the frozen empirical prior loses its edge.**
  `em_frozen` — the headline winner in-distribution (§2) — drops to the *worst* GP method
  here (rank 4.53, @30 0.58): a frozen empirical mean fit to a dissimilar cluster is a
  poor mean for the eval cluster.
- **`em_finetuned` is the OOD winner** (rank 3.12, reaches 0 regret): adding a
  fully-trainable additive base kernel lets the model *correct* the mis-specified prior
  from observed data — exactly what you want when transfer is unreliable.
- **The ranking compresses** (all GP methods reach ≤0.65 by @30, SEMs overlap heavily) —
  with a poor prior the problem reduces toward vanilla BO. Net message for the paper:
  **the empirical prior's advantage is real but distribution-dependent; report OOD to be
  honest, and prefer the finetuned variant when transfer cannot be assumed.**

---

## 5I. EM-variant comparison (additive base kernels) **[legacy]**

*Setup: 4×4, 200 configs, 30 iters, `n_pretrain=25` (in-distribution eval).* Compares the
plain frozen empirical prior against the additive-base-kernel variants. Regret @10 / @30
and avg-rank.

| method | @10 | @30 | avg_rank | win% |
|---|---|---|---|---|
| hyperbo_frozen | 0.40±0.33 | **0.00±0.00** | **2.62** | 36 |
| em_frozen | **0.12±0.04** | 0.07±0.03 | 3.03 | 14 |
| em_finetuned | 0.13±0.04 | 0.08±0.03 | 3.16 | 14 |
| em_additive_freshbase | 0.37±0.28 | 0.07±0.04 | 3.19 | 18 |
| em_additive_3way | 1.22±0.53 | 0.10±0.04 | 3.59 | 14 |
| random | 3.43±0.47 | 1.43±0.49 | 5.41 | 3 |

**Takeaways.**
- **The additive base-kernel variants give no clear gain over plain `em_frozen`** at this
  scale — `em_frozen`, `em_finetuned`, and `em_additive_freshbase` cluster within SEM
  (@30 ≈ 0.07–0.08, ranks 3.0–3.2), and **`em_additive_3way` is nominally the *weakest*
  EM variant early** (@10 1.22, driven by one bad-fit run). This does **not** support
  promoting `em_additive_3way` to the headline; the plain frozen empirical prior remains
  the best cost/quality EM choice in-distribution. (The additive variants *do* earn their
  keep OOD — §5H — where `em_finetuned` leads.)
- HyperBO tops the rank here only because on these 4 datasets at `n_pretrain=25` it
  reaches exactly 0 by @20 (consistent with §5F at high budget).

---

## 5J. Dataset-clustered rank statistics + the definitive high-power table **[committed]**

Every study reports **dataset-clustered average ranks** (per-run rank by the metric,
ties averaged) and **win-rate**. Conventions are pinned in `results/README.md`;
the code is `analyze.py` (previously ad-hoc chat-side analysis, now in-tree).

**Definitive high-power table — recomputed from scratch.** 12 eval datasets × 16 seeds =
**192 runs**, 300 configs, 40 iters, `meta_iters=2000`, 9 methods. *Pre-training uses
**23** datasets, not the 25 requested: LCBench has 35, so `12 eval + 25 pretrain` overruns
and the slice truncates. §5F shows HyperBO is data-hungry in this range, so the exact
count matters.*
Eval: albert, bank-marketing, jannis, sylvine, kr-vs-kp, segment, christine,
mfeat-factors, blood-transfusion-service-center, KDDCup09_appetency, cnae-9,
Amazon_employee_access. Simple regret ± **clustered** SEM; sorted by average rank on
final regret. (The design is balanced, so clustered and per-run average ranks coincide.)

| method | @1 | @5 | @10 | @20 | @40 | AUC | rank | win% |
|---|---|---|---|---|---|---|---|---|
| em_noisefit | 2.490±0.852 | 1.343±0.818 | 0.808±0.524 | 0.276±0.193 | **0.093±0.057** | **0.669** | **4.22** | 13 |
| em_frozen | 2.829±0.827 | 1.335±0.724 | **0.621±0.318** | **0.181±0.084** | 0.109±0.058 | 0.639 | 4.32 | 15 |
| em_finetuned | 2.613±0.858 | 1.239±0.647 | 0.906±0.531 | 0.351±0.246 | 0.144±0.078 | 0.744 | 4.43 | 13 |
| hyperbo_adapt | **2.490±0.762** | 1.237±0.586 | 0.757±0.512 | 0.539±0.410 | 0.312±0.233 | 0.820 | 4.54 | 13 |
| hyperbo_frozen | 2.618±0.829 | 1.372±0.715 | 0.714±0.420 | 0.410±0.228 | 0.208±0.124 | 0.810 | 4.78 | 11 |
| vanilla_gp | 4.257±0.922 | 2.370±0.648 | 1.482±0.409 | 0.844±0.323 | 0.278±0.122 | 1.276 | 4.96 | 11 |
| ablr | 3.236±0.434 | **0.828±0.329** | 0.431±0.190 | 0.279±0.135 | 0.217±0.129 | **0.619** | 5.04 | 9 |
| pacoh_frozen | 4.587±1.068 | 2.620±0.800 | 1.746±0.769 | 0.937±0.295 | 0.368±0.169 | 1.397 | 5.64 | 8 |
| random | 5.623±0.893 | 3.134±0.602 | 1.965±0.423 | 1.425±0.330 | 0.971±0.270 | 1.926 | 7.07 | 7 |

**Figure:** `results/figures/lcbench_highpower_trajectories.png` — the main figure, full
mean-regret trajectories with shaded clustered SEM.

**Statistical significance (Friedman + Nemenyi, N=192, k=9, CD = 0.87).**

- **Final regret** — χ²(8) = 163.2, p ≈ 3e-31. But the top clique is
  {em_noisefit, em_frozen, em_finetuned, hyperbo_adapt, hyperbo_frozen, **vanilla_gp**,
  **ablr**}: by @40 the finite 300-config pool is largely exhausted and even a
  from-scratch GP catches up. **Final-regret rank is a weak discriminator** — this
  reproduces the earlier finding at 1.5× the runs.
- **Regret-AUC (the discriminating metric)** — χ²(8) = 360.5, p ≈ 5e-73. Ranks:
  em_noisefit **3.49**, em_frozen 4.00, hyperbo_adapt 4.15, em_finetuned 4.19,
  ablr 4.58, hyperbo_frozen 4.59 ≫ vanilla_gp 6.08, pacoh_frozen 6.60, random 7.32.
  Tied-best clique = **{em_noisefit, em_frozen, hyperbo_adapt, em_finetuned}**, and
  **all six meta-priors are significantly ahead of vanilla_gp, PACOH and random**
  (gaps of 1.5–3.8 rank units vs CD 0.87). `em_noisefit` is significantly ahead of
  `ablr` and `hyperbo_frozen` (1.09 > CD).
- **@5 / @10 / @20** show the same structure: a six-method meta-prior clique, then a
  large gap to {vanilla_gp, pacoh_frozen, random}.

**Takeaways.**
- **The meta-prior advantage is real, large and statistically significant — measured
  early or by AUC.** It vanishes at final regret only because the pool saturates.
- **`em_noisefit` is the best single method** (AUC rank 1, best @40 regret). Fitting one
  scalar — the per-task likelihood noise — on top of the frozen empirical prior beats
  both plain `em_frozen` and the far more expensive `em_finetuned`. §14 shows why: it is
  a *calibration* fix, and `em_frozen`'s miscalibration grows with data.
- **EM owns the early regime**, HyperBO closes in by @10, and `hyperbo_adapt`'s early
  strength (best @1) does not persist (@40 0.312, the worst of the four EM/HyperBO arms).
- **ABLR is a genuinely strong baseline with a distinctive profile**: worst-but-one @1
  (3.24 — its BLR head has a zero prior mean) yet **best @5 in absolute terms** (0.828)
  and the **lowest mean AUC of any method** (0.619). Its *rank* is only 4.58 because it
  is reliably good rather than often best — a mean-vs-rank divergence worth reporting.
- **PACOH is weak, confirmed at 192 runs** (AUC rank 6.60, barely ahead of random).

*Differences vs the earlier 8×16 = 128-run table (legacy, raw data lost): the eval set is
50% wider and `em_noisefit`/`ablr` are now in the main comparison. The qualitative story
is unchanged — meta-priors separate early and tie at the frontier — but the nominal
rank-1 moved from `hyperbo_adapt` to `em_noisefit`, and vanilla_gp now joins the
final-regret top clique more clearly.*

---

## 5K. Observation-noise robustness **[legacy]**

*Setup: 4 eval × 6 seeds = 24 runs, 200 configs, 30 iters, `n_pretrain=25`. Inject i.i.d.
Gaussian noise (std σ in standardized units, `--obs-noise`) into the observations each
surrogate **conditions on** (and into the EI incumbent); **regret is always measured on
the true objective.** LCBench is near-noiseless, so this isolates uncertainty handling.*

Final simple regret @30 ± SEM:

| σ_obs | em_frozen | hyperbo_frozen | pacoh_frozen | vanilla_gp | random\* |
|---|---|---|---|---|---|
| 0.0  | 0.08±0.03 | **0.00±0.00** | 0.64±0.25 | 0.22±0.10 | 1.76±0.45 |
| 0.1  | 0.09±0.03 | **0.02±0.02** | 0.55±0.20 | 0.69±0.27 | 1.24±0.36 |
| 0.25 | 0.15±0.05 | **0.03±0.02** | 0.55±0.20 | 0.58±0.19 | 1.24±0.36 |
| 0.5  | 0.40±0.18 | **0.02±0.02** | 0.46±0.18 | 1.68±0.48 | 1.24±0.36 |

Dataset-clustered average rank @final (lower = better):

| σ_obs | em_frozen | hyperbo_frozen | pacoh_frozen | vanilla_gp | random |
|---|---|---|---|---|---|
| 0.0  | 2.60 | **2.12** | 2.98 | 2.96 | 4.33 |
| 0.1  | 2.56 | **2.15** | 2.88 | 3.23 | 4.19 |
| 0.25 | 2.58 | **1.96** | 2.71 | 3.62 | 4.12 |
| 0.5  | 2.94 | **1.90** | 2.62 | 3.62 | 3.92 |

**Takeaways.**
- **HyperBO is the most noise-robust** (final regret flat at ~0.00–0.02 across all σ; its
  avg-rank actually *improves* 2.12→1.90 as noise grows). Its jointly-learned GP noise
  variance absorbs observation noise — **HyperBO's edge over EM widens under noise.**
- **`em_frozen` degrades gracefully but noticeably** (0.08→0.40): it conditions with a
  *frozen* likelihood noise (`COND_NOISE`) that is not adapted, so it cannot down-weight
  heavy noise; still 2nd-tier through σ=0.25.
- **`vanilla_gp` is the most noise-fragile** (0.22→1.68 — the *worst* GP method by σ=0.5):
  a from-scratch GP over-fits the noisy observations.
- **PACOH is ~flat** (0.64→0.46) — already over-dispersed/weak, so extra noise barely
  moves it.
- \*The `random` reference shifts slightly across the sweep because drawing the noise
  tensor consumes the shared RNG before the pick loop; this does **not** affect the
  meta-method trajectories (their picks are acquisition-argmax, RNG-independent post-init).

**For the paper:** obs-noise is the regime where HyperBO's *adaptive* noise model earns
its keep — complementary to EM's noiseless-regime win (§2). The natural follow-up is a
noise-adapted EM (`em_finetuned`, which fits an additive base kernel + noise) under the
same sweep.

**Noise-adapted EM (`em_noisefit`) — does per-task noise fitting close HyperBO's edge?**
*Focused sweep (4×6 = 24 runs, 200 configs, 30 iters, `n_pretrain=25`): `em_frozen` (frozen
`COND_NOISE`) vs `em_noisefit` (freeze the empirical mean+Σ, fit ONLY the likelihood noise per
task) vs `hyperbo_frozen`.* Final regret @30 / avg-rank / 95%-coverage vs σ:

| σ_obs | em_frozen | em_noisefit | hyperbo_frozen |
|---|---|---|---|
| 0.0  | 0.08 / 2.35 / 0.82 | 0.07 / 2.12 / 0.89 | **0.00 / 1.81 / 0.99** |
| 0.1  | 0.09 / 2.31 / 0.68 | 0.07 / 2.21 / 0.91 | 0.02 / 1.90 / 0.99 |
| 0.25 | 0.15 / 2.44 / 0.54 | 0.09 / 2.12 / 0.94 | 0.03 / 1.85 / 0.98 |
| 0.5  | 0.40 / 2.65 / **0.33** | **0.06 / 2.04 / 0.96** | 0.02 / 1.77 / 0.93 |

**Verdict (a clean positive result).** Fitting *only* the likelihood noise (`em_noisefit`),
freezing the empirical mean+covariance, **closes EM's noise-regime deficit**: final regret stays
flat (0.07→0.06) where `em_frozen` degrades ~5× (0.08→0.40), and its avg-rank overtakes
`em_frozen` by a widening margin as σ grows. The mechanism is **calibration** — `em_frozen`'s
frozen `COND_NOISE` cannot absorb the injected noise, so its 95% coverage collapses (0.82→0.33)
and NLL explodes (0.5→23.7 std-units); `em_noisefit` restores near-nominal coverage (0.89→0.96,
NLL ≤0.4), matching HyperBO's learned-noise calibration (0.93–0.99). **This confirms the §5K
attribution: HyperBO's noise-regime advantage over frozen-EM is the *noise adaptation*, and EM
recovers it with a single fitted scalar** (no base-kernel fit). `em_noisefit` is the recommended
EM variant under observation noise.

---

## 5L. Batch BO: greedy top-q vs kriging-believer vs fantasy qLogEI **[committed]**

*The earlier batch study used only a finite-pool greedy top-q selection (no fantasies)
and flagged "a fantasy-based qLogEI is the stronger comparison" as the open follow-up.
That is now implemented and run.* `--batch-mode {topq,kb,fantasy}`:

- **topq** — top-q by acquisition value; ignores within-batch redundancy.
- **kb** — sequential greedy, *kriging believer*: after each pick, condition on a
  deterministic fantasy at the posterior mean.
- **fantasy** — sequential greedy with `--n-fantasies` Monte-Carlo draws, i.e. the proper
  qEI/qLogEI construction `a_j(x) = E_y[a(x | D ∪ {(x_i,y_i)}_{i<j})]`, averaged in EI
  space via logsumexp.

A *joint* qLogEI is deliberately not used: PACOH's mixture posterior and ABLR's analytic
linear head expose **marginals only**, so a joint-MC acquisition could not be applied
uniformly across the method set. Sequential greedy with fantasies is BoTorch's own
`sequential=True` strategy and keeps every surrogate on identical footing.

*Setup: 4 eval × 6 seeds = 24 runs, 200 configs, 20 rounds, `n_pretrain=25`,
`n_fantasies=4`. A round is q evaluations, so q=4 @round-20 = 80 evaluations; compare
modes **within** a q, not across.* **Regret-AUC** (lower = better) and online ms/run:

| q | mode | em_frozen AUC | hyperbo AUC | vanilla AUC | em ms/run | vanilla ms/run |
|---|---|---|---|---|---|---|
| 1 | (sequential) | 0.870 | 1.197 | 2.150 | 132 | 4 892 |
| 2 | topq | 0.741 | 0.771 | **1.236** | **133** | 7 143 |
| 2 | **kb** | **0.574** | **0.658** | 1.350 | 354 | 13 669 |
| 2 | fantasy | 0.659 | 0.663 | 1.361 | 807 | 35 279 |
| 4 | topq | 0.442 | 0.521 | **0.722** | **135** | 9 204 |
| 4 | **kb** | **0.407** | 0.436 | 0.753 | 686 | 37 522 |
| 4 | fantasy | 0.410 | **0.381** | 0.977 | 2 108 | 122 339 |

**Takeaways.**
- **Fantasies genuinely help the meta-priors.** At q=2 they cut EM's regret-AUC by 21%
  (0.741 → 0.574 kriging-believer) and HyperBO's by 15%; at q=4 HyperBO improves 27%
  (0.521 → 0.381 with MC fantasies). So the earlier top-q caveat was well-founded — the
  greedy batch *was* leaving something on the table.
- **Fantasies actively hurt the from-scratch GP** (q=4: 0.722 top-q → 0.977 fantasy,
  −35%). This is the interesting mechanism: a fantasy is only as good as the posterior
  it is drawn from. A well-calibrated transferred prior produces informative fantasies;
  a from-scratch GP fit to a handful of points produces misleading ones and compounds
  its own error across the batch. **Fantasy-based batching is a benefit that transfer
  learning unlocks, not a free win.**
- **Kriging believer is the sweet spot.** It captures most of the gain (best for EM at
  both q) at 2.7–5× the online cost, whereas MC fantasies cost 6–16× and only beat KB
  for HyperBO at q=4. With `n_fantasies=4` and q=4, `vanilla_gp` costs **122 s/run** —
  25× the sequential baseline — for a *worse* result.
- **The method ranking is preserved at every q and mode** (EM/HyperBO ≫ vanilla >
  random), so all conclusions transfer to parallel BO.
- `random` is byte-identical across all three modes at a given q, confirming the batch
  logic does not perturb the shared RNG stream.

---

## 6. Insights for the paper

1. **The empirical (EM) GP is the strongest and cheapest method** on both 7D BO and 1D
   extrapolation — best early, best/again-tied at the frontier, well-calibrated, ~0 extra
   online cost.
2. **HyperBO is a strong, honest baseline** — statistically tied with EM at the 7D frontier
   and competitive on 1D; its only deficit is slower early progress (it must learn what EM
   gets for free from the shared-pool empirical mean). Report it with ≥2000 pretrain iters
   and per-task noise adaptation.
3. **The value is largely in the meta-learned *mean* (zero-shot, §2.1):** a single no-data
   pick from the prior mean (EM 1.71) beats random/vanilla after 10–20 evals, and flat-mean
   models collapse to random at @0. **PACOH's mean is actually fine** (2nd-best zero-shot);
   its full-loop weakness is the SVGD-mixture *uncertainty*, not the mean — a clean
   mean-vs-uncertainty story.
4. **PACOH-GP is genuinely weak in the loop** — verified across *every* faithful axis
   (hyper-prior scale, SVGD length-scale, iterations, **MLP width**, and **K=1 MAP vs K=10
   SVGD**, §5/§5A). MAP ≈ SVGD at scale, so it is not simply an SVGD-collapse artifact;
   the deep-kernel prior itself underperforms the empirical/EM prior here. Its *mean* is,
   however, fine (§2.1, §5B) — the loop weakness is uncertainty/modeling, not the mean.
5. **Acquisition-robust ranking + mean-vs-uncertainty split (§5B):** EM/HyperBO ≫ vanilla >
   PACOH holds under greedy/LogEI/UCB. HyperBO is worst under greedy (weak mean) but best
   under LogEI/UCB (exploration rescues it); EM is mean-driven and robust.
6. **The meta-learned prior transfers to unseen configs (§5D)** — no degradation on a
   held-out config pool — so EM's edge is not a shared-pool artifact.
7. **Objective choice is regime-specific:** EKL helps HyperBO's 1D point accuracy but
   **hurts in 7D** (§5C); use NLL for 7D. On 1D, EM ≈ HyperBO-adapt across all context
   fractions, and both widen their lead over a vanilla GP as context grows (§5E).
8. **Compute matters:** present the perf-vs-cost Pareto. Per-step full-GP refits
   (`vanilla_gp`, `em_finetuned`) cost 15–21 s/run for no gain; frozen priors are ~50× cheaper.
9. **Meta-data scaling (§5F):** EM stabilizes with the fewest pre-training tasks (~0 regret
   by 10 datasets); HyperBO is data-hungry (erratic <15, wins only at ≥15–25); PACOH never
   catches up. Report the regret-vs-#tasks curve — it is the canonical meta-learning figure.
10. **`n_init` sweep (§5G):** the meta-prior's advantage is concentrated at small init
    budgets and shrinks as random-init data grows — quantifies *why* BO (scarce evals) is
    the right regime for these priors. EM owns the no-data pick; HyperBO wins converged rank.
11. **OOD honesty (§5H):** under a deliberate response-shape distribution shift, the *frozen*
    empirical prior loses its edge (worst GP method), but **`em_finetuned` leads** — the
    trainable additive base kernel corrects a mis-specified prior. In-distribution vs OOD is
    the frozen-vs-finetuned trade-off; report both.
12. **Additive EM variants (§5I):** in-distribution the additive-base variants (incl.
    `em_additive_3way`) give **no gain** over plain `em_frozen` and are nominally weaker
    early — keep `em_frozen` as the headline EM; reserve `em_finetuned` for the OOD story.
13. **Observation-noise robustness (§5K):** HyperBO is the most noise-robust (final regret
    flat ~0.02; its rank improves as σ grows) because it *learns* the GP noise; `em_frozen`
    degrades gracefully with its frozen conditioning noise; `vanilla_gp` is the most
    noise-fragile. **HyperBO's edge over EM widens under observation noise** — complementary
    to EM's noiseless-regime win.
14. **Batch BO (§5L) [committed]:** the ranking is preserved under batching, and
    **fantasy-based qLogEI genuinely helps the meta-priors** (EM regret-AUC −21%,
    HyperBO −27%) while **hurting a from-scratch GP** (35% worse) — a fantasy is only as
    good as the posterior it is drawn from, so fantasy batching is a benefit that
    transfer *unlocks*. Kriging-believer captures most of the gain at 3–5× the online
    cost; full MC fantasies cost 6–16× and only pay off for HyperBO at q=4.
15. **ABLR transfer baseline (§10):** a strong, competitive LCBench reference (rank 2, inside
    the EM/HyperBO cluster) that validates the harness — yet it too gives *no* benefit on the
    under-powered PD1-matched pool (below random/vanilla), corroborating the PD1 negative from
    a different method family.
16. **Noise-adapted EM (`em_noisefit`, §5K):** fitting *only* the per-task likelihood noise on
    the frozen empirical prior closes HyperBO's noise-regime edge (final regret 0.40→0.06 at
    σ=0.5) by restoring calibration — cheap noise adaptation recovers most of the deep kernel's
    robustness.
17. **Early-horizon significance (§5J) [committed, 192 runs]:** the meta-prior advantage
    is *statistically real* — on regret-AUC (Friedman χ²(8)=360, p≈5e-73; Nemenyi CD 0.87)
    all six meta-priors separate from vanilla/PACOH/random by 1.5–3.8 rank units, and
    **`em_noisefit` is rank-1** (3.49), significantly ahead of ABLR and frozen HyperBO.
    The "everything ties" holds only at *final* regret, after the finite pool is exhausted.
18. **Calibration tracks benchmark difficulty (§11, §9.2–§9.4):** all methods calibrate on smooth
    LCBench (95% coverage 0.76–1.00) but collapse on PD1's bimodal objective (0.16–0.33). A
    Gaussian output warp roughly halves PD1's miscalibration (HyperBO 0.29→0.58) but does not by
    itself let transfer beat random — the low-signal/short-horizon *regime* dominates the PD1
    result, not calibration alone.
19. **PD1 reconciles with Wang et al. — it was a data/setup artifact (§12).** A fair, discriminating
    leave-one-out comparison (subgroup-matched 73-config pool, EM included) shows transfer's PD1
    edge is at most a fragile ~1-rank effect, present only un-warped (§12.2); reading the paper
    (§12.4) found ≥3 deviations — the `−log(error)` output warp, PD1Lite's ~50–73 matched configs
    (vs the paper's ~500), and ~25× less HyperBO pre-training. Isolating the paper's warp (§12.5)
    shows it *hurts* on our tiny pool, localizing the cause to **data + training budget, not the
    method**. Meta-learned GPs do win on PD1 in the full-data regime; PD1Lite just lacks the data
    to show it (LCBench §5J independently confirms the real, significant advantage).
20. **Structured-correction design (§13):** for the conditioning-time base kernel, **additive
    (`K_emp + β·B`) is the robust best** across 6 datasets × history-size × context (best mean
    and worst-case NLL, best rank); fixed/learned convex and the free two-scale (`α·K_emp +
    β·B`) are dominated — any design that can *down-weight* the empirical part overfits when its
    coefficient is fit on the sparse target head, whereas additive's pinned empirical weight
    only adds calibrated variance. The empirical prior's mean is excellent; only its variance
    needs fixing.
21. **EM shrinkage and the additive base are complementary, not redundant (§13.1):** EM's
    `covariance_shrinkage` is *necessary* for incomplete-curve imputation (α=0 triples tail
    RMSE, and the conditioning-time additive base cannot recover it), while additive is best for
    target adaptation. Convex-regularize an *estimate* (EM, rich corpus, fixed coeff);
    additively-augment a *prior* (conditioning, per-target). The asymmetry is principled.

22. **Zero-shot transfer is an extreme-region property, not a global-ranking one (§14.1).**
    EM and HyperBO order the whole pool equally well (Spearman 0.782 vs 0.771), yet EM's
    no-data pick is 3.7× better and its top-10 precision 2.8× higher. PACOH is far worse
    globally but better at the top than HyperBO — which is exactly why its zero-shot pick
    beats HyperBO's (§2.1). Report top-1/top-k precision, not rank correlation.
23. **`em_frozen`'s miscalibration grows with data (§14.2)** — σ/RMSE 1.06→0.15 and 95%
    coverage 0.95→0.43 from 5 to 50 observations, NLL up to 15. Frozen conditioning noise
    plus a rank-limited empirical covariance keeps shrinking the posterior while RMSE
    plateaus. `em_noisefit` repairs it with one fitted scalar. **Recommendation upgraded:
    make `em_noisefit` the default EM variant, not just the noisy-regime one.**
24. **Cost-normalized Pareto (§15):** amortized over 50 tasks, EM is simultaneously the
    most accurate and the cheapest (regret 0.09–0.11 at 0.7–1.2 s/task) while per-step
    refits cost 5–20× for worse regret. Charged cold to a single task, a from-scratch GP
    reaches a usable regret first — report both panels.

25. **EM shrinkage is an in-distribution-vs-transfer dial, not a tuning knob (§16).** The
    EM prior covariance has effective rank **2.9 of 200** with the bulk of directions
    below the conditioning-noise floor. In-distribution that anisotropy is a *useful
    inductive bias* — correcting it with shrinkage significantly degrades BO (@10 z=2.6,
    3W/19L) while dramatically improving calibration. Under the OOD split the sign flips:
    shrinkage significantly helps frozen EM early (@5 z=−2.5, 25W/7L) and lifts it from
    worse-than-`vanilla_gp` to better. Keep α=0 in-distribution; use α>0 only for a frozen
    prior under shift.
26. **Shrinkage and target adaptation are substitutes, not complements (§16.4).** Adding
    shrinkage on top of `em_noisefit`/`em_finetuned` over-corrects and significantly hurts
    final regret OOD (z=2.5, z=3.2). Use one mechanism, chosen by whether target data is
    available to adapt with.
27. **Calibration is not a proxy for BO utility (§16.2 vs §16.3).** Within a single method,
    NLL improved ~15 nats and 95% coverage went 0.43→0.94 while BO regret significantly
    worsened. The calibration tables (§11, §14.3) are posterior diagnostics, not a
    ranking of optimization quality.

28. **The PD1 negative was a data artifact; the full release overturns it (§17).** On the
    complete public PD1 (400 matched configs across 23 tasks, vs PD1Lite's 50), with
    leave-one-out over all tasks and the paper's 50k-step HyperBO budget, **all six
    transfer methods form the tied-best clique and every one is significantly ahead of
    `vanilla_gp`, PACOH and random** (Friedman χ²(8)=315, p≈3e-63; best-to-random gap 4.4
    rank units vs CD 1.12). Random went from rank 1–3 on PD1Lite to **last**. §12.5's
    diagnosis was right.
29. **EM transfers on PD1 after all (§17.4).** `em_noisefit` and `em_finetuned` sit inside
    the top clique, indistinguishable from ABLR and HyperBO — retiring §9.1's "the
    empirical mean does not transfer on PD1", which was the 50-config pool talking.
30. **The `−log` warp behaves as the paper intends once the pool is dense (§17.2).** It
    lifts HyperBO relative to EM at 400 configs, the opposite of its effect at 50–73
    (§12.2/§12.5) — the amplification of the good-config region only pays off when there
    are enough points to model the peaked objective. The transfer-vs-random conclusion
    itself is robust to the transform.
31. **The two benchmarks now agree (§17.4).** {EM variants, HyperBO, ABLR} ≫ vanilla_gp >
    PACOH > random on both LCBench and PD1. PACOH's weakness is the one ranking that never
    moved across benchmarks, pool sizes or transforms.

32. **HyperBO was under-trained throughout (§19).** 2 000 pre-training steps sits far
    below the ~5 000-step knee; going to 25 000 cuts its regret-AUC 33 % (0.810→0.539).
    The 2k budget was originally chosen on *final* regret, which §5J later showed is
    saturated — a criterion error that propagated into every HyperBO comparison.
33. **One prior, many runs: our per-run error bars overstated significance (§19.4).**
    Neural meta-learners share a single pre-trained prior across all runs, so per-run
    paired tests treat prior-draw luck as evidence. Measured over 5 independent priors,
    prior-draw sd is 0.037–0.048 AUC. Report meta-learners over multiple pre-training
    seeds and treat the prior as the unit of analysis.
34. **ABLR's advantage over our HyperBO is robust, not luck (§19.4)** — 5/5 independent
    priors, paired t=+5.53. Under-training explains much of the gap but not all of it;
    the paper's ordering is still unreproduced, and the leading suspect is H-EKL, their
    strongest variant, which we had hardcoded off in the PD1 drivers and were optimizing
    with Adam instead of their L-BFGS.

35. **RETRACTED (§27.3).** ~~The PD1 protocol, not the models, decided the ordering
    (§21.3).~~ The claim that searching each task's own ~2040 configurations flips HyperBO
    from last of the transfer methods to first rested on **one pre-training prior**. Two
    of three replicate priors put EM above HyperBO, and pooled + paired over 23 task
    clusters every transfer method is tied. See §27.3.
36. **Our matched pool could not contain their answer (§21.2).** The matched oracle
    (20.86% on cifar100_wrn) is *worse* than the paper's reported 20.48%; on the full
    pool it is 20.40%. Any comparison in which the baseline's published result is
    unreachable in your candidate set is not a comparison.
37. **Prior work never used regret-AUC (§22.1).** Zero mentions. Their headline is a
    performance profile — fraction of tasks solved to within absolute regret
    0.05/0.01/0.001 at each iteration. Ranking on AUC was a second, independent reason
    our ordering disagreed with theirs.
38. **AUC and budget-to-target disagree on the fine ordering (§22).** `em_finetuned` is
    the worst transfer method by AUC (0.786) and among the best by evaluations-to-target
    (3.0) — the explore-then-converge profile AUC mis-scores. ABLR is the reverse: best
    AUC, slowest to the first target, because its zero prior mean gives no informed first
    pick. Report the metric with the claim.
39. **The large separations are metric-invariant and are the safe claim.** Every transfer
    method reaches the 5% target in 2.7–4 evaluations against 13 for a from-scratch GP,
    13.7 for PACOH and 10 for random (35% solve rate) — a 3–5× sample-efficiency gain.
40. **Baselines were handicapped on data, not just budget (§21.2).** Wang et al. train
    H-NLL/FSBO/ABLR/MIMO on matched **and** unmatched data (~1600 pts/task) and restrict
    only H-EKL to matching inputs. We had been giving every baseline 400.
41. **H-EKL is not the missing piece (§19.6).** With its correct L-BFGS optimizer it is
    ~3× worse than H-NLL on LCBench (5/5 priors, t=+5.56) and loses decisively on PD1
    (z=+7.48) — the setting the paper reports it winning. This *vindicates* §5C. Its one
    real advantage is cost: 63–109× cheaper to pre-train.
42. **A research variant can be a silent no-op (§21.1).** The additive EM variants hooked
    `_effective_Sigma_inducing`, which the active interpolation path never calls, so they
    were bit-identical to `em_noisefit` and every "additive vs convex" conclusion drawn
    from them was vacuous. Bit-identical trajectories across nominally different models
    should be treated as a failed assertion, not a coincidence.
43. **Once the additive base actually runs, the plain version wins (§21.1).** Library
    `base_covar_module` (`em_finetuned`, 0.573) beats the research variants that add a
    free empirical scale (freshbase 0.677, 3-way 0.796). EM's levers — shrinkage, per-task
    noise, additive base — are now all pulled; only `coordinate_ascent_em` and the
    pre-training noise remain untried.

44. **Shrinkage is worth up to 14 nats on 7D NLL (§23.1).** `em_frozen` goes from 14.107
    to −0.226. The published −0.415 is reproducible only with the additive base AND
    shrinkage together — neither of which the manuscript describes.
45. **More inducing points is better, not worse (§23.2).** NLL improves monotonically in
    M (50 → 100 → 200), refuting the rank-to-dimension hypothesis and making the paper's
    M=100 suboptimal.
46. **Conditioning noise is a no-op for the variants that matter (§23.2)** — they fit the
    noise, so the frozen value is overwritten.
47. **The EM-vs-HyperBO gap is regime-dependent (§23.3).** EM wins at n=5, ties at n=20,
    loses at n≥50. This confirms §5.5's *scoped* data-efficiency claim while refuting
    Table 2's dominance at all n.
48. **Optimal shrinkage grows with the target budget (§23.3)** — 0, 0.1, 0.2, 0.3+ — as
    the MAP-EM reading predicts.
49. **For multi-output/multi-task, the additive base is the whole story (§23.4).** It
    fixes the NLL blow-up in both; afterwards multi-task ties single-task and
    multi-output still loses to independent per-output models.

50. **The rank ceiling is not the constraint (§24.1).** Shrinkage and the additive base
    both make Σ full rank, and the near-full-rank M=30 arm is the *worst* configuration.
51. **Variance miscalibration is not the constraint (§24.2).** Optimal variance rescaling
    WIDENS the EM-HyperBO gap, so no post-hoc calibration can close it.
52. **A confounded experiment is worse than none (§24.3).** Subsetting datasets instead of
    passing `inducing_points` starved every method; HyperBO's NLL hit 12.3 and the
    off-grid question is still open.
53. **A learned kernel metric closes 38% of the gap (§24.4)** and gives EM a better RMSE
    than HyperBO at n=100 (0.162 vs 0.176).
54. **A representational guarantee is not an optimization guarantee (§24.4).** The deep
    kernel provably contains the plain one, yet is worse at n=20 because the fit does not
    find it.

55. **The n=5 EM win does not replicate (§25.2).** At 10 eval datasets instead of 5, EM is
    *worse* at n=5 and never significantly beats HyperBO at any budget; the crossover to
    significant HyperBO advantage is n=8.
56. **The eval split moves results more than the effects under study (§25.3).** HyperBO's
    n=100 NLL swung -0.995 -> -0.200 from changing n_pretrain 30->25 and the split. Treat
    all single-split orderings in this document as provisional.
57. **EM is mid-tier in BO (§25.4)** — 5th of 10, clearly in the transfer tier, but ABLR
    leads by ~4x its own prior-draw sd.
58. **EM's prior-draw sd is exactly zero (§25.4)**, because EM pre-training is
    deterministic. The neural baselines carry 0.03-0.27.
59. **EM beats HyperBO on budget-to-target while losing on AUC (§25.4)** — the metric, not
    the method, decides the ordering.

60. **Pair your comparisons, and cluster by split (§26.1).** Unclustered per-method SEMs
    (±0.13-0.17 AUC) make everything look tied; the paired-by-split test recovers the
    signal and shows em_finetuned significantly beats ABLR/PACOH.
61. **The §25.4 "ABLR leads, EM 5th" headline was a single-split artifact (§26.1).** ABLR
    swings 0.537 -> 1.069 across splits. EM is top-tier and tied with HyperBO.
62. **EM is Pareto-dominant (§26.2)** — best budget-to-target AND 44-629x cheaper
    pre-training. This margin is too large for split noise to overturn, unlike every
    accuracy ordering in this document.
63. **HyperBO's kernel helps as an ADDITIVE component, not as the canonical one
    (§26.3).** em_additive_hyperbo tops all four regression cells and beats standalone
    HyperBO on both metrics; fitted weight on-grid, frozen off-grid.
64. **Silent no-ops are the recurring failure mode of this stack (§27.1).** Fifth
    occurrence: `--pd1-candidate-pool full` appeared to fall back to `matched`. Always
    verify a flag CHANGES the output before trusting a comparison built on it.
65. **That fifth occurrence was misdiagnosed, and the misdiagnosis cost more than the bug
    (§27.1).** The cause was not the loader but two sweep scripts omitting `--benchmark
    pd1_loo`, so eight "PD1" cells silently ran LCBench. The wrong diagnosis then declared
    the highest-value experiment blocked while its results sat finished on disk.
    **Check the run's own recorded config before theorising about the code** — every one
    of those JSONs says `benchmark: lcbench`.
66. **"Everything on disk has been analyzed" needs a check, not an assertion (§27).** A
    48-file sweep, including the replicate priors written specifically to test the
    project's shakiest claim, had never been read. Twice now the largest available
    correction has come from analysing data already collected rather than collecting more.
67. **The heterogeneity win is synthetic-sparsity-only (§27.4).** EM's graceful
    degradation vs HyperBO's collapse (z=+3.3 → −3.5) does NOT replicate on PD1's natural
    heterogeneity, where every method ties. Do not generalize §24/§25 beyond LCBench.
68. **Tying under a handicap is the strongest claim available (§27.5).** In arm C the
    baselines get 5× more pre-training data than EM, which is confined to the matched
    grid, and EM still ties them at 44–629× lower cost. That is more durable than any
    accuracy ordering in this document, all of which move with the prior or the split.

---

## 7. Reproduction

**Everything is driven by the committed `results/run_all.sh`** — this replaced an
uncommitted `/tmp/wave1.sh` that was lost, taking the exact invocations for §5F–§5I with
it. Waves run their jobs concurrently, capped at `THREADS` torch threads each.

```bash
cd <repo>
# recompute the committed sections end-to-end (headline, batch BO, diagnostics, figures)
pytorch/botorch/_scratch_bo/results/run_all.sh headline batch diagnose figures

# legacy sections -- invocations reconstructed, not yet re-run
pytorch/botorch/_scratch_bo/results/run_all.sh scaling ninit ood noise pd1 curve
```

| wave | section | scale |
|---|---|---|
| `headline` | §5J | 12 datasets × 16 seeds = 192 runs, 300 configs, 40 iters, 9 methods |
| `batch` | §5L | q∈{1,2,4} × {topq, kb, fantasy}, 4×6 = 24 runs, 20 rounds |
| `diagnose` | §14 | 6 eval datasets, `n_obs ∈ {5,10,20,50}`, 9 methods |
| `figures` | §15 | aggregation + all plots from the raw JSONs |
| `shrinkage` / `ood_shrinkage` / `spectrum` | §16 | EM shrinkage in- and out-of-distribution, plus the covariance eigenspectrum |
| `results/run_pd1_full.sh neglog none` | §17 | faithful PD1 LOO on the full release (sharded; merge with `analyze --result a.json,b.json,...`) |
| `scaling` / `ninit` / `ood` / `noise` / `pd1` / `curve` | §5F–§5K, §9–§12, §3/§5E | see the script |

Outputs land in `results/{raw,logs,figures}/` with an append-only `MANIFEST.tsv` logging
job name, UTC start and the full command. Raw JSON stores full **per-run trajectories**
(`per_run.traj_raw`), so rank/CD statistics at any horizon, regret-AUC and every figure
are reproducible after the fact without re-running anything. Timings (pre-training +
per-method online) are under `"timings"`.

Statistics come from `analyze.py`; conventions (clustered SEM, tie-averaged ranks,
fractional win-credit, Friedman/Nemenyi with N = #runs) are documented in
`results/README.md`.

**Long runs must be detached** (`setsid ... </dev/null &`): the headline sweep takes
~75 min and a plain `nohup ... &` from a short-lived shell gets reaped with its parent.

---

## 8. Recommendations / open improvements

See the companion review (chat) for the full setup critique. Highest-value faithful
follow-ups, roughly in priority order:

1. **Give PACOH its strongest honest shot:** sweep **MLP width** (`(8,)`, `(16,)`) — smaller
   particle dimension should reduce SVGD collapse — before finalizing the "PACOH is weak"
   claim. Also try a PACOH-MAP (K=1) sanity point.
2. **Acquisition sweep (A.4a):** rerun the headline under greedy-posterior-mean (pure
   exploit) and UCB/Thompson (explore) in addition to LogEI. This formalizes the
   mean-vs-uncertainty attribution from §2.1 (expect PACOH to look better under greedy-mean,
   worse under UCB) and proves the ranking is not a LogEI artifact.
3. **HyperBO EKL in 7D:** the pool is shared across datasets, so EKL's matched-input
   requirement is satisfied in 7D too (currently only NLL is used there). Untried lever.
4. **Scale up** the headline (more eval datasets, ≥16 seeds) and add **OOD split**
   (`--ood-split`) plus a **novel-config split** (eval configs held out of the pretrain
   pool) — the latter is the discriminating test of the empirical mean's transfer.
5. **1D context sweep (A.4b)** (0.1/0.2/0.4) for the accuracy-vs-context diagnostic curve.
6. **Cost-normalized main figure** (regret vs wall-clock) using the recorded timings.
7. Consider dropping `em_finetuned`/`vanilla_gp` per-step refits or bounding restarts —
   they dominate wall-clock with no benefit.

**Done [committed — raw data, invocations and figures in `results/`]:**
**wider-eval-set high-power table with ranks / Friedman / Nemenyi CD at every horizon
and on regret-AUC (§5J, 192 runs)**, **batch BO with fantasy-based qLogEI vs
kriging-believer vs greedy top-q (§5L)**, **all-method surrogate diagnostics: prior-mean
extreme-region precision + calibration vs #observations (§14)**, **the figure set:
shaded-SEM trajectory main figure, CD diagrams, Dolan-Moré performance profile, data
profile, and the cost-normalized regret-vs-wall-clock Pareto (§15)**.

**Done [legacy — numbers as recorded, raw data lost, invocations reconstructed in
`run_all.sh`]:** zero-shot `n_init=0` (§2.1), HyperBO/PACOH perf-vs-cost Pareto (§4),
PACOH single-knob sweep (§5), PACOH capacity+MAP ablation (§5A), acquisition ablation
(§5B), HyperBO EKL-vs-NLL 7D (§5C), novel-config split (§5D), 1D context sweep (§5E),
meta-data scaling (§5F), `n_init` 0→10 sweep (§5G), OOD dataset split (§5H), EM-variant
comparison (§5I), observation-noise robustness (§5K), second benchmark PD1 (§9–§9.4),
ABLR transfer-BO baseline (§10), trajectory calibration (§11 — superseded by §14.3), the
fair discriminating PD1 leave-one-out comparison and Wang-et-al. protocol reconciliation
(§12), structured-correction design (§13).

**Remaining (optional):**
1. **Re-run the legacy sections through the committed pipeline** so every number in this
   document is backed by a stored artifact. `run_all.sh` already defines the waves; this
   is compute, not work. Highest value: §5K (noise) and §9–§12 (PD1), whose conclusions
   the new calibration findings in §14.2 bear directly on.
2. **Faithful PD1 reproduction on the FULL PD1 dataset** (~500 matched + ~1500
   unmatched/task) with the `−log` warp + 50k-step HyperBO pre-training. PD1Lite is the
   binding constraint (§12.5), so this needs the data, not more code.
3. **Promote `em_noisefit` to the headline EM variant** throughout the paper: it is AUC
   rank-1 (§5J), fixes the data-growing miscalibration (§14.2), and costs one scalar fit.
   This supersedes recommendation 12's "keep `em_frozen` as the headline EM".
4. Fantasy-based batching for the *remaining* surrogates (PACOH/ABLR were left out of the
   batch sweep for cost); and a joint qLogEI restricted to the models that expose a joint
   posterior, as a cross-check on the sequential-greedy approximation.

---

## 9. Second benchmark: PD1 (HyperBO's home turf) **[legacy — superseded by §17]**

To test whether EM's win is LCBench-specific, the whole pipeline was ported to **PD1**
(Wang et al. 2021) — HyperBO's *own* benchmark — via `--benchmark pd1` (matched-config
loader from internal object storage; objective `-valid/error_rate`, higher = better). EM/HyperBO's
shared-input requirement forces the **intersection** of hyperparameter configs across the
23 tasks: **50 matched configs, d=4.** *Setup: 4 eval × 6 seeds = 24 runs, 50 configs,
30 iters, `n_pretrain=15`, `meta_iters=2000`.* Simple regret ± SEM (rank/win% by final).

| method | @1 | @5 | @10 | @20 | @30 | avg_rank | win% |
|---|---|---|---|---|---|---|---|
| em_frozen | 0.090±0.039 | 0.018±0.007 | 0.006±0.002 | 0.003±0.001 | 0.001±0.001 | **3.27** | 23 |
| em_finetuned | 0.089±0.040 | 0.022±0.008 | 0.012±0.007 | 0.003±0.002 | 0.000±0.000 | 3.54 | 16 |
| random | 0.062±0.026 | 0.015±0.004 | 0.009±0.003 | 0.004±0.001 | 0.001±0.001 | 3.92 | 17 |
| hyperbo_frozen | 0.096±0.040 | 0.031±0.009 | 0.010±0.003 | 0.004±0.001 | 0.001±0.000 | 4.12 | 10 |
| hyperbo_adapt | 0.086±0.037 | 0.033±0.009 | 0.013±0.004 | 0.004±0.001 | 0.001±0.000 | 4.12 | 10 |
| vanilla_gp | 0.045±0.015 | 0.019±0.006 | 0.008±0.003 | 0.005±0.002 | 0.001±0.001 | 4.15 | 18 |
| pacoh_frozen | 0.068±0.030 | 0.032±0.007 | 0.027±0.008 | 0.006±0.002 | 0.001±0.001 | 4.88 | 6 |

**Takeaways (honest — the matched-config subset is under-powered).**
- **The PD1 matched pool is tiny (50 configs) and easy.** With 30 iters (66% of the pool)
  every method reaches ~0.001 regret; even @5 the methods are within SEM (em 0.018, random
  0.015, vanilla 0.019). **The LCBench empirical-prior advantage does not replicate here —
  there is almost no room to be sample-efficient.**
- **`em_frozen` is nominally rank-1** (3.27, 23% win) so EM does not *hurt* on PD1 — but
  **`random` ranks 3rd** (above HyperBO/vanilla/PACOH), the tell-tale sign of a *saturated*
  benchmark rather than a good random search. Read the ordering as "tied within noise."
- **PACOH is again last** (rank 4.88) — the one ordering consistent with LCBench.
- **Root cause & options.** EM/HyperBO's matched-input requirement collapses PD1's per-task
  configs to a 50-point intersection. A discriminating PD1 test needs either (a) a **shorter
  horizon** (early regime — §9.1: does not rescue it), (b) a larger matched pool, or (c) PD1's *full*
  unmatched space with a matched-input-free surrogate (which rules out frozen-EM) — **tested
  in §9.2: transfer still does not clearly beat random, so the matched-input constraint was
  not the cause; PD1 is a genuinely low-signal transfer benchmark.** Net
  honest statement: **on the fair matched-config PD1 the methods are statistically tied and
  the benchmark is under-powered at 50 configs — EM's LCBench win neither replicates nor is
  contradicted.** Pretrain cost transfers (em 4 s, hyperbo 62 s, pacoh 96 s).

### 9.1 PD1 early regime (short horizon, `n_iters=10`)

*To rule out saturation (30 iters covers 66% of the 50-config pool), re-ran at `n_iters=10`
(≈13/50 configs evaluated), 4 eval × 10 seeds = 40 runs.* Regret @1/@3/@5/@10 + rank/win%.

| method | @1 | @3 | @5 | @10 | avg_rank | win% |
|---|---|---|---|---|---|---|
| vanilla_gp | 0.040±0.010 | 0.028±0.007 | 0.018±0.005 | 0.010±0.003 | **3.44** | 23 |
| hyperbo_frozen | 0.087±0.029 | 0.040±0.009 | 0.026±0.006 | 0.009±0.002 | 3.58 | 14 |
| random | 0.049±0.016 | 0.028±0.012 | 0.014±0.003 | 0.008±0.002 | 3.58 | 25 |
| hyperbo_adapt | 0.081±0.028 | 0.041±0.009 | 0.027±0.006 | 0.011±0.003 | 3.86 | 11 |
| em_frozen | 0.079±0.029 | 0.054±0.024 | 0.016±0.005 | 0.006±0.002 | 3.94 | 14 |
| em_finetuned | 0.080±0.029 | 0.039±0.017 | 0.019±0.005 | 0.010±0.004 | 4.05 | 9 |
| pacoh_frozen | 0.053±0.018 | 0.029±0.005 | 0.028±0.005 | 0.022±0.005 | 5.56 | 4 |

**Conclusion (the early regime does not rescue PD1 — it sharpens the negative).**
`vanilla_gp` (rank 3.44) and `random` (3.58) *lead*; `em_frozen`/`em_finetuned` rank 5th–6th,
and their **zero-shot @1 picks are worse than random** (EM 0.079 vs random 0.049 vs vanilla
0.040). So on the matched-config PD1 subset the empirical mean **does not transfer** — unlike
LCBench, PD1's cross-task config→error structure is not shared enough for the prior mean to
help (the 15 image/LM pretrain tasks are heterogeneous, and the matched intersection is only
50 configs). PACOH is again last (5.56), the sole ordering consistent with LCBench.

**Honest bottom line:** EM's sample-efficiency win is **LCBench-real but does not generalize
to PD1-matched** — we report this as a genuine limitation, not a strength. The likely culprit
is the matched-input constraint interacting with heterogeneous PD1 tasks; a matched-input-free
transfer baseline (ABLR) is the fairer PD1 comparison and is the natural next step.

### 9.2 Full (unmatched) PD1 — was it the matched-input constraint?

*The §9/§9.1 negative could be an artifact of collapsing PD1 to 50 matched configs.
`--benchmark pd1_full` gives the matched-input-free methods their REAL per-task design
space (each task its own pool, capped at 200 trials; EM / pretrained-GP / warm-start are
excluded — they require a shared pool). 4 eval × 6 seeds = 24 runs, 30 iters,
n_pretrain=15.* Regret ± SEM + rank/win%:

| method | @1 | @5 | @10 | @20 | @30 | avg_rank | win% |
|---|---|---|---|---|---|---|---|
| pacoh_frozen | 0.006±0.001 | 0.004±0.001 | 0.001±0.001 | 0.000±0.000 | 0.000±0.000 | **3.10** | 21 |
| ablr | 0.006±0.001 | 0.003±0.001 | 0.003±0.001 | 0.002±0.001 | 0.000±0.000 | 3.12 | 24 |
| random | 0.010±0.004 | 0.002±0.001 | 0.002±0.001 | 0.001±0.000 | 0.001±0.000 | 3.56 | 18 |
| vanilla_gp | 0.014±0.006 | 0.003±0.001 | 0.002±0.001 | 0.001±0.000 | 0.001±0.000 | 3.71 | 17 |
| hyperbo_frozen | 0.006±0.002 | 0.004±0.001 | 0.001±0.000 | 0.001±0.000 | 0.000±0.000 | 3.75 | 10 |
| hyperbo_adapt | 0.006±0.002 | 0.002±0.001 | 0.001±0.000 | 0.001±0.000 | 0.000±0.000 | 3.75 | 10 |

1-step-ahead posterior calibration at acquired points (standardized units): pacoh **NLL
1.6 / cov 0.91**; hyperbo_frozen 10.6 / **0.20**; hyperbo_adapt 10.7 / 0.22; ablr 12.2 /
0.33; vanilla 15.9 / 0.74.

**Takeaways — the matched-input constraint was NOT the culprit.**
- **Even on PD1's full design space, transfer does not clearly win.** ABLR (3.12) and PACOH
  (3.10) edge random (3.56) / vanilla (3.71) only *marginally* (all @30 ≈ 0, SEMs overlap),
  and **HyperBO ranks last (3.75), below random.** So the §9 negative is not an artifact of
  the 50-config matched pool — **PD1 is a genuinely low-signal transfer benchmark**: its
  heterogeneous image/LM tasks share too little config→error structure and each pool is
  easily near-optimized (tiny regrets).
- **Calibration explains the failure.** At the acquired points HyperBO and ABLR are
  *severely overconfident* — 95% coverage of only **0.20–0.33** (predictive σ far too small
  for PD1's cross-task shift) — while PACOH's broad SVGD mixture stays calibrated (0.91).
  Overconfident transfer posteriors provide no useful exploration signal; the one method
  that *is* calibrated (PACOH) is the nominal rank-1. Clean mechanistic link to §11.
- Net: this **strengthens** the honest narrative — the empirical/deep-kernel priors' PD1
  non-benefit is a real property of PD1 (weak, heterogeneous transfer + easy per-task
  optimization), robust to removing the matched-input constraint. Present PD1 as the honest
  boundary of where meta-BO helps.

### 9.3 Investigation: why doesn't HyperBO win on PD1 here, when Wang et al. 2021 report it does?

This counterintuitive result was investigated systematically (hypothesis → minimal test).
**It is not a HyperBO bug** — the same code is well-calibrated and competitive on LCBench
(§2, §5J), and on PD1 the meta-priors *do* give a real (if small) early-pick edge (see below).
Two candidate bugs were **tested and rejected**:

- **Global vs per-task Y-standardization (`--per-task-standardize`):** no material change —
  HyperBO coverage 0.20→0.33, ranking unchanged, transfer still ≈ random. *Rejected.*
- **Objective = last-epoch vs best-over-curve error (`--pd1-best-metric`):** the two objective
  distributions are near-identical (diverged configs stay at ~0.9 error), and results/calibration
  are unchanged. The diverged trials diverge *from the start* (bad learning rates that never
  train), so "best validation error" doesn't smooth them. *Rejected.*

**What we found instead (root causes of the discrepancy).**
1. **PD1's objective is intrinsically bimodal / heavy-tailed:** per-task, 10–48% of the
   randomly-sampled configs *diverge* (error ≈ 0.9, i.e. random-guessing) while the rest are
   good (≤0.4). A plain GP cannot model this sharp good/diverged boundary, so **every** method
   is badly miscalibrated at acquired points — including from-scratch `vanilla_gp` (95% coverage
   0.72) and HyperBO/ABLR (coverage 0.17–0.30). Miscalibrated posteriors → broken exploration.
2. **Our regime washes out transfer's advantage.** With ≤200-config pools and 30 BO iters
   (15–30% of the pool) and >50% good configs, **random reaches ~0 regret within ~10 evals**, so
   there is almost no room to win at the horizon we score. The meta-prior's *real* benefit shows
   only at the **first pick** (@1 regret: ABLR 0.026 / PACOH 0.038 vs random 0.050 — ~2× better)
   but is gone by @10 and does not win on regret-AUC either.
3. **The one calibrated method is PACOH** (coverage 0.90) — its broad SVGD mixture tolerates the
   bimodal landscape — and it is (nominally) the best PD1 method, reinforcing that **calibration,
   not the mean, is what PD1 rewards.**

**Most likely "missing piece" vs the paper (untested here, recommended next).** The HyperBO
reference pipeline applies a **nonlinear output warp** (Gaussian/quantile transform) to each
task's objective before fitting — precisely to tame non-Gaussian/heavy-tailed objectives like
PD1's divergence tail. We use plain standardization, so no GP calibrates on PD1. The paper also
reports **leave-one-out across all 24 tasks with normalized regret emphasizing early iterations**,
a regime that credits the divergence-avoidance our @1 numbers already show. *(Caveat: the paper's
exact protocol could not be re-fetched here; these are informed by the HyperBO method/codebase,
not verified line-by-line.)* **Recommended confirmatory experiment: add a per-task Gaussian/rank
output warp and re-run PD1** — expected to fix calibration and let HyperBO's prior separate from
random, reconciling with the paper. **(Tested in §9.4: the warp roughly halves the
miscalibration — HyperBO coverage 0.29→0.58 — but does *not* by itself let transfer beat
random; the evaluation regime, not just the warp, is the dominant factor.)**

---

### 9.4 PD1 with a Gaussian output warp (testing the §9.3 hypothesis)

*§9.3 hypothesized the missing piece is a per-task output warp. Added `--output-warp gaussian`
(rank/copula map of each task's objective to N(0,1)) and re-ran the same 8-eval-task `pd1_full`
config (per-task standardized). Regret is then in warped units; compare **calibration** and
**rank** to the matched no-warp run.*

95%-coverage at acquired points (no-warp → Gaussian-warp):

| method | no-warp | warp |
|---|---|---|
| hyperbo_frozen | 0.29 | **0.58** |
| hyperbo_adapt | 0.30 | 0.57 |
| ablr | 0.16 | 0.31 |
| pacoh_frozen | 0.90 | 0.92 |
| vanilla_gp | 0.72 | 0.73 |

avg-rank @final under the warp: random **3.07**, pacoh 3.24, vanilla 3.38, hyperbo_frozen 3.47,
hyperbo_adapt 3.80, ablr 4.05 (random still rank-1; transfer still does not separate).

**Verdict — partial reconciliation.** The warp does what §9.3 predicted *directionally*: it
**roughly doubles HyperBO's coverage (0.29→0.58)** and improves ABLR (0.16→0.31), confirming the
un-warped heavy-tailed objective was a real source of miscalibration. **But it is not sufficient**
— coverage still falls short of ~0.9 and **transfer still does not beat random** at the scored
horizon. So the output warp is *a* contributing factor, not the whole story. The **dominant reason
our PD1 doesn't reproduce Wang et al. is the evaluation regime**: small per-task pools (≤200), a
short 30-iter horizon (random reaches ~0 regret by @10 because >50% of configs are good), and an
eval split dominated by easy mnist/fashion tasks. A faithful reproduction needs the warp **plus**
the paper's protocol (leave-one-out over all 24 tasks, a longer horizon relative to the search
space, normalized early-iteration regret). Our LCBench results (§2, §5J) already show the
meta-prior advantage is real and statistically significant when those conditions hold.

---

## 10. ABLR transfer-BO baseline (Perrone et al. 2018) **[legacy]**

Reviewers expect a classic transfer-BO reference beyond EM/HyperBO/PACOH. **ABLR**
("Scalable Hyperparameter Transfer Learning") learns a **shared MLP feature map** φ across
the pretrain tasks with a **Bayesian linear-regression head** on top (analytic posterior);
we learn shared prior/noise precisions (α, β) jointly with φ by maximizing the summed
per-task evidence. Unlike frozen-EM it is **matched-input-free** (no shared-config
requirement) — the fairer transfer baseline on PD1. Added as `--methods ablr`.

**LCBench** (4 eval × 6 seeds = 24 runs, 200 configs, 30 iters, `n_pretrain=25`; ABLR
pretrain 165 s). Simple regret ± SEM, sorted by avg-rank:

| method | @1 | @5 | @10 | @20 | @30 | avg_rank | win% |
|---|---|---|---|---|---|---|---|
| hyperbo_frozen | 4.05±0.40 | 1.37±0.48 | 0.51±0.30 | **0.00±0.00** | **0.00±0.00** | **3.40** | 22 |
| **ablr** | 4.14±0.64 | 0.92±0.32 | 0.19±0.10 | 0.12±0.10 | 0.02±0.02 | **3.65** | 16 |
| hyperbo_adapt | 4.19±0.46 | 0.72±0.33 | 0.27±0.14 | 0.16±0.10 | 0.12±0.10 | 3.83 | 17 |
| em_frozen | **1.71±0.46** | 1.44±0.46 | 0.33±0.22 | 0.09±0.03 | 0.08±0.03 | 4.27 | 11 |
| em_finetuned | 1.70±0.46 | 1.44±0.46 | 0.13±0.04 | 0.11±0.03 | 0.09±0.03 | 4.27 | 11 |
| vanilla_gp | 4.42±0.52 | 3.16±0.45 | 1.78±0.39 | 0.60±0.25 | 0.22±0.10 | 4.77 | 10 |
| pacoh_frozen | 4.13±0.36 | 3.27±0.51 | 3.07±0.46 | 1.17±0.44 | 0.64±0.25 | 4.81 | 11 |
| random | 4.53±0.50 | 3.83±0.49 | 3.42±0.46 | 2.00±0.47 | 1.76±0.45 | 7.00 | 2 |

**PD1** (matched-config; 4 eval × 6 seeds = 24 runs, 50 configs, 30 iters, `n_pretrain=15`):

| method | @1 | @5 | @10 | @30 | avg_rank | win% |
|---|---|---|---|---|---|---|
| em_frozen | 0.090±0.039 | 0.018±0.007 | 0.006±0.002 | 0.001±0.001 | **3.54** | 22 |
| em_finetuned | 0.089±0.040 | 0.022±0.008 | 0.012±0.007 | 0.000±0.000 | 3.88 | 15 |
| random | 0.062±0.026 | 0.015±0.004 | 0.009±0.003 | 0.001±0.001 | 4.29 | 16 |
| vanilla_gp | 0.045±0.015 | 0.019±0.006 | 0.008±0.003 | 0.001±0.001 | 4.58 | 17 |
| hyperbo_frozen | 0.096±0.040 | 0.031±0.009 | 0.010±0.003 | 0.001±0.000 | 4.62 | 9 |
| hyperbo_adapt | 0.086±0.037 | 0.033±0.009 | 0.013±0.004 | 0.001±0.000 | 4.62 | 9 |
| ablr | 0.065±0.030 | 0.029±0.010 | 0.009±0.002 | 0.002±0.001 | 4.98 | 5 |
| pacoh_frozen | 0.068±0.030 | 0.032±0.007 | 0.027±0.008 | 0.001±0.001 | 5.48 | 6 |

**Takeaways (honest).**
- **On LCBench ABLR is a strong baseline** — rank 3.65 (2nd), slotting *into* the top cluster
  between the two HyperBO variants and clearly ahead of vanilla/PACOH (@10=0.19, @30=0.02).
  Like HyperBO it has a poor zero-shot @1 (4.14 — the BLR prior mean is 0 with no data) but is
  very sample-efficient with a few points. This **validates the harness**: the standard
  transfer-BO reference performs as expected and is competitive with EM/HyperBO. EM still owns
  the zero-shot pick (@1 = 1.71) and is the cheapest (ABLR's 165 s pretrain is the heaviest).
- **On PD1 ABLR does not rescue transfer either** — rank 4.98 (7th of 8), *below* random (4.29)
  and vanilla (4.58), like every other transfer method. Even the matched-input-free baseline
  gains nothing on the 50-config matched pool, which **independently confirms §9/§9.1**: PD1-
  matched is under-powered/saturated, not merely a matched-input artifact. `em_frozen` stays
  nominally rank-1.
- Net: ABLR both **strengthens the LCBench baseline set** (a competitive top-cluster reference)
  and **corroborates the honest PD1 negative** from a different method family.

---

## 11. Posterior calibration along the BO trajectory **[legacy — superseded by §14.3]**

*1-step-ahead predictive calibration logged at each acquired point during the high-seed LCBench
run (§5J, 128 runs): Gaussian NLL and 95% coverage of the true (near-noiseless) objective under
each surrogate's latent posterior (standardized units).*

| method | 95% coverage | mean NLL |
|---|---|---|
| pacoh_frozen | 1.00 | 1.23 |
| ablr | 0.99 | -0.56 |
| hyperbo_frozen | 0.98 | -0.46 |
| hyperbo_adapt | 0.89 | -0.33 |
| em_noisefit | 0.86 | -0.20 |
| vanilla_gp | 0.84 | 2.61 |
| em_frozen | 0.76 | 1.72 |

**Takeaways.**
- **On LCBench every method is reasonably calibrated** (coverage 0.76–1.00) — the smooth accuracy
  objective is GP-friendly. This is the sharp contrast with **PD1, where the same surrogates
  collapse to 0.16–0.33 coverage** (§9.2–§9.4): **calibration tracks benchmark difficulty /
  objective shape, not the method** — PD1's bimodal divergence tail is what breaks it.
- **`em_frozen` is slightly under-covered (0.76, NLL 1.72)** — its frozen `COND_NOISE` is a touch
  tight even on near-noiseless LCBench; **`em_noisefit` fixes this (0.86, NLL −0.20)**, the same
  mechanism that rescued it under injected noise (§5K).
- **PACOH is (over-)covered (1.00)** — its broad SVGD mixture is conservative; combined with its
  weak in-loop rank (§5J) this is the "well-hedged but too diffuse to exploit" profile. HyperBO
  and ABLR are near-nominal (0.89–0.99), consistent with their strong early ranks.
- Net: the mean-vs-uncertainty story (§2.1) now has hard numbers — the methods that win early
  (EM/HyperBO) are well-calibrated on the objective they optimize; miscalibration (PD1) is exactly
  where transfer BO breaks down.

---

## 12. Fair, comprehensive EM-vs-baselines on PD1 **[legacy — superseded by §17]**

The earlier PD1 sections had a structural gap: EM was either on a *saturated* 50-config matched
pool (§9/§9.1) or *excluded* from the discriminating full-pool setup (§9.2/§9.4). This section
closes it with the fairest discriminating comparison we can construct: **leave-one-out over the
batch-size-256 subgroup's matched pool (73 shared configs across 10 tasks — 46% larger than the
all-task intersection, and non-saturated), EM + all 7 baselines on the SAME pool**, per-task
Gaussian output warp + per-task standardization. Metrics are **per-task-normalized** (regret ÷
that task's mean initial gap) then averaged across the 10 LOO folds; ranks/win% are
dataset-clustered. `--benchmark pd1_loo --pd1-group 256`.

### 12.1 Headline — leave-one-out, Gaussian warp, EM included (50 runs)

| method | nReg@1 | nReg@5 | nReg@10 | nReg@final | AUC | avg_rank | win% | cov95 |
|---|---|---|---|---|---|---|---|---|
| ablr | 0.91 | 0.67 | 0.49 | 0.24 | 0.47 | **4.26** | 13 | 0.26 |
| em_frozen | 0.89 | 0.67 | 0.50 | **0.21** | 0.48 | 4.28 | 15 | 0.02 |
| hyperbo_adapt | 0.88 | 0.53 | 0.43 | 0.23 | **0.39** | 4.36 | 12 | 0.50 |
| random | 0.91 | 0.71 | 0.50 | 0.24 | 0.47 | 4.37 | 15 | — |
| hyperbo_frozen | 0.86 | 0.51 | 0.44 | 0.24 | 0.40 | 4.43 | 10 | 0.51 |
| vanilla_gp | 0.91 | 0.63 | 0.54 | 0.27 | 0.48 | 4.60 | 13 | 0.73 |
| em_noisefit | 0.85 | 0.62 | 0.48 | 0.25 | 0.46 | 4.61 | 11 | 0.30 |
| pacoh_frozen | 0.83 | 0.58 | 0.48 | 0.29 | 0.47 | 5.09 | 12 | 0.91 |

**Verdict — the fair comparison confirms the honest negative (robustly).** Even on a
*discriminating* pool (final normalized regret 0.21–0.29, so **not** saturated), with the
leave-one-out protocol, the output warp, and **EM in the comparison**, **no transfer method
reliably beats random**: `em_frozen` (rank 4.28) and `ablr` (4.26) are nominally best but sit
within ~0.1 rank of `random` (4.37), and HyperBO's only visible edge is a slightly lower
regret-AUC (0.39–0.40 vs 0.47) from faster early progress. The transfer advantage that is large
and statistically significant on LCBench (§5J) simply does **not** materialize on PD1 even under
the fairest setup — a robust property of the benchmark (heterogeneous tasks + a divergence-heavy
objective), not an artifact of pool size or EM's exclusion. Calibration stays the weak point:
the warp lifts HyperBO to ~0.50 coverage, but `em_frozen` remains overconfident (0.02) via its
frozen conditioning noise (`em_noisefit` → 0.30; PACOH is well-calibrated at 0.91 yet weakest on
rank). **However, this gaussian-warp headline *under*-states transfer — see §12.2: without the
warp, HyperBO clearly beats random on this same discriminating pool.**

### 12.2 Output-transform ablation (none vs Gaussian vs winsor)

Avg-rank (lower = better) / 95%-coverage per method, identical LOO setup, three output
transforms (the winsor run used a reduced method set):

| method | none (rank/cov) | gaussian (rank/cov) | winsor (rank/cov) |
|---|---|---|---|
| hyperbo_frozen | **3.49** / 0.68 | 4.43 / 0.51 | 3.26 / 0.38 |
| hyperbo_adapt | **3.62** / 0.67 | 4.36 / 0.50 | — |
| ablr | 4.38 / 0.13 | 4.26 / 0.26 | 3.09 / 0.12 |
| em_frozen | 4.60 / 0.17 | 4.28 / 0.02 | 2.83 / 0.04 |
| random | 4.69 / — | 4.37 / — | 2.87 / — |
| em_noisefit | 4.92 / 0.78 | 4.61 / 0.30 | — |
| pacoh_frozen | 5.07 / 0.87 | 5.09 / 0.91 | — |
| vanilla_gp | 5.23 / 0.75 | 4.60 / 0.73 | 2.95 / 0.76 |

**Key finding — the warp does NOT help, and *without* it HyperBO shows a real edge.** Under **no
transform**, `hyperbo_frozen`/`hyperbo_adapt` rank **3.49 / 3.62 vs random 4.69** — a modest but
clear ~1-rank advantage on the discriminating bs=256 LOO pool, in the direction the PD1 paper
reports. The Gaussian rank warp *flattens* the objective and **erases** this edge (HyperBO → 4.43
≈ random 4.37); winsor sits in between. So the §9.3/§9.4 hypothesis that a *missing output warp*
explained PD1 was **wrong in sign** — the plain objective is where HyperBO's transfer signal is
strongest; the warp mainly shifts marginal calibration without helping (indeed hurting) rank.
`em_frozen` does **not** share HyperBO's edge here (≈ random under every transform).

### 12.3 EM on the full per-task pools (#3)

EM (matched-config pretrain, off-grid eval) vs the matched-input-free baselines on PD1's full
per-task pools (8 eval tasks × 6 seeds, per-task-normalized regret):

| method | nReg@1 | nReg@5 | nReg@10 | nReg@final | AUC | avg_rank | win% | cov95 |
|---|---|---|---|---|---|---|---|---|
| em_noisefit | 0.91 | 0.69 | 0.46 | **0.18** | 0.44 | **3.46** | 15 | 0.87 |
| random | 0.89 | 0.63 | 0.54 | 0.23 | 0.47 | 3.66 | 26 | — |
| em_frozen | 0.93 | 0.64 | 0.48 | 0.22 | 0.45 | 3.72 | 16 | 0.57 |
| vanilla_gp | 0.87 | 0.67 | 0.54 | 0.25 | 0.48 | 4.08 | 9 | 0.74 |
| hyperbo_frozen | 0.94 | 0.58 | 0.48 | 0.27 | 0.48 | 4.15 | 16 | 0.62 |
| pacoh_frozen | 0.92 | 0.67 | 0.51 | 0.29 | 0.50 | 4.29 | 11 | 0.96 |
| ablr | 0.90 | 0.63 | 0.45 | 0.32 | 0.47 | 4.65 | 6 | 0.31 |

**Finding — EM is competitive on the full pools.** `em_noisefit` is the **best-ranked method
(3.46)**, ahead of random (3.66) and every baseline, with good calibration (0.87); `em_frozen`
(3.72) ≈ random. So EM — even restricted to a matched-config pretraining grid and predicting
off-grid — transfers at least as well as the matched-input-free methods on PD1's real design
space, contradicting the notion that EM simply "can't do PD1".

**Revised overall PD1 verdict (§12).** The picture is **noisy and setting-dependent, not a clean
negative**: on the discriminating bs=256 LOO, *no-warp* HyperBO beats random by ~1 rank; on the
full per-task pools, *em_noisefit* is best-ranked; yet under the Gaussian warp the LOO edges
vanish. Transfer on PD1 is **real but small and fragile** (single-rank, near SEM at these sample
sizes) — far from LCBench's large, robust, statistically-significant separation (§5J). Given
published reports of *dramatic* PD1 gains, this fragility means we are likely still missing a
protocol detail; **a direct cross-check against the paper's exact PD1 setup is the right next
step before finalizing — conclusions HELD pending that review.**

### 12.4 Protocol diff vs Wang et al. 2021 (resolves the discrepancy)

Reading the paper's LaTeX source (`exp-setup.tex`, `exp-online.tex`) against our setup found
**≥3 substantive deviations** that explain why we saw a fragile null where the paper reports
dramatic PD1 gains:

| Dimension | Paper (Wang et al.) | Ours | Impact |
|---|---|---|---|
| **Output warp** | **`r ← −log(error + 1e-10)`** (offline PD1) | `−error` / Gaussian-rank / winsor | **HIGH** |
| **Matched pool** | **~500 pts/task** (242 non-divergent) | 50 (all-24) / 73 (bs256) | **HIGH** |
| **Dataset** | full PD1 (~500 matched + ~1500 unmatched) | **PD1Lite** (reduced internal object storage subset) | **HIGH** |
| **HyperBO training** | Adam **50,000 steps**, batch 50 | `meta_iters` ~1500–2000 | **HIGH** |
| Acquisition | thresholded PI `(μ−(max+0.1))/σ` | LogEI | MED |
| Text-task objective | cross-entropy loss | `valid/error_rate` | MED |
| Aggregation | regret in warped space; mean-over-tasks, median±20/80 over seeds; ranking/perf-profiles | per-run final/AUC + clustered ranks | MED |

**Key insight:** the `−log(error)` warp *amplifies* resolution among good configs while compressing
the divergence tail — the opposite of our Gaussian-rank warp, which flattens it (and §12.2 showed
flattening *hurts*, while the un-warped run was HyperBO's best). Combined with a ~10× larger matched
pool and ~25× more pre-training, these are exactly the levers that convert our fragile ~1-rank edge
into a large gain. **Conclusion: our PD1 null is a setup/data artifact (PD1Lite + wrong warp +
under-training), NOT evidence against meta-learned GPs.** The definitive reconciliation is to
re-run PD1 with (a) the `−log` output warp, (b) 50k-step HyperBO pre-training, and (c) the full PD1
data if obtainable (PD1Lite may be the binding constraint). §5J's LCBench result already
demonstrates the real, statistically-significant meta-prior advantage.

### 12.5 The paper's `−log(error)` output warp, isolated (neglog)

Re-ran the identical bs=256 LOO with only the output transform changed to the paper's
`−log(error + 1e-10)` (`--output-warp neglog`). Avg-rank (lower = better) / 95%-coverage across
the three transforms (same folds/seeds/methods, warp is the only change):

| method | none (rank/cov) | neglog (rank/cov) | gaussian (rank/cov) |
|---|---|---|---|
| hyperbo_frozen | **3.49** / 0.68 | 4.53 / 0.36 | 4.43 / 0.51 |
| hyperbo_adapt | **3.62** / 0.67 | 4.34 / 0.35 | 4.36 / 0.50 |
| em_frozen | 4.60 / 0.17 | 4.23 / 0.09 | 4.28 / 0.02 |
| em_noisefit | 4.92 / 0.78 | 4.38 / 0.74 | 4.61 / 0.30 |
| ablr | 4.38 / 0.13 | 5.30 / 0.15 | 4.26 / 0.26 |
| pacoh_frozen | 5.07 / 0.87 | 4.94 / 0.91 | 5.09 / 0.91 |
| vanilla_gp | 5.23 / 0.75 | 4.22 / 0.73 | 4.60 / 0.73 |
| random | 4.69 / — | **4.06** / — | 4.37 / — |

**Verdict — on PD1Lite the warp is NOT the binding constraint; the DATA is.** Surprisingly the
paper's `−log` warp *hurts* here, like the Gaussian one: **random becomes rank-1 (4.06)** and
HyperBO's clear no-warp edge (3.49/3.62) is erased (→ 4.53/4.34, now below random). On our tiny
matched pool (50–73 configs) the warp's amplification of the good-config region backfires — the GP
cannot model the now-peaked objective from so few points (HyperBO coverage 0.68 → 0.36), so
acquisition degrades and random wins. This is the *opposite* of the paper's result, where the
warp helps **precisely because** their matched pool is ~500 configs.

**Reconciled PD1 conclusion.** Across every transform (none / gaussian / winsor / neglog),
transfer's PD1 signal is at most a fragile ~1-rank effect (present only un-warped, §12.2), never
the paper's dramatic gain. The protocol diff (§12.4) plus this ablation localize the cause to the
**data and training budget, not the method or the warp**: PD1Lite yields ~50–73 matched configs
(vs the paper's ~500) and we pre-train HyperBO ~25× less (2k vs 50k Adam steps). On reduced
PD1Lite, no output-warp recovers the paper's result — a faithful reproduction needs the **full PD1
dataset**. This reconciles our observations with Wang et al.: **meta-learned GPs do win on PD1 in
the paper's full-data / full-training regime; our PD1Lite setup simply lacks the data to surface
it**, while LCBench (§5J) independently confirms the real, statistically-significant meta-prior
advantage. **Net: the earlier PD1 "null" was a data/setup artifact, not evidence against
empirical/meta-learned GPs.**

---

## 13. Structured-correction design: additive base vs convex shrinkage **[legacy]**

Two design questions settled empirically here: (i) is the EM M-step `covariance_shrinkage`
necessary, and (ii) what is the best structured correction for the (rank-limited) empirical
covariance at conditioning time — **additive** (`K_emp + β·B`, the current
`BaseAugmentedEmpiricalKernel`), **convex** (`(1−t)·K_emp + t·c·B`, `c=tr(K_emp)/tr(B)`,
the EM shrinkage form), or a **free two-scale** (`α·K_emp + β·B`)?

### 13.1 EM `covariance_shrinkage` is necessary (incomplete-curve imputation)

*Setup: tutorial Section A/B — LCBench Fashion-MNIST, 100 historical curves 80%-truncated
(early-stopping simulation), 5 held-out test curves each observed on their first 30% of
epochs; score the unobserved tail after EM pretraining with `covariance_shrinkage=α`. 30 EM
iters, base Matern(nu=2.5, ls=10).*

| α (covariance_shrinkage) | tail RMSE | tail NLL | 95% cov |
|---|---|---|---|
| 0.0 | 5.41 | 3.23 | 0.85 |
| 0.1 | 2.84 | 2.94 | 0.90 |
| **0.3** (tutorial) | **1.72** | 2.90 | 0.97 |
| 0.5 | 1.61 | 2.93 | 0.99 |
| 0.0 + per-target additive base (a) | 5.22 | 3.10 | 0.85 |

**Verdict — keep it.** Convex shrinkage during EM is load-bearing: dropping it (α=0) **triples
tail RMSE** (5.4 vs 1.7) and wrecks calibration (coverage 0.85 vs 0.97). Critically, the
conditioning-time **additive base does NOT recover it** (5.22 / 0.85 ≈ α=0) — it fits the base
on the target's *observed head* and so cannot regularize the prior's *unobserved, data-starved
tail*. This is a pretraining-time estimation problem, not a target-adaptation problem.

### 13.2 Additive vs convex vs two-scale for the conditioning-time base kernel

*Setup: matrix-level exact-Gaussian conditioning on held-out 1d targets, all candidates built
from the SAME base B (Matern nu=2.5, ls=10) so the comparison isolates the blend; 6 LCBench
datasets × K∈{10,30,100} history × obs∈{0.1,0.3,0.5} × 3 seeds × 5 test curves; per-target
hyperparameters (α/β/t/noise) selected by observed-head marginal likelihood; score unobserved
tail. (Standardized targets.)*

Overall, averaged over all scenarios (lower RMSE/NLL/rank/worst-NLL better; cov target 0.95):

| candidate | mean RMSE | mean NLL | mean cov | mean rank | worst NLL | win% |
|---|---|---|---|---|---|---|
| empirical | **0.225** | 4.061 | 0.811 | 3.33 | 182.3 | 30.9 |
| **additive** (α=1, β fit) | 0.328 | **0.446** | 0.920 | **2.81** | **24.5** | 13.6 |
| convex=0.1 (fixed t) | 0.267 | 1.543 | 0.954 | 3.39 | 99.2 | 13.6 |
| convex=0.3 (fixed t) | 0.322 | 1.494 | 0.972 | 4.43 | 65.9 | 3.1 |
| learned_convex (t fit) | 0.381 | 1.324 | 0.928 | 3.94 | 36.8 | 14.8 |
| learned_alpha (α,β fit) | 0.346 | 0.964 | 0.885 | 3.10 | 53.6 | 24.1 |

**Takeaways.**
- **Additive is the robust winner** — best mean NLL, best worst-case NLL, best mean rank; it
  never blows up and dominates the hardest few-shot rank-limited cells (K=10).
- **The empirical prior's *mean* is excellent** (best RMSE); only its *variance* needs fixing
  (empirical is badly overconfident: worst-NLL 182, coverage 0.81). Corrections trade a little
  RMSE for calibration.
- **Any design that can DOWN-WEIGHT the empirical part** (convex `t`, or free `α`), when its
  coefficient is fit on the sparse observed head, **over-shrinks → worse tail RMSE and NLL
  blow-ups.** Additive's empirical coefficient is pinned at 1, so it can only *add* calibrated
  variance — that constraint is exactly why it is robust.
- **Learning the convex coefficient** beats fixed-t (adapting helps within the convex family)
  but is still dominated by additive; **the free two-scale (`learned_alpha`) is not
  best-of-all-worlds** — worse than additive on every calibration metric and less robust.

### 13.3 Design decision (evidence-based)

- **Conditioning-time / 1d empirical kernels → additive-only** (`BaseAugmentedEmpiricalKernel`,
  α≡1, β fit per target). Chosen on *performance/robustness*, not merely simplicity.
- **EM prior learning → keep convex `covariance_shrinkage` (fixed α=0.3) as the robust
  default.** Different problem: regularizing a covariance *estimate* from incomplete data for
  E-step imputation, where *some* structured M-step regularization is essential (unregularized
  EM is catastrophic — §13.4). An *additive* M-step regularizer turns out to be a competitive
  alternative (§13.4); the "additive collapses" caveat applies only to *MLL-fitting* the base
  (in-sample, or on the sparse target head — §13.1), not to a fixed/held-out-tuned regularizer.
- **The asymmetry is principled, not an oversight:** convex = *regularize an estimate toward
  structure* (bias–variance, on the rich historical corpus, fixed coefficient); additive =
  *augment a trusted prior for a new target* (add capacity, per-target fit). Do not unify.
- **Do NOT add learned-α or a learned/convex coefficient to the base kernel** — the freedom to
  shrink the transferred prior on sparse target data consistently hurts.

*Caveats: LCBench val-accuracy only; one base kernel (Matern ls=10); head-MLL grid selection.
Ablations run via ad-hoc grid-level scratch scripts (recipe above), not committed.*

Both EM follow-ups were run — see §13.4.

### 13.4 EM M-step regularizer: additive vs convex, and auto-tuned α (results)

*Hand-rolled minimal EM (E-step GP curve-completion + M-step expected-outer-product covariance
+ regularizer), 6 LCBench datasets × 2 seeds, truncated historical curves (80% early-stopped).
The regularizer coefficient is selected on held-out VAL curve tails; TEST tail metrics reported.
Prototype validated: reproduces §13.1's Fashion-MNIST trend (none 2.81 → convex-0.3 0.86 →
tuned better).*

| arm | tail RMSE | tail NLL | 95% cov | sel. coeff |
|---|---|---|---|---|
| none (α=0, unregularized) | 0.534 | 44.35 | 0.705 | — |
| convex fixed 0.3 (current default) | 0.553 | 3.72 | 0.909 | 0.30 |
| convex auto-tuned α | 0.630 | 4.02 | 0.893 | 0.10 |
| **additive auto-tuned β** | **0.460** | **3.66** | **0.917** | 0.32 |

**Findings.**
- **Structured M-step regularization is essential** — unregularized EM is catastrophic (NLL
  44, coverage 0.71); this reinforces §13.1.
- **(2) An additive M-step regularizer is NOT inferior to convex — it is the best arm here**
  (RMSE 0.46, NLL 3.66, cov 0.92), *correcting the earlier prediction that convex would win*.
  Key distinction from the failed conditioning-additive (§13.1): here β is a **fixed /
  held-out-selected regularizer** of the iterative M-step covariance (not an in-sample or
  sparse-head MLL fit), so it fills the data-starved tail during EM just as convex does and
  does **not** collapse. Dataset-dependent (convex is better on MiniBooNE).
- **(1) Auto-tuning α is mixed — it does NOT robustly beat fixed 0.3.** It helps on most
  datasets (Fashion-MNIST 0.86→0.32, plus APSFailure/Australian/KDDCup) but the held-out-val
  selection *overfits* on a couple (Amazon, MiniBooNE), so mean NLL (4.02) is no better than
  fixed 0.3 (3.72). The 20-curve val set makes selection noisy.

**Decision.** Keep **convex, fixed α=0.3** as the EM default — robust, tuning-free, and within
noise of the best tuned arm. Neither auto-tuned α (selection overfits) nor an additive M-step
(competitive but not clearly better, and a code change) justifies switching. Useful corollary:
since the additive M-step regularizer *is* competitive, unifying on the additive base for
conceptual consistency would not cost EM imputation quality if ever desired — but there is no
performance reason to do it now. Refined theory: additive "collapse" is a property of
*MLL-fitting* β, not of a fixed structured regularizer.

**Confirmatory check — trace-inflation is the *sole* difference.** Adding an
`additive_tracematched` arm (`Σ_emp + β·B`, then renormalized back to `tr(Σ_emp)`) — which
equals convex with `α = β·tr(B)/(tr(Σ_emp)+β·tr(B))` — tracks `convex` almost exactly (mean
tail NLL 4.017 vs 4.024; per-dataset within noise: Australian 0.29/0.29, Fashion-MNIST
0.32/0.32, Amazon 16.4/16.7, MiniBooNE 2.32/2.40), while trace-*inflating* `additive` is the
outlier (3.66). So **convex ≡ trace-matched additive**; the two regularizers span the *same*
`α·K_emp + β·B` family and differ only by whether the total trace is preserved (convex) or
inflated (additive). Consequence for design: one shared "empirical + base" covariance combiner
(mode ∈ {additive, trace-matched shrinkage}) can serve *both* the EM M-step and the
conditioning base kernel — same math, each family defaulting to its evidence-backed special
case (EM → shrinkage α=0.3; conditioning → additive, empirical weight fixed at 1).

### 13.5 Theory — shrinkage as MAP-EM under an Inverse-Wishart prior, and fixed points

The empirical results above have a clean probabilistic backing that also explains *why*
the unregularized EM fails and why convex behaves better than additive.

**Model.** Per historical dataset `k`, a latent full curve `θ_k ∈ R^M` at the inducing
points, observed through a selection `O_k` with noise:
`θ_k ~ N(μ, Σ)`, `y_k = O_k θ_k + ε_k`, `ε_k ~ N(0, σ²I)`. The E-step yields
`q_k = N(m_k, C_k)`; the plain (ML) M-step is
`Σ_ML = (1/K) Σ_k [ (m_k − μ)(m_k − μ)ᵀ + C_k ]`.

**Convex shrinkage = MAP-EM with an Inverse-Wishart prior (exact).** With `Σ ~ IW(Ψ, ν)`
the conjugate MAP M-step is `Σ_MAP = (S + Ψ) / (K + ν + M + 1)`, where
`S = Σ_k[(m_k−μ)(m_k−μ)ᵀ + C_k]` is the expected scatter (this is the code's IW path,
`(scatter + Psi)/(K + nu + N_obs + 1)`). Setting the scale to
`Ψ = (ν + M + 1) · s · B` with `s = tr(Σ_ML)/tr(B)` gives *exactly*

    Σ_MAP = (1 − α) · Σ_ML + α · s · B,   α = (ν + M + 1) / (K + ν + M + 1),

i.e. `trace_matched_shrinkage`. So the **model assumption** behind `covariance_shrinkage`
is: *a priori `Σ` concentrates around a scaled version of the parametric base-kernel gram
`B = K(Z, Z)`* — "the covariance should look like the smooth parametric kernel" — with `α`
(equivalently the IW dof `ν`) as the prior strength.

Two caveats make it *empirical-Bayes / penalized-likelihood* rather than strict fixed-prior
Bayes: (i) fixing `α` as a constant (rather than `ν`) makes the effective pseudo-count
`ν + M + 1 = αK/(1−α)` **scale with `K`**; (ii) trace-matching sets the prior *scale* `s`
from the data each iteration. The fixed-`Ψ` IW mode (no trace-match, fix `ν`) is the
bona-fide fixed-prior MAP.

**Additive `Σ ← Σ_ML + βB` is a ridge heuristic**, not a conjugate-prior MAP. (The
sum-of-GPs generative reading `θ_k = a_k + b_k`, `a_k ~ N(μ,Σ)`, `b_k ~ N(0, βB)` implies
`θ_k ~ N(μ, Σ+βB)`, but its correct M-step would *deconvolve* the fixed `b`-component, not
simply add `βB`.)

**Fixed points.**
- **Fixed-penalty IW-MAP** (fixed `Ψ, ν`): exact MAP-EM ⇒ monotonically increases the
  penalized log-likelihood `ℓ(Y;μ,Σ) + log p(Σ)`, which is bounded above (the IW log-density
  → −∞ as `Σ` → singular or `‖Σ‖ → ∞`) ⇒ converges to a stationary point, same guarantee as
  ordinary EM.
- **Trace-matched convex** (the `covariance_shrinkage` knob): the target rescales each
  iteration, so there is no single fixed objective to ascend — but the update map
  `T = shrink ∘ M ∘ E` is continuous and, being **trace-preserving**, maps the compact convex
  set `{Σ ⪰ 0 : tr Σ ≤ c}` into itself, so a fixed point **exists** (Brouwer); it converges
  in practice. In directions unconstrained by any observed data it contracts to a **unique
  bounded** value pinned to `B`, whereas plain ML (`α=0`) has a *continuum* of stationary
  points there — the unidentified tail is arbitrary/overconfident, which is why the "none"
  arm was catastrophic (§13.4, NLL 44).
- **Additive** `Σ + βB`: no MAP interpretation and **not trace-bounded** ⇒ it can **diverge**.
  A direction observed by no curve has `Σ_ML,uu = Σ_old,uu`, so `Σ_new,uu = Σ_old,uu + βB_uu`
  grows by `βB_uu` every iteration → ∞. A finite fixed point is only guaranteed when every
  inducing direction is constrained by at least one (e.g. fully-observed) curve — which holds
  in the tutorial (the 20% complete curves cover all epochs), so it converged here, but the
  guarantee is strictly weaker than convex.

**Takeaway.** Convex/IW shrinkage has *both* a clean probabilistic derivation (MAP under an
`IW(∝B, ν)` prior; empirical-Bayes when trace-matched) *and* the strongest fixed-point
guarantee (unique, bounded, pinning the data-starved tail to the base). This is theory-side
support for keeping it as the EM default — complementing the empirical result — while additive
is a heuristic that only stays finite when the data constrains every direction. (Orthogonal to
the *conditioning-time* choice, where additive wins because the base is fit to the target
rather than regularizing an EM estimate — §13.2.)

### 13.6 Conditioning-time additive base: full K_base(X,X) vs Nyström-limited (Option A)

**Issue (found via diff review).** The optional `base_covar_module` was added to the EM
covariance *only at the inducing points* (`Σ_eff = Σ_Z + K_base(Z,Z)`) and then carried to
query points through the same Nyström/shift-interpolation map `W = K(X,Z)K(Z,Z)⁻¹`. So at
query points the base contributed `W·K_base(Z,Z)·Wᵀ` — the rank-≤M Nyström *projection* of the
base kernel — instead of its full `K_base(X,X)`. This under-counts base variance away from the
inducing set (exact only at `X≈Z`, where `W≈I`), yielding over-confident posteriors and
contradicting the "degrades gracefully toward a standard GP" claim. The gap is worst in higher
dimensions (e.g. the 7D Section F), where the inducing set is sparse.

**Fix (Option A, chosen over documenting the limitation).** Interpolate only the (rank-limited,
inducing-defined) EM covariance through `W`, and add the base kernel's *full* covariance directly
at the query points:

    Σ(X,X) = Λ(X) + W·Σ_Z·Wᵀ + K_base(X,X)

Backward-compatible at `X=Z` (`W=I`), consistent with the direct-indexing path, and validated by
a new off-grid unit test (`test_base_covar_module_adds_full_kernel_offgrid`) asserting
`Σ_with_base − Σ_no_base == K_base(X_off, X_off)`.

**Empirical effect: none on the reported experiments.** Re-ran Section F (7D EM GP) and Section G
(finite-pool BO) under Option A: EM-EGP RMSE and NLL are **identical** to the Nyström-limited
version, and every BO regret trajectory (incl. `EM-EGP (fine-tuned)`) is **identical** (only
sub-0.01 RNG wiggles in the non-base Vanilla-GP NLL row). The fitted base `outputscale` stays
small (the EM prior already fits well) and/or the pool is queried on-grid, so `K_base(X,X) ≈
W·K_base(Z,Z)·Wᵀ` here.

**Takeaway.** Option A is a correctness/robustness fix for `base_covar_module` fine-tuning — it
matters when the base contributes materially and queries fall off-grid (sparse-inducing / higher
dimensions) — but it does **not** change any conclusion in §§5–13: the empirical prior remains the
most sample-efficient, frozen ≈ fine-tuned, and kernel-hyperparameter transfer stays weak.

---

## 14. Surrogate diagnostics across all methods **[committed]**

*`bo_diagnose.py` used to cover 4 methods; it now covers all 9 and separates "why does a
method win the loop?" from the loop itself. 6 eval datasets, 200 configs,
`n_pretrain=25`, `meta_iters=2000`. Raw: `results/raw/diagnose.json`.*

### 14.1 Prior-mean informativeness — global ordering vs the extremes

No observations at all. `spearman` = rank correlation of the prior mean with the truth
over the whole pool; `top1_reg` = regret (standardized units) of the **argmax** of the
prior mean, i.e. the zero-shot BO pick; `top10_prec` = overlap of the prior's top-10
with the true top-10.

| method | spearman | top1_reg | top10_prec |
|---|---|---|---|
| **em_frozen** / em_finetuned / em_noisefit | **+0.782** | **0.048** | **0.37** |
| hyperbo_frozen / hyperbo_adapt | +0.771 | 0.176 | 0.13 |
| pacoh_frozen | +0.526 | 0.107 | 0.20 |
| ablr / pretrained_gp_frozen / vanilla_gp | +0.000 | 0.427 | 0.13 |

**This resolves a puzzle in §2.1.** EM and HyperBO have *statistically indistinguishable
global orderings* (0.782 vs 0.771) — yet EM's zero-shot pick is **3.7× better**
(top1 regret 0.048 vs 0.176) and its top-10 precision is **2.8× higher** (0.37 vs 0.13).
So the zero-shot advantage is **not** better global ranking; it is accuracy **in the
extreme region**, which is the only part the argmax and LogEI ever touch.

It also explains the previously-odd observation that PACOH's zero-shot pick beats
HyperBO's: PACOH is far worse globally (0.526) but better at the top (top1 0.107 vs
0.176, top10 0.20 vs 0.13). **Global rank correlation is the wrong diagnostic for
zero-shot BO; extreme-region precision is the right one.** Flat-mean models (vanilla,
pretrained_gp, and ABLR — whose BLR head has a zero prior mean) score exactly 0 and pick
at chance (0.427).

### 14.2 Ranking quality and calibration vs #observations

Predictive-mean Spearman on the *unobserved* pool, mean σ, RMSE, calibration ratio
σ/RMSE (≪1 = overconfident), Gaussian NLL and 95% coverage, after `n_obs` random
observations.

| n_obs | method | rank_corr | calib σ/RMSE | NLL | cov95 |
|---|---|---|---|---|---|
| 5 | em_frozen | +0.699 | 1.06 | 0.03 | 0.95 |
| 5 | hyperbo_frozen | +0.766 | 1.15 | 0.22 | 0.96 |
| 5 | ablr | **+0.788** | 1.40 | 0.16 | 0.98 |
| 5 | vanilla_gp | +0.469 | 1.19 | 0.51 | 0.98 |
| 10 | em_frozen | +0.750 | 0.63 | 0.05 | 0.86 |
| 10 | hyperbo_frozen | **+0.819** | **1.02** | 0.06 | **0.93** |
| 10 | vanilla_gp | +0.630 | 0.58 | 2.19 | 0.78 |
| 20 | em_frozen | +0.790 | 0.27 | 3.89 | 0.63 |
| 20 | em_noisefit | +0.811 | 0.52 | −0.09 | 0.84 |
| 20 | hyperbo_frozen | **+0.834** | 0.83 | 0.10 | **0.89** |
| 50 | em_frozen | +0.825 | **0.15** | **14.96** | **0.43** |
| 50 | em_noisefit | +0.845 | 0.45 | 0.20 | 0.77 |
| 50 | hyperbo_adapt | **+0.874** | 0.45 | 1.02 | 0.77 |
| 50 | ablr | +0.845 | 0.64 | 0.20 | 0.83 |

**A new finding: `em_frozen`'s miscalibration *grows with data*.** Its calibration ratio
collapses 1.06 → 0.63 → 0.27 → **0.15** and coverage 0.95 → 0.86 → 0.63 → **0.43** as
`n_obs` goes 5 → 50, with NLL exploding to **14.96**. The mechanism: the frozen
`COND_NOISE` plus a rank-limited empirical covariance means posterior variance keeps
shrinking as points are added, while RMSE plateaus at ~0.25 because the residual
structure is outside the empirical basis. The model becomes *more* confident exactly
where it cannot improve.

`em_noisefit` largely repairs this (0.45 / cov 0.77 / NLL 0.20 at n_obs=50) — the same
single fitted scalar that rescued EM under injected noise (§5K). **This is the mechanistic
explanation for why `em_noisefit` is the AUC rank-1 method in §5J**, and it strengthens
the §5K recommendation from "use it under observation noise" to "use it by default".

Two more results worth keeping:
- **HyperBO has the best-conditioned posterior throughout** (best rank_corr at every
  n_obs ≥ 10, calibration nearest 1.0). Its loop weakness is purely the prior mean's
  extreme region (§14.1) — the exact complement of EM.
- **`pretrained_gp_frozen` is diagnostically dead** (rank_corr 0.29 at n_obs=10, σ/RMSE
  1.4–2.0): transferring kernel hyperparameters alone conveys almost no usable ranking
  information. This is the cleanest single number behind "kernel transfer does not work".

### 14.3 Trajectory calibration at the recomputed scale

1-step-ahead calibration at acquired points from the 192-run headline (supersedes §11's
128-run numbers):

| method | cov95 | mean NLL |
|---|---|---|
| pacoh_frozen | 1.00 | +1.31 |
| hyperbo_frozen | 0.95 | **−0.08** |
| ablr | 0.94 | −0.17 |
| hyperbo_adapt | 0.85 | +1.09 |
| em_noisefit | 0.85 | +0.84 |
| em_finetuned | 0.85 | +1.02 |
| vanilla_gp | 0.85 | +7.40 |
| em_frozen | **0.64** | **+10.68** |

`em_frozen` is markedly worse here than in the legacy §11 table (0.64/10.68 vs
0.76/1.72) — consistent with §14.2, since the longer 40-iteration horizon accumulates
more observations and drives the frozen-noise pathology further. `em_noisefit` and
`hyperbo_frozen` are the well-calibrated options.

---

## 15. Committed figures

All under `results/figures/`, `.png` (180 dpi) and `.pdf`, regenerated by
`run_all.sh figures`.

| figure | what it shows |
|---|---|
| `lcbench_highpower_trajectories` | **main figure** — mean regret vs evaluations, shaded clustered SEM, 9 methods |
| `lcbench_highpower_cd_auc` | critical-difference diagram on regret-AUC — the discriminating test |
| `lcbench_highpower_cd_final` | CD on final regret — shows the saturated top clique |
| `lcbench_highpower_cd_at{1,5,10,20,40}` | CD at each horizon |
| `lcbench_highpower_perfprofile` | Dolan-Moré performance profile on final regret |
| `lcbench_highpower_dataprofile` | fraction of runs solved to within 5% of the initial gap vs budget |
| `lcbench_highpower_cost` | **cost-normalized Pareto** — regret vs wall-clock, cold-start and amortized panels |
| `*_cost_quality` | **quality vs cost scatter** — regret-AUC against pre-training and against inference cost, one point per method |

**On the cost figure.** Charged cold (all pre-training billed to one task) every
meta-learner is a vertical line at 32–80 s and `vanilla_gp` reaches a usable regret
sooner — the honest answer if you solve exactly one problem. Amortized over 50 tasks the
picture inverts and the Pareto frontier is clean: `em_frozen` reaches regret 0.11 at
≈0.7 s/task, `em_noisefit` 0.09 at ≈1.2 s, `hyperbo_frozen` 0.21 at ≈1.1 s, `ablr` 0.22
at ≈1.7 s, `em_finetuned` 0.14 but at ≈6 s, and `vanilla_gp` needs ≈14 s to reach 0.28.
**EM is simultaneously the most accurate and the cheapest amortized method**; per-step
refits (`em_finetuned`, `vanilla_gp`) cost 5–20× more for worse regret.

Pre-training (192-run headline): em **32 s**, hyperbo 39 s, ablr 76 s, pacoh 80 s.
Online per run: ablr 55 ms, hyperbo_frozen 305 ms, em_frozen 324 ms, pacoh 572 ms,
hyperbo_adapt 1.16 s, em_noisefit 1.21 s, em_finetuned 5.8 s, vanilla_gp 13.7 s.

---

## 16. EM covariance shrinkage in 7D: an in-distribution vs transfer dial **[committed]**

*Motivated by a direct question: §13.3 recommends `covariance_shrinkage=0.3` as the EM
default, but was that ever used in 7D?* **It was not.** The tutorial applies α=0.3 only in
Section B (1D, incomplete curves); Sections F (7D) and G (BO) pre-train with **no
shrinkage** and rely solely on the conditioning-time additive base — and the headline
`em_frozen` arm has `base_covar_module=None`, so it gets neither. `bo_experiment.py`
inherited exactly this. §13.3's recommendation was derived in the 1D incomplete-data
setting and had never been tested in 7D.

Exposed as `--em-shrinkage` (plus `--cond-noise`, `--split-seed`); the EM covariance
eigenspectrum is now reported by `bo_diagnose.py`.

### 16.1 Why it could matter: the prior covariance is extremely anisotropic

EM's M-step covariance is a sum of K rank-1 task deviations plus the E-step posterior
term. With **complete** pre-training data the posterior term nearly vanishes, so the
covariance concentrates in a handful of directions. Measured (200 configs, K=25 tasks):

| α | effective rank | λ_max | median eig | min eig | frac. trace in top 22 |
|---|---|---|---|---|---|
| 0.0 | **2.89** | 130.3 | **1.99e-4** | 1.88e-4 | **0.99978** |
| 0.1 | 3.70 | 125.4 | 8.04e-3 | 1.02e-3 | 0.98190 |
| 0.3 | 5.05 | 115.8 | 1.61e-2 | 1.43e-3 | 0.96242 |

**Effective rank 2.9 out of 200**, with 99.98% of the trace in the top 22 modes and the
bulk directions at variance 2e-4 — *below* the 1e-3 conditioning-noise floor. Because the
BO pool is the same on-grid set the prior was trained on (`X ⊆ Z`, so `W ≈ I` and the
Nyström residual vanishes — corroborated by §13.6), this anisotropy is inherited exactly
at the BO candidates. Shrinkage lifts the bulk ~80× and raises effective rank to 5.1.

### 16.2 Shrinkage fixes calibration completely

`em_frozen` on the unobserved pool after 50 observations:

| α | σ/RMSE | NLL | 95% cov | RMSE |
|---|---|---|---|---|
| 0.0 | 0.15 | **14.97** | **0.43** | 0.251 |
| 0.1 | 0.65 | −0.07 | 0.88 | 0.228 |
| 0.3 | **0.91** | **−0.16** | **0.94** | **0.226** |

Coverage 0.43 → 0.94, NLL improves ~15 nats, and RMSE gets slightly *better*. At α=0.3
`em_frozen` is better calibrated than `em_noisefit` at α=0 — shrinkage **subsumes** the
noise-fitting fix (§5K, §14.2).

### 16.3 …and yet it makes in-distribution BO significantly WORSE

*8 eval × 8 seeds = 64 runs, 300 configs, 40 iters, `--split-seed 7` (a dev split
disjoint from the §5J test datasets). Paired per-run differences vs α=0; **positive =
shrinkage worse**. `hyperbo_frozen`/`vanilla_gp`/`random` come out bit-identical across α,
confirming the knob is EM-only and the pairing is valid.*

| method | @5 | @10 | AUC | final |
|---|---|---|---|---|
| em_frozen α=0.1 | +0.29±0.19 | +0.23±0.12 (z=2.0) | −0.01 (null) | −0.07 |
| em_frozen α=0.3 | +0.58±0.29 (z=2.0) | **+0.62±0.23 (z=2.6), 3W/19L** | +0.23±0.16 | +0.01 (null) |
| em_noisefit α=0.3 | +0.62±0.27 (z=2.3) | **+0.70±0.22 (z=3.2), 2W/20L** | +0.31±0.14 (z=2.2) | +0.11 |

**Fixing the calibration makes the optimizer worse.** The tight off-span variance is not a
defect in-distribution — it is an **inductive bias** that confines LogEI to the
low-dimensional subspace the pre-training tasks span, which is exactly where the good
configs live. Inflating it sends the acquisition exploring ~197 directions of noise.

### 16.4 OOD: the sign flips

*OOD split (§5H clustering; cross-cluster correlation 0.224 vs ~0.50 within), 6 eval × 10
seeds = 60 runs. Same paired convention — **negative = shrinkage better**.*

| method | @5 | @10 | AUC | final |
|---|---|---|---|---|
| em_frozen α=0.1 | **−1.05±0.52 (z=−2.0)** | −0.19±0.24 | −0.10±0.19 | +0.13 |
| em_frozen α=0.3 | **−1.33±0.53 (z=−2.5), 25W/7L** | −0.39±0.28, 26W/8L | −0.29±0.25 | +0.19 |
| em_noisefit α=0.3 | −0.18±0.11 | +0.14±0.18 | +0.26±0.15 | **+0.55±0.22 (z=2.5)** |
| em_finetuned α=0.3 | −0.24±0.17 | +0.06±0.17 | +0.29±0.15 (z=2.0) | **+0.64±0.20 (z=3.2)** |

Absolute regret-AUC (lower = better):

| method | α=0.0 | α=0.1 | α=0.3 |
|---|---|---|---|
| em_noisefit | **0.826** | 1.050 | 1.090 |
| em_finetuned | 0.849 | 1.192 | 1.141 |
| em_frozen | 1.471 | 1.372 | **1.178** |
| vanilla_gp | 1.388 | 1.388 | 1.388 |
| hyperbo_frozen | 2.085 | 2.085 | 2.085 |

95% coverage at acquired points, `em_frozen`: 0.56 → 0.91 → 0.95.

**Two findings.**
1. **For `em_frozen` the effect reverses sign**: shrinkage is *significantly better* early
   OOD (@5, z=−2.5, 25W/7L) and improves AUC monotonically (1.471 → 1.178), lifting it
   from **worse than `vanilla_gp`** — reproducing §5H's embarrassing result — to better.
   Same knob, same code, opposite conclusion depending on distribution shift.
2. **Shrinkage and target adaptation are substitutes, not complements.** For
   `em_noisefit` and `em_finetuned`, which already correct using the target's own data,
   adding shrinkage *over*-corrects and significantly hurts final regret (z=2.5, z=3.2).
   Neither ever beats its own α=0 setting on AUC.

### 16.5 Conclusions

- **Keep α=0 for in-distribution 7D BO.** The historical default was right; §13.3's α=0.3
  is correct for its own setting (1D incomplete-curve imputation, where the E-step must
  impute) and wrong here. The notebook's split — shrinkage in Section B, none in F/G —
  is correct.
- **Use shrinkage when you cannot adapt to the target**: a frozen prior under
  distribution shift. It is the only full-rank correction that needs no target data,
  since the additive base fits its coefficient on target observations (§13.1 showed that
  fails on a sparse head). When target adaptation *is* available, it dominates and
  shrinkage becomes harmful.
- **α is therefore a principled in-distribution-vs-transfer dial**, not a tuning nuisance:
  it trades "trust the meta-learned subspace" against "hedge outside it".
- **Calibration is not a proxy for BO utility.** §16.2 vs §16.3 is a direct counterexample
  *within a single method*: NLL improves by 15 nats and coverage more than doubles while
  BO regret significantly degrades. Read §11 and §14.3's calibration tables as diagnostics
  of the posterior, **not** as a ranking of optimization quality.
- Consequently shrinkage **drops out of the hyperparameter-tuning protocol** for the
  in-distribution headline (α=0 confirmed), and should instead be reported as part of the
  OOD story.

*Caveat: the in-distribution and OOD sweeps use different scales (64 vs 60 runs, 40 vs 30
iterations, 300 vs 200 configs), so compare signs and paired z-scores across the two, not
absolute regrets.*

---

## 17. PD1, faithfully: the full release overturns the negative **[committed]**

§9–§12 concluded that transfer does not beat random on PD1, and §12.4/§12.5 localized
that to the **data**: PD1Lite yields ~50 matched configs across all tasks (73 within the
bs=256 subgroup) against the paper's ~500, and we pre-trained HyperBO ~25× less than the
paper. Both constraints are now removed.

**Data.** The full public PD1 release (Wang et al. 2021, CC-BY 4.0) read by the new
`pd1_full_data.py` (`--pd1-source full`). It shares an **identical 400-point Halton grid
across all 23 evaluable tasks** — 8× PD1Lite's matched pool — and the grid exactly fills
the documented 4-D search space. Objective `best_valid/error_rate` (what the paper
optimizes); the 1348 matched rows lacking an evaluation are all `status="diverged"` and
are assigned error 1.0 rather than dropped, since discarding them would delete the very
divergence tail that makes PD1 hard (§9.3). Phase-0/phase-1 duplicates (2277 pairs)
resolve to phase 1.

**Protocol.** Leave-one-out over all 23 tasks (no batch-size subgroup needed now that the
all-task intersection is 400), 5 seeds/fold = **115 runs**, 50 BO iterations on the
400-config pool (12.5% of it, so nowhere near the saturation that made PD1Lite
uninformative), per-task standardization, `n_init=3`. **Per-method pre-training budgets
are decoupled** (`--hyperbo-iters/--pacoh-iters/--ablr-iters`) so no baseline is forced
onto another's budget: HyperBO **50 000** steps as the paper specifies, PACOH and ABLR
10 000. Cost per fold: HyperBO 763 s, PACOH 453 s, ABLR 259 s, EM 52 s.

### 17.1 Headline — the paper's `−log(error)` warp

Simple regret ± clustered SEM, 115 runs, 9 methods, Nemenyi **CD = 1.12**.

| method | @1 | @5 | @10 | @25 | @50 | AUC | rank(AUC) |
|---|---|---|---|---|---|---|---|
| **ablr** | **0.099±0.011** | 0.054±0.007 | 0.037±0.006 | 0.018±0.005 | 0.005±0.002 | **0.033** | **3.69** |
| em_noisefit | 0.145±0.033 | 0.062±0.014 | 0.037±0.008 | 0.016±0.004 | 0.008±0.003 | 0.035 | 3.76 |
| em_finetuned | 0.146±0.032 | 0.054±0.009 | **0.033±0.007** | **0.014±0.004** | 0.009±0.003 | 0.033 | 3.88 |
| hyperbo_adapt | 0.127±0.029 | 0.072±0.018 | 0.042±0.010 | 0.020±0.006 | 0.007±0.002 | 0.038 | 4.03 |
| hyperbo_frozen | 0.128±0.028 | 0.073±0.017 | 0.048±0.011 | 0.020±0.006 | 0.008±0.003 | 0.040 | 4.18 |
| em_frozen | 0.158±0.033 | 0.082±0.021 | 0.057±0.014 | 0.027±0.007 | 0.011±0.003 | 0.045 | 4.40 |
| vanilla_gp | 0.384±0.067 | 0.126±0.021 | 0.067±0.010 | 0.031±0.006 | 0.012±0.003 | 0.062 | 6.20 |
| pacoh_frozen | 0.282±0.050 | 0.140±0.025 | 0.070±0.014 | 0.036±0.008 | 0.016±0.004 | 0.065 | 6.80 |
| random | 0.330±0.050 | 0.156±0.018 | 0.115±0.015 | 0.075±0.012 | 0.055±0.010 | 0.102 | **8.07** |

**Friedman χ²(8) = 315.0, p ≈ 3e-63 on regret-AUC.** The tied-best clique is
**{ablr, em_noisefit, em_finetuned, hyperbo_adapt, hyperbo_frozen, em_frozen}** — *all six
transfer methods* — and every one of them is significantly ahead of `vanilla_gp` (6.20),
`pacoh_frozen` (6.80) and `random` (8.07). The gap from the best transfer method to
random is **4.4 rank units against a CD of 1.12**.

**This overturns §9/§9.1/§9.2/§12.** On PD1Lite `random` ranked *1st–3rd* and no transfer
method beat it; on the full release `random` is **last by a wide margin** and transfer is
3× better on regret-AUC (0.033 vs 0.102) and 2–4× better on the first pick. §12.5's
diagnosis was correct: **the PD1 null was a data artifact of PD1Lite, not a property of
the methods or of PD1.**

### 17.2 Output-transform ablation, re-run at proper scale

Regret-AUC average rank under the paper's `−log` warp vs no transform:

| method | neglog | none |
|---|---|---|
| ablr | **3.69** | 3.93 |
| em_noisefit | 3.76 | **3.53** |
| em_finetuned | 3.88 | 3.82 |
| hyperbo_adapt | 4.03 | 4.18 |
| hyperbo_frozen | 4.18 | 4.13 |
| em_frozen | 4.40 | 4.10 |
| vanilla_gp | 6.20 | 5.92 |
| pacoh_frozen | 6.80 | 7.58 |
| random | **8.07** | 7.82 |

The transfer-beats-everything-else conclusion is **robust to the transform** — both arms
put all six transfer methods in the top clique and random last (χ²=315 and χ²=343).

The warp does shift the *internal* ordering in the direction the paper implies: under
`neglog`, HyperBO improves relative to EM (4.03/4.18 vs EM-frozen 4.40), while un-warped
EM edges ahead (em_frozen 4.10 vs hyperbo 4.13/4.18). **This is the opposite of §12.2/§12.5
on PD1Lite**, where the warp *erased* HyperBO's edge — confirming that explanation: the
`−log` warp amplifies resolution among good configs, which only pays off once the pool is
dense enough to model the resulting peaked objective. At 50–73 configs it backfired; at
400 it behaves as designed.

### 17.3 Calibration

95% coverage at acquired points:

| method | neglog | none |
|---|---|---|
| hyperbo_adapt | 0.94 | 0.95 |
| hyperbo_frozen | 0.93 | 0.94 |
| ablr | 0.91 | 0.94 |
| em_finetuned | 0.86 | 0.91 |
| em_noisefit | 0.84 | 0.90 |
| vanilla_gp | 0.85 | 0.89 |
| pacoh_frozen | 0.98 | 0.99 |
| **em_frozen** | **0.45** | **0.57** |

Every method is now well calibrated on PD1 (0.84–0.99), versus the 0.16–0.33 collapse
reported in §9.2/§9.4 on PD1Lite — so **that miscalibration was also a small-pool
artifact**, not PD1's bimodality per se. The one exception is `em_frozen` (0.45), which is
the §16 rank-deficiency pathology, not a PD1 effect; `em_noisefit`/`em_finetuned` fix it.

### 17.4 Cost: two different axes, two different winners

Leave-one-out re-trains a prior per fold, so pre-training is a **per-task** cost here
(total / 23), not a one-off. Budgets are each method's own converged setting.

| method | pre-train / task | × EM | online ms / BO run | regret-AUC |
|---|---|---|---|---|
| **EM** (`em_*`) | **52 s** | 1.0× | 763 (frozen) – 8087 (finetuned) | 0.033–0.045 |
| ABLR | 259 s | 4.9× | **66** | **0.033** |
| PACOH | 453 s | 8.6× | 1317 | 0.065 |
| HyperBO | **763 s** | **14.6×** | 438 | 0.038–0.040 |
| vanilla_gp | 0 | — | 7870 | 0.062 |

**There is no single cost winner — the two axes have different answers.**

- **Meta-training axis: EM dominates outright.** `em_finetuned`/`em_noisefit` reach the
  best regret-AUC (0.033–0.035) at **1/15th of HyperBO's pre-training** and 1/5th of
  ABLR's. Nothing else is simultaneously best-in-quality and cheapest-to-train.
- **Inference axis: ABLR dominates outright.** 66 ms/run — **6.6× cheaper than HyperBO,
  12× cheaper than frozen EM, 120× cheaper than a from-scratch GP** — at tied-best
  quality. Its analytic Bayesian-linear head costs O(D³) in a *fixed* basis dimension,
  independent of the number of observations, where every GP here pays O(n³).
- PACOH is the worst of both worlds: 8.6× EM's pre-training, 20× its inference, worst
  quality of the transfer methods.

The same pattern holds on LCBench (§15): ABLR is again the cheapest GP method online
(55 ms/run vs HyperBO 305 ms, `em_frozen` 324 ms, `vanilla_gp` 13.7 s).

**For the paper:** "matches specialized deep-kernel meta-BO at 1/15th the pre-training
cost" is the defensible EM claim, and it is stronger than a quality claim because EM only
*ties* on quality here. Report both axes — a single "total cost" number hides the
ABLR inference result entirely.

Figure: `results/figures/pd1full_neglog_cost_quality.pdf`.

### 17.5 Conclusions

1. **PD1 now supports the paper's claim and ours.** Meta-learned priors beat both a
   from-scratch GP and random search on PD1 by a large, highly significant margin. The
   earlier negative was PD1Lite.
2. **EM transfers on PD1.** `em_noisefit` (3.76) and `em_finetuned` (3.88) sit inside the
   top clique, statistically indistinguishable from ABLR and HyperBO. This retires §9.1's
   "the empirical mean does not transfer on PD1" — that was the 50-config pool talking.
3. **ABLR is the strongest single method on PD1** (AUC rank 3.69) as it was competitive on
   LCBench, making it the most consistent baseline across both benchmarks.
4. **PACOH is weak on PD1 too** (6.80, barely ahead of random) — the only method whose
   ranking is unchanged from PD1Lite, and consistent with LCBench. Its weakness is a
   genuine property.
5. **LCBench and PD1 now agree**: {EM variants, HyperBO, ABLR} ≫ vanilla_gp > PACOH >
   random. The two benchmarks tell the same story once PD1 is given its real data.

*Sections §9–§12 are retained as the record of the PD1Lite investigation and of how the
artifact was diagnosed, but **§17 supersedes their conclusions**.*

---

## 18. Why is ABLR so strong here, when Wang et al. outperformed it? **[analysis]**

ABLR is the best-ranked method on PD1 (§17) and top-clique on LCBench (§5J), which is
surprising: the HyperBO paper compares against ABLR and **beats it**. Reading their
source (`exp-setup.tex`, `exp-hold-out-related.tex`, `exp_hpob_normalized.tex`) resolves
most of the discrepancy, and one part of it is a caution about *our* setup rather than a
result about ABLR.

**What they did.** Their ABLR is *"a special case of HyperBO with **zero mean** and linear
kernel `k(x,x') = b² + φ(x)ᵀφ(x')/s²`, where φ is the **feature layer of a 2-hidden-layer
network of size (32,32)**"*, optimized like H-NLL. Their diagnosis of its weakness is
explicit and twofold:

> "ABLR used a **much more constrained kernel function** than other HyperBO variants,
> resulting in worse performance."

> "…**overly confident predictions can lead to less exploration** in BO and as a result,
> poor BO performance. **ABLR also uses Bayesian linear regression and likely suffered
> from the same lack of exploration issue.**"

**Why ours does not hit that failure mode.**

1. **Their diagnosed failure is over-confidence; ours is measurably not over-confident.**
   We logged 95% coverage at acquired points: ABLR **0.94** on LCBench (NLL −0.17) and
   0.91–0.94 on PD1, with σ/RMSE 1.09–1.40 in §14.2 — if anything slightly *under*
   confident. We learn the prior and noise precisions (α, β) by maximizing the summed
   per-task evidence, which directly calibrates predictive variance.
2. **Our basis is less constrained** — a dedicated 50-dimensional tanh output layer,
   versus their use of a 32-d *hidden* layer as φ. That speaks straight to their
   "much more constrained kernel" criticism.
3. **Acquisition.** We use LogEI; they use thresholded PI. Under-exploration is punished
   harder by a greedier acquisition, so the same over-confidence costs them more.
4. **Data regime favours ABLR.** Their own scaling study finds *"H-NLL and ABLR had the
   most amount of improvements when increasing the number of datapoints per task from 10
   to 100"*. We give it 400 configs × 22 tasks ≈ 8.8k points — the regime where they
   report ABLR gains most.

**The caution, stated plainly.** Their best variant is **H-EKL**, and §5C found EKL
*hurts* in our 7D setup, so we run HyperBO under NLL. **ABLR beating our HyperBO does not
establish ABLR > HyperBO in general — it may partly mean our HyperBO is under-powered.**
Two concrete reasons to suspect this: (i) until §17 we trained HyperBO at 2 000 steps
against the paper's 50 000; (ii) we never reproduced their H-EKL configuration, which is
their strongest. We also do not implement their FSBO or MIMO baselines, so our comparison
set is narrower than theirs.

**Consequences.**
- Report ABLR as a **strong, cheap, well-calibrated** baseline — it is, on both benchmarks
  — but do **not** claim it beats HyperBO in general; scope the claim to our
  configuration and note the H-EKL gap.
- ABLR's genuinely robust and novel-to-us finding is the **cost** result (§17.4): tied-best
  quality at 6–120× cheaper inference, because its analytic head is O(D³) in a fixed basis
  rather than O(n³) in the observation count. That is an architectural property, not a
  tuning artifact, and it holds on both benchmarks.
- ABLR has still **never been tuned** here (`feat_dim` fixed at 50; per-task precisions
  only just implemented, matching Perrone et al.). Its numbers may improve further, which
  makes the "our HyperBO may be under-powered" caveat *more* pressing, not less.

---

## 19. Debugging HyperBO: under-training found and fixed; a gap to ABLR remains **[committed]**

§18 flagged that ABLR outranking HyperBO contradicts Wang et al., who report HyperBO
beating ABLR on ~11 of 23 PD1 tasks while ABLR wins 1. That is a large enough
disagreement to be a bug in our setup rather than a result, so it was investigated
systematically. Two hypotheses were pre-registered and tested; a third problem was found
along the way that invalidates part of our earlier statistics.

### 19.1 Localizing the failure

Paired per-run tests put HyperBO significantly behind ABLR on regret-AUC — z=+2.81
(LCBench, 192 runs) and z=+2.20 (PD1, 115 runs). But HyperBO *wins the majority of
head-to-head runs* (41 vs 15 at @10) while losing on the mean, and per-dataset
breakdown shows the whole deficit comes from **2 of 12 eval datasets**:

| dataset | HyperBO AUC | ABLR AUC | ratio |
|---|---|---|---|
| sylvine | 2.415 | 1.037 | 2.3× |
| kr-vs-kp | 3.672 | 2.164 | 1.7× |
| *(other 10)* | — | — | 0.3–1.1× |

Those two are also the hardest tasks (largest residual regret for every method), and
together they more than account for the total gap; on the other ten HyperBO nets ahead.
Since the LCBench prior is trained once and shared, this is not an unlucky fit.

### 19.2 H1 — under-training. **Confirmed, and it was large.**

We used 2 000 Adam steps; the paper uses **50 000**. §4 had already shown @10 regret
improving 7× from 2k to 5k, but that was dismissed because the *final*-regret column was
flat — the criterion §5J later showed is saturated. Budget sweep, 192 runs per point,
all paired, identical data and BO seeds:

| steps | pre-train | HyperBO AUC | paired vs ABLR |
|---|---|---|---|
| 2 000 | 34 s | 0.810 | +0.342 (z=+3.32) worse |
| 5 000 | 85 s | 0.574 | +0.075 (z=+2.75) |
| 10 000 | 170 s | 0.589 | +0.127 (z=+3.05) |
| **25 000** | 421 s | **0.539** | −0.111 (z=−2.09) |
| 50 000 | 837 s | 0.637 | +0.120 (z=+2.08) |

**2 000 steps is far below the knee** (~5k), and it is the budget every earlier result in
this document used. Figure: `results/figures/hyperbo_budget_budget_pareto.pdf`.

### 19.3 H2 — is the learned mean hurting? **Refuted.**

HyperBO uses μ(x)=W·φ(x); ABLR has a zero prior mean and so cannot be misled. Running
FSBO (= HyperBO with `--hyperbo-zero-mean`):

| steps | H-NLL | FSBO | paired |
|---|---|---|---|
| 2 000 | 0.810 | 0.874 | −0.064 (z=−1.04), mean *helps* |
| 50 000 | 0.637 | 0.564 | +0.073 (z=+1.66), mean *hurts* |

Neither is significant and the effect (±0.07) is dwarfed by the budget effect (0.27). At
the low budget the direction matches the paper's H-NLL > FSBO. **The mean is not the
problem.**

### 19.4 A statistics bug this uncovered: one prior, 192 "independent" runs

The neural meta-learners train **one prior** that is then reused by every BO run, so
per-run paired tests treat prior-draw luck as free evidence — the effective sample size
for comparing *priors* is **1**, not 192. EM is unaffected (its pre-training is
deterministic; measured spread exactly 0.000 across arms), as are vanilla_gp and random.

Added `--pretrain-seed`, independent of `--split-seed`, and re-ran with **5 independent
priors** (HyperBO @25k, ABLR @10k, everything else identical):

| prior seed | HyperBO | ABLR | winner |
|---|---|---|---|
| 0 | 0.589 | 0.459 | ablr |
| 1 | 0.646 | 0.522 | ablr |
| 2 | 0.602 | 0.412 | ablr |
| 3 | 0.551 | 0.496 | ablr |
| 4 | 0.631 | 0.527 | ablr |
| **mean ± sd** | **0.604 ± 0.037** | **0.483 ± 0.048** | **ablr 5/5** |

Paired over priors: **+0.121 ± 0.022, t = +5.53 (df=4), significant.**

So the gap is **not** prior-draw luck: ABLR robustly beats our HyperBO. (An earlier
apparent 0.188 ABLR spread was an artifact of the budget sweep — varying HyperBO's
budget shifted the shared RNG state that ABLR's pre-training subsequently consumed.
`--pretrain-seed` removes that confound; true prior-draw sd is 0.048.)

### 19.5 Where this leaves the comparison

- **Under-training was real and material.** HyperBO's regret-AUC improves 33 % from 2k to
  25k steps (0.810 → 0.539). **Every HyperBO number in §2–§18 was produced at 2 000
  steps and therefore under-sells it.** On the full 9-method headline at converged
  budgets, HyperBO improves 0.810 → 0.637 (z=−1.8) and lands statistically **tied** with
  both ABLR (z=+1.72) and EM (z=−0.04).
- **But under-training does not explain everything.** Even at its best budget with 5
  independent priors, HyperBO loses to ABLR 5/5. The paper's ordering is still not
  reproduced, so at least one implementation or protocol difference remains.
- **The leading remaining candidate is H-EKL**, the paper's strongest variant, which we
  have never run: our EKL was (i) hardcoded off in both PD1 drivers and (ii) optimized
  with Adam, where Wang et al. use **L-BFGS for 100 iterations** — they note explicitly
  that the low-rank-Matérn problem motivating Adam for NLL "did not seem to occur for
  EKL". Both are now fixed (`--ekl-optimizer lbfgs`) but unrun.
- **Consequence for EM's claims.** At converged budgets EM, HyperBO and ABLR are
  statistically indistinguishable on LCBench quality. EM's defensible advantage becomes
  **cost**: 29 s of pre-training vs HyperBO's 959 s (**33×**) for tied quality — which is
  the framing §17.4 already recommends for PD1.
- **Methodological rule going forward:** report neural meta-learner results over
  **multiple pre-training seeds**, and treat the prior as the unit of analysis. Per-run
  error bars on a single prior are not evidence about the method.

### 19.6 H-EKL with the paper's L-BFGS optimizer: tested, and it is not the answer

The last untested lead from §19.5. Both defects were fixed -- EKL was hardcoded off in
the PD1 drivers, and was being optimized with Adam where Wang et al. use **L-BFGS for
100 iterations** -- and then run on both benchmarks.

**LCBench** (5 independent pre-training priors, regret-AUC):

| variant | AUC | pre-train |
|---|---|---|
| H-NLL @25k (Adam) | **0.604 ± 0.037** | 409 s |
| H-EKL (L-BFGS ×100) | 1.688 ± 0.451 | **6.5 s** |
| ABLR | 0.494 ± 0.057 | 76 s |

H-EKL is ~3× worse and an order of magnitude more variable (paired over priors
t=+5.56, df=4). **This vindicates §5C**: "NLL beats EKL in 7D" was a real finding, not an
artifact of the wrong optimizer.

**PD1** (full release, 23-task LOO, 115 runs, paired on folds and seeds):

| | H-NLL @50k | H-EKL / L-BFGS |
|---|---|---|
| hyperbo_frozen AUC | **0.0403** | 0.0714 |
| hyperbo_adapt AUC | **0.0383** | 0.0795 |
| HyperBO pre-train (23 folds) | 17 555 s | **162 s** |

EKL − NLL = +0.031 ± 0.004 (z=+7.48) for frozen and +0.041 ± 0.006 (z=+6.87) for adapt;
against ABLR, H-EKL is worse by +0.039 ± 0.005 (z=+7.28). **H-EKL loses decisively on
PD1 as well**, which is precisely where the paper reports it winning.

**Where this leaves the HyperBO question.** Every lead we could identify has now been
tested: the budget (H1, confirmed and fixed — worth 33 % of regret-AUC), the learned mean
(H2, refuted), prior-draw luck (§19.4, refuted at 5/5 priors), and the EKL objective with
its correct optimizer (refuted on both benchmarks). Our architecture matches their
published spec — tanh MLP, linear mean `W·φ(x)`, ARD Matérn-5/2 on the feature layer,
Adam @ 1e-3 — so the remaining explanations are (i) a subtler implementation difference
we have not isolated, (ii) their acquisition (thresholded PI) versus our LogEI, or
(iii) a genuine difference in what the two evaluation protocols reward.

**Honest reporting position:** state that our HyperBO is competitive but does not
reproduce the paper's margin over ABLR, list what was ruled out, and avoid claiming a
general ordering between HyperBO and ABLR from our numbers alone. The one unambiguous
EKL finding is a **cost** result: at 6.5 s (LCBench) and 162 s across 23 PD1 folds it
pre-trains **63–109× cheaper** than NLL, so it is the right choice when meta-training
budget dominates and BO quality is secondary.


---

## 20. Definitive re-run at corrected budgets **[committed]**

Every study contaminated by the §19 bugs was re-run overnight: 42 jobs at **HyperBO
25 000 / PACOH 10 000 / ABLR 10 000 steps with ABLR's faithful per-task precisions**,
written to `results/raw/rerun/` so the contaminated originals stay auditable alongside.
Driver: `results/run_rerun_all.sh`. Zero failures.

### 20.1 The headline, corrected — and it changes the ranking

3 independent pre-training priors × 192 runs. `sd(prior)` is the spread across priors,
the unit of analysis §19.4 established as the correct one.

| method | AUC | sd(prior) | @5 | @10 | old (2k) | change |
|---|---|---|---|---|---|---|
| **ablr** | **0.532** | 0.016 | **0.587** | **0.396** | 0.619 | **−14 %** |
| hyperbo_adapt | 0.599 | 0.027 | 0.948 | 0.617 | 0.820 | **−27 %** |
| ablr_adapt | 0.611 | 0.042 | 0.881 | 0.459 | — | new |
| hyperbo_frozen | 0.612 | 0.030 | 0.976 | 0.626 | 0.810 | **−24 %** |
| em_frozen | 0.639 | 0.000 | 1.322 | 0.689 | 0.639 | 0 % |
| em_noisefit | 0.671 | 0.000 | 1.342 | 0.824 | 0.669 | 0 % |
| em_finetuned | 0.786 | 0.000 | 1.386 | 0.923 | 0.744 | +6 % |
| vanilla_gp | 1.250 | 0.000 | 2.370 | 1.482 | 1.276 | −2 % |
| pacoh_frozen | 1.467 | 0.248 | 2.601 | 2.074 | 1.397 | +5 % |
| random | 1.926 | 0.000 | 3.134 | 1.965 | 1.926 | 0 % |

Paired over the 3 priors (df=2, so |t| > 4.30 for significance):

| comparison | Δ AUC | t | verdict |
|---|---|---|---|
| ablr − em_frozen | −0.107 | −11.93 | **ABLR significantly better** |
| ablr − em_noisefit | −0.139 | −15.48 | **ABLR significantly better** |
| ablr − hyperbo_frozen | −0.080 | −3.50 | tied |
| hyperbo_frozen − em_frozen | −0.027 | −1.57 | tied |
| ablr_adapt − ablr | +0.079 | +2.75 | tied (per-target precision refit does not help) |

**This overturns §5J's ranking.** With the baselines properly trained, the meta-learners
gain 24–27 % while EM is unchanged (its pre-training is deterministic — `sd(prior)` is
exactly 0.000), and the order becomes **ABLR ≥ HyperBO ≥ EM** rather than EM first. ABLR
now *significantly* beats every EM variant; HyperBO and EM are statistically tied.

**EM's LCBench claim must therefore be restated.** It is no longer "best BO quality". The
defensible claims are (i) **cost** — 29 s of pre-training vs HyperBO's 409–959 s, for
tied quality; (ii) **determinism** — zero prior-draw variance where HyperBO's is 0.030
and PACOH's 0.248; and (iii) the calibration and zero-shot properties of §14.

*Caveat: 3 priors gives df=2 and low power, so "tied" partly reflects that. The
ABLR-vs-EM comparisons are significant because EM contributes no prior variance.*

### 20.2 Which earlier conclusions survived

| study | conclusion | status |
|---|---|---|
| §5L batch BO | fantasy helps meta-priors, hurts from-scratch GPs | **survives, strengthened** — em −0.018, hyperbo −0.074, ablr −0.104; vanilla **+0.268** |
| §16 shrinkage | hurts in-distribution, helps OOD for `em_frozen` | **directions survive** (in-dist +0.206/+0.275; OOD `em_frozen` −0.293); AUC-level significance weaker in the smaller re-run, original claims were at @5/@10 |
| §5C EKL vs NLL in 7D | NLL beats EKL | **survives** — and is now stronger, since §19.6 confirms it with EKL's correct L-BFGS optimizer |
| §5F meta-data scaling | HyperBO is the data-hungry one | **does not survive** — at n_pretrain 10 and 15 HyperBO is now the *best* method (0.728, 0.542). Its apparent data-hunger was partly under-training. Curve is noisy (one prior per point) |
| §14 diagnostics | EM's prior mean wins the extreme region | re-run committed as `rerun/diagnose.json`; EM's numbers are unchanged by construction |
| §17 PD1 | transfer ≫ random; all six transfer methods tied | **unaffected** — already ran at 50k |

### 20.3 Provenance

`raw/rerun/` holds all 42 corrected outputs; `raw/*.json` holds the contaminated
originals. Anything quoted in the paper should come from `raw/rerun/`, `raw/pd1full/`, or
the 50k headline — never from the 2 000-step files.

---

## 21. EM levers, and why the published ordering cannot be reproduced **[committed]**

### 21.1 A silent no-op: the additive-kernel variants were never additive

`em_additive_freshbase` and `em_additive_3way` produced **bit-identical trajectories to
`em_noisefit`** across all 744 points. Root cause: both variants installed their kernel by
overriding `_effective_Sigma_inducing`, which is only consulted by
`_get_prior_at_indices`. With `enable_interpolation=True` — which every run uses — the live
path is `_interpolate_prior_to_X`, so **the override was never called** and the MLL fit
could only move the likelihood noise, collapsing both variants onto `em_noisefit`.

This invalidates every §13/§5I conclusion that compared "additive vs convex vs 3-way"
through these harness variants. `em_finetuned` was unaffected — it uses the library's
supported `base_covar_module`/`_additive_base` hook, which *is* honoured on both paths.

**Fix.** The variants now set `_additive_base` (an `AdditiveKernel` for the 3-way case),
and the library gained `_log_sigma_scale`, read and exponentiated on every forward, so the
empirical block's magnitude `exp(s)·Σ_emp + K_base` is learnable on both code paths rather
than only the on-grid one.

**Result once it actually runs** (4 datasets × 6 seeds, corrected budgets; variants are now
provably distinct):

| method | regret-AUC | @10 |
|---|---|---|
| ablr | **0.517** | 0.110 |
| **em_finetuned** | **0.573** | 0.131 |
| em_noisefit | 0.577 | 0.131 |
| em_frozen | 0.616 | 0.334 |
| em_additive_freshbase | 0.677 | 0.486 |
| em_additive_3way | 0.796 | 1.188 |

So the lever is real but **the extra free empirical scale hurts**: `em_finetuned` — the
library's plain `base_covar_module`, and exactly the configuration the paper's 7D results
use — is the best EM variant. The research variants that add `s1` on top over-parameterize
a small target design. **EM's remaining levers are therefore already pulled**: shrinkage
(§16, α=0 correct in-distribution), per-task noise (`em_noisefit`), and the additive base
(`em_finetuned`). The untried ones are `coordinate_ascent_em` (joint EM + kernel fitting,
never wired into the harness) and the pre-training likelihood noise (fixed at 1e-2).

### 21.2 Why the published ordering is unreproducible: we are solving a different problem

Comparing our absolute PD1 best-validation-error against the paper's Table (both in %):

| task | our pool oracle | our EM | our HyperBO | our ABLR | paper's best |
|---|---|---|---|---|---|
| cifar100_wrn | 20.86 | 20.86 | 20.86 | 20.86 | **20.48** |
| imagenet_resnet50 | 22.99 | 22.99 | 22.99 | 22.99 | **22.53** |
| lm1b_transformer | 62.08 | 62.11 | 62.11 | 62.08 | **61.81** |
| uniref50_transformer | 78.95 | 78.95 | 78.95 | 78.95 | **78.72** |
| wmt15_de_en | 34.11 | 34.11 | 34.11 | 34.11 | **33.91** |
| svhn_wrn | 3.96 | 3.96 | 4.07 | 4.07 | **3.89** |

Two things are visible at once:

1. **Every method reaches our pool's oracle.** With 50 iterations over 400 configs the
   finite pool is exhausted, so all methods return the identical best configuration and
   final regret carries no signal — the ordering is decided by early-iteration noise.
2. **The paper achieves a LOWER error than the best point in our pool, on every task.**
   That is only possible if they are not searching our candidate set.

The data confirms it: each PD1 task holds **~1 959 configurations**, of which only **400**
are shared across tasks. The paper's baselines (HyperBO, ABLR, FSBO, MIMO) need no matched
inputs, so they search each task's **full ~5× larger space**; we restricted the pool to the
matched intersection because the *empirical* prior needs shared inducing points.

**This, not a modelling bug, is the primary reason our ordering differs.** Our protocol is
strictly easier and saturates. It also explains §19's puzzle — no amount of HyperBO
tuning can separate methods on a problem every method solves exactly.

**The fix, and it plays to EM's strengths.** The EM prior is defined on inducing points
`Z` and interpolated to arbitrary `X` (`enable_interpolation=True`), so EM *can* score
off-grid candidates. The faithful protocol is therefore: **pre-train the empirical prior on
the 400 matched configs, then run BO over each task's full ~1 959-config pool.** That
matches the paper's search space, removes the saturation, and exercises the
continuous-domain interpolation that is itself a contribution of the paper. This is the
single highest-value experiment remaining and is not yet implemented.

---

## 22. Regret-AUC is the wrong headline metric **[committed]**

Every ranking in §5J/§17/§20 is driven by regret-AUC. AUC integrates the whole
trajectory, but the operational question is **"how many evaluations until I reach a
good-enough configuration"** — and a method that explores early then jumps to the
optimum can lose on AUC while reaching every target sooner than a steadily-improving
competitor.

*(Worth stating precisely, since it is easy to conflate: our trajectories are
**best-so-far**, so simple regret is monotone and a bad exploratory trial costs nothing
directly. AUC penalizes slowness, not exploration. The bias is subtler — AUC is dominated
by the long flat tail after a method has already converged.)*

**Budget-to-target.** For each run, the first evaluation at which regret falls below
`tol x` that run's own initial gap (scale-free across tasks). Runs that never reach it are
right-censored, so the median is taken over solved runs and the solve rate reported
alongside — a median without its solve rate flatters methods that only solve easy runs.
Corrected LCBench headline, averaged over 3 pre-training priors:

| method | AUC | ->50% | ->10% | ->5% | solve@5% |
|---|---|---|---|---|---|
| hyperbo_adapt | 0.599 | **1.0** | **2.0** | **2.7** | 82% |
| **em_finetuned** | 0.786 *(worst transfer)* | **1.0** | 3.0 | **3.0** | 82% |
| em_noisefit | 0.671 | **1.0** | **2.0** | 3.0 | 84% |
| **ablr** | **0.532** *(best AUC)* | 2.0 | 2.3 | 3.0 | 83% |
| hyperbo_frozen | 0.612 | **1.0** | 2.3 | 3.3 | 82% |
| ablr_adapt | 0.611 | 2.0 | 3.0 | 3.3 | 80% |
| em_frozen | 0.639 | **1.0** | 3.0 | 4.0 | 82% |
| vanilla_gp | 1.250 | 5.0 | 10.0 | 13.0 | 71% |
| pacoh_frozen | 1.467 | 5.7 | 12.7 | 13.7 | 62% |
| random | 1.926 | 6.0 | 10.0 | 10.0 | 35% |

**The two metrics disagree on the fine ordering.**

- By AUC: `ablr < hyperbo_adapt < ablr_adapt < hyperbo_frozen < em_frozen`
- By budget-to-5%: `hyperbo_adapt < em_finetuned ~ em_noisefit ~ ablr < hyperbo_frozen`

Two reversals matter:

1. **`em_finetuned` is the *worst* transfer method on AUC (0.786) and among the *best* on
   budget-to-target (3.0).** Exactly the explore-then-converge profile AUC mis-scores: it
   refits a base kernel each step so its early trajectory is ragged, but it reaches any
   given target as fast as anything else.
2. **ABLR is best on AUC and the *slowest* to the first target** — 2 evaluations to halve
   the gap, where every EM and HyperBO variant needs 1. That follows directly from ABLR's
   **zero prior mean**: it has no informed first pick. It matches §14.1, where EM's
   zero-shot top-1 regret is 3.7x better than HyperBO's.

### 22.1 The prior work never used AUC

Checked against the HyperBO source: **AUC appears zero times in Wang et al. (2021).**
Their headline metric is a *performance profile* — "the fraction of all test tasks that
each method is able to solve by reaching at most **0.05, 0.01 and 0.001** regrets" at each
BO iteration (`exp-hold-out-related.tex`) — plus final simple regret at iteration 100 and
average-rank-vs-iteration plots. Their solve thresholds are **absolute** regret values,
not fractions of an initial gap.

So the metric we have been ranking on is not the one the field, or the paper we are
comparing against, actually uses — which is a second reason our ordering did not
reproduce theirs, independent of the protocol issues in §21.2. `analyze.py` now supports
absolute thresholds (pass a negative `tol`, e.g. `-0.05` for their C=0.05) in both
`budget_to_target` and the data profile, so our profiles are directly comparable to
theirs.

**Recommended headline going forward:** performance profiles at fixed regret thresholds,
with budget-to-target as the tabular summary. Keep AUC only as a secondary "average
quality of the search" number, clearly labelled as such.

**Consequences.**

- **§20.1's "ABLR significantly beats every EM variant" is AUC-specific.** On
  budget-to-target they are tied, and EM is strictly faster to the 50% target. That claim
  must be qualified by metric.
- The **large** separations are metric-invariant and can be stated without qualification:
  every transfer method reaches the 5% target in 2.7-4 evaluations against 13 for
  `vanilla_gp`, 13.7 for PACOH and 10 for random (which solves only 35% of runs at all).
  A **3-5x sample-efficiency gain** is the claim worth making in the paper.
- **Report both.** AUC answers "how good was the search on average"; budget-to-target
  answers "how long until I can stop". `analyze.py` now emits both, and the summary JSON
  carries `budget_to_target` at the 50/10/5% levels.

---

## 23. Improving the Empirical GP: what worked, what was refuted **[committed]**

§20–§22 measured the EM model against properly-trained baselines. This section tries to
*improve* it, on the 7D regression task and on the multi-output / multi-task variants.
Metrics throughout are accuracy **and** calibration — RMSE alone hides overconfidence,
NLL alone conflates the two, and `sigma/RMSE` (1.0 = calibrated) separates them.

### 23.1 Shrinkage repairs the 7D NLL, and the fix is large

The refreshed 7D run left `em_frozen`'s NLL diverging with data (1.5 → 6.9 → **14.1** at
n=20/50/100) while its RMSE stayed fine (0.33 → 0.20) — pure overconfidence. Sweeping
M-step shrinkage over the full variant set (5 α × 6 variants):

| variant | α=0 @n=100 | best | at α |
|---|---|---|---|
| **em_frozen** | **14.107** | **−0.226** | 0.5 |
| em_noisefit | 0.394 | −0.398 | 0.3 |
| **em_finetuned** | −0.145 | **−0.417** | 0.1 |
| em_additive_3way | −0.278 | −0.376 | 0.1 |

`em_frozen` improves by **14.3 nats**. The best configuration, `em_finetuned` + α=0.1 at
−0.417, essentially reproduces the manuscript's reported −0.415 — but only with **both**
levers, the additive base *and* shrinkage, neither of which the manuscript currently
describes. Shrinkage buys calibration and costs a little accuracy: coverage 0.80 → 0.90,
RMSE 0.190 → 0.198.

`em_additive_freshbase` is catastrophic at small n (**NLL 54.8** at n=5) where
`em_additive_warmbase` is fine (0.32): a freshly-initialized base kernel overfits a
5-point design. This is direct evidence for the "freeze the lengthscales for transfer,
fit only the outputscale" guidance in \Cref{apd:base_augmented}.

### 23.2 Two structural hypotheses for the residual gap — both refuted

At n=100 the tuned EM model still trailed HyperBO (−0.417 vs −0.995). Two candidate
causes were pre-registered and tested.

**H1, rank versus dimension: refuted, and backwards.** Σ has rank ≤ K−1 = 29 in an
M-dimensional space, so we predicted that shrinking M toward K would help. It does the
opposite — NLL improves monotonically with M (M=50: +0.082, M=100: −0.134, M=200:
−0.280). Extra inducing points buy resolution of the empirical mean and covariance over
the 7D config space, and that outweighs the enlarged null space. **The paper's M=100 is
suboptimal; M=200–300 is better.**

**H2, conditioning noise: no effect.** NLL is identical across `cond_noise` ∈
{0.001, 0.01, 0.05, 0.2}. The reason is structural: `em_finetuned` and `em_noisefit` both
*fit* the noise, so the frozen initial value is immediately overwritten. The knob can only
matter for `em_frozen`, which is never the best variant. (Exposed as `--cond-noise`
anyway, since its absence was itself a gap.)

### 23.3 The gap is regime-dependent, and EM wins where the paper claims it does

Tuning α per budget at M=300:

| n_train | best EM NLL | config | HyperBO NLL | gap |
|---|---|---|---|---|
| **5** | **−0.155** | em_frozen α=0 | +0.037 | **−0.192 (EM wins)** |
| 20 | −0.359 | em_noisefit α=0.1 | −0.443 | +0.084 (near-tie) |
| 50 | −0.415 | em_noisefit α=0.2 | −0.795 | +0.380 |
| 100 | −0.371 | em_finetuned α=0.2 | −0.995 | +0.624 |

**EM dominates the sparse regime, HyperBO the data-rich regime, with the crossover near
n=20.** That is precisely the claim §5.5 already scopes ("strong data efficiency in the
sparse-data regime, n_train ≤ 20"), so these results *confirm the scoped claim* while
showing that Table `tab:hd_nll`'s dominance at **all** n does not survive a
properly-trained HyperBO. Note the n=5 win is a **calibration** win, not an accuracy one:
HyperBO still has lower RMSE there (0.242 vs 0.257) but is wildly under-confident
(sigma/RMSE 1.76, coverage 1.00).

**The optimal α rises monotonically with the target budget** — 0.0, 0.1, 0.2, (0.3+) —
which is what the MAP-EM reading predicts: as the target supplies more data the
rank-deficient prior's null-space overconfidence becomes the dominant error term. At
α=0.2 the model is still overconfident at n≥50 (sigma/RMSE 0.51–0.64 vs HyperBO's 0.89),
so α=0.2 is a **boundary solution, not an optimum**; an extension to {0.3, 0.5, 0.7} is
running.

### 23.4 Multi-output and multi-task: the additive base is what mattered

Both variants were evaluated against the baseline each must beat to justify itself —
independent per-output GPs, and a single-task GP on the target alone — with and without a
fitted additive base (8 datasets × 8 seeds, 2048 scored rows, 0 errors).

| model | RMSE | NLL | cov95 | sigma/RMSE |
|---|---|---|---|---|
| multioutput | 0.064 | −1.438 | 0.96 | 3.65 |
| multioutput **+base** | **0.038** | −1.224 | 0.88 | 2.03 |
| independent | 0.028 | −1.132 | 0.93 | 4.31 |
| independent **+base** | **0.026** | **−2.558** | 0.96 | 4.07 |
| multitask | 0.189 | 1.084 | 0.94 | 3.95 |
| multitask **+base** | 0.158 | **−0.609** | 0.93 | 2.49 |
| singletask | 0.151 | 3.893 | 0.86 | 2.72 |
| singletask **+base** | 0.155 | **−0.789** | 0.95 | 2.77 |

- **The additive base is the dominant effect.** It cuts multi-output RMSE 0.064 → 0.038
  (z=−2.98) and converts the multi-task NLL blow-up into a healthy value (1.084 →
  −0.609; single-task 3.893 → −0.789).
- **Multi-task ≈ single-task once both have a base** — every paired comparison is
  insignificant (|z| < 2). The dramatic NLL gap seen without a base was the *missing base
  kernel*, not cross-task structure.
- **Multi-output still loses to independent per-output models** even with a base
  (RMSE z=+4.21). The dense cross-output covariance does not pay for itself at K=40
  historical curves.
- These models are **under**-confident (sigma/RMSE 2–4), the opposite of the 7D EM
  regime, so the corrective levers differ by regime and should not be transplanted.

**For the paper:** do not claim either variant improves performance. The appendix text in
`PAPER_ADDITIONS.tex` describes the constructions without claiming superiority, which is
the defensible framing. If results are included, the honest statement is that multi-output
trades accuracy for a dense covariance and is not recommended at these corpus sizes, while
multi-task matches single-task once both are base-augmented.

---

## 24. Closing the gap: three refuted hypotheses and one that worked **[committed]**

§23 left the EM model trailing HyperBO at n≥20 on 7D NLL. This section diagnoses why.
The value of the section is mostly the **refutations** — each was plausible, each is now
measured, and none should be re-attempted.

### 24.1 Refuted: the rank ceiling

The continuous-domain covariance is `Σ(X) = Λ(X) + W Σ_Z Wᵀ` with Nyström residual
`Λ(X) = K(X,X) − W K(Z,X)`, and `Λ ≡ 0` whenever the evaluation points lie in the
reference set — which they did, since the harness anchored Z on the whole pool. That
suggested a hard rank-(K−1) = rank-29 ceiling.

**It is not the constraint.** Two independent refutations:
- Shrinkage adds `α·s·B` with `B` a full-rank Matérn gram, and the additive base adds a
  full-rank `k_θ` at the query points. Both configurations are already full rank, and
  they are the best-performing ones. Measured: 300 non-negligible eigenvalues.
- The M=30 arm gives a 30×30 covariance of rank 29 — nearly full rank, the best possible
  rank-to-dimension ratio — and is **the worst configuration by a wide margin**
  (NLL ≈ 1.0 at every budget).

### 24.2 Refuted: variance scale

If the model were merely overconfident, a single optimal variance rescale would fix it.
Adding `nll_recalibrated` (rescale predictive sd by the ML-optimal τ) shows the opposite:

| n | EM raw → recal | HyperBO raw → recal | gap raw → recal |
|---|---|---|---|
| 20 | −0.070 → −0.249 | −0.446 → −0.808 | +0.376 → **+0.560** |
| 100 | −0.398 → −0.507 | −1.000 → −1.248 | +0.602 → **+0.741** |

Perfect recalibration **widens** the gap — HyperBO gains more from it than EM does. No
post-hoc calibration, and by extension no amount of α tuning, can close this.

### 24.3 Confounded, still open: off-grid capacity

The first `--n-inducing` implementation subsetted the *datasets*, so every method —
including HyperBO, which does not use the EM prior at all — pre-trained on only M points
per task. The tell was HyperBO's NLL hitting **12.324** at M=30 versus −0.995 in control.
That sweep measured a data reduction, not off-grid capacity. Fixed by passing
`inducing_points` to `pretrain_em_prior` so observations stay complete. **The off-grid
question is unanswered and needs a re-run.**

### 24.4 What worked: a learned kernel metric

With optimal rescaling the residual lives in `mean(log σᵢ)` — *where* uncertainty is
placed — and RMSE explains only ~0.12 of 0.74 nats. HyperBO's Matérn acts on a learned
32-d feature map; the EM canonical kernel had 7 ARD lengthscales. Since the canonical
kernel drives `W`, `Λ` and the additive base, it is exactly the object that sets variance
shape. `_DeepKernel` gives it `φ(x) = [x, MLP(x)]` fed to an ARD Matérn, meta-trained by
Adam across pre-training tasks. The skip connection keeps the embedding injective (a
plain MLP collapsed it, `NotPSDError`) and makes the deep kernel a strict generalization
of the plain one.

| n | best DEEP | best PLAIN | HyperBO | deep − HyperBO |
|---|---|---|---|---|
| 5 | −0.151 | −0.155 | +0.037 | **−0.188 (EM wins)** |
| 20 | −0.260 | −0.359 | −0.443 | +0.183 |
| 50 | −0.478 | −0.434 | −0.795 | +0.317 |
| 100 | **−0.624** | −0.396 | −0.995 | +0.370 |

At n=100 the gap narrows **+0.602 → +0.370 (38%)** and EM's RMSE overtakes HyperBO's
(**0.162 vs 0.176**). Two honest caveats:
- A 2-dataset smoke test reported −0.744 / 0.143; at 5 datasets it is −0.624 / 0.165. The
  small-scale test overstated the win by ~0.12 nats.
- **At n=20 the deep kernel is worse than plain** (−0.260 vs −0.359). The skip connection
  guarantees it can *represent* the plain kernel; 2000 Adam steps over 30 meta-training
  tasks does not reliably *find* it. A representational guarantee is not an optimization
  guarantee — the deep fit needs regularization (early stopping on held-out tasks).

### 24.5 Where this leaves the story

EM wins at n=5, HyperBO at n≥20, and the crossover is somewhere in between — our grid is
too coarse to say where. Since meta-learning is most valuable exactly where evaluations
are expensive, **locating that crossover precisely, with error bars, is the highest-value
remaining experiment**, and is running (§25).

---

## 25. Locating the crossover: the low-sample story does not replicate **[committed]**

§24 ended with EM winning at n=5 and losing at n≥20, and the crossover unlocated. Because
meta-learned priors matter most where evaluations are expensive, the low-sample regime is
the one that decides whether there is a story. This section pins the crossover down at
higher statistical power — and the answer is negative.

### 25.1 Design

Dense grid n ∈ {1,2,3,4,5,6,8,10,12,15,20,25,30,50,100}; evaluation datasets raised 5 → 10
(pre-training tasks 25); both canonical kernels; α ∈ {0, 0.1}; HyperBO at its converged
25k steps in every arm. `bo_diagnose` was extended to emit `per_dataset_nll` /
`per_dataset_rmse` so the comparison is **paired by evaluation dataset** with a
**dataset-clustered SEM** — runs on one dataset share a historical corpus and are not
independent, so a pooled standard error would overstate significance.

### 25.2 EM never significantly wins

| n | best EM | HyperBO | diff ± SEM | z | verdict |
|---|---|---|---|---|---|
| 1 | 0.683 | 0.619 | +0.063 ± 0.056 | 1.13 | tie |
| 3 | 0.400 | 0.371 | +0.029 ± 0.092 | 0.31 | tie |
| 5 | 0.340 | 0.191 | +0.149 ± 0.099 | 1.51 | tie |
| **8** | 0.155 | −0.009 | +0.165 ± 0.079 | 2.09 | **HyperBO** |
| 15 | 0.088 | −0.236 | +0.324 ± 0.106 | 3.06 | **HyperBO** |
| 25 | −0.069 | −0.383 | +0.314 ± 0.073 | 4.32 | **HyperBO** |
| 100 | −0.260 | −0.200 | −0.060 ± 0.305 | −0.20 | tie |

**The §24 result that EM wins at n=5 (−0.155 vs +0.037) does not replicate.** At 10
evaluation datasets EM is *worse* at n=5. The earlier win was an artifact of the 5-dataset
evaluation set. EM is statistically tied from n=1–6 and significantly behind from n=8.

### 25.3 The split matters more than the effects we were chasing

HyperBO's n=100 NLL moved from **−0.995 to −0.200** purely from changing `n_pretrain`
30 → 25 and the evaluation split. That is larger than almost every effect measured in
§23–§24. **Any fine-grained ordering on this benchmark needs multiple splits before it can
be trusted**, and single-split orderings — including several reported earlier in this
document — should be treated as provisional.

### 25.4 BO: EM is competitive but mid-tier

LCBench, 8 independent pre-training priors, regret-AUC (lower better):

| method | regret-AUC | sd | iters to r ≤ 0.5 |
|---|---|---|---|
| **ablr** | **0.537** | 0.026 | **7.8** |
| hyperbo_adapt | 0.601 | 0.037 | 13.6 |
| ablr_adapt | 0.602 | 0.046 | 9.6 |
| hyperbo_frozen | 0.610 | 0.036 | 13.8 |
| **em_frozen** | **0.639** | **0.000** | **12.0** |
| em_noisefit | 0.671 | 0.000 | 14.0 |
| em_finetuned | 0.786 | 0.000 | 21.0 |
| vanilla_gp | 1.250 | 0.000 | 29.0 |
| pacoh_frozen | 1.559 | 0.266 | 39.2 |
| random | 1.926 | 0.000 | 41.0 |

Three observations:
- EM is **5th of 10** — solidly in the transfer tier, ~2× better than vanilla GP and 3×
  better than random, but ABLR leads it by ~4× ABLR's own prior-draw sd, so the gap is
  real rather than a seeding artifact.
- **EM's prior-draw sd is exactly 0.000** because EM pre-training is deterministic given
  the data. Its number is exact where the neural baselines carry ±0.03–0.27.
- **On budget-to-target EM beats HyperBO** (12.0 vs 13.8 iterations to r ≤ 0.5) while
  losing on AUC — the §22 metric-dependence again. Which method looks better depends on
  the metric, and the paper's own metric is not AUC.
- In BO the **simplest** EM variant wins (frozen < noisefit < finetuned), the reverse of
  the regression ordering. Consistent with §23.3: BO spends its budget at low n, where
  α = 0 and a frozen prior are optimal.

### 25.5 What the evidence now supports

Three independent lines — the corrected HyperBO budget (§19), the failed n=5 replication
(§25.2), and ABLR's BO lead (§25.4) — all point away from "EM is state of the art". The
claims that *are* supported:

- EM is **competitive within the transfer tier** on both benchmarks, and far ahead of
  vanilla GP, PACOH and random.
- EM pre-training is **deterministic** (prior-draw sd exactly zero) and **63–109× cheaper**
  than HyperBO's, where the neural baselines are neither.
- EM **wins on budget-to-target** against HyperBO even where it loses on AUC.

These are real contributions; they are simply different from a state-of-the-art claim, and
the framing should be settled before more compute is spent.

---

## 26. The definitive BO measurement, the cost Pareto, and the HyperBO hybrid **[committed]**

Three sweeps had completed but sat unanalyzed. Analyzing them **overturns the §25.4
headline** and produces the strongest positive result in this document.

### 26.1 Paired, multi-split BO: EM is top-tier

§25.4 reported ABLR first and EM fifth. That used 8 pre-training priors on a **single
dataset split** — and §25.3 had already shown the split is the dominant noise source.
Re-run over **5 splits x 2 priors** and compared **paired on the same splits**:

| vs `em_finetuned` | Δ regret-AUC | z | Δ budget→r≤0.5 | z | verdict |
|---|---|---|---|---|---|
| ablr | −0.500 | −3.13 | −8.9 | −2.78 | **EM better** |
| ablr_adapt | −0.289 | −3.09 | −9.1 | −3.59 | **EM better** |
| pacoh_frozen | −0.426 | −5.43 | −10.0 | −2.75 | **EM better** |
| em_frozen | −0.277 | −2.65 | −15.0 | −2.03 | EM better |
| em_noisefit | −0.121 | −1.27 | −6.0 | −0.91 | tied |
| hyperbo_adapt | −0.160 | −1.24 | −3.2 | −0.88 | tied |
| hyperbo_frozen | −0.168 | −1.26 | −3.2 | −0.84 | tied |
| vanilla_gp | −0.982 | −5.43 | −19.8 | −8.26 | **EM better** |
| random | −2.363 | −5.70 | −31.4 | −10.73 | **EM better** |

**`em_finetuned` is statistically tied with both HyperBO variants and significantly
better than ABLR, ABLR-adapt, PACOH, vanilla GP and random.** ABLR swung from
regret-AUC 0.537 (one split) to 1.069 (five splits) — larger than any effect chased in
§19–§25. Note also that the best EM variant flips with the protocol: `em_frozen` in the
single-split analysis, `em_finetuned` here.

**Unclustered SEMs are misleading.** Independent per-method SEMs are ±0.13–0.17 on AUC,
which would make everything look tied. The paired test removes the shared split effect
and is what recovers the signal. Always pair.

### 26.2 EM is Pareto-dominant on performance vs cost

Pre-training cost measured on the same 5×2 cells (medians, n=10):

| method | budget→r≤0.5 | pre-train | × EM cost |
|---|---|---|---|
| **em_finetuned** | **9.6** | **20.1 s** | **1.0×** |
| hyperbo_{frozen,adapt} | 12.8 | 3431.1 s | 171× |
| ablr / ablr_adapt | 18.5 / 18.7 | 873.5 s | 44× |
| pacoh_frozen | 19.6 | 12616.0 s | **629×** |

EM has the **best budget-to-target and the lowest cost** — the only non-trivial point on
the frontier. Unlike the accuracy orderings, this margin (44–629×) is far too large for
split noise to overturn, which makes it the most durable claim available.

### 26.3 HyperBO's pre-trained kernel as an ADDITIVE component is the best variant

Reusing HyperBO's 25k-step kernel as an additive component at conditioning
(`k_emp + s·k_hb`, with `hyperbo_kernel_from_prior` freezing its internals so exactly one
scalar is fit) tops **all four** LCBench regression cells:

| cell | best | NLL / RMSE | vs standalone hyperbo_frozen |
|---|---|---|---|
| on-grid, sumll canonical | `em_additive_hyperbo` | −0.465 / 0.266 | −0.299 / 0.277 |
| on-grid, hyperbo canonical | `em_additive_hyperbo` | −0.523 / 0.266 | −0.299 / 0.277 |
| off-grid, sumll | `em_additive_hyperbo_frozen` | 0.311 / 0.346 | 1.526 / 0.340 |
| off-grid, hyperbo | `em_additive_hyperbo_frozen` | −0.107 / 0.320 | 1.526 / 0.340 |

It beats standalone HyperBO on **both** metrics everywhere. The **fitted** weight wins
on-grid; the **frozen** weight wins off-grid. This is the hybrid that works — note it is
the *additive* door, not the canonical one.

### 26.4 The PD1 "pool no-op" was a mislabelling, not a code bug

The PD1 `matched` and `full` cells in `canon_matrix` and `hb_hybrid` are **bit-identical
to four decimals across every method**. The original diagnosis — that the full-pool loader
returns `None` and `bo_experiment.py:1413` silently falls back — is **wrong**. See §27.1
for the actual cause: those eight runs never touched PD1 at all.

---

## 27. PD1 natural heterogeneity: the experiment was already on disk **[committed]**

§21.2 named the full-pool PD1 run "the single highest-value experiment remaining", and
§26.4 recorded it as blocked by a loader bug. Both were wrong: the sweep **ran on
2026-08-08** (`results/raw/pd1pool/`, 48 files, `run_pd1_pool.sh` + `run_armc_seeds.sh`),
the loader worked, and the results had simply never been analysed. This section fixes the
misdiagnosis and reports the analysis.

Entry point: `analyze_pd1pool.py` (`//pytorch/botorch/_scratch_bo:analyze_pd1pool`).

### 27.1 The real cause: two sweep scripts omitted `--benchmark`

`run_canon_matrix.sh` and `run_hb_hybrid.sh` passed `--pd1-source full
--pd1-candidate-pool <pool> --pd1-pretrain-pool <pool>` but **never passed
`--benchmark pd1_loo`**. The default is `lcbench` (`bo_experiment.py:2015`), and the PD1
pool flags are read only by the PD1 code paths. All eight files prove it in their own
metadata:

| evidence | value |
|---|---|
| `config.benchmark` in all 8 JSONs | `lcbench` |
| `eval_datasets` length | 6 (an LCBench sweep), not 23 (the PD1 task count) |
| first line of every log | `Loading 35 LCBench datasets ...` |

This explains every symptom exactly: the pool flags were unread, so `matched` and `full`
were the *same LCBench run twice*; `--em-canonical` still worked because it is not
PD1-specific.

The loader is fine. `logs/pd1pool_fullboth_p1_s0.txt`:

```
per-task pools: 1666-3084 configs (mean 2040) vs 400 matched | candidates=full pretrain=full
```

**Consequence.** Those eight files are duplicate LCBench runs under PD1 filenames —
**retracted, do not cite**. They are now `RETRACTED_ranLCBench_<pool>_<canon>.json` in
`canon_matrix/` and `hb_hybrid/` (logs likewise), renamed rather than deleted so the
evidence for this section stays in-tree. The rename drops the `pd1` substring, so a
`*pd1*` glob can no longer pick them up — a note in a doc does not survive a glob. Their
LCBench content is redundant with `lcb_*` / `reg_*` cells we already hold under correct
names, so no LCBench conclusion moves; §26.3's four regression cells are `bo_diagnose`
runs and are unaffected. Nothing else is touched: `pd1pool` and `pd1full` both pass
`--benchmark pd1_loo` explicitly.

The runner scripts keep their original output paths, so re-running them now regenerates
genuine PD1 results under the correct names.

**Fixed.** Both scripts now pass `--benchmark pd1_loo`, and `bo_experiment.py` raises if
any `--pd1-*` flag is passed with a non-PD1 benchmark, detecting explicit use from `argv`
because each of those flags has a non-`None` default that a value check cannot
distinguish from disuse. This class of bug is now structurally impossible.

### 27.2 Design: a two-factor protocol ablation, replicated over priors

All arms are leave-one-out over the **23 PD1 tasks**, `--pd1-source full`, neglog warp,
50 iterations, per-task standardization.

| arm | candidates | baseline pre-training | files | runs |
|---|---|---|---|---|
| A | matched (400) | matched | `pd1full/pd1_loo_neglog_s*` | 115 |
| B | **full (~2040/task)** | matched | `pd1pool/fullcand_s*` | 115 |
| C | **full** | **full (~2040/task)** | `pd1pool/fullboth[_p1,_p2]_s*` | 115 + 69 + 69 |

A→B isolates the search space; B→C adds the baselines' data advantage. Arm C is
replicated over **three pre-training priors** (p0/p1/p2) because §19.4 established that a
single prior cannot rank the neural meta-learners — `run_armc_seeds.sh` was written for
exactly this test and its output had never been read.

Two guards on the arms being real, both passing: every arm's mean final regret differs,
and `vanilla_gp`/`random` — which use no pre-training data — are **bit-identical between
B and C-p0**, which is precisely what changing only the pre-training pool must produce.

**Caveats.** Arm A used `--hyperbo-iters 50000` against 25000 in B/C, so A-vs-B/C HyperBO
comparisons are confounded; only within-arm orderings are used. The candidate pools differ
across arms, so absolute regret is not commensurable between them either. EM always
pre-trains on the matched grid (its inducing points require it), so arm C hands the
baselines 5× more data than EM gets.

### 27.3 The §21.3 "HyperBO flips to first" claim does not survive replication

Insight 35 claimed that searching each task's own ~2040 configurations flips HyperBO from
last of the transfer methods to first, reproducing Wang et al.'s ordering. It rests on one
prior. The three arm-C priors disagree on the ordering:

| arm | 1st | 2nd | 3rd | where `hyperbo_frozen` lands |
|---|---|---|---|---|
| A matched | ablr | hyperbo_adapt | em_noisefit | 4th |
| B full/matched | **hyperbo_frozen** | hyperbo_adapt | ablr | **1st** |
| C p0 | ablr | hyperbo_adapt | hyperbo_frozen | 3rd |
| C p1 | **em_frozen** | **em_finetuned** | hyperbo_adapt | **5th** |
| C p2 | ablr | em_frozen | hyperbo_adapt | 6th |

Two of the three arm-C priors put EM **above** HyperBO. The flip is a prior artifact of
the same kind §19.4 and §26.1 already caught twice.

Pooling the three priors (253 runs) and pairing by task, **every transfer method is tied
on final regret**:

| comparison | difference | z | verdict |
|---|---|---|---|
| `hyperbo_frozen` − `em_finetuned` | +0.00072 | +0.08 | tied |
| `hyperbo_frozen` − `em_frozen` | −0.00019 | −0.02 | tied |
| `ablr` − `em_finetuned` | −0.00353 | −0.94 | tied |
| `em_finetuned` − `vanilla_gp` | −0.00894 | −1.98 | tied (nominal EM win) |

Ordering by mean, pooled arm C: ablr 0.02819, hyperbo_adapt 0.02960, **em_finetuned
0.03172**, hyperbo_frozen 0.03245, em_frozen 0.03264, em_noisefit 0.03430, vanilla_gp
0.04067, random 0.07891 — a 0.004 spread against clustered SEMs of ±0.006.

**Insight 35 is retracted.**

### 27.4 The honest result: EM ties on real heterogeneity, and does not win

The LCBench sweep (§24/§25) found EM degrades gracefully under sparse/irregular
observations while HyperBO collapses (z=+3.3 → −3.5). That sparsity was **synthetic**, and
PD1's ~2040-configs-per-task-with-400-shared structure is the natural version. It does
**not** replicate. On PD1, EM is statistically tied — it neither wins nor loses.

The one place a consistent trend survives all three priors is the **solve rate at tight
targets**, where the neural meta-learners lead:

| target | em_finetuned | hyperbo_frozen | hyperbo_adapt | ablr |
|---|---|---|---|---|
| \|regret\| ≤ 0.05 | 72% | 71% | 74% | 76% |
| \|regret\| ≤ 0.01 | 37% | 46% | 50% | 48% |
| \|regret\| ≤ 0.001 | 19% | 29% | 35% | 28% |

The direction is stable across p0/p1/p2 and is consistent with the interpolation-error
mechanism §21.3 proposed: off-grid, EM must interpolate its empirical prior to candidates
it was never defined on, while a deep kernel is a genuine continuous function. But it is
**not statistically significant** — paired and clustered by task, only `ablr − em_finetuned`
at 0.01 reaches z=+2.02, one marginal result out of nine comparisons, which no
multiplicity correction survives. Report it as a trend worth a targeted follow-up, not a
finding.

### 27.5 What this does to the story

- **Kept, and now stronger.** EM is Pareto-dominant (§26.2). It ties the neural baselines
  on a *real*, heterogeneous, off-grid benchmark at 44–629× lower pre-training cost —
  while being handed 5× less pre-training data than they get in arm C. Tying under a
  handicap is the most robust claim this project has.
- **Lost.** "EM wins under heterogeneity" is LCBench-and-synthetic-sparsity only. Do not
  generalize it to PD1; §27.4 is the counter-example a reviewer would find.
- **Open.** The tight-target solve-rate gap is the one real off-grid signal. Testing it
  properly needs more task clusters or more priors, not more seeds — 23 clusters is the
  binding constraint on every z in this section.
- **The one genuine coverage gap.** No PD1 arm has ever run the HyperBO hybrid: every
  file in `pd1pool` and `pd1full` records `em_canonical: None` and none of them include
  `em_additive_hyperbo*` in `methods`. §26.3's additive-hybrid win is LCBench-regression
  only, and arm C is where a genuine continuous kernel should pay off, since that is
  exactly the off-grid regime §27.4's solve-rate trend points at. This is the experiment
  worth the next compute slot — not a re-run of the eight retracted cells, whose LCBench
  content is redundant.

### 27.6 Power ceiling: PD1 BO cannot resolve these methods, at any budget

Before spending an overnight slot on the arm-C hybrid sweep (§5 item 3), two pre-flight
checks. One passed; the other kills the experiment as designed.

**Pre-flight 1 — RNG safety: PASS.** Adding `em_additive_hyperbo` to the method list
leaves `em_frozen` and `hyperbo_frozen` **bit-identical** to a run without it, so a new
sweep can be merged with the existing arm-C data rather than forcing a full re-run.

**Pre-flight 2 — statistical power: FAIL, and not fixable with compute.** The SEM of a
task-clustered paired test is `sd(cluster means)/sqrt(T)`, where

    Var(cluster mean) = sigma_between^2 + sigma_within^2 / n_per_task

Priors and seeds only shrink the second term. Decomposing the pooled arm-C data
(`analyze_pd1pool.py`, `variance_decomposition`) gives the SEM **floor** at infinite
priors — and for every method-vs-method contrast that floor is already too large:

| contrast | metric | \|effect\| | SEM now | SEM floor | SEM needed for z=2 |
|---|---|---|---|---|---|
| `hyperbo_frozen` − `em_finetuned` | final regret | 0.00072 | 0.00858 | 0.00841 | 0.00036 |
| `hyperbo_adapt` − `em_finetuned` | final regret | 0.00212 | 0.00906 | 0.00889 | 0.00106 |
| `ablr` − `em_finetuned` | final regret | 0.00353 | 0.00376 | 0.00309 | 0.00177 |
| `hyperbo_adapt` − `em_finetuned` | solved to 0.001 | 0.15415 | 0.08954 | 0.08578 | 0.07708 |
| `em_finetuned` − `vanilla_gp` | final regret | 0.00894 | 0.00451 | 0.00390 | 0.00447 |

Only the last is reachable, and only because EM-vs-`vanilla_gp` is a large effect; it
needs ~1.1× the current runs. **Every transfer-method comparison is unreachable.**

**The binding constraint is the number of TASKS, not runs.** `floor = sigma_between /
sqrt(T)` and PD1 has T=23. The closest near-miss (`hyperbo_adapt` − `em_finetuned` at
the 0.001 target) would need T≈29. There is no 29-task PD1.

**So: do not run the arm-C sweep to test the hybrid's mean effect.** It would burn an
overnight slot to produce another "tied", and that tie would be a statement about PD1's
task count, not about the methods.

**What the decomposition also says, and this is the interesting part.** The between-task
sd of the *paired difference* is 0.040 against a mean effect of 0.002 — the per-task
effects are ~20× the average, in both directions. The methods are not equivalent; they
**disagree by task**, and averaging destroys the signal. "Which tasks favour EM over
HyperBO, and what distinguishes them" is a well-powered question on exactly this data,
and it does not need a single new BO run.

### 27.7 Correction: §27.6 answered only one of two questions, and the other is resolvable

§27.6's "UNREACHABLE" verdicts are correct but **incomplete**, and stating them without
qualification overstated the case. They describe a single estimand. There are two:

| estimand | question | between-task spread is | precision limit |
|---|---|---|---|
| random-effects (task-clustered) | which method wins on a **new, unseen** task? | irreducible noise | floored by T=23 |
| **fixed-task** | which method wins **on this suite**? | structure, not noise | `1/n`, **no floor** |

Benchmark comparisons normally report the second. There the 23 tasks *are* the
population, so `Var = (1/T^2) sum_t s_t^2/n_t` shrinks without limit as priors and seeds
are added. §27.6 only computed the first and declared the question closed. It is not.

Under the fixed-task estimand (`analyze_pd1pool.py`, `fixed_effects_test`):

| contrast | metric | effect | SE | z | verdict |
|---|---|---|---|---|---|
| `em_finetuned` − `vanilla_gp` | final regret | −0.00894 | 0.00227 | **−3.94** | **EM better** |
| `hyperbo_frozen` − `em_finetuned` | solved to 0.01 | +0.09091 | 0.02628 | **+3.46** | **HyperBO better** |
| `hyperbo_frozen` − `em_finetuned` | solved to 0.001 | +0.09881 | 0.02487 | **+3.97** | **HyperBO better** |
| `hyperbo_adapt` − `em_finetuned` | solved to 0.01 | +0.13439 | 0.02744 | **+4.90** | **HyperBO better** |
| `hyperbo_adapt` − `em_finetuned` | solved to 0.001 | +0.15415 | 0.02568 | **+6.00** | **HyperBO better** |
| `ablr` − `em_finetuned` | final regret | −0.00353 | 0.00215 | −1.64 | tied (needs 1.5× runs) |
| `hyperbo_adapt` − `em_finetuned` | final regret | −0.00212 | 0.00175 | −1.21 | tied (needs 2.7× runs) |
| `hyperbo_frozen` − `em_finetuned` | final regret | +0.00072 | 0.00173 | +0.42 | tied (needs 23× runs) |

**This retracts §27.4's "not statistically significant".** On the tight-target solve
rate, HyperBO beats EM on this suite decisively — z up to 6.0, not a trend. §27.4 was
right that the task-clustered test does not reach significance, but wrong to present that
as the only reading. **EM is materially worse at reaching tight targets off-grid.**

**A bug found while doing this, worth recording.** The first version of
`fixed_effects_test` applied the solve-rate direction convention (higher is better) to
final regret (lower is better), which inverted half the verdicts and briefly reported
`vanilla_gp` beating `em_finetuned` — the reverse of the pooled means in §27.3. Any
metric-agnostic comparison helper needs the direction passed in explicitly.

**Consequences.**

- The remaining final-regret ties are now **finite**, not unreachable: 1.5× more runs for
  ABLR, 2.7× for `hyperbo_adapt`. More priors genuinely do buy resolution here, which is
  the opposite of §27.6's advice.
- The arm-C hybrid sweep is therefore worth more than §27.6 credited: its three extra
  priors move the hybrid contrasts toward fixed-task resolution, not just cost.
- Every claim in this document must now say **which estimand it uses**. "Tied" under
  random effects and "tied" under fixed effects are different statements, and §27
  conflated them.

### 27.8 §26.1 vs §27.7: it is the BENCHMARK, not the metric — and EM does worse than §27.7 said

§26.1 found `em_finetuned` top-tier on **LCBench**. §27.7 finds EM worse than HyperBO on
**PD1**. Three things differ: benchmark, metric, estimand. Isolating them.

**On the primary metric — evaluations to a very good configuration (see
`.llms/rules/metrics.md`), fixed-task estimand, PD1 arm C:**

| contrast | target | Δ evals | SE | z | verdict |
|---|---|---|---|---|---|
| `hyperbo_frozen` − `em_finetuned` | 0.01 | −4.16 | 0.66 | **−6.32** | HyperBO better |
| `hyperbo_adapt` − `em_finetuned` | 0.01 | −4.69 | 0.66 | **−7.13** | HyperBO better |
| `ablr` − `em_finetuned` | 0.01 | −2.18 | 0.70 | **−3.11** | ABLR better |
| `hyperbo_adapt` − `em_finetuned` | 0.001 | −2.03 | 0.65 | **−3.12** | HyperBO better |
| `ablr` − `em_finetuned` | 0.001 | −1.72 | 0.71 | **−2.43** | ABLR better |
| `em_finetuned` − `vanilla_gp` | 0.01 | +0.70 | 0.71 | +0.99 | **tied** |
| `em_finetuned` − `vanilla_gp` | 0.001 | −0.98 | 0.70 | −1.40 | **tied** |

**On PD1, EM reaches a very good configuration 2–5 evaluations later than HyperBO and
ABLR, and is not distinguishable from a from-scratch GP.** That is a worse result for EM
than §27.7 reported, and it is the metric that matters.

**It is not a metric artifact — but AUC did flatter EM.** Regret-AUC on the same runs
(cross-check only, §22): HyperBO beats EM (z = −2.80, −4.01), ABLR ties — same direction.
But AUC says EM crushes `vanilla_gp` at **z = −9.93**, where budget-to-target says
**tied**. Leading with AUC, as the first draft of this section did, would have claimed a
headline EM advantage the primary metric does not support. Hence the metric policy now in
`.llms/rules/metrics.md`.

**So the disagreement with §26.1 is the benchmark.** LCBench is dense and on-grid, where
EM is top-tier. PD1 arm C is off-grid with naturally heterogeneous pools, where EM loses.

**This partially rehabilitates the §21.3 mechanism.** That section argued EM must
interpolate its empirical prior to candidates it was never defined on, while a deep kernel
is a genuine continuous function. §27.3 retracted the *ranking* built on it (one prior,
two of three replicates disagreed), but the mechanism now has properly-powered support
from an independent direction: PD1 is the off-grid benchmark and is exactly where EM
loses, on the metric we care about.

**What survives untouched.**

- §26.1 and §26.2 are LCBench claims; nothing here contradicts them.
- **Cost.** EM still pre-trains 44–629× cheaper. On PD1 that now buys parity with a
  from-scratch GP rather than a win, so state the Pareto claim as *cost*, not
  *cost-and-accuracy*, whenever the benchmark is PD1.

**Not yet measured: the additive hybrid.** None of the above includes
`em_additive_hyperbo*`, which has never run on PD1 — that is the in-flight arm-C sweep.
§26.3 found the hybrid beats standalone HyperBO on LCBench *regression*. Whether it closes
this PD1 gap is now the sharpest open question in the project, because §27.8 says the gap
is real and the mechanism (§21.3) predicts a continuous kernel is exactly the fix.

**A second direction bug, caught the same way.** The first AUC run returned numbers
byte-identical to final regret across all four contrasts, because the `metric="auc"`
branch had been edited into the wrong function — the surrounding code was identical in two
places — so the label printed `[regret-AUC]` while the value was still final regret.
Impossible-looking agreement between two metrics is a bug signature, not a finding.

### 27.9 EM has been run UNTUNED in every BO comparison, and α alone does not fix it

Prompted by the question of whether EM's weak PD1 showing is a tuning artifact. It partly
is, but not in the way expected.

**Every headline BO run used `--em-shrinkage 0.0` — the default, i.e. shrinkage OFF.**
That covers `bo_multisplit` (LCBench, §26.1/§26.2) and all of `pd1pool` (PD1, §27).
Across all BO runs on disk: 176 at α=0, 19 at α=0.1, 7 at α=0.3. Meanwhile §23.1's
"shrinkage is worth up to 14 nats" is a **7D regression NLL** result that was never
carried into a BO comparison. EM's largest known improvement has never been switched on
where the headline claims are made.

**But α on its own does not help PD1 BO.** `results/raw/followup/pd1_shrink_full_a*.json`
already sweeps it (arm B, 6 tasks × 3 seeds, previously unanalysed). Mean evaluations to
target, paired, primary metric:

| method | target | α=0.0 | α=0.1 | α=0.3 |
|---|---|---|---|---|
| `em_frozen` | 0.01 | **38.4** | 39.5 | 40.4 |
| `em_finetuned` | 0.01 | **37.3** | 39.9 | 37.6 |
| `em_frozen` | 0.001 | **41.0** | 42.4 | 44.6 |
| `em_noisefit` | 0.001 | **39.9** | 42.4 | 44.5 |
| `em_finetuned` | 0.001 | **40.1** | 42.2 | 44.6 |

Every paired difference is positive — α makes it *worse* — monotonically so at the tight
target (+3.6 to +4.6 evaluations at α=0.3), though none reaches significance (|z| ≤ 1.51).
Control: `hyperbo_frozen`, `vanilla_gp` and `random` are identical across α, as they must
be since α touches only EM.

**Why this does not close the question — the α × target interaction.** The M-step is

    Sigma <- (1 - a) * Sigma_ML + a * s * B

**At α=0 the target `B` drops out of the equation entirely.** So:

1. Every PD1 run, and the in-flight hybrid sweep, has the shrinkage target as a
   *mathematical no-op*. Setting `--em-canonical hyperbo` at α=0 exercises only the
   interpolation and EM-init doors — **not** the shrinkage-target door, which the
   canon_decomp decomposition found to be the **largest** of the three (4.10 nats).
2. The α screen used the DEFAULT target (`sumll`). That target **is** pre-trained — a
   shared ARD Matérn-5/2 fitted across all pre-training tasks by summed marginal
   likelihood (`build_shared_kernel`), genuinely analogous to the tutorial's pre-trained
   canonical GP baseline. It differs from HyperBO's kernel in **capacity** (plain Matérn
   vs MLP+Matérn trained 25k steps on NLL/EKL), not in whether it was trained.
   **But it was fitted on the wrong data.** The α screen is **arm B**
   (`pd1_pretrain_pool: matched`), so `build_shared_kernel` saw only the 400-config
   matched grid, not the ~2040-config full pools. `bo_experiment.py:1463` calls that
   restriction "an unnecessary handicap ... and it matters once the candidate pool goes
   off-grid, because this kernel is exactly what drives the Nyström map
   `W = K(X,Z)K(Z,Z)^-1`". So α was tested against a target deliberately starved of the
   off-grid data it exists to interpolate over.
3. **Untested on PD1: α > 0 in arm C**, where the target is fitted on the full pools, and
   **α > 0 with `B` = HyperBO's pre-trained kernel**, the highest-capacity target
   available. The decomposition predicts the latter is the largest of the three doors.

α and the target are only meaningful jointly. Sweeping either alone — which is all that
has ever been done on PD1 — cannot find the interaction. Any comprehensive EM tuning
evaluation must cross them, **and must do it in arm C**, because arm B under-trains the
very kernel that shrinkage blends toward.

**Correction to the handoff's "do not bother" list.** It said "more α tuning (interior
optimum found, exhausted)". That conclusion came from **regression** cells at the default
target. It does not transfer to BO, and it does not cover the α × HyperBO-target cell.

### 27.10 The hybrid sweep cannot be merged with the 2026-08-08 arm-C data: code drift

The arm-C hybrid sweep completed 36/36 (3 priors, no failed shards). Merge validation
against the existing `fullboth*.json` **FAILED on all 72 control comparisons** —
`em_frozen` and `hyperbo_frozen` differ on every shard of every prior, typically within
the first few BO steps. `validate_hybrid_merge.py` gates this and exits non-zero.

**Diagnosis, by elimination rather than assumption.**

| candidate | test | result |
|---|---|---|
| my `--em-canonical-pool` refactor | prior 0 ran entirely *before* that edit and still differs | **cleared** |
| dropping `pacoh_frozen`/`ablr` from `--methods` | same binary, with vs without the extras | **cleared** — both controls bit-identical |
| code drift since 2026-08-08 | remaining explanation | **the cause** |

The second test is the informative one: under the *current* binary, dropping PACOH and
ABLR leaves `em_frozen` and `hyperbo_frozen` byte-for-byte unchanged. The BO loop seeds
explicit local generators from `(held, seed)`, so the method list genuinely cannot move
the trajectories. The divergence is therefore between the code of 2026-08-08 and the code
of today, not between the two method lists.

**Separately verified: the `--em-canonical-pool` refactor is a no-op in arm C.** Re-running
the saved smoke configuration on the current binary reproduces `em_frozen`,
`hyperbo_frozen` **and** `em_additive_hyperbo` bit-identically. That matters for two
reasons: the refactor is safe, and the sweep's priors 0–1 (old binary) are consistent with
prior 2 (which may have rebuilt mid-run), so **the sweep is internally valid**.

**What is lost, and what is not.**

- **Lost:** any comparison that puts hybrid numbers next to `fullboth*.json` numbers —
  including anything that would place the hybrid into §27.3–§27.8's tables.
- **Not lost:** the hybrid question itself. The sweep carries `em_frozen` and
  `hyperbo_frozen` as internal reference arms, so `em_additive_hyperbo*` can be compared
  against them entirely within the sweep, on a single binary. Including those controls
  was what saved it.
- **Open risk:** §27's conclusions rest on the 2026-08-08 data. They remain internally
  consistent, but whether the drift changes the *ordering* of methods is untested. If it
  does, §27 needs re-running on current code before anything is published.

**Rule this establishes.** Every sweep must carry its own reference arms, and no two
sweeps may be compared without a bit-identity check on a shared arm first. Bit-identity
across code versions is not a safe default assumption in this codebase — it has now
failed once, silently, and only an explicit control caught it.

### 27.11 The drift changes numbers, not most conclusions — but one §27.8 row is retracted

Stage A re-ran the arm-C configuration (prior 0, same seeds, same config) on current code
so §27.10's drift could be assessed by its effect on conclusions rather than on bits.

**Where the drift lives.** `vanilla_gp` and `random` are **bit-identical** across the two
codebases — 39.35/39.35 evaluations to 0.01, 0.03973 final regret, to every decimal.
Those are the only two methods with no pre-training. **The drift is confined to the
pre-training path** (EM prior, HyperBO, ABLR), which is consistent with a change to
per-baseline RNG seeding, and rules out the data loader, the BO loop and the metric code.

**Orderings changed on all three metrics**, but almost entirely by reshuffling near-ties:
at the 0.01 target `em_finetuned` moved 39.91 → 39.09 and crossed `vanilla_gp` at 39.35.
Final regret shows a genuine sign flip — `em_finetuned` − `hyperbo_frozen` was +0.0021
(EM worse), now −0.0025 (EM better).

**The verdicts, which are what actually matter — 7 of 8 survive:**

| contrast | target | old | new | |
|---|---|---|---|---|
| `hyperbo_frozen` − `em_finetuned` | 0.01 | HyperBO, z=−6.29 | **tied, z=−1.31** | **CHANGED** |
| `hyperbo_adapt` − `em_finetuned` | 0.01 | HyperBO, z=−6.90 | HyperBO, z=−2.39 | survives |
| `ablr` − `em_finetuned` | 0.01 | ABLR, z=−5.54 | ABLR, z=−4.03 | survives |
| `em_finetuned` − `vanilla_gp` | 0.01 | tied | tied | survives |
| `hyperbo_frozen` − `em_finetuned` | 0.001 | tied | tied | survives |
| `hyperbo_adapt` − `em_finetuned` | 0.001 | HyperBO, z=−3.08 | HyperBO, z=−2.82 | survives |
| `ablr` − `em_finetuned` | 0.001 | ABLR, z=−4.82 | ABLR, z=−6.53 | survives |
| `em_finetuned` − `vanilla_gp` | 0.001 | tied | tied | survives |

**What this means for §27.8.**

- **Survives:** HyperBO and ABLR reach a very good configuration significantly sooner
  than EM on PD1. `hyperbo_adapt` and `ablr` clear |z|>2 at both targets under both
  codebases. The direction is robust to the drift.
- **Survives:** EM is not distinguishable from a from-scratch GP on budget-to-target.
  Tied in all four comparisons, both codebases.
- **RETRACTED:** the `hyperbo_frozen` − `em_finetuned` row at the 0.01 target. It reads
  z=−6.32 in §27.8 and does not reproduce (z=−1.31, tied).
- **Weakened:** every surviving magnitude shrank. §27.8's "2–5 evaluations later" should
  be stated as **1–4** on current code, and the effect sizes in that table should be
  treated as upper bounds.

**Practical rule.** §27's *conclusions* can be cited; §27's *numbers* cannot, until
re-measured on current code. The distinction is now load-bearing, so state which one a
claim depends on.

### 27.12 BLOCKER FIXED: `--em-canonical hyperbo` hung the factorial for 62 hours

The α × target factorial completed its three `sumll` cells (36 shards) and then **hung on
`a0.0_hyperbo` for ~62 hours**. All 12 shards were alive, burning CPU, with no log output
since the line `PD1: seeding EM canonical kernel from a pre-trained HyperBO`.

**Cause.** `_pd1_canonical` called `pretrain_hyperbo` **without `subsample_size` or
`task_batch_size`**, which the main gate's call always passed. Without them HyperBO
pre-trains with an exact-GP MLL over the **full ~2040-config pools × 22 tasks** — cubic
per task per step, at 25 000 steps. That is not slow, it is intractable.

The bug predates this work: `_pd1_canonical` never passed those arguments. It had simply
never been executed, because **no PD1 run had ever set `--em-canonical hyperbo`** (every
recorded config shows `em_canonical: None`). The factorial is the first thing to exercise
that path, and it exposed the defect immediately.

**Fixed** by passing `subsample_size=args.meta_subsample` and
`task_batch_size=args.meta_task_batch`, matching the main gate.

**State of `raw/armc_factorial/`: 3 of 6 cells, 36 of 72 shards.** Present and valid:
`a0.0_sumll`, `a0.1_sumll`, `a0.3_sumll`. Missing: all three `hyperbo`-target cells —
which are the *interesting half*, since §27.9 showed the target only matters when α > 0
and the HyperBO kernel is the highest-capacity target available. **The factorial's central
question is still unanswered.**

**Lesson.** A code path that no experiment has ever run is untested code, whatever its
age. Before committing an overnight slot to a new flag combination, run one shard of it
to completion first — a 10-minute check would have saved 62 hours of wall-clock and a
saturated box.

### 27.13 LCBench shows NO code drift — and my drift check was misconfigured

Stage C re-ran two `bo_multisplit` cells on current code to test whether §26.1/§26.2 —
the surviving headline claims — are reproducible.

**Result: 7 of 9 methods are bit-identical on both cells**, including every EM variant,
both HyperBO variants, `vanilla_gp` and `random`. Only `ablr` and `ablr_adapt` differ.

**And that difference is my error, not drift.** Comparing the two configs:

| key | bo_multisplit | my re-run |
|---|---|---|
| `ablr_iters` | 25000 | **10000** |
| `ablr_per_task_precision` | False | **True** |

I copied ABLR's flags from the PD1 runner instead of from `bo_multisplit`'s recorded
config, while the script header claimed they were copied verbatim. ABLR re-seeds
independently (`pretrain_seed * 1009 + 17`), so dropping PACOH cannot explain it; the
changed budget and precision setting fully do. Script corrected.

**Conclusions.**

- **§26.1 and §26.2 are bit-reproducible on current code** for EM, HyperBO, `vanilla_gp`
  and `random`. The drift that invalidated the PD1 comparison (§27.10) **does not affect
  LCBench**, so the project's headline claims stand on reproducible data.
- **ABLR drift remains untested on LCBench.** §26.1's "EM significantly beats ABLR
  (z=−3.13)" involves a method whose reproducibility is still unknown. Re-run with
  `--ablr-iters 25000` and no per-task precision to close it.
- The drift is therefore **PD1-specific**, which narrows §27.10's suspect list
  considerably: it points at the PD1 leave-one-out pre-training path rather than at
  anything shared with LCBench.

**Lesson, and it is the same one twice in this section.** Both defects came from copying
a config or a call-signature from a neighbouring context without diffing it against the
thing being reproduced. A reproduction run must diff its full recorded config against the
original before its result is interpreted — `analyze_pd1pool.py`-style config dumps make
this a one-liner.

### 27.14 The additive hybrid is the first thing to significantly improve EM on PD1

The arm-C hybrid sweep (3 priors, 253 runs) analysed within-sweep only, since §27.10
established it cannot be compared against `fullboth*.json`. It carries `em_frozen` and
`hyperbo_frozen` as internal reference arms. Primary metric, fixed-task estimand.

**Evaluations to reach |regret| ≤ 0.01:**

| method | mean evals |
|---|---|
| `hyperbo_frozen` | **36.76** |
| `em_additive_hyperbo` | 38.38 |
| `em_additive_hyperbo_frozen` | 38.43 |
| `em_frozen` | 39.81 |

| contrast | Δ evals | z | verdict |
|---|---|---|---|
| `em_additive_hyperbo` − `em_frozen` | −1.43 | **−2.33** | **hybrid better, wins 18/23 tasks** |
| `em_additive_hyperbo_frozen` − `em_frozen` | −1.38 | **−2.35** | **hybrid better** |
| `em_additive_hyperbo` − `hyperbo_frozen` | +1.62 | **+2.33** | HyperBO still better |
| `em_frozen` − `hyperbo_frozen` | +3.06 | **+4.47** | HyperBO better |

**The hybrid closes roughly half the gap.** Plain EM trails `hyperbo_frozen` by 3.06
evaluations; the hybrid trails by 1.62. Both hybrid variants significantly beat plain EM —
**the first intervention in this project to significantly improve EM on PD1.** Neither
overtakes standalone HyperBO.

**At the tighter 0.001 target the picture changes, informatively:**

| contrast | Δ evals | z | verdict |
|---|---|---|---|
| `em_additive_hyperbo_frozen` − `em_frozen` | −1.24 | **−2.03** | **frozen hybrid better** |
| `em_additive_hyperbo` − `em_frozen` | +0.92 | +1.37 | tied (nominally worse) |
| `em_frozen` − `hyperbo_frozen` | +0.37 | +0.50 | **tied — no gap to close here** |

**The FROZEN variant is the robust one**: it beats plain EM at both targets, while the
fitted variant helps only at 0.01 and is nominally worse at 0.001. That is exactly what
§26.3 predicted from LCBench regression — *"fitted weight wins on-grid, frozen off-grid"* —
and PD1 arm C is the off-grid regime. A prediction made on a different benchmark, a
different metric and a different task family holds here, which is the strongest
corroboration any mechanism in this project has received.

**Recommendation.** `em_additive_hyperbo_frozen` should be the default EM variant for
off-grid work. It is the only configuration that significantly improves on plain EM at
both targets.

**Caveats.** Within-sweep only — do not place these numbers beside §27.3–§27.8's.
`hyperbo_frozen` is still ahead at 0.01, so this narrows the gap rather than closing it,
and it costs an extra HyperBO pre-training to obtain.

### 27.15 The hybrid IS state-of-the-art on LCBench BO — the PD1 shortfall is off-grid-specific

Investigating why §27.14's hybrid improves EM on PD1 but does not reach standalone
HyperBO, when §26.3 reported the hybrid beating everything on LCBench.

**Finding 1 — the training-stochasticity confound does not exist for the additive base.**
In the LOO path there is exactly ONE `pretrain_hyperbo` call, and
`HYPERBO_BASE_KERNEL = hyperbo_kernel_from_prior(hyperbo_prior)` derives from that same
object. `hyperbo_kernel_from_prior` rebuilds the modules and `load_state_dict`s the
pre-trained weights, then freezes them — effectively the deepcopy one would want.
**`hyperbo_frozen` and `em_additive_hyperbo` already share one pre-trained HyperBO**, so
their comparison is not confounded by separate training draws.

**Finding 2 — but that sharing does NOT extend to the canonical kernel.**
`_pd1_canonical` calls `pretrain_hyperbo` **again**, independently. So under
`--em-canonical hyperbo` the canonical kernel and `hyperbo_frozen` come from *different*
training draws. **The α × target factorial's three `hyperbo` cells carry this confound**,
which conflates "different kernel role" with "different random draw" — and it doubles the
cost of the most expensive component. Fix: reuse the single pre-trained prior.

**Finding 3 — §26.3 is REGRESSION; the LCBench BO cells were unanalysed.**
`hb_hybrid/bo_lcb_*` (4 cells, 2 canonical × 2 priors, 36 runs each) had **zero mentions**
in this document. Analysed now, primary metric, fixed-task estimand, canonical=sumll:

| method | evals to target |
|---|---|
| **`em_additive_hyperbo`** | **8.89** |
| `em_noisefit` | 9.97 |
| `em_frozen` | 10.42 |
| `em_additive_hyperbo_frozen` | 11.75 |
| `hyperbo_frozen` | 14.43 |
| `ablr` | 15.22 |
| `vanilla_gp` | 20.64 |
| `random` | 37.17 |

| contrast | Δ evals | z | verdict |
|---|---|---|---|
| `em_additive_hyperbo` − `hyperbo_frozen` | −5.54 | **−7.50** | **hybrid better, 5/6 tasks** |
| `em_additive_hyperbo_frozen` − `hyperbo_frozen` | −2.68 | **−3.25** | hybrid better |
| `em_additive_hyperbo` − `em_frozen` | −1.53 | **−2.58** | hybrid better |

**On LCBench BO the hybrid is the best method by a wide margin, beating standalone
HyperBO by 5.5 evaluations.** The recollection that "a variant of EM + HyperBO
outperformed everything" is correct, and now verified on BO rather than only regression.

*(The 0.01 and 0.001 tables are identical, which looks like a bug and is not: LCBench
regret is either exactly 0 — 33/36 runs find the pool optimum — or ≥0.01, with nothing in
between, so both targets are crossed at the same step. Checked before reporting.)*

**So the contrast is benchmark-specific, not a confound:**

| | LCBench BO | PD1 arm C BO |
|---|---|---|
| candidates | **on-grid** (200-config pool = the prior's grid) | **off-grid** (~2040 vs 400 matched) |
| hybrid vs standalone HyperBO | **−5.54 evals (wins)** | +1.62 evals (loses) |

**Leading explanation: the hybrid transfers HyperBO's KERNEL but not its MEAN.**
`hyperbo_frozen` builds `HyperBOModel.from_pretrained` with a meta-learned *linear mean*;
`em_additive_hyperbo` builds `EMEmpiricalGaussianProcess.from_pretrained` with EM's
*empirical* mean plus the additive base kernel. On-grid the empirical mean is exact at
every candidate, so losing HyperBO's mean costs nothing and the added kernel is pure gain.
Off-grid the empirical mean must be interpolated to points it was never defined on — and
insight 38 already established that the prior mean drives budget-to-target ("ABLR's zero
prior mean gives no informed first pick"). That predicts exactly the observed pattern:
hybrid wins on-grid, falls short off-grid.

**The untested lever this implies: give the hybrid HyperBO's learned MEAN as well as its
kernel.** No variant has ever done this — every `em_additive_*` transfers only covariance.
It is the most direct attack on the off-grid deficit and is cheap, since the mean is
already pre-trained and sitting in the same prior object.

### 27.16 The EM model has four never-exposed knobs; two look immediately promising

Prompted by the observation that EM's base mean was always a `ConstantMean`. Auditing
`pretrain_em_prior`'s full signature against what `bo_experiment.py` exposed:

| knob | status before 2026-08-17 | now |
|---|---|---|
| `covariance_shrinkage` (α) | explored (§27.9) | — |
| `mean_module` | **only `ConstantMean` ever** | `--em-base-mean {constant,linear}` |
| `use_covar_prior` | **never exposed** | `--use-covar-prior` |
| `use_mean_prior` | **never exposed** | `--use-mean-prior` |
| `iw_nu` (Inverse-Wishart dof) | **never exposed** | `--iw-nu` |
| `init_mode` | **never exposed** (always `"kernel"`) | `--em-init-mode {kernel,naive}` |

Why the mean matters: EM interpolates as `mu(X) = m(X) + W @ delta_mu`, so with a
`ConstantMean` the extrapolated mean is **flat** away from the inducing grid. That is a
plausible off-grid handicap and it has never been varied.

**Non-no-op screen** (`em_frozen`, PD1 arm C, 1 shard; posterior NLL, since a 5-iteration
trajectory is too coarse to flip a discrete argmax):

| knob | trajectory moved | posterior NLL | verdict |
|---|---|---|---|
| control | — | 0.952804 | baseline |
| `--em-base-mean linear` | yes | **0.846790** | LIVE, better |
| `--use-covar-prior` | yes | **0.589271** | LIVE, **much better** |
| `--em-init-mode naive` | no | 0.937054 | LIVE (posterior only) |
| `--iw-nu 50` | no | 0.952804 | **SILENT NO-OP** |

**`--iw-nu` alone does nothing** — occurrence #7. `pretrain_em_prior` reads `iw_nu` only
inside `if use_covar_prior:`, and valid values satisfy `nu > M-1` (M=400 on PD1's matched
grid), so `--iw-nu 50` would have *raised* had the prior been enabled. A guard now rejects
`--iw-nu` without `--use-covar-prior` rather than accepting a flag that cannot act.

**The single-shard NLL improvements are suggestive, not results.** One shard, one seed, no
pairing — they say the knobs are live and worth including in the matrix, nothing more.
`--use-covar-prior` in particular deserves attention: it is a principled Bayesian
regulariser of the EM covariance, and §27.9 found the ad-hoc `--em-shrinkage` blend does
*not* help on PD1. If shrinkage was the right idea implemented the wrong way, this is
where that would show up.

**Also verified: the mean transfer moves the posterior.** §27.15's `--em-mean hyperbo`
left `em_frozen`'s BO trajectory identical, which looked like a no-op. Its posterior NLL
moves (0.725335 → 0.781655), so the transfer is live and the identical trajectory was
smoke coarseness. Posterior NLL, not the trajectory, is the right non-no-op probe for
mean-only changes.

### 27.17 RETRACTION: §27.14's hybrid result does not survive correct pooling

The 2026-08-17 adversarial review found that `PooledStudy` weighted priors by **run
count**, not equally. Arm C's p0 carries 5 seeds (115 runs) against p1/p2's 3 (69 each),
so p0 held **45.5%** of the weight instead of 33.3%. §27.3 established that priors
disagree on ordering, so this is a real bias, not a rounding detail. A second defect
counted **ties as wins** for the first method in the lower-is-better branch only.

Both are fixed (`PooledStudy.wmean` / `task_wmean` equalise priors overall and within
each task; ties are counted and reported separately). Re-derived, primary metric,
fixed-task, target 0.01:

| contrast | §27.14 as published | corrected | |
|---|---|---|---|
| `em_additive_hyperbo` − `em_frozen` | −1.43, z=**−2.33**, 18/23 wins | −1.12, z=**−1.83**, 11/23 (6 ties) | **now TIED** |
| `em_additive_hyperbo_frozen` − `em_frozen` | −1.38, z=**−2.35**, 15/23 | −1.01, z=**−1.71**, 9/23 (6 ties) | **now TIED** |
| `em_additive_hyperbo` − `hyperbo_frozen` | +1.62, z=+2.33 | +2.06, z=**+2.97** | HyperBO better (strengthened) |
| `em_frozen` − `hyperbo_frozen` | +3.06, z=+4.47 | +3.19, z=**+4.66** | HyperBO better (strengthened) |

**RETRACTED: "the additive hybrid is the first intervention to significantly improve EM
on PD1."** It does not. Both variants are now tied with plain EM at the 0.01 target. The
apparent significance came from over-weighting one prior and from crediting ties as wins
— the inflated "18/23 tasks" becomes 11/23 with 6 ties once ties are counted honestly.

**What survives, and is now stronger:** `hyperbo_frozen` beats every EM variant on PD1
budget-to-target, z=+2.97 to +4.66. The gap the hybrid was supposed to close is real and
the hybrid does not close it.

**Unaffected:** §27.15's LCBench BO result (single-prior cells, no pooling) and §27.11 /
§27.13 (single prior each). The bias only ever affected multi-prior pooled statistics,
which is §27.14 and §27.3–§27.8.

**§27.3–§27.8 are now SUSPECT and must be re-derived.** They use the same pooled
estimator over `fullboth*` (3 priors, same 5/3/3 seed imbalance). Their conclusions
already carried the §27.11 caveat that numbers do not reproduce; they now additionally
carry a known estimator bias in the same direction.

**Lesson.** Every pooled statistic in this project silently assumed equal seed counts
across priors. Nothing checked it, and the imbalance was visible on disk the whole time
(115/69/69). A one-line assertion at pool construction would have caught it two weeks ago;
there is now a test that does exactly that.

### 27.18 §27.9's shrinkage conclusion is SUSPECT: the target was on the wrong scale

The round-3 adversarial review (`ADVERSARIAL_REVIEW_ROUND3_2026-08-17.md`, D2) found that
`build_shared_gp_model_list` constructed its `SingleTaskGP`s without an
`outcome_transform`, so BoTorch's default applied a **per-task `Standardize`**. Every
task's `Y` was re-standardised inside its own GP before the shared MLL was evaluated,
while `pretrain_em_prior` consumes **globally**-standardised `Y`.

So `covar_s.outputscale` was calibrated to unit per-task variance on a corpus whose
per-task variance is *not* 1 (whenever between-task means differ, which is the default
non-`--per-task-standardize` path). That makes `Sigma_init = K(Z,Z)`, the EM kernel
initialisation, **and the `--em-shrinkage` trace-matched target `s·B`** all wrongly
scaled.

**Consequence.** §27.9 concluded that shrinkage toward the canonical kernel does not help
on PD1, and the α screen in §27.16 was read the same way. Both were measured with a
mis-scaled target. A shrinkage target at the wrong scale is expected to hurt regardless of
whether the mechanism is sound, so **the finding does not distinguish "shrinkage does not
help" from "the target was mis-scaled".**

**Status: SUSPECT, not retracted.** The α=0 cells are unaffected (at α=0 the target drops
out entirely, §27.9), so any conclusion resting only on α=0 still stands. Every claim that
compares α>0 against α=0 must be re-derived on the fixed code.

This also raises the prior probability that shrinkage *does* help once correctly scaled,
which makes the α × target factorial more interesting rather than less — and it is
another reason the whole matrix is being re-run rather than reusing anything.

### 27.19 CORRECTION to §27.18: D2 hits LCBench, not PD1

§27.18 flagged §27.9's PD1 shrinkage conclusion as suspect because of the hidden per-task
`Standardize` (round-3 D2). **That attribution was wrong, and checking the recorded
configs shows why.**

`--per-task-standardize` rescales each pre-training task to exactly zero mean and unit
variance **before** `build_shared_kernel` ever sees it. A subsequent per-task
`Standardize` then computes mean≈0, std≈1 and is a **near-identity**. So D2 is inert
wherever that flag is set.

What the runs on disk actually used:

| run | `per_task_standardize` | D2 impact |
|---|---|---|
| `pd1pool/fullboth*` (§27.3–§27.8) | **True** | inert |
| `pd1pool/hybrid_*` (§27.14, §27.17) | **True** | inert |
| `armc_factorial/*` (α screen, §27.16) | **True** | inert |
| `drift_check/*` (§27.11) | **True** | inert |
| `bo_multisplit/*` (§26.1, §26.2) | **False** | **AFFECTED** |

**Corrected conclusions.**

- **§27.9 / §27.16 are NOT suspect from D2.** Every PD1 run standardises per task, so the
  shrinkage target was on the right scale. "Shrinkage does not help on PD1" stands on this
  ground. §27.18's suspect flag is withdrawn.
- **The LCBench sections are the affected ones.** `bo_multisplit` (§26.1, §26.2) and the
  `bo_diagnose` regression cells (§26.3) run on globally-standardised targets, so their
  shared kernel *was* fitted against per-task-restandardised data while the EM prior
  consumed the global scale. Those are the numbers to re-derive.
- D2 remains a real defect and the fix stands; only its blast radius was mis-stated.

**Lesson, and it is a repeat.** §27.18 inherited a severity claim from a review instead of
checking it against the configs on disk — the same mistake as §27.13, where a "drift"
finding turned out to be a config mismatch. **A severity claim is a claim; check it
against `config` in the recorded JSON before propagating it into the handoff.**

### 27.20 With between-prior variance included, almost nothing on PD1 is significant

Round-4 S1 found that the fixed-task SE omitted the **between-prior** variance term.
Seeds within a prior share the same EM prior, HyperBO prior, pre-training data and
canonical kernel — they differ only in the initial design — so `s_p²/n_p` estimated
initial-design noise *only*. The variance of an average over P priors is

    Var = (1/P²) Σ_p s_p²/n_p  +  σ²_between-prior / P

and the second term was entirely absent. §27.3 established that priors **disagree on
ordering**, which is precisely why that term is the one that must not be dropped.

Arm-C hybrid sweep, target 0.01, recomputed with the corrected SE:

| contrast | effect | old z | **corrected z** | verdict |
|---|---|---|---|---|
| `em_additive_hyperbo` − `em_frozen` | −1.12 | −1.66 | **−1.28** | tied |
| `em_additive_hyperbo` − `hyperbo_frozen` | +2.06 | **+2.97** | **+1.72** | **now TIED** |
| `em_frozen` − `hyperbo_frozen` | +3.19 | **+4.66** | **+2.63** | still HyperBO |

**Two of three contrasts are now ties.** Only `em_frozen` vs `hyperbo_frozen` clears
|z|>2, and it does so at 2.63 rather than 4.66.

**And it does not survive the LOO-dependence caveat.** Leave-one-out folds share 21 of 22
pre-training tasks, so the per-task differences are positively equicorrelated and
`sqrt(Σ Var_t)/T` is anti-conservative (round-4 S2). Every contrast now prints what it
would be under modest correlation:

    em_frozen - hyperbo_frozen:  z = +2.63  ->  +1.82 at rho=0.05,  +1.27 at rho=0.15

So even the last surviving PD1 claim is **not robust to a correlation of 0.05**, which is
a small number for folds sharing 95% of their pre-training data.

**Honest summary of the PD1 arm-C evidence as it now stands.** With three priors, 23
tasks and correlated folds, this sweep can support *directional* statements and cannot
support significance claims. §27.14's retraction (§27.17) was the first step; this is the
second. The right response is more priors — between-prior variance enters as `σ²/P`, so
it is the only term more replicates actually reduce — not more seeds, which only shrink
the term that was already small.

**This changes what the experiment matrix must do.** Stage 3's replication plan (≥3
priors) is now the *minimum*, not a safety margin, and the stage-1/2 screens should be
read as ranking devices only, exactly as `EXPERIMENT_PLAN.md` already specifies.

### 27.21 CORRECTION to §27.20: the between-prior fix double-counted, inflating every SE

§27.20 added a between-prior variance term *on top of* the within term. Round-6 S1 showed
that is wrong. With `ȳ_p = θ_t + b_p + e_p`:

    E[ s²(ȳ_1..ȳ_P) ] = (1/P) Σ_p Var(ȳ_p)   ⇒   E[ s²/P ] = Var(θ̂_t)   exactly

**The spread of the per-prior means already IS the whole variance**, because each `ȳ_p`
is itself a noisy mean. Adding a separate within term counted it twice and inflated the
SE by up to √2 at arm C's n=(5,3,3). Corrected: `task_wvar` returns the between-prior
spread alone for P>1, falling back to the within form only at P=1.

Arm C, target 0.01, all three estimators side by side:

| contrast | pre-§27.20 (within only) | §27.20 (double-counted) | **corrected** |
|---|---|---|---|
| `em_additive_hyperbo` − `em_frozen` | −1.66 | −1.28 | **−2.01, hybrid better** |
| `em_additive_hyperbo_frozen` − `em_frozen` | −1.80 | −1.06 | −1.31, tied |
| `em_additive_hyperbo` − `hyperbo_frozen` | +2.97 | +1.72 | **+2.03, HyperBO better** |
| `em_frozen` − `hyperbo_frozen` | +4.66 | +2.63 | **+3.06, HyperBO better** |

**§27.20's "two of three contrasts became ties" is withdrawn.** It was an artefact of the
inflated SE. The corrected picture sits between the two earlier ones, which is what an
unbiased estimator between an under- and an over-estimate should look like.

**What now stands, with its caveat attached:**

- `hyperbo_frozen` beats plain EM on PD1 budget-to-target: **z=+3.06**, but **+2.11 at
  ρ=0.05** under the LOO-dependence caveat. Survives a small correlation, marginally.
- `hyperbo_frozen` beats the additive hybrid: **z=+2.03**, which does **not** survive
  ρ=0.05. Directional only.
- The additive hybrid beats plain EM: **z=−2.01**, same caveat. §27.17's retraction of
  §27.14 stands in substance — the effect is marginal and not robust — but the sign is
  now consistently in the hybrid's favour across all three estimators.

**Three different SEs in three rounds, on the same data.** The numbers moved because the
estimator was wrong twice, not because the data changed. That is an argument for
reporting the estimator alongside every z from here on, which `fixed_effects_test` now
does implicitly by printing the LOO caveat on every line.

### 27.22 With correct degrees of freedom, NOTHING on PD1 arm C is significant

Round-6 S3: the fixed-task SE is dominated by the between-prior spread, which is
estimated from **P prior draws**. All 23 per-task terms come from the same P draws, so
that component carries **P−1 degrees of freedom** — 2 at the project's usual P=3, not the
~46 a per-task count suggests. The threshold is therefore **t(2) = 4.30**, not 1.96.

Arm C, target 0.01, corrected SE against the corrected threshold:

| contrast | t | crit t(2) | verdict |
|---|---|---|---|
| `em_frozen` − `hyperbo_frozen` | +3.06 | 4.30 | **tied** |
| `em_additive_hyperbo` − `hyperbo_frozen` | +2.03 | 4.30 | tied |
| `em_additive_hyperbo` − `em_frozen` | −2.01 | 4.30 | tied |

**Every PD1 arm-C contrast is now a tie.** The last surviving significance claim —
`hyperbo_frozen` beats plain EM — does not clear a t(2) threshold, and it did not survive
the LOO-dependence caveat either (+2.11 at ρ=0.05).

**This is not a new measurement.** The data has not changed since §27.14; the *estimator*
has been wrong in three successive ways — within-only (too small), within+between
(double-counted), and then the right variance against the wrong critical value. §27.22 is
the first version where both the SE and the threshold match the design.

**The honest state of the PD1 evidence: directional only.** With 3 priors and 23 tasks,
arm C can rank methods and cannot certify differences. HyperBO is ahead of every EM
variant on budget-to-target in every estimator tried, which is worth stating as a
direction; it is not a significant result and must not be written as one.

**What would change that.** Between-prior variance enters as `σ²_B/P`, and the threshold
relaxes fast in P: t(2)=4.30, t(4)=2.78, t(9)=2.26. Going from 3 priors to 10 buys both a
smaller SE and a threshold near the normal value. **More priors, not more seeds** — seeds
only shrink the within component, which the corrected estimator shows was never the
binding term. `EXPERIMENT_PLAN.md`'s stage 3 should be re-scoped from "≥3 priors" to
"≥10 priors for any claim that needs significance".

### 27.23 The end-to-end smoke found what seven review rounds and 83 tests did not

`results/smoke_pipeline.sh` runs the real pipeline — `run_cell_queue` → `summarize_stage`
— on one BO regime and one regression regime at toy settings, and asserts on artifacts.

First run, it found a defect no review round or unit test had caught: **`run_cell_queue`
writes `results/raw/$STUDY` while `summarize_stage` hardcoded `results/raw/v2/<stage>`.**
Both halves are individually correct; only their *coupling* was wrong, and the
orchestrator made them agree by convention rather than by construction. A mismatch
summarises nothing, the orchestrator tolerates that with `|| true`, and the `.DONE` gate
then fails forever on the missing SUMMARY.json.

It also **confirmed two round-7 fixes work in production**, which reading could not: the
`bo_diagnose --pretrain-seed` path (R7-1 said 96 cells would crash-loop) completed 2/2,
and `by_n_obs` metrics (R7-2) now survive into the summary alongside `prior_mean`.

**Ten minutes of execution beat seven rounds of reading.** The reviews were valuable —
125 defects — but every one of the guaranteed criticals in rounds 6 and 7 was an
integration failure that running the pipeline once would have surfaced immediately.
Run the smoke before any launch, and after any change to the runner or summariser.

### 27.24 LCBench divergence: our stack never filtered, and it should not start

Prompted by a parallel stack (tutorial section G) finding that its LCBench curve filter
**drops an entire dataset when only a few curves diverge**. Two questions: does that
affect our results, and should we adopt per-curve filtering?

**Fact 1 — our workstream applies NO LCBench filtering at all.** `load_pool`
(`bo_experiment.py`) loads all 35 datasets × 2000 configs and takes final-epoch accuracy,
with no NaN check, no divergence predicate, no dataset exclusion. `bo_diagnose` imports
the *same function object*. So **no existing result in this workstream is affected by the
dataset-dropping bug** — there was nothing to drop.

**Fact 2 — measured divergence in LCBench (all 35 datasets, 70 000 curves):**

| quantity | count | share |
|---|---|---|
| curves with any NaN | **0** | 0.000% |
| curves with NaN at final epoch | **0** | 0.000% |
| perfectly flat curves (no learning across 50 epochs) | **1261** | 1.80% |

So "diverged" in LCBench is **not** NaN — it is a *flat* curve, a run that never learned.
Rates are very uneven: `vehicle` 88/2000 (4.4%), `jasmine` 49, `kc1` 42, `kr-vs-kp` 40,
down to `helena` 1.

**Fact 3 — the dataset-level rule is catastrophic here.** "Drop the dataset if any curve
diverged" would remove **26 of 35 datasets — 74% of the benchmark — to excise 1.8% of
curves.** The parallel agent's diagnosis is exactly right, and per-curve is strictly
better than per-dataset wherever filtering is warranted at all.

**Decision: do NOT filter for the BO experiments. Three independent reasons.**

1. **It changes the task.** BO here searches a *finite pool* of 2000 configurations. A
   flat/diverged config is a legitimate member of that pool — practitioners do propose
   configurations that fail. Deleting them makes the benchmark easier and less
   representative, and inflates every method's apparent performance.
2. **It is differentially biased toward our own hypothesis, which is the dangerous part.**
   §23/§27 document that EM's calibration collapses precisely on the divergence tail
   (PD1's bimodal error distribution is the stated root cause). Removing that tail would
   selectively help the method under test. Filtering here would be a way to confirm our
   own hypothesis by deleting its counter-evidence.
3. **It breaks the PD1 comparison.** `pd1_full_data.py` *deliberately keeps* diverged
   trials at error 1.0, with the recorded rationale that dropping them "would silently
   delete precisely the divergence tail that makes PD1 hard". Filtering LCBench while
   keeping PD1 would make the suites non-comparable — and cross-suite comparison is the
   entire point of stage 4.1.

**But divergence is scientifically valuable — as a MODERATOR, not a preprocessing step.**
The flat-curve rate varies 88× across datasets. That is a free, pre-registerable covariate:
*does the EM-vs-HyperBO gap depend on how much of the pool is degenerate?* If EM's deficit
concentrates on high-divergence datasets, that localises the mechanism to the tail and is a
far stronger claim than any average. It costs **zero extra compute** — the divergence
fraction is computable from data already on disk, and the per-task effects already exist.

**Where filtering IS legitimate: curve modelling.** A model of *learning curves*
(`curve_experiment.py`) has a real argument for excluding non-curves. That is a different
experiment from ours, which models final accuracy, and it is where the parallel agent's
per-curve fix belongs.

**Guardrail added to the plan.** Any future filter must be per-curve, must be recorded in
the cell config, and must never remove a whole dataset silently — a rule the 26-of-35
result makes concrete.

### 27.25 CORRECTION to §27.24: LCBench divergence is real — it lives in the LOSS metric

§27.24 reported "0 NaN, 0 extreme values, divergence = flat curves" and used that to argue
no filtering is needed. **That measurement was taken on `Train/val_accuracy` only, which
is bounded [0, 100] and therefore cannot exhibit divergence at all.** The parallel stack's
report of "truly diverged curves with extremely large values" is correct; I measured a
metric incapable of showing it.

Measured across all 12 LCBench metrics, 70 000 curves each:

| metric | \|v\|>1e3 | \|v\|>1e6 | largest \|v\| | datasets hit |
|---|---|---|---|---|
| **`Train/loss`** | **143 (0.204%)** | **89 (0.127%)** | **`inf`** (in `shuttle`) | **5/35** |
| `Train/test_cross_entropy` | 0 | 0 | 5.87 | 0/35 |
| `Train/val_accuracy` (ours) | 0 | 0 | 100 | 0/35 |
| all others | 0 | 0 | bounded | 0/35 |

So **89 curves reach \|loss\| > 10⁶ and at least one is literally `inf`.** Divergence in
LCBench is a *loss* phenomenon, invisible in accuracy.

**Does the §27.24 conclusion survive? Yes — but for a different and better reason.**

Not "there is no divergence" (false), but: **our objective is `Train/val_accuracy`, and a
run whose loss diverged to `inf` still has a finite, bounded, genuinely-bad accuracy.** The
accuracy metric maps the divergence onto a low score, which is exactly the behaviour we
want from an objective — the config is bad, and BO should learn that. Nothing propagates
`inf` into our pipeline; the earlier probe confirmed 0 non-finite values in the accuracy
tensor, and that part stands.

**What changes:**

1. **Any future work on loss curves must filter or clamp**, per-curve. `curve_experiment.py`
   currently uses accuracy and is safe; a loss-curve variant would hit `inf` on `shuttle`
   and produce NaN gradients immediately.
2. **The parallel stack's per-curve fix is right and should be adopted there.** Dropping a
   dataset because 89 of its 2000 loss curves blew up is still the wrong granularity —
   `shuttle` would be removed entirely.
3. **§27.24's headline number (1.80% flat curves) is not the divergence rate.** Flat
   accuracy curves and diverged losses are different phenomena; 0.204% of curves have
   \|loss\| > 1e3.

**Lesson, and it is the same one as §27.13 and §27.19.** I answered "is there divergence?"
by measuring the one metric that cannot express it, and reported the null result as
settling the question. Check that the instrument can register the effect before quoting a
zero.

### 27.26 Variance allocation: σ²_within often DOMINATES σ²_between, and priors are still the better buy

Prompted by asking why priors had unequal seed counts (5/3/3). That imbalance was inherited
— `run_hybrid_armc.sh` says "that asymmetry is in the existing data" — because arm C grew
incrementally: prior 0 first at 5 seeds, priors 1–2 added later as cheaper replicates. It
is **not** present in the new matrix, where `--priors` varies only `--pretrain-seed` and
every cell inherits the same `--n-seeds`.

But measuring the two components (arm-C hybrid sweep, target 0.01) contradicts an
assumption §27.21 stated as fact:

| method | σ²_W (within prior) | σ²_B (between priors) | ratio B/W |
|---|---|---|---|
| `em_additive_hyperbo` | 48.6 | **0.0** | 0.00 |
| `em_frozen` | 51.7 | 7.0 | 0.14 |
| `em_additive_hyperbo_frozen` | 28.4 | 12.2 | 0.43 |
| `hyperbo_frozen` | 43.3 | 37.3 | 0.86 |

**§27.21's claim that "the between-prior spread is the dominant variance component" is not
supported.** It dominates for `hyperbo_frozen` and is a *minority* of the total for the EM
variants. With P=3 these estimates carry 2 df and the σ²_B≈0 entry is clamped at zero, so
treat the ratios as indicative, not precise.

**The allocation result holds regardless, and is worth acting on.** For a fixed budget of
B = P·n runs per task:

    Var(θ̂) = σ²_B/P + σ²_W/(P·n) = σ²_B/P + σ²_W/B

The second term depends **only on the total budget**; the first depends **only on the
number of priors**. So at fixed compute, more priors is free variance reduction, and n=1 is
optimal. At B=9:

| method | P=3, n=3 | P=9, n=1 | reduction |
|---|---|---|---|
| `em_frozen` | 8.07 | 6.52 | 19% |
| `hyperbo_frozen` | 17.24 | 8.95 | **48%** |

**Recommendation for the matrix: `--priors 9 --n-seeds 1` instead of `--priors 3
--n-seeds 3`.** Identical compute, strictly lower variance for every method, and it lifts
the t-critical value from t(2)=4.30 to t(8)=2.31 — which is the difference between a design
that cannot clear its own significance bar (§27.21) and one that can.

### 27.27 FUTURE STUDY: the diverged loss metrics are auxiliary signal, not noise

§27.25 established that LCBench divergence lives in `Train/loss` (89 curves > 10⁶, at
least one `inf`) and is **invisible** in the `Train/val_accuracy` we optimise. That framing
treats the loss metric as a nuisance to be avoided. **The opposite framing is the
interesting one.**

LCBench records 12 metrics per configuration, of which we use exactly one:

    Train/loss, Train/lr, Train/test_balanced_accuracy, Train/test_cross_entropy,
    Train/test_result, Train/train_accuracy, Train/train_balanced_accuracy,
    Train/train_cross_entropy, Train/val_accuracy ← ours, Train/val_balanced_accuracy,
    Train/val_cross_entropy, time

**The hypothesis.** A configuration whose loss diverges is *identifiable as bad from the
loss alone*, often earlier and more sharply than from accuracy — accuracy saturates near
chance and compresses the signal, while loss blows up by six orders of magnitude. A
**multi-output** model over (accuracy, loss, cross-entropy, …) could therefore predict
accuracy better than a single-output model, precisely on the degenerate tail where §23/§27
document EM's calibration collapsing.

**Why this is a natural fit for the empirical-GP line of work specifically.** The EM
empirical prior estimates a full covariance across the inducing set. Extending it across
*outputs* is the same machinery one dimension up, and the auxiliary metrics are free —
they are already in the parquet, no new evaluations needed. The MultiOutput Empirical 1D GP
already in the OSS stack is the obvious starting point.

**Concrete design when someone picks this up:**

1. Targets: `val_accuracy` (the objective) + `Train/loss` and `Train/val_cross_entropy`
   (auxiliaries). Clamp or log-transform the loss first — `inf` is present and will produce
   NaN gradients immediately.
2. Baseline: the same EM/HyperBO variants on accuracy alone, which is the current study.
3. Metric: budget-to-target as always, but **stratified by `p_div`** (§27.26's moderator),
   because the whole hypothesis is that the gain concentrates on degenerate-heavy tasks.
4. Watch for leakage: the auxiliary metrics must be observed only for configurations the
   BO loop has actually evaluated, never for the candidate pool.

**Why it is not in the current matrix.** It changes the model class rather than a
hyperparameter, so it belongs in a separate study with its own baselines. Recording it here
so the observation is not lost — it emerged only because a divergence question forced a
look at metrics we had never loaded.

### 27.28 FIRST v2 RESULT: stage-0 LCBench baselines, and the 9-prior design earns its keep

`stage0_lcb_bo` complete: 9 cells (1 baseline config × 9 priors), 0 errors, 0 incomplete,
`n_priors = 9`, 54 runs/method (6 eval datasets × 1 seed × 9 priors). Wall clock **3 h 41 m**
for 9 cells at `CELLS_PARALLEL=3`, i.e. ~74 min/cell — better than the 2.5–3 h §0 feared,
worse than the 16.7 min PD1 cell, exactly because LCBench runs one serial shard.

**Pooled budget-to-target @0.01 (lower is better), and final regret:**

| method | budget@0.01 | final regret |
|---|---|---|
| `em_finetuned` | **9.44** | 0.0232 |
| `em_noisefit` | 9.81 | 0.0185 |
| `em_frozen` | 9.87 | **0.0093** |
| `hyperbo_frozen` | 12.00 | 0.0278 |
| `ablr` | 15.06 | 0.0357 |
| `vanilla_gp` | 21.33 | 0.0986 |
| `random` | 34.35 | 1.3646 |

**Paired across the 9 priors** (both methods see the same priors, so this is the right
test), against `t(8) = 2.31`:

| contrast | mean d | SE | t | verdict |
|---|---|---|---|---|
| `em_finetuned` − `hyperbo_frozen` | −2.56 | 1.11 | **−2.31** | **at the boundary** |
| `em_frozen` − `hyperbo_frozen` | −2.13 | 1.20 | −1.78 | tied |
| `em_noisefit` − `hyperbo_frozen` | −2.19 | 1.37 | −1.59 | tied |
| `ablr` − `hyperbo_frozen` | +3.06 | 1.39 | +2.19 | tied |
| `vanilla_gp` − `hyperbo_frozen` | +9.33 | 1.80 | +5.17 | HyperBO better |
| `random` − `hyperbo_frozen` | +22.35 | 1.37 | +16.28 | HyperBO better |

**Read this as: all three EM variants are directionally ahead of HyperBO, none is
convincingly so.** `em_finetuned` lands at t = −2.31 against a critical value of 2.31 —
a coin-flip call, and it would be dishonest to write it up as significant. What *is* solid
is the gap to `vanilla_gp` and `random`. This reproduces §26.1's "EM is top-tier, tied with
HyperBO" on a completely re-run pipeline, which is reassuring after seven rounds of fixes.

**The 9-prior design change (§27.26) is doing exactly what it was meant to.** At P=3 the
threshold is t(2)=4.30 and *nothing here would clear it* — not even the 16σ gap to random
would have been expressible with the old estimator's df. At P=9 the threshold is 2.31 and
the strong contrasts separate cleanly. The extra priors bought real resolution at
identical compute, as predicted.

**Caveat.** These are stage-0 *baselines* — one configuration, no OFAT factors. The point
of stage 0 is to verify the pipeline end-to-end and calibrate cost, both of which it did.

### 27.29 STAGE 0 COMPLETE: all five regimes, and regression quality dissociates from BO

Stage 0 finished in **7 h 34 m** (09:59→17:33 UTC), inside the 6–8 h estimate. Health is
perfect across every regime: **5/5 `.DONE`, 243 shards, 0 incomplete, 0 error cells,
`n_priors = 9` everywhere, and a `between_prior` block on all five — including both
regression regimes**, which before the R7-2 fix would have reported COMPLETE while
carrying nothing.

**BO, budget-to-target @0.01 (lower better), paired across 9 priors vs `hyperbo_frozen`,
`t(8)=2.31`:**

| regime | best method | EM vs HyperBO | verdict |
|---|---|---|---|
| `lcb_bo` (on-grid) | `em_finetuned` 9.44 | d=−2.56, **t=−2.31** | EM ahead, at the boundary |
| `pd1_bo_armA` (on-grid, matched) | `em_finetuned` 19.99 | d=−1.82, t=−1.73 | **tied** |
| `pd1_bo_armC` (off-grid, full) | `ablr` 36.85 | d=+2.27…+2.98, **t=+2.35…+2.98** | **HyperBO ahead** |

**Regression, rank correlation at n=5 observations (higher better):**

| regime | `em_frozen` | `hyperbo_frozen` | gap |
|---|---|---|---|
| `lcb_reg` | **0.7307** | 0.7118 | +0.019 |
| `pd1_reg` | **0.7236** | 0.5789 | **+0.145** |

**The headline finding is a dissociation.** On PD1, EM's *regression* is far better than
HyperBO's (0.724 vs 0.579, a large gap) while HyperBO's *BO* is better (t=+2.35 to +2.98
on arm C). **Better posterior ranking does not translate into better optimisation here.**
That is precisely the question stage 4.1 was written to ask, and stage 0 already supplies
preliminary evidence for a negative answer. It also means any argument of the form "EM
models the tasks better, therefore it should optimise better" is empirically unsupported
on PD1.

**Second observation: on arm C, transfer buys almost nothing.** `vanilla_gp` (38.04,
t=+0.44) and `ablr` (36.85, t=−0.51) are both **tied with `hyperbo_frozen`**, and `ablr`
is nominally the best method in the regime. When a plain GP with no meta-learning matches
the meta-learned baselines, the off-grid arm-C setting is not rewarding transfer at all —
which reframes "HyperBO beats EM on arm C" as a smaller claim than it sounds.

**Consistency with history.** Arm C reproduces §27.21's direction (HyperBO ahead of EM)
and `lcb_bo` reproduces §26.1's ("EM top-tier, tied with HyperBO") on a completely re-run
pipeline after seven rounds of fixes. Nothing has silently drifted.

**On-grid vs off-grid is the axis that organises all of this:** EM is ahead or tied on the
two on-grid regimes and behind only on the off-grid one, consistent with §27's long-running
theme that the Nyström interpolation is where EM pays a price.

### 27.30 The EM+HyperBO hybrids: they close the arm-C gap, but the improvement is not certified

Re-analysed the arm-C hybrid sweep with the **same paired-across-priors estimator** stage 0
uses, so the hybrids can be compared with the stage-0 baselines on equal footing.
P=3 priors, so `t(2) = 4.30` — a strict bar.

| contrast | mean d | t | verdict |
|---|---|---|---|
| `em_frozen` − `hyperbo_frozen` | +3.19 | **+7.80** | **HyperBO better** |
| `em_additive_hyperbo` − `hyperbo_frozen` | +2.06 | +1.70 | **tied** |
| `em_additive_hyperbo_frozen` − `hyperbo_frozen` | +2.18 | +1.56 | **tied** |
| `em_additive_hyperbo` − `em_frozen` | −1.12 | −1.29 | tied |
| `em_additive_hyperbo_frozen` − `em_frozen` | −1.01 | −0.96 | tied |

**The interesting structure: the hybrid converts a clear loss into a tie.** Plain EM loses
to HyperBO on arm C decisively — t=+7.80 clears even the strict t(2)=4.30 bar, the single
most solid contrast anywhere in this study. Adding the HyperBO kernel as an **additive
component** moves the deficit from +3.19 to +2.06/+2.18 and the verdict from "HyperBO
better" to **tied**.

**But the improvement itself is not certified** (t=−1.29, −0.96). So the honest statement
is: *the hybrid is not distinguishable from plain EM, and unlike plain EM it is also not
distinguishable from HyperBO.* Whether that is a real repair or a variance increase that
merely blurs the comparison cannot be settled at P=3 — precisely the resolution problem
§27.26 addresses.

**Coverage of the three hybrid mechanisms in the running matrix:**

| mechanism | flag | where it is tested |
|---|---|---|
| HyperBO as **additive component** | methods `em_additive_hyperbo{,_frozen}` | **every stage-1 OFAT cell** (in `EM_METHODS`) |
| HyperBO as **base/canonical kernel** | `--em-canonical hyperbo` | stage-1 factor `canon_hyperbo` |
| HyperBO as **shrinkage target** | `--em-shrinkage` *with* `--em-canonical hyperbo` | ⚠️ **only in stage 2** — OFAT varies one factor at a time, so `shrink_0.1/0.3` currently shrink toward the *default* canonical kernel, not HyperBO's |

The third row is a genuine gap for the question as posed: "shrinkage target = HyperBO"
requires two flags together, and a one-factor-at-a-time screen cannot express it. Stage 2
(combinations) is where it lands. Note §27.24's mechanical result — trace-matched
shrinkage is trace-preserving and so mostly *downscales* the informative directions — which
predicts this combination will underperform the additive one regardless of target.

**Stage 1 will settle the first two at P=9**, where the threshold falls to t(8)=2.31 and
the minimum detectable difference improves ~3.2× (§27.26). The additive hybrids appear in
every OFAT cell, so they get 9 priors × 14 configs × 3 BO regimes of evidence.

### 27.31 STAGE 1 / LCBench: only the ADDITIVE hybrid helps — kernel and mean transfer do nothing

`stage1_lcb_bo` complete: **14 configs × 9 priors = 126 cells, 0 errors**, in **3 h 52 m**
— against 3 h 41 m for stage 0's *9* cells. A 14× larger stage in the same wall clock,
because the prior cache turns ~61 min of HyperBO pre-training per cell into a lookup.

**Consistency check that passes:** `hyperbo_frozen` = 12.00 in **all 14 configs** and in
stage 0. It is unaffected by every EM factor, exactly as it should be — the OFAT factors
are not leaking into the control.

**Best EM variant per config** (budget@0.01, lower better; `em_additive_hyperbo` wins in
13 of 14):

| config | best EM | config | best EM |
|---|---|---|---|
| `shrink_0.1` | **9.17** | `meanxfer_blend` | 9.74 |
| `meanxfer_full` | 9.28 | `shrink_0.3` | 10.06 |
| `mean_linear+meanxfer` | 9.28 | `covprior_nu` | 10.72 |
| `base` | 9.30 | `covprior` | 10.81 |
| `meanprior` | 9.35 | `mean_linear` | 11.89 |
| `canon_deep` / `init_naive` | 9.54 | `mean_linear+covprior` | 13.31 |
| `canon_hyperbo` | 9.70 | *(`hyperbo_frozen`)* | *12.00* |

**Paired across the 9 priors, `t(8) = 2.31`:**

| question | contrast | d | t | verdict |
|---|---|---|---|---|
| Does additive beat pure HyperBO? | `em_additive_hyperbo` − `hyperbo_frozen` | −2.70 | **−2.42** | **YES** |
| Does plain EM? | `em_frozen` − `hyperbo_frozen` | −2.13 | −1.78 | no, tied |
| Does additive beat plain EM? | `em_additive_hyperbo` − `em_frozen` | −0.57 | −0.74 | tied |
| Does HyperBO **as base kernel** help? | `canon_hyperbo` − `base` (em_frozen) | +0.06 | +0.11 | **no effect** |
| " (additive method) | `canon_hyperbo` − `base` (em_add) | +0.41 | +0.73 | **no effect** |
| Does **mean transfer** help? | `meanxfer_full` − `base` | −0.02 | −0.05 | **no effect** |
| " blended | `meanxfer_blend` − `base` | +0.44 | +0.76 | no effect |
| " with linear base mean | `mean_linear+meanxfer` − `base` | −0.02 | −0.05 | **no effect** |

**The expectation that "EM + HyperBO kernel should be purely better" is not supported.**
Three distinct ways of handing EM the HyperBO information behave very differently:

* **Additive composition** (`K_em + K_hyperbo`) is the *only* one that does anything. It
  is the sole EM variant that significantly beats `hyperbo_frozen` (t=−2.42) — though it
  remains statistically indistinguishable from plain EM, so the effect is modest.
* **Replacing EM's base kernel with HyperBO's** (`--em-canonical hyperbo`) has
  **literally no effect** (t=0.11 / 0.73). This is the variant intuition says should be
  best, and on LCBench it changes nothing measurable.
* **Transferring the mean** (`--em-mean hyperbo`, blended, or on top of a linear base) has
  **no effect either** (|t| ≤ 0.76, and d = −0.02 for full transfer).

`meanxfer_full` and `mean_linear+meanxfer` give identical numbers (both −0.02), which is
the expected consistency: a full mean replacement makes `--em-base-mean linear` irrelevant.
That the two agree exactly is a check passing, not a no-op.

**Reading.** On LCBench, EM's empirical covariance is already doing the work; HyperBO's
kernel adds value only as an *extra additive component*, not as a substitute for EM's own
canonical kernel or mean. Note also that the two *worst* configs are `mean_linear` (11.89)
and `mean_linear+covprior` (13.31) — a non-constant base mean actively **hurts** here,
which is worth remembering before assuming richer priors help.

**Caveat:** LCBench is on-grid. §27.29 showed the on-grid/off-grid axis organises these
results, and arm C is where EM struggles. `stage1_pd1_bo_armC` is running now and is the
real test of whether the additive hybrid repairs the off-grid deficit.

### 27.32 AUDIT: we do NOT have the missing-observation-noise bug — verified empirically

A parallel session found an evaluation bug where observation noise was omitted from the
posterior, making every model look badly overconfident. Audited both harnesses for it.
**We are clean, and the reason is structural rather than lucky.**

**The three-step argument:**

1. **Our targets carry no observation noise.** LCBench and PD1 are deterministic table
   lookups. `--obs-noise` defaults to `0.0` and is **never set anywhere in the matrix**
   (`grep -c obs-noise gen_stage_queue.py` → 0), so `pool_Y_cond = pool_Y` exactly.
2. **Both harnesses use the LATENT posterior.** `_posterior_moments` calls
   `model.posterior(cand)` with no `observation_noise`, and `bo_diagnose` imports that same
   function — so BO calibration and regression metrics share one code path. Repo-wide,
   **no call site sets `observation_noise=True`.**
3. **Latent posterior vs noiseless target is the correct pairing.** The bug is comparing
   *noisy* observations against a *latent* posterior; with exact targets there is no noise
   term to add. Adding one would make our models artificially **under**-confident.

**Empirically verified rather than argued** — reading the code cannot settle this, because
each method reaches `.posterior()` through a different class, and a surrogate that
silently returned a noise-inclusive predictive would get a wider σ for free:

| method | sd[f] (what we use) | sd[y] | gap = σ²_noise | |
|---|---|---|---|---|
| `em_frozen` | 0.0805 | 0.0865 | **1.00e-03** | = `COND_NOISE` exactly |
| `pretrained_gp_frozen` | 0.7370 | 0.7377 | **1.00e-03** | = `COND_NOISE` exactly |
| `vanilla_gp` | 0.7267 | 0.7311 | 6.43e-03 | = its *fitted* noise |

Every method **distinguishes** the two and none bakes noise in; the gaps recover exactly
the likelihood noise each model was given. The comparison is internally consistent.

**Two things this audit surfaced that are worth knowing:**

* **The latent/observation choice is not neutral across methods.** For `em_frozen` the
  noise variance is 1e-3 against a predictive variance of 0.00648 — **15% of the total**.
  For `vanilla_gp` it is 6.4e-3 against 0.528, about **1%**. So had we needed
  observation-level calibration, EM's numbers would have moved ~15× more in relative terms
  than the GP baselines'. Our targets are noiseless so latent is right, but this is exactly
  the sort of asymmetry that turns a shared convention into a biased comparison, and it is
  why the check had to be per-method rather than global.
* **EM really is far more confident, and it is not an artifact.** `sd[f]` = 0.0805 for
  `em_frozen` against 0.7267 for `vanilla_gp` — a **9×** difference on identical data, with
  noise handling now ruled out as the cause. This corroborates §27's overconfidence finding
  through an independent route.

**Residual conservatism, in the safe direction.** Models condition with `COND_NOISE`
(1e-3, a variance) on data that is actually noiseless. That slightly *widens* the latent
posterior, so if anything our calibration numbers understate confidence. It cannot
manufacture the overconfidence the parallel session observed.

**One real defect found while auditing:** the §27.23 units correction (`--cond-noise` is a
VARIANCE, not a std) was applied to `bo_experiment` but **not propagated to
`bo_diagnose`**, whose help text still reads "std, standardized units". Documentation only
— both harnesses pass the value to the same GPyTorch likelihood, so the *behaviour* is
identical and no number is affected. Queued in the deferred-cleanup list rather than
patched, because `bo_diagnose` is a live dependency of the running stage.

### 27.33 CORRECTION to §27.32: the model's own noise term belongs in the predictive

§27.32 concluded "targets are deterministic, so the latent posterior is correct". That
argument is too quick, and the objection is right.

**A GP's fitted σ² is not measurement noise.** On a deterministic benchmark it is
**misspecification absorbed into the noise term** — variation the kernel could not
explain. `vanilla_gp` fits σ² = 6.4e-3 on exact LCBench lookups precisely because the
Matérn kernel cannot represent everything. **That unexplained structure is present at test
points too**, so the model's own belief about a new observable is `N(μ, k** + σ²)`, and
scoring `N(μ, k**)` scores a distribution the model does not claim describes the data.

This bites `calib_ratio = σ/RMSE` hardest: RMSE measures the *actual* error, which contains
the absorbed structure. Excluding σ² from the numerator reports overconfidence by
construction.

**Measured both ways** (LCBench, 5 pre-train tasks):

| method | n_obs | RMSE | ratio_f | ratio_y | nll_f | nll_y |
|---|---|---|---|---|---|---|
| `em_frozen` | 5 | 2.061 | 0.062 | 0.065 | 81.07 | **74.44** |
| `em_frozen` | 20 | 0.655 | 0.089 | 0.102 | 56.26 | **41.62** |
| `vanilla_gp` | 20 | 0.928 | 0.484 | 0.493 | 2.398 | 2.323 |
| `pretrained_gp_frozen` | 20 | 0.661 | 0.818 | 0.820 | 1.070 | 1.069 |

**Two conclusions, and they point in different directions:**

1. **Including the noise does NOT rescue calibration.** `ratio_y` is barely better than
   `ratio_f` everywhere (0.062→0.065, 0.818→0.820). The models are overconfident by
   factors of 1.2–16×, and σ² accounts for only a few percent of that. **So §27's
   overconfidence finding stands** — it is not an artefact of this choice, which is the
   reassuring half.
2. **But the reported NLL moves a lot, and differentially.** `em_frozen` improves **26%**
   (56.26 → 41.62) while `vanilla_gp` improves **3%**. An 8× differential means the choice
   is **not neutral in a comparison** — it systematically flatters whichever method has the
   larger σ² relative to its predictive variance, which here is EM. Our published NLL gaps
   between EM and the GP baselines were therefore **overstated**.

**Resolution: record both, do not pick.** `bo_diagnose` now emits `nll_obs`,
`mean_sigma_obs`, `coverage95_obs` and `calib_ratio_obs` alongside the latent versions, at
the cost of one extra posterior call. The two answer genuinely different questions —
*"how well is f known"* (latent, defensible because the benchmark really is
deterministic) versus *"how well is the next observation predicted"* (noise-inclusive, the
standard in GP benchmarking) — and with both on disk the corpus can be read either way
without a re-run.

**Scope of the correction.** BO results are **unaffected**: acquisition uses the latent
posterior, which is correct and standard for EI/UCB. Only the regression calibration
metrics are involved, and the regression regimes had not yet started when this landed, so
the entire v2 regression corpus will carry both.

**What I got wrong in §27.32.** I checked that the *targets* were noiseless and stopped
there, treating "no measurement noise" as "no noise term belongs in the predictive". The
model's σ² is a statement about its own residual uncertainty, and a deterministic target
does not make that statement vacuous.

### 27.34 THE HYPERBO KERNEL TRANSFER WORKS — off-grid, which is exactly where EM was losing

`stage1_pd1_bo_armC` (13 h 49 m) and `stage1_pd1_bo_armA` (3 h 57 m) complete: 126 cells
each, 9 priors, **0 errors**. This **overturns §27.31's headline**, which was drawn from
LCBench alone.

**Paired across 9 priors, `t(8) = 2.31`:**

| mechanism | LCBench (on-grid) | PD1 arm A | PD1 arm C (off-grid) |
|---|---|---|---|
| `canon_hyperbo` vs `base` (em_frozen) | +0.06, t=+0.11 — **nothing** | −0.93, t=−1.58 tied | **−3.58, t=−3.36 HELPS** |
| `canon_hyperbo` vs `base` (em_additive) | +0.41, t=+0.73 — **nothing** | **−1.52, t=−2.52 HELPS** | **−4.27, t=−5.21 HELPS** |
| `meanxfer_full` vs `base` (em_additive) | −0.02, t=−0.05 — **nothing** | **−1.28, t=−2.58 HELPS** | **−4.08, t=−4.64 HELPS** |

**And it reverses the headline comparison against HyperBO itself:**

| | arm C | arm A |
|---|---|---|
| `base` `em_frozen` − `hyperbo_frozen` | +2.27, **t=+2.35 → HyperBO wins** | −0.15, t=−0.19 tied |
| `canon_hyperbo` `em_additive` − `hyperbo_frozen` | −1.79, t=−1.76 → **tied** | **−3.05, t=−4.28 → EM WINS** |

So on arm C, plain EM **loses** to HyperBO (the §27.21/§27.29 finding, reproduced) — and
giving EM the HyperBO kernel as its canonical kernel **erases that deficit**. On arm A the
same change turns a tie into a **decisive EM win** (t=−4.28).

**The organising principle, now with direct evidence.** §27.29 established that on-grid vs
off-grid separates these results; this says *why*. EM's weakness is its canonical kernel's
Nyström extrapolation off-grid — so substituting a kernel learned across tasks helps
precisely there, and does nothing on LCBench where EM's own kernel is already adequate.
The three mechanisms are not interchangeable and their value is **regime-dependent**:

* **On-grid (LCBench):** only the *additive* composition does anything (§27.31).
* **Off-grid (PD1):** the *canonical kernel substitution* and the *mean transfer* both
  help substantially, and the additive variant is the best method in every top config.

**§27.31's claim "the expectation that EM + HyperBO kernel should be purely better is not
supported" is WITHDRAWN as stated.** It is supported on PD1 — strongly, on both arms. It
is unsupported only on LCBench, and generalising from one on-grid suite was exactly the
error §27.29 warned about. The honest formulation: *the HyperBO kernel transfer is
inert where EM is already strong and substantial where EM is weak.*

**Best configs.** Arm C: `canon_deep` 34.84, `canon_hyperbo` 35.68, `meanxfer_full` 35.86
(vs `hyperbo_frozen` 37.47). Arm A: `canon_hyperbo` 18.75, `meanxfer_full` 19.00 (vs
21.81). `em_additive_hyperbo` is the best EM method in every top config on both arms.

**Consistent across all three suites:** the two worst configs everywhere are `mean_linear`
and `mean_linear+covprior`. A non-constant *base* mean hurts regardless of regime — which
is not the same as the *transferred* mean, which helps off-grid.

**Cost note.** Arm A took 3 h 57 m against arm C's 13 h 49 m for identical cell counts;
the prior cache (423 entries, 16 MB) is now warm for the matched pool. `stage1_lcb_reg` is
running and will be the first regression regime to carry the §27.33 dual calibration
metrics.

### 27.35 THE EM PRE-TRAINING NOISE IS HARDCODED, UNEXPLORED, AND BADLY WRONG

`pretrain_em_prior(..., likelihood_noise=1e-2)` appears at **5 call sites, hardcoded, with
no CLI flag, never fitted**. It is a **variance**, so on unit-variance standardized data it
asserts σ = 0.1. (`em_noisefit` fits the *BO-time* `COND_NOISE`, a different parameter.
There is also a 10× internal inconsistency: pre-train at 1e-2, condition at 1e-3.)

**Scan on LCBench** (6 pre-train tasks, held-out task 0, `em_frozen`):

| σ² | σ | rank_corr @5 | RMSE @5 | rank_corr @20 | RMSE @20 |
|---|---|---|---|---|---|
| 1e-6 | 0.001 | 0.6855 | 2.821 | 0.8154 | 0.662 |
| 1e-3 | 0.032 | 0.6857 | 2.793 | 0.8166 | 0.660 |
| **1e-2 (current)** | **0.100** | **0.6917** | **2.553** | **0.8212** | **0.647** |
| 1e-1 | 0.316 | 0.7513 | 1.472 | 0.8426 | 0.581 |
| 3e-1 | 0.548 | 0.8130 | 0.920 | 0.8589 | 0.539 |
| **1.0** | **1.000** | **0.8546** | 0.629 | **0.8642** | **0.526** |
| 3.0 | 1.732 | 0.8531 | **0.561** | 0.8586 | 0.531 |
| 10.0 | 3.162 | 0.8036 | 0.610 | 0.8262 | 0.568 |

**The optimum is σ² ≈ 1–3, two to three orders of magnitude above the hardcoded value.**
Against the current setting:

* **n=5: rank corr +23.5% (0.692 → 0.855), RMSE −78% (2.553 → 0.561)**
* n=20: rank corr +5.2%, RMSE −19%

The gain is largest in the **low-data regime**, which is exactly where BO spends its early
iterations, so this plausibly affects budget-to-target and not just regression metrics.

**Mechanism.** The E-step posterior at the inducing points is `K(K+σ²I)⁻¹y`. Raising σ²
shrinks each task's posterior toward the prior mean, so `Σ_ind` stops chasing the
pre-training tasks' idiosyncrasies. That the optimum sits near σ² ≈ 1 on *unit-variance*
data says the empirical covariance was **massively overfitting** and needed regularization
comparable to the signal itself. This dovetails with §27's long-running overconfidence
finding: the prior was too tight, and 1e-2 was nowhere near enough to fix it.

**⚠️ The critical caveat, which must be tested before anyone celebrates.** σ² ≈ 1 is
suspiciously large. It may mean the empirical Σ is being **switched off**, leaving EM to
fall back on its base kernel — in which case this is not "EM works better" but "EM
degenerates into a simpler model that happens to beat it". **The decisive check is whether
`em_frozen` at σ²=1 becomes numerically indistinguishable from `pretrained_gp_frozen`.**
Until that is run, treat the table as *"the current value is wrong"* and **not** as
*"σ²=1 is the answer"*.

**Status: not yet actionable in code.** `bo_experiment` is a live dependency of the running
regression stage (`bo_diagnose` imports it, so editing it rebuilds both mid-stage), so the
flag is not added yet. Planned, in order:

1. `--em-likelihood-noise` (default 1e-2, preserving every existing result).
2. The degeneracy check above.
3. A noise sweep as its own stage, on all regimes — the effect is regime-dependent
   elsewhere (§27.34) and there is no reason to assume it is not here.

This is a single hardcoded constant that has been silently degrading every EM number in
the study, and it was found only by questioning why it was 1e-2 at all.

### 27.36 DEGENERACY CHECK: §27.35's gain is REAL — EM does not collapse into the GP

§27.35 flagged the obvious alternative: raising the EM pre-training noise shrinks every
task's E-step posterior toward the prior mean, so at some point `Σ_ind` carries nothing and
`em_frozen` becomes its own base kernel — i.e. `pretrained_gp_frozen`. If so the
"improvement" would be EM **switching itself off**. Tested directly (LCBench, 6 pre-train
tasks, n_obs=20, both models on identical observations and the identical shared kernel):

| σ² | corr(μ_em, μ_gp) | max\|Δμ\| | sd ratio | ρ_em | ρ_gp | RMSE_em | RMSE_gp |
|---|---|---|---|---|---|---|---|
| 1e-2 *(old default)* | 0.621 | 2.96 | 0.089 | 0.8212 | 0.6796 | 0.6474 | 0.7308 |
| 1e-1 | 0.667 | 2.61 | 0.236 | 0.8426 | 0.6796 | 0.5812 | 0.7308 |
| **1.0 (optimum)** | **0.762** | **1.66** | **0.507** | **0.8642** | 0.6796 | **0.5255** | 0.7308 |
| 3.0 | 0.808 | 1.41 | 0.648 | 0.8586 | 0.6796 | 0.5309 | 0.7308 |
| 10.0 | 0.896 | 1.04 | 0.785 | 0.8262 | 0.6796 | 0.5682 | 0.7308 |
| 100.0 | **0.984** | **0.38** | **0.954** | 0.7189 | 0.6796 | 0.6901 | 0.7308 |

**Verdict: not degenerate at the optimum.** At σ²=1 the two models remain plainly
distinct — correlation 0.76 (not 1), predictions differing by up to 1.66, and EM's
uncertainty **half** the GP's. And EM is decisively *better* there than the GP it might
have collapsed into: **RMSE 0.5255 vs 0.7308 (−28%)** and **ρ 0.864 vs 0.680**.

**Degeneracy is real but lives far beyond the optimum.** At σ²=100 the diagnostics do
converge (corr 0.984, Δμ 0.38, sd ratio 0.954) *and performance collapses toward the GP's*
(ρ 0.719 → 0.680, RMSE 0.690 → 0.731). So the failure mode I hypothesised exists — it is
simply not what is happening at σ² ≈ 1–3.

The full picture is a clean inverted U:

* **σ² too small (1e-2):** `Σ_ind` overfits the pre-training tasks. RMSE 0.647.
* **σ² ≈ 1–3:** regularised but still informative, and maximally distinct from the GP.
  **RMSE 0.526.**
* **σ² too large (100):** `Σ_ind` is washed out, EM converges to its base kernel. RMSE 0.690.

**Internal consistency check that passes:** `ρ_gp` and `RMSE_gp` are *identical* (0.6796 /
0.7308) on every row, as they must be — `pretrained_gp_frozen` does not use the EM prior,
so the EM noise cannot touch it. That confirms the comparison is fair and the sweep is
varying only what it claims to.

**Caveat on the trace column** (omitted above): the `Σ_ind` trace did not shrink toward
zero — it drifted *up* slightly. I could not confirm which attribute the probe read, so
that measurement is inconclusive and the prediction-level evidence is what carries the
conclusion. The mechanism is better described as *`Σ_ind` losing informative structure*
than *losing magnitude*.

**Consequence: §27.35 is upheld and the noise sweep is justified.** The hardcoded 1e-2 was
costing real accuracy, and the fix is genuine regularisation of the empirical covariance,
not a disguised fallback. `--em-likelihood-noise` is already in place (default 1e-2,
verified bit-identical), so the sweep can run as its own stage.

### 27.37 STAGE 1 / LCBench regression: the noise-inclusive correction changes nothing here

`stage1_lcb_reg` complete: 14 configs × 9 priors, 0 errors, and the **first corpus carrying
the §27.33 dual calibration metrics**.

**Latent vs noise-inclusive, `base` config at n=5:**

| method | nll | nll_obs | Δ | calib | calib_obs | Δ |
|---|---|---|---|---|---|---|
| `em_additive_hyperbo_frozen` | 0.204 | 0.206 | +1.1% | 1.219 | 1.220 | +0.2% |
| `hyperbo_frozen` | 0.218 | 0.223 | +2.3% | 1.092 | 1.096 | +0.4% |
| `em_additive_hyperbo` | 0.496 | 0.490 | −1.2% | 0.711 | 0.727 | +2.3% |
| `em_frozen` | 0.507 | 0.496 | −2.2% | 0.720 | 0.727 | +1.0% |

**The ranking is identical under both.** The differential between EM and the HyperBO
control is ~4.5 percentage points, not the **23** the §27.33 probe suggested.

**Why the probe overstated it, which is worth understanding.** That probe used 5–6
pre-training tasks and got EM to `sd[f]` = 0.058, so a 1e-3 noise term was ~15% of the
predictive variance. At the real stage-1 scale (25 pre-training tasks, 200 configs) EM's
`calib_ratio` is 0.72 — its predictive sd is far larger relative to the noise, so the same
absolute noise is a much smaller fraction. **§27.33's correction is methodologically right
and practically inert on this regime.** Recording both remains the correct call, but no
existing conclusion moves.

**Regression results** (rank corr @ n=5, `hyperbo_frozen` = 0.7118 control):
`em_additive_hyperbo_frozen` is the best EM method in **13 of 14 configs** and beats the
control everywhere (0.688–0.764). Best: `canon_hyperbo`, `meanxfer_full`,
`meanxfer_blend`, `mean_linear+meanxfer` all at **0.7642**. Worst: `meanprior` 0.688 and
`mean_linear+covprior` 0.692 — the same two families that lose on BO.

Note the calibration ordering differs from the accuracy ordering: `hyperbo_frozen` is the
best *calibrated* (1.092, nearest 1), `em_frozen` is overconfident (0.720), and
`em_additive_hyperbo_frozen` **overshoots into under-confidence** (1.219) while having the
best NLL. Accuracy and calibration are not tracking each other, which is consistent with
§27.29's regression-vs-BO dissociation.

### 27.38 BUG: the RNG re-seed guard was never ported to `bo_diagnose`

`hyperbo_frozen` should be **identical across all 14 configs** — it does not use the EM
prior, so no EM factor can touch it. On the BO regimes it is (verified: 1 distinct value).
On `lcb_reg` there are **3**:

| value | configs |
|---|---|
| 0.7118 | 11 configs (clean) |
| 0.7087 | `mean_linear`, `mean_linear+covprior` |
| 0.7082 | `canon_deep` |

Exactly the configs that construct **extra parameterised modules** — `--em-base-mean
linear` (a `LinearMean`) and `--deep-kernel 32,32` (an MLP). They consume RNG draws, which
shifts the stream and silently re-initialises the neural baselines downstream.

`bo_experiment` already guards this with `torch.manual_seed(pretrain_seed * 1009 + 11)`
before each baseline, added precisely because *"changing the canonical kernel (which builds
an MLP) silently reinitializes every neural baseline, moving arms that should be
bit-identical controls."* **`bo_diagnose` has zero occurrences of that guard** — the fix was
never propagated, the same failure mode as the §27.23 units correction.

**Impact is small but real:** 0.7118 → 0.7082/0.7087, about 0.5% relative, affecting 3 of
14 configs. It does not change any ranking above. But those three configs are being
compared against a slightly different control than the other eleven, and the same defect
will recur on every future regression run.

**Not fixed yet:** `bo_diagnose` is live (`stage1_pd1_reg` at 122/126). Queued to fix
before stage 3, together with re-checking whether `pd1_reg` shows the same 3-value split.

### 27.39 ARM B SETTLES IT: the deficit is interpolation, and richer pre-training half-repairs it

Arm B complete (126/126 cells, 9 priors). It holds pre-training **matched** as in arm A
while searching the **full** pool as in arm C, so the two flags A and C changed together
can finally be separated.

**Base config, budget@0.01 (lower better):**

| arm | candidates | pre-train | `em_frozen` | `em_additive` | `hyperbo_frozen` |
|---|---|---|---|---|---|
| A | matched | matched | 21.66 | 20.28 | 21.81 |
| **B** | **full** | **matched** | **41.30** | 40.79 | 36.33 |
| C | full | full | 39.73 | 39.94 | 37.47 |

**Paired across 9 priors, `t(8)=2.31`:**

| | arm A | **arm B** | arm C |
|---|---|---|---|
| `canon_hyperbo` − `base` (em_frozen) | −0.93, tied | **−4.20, t=−6.62** | −3.58, t=−3.36 |
| `em_frozen` − `hyperbo_frozen` | −0.15, tied | **+4.97, t=+4.41** | +2.27, t=+2.35 |

**A → B isolates the interpolation effect, and it is the whole story.** Pre-training is
*identical* between these two; only the candidate pool moves on-grid → off-grid. That
single change takes EM from **tied with HyperBO (−0.15)** to **losing by 4.97 (t=+4.41)**,
and takes the `canon_hyperbo` repair from **inert (−0.93, tied)** to **the strongest effect
anywhere in the study (−4.20, t=−6.62)**.

This confirms §27.34's mechanism **directly** rather than by inference: EM's weakness is
the Nyström extrapolation off the inducing set, and substituting a cross-task kernel
repairs precisely that.

**B → C isolates the pre-training pool, and it partially compensates.** Candidates are
fixed off-grid; only the pre-training data changes matched → full. EM's deficit **halves**,
from +4.97 to +2.27. So giving EM richer per-task pre-training data recovers about half of
what going off-grid costs it — a second, independent lever.

**Arm B is the hardest setting in the study for EM** (`em_frozen` 41.30, worse than arm C's
39.73): off-grid candidates *without* the richer pre-training that would partly offset
them. That it was the missing cell is precisely why A-vs-C was ambiguous.

**Summary of the causal picture, now measured end to end:**

1. On-grid, EM ties HyperBO and the kernel transfer does nothing (arm A).
2. Going off-grid costs EM ~5 evaluations and is the source of the entire deficit (A→B).
3. Richer pre-training returns ~2.7 of those (B→C).
4. Substituting HyperBO's kernel returns ~3.6–4.2 wherever the interpolation is active
   (B and C), and nothing where it is not (A).

### 27.40 WHY the optimal EM noise is so large: it is a spectral truncation, not a bug

The large optimum (σ² ≈ 1–3 on unit-variance data) is a fair thing to be suspicious of — it
reads as "the observations are essentially pure noise", which is absurd for a deterministic
benchmark. Two hypotheses:

* **H2 — scale compensation.** `Σ_ind` or the canonical kernel is mis-scaled and σ² is
  silently absorbing the mismatch. Then tuning σ² papers over a bug.
* **H1 — genuine regularisation.** `Σ_ind` is a rank-limited empirical estimate that
  overfits, and heavy shrinkage generalises better.

**H2 is refuted. Every scale checks out:**

| quantity | measured | expected |
|---|---|---|
| per-task variance | 1.0000 | 1 (standardisation) |
| canonical kernel outputscale | 1.0985 | ~1 |
| K diagonal mean | 1.0985 | ~1 |
| **cross-task variance at a point** | **0.4461** | — |
| **`Σ_ind` diagonal mean** | **0.373–0.410** | should match the row above |

`Σ_ind` recovers the actual cross-task variance (0.37–0.41 vs 0.4461). Nothing is
mis-scaled.

**H1 is confirmed, and the mechanism is specific — σ² acts as a spectral truncation.** The
E-step retains each eigendirection of K with weight `λ/(λ+σ²)`:

| σ² | mean weight retained | directions kept >50% |
|---|---|---|
| 1e-2 | 0.953 | **60 / 60** |
| 1e-1 | 0.722 | 50 / 60 |
| **1.0** | 0.314 | **14 / 60** |
| 3.0 | 0.170 | 5 / 60 |

**Now the decisive fact: `Σ_ind` estimated from `T` pre-training tasks has rank ≤ T−1.**
This probe used 6 tasks, so the true rank is **≤ 5**. At σ²=1e-2 the E-step faithfully
retains all **60** directions of a rank-5 object — 55 of them are pure estimation noise, and
EM dutifully fits them. At σ²=3 only 5 survive, which is exactly the sample rank, and σ²=3
was the best RMSE in the §27.35 scan.

**So σ² is not modelling noise at all — it is choosing an effective rank.** The hardcoded
1e-2 was implicitly asserting rank 60 for a rank-5 estimate.

**This yields a principled selection rule and a falsifiable prediction.**

Rule: pick σ² so the retained rank ≈ the sample rank, i.e. roughly the number of
pre-training tasks. **Prediction: the optimal σ² should DECREASE as `n_pretrain` grows.**
With 6 tasks the optimum was ~3; the real runs use 25 tasks (rank ≤ 24), which by the
retention table implies σ² ≈ 0.1–0.5 rather than 3.

The running sweep can test this directly — regimes differ in `n_pretrain`, so if the
optimum tracks it, the rule holds; if the optimum is the same everywhere, the rule is wrong
and the story is something else.

**Better still: stop choosing it by hand.** Three principled options, in order of merit:

1. **Fit σ² in the M-step.** Textbook EM estimates the noise; we fix it. That the optimum
   sits 2–3 orders from the fixed value is precisely the symptom of a parameter that should
   be learned. This is the real fix.
2. **Leave-one-task-out marginal likelihood** over the pre-training tasks — directly
   optimises the quantity we care about, at T extra fits.
3. **The rank heuristic above** as a cheap default when neither is available.

**Answering "was the study comprehensive?" — no, and it still is not.** §27.35 scanned one
regime, one held-out task, one metric, at two `n_obs` values. The running sweep widens this
to 7 values × 9 priors × 5 regimes, which is far better, but it remains a **grid search over
a parameter that should be estimated**. The grid tells us how much is on the table; it is
not a method for choosing σ² on a new problem, and it should not be presented as one.

### 27.41 SELECTORS: my rank-matching prediction is REFUTED; the optimum is scale-matched

§27.40 predicted the optimal σ² would **fall as `n_pretrain` rises**, because `Σ_ind`'s
sample rank rises. Built the selectors and tested it. **The prediction is wrong.**

| T (pre-train tasks) | grid optimum | effective-rank match | Marchenko–Pastur edge | Ledoit–Wolf α | OAS α |
|---|---|---|---|---|---|
| 5 | **1.0** | 17.7 | 1.91 | 0.512 | 0.616 |
| 10 | **1.0** | 4.54 | 0.901 | 0.499 | 0.533 |
| 17 | **1.0** | 1.78 | 1.04 | 0.449 | 0.403 |

**The grid optimum is 1.0 at every T.** It does not track the sample rank at all, so
"choose σ² to match the effective rank to T−1" is **not** the right rule. The
effective-rank matcher does move in the predicted direction (17.7 → 4.54 → 1.78) — it is
faithfully implementing my hypothesis, and the hypothesis is simply false.

**What the optimum actually tracks: the data scale.** Per-task standardisation gives unit
variance, and the optimum sits at σ² ≈ 1 — i.e. **shrink the dominant directions by about
half**, independent of T. Ledoit–Wolf agrees from a completely different derivation:
**α ≈ 0.40–0.51**, "shrink roughly halfway toward the target", again nearly flat in T. Two
independent estimators converging on *shrink by half* is far better evidence than either
alone.

So §27.40's mechanism (σ² is a spectral truncation) stands — that was measured. Its
*corollary* (therefore the optimum tracks rank) does not survive contact with data.

**Best parameter-free selector: the Marchenko–Pastur edge.** 1.91 / 0.901 / 1.04 against a
true optimum of 1.0 — within 2× at T=5 and within 10% at T≥10, with **no free parameters
and no held-out data**. That is good enough to replace the hardcoded default outright.

**The better fix remains structural.** Ledoit–Wolf/OAS shrink `Σ_ind` *directly*, with a
closed-form intensity, which:

* separates the two concerns — σ² goes back to meaning observation noise, and
  regularisation is done by the estimator built for it;
* removes the coupling to `--em-shrinkage` and `--use-covar-prior`, which currently
  regularise the same object through three different knobs;
* needs no grid, no held-out data, and no tuning.

`em_noise_selection.py` implements all four (LW, OAS, effective-rank, MP edge) so the next
step is wiring the chosen one in rather than re-deriving it.

**Recommended sequence:** (1) default σ² to the MP edge instead of 1e-2 — parameter-free
and demonstrably near-optimal; (2) implement LW shrinkage on `Σ_ind` and check whether it
beats *any* σ², which would let σ² revert to a true noise term; (3) only then consider
M-step estimation, which is more invasive and now looks less necessary given (1) and (2).

**Caveat.** One benchmark (LCBench), one held-out task, RMSE at n_obs=20. The running
sweep covers 5 regimes × 9 priors and will show whether the flat optimum at 1.0 holds
outside this slice — in particular whether PD1, with its very different eigenspectrum,
also lands there.

### 27.42 WHY σ²≈1 is principled: it is a ridge coefficient at the 50/50 point, not a noise level

The value looks absurd only if σ² is read as observation noise. It is not — and once that
is fixed, the value is not merely defensible but close to the textbook answer.

**1. `K(K+σ²I)⁻¹` is ridge regression.** The E-step posterior mean at the inducing points is
exactly a Tikhonov-regularised solve, and σ² is the ridge coefficient. Ridge coefficients
are set by a **bias–variance tradeoff**, not by the physical noise of the instrument. There
is no reason for one to be small just because the data are clean.

**2. The optimum sits precisely at the 50/50 point.** With the measured scales:

| direction | λ | weight `λ/(λ+σ²)` at σ²=1 |
|---|---|---|
| typical (= K diagonal) | 1.099 | **0.523** |
| top eigendirection | 17.18 | 0.945 |
| median eigendirection | 0.281 | 0.219 |

The σ² giving *exactly* 50/50 on the typical direction is **σ² = K_diag = 1.0985**; the
measured optimum was **1.0**. And Ledoit–Wolf, derived independently with no reference to
this kernel, keeps **0.49–0.55** of the sample covariance. **Three routes, one answer:
weight the empirical estimate and the prior about equally.**

**3. Why equal weighting is the *right* answer here.** Each pre-training task contributes
**exactly one draw** of the task-level deviation. Estimating a covariance direction from one
contrast per task gives a signal-to-noise ratio of order 1, and the empirical-Bayes /
James–Stein result for that regime is to weight likelihood and prior equally. σ²=1 is that
prescription, not a claim that the data are noise.

**4. This also explains the flatness in T that killed the rank hypothesis (§27.41).**
Adding pre-training tasks increases the **number of estimable directions** (rank ≤ T−1) but
does **not** improve the SNR *within* any direction — each remains one contrast per task. So
the optimal per-direction shrinkage is invariant to T, and σ\* is flat. The rank hypothesis
predicted movement because it conflated "how many directions exist" with "how well each is
estimated". Only the first grows with T.

**5. What σ² is really absorbing: model discrepancy.** The tasks are not draws from
GP(m, K) — K is a Matérn fitted to a handful of tasks. The gap between the true task
function and what the kernel can represent behaves exactly like noise in the E-step. This
is the standard Kennedy–O'Hagan model-discrepancy term, and its natural scale is the
*unexplained* share of the variance, which on unit-variance data is O(1). σ²≈1 says the
kernel explains roughly half — entirely consistent with the measured cross-task variance of
0.4461 against a total of 1.0.

**The real defect is the parameterisation, not the value.** σ² is scale-dependent (it must
change if the data or the kernel outputscale is rescaled), it is typed and named as noise,
and it silently overlaps `--em-shrinkage` and `--use-covar-prior`. The fix:

> Expose **`shrinkage_weight` w = λ̄/(λ̄+σ²) ∈ [0,1]** instead of σ², with **default 0.5**.

That is scale-free (invariant to standardisation and to the kernel outputscale),
interpretable ("trust the empirical covariance this much"), theory-backed at 0.5, and it
makes the overlap with the other two regularisers explicit rather than accidental.
Internally it maps straight back: σ² = λ̄(1−w)/w.

**Summary answer to "how can such a large noise term be justified?"** It cannot be, as
noise. As a ridge coefficient it is not large at all — it is the value that weights one
noisy empirical draw equally against the prior, which is what one draw per task warrants.
The number is right; the name is wrong.

### 27.43 CORRECTIONS to §27.42: what the E-step shrinks toward, and why the flatness claim was wrong

Two questions exposed an imprecision and an error.

**(a) Which "prior"? The current EM MEAN — not the base kernel.** From the source:
`residual = y - mu_S`, with the comment *"target moments are always the current
(mu, Sigma)"*. So the E-step pulls each task's curve toward **μ**, the running estimate of
the across-task mean. A vector, not a covariance.

**How that differs from `--em-shrinkage`,** which is a fair thing to suspect is the same
knob:

| | E-step σ² | `--em-shrinkage` |
|---|---|---|
| target | the **mean** μ (a vector) | a scaled **base-kernel gram** (a matrix) |
| stage | *before* `Σ_ind` is formed | *after* it is formed |
| effect on trace | **reduces** it | **trace-preserving** (§27.24) |
| per-direction | weight `λ/(λ+σ²)`, kernel-dependent | uniform blend |

They are genuinely different operations, and the trace row is the important one: §27.9
found `--em-shrinkage` unhelpful precisely because a trace-preserving blend cannot reduce
an overfitted magnitude — it only redistributes. σ² *can*, which is why it works.

**And a numerical correction: "50/50" describes the MEAN, not the covariance.** Verified —
shrinking deviations by `w` scales the covariance by exactly `w²`:

| w | measured cov scale | w² |
|---|---|---|
| 0.75 | 0.5625 | 0.5625 |
| 0.50 | **0.2500** | 0.2500 |
| 0.25 | 0.0625 | 0.0625 |

So σ²≈1 is a 50/50 shrink of the mean and a **4× shrink of the covariance**. §27.42 said
"weight the empirical estimate and the prior equally" without distinguishing these.

**(b) §27.42's explanation of the flatness in T is REFUTED.** I claimed more tasks add
directions but do not improve the estimate *within* a direction. Tested on a fixed
direction `u`, 400 replicates:

| T | measured rel. error | √(2/T) |
|---|---|---|
| 5 | 0.545 | 0.633 |
| 10 | 0.367 | 0.447 |
| 17 | 0.282 | 0.343 |
| 160 | 0.090 | 0.112 |

**More tasks clearly do sharpen each direction, at the textbook 1/√T rate.** My claim was
simply false.

**So why did the optimum look flat?** From T=5 to T=17 the estimation error improves 1.93×.
Our grid near the optimum is 0.3 → 1 → 3, i.e. **3.3× steps** — a ~2× shift in σ\* is
*smaller than one grid step* and therefore invisible. The honest statement is: **the data
are consistent with either a flat optimum or a decrease of up to ~2×, and this grid cannot
distinguish them.** Resolving it needs a finer grid (e.g. 0.5, 0.7, 1.0, 1.4, 2.0) at two
well-separated T values, which is cheap and worth doing.

**(c) "But the data are noiseless — why shrink at all?"** The sharpest framing yet:

> With noiseless observations the **inferentially correct σ² is exactly 0**. The E-step
> would return `f_t = y_t`, and `Σ_ind` would be the exact sample covariance of the
> pre-training curves. Every σ² > 0 is a **deliberate bias accepted for variance
> reduction** — regularisation, not inference.

That is the cleanest statement of why this parameter feels wrong: it is doing estimation
control through a slot the model reserves for measurement error. The estimand is not noisy;
the **estimator** is high-variance, because `Σ_ind` is a T-sample covariance in K
dimensions.

**Does more data per task help? No.** Each task is already fully observed on the matched
grid, so per-task data is not the limiting resource — `Σ_ind`'s error depends on **T, the
number of tasks**, and nothing else. The only ways forward are more tasks, or a better
estimator for the same T (Ledoit–Wolf, §27.41), which is why the analytic-shrinkage route
is the more promising one.

### 27.44 LEDOIT–WOLF REPLACES THE σ² HACK — and beats it, with no tuning

Two open questions from §27.43, settled together.

**Q1 — the optimum really is flat in T, and it is 0.6, not 1.0.** A 1.4×-spaced grid
(vs the old 3.3×) at two well-separated T:

| σ² | RMSE @ T=6 | RMSE @ T=17 |
|---|---|---|
| 0.30 | 0.5440 | 0.4935 |
| 0.42 | 0.5391 | 0.4769 |
| **0.60** | **0.5390** | **0.4726** |
| 0.85 | 0.5414 | 0.4744 |
| 1.20 | 0.5446 | 0.4774 |
| 3.40 | 0.5599 | 0.5021 |

**Same optimum, σ² = 0.6, at both T.** So the flatness is real and not a resolution
artefact — §27.43 was right to withhold judgement, and the answer is that it does not move.
Note also that the coarse grid's "1.0" was itself an artefact: with proper resolution the
optimum is **0.6**.

**Q2 — Ledoit–Wolf on `Σ_ind` beats the best σ², at essentially zero noise.** `Σ_ind` was
patched by rebuilding the frozen `EMPriorContainer`, so the arms differ in exactly one
matrix:

| | RMSE @ T=6 | RMSE @ T=17 |
|---|---|---|
| (a) σ²=1e-3, raw `Σ_ind` — no regularisation | 0.7518 | 1.1261 |
| (b) σ²=0.6, raw `Σ_ind` — the current hack | 0.5390 | 0.4726 |
| **(c) σ²=1e-3 + LW shrink — the proposed fix** | **0.5296** | **0.4568** |
| (d) σ²=0.6 + LW shrink — both | 0.5300 | 0.4769 |

**(c) wins at both T**, by 1.7% and 3.3% over the best hand-tuned σ². And it does so with
σ² at its *inferentially correct* value for noiseless data (≈0), with the shrinkage
intensity computed in **closed form from the pre-training data** — no grid, no held-out
set, no tuning.

**(d) shows they are substitutes, not complements.** Applying both is no better at T=6 and
distinctly *worse* at T=17 (0.4769 vs 0.4568) — the two regularisers stack into
over-shrinkage. So this is a replacement, not an addition.

**(a) shows the regularisation is essential**, and increasingly so with T: raw `Σ_ind`
degrades from 0.75 to 1.13 as T rises from 6 to 17. More tasks means more estimable
directions, and without shrinkage the extra directions are fitted noise. That is the
correct version of the intuition §27.40 got wrong.

**This closes the loop.** σ² was doing regularisation through a slot reserved for
measurement error, which made it scale-dependent, misnamed, uninterpretable, and
overlapping with two other knobs. Replacing it with Ledoit–Wolf:

* **is better** (1.7–3.3% RMSE at both T tested);
* **needs no tuning** — closed form, pre-training data only;
* **lets σ² revert to meaning noise**, which for these deterministic benchmarks is ~0;
* **removes the overlap** with `--em-shrinkage` and `--use-covar-prior`.

**Recommended implementation:** add `--em-covar-shrinkage {none,ledoit_wolf,oas}` defaulting
to `none` (preserving every committed result), apply it to `Σ_ind` inside
`pretrain_em_prior`, and sweep `ledoit_wolf` against the current best σ² on all regimes
before switching the default.

**Caveats.** One benchmark (LCBench), one held-out task, RMSE at n_obs=20, two T values.
The gain is small in absolute terms (1.7–3.3%) though consistent in sign, and the
comparison holds everything else fixed. It should be confirmed on PD1 — particularly arm C,
where §27.39 showed the off-grid interpolation dominates and the covariance estimate matters
most.

### 27.45 SYNTHETIC GROUND TRUTH: §27.42's "model discrepancy" explanation is REFUTED

Scepticism about whether so much regularisation could be legitimate was well placed. On
real data we cannot separate "regularisation is genuinely needed" from "something else is
broken", so this generates data where the truth is known and runs the **real**
`pretrain_em_prior` on it.

**The prediction under my §27.42 story:** data drawn from *exactly* the model EM assumes —
tasks ~ GP(0, K) with the same kernel EM is given, observed **noiselessly** — has no
observation noise and no discrepancy, so the optimal σ² should be ≈ 0.

| data source | T | optimal σ² | RMSE | ‖Σ̂−Σ_true‖/‖Σ_true‖ | LW α |
|---|---|---|---|---|---|
| **well-specified** (matched kernel) | 6 | **2** | 0.843 | 1.543 | 0.627 |
| **well-specified** | 17 | **1** | 0.767 | 1.429 | 0.507 |
| misspecified (EM lengthscale 3.3× too long) | 6 | 2 | 1.025 | 2.362 | 0.678 |
| misspecified | 17 | 2 | 0.972 | 1.523 | 0.850 |
| non-stationary (task-specific step) | 6 | 2 | 1.169 | 1.542 | 0.566 |
| non-stationary | 17 | 2 | 0.745 | 1.367 | 0.498 |

**The well-specified case wants σ² = 1–2, not 0. §27.42's discrepancy explanation is
wrong.** There is no discrepancy to absorb, and it still wants heavy damping.

**But the implementation is NOT broken.** The recovery error looked alarming — a relative
Frobenius error above 1 means the estimate is worse than predicting zero — so it was checked
against finite-sample theory, `E‖S−Σ‖²/‖Σ‖² ≈ (r_eff+1)/T` with `r_eff = tr(Σ)²/tr(Σ²)`:

| T | measured | theory √((r_eff+1)/T) |
|---|---|---|
| 6 | 1.469 | 1.394 |
| 17 | 0.848 | 0.828 |
| 50 | 0.483 | 0.483 |
| 200 | 0.239 | 0.242 |

(`r_eff = 10.66` in 60 ambient dimensions.) **Agreement to within a few percent at every
T.** The sample covariance really is that bad; nothing in the code is misbehaving.

**So the correct explanation is finite-sample severity alone — and it is stark.** At T=6
the empirical covariance has 147% relative error. Shrinking it hard is not compensation for
a bug or for misspecification; it is the right response to an estimate that poor. That also
explains the flatness in T: over 6→17 the error only improves 1.47→0.85, still far too
large to trust, so the optimal damping barely moves.

**Three consequences worth stating plainly:**

1. **LCBench is not anomalous.** Its optimum (σ²≈0.6) is *smaller* than the synthetic
   ideal case needs (1–2), so if anything the real data is better conditioned than the
   idealised test. Nothing about that benchmark is suspect.
2. **Misspecification adds on top of, rather than causing, the effect.** The misspecified
   arm is worse at every T (RMSE 0.97–1.03 vs 0.77–0.84) and drives LW α to 0.85. So
   discrepancy matters — it is just not the *source*.
3. **This bounds what the method can deliver.** EM's empirical covariance is fundamentally
   under-determined at the T available here (6–25 pre-training tasks against
   `r_eff ≈ 10` and hundreds of inducing points). The heavy regularisation is a symptom of
   that, and no amount of tuning removes it. Materially better `Σ_ind` needs **more tasks**
   or a **structurally constrained estimator** — which is exactly why the Ledoit–Wolf result
   (§27.44) matters more than finding a better σ².

**Correction log for this thread:** §27.40's rank-matching corollary — refuted (§27.41).
§27.42's per-direction-SNR claim — refuted (§27.43). §27.42's model-discrepancy explanation
— refuted here. What survives is: σ² is a ridge coefficient acting as a spectral truncation
(§27.40, measured), the optimum is flat at ≈0.6 on LCBench (§27.44), and Ledoit–Wolf does
the job better with no tuning (§27.44).

### 27.46 IS LEDOIT–WOLF THE RIGHT ESTIMATOR? Controlled study across K, T and spectrum

Metric is **PRIAL** — the fraction of the sample covariance's error an estimator removes
(0 = no better than `S`, 1 = perfect). `ORACLE` is linear shrinkage that is *allowed to see*
`Σ_true`, so it upper-bounds every linear method.

**Low-rank spectrum (`r_eff = 5.8`, matching our setting):**

| K | T | LW α | LW | OAS | clip | ORACLE | LW/oracle |
|---|---|---|---|---|---|---|---|
| 20 | 6 | 0.469 | 0.545 | **0.607** | 0.000 | 0.658 | 83% |
| 20 | 17 | 0.348 | 0.322 | 0.338 | −0.007 | 0.426 | 76% |
| 20 | 200 | 0.045 | 0.035 | 0.035 | −0.029 | 0.123 | **28%** |
| 60 | 6 | 0.416 | 0.449 | **0.490** | 0.000 | 0.536 | 84% |
| 60 | 1000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.060 | **0%** |
| 150 | 6 | 0.412 | 0.515 | **0.579** | 0.000 | 0.639 | 81% |
| 150 | 200 | 0.034 | 0.056 | 0.056 | 0.000 | 0.150 | 37% |

**Flat spectrum (`r_eff = K`, the hardest case for shrinkage):** LW reaches 0.88–1.00 and
OAS 0.97–1.00, both essentially at the oracle.

**Q1 — does LW behave correctly as T grows? Yes, and in both directions.** For the low-rank
spectrum α falls 0.469 → 0.348 → 0.158 → 0.045 → 0.009, correctly ceding to the sample
covariance. For the flat spectrum α *rises* to 0.97 — which is also right, because there
`Σ_true = μI` **is** the target, so shrinking fully is optimal. The estimator is not
merely damping; it is tracking how informative the target is.

**Q2 — size dependence.** PRIAL is governed by **T relative to `r_eff`**, not by K. At
`r_eff = 5.8` the gain is already small by T=200 (0.035–0.056) *because the sample
covariance is by then good on its own* — there is little left to remove. This answers the
"surely T=200 suffices for a 30×30 matrix" intuition: it does, and LW correctly gets out of
the way. The earlier §27.45 residual error of 24% at T=200 was for `r_eff = 10.7`, and is
the irreducible sampling error, not a shrinkage failure.

**Q3 — LW is good in our regime and mediocre outside it.** At T=6–17 it captures **76–84%**
of the oracle. By T=200–1000 it captures **0–37%**: it systematically **under-shrinks** once
the sample covariance looks reasonable. Since our T is 6–25, this is acceptable — but it is
not a general-purpose "always near-optimal" estimator, and should not be adopted as one.

**Q4 — what beats it.**

* **OAS beats LW essentially everywhere, and by the most at small T** (0.607 vs 0.545,
  0.490 vs 0.449, 0.579 vs 0.515). Our regime *is* small T. **OAS should be the default,
  not LW.**
* **Eigenvalue clipping is useless here** — PRIAL ≈ 0 or negative on the low-rank spectrum.
  My Marchenko–Pastur edge estimate uses the median eigenvalue as the noise level, which is
  meaningless when the spectrum decays fast. Nonlinear shrinkage is genuinely better in the
  literature, but requires a real spectral-density estimator (QuEST), not this shortcut.
  **Recorded as not-yet-tested rather than as a negative result.**
* **A structured target with the wrong eigenvectors is worse than isotropic** (kern 0.474
  vs oracle 0.658 at K=20/T=6). **Important caveat:** my simulated "kernel-like" target had
  the right eigenvalues under a *random rotation*, which is a pessimistic stand-in. The real
  canonical kernel gram shares inputs with `Σ_true` and so should share eigenvectors to some
  degree. **This test was unfair to the kernel target and must be redone on real data
  before concluding anything.**

**The Bayesian justification, which connects to code we already have.** LW/OAS are
frequentist — they minimise Frobenius risk over the linear-shrinkage class. The Bayesian
counterpart is exactly an **Inverse-Wishart prior**: for `Σ ~ IW(Ψ, ν)` the posterior mean is

> `E[Σ | data] = (T·S + Ψ) / (T + ν − K − 1)`

which **is** linear shrinkage toward `Ψ/(ν−K−1)`, with intensity `α = (ν−K−1)/(T+ν−K−1)`.
So *choosing α by LW/OAS is precisely empirical-Bayes selection of ν*. We already expose
`--use-covar-prior` and `--iw-nu`; today ν is hand-set. **Setting ν from the OAS α turns an
arbitrary knob into an estimated one using existing machinery**, and unifies three
overlapping regularisers (`likelihood_noise`, `--em-shrinkage`, `--use-covar-prior`) under
one principle.

**Revised recommendation:**

1. **Use OAS, not LW** — it dominates at the T we actually have.
2. **Implement it as ν for the existing IW prior**, not as a new independent knob.
3. **Re-test the kernel-gram target on real data** — the synthetic test was rigged against it.
4. Treat nonlinear shrinkage (QuEST) as open, not dismissed.

### 27.47 PRIMARY REFERENCES for the covariance-shrinkage work

The estimators in §27.44/§27.46 are not ad hoc — they are the standard tools for exactly
this problem (a rank-deficient sample covariance from `T ≪ K` samples), and the sources
should be cited rather than the methods re-derived.

**Chen, Wiesel, Eldar & Hero, "Shrinkage Algorithms for MMSE Covariance Estimation",
IEEE Trans. Signal Processing 58(10), 2010 — <https://arxiv.org/abs/0907.4698>**

The primary source for **OAS**, which §27.46 measured as the better choice at our sample
sizes. Verified that our implementation matches their eq. (23) exactly:

> `ρ_OAS = [(1 − 2/p)·tr(S²) + tr(S)²] / [(n + 1 − 2/p)·(tr(S²) − tr(S)²/p)]`, clipped to [0,1]

with `p` the dimension and `n` the sample count. The paper's framing is also the one we
arrived at independently in §27.42: shrinkage intensity as an **MMSE** problem, choosing
where to sit between a high-variance sample estimate and a low-variance biased target —
not as a noise level.

It also gives **RBLW** (Rao–Blackwellised Ledoit–Wolf, eq. 17), which under Gaussianity
dominates plain LW and, unlike OAS, is closed-form rather than the fixed point of an
iteration. **Implemented as `rblw_alpha` but NOT yet benchmarked** — §27.46 compared only
LW and OAS. It sits between them in the paper and is cheap, so it belongs in the next
sweep. Recorded as untested rather than recommended.

**Ledoit & Wolf, "A well-conditioned estimator for large-dimensional covariance matrices",
J. Multivariate Analysis 88(2), 2004** — the earlier distribution-free estimator that OAS
improves on by assuming Gaussianity. That assumption is what buys the gain at small `T`
(§27.46), and is worth remembering as a caveat: our task curves are standardised learning
curves, not obviously Gaussian, so OAS's advantage rests on an assumption we have not
checked. LW's distribution-free guarantee is the conservative fallback if that ever matters.

**Why this connects to the Bayesian framing (§27.46).** All three are linear shrinkage
`(1−α)S + α·target`, which is exactly the Inverse-Wishart posterior mean with
`α = (ν−K−1)/(T+ν−K−1)`. So the frequentist MMSE literature and our existing
`--use-covar-prior` machinery are solving the same problem in different notation, and
`estimate_iw_nu` is the bridge.

**Open, and worth a literature pass rather than invention:** nonlinear shrinkage
(Ledoit–Wolf 2012/2015, QuEST), which shrinks each eigenvalue separately and is the
strongest known family for large-dimensional covariance estimation. §27.46's clipping
attempt failed on an implementation detail, not on the idea.

---

## 28. The v2 sweeps, analysed: one design failure, one inverted optimum, one confound

**Written 2026-08-22**, after a machine reboot killed every running unit. The reboot is
incidental; what matters is that ~1500 finished cells had never been read. Consistent with
this project's signature failure mode (§27, §27.13, §27.15), **the largest corrections
available again came from data already on disk**, not from new compute.

Scope: `stage1_*` (6 regimes × 126 cells), `stage2means_*` (4 complete regimes × 90),
`stage2noise_*` (5 regimes × 63), `stage3_*` (3 regimes × 27). All were `UNANALYSED`.

### 28.1 STAGE 3 REPLICATED NOTHING — it is a byte-for-byte copy of stage 1

**Verified directly, not inferred.** Comparing every stage-3 shard against its stage-1
namesake, excluding only `timings` and the output path:

| pair | shards | identical | differing |
|---|---|---|---|
| `stage1_lcb_bo` → `stage3_lcb_bo` | 27 | **27** | 0 |
| `stage1_pd1_bo_armA` → `stage3_pd1_bo_armA` | 324 | **324** | 0 |
| `stage1_pd1_bo_armC` → `stage3_pd1_bo_armC` | 324 | **324** | 0 |

**675 / 675 identical. Δ(stage3 − stage1) = +0.000 ± 0.000** for every config × method ×
metric.

**Cause.** `select_top_configs` correctly picked stage 1's top 3 per regime, and stage 3
then re-ran them with the *identical* flag strings and the *identical* `--pretrain-seed
0..8`. The only `bo_experiment.py` change across the intervening commits was exposing
`--em-likelihood-noise` with its history-preserving default — behaviour-preserving by
construction. Deterministic code, same inputs, same outputs.

**Consequence, and it is the most important sentence in this section.**
`EXPERIMENT_PLAN.md` §4 rules that "only Stage 3 numbers may carry significance claims."
**Stage 3 therefore certifies nothing**, and the observed shrinkage of exactly 0.000 must
**never** be reported as "the winners replicated." It is not evidence of stability; it is
evidence that no second measurement was taken.

**Winner's curse, estimated properly.** With no independent replicate available, optimism
was estimated by leave-one-prior-out cross-fitting *within* stage 1 — select on 8 priors
using the exact `select_top_configs` statistic, score on the held-out prior:

| regime | metric | selection gap | top-1 optimism | verdict |
|---|---|---|---|---|
| `lcb_bo` | b@0.01 | −0.917 | **+1.211 ± 0.444, t=+2.73** | over-optimistic |
| `pd1_bo_armA` | b@0.01 | −1.371 | +0.432 ± 0.865, t=+0.50 | stable |
| `pd1_bo_armB` | b@0.01 | −3.619 | +0.000 ± 0.491 | **stable** |
| `pd1_bo_armC` | b@0.01 | −2.099 | +0.342 ± 0.735, t=+0.47 | stable |
| `pd1_bo_armC` | b@0.001 | −0.641 | **+1.960 ± 0.416, t=+4.71** | **entirely selection noise (186%)** |

A no-selection control (the same LOO contrast on the fixed `base` config) is
+0.000 ± 0.35–0.82 everywhere, so this is selection, not fold noise.

Read: at the **0.01** target on arms B and C the winner is genuinely stable (`canon_deep`
wins 9/9 folds on both). At the **0.001** target on arm C the winning set changes in 5/9
folds and the whole 0.641-eval advantage is optimism — **that ranking is not supportable**.

**Fix required before any stage 3 is re-run:** the replicate must use a seed range disjoint
from its source stage, and the generator must refuse to emit an overlapping queue.

### 28.2 The σ² optimum is OFF-GRID-SPECIFIC, and it INVERTS on-grid

7 σ² values × 9 priors × 5 regimes, all 315 cells present. Grid is
{1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10}; **σ²=0.6 is not on it** and is bracketed by 0.3 and 1.
Baseline 1e-2. `em_frozen`, budget-to-0.01, per-prior paired, t(8), crit 2.306.

**PD1 arm C (off-grid)** — a clean interior optimum, bracketed on both sides:

| σ² | 1e-3 | 1e-2 | 1e-1 | **0.3** | **1** | 3 | 10 |
|---|---|---|---|---|---|---|---|
| budget→0.01 | 39.69 | 39.76 | 39.44 | **37.55** | **36.76** | 38.83 | 39.42 |
| t(8) vs 1e-2 | −0.15 | — | −0.64 | **−4.41** | **−3.50** | −1.35 | −0.43 |
| final regret | .0359 | .0324 | .0318 | .0283 | **.0273 (t=−3.69)** | .0320 | .0403 |

9/9 priors put their individual argmin at 0.3 or 1. The companion metric agrees.

**PD1 arm A (on-grid)** — a *monotone penalty*, argmin at the low grid edge:

| σ² | 1e-3 | 1e-2 | 1e-1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|---|
| budget→0.01 | **21.58** | 21.66 | 21.91 | 22.90 | 23.49 | 24.40 | 25.24 |
| t(8) | −0.20 | — | +0.45 | +1.40 | **+2.64** | **+4.74** | **+6.24** |

**The sign flips, and the flip is a tested interaction, not an impression.** Paired by
prior on the *same 23 tasks* with only the two pool flags differing:
`[armC(σ²)−armC(1e-2)] − [armA(σ²)−armA(1e-2)]` = **−3.45 evals at 0.3 (t=−3.01)** and
**−4.83 at 1 (t=−4.10)**. `em_additive_hyperbo` reproduces it (−2.75 at σ²=1, t=−2.60).

This is mechanically what `METHODS.md` §2 predicts: when X ⊆ Z the Nyström residual
Λ(X)=0 and `Σ_ind` is used directly, so shrinking it **discards signal**; off-grid
everything passes through W and shrinkage pays.

**Is arm C's optimum distinguishable from LCBench's 0.6?** No — 0.3 vs 1 is
d=+0.79 ± 0.70, t=+1.12. The resolved optimum is the **interval [0.3, 1]**, which contains
0.6. So the LCBench value transfers off-grid within noise, and does not transfer on-grid
at all.

**⚠️ The §27.35 anchor does not reproduce.** On `lcb_reg` at n_obs=5 the argmax/argmin is
**1e-2**, and σ²=1 is **−8.6% rank corr and +18.5% RMSE** — sign-flipped against the
recorded "+23.5% / −78%". It *does* reproduce at n20 (+9.2% / −29.0%) and n50
(+5.7% / −17.5%). Since `em_noise.md` rule 2 already establishes flatness in T, this is not
a T effect but an **unexplained discrepancy in the anchor result**. Do not cite σ²≈0.6 for
the small-n regime until it is resolved.

**Estimand.** All of the above is **fixed-task**. Clustered by task (T=23, df=22), the
headline arm-C contrast is σ²=1 vs 1e-2 = −2.87 ± 1.90, **t(22)=−1.51, tied**. "σ²≈0.6–1
buys ~3 evaluations off-grid" is a claim about this suite, not about a new task.

### 28.3 The mean sweep is CONFOUNDED: canonical kernel ≡ mean transfer

`--em-canonical hyperbo` is set on **exactly** the 6 configs whose `--em-mean` is not
`empirical` (it is required at runtime). There is **no** `canonical=hyperbo × mean=empirical`
cell. The two factors are perfectly confounded.

On arm C the ranking is *precisely* that split:

| group | configs | budget→0.01 |
|---|---|---|
| canonical = hyperbo | `const_hb, lin_hb, const_b25, const_b50, const_b75, lin_b50` | 35.4 – 36.3 (ranks 1–6) |
| canonical = sumll | `const_emp, lin_emp, const_prior, lin_prior` | 39.5 – 39.9 (ranks 7–10) |

Between-group gap **3.93 ± 1.10, t=−3.57**; within-group spread 0.92 and 0.43. Meanwhile
every *within*-mean contrast is null: blend weight |t| ≤ 0.92, hyperbo-vs-blend0.5
t=−0.81, base mean t=−1.13, `--use-mean-prior` t=+0.37.

**So the data attributes the whole effect to the kernel at least as well as to the mean.**
Any "mean transfer helps off-grid" claim from *this sweep* is unidentified.

**RESOLVED from stage-1 OFAT, with zero new compute — and the answer is the kernel.**
The means sweep is confounded, but the stage-1 screen happens to contain both cells
separately: `canon_hyperbo` is `--em-canonical hyperbo` alone, while `meanxfer_full` is
`--em-canonical hyperbo --em-mean hyperbo`. Their paired difference isolates the mean
transfer at fixed kernel:

| regime | `meanxfer_full − canon_hyperbo`, `em_frozen` | t(8) |
|---|---|---|
| `pd1_bo_armC` (off-grid) | **+0.184 ± 0.476** | +0.39 |
| `pd1_bo_armB` (off-grid) | +0.034 ± 0.576 | +0.06 |
| `pd1_bo_armA` (on-grid) | +0.285 ± 0.136 | +2.10 |
| `lcb_bo` (on-grid) | −0.019 ± 0.168 | −0.11 |

Against a *base* contrast of −3.58 ± 1.07 for `canon_hyperbo` alone and −3.40 ± 1.36 for
`meanxfer_full`. **The mean transfer contributes nothing — if anything it is very slightly
harmful — and `--em-canonical hyperbo` carries the entire off-grid win.** The same holds
for `em_additive_hyperbo` (+0.184 ± 0.700, t=+0.26 on arm C) and for the blend variant.

Two consequences: (a) **write the off-grid result as a KERNEL-transfer result, never a
mean-transfer result**; (b) the added `mean_const_hbk` cell is still needed to make the
*means sweep* internally interpretable, but the scientific question no longer waits on it.

**Regime dependence reproduces the §27.15 prediction exactly.** The clean contrast
`mean_const_hb − mean_const_emp` (constant base mean on both sides):

| regime | geometry | Δ evals | t(8) | |
|---|---|---|---|---|
| `lcb_bo` | on-grid | **+0.037** | +0.07 | **inert** |
| `pd1_bo_armA` | on-grid | −0.647 | −1.07 | ns |
| **`pd1_bo_armC`** | **off-grid** | **−4.138** | **−3.66** | **SIG** |
| `lcb_reg` | on-grid | +0.0162 rmse | — | worse |

Difference-in-differences armC − armA = **−3.49 ± 1.12, t=−3.13**. Do not report a pooled
mean-function winner; the on-grid effect is +0.04 evals and the off-grid effect is −4.14.

**⚠️ A trap worth recording.** Pooling all 6 hyperbo-canonical against all 4 sumll configs
on `lcb_bo` gives Δ=−4.50, t=−3.30, which *looks* like an on-grid win. It is an artifact:
the sumll group contains `lin_emp`/`lin_prior`, which are catastrophic on LCBench
(+9.3 evals). Restricted to a constant base mean the on-grid effect collapses to
**−0.08 ± 0.59, t=−0.14**.

**Three embedded no-ops** (the standing rule is to report these loudly):
* `--em-base-mean` is a **no-op whenever `--em-mean hyperbo` is set** — `mean_const_hb` and
  `mean_lin_hb` are bit-identical on all 9 priors in 3 of 4 regimes. Two of ten configs are
  one config.
* `mean_const_b75` collapses onto both of them on `lcb_bo`.
* `--use-mean-prior` is null on all three BO regimes (t = 0.29 / 1.27 / 0.37).

### 28.4 The arm decomposition reproduces at 126 cells — and it is fixed-task only

Raw arm differences confound "EM got worse" with "the pool got harder" (400 → ~2040
candidates moves `pool_max`), so the EM-specific cost is the **difference-in-differences
against `hyperbo_frozen`**, which sees the same pool change but never touches the EM prior.

| contrast | raw Δ (`em_frozen`) | **DiD vs `hyperbo_frozen`** | §27.39 |
|---|---|---|---|
| **A → B** going off-grid | +19.64 ± 0.56, t=+35.1 | **+5.121 ± 0.866, t=+5.92** | 4.97 |
| **B → C** richer pre-training | −1.57 ± 0.53, t=−2.97 | **−2.705 ± 1.161, t=−2.33** | ~half |
| A → C net | +18.08 ± 0.55 | +2.415 ± 1.044, t=+2.31 | — |

§27.39 reproduces: off-grid alone costs **5.12 evals** (was 4.97, +3%) with a *larger* t on
the 126-cell runs, and B→C returns **52.8%** — "about half" is exact. `final_regret_mean`
agrees (A→B DiD +0.008, t=+2.41; B→C −0.012, t=−3.59).

**The penalty is config-conditional, which localises the mechanism:**

| A→B DiD | configs |
|---|---|
| **+3.5 to +6.9** | `shrink_0.1`, `shrink_0.3`, `mean_linear`, `init_naive`, `base`, `meanprior` — plain EM interpolation |
| **+1.6 to +1.9** | `canon_hyperbo`, `meanxfer_*` — non-trivial canonical kernel, penalty cut to ~⅓ |
| **−2.2 to −0.1** | `canon_deep`, `covprior`, `covprior_nu`, `mean_linear+covprior` — **penalty eliminated** |

B→C is flat across all 14 configs (−1.93 to −3.40, median −2.11): the pre-training-pool
benefit is a property of the data, not of the EM configuration.

**⚠️ Estimand.** Task-clustered (T=23, df=22, crit 2.074) **both effects are tied**:
A→B DiD +5.121 ± 2.875 (t=+1.78), B→C −2.705 ± 1.698 (t=−1.59). Point estimates identical;
only the SE inflates 3.3×. Write this as a fixed-task claim about this suite.

Comparability was checked, not assumed: `bo_experiment.py` is **identical** across the
three arm commits, so the contrast is not contaminated by binary drift.

### 28.5 Stage-1 screen: what is live, and factor E is the screen's worst

Per-prior paired vs `base`, `em_frozen`, budget@0.01 (negative = better):

| factor | `lcb_bo` | `armA` | `armB` | `armC` |
|---|---|---|---|---|
| A canonical = hyperbo | +0.06±0.51 | −0.93±0.59 | **−4.20±0.64** | **−3.58±1.07** |
| A canonical = deep 32,32 | +0.00±0.69 | −0.21±0.45 | **−5.40±1.18** | **−4.68±0.65** |
| B base mean = linear | **+9.26±2.00** | −0.08±0.20 | +0.47±0.53 | −0.22±0.61 |
| C mean transfer = hyperbo | +0.04±0.55 | −0.65±0.60 | **−4.17±0.90** | **−3.40±1.36** |
| D shrinkage 0.1 | +0.46±0.62 | **−2.19±0.45** | −0.46±0.30 | −0.64±0.58 |
| **E covariance prior** | **+3.78±0.93** | **+7.60±0.79** | +0.82±1.07 | +1.50±0.71 |
| F mean prior | +0.20±0.71 | +1.31±1.04 | −0.28±0.61 | +0.21±0.64 |
| G EM init = naive | −0.09±0.68 | −0.35±0.18 | −0.20±0.23 | −0.05±0.43 |
| B × E | **+11.80±2.52** | **+7.85±0.77** | +0.53±0.80 | **+1.15±0.46** |

* **Inert everywhere: G (`init_naive`)** — |Δ| ≤ 0.35 with SE ≤ 0.68 on both methods in all
  four regimes. **Drop it.** Also **F (mean prior)** and **D shrinkage 0.3**.
* **The only large consistent win is A/C, and only off-grid** — 3.4–5.4 evals on arms B
  and C, nothing (|t| ≤ 1.58) on arm A or LCBench. With the arm-A on-grid PD1 control in
  place this is **"regime", not "benchmark"** — the confound §9.2 of the plan existed to
  break.
* **⚠️ Factor E is the best NLL mover of §27.16 and the WORST factor here** — +3.8 evals on
  `lcb_bo`, +7.6 on arm A, null off-grid. This is the sharpest instance yet of regression
  quality dissociating from BO performance (§27.29), and it is a direct warning against
  §27.16's NLL-based promotion of `--use-covar-prior`.

**Two caveats on the screen.** On `lcb_bo`, `budget_to_0.01 == budget_to_0.001` in **all
504** cell×method slots (LCBench regret hits exactly 0 once the pool max is found), so the
0.001 target is not an independent measurement there. And `lcb_bo` averages over **6** eval
tasks vs 23 for PD1, so its SEs are ~2× larger at equal per-task noise.

**The selection statistic is not the factor effect.** On `lcb_bo`, `shrink_0.1` ranks 1st
on `select_top_configs`'s pooled statistic (9.167) while its `em_frozen` effect is
+0.46 ± 0.62 (t=+0.74). `select_top_configs` minimises over methods *and* configs — a
double-selection statistic, which is exactly where §28.1's LOO optimism is largest.

### 28.6 TWO defects in `bo_diagnose`, and the known one is the smaller

**Defect 1 — the missing RNG re-seed guard (§27.38), confirmed and quantified.**
`bo_experiment.py:3121/3160/3188` calls `torch.manual_seed(pretrain_seed*1009 + {11,13,17})`
before each neural baseline. `bo_diagnose.py:523` calls `pretrain_hyperbo` with **no**
re-seed, so its RNG state is whatever `build_shared_kernel` + `pretrain_em_prior` left.

`--em-base-mean linear` and `--deep-kernel 32,32` are the only two OFAT levels that consume
draws in that window, which is why the control takes exactly **3** values, not 14 — and the
partition is **identical on both regression regimes**:

| group | n | configs |
|---|---|---|
| 1 | 11 | `base`, `canon_hyperbo`, `covprior`, `covprior_nu`, `init_naive`, `meanprior`, `meanxfer_full`, `meanxfer_blend`, `mean_linear+meanxfer`, `shrink_0.1`, `shrink_0.3` |
| 2 | 2 | `mean_linear`, `mean_linear+covprior` |
| 3 | 1 | `canon_deep` |

Control shift on `n5_rank_corr`: `lcb_reg` max 0.0036 (**0.51%**); `pd1_reg` max 0.0159
(**2.74%**). **BO regimes are provably clean** — `hyperbo_frozen` is bit-identical across
all 14 configs at 0 / 126 / 1512 / 1512 mismatches for `lcb_bo` / each PD1 arm.

Artifact-to-effect ratio: on `lcb_reg` ≤ 33% and never itself significant (|t| ≤ 1.39). On
`pd1_reg` **`canon_deep` is not interpretable** — ratio 1.09 on `n20_rank_corr`, **1.31** on
`n5_rmse` — and `mean_linear`'s control shift is itself significant (−0.01586 ± 0.00646,
t=−2.46).

**Defect 2 — `--pretrain-seed` is a NO-OP for EM in `bo_diagnose`, and this is worse.**
`bo_diagnose.py:702` seeds the observation subset with `torch.manual_seed(1000 * ei)` —
**no `pretrain_seed` term**. `bo_experiment.py:3240` keys its initial design on the prior
(`_bo_seed = 1000*ei + seed + 100003*pretrain_seed`) *explicitly to prevent this*;
`bo_diagnose` never received the fix.

EM pre-training is deterministic (§25.4), so with the seed absent from the data path too,
`em_frozen`'s entire payload is **bit-identical across all 9 priors for 7 of 14 configs**
(`base`, `covprior`, `covprior_nu`, `init_naive`, `meanprior`, `shrink_0.1`, `shrink_0.3`).
Between-prior variance is **exactly 0**, so every per-prior paired SE is 0 and t is 0/0 —
visible in raw output as `t = -12925`, `t = +884990`, `t = 2.5e16`.

**This invalidates the df=8 statistics on both regression regimes far more broadly than the
control drift does — 7/14 configs versus 3/14 — and installing the RNG guard does not fix
it.** It also means the §28.2 σ²-sweep regression rows cannot use the pre-registered
prior-pooling at all; per-task pooling is the only defensible route there, and `rank_corr`
has no per-task breakdown, so **`rank_corr` cannot be error-barred in that sweep**.

**A note that also applies to the clean BO regimes.** `pretrain_seed` never resamples the
EM prior *anywhere* — EM pre-training is deterministic. In BO it resamples the **initial
design**. So for EM methods "9 priors" means 9 initial-design replicates, while for
HyperBO/ABLR/PACOH it genuinely means 9 prior draws. Valid as a replication unit for a
fixed-task claim, but the label overstates what varies, and **EM's reported uncertainty
contains no prior-draw component**.

**Defect 3 (latent, inert today).** `select_top_configs`
(`gen_stage_queue.py:301-307`) takes `keys[0]` of `sorted({'n20_rank_corr','n50_rank_corr',
'n5_rank_corr'})`, which is **`n20_rank_corr`** — not `n5` as its own comment asserts
("smallest n_obs: the regime BO actually operates in"). Harmless until a regression stage 3
runs, at which point it silently mis-selects.

### 28.7 Process findings

* **The plan's "single binary" hazard mitigation is violated.** The 11 stage directories
  carry **9 distinct commits**. The two comparisons §28.1 and §28.4 rely on were verified
  safe by diffing the source; **no other cross-stage comparison has been checked.**
* `PROVENANCE.json` records `date: null` in every stage directory, so run ordering is only
  recoverable from commit dates. Worth fixing.
* `pd1_bo_armC` is **52–66% censored** at the 51-eval cap. Budget-to-target there is
  substantially a rescaled solve rate; implied E[t | solved] actually *rises* with σ²
  (19.6 → 21.5 → 23.1). Report the solve rate alongside, per `.llms/rules/metrics.md` rank 3.
* `stage2means_pd1_reg` holds only the baseline config (9/90) — **zero mean variants ran**,
  so it contributes nothing and is excluded from every number above.

---

## 29. STAGE 4 SYNTHESIS — implemented at last, and 4.1 inverts off-grid

**Written 2026-08-22.** `EXPERIMENT_PLAN.md` §4 and §10 specified four cross-stage
analyses. None had ever been written — a grep for `p_div|stage4|regression_predicts_bo`
returned nothing across the whole directory. All four are now in `stage4.py`, run on data
already on disk, and cost no compute.

`stage4.py` is deliberately **not** in the `lib` srcs and has its own buck target: anything
in `lib` rebuilds `bo_experiment` and `bo_diagnose`, and an analysis script must never be
able to change the binary a running sweep executes.

### 29.1 (4.1) Regression quality predicts BO ON-GRID and ANTI-PREDICTS IT OFF-GRID

Unit = config, joined on the flag string across the two harnesses. Restricted to the
11-config clean block (§28.6); `canon_deep`, `mean_linear`, `mean_linear+covprior`
excluded. Point estimates only — the prior axis is dead for 7/14 configs until the
`bo_diagnose` re-run lands, so none of these carry a standard error.

| Spearman vs budget-to-0.01 | LCBench (on-grid) | PD1 arm C (off-grid) | expected |
|---|---|---|---|
| `n5_rank_corr` | **−0.689** | **+0.635** | negative |
| `n5_rmse` | +0.361 | +0.324 | positive |
| `n5_nll` | +0.041 | **−0.616** | positive |

**On-grid the relationship has the expected sign; off-grid it reverses on two of three
metrics.** The off-grid table shows it plainly:

| config | `n5_rank_corr` | `n5_nll` | budget |
|---|---|---|---|
| `canon_hyperbo` | 0.7065 | 1.4722 | **36.155** |
| `meanxfer_blend` | 0.7067 | 1.4720 | 36.222 |
| `shrink_0.3` | **0.8109** | **0.5225** | 38.826 |
| `shrink_0.1` | 0.7900 | 0.4623 | 39.092 |
| `covprior_nu` | 0.7465 | 1.1017 | 41.304 |

The configs with the **best** regression quality (`shrink_*`: rank corr 0.79–0.81, NLL
0.46–0.52) are mid-tier at BO, while the **best BO configs** (`canon_hyperbo`,
`meanxfer_*`) have mediocre rank correlation and nearly 3× the NLL.

**This is the strongest available warning against §27.16's methodology**, which promoted
`--use-covar-prior` on the basis of being the largest single-shard NLL mover. §28.5 already
showed it is the worst factor in the screen on the primary metric (+7.6 evals on arm A);
§29.1 now shows the *general* practice of screening EM variants by regression quality is
unsound in exactly the off-grid regime the paper cares about. Instrument check: budget
spread 4.96 (LCBench) and 5.15 (PD1) over 11 configs, so the axis is live.

### 29.2 (4.2) Which factors are regime-dependent, quantified

DiD = (arm C effect) − (arm A effect), paired per prior, `em_frozen`, budget@0.01.

| factor | on-grid (armA) | off-grid (armC) | **DiD** | |
|---|---|---|---|---|
| `covprior` | +7.599 (t=+9.62) | +1.502 (t=+2.13) | **−6.097 (t=−8.68)** | * |
| `mean_linear+covprior` | +7.845 (t=+10.13) | +1.150 (t=+2.52) | **−6.696 (t=−7.47)** | * |
| `covprior_nu` | +7.676 (t=+9.65) | +1.570 (t=+2.30) | **−6.106 (t=−8.51)** | * |
| `canon_deep` | −0.208 (t=−0.46) | −4.676 (t=−7.22) | **−4.469 (t=−5.94)** | * |
| `canon_hyperbo` | −0.932 (t=−1.58) | −3.580 (t=−3.36) | −2.647 (t=−2.06) | |
| `meanxfer_blend` | −0.705 (t=−1.22) | −3.512 (t=−2.54) | −2.807 (t=−1.93) | |
| `shrink_0.1` | −2.193 (t=−4.86) | −0.643 (t=−1.10) | +1.551 (t=+1.86) | |
| `init_naive` | −0.348 (t=−1.91) | −0.053 (t=−0.12) | +0.295 (t=+0.62) | |
| `mean_linear` | −0.082 (t=−0.42) | −0.222 (t=−0.36) | −0.140 (t=−0.23) | |

Four factors are significantly regime-dependent and **must never be reported as a single
pooled effect**. Note `shrink_0.1` runs the *other* way — it is the one factor that helps
on-grid (−2.19, t=−4.86) and does nothing off-grid, mirroring the σ² inversion of §28.2.

### 29.3 (4.3) The cost Pareto: EM's claim is performance+cost on-grid, cost only off-grid

Pre-training seconds attributed per method (EM methods pay `em_pretrain_s`; the additive
hybrids also pay for the HyperBO kernel they borrow), against budget@0.01.

**LCBench, `base`:**

| method | pretrain s | budget | |
|---|---|---|---|
| `em_additive_hyperbo` | 3455.2 | **9.296** | Pareto-optimal |
| `em_frozen` | **26.7** | 9.870 | **Pareto-optimal** |
| `em_additive_hyperbo_frozen` | 3455.2 | 11.889 | dominated |
| `hyperbo_frozen` | 3428.4 | 12.000 | dominated by `em_frozen` |

`em_frozen` reaches the target in **fewer** evaluations than HyperBO at **128× less**
pre-training. §26.2 reproduces.

**PD1 arm C, `base`:**

| method | pretrain s | budget | |
|---|---|---|---|
| `hyperbo_frozen` | 845.1 | **37.469** | Pareto-optimal |
| `em_frozen` | **405.5** | 39.734 | Pareto-optimal (cost only) |
| `em_additive_hyperbo_frozen` | 1250.6 | 38.976 | dominated |
| `em_additive_hyperbo` | 1250.6 | 39.942 | dominated by all three |

Off-grid EM is **2.1× cheaper and 2.3 evaluations worse**. This confirms the standing
instruction to state the PD1 Pareto claim as **cost only** (§27.8), and it is now measured
rather than asserted.

### 29.4 (4.4) The pre-registered divergence moderator: the tail explanation is NOT supported

`d_t = a + b · p_div_t`, where `d_t = budget(em_frozen) − budget(hyperbo_frozen)` per task
and `p_div` is the degenerate fraction of the task's pool (LCBench: flat curves; PD1:
error ≥ 0.8, the threshold `pd1_full_data.py` already uses). Both readings were fixed in
advance in `EXPERIMENT_PLAN.md` §10.

**PD1 arm C — the properly powered test, and it is the answer:**

* 23 tasks, `p_div` spanning **0.02 to 0.96** (sd 0.194) — a genuinely wide covariate.
* `b = +11.88 ± 10.19`, **t(21) = +1.17**, 95% CI **[−9.42, +33.18]**.
* Median split: low `p_div` mean `d_t` = +1.81, high = +2.77.
* Leave-one-task-out: slope stays in [+3.01, +17.04], significant in **0/23** folds.

**`b` is not distinguishable from zero, consistently. Under the pre-registered reading
this means the EM deficit is UNIFORM across tasks and the tail/degeneracy explanation
offered in §23 and §27 is NOT supported.** The direction is positive but the effect is
small relative to its noise, and the null is stable under resampling.

**LCBench — significant in the full sample, but it does not survive resampling:**

* Only 6 tasks, and `p_div` spans **0.0000 to 0.0200** — a covariate range of two
  percentage points.
* `b = +365.9 ± 120.9`, t(4) = +3.03 vs crit 2.78, CI [+29.7, +702.1].
* Leave-one-task-out: slope stable in sign (+273 to +472) but significant in only
  **2 of 6 folds**.

The LOO check was added precisely because a large slope over a 0.02-wide covariate on 6
points is the shape of an artifact, and it earned its keep on the first run. **Report the
direction, make no claim.** LCBench simply does not have enough degeneracy to moderate on
— which is consistent with §27.24/§27.25, where the divergence was found to live in the
loss metric and not in the accuracy we optimise.

**Net:** the moderator was pre-registered to rescue value from a design that cannot support
significance on the average, and it does that honestly — by returning a clean, stable null
on the benchmark that can actually test it.

---

## §30 — The prior cache as an instrument (2026-08-22)

`prior_cache.py` was built to make pre-training reusable, and its key deliberately
includes `--pretrain-seed` so that repeated fits are RETAINED SEPARATELY rather than
overwriting. Its own docstring says this is "what allows the variance of the trained
prior to be measured rather than assumed." That measurement had never been taken, and
`describe()` — written for exactly this purpose — had no caller outside the tests.
Reading the cache back turned out to answer a question the result files cannot, and to
expose a defect in a cost claim published hours earlier in §29.3.

### §30.1 — Cache hits were being recorded as zero-cost pre-training

The timing bracket in `bo_experiment.py` wrapped `prior_cache.cached(...)` rather than
the compute lambda, so a cache hit recorded the `torch.load` time as the cost of
pre-training. Measured on `stage1_pd1_bo_armC`:

| config | median `hyperbo_pretrain_s` | cells under 5 s |
|---|---|---|
| `base` | 882.4 | 0 / 108 |
| `canon_hyperbo` | 0.031 | 108 / 108 |
| `shrink_0.1` | 0.030 | 108 / 108 |
| `init_naive`, `meanprior`, `covprior`, `mean_linear` | ~0.03 | 108 / 108 each |

`base` is the first config run for each prior, so it always misses and always pays; every
later config hits. **Any cost analysis over a non-`base` config therefore reports HyperBO
pre-training as approximately free.**

§29.3 survives, but by luck rather than design: `analysis_43` defaults to `config="base"`.
Re-running it after the fix reproduces the published numbers exactly — LCBench
`em_frozen` 26.7 s vs `hyperbo_frozen` 3428.4 s, armC 405.5 s vs 845.1 s, both with
0/108 cells excluded. The claim stands; the instrument that produced it did not.

Three fixes:

1. `prior_cache.cached()` now times the compute itself and reports `hit` / `compute_s`
   through an optional `stats` dict. Pre-training times in `timings` are now REAL fitting
   seconds, with `prior_cache_hits` / `prior_cache_calls` recorded alongside.
2. On the LCBench path `em_pretrain_s` started its clock BEFORE the optional
   "HyperBO first" block whose kernel seeds EM, so `--em-canonical hyperbo` charged
   HyperBO's fit to EM. That fit is now attributed to HyperBO and subtracted from EM.
3. `stage4.pretrain_cost` refuses cache-served cells instead of averaging their zeros,
   and reports `NO COST DATA` for the affected method.

Fix 3 took three attempts, each failing in a way this project has a name for:
- `val <= 0.0` — never fired, because a hit records 0.03 s, not 0.0. A test written
  against it would have passed. This is "an instrument that cannot register the effect."
- comparing the TOTAL — `em_additive_hyperbo` pays `em + hyperbo`; with only the HyperBO
  half cached the total is still ~416 s and passes a total-based threshold while
  understating the true cost by ~845 s. Now every cacheable COMPONENT is tested.
- the surviving version uses a `CACHE_HIT_SECONDS = 1.0` threshold, justified by a
  four-order-of-magnitude gap: hits are 0.03–0.05 s, the cheapest real fit in this
  project is 35 s.

### §30.2 — The cache now covers both harnesses

`bo_diagnose.py` had no cache, and both drivers deliberately withheld the flag because an
unknown flag once crash-looped 96 regression cells (R7-1). So all 252 regression cells
re-fitted a HyperBO prior they could have loaded.

The precondition was the RNG re-seed guard added earlier this session (§28.6). A cache hit
skips a pre-training block; without a guard that leaves the global RNG in a different
state than a miss, and every downstream draw shifts. With the guard, each baseline
re-seeds to `pretrain_seed * 1009 + {11,13,17}` before its own block, and all
post-pre-training randomness uses local `torch.Generator()` objects, so hit and miss are
observationally identical. **Verified rather than argued:**

```
cold (miss, writes cache) == warm (hit) == cache-off      : True, True
config keys that differ                                    : {out, prior_cache_mode}
HyperBO pre-training: 2.4 s cold  ->  0.0 s warm
```

The warm run was confirmed to have actually hit the cache before the equality was
believed; an equality test against a run that silently missed would have proved nothing.

`bo_diagnose` gets its own `_diag_hyperbo_params`, NOT `bo_experiment`'s. That harness
hardcodes `loss_type="NLL"` and passes `args.meta_iters`, and does not define the
`--hyperbo-*` or `--ekl-*` flags at all; keying a fit under a description it does not
satisfy would hand a run someone else's model, invisibly. PACOH and ABLR are cached on
this harness too.

A second benefit beyond speed: served from cache, `hyperbo_frozen` is byte-identical
across configs BY CONSTRUCTION, rather than merely expected to be. That is the §27.38
control-invariance property the task-7 acceptance gate was written to check.

### §30.3 — HyperBO's weights are unstable; its behaviour is not

`prior_stability.py` (new) reads the 423 cached fits, groups by every key field except
the seed to recover independent fits of the same model, and compares them.

**Parameter space** — 47 distinct models, each fitted 9 times, 3687 free parameters after
masking the constraint buffers that hold ±inf:

```
hyperbo   n_models=47   median relative L2 = 1.371   range [1.358, 1.402]
```

Relative L2 is `‖a−b‖ / mean(‖a‖,‖b‖)`. Two random vectors of equal norm sit at √2 ≈
1.414. **At 1.371 the fits are close to mutually orthogonal: two HyperBO pre-training runs
that differ only in seed land in almost entirely different places in parameter space.**

**Result space** — two-way decomposition on `stage1_pd1_bo_armC`, budget-to-0.01, 9 priors
× 12 task shards:

| method | sd_prior | sd_task | sd_resid | prior frac |
|---|---|---|---|---|
| `hyperbo_frozen` | 2.113 | 12.801 | 4.623 | 0.17 |
| `em_frozen` | 1.428 | 14.299 | 5.858 | 0.06 |
| `em_additive_hyperbo` | 1.376 | 13.640 | 5.587 | 0.06 |
| `em_additive_hyperbo_frozen` | 0.822 | 12.516 | 4.715 | 0.03 |

**Near-orthogonal weights produce nearly identical behaviour.** The 25k-step fit is
massively non-identifiable — the data does not pin down the parameters at all — yet the
predictive function it induces is stable to within a couple of budget units, against task
heterogeneity of 12–14. The model is heavily over-parameterised relative to what the
pre-training data constrains.

**The `em_frozen` control.** `--pretrain-seed` also enters the BO seed
(`_bo_seed = 1000*ei + seed + 100003*pretrain_seed`), so the prior axis is NOT pure: it
moves the initial design as well. EM pre-training is deterministic given the data, so
`em_frozen`'s sd_prior of 1.428 is a pure evaluation-noise floor. HyperBO's genuine
pre-training contribution is the excess: √(2.113² − 1.428²) ≈ **1.56 budget units**, about
one eighth of the task effect.

**Consequence for §27.26.** Replicating over priors instead of seeds was adopted on the
belief that prior-draw variance dominates. On armC it does not — it is the SMALLEST of the
three components. The 9-prior axis is still the right error bar for "the method as
deployed", because it varies prior and initial design together the way a real user would
experience them, but it should no longer be described as isolating pre-training
randomness. Isolating that requires varying `pretrain_seed` with `_bo_seed` held fixed,
which no current cell does.

Not measurable on LCBench: one shard per prior, so the two-way layout is unidentified
there.

### §30.4 — Not done

- PACOH and ABLR are cached on `bo_diagnose` but still uncached on `bo_experiment`
  (6 raw call sites).
- `run_pd1_full` still bypasses the cache entirely — the only HyperBO call site in
  `bo_experiment.py` that does.
- EM is deliberately NOT cached: `EMPriorContainer` holds live modules, Round-4 E6
  records an aliasing hazard, `_BlendedMean.beta` is lost through `load_state_dict`, and
  EM is the cheap one.
- `test_wired_into_the_expensive_pretraining_path` still greps source text rather than
  observing behaviour.

### §30.5 — A PACOH shape bug in the noise-inclusive predictive

**Scope correction, written after the fact.** I first reported this as a launch blocker
that would have crash-looped all 252 regression cells. That was wrong, and the error is
worth recording because it is the same one this section is about. I smoke-tested the
`pd1_reg` regime by passing `REGIMES['pd1_reg']['flags']` and NO `--methods`, so the cell
ran `bo_diagnose.DEFAULT_METHODS`, which includes `pacoh_frozen`. The driver never does
that: `gen_stage_queue` passes `EM_METHODS` (stages 1-3) or `BASELINE_METHODS` (stage 0),
and **neither contains PACOH** — a grep for `pacoh` across both lists and all five regime
flag strings returns zero. Every existing `stage*_*_reg` shard confirms it, carrying only
`em_frozen, em_additive_hyperbo, em_additive_hyperbo_frozen, hyperbo_frozen`.

So the bug is real, and fatal for anyone invoking `bo_diagnose` with default methods, but
it blocked nothing that was queued. I measured the harness with flags the harness does not
use, then reported the result as if it described the harness.

Found by smoke-testing one real `pd1_reg` cell before launching the 252-cell re-run,
rather than by launching it.

```
RuntimeError: The size of tensor a (195) must match the size of tensor b (1950)
```

1950 = 195 × 10 = points × PACOH particles. `_posterior_moments`
(`bo_experiment.py:1344`) documents that PACOH returns a `GaussianMixturePosterior`
whose `.variance` keeps the per-particle MCMC dim, and marginalizes it with
`mixture_variance`. The noise-inclusive block added later for §27.33 called
`.variance.reshape(-1)` directly and did not. The latent path marginalized; its twin,
written from the same posterior four lines below, did not.

It was fatal rather than skipped because the shape mismatch does not surface at the
posterior call — that succeeds and returns 1950 elements. It surfaces ~40 lines later at
the `nll_obs` division, which sits OUTSIDE the per-method `except`, so one bad method
killed the entire cell instead of being dropped with a warning.

Reproduced with no cache flag at all, so it is not caused by the §30.2 caching work; it
dates from the commit that added the noise-inclusive predictive. Fixed by mirroring the
mixture handling. After the fix all 9 methods complete and `nll_obs` is genuinely
distinct from `nll` (`hyperbo_adapt` 0.919 → 0.388, `pacoh_frozen` 1.665 → 1.749).

**A second, quieter problem in the same block.** The `except` was bare, and on failure
`nll_obs` falls back to the latent `nll`. `ABLRSurrogate.posterior()` takes no
`observation_noise` kwarg, so **every** ABLR cell in the corpus records a
"noise-inclusive" NLL that is really the latent one, and it is indistinguishable from a
genuine equality — `ablr 1.065 / 1.065` above is not a finding, it is a missing
measurement. The handler now says so at runtime. Any cross-method reading of `nll_obs`
must exclude ABLR until its surrogate accepts the kwarg.

**Method note.** My first pass concluded the aggregation ALSO dropped these metrics,
because `nll_y` was `None` for all nine methods. That was wrong: the local is `nll_y`,
the output key is `nll_obs`, and `_mean(agg[m], 7..9)` writes them correctly. I had
measured with the wrong instrument — the same failure this section is about.

### §30.6 — The induced kernel is 8x steadier than the weights that encode it

**Question.** Would a pre-trained GP's kernel be a better Inverse-Wishart scale matrix Ψ
(shrinkage target) than the current spherical μI? For Σ ~ IW(Ψ, ν) the posterior mean is
`(T·S + Ψ)/(T + ν − K − 1)`, i.e. linear shrinkage toward `Ψ/(ν−K−1)` — so Ψ already *is*
the target, and swapping it is a change of target, not a new regulariser. Ledoit-Wolf's
optimal intensity is `α* = ΣVar(s_ij) / (ΣVar(s_ij) + ‖Σ − F‖²_F)`: a better-specified
target *raises* optimal α, so you shrink harder and buy more variance reduction.

**The blocker this measures.** §30.3 found HyperBO's parameters near-orthogonal across
pre-training seeds (relative L2 1.371 against √2 ≈ 1.414 for orthogonal). If the induced
kernel were that unstable, using it as Ψ would inject pre-training noise into every fit.
But weight instability does not imply functional instability — a deep kernel is heavily
over-parameterised, so many weight vectors induce nearly the same covariance. §30.3
measured the parameters; it could not speak to the function.

**Method.** `prior_stability.py --kernel-stability`. For each group of ≥3 independent fits
of the same model (differing only in `pretrain_seed`), rebuild the covariance through the
library's own `HyperBOModel.from_pretrained` — not a reimplementation — evaluate it on a
FIXED X shared across all fits in the group, and take the same pairwise relative L2 used
for weights. Two numbers, because they answer different questions: raw K includes
outputscale, while corr K = K_ij/√(K_ii·K_jj) divides amplitude out. **corr K is the one
that matters for a shrinkage target**, since ν (through α) already sets how much total
mass the prior contributes.

**Result.** 48 models, 9 fits each, zero skipped.

| quantity | median | read |
|---|---|---|
| weight relative L2 | **1.370** | near-orthogonal — reproduces §30.3 |
| raw K relative L2 | **0.169** | |
| corr K relative L2 | **0.165** | IQR [0.147, 0.193], range [0.115, 0.224] |

**The induced kernel is ~8.3x steadier than the parameters encoding it**, and the spread
is tight — the worst of 48 models is 0.224, so this is not an average concealing failures.
The one dim=7 model is instructive: raw K 0.389 against corr K 0.218, so most of its
apparent instability is amplitude, which Ψ's scale absorbs anyway. That is why both are
reported.

**Verdict: green light, with a caveat.** 0.165 is "moderate", not "near-identical"
(<0.05). A pre-trained Ψ carries roughly 17% relative variation across pre-training seeds
— usable, but not free, and any gain must clear that noise floor.

**Two things still unresolved, and neither is measured here.**

1. **OAS as published does not transfer.** Its derivation assumes the spherical target
   (`rblw_alpha` computes `mu = S.diagonal().mean()`). Ledoit-Wolf generalises to
   structured targets; OAS does not. The clean repair is to whiten —
   `Σ̂ = F^½ · shrink(F^{−½} S F^{−½}) · F^½` — because in whitened coordinates the target
   IS the identity, so the existing estimator applies unchanged. Marchenko-Pastur is
   likewise inapplicable directly, since MP-based nonlinear shrinkage is
   rotation-equivariant precisely because it assumes no basis knowledge; an informed
   target is basis knowledge. It does apply to the whitened matrix.
2. **Data reuse.** HyperBO is pre-trained on the same tasks that form S, so
   `Cov(s_ij, f_ij) ≠ 0` and the fixed-target formula is biased. Cross-fitting across
   pre-training tasks would fix it, but T = 23 leaves little room.

Neither is a reason not to try it; both are reasons the first result will need care.

### §30.7 — Whitened-target OAS: naive whitening fails, ridged whitening wins ~21%

**Prototype.** `prior_stability.py --whitened-oas`. Σ̂ = F^½·shrink(F^−½ S F^−½)·F^½, so
that OAS's published derivation — which assumes the spherical target, baked in via
`mu = S.diagonal().mean()` — applies verbatim in coordinates where the target *is* the
identity. Implemented by whitening the DATA ROWS so `oas_alpha` is reused rather than
reimplemented; the whole argument for whitening is that it inherits OAS's guarantees,
which only holds if it calls the same code.

The algebra collapses to `Σ̂ = (1−α)S + α·μ_w·F`, ordinary linear shrinkage toward a
scaled F.

**Structural checks (all exact).**

| check | result | |
|---|---|---|
| F=I reduces to plain OAS | dα = 0.00e+00, dΣ = 0.00e+00 | **exact**, not approximate |
| invariant to scale of F | dα = 1.8e-12 | only F's SHAPE matters |
| affine equivariance of α | dα = 1.2e-10 | |
| Σ̂ stays PD | min eig 9.0e-06 | |

The scale-invariance result matters beyond hygiene: it means the estimator ignores F's
amplitude entirely. §30.6 found the pre-trained kernel's amplitude is its *least* stable
part (raw K 0.169, and 0.389 for the dim=7 model) while its shape is steadier (corr K
0.165). **The part of the pre-trained kernel that is unstable is exactly the part this
estimator discards.**

**Naive whitening is catastrophic, and that is the finding.** First pass, pure F: the
*informed* target came out **+2947% worse** than spherical. Not a weak effect — a
diagnostic. An RBF Gram matrix on 20 points has cond(F) = 1.4e7, numerically
rank-deficient, so F^−½ has eigenvalues ~1e3 and the whitened S_w is dominated by
directions where F carries no signal at all. μ_w averages over those blown-up directions.
Whitening presumes an invertible target; a GP kernel is the opposite of one. My earlier
note that F "needs jitter" badly understated this.

**Ridged whitening.** F_τ = (1−τ)·F̂ + τ·I on a unit-mean-diagonal F̂, so τ=1 *is* the
spherical control and the sweep contains its own baseline. K=20, T=15, 200 paired trials,
Frobenius error to the true Σ:

| target | τ=0 | 0.01 | 0.05 | 0.20 | 0.50 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| informed F | +2947% | −19.4% | **−21.2%** | −12.9% | −5.9% | −0.8% | 0.0% |
| uninformative F | +1443% | +71.9% | +11.5% | −0.3% | −0.7% | −0.1% | 0.0% |

Baseline (spherical OAS) 5.645; best informed cell 4.451 ± 0.101 sem, paired.

Three readings. **A well-specified target buys ~21%**, consistent with Ledoit-Wolf theory
where a better target raises optimal α (α climbs 0.232 → 0.299). **The optimum in τ is
interior**, so this introduces a hyperparameter that did not exist before — the honest
cost of the method. **Failure is bounded but only if τ is not tiny**: an uninformative
target at τ≥0.2 reverts to spherical (−0.7% to −0.1%) rather than doing harm, but at
τ=0.01 it still costs +72%.

**Caveat on scope.** This is a simulation with known Σ_true and F drawn from the same
kernel family — a best case for target specification. It establishes the estimator is
correct and that an informed target *can* pay, not that HyperBO's kernel *will*. The
data-reuse bias of §30.6 (F pre-trained on the same tasks forming S) is still unaddressed
and cannot be seen in this design at all.

**Open question, not resolved here:** τ is currently hand-set. It may be selectable from
F's spectrum (e.g. from its effective rank) rather than swept, which would remove the new
hyperparameter. Worth settling before this goes into a real sweep.

## §31 — The iw-nu sweep: a saturated knob, and why the null is not the finding

1080/1080 shards, 0 regime failures. 540 per arm, all carrying
`provenance.code_commit = 6f583e3e1590`, zero unstamped. `code_dirty` is true on all of
them; the dirt is `BUCK` (adding `:em_noise_selection` to `prior_stability`'s deps) and
generated result files — `sl status` on the harness sources themselves is empty, so the
binary was built from committed source.

**Estimand.** Fixed-task. Each cell is 9 priors x 12 shards over the same 23 PD1 tasks;
replication is over PRIORS, so t(8)=2.31 and the interval is between-prior. This is not
the task-clustered estimand and does not support claims about new tasks, where T=23 floors
the achievable precision (§27.29). armA and armC are kept separate throughout.

### 31.1 The headline: ν selection changes nothing, because it cannot

Budget-to-target, paired by prior against `iwnu_manual`, armA:

| method | control | oas | ledoit | whiten005 | whiten020 |
|---|---|---|---|---|---|
| em_additive_hyperbo | 23.81 (sd 1.58) | −0.02 ±0.04 | +0.00 ±0.00 | +0.03 ±0.09 | −0.02 ±0.04 |
| em_additive_hyperbo_frozen | 22.57 (sd 2.38) | +0.00 ±0.00 | +0.00 ±0.00 | +0.00 ±0.00 | +0.00 ±0.00 |
| em_frozen | 29.26 (sd 2.04) | +0.00 ±0.00 | +0.00 ±0.00 | +0.02 ±0.03 | +0.01 ±0.03 |
| hyperbo_frozen | 21.81 (sd 1.83) | +0.00 ±0.00 | +0.00 ±0.00 | +0.00 ±0.00 | +0.00 ±0.00 |

Null, and an extremely tight one: effects larger than ~0.1 evaluations are excluded
against a between-prior sd of ~2. `budget_to_0.001` is the same picture. But the exact
zeros are the tell, and they are not a coincidence — **the raw trajectories are
bit-identical** between `iwnu_manual` (ν unset) and `iwnu_whiten005` (ν=442.2) for every
method, with `use_covar_prior=True` in both.

So this is not "ν selection does not matter." It is "ν did not reach the outcome at all."

### 31.2 Why: Ψ is scaled BY ν, so the intensity is pinned near 1

The M-step (`em_empirical_gp.py:427`) is `Sigma_new = (scatter + Psi) / (K + nu + N + 1)`
with `Psi = (nu + N + 1) * Sigma_init`. Writing `c = nu + N + 1`:

    Sigma_new = [K/(K+c)] * (scatter/K) + [c/(K+c)] * Sigma_init

so the realized shrinkage intensity toward `Sigma_init` is **α = (ν+N+1)/(K+ν+N+1)**,
where N = N_inducing ≈ 400 and K = 25 pre-training tasks. And ν is *validated* to exceed
N−1, so ν ≥ 399 always. That floors α at 0.970 and it rises from there:

| ν | 400 | 500 | 1 000 | 10 000 |
|---|---|---|---|---|
| realized α | 0.9697 | 0.9730 | 0.9825 | 0.9976 |

Against what the estimators asked for:

| cell | intended α | ν | realized α (armA) |
|---|---|---|---|
| iwnu_ledoit | 0.075 | 402.7 | 0.9698 |
| iwnu_oas | 0.133 | 404.4 | 0.9699 |
| iwnu_whiten020 | 0.575 | 430.8 | 0.9707 |
| iwnu_whiten005 | 0.652 | 442.2 | 0.9710 |

**Intended α spans 0.075–0.652, a 9x range. Realized α spans 0.9698–0.9710, a range of
0.0012.** The IW prior as parameterised cannot express anything but near-total shrinkage
to `Sigma_init`, for any ν a user is permitted to pass.

This also invalidates the inversion underlying all three estimators, mine included.
`estimate_iw_nu` documents α = (ν−K−1)/(T+ν−K−1) and inverts it to get ν. The
implementation's actual α is (ν+N+1)/(K+ν+N+1). Those are different functions, so the ν we
computed was the right answer to the wrong equation. **§30.8's premise was correct — the
target really is the kernel — but the knob I wired it to is inert.**

### 31.3 The one contrast that is not degenerate: the prior itself is costly

Because ν is inert, `iwnu_manual` is simply "covar prior ON". Against the covar-prior-OFF
`stage1 base`, NON-PAIRED across sweeps, budget_to_0.01:

| method | armA OFF | armA ON | delta | armC OFF | armC ON | delta |
|---|---|---|---|---|---|---|
| em_frozen | 21.66 | 29.26 | **+7.60 ±1.88** | 39.73 | 41.24 | +1.50 ±1.62 |
| em_additive_hyperbo | 20.28 | 23.81 | **+3.53 ±2.06** | 39.94 | 38.67 | −1.28 ±1.49 |
| em_additive_hyperbo_frozen | 21.83 | 22.57 | +0.74 ±2.24 | 38.98 | 38.14 | −0.84 ±1.18 |
| hyperbo_frozen | 21.81 | 21.81 | +0.00 ±1.99 | 37.47 | 37.47 | +0.00 ±2.09 |

Positive = worse. On armA, enabling the covariance prior **costs up to 7.6 evaluations**,
which is what α≈0.97 predicts: the EM's learned covariance is almost entirely discarded in
favour of the parametric `Sigma_init`. armC is smaller and mixed (−1.97 ±1.19 for
em_additive_hyperbo at 0.001, +1.95 ±1.53 for em_frozen), consistent with §27.29's finding
that the arms behave differently.

The cross-sweep caveat is weaker than usual here, and the data says why: `hyperbo_frozen`
is **exactly 0.00** in both arms. HyperBO never touches the EM covariance prior, so it is a
negative control that ran in both sweeps and returned bit-identical numbers. That
simultaneously shows the two sweeps are run-comparable for untouched methods AND that the
instrument registers a real +7.60 when there is one to register. The null in 31.1 is a
property of ν, not of the measurement.

### 31.4 What this changes

The whitened estimator is not refuted — it was never given a chance to act. §30.7's
simulation result (−21.2% Frobenius error at τ=0.05) stands on its own; what fails is the
delivery mechanism.

The fix is already in the codebase and is not ν. `_run_em_algorithm` takes
`covariance_shrinkage`, described in-line as "a free intensity in [0,1], decoupled from the
number of inducing points" with `shrinkage_target=K_kernel_inducing` — exactly the
parameterisation the IW route cannot reach. It is currently hard-wired off
(`EM_SHRINKAGE = 0.0`). The next experiment should feed the whitened-OAS α into
`covariance_shrinkage` directly and drop the ν detour entirely.

Two cautions for that run. Given 31.3, the interesting region is **small** α — α≈0.97 is
already known to cost 7.6 evaluations on armA, so a sweep should cover roughly 0.0–0.3 and
must include α=0 as the control. And whether the estimated α (0.13 spherical, 0.65
whitened) is any good is still untested: the data-reuse bias from §30.6 is unaddressed,
and 31.3 suggests the whitened estimate of 0.65 may be far too aggressive for this problem.

## 31.5. Interlude: the target is fitted on the data it is evaluated against

Prototype in `prior_stability.py --holdout-alpha`, committed in `7cc9c4f1cb2d`.

Ledoit-Wolf's optimal intensity is `alpha* = sum Var(s_ij) / (sum Var(s_ij) + ||Sigma - F||^2_F)`,
which assumes the target `F` is FIXED. HyperBO's kernel is pre-trained on the very tasks
that form `S`, so `||S - F||^2` is shrunk by construction, the denominator collapses, and
alpha is biased upward. Refitting `F` on disjoint tasks should remove the inflation.

K=20, T=25, rank-5 target, 300 trials. alpha / Frobenius-to-`Sigma_true`:

| tau | in-sample F | held-out F | inflation |
|------|----------------|----------------|-----------|
| 0.05 | 0.3913 / 4.890 | 0.2462 / 4.101 | +58.9% |
| 0.20 | 0.3110 / 4.686 | 0.2228 / 4.225 | +39.6% |
| 0.50 | 0.2527 / 4.521 | 0.2015 / 4.277 | +25.4% |

Reference rows: no shrinkage 4.341, spherical OAS 0.1523 / 4.325.

Reuse inflates alpha by 25-59%, and the inflation shrinks as the ridge grows because
ridging dilutes `F` toward `I` and with it the data dependence. The inflation is not
cosmetic: in-sample `F` (4.890) is WORSE THAN NOT SHRINKING AT ALL (4.341), while held-out
`F` (4.101) is the only target that beats both no-shrinkage and spherical OAS (4.325). The
whitened target is not a bad idea; it is a good idea evaluated on its own training data.

Honest limit: +59% does not explain the full PD1 gap in 32.3 below (0.633 vs ~0.10 is ~6x).
Data reuse is A cause, not demonstrably the only one.

Two self-inflicted errors on the way, both re-runs of failure modes already in this file.
The first version used the raw sample covariance as the target, so `F == S`, whitening
returned the identity, alpha saturated at 0.999 and `Sigma_hat` degenerated to plain `S` --
a confident +385% that measured a degenerate case. The second omitted the ridge that
`estimate_iw_nu_whitened` applies internally; the rank-5 target is singular, `F^-0.5`
exploded, Frobenius came out at 2.5e5. Neither was visible until the no-shrinkage and
spherical-OAS reference rows were added. A metric column with nothing to compare against
cannot support a claim.

## 32. Shrinking toward the kernel: a real gain on armA, nothing on armC

`stage2shrink_pd1_bo_armA` and `..._armC`, 756 shards each, 0 regime failures.
Seven cells x 9 priors x 12 shards. Estimand is FIXED-TASK: the same 23 PD1 tasks
throughout, replication over the 9 priors, t(8)=2.31, between-prior intervals, paired by
prior against `shrink000`. Arms kept separate per 27.29.

No cell sets `--use-covar-prior`. 31.3 measured that prior at +7.60 +/- 1.88 evals WORSE
on armA, and leaving it on would pin alpha at 0.970 underneath whatever the sweep sets.
`shrink000` (alpha=0) is therefore the control.

### 32.1. The gates, run before any table

31 measured an inert knob for hours because it never checked the knob was live. All four
checks here passed first:

| gate | armA | armC |
|------|------|------|
| alpha=0 vs 0.3 changes the trajectory | yes, all 3 EM methods | yes, all 3 EM methods |
| `hyperbo_frozen` negative control | exactly 0.00 +/- 0.00, all 6 cells | exactly 0.00 +/- 0.00, all 6 cells |
| provenance stamped | 756/756 | 756/756 |
| control reproduces `stage1 base` | EXACT to 4 dp | NO -- see 32.4 |

`hyperbo_frozen` never touches the EM covariance, and it reads exactly zero in every cell
of both arms. That is what licenses reading the non-zero numbers as real.

### 32.2. armA: modest shrinkage helps, and it replicates across tolerances

`em_frozen`, budget-to-target, negative = better:

| alpha | @0.01 | @0.001 |
|-------|-------|--------|
| 0 (control) | 21.66, sd 1.35 | 28.29, sd 1.86 |
| 0.05 | -1.16 +/- 1.40 | **-1.30 +/- 1.23** |
| **0.10** | **-2.19 +/- 1.04** | **-3.14 +/- 1.64** |
| 0.20 | **-1.97 +/- 1.34** | -1.84 +/- 2.15 |
| 0.30 | -1.19 +/- 1.54 | -0.75 +/- 1.77 |
| OAS, alpha=0.133 | **-1.73 +/- 1.33** | **-2.31 +/- 1.73** |
| whitened, alpha=0.633 | **+2.13 +/- 1.00** | +1.01 +/- 1.70 |

alpha ~ 0.10 buys 10-11% fewer evaluations. MULTIPLICITY: 9 of 56 armA tests clear t(8)
where ~3 are expected by chance, so some are noise -- but the surviving claim is the one
that replicates across BOTH tolerances with neighbouring alphas trending the same way,
which isolated false positives do not do. `em_additive_hyperbo` shows only a single
alpha=0.05 effect that vanishes at 0.001; treat it as unsupported.

### 32.3. The analytic alphas: spherical lands on the optimum, whitened overshoots

`config.resolved_em_shrinkage`, n=108 per arm:

| cell | armA | armC |
|------|------|------|
| `shrink_oas` | 0.1326 +/- 0.0021 | 0.1326 +/- 0.0021 |
| `shrink_whiten` | 0.6332 +/- 0.0541 | 0.6654 +/- 0.0431 |

This is what the sweep existed to answer. Spherical OAS estimates 0.133, essentially ON
the empirical optimum, and delivers -1.73 / -2.31 without any tuning. The whitened
estimator -- the method 30.7 built and simulated at a 21% WIN -- estimates 0.633, a ~5x
overshoot, and significantly HURTS (+2.13 +/- 1.00). It is also 25x more variable
(sd 0.054 vs 0.002). 31.5 gives a mechanism that explains part of the gap but not all of it.

The whitened target is refuted in practice on PD1, by the very thing it was built to beat.

### 32.4. armC: a plain null, and a control that does not reproduce

Every `em_frozen` delta on armC is within its interval at both tolerances: @0.01 ranges
-1.08 +/- 1.24 to +0.50 +/- 1.80, @0.001 -0.29 +/- 1.46 to +0.15 +/- 1.23. The single
significant cell (`em_additive_hyperbo` / `shrink_whiten`, -1.67 +/- 1.65 at 0.01) does not
replicate at 0.001 (-1.24 +/- 1.90) and is best read as one of the ~3 expected false
positives. The armA effect does NOT transfer to armC. Consistent with 27.29: on-grid vs
off-grid is the organising axis, and armC's budgets sit near the censoring ceiling
(39-47 of 50), which compresses any effect.

A caveat found by checking rather than assuming. armA's `shrink000` reproduces
`stage1 base` EXACTLY (21.6570 / 20.2754 to 4 dp). armC's does not (39.7585 vs 39.7343;
39.4058 vs 39.9420), with 27/36 method-runs diverging by up to 2e-1. This is NOT
nondeterminism: two identical armC runs launched by hand came back bit-identical, and
`hyperbo_frozen` matches across stages, so neither the RNG stream nor the data moved. The
divergence is confined to the EM path and is a genuine harness change somewhere in the
30-32 interval (`em_likelihood_noise` is recorded as `None` in stage1 and 0.01 here).
Consequence: on armC only WITHIN-stage comparisons are valid, which is what 32.4 reports.
armA is unaffected.

Related: armC shards carry three `code_commit` values (502 / 252 / 2) because commits
landed while it ran. `bo_experiment.py`, `em_noise_selection.py`, `bo_diagnose.py` and
`BUCK` are byte-identical across all three, so the corpus is homogeneous; the hashes record
when a shard ran, not what ran.

### 32.5. Where this leaves the shrinkage programme

Shrinking the EM covariance toward the base kernel is worth something on armA and nothing
on armC, and the free empirical-Bayes estimate that captures it is the SPHERICAL one that
the whitened work was trying to improve on. Follow-ups queued as `stage3shrink2` (33):
alpha 0.08/0.12/0.15 to see whether the armA optimum is a point or a flat bottom -- given
a between-prior sd of 1.4-1.9 a flat bottom is the more plausible answer -- and ridge
tau 0.20/0.50/0.80 with plain OAS as the tau=1 anchor IN THE SAME STAGE. tau=1 is exactly
spherical OAS, which won; tau=0.05 lost. If budget-to-target improves monotonically toward
tau=1, whitening adds nothing on PD1 and should be dropped.

## 33. The optimum is a plateau, and whitening is only a bad way of choosing alpha

`stage3shrink2_pd1_bo_armA` and `..._armC`, 864 shards each, 0 regime failures, 864/864
stamped per arm. Eight cells x 9 priors x 12 shards. Estimand FIXED-TASK, replication over
the 9 priors, t(8)=2.31, paired by prior against `s2_ctrl`. No cell sets
`--use-covar-prior`. Arms separate per 27.29.

Gates first, all passed: `s2_ctrl` vs `s2_a015` differ on all three EM methods on both
arms, and `hyperbo_frozen` reads exactly 0.00 +/- 0.00 in all fourteen cells across both
arms.

One gate is stronger than it was in 32. `s2_ctrl` reproduces 32's `shrink000`
BIT-IDENTICALLY, 108/108 shards, on BOTH arms. So the harness change diagnosed in 32.4 sits
between stage1 and stage2 and has not recurred, and stage2 and stage3 are directly
comparable -- which licenses pooling them onto one axis below.

### 33.1. Q1: the armA optimum is a plateau, not a point

`em_frozen`, armA, budget-to-target vs `s2_ctrl`:

| alpha | @0.01 | @0.001 |
|-------|-------|--------|
| 0.08 | **-2.11 +/- 0.72** | **-2.43 +/- 1.38** |
| 0.12 | **-2.04 +/- 1.25** | **-2.30 +/- 1.99** |
| 0.15 | **-2.09 +/- 1.31** | **-2.88 +/- 1.82** |

With 32's 0.10 (-2.19 +/- 1.04) and 0.20 (-1.97 +/- 1.34), the five points from 0.08 to
0.20 span -1.97 to -2.19 at tol 0.01 -- a range of 0.22 against intervals of 0.7-1.3. They
are indistinguishable. THE MINIMUM IS NOT RESOLVED AND CANNOT BE from this data; what is
resolved is a FLAT BOTTOM across roughly 0.08-0.20. That is the useful answer: any alpha in
that band buys the same ~10% saving, so the parameter does not need tuning, only
order-of-magnitude placement. Reporting an argmin here would be reading noise.

### 33.2. Q2: the whitened cells lie on the plain alpha curve

Because the controls are bit-identical across stages, every cell from 32 and 33 can go on
one axis. armA, `em_frozen`, budget@0.01, ordered by the alpha each cell actually used:

| alpha | cell | delta vs control |
|-------|------|------------------|
| 0.000 | control | 0.000 |
| 0.050 | manual | -1.16 +/- 1.40 |
| 0.080 | manual | **-2.11 +/- 0.72** |
| 0.100 | manual | **-2.19 +/- 1.04** |
| 0.120 | manual | **-2.04 +/- 1.25** |
| 0.133 | **spherical OAS (tau=1)** | **-1.73 +/- 1.33** |
| 0.150 | manual | **-2.09 +/- 1.31** |
| 0.200 | manual | **-1.97 +/- 1.34** |
| 0.249 | **whitened tau=0.80** | **-1.27 +/- 1.12** |
| 0.300 | manual | -1.19 +/- 1.54 |
| 0.405 | **whitened tau=0.50** | **-1.44 +/- 0.65** |
| 0.566 | **whitened tau=0.20** | **+1.27 +/- 0.91** |
| 0.633 | **whitened tau=0.05** | **+2.13 +/- 1.00** |

The whitened cells interleave with the manual cells and land where manual cells of the same
alpha land. tau=0.80 (alpha=0.249) gives -1.27, sitting between manual 0.20 (-1.97) and
manual 0.30 (-1.19). tau=0.20 and tau=0.05 sit on the rising branch because their alphas
are 0.57 and 0.63.

So the answer to Q2 is plainly NO. Budget-to-target improves monotonically toward tau=1,
and tau=1 is exactly spherical OAS. There is no interior optimum. The ridge helps only
because raising it lowers alpha (0.566 -> 0.405 -> 0.249 -> 0.133 as tau goes
0.20 -> 0.50 -> 0.80 -> 1), walking the estimate back toward the plateau the manual grid
already found. WHITENING CONTRIBUTES NOTHING ON PD1 BEYOND MIS-SETTING ALPHA, and should be
dropped. `--em-shrinkage-mode oas_whitened` and `--iw-target-ridge` stay in the code as
recorded negative results, not as recommended settings.

31.5 predicted larger tau would do better, and it does -- but for a reason that removes the
motivation rather than supporting it. The prediction was that ridging reduces data-reuse
inflation; what the sweep shows is that alpha alone explains the ordering, with no residual
benefit attributable to whitening. The mechanism transfers; the method does not.

### 33.3. armC: nothing, again

ZERO of 64 armA-style tests clear t(8) on armC, at either tolerance. `em_frozen` @0.01
ranges -1.31 +/- 1.44 to +0.05 +/- 2.04; @0.001 ranges -0.04 +/- 1.28 to +0.95 +/- 2.09.
This is a clean null and it replicates 32.4's null on the same arm with a different cell
set. Two independent sweeps now agree that armC is unmoved by covariance shrinkage.

Multiplicity: 11 of 64 armA tests clear t(8) against ~3 expected by chance; on armC, 0 of
64 against the same ~3 expected, which is itself mild evidence the armC null is real rather
than underpowered. The armA claims that survive are the ones replicating across both
tolerances with neighbouring alphas agreeing -- 33.1's plateau and 33.2's ordering.

### 33.4. Estimated alphas

`config.resolved_em_shrinkage`, n=108 per cell per arm:

| cell | armA | armC |
|------|------|------|
| `s2_tau020` | 0.5658 +/- 0.0457 | 0.6039 +/- 0.0461 |
| `s2_tau050` | 0.4053 +/- 0.0255 | 0.4588 +/- 0.0408 |
| `s2_tau080` | 0.2489 +/- 0.0118 | 0.2906 +/- 0.0272 |
| `s2_oas` | 0.1326 +/- 0.0021 | 0.1326 +/- 0.0021 |

alpha falls monotonically with tau on both arms, exactly as 31.5's simulation predicted.
The spherical estimate is also 10-20x more stable than any whitened one (sd 0.002 vs
0.012-0.046), and it is identical across arms because it depends only on S.

### 33.5. Where the shrinkage programme stands

Settled: shrinking the EM covariance toward the base kernel is worth ~10% of the evaluation
budget for `em_frozen` on armA, nothing on armC, and any alpha in 0.08-0.20 collects it.
Spherical OAS estimates 0.133 for free, needs no tuning, is the most stable estimator in
the sweep, and is within noise of the best manual value. That is the recommendation.

Closed: the whitened-target line from 30.7 through 32 to here. It was a reasonable idea,
it validated in simulation, and it lost on the benchmark to the simpler estimator it was
built to beat. Task 15 (sample-splitting the kernel fit) was explicitly conditional on an
interior tau optimum; there is none, so it should be closed unattempted rather than built.

Still open and untouched: why armA and armC disagree so completely. Every shrinkage result
in 32 and 33 is an armA result. 27.29 attributes the split to on-grid vs off-grid, and
armC's budgets sit at 39-47 of 50 near the censoring ceiling, but neither has been tested
directly. That, not more shrinkage variants, is where the next experiment belongs.


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
