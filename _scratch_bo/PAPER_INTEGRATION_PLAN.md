# Integrating this stack into the ICML 2026 camera-ready

Audit of what the stack contains versus what `_ICML_2026__Empirical_GPs.zip/main.tex`
currently says, and a prioritized integration plan. Section numbers below refer to
`EXPERIMENTAL_RESULTS.md` unless prefixed with "paper".

---

## 0. Two things to fix regardless of what else is added

**0.1 — The 7D NLL table is produced by a model variant the paper never describes.**
Paper Table `tab:hd_nll` reports EM-EGP NLL of −0.039 … −0.415, dominating HyperBO. The
matching experiment (tutorial Section F: same `M=100` anchors, same
`n_train ∈ {5,10,20,50,100}`) constructs the model with
`EMEmpiricalGaussianProcess.from_pretrained(..., base_covar_module=base)` — i.e. **with a
fitted additive base kernel**, which the paper does not mention anywhere (0 occurrences of
"additive kernel"/"base kernel").

This matters because §14.2 measured the *bare* EM prior and found its NLL **grows** with
data (0.05 → 14.97 from 5 to 50 observations, 95% coverage 0.95 → 0.43). A reader who
instantiates `EMEmpiricalGaussianProcess` without `base_covar_module` will not reproduce
Table `tab:hd_nll`. **Verify which configuration produced the table and describe it.**
This is the highest-priority item — it is a reproducibility defect, not an enhancement.

**0.2 — `_ICML_2026__Empirical_GPs.zip` is sitting inside the Python package**
`botorch/models/empirical_gps/`. Harmless to the build (BUCK `srcs` are explicit) but it
will show up as untracked source-tree clutter and could be committed by accident. Move it
out of the package.

---

## 1. Technical ingredients in the stack vs the paper

| Ingredient | Implementation | In paper? |
|---|---|---|
| EM with closed-form updates | `em_empirical_gp.pretrain_em_prior` | ✅ core contribution |
| Shift/residual interpolation (continuous domain) | `em_empirical_gp` | ✅ paper §4.5 |
| **M-step covariance shrinkage** (trace-matched → base gram) | `utils.trace_matched_shrinkage`, `covariance_shrinkage=` | ❌ **derivation commented out (main.tex 647–649); listed as an open LIMITATION in the Conclusion** |
| **Additive base kernel** at conditioning | `BaseAugmentedEmpiricalKernel` | ❌ absent — yet load-bearing for Table `tab:hd_nll` (§0.1) |
| **Multi-output EGP** | `MultiOutputEmpiricalOneDimensionalGP` | ❌ absent |
| **Multi-task EGP** | `MultiTaskEmpiricalOneDimensionalGP` | ❌ absent |
| **EM prior → basis curves** (KL expansion, rank dial) | `utils.em_prior_to_basis_curves` | ❌ absent |
| **Coordinate-ascent EM / KL prior fit** | `em_coordinate_ascent` | ❌ absent |
| Joint kernel fit across datasets | `build_shared_gp_model_list` | ❌ absent |

## 2. Experimental results vs the paper

| Result | Scale / evidence | In paper? |
|---|---|---|
| 1D LC extrapolation vs power law / LC-PFN | — | ✅ main |
| GIFT-Eval | — | ✅ main |
| 7D modeling NLL vs HyperBO / pretrained / vanilla | — | ✅ main |
| 1D vs HyperBO / PACOH | — | ✅ appendix |
| **7D finite-pool Bayesian optimization** (§5J) | 192 runs, 9 methods, Friedman χ²=360, p≈5e-73 | ❌ **paper has 0 mentions of BO, regret or acquisition** |
| **PD1, faithful** (§17) | 115 runs, 23-task LOO, χ²=315, p≈3e-63 | ❌ |
| **ABLR baseline** | both benchmarks | ❌ |
| **Shrinkage as in-distribution↔transfer dial** (§16) | paired, 64 + 60 runs | ❌ |
| **Additive vs convex correction design** (§13) | 6 datasets × 3 history × 3 obs | ❌ |
| **Prior-mean extreme-region diagnostics** (§14) | 9 methods | ❌ |
| Noise robustness, OOD, meta-data scaling, `n_init`, batch BO | §5F–§5L | ❌ |
| Cost-normalized Pareto | §15 | ❌ |

---

## 3. Prioritized plan

### Tier A — main text. These change what the paper claims.

**A1. A Bayesian optimization experiment.** *(highest value)*

The paper argues for better GP priors and never demonstrates the downstream task where
priors matter most; "Bayesian optimization", "regret" and "acquisition" appear zero times.
For an ICML paper whose motivation is prior quality, this is the most likely reviewer
question, and we now have a strong, statistically-tested answer.

- Content: LCBench 7D finite-pool BO, 192 runs, 9 methods. On regret-AUC the meta-priors
  separate from `vanilla_gp`/PACOH/random with Friedman χ²(8)=360, p≈5e-73, Nemenyi CD 0.87.
- Figure: `results/figures/lcbench_highpower_trajectories.pdf` (shaded clustered SEM) and
  `..._cd_auc.pdf`.
- Honest framing to preserve: **at final regret everything ties** because the 300-config
  pool saturates; the advantage is an early-regime/AUC phenomenon. Say so — it is a more
  defensible claim than a raw win, and §5J makes the case rigorously.
- Cost: ~1 page + 1–2 figures.

**A2. PD1 as a second benchmark.** *(highest value)*

Every current experiment except GIFT-Eval is LCBench. PD1 is HyperBO's *own* benchmark, so
it is the maximally fair venue, and it directly answers "is this LCBench-specific?".

- Content: leave-one-out over 23 tasks on the 400-config matched pool, the paper's
  `−log(error)` warp, HyperBO at its specified 50k steps. All six transfer methods form the
  tied-best clique; every one is significantly ahead of `vanilla_gp`, PACOH and random.
- **Frame around cost, not victory.** EM *ties* the specialized meta-BO methods here — but
  at 52 s/fold pre-training against HyperBO's 763 s (**15×**) and PACOH's 453 s. "Matches
  specialized deep-kernel meta-BO on its home benchmark at a fraction of the pre-training
  cost" is both true and strong.
- Figure: `results/figures/pd1full_neglog_cd_auc.pdf`.
- Cost: ~0.75 page + 1 figure.

**A3. Covariance shrinkage: turn a stated limitation into a contribution.**

The Conclusion currently says the empirical covariance "can be singular, possibly requiring
shrinkage regularization such as an Inverse-Wishart prior", and the IW-MAP derivation is
**commented out** in the source. We have both halves:

- *Theory* (§13.5): trace-matched convex shrinkage is **exactly** MAP-EM under
  `Σ ~ IW(Ψ, ν)` with `Ψ = (ν+M+1)·s·B`, giving
  `Σ_MAP = (1−α)Σ_ML + α·s·B`, `α = (ν+M+1)/(K+ν+M+1)`. Plus fixed-point analysis: convex
  shrinkage is trace-bounded so a fixed point exists by Brouwer, while the additive variant
  is not trace-bounded and can diverge.
- *Empirics* (§16): the EM prior covariance has **effective rank 2.9 of 200**, and shrinkage
  is an **in-distribution ↔ transfer dial** — it significantly *hurts* in-distribution BO
  (@10 z=2.6) while significantly *helping* under distribution shift (@5 z=−2.5), and it
  fully repairs calibration (coverage 0.43 → 0.94).
- This is a clean theory+experiment package that removes a limitation. Uncomment and finish
  the derivation; put the dial result in main text or appendix depending on space.
- Cost: ~0.5 page main + appendix derivation (mostly already written in §13.5).

### Tier B — main text if space allows, otherwise appendix (but B1 is mandatory somewhere)

**B1. Additive base kernel — mandatory (see §0.1).** At minimum a methods paragraph
describing `BaseAugmentedEmpiricalKernel` (`K_emp + β·B`, β fit by MLL) and stating that
the 7D results use it. Ideally also §13.2's design study: additive beats fixed/learned
convex and the free two-scale on mean NLL, worst-case NLL and rank, because pinning the
empirical weight at 1 prevents over-shrinking on a sparse target head. Also worth including
the §13.6 correctness fix (adding the base's *full* `K_base(X,X)` at query points rather
than its Nyström projection).

**B2. Multi-output and multi-task EGP.** Two new model classes with no paper presence.
Both extend the framework meaningfully — multi-task handles partial observations and
heterogeneous per-task domains, which is exactly the "heterogeneous observation locations"
the abstract advertises. A compact appendix subsection with the tutorial Section D/E
results is the cheap option; promoting to main text is justified only if you want to claim
a model *family* rather than a single model.

### Tier C — appendix

- **C1. ABLR** (Perrone et al. 2018) as a third meta-learning baseline. Note: our
  implementation currently deviates from the paper (shared rather than per-task
  precisions); a faithful version is implemented but the tuning sweep has not been run yet,
  so hold this until it is.
- **C2. Mechanism diagnostics (§14).** The best single explanation of *why* the empirical
  prior wins zero-shot: EM and HyperBO order the whole pool equally well (Spearman 0.782 vs
  0.771) but EM's top-10 precision is 2.8× higher and its no-data pick 3.7× better. Global
  rank correlation is the wrong diagnostic; extreme-region precision is the right one.
- **C3. Robustness suite:** observation noise (§5K), OOD split (§5H), meta-data scaling
  (§5F — the canonical "regret vs #pre-training tasks" figure), `n_init` sweep (§5G),
  batch BO with fantasy qLogEI (§5L).
- **C4. Cost-normalized Pareto (§15).** Amortized over 50 tasks, EM is simultaneously the
  most accurate and the cheapest. Complements A2's cost framing.
- **C5. Implementation notes:** `em_prior_to_basis_curves` (rank-truncation dial),
  `coordinate_ascent_em`, `build_shared_gp_model_list`.

---

## 4. Suggested narrative change

The abstract currently ends "competitive performance on learning curve extrapolation and
time series forecasting benchmarks" — both *regression* tasks. With A1 and A2 the claim
can become materially stronger and still be honest:

> …and that the resulting priors translate into **more sample-efficient Bayesian
> optimization**, matching specialized deep-kernel meta-BO methods on their own benchmark
> at a fraction of the pre-training cost.

That reframes the paper from "a better way to build GP priors" to "a better way to build
GP priors, *and* evidence that it matters downstream" — which is what separates an accept
from a strong accept for this kind of contribution.

---

## 5. Caveats to carry into the paper

1. **Final-regret ties.** On LCBench everything converges by @40 once the finite pool is
   exhausted. Report early-horizon/AUC results and say why.
2. **EM ties rather than wins on PD1.** Lead with cost.
3. **Calibration ≠ optimization quality.** §16 is a counterexample *within one method*:
   NLL improved 15 nats and coverage doubled while BO regret significantly worsened. Do not
   present calibration tables as a quality ranking.
4. **`em_frozen` is overconfident and worsens with data** (§14.2). Whichever variant the
   paper reports, name it explicitly.
5. **Provenance.** §5J, §5L, §14, §16, §17 are backed by committed raw data and figures in
   `_scratch_bo/results/`; §2–§5I and §9–§13 are legacy numbers whose raw outputs were lost
   and have not been re-run under the committed pipeline. Anything quoted in the paper
   should come from the committed set, or be re-run first.

---

## 6. Ready-to-paste LaTeX: `PAPER_ADDITIONS.tex` (appendix-only)

All four blocks are written as **appendix** subsections under one new
`\section{Model Variants and Regularization}`, to avoid main-text space pressure.

| block | content | label |
|---|---|---|
| B.1 | Covariance shrinkage + MAP-EM justification + `tab:shrinkage_spectrum` | `apd:shrinkage` |
| B.2 | Base-augmented conditioning (inference-time adaptation) | `apd:base_augmented` |
| B.3 | Multi-output EGP (dense, non-separable cross-output covariance) | `apd:multioutput` |
| B.4 | Multi-task EGP (long format, partial observation) | `apd:multitask` |

**Three main-text pointer sentences** are given verbatim at the bottom of the
`.tex` file. Each is one sentence. The third is **required, not optional**: it
names the variant that produced `tab:hd_nll`. Without it the table is not
reproducible, because those numbers come from the additive-base configuration
that the manuscript never describes.

**Also worth a one-line edit:** the Conclusion currently lists shrinkage as future
work ("…can be singular, possibly requiring shrinkage regularization such as an
Inverse-Wishart prior"). That is now `\Cref{apd:shrinkage}`, with the IW
connection made exact — the sentence should point there instead of deferring.

### Numbers in B.1 — corrected, and their provenance

An earlier draft of this block quoted coverage `0.43 -> 0.94` and "~15 nats".
**Those were wrong** — they came from the §14.2 diagnostic grid (how `em_frozen`
degrades as observations accumulate), not from the shrinkage study. The table now
uses the shrinkage runs directly:

| quantity | α=0 | α=0.3 | source |
|---|---|---|---|
| effective rank (of 200) | 2.89 | 5.05 | `raw/spectrum_a{0.0,0.3}.json` |
| trace in top 22 modes | 0.9998 | 0.9624 | same |
| median eigenvalue | 2.0e-4 | 1.6e-2 | same |
| 95% coverage | 0.65 | 0.98 | `raw/rerun/shrink_a{0.0,0.3}.json` |
| predictive NLL | 8.30 | 0.10 | same |

Coverage/NLL are from the **corrected-budget re-run**; the originals agree
closely (0.653/7.198 → 0.982/0.102), so the claim is stable across both. The
spectrum is unchanged by the re-run because EM pre-training is deterministic.

**Deliberately NOT drafted:** the BO and PD1 experiments (§20–§22). Higher value,
but their fine orderings are still moving as multi-prior replicates land, and a
claim that shifts after submission is worse than one omitted. If you want one
sentence, the metric-invariant result is stable: transfer methods reach a 5%
regret target in 2.7–4 evaluations vs 13 for a from-scratch GP.
