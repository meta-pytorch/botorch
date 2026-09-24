# METHODS — what is actually being computed, and why

`EXPERIMENTAL_RESULTS.md` is a 5000-line chronological log and `HANDOFF.md` is a resume
guide. Neither explains **the method**. This file does, so that someone arriving cold can
read a result and know what it means.

Written 2026-08-21. Everything here was verified against the source, not recalled.

---

## 1. The problem

Meta-learning a Gaussian-process prior for Bayesian optimisation. We have `T` **pre-training
tasks** (e.g. 25 LCBench datasets), each fully observed on a shared grid of `M` **inducing
points** `Z`. A new **held-out task** arrives and we must optimise it in as few evaluations
as possible. The question is whether a prior learned from the other tasks helps, and how it
compares to the published meta-BO baselines.

Evaluation is **leave-one-out**: every task takes a turn as the held-out one.

---

## 2. The EM-Empirical GP

The method estimates, by expectation-maximisation over the pre-training tasks, a mean
vector and covariance matrix **at the inducing points**:

* `mu_inducing` — (M,) the EM-converged mean at `Z`
* `Sigma_inducing` — (M, M) the EM-converged covariance at `Z`
* `delta_mu = mu_inducing − m(Z)` — the *shift* away from the parametric mean

**E-step.** For each task, compute its posterior at the inducing points given its data.
In the source this is `residual = y - mu_S` followed by a solve against
`Sigma_SS + likelihood_noise·I`. So each task's curve is pulled toward the **current mean
estimate μ**, with eigendirection `λ` retained at weight `λ/(λ+σ²)`.

**M-step.** Re-estimate `μ` and `Σ` from those posteriors, optionally under an
Inverse-Wishart prior on `Σ` (`--use-covar-prior`, `--iw-nu`).

**Interpolation to arbitrary inputs** is where the method lives or dies. For a new input
set `X`:

```
mu(X)    = m(X) + W @ delta_mu
Sigma(X) = Lambda(X) + W @ Sigma_inducing @ W^T   [+ K_base(X, X)]

W        = K(X, Z) @ K(Z, Z)^-1              <- the Nystrom map
Lambda(X)= K(X, X) - W @ K(Z, X)             <- the Nystrom residual
```

**The single most important consequence:** if `X ⊆ Z`, then `W` is a selection matrix and
`Lambda(X) = 0`, so the empirical covariance is used *directly* and the canonical kernel
barely matters. Off the grid, every prediction is extrapolated **through** `K`, and quality
becomes hostage to it. That is the on-grid/off-grid axis that organises nearly every result
(§27.29, §27.34, §27.39).

---

## 3. The benchmarks, and what the arms mean

| | LCBench | PD1 |
|---|---|---|
| tasks | 35 datasets | 23 evaluable |
| shared grid | 2000 configs | 400-point Halton |
| per-task extra | — | ~1959 more configs |
| geometry | **on-grid** | **off-grid available** |

PD1 has two pool choices, and crossing them gives the arms. Only two flags differ:

| arm | `--pd1-candidate-pool` | `--pd1-pretrain-pool` | isolates |
|---|---|---|---|
| **A** | matched | matched | fully on-grid |
| **B** | full | matched | *interpolation alone* (vs A) |
| **C** | full | full | matches Wang et al.; *pre-training pool* (vs B) |

Arm B exists because A and C differ in **two** flags at once, so A-vs-C confounds the two
effects. §27.39 used B to separate them: going off-grid alone costs EM 4.97 evaluations
(t=+4.41); richer pre-training returns about half of it.

**Caveat on arm A:** the matched pool saturates within 50 iterations, so *final regret* is
near-uninformative there and **budget-to-target** is the metric that discriminates.

---

## 4. Methods compared

**EM variants** — `em_frozen` (pure empirical prior, no fitting), `em_finetuned` (adds a
trainable base kernel), `em_additive_hyperbo[_frozen]` (empirical Σ **plus** HyperBO's
kernel additively), `em_noisefit` (fits only the conditioning noise), `em_then_vanilla_k`.

**Baselines** — `hyperbo_frozen`/`_adapt`, `ablr`/`_adapt`, `pacoh`/`_adapt`,
`pretrained_gp_frozen`/`_tuned`, `vanilla_gp`, random.

`hyperbo_frozen` doubles as a **control**: it never touches the EM prior, so it must be
identical across configs within a regime. It is on the BO regimes; it is **not** on
`lcb_reg`, which is how the §27.38 RNG bug was found.

---

## 5. Metrics

**Primary: `budget_to_target`** — evaluations to reach a fixed regret threshold (0.01).
Lower is better. Preferred over final regret because of the arm-A saturation above.

**Regression metrics** (`bo_diagnose`): `rank_corr`, `rmse`, `nll`, `calib_ratio = σ/RMSE`,
`coverage95`, each at `n_obs ∈ {5, 20, 50}`.

**Dual calibration (§27.33).** Every v2 regression result carries *both* the latent
posterior `N(μ, k**)` and the noise-inclusive `N(μ, k** + σ²)` — the `*_obs` variants. They
answer different questions ("how well is f known" vs "how well is the next observation
predicted") and recording both means the corpus reads either way without a re-run.

**Statistics.** Paired across `P = 9` pre-training seeds ("priors"), `t = mean(d)/(s_d/√P)`
with `df = P − 1`, so `t(8) = 2.31`. The variance decomposition is
`Var(θ̂) = σ²_B/P + σ²_W/B`: the first term depends only on the number of priors, the second
only on total budget. That is why P was raised to 9 rather than adding seeds.

---

## 6. The regularisation story, in one place

`likelihood_noise` in the E-step is **not** an observation-noise level — it is a **ridge
coefficient** that selects an effective rank for `Sigma_inducing`. It was hardcoded at 1e-2
for the entire study before §27.35.

* `Sigma_inducing` from `T` tasks has **rank ≤ T−1**, but lives in `M` dimensions.
* At σ²=1e-2 the E-step retains ~all directions of what is often a rank-5 object.
* The optimum is **σ² ≈ 0.6**, flat in `T` (§27.44) — worth −78% RMSE at n_obs=5.
* It is **not** misspecification: data drawn from exactly the assumed model, observed
  noiselessly, still wants σ²=1–2 (§27.45). It is finite-sample severity — `Sigma_inducing`
  has **147% relative error at T=6**, matching `√((r_eff+1)/T)`.
* **The better fix is not to tune it.** Ledoit–Wolf/OAS shrinkage applied to
  `Sigma_inducing` directly beats the best tuned σ² at ~zero noise (§27.44), and is
  equivalent to empirical-Bayes selection of the Inverse-Wishart `ν` we already expose:
  `α = (ν−K−1)/(T+ν−K−1)`. Implemented as `--iw-nu-mode`.

See `.llms/rules/em_noise.md` for the operating rules and the refuted hypotheses, and
§27.47 for the references (Chen et al. 2010 for OAS/RBLW; Ledoit–Wolf 2004).

---

## 7. Where things are

| | |
|---|---|
| `bo_experiment.py` | the BO harness — all arms, all methods |
| `bo_diagnose.py` | the regression harness. **Imports `bo_experiment`**, so editing either rebuilds both |
| `em_noise_selection.py` | LW / OAS / RBLW / MP-edge / `estimate_iw_nu`. Own target, deliberately not in `lib` |
| `gen_stage_queue.py` | regime and cell definitions; `--kind {baseline,ofat,replicate,means,noise}` |
| `summarize_stage.py`, `analyze_pd1pool.py` | analysis |
| `results/run_*.sh` | the runners; all use `wait_for_unit`, which treats an unreadable systemctl state as "keep waiting" |
| `results/raw/v2/<stage>/` | per-stage shards, `SUMMARY.json`, `PROVENANCE.json` |

**Reading order for a newcomer:** this file → `HANDOFF.md` §0 (live state) and §2 (current
claims) → `EXPERIMENT_PLAN.md` (design) → `EXPERIMENTAL_RESULTS.md` only for the specific
§ numbers cited above. Do not read the results log front to back; it is a chronological log
including retracted work, and §3 of `HANDOFF.md` lists what not to cite.
