# Adversarial review, ROUND THREE — 2026-08-17

Rounds 1 and 2 covered orchestration, metrics and flag plumbing (49 defects). Round 3
examined the layer neither had touched: **whether the harness uses the model API
correctly, and whether the science it computes is what it claims.**

It found 11 defects, two critical. The most consequential defect of all three rounds is
here (D2) — it sits underneath every EM result produced so far.

**Status key:** ✅ fixed · ⬜ open

---

## D2 — CRITICAL: the shared kernel and base mean are fitted under a hidden per-task `Standardize` ✅

`build_shared_gp_model_list` constructed `SingleTaskGP`s **without** an
`outcome_transform`, so BoTorch's `DEFAULT` applied `Standardize(m=1)` **per task**. Every
task's `Y` was re-standardised *inside its own GP* before the shared MLL was evaluated.
Verified at source: `em_empirical_gp.py` omits the argument, `gp_regression.py:145-147`
substitutes `Standardize`.

Consequences, all silent:

- **`--em-base-mean linear` was fitted against targets whose per-task mean is exactly 0**,
  so its bias collapses and its slope is scaled by `1/std_task`. It *is* in the optimiser
  (round 2 confirmed that), but it learns the slope of the wrong space.
- **`covar_s.outputscale` was calibrated to unit-variance targets** while
  `pretrain_em_prior` consumes globally-standardised `Y`. So `Sigma_init = K(Z,Z)`, the EM
  kernel init, **and the `--em-shrinkage` trace-matched target** were all on the wrong
  scale.
- `pretrained_gp_*` / `warmstart_gp` copy `raw_outputscale` into GPs built with
  `outcome_transform=None`, transplanting the mismatch into the baselines.

**This is a plausible mechanical explanation for §27.9's finding that shrinkage "hurts".**
That conclusion should be re-derived, not cited.

Fixed as an **opt-in parameter** (`outcome_transform=DEFAULT`) rather than a change to the
library default, because `em_empirical_gp.py` is part of the OSS stack and other callers
may rely on current behaviour. The harness passes `None`; the default is unchanged and
flagged for the OSS review.

## D1 — CRITICAL: `--em-mean` silently contaminated the `pretrained_gp_frozen` control ✅

`_apply_em_mean` rebound `mean_s` to a `_BlendedMean` carrying HyperBO's meta-learned mean,
and `mean_s` was then handed to `make_surrogate` for **every** method. With
`--em-mean hyperbo|blend`, the `pretrained_gp_frozen` control silently became
"SumMLL kernel + HyperBO's mean" — not a control, a second transfer method. Fixed by
threading a pre-blend `mean_baseline`.

*(Round 3 also answered the question that prompted it: `beta` **is** preserved through
`deepcopy` and through `EMPriorContainer`, and `_interpolate_prior_to_X` reads the live
module — so the blend is active at prediction time. It would be lost only via
`load_state_dict`, which this harness never uses on the mean. Now covered by a test.)*

## D3 — HIGH: `run_pd1_full` ignored eight EM flags ✅

`--em-base-mean`, `--deep-kernel`, `--em-canonical`, `--em-mean`, `--iw-nu`,
`--use-mean-prior`, `--use-covar-prior`, `--em-init-mode` were all accepted and did
nothing on `--benchmark pd1_full`. **Silent no-op #8.** Brought to parity, and the HyperBO
gate widened so the prior exists when the mean transfer needs it.

## D4 — HIGH: `--output-warp gaussian` applied two different warps in arm C ✅

`_gaussian_rank_warp` is **rank-based and therefore pool-dependent**. Arm C warps the
matched grid and the full per-task pools separately, so EM's inducing values and the
candidates end up on two different monotone scales — precisely at the off-grid points arm
C exists to test. `winsor`/`neglog` are pointwise and unaffected. Now **refused** with an
explanatory error rather than silently mis-scaled.

## D5 — HIGH: `bo_diagnose` crashed on the new mean flags ✅

The `mean_constant` dict value was guarded for `None`, but the `print` two lines later
formatted it unconditionally → `TypeError` **after all pre-training had been paid for**.

## D9 — MEDIUM: the deep-kernel fit kept the *bad* parameters ✅

The loop validated `θ_i` then snapshotted `θ_{i+1}`, so a non-finite loss on the next
iteration kept the parameters that **produced** it — the opposite of the comment's claim.
Snapshot moved before `opt.step()`.

## D6 — MEDIUM: undeclared module globals turned a `ValueError` into a swallowed `NameError` ✅

`HYPERBO_BASE_KERNEL`, `EM_CANONICAL`, `DEEP_KERNEL` were only bound inside `main()`.
`bo_diagnose` imports `make_surrogate` directly, so a missing prior raised `NameError`,
which its broad `except` swallowed — **the method vanished from the results table with no
message**, and its competitors were then averaged over a different subset of eval tasks.
Globals bound at module scope; the handler now logs the method and exception.

## D10 — MEDIUM: hyperparameter transplants crashed on composite canonical kernels ✅

`covar_s.raw_outputscale` does not exist on `HyperBODeepKernel` (its ScaleKernel is one
level down) or `_DeepKernel` (property only), and `raw_lengthscale` shapes mismatch when
ARD dims differ (39 vs 7). Both raised **outside any `try`**, aborting the whole run for
`--em-canonical hyperbo` or `--deep-kernel` combined with any `pretrained_gp_*` method.
Replaced with `_safe_copy_outputscale` / `_safe_copy_lengthscale`.

## D11 — MEDIUM: `--em-mean` was unusable on the LCBench path ✅

`_apply_em_mean` was called with `hyperbo_prior_early`, non-`None` only under
`--em-canonical hyperbo`, while the real prior is trained ~30 lines later. The error text
advised "include a `hyperbo_*` method", which does not work on that path. The gate now
folds in `args.em_mean`, matching `run_pd1_matched_loo`.

## D7 / D8 — MEDIUM: cross-suite regression numbers are not comparable ⬜

- **D7:** `bo_diagnose` NLL is in globally-standardised units and has no
  `--per-task-standardize`. Gaussian NLL contains `log σ`, so
  `NLL_reported = NLL_raw + log(ystd_benchmark)`. LCBench (accuracy, %) and PD1
  (`-error_rate ∈ [-1,0]`) differ by more than an order of magnitude in `ystd`, so **a
  constant offset separates the two suites' NLL columns**. Worse for PD1: a single global
  scale over heterogeneous tasks inflates per-task `z` by scale mismatch, not
  miscalibration — biased in the direction that makes PD1 look worse.
  **Report `nll_recalibrated` + `tau` for cross-suite claims; raw `nll` is scale-relative.**
- **D8:** the 1-step-ahead calibration in `run_bo` is recorded **only at acquired points**,
  which differ per method — a method that probes near its incumbent gets an easier
  calibration set. That is a selection effect, not a calibration measurement.

---

## Verified correct (so they are not re-litigated)

- **`pool_max` is the searched set** in all four paths — arm C reassigns `raw = Yc` with
  `Xq = Xc`, arm A uses `Y_all[held]` with `Xn`, `pd1_full` uses `datasets[ei].Y`, LCBench
  uses the BO pool (not the pretrain pool) under `--novel-config-split`.
- **Initial design is paired** across methods in all four paths (`1000*ei + seed`), and
  candidate exclusion is correct in both the `topq` and fantasy paths.
- **`_BlendedMean` broadcasting** — `HyperBOLinearMean.forward` squeezes its trailing
  singleton, so the blend yields `(n,)`, not an `(n,n)` outer product.
- **PD1 coordinate consistency** — the matched inducing grid and the per-task pools use
  the same declared `PARAM_SPECS` bounds, so EM's inducing points really are a subset of
  the arm-C candidates.
- **`_additive_base` / `_log_sigma_scale`** are properly registered and honoured by both
  `_effective_Sigma_inducing` and `_interpolate_prior_to_X`.
- **`_update_cache` is not called on the frozen path**, so the pre-trained `delta_mu`
  survives and `_BlendedMean` is not silently re-baselined.

---

## Remaining open

| id | issue |
|---|---|
| D7 | raw `nll` not comparable across suites; use `nll_recalibrated` + `tau`, or add per-task standardization to `bo_diagnose` |
| D8 | acquired-point calibration is a selection effect |
| — | **§27.9's "shrinkage hurts" must be re-derived**: D2 put the shrinkage target on the wrong scale |
| — | censoring estimator (round 2): rename to RMTT@T, pair with solve rate, add a `2T` sensitivity row |
