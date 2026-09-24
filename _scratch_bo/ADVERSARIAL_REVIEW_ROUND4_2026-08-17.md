# Adversarial review, ROUND FOUR — 2026-08-17

Run after rounds 1-3 (60 defects, all believed fixed, 55 tests guarding them). Goal: find
what all three missed, and whether the round-3 fixes regressed anything.

**They had.** Three round-3 "fixed" items were inert or partial, and one converted a loud
crash into a silent wrong baseline — this project's signature failure mode, introduced by
a hardening fix. 23 new findings.

**Fixed in this pass:** R1 (mean_baseline was dead — now threaded through all three call
sites and pre-bound), R2 (safe-copy wired at 1 of 4 sites — now all 4), R3 (silent
degradation — helpers now warn, and a missing shared kernel raises).

**Still open:** R4-R6, E1-E6, S1-S4, N1-N4. See the table below; these are the queue.

Verbatim agent output follows.

---

# FOURTH-ROUND ADVERSARIAL REVIEW â 2026-08-17

## Direct answer

**Yes â three of the roundâ3 fixes are inert or halfâapplied, and one converted a loud crash into a silent wrong baseline.** I found **23 new defects**, of which 4 are critical, 8 high. Two roundâ3 "fixed â" items (D1, D10) and one roundâ3 "verified correct" claim are wrong as written.

---

## Severity summary

| id | area | severity | one line |
|---|---|---|---|
| R1 | roundâ3 regression | **CRITICAL** | `mean_baseline` is dead code â D1 is **not fixed** |
| R2 | roundâ3 regression | **CRITICAL** | `_safe_copy_*` wired at 1 of 4 crash sites â D10 **Â¾ unfixed** |
| R3 | roundâ3 regression | **CRITICAL** | the one site that *was* fixed now fails **silently** instead of crashing |
| R4 | roundâ3 regression | **HIGH** | D4 guard keys on the wrong precondition; two unguarded doubleâwarp paths remain |
| R5 | roundâ3 regression | HIGH | D1/D11 not applied to `bo_diagnose` at all |
| R6 | roundâ3 regression | **HIGH** | 6 of 30 stageâ1 OFAT cells are guaranteed to abort at runtime; stage never reaches `.DONE` |
| E1 | EM model | **HIGH** | `--cond-noise` documented as *std*, used as *variance* (31Ã error) |
| E2 | EM model | HIGH | shrinkage is **traceâpreserving** â it cannot "fill" the null space, it only rescales |
| E3 | EM model | MEDIUM | roundâ3's "`_update_cache` not called on the frozen path" is **false** |
| E4 | EM model | MEDIUM | `enable_interpolation=False` is silently ignored when inducing points exist |
| E5 | EM model | MEDIUM | `forward()` and the EM MLL use **different priors** for the same model |
| E6 | EM model | LOW | container tensors/modules are shared & mutated across every surrogate |
| S1 | statistics | **CRITICAL** | fixedâtask SE omits **betweenâprior** variance entirely â z inflated |
| S2 | statistics | **HIGH** | LOO folds are not independent; SE is antiâconservative, VIF â 1+(Tâ1)Ï |
| S3 | statistics | HIGH | weighted mean + unweighted SEM in the headline table |
| S4 | statistics | MEDIUM | budgetâtoâtarget omits the `n_init` initial evaluations |
| S5 | statistics | MEDIUM | two different `budget_to_target` functions, same name, different estimator |
| T1..T7 | tests | see Â§4 | vacuous / tautological / fixtureâasserting tests |
| N1 | never looked at | **HIGH** | `--batch-mode` / `--n-fantasies` are silent noâops on **both** PD1 paths |
| N2 | never looked at | MEDIUM | shardâcount *increase* silently mixes two shardings |
| N3 | never looked at | MEDIUM | NaN posterior â silent deterministic pick, no finite check |
| N4 | never looked at | LOW | latent seed collision `7k+13` â initâdesign seeds |
| N5 | never looked at | â | `_shard_folds` **verified correct** |

---

## 1. Regressions and halfâfixes from round 3

### R1 â CRITICAL: `mean_baseline` is dead code. D1 is **not fixed**.

`make_surrogate` grew the parameter and the branch:

- `<botorch>/botorch/../_scratch_bo/bo_experiment.py:853` â `mean_baseline=None,`
- `.../bo_experiment.py:916` â `base_mean_src = mean_s if mean_baseline is None else mean_baseline`

and the three producers assign it:

- `.../bo_experiment.py:1761` (`run_pd1_matched_loo`)
- `.../bo_experiment.py:2080` (`run_pd1_full`)
- `.../bo_experiment.py:2839` (`main`, LCBench)

**But none of the three `surrogate_fn` closures pass it:**

- `.../bo_experiment.py:1812-1821`
- `.../bo_experiment.py:2106-2115`
- `.../bo_experiment.py:2955-2964`

All three call `make_surrogate(method, X, Y, em_prior, mean_s, covar_s, hyperbo_prior, pacoh_prior, ablr_prior)` â nine positional args, no tenth. So `mean_baseline` is always `None` inside `make_surrogate`, `base_mean_src` always resolves to `mean_s`, and `pretrained_gp_frozen` still receives the `_BlendedMean` under `--em-mean hyperbo|blend`. **The control is still a second transfer method.** `mean_baseline` is a writeâonly local in all three functions (flake8 `F841` would flag it).

Fix: add `mean_baseline` to each closure. There is no test that would catch this â see T4.

### R2 â CRITICAL: `_safe_copy_*` is called at exactly one of four crash sites. D10 is Â¾ unfixed.

Guarded (1 site): `.../bo_experiment.py:922-923`.

Still raw and **outside any `try`**:

| site | lines | expression |
|---|---|---|
| `warmstart_gp` | `bo_experiment.py:967-972` | `cc.raw_outputscale.data.copy_(covar_s.raw_outputscale.data)` and `cc.base_kernel.raw_lengthscale...` |
| `em_additive_warmbase` | `bo_experiment.py:~1017-1024` | `model.initial_covar_module.base_kernel.raw_lengthscale` / `.raw_outputscale` |
| `em_additive_3way` | `bo_experiment.py:1122-1128` | same two attributes on `model.initial_covar_module` |

All three fail identically under `--em-canonical hyperbo` (`HyperBODeepKernel` has no `raw_outputscale`; its `.base_kernel` is a `ScaleKernel`, which has no `raw_lengthscale`) and under `--deep-kernel 32,32` (`_DeepKernel.outputscale` is a property only; ARD dims 39 vs 7). `em_additive_3way` is worse than `warmstart_gp` because `model.initial_covar_module` is the container's canonical kernel, so it hits both attribute errors.

Concretely: OFAT cells `canon_hyperbo` and `canon_deep` (`gen_stage_queue.py:82-83`) run with `EM_METHODS` (`gen_stage_queue.py:75`), which does not include these three methods â so the plan as generated dodges it. Any stageâ2/3 cell that adds `em_additive_3way` or `warmstart_gp` to a deep/hyperbo canonical arm will abort the whole shard.

### R3 â CRITICAL: the one site that *was* fixed now fails **silently** rather than loudly.

`_safe_copy_outputscale` / `_safe_copy_lengthscale` **return `False` and do nothing** on mismatch (`bo_experiment.py:822-827`, `835-840`). At `bo_experiment.py:922-923` there is no logging, no counter, no raise. Consequences:

1. With `--em-canonical hyperbo` or `--deep-kernel`, `pretrained_gp_frozen` / `pretrained_gp_tuned` silently become a **defaultâinitialised** `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))` â zero transfer â while still being reported in the results table as the preâtrainedâGP baseline.
2. Worse: in `run_pd1_matched_loo`, `mean_s = covar_s = None` (`bo_experiment.py:~1694`) whenever **no `em*` method is requested**. Then `_safe_copy_outputscale(cc, None)` â `False`, `_safe_copy_lengthscale(cc.base_kernel, None)` â `False` (the new `if dst is None or src is None` at line 832), `base_mean_src = None` â `getattr(None,"constant",None)` is `None` â `mm = copy.deepcopy(None) = None` â `SingleTaskGP(..., mean_module=None)` falls back to a fresh `ConstantMean`. **`pretrained_gp_frozen` becomes a completely untrained GP with a frozen noise, with no message.** Before the roundâ3 fix this raised `AttributeError` and the run died.

This is the project's canonical failure mode (crash â silent wrong number) being *introduced* by a hardening fix. At minimum both helpers must `print`/`raise` when they decline, and the `covar_s is None` case must raise.

### R4 â HIGH: the D4 gaussianâwarp guard fires on the wrong condition. Two unguarded paths remain.

Guard: `bo_experiment.py:2651` â `if args.output_warp == "gaussian" and args.pd1_candidate_pool == "full":`

The actual precondition for the defect is *"the rank warp is applied to more than one pool"*. That happens in three places, only one of which is guarded:

**(a) Unguarded â `--pd1-pretrain-pool full` or `--em-canonical-pool full` with matched candidates.** `task_pools` is loaded whenever `"full" in (candidate_pool, pretrain_pool)` (`bo_experiment.py:1585`) and each perâtask pool is warped **separately** at `bo_experiment.py:1596`, while the matched grid is warped at `bo_experiment.py:1579`. With `--output-warp gaussian --pd1-candidate-pool matched --pd1-pretrain-pool full`, EM preâtrains on matchedâpoolâranked targets and HyperBO/PACOH/ABLR preâtrain on fullâpoolâranked targets **for the same tasks**. This is strictly worse than the blocked case: it differentially rescales one arm of the comparison. The guard passes it.

**(b) Unguarded â `--benchmark pd1_full`.** `run_pd1_full` warps the perâtask eval pools at `bo_experiment.py:1955` and the *matched* EM preâtraining subset at `bo_experiment.py:2072`, using two different reference sets. `pd1_candidate_pool` is never read on that path, so it keeps its `"matched"` default and the guard never fires. This is D4 verbatim, in the function round 3 brought "to parity".

**(c) False positive:** the guard also fires for `--benchmark pd1` (the nonâLOO matched path in `main()`), which never reads `pd1_candidate_pool` at all.

Does it wrongly block valid armâA configs? **No** â arm A is `--pd1-candidate-pool matched --pd1-pretrain-pool matched` (`gen_stage_queue.py:52-56`) and uses `--output-warp neglog`, so it is untouched. The guard's problem is underâcoverage, not overâcoverage.

Correct condition: `output_warp == "gaussian"` and (`task_pools` will be loaded **or** `benchmark == "pd1_full"`).

### R5 â HIGH: D1 and D11 were never applied to `bo_diagnose`.

- `bo_diagnose.py:529-540` â `build()` passes nine positional args, no `mean_baseline`. `DEFAULT_METHODS` (`bo_diagnose.py:56-59`) **includes `pretrained_gp_frozen`**, so on the regression harness the D1 contamination is live and unmitigated.
- `bo_diagnose.py:408` â `_apply_em_mean(mean_s, args, hyperbo_prior_early)`, and `hyperbo_prior_early` is nonâ`None` only under `--em-canonical hyperbo` (`bo_diagnose.py:385`). The methodâdriven HyperBO prior is trained 74 lines later at `bo_diagnose.py:482-498`. This is **exactly D11**, on the other harness, unfixed â and the error message the user sees ("Include a `hyperbo_*` methodâ¦", `bo_experiment.py:379-382`) is wrong advice here.

### R6 â HIGH: three OFAT cells Ã two regression regimes are guaranteed to abort, and permanently block `.DONE`.

`gen_stage_queue.OFAT` contains `meanxfer_full` (`--em-mean hyperbo`), `meanxfer_blend`, and `mean_linear+meanxfer` (`gen_stage_queue.py:87,88,104`). None pass `--em-canonical hyperbo`. Every OFAT cell runs on **every** regime (`run_autonomous.sh:175-177`), including `lcb_reg` and `pd1_reg` (harness `bo_diagnose`). By R5, `_apply_em_mean` raises `ValueError` at `bo_diagnose.py:408` on those six cells.

Downstream: the cell's shards never appear â `run_cell` returns 1 (`run_cell_queue.sh:121-122`) â `.incomplete` â `qrc != 0` â the `.DONE` gate at `run_autonomous.sh:143-155` fails â **the stage is reârun in full on every restart and fails again forever**. The `test_every_ofat_cell_parses_on_every_harness` test (T2) passes because argparse accepts the flag; only the runtime guard rejects it.

---

## 2. The EM model itself

### E1 â HIGH: `--cond-noise` is documented in **std** units and used as a **variance**.

- `bo_experiment.py:79-81` â `COND_NOISE = 1e-3  # frozen conditioning-likelihood noise (std units)`
- `bo_experiment.py:2378-2381` (`--cond-noise` help) â "(std units)"
- `bo_diagnose.py:196-201` â "(std, standardized units)"
- Used at `bo_experiment.py:856, 940, 998, 1064, 1099` as `lik.noise = torch.tensor(COND_NOISE)`.

GPyTorch's `GaussianLikelihood.noise` is the noise **variance** (`HomoskedasticNoise` puts it straight on the diagonal). So the actual conditioning Ï is `sqrt(1e-3) â 0.0316` in standardized units, **31Ã the documented value**. Everything the frozenâprior arms report about uncertainty is affected: `mean_sigma`, `calib_ratio`, `nll`, `coverage95` in `bo_diagnose.py:724-739`, and the LogEI exploration/exploitation balance in `_acq_score` (`bo_experiment.py:1273`). Any statement of the form "frozen EM is overconfident (Ï/RMSE âª 1)" was measured with 31Ã more conditioning noise than claimed â i.e. it is *understating* the overconfidence. Note the inconsistency with `--obs-noise`, which **is** a std (`bo_experiment.py:1412`).

### E2 â HIGH: the shrinkage blend is traceâpreserving, so its stated purpose is impossible.

`trace_matched_shrinkage` (`utils.py:690-719`) returns `(1-Î±)Â·cov + Î±Â·(tr(cov)/tr(target))Â·target`, whose trace is exactly `tr(cov)`. The motivation recorded at `bo_experiment.py:82-86` is:

> "â¦that leaves the remaining directions with only COND_NOISE of variance. Shrinkage blends in a scaled base-kernel gram to **fill** them."

It cannot fill them. Any variance placed in the rankâdeficient tail is taken, oneâforâone, out of the dominant EM directions: at Î± = 0.3 the top eigenvalues lose 30 % while the tail gains `Î±Â·(tr Î£/tr K)Â·Î»_i(K)`. And `K(Z,Z)` for a MatÃ©rnâ5/2 on a 400âpoint grid is *itself* strongly lowârankâdominant, so the tail gain is a small fraction of 30 %. Net effect at realistic Î±: mostly a **downscale of the informative directions**. This is an independent, mechanical explanation of Â§27.9's "shrinkage hurts" that has nothing to do with D2, and it means reâderiving Â§27.9 after the D2 fix will probably reproduce the same conclusion.

The MAPâEM equivalence claimed at `em_empirical_gp.py:852-858` is correct *only* for a fixed Î¨; reâmatching the trace each step makes the "prior" dataâdependent, so `Î± = (Î½+M+1)/(K+Î½+M+1)` is not a priorâstrength interpretation.

Also: `_m_step` applies shrinkage **after** the IW update (`em_empirical_gp.py:427` then `434-435`), so `--use-covar-prior` + `--em-shrinkage` doubleâregularise with no warning.

### E3 â MEDIUM: roundâ3's "`_update_cache` is not called on the frozen path" is false.

`EMEmpiricalMarginalLogLikelihood.forward` calls `self.model._update_cache()` **unconditionally** (`em_empirical_gp.py:1653`), outside the `if not self.model._using_pretrained_prior:` block at line 1645. `_update_cache` overwrites `_cached_L_kernel_inducing` and `_cached_delta_mu` (`em_empirical_gp.py:1193-1196`). Benign only because (a) the harness fits with `ExactMarginalLogLikelihood`, not this class, and (b) with `learnable_inducing_points=False` the recomputation is numerically identical. It becomes a real bug the moment `learnable_inducing_points=True` is used with this MLL. The roundâ3 doc claim should be corrected â it is currently loadâbearing for anyone reasoning about `_BlendedMean` survival.

### E4 â MEDIUM: `enable_interpolation=False` silently does nothing.

`em_empirical_gp.py:1420` â `if self.enable_interpolation or self._use_inducing_points:`. With explicitly provided inducing points, `enable_interpolation=False` is ignored. The docstring (`em_empirical_gp.py:838`, `602`) advertises it as a switch. `bo_diagnose --n-inducing K` takes this path.

### E5 â MEDIUM: `forward()` and the EM MLL disagree about which prior they use.

`forward` branches on `enable_interpolation or _use_inducing_points` (line 1420); `EMEmpiricalMarginalLogLikelihood.forward` branches on `_use_inducing_points` **alone** (`em_empirical_gp.py:1672`). With `enable_interpolation=True` and no explicit inducing points â the harness's default via `pretrain_em_prior(..., enable_interpolation=True)` â the MLL scores the datasets through `_get_prior_at_indices` (direct indexing, which reads `_effective_Sigma_inducing()` and thus `K_base(Z,Z)`) while `forward` scores queries through `_interpolate_prior_to_X` (which adds `K_base(X,X)`). Algebraically equal at `X â Z`, but numerically they go through a jittered Cholesky vs. plain indexing, so the objective being maximised is not the predictor being used.

### E6 â LOW: shared, mutated container state.

`_init_from_container` aliases (does not copy) `container.mean_module`, `container.covar_module`, `container.mu_inducing`, `container.Sigma_inducing` (`em_empirical_gp.py:997-1047`), and registers the same `Mean` object under two names (`initial_mean_module` and `mean_module`). `freeze_pretrained_parameters()` therefore mutates `requires_grad` on the **caller's** `covar_s`/`mean_s`: after the first `em_*` surrogate is built in a fold, the canonical kernel handed to `pretrained_gp_*`/`warmstart_gp` is globally frozen. Currently harmless (those branches copy `.data`), but it makes method ordering in `--methods` semantically significant.

**Verified clean in the EM model:** `psd_safe_cholesky` *is* used for `K(Z,Z)` in both the Eâstep (`em_empirical_gp.py:286`) and the interpolation cache (line 747), so the "Cholesky with no jitter" concern does not apply. `Î£(X) = Î(X) + W Î£_Z Wáµ` (line 1538â1539) is the PSDâsafe decomposition, not the cancelling one. `_compute_observation_factors` builds a new tensor rather than mutating `Sigma_SS` (line 204). `project_psd` symmetrises (`utils.py:687`). No inâplace mutation of cached tensors was found.

---

## 3. Statistical validity

### S1 â CRITICAL: the fixedâtask SE omits **betweenâprior** variance, which the analysis exists because it is nonâzero.

`analyze_pd1pool.py:326` â `se = math.sqrt(sum(var_terms)) / len(tasks)` with `var_terms = [pooled.task_wvar(d, t) â¦]` (line 321), and `task_wvar` (lines 165â190) is

```
Var = (1/PÂ²) Â· Î£_p  s_pÂ² / n_p
```

where `s_pÂ²` is the variance **across seeds within prior p**. Seeds within a fold share the *same* EM prior, the *same* HyperBO prior, the *same* preâtraining data and the *same* canonical kernel â they differ only in the initial design and the fantasy RNG. So `s_pÂ²/n_p` estimates initialâdesign variance only.

`task_wmean` averages P perâprior means. The variance of that average is

```
Var = (1/PÂ²) Î£_p s_pÂ²/n_p  +  ÏÂ²_between-prior / P
```

The second term is **entirely absent**. And the whole `PooledStudy` machinery exists because **S27.3 established that priors disagree on ordering** (`analyze_pd1pool.py:105-106, 118-121`) â i.e. ÏÂ²_between is known to be large. The estimator therefore treats the one source of variance the authors already documented as exactly zero.

Direction and magnitude: strictly **antiâconservative**. If ÏÂ²_between â ÏÂ²_within (a modest assumption given S27.3), the true variance is `(1/PÂ²)Î£ sÂ²/n + ÏÂ²_w/P â 3Ã to 4Ã` the reported value at P=3, n=3â5 â i.e. **the reported |z| is inflated by â1.7â2.0Ã**. A headline z = +3.5 is really z â 1.8â2.1, i.e. at or below the Â±2 threshold the code uses for its verdict strings (`analyze_pd1pool.py:351-353`). Every "better" verdict in the hybrid and LCBenchâhybrid sections needs reâderivation.

Fix: `task_wvar` should be the variance of the P perâprior means (`var(per_prior_means, ddof=1)/P`) plus the within term, or the whole analysis should bootstrap over priors.

### S2 â HIGH: leaveâoneâout folds are not independent, so `sqrt(Î£_t Var_t)/T` is wrong.

The formula documented at `analyze_pd1pool.py:299` (`Var(Î¸Ì) = (1/TÂ²) Î£_t s_tÂ²/n_t`) assumes `Cov(Î¸Ì_t, Î¸Ì_{t'}) = 0`. Under leaveâoneâout (`bo_experiment.py:1622` â `pre_ix = [i for i in range(T) if i != held]`), folds *t* and *t'* share **Tâ2 of Tâ1** preâtraining tasks (21 of 22 at T=23) plus the identical global seed derivation `pretrain_seed*100003 + held`.

**Does it invalidate the SE? Yes, and in the antiâconservative direction.** For paired differences `d_t = A_t â B_t`, part of the sharedâdata effect cancels *within* a fold, but not across folds: a preâtraining corpus (or prior draw) that systematically favours method A does so on all 23 folds. That induces positive equicorrelation Ï between the `d_t`, and

```
Var(mean d) = (ÏÂ²/T)Â·[1 + (Tâ1)Ï]
```

At T = 23, even Ï = 0.05 gives a variance inflation factor of 2.1 (SE Ã1.45); Ï = 0.15 gives VIF 4.3 (SE Ã2.1). **Reported |z| is too large by âVIF.** Combined multiplicatively with S1, plausible total inflation of |z| is 2.5â4Ã.

The true estimand under LOO with a shared corpus is closer to "which method wins given *this* 23âtask corpus", and the honest SE is a leaveâoneâ*cluster*âout jackknife or a movingâblock/cluster bootstrap over tasks, not `sqrt(Î£ Var)/T`. At minimum the docstring at `analyze_pd1pool.py:296-302` must state the independence assumption and that LOO violates it. Note `clustered_paired_z` (`analyze_pd1pool.py:56-75`) has the same problem but is *less* affected, because it takes the sd of the 23 perâtask means directly and therefore absorbs some of the sharedâcorpus variance into `means.std(ddof=1)`.

**Quantified direction: both S1 and S2 make results look MORE significant than they are. Neither can make a null look significant in the conservative direction.**

### S3 â HIGH: the headline table divides a weighted mean by an unweighted SEM.

`analyze_pd1pool.py:742` â
```python
rows = [(m, pooled.wmean(pooled.at(m)), pooled.clustered_sem(pooled.at(m))) for m in pooled.methods]
```
`wmean` equalises priors (`analyze_pd1pool.py:160-163`); `clustered_sem` (line 218â222) is a **plain** mean over runs. This is precisely the error the code warns about 420 lines earlier ("The SE must describe the SAME estimator as the effect, otherwise z divides a weighted numerator by an unweighted denominator", line 319â320). Same halfâfix at:

- `analyze_pd1pool.py:757` â `clustered_paired_z_values(pooled.ds_idx, pooled.at(a) - pooled.at(b))` on pooled multiâprior data, unweighted â p0 gets 45.5 %.
- `analyze_pd1pool.py:773-774` â solve rate printed with `wmean` (line 767) but tested unweighted.
- `variance_decomposition` (`analyze_pd1pool.py:386-388`) â entirely unweighted, yet it is what decides "is more compute worth it".

The roundâ2 equalâprior fix reached `wmean`/`task_wmean`/`fixed_effects_test` and stopped.

### S4 â MEDIUM: budgetâtoâtarget understates the true budget by `n_init`.

Trajectory index 0 is the state *after* the initial design (`bo_experiment.py:1421` â `traj = [best_true]`). `budget_censored` returns `np.argmax(hit, axis=1)` (`analyze_pd1pool.py:267`) and `analyze.py:367` plots `x = np.arange(study.n_pts)` labelled "evaluations". A method solving at index 0 is reported as **0 evaluations** having actually spent `n_init = 3`. Differences between methods are unaffected (shared design), but every absolute "evaluations to target" number in the paper is low by 3.

### S5 â MEDIUM: two different primary metrics share one name.

- `summarize_stage.budget_to_target(traj, pool_max, tol) -> float` (`summarize_stage.py:57-71`) â **censored mean** over all runs.
- `analyze.budget_to_target(study, method, tol) -> (median, _, solve_rate)` (`analyze.py:781`) â **median over solved runs only**, the estimator `analyze_pd1pool.solved_indicator`'s docstring explicitly warns about (lines 78â83).

`analyze_pd1pool.main()` prints the second at line 711 under the banner "BUDGET TO TARGET" while every `fixed_effects_test(..., metric="budget")` uses the first. The unit test imports only the `summarize_stage` one.

### S6 â MEDIUM: dead code resurrects a retracted claim.

`analyze_pd1pool.py:271-283` is unreachable (after `return first` at line 270) and contains a second docstring + body whose text is the claim round 2 explicitly retracted: *"so a method cannot look fast by solving only the easy runs"*. `CensoringSemanticsTest.test_docstring_names_the_estimator_and_its_bias` checks only `budget_censored.__doc__`, so the retracted sentence survives in the same file, one function away, invisible to the test that exists to keep it out.

---

## 4. Test quality â 55 tests, and these do not do what they claim

### T1 â `GuardTest.test_iw_nu_without_covar_prior_raises` is **vacuous**.
`tests/test_experiment_setup.py:125-133`. Despite the name, it never invokes the guard. It parses args, asserts `args.iw_nu is not None`, asserts `not args.use_covar_prior`, then asserts `args.iw_nu is not None and not args.use_covar_prior` â the logical conjunction of the two preceding assertions. **Delete the entire guard block at `bo_experiment.py:2659-2665` and this test still passes.** The `bo_diagnose` counterpart guard (`bo_diagnose.py:288-292`) has no test at all.

### T2 â `test_every_ofat_cell_parses_on_every_harness` validates a string that is never generated.
`tests/test_experiment_setup.py:62-86` parses `flags.split()`. But `gen_stage_queue.main()` writes `f"{f} --methods {EM_METHODS}".strip()` (`gen_stage_queue.py:187`), and `run_cell_queue.sh:96-97` further appends `$COMMON`, `--threads`, `--loo-shard` and `--out`. The test therefore never checks (a) the `--methods` suffix, (b) regime flags combined with cell flags, or (c) `--loo-shard`, which **`bo_diagnose` does not define** and which `run_cell_queue.sh:90` injects on the sole condition `SHARDS -gt 1` â with no `HARNESS` check, despite the comment two lines above saying bo_diagnose must not receive it. And, per **R6**, the test passes for six cells that are guaranteed to abort at runtime.

### T3 â `DriverExecutionTest` exercises the bookkeeping but **not the flag plumbing**.
`tests/test_experiment_setup.py:506-624`. The good half is real: `_STUB_OK`/`_STUB_FAIL` genuinely drive the shell, and the corruptâquarantine, resumability and `.incomplete` assertions are behavioural. The gaps:

- `_STUB_OK` (lines 560â567) **discards every argument except `--out`**. A driver that dropped `$COMMON` and `$flags` from the `the build tool run` line entirely would pass all six tests. Given that this project's signature failure is "a cell ran the wrong configuration" (the LCBenchâunderâPD1âfilenames incident), this is the single most important untested contract in the driver. Fix: have the stub write `"$@"` to a sidecar file and assert the cell's flags and `--loo-shard i/N` are present.
- `HARNESS` is never set in `_run` (lines 537â549), so the default `bo_experiment` is the only path covered; the `--loo-shard`âonâ`bo_diagnose` hazard is untested.
- The queue fixture (`"c1\t--x\n"`) has no `#`-comment line, though `gen_stage_queue.py:190` always writes one.
- `test_valid_shard_is_not_rerun` (lines 605â617) *is* genuine: with SHARDS=1 and a valid shard, `pids` stays empty and `_STUB_FAIL` is never invoked.

### T4 â No test covers the D1 fix at all, which is why R1 went unnoticed.
`BlendedMeanPersistenceTest` (lines 627â677) tests `deepcopy`/`state_dict`/`named_children` of `_BlendedMean`. Nothing asserts that `pretrained_gp_frozen` receives the *preâblend* mean. A oneâline test â build a surrogate with `mean_s=_BlendedMean(...)`, `mean_baseline=ConstantMean()`, assert `type(gp.mean_module) is ConstantMean` â would have failed immediately.

Similarly, `SafeHyperparamCopyTest` (lines 887â906) tests the two helpers **in isolation**. They pass while three of four call sites still crash (R2) and while the fourth fails silently (R3). This is a test that provides *false* assurance: it makes the fix look guarded when it is not wired.

### T5 â `PoolingWeightTest` asserts on its own fixture, and was left in place.
`tests/test_experiment_setup.py:278-324`. `_FakePooled.__init__` (lines 291â301) **reâimplements the production weight loop** (`analyze_pd1pool.py:144-149`) and then borrows `wmean`/`task_wmean` as unbound functions. `test_weights_sum_to_one_per_prior` (lines 312â316) asserts a property of three lines written inside the test. `RealPooledStudyTest` (lines 371â444) was added in round 3 to fix exactly this â and its own docstring says so â but the vacuous class was **not deleted**. Deleting `analyze_pd1pool.py:144-149` breaks `RealPooledStudyTest` and leaves `PoolingWeightTest` green.

### T6 â `test_task_variance_matches_the_weighted_estimator` is tautological.
`tests/test_experiment_setup.py:414-434`. The "independent" recomputation (lines 427â434) is `sum(s_kÂ²/n_k)/len(terms)Â²` â characterâforâcharacter the production formula at `analyze_pd1pool.py:183-190`. It locks the estimator in; it cannot detect that the estimator is **wrong** (S1: no betweenâprior term). A validating test would generate data with a known betweenâprior effect and assert the SE grows.

### T7 â `SummariserAlignmentTest.test_corrupt_shard_does_not_kill_the_whole_cell` contains no corrupt shard.
`tests/test_experiment_setup.py:361-368`. The body writes **one good shard** and asserts `n_runs == 1`. It is a duplicate of the happy path with a misleading name. (The real version exists separately as `SummariserRobustnessTest`, lines 480â503 â that one is correct.)

### T8 â `OutputBoundaryTest` cannot fail, and does not guard the boundary that matters.
`tests/test_experiment_setup.py:261-276`. `write_provenance` hardcodes `os.path.join(ROOT, "results/raw/v2", stage)` (`gen_stage_queue.py:140`), so asserting `"raw/v2" in path` is a restatement of that literal. The boundary that can actually be broken is `STUDY="v2/$tag"` in `run_autonomous.sh:122` â `"$R/raw/$STUDY"` in `run_cell_queue.sh:75`; `DriverExecutionTest` sets `STUDY="unit"`, so **no test asserts that shard files land under `v2/`**. Drop the `v2/` prefix in `run_autonomous.sh` and the whole suite stays green.

### T9 â `test_docstring_names_the_estimator_and_its_bias` is a lint, not a test.
`tests/test_experiment_setup.py:860-868` asserts substrings in a docstring. It is why S6 (the retracted claim surviving in dead code twelve lines below) is invisible.

**Tests that would still pass if the corresponding production code were deleted:** T1 (the `--iw-nu` guard), T5 (`PooledStudy.run_w` construction), T8 (`run_autonomous`'s `v2/` prefix), and â for the flagâplumbing half only â T3 (`$COMMON`/`$flags`/`--loo-shard` in the `the build tool run` line).

---

## 5. Things nobody has looked at

### N1 â HIGH: `--batch-mode` and `--n-fantasies` are silent noâops on **both** PD1 paths.

`run_bo`'s signature takes `batch_mode="topq"` and `n_fantasies=0` (`bo_experiment.py:1394-1396`). Only `main()` forwards them:

| caller | line | forwards `batch_mode` / `n_fantasies`? |
|---|---|---|
| `run_pd1_matched_loo` | `bo_experiment.py:1842-1846` | **no** |
| `run_pd1_full` | `bo_experiment.py:2150-2154` | **no** |
| `main` (LCBench) | `bo_experiment.py:3002-3008` | yes |

So `--batch-mode fantasy --n-fantasies 8 --benchmark pd1_loo --batch-q 4` runs plain topâq with zero fantasies and reports it as a fantasy qâbatch. This is silent noâop **#9**, in the same function family as #1â#8, and it is not covered by the `pd1_only` argv guard (`bo_experiment.py:2672-2680`) because that guard is oneâdirectional: it catches PD1 flags on LCBench but never LCBenchâonly flags on PD1. `--novel-config-split`, `--ood-split`, `--n-configs`, `--n-pretrain` and `--split-seed` are all in the same category on `pd1_loo`.

### N2 â MEDIUM: `_shard_folds` is correct, but the shard **count** can silently mix two shardings.

`_shard_folds` (`bo_experiment.py:â1539-1551`): `[f for f in range(n_folds) if f % n == i]` for `0 â¤ i < n`. Every fold satisfies `f % n == f % n` for exactly one `i`, so the shards **partition the folds exactly once â no gaps, no overlaps. Verified correct.** The `0/1` shortâcircuit and the `0 â¤ i < n` validation are both right.

The gap is one level up. `run_cell_queue.sh:112-116` refuses when `have > SHARDS` (a *shrink*), but not when `have < SHARDS` (a *grow*). Reârunning a cell that has 12 valid `_s0..s11` files with `SHARDS=24`: shards 0â11 are valid JSON so they are **skipped** (`run_cell_queue.sh:79-83`) while still containing the `f%12==i` fold sets; shards 12â23 are then run with `f%24==i`. Folds are duplicated and missing simultaneously, and `Study` (`analyze.py:161-164`) concatenates all 24 with no foldâidentity check. `analyze.py:137-139` even documents the assumption ("the folds are disjoint") without verifying it. A cheap guard: also refuse when any existing shard's `config.loo_shard` denominator â  `SHARDS`.

### N3 â MEDIUM: a NaN posterior becomes a silent deterministic pick.

`run_bo` never checks `mean`/`sigma` for finiteness before `torch.topk(score, q)` (`bo_experiment.py:~1469`) or `torch.argmax(score)` (`bo_experiment.py:~1344`). `_posterior_moments` only does `clamp_min(1e-12)` on the variance (`bo_experiment.py:1254-1257`). A failed Cholesky, a saturated `_DeepKernel` embedding, or a degenerate `Sigma_inducing` produces NaN, `argmax`/`topk` return an arbitraryâbutâdeterministic index, and the run completes with a plausibleâlooking trajectory. `_pd1_probe` has exactly this finiteness check (`bo_experiment.py:261-266`) â the production loop does not.

### N4 â LOW: latent seed collision between the acquisition and initialâdesign streams.

`g = Generator().manual_seed(1000*held + seed)` (init design) and `gm = Generator().manual_seed(7*(1000*held+seed) + 13)` (acquisition/fantasy) â `bo_experiment.py:1824, 1829`. These collide when `7k+13 = k'`, i.e. run `(h, s)`'s acquisition stream equals run `(h, s+13)`'s initâdesign stream. At `n_seeds = 3` this never fires. It fires at `n_seeds â¥ 14`.

**On the collision the question raised:** `pretrain_seed*100003 + held` is passed to `torch.manual_seed` (global RNG, `bo_experiment.py:1694`) while `1000*ei + seed` is passed to a private `torch.Generator()`. These are **different RNG objects** â no shared stream, no collision, regardless of numerical equality. Within the global stream, `100003Â·Îp = Îheld` requires `|Îheld| â¥ 100003 â« T=23`, so the perâfold global seeds are distinct across every `(pretrain_seed, held)`. **Verified clean.**

### N5 â float32/float64: verified clean.

`torch.set_default_dtype(torch.float64)` at `bo_experiment.py:66` and `bo_diagnose.py:54`; `_DeepKernel.net` â `.to(dtype=torch.double)` (line 518); `hyperbo_kernel_from_prior(...).to(dtype=torch.double)` (line 494); `hyperbo_mean_from_prior(...).to(dtype=torch.double)` (line 370); `_make_ablr_net(...).double()` (line 680); ABLR precisions `dtype=torch.double`; `_blr_neg_log_evidence` derives `dtype` from `Phi`. No mixing found. Minor: setting the global default dtype at **module import** is a moduleâscope side effect that also silently reconfigures the unitâtest process (`tests/test_experiment_setup.py:29-34` imports `bo_experiment`), which conflicts with the <repo> lazyâimport rule.

### N6 â Acquisition function and its optimization: verified correct, one caveat.

`_acq_score` (`bo_experiment.py:1261-1273`) computes `_log_ei_helper((Î¼-best)/Ï) + log Ï`, which is argmaxâidentical to BoTorch's analytic `LogExpectedImprovement`. The incumbent is the best **noisy** observed value (`pool_Y_cond`), correctly separated from the regret's `best_true` (`bo_experiment.py:1418-1420`). Because the domain is a finite pool, the acquisition is **exhaustively enumerated** â there is no inner optimizer, so none of the usual multiâstart/gradient failure modes apply, and the `argmax` is exact. `_posterior_moments` uses the **latent** posterior variance (no observation noise), which is the correct convention for analytic EI. The `logsumexp` averaging for LogEI fantasies (`bo_experiment.py:1339-1341`) correctly averages in EI space, not log space.

Caveat: because the pool is enumerated, `sigma` never comes from an observed point, so the `1e-12` clamp is the only guard against `(Î¼-best)/Ï â Â±â`; see N3.

### N7 â Duplicate `n_init` points: cannot occur by index; **unverified at the value level**.

All three paths use `torch.randperm(...)[:n_init]` (`bo_experiment.py:1825, 2131, 2981`), which yields distinct **indices**, so `remaining.remove(pick)` can never fail. However, nothing checks that the PD1 full perâtask pools contain distinct **X rows**. If a task's ~1959âconfig pool contains repeated hyperparameter vectors (repeats/seeds are common in PD1), then (a) two pool indices map to the same input, (b) `vanilla_gp` conditions on an exactly duplicated design (survives only on jitter), (c) `build_unique_inputs` (`utils.py:634`) dedups them for EM but the candidate set does not, and (d) duplicates of the optimum inflate every method's solve rate uniformly. Oneâline check worth running: `torch.unique(Xc, dim=0).shape[0] == Xc.shape[0]` for each `task_pools` entry at `bo_experiment.py:1588-1591`.

---

## Recommended order of work

1. **R1, R2, R3** â the roundâ3 fixes are not in the execution path. Nothing downstream can be trusted until they are, and R3 is actively producing a wrong baseline where it *is* wired.
2. **R6** â six cells will abort on every run; the autonomous driver will never reach `.DONE` on either regression regime.
3. **S1 + S2** â every significance verdict in `analyze_pd1pool` is inflated by an estimated 2.5â4Ã in |z|, in the falseâpositive direction. Reâderive before any claim is published.
4. **R4, N1** â two more silent noâops of the exact class this project has now hit nine times.
5. **E1** â one line, but it silently rescales every calibration number in `bo_diagnose`.
6. **T1, T3, T5, T7, T8** â delete or repair; these are the tests that let rounds 1â3 believe the work was done.


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
