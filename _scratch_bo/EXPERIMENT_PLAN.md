# Comprehensive EM evaluation plan — all settings × both metrics × both benchmarks

**Written:** 2026-08-17 · **Status: EXECUTED AND COMPLETE** (was "proposed, not yet
started"; corrected 2026-09-08). All of Stage 0–4 ran: 32/32 v2 stage directories,
2,079/2,079 cells, 0 failures or pending cells in the current `SUMMARY.json` files.

> ⚠️ **One planned element did not deliver what it promised.** The original Stage 3
> "replication" produced 675/675 shards **byte-identical** to Stage 1. It is
> computationally complete but is **not independent replication evidence** and must not be
> cited as such. See `EXPERIMENTAL_RESULTS.md` §29 and `EMPIRICAL_GP_INDEX.md` §4.
>
> Outcomes since this plan was written are recorded in `EXPERIMENTAL_RESULTS.md` §28–§33;
> several of this plan's factors were subsequently shown to be **inert instruments** rather
> than null results — notably the IW-ν path (§31), which was saturated at α ≈ 0.97
> regardless of the requested value.

**Start here instead:** `EMPIRICAL_GP_INDEX.md` is the router for the whole project.

**Owner doc:** this file is the plan; `EXPERIMENTAL_RESULTS.md` records outcomes;
`HANDOFF.md` §9 tracks which directory backs which claim.

Read `.llms/rules/metrics.md` first — it fixes the metric priority and the estimand
requirement, and it overrides habit.

---

> **New here?** Read `METHODS.md` first — it explains what is being computed.
> This file covers the experimental *design* and assumes that context.

## 1. Why this exists

The project's picture of the EM model is built on a **narrow slice** of its configuration
space, and recent findings suggest the slice was unrepresentative:

- **§27.16** — four `pretrain_em_prior` knobs were never exposed at all
  (`use_covar_prior`, `use_mean_prior`, `iw_nu`, `init_mode`), and the base mean was
  **always** `ConstantMean`. A one-shard screen moved posterior NLL from 0.9528 to
  **0.5893** with `--use-covar-prior` and to **0.8468** with `--em-base-mean linear`.
- **§27.15** — every `em_additive_*` variant transferred HyperBO's *covariance* only. The
  mean transfer now exists and is verified live.
- **§27.9** — the ad-hoc `--em-shrinkage` blend does not help on PD1. `use_covar_prior`
  is the principled version of the same idea and has never been run.
- **§27.14 / §27.15** — the hybrid is **SOTA on LCBench BO** (−5.54 evals vs standalone
  HyperBO) but **loses on PD1 BO** (+1.62). On-grid vs off-grid is the leading
  explanation.

So: **plain EM with a better base mean and a proper covariance prior may be a much
stronger baseline than anything measured so far**, independent of the HyperBO hybrids.
That possibility is currently untested and is the main motivation.

**Hard constraint from §27.15:** hoisting HyperBO pre-training changed the global RNG
order. **Nothing produced before 2026-08-17 is comparable with anything produced after.**
Every number in this plan must come from the current binary. This is why baselines are
re-run rather than reused.

---

## 2. Factor inventory

Everything the EM model can vary, and its status.

| # | factor | flag | levels | previously explored? |
|---|---|---|---|---|
| A | canonical / interpolation kernel | `--em-canonical`, `--deep-kernel` | `sumll`, `hyperbo`, `deep 32,32` | partly — `deep` never off-grid with a non-constant mean |
| B | **base mean** | `--em-base-mean` | `constant`, `linear` | **no — always constant** |
| C | **mean transfer** | `--em-mean`, `--em-mean-weight` | `empirical`, `blend 0.5`, `hyperbo` | **no** |
| D | covariance shrinkage (ad-hoc) | `--em-shrinkage` | `0`, `0.1`, `0.3` | yes (§27.9, unhelpful) |
| E | **covariance prior (principled)** | `--use-covar-prior`, `--iw-nu` | off, on, on+`nu` | **no** |
| F | **mean prior** | `--use-mean-prior` | off, on | **no** |
| G | **EM init** | `--em-init-mode` | `kernel`, `naive` | **no** |
| H | additive base kernel | `--methods em_additive_hyperbo*` | none, fitted, frozen | yes |
| I | EM conditioning variant | `--methods` | `em_frozen`, `em_noisefit`, `em_finetuned` | yes |
| J | EM iterations / noise | `--n-em`, `--cond-noise` | defaults | barely |

Full crossing is ~10⁴ cells. The plan therefore screens one-factor-at-a-time (OFAT)
before any factorial.

---

## 3. Benchmarks and regimes

| suite | harness | regime | tasks |
|---|---|---|---|
| LCBench BO | `bo_experiment --benchmark lcbench` | **on-grid** | 35 datasets, 200-config pool |
| LCBench regression | `bo_diagnose` | on-grid **and** off-grid (`--n-inducing`) | same |
| PD1 BO arm C | `bo_experiment --benchmark pd1_loo` (full/full) | **off-grid** | 23 tasks, ~2040 configs |
| PD1 BO arm A | same, matched/matched | on-grid | 23 tasks, 400 configs |
| **PD1 regression** | **to be built in Stage 0.2** | on-grid and off-grid | 23 tasks |

**Infrastructure gap.** `bo_diagnose.py` is LCBench-only. Answering "is regression
performance indicative of BO performance?" on *both* benchmarks requires a PD1 regression
path. That is new code, scoped in Stage 0, and is the single largest build item here.

The on-grid/off-grid axis is the plan's main scientific lever: LCBench BO is on-grid, PD1
arm C is off-grid, and §27.15 predicts the mean/interpolation factors (B, C, G) matter
**only** off-grid. Including PD1 arm A gives an on-grid PD1 control, separating
"benchmark" from "regime" — a confound no previous experiment has broken.

---

## 4. Stages

### Stage 0 — infrastructure and baselines (blocking)

0.1 **Re-run the α × target factorial on the current binary.** 6 cells
    (α ∈ {0, 0.1, 0.3} × target ∈ {sumll, hyperbo}), arm C, 1 prior. The three existing
    `sumll` cells are pre-RNG-change and must be discarded. Write to
    `raw/factorial_v2/`, leaving `raw/armc_factorial/` as retracted history.
    *Built-in controls:* `hyperbo_frozen`/`vanilla_gp`/`random` invariant to α; the two
    α=0 cells must differ.

0.2 **Build the PD1 regression harness.** Extend `bo_diagnose.py` to accept
    `--benchmark pd1_loo`, reusing the arm A/B/C pool logic. Report the same NLL / RMSE /
    calibration it reports for LCBench. **Gate:** on LCBench it must reproduce existing
    `bo_diagnose` numbers bit-identically, or the refactor changed behaviour.

0.3 **Baseline re-measurement.** `em_frozen`, `em_noisefit`, `em_finetuned`,
    `hyperbo_frozen`, `ablr`, `vanilla_gp`, `random` on all **five** regime cells
    (LCBench BO / LCBench regression / PD1 BO arm A / PD1 BO arm C / PD1 regression, the
    last gated on 0.2). Everything later is measured against these.

### Stage 1 — OFAT screen (the bulk of the compute)

From a fixed baseline (`em_frozen`, `sumll`, constant mean, no priors, α=0, `kernel`
init), vary **one factor at a time** across all levels in §2, on **every** regime cell.

- ~24 configs × 5 regime cells × **9 priors** ≈ **1080 cells**, but each is now a SINGLE
  seed rather than 3-6, so the shard count per regime is unchanged and the compute is
  comparable. §27.26: at a fixed budget B = P·n, `Var = σ²_B/P + σ²_W/B` — the second term
  depends only on B and the first only on P, so replicating over priors instead of seeds
  is free variance reduction (19–48% measured) and lifts the critical value from
  t(2)=4.30 to t(8)=2.31.
- **Prerequisite, and it is not optional:** the initial design must vary with the prior.
  It previously keyed on `1000·task + seed` only, so at `--n-seeds 1` every prior would
  have shared one initial design, the per-prior noise would collapse to a single shared
  term, and the between-prior spread would understate `Var` by σ²_W — the larger
  component. Fixed, and guarded by `ReplicationIndependenceTest`.
- Purpose: find which factors are live and which direction they move, per regime.
- **Every config must first pass the non-no-op screen** (§27.16's method: posterior NLL
  for mean-only changes, trajectory for others). A config that cannot be shown to change
  the output is dropped before it consumes a cell — this has caught 7 no-ops.

### Stage 2 — focused factorial

Cross only the factors Stage 1 shows live **and** promising (expected: B × C × E, i.e.
base mean × mean transfer × covariance prior), plus H (additive) since it interacts with
C by construction.

- ≈ 3 × 3 × 3 × 2 = 54 configs, but only on the two decisive regimes
  (LCBench BO on-grid, PD1 arm C off-grid) ≈ **108 cells**, 1 prior.

### Stage 3 — replication

Top 3 configs per regime at **≥3 pre-training priors** (§27.3: one prior cannot rank
these methods), all five regime cells. ≈ 3 × 3 × 5 = **45 cells**.

Only Stage 3 numbers may carry significance claims. Stages 1–2 rank candidates.

### Stage 4 — synthesis

4.1 **Does regression predict BO?** Correlate each config's regression NLL/RMSE against
    its BO budget-to-target, across configs, within each benchmark. This is a direct
    answer to a question the project has assumed rather than tested — §26.3's hybrid win
    was regression and was silently taken as evidence for BO.
4.2 **On-grid vs off-grid.** Compare factor effects between LCBench BO / PD1 arm A
    (on-grid) and PD1 arm C (off-grid). Tests §27.15's prediction that B/C/G matter only
    off-grid.
4.3 **Cost Pareto** across all configs (`.llms/rules/metrics.md` rank 4; no sampling
    noise, so the most durable output of the whole exercise).

---

## 5. Metrics and analysis

Per `.llms/rules/metrics.md`, in order: **budget-to-target** (0.01, 0.001, censored),
best-so-far regret, solve rate *alongside* budget-to-target, cost. **Regret-AUC is a
cross-check only.** Regression cells report NLL, RMSE, and 95% coverage.

Every claim states its **estimand** (§27.7): fixed-task for "wins on this suite",
task-clustered for "wins on a new task". Expect most task-clustered results to be "tied";
that is a property of T=23, not of the methods (§27.6).

---

## 6. Execution and parallelism

**The first version of this plan said ~9 days. That was wrong twice** — per-cell cost was
overestimated ~2.4×, and cross-cell parallelism was ignored entirely. Corrected against
measurement:

| quantity | first estimate | **measured** |
|---|---|---|
| cell wall-clock (12 shards, 3 threads, PD1 arm C, 3 seeds) | ~60 min | **~25 min** |
| shard span within a cell | — | ~12 min |
| threads used per cell | 36 | 36 |
| **cores available** | (not considered) | **96** |
| cells run concurrently | 1 | **3** |

Measured from the mtimes of the hybrid sweep and the three `sumll` factorial cells:
cell-to-cell spacing was 24.2, 24.5, 24.7, 25.0 and 20.5 min. Every runner to date used
12 shards × 3 threads = **36 of 96 cores, 37% utilisation**, with cells strictly
sequential.

`results/run_cell_queue.sh` fixes this: a generic queue that runs `CELLS_PARALLEL` cells
at once, each still 12 shards. At the default 3 × 12 × 3 = 108 threads it mildly
oversubscribes 96 cores, which is normally a net win because shards idle during buck
startup and I/O. RAM is not a constraint (235 GB total, ~120 GB free). Drop to 2 if load
exceeds ~110 or per-cell runtime regresses.

**Revised sizing at ~25 min/cell and 3 concurrent cells (~8.3 min/cell effective):**

| stage | cells | wall-clock |
|---|---|---|
| 0 | ~12 + harness build | ~2 h machine, plus dev time for 0.2 |
| 1 | ~120 | **~17 h** |
| 2 | ~108 | **~15 h** |
| 3 | ~45 | **~6 h** |

**≈ 40 hours — under two days of machine time**, down from the 9-day figure.

Two caveats on the revision, since it is an extrapolation:

- The 25 min/cell is measured on **PD1 arm C BO only**. LCBench BO cells (smaller pools,
  more seeds) and the `bo_diagnose` regression cells are unmeasured and may differ in
  either direction. Stage 0 should be timed and the estimate re-derived before Stage 1
  is sized for real.
- Oversubscription at 108 threads is an assumption, not a measurement. Stage 0 doubles as
  the calibration run: record load and per-cell time at `CELLS_PARALLEL` 2 and 3, then fix
  the value for Stages 1–3.

Further compression is available if needed but not assumed here: raising `SHARDS` from 12
to 23 gives one LOO fold per shard and roughly halves per-cell latency, at the cost of
using ~69 threads for a single cell. That trades throughput for latency, so it suits a
single urgent cell rather than a long queue.

Runs go under `systemd-run --user --slice=user.slice` with **absolute** script paths (a
relative path exits silently — §27.12), each stage a separate unit.

## 7. Recording, so a future agent can use this without context

Non-negotiable, because unanalysed sweeps are this project's signature failure (§27,
§27.13, §27.15 each found one):

1. **One directory per stage**, `results/raw/<stage>/`, never mixed.
2. **`HANDOFF.md` §9 updated in the same commit** that lands the data, with status
   (CURRENT / PARTIAL / UNANALYSED / RETRACTED) and what it may be compared against.
3. **A runner script per stage**, committed, whose header states *why* the stage exists.
4. **Analysis committed with the data**, as an `EXPERIMENTAL_RESULTS.md` section.
5. **A machine-readable summary** — `results/raw/<stage>/SUMMARY.json` with one record per
   cell: config, metrics, estimand, verdict. Stage 4 consumes these rather than re-parsing
   raw trajectories.
6. **Never claim "everything is analysed"** without re-deriving it from `ls results/raw/*/`.

---

## 8. Hazards

| hazard | mitigation |
|---|---|
| Silent no-ops (7 so far) | Non-no-op screen before every config enters the matrix |
| RNG-order incomparability (§27.15) | Single binary for the whole plan; re-run baselines; record the commit hash per stage |
| Cross-sweep comparison (§27.10) | Every sweep carries its own reference arms |
| Config drift between runs (§27.13) | Diff the recorded config against the intended one before interpreting |
| Editing source mid-run | No source edits while a stage is active |
| Underpowered "tied" (§27.6) | State the estimand; significance only from Stage 3 |
| Unanalysed data | §7 rules; Stage N+1 does not start until Stage N has a committed analysis |

---

## 9. Decisions taken (2026-08-17)

All three previously-open items are **confirmed IN SCOPE**. Recorded rather than deleted,
so a future reader can see they were chosen deliberately, and why.

1. **Build the PD1 regression harness (Stage 0.2).** Confirmed. `bo_diagnose.py` is
   LCBench-only, so without it Stage 4.1 ("does regression predict BO?") is answerable on
   one benchmark — and a one-benchmark answer to exactly that question is what produced
   the §26.3 over-generalisation in the first place. Largest build item; blocks the PD1
   regression cell.
2. **Include PD1 arm A.** Confirmed. Without it "benchmark" and "regime" are perfectly
   confounded: LCBench is on-grid, PD1 arm C is off-grid, and no experiment to date can
   separate them. Arm A is on-grid PD1, so it is the *only* way to test §27.15's
   prediction that the mean/interpolation factors matter off-grid rather than merely on
   PD1. Worth the ~20% extra cells.
3. **Include `--deep-kernel 32,32` as a factor-A level, Stage 1 only.** Confirmed. The
   handoff's "do not bother" verdict came from on-grid BO *before* the base mean could
   vary, and §26 retracted the mechanism attribution behind it anyway. A deep kernel
   changes the Nyström map `W`, so its off-grid behaviour with a linear mean is untested.
   Restricted to Stage 1; it earns a Stage 2 place only if the screen shows it live and
   promising.

**No open decisions remain. Stage 0 is cleared to start.**

---

## 10. Divergence policy and the divergence moderator (added 2026-08-18)

**Policy: no divergence filtering on either suite.** §27.24 measured the accuracy metric
at 0 NaNs and 1.80% flat curves, and **§27.25 then corrected the picture**: divergence in
LCBench is real and lives in `Train/loss` (89 curves > 10⁶, at least one `inf`), invisible
in the `val_accuracy` we optimise. A dataset-level rule would still delete 26 of 35
datasets. Beyond that arithmetic, filtering is rejected on principle for the BO cells: a
degenerate configuration is a legitimate member of the search pool; the divergence tail is
exactly where EM's calibration is documented to fail, so removing it would selectively
favour the method under test; and PD1 deliberately retains its diverged trials, so
filtering one suite would break the cross-suite comparison stage 4.1 exists for.

**A run whose loss diverged still has a finite, bad accuracy** — the objective maps the
failure onto a low score, which is the behaviour we want. Any future work on *loss* curves
must clamp or filter per-curve before modelling; see §27.27 for the multi-output study
that would use those metrics as signal rather than avoiding them.

**Stage 4.4 — the divergence moderator (new, zero extra compute).**

For each eval task compute `p_div`, the fraction of its pool that is degenerate:

* LCBench: flat curves, `max(curve) - min(curve) <= 0`, per dataset.
* PD1: `error >= 0.8`, the threshold `pd1_full_data.py` already uses for reporting.

Then regress the **per-task paired effect** `d_t` (already produced by
`fixed_effects_test`) on `p_div`:

    d_t = a + b * p_div_t + e_t

and report `b` with its CI, plus `d_t` split at the median `p_div`.

* `b ≈ 0` ⇒ the EM deficit is uniform, and the tail explanation in §23/§27 is wrong.
* `b > 0` ⇒ the deficit concentrates on degenerate-heavy tasks, which **localises the
  mechanism** and is a substantially stronger claim than any average effect.

This is pre-registered here specifically so it cannot become a post-hoc slice: the
covariate, the model, and both interpretations are fixed before the data exists. It also
partially rescues value from a design that (§27.21) cannot support significance claims on
the average — a moderator with a large effect can be detectable where a mean difference is
not, because it uses the between-task spread as signal instead of noise.

**Guardrail.** If a filter is ever introduced it must be per-curve, recorded in the cell
config, and must never silently remove a whole dataset. A test asserts the dataset count
is unchanged by loading.
