# Fitted-GP FPBI comparison

FPBI completed successfully on these fitted posterior means, but produced lower
median **true** hypervolume than NSGA-II on all four cases and took 9.4–12.2 times
longer than the approximately query-matched arm (ratios of per-case median
generation times). This adds a real surrogate-model check; it does not establish
a human-in-the-loop benefit.

## Shared models and comparison

There are 12 shared GP fits (four analytic cases × seeds 0, 1, 2) and 36 generation
runs. Each case/seed fits one multi-output `SingleTaskGP` on 24 training designs
for two objectives or 48 for three objectives, with Standardize and known fixed
observation variances. Independent Gaussian observation noise has SD equal to 2%
of each known analytic Pareto outcome range. The three arms reuse this same fit:
FPBI, NSGA-II with 20 generations, and NSGA-II with its generation count rounded
to FPBI's queried-vector count (population 32, maximum 300 generations).

All table cells below are **FPBI / query-matched NSGA-II**, separately taking the
median over the three seeds. HV is divided by the known continuous front's
hypervolume in normalized cost coordinates; higher is better. Predicted HV uses
posterior means; true HV audits the returned designs on analytic truth. Mean
latent SD is normalized by the known objective spans. A queried vector is one
decision design, including finite-difference/restart queries; a posterior call
can batch many vectors.

| Case | True HV | Predicted HV | Generation s | Queried vectors | Posterior calls | Mean latent SD | Returned designs |
|---|---:|---:|---:|---:|---:|---:|---:|
| convex_2d | 0.926 / 0.949 | 0.922 / 0.948 | 0.886 / 0.095 | 501 / 512 | 501 / 16 | 0.0091 / 0.0079 | 6 / 7 |
| concave_2d | 0.852 / 0.874 | 0.848 / 0.871 | 0.941 / 0.080 | 516 / 512 | 516 / 16 | 0.0119 / 0.0120 | 6 / 7 |
| constrained_2d | 0.894 / 0.923 | 0.893 / 0.922 | 0.841 / 0.089 | 456 / 448 | 456 / 14 | 0.0096 / 0.0079 | 6 / 7 |
| sphere_3d | 0.483 / 0.681 | 0.494 / 0.679 | 6.214 / 0.510 | 3587 / 3584 | 3587 / 112 | 0.0213 / 0.0137 | 12 / 12 |

The common upper output cap is seven designs in two objectives and twelve in
three. FPBI returned six versus NSGA-II's seven in two objectives; this cardinality
difference can contribute to the quality gap. Both return twelve in three
objectives. The fixed-20-generation NSGA-II arm also has higher median true HV
on all cases; its full results are in summary.csv. Query matching is rounded,
and actual budget ratios are retained, rather than claiming identical effort.
NSGA-II batches its queries; FPBI makes many separate posterior calls.

## Functional checks and uncertainty

All 36 generation runs completed. All 72 nominal FPBI rays succeeded, and all
369 individual solver/restart attempts were accepted (zero rejected attempts).
Maximum returned constraint violation was 1.11e-16. These functional successes
coexist with the negative quality/runtime comparison above.

Held-out diagnostics use 512 separately seeded Sobol designs without the fixed
training endpoint anchors. Below are per-case median GP diagnostics and totals
over three FPBI runs. Coverage is for true latent outcomes, not noisy observations.

| Case | Held-out normalized RMSE | Latent 95% coverage | FPBI rays succeeded | Accepted solver attempts |
|---|---:|---:|---:|---:|
| convex_2d | 0.0101 | 90.5% | 15/15 | 72/72 |
| concave_2d | 0.0166 | 84.4% | 15/15 | 72/72 |
| constrained_2d | 0.0097 | 90.1% | 15/15 | 72/72 |
| sphere_3d | 0.0193 | 93.0% | 27/27 | 153/153 |

The measured coverage does not justify calibrated-safety claims. FPBI's returned
designs have higher median posterior SD than the matched comparator here,
especially in three objectives; the table describes that association without
assigning it a causal explanation.

## Reproduction and validation

Run the current harness from a BoTorch source checkout containing
`optimize_with_nsgaii` and the contributed modules, with the dependencies in
protocol.json (including optional benchmark dependency pymoo 0.6.2):

```text
python -m notebooks_community.pareto_decision_making.experiments.benchmark_gp_fpbi --output-dir NEW_OUTPUT --seeds 0 1 2 --train-2d 24 --train-3d 48 --noise-sd 0.02 --fit-maxiter 75 --steps-2d 5 --steps-3d 3 --restarts 2 --maxiter 100 --population 32 --generations 20
```

BoTorch's version string alone is insufficient: one tested 0.18.1 installation
lacked the NSGA-II helper. The exact measured helper module is bundled in
source/installed-botorch-optimize.py.txt; upstream-helper-provenance.json pins its
public source commit and file/function hashes. Measured algorithm and experiment
snapshots are under source/. The current runnable GP harness later gained a fix
for a failed FPBI arm followed by a successful fixed-budget NSGA-II arm; that fix
does not affect these 36 successful recorded runs. The original measurement
protocol, source snapshots and raw results are preserved; no new timings are
attributed to the later revision.

An independent retained-record audit verified all source hashes at measurement
revision, the shared training hashes, common output caps, true outcomes and
54 independent two-dimensional hypervolume calculations. See metrics-audit.json.
Eleven focused harness tests pass, including batched GP query counters, held-out
anchor exclusion, independent predicted/true audit endpoints, retained work on
solver failure, and the later partial-failure regression. They construct tiny
real GPs without optimizer fits. Black and the upstream Flake8 configuration pass.

## Limits

This is a descriptive, four-case, three-seed CPU diagnostic on one Windows
machine with concurrent work. Generation timing excludes the shared GP fit,
held-out diagnostics, common capping and true-outcome audit. The benchmark uses
known analytic normalization ranges, fixed known noise and simple low-dimensional
models. It does not test a BO trajectory, acquisition policy, uncertain real
ratings, deployment decisions, human behavior or MPC guarantees. Cross-platform
timing and exact fitted-model reproduction are unverified.
