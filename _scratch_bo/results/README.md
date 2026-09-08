# Committed results for the empirical-GP BO study

Everything in `EXPERIMENTAL_RESULTS.md` that is marked **[committed]** is regenerated
from the artifacts in this directory. An earlier iteration of this study kept its raw
outputs and driver script in `/tmp`; those were lost when `/tmp` was cleaned, which is
why results, the exact invocations, and the aggregation code now all live in-tree.

## Layout

```
results/
  run_all.sh        driver -- every sweep's exact invocation, grouped into waves
  run_pd1_full.sh   faithful PD1 reproduction on the full public release (sharded LOO)
  MANIFEST.tsv      append-only log: job name, UTC start, full command
  raw/*.json        bo_experiment / curve_experiment / bo_diagnose outputs
  raw/pd1full/      per-shard leave-one-out outputs (concatenated by analyze.py)
  logs/*.txt        stdout of each job (pre-training times, per-run progress)
  figures/          plots (.png + .pdf) and *_summary.json aggregation output
  *.md              markdown tables emitted by analyze.py, pasted into the write-up
```

**PD1 data.** `run_pd1_full.sh` needs the full PD1 release (Wang et al. 2021, CC-BY 4.0)
extracted on disk; point `--pd1-root` at the directory holding
`pd1_matched_phase*_results.jsonl.gz`. It is *not* the PD1Lite mirror that
`pd1_data.py` reads from internal object storage: PD1Lite yields ~50 matched configs across all tasks,
the full release 400, which §12.5 identified as the binding constraint on reproducing
the paper. `--pd1-source lite` keeps the old path reproducible.

**Sharding.** Leave-one-out re-pre-trains every prior per fold (~10 h sequential once
HyperBO gets the paper's 50k steps), so `--loo-shard i/n` splits folds round-robin
across processes. `analyze.py --result` accepts a comma-separated list and concatenates
shards, summing timings and recombining calibration weighted by point count.

`raw/*.json` is the authoritative artifact — it carries the full configuration, per-run
trajectories, calibration and timings, and every table and figure is derived from it.
Logs are a convenience copy of stdout and carry nothing the JSON does not; the batch-wave
logs from the first run were lost to a shell mishap and were not worth ~1 h of compute to
regenerate. (`.txt`, not `.log`, because `*.log` is gitignored repo-wide.)

## Reproducing

```bash
cd <repo>
pytorch/botorch/_scratch_bo/results/run_all.sh headline batch diagnose figures
```

Waves run their jobs concurrently, each capped at `THREADS` (default 4) intra-op torch
threads. Override the output location with `RESULTS_DIR=...`. Individual waves:

| wave | what it produces |
|---|---|
| `headline` | high-power LCBench table + the main figure (§5J) |
| `batch` | batch BO: top-q vs kriging-believer vs MC-fantasy qLogEI (§5L) |
| `diagnose` | prior-mean informativeness + calibration vs #observations (§14) |
| `figures` | aggregation, statistics and all plots from the raw JSONs |
| `scaling` | meta-data scaling, regret vs #pre-training datasets (§5F) |
| `ninit` | initial-design sweep, zero-shot to warm start (§5G) |
| `ood` | OOD dataset split (§5H) and EM-variant comparison (§5I) |
| `noise` | observation-noise robustness (§5K) |
| `pd1` | PD1 matched / full / leave-one-out with output warps (§9–§12) |
| `curve` | 1D learning-curve extrapolation, context sweep (§3, §5E) |

## What the raw JSON contains

`bo_experiment.py` writes, per sweep:

- `summary[method]` — mean best-so-far, mean regret and regret SEM per trajectory index.
- `per_run.traj_raw[method]` — the **full per-run best-so-far trajectory** (raw objective
  units). This is what makes rank/CD statistics at *any* horizon, regret-AUC, and the
  shaded-SEM figures reproducible after the fact.
- `per_run.final_regret[method]`, `per_run.pool_max_raw`, `per_run.eval_dataset_idx` —
  per-run finals plus the dataset tag needed for dataset-clustered statistics.
- `calibration[method]` — 1-step-ahead NLL and 95% coverage at acquired points.
- `timings` — one-time pre-training seconds per meta-learner and total online BO
  seconds per method, which drive the cost-normalized figure.
- `config` — the full argument namespace, so a sweep is self-describing.

## Aggregation conventions

Implemented in `analyze.py` (previously ad-hoc and uncommitted). Stated explicitly
because the exact convention changes the numbers:

- **Clustered SEM** — standard error over *per-dataset means*, not over runs, so
  correlated seeds within a dataset are not counted as independent evidence.
- **Ranks** — per run, methods ranked by the metric with **ties averaged**.
  `avg_rank` averages over runs; `avg_rank_clustered` averages within a dataset first,
  then over datasets. Tables report the clustered figure.
- **win%** — a run's winning method gets 1 credit, split evenly among ties, so the
  column sums to 100.
- **Friedman / Nemenyi** — N = number of runs, k = number of methods,
  `CD = q_alpha * sqrt(k(k+1)/(6N))` with `q_alpha` from Demsar (2006) Table 5. Two
  methods are statistically distinguishable iff their average ranks differ by > CD.
- **Regret-AUC** — mean regret over the whole trajectory; the discriminating metric once
  a finite pool is largely exhausted and final regret saturates.
- **Performance profile** — Dolan-Moré on final regret, with regret regularized by 0.1%
  of that run's initial gap so the ratio stays finite when a method reaches exactly zero.
