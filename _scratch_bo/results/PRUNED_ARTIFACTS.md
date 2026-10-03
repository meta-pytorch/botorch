# Pruned artifacts, and how to regenerate them

The research tree originally carried ~700 MB of experiment output. Most of that
is per-shard raw output and captured console logs, which cannot be reviewed in a
pull request and are reproducible from committed code plus public data. They were
pruned when this work was open-sourced.

**Everything needed to read, check, or continue the research is still here.** What
was removed is bulk, not insight.

## What was kept

| | count | size |
|---|---:|---:|
| `raw/v2/*/SUMMARY.json` — per-cell aggregates every claim is computed from | 32 | small |
| `raw/v2/*/PROVENANCE.json` — which code, config and seeds produced each cell | 32 | small |
| `raw/v2/STATUS.txt` — stage completion ledger | 1 | small |
| `figures/` — every committed figure | 203 | 9 MB |
| `MANIFEST.tsv` — the launch ledger: every command that was run, with timestamps | 1 | 7.4 MB |
| `*.sh` — every sweep runner, unmodified | 38 | small |
| all analysis code, all documents | — | — |

The `SUMMARY.json` files are the layer the analysis scripts actually consume, so
`analyze.py`, `analyze_pd1pool.py`, `stage4.py` and `summarize_stage.py` still run
against what is here.

## What was pruned

| | count | size | why |
|---|---:|---:|---|
| `logs/**` | 14,838 | 71 MB | captured stdout/stderr: build noise, progress lines, warnings. No result is derived from these. |
| `raw/**/*.json` (per-shard, excluding SUMMARY/PROVENANCE) | 14,831 | 575 MB | one file per (cell, prior, seed) shard. `SUMMARY.json` is their aggregate, and it is what every reported number comes from. |
| `prior_cache/**` | 450 | 17 MB | cached pre-trained priors. Pure cache — regenerating costs time, not information. |

## Regenerating

The shards and the cache are recomputed by re-running the sweeps. Both benchmarks
are public: LCBench loads through `botorch.utils.lcbench`, and PD1 is a separate
CC-BY-4.0 download (see `PD1_DATA.md`).

```bash
# full staged sequence; existing shards are skipped, so this resumes
STAGES=0 CELLS_PARALLEL=3 bash _scratch_bo/results/run_autonomous.sh

# a single stage
bash _scratch_bo/results/run_cell_queue.sh      # see the script for env vars
```

`MANIFEST.tsv` records the exact command line for every cell that was run, so any
individual result can be reproduced without re-running the whole campaign. Read it
as a launch ledger, not a success ledger — a row proves a command started, not that
it finished.

Two caveats carried over from `EMPIRICAL_GP_INDEX.md`, because they bite anyone
re-running this:

- **`.DONE` files are not a reliable completion test.** Use
  `cells_present == cells_expected` in each `SUMMARY.json`.
- **A sweep that suddenly gets fast has stopped computing**, usually because the
  benchmark cache is missing.

Exact numerical reproduction is not guaranteed: §32.4 records that the arm-C
control did not reproduce across stages because of a harness change, which is why
arm-C comparisons are valid only within a stage.
