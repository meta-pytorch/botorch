# `analysis/` — one-off analysis scripts

Scripts that produced specific findings in `EXPERIMENTAL_RESULTS.md`. They were
originally written under `/tmp` and are kept here because that has already cost this
project once: §5F–§5I of `EXPERIMENTAL_RESULTS.md` records that `/tmp/wave1.sh` was lost
on a reboot, taking the exact invocations for those sections with it. Anything whose
output is cited in a document belongs in the repository.

These are **not** part of `:lib`. Nothing in `lib` may depend on them, because `lib` is a
dependency of `bo_experiment` and `bo_diagnose`, and an edit to an analysis script must
never change the binary a running sweep is executing mid-stage. Run them with `python3`
directly.

| Script | Produced | Question it answers |
|---|---|---|
| `sig2_inversion.py` | §28.3 | Does the σ² sweep invert the EM-vs-HyperBO ordering? |
| `sig2_report.py` | §28.3 | Summary table of the σ² sweep across regimes. |
| `sig2_pertask.py` | §28.3 | Per-task breakdown — is the inversion carried by a few tasks? |
| `sig2_extra.py` | §28.3 | Robustness of the inversion to the target threshold. |
| `means_confound.py` | §28.5 | Separating the canonical-kernel factor from the mean-transfer factor. |
| `means_factors.py` | §28.5 | Factor-level decomposition of the mean/kernel OFAT arms. |
| `rblw_check.py` | §28.4 | Is the RBLW/OAS shrinkage target behaving as documented? |

Reusable analysis lives elsewhere and should not be added here:

- `stage4.py` — the four Stage 4 syntheses (§29), a real buck target.
- `analyze.py` — the standing report over `results/raw/v2`.
- `prior_stability.py` — variance decomposition across pre-training runs (§30.3).
