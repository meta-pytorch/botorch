# Selector before/after performance

Full selector indices matched 396/396 comparisons. Both indices and knee masks matched exactly in every case: True.

Values below are medians of 5 timed calls after 2 warmup calls. Speedup is before/after: values below one mean the candidate is slower.

| Device | dtype | (batch, n, m) | Before ms | After ms | Speedup | Exact indices |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cpu | float32 | (1, 64, 2) | 1.401 | 1.139 | 1.229x | True |
| cuda | float32 | (1, 64, 2) | 1.517 | 1.584 | 0.957x | True |
| cpu | float64 | (1, 64, 2) | 0.4801 | 0.4637 | 1.035x | True |
| cuda | float64 | (1, 64, 2) | 1.873 | 1.955 | 0.958x | True |
| cpu | float32 | (1, 256, 3) | 4.003 | 3.885 | 1.030x | True |
| cuda | float32 | (1, 256, 3) | 1.575 | 1.689 | 0.933x | True |
| cpu | float64 | (1, 256, 3) | 5.235 | 3.789 | 1.381x | True |
| cuda | float64 | (1, 256, 3) | 1.611 | 1.485 | 1.085x | True |
| cpu | float32 | (32, 64, 3) | 8.432 | 8.355 | 1.009x | True |
| cuda | float32 | (32, 64, 3) | 1.506 | 1.447 | 1.041x | True |
| cpu | float64 | (32, 64, 3) | 7.848 | 7.837 | 1.001x | True |
| cuda | float64 | (32, 64, 3) | 1.445 | 1.495 | 0.967x | True |
| cpu | float32 | (32, 256, 3) | 68.33 | 66.78 | 1.023x | True |
| cuda | float32 | (32, 256, 3) | 97.36 | 2.337 | 41.664x | True |
| cpu | float64 | (32, 256, 3) | 72.51 | 77.44 | 0.936x | True |
| cuda | float64 | (32, 256, 3) | 92.27 | 2.328 | 39.629x | True |
| cpu | float32 | (32, 256, 5) | 89.81 | 89.16 | 1.007x | True |
| cuda | float32 | (32, 256, 5) | 224 | 6.561 | 34.143x | True |
| cpu | float64 | (32, 256, 5) | 94.37 | 89.67 | 1.052x | True |
| cuda | float64 | (32, 256, 5) | 145.7 | 9.803 | 14.860x | True |
| cpu | float32 | (1, 1024, 3) | 13.41 | 11.43 | 1.174x | True |
| cuda | float32 | (1, 1024, 3) | 93.1 | 2.578 | 36.106x | True |
| cpu | float64 | (1, 1024, 3) | 12.26 | 11.86 | 1.034x | True |
| cuda | float64 | (1, 1024, 3) | 146 | 5.64 | 25.882x | True |

The candidate was slower in 5/24 full-selection configurations. No settings were tuned after observing these measurements. All per-function timings, ranges, repetitions and CUDA allocator peaks are preserved in timings.csv/timings.json.

## Protocol and limits

- Seed 17; standard normal objective tensors; positive batched preferences; alpha=0.5; maximization. Each dtype/device pair receives identical CPU-generated input values. Source snapshots and hashes are saved beside the protocol before timing.
- Both complete public functions run their original validation and Pareto processing. The candidate reuses installed BoTorch Pareto dispatch. Shapes that trigger its loop fallback can introduce CPU/GPU synchronization and regressions; none are hidden by alternative kernels.
- Both functions use the same interpreter, Torch/BoTorch versions, hardware and 1 Torch CPU threads. Before/after order is counterbalanced deterministically across case/dtype/device/function. GPU execution is synchronized immediately before and after each measured call.
- CUDA memory is peak live allocator bytes, plus the increase above the per-call live baseline. It is not total device/process usage or reserved cache memory. CPU memory is not measured.
- 5 repetitions on one machine give descriptive ranges, not confidence intervals or a cross-hardware speed claim. Import/context initialization, input construction, copies, equality checks and serialization are outside the timing. A warmup does not eliminate every effect of GPU clocking or background activity.
- Random continuous inputs avoid deliberately introduced ties. Exact mismatches, if any, remain visible; this benchmark does not silently accept near-threshold differences.

Environment: Python 3.13.12, Torch 2.13.0+cu132, BoTorch 0.18.1; CUDA device NVIDIA GeForce RTX 4050 Laptop GPU. See provenance.json for source hashes and full environment information.

Observed conditions: Post-audit GPU tiling candidate; old protocol/source preserved. Rust wheel packaging and documentation preparation run concurrently; no machine isolation or universal speedup claim.
