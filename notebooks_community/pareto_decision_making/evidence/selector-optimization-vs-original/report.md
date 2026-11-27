# Selector before/after performance

Full selector indices matched 396/396 comparisons. Both indices and knee masks matched exactly in every case: True.

Values below are medians of 5 timed calls after 2 warmup calls. Speedup is before/after: values below one mean the candidate is slower.

| Device | dtype | (batch, n, m) | Before ms | After ms | Speedup | Exact indices |
| --- | --- | --- | ---: | ---: | ---: | --- |
| cpu | float32 | (1, 64, 2) | 2.233 | 1.359 | 1.643x | True |
| cuda | float32 | (1, 64, 2) | 10.4 | 6.831 | 1.523x | True |
| cpu | float64 | (1, 64, 2) | 1.58 | 2.125 | 0.743x | True |
| cuda | float64 | (1, 64, 2) | 15.13 | 5.091 | 2.973x | True |
| cpu | float32 | (1, 256, 3) | 14.39 | 10.11 | 1.423x | True |
| cuda | float32 | (1, 256, 3) | 14.23 | 10.36 | 1.373x | True |
| cpu | float64 | (1, 256, 3) | 9.453 | 11.88 | 0.796x | True |
| cuda | float64 | (1, 256, 3) | 9.501 | 8.774 | 1.083x | True |
| cpu | float32 | (32, 64, 3) | 59.96 | 22.57 | 2.656x | True |
| cuda | float32 | (32, 64, 3) | 195.3 | 10.13 | 19.282x | True |
| cpu | float64 | (32, 64, 3) | 79.67 | 26.57 | 2.998x | True |
| cuda | float64 | (32, 64, 3) | 155.1 | 10.92 | 14.205x | True |
| cpu | float32 | (32, 256, 3) | 358.7 | 266.5 | 1.346x | True |
| cuda | float32 | (32, 256, 3) | 89.74 | 8.169 | 10.986x | True |
| cpu | float64 | (32, 256, 3) | 216.5 | 156.6 | 1.383x | True |
| cuda | float64 | (32, 256, 3) | 82.22 | 9.251 | 8.888x | True |
| cpu | float32 | (32, 256, 5) | 232.2 | 223.9 | 1.037x | True |
| cuda | float32 | (32, 256, 5) | 89.55 | 10.9 | 8.219x | True |
| cpu | float64 | (32, 256, 5) | 235.6 | 211.1 | 1.116x | True |
| cuda | float64 | (32, 256, 5) | 117.7 | 17.7 | 6.650x | True |
| cpu | float32 | (1, 1024, 3) | 99.6 | 36.21 | 2.750x | True |
| cuda | float32 | (1, 1024, 3) | 9.82 | 7.98 | 1.231x | True |
| cpu | float64 | (1, 1024, 3) | 78.26 | 39.64 | 1.975x | True |
| cuda | float64 | (1, 1024, 3) | 8.285 | 5.688 | 1.457x | True |

The candidate was slower in 2/24 full-selection configurations. No settings were tuned after observing these measurements. All per-function timings, ranges, repetitions and CUDA allocator peaks are preserved in timings.csv/timings.json.

## Protocol and limits

- Seed 17; standard normal objective tensors; positive batched preferences; alpha=0.5; maximization. Each dtype/device pair receives identical CPU-generated input values. Source snapshots and hashes are saved beside the protocol before timing.
- Both complete public functions run their original validation and Pareto processing. The candidate reuses installed BoTorch Pareto dispatch. Shapes that trigger its loop fallback can introduce CPU/GPU synchronization and regressions; none are hidden by alternative kernels.
- Both functions use the same interpreter, Torch/BoTorch versions, hardware and 1 Torch CPU threads. Before/after order is counterbalanced deterministically across case/dtype/device/function. GPU execution is synchronized immediately before and after each measured call.
- CUDA memory is peak live allocator bytes, plus the increase above the per-call live baseline. It is not total device/process usage or reserved cache memory. CPU memory is not measured.
- 5 repetitions on one machine give descriptive ranges, not confidence intervals or a cross-hardware speed claim. Import/context initialization, input construction, copies, equality checks and serialization are outside the timing. A warmup does not eliminate every effect of GPU clocking or background activity.
- Random continuous inputs avoid deliberately introduced ties. Exact mismatches, if any, remain visible; this benchmark does not silently accept near-threshold differences.

Environment: Python 3.13.12, Torch 2.13.0+cu132, BoTorch 0.18.1; CUDA device NVIDIA GeForce RTX 4050 Laptop GPU. See provenance.json for source hashes and full environment information.

Observed conditions: Fixed GPU tiling candidate compared directly with original per-batch prototype; prior measurements preserved; other packaging/docs tasks run concurrently.
