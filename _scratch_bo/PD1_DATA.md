# PD1 data: the two sources, and how to recreate the reduced one

There are **two different PD1 datasets** in this project. They are not
interchangeable, and every result changes depending on which one produced it.
This file records what each contains and — because the reduced one is
Meta-internal and will not survive open-sourcing — exactly how to rebuild it
from the public release.

Last updated 2026-09-08.

---

## 1. The two sources

| | `--pd1-source full` (**default since 2026-09-08**) | `--pd1-source lite` |
|---|---|---|
| Loader | `_scratch_bo/pd1_full_data.py` | `botorch_fb/.../pd1_data.py` (internal) |
| Origin | Public release, CC-BY 4.0, Wang et al. 2021, arXiv:2109.08215 | Reduced mirror on internal object storage, `tree/datasets/PD1Lite/<task>/{parameters,metrics}.parquet.gzip` |
| Availability | Anywhere (separate download) | **Meta-internal only** |
| Tasks | 23 evaluable | 23 (same list, see §3) |
| Matched config pool | ~400 (shared Halton grid) | ~50 |
| Objective column | `best_valid/error_rate` (best over the curve) | `valid/error_rate` |
| Sign | negated, higher-is-better | negated, higher-is-better |

### Why the default changed

`--pd1-source` defaulted to `lite` until 2026-09-08. Two reasons it no longer does:

1. **Every committed sweep script passes `--pd1-source full` explicitly** (12/12 in
   `results/*.sh`), so every result in `EXPERIMENTAL_RESULTS.md` from §17 onwards
   was produced against the public release. The default described a path no
   published number used.
2. **§12.4/§12.5 localized the failure to reproduce the paper's PD1 result to the
   pool size, not the method.** PD1Lite's ~50 shared configurations were the
   binding constraint; the full release's ~400-point shared grid is an 8x larger
   matched pool.

Outside Meta the `lite` path cannot work at all. `_import_pd1_lite()` in
`bo_experiment.py` now raises an `ImportError` that says so and points here.

---

## 2. The three differences that change the numbers

Do not assume the two loaders agree after pointing them at the same trials. They
differ in three ways that each move results:

1. **Objective column.** Lite reads `valid/error_rate`; full reads
   `best_valid/error_rate`, the best validation error over the training curve,
   which is what the paper optimizes. On a task with a non-monotone curve these
   differ substantially.
2. **Config-matching precision.** Lite normalizes X to [0,1] and rounds to
   **4 decimals** to key a configuration (`pd1_data.py`, "rounded to 4 decimals").
   Full rounds the **raw** parameter values to **10 decimals** (`_ROUND = 10` in
   `pd1_full_data.py`) and only then min-max scales, using the *declared* search
   bounds rather than observed extrema, so a subgroup and the full task set embed
   identically.
3. **Diverged trials.** Full assigns error `1.0` to the 1348 matched rows with
   `status == "diverged"` and no recorded evaluation, rather than dropping them —
   dropping would shrink the matched pool and delete exactly the divergence tail
   that makes PD1 hard (§9.3). Lite's handling is not equivalent.

Full additionally de-duplicates the 2277 `(task, config)` pairs present in both
collection phases, preferring phase 1 as the scaled-up primary run.

---

## 3. The 23 PD1Lite tasks

Identifiers are `shortname,dataset,model,variant,batch_size`. This is
`DATASET_NAMES` in `pd1_data.py`, reproduced so it survives the internal loader:

```
cifar100_wrn,cifar100,wide_resnet,wrn,2048
cifar100_wrn,cifar100,wide_resnet,wrn,256
cifar10_wrn,cifar10,wide_resnet,wrn,2048
cifar10_wrn,cifar10,wide_resnet,wrn,256
fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_relu,2048
fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_relu,256
fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_tanh,2048
fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_tanh,256
fashion_smpl_cnn,fashion_mnist,simple_cnn,simple_cnn,2048
fashion_smpl_cnn,fashion_mnist,simple_cnn,simple_cnn,256
imagenet_resnet50,imagenet,resnet,resnet50,256
imagenet_resnet50,imagenet,resnet,resnet50,512
lm1b_trfmr,lm1b,transformer,transformer,2048
mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_relu,2048
mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_relu,256
mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_tanh,2048
mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_tanh,256
mnist_simple_cnn,mnist,simple_cnn,simple_cnn,2048
mnist_simple_cnn,mnist,simple_cnn,simple_cnn,256
svhn_noextra_wrn,svhn_no_extra,wide_resnet,wrn,1024
svhn_noextra_wrn,svhn_no_extra,wide_resnet,wrn,256
uniref50_trfmr,uniref50,transformer,transformer,128
wmt15_de_en_xfmr,translate_wmt,xformer_translate,xformer,64
```

The task list is the **same 23** the full release exposes, so the two sources
differ in their *configuration pool*, not their task set.

The four search-space coordinates, in order, with the declared bounds the full
loader scales against (`PARAM_SPECS` in `pd1_full_data.py`):

| parameter | lower | upper | log |
|---|---|---|---|
| `hps.lr_hparams.initial_value` | 1e-5 | 1e1 | yes |
| `hps.lr_hparams.power` | 0.1 | 2.0 | no |
| `hps.opt_hparams.one_minus_momentum` | 1e-3 | 1e0 | yes |
| `hps.lr_hparams.decay_steps_factor` | 0.01 | 0.99 | no |

`one_minus_momentum` is derived as `1 - hps.opt_hparams.momentum`.

---

## 4. Recreating PD1Lite from the public release

The reduced pool is a *derived* artifact, so it can be rebuilt without internal object storage:

1. Obtain the public release (see `pd1_full_data.py` for the download note) and
   read the two **matched** files,
   `pd1_matched_phase{0,1}_results.jsonl.gz`.
2. Restrict to the 23 `study_group` identifiers in §3.
3. Take the objective as **`valid/error_rate`**, not `best_valid/error_rate` — this
   is the lite convention and it is the one that reproduces lite's numbers.
4. Extract the four coordinates in the §3 order, normalize each to [0,1].
5. Key each configuration by the tuple of coordinates **rounded to 4 decimals**.
6. Intersect the key sets across all 23 tasks; keep only configurations present in
   every task. That intersection is the ~50-config matched pool.

**Verify before trusting a rebuild.** The pool size is the whole point of the
lite/full distinction, so check it: a correct rebuild yields roughly 50 shared
configurations across all 23 tasks (and ~73 within the batch-size-256 subgroup,
per `pd1_full_data.py`'s docstring). If you get ~400 you have accidentally
reproduced the full grid — most likely by rounding raw rather than normalized
values, or at 10 decimals rather than 4.

**This has not been executed end to end.** The recipe is reconstructed from
`pd1_data.py`, not from a verified round-trip, so treat the config count as the
acceptance test rather than an assumption.

---

## 5. Should you bother?

Probably not, for new work. `lite` exists only so the pre-§17 results
(§9–§12) remain reproducible, and those sections are **superseded** — see
`EMPIRICAL_GP_INDEX.md` §4, retraction 2. Use `full` unless you are specifically
re-deriving a legacy number.


---

*Internal identifiers in this document (code-review diff IDs, object-storage
paths, host paths, internal tool and site names) were replaced with stable
placeholders when the research was open-sourced. Distinct originals map to
distinct placeholders, so cross-references within these documents still
resolve; they simply no longer point at anything outside this repository.*
