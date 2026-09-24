#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""How stable is a pre-trained prior across independent pre-training runs?

Every error bar this project reports mixes two sources of randomness:

  1. PRE-TRAINING -- which prior you happened to fit (the `--pretrain-seed` axis);
  2. EVALUATION   -- which init points / observation subset the BO or regression run
     happened to draw.

S27.26 switched replication onto the prior axis precisely because (1) was suspected to
dominate, but that was an inference from result spread, never a direct measurement. It
can be measured directly, because `pretrain_seed` is part of the prior-cache key: the
9-prior axis has already written 9 independent fits of the same model to disk. Nothing
had ever read them back.

This reads the cache, groups by every key field EXCEPT the seed to recover sets of
independent fits of one model, and reports how far apart those fits are in parameter
space. It then decomposes the observed result variance into a between-prior and a
within-prior component, so the project can finally say which one its error bars are made
of.

Run:  python3 prior_stability.py [--cache DIR] [--stage STAGE] [--config CONFIG]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# Reuse the PUBLISHED estimator rather than reimplementing eq. (23) here. The whitened
# variant's whole argument is that it inherits OAS's guarantees, which only holds if it
# calls the same code the spherical path does.
from _scratch_bo.em_noise_selection import oas_alpha

try:
    from _scratch_bo import prior_cache
except ImportError:  # pragma: no cover - direct script use
    import prior_cache


ROOT = os.environ.get(
    "SCRATCH_BO_ROOT",
    str(Path(__file__).resolve().parent),
)
RAW = os.path.join(ROOT, "results/raw/v2")
DEFAULT_CACHE = os.path.join(ROOT, "results/prior_cache")

# t critical values, two-sided 0.05, indexed by degrees of freedom.
_T = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    return _T.get(df, 1.96 + 2.4 / max(df, 1))


# ---------------------------------------------------------------------------------
# Parameter-space distance between two fits
# ---------------------------------------------------------------------------------
def _flatten(obj: Any, out: list[torch.Tensor]) -> None:
    """Collect every float tensor reachable from a saved prior object."""
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            out.append(obj.detach().reshape(-1).double())
        return
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            _flatten(obj[k], out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
        return
    sd = getattr(obj, "state_dict", None)
    if callable(sd):
        try:
            _flatten(sd(), out)
            return
        except Exception:
            pass
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        _flatten(d, out)


def param_vector(blob: Any) -> tuple[torch.Tensor | None, int]:
    """(flat vector of every parameter in a saved fit, count of non-finite entries).

    Concatenation order is deterministic (dict keys sorted) so two fits of the SAME
    architecture line up elementwise. Fits whose vectors differ in length are not
    comparable and are dropped by the caller rather than silently truncated.

    Non-finite entries are REPORTED, not propagated. A single NaN anywhere makes every
    norm NaN, and the first version of this script printed a column of `nan` next to the
    verdict "LARGE -- fits land in very different places", which is a conclusion drawn
    from an instrument that had failed.
    """
    parts: list[torch.Tensor] = []
    _flatten(blob, parts)
    if not parts:
        return None, 0
    v = torch.cat(parts)
    bad = int((~torch.isfinite(v)).sum())
    return v, bad


def rel_spread(vs: list[torch.Tensor]) -> tuple[float, float, int]:
    """(mean pairwise relative L2 distance, mean coefficient of variation, n_masked).

    Relative distance is ||a-b|| / mean(||a||,||b||): scale-free, so 0.9 means two
    independently fitted priors are nearly orthogonal and 0.01 means they landed in
    essentially the same place.

    Positions that are non-finite in ANY fit are masked out of ALL fits. Saved GPyTorch
    modules carry constraint buffers holding +/-inf, which are constants rather than
    fitted parameters; leaving them in makes every norm NaN. An earlier version dropped
    whole fits instead, which discarded all 47 models and printed an empty table.
    """
    stacked = torch.stack(vs)
    finite = torch.isfinite(stacked).all(dim=0)
    n_masked = int((~finite).sum())
    stacked = stacked[:, finite]
    if stacked.shape[1] == 0:
        return float("nan"), float("nan"), n_masked
    n = stacked.shape[0]
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = stacked[i], stacked[j]
            denom = 0.5 * (a.norm().item() + b.norm().item())
            if denom > 0:
                ds.append((a - b).norm().item() / denom)
    mu = stacked.mean(0)
    sd = stacked.std(0)
    scale = mu.abs().mean().item()
    cv = (sd.mean().item() / scale) if scale > 0 else float("nan")
    return (st.mean(ds) if ds else float("nan")), cv, n_masked


# ---------------------------------------------------------------------------------
# 1. Prior-space stability, straight from the cache
# ---------------------------------------------------------------------------------
def group_key(rec: dict) -> tuple:
    """Everything that identifies a MODEL, with the seed removed.

    Two entries sharing this key are independent fits of the same model on the same
    data, which is exactly the comparison `describe()` was written to enable.
    """
    return tuple(
        sorted(
            (k, str(v))
            for k, v in rec.items()
            if k not in ("path", "param.pretrain_seed")
        )
    )


def analyse_cache(cache_dir: str, min_fits: int = 3) -> None:
    print("\n" + "=" * 86)
    print("1. PRIOR-SPACE STABILITY  (independent fits of the same model)")
    print("=" * 86)
    if not os.path.isdir(cache_dir):
        print(f"    no cache at {cache_dir}")
        return
    recs = prior_cache.describe(cache_dir)
    print(f"    {len(recs)} cached fits at {cache_dir}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    usable = {k: v for k, v in groups.items() if len(v) >= min_fits}
    print(
        f"    {len(groups)} distinct models; "
        f"{len(usable)} fitted >= {min_fits} times (the measurable ones)\n"
    )
    if not usable:
        print("    NOT MEASURABLE: no model has enough independent fits.")
        return

    print(
        f"    {'kind':<10}{'iters':>8}{'fits':>6}{'params':>9}"
        f"{'rel L2':>10}{'CV':>9}   interpretation"
    )
    rows = []
    shown = 0
    for _k, v in sorted(usable.items(), key=lambda kv: -len(kv[1])):
        vecs = []
        nonfinite = 0
        for r in v:
            try:
                blob = torch.load(r["path"], map_location="cpu", weights_only=False)
            except Exception:
                continue
            pv, bad = param_vector(blob.get("obj", blob))
            if pv is None:
                continue
            if bad:
                nonfinite += 1
            vecs.append(pv)
        if len(vecs) < min_fits:
            continue
        # Architecture must match elementwise for the comparison to mean anything.
        sizes = {int(x.numel()) for x in vecs}
        if len(sizes) != 1:
            print(
                f"    {v[0].get('kind', '?'):<10}{'-':>8}{len(vecs):>6}"
                f"{'MIXED':>9}{'-':>10}{'-':>9}   sizes {sorted(sizes)}, not comparable"
            )
            continue
        d, cv, n_masked = rel_spread(vecs)
        if not math.isfinite(d):
            print(
                f"    {v[0].get('kind', '?'):<10}{'-':>8}{len(vecs):>6}"
                f"{next(iter(sizes)):>9}{'-':>10}{'-':>9}   distance not finite"
            )
            continue
        kind = str(v[0].get("kind", "?"))
        iters = v[0].get("param.num_iterations", "?")
        if d < 0.05:
            note = "near-identical -- prior draw is not a real axis"
        elif d < 0.30:
            note = "moderate spread"
        else:
            note = "LARGE -- fits land in very different places"
        # One line per distinct model would be hundreds of near-identical rows, because
        # each task shard has its own data digest. Summarise instead.
        if shown < 12:
            print(
                f"    {kind:<10}{str(iters):>8}{len(vecs):>6}"
                f"{next(iter(sizes)) - n_masked:>9}"
                f"{d:>10.3f}{cv:>9.3f}   {note}"
            )
            shown += 1
        rows.append((kind, d, cv, len(vecs)))
    if rows:
        by_kind: dict[str, list[float]] = defaultdict(list)
        for kind, d, _cv, _n in rows:
            by_kind[kind].append(d)
        print(f"\n    ---- across all {len(rows)} models with >= {min_fits} fits ----")
        for kind, ds in sorted(by_kind.items()):
            lo, hi = min(ds), max(ds)
            print(
                f"    {kind:<10} n_models={len(ds):<4} "
                f"median rel L2 = {st.median(ds):.3f}   range [{lo:.3f}, {hi:.3f}]"
            )
        print(
            "\n    Read this as: how much of the trained model is fixed by the data"
            "\n    rather than by the seed. It bounds what the prior axis CAN explain."
        )


# ---------------------------------------------------------------------------------
# 2. Variance decomposition of the results themselves
# ---------------------------------------------------------------------------------
def reduce_metric(summary_entry: dict, metric: str) -> float | None:
    """Collapse a per-iteration trace to one number.

    `summary` stores TRACES (`mean_regret` is a list over BO iterations), not scalars.
    Reading one as a scalar silently yields nothing, which is how the first version of
    this script reported "no data" for all 108 cells while the key was plainly present.

    `budget_to_<t>` follows the project's primary metric (.llms/rules/metrics.md): the
    number of evaluations until mean regret first falls to <= t, censored at the horizon
    when it never does. `final_regret` and `final_best` take the last entry.
    """
    if metric.startswith("budget_to_"):
        try:
            target = float(metric.split("budget_to_")[1])
        except (IndexError, ValueError):
            return None
        tr = summary_entry.get("mean_regret")
        if not isinstance(tr, list) or not tr:
            return None
        for i, v in enumerate(tr):
            if isinstance(v, (int, float)) and v <= target:
                return float(i + 1)
        return float(len(tr))  # censored
    key = {"final_regret": "mean_regret", "final_best": "mean_best"}.get(metric, metric)
    tr = summary_entry.get(key)
    if isinstance(tr, list) and tr:
        last = tr[-1]
        return float(last) if isinstance(last, (int, float)) else None
    if isinstance(tr, (int, float)) and not isinstance(tr, bool):
        return float(tr)
    return None


def cell_matrix(
    stage: str, config: str, method: str, metric: str
) -> dict[tuple[int, int], float]:
    """metric value keyed by (shard, prior).

    Shards are NOT repeated evaluations of the same thing: on the off-grid arms each
    shard is a different subset of tasks. Pooling across shards at fixed prior would
    therefore measure task heterogeneity and call it evaluation noise, so the two are
    kept as separate factors and the layout stays two-way.
    """
    out: dict[tuple[int, int], float] = {}
    for path in sorted(glob.glob(os.path.join(RAW, stage, f"{config}_p*_s*.json"))):
        base = os.path.basename(path)
        try:
            tail = base.split("_p")[-1]
            prior = int(tail.split("_s")[0])
            shard = int(tail.split("_s")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        try:
            with open(path) as f:
                d = json.load(f) or {}
        except (OSError, ValueError):
            continue
        entry = (d.get("summary") or {}).get(method)
        if not isinstance(entry, dict):
            continue
        v = reduce_metric(entry, metric)
        if v is not None:
            out[(shard, prior)] = v
    return out


def decompose(stage: str, config: str, method: str, metric: str) -> None:
    """Two-way decomposition of a metric into prior and task-shard components.

    With one observation per (shard, prior) cell the interaction cannot be separated
    from measurement noise, so the residual carries both. That is fine for the question
    being asked, which is whether the PRIOR or the TASK dominates the spread.
    """
    m = cell_matrix(stage, config, method, metric)
    if not m:
        print(f"    {method:<28} no data")
        return
    priors = sorted({p for _s, p in m})
    shards = sorted({s for s, _p in m})
    if len(priors) < 3:
        print(f"    {method:<28} NOT MEASURABLE (only {len(priors)} priors)")
        return
    # Balanced cells only: an unbalanced layout would confound the main effects.
    full = [s for s in shards if all((s, p) in m for p in priors)]
    if len(full) < 2:
        print(f"    {method:<28} NOT MEASURABLE (only {len(full)} complete shards)")
        return
    vals = [m[(s, p)] for s in full for p in priors]
    grand = st.mean(vals)
    a = {p: st.mean([m[(s, p)] for s in full]) - grand for p in priors}  # prior effect
    b = {s: st.mean([m[(s, p)] for p in priors]) - grand for s in full}  # shard effect
    resid = [m[(s, p)] - grand - a[p] - b[s] for s in full for p in priors]

    v_prior = st.pvariance(list(a.values()))
    v_shard = st.pvariance(list(b.values()))
    v_resid = st.pvariance(resid)
    denom = v_prior + v_resid
    frac = (v_prior / denom) if denom > 0 else float("nan")

    if not math.isfinite(frac):
        note = "degenerate"
    elif frac > 0.5:
        note = "PRIOR-dominated -- replicating priors is the right axis (S27.26)"
    elif frac < 0.15:
        note = "prior barely matters vs residual"
    else:
        note = "mixed"
    print(
        f"    {method:<28}{len(priors):>4}{len(full):>7}"
        f"{math.sqrt(v_prior):>10.3f}{math.sqrt(v_shard):>10.3f}"
        f"{math.sqrt(v_resid):>10.3f}{frac:>8.2f}   {note}"
    )


def analyse_results(stage: str, config: str, metric: str) -> None:
    print("\n" + "=" * 86)
    print(f"2. RESULT VARIANCE DECOMPOSITION  [{stage}, {config}, {metric}]")
    print("=" * 86)
    print("    two-way: which PRE-TRAINING RUN vs which TASK SHARD vs residual")
    print("    frac = prior variance / (prior + residual); task effect is reported")
    print("    separately because shards hold different tasks, not repeated draws.\n")
    stage_dir = os.path.join(RAW, stage)
    if not os.path.isdir(stage_dir):
        print(f"    no such stage: {stage_dir}")
        return
    methods: set[str] = set()
    all_cells = sorted(glob.glob(os.path.join(stage_dir, f"{config}_p*_s*.json")))
    print(f"    {len(all_cells)} cells under {stage_dir}")
    for path in all_cells[:40]:
        try:
            with open(path) as f:
                methods |= set((json.load(f) or {}).get("summary") or {})
        except (OSError, ValueError):
            continue
    if not methods:
        print(f"    no cells for config={config}")
        return
    print(
        f"    {'method':<28}{'prio':>4}{'shard':>7}"
        f"{'sd_prior':>10}{'sd_task':>10}{'sd_resid':>10}{'frac':>8}   verdict"
    )
    for mm in sorted(methods):
        decompose(stage, config, mm, metric)
    print(
        "\n    CAVEAT: --pretrain-seed also enters the BO seed"
        " (_bo_seed = 1000*ei + seed + 100003*pretrain_seed),"
        "\n    so sd_prior confounds the prior draw with the initial design. EM"
        " pre-training is\n    deterministic given the data, so em_frozen's sd_prior is"
        " a pure evaluation-noise\n    floor; the pre-training contribution of a neural"
        " baseline is the EXCESS over it."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--stage", default="stage1_pd1_bo_armC")
    ap.add_argument("--config", default="base")
    ap.add_argument("--metric", default="budget_to_0.01")
    ap.add_argument(
        "--whitened-oas",
        action="store_true",
        help="Validate the whitened-target OAS prototype: Sigma = F^.5 shrink(F^-.5 S "
        "F^-.5) F^.5, shrinking toward a pre-trained kernel instead of mu*I.",
    )
    ap.add_argument(
        "--holdout-alpha",
        action="store_true",
        help="Test the S32 data-reuse hypothesis: is whitened alpha inflated when the "
        "target F is fitted to the same tasks that form S? Compares in-sample, "
        "held-out and true targets.",
    )
    ap.add_argument("--min-fits", type=int, default=3)
    ap.add_argument(
        "--kernel-stability",
        action="store_true",
        help="Also measure whether the INDUCED KERNEL is stable across pre-training "
        "seeds, not just the weights. Answers whether a pre-trained K is a viable "
        "Inverse-Wishart scale matrix Psi (shrinkage target).",
    )
    a = ap.parse_args()
    print("=" * 86)
    print(
        "PRE-TRAINING STABILITY -- separating prior randomness from evaluation randomness"
    )
    print("=" * 86)
    if a.holdout_alpha:
        for t in (0.05, 0.20, 0.50):
            validate_holdout_alpha(tau=t)
        print()
        return
    if a.whitened_oas:
        validate_whitened_oas()
        print()
        return
    analyse_cache(a.cache, a.min_fits)
    if a.kernel_stability:
        analyse_kernel_stability(a.cache, a.min_fits)
    analyse_results(a.stage, a.config, a.metric)
    print()


# ---------------------------------------------------------------------------------
# 4. Kernel-space stability: is the INDUCED KERNEL MATRIX stable across pre-training
#    seeds, even though the weights are not?
# ---------------------------------------------------------------------------------
#
# Motivation. S30.3 measured PARAMETER-space spread and found HyperBO's fits nearly
# orthogonal (median relative L2 1.371, with sqrt(2) ~ 1.414 being orthogonal) while the
# BO behaviour they produced was stable. Weights being unstable does not imply the
# FUNCTION is unstable -- a GP kernel is massively over-parameterised, so many weight
# vectors induce nearly the same covariance.
#
# This matters for a concrete proposal: using a pre-trained GP's kernel as the
# Inverse-Wishart scale matrix Psi (the shrinkage target) instead of the current
# spherical mu*I. For Sigma ~ IW(Psi, nu) the posterior mean shrinks toward
# Psi/(nu-K-1), so Psi IS the target. A better-specified target raises the optimal
# shrinkage intensity and buys more variance reduction -- but only if the target is
# STABLE. If K varies as much as the weights do, the target injects pre-training noise
# into every fit and the idea is dead before it costs a sweep.
#
# Two numbers are reported because they answer different questions:
#   raw K   -- includes outputscale, so a fit differing only in overall amplitude counts
#              as different.
#   corr K  -- K_ij / sqrt(K_ii K_jj), the SHAPE of the kernel with amplitude divided
#              out. This is the one that matters for a shrinkage target, because nu (via
#              alpha) already controls how much total mass the prior contributes.
def _find_kernel(obj):
    """The covariance module of a saved prior, or None with a reason.

    The cache stores a ``HyperBOPriorContainer`` -- a frozen dataclass of state dicts,
    not a live module -- so rebuild through the library's own ``from_pretrained`` rather
    than reimplementing the architecture here. HyperBO is a DEEP kernel: the MLP feature
    extractor is part of the covariance, which is exactly why parameter-space distance
    overstates functional distance.
    """
    for attr in ("covar_module", "covar", "kernel"):
        k = getattr(obj, attr, None)
        if k is not None:
            return k, ""
    if hasattr(obj, "forward") and hasattr(obj, "lengthscale"):
        return obj, ""
    if hasattr(obj, "covar_module_state") and hasattr(obj, "input_dim"):
        try:
            from botorch.models.empirical_gps import HyperBOModel

            d = int(obj.input_dim)
            # from_pretrained needs conditioning data; the PRIOR covariance does not
            # depend on it, and we only ever read covar_module.
            g = torch.Generator().manual_seed(7)
            tx = torch.rand(4, d, generator=g, dtype=torch.double)
            ty = torch.zeros(4, 1, dtype=torch.double)
            m = HyperBOModel.from_pretrained(obj, tx, ty)
            return m.covar_module, ""
        except Exception as e:
            return None, f"from_pretrained: {type(e).__name__}: {e}"
    if isinstance(obj, dict):
        ks = [k for k in obj if isinstance(k, str)]
        return None, f"state_dict keys={sorted(ks)[:6]}"
    return None, f"no covar_module on {type(obj).__name__}"


def _kernel_dim(kern) -> int | None:
    for attr in ("lengthscale", "base_kernel"):
        o = getattr(kern, attr, None)
        if o is None:
            continue
        ls = o if attr == "lengthscale" else getattr(o, "lengthscale", None)
        if ls is not None and ls.numel() >= 1:
            return int(ls.shape[-1])
    return None


def analyse_kernel_stability(
    cache_dir: str, min_fits: int = 3, n_pts: int = 48
) -> None:
    print("\n" + "=" * 86)
    print("4. KERNEL-SPACE STABILITY  (induced K across pre-training seeds)")
    print("=" * 86)
    if not os.path.isdir(cache_dir):
        print(f"    no cache at {cache_dir}")
        return
    recs = prior_cache.describe(cache_dir)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    usable = {k: v for k, v in groups.items() if len(v) >= min_fits}
    print(f"    {len(usable)} models fitted >= {min_fits} times\n")
    if not usable:
        print("    NOT MEASURABLE.")
        return

    print(
        f"    {'kind':<10}{'fits':>6}{'dim':>6}{'wt relL2':>11}"
        f"{'K relL2':>10}{'corrK relL2':>13}   interpretation"
    )
    raw_all, corr_all, wt_all, skipped = [], [], [], defaultdict(int)
    for _k, v in sorted(usable.items(), key=lambda kv: -len(kv[1])):
        mats, corrs, wts, dim = [], [], [], None
        for r in v:
            try:
                blob = torch.load(r["path"], map_location="cpu", weights_only=False)
            except Exception as e:
                skipped[f"load: {type(e).__name__}"] += 1
                continue
            obj = blob.get("payload", blob.get("obj", blob))
            kern, why = _find_kernel(obj)
            if kern is None:
                skipped[why] += 1
                continue
            d = getattr(obj, "input_dim", None) or _kernel_dim(kern)
            if d is None:
                skipped["no lengthscale -> unknown input dim"] += 1
                continue
            d = int(d)
            dim = d
            # The SAME X for every fit in the group, or the comparison is meaningless.
            g = torch.Generator().manual_seed(20260822)
            X = torch.rand(n_pts, d, generator=g, dtype=torch.double)
            try:
                with torch.no_grad():
                    K = kern(X).to_dense().double()
            except Exception as e:
                skipped[f"eval: {type(e).__name__}"] += 1
                continue
            if not torch.isfinite(K).all():
                skipped["non-finite K"] += 1
                continue
            dg = K.diagonal().clamp_min(1e-12).sqrt()
            mats.append(K.reshape(-1))
            corrs.append((K / dg.outer(dg)).reshape(-1))
            pv, _bad = param_vector(obj)
            if pv is not None:
                wts.append(pv)
        if len(mats) < min_fits:
            continue
        d_raw, _, _ = rel_spread(mats)
        d_corr, _, _ = rel_spread(corrs)
        d_wt = (
            rel_spread(wts)[0]
            if len(wts) == len(mats) and len({x.numel() for x in wts}) == 1
            else float("nan")
        )
        if not math.isfinite(d_corr):
            continue
        raw_all.append(d_raw)
        corr_all.append(d_corr)
        if math.isfinite(d_wt):
            wt_all.append(d_wt)
        if d_corr < 0.05:
            note = "SHAPE STABLE -- viable shrinkage target"
        elif d_corr < 0.30:
            note = "moderate"
        else:
            note = "UNSTABLE -- target would inject pretraining noise"
        kind = str(v[0].get("kind", "?"))
        print(
            f"    {kind:<10}{len(mats):>6}{dim if dim else '?':>6}"
            f"{d_wt:>11.3f}{d_raw:>10.3f}{d_corr:>13.3f}   {note}"
        )
    if corr_all:
        print(
            f"\n    MEDIAN across {len(corr_all)} models:"
            f"  weights {st.median(wt_all) if wt_all else float('nan'):.3f}"
            f"   raw K {st.median(raw_all):.3f}"
            f"   corr K {st.median(corr_all):.3f}"
        )
        print(
            "    Read: sqrt(2) ~ 1.414 is orthogonal. If corr K is far below the weight\n"
            "    figure, the induced kernel is far steadier than the parameters that\n"
            "    encode it, and a pre-trained K is a usable Psi."
        )
    if skipped:
        # Never silently drop: a skipped model is a model we cannot speak about.
        print("\n    SKIPPED (not silently):")
        for why, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"      {n:>4}  {why}")


# ---------------------------------------------------------------------------------
# 5. PROTOTYPE: whitened-target OAS
# ---------------------------------------------------------------------------------
#
# Sigma_hat = F^(1/2) . shrink( F^(-1/2) S F^(-1/2) ) . F^(1/2)
#
# WHY WHITEN INSTEAD OF JUST SWAPPING THE TARGET. Ledoit-Wolf generalises to a structured
# target, but OAS does not: Chen et al. eq. (23) is derived for the spherical target, and
# `oas_alpha` bakes that in via `mu = S.diagonal().mean()`. Whitening sidesteps the
# re-derivation entirely -- in F-coordinates the target IS the identity, so the published
# estimator applies verbatim and we inherit its guarantees rather than inventing an
# unvalidated variant.
#
# THE ALGEBRA COLLAPSES. Expanding,
#     F^.5[(1-a) S_w + a mu_w I]F^.5  =  (1-a) S + a mu_w F
# so this is ordinary linear shrinkage toward a SCALED F. Two consequences worth stating
# because they are testable:
#   * F = I must reproduce plain OAS EXACTLY, not approximately.
#   * The result is invariant to the SCALE of F -- mu_w absorbs it. Only F's SHAPE
#     matters. That lines up with S30.6, which found the induced kernel's shape (corr K,
#     0.165) far steadier than its amplitude (raw K, 0.169, and 0.389 for the dim=7
#     model). The part of the pre-trained kernel that is unstable is precisely the part
#     this estimator ignores.
#
# Implemented by whitening the DATA ROWS (X_w = X W) rather than the covariance, so
# `oas_alpha` is reused verbatim instead of reimplemented.
def _inv_sqrt(F: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """F^(-1/2) via symmetric eigendecomposition, floored to stay finite."""
    F = 0.5 * (F + F.T)
    w, V = torch.linalg.eigh(F.double())
    w = w.clamp_min(eps * float(w.max().clamp_min(eps)))
    return (V * w.rsqrt()) @ V.T


def whitened_oas(
    X: torch.Tensor, F: torch.Tensor | None
) -> tuple[float, float, float, torch.Tensor]:
    """(alpha, mu_w, nu, Sigma_hat) shrinking toward F instead of mu*I.

    ``F=None`` is the spherical control and must agree with plain ``oas_alpha``.
    """
    X = X.double()
    T, K = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    S = (Xc.T @ Xc) / max(T, 1)
    if F is None:
        alpha, mu = oas_alpha(X)
        sigma = (1.0 - alpha) * S + alpha * mu * torch.eye(K, dtype=S.dtype)
    else:
        W = _inv_sqrt(F.double())
        # Whitening the rows gives exactly S_w = W S W, since W is symmetric.
        alpha, mu = oas_alpha(Xc @ W)
        sigma = (1.0 - alpha) * S + alpha * mu * F.double()
    alpha = min(alpha, 0.999)
    nu = (K + 1.0) + alpha * T / (1.0 - alpha)
    return float(alpha), float(mu), float(nu), sigma


def _rbf(x: torch.Tensor, ell: float) -> torch.Tensor:
    d = (x[:, None] - x[None, :]) ** 2
    return torch.exp(-0.5 * d / (ell * ell))


def validate_whitened_oas(n_trials: int = 200, K: int = 20, T: int = 15) -> None:
    print("\n" + "=" * 86)
    print("5. WHITENED-TARGET OAS  (prototype validation)")
    print("=" * 86)
    g = torch.Generator().manual_seed(20260822)
    eye = torch.eye(K, dtype=torch.double)

    # --- Structural properties. Each is a claim that could be false. ---
    x = torch.linspace(0, 1, K, dtype=torch.double)
    Sig = _rbf(x, 0.25) + 1e-6 * eye
    L = torch.linalg.cholesky(Sig)
    Z = torch.randn(T, K, generator=g, dtype=torch.double)
    Xs = Z @ L.T

    a_sph, mu_sph, _, sig_sph = whitened_oas(Xs, None)
    a_eye, _, _, sig_eye = whitened_oas(Xs, eye)
    d_alpha = abs(a_sph - a_eye)
    d_sig = float((sig_sph - sig_eye).abs().max())
    print(
        f"    [1] F=I reduces to plain OAS      d_alpha={d_alpha:.2e}  "
        f"d_Sigma={d_sig:.2e}  {'PASS' if d_alpha < 1e-9 and d_sig < 1e-9 else 'FAIL'}"
    )

    F = _rbf(x, 0.4) + 1e-6 * eye
    a1, _, _, s1 = whitened_oas(Xs, F)
    a2, _, _, s2 = whitened_oas(Xs, 137.0 * F)
    print(
        f"    [2] invariant to scale of F       d_alpha={abs(a1 - a2):.2e}  "
        f"d_Sigma={float((s1 - s2).abs().max()):.2e}  "
        f"{'PASS' if abs(a1 - a2) < 1e-9 and float((s1 - s2).abs().max()) < 1e-8 else 'FAIL'}"
    )

    A = torch.randn(K, K, generator=g, dtype=torch.double) * 0.3 + eye
    aA, _, _, _ = whitened_oas(Xs @ A.T, A @ F @ A.T)
    print(
        f"    [3] affine equivariance of alpha  d_alpha={abs(a1 - aA):.2e}  "
        f"{'PASS' if abs(a1 - aA) < 1e-6 else 'FAIL'}"
    )

    ev = float(torch.linalg.eigvalsh(s1).min())
    print(
        f"    [4] Sigma_hat stays PD            min_eig={ev:.3e}  "
        f"{'PASS' if ev > 0 else 'FAIL'}"
    )

    # --- Does it actually help? The only question that matters. ---
    #
    # First pass ran pure F and the "good" target came out 3128% WORSE than spherical.
    # That is not a weak effect, it is a diagnostic: an RBF Gram matrix on 20 points has
    # a spectrum that decays to numerical zero, so F^(-1/2) has eigenvalues ~1e3 and the
    # whitened S_w is dominated by directions where F carries no signal. mu_w is a mean
    # over those blown-up directions, so alpha*mu_w*F is enormous. Whitening presumes an
    # invertible target, and a GP kernel is the opposite of that.
    #
    # So sweep a ridge: F_tau = (1-tau)*F_hat + tau*I on a unit-mean-diagonal F_hat.
    # tau=1 IS the spherical control, so the sweep contains its own baseline and shows
    # whether ANY mixture of kernel-shape and identity beats plain OAS.
    def _unit_diag(M: torch.Tensor) -> torch.Tensor:
        return M / float(M.diagonal().mean())

    Fg_raw = _unit_diag(_rbf(x, 0.4) + 1e-6 * eye)
    R = torch.randn(K, K, generator=g, dtype=torch.double)
    Fb_raw = _unit_diag(R @ R.T / K + 1e-6 * eye)
    print(
        f"\n    cond(Sigma_true)={float(torch.linalg.cond(Sig)):.2e}   "
        f"cond(F_good)={float(torch.linalg.cond(Fg_raw)):.2e}   "
        f"cond(F_bad)={float(torch.linalg.cond(Fb_raw)):.2e}"
    )
    taus = [0.0, 0.01, 0.05, 0.2, 0.5, 0.9, 1.0]
    err = {("good F", t): [] for t in taus}
    err.update({("bad F", t): [] for t in taus})
    err[("sample", -1.0)] = []
    alphas = defaultdict(list)
    for _ in range(n_trials):
        Z = torch.randn(T, K, generator=g, dtype=torch.double)
        Xt = Z @ L.T
        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        err[("sample", -1.0)].append(float(((Xc.T @ Xc) / T - Sig).norm()))
        for name, Fr in (("good F", Fg_raw), ("bad F", Fb_raw)):
            for t in taus:
                Ft = (1.0 - t) * Fr + t * eye
                a, _, _, sg = whitened_oas(Xt, Ft)
                err[(name, t)].append(float((sg - Sig).norm()))
                alphas[(name, t)].append(a)
    base = st.mean(err[("good F", 1.0)])  # tau=1 is exactly spherical OAS
    print(f"\n    Frobenius error to true Sigma, K={K} T={T}, {n_trials} paired trials")
    print(f"    baseline = spherical OAS (tau=1) = {base:.3f}")
    print(
        f"\n    {'target':<9}{'tau':>6}{'mean err':>10}{'sem':>8}{'vs spherical':>14}{'alpha':>8}"
    )
    m = st.mean(err[("sample", -1.0)])
    print(
        f"    {'sample':<9}{'--':>6}{m:>10.3f}{st.stdev(err[('sample', -1.0)]) / math.sqrt(n_trials):>8.3f}"
        f"{100 * (m - base) / base:>13.1f}%{'--':>8}"
    )
    for name in ("good F", "bad F"):
        for t in taus:
            e = err[(name, t)]
            m = st.mean(e)
            sem = st.stdev(e) / math.sqrt(n_trials)
            print(
                f"    {name:<9}{t:>6.2f}{m:>10.3f}{sem:>8.3f}"
                f"{100 * (m - base) / base:>13.1f}%{st.mean(alphas[(name, t)]):>8.3f}"
            )
    print(
        "\n    Negative 'vs spherical' = whitened target WINS. Paired on identical draws,\n"
        "    so the contrast is the target and nothing else. tau=1 is the spherical\n"
        "    control by construction, which is why its row reads exactly 0.0%."
    )


# =============================================================================
# 6. Does an IN-SAMPLE target inflate alpha? (the S32 data-reuse hypothesis)
# =============================================================================


def validate_holdout_alpha(
    n_trials: int = 300,
    K: int = 20,
    T: int = 25,
    rank: int = 5,
    tau: float = 0.05,
) -> None:
    """Test the one explanation S32 left open, in the only design that can see it.

    S32 measured whitened alpha at 0.633 on PD1 against an empirical optimum near 0.10,
    a 5x overshoot that significantly HURT (+2.13 +/- 1.00 evals). S30.7's simulation had
    predicted a 21% WIN. The difference between the two settings is that the simulation
    drew F independently of the data, whereas HyperBO's kernel is pre-trained on the very
    tasks that form S.

    Ledoit-Wolf's alpha* = sum Var(s_ij) / (sum Var(s_ij) + ||Sigma - F||^2_F) assumes F
    is FIXED. If F was fitted to the same draw, ||S - F||^2 is shrunk by construction, the
    denominator collapses, and alpha is biased UP.

    The fitted kernel is modelled as a RANK-``rank`` TRUNCATION of the sample covariance,
    not the sample covariance itself. Using the raw S makes F == S, whitening returns the
    identity exactly, alpha saturates at 1 and Sigma-hat degenerates back to S -- a
    degenerate case that answers a question nobody asked. A truncation is smooth and has
    far fewer degrees of freedom than S, which is what makes a parametric kernel fit a
    fit rather than a copy.

    Reports two reference rows, because a Frobenius column with nothing to compare it
    against cannot support any claim.
    """
    if tau is None:
        raise ValueError("tau is required; whitening without a ridge blows up (S30.7)")
    g = torch.Generator().manual_seed(20)
    x = torch.linspace(
        0, 1, K, dtype=torch.double
    )  # _rbf wants 1-D; (K,1) goes batched
    Sig = _rbf(x, 0.25) + 1e-6 * torch.eye(K, dtype=torch.double)
    L = torch.linalg.cholesky(Sig)

    def _cov(Z: torch.Tensor) -> torch.Tensor:
        Zc = Z - Z.mean(dim=0, keepdim=True)
        return (Zc.T @ Zc) / Z.shape[0]

    def _ridge(F: torch.Tensor, t: float) -> torch.Tensor:
        """Production normalisation: unit mean diagonal, then ridge toward I.

        A rank-deficient target has near-zero eigenvalues, so F^-0.5 amplifies them
        without bound -- the S30.7 blow-up. estimate_iw_nu_whitened does this internally;
        omitting it here produced Frobenius errors of 2.5e5.
        """
        F = F / float(F.diagonal().mean().clamp_min(1e-300))
        return (1.0 - t) * F + t * torch.eye(K, dtype=torch.double)

    def _fitted_kernel(Z: torch.Tensor) -> torch.Tensor:
        """Rank-truncated sample covariance: a smooth, low-dof fit to Z."""
        w, V = torch.linalg.eigh(_cov(Z))
        w = w.clamp_min(0.0)
        w[: K - rank] = 0.0
        return (V * w) @ V.T + 1e-8 * torch.eye(K, dtype=torch.double)

    keys = ("no shrinkage", "spherical OAS", "in-sample F", "held-out F", "true F")
    rows = {k: {"a": [], "e": []} for k in keys}
    for _ in range(n_trials):
        Xa = torch.randn(T, K, generator=g, dtype=torch.double) @ L.T
        Xb = torch.randn(T, K, generator=g, dtype=torch.double) @ L.T
        S = _cov(Xa)
        rows["no shrinkage"]["a"].append(0.0)
        rows["no shrinkage"]["e"].append(float((S - Sig).norm()))
        a, _mu, _nu, sg = whitened_oas(Xa, None)  # spherical target
        rows["spherical OAS"]["a"].append(a)
        rows["spherical OAS"]["e"].append(float((sg - Sig).norm()))
        for tag, F in (
            ("in-sample F", _fitted_kernel(Xa)),  # fitted to the SAME draw that forms S
            ("held-out F", _fitted_kernel(Xb)),  # same estimator, disjoint tasks
            ("true F", Sig),  # the unreachable ideal
        ):
            a, _mu, _nu, sg = whitened_oas(Xa, _ridge(F, tau))
            rows[tag]["a"].append(a)
            rows[tag]["e"].append(float((sg - Sig).norm()))

    print(
        f"\n    Data reuse in the shrinkage target, K={K} T={T} rank={rank}, "
        f"{n_trials} trials"
    )
    print(f"    {'target':<16}{'mean alpha':>12}{'sd':>8}{'Frob err':>11}{'sem':>8}")
    for tag in keys:
        a, e = rows[tag]["a"], rows[tag]["e"]
        sd = st.stdev(a) if len(set(a)) > 1 else 0.0
        print(
            f"    {tag:<16}{st.mean(a):>12.4f}{sd:>8.4f}"
            f"{st.mean(e):>11.3f}{st.stdev(e) / math.sqrt(len(e)):>8.3f}"
        )
    ai, ah = st.mean(rows["in-sample F"]["a"]), st.mean(rows["held-out F"]["a"])
    ei, eh = st.mean(rows["in-sample F"]["e"]), st.mean(rows["held-out F"]["e"])
    print(
        f"\n    inflation from reuse: alpha {ai:.4f} -> {ah:.4f} "
        f"({100 * (ai - ah) / max(ah, 1e-9):+.1f}%), "
        f"Frobenius {ei:.3f} -> {eh:.3f} ({100 * (eh - ei) / ei:+.1f}% from holding out)"
    )


if __name__ == "__main__":
    main()
