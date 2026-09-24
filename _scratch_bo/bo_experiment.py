#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Finite-candidate Bayesian optimization on LCBench using empirical-GP surrogates.

Setup
-----
LCBench exposes the same pool of 2000 hyperparameter configs (7D) evaluated on many
datasets. We treat each dataset as a black-box: maximize final validation accuracy over
the *finite* pool of configs. The surrogate is used ONLY to pick which pool config to
query next (via analytic LogEI); the objective is a data lookup (no fine-tuning of the
objective, just candidate generation).

Methods
-------
- em_frozen        : EMEmpiricalGaussianProcess.from_pretrained (meta-learned EM prior),
                     frozen, NO marginal-likelihood fit on incoming data.
- em_finetuned     : same, but adds a fresh additive base kernel + fits it + noise via
                     fit_gpytorch_mll each iteration.
- pretrained_gp    : SingleTaskGP reusing the shared SumMLL kernel (frozen
                     lengthscales), mean+noise fit on incoming data.
- vanilla_gp       : SingleTaskGP fit from scratch on incoming data (no transfer).
- random           : random pick from the remaining pool.

All methods share the same random initial design per (dataset, seed) for a paired
comparison. Trajectories track best-accuracy-so-far vs #evaluations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable

import torch
from _scratch_bo.em_noise_selection import estimate_iw_nu, estimate_iw_nu_whitened
from _scratch_bo.lcbench_io import (
    LCBENCH_DATASET_NAMES as DATASET_NAMES,
    load_lcbench_data,
)
from botorch.acquisition.analytic import _log_ei_helper
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps import (
    EMEmpiricalGaussianProcess,
    HyperBOModel,
    PACOHGPConfig,
    PACOHGPModel,
    pretrain_em_prior,
    pretrain_hyperbo,
    pretrain_pacoh_gp,
)
from botorch.models.empirical_gps.em_empirical_gp import build_shared_gp_model_list
from botorch.models.empirical_gps.utils import ExperimentDataset
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean, Mean
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

try:  # works both as a Buck target and as a plain script
    from _scratch_bo import prior_cache
except ImportError:  # pragma: no cover
    import prior_cache

METRIC = "Train/val_accuracy"
_HP_LOG: list = []  # (method, dataset, fresh_min_lengthscale, fresh_outputscale)
_CUR_DS: str = ""  # current eval dataset name (set by main loop)

# Module-scope defaults. Without these, importing make_surrogate from another module
# (bo_diagnose does) raised NameError instead of the intended ValueError, and
# bo_diagnose's bare `except Exception` swallowed it so the method just vanished
# from the results table (D6).
HYPERBO_BASE_KERNEL = None
EM_CANONICAL = "sumll"
DEEP_KERNEL = None
# GPyTorch's GaussianLikelihood.noise is the noise VARIANCE (HomoskedasticNoise puts it
# straight on the diagonal), so this value is a variance: 1e-3 means sigma = sqrt(1e-3)
# ~= 0.0316 in standardized units, NOT 1e-3. It was documented as std for a long time,
# which overstated the frozen arms' confidence by 31x wherever that was quoted (E1).
COND_NOISE = (
    1e-3  # frozen conditioning-likelihood noise (VARIANCE units); obs are ~noiseless
)
# M-step covariance shrinkage used when pre-training the EM prior. The EM covariance is
# a rank-<=(K-1) empirical estimate; on a shared on-grid pool (W ~ I, so the Nystrom
# residual vanishes) that leaves the remaining directions with only COND_NOISE of
# variance.
# NOTE: trace_matched_shrinkage returns (1-a)*cov + a*(tr cov / tr target)*target, whose
# trace is exactly tr(cov). It therefore CANNOT fill the tail: whatever the tail gains is
# taken one-for-one from the dominant EM directions, and K(Z,Z) is itself
# low-rank-dominant so the tail gain is a small fraction of what the top loses. At
# realistic alpha the net effect is mostly a DOWNSCALE of the informative directions --
# an independent mechanical explanation of S27.9's "shrinkage hurts" (E2). Historically 0.
EM_SHRINKAGE = 0.0

# PD1 data source: "lite" = the reduced PD1Lite internal object storage mirror (~50 matched configs),
# "full" = the full public release read from PD1_ROOT (400 matched configs, 23 tasks).
PD1_SOURCE = "lite"
PD1_ROOT = ""


def _set_cur_ds(name: str) -> None:
    global _CUR_DS
    _CUR_DS = name


def _log_hp(method: str, scale_kernel) -> None:
    """Record a fresh base kernel's min lengthscale + outputscale for diagnostics.

    Tolerates composite kernels (e.g. ``AdditiveKernel``), where there is no single
    base_kernel/outputscale pair -- the first ScaleKernel-like component is logged.
    """
    k = scale_kernel
    if not hasattr(k, "outputscale"):
        sub = [c for c in k.modules() if hasattr(c, "outputscale")]
        if not sub:
            return
        k = sub[0]
    try:
        _HP_LOG.append(
            (
                method,
                _CUR_DS,
                float(k.base_kernel.lengthscale.min().detach()),
                float(k.outputscale.detach()),
            )
        )
    except AttributeError:
        pass  # diagnostics only; never fail a run over a log line


def load_pool(names: list[str]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return normalized shared configs Xn (N,7) and per-dataset final accuracies."""
    X_all, Y_all = [], []
    for nm in names:
        data = load_lcbench_data(nm, METRIC, dtype=torch.double)
        X_all.append(data.parameters)  # (2000, 7)
        Y_all.append(data.metrics[:, -1:])  # final-epoch val accuracy (2000, 1)
    X_ref = X_all[0]
    xmin = X_ref.min(0).values
    xrng = (X_ref.max(0).values - xmin).clamp_min(1e-8)
    Xn = (X_ref - xmin) / xrng
    return Xn, Y_all


def load_pd1_pool(
    dataset_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Return shared matched configs Xn (N,d) and per-dataset objective values.

    Dispatches on ``PD1_SOURCE``: ``"lite"`` reads the reduced PD1Lite mirror from
    internal object storage (~50 matched configs across all tasks), ``"full"`` reads the full public
    PD1 release from ``PD1_ROOT`` (400 matched configs across 23 tasks). See
    ``pd1_full_data`` for why that difference matters.
    """
    if PD1_SOURCE == "full":
        from _scratch_bo.pd1_full_data import load_pd1_full_pool

        return load_pd1_full_pool(root=PD1_ROOT, dataset_names_filter=dataset_names)
    return _load_pd1_lite_pool(dataset_names)


def _import_pd1_lite():
    """The PD1Lite loader, or a clear error explaining why it is unavailable.

    PD1Lite is a reduced mirror of PD1 hosted on Meta-internal storage; it is not
    part of the public release and cannot be fetched outside Meta. Every result in
    EXPERIMENTAL_RESULTS.md from S17 onwards was produced with ``--pd1-source full``
    against the public release, which S12.5 showed is the source that matters: the
    reduced pool, not the method, was the binding constraint.
    """
    try:
        from botorch_fb.models.empirical_gaussian_processes import pd1_data
    except ImportError as e:  # pragma: no cover - depends on the checkout
        raise ImportError(
            "--pd1-source lite requires the Meta-internal PD1Lite mirror, which is "
            "not available in this checkout. Use --pd1-source full, which reads the "
            "public PD1 release (Wang et al. 2021, arXiv:2109.08215, CC-BY 4.0) and "
            "is what every committed sweep script uses."
        ) from e
    return pd1_data


def _pd1_dataset_names() -> list[str]:
    """All PD1 task identifiers for the active source."""
    if PD1_SOURCE == "full":
        from _scratch_bo.pd1_full_data import dataset_names

        return dataset_names(PD1_ROOT)
    return list(_import_pd1_lite().DATASET_NAMES)


def _load_pd1_lite_pool(
    dataset_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Return shared matched configs Xn (N,d) and per-dataset objective values.

    PD1 (Wang et al. 2021) is HyperBO's own benchmark. We use the matched-config
    subset -- the shared hyperparameter grid across tasks -- so the empirical-prior
    assumptions hold exactly as for LCBench. Restricting ``dataset_names`` to a
    homogeneous subgroup (e.g. same batch size or study) yields a MUCH larger shared
    grid than intersecting all 24 heterogeneous tasks. PD1's raw metric is
    ``valid/error_rate`` (lower is better); the harness maximizes, so we return
    ``-error_rate`` (higher is better), consistent with LCBench accuracy.
    """
    load_pd1_datasets_with_matching_inputs = (
        _import_pd1_lite().load_pd1_datasets_with_matching_inputs
    )

    datasets, matching_inputs, names_used = load_pd1_datasets_with_matching_inputs(
        dataset_names=dataset_names, normalize=True, dtype=torch.double
    )
    if matching_inputs.shape[0] == 0:
        raise RuntimeError("PD1 matched-config intersection is empty across datasets.")
    Y_all = [-ds.Y for ds in datasets]  # error_rate -> higher-is-better objective
    return matching_inputs, Y_all, names_used


def _pd1_probe(args) -> None:
    """Groundwork diagnostic: matched-pool size per grouping (#1) + EM off-grid (#3)."""
    from collections import defaultdict

    _pd1_lite = _import_pd1_lite()
    DATASET_NAMES = _pd1_lite.DATASET_NAMES
    load_pd1_datasets_with_matching_inputs = (
        _pd1_lite.load_pd1_datasets_with_matching_inputs
    )

    def matched_size(names):
        try:
            _, mi, used = load_pd1_datasets_with_matching_inputs(
                dataset_names=list(names), normalize=True, dtype=torch.double
            )
            return mi.shape[0], len(used)
        except Exception as e:  # noqa: BLE001
            return -1, str(e)[:50]

    print("=== PD1 matched-pool size by grouping ===", flush=True)
    n, t = matched_size(DATASET_NAMES)
    print(f"ALL {len(DATASET_NAMES)} tasks -> matched={n}, used={t}", flush=True)
    by_bs, by_study = defaultdict(list), defaultdict(list)
    for nm in DATASET_NAMES:
        by_bs[nm.split(",")[-1]].append(nm)
        by_study[nm.split(",")[0]].append(nm)
    print("-- by batch size --", flush=True)
    for bs, names in sorted(by_bs.items(), key=lambda kv: -len(kv[1])):
        n, t = matched_size(names)
        print(f"  bs={bs:6s} ntasks={len(names)} -> matched={n}, used={t}", flush=True)
    print("-- by study prefix (>=2 tasks) --", flush=True)
    for st, names in sorted(by_study.items(), key=lambda kv: -len(kv[1])):
        if len(names) >= 2:
            n, t = matched_size(names)
            print(
                f"  {st:22s} ntasks={len(names)} -> matched={n}, used={t}", flush=True
            )

    print("=== EM off-grid prediction test ===", flush=True)
    datasets, mi, used = load_pd1_datasets_with_matching_inputs(
        normalize=True, dtype=torch.double
    )
    Y_all = [-ds.Y for ds in datasets]
    ycat = torch.cat(Y_all)
    ymean, ystd = ycat.mean(), ycat.std().clamp_min(1e-8)
    pretrain_ds = [ExperimentDataset(X=mi, Y=(Y - ymean) / ystd) for Y in Y_all]
    mean_s, covar_s = build_shared_kernel(
        pretrain_ds, deep_hidden=DEEP_KERNEL, base_mean=args.em_base_mean
    )
    mean_s, covar_s = _pd1_canonical(pretrain_ds, args, mean_s, covar_s)
    em_prior = pretrain_em_prior(
        datasets=pretrain_ds,
        mean_module=mean_s,
        covar_module=covar_s,
        likelihood_noise=torch.tensor(args.em_likelihood_noise, dtype=torch.double),
        covariance_shrinkage=_resolve_em_shrinkage(args, pretrain_ds, covar_s),
        num_em_iterations=args.n_em,
        enable_interpolation=True,
        iw_nu=_resolve_iw_nu(args, pretrain_ds, covar_s),
        use_mean_prior=args.use_mean_prior,
        use_covar_prior=args.use_covar_prior,
        init_mode=args.em_init_mode,
    )
    Xobs, Yobs = mi[:5], (Y_all[0][:5] - ymean) / ystd
    model = make_surrogate("em_frozen", Xobs, Yobs, em_prior, mean_s, covar_s)
    on_grid = mi[10:15]
    off_grid = (mi[10:15] + 0.03 * torch.randn_like(mi[10:15])).clamp(0.0, 1.0)
    with torch.no_grad():
        pg, pog = model.posterior(on_grid), model.posterior(off_grid)

    def ok(p):
        m, v = p.mean.reshape(-1), p.variance.reshape(-1)
        return bool(
            torch.isfinite(m).all() and torch.isfinite(v).all() and (v > 0).all()
        )

    print(
        f"  on-grid  finite&posvar={ok(pg)}  mean0={pg.mean.reshape(-1)[0]:.3f}",
        flush=True,
    )
    print(
        f"  off-grid finite&posvar={ok(pog)} mean0={pog.mean.reshape(-1)[0]:.3f}",
        flush=True,
    )
    # do on-grid vs off-grid predictions DIFFER? (if identical, EM is snapping to grid)
    diff = (pg.mean.reshape(-1) - pog.mean.reshape(-1)).abs().max().item()
    print(
        f"  max|on-off mean diff|={diff:.4f}  => EM off-grid usable: {ok(pog)}",
        flush=True,
    )


def _hyperbo_cache_params(args, input_dim: int, **extra) -> dict:
    """Everything that determines a HyperBO fit, besides the data itself.

    The data digest is supplied separately by prior_cache and is the primary guard; this
    covers hyperparameters that change the FIT without changing the inputs. pretrain_seed
    is included deliberately -- it makes repeated fits separate retained entries, which is
    what allows the variance of the trained prior to be measured rather than assumed.
    """
    p = {
        "input_dim": int(input_dim),
        "hidden_dims": str(args.hyperbo_hidden),
        "use_linear_mean": not args.hyperbo_zero_mean,
        "loss_type": args.hyperbo_loss,
        "ekl_optimizer": getattr(args, "ekl_optimizer", None),
        "ekl_lbfgs_iterations": getattr(args, "ekl_lbfgs_iters", None),
        "num_iterations": args.hyperbo_iters or args.meta_iters,
        "learning_rate": args.meta_lr,
        "subsample_size": args.meta_subsample,
        "task_batch_size": args.meta_task_batch,
        "pretrain_seed": args.pretrain_seed,
    }
    p.update(extra)
    return p


def _needs_hyperbo_base_kernel(methods) -> bool:
    """Do any requested methods need HyperBO's pre-trained kernel as an additive base?

    ``em_additive_hyperbo*`` starts with "em", so the ``startswith("hyperbo")``
    gates that decide whether to pre-train a HyperBO prior do not match it.
    Without this the prior is never trained, ``HYPERBO_BASE_KERNEL`` stays
    ``None``, and the additive variants raise.
    """
    return any(m.startswith("em_additive_hyperbo") for m in methods)


def _pd1_full_pool_datasets(pre_ix, names, task_pools, args, emean, estd):
    """Each pre-training task's FULL ~2040-config pool, standardized like the caller.

    Shared by the baselines' pre-training pool and the canonical-kernel pool, which are
    separate decisions -- see the call sites in ``run_pd1_matched_loo``.
    """
    out = []
    for i in pre_ix:
        Xi, Yi = task_pools[names[i]]
        if args.per_task_standardize:
            Yi = (Yi - Yi.mean()) / Yi.std().clamp_min(1e-8)
        else:
            Yi = (Yi - emean) / estd
        out.append(ExperimentDataset(X=Xi, Y=Yi))
    return out


class _BlendedMean(Mean):
    """``(1 - beta) * m_em(x) + beta * m_hb(x)`` -- the mean analogue of the additive
    kernel door.

    EM interpolates its mean off-grid as ``mu(X) = m(X) + W @ delta_mu``, where ``W`` is
    the Nystrom map from the canonical kernel and ``m`` is this parametric base mean.
    Until now ``m`` was always EM's own ConstantMean, so the additive/canonical hybrids
    transferred HyperBO's COVARIANCE only. S27.15 argued that is the likely reason the
    hybrid wins on-grid (where the empirical mean is exact at every candidate) but falls
    short off-grid (where it must be interpolated).

    ``beta`` spans the family in one lever: 0 = EM's mean (historical behaviour),
    1 = HyperBO's learned mean outright, in between = a shrinkage blend. Replacement is
    the degenerate end of the blend rather than a separate code path.

    Note ``delta_mu = mu_Z - m(Z)`` is recomputed against whichever base mean is used, so
    the prior still reproduces the EM solution exactly at the inducing points for any
    beta; the blend only changes how the mean EXTRAPOLATES away from them.
    """

    def __init__(self, mean_em: Mean, mean_hb: Mean, beta: float) -> None:
        super().__init__()
        self.mean_em = mean_em
        self.mean_hb = mean_hb
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta >= 1.0:
            return self.mean_hb(x)
        if self.beta <= 0.0:
            return self.mean_em(x)
        return (1.0 - self.beta) * self.mean_em(x) + self.beta * self.mean_hb(x)


def hyperbo_mean_from_prior(prior) -> Mean:
    """Rebuild HyperBO's PRE-TRAINED linear mean as a plain GPyTorch Mean.

    The exact counterpart of ``hyperbo_kernel_from_prior``, and the piece that was
    missing: every ``em_additive_*`` variant to date transferred HyperBO's covariance
    while leaving the mean as EM's ConstantMean.
    """
    from botorch.models.empirical_gps.hyperbo import (
        HyperBOLinearMean,
        MLPFeatureExtractor,
    )

    if not prior.use_linear_mean or prior.mean_module_state is None:
        raise ValueError(
            "This HyperBO prior has no learned mean (use_linear_mean=False), so there "
            "is nothing to transfer. Re-train without --hyperbo-zero-mean."
        )
    fe = MLPFeatureExtractor(
        input_dim=prior.input_dim, hidden_dims=tuple(prior.hidden_dims)
    )
    fe.load_state_dict(prior.feature_extractor_state)
    mean = HyperBOLinearMean(feature_extractor=fe)
    mean.load_state_dict(prior.mean_module_state)
    for prm in mean.parameters():
        prm.requires_grad_(False)  # frozen: a pre-trained prior, not a fit
    return mean.to(dtype=torch.double)


def _apply_em_mean(mean_s, args, hyperbo_prior):
    """Apply --em-mean / --em-mean-weight to EM's parametric base mean."""
    if args.em_mean == "empirical":
        return mean_s
    if hyperbo_prior is None:
        raise ValueError(
            f"--em-mean {args.em_mean} needs a pre-trained HyperBO prior, but none was "
            "built. Include a hyperbo_* method, an em_additive_hyperbo* method, or "
            "--em-canonical hyperbo. Refusing to fall back to the empirical mean."
        )
    beta = 1.0 if args.em_mean == "hyperbo" else args.em_mean_weight
    print(
        f"  EM base mean: blend beta={beta:g} toward HyperBO's learned mean", flush=True
    )
    return _BlendedMean(mean_s, hyperbo_mean_from_prior(hyperbo_prior), beta)


def _resolve_em_shrinkage(args, datasets, covar_module=None):
    """--em-shrinkage, or an analytic estimate of it when --em-shrinkage-mode says so.

    THIS IS THE KNOB THE IW ROUTE COULD NOT REACH. S31 measured the Inverse-Wishart path
    as inert: Psi = (nu+N+1)*Sigma_init makes the realised intensity
    alpha = (nu+N+1)/(K+nu+N+1), and since nu is validated to exceed N_inducing-1 that is
    floored at 0.970 -- a 9x range of intended alpha collapsed to a 0.0012 range of
    realised alpha, and trajectories came back bit-identical. ``covariance_shrinkage``
    takes alpha directly, in [0, 1], decoupled from N_inducing.

    So estimate alpha and USE it, rather than inverting it to a nu that cannot express it.
    ``trace_matched_shrinkage`` rescales the target to match tr(Sigma) anyway, which is
    the same scale-invariance the whitened estimator has (S30.7) -- the two agree that
    only the target's correlation structure should be imposed.
    """
    mode = getattr(args, "em_shrinkage_mode", "manual")
    if mode == "manual":
        return args.em_shrinkage
    rows = [d.Y.reshape(-1) for d in datasets]
    if len(rows) < 3:
        return args.em_shrinkage
    X = torch.stack(rows).double()
    if mode == "oas_whitened":
        if covar_module is None:
            print(
                "    WARNING: --em-shrinkage-mode oas_whitened got no covar_module; "
                "falling back to spherical oas. This cell is NOT a whitened result.",
                flush=True,
            )
            _nu, alpha = estimate_iw_nu(X, method="oas")
        else:
            Xg = datasets[0].X
            F = covar_module(Xg, Xg).to_dense()
            _nu, alpha = estimate_iw_nu_whitened(X, F, ridge=args.iw_target_ridge)
    else:
        _nu, alpha = estimate_iw_nu(X, method=mode)
    alpha = float(min(max(alpha, 0.0), 1.0))
    args.resolved_em_shrinkage = alpha
    print(
        f"[em-shrinkage-mode={mode}] alpha={alpha:.4f} "
        f"from {X.shape[0]} tasks x {X.shape[1]} points",
        flush=True,
    )
    return alpha


def _resolve_iw_nu(args, datasets, covar_module=None):
    """--iw-nu, or an empirical-Bayes estimate of it when --iw-nu-mode says so.

    Kept out of the call sites so all three stay in sync; S27.38 is a standing reminder
    of what happens when a fix lands in one harness and not the others.
    """
    mode = getattr(args, "iw_nu_mode", "manual")
    if mode == "manual" or not args.use_covar_prior:
        return args.iw_nu
    rows = [d.Y.reshape(-1) for d in datasets]
    if len(rows) < 3:
        return args.iw_nu
    X = torch.stack(rows).double()
    if mode == "oas_whitened":
        # Estimate alpha against the target the EM ACTUALLY uses. With the default
        # init_mode="kernel", Psi = (nu+N+1)*K(Z,Z), so the spherical oas_alpha was
        # calibrating for a target that is not the one in force.
        if covar_module is None:
            # Loud, because falling back silently would report a "whitened" number that
            # is really the spherical one -- indistinguishable from a genuine null.
            print(
                "    WARNING: --iw-nu-mode oas_whitened got no covar_module; "
                "falling back to spherical oas. This cell is NOT a whitened result.",
                flush=True,
            )
            nu, alpha = estimate_iw_nu(X, method="oas")
        else:
            Xg = datasets[0].X
            F = covar_module(Xg, Xg).to_dense()
            nu, alpha = estimate_iw_nu_whitened(X, F, ridge=args.iw_target_ridge)
    else:
        nu, alpha = estimate_iw_nu(X, method=mode)
    args.resolved_iw_nu = float(nu)
    args.resolved_iw_alpha = float(alpha)
    print(
        f"[iw-nu-mode={mode}] estimated nu={nu:.2f} (alpha={alpha:.3f}) "
        f"from {X.shape[0]} tasks x {X.shape[1]} points",
        flush=True,
    )
    return nu


def _pd1_canonical(pretrain_ds, args, mean_s, covar_s, hyperbo_prior=None):
    """Optionally swap in HyperBO's pre-trained kernel as the PD1 canonical kernel.

    PD1 is the interesting case for this hybrid: its tasks carry ~2040 configs of which
    only ~400 are shared, so with the full pool the evaluation points are genuinely
    off-grid and the Nystrom residual Lambda is non-zero. That is the only regime where
    the canonical kernel can influence the EM posterior at all.

    ``hyperbo_prior`` is the caller's already-trained prior. Reusing it is REQUIRED for a
    clean comparison: training a second one here would make the canonical kernel and
    ``hyperbo_frozen`` differ by random draw as well as by role, and would pay for the
    most expensive component twice (S27.15). Training locally is kept only as a fallback
    for callers that have no prior of their own.
    """
    if EM_CANONICAL != "hyperbo":
        return mean_s, covar_s
    if hyperbo_prior is not None:
        print("  PD1: canonical kernel from the SHARED pre-trained HyperBO", flush=True)
        return mean_s, hyperbo_kernel_from_prior(hyperbo_prior)
    print("  PD1: seeding EM canonical kernel from a pre-trained HyperBO", flush=True)
    prior = pretrain_hyperbo(
        datasets=pretrain_ds,
        input_dim=pretrain_ds[0].X.shape[-1],
        hidden_dims=args.hyperbo_hidden,
        use_linear_mean=not args.hyperbo_zero_mean,
        loss_type=args.hyperbo_loss,
        num_iterations=args.hyperbo_iters or args.meta_iters,
        learning_rate=args.meta_lr,
        # REQUIRED, not optional. Without these this fits an exact-GP MLL over the FULL
        # per-task pools (~2040 configs x 22 tasks), which is cubic per task per step at
        # 25k steps -- not slow, intractable. It hung 12 shards for 62 h before being
        # caught. The main gate's pretrain_hyperbo call always passed them; this one
        # never did, and was simply never exercised because no PD1 run had ever set
        # --em-canonical hyperbo.
        subsample_size=args.meta_subsample,
        task_batch_size=args.meta_task_batch,
    )
    return mean_s, hyperbo_kernel_from_prior(prior)


def _code_provenance() -> dict:
    """Commit + dirty flag as of THIS PROCESS, cached for the run.

    PROVENANCE.json records the commit at QUEUE-GENERATION time, but cells run hours to
    days after the queue is written and the binary can be rebuilt in between (S27.15). So
    the stage-level commit is not a claim about what produced a given shard; this is.

    Failures are recorded as a value, never raised and never omitted: a missing key would
    be indistinguishable from an old unstamped shard, whereas "sl-failed" says we tried.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        return _PROVENANCE_CACHE
    import subprocess as _sp

    here = os.path.dirname(os.path.abspath(__file__))
    commit, dirty = "unknown", None
    try:
        commit = (
            _sp.run(
                ["sl", "whereami"], cwd=here, capture_output=True, text=True, timeout=30
            ).stdout.strip()
            or "unknown"
        )
        st = _sp.run(
            ["sl", "status", "-mard"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=30,
        )
        dirty = bool(st.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        commit, dirty = f"sl-failed: {type(exc).__name__}", None
    _PROVENANCE_CACHE = {
        "code_commit": commit,
        "code_dirty": dirty,
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return _PROVENANCE_CACHE


_PROVENANCE_CACHE: dict | None = None


def atomic_write_json(path: str, payload, default=None) -> None:
    """Write JSON so a crash can never leave a truncated file behind.

    ``open(path, "w")`` truncates immediately and then streams hundreds of KB, so the
    window in which the file exists but is invalid is the whole serialisation, not a
    narrow race. A shard left in that state was skipped forever by the runner's
    ``[ -f ]`` check and turned its entire cell into an error (review B3/N17).

    ``default`` is None (strict) to preserve the harnesses' original semantics: a value
    json cannot serialise must RAISE, not be silently stringified into a result file.
    Callers whose payloads legitimately contain non-JSON scalars pass ``default=str``.

    Stamps provenance HERE rather than at the four call sites, so a future writer cannot
    forget it -- the failure mode this project keeps hitting is the call site that was
    missed, not the helper that was wrong. Keyed on the presence of ``config`` because
    every result payload carries ``"config": vars(args)`` and nothing else does; that
    ties the behaviour to the data's shape rather than to caller discipline, and leaves
    this helper's contract unchanged for the non-result JSON it also writes.
    """
    import os as _os
    import tempfile as _tempfile

    if (
        isinstance(payload, dict)
        and "config" in payload
        and "provenance" not in payload
    ):
        payload = {**payload, "provenance": _code_provenance()}

    d = _os.path.dirname(_os.path.abspath(path)) or "."
    _os.makedirs(d, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(dir=d, suffix=".partial")
    try:
        with _os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=default)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, path)  # atomic within a filesystem
    except BaseException:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        raise


def hyperbo_kernel_from_prior(prior) -> Kernel:
    """Rebuild HyperBO's PRE-TRAINED deep kernel as a plain GPyTorch Kernel.

    HyperBO's kernel is a different object from anything the EM model has used so far:
    `em_additive_freshbase` fits a fresh Matern on the target, `warmbase` warm-starts our
    own SumMLL canonical kernel, and `_DeepKernel` trains an MLP for 2k Adam steps against
    the summed marginal likelihood. HyperBO's is an MLP + ARD Matern fit for 25k steps
    against its own NLL/EKL objective across the pooled tasks -- far more pre-training
    compute, and the representation that makes HyperBO strong on dense grids.

    Reusing it as the EM model's canonical kernel is the natural hybrid: HyperBO's learned
    metric driving the interpolation weights W and the Nystrom residual Lambda, with EM's
    closed-form empirical mean and covariance on top. It is only expected to matter
    OFF-grid -- on a shared grid Lambda == 0 and W == I, so the canonical kernel is
    provably a no-op there.
    """
    from botorch.models.empirical_gps.hyperbo import (
        HyperBODeepKernel,
        MLPFeatureExtractor,
    )

    fe = MLPFeatureExtractor(
        input_dim=prior.input_dim, hidden_dims=tuple(prior.hidden_dims)
    )
    fe.load_state_dict(prior.feature_extractor_state)
    kern = HyperBODeepKernel(
        feature_extractor=fe,
        base_kernel=ScaleKernel(MaternKernel(nu=2.5)),
    )
    kern.load_state_dict(prior.covar_module_state)
    for prm in kern.parameters():
        prm.requires_grad_(False)  # frozen: this is a pre-trained prior, not a fit
    return kern.to(dtype=torch.double)


class _DeepKernel(Kernel):
    """ARD Matern-5/2 applied to a learned MLP embedding: k(x,x') = k_M(phi(x), phi(x')).

    The canonical kernel is what sets the EM model's predictive-variance SHAPE -- it
    drives the interpolation weights W = K(X,Z)K(Z,Z)^-1, the Nystrom residual Lambda,
    and the additive base. Diagnostics isolated that shape as the residual gap to
    HyperBO: with both models optimally variance-rescaled the gap WIDENS (+0.60 ->
    +0.74 nats), and RMSE explains only ~0.12 of it, so what remains lives in
    mean(log sigma_i) -- knowing *where* to be uncertain. A plain ARD Matern has only
    d=7 lengthscales with which to express that; HyperBO's Matern acts on a learned
    32-dimensional feature map. This gives the empirical prior the same metric-learning
    capacity while keeping its closed-form mean and covariance.
    """

    def __init__(self, input_dim: int, base: Kernel, hidden=(32, 32)) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
            prev = h
        self.net = torch.nn.Sequential(*layers).to(dtype=torch.double)
        # Small init keeps the embedding out of tanh saturation, where many inputs map
        # to nearly the same point and the gram matrix goes singular (NotPSDError).
        with torch.no_grad():
            for mod in self.net:
                if isinstance(mod, torch.nn.Linear):
                    mod.weight.mul_(0.1)
                    mod.bias.mul_(0.0)
        self.base = base

    @property
    def base_kernel(self):
        # Downstream helpers (hyperparameter logging, additive-variant construction)
        # reach for covar_module.base_kernel; delegate to the Matern inside the
        # ScaleKernel so the deep kernel is a drop-in for a plain ScaleKernel.
        return self.base.base_kernel

    @property
    def outputscale(self):
        return self.base.outputscale

    def _embed(self, x):
        # Skip connection: phi(x) = [x, MLP(x)]. Two reasons. It keeps the embedding
        # injective, so the gram cannot collapse. And it makes the deep kernel a strict
        # GENERALIZATION of the plain ARD Matern -- if the learned features are useless
        # their ARD lengthscales grow and the kernel reduces to the original, so this
        # cannot be worse than the baseline at the optimum.
        return torch.cat([x, self.net(x)], dim=-1)

    def forward(self, x1, x2, diag: bool = False, **params):
        return self.base.forward(self._embed(x1), self._embed(x2), diag=diag, **params)


def build_shared_kernel(
    pretrain_ds: list[ExperimentDataset],
    max_pts: int = 800,
    deep_hidden: tuple[int, ...] | None = None,
    deep_iters: int = 2000,
    base_mean: str = "constant",
):
    """Fit one ARD Matern-5/2 jointly across the pre-training tasks (shared SumMLL).

    This is the EM model's canonical/interpolation kernel: it becomes
    ``initial_covar_module`` and so determines how the empirical covariance is carried
    to inputs away from the inducing grid. ``max_pts`` subsamples each task, because an
    exact-GP MLL is cubic per task and the full PD1 pools run to ~3000 points.
    """
    if max_pts > 0:
        pretrain_ds = [
            (
                ds
                if ds.X.shape[0] <= max_pts
                else ExperimentDataset(
                    X=ds.X[(idx := torch.randperm(ds.X.shape[0])[:max_pts])],
                    Y=ds.Y[idx],
                )
            )
            for ds in pretrain_ds
        ]
    d = pretrain_ds[0].X.shape[-1]
    # EM interpolates as mu(X) = m(X) + W @ delta_mu, so a ConstantMean makes the
    # extrapolated mean flat away from the inducing grid. LinearMean gives it a slope
    # and is fitted jointly by the same SumMLL. Never explored before 2026-08-17.
    mean = LinearMean(input_size=d) if base_mean == "linear" else ConstantMean()
    if deep_hidden:
        # Matern lives in the learned embedding, so ARD is over feat_dim, not d.
        feat = d + deep_hidden[-1]  # skip connection: [x, MLP(x)]
        covar = _DeepKernel(
            input_dim=d,
            hidden=tuple(deep_hidden),
            base=ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=feat,
                    lengthscale_constraint=LogTransformedInterval(
                        0.01, 100.0, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    0.01, 100.0, initial_value=1.0
                ),
            ),
        )
        model_list, mll = build_shared_gp_model_list(
            pretrain_ds, mean, covar, outcome_transform=None
        )
        # L-BFGS is unreliable through an MLP; Adam matches how HyperBO fits its deep
        # kernel. SumMarginalLogLikelihood needs explicit (outputs, targets), so the
        # model list is evaluated on its own training inputs each step.
        mll.train()
        # lr 1e-3 + gradient clipping: at 1e-2 Adam steps the raw noise far enough
        # negative that softplus underflows and the LogNormal prior's support check
        # fails. Non-finite losses abort the fit and keep the last good parameters
        # rather than poisoning the prior.
        opt = torch.optim.Adam(mll.parameters(), lr=1e-3, foreach=True)
        best = {k: v.detach().clone() for k, v in mll.state_dict().items()}
        for _ in range(deep_iters):
            opt.zero_grad()
            try:
                out = model_list(*model_list.train_inputs)
                loss = -mll(out, model_list.train_targets).sum()
                if not torch.isfinite(loss):
                    break
                loss.backward()
            except Exception:  # noqa: BLE001 - keep the last finite parameters
                break
            torch.nn.utils.clip_grad_norm_(mll.parameters(), 10.0)
            # Snapshot BEFORE stepping: snapshotting after meant `best` held the
            # parameters produced by the step whose loss had not yet been validated, so
            # a non-finite loss on the next iteration kept the bad parameters -- the
            # opposite of what the comment claimed (D9).
            best = {k: v.detach().clone() for k, v in mll.state_dict().items()}
            opt.step()
        mll.load_state_dict(best)
        mll.eval()
        return mean, covar
    covar = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_constraint=LogTransformedInterval(
                0.01, 100.0, initial_value=1.0
            ),
        ),
        outputscale_constraint=LogTransformedInterval(0.01, 100.0, initial_value=1.0),
    )
    # outcome_transform=None: the caller already standardized globally, and the
    # per-task default would refit this mean against zero-mean targets (D2).
    _, mll = build_shared_gp_model_list(
        pretrain_ds, mean, covar, outcome_transform=None
    )
    fit_gpytorch_mll(mll)
    return mean, covar


# ---------------------------------------------------------------------------
# ABLR (Perrone et al. 2018, "Scalable Hyperparameter Transfer Learning"): a shared
# MLP feature map meta-trained across the pretrain tasks + a Bayesian linear-regression
# head. The standard transfer-BO reference. We learn a shared prior/noise precision
# (alpha, beta) jointly with the features (a stable variant of the original per-task
# precisions); the latent predictive posterior is analytic (dual BLR, Bishop 3.5).
# ---------------------------------------------------------------------------
class _ABLRPrior:
    def __init__(self, net, log_alpha, log_beta):
        self.net = net
        self.log_alpha = log_alpha
        self.log_beta = log_beta


class _ABLRPosterior:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


def _make_ablr_net(input_dim: int, feat_dim: int, hidden) -> torch.nn.Sequential:
    layers: list = []
    prev = input_dim
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
        prev = h
    layers += [torch.nn.Linear(prev, feat_dim), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers).double()


def _blr_neg_log_evidence(Phi, y, alpha, beta):
    # Dual (D-dim) BLR negative log evidence up to an additive constant (Bishop 3.5).
    n, D = Phi.shape
    A = alpha * torch.eye(D, dtype=Phi.dtype) + beta * (Phi.transpose(-1, -2) @ Phi)
    L = torch.linalg.cholesky(A)
    Phi_y = Phi.transpose(-1, -2) @ y  # (D,1)
    v = torch.cholesky_solve(Phi_y, L)  # A^{-1} Phi^T y
    quad = (
        beta * (y * y).sum() - (beta * beta) * (Phi_y.transpose(-1, -2) @ v).squeeze()
    )
    logdetA = 2.0 * torch.log(torch.diagonal(L)).sum()
    log_ev = 0.5 * (D * torch.log(alpha) + n * torch.log(beta) - quad - logdetA)
    return -log_ev


def pretrain_ablr(
    datasets,
    input_dim: int,
    hidden=(32, 32),
    feat_dim: int = 50,
    num_iterations: int = 2000,
    learning_rate: float = 1e-3,
    subsample_size: int | None = None,
    task_batch_size: int | None = None,
    per_task_precision: bool = False,
) -> _ABLRPrior:
    """Meta-train the shared feature map on the summed per-task BLR evidence.

    ``per_task_precision`` gives every pre-training task its own ``(alpha_t, beta_t)``,
    learned jointly with the shared features, which is what Perrone et al. (2018)
    prescribe. A single shared pair -- the original behaviour here, kept as the default
    so earlier results stay reproducible -- forces one noise level and one signal scale
    onto heterogeneous tasks and handicaps the method. The returned prior carries the
    mean of the learned precisions, which ``ABLRSurrogate`` uses as-is (``ablr``) or as
    the initialization for a per-target fit (``ablr_adapt``).

    ``task_batch_size`` samples a subset of tasks per step with an unbiased rescaling,
    matching what ``pretrain_hyperbo`` / ``pretrain_pacoh_gp`` already accept; without it
    ABLR was the only meta-learner denied stochastic task sampling.
    """
    net = _make_ablr_net(input_dim, feat_dim, hidden)
    n_tasks = len(datasets)
    shape = (n_tasks,) if per_task_precision else ()
    log_alpha = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    log_beta = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    opt = torch.optim.Adam(
        list(net.parameters()) + [log_alpha, log_beta], lr=learning_rate
    )
    Xs = [ds.X for ds in datasets]
    Ys = [ds.Y for ds in datasets]
    for _ in range(num_iterations):
        opt.zero_grad()
        if task_batch_size is not None and task_batch_size < n_tasks:
            sel = torch.randperm(n_tasks)[:task_batch_size].tolist()
            scale = n_tasks / len(sel)  # unbiased estimate of the full-batch sum
        else:
            sel, scale = range(n_tasks), 1.0
        loss = torch.zeros((), dtype=torch.double)
        for ti in sel:
            X, y = Xs[ti], Ys[ti]
            if subsample_size is not None and X.shape[0] > subsample_size:
                idx = torch.randperm(X.shape[0])[:subsample_size]
                X, y = X[idx], y[idx]
            a = (log_alpha[ti] if per_task_precision else log_alpha).clamp(-9.0, 9.0)
            b = (log_beta[ti] if per_task_precision else log_beta).clamp(-9.0, 13.0)
            loss = loss + _blr_neg_log_evidence(net(X), y, a.exp(), b.exp())
        (loss * scale).backward()
        opt.step()
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return _ABLRPrior(net, log_alpha.detach().mean(), log_beta.detach().mean())


class ABLRSurrogate:
    """Frozen shared features + per-task Bayesian linear head; analytic posterior.

    ``fit_precisions`` re-fits ``(alpha, beta)`` on the target task's own observations by
    maximizing its evidence, starting from the meta-learned values. That is the per-task
    adaptation Perrone et al. (2018) describe, and it is cheap -- two scalars, with the
    frozen features giving a fixed design matrix. Without it the target inherits the
    average precisions of the pre-training tasks (the ``ablr`` arm).
    """

    def __init__(
        self,
        prior: _ABLRPrior,
        X: torch.Tensor,
        Y: torch.Tensor,
        fit_precisions: bool = False,
        n_steps: int = 100,
    ):
        self._net = prior.net
        with torch.no_grad():
            Phi = self._net(X)  # (n,D); features are frozen
        log_a, log_b = prior.log_alpha.clone(), prior.log_beta.clone()
        if fit_precisions and X.shape[0] >= 2:
            la = log_a.detach().clone().requires_grad_(True)
            lb = log_b.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([la, lb], lr=0.1)
            try:
                for _ in range(n_steps):
                    opt.zero_grad()
                    loss = _blr_neg_log_evidence(
                        Phi, Y, la.clamp(-9.0, 9.0).exp(), lb.clamp(-9.0, 13.0).exp()
                    )
                    loss.backward()
                    opt.step()
                log_a, log_b = la.detach(), lb.detach()
            except Exception:
                pass  # ill-conditioned tiny design -> keep the meta-learned precisions
        alpha = log_a.clamp(-9.0, 9.0).exp()
        self._beta = log_b.clamp(-9.0, 13.0).exp()
        D = Phi.shape[-1]
        A = alpha * torch.eye(D, dtype=X.dtype) + self._beta * (
            Phi.transpose(-1, -2) @ Phi
        )
        self._L = torch.linalg.cholesky(A)
        self._mbeta = self._beta * torch.cholesky_solve(
            Phi.transpose(-1, -2) @ Y, self._L
        )  # (D,1)

    def posterior(self, cand: torch.Tensor) -> _ABLRPosterior:
        Phi_c = self._net(cand)  # (m,D)
        mean = Phi_c @ self._mbeta  # (m,1)
        V = torch.cholesky_solve(Phi_c.transpose(-1, -2), self._L)  # A^{-1} Phi_c^T
        var = (Phi_c * V.transpose(-1, -2)).sum(-1, keepdim=True)  # diag latent var
        return _ABLRPosterior(mean, var.clamp_min(1e-12))


def _safe_copy_outputscale(dst, src) -> bool:
    """Copy raw_outputscale when BOTH modules expose it.

    Composite canonical kernels do not: HyperBODeepKernel keeps its ScaleKernel one
    level down and _DeepKernel exposes `outputscale` only as a property. The
    unconditional copy raised AttributeError outside any try, aborting the whole run
    whenever --em-canonical hyperbo or --deep-kernel met a pretrained_gp_* method (D10).
    """
    if src is None:
        raise ValueError(
            "_safe_copy_outputscale received src=None: the shared kernel was never "
            "fitted. That happens when a pretrained_gp_*/warmstart_gp method is "
            "requested with no em* method, so build_shared_kernel never ran. Refusing "
            "to hand back a silently untrained 'pre-trained' baseline."
        )
    s = getattr(src, "raw_outputscale", None)
    d = getattr(dst, "raw_outputscale", None)
    if s is None or d is None:
        # Declining is legitimate for composite kernels, but it means this baseline is
        # NOT warm-started, and reporting it as pre-trained would be a silent wrong
        # number -- the failure mode this project keeps shipping (R3).
        print(
            f"  WARNING: cannot transfer outputscale from {type(src).__name__} to "
            f"{type(dst).__name__}; this baseline is NOT warm-started",
            flush=True,
        )
        return False
    with torch.no_grad():
        d.data.copy_(s.data)
    return True


def _safe_copy_lengthscale(dst, src) -> bool:
    """Copy raw_lengthscale when both exist AND their ARD dimensions agree."""
    if dst is None or src is None:
        print(
            "  WARNING: no lengthscale source/target; this baseline is NOT warm-started",
            flush=True,
        )
        return False
    s = getattr(src, "raw_lengthscale", None)
    d = getattr(dst, "raw_lengthscale", None)
    if s is None or d is None or s.shape != d.shape:
        print(
            f"  WARNING: cannot transfer lengthscale from {type(src).__name__} to "
            f"{type(dst).__name__} (shapes "
            f"{None if s is None else tuple(s.shape)} vs "
            f"{None if d is None else tuple(d.shape)}); this baseline is NOT "
            "warm-started",
            flush=True,
        )
        return False
    with torch.no_grad():
        d.data.copy_(s.data)
    return True


def make_surrogate(
    method: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    em_prior,
    mean_s,
    covar_s,
    hyperbo_prior=None,
    pacoh_prior=None,
    ablr_prior=None,
    mean_baseline=None,
):
    """Build (and, where applicable, fit) a surrogate on observed (X, Y std-units)."""
    d = X.shape[-1]  # input dimension (benchmark-agnostic)
    if method in ("em_frozen", "em_finetuned", "em_noisefit"):
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        # em_finetuned augments the empirical prior with a fresh, fully-trainable
        # additive base kernel (Sigma + K_base) fit on the observed data; em_frozen
        # uses the pure empirical prior (no base, no fit).
        base = (
            ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=d,
                    lengthscale_constraint=LogTransformedInterval(
                        0.01, 100.0, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    0.01, 100.0, initial_value=1.0
                ),
            )
            if method == "em_finetuned"
            else None
        )
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=X,
            train_Y=Y,
            likelihood=lik,
            base_covar_module=base,
        )
        if method == "em_finetuned":
            model.train()
            lik.train()
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(lik, model))
            except Exception:
                pass  # fall back to frozen prior on a failed fit
            _log_hp(method, model._additive_base)
        elif method == "em_noisefit":
            # Freeze the empirical mean+Sigma; fit ONLY the likelihood noise on the
            # incoming task -- isolates whether per-task noise adaptation alone explains
            # HyperBO's noise-regime edge (see obs-noise sweep, section 5K).
            for p in model.parameters():
                p.requires_grad_(False)
            lik.raw_noise.requires_grad_(True)
            model.train()
            lik.train()
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(lik, model))
            except Exception:
                pass  # borderline fit -> keep the frozen conditioning noise
        model.eval()
        lik.eval()
        return model

    if method in ("pretrained_gp_frozen", "pretrained_gp_tuned"):
        # The NON-EM baselines must use the pre-blend mean. With --em-mean, mean_s is a
        # _BlendedMean carrying HyperBO's meta-learned mean, so reusing it here turned
        # this control into a second transfer method (D1). mean_baseline defaults to
        # mean_s so callers that never blend are unaffected.
        base_mean_src = mean_s if mean_baseline is None else mean_baseline
        # Reuse the shared SumMLL kernel (fitted lengthscales + outputscale).
        cc = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
        # Composite canonical kernels (HyperBODeepKernel, _DeepKernel) do not expose
        # raw_outputscale, and ARD dims differ, so the unguarded copies raised outside
        # any try and aborted the whole run (D10).
        _safe_copy_outputscale(cc, covar_s)
        _safe_copy_lengthscale(cc.base_kernel, getattr(covar_s, "base_kernel", None))
        if method == "pretrained_gp_frozen":
            # Pure reuse of the pre-trained hyperparameters: also inherit the shared
            # ConstantMean and a fixed conditioning noise (matching em_frozen); NO fit.
            # Inherit the shared mean. Copying `.constant` assumed a ConstantMean, so
            # --em-base-mean linear raised AttributeError and took this baseline down
            # with it (review, integration #6). Reuse the shared module directly when
            # it is not a ConstantMean.
            if getattr(base_mean_src, "constant", None) is not None:
                mm = ConstantMean()
                with torch.no_grad():
                    mm.constant.data.copy_(base_mean_src.constant.data)
            else:
                mm = copy.deepcopy(base_mean_src)
                for _p in mm.parameters():
                    _p.requires_grad_(False)
            lik = GaussianLikelihood()
            lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
            gp = SingleTaskGP(
                X,
                Y,
                likelihood=lik,
                covar_module=cc,
                mean_module=mm,
                outcome_transform=None,
            )
            gp.eval()
            return gp
        # pretrained_gp_tuned: freeze transferred lengthscales, fit mean + noise
        # + scale.
        gp = SingleTaskGP(X, Y, covar_module=cc, outcome_transform=None)
        gp.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass  # borderline fit on tiny design -> keep unfitted init
        gp.eval()
        return gp

    if method == "warmstart_gp":
        # Warm start: initialize at the shared SumMLL lengthscales + outputscale,
        # then fit EVERYTHING (lengthscales included) on the incoming data. Tests
        # whether the pre-trained kernel is a useful *initialization* even though it
        # is a poor *frozen* choice.
        cc = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
        # Guarded: composite canonical kernels expose neither raw_outputscale nor a
        # shape-compatible raw_lengthscale, and the raw copies raised outside any try,
        # aborting the whole shard (R2).
        _safe_copy_outputscale(cc, covar_s)
        _safe_copy_lengthscale(cc.base_kernel, getattr(covar_s, "base_kernel", None))
        gp = SingleTaskGP(X, Y, covar_module=cc, outcome_transform=None)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass  # borderline fit on tiny design -> keep unfitted init
        gp.eval()
        return gp

    if method == "vanilla_gp":
        gp = SingleTaskGP(X, Y, outcome_transform=None)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass  # borderline fit on tiny design -> keep unfitted init
        gp.eval()
        return gp

    if method in ("em_additive_freshbase", "em_additive_warmbase"):
        # Additive (sum-of-kernels) research variant with an explicit empirical scale:
        #   Sigma_used = s1 * Sigma_empirical + K_base(Z, Z)
        # s1 is a free positive scalar (empirical magnitude); the base is a ScaleKernel
        # whose outputscale is the second free scalar s2. Each component's magnitude is
        # learned directly, and s1 can recalibrate the empirical-prior variance.
        # (The shipped library uses base_covar_module without the extra s1.)
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=X,
            train_Y=Y,
            likelihood=lik,
        )
        base = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=d,
                lengthscale_constraint=LogTransformedInterval(
                    0.01, 100.0, initial_value=1.0
                ),
            ),
            outputscale_constraint=LogTransformedInterval(
                0.01, 100.0, initial_value=1.0
            ),
        )
        if method == "em_additive_warmbase":
            # See R2: model.initial_covar_module is the canonical kernel, so under
            # --em-canonical hyperbo / --deep-kernel it has neither attribute.
            _safe_copy_lengthscale(
                base.base_kernel,
                getattr(model.initial_covar_module, "base_kernel", None),
            )
            _safe_copy_outputscale(base, model.initial_covar_module)
        # Use the library's supported hooks. An earlier version overrode
        # _effective_Sigma_inducing, which is only consulted by _get_prior_at_indices;
        # with enable_interpolation=True the live path is _interpolate_prior_to_X, so the
        # override was never called and this variant silently collapsed onto em_noisefit
        # (bit-identical trajectories). _additive_base and _sigma_scale are honoured by
        # both paths.
        model._additive_base = base
        # Registered Parameter, read (and exponentiated) by the model on every forward.
        model._log_sigma_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        model.train()
        lik.train()
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(lik, model))
        except Exception:
            pass  # borderline fit -> keep init
        _log_hp(method, model._additive_base)
        model.eval()
        lik.eval()
        return model

    if method in ("em_additive_hyperbo", "em_additive_hyperbo_frozen"):
        # HyperBO's PRE-TRAINED kernel as an additive component at conditioning:
        #     k(x,x') = k_emp(x,x') + s * k_hyperbo(x,x')
        # hyperbo_kernel_from_prior() already froze every internal parameter, so the
        # ScaleKernel wrapper contributes EXACTLY ONE trainable scalar s. That matters:
        # em_additive_freshbase fits a whole ARD Matern on the target and reaches NLL
        # 54.8 at n=5, while a single magnitude cannot overfit a 5-point design.
        # The _frozen variant pins s = 1 so the kernel enters as a pure prior with no
        # target adaptation at all, separating "is the representation useful" from
        # "does adapting its weight help".
        if HYPERBO_BASE_KERNEL is None:
            raise ValueError(
                f"{method} needs a pre-trained HyperBO prior, but none was built. "
                "Include a hyperbo_* method so one is trained. Note that on the PD1 "
                "leave-one-out path `--em-canonical hyperbo` does NOT satisfy this: it "
                "builds a kernel for the EM prior without publishing it as the "
                "additive base."
            )
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=X,
            train_Y=Y,
            likelihood=lik,
        )
        base = ScaleKernel(
            HYPERBO_BASE_KERNEL,
            outputscale_constraint=LogTransformedInterval(
                0.01, 100.0, initial_value=1.0
            ),
        )
        if method == "em_additive_hyperbo_frozen":
            base.raw_outputscale.requires_grad_(False)
        model._additive_base = base
        model._log_sigma_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        model.train()
        lik.train()
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(lik, model))
        except Exception:  # noqa: BLE001 - borderline fit -> keep init
            pass
        _log_hp(method, model._additive_base)
        model.eval()
        lik.eval()
        return model

    if method == "em_additive_3way":
        # Three-way additive kernel: empirical (frozen, scaled by s1) + pre-trained
        # kernel (frozen lengthscales, free outputscale) + fresh kernel (all free).
        # MLL fits {s1, pretrained outputscale, fresh lengthscales+outputscale, noise}.
        # Blends all transfer regimes; fresh = vanilla fallback, pretrained = structured
        # mid-point that may regularize the fresh component.
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=X,
            train_Y=Y,
            likelihood=lik,
        )

        def _mk_base():
            return ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=d,
                    lengthscale_constraint=LogTransformedInterval(
                        0.01, 100.0, initial_value=1.0
                    ),
                ),
                outputscale_constraint=LogTransformedInterval(
                    0.01, 100.0, initial_value=1.0
                ),
            )

        base_pre = _mk_base()
        # See R2.
        _safe_copy_lengthscale(
            base_pre.base_kernel,
            getattr(model.initial_covar_module, "base_kernel", None),
        )
        _safe_copy_outputscale(base_pre, model.initial_covar_module)
        base_pre.base_kernel.raw_lengthscale.requires_grad_(False)  # frozen transfer
        base_fresh = _mk_base()
        # Both bases go through the supported _additive_base hook as a single
        # AdditiveKernel; overriding _effective_Sigma_inducing does nothing on the
        # interpolation path (see the freshbase branch).
        model._additive_base = AdditiveKernel(base_pre, base_fresh)
        model._log_sigma_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        model.train()
        lik.train()
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(lik, model))
        except Exception:
            pass  # borderline fit -> keep init
        _log_hp(method, model._additive_base)
        model.eval()
        lik.eval()
        return model

    if method in ("hyperbo_frozen", "hyperbo_adapt"):
        # HyperBO (Wang et al. 2021): deep-kernel GP whose MLP feature extractor +
        # Matern kernel + linear mean are meta-learned across the pretrain datasets.
        #   frozen: use the pretrained prior exactly as the paper prescribes (all
        #           hyperparameters, including the meta-learned noise, frozen).
        #   adapt : keep features/kernel frozen (paper: prior is fixed after
        #           pre-training) but re-fit ONLY the Gaussian likelihood noise on the
        #           incoming task, mirroring pretrained_gp_tuned / em_finetuned's
        #           per-task noise adaptation.
        model = HyperBOModel.from_pretrained(
            hyperbo_prior, train_X=X, train_Y=Y, freeze_pretrained=True
        )
        if method == "hyperbo_adapt":
            model.likelihood.raw_noise.requires_grad_(True)
            model.train()
            model.likelihood.train()
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass  # borderline fit on tiny design -> keep pretrained noise
        model.eval()
        model.likelihood.eval()
        return model

    if method in ("pacoh_frozen", "pacoh_adapt"):
        # PACOH-GP (Rothfuss et al. 2021): PAC-Bayesian meta-learning over the HyperBO
        # deep-kernel prior; the hyper-posterior is approximated by K SVGD particles and
        # prediction is a mixture-of-GPs (GaussianMixturePosterior).
        #   frozen: all K particles frozen (paper-faithful).
        #   adapt : re-fit only the (batched) likelihood noise on the incoming task.
        model = PACOHGPModel.from_pretrained(
            pacoh_prior, train_X=X, train_Y=Y, freeze_pretrained=True
        )
        if method == "pacoh_adapt":
            # Adapt the shared observation noise across all K particles with a short
            # Adam refit of the summed per-particle marginal log-likelihood.
            model.likelihood.raw_noise.requires_grad_(True)
            model.train()
            Yt = Y.squeeze(-1) if (Y.ndim == 2 and Y.shape[-1] == 1) else Y
            opt = torch.optim.Adam([model.likelihood.raw_noise], lr=0.1, foreach=True)
            try:
                for _ in range(50):
                    opt.zero_grad()
                    out = model.likelihood(model.forward(X))
                    loss = -out.log_prob(Yt).sum()
                    loss.backward()
                    opt.step()
            except Exception:
                pass  # borderline fit -> keep pretrained noise
        model.eval()
        model.likelihood.eval()
        return model

    if method in ("ablr", "ablr_adapt"):
        # ABLR: frozen shared MLP features + analytic Bayesian linear-regression head.
        #   ablr      : inherit the meta-learned precisions (alpha, beta).
        #   ablr_adapt: re-fit them on the incoming task's evidence, per Perrone et al.
        return ABLRSurrogate(ablr_prior, X, Y, fit_precisions=(method == "ablr_adapt"))

    raise ValueError(f"unknown method {method}")


# A method's frozen-prior surrogate for a zero-shot (no-data) first pick, or None if
# it has no informative prior mean (-> random first pick, the honest n_init=0 baseline
# for constant-mean models like vanilla_gp / pretrained_gp_frozen).
def _zero_shot_prior_method(method: str) -> str | None:
    for pref, frozen in (
        ("em_", "em_frozen"),
        ("hyperbo_", "hyperbo_frozen"),
        ("pacoh_", "pacoh_frozen"),
    ):
        if method.startswith(pref):
            return frozen
    return None


def _prior_mean_pick(
    frozen_method: str,
    pool_X: torch.Tensor,
    remaining: list[int],
    surrogate_fn: Callable,
    topq: int = 1,
) -> list[int]:
    """Zero-shot pick(s): top-q of the surrogate's PRIOR mean (no observations).

    Builds the frozen prior model with a single dummy point (used only to construct it)
    and queries ``model.forward(cand)`` -- the GP prior at the candidates. For EM this
    is the shift-interpolated empirical mean ``m(X) + W @ Delta_mu``; for HyperBO it is
    the linear-mean-on-features; for PACOH it is the per-particle prior mean, averaged
    over particles for the mixture prior mean.
    """
    dummy_X = pool_X[remaining[:1]]
    dummy_Y = torch.zeros(1, 1, dtype=pool_X.dtype, device=pool_X.device)
    model = surrogate_fn(frozen_method, dummy_X, dummy_Y)
    cand = pool_X[remaining]
    with torch.no_grad():
        prior = model.forward(cand)
        pmean = prior.mean
        if pmean.ndim > 1:  # PACOH: batched over K particles -> mixture prior mean
            pmean = pmean.mean(0)
        pmean = pmean.reshape(-1)
    # The zero-shot pick (n_init=0) is exactly the quantity behind the "EM owns the
    # zero-shot pick" claim, and it was the one selection site left unguarded -- a NaN
    # prior mean here still produced a silent, deterministic, arbitrary index (R6-B3).
    pmean = _assert_finite_scores(pmean, "prior_mean", "zero-shot prior-mean pick")
    top = torch.topk(pmean, min(topq, pmean.numel())).indices.tolist()
    return [remaining[i] for i in top]


def _posterior_moments(model, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginal predictive (mean, sigma) at candidates, flattened to 1D.

    PACOH returns a GaussianMixturePosterior whose ``.mean``/``.variance`` keep the
    per-particle MCMC dim; use the marginalized mixture moments. Standard BoTorch
    posteriors and ABLR's analytic head fall through.
    """
    post = model.posterior(cand)
    if hasattr(post, "mixture_mean"):
        mean = post.mixture_mean.reshape(-1)
        sigma = post.mixture_variance.clamp_min(1e-12).sqrt().reshape(-1)
    else:
        mean = post.mean.reshape(-1)
        sigma = post.variance.clamp_min(1e-12).sqrt().reshape(-1)
    return mean, sigma


def _assert_finite_scores(score, method: str, where: str):
    """A NaN posterior otherwise becomes a silent, deterministic, arbitrary pick.

    argmax/topk on a NaN tensor return an index without complaint, so a failed
    Cholesky or a saturated deep-kernel embedding produced a plausible-looking
    trajectory instead of an error (round-4 N3). _pd1_probe already checked this; the
    production loop did not.
    """
    if not torch.isfinite(score).any():
        raise FloatingPointError(
            f"{method}: every acquisition score is non-finite at {where}. This is a "
            "model failure (Cholesky, degenerate Sigma, saturated embedding), not a "
            "tie -- refusing to return an arbitrary argmax."
        )
    if torch.isposinf(score).any():
        # +inf is a model blow-up, not the best candidate. Mapping it to -inf would
        # silently rank it LAST; mapping it to +inf would make it win. Neither is a
        # measurement, so refuse (round-6 B2).
        raise FloatingPointError(
            f"{method}: acquisition produced +inf at {where}, which is a model blow-up "
            "rather than an optimum. Refusing to rank it."
        )
    return torch.nan_to_num(score, nan=-float("inf"))


def _acq_score(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    best: float,
    acquisition: str,
    ucb_beta: float,
) -> torch.Tensor:
    """Finite-pool acquisition value (argmax-identical to BoTorch's analytic form)."""
    if acquisition == "greedy":
        return mean  # pure exploitation -> isolates prior-mean quality
    if acquisition == "ucb":
        return mean + ucb_beta * sigma  # exploration -> stresses uncertainty
    return _log_ei_helper((mean - best) / sigma) + sigma.log()


def _fantasy_batch_pick(
    eff_method: str,
    pool_X: torch.Tensor,
    observed: list[int],
    pool_Y_cond: torch.Tensor,
    remaining: list[int],
    best: float,
    q: int,
    surrogate_fn: Callable,
    acquisition: str,
    ucb_beta: float,
    n_fantasies: int,
    rng: torch.Generator,
) -> tuple[list[int], torch.Tensor, torch.Tensor, int]:
    """Sequential-greedy q-batch with fantasies (the standard qEI/qLogEI construction).

    The j-th batch member is scored by its acquisition value *marginalized over
    fantasized outcomes* for the j-1 already-selected points:
    ``a_j(x) = E_y[ a(x | D u {(x_i, y_i)}_{i<j}) ]``, a Monte-Carlo average over
    ``n_fantasies`` posterior draws (``n_fantasies=0`` = "kriging believer": one
    deterministic fantasy pinned at the posterior mean). Averaging happens in EI space
    via ``logsumexp`` for LogEI, and in score space for greedy/UCB.

    This is BoTorch's ``sequential=True`` q-batch strategy. A *joint* qLogEI is not
    available uniformly across this method set -- PACOH's mixture posterior and ABLR's
    analytic linear head expose marginals only, not a joint posterior over q points --
    so the fantasy construction is what keeps every surrogate on the same footing.

    Returns ``(picks, mean0, sigma0, idx0)``: the batch, plus the *unconditioned*
    posterior moments over ``remaining`` and the index (into ``remaining``) of the first
    pick, so the caller can log 1-step-ahead calibration exactly as in the q=1 path.
    """
    kriging_believer = n_fantasies <= 0
    n_scen = 1 if kriging_believer else n_fantasies
    X_obs, Y_obs = pool_X[observed], pool_Y_cond[observed]
    # Until the first fantasy is drawn every scenario is identical, so share one model.
    base_model = surrogate_fn(eff_method, X_obs, Y_obs)
    models = [base_model]
    scen_X: list[torch.Tensor] = [X_obs]
    scen_Y: list[torch.Tensor] = [Y_obs]
    incumbents: list[float] = [best]
    diverged = False  # True once scenarios differ (after the first fantasy)

    picks: list[int] = []
    avail = list(remaining)
    mean0 = sigma0 = None
    idx0 = 0
    for j in range(q):
        cand = pool_X[avail]
        with torch.no_grad():
            scores = [
                _acq_score(
                    *_posterior_moments(models[f], cand),
                    incumbents[f],
                    acquisition,
                    ucb_beta,
                )
                for f in range(len(models))
            ]
            if j == 0:
                mean0, sigma0 = _posterior_moments(base_model, cand)
            if len(scores) == 1:
                score = scores[0]
            elif acquisition == "logei":
                S = torch.stack(scores)  # (F, M) log-EI per scenario
                score = torch.logsumexp(S, dim=0) - math.log(S.shape[0])
            else:
                score = torch.stack(scores).mean(0)
        score = _assert_finite_scores(score, eff_method, "fantasy batch selection")
        bi = int(torch.argmax(score))
        if j == 0:
            idx0 = bi
        picks.append(avail[bi])
        pick_X = pool_X[avail[bi]].unsqueeze(0)
        avail.pop(bi)
        if j == q - 1 or not avail:
            break
        # Fantasize this pick's outcome in every scenario, then re-condition.
        if not diverged:
            scen_X = [scen_X[0] for _ in range(n_scen)]
            scen_Y = [scen_Y[0] for _ in range(n_scen)]
            incumbents = [incumbents[0] for _ in range(n_scen)]
            models = [models[0] for _ in range(n_scen)]
            diverged = True
        new_models = []
        for f in range(n_scen):
            with torch.no_grad():
                mu_f, sd_f = _posterior_moments(models[f], pick_X)
            y_f = mu_f.reshape(1, 1)
            if not kriging_believer:
                eps = torch.randn(1, 1, generator=rng, dtype=pool_X.dtype)
                y_f = y_f + sd_f.reshape(1, 1) * eps
            scen_X[f] = torch.cat([scen_X[f], pick_X], dim=0)
            scen_Y[f] = torch.cat([scen_Y[f], y_f], dim=0)
            incumbents[f] = max(incumbents[f], y_f.item())
            new_models.append(surrogate_fn(eff_method, scen_X[f], scen_Y[f]))
        models = new_models
    return picks, mean0, sigma0, idx0


def run_bo(
    method: str,
    pool_X: torch.Tensor,
    pool_Y: torch.Tensor,  # standardized (N,1)
    init_idx: list[int],
    n_iters: int,
    rng: torch.Generator,
    surrogate_fn: Callable,
    acquisition: str = "logei",
    ucb_beta: float = 2.0,
    obs_noise: float = 0.0,
    batch_q: int = 1,
    batch_mode: str = "topq",
    n_fantasies: int = 0,
    calib_sink: list | None = None,
) -> list[float]:
    """Best-so-far trajectory (standardized units), indexed by acquisition round.

    Length is ``n_iters+1`` normally, or ``n_iters`` when ``init_idx`` is empty
    (n_init=0). ``obs_noise`` (std, standardized units) adds Gaussian noise to the
    observations the surrogate conditions on and to the EI incumbent; regret is always
    tracked on the TRUE objective.

    ``batch_q`` selects q candidates per round -- with batch_q>1 each round is q
    parallel evaluations, so the trajectory is indexed by round, not by #evaluations.
    ``batch_mode`` picks how the batch is formed:

    - ``topq``    : top-q by acquisition value, no fantasies (cheap, ignores
                    within-batch redundancy).
    - ``kb``      : sequential greedy, kriging believer (one deterministic fantasy at
                    the posterior mean per selected point).
    - ``fantasy`` : sequential greedy with ``n_fantasies`` Monte-Carlo fantasy draws --
                    the proper qEI/qLogEI construction.
    """
    n = pool_X.shape[0]
    # Noisy conditioning targets (surrogate + EI incumbent); regret uses the truth.
    if obs_noise > 0.0:
        noise = obs_noise * torch.randn(n, generator=rng, dtype=pool_Y.dtype)
        pool_Y_cond = pool_Y + noise.reshape(pool_Y.shape)
    else:
        pool_Y_cond = pool_Y
    observed = list(init_idx)
    remaining = [i for i in range(n) if i not in set(observed)]
    # best_true drives the regret trajectory; best (noisy) is the EI incumbent.
    best_true = pool_Y[observed].max().item() if observed else None
    best = pool_Y_cond[observed].max().item() if observed else None
    traj: list[float] = [best_true] if best_true is not None else []
    for it in range(n_iters):
        if not remaining:
            traj.append(best_true if best_true is not None else float("nan"))
            continue
        q = min(batch_q, len(remaining))
        if best is None:
            # Zero-shot round (n_init=0): top-q of the prior mean for methods with an
            # informative meta-learned prior; random otherwise.
            zs = _zero_shot_prior_method(method)
            if zs is not None:
                picks = _prior_mean_pick(zs, pool_X, remaining, surrogate_fn, topq=q)
            else:
                perm = torch.randperm(len(remaining), generator=rng)[:q].tolist()
                picks = [remaining[j] for j in perm]
        elif method == "random":
            perm = torch.randperm(len(remaining), generator=rng)[:q].tolist()
            picks = [remaining[j] for j in perm]
        else:
            eff = method
            if method.startswith("em_then_vanilla_k"):
                # Ablation: EM-EGP surrogate for the first k acquisitions, then a
                # vanilla GP thereafter.
                k = int(method[len("em_then_vanilla_k") :])
                eff = "em_frozen" if it < k else "vanilla_gp"
            if q > 1 and batch_mode in ("kb", "fantasy"):
                picks, mean, sigma, top0 = _fantasy_batch_pick(
                    eff,
                    pool_X,
                    observed,
                    pool_Y_cond,
                    remaining,
                    best,
                    q,
                    surrogate_fn,
                    acquisition,
                    ucb_beta,
                    0 if batch_mode == "kb" else n_fantasies,
                    rng,
                )
                top = [top0]  # calibrate on the first (unconditioned) pick only
            else:
                model = surrogate_fn(eff, pool_X[observed], pool_Y_cond[observed])
                cand = pool_X[remaining]  # (M, d)
                with torch.no_grad():
                    mean, sigma = _posterior_moments(model, cand)
                    score = _acq_score(mean, sigma, best, acquisition, ucb_beta)
                score = _assert_finite_scores(score, method, "top-q batch selection")
                # Finite-pool batch = top-q by acquisition (no fantasies).
                top = torch.topk(score, q).indices.tolist()
                picks = [remaining[i] for i in top]
            if calib_sink is not None:
                # 1-step-ahead predictive calibration at the acquired point(s):
                # Gaussian NLL + 95% coverage of the TRUE (noiseless) target under the
                # surrogate's latent posterior (standardized units).
                for i in top:
                    mu = mean[i].item()
                    sg = max(sigma[i].item(), 1e-6)
                    yt = pool_Y[remaining[i]].item()
                    nll = 0.5 * math.log(2 * math.pi * sg * sg) + (yt - mu) ** 2 / (
                        2 * sg * sg
                    )
                    covered = 1.0 if abs(yt - mu) <= 1.96 * sg else 0.0
                    calib_sink.append((nll, covered))
        for pick in picks:
            observed.append(pick)
            remaining.remove(pick)
            yt = pool_Y[pick].item()
            yc = pool_Y_cond[pick].item()
            best_true = yt if best_true is None else max(best_true, yt)
            best = yc if best is None else max(best, yc)
        traj.append(best_true)
    return traj


def _gaussian_rank_warp(Y: torch.Tensor) -> torch.Tensor:
    """Per-task Gaussian-copula (rank) warp: map each objective value to its standard-normal
    quantile. Monotonic (preserves the optimum and config ordering) but reshapes any marginal
    -- including PD1's bimodal good/diverged distribution -- to ~N(0,1), so a Gaussian-
    likelihood GP is well-specified. This is the output transform the HyperBO pipeline uses to
    handle heavy-tailed objectives; we omit it by default (plain standardization).
    """
    y = Y.reshape(-1)
    n = y.shape[0]
    ranks = torch.empty(n, dtype=y.dtype)
    ranks[y.argsort()] = torch.arange(1, n + 1, dtype=y.dtype)
    u = (ranks - 0.5) / n  # plotting positions in (0,1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)  # inverse standard-normal CDF
    return z.reshape(Y.shape)


def _apply_output_warp(Y_list: list[torch.Tensor], mode: str) -> list[torch.Tensor]:
    """Per-task output transform (list of (n,1) objective tensors, higher=better).
    - none: identity.
    - gaussian: rank/copula map to N(0,1) (tames heavy tails, tail-insensitive).
    - winsor: clamp the divergence tail (cap error at 0.5, i.e. -error at -0.5), collapsing
      the bimodal spike to one value before standardization -- a robust alternative that
      keeps the good-config spread linear (gaussian flattens it).
    """
    if mode == "gaussian":
        return [_gaussian_rank_warp(Y) for Y in Y_list]
    if mode == "winsor":
        return [Y.clamp_min(-0.5) for Y in Y_list]
    if mode == "neglog":
        # Paper's PD1 output warp: error r -> -log(r + 1e-10) (maximization). Our Y is
        # -error, so r = -Y; this AMPLIFIES resolution among low-error (good) configs while
        # compressing the divergence tail -- the opposite of the gaussian rank warp.
        return [-torch.log((-Y).clamp_min(0.0) + 1e-10) for Y in Y_list]
    return Y_list


def _shard_folds(n_folds: int, spec: str) -> list[int]:
    """Folds belonging to shard ``i/n`` of a leave-one-out sweep (round-robin).

    Leave-one-out re-pre-trains every prior per fold, which dominates the cost when the
    meta-learners are given their converged budgets. Folds are independent, so sharding
    across processes is the natural parallelism; ``analyze.py`` concatenates the shards.
    """
    if not spec or spec == "0/1":
        return list(range(n_folds))
    i, n = (int(v) for v in spec.split("/"))
    if not (0 <= i < n):
        raise ValueError(f"--loo-shard {spec}: need 0 <= i < n")
    return [f for f in range(n_folds) if f % n == i]


def run_pd1_matched_loo(args, methods) -> None:
    """Fair, comprehensive EM-vs-baselines on PD1 (#1+#2). Leave-one-out over a
    homogeneous subgroup's MATCHED-config pool -- larger + shared, so EM competes on the
    same discriminating pool as every baseline. Optional Gaussian output warp
    (--output-warp) and per-task standardization. Stores per-run trajectories + calibration
    so downstream analysis computes per-task-normalized regret, regret-AUC, ranks/CD.
    """
    # Required: this function assigns HYPERBO_BASE_KERNEL per fold. Without the
    # declaration the assignment binds a local and the additive-hybrid methods keep
    # seeing None -- the exact mechanism of the DEEP_KERNEL silent no-op (gotcha #1).
    global HYPERBO_BASE_KERNEL

    group = args.pd1_group
    all_names = _pd1_dataset_names()
    # "all": leave-one-out across every task. On the full PD1 release the matched grid
    # is shared by all 23 tasks (400 configs), so no batch-size subgroup is needed --
    # that restriction only existed because PD1Lite's all-task intersection was 50.
    names_all = (
        all_names
        if group == "all"
        else [n for n in all_names if n.split(",")[-1] == group]
    )
    if len(names_all) < 3:
        raise ValueError(f"--pd1-group {group}: too few tasks ({len(names_all)})")
    Xn, Y_all, names = load_pd1_pool(dataset_names=names_all)  # matched within subgroup
    Y_all = _apply_output_warp(Y_all, args.output_warp)
    T, n_conf = len(names), Xn.shape[0]
    n_init = min(args.n_init, max(1, n_conf - 1))
    # Per-task pools over ALL of each task's configurations, loaded once when either the
    # candidate set or the baseline pre-training set is set to "full".
    task_pools = None
    if "full" in (args.pd1_candidate_pool, args.pd1_pretrain_pool):
        if PD1_SOURCE != "full":
            raise ValueError(
                "--pd1-candidate-pool/--pd1-pretrain-pool full requires "
                "--pd1-source full"
            )
        from _scratch_bo.pd1_full_data import load_pd1_task_pools

        Xs, Ys, pool_names = load_pd1_task_pools(
            root=PD1_ROOT, dataset_names_filter=names
        )
        task_pools = {
            n: (Xs[j], _apply_output_warp([Ys[j]], args.output_warp)[0])
            for j, n in enumerate(pool_names)
        }
        sizes = [task_pools[n][0].shape[0] for n in names if n in task_pools]
        print(
            f"  per-task pools: {min(sizes)}-{max(sizes)} configs "
            f"(mean {sum(sizes) / len(sizes):.0f}) vs {n_conf} matched | "
            f"candidates={args.pd1_candidate_pool} pretrain={args.pd1_pretrain_pool}",
            flush=True,
        )
    print(
        f"PD1-LOO bs={group}: {T} tasks, matched pool {n_conf} configs, d={Xn.shape[1]}, "
        f"warp={args.output_warp}, per_task_std={args.per_task_standardize}",
        flush=True,
    )

    results = {m: [] for m in methods}
    calib_records = {m: [] for m in methods}
    run_ds: list[int] = []
    pool_max_raw: list[float] = []
    method_bo_s = {m: 0.0 for m in methods}
    ptime = {"em": 0.0, "hyperbo": 0.0, "pacoh": 0.0, "ablr": 0.0}
    # Cache hits are tracked alongside the timings so that a downstream cost analysis
    # can tell "cheap to fit" apart from "already fitted" (S30.1).
    pcache_hits = {"em": 0, "hyperbo": 0, "pacoh": 0, "ablr": 0}
    pcache_calls = {"em": 0, "hyperbo": 0, "pacoh": 0, "ablr": 0}
    _t0 = time.time()
    folds = _shard_folds(T, args.loo_shard)
    print(f"  folds this shard ({args.loo_shard}): {folds}", flush=True)
    for held in folds:
        pre_ix = [i for i in range(T) if i != held]
        if args.per_task_standardize:
            pretrain_ds = [
                ExperimentDataset(
                    X=Xn,
                    Y=(Y_all[i] - Y_all[i].mean()) / Y_all[i].std().clamp_min(1e-8),
                )
                for i in pre_ix
            ]
            emean = Y_all[held].mean()
            estd = Y_all[held].std().clamp_min(1e-8)
        else:
            yc = torch.cat([Y_all[i] for i in pre_ix])
            gmean, gstd = yc.mean(), yc.std().clamp_min(1e-8)
            pretrain_ds = [
                ExperimentDataset(X=Xn, Y=(Y_all[i] - gmean) / gstd) for i in pre_ix
            ]
            emean, estd = gmean, gstd
        dY = (Y_all[held] - emean) / estd
        raw = Y_all[held].squeeze(-1)

        # Candidate pool for the held-out task. "full" searches that task's OWN ~1959
        # configurations instead of the 400-config shared intersection, which is what
        # Wang et al. do. The intersection saturates within 50 iterations -- every
        # method returns the identical configuration -- so it cannot discriminate
        # (§21.2). EM scores the extra off-grid candidates through its shift
        # interpolation from the matched inducing points, which is exactly the
        # continuous-domain mechanism the paper contributes.
        if args.pd1_candidate_pool == "full" and task_pools is not None:
            Xc, Yc = task_pools[names[held]]
            dY = (Yc - emean) / estd
            raw = Yc.squeeze(-1)
            Xq = Xc
        else:
            Xq = Xn

        # Pre-training pool for the NON-EM meta-learners. Wang et al. train H-NLL /
        # FSBO / ABLR / MIMO on matched + unmatched data (~1600 points/task) and
        # restrict only H-EKL to matching inputs. EM always needs the shared grid for
        # its inducing points, so it keeps `pretrain_ds`.
        pretrain_ds_base = pretrain_ds
        if args.pd1_pretrain_pool == "full" and task_pools is not None:
            pretrain_ds_base = _pd1_full_pool_datasets(
                pre_ix, names, task_pools, args, emean, estd
            )

        # Data the CANONICAL / shrinkage-target kernel is fitted on. This is a separate
        # decision from the baselines' pre-training budget above. A shared-GP MLL has
        # no matched-input requirement, so restricting this kernel to the 400-config
        # grid is a pure handicap with no methodological justification -- and it is the
        # kernel
        # that drives the Nystrom map W = K(X,Z)K(Z,Z)^-1 carrying the empirical
        # covariance OFF-grid, exactly where arm B evaluates. The alpha screen in
        # S27.9 was run this way and found shrinkage hurting, which the handicap may
        # explain. `full` fits it on the richest pools available regardless of
        # --pd1-pretrain-pool; `inherit` keeps the historical behaviour so existing
        # results stay reproducible.
        canon_ds = pretrain_ds_base
        if args.em_canonical_pool == "full":
            if task_pools is None:
                raise ValueError(
                    "--em-canonical-pool full requires the per-task pools, which are "
                    "loaded only when --pd1-candidate-pool or --pd1-pretrain-pool is "
                    "'full' (and --pd1-source full). Refusing to fall back silently."
                )
            canon_ds = _pd1_full_pool_datasets(
                pre_ix, names, task_pools, args, emean, estd
            )

        # Re-seed per fold so --pretrain-seed resamples the priors independently of the
        # split; §19.4 showed one prior per arm cannot rank the neural meta-learners.
        torch.manual_seed(args.pretrain_seed * 100003 + held)
        em_prior = hyperbo_prior = pacoh_prior = ablr_prior = None
        # mean_baseline mirrors mean_s for the non-EM baselines. Initialised here
        # because both are assigned only inside the EM branch, and surrogate_fn
        # closes over them regardless of which methods were requested.
        mean_s = covar_s = mean_baseline = None

        # HyperBO is pre-trained FIRST, once, and the single resulting prior serves all
        # three roles that can need it: the `hyperbo_*` methods, the additive base kernel
        # (`em_additive_hyperbo*`), and the canonical/shrinkage target under
        # --em-canonical hyperbo. Previously `_pd1_canonical` trained its OWN prior, so
        # the canonical kernel and `hyperbo_frozen` came from DIFFERENT random draws --
        # confounding "different kernel role" with "different training draw" -- and it
        # paid for the most expensive component twice (S27.15).
        #
        # Hoisting it above the EM branch changes the global RNG consumption order, so
        # results are NOT comparable with runs made before this change. Every cell of a
        # comparison must be produced by this version.
        want_hyperbo = (
            any(m.startswith("hyperbo") for m in methods)
            or _needs_hyperbo_base_kernel(methods)
            or (EM_CANONICAL == "hyperbo" and any(m.startswith("em") for m in methods))
            or (
                args.em_mean != "empirical" and any(m.startswith("em") for m in methods)
            )
        )
        if want_hyperbo:
            t = time.time()
            # Pre-training is the dominant cost and is unchanged by most OFAT factors, so
            # it is cached content-addressably. The key includes the DATA digest (so a
            # forgotten flag cannot cause a false hit), the fit hyperparameters, the
            # held-out fold, and the prior seed -- the seed being part of the key is what
            # keeps repeated fits as separate retained entries instead of overwrites.
            _hb_params = {
                "input_dim": int(Xn.shape[-1]),
                "hidden_dims": str(args.hyperbo_hidden),
                "use_linear_mean": not args.hyperbo_zero_mean,
                "loss_type": args.hyperbo_loss,
                "ekl_optimizer": args.ekl_optimizer,
                "ekl_lbfgs_iterations": args.ekl_lbfgs_iters,
                "num_iterations": args.hyperbo_iters or args.meta_iters,
                "learning_rate": args.meta_lr,
                "subsample_size": args.meta_subsample,
                "task_batch_size": args.meta_task_batch,
                "pretrain_seed": args.pretrain_seed,
                "held_out": int(held),
            }
            _hb_stats: dict = {}
            hyperbo_prior = prior_cache.cached(
                args.prior_cache,
                args.prior_cache_mode,
                "hyperbo",
                pretrain_ds_base,
                _hb_params,
                # Loop variables are bound as defaults rather than captured. cached()
                # invokes this synchronously today, so late binding is not a live bug --
                # but it would become one silently the moment anything defers or
                # parallelises the call, and this file has been bitten before by code
                # that was only incidentally correct.
                lambda _ds=pretrain_ds_base, _dim=int(Xn.shape[-1]): pretrain_hyperbo(
                    datasets=_ds,
                    input_dim=_dim,
                    hidden_dims=args.hyperbo_hidden,
                    use_linear_mean=not args.hyperbo_zero_mean,
                    # The LOO pool is matched by construction, so EKL is applicable
                    # here -- this is the setting Wang et al. report H-EKL as their
                    # strongest variant in. Previously hardcoded to NLL, which
                    # silently ignored --hyperbo-loss on PD1.
                    loss_type=args.hyperbo_loss,
                    ekl_optimizer=args.ekl_optimizer,
                    ekl_lbfgs_iterations=args.ekl_lbfgs_iters,
                    num_iterations=args.hyperbo_iters or args.meta_iters,
                    learning_rate=args.meta_lr,
                    subsample_size=args.meta_subsample,
                    task_batch_size=args.meta_task_batch,
                ),
                stats=_hb_stats,
            )
            # Attribute only REAL fitting time. On a cache hit compute_s is 0.0 and the
            # hit is counted separately, so a warm cache can never be mistaken for a
            # free model (S30.1).
            ptime["hyperbo"] += _hb_stats["compute_s"]
            pcache_hits["hyperbo"] += int(_hb_stats["hit"])
            pcache_calls["hyperbo"] += 1
            # Republish the additive base every fold: leave-one-out retrains the prior
            # per held-out task, so a kernel built once would leak the first fold's
            # pre-training into all the others.
            if hyperbo_prior is not None:
                HYPERBO_BASE_KERNEL = hyperbo_kernel_from_prior(hyperbo_prior)

        if any(m.startswith("em") for m in methods):
            t = time.time()
            # The canonical kernel is fitted on `pretrain_ds_base`, which is the FULL
            # per-task data when --pd1-pretrain-pool=full. A shared-GP MLL has no
            # matching-input requirement, so restricting it to the 400-config grid was
            # an unnecessary handicap -- and it matters once the candidate pool goes
            # off-grid, because this kernel is exactly what drives the Nystrom map
            # W = K(X,Z)K(Z,Z)^-1 and the residual Lambda(X) = K(X,X) - W K(Z,X) used to
            # interpolate the empirical covariance to new inputs. Only the empirical
            # Sigma itself needs the matched grid, so pretrain_em_prior keeps
            # `pretrain_ds`.
            mean_s, covar_s = build_shared_kernel(
                canon_ds, deep_hidden=DEEP_KERNEL, base_mean=args.em_base_mean
            )
            mean_s, covar_s = _pd1_canonical(
                canon_ds, args, mean_s, covar_s, hyperbo_prior=hyperbo_prior
            )
            # See D1: baselines must not inherit the blended mean.
            mean_baseline = mean_s
            mean_s = _apply_em_mean(mean_s, args, hyperbo_prior)
            em_prior = pretrain_em_prior(
                datasets=pretrain_ds,
                mean_module=mean_s,
                covar_module=covar_s,
                likelihood_noise=torch.tensor(
                    args.em_likelihood_noise, dtype=torch.double
                ),
                covariance_shrinkage=_resolve_em_shrinkage(args, pretrain_ds, covar_s),
                num_em_iterations=args.n_em,
                enable_interpolation=True,
                iw_nu=_resolve_iw_nu(args, pretrain_ds, covar_s),
                use_mean_prior=args.use_mean_prior,
                use_covar_prior=args.use_covar_prior,
                init_mode=args.em_init_mode,
            )
            ptime["em"] += time.time() - t
        if any(m.startswith("pacoh") for m in methods):
            t = time.time()
            pacoh_prior = pretrain_pacoh_gp(
                datasets=pretrain_ds_base,
                input_dim=Xn.shape[-1],
                config=PACOHGPConfig(
                    num_particles=args.pacoh_particles,
                    hidden_dims=args.pacoh_hidden or args.hyperbo_hidden,
                    hyper_prior_std=args.pacoh_hyper_prior_std,
                    svgd_length_scale=args.pacoh_svgd_lengthscale,
                ),
                num_iterations=args.pacoh_iters or args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
                task_batch_size=args.meta_task_batch,
            )
            ptime["pacoh"] += time.time() - t
        if any(m.startswith("ablr") for m in methods):
            t = time.time()
            ablr_prior = pretrain_ablr(
                datasets=pretrain_ds_base,
                input_dim=Xn.shape[-1],
                hidden=args.ablr_hidden or tuple(args.hyperbo_hidden),
                feat_dim=args.ablr_feat_dim,
                num_iterations=args.ablr_iters or args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
                task_batch_size=args.meta_task_batch,
                per_task_precision=args.ablr_per_task_precision,
            )
            ptime["ablr"] += time.time() - t

        def surrogate_fn(method, X, Y):
            return make_surrogate(
                method,
                X,
                Y,
                em_prior,
                mean_s,
                covar_s,
                hyperbo_prior,
                pacoh_prior,
                ablr_prior,
                mean_baseline=mean_baseline,
            )

        _set_cur_ds(names[held])
        for seed in range(args.n_seeds):
            # The initial design must vary with the PRIOR, not only with the seed.
            # Otherwise at --n-seeds 1 every prior shares one initial design, the
            # per-prior noise e_p becomes a single shared e, and the between-prior
            # spread estimates only sigma_B^2 -- understating Var by sigma_W^2,
            # which S27.24 measured as the LARGER component for most methods. The
            # replication axis has to be independent to be worth anything.
            _bo_seed = 1000 * held + seed + 100003 * args.pretrain_seed
            g = torch.Generator().manual_seed(_bo_seed)
            init_idx = torch.randperm(Xq.shape[0], generator=g)[:n_init].tolist()
            pool_max_raw.append(raw.max().item())
            run_ds.append(held)
            for m in methods:
                gm = torch.Generator().manual_seed(7 * _bo_seed + 13)
                tb = time.time()
                traj = run_bo(
                    m,
                    Xq,
                    dY,
                    init_idx,
                    args.n_iters,
                    gm,
                    surrogate_fn,
                    acquisition=args.acquisition,
                    ucb_beta=args.ucb_beta,
                    obs_noise=args.obs_noise,
                    batch_q=args.batch_q,
                    # Forwarded on the PD1 paths too: only main() passed these, so
                    # --batch-mode fantasy --n-fantasies N silently ran plain top-q on
                    # PD1 and reported it as a fantasy q-batch (silent no-op #9, N1).
                    batch_mode=args.batch_mode,
                    n_fantasies=args.n_fantasies,
                    calib_sink=calib_records[m],
                )
                method_bo_s[m] += time.time() - tb
                results[m].append([v * estd.item() + emean.item() for v in traj])
        print(
            f"  fold {held + 1}/{T} ({names[held].split(',')[0]}) | "
            f"{time.time() - _t0:.0f}s elapsed",
            flush=True,
        )

    pool_max_mean = sum(pool_max_raw) / len(pool_max_raw)
    summary = {}
    for m in methods:
        Tt = torch.tensor(results[m])
        pm = torch.tensor(pool_max_raw).unsqueeze(1)
        ra = pm - Tt
        summary[m] = {
            "mean_best": Tt.mean(0).tolist(),
            "mean_regret": ra.mean(0).tolist(),
            "regret_sem": (ra.std(0) / (Tt.shape[0] ** 0.5)).tolist(),
        }
    per_run = {
        "eval_dataset_idx": run_ds,
        "pool_max_raw": pool_max_raw,
        "final_regret": {
            m: [pool_max_raw[r] - results[m][r][-1] for r in range(len(pool_max_raw))]
            for m in methods
        },
        "traj_raw": {m: results[m] for m in methods},
    }
    calibration = {}
    for m in methods:
        recs = calib_records[m]
        if recs:
            nl = [r[0] for r in recs]
            cv = [r[1] for r in recs]
            calibration[m] = {
                "mean_nll": sum(nl) / len(nl),
                "coverage95": sum(cv) / len(cv),
                "n_points": len(recs),
            }
    out = {
        "config": vars(args),
        "eval_datasets": names,
        "n_runs": len(pool_max_raw),
        "pool_max_mean": pool_max_mean,
        "init_evals": n_init,
        "summary": summary,
        "per_run": per_run,
        "calibration": calibration,
        "timings": {
            "em_pretrain_s": ptime["em"],
            "hyperbo_pretrain_s": ptime["hyperbo"],
            "pacoh_pretrain_s": ptime["pacoh"],
            "ablr_pretrain_s": ptime["ablr"],
            # A pre-training time of 0.0 is ambiguous on its own -- it means either
            # "not requested" or "served from the prior cache". These disambiguate it,
            # so a cost analysis can refuse cache-served cells instead of averaging
            # their zeros into a conclusion that pre-training is free (S30.1).
            "prior_cache_hits": dict(pcache_hits),
            "prior_cache_calls": dict(pcache_calls),
            "meta_iters": args.meta_iters,
            "method_bo_s": method_bo_s,
            "n_runs": len(pool_max_raw),
        },
    }
    atomic_write_json(args.out, out)
    print(
        f"\n=== PD1-LOO (bs={group}) done: {T} folds ===\nWrote {args.out}", flush=True
    )


def run_pd1_full(args, methods) -> None:
    """Full (unmatched) PD1: each task keeps its OWN config pool (no shared-input
    intersection), so matched-input-free transfer methods (HyperBO / PACOH / ABLR /
    vanilla / random) use PD1's real design space. Disentangles 'PD1 is a hard transfer
    problem' from 'the 50-config matched intersection crippled it'. EM / pretrained-GP /
    warm-start REQUIRE a shared pool and are skipped here.
    """
    load_pd1_datasets = _import_pd1_lite().load_pd1_datasets

    allowed = {
        "em_frozen",
        "em_noisefit",
        "hyperbo_frozen",
        "hyperbo_adapt",
        "pacoh_frozen",
        "pacoh_adapt",
        "ablr",
        "vanilla_gp",
        "random",
    }
    skipped = [m for m in methods if m not in allowed]
    methods = [m for m in methods if m in allowed]
    if skipped:
        print(f"pd1_full: skipping unsupported methods {skipped}", flush=True)
    if not methods:
        raise ValueError("pd1_full: no matched-input-free methods requested")

    print("Loading full (unmatched) PD1 datasets ...", flush=True)
    ds_raw, names = load_pd1_datasets(
        normalize=True,
        max_trials_per_task=args.n_configs,
        dtype=torch.double,
        reduction=("best" if args.pd1_best_metric else "last"),
    )
    # objective = -error_rate (higher = better), matching the matched-PD1 path
    datasets = [ExperimentDataset(X=r.X, Y=-r.Y) for r in ds_raw]
    if args.output_warp != "none":
        # Per-task output transform (tames PD1's bimodal divergence tail). Monotonic for
        # gaussian/winsor, so regret/ranking still reflect finding the best config;
        # calibration is then measured in the transformed space.
        _wY = _apply_output_warp([dd.Y for dd in datasets], args.output_warp)
        datasets = [ExperimentDataset(X=dd.X, Y=y) for dd, y in zip(datasets, _wY)]
    input_dim = datasets[0].X.shape[-1]
    sizes = [x.X.shape[0] for x in datasets]
    print(
        f"PD1-full: {len(names)} tasks, d={input_dim}, pool sizes "
        f"{min(sizes)}..{max(sizes)}",
        flush=True,
    )
    for i, nm in enumerate(names):
        y = -datasets[i].Y.squeeze(-1)  # back to raw error_rate
        q = torch.quantile(y, torch.tensor([0.0, 0.5, 1.0], dtype=y.dtype))
        print(
            f"  task {i:2d} {nm.split(',')[0]:16s} err min/med/max="
            f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} std={y.std():.3f} "
            f"frac_err>0.5={(y > 0.5).float().mean():.2f}",
            flush=True,
        )

    torch.manual_seed(args.split_seed)
    dperm = torch.randperm(len(names))
    eval_ix = dperm[: args.n_eval].tolist()
    pre_ix = dperm[args.n_eval : args.n_eval + args.n_pretrain].tolist()

    def _standardize(Yt):
        # Per-task standardization (paper-faithful) when --per-task-standardize, else a
        # single global scale from the pretrain pool. Heterogeneous PD1 tasks land on very
        # different scales under global scaling -> miscalibrated transferred posteriors.
        if args.per_task_standardize:
            m, s = Yt.mean(), Yt.std().clamp_min(1e-8)
        else:
            m, s = gmean, gstd
        return (Yt - m) / s, m, s

    Ypool = torch.cat([datasets[i].Y for i in pre_ix])
    gmean, gstd = Ypool.mean(), Ypool.std().clamp_min(1e-8)
    pretrain_ds = [
        ExperimentDataset(X=datasets[i].X, Y=_standardize(datasets[i].Y)[0])
        for i in pre_ix
    ]
    print(f"Pretrain on {len(pre_ix)} tasks; evaluate on {len(eval_ix)}", flush=True)

    hyperbo_prior = pacoh_prior = ablr_prior = None
    hyperbo_pretrain_s = pacoh_pretrain_s = ablr_pretrain_s = 0.0
    # The additive base, the canonical target and the mean transfer all need this prior,
    # so gating only on hyperbo_* methods left them raising or silently unset (D3/D11).
    if (
        any(m.startswith("hyperbo") for m in methods)
        or _needs_hyperbo_base_kernel(methods)
        or (EM_CANONICAL == "hyperbo" and any(m.startswith("em") for m in methods))
        or (args.em_mean != "empirical" and any(m.startswith("em") for m in methods))
    ):
        t1 = time.time()
        print("Pre-training HyperBO (unmatched tasks) ...", flush=True)
        hyperbo_prior = pretrain_hyperbo(
            datasets=pretrain_ds,
            input_dim=input_dim,
            hidden_dims=args.hyperbo_hidden,
            use_linear_mean=not args.hyperbo_zero_mean,
            # Forced to NLL: pd1_full gives every task its own pool, so inputs are not
            # matched and the EKL objective does not apply.
            loss_type="NLL",
            num_iterations=args.hyperbo_iters or args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
        )
        hyperbo_pretrain_s = time.time() - t1
    if any(m.startswith("pacoh") for m in methods):
        t1 = time.time()
        print("Pre-training PACOH (unmatched tasks) ...", flush=True)
        pacoh_prior = pretrain_pacoh_gp(
            datasets=pretrain_ds,
            input_dim=input_dim,
            config=PACOHGPConfig(
                num_particles=args.pacoh_particles,
                hidden_dims=args.pacoh_hidden or args.hyperbo_hidden,
                hyper_prior_std=args.pacoh_hyper_prior_std,
                svgd_length_scale=args.pacoh_svgd_lengthscale,
            ),
            num_iterations=args.pacoh_iters or args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
        )
        pacoh_pretrain_s = time.time() - t1
    if any(m.startswith("ablr") for m in methods):
        t1 = time.time()
        print("Pre-training ABLR (unmatched tasks) ...", flush=True)
        ablr_prior = pretrain_ablr(
            datasets=pretrain_ds,
            input_dim=input_dim,
            hidden=args.ablr_hidden or tuple(args.hyperbo_hidden),
            feat_dim=args.ablr_feat_dim,
            num_iterations=args.ablr_iters or args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
            per_task_precision=args.ablr_per_task_precision,
        )
        ablr_pretrain_s = time.time() - t1

    em_prior = None
    mean_s = covar_s = mean_baseline = None
    em_pretrain_s = 0.0
    if any(m.startswith("em") for m in methods):
        # #3: EM needs matched inputs to *pretrain*, but its surrogate predicts off-grid
        # (verified), so pretrain the empirical prior on the matched-config subset of the
        # pretrain tasks and let it evaluate on the FULL per-task eval pools.
        t1 = time.time()
        print(
            "Pre-training EM on the matched-config subset of pretrain tasks ...",
            flush=True,
        )
        pre_names = [names[i] for i in pre_ix]
        Xn_m, Y_all_m, used_m = load_pd1_pool(dataset_names=pre_names)
        if args.output_warp != "none":
            Y_all_m = _apply_output_warp(Y_all_m, args.output_warp)
        em_pre_ds = [ExperimentDataset(X=Xn_m, Y=_standardize(Y)[0]) for Y in Y_all_m]
        # Parity with run_pd1_matched_loo and main(): without these, --em-base-mean,
        # --deep-kernel, --em-canonical, --em-mean, --iw-nu, --use-mean-prior,
        # --use-covar-prior and --em-init-mode were all accepted and silently did
        # nothing on --benchmark pd1_full (D3, silent no-op #8).
        mean_s, covar_s = build_shared_kernel(
            em_pre_ds, deep_hidden=DEEP_KERNEL, base_mean=args.em_base_mean
        )
        mean_s, covar_s = _pd1_canonical(
            em_pre_ds, args, mean_s, covar_s, hyperbo_prior=hyperbo_prior
        )
        mean_baseline = mean_s
        mean_s = _apply_em_mean(mean_s, args, hyperbo_prior)
        em_prior = pretrain_em_prior(
            datasets=em_pre_ds,
            mean_module=mean_s,
            covar_module=covar_s,
            likelihood_noise=torch.tensor(args.em_likelihood_noise, dtype=torch.double),
            covariance_shrinkage=_resolve_em_shrinkage(args, em_pre_ds, covar_s),
            num_em_iterations=args.n_em,
            enable_interpolation=True,
            iw_nu=_resolve_iw_nu(args, em_pre_ds, covar_s),
            use_mean_prior=args.use_mean_prior,
            use_covar_prior=args.use_covar_prior,
            init_mode=args.em_init_mode,
        )
        em_pretrain_s = time.time() - t1
        print(
            f"  EM matched-pretrain: {Xn_m.shape[0]} configs, {len(used_m)} tasks, "
            f"{em_pretrain_s:.1f}s (predicts on full eval pools off-grid)",
            flush=True,
        )

    def surrogate_fn(method, X, Y):
        return make_surrogate(
            method,
            X,
            Y,
            em_prior,
            mean_s,
            covar_s,
            hyperbo_prior,
            pacoh_prior,
            ablr_prior,
            mean_baseline=mean_baseline,
        )

    results = {m: [] for m in methods}
    calib_records = {m: [] for m in methods}
    run_ds: list[int] = []
    pool_max_raw: list[float] = []
    method_bo_s = {m: 0.0 for m in methods}
    _t_start = time.time()
    _n_total = len(eval_ix) * args.n_seeds
    _run_i = 0
    for ei in eval_ix:
        _set_cur_ds(names[ei])
        Xi = datasets[ei].X
        dY, eymean, eystd = _standardize(datasets[ei].Y)
        raw = datasets[ei].Y.squeeze(-1)
        n_conf = Xi.shape[0]
        n_init = min(args.n_init, max(1, n_conf - 1))
        for seed in range(args.n_seeds):
            # The initial design must vary with the PRIOR, not only with the seed.
            # Otherwise at --n-seeds 1 every prior shares one initial design, the
            # per-prior noise e_p becomes a single shared e, and the between-prior
            # spread estimates only sigma_B^2 -- understating Var by sigma_W^2,
            # which S27.24 measured as the LARGER component for most methods. The
            # replication axis has to be independent to be worth anything.
            _bo_seed = 1000 * ei + seed + 100003 * args.pretrain_seed
            g = torch.Generator().manual_seed(_bo_seed)
            init_idx = torch.randperm(n_conf, generator=g)[:n_init].tolist()
            pool_max_raw.append(raw.max().item())
            run_ds.append(int(ei))
            for m in methods:
                gm = torch.Generator().manual_seed(7 * _bo_seed + 13)
                _t = time.time()
                traj_std = run_bo(
                    m,
                    Xi,
                    dY,
                    init_idx,
                    args.n_iters,
                    gm,
                    surrogate_fn,
                    acquisition=args.acquisition,
                    ucb_beta=args.ucb_beta,
                    obs_noise=args.obs_noise,
                    batch_q=args.batch_q,
                    # Forwarded on the PD1 paths too: only main() passed these, so
                    # --batch-mode fantasy --n-fantasies N silently ran plain top-q on
                    # PD1 and reported it as a fantasy q-batch (silent no-op #9, N1).
                    batch_mode=args.batch_mode,
                    n_fantasies=args.n_fantasies,
                    calib_sink=calib_records[m],
                )
                method_bo_s[m] += time.time() - _t
                results[m].append([v * eystd.item() + eymean.item() for v in traj_std])
            _run_i += 1
            print(
                f"    progress {_run_i}/{_n_total} ({names[ei]} seed {seed}) | "
                f"{time.time() - _t_start:.0f}s elapsed",
                flush=True,
            )

    pool_max_mean = sum(pool_max_raw) / len(pool_max_raw)
    summary = {}
    for m in methods:
        T = torch.tensor(results[m])
        pm = torch.tensor(pool_max_raw).unsqueeze(1)
        regret_all = pm - T
        summary[m] = {
            "mean_best": T.mean(0).tolist(),
            "mean_regret": regret_all.mean(0).tolist(),
            "regret_sem": (regret_all.std(0) / (T.shape[0] ** 0.5)).tolist(),
        }
    per_run = {
        "eval_dataset_idx": run_ds,
        "pool_max_raw": pool_max_raw,
        "final_regret": {
            m: [pool_max_raw[r] - results[m][r][-1] for r in range(len(pool_max_raw))]
            for m in methods
        },
        "traj_raw": {m: results[m] for m in methods},
    }
    calibration = {}
    for m in methods:
        recs = calib_records[m]
        if recs:
            nl = [r[0] for r in recs]
            cv = [r[1] for r in recs]
            calibration[m] = {
                "mean_nll": sum(nl) / len(nl),
                "coverage95": sum(cv) / len(cv),
                "n_points": len(recs),
            }
    out = {
        "config": vars(args),
        "eval_datasets": [names[i] for i in eval_ix],
        "n_runs": len(pool_max_raw),
        "pool_max_mean": pool_max_mean,
        "init_evals": args.n_init,
        "summary": summary,
        "per_run": per_run,
        "calibration": calibration,
        "timings": {
            "em_pretrain_s": em_pretrain_s,
            "hyperbo_pretrain_s": hyperbo_pretrain_s,
            "pacoh_pretrain_s": pacoh_pretrain_s,
            "ablr_pretrain_s": ablr_pretrain_s,
            "meta_iters": args.meta_iters,
            "method_bo_s": method_bo_s,
            "n_runs": len(pool_max_raw),
        },
    }
    atomic_write_json(args.out, out)
    print("\n=== BO on PD1-FULL (per-task unmatched pools) ===", flush=True)
    print(f"Wrote {args.out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Split out of main() so tests can validate flag
    strings (notably EXPERIMENT_PLAN's REGIMES table) without executing anything."""
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-configs", type=int, default=200)
    p.add_argument("--n-iters", type=int, default=40)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--n-em", type=int, default=50)
    p.add_argument("--n-pretrain", type=int, default=25)
    # Meta-learner (HyperBO / PACOH) knobs. Defaults follow the papers' primary
    # settings: a (32, 32) tanh MLP deep kernel, NLL pre-training, K=10 SVGD
    # particles. These are the "strongest faithful" dials to sweep.
    p.add_argument(
        "--hyperbo-hidden",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(32, 32),
        help="comma-separated MLP hidden dims for HyperBO/PACOH deep kernel",
    )
    p.add_argument("--meta-iters", type=int, default=2000)
    p.add_argument("--meta-lr", type=float, default=1e-3)
    p.add_argument("--meta-subsample", type=int, default=None)
    p.add_argument(
        "--meta-task-batch",
        type=int,
        default=None,
        help="tasks sampled per meta-training step (stochastic over tasks; "
        "unbiased-rescaled). None = full batch.",
    )
    p.add_argument("--pacoh-particles", type=int, default=10)
    p.add_argument("--pacoh-hyper-prior-std", type=float, default=1.0)
    p.add_argument("--pacoh-svgd-lengthscale", type=float, default=None)
    p.add_argument(
        "--pacoh-hidden",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=None,
        help="MLP hidden dims for the PACOH deep kernel; defaults to --hyperbo-hidden. "
        "Smaller widths lower the SVGD particle dimension (capacity ablation).",
    )
    p.add_argument(
        "--hyperbo-loss",
        type=str,
        default="NLL",
        choices=["NLL", "EKL"],
        help="HyperBO pre-training objective. EKL is valid in 7D too because the "
        "config pool is shared across datasets (matched inputs).",
    )
    p.add_argument(
        "--acquisition",
        type=str,
        default="logei",
        choices=["logei", "greedy", "ucb"],
        help="Finite-pool acquisition: logei (balanced), greedy (mean-only, isolates "
        "prior-mean quality), ucb (mean + beta*sigma, stresses uncertainty).",
    )
    p.add_argument("--ucb-beta", type=float, default=2.0)
    p.add_argument(
        "--obs-noise",
        type=float,
        default=0.0,
        help="Std (standardized units) of Gaussian observation noise added to what the "
        "surrogate conditions on; regret is still measured on the true objective.",
    )
    p.add_argument(
        "--batch-q",
        type=int,
        default=1,
        help="Batch size per acquisition round. With q>1 the trajectory is indexed by "
        "round (each round = q evaluations).",
    )
    p.add_argument(
        "--batch-mode",
        type=str,
        default="topq",
        choices=["topq", "kb", "fantasy"],
        help="How a q>1 batch is formed: topq (top-q by acquisition, no fantasies); "
        "kb (sequential greedy, kriging believer -- fantasy pinned at the posterior "
        "mean); fantasy (sequential greedy with --n-fantasies MC draws, the proper "
        "qEI/qLogEI construction).",
    )
    p.add_argument(
        "--n-fantasies",
        type=int,
        default=8,
        help="Monte-Carlo fantasy draws per selected point for --batch-mode fantasy.",
    )
    p.add_argument(
        "--novel-config-split",
        action="store_true",
        help="Use DISJOINT config pools: pre-train the prior on one set of configs and "
        "run BO on a held-out set (tests transfer of the empirical mean to unseen "
        "configs). Requires 2*n_configs <= pool size.",
    )
    p.add_argument(
        "--iw-nu-mode",
        type=str,
        default="manual",
        choices=["manual", "oas", "ledoit_wolf", "oas_whitened"],
        help="How to choose the Inverse-Wishart degrees of freedom. 'manual' uses "
        "--iw-nu as given (the historical behaviour, and the default). 'oas' and "
        "'ledoit_wolf' instead ESTIMATE it from the pre-training tasks: the IW posterior "
        "mean is linear shrinkage with intensity alpha = (nu-K-1)/(T+nu-K-1), so picking "
        "alpha analytically and inverting is empirical Bayes for nu. S27.46 measured OAS "
        "as the better of the two at T = 6-25, which is our regime. 'oas_whitened' fixes "
        "a mismatch in the other two: Psi = (nu+N+1)*K(Z,Z) under the default "
        "init_mode='kernel', so the IW target is the parametric KERNEL, but oas_alpha is "
        "derived for a SPHERICAL target -- intensity chosen for one target, applied to "
        "another. Whitening by K puts the estimator in coordinates where the target is "
        "the identity (S30.7). Only has an effect with --use-covar-prior.",
    )
    p.add_argument(
        "--em-shrinkage-mode",
        type=str,
        default="manual",
        choices=["manual", "oas", "ledoit_wolf", "oas_whitened"],
        help="How to choose the EM M-step shrinkage intensity alpha. 'manual' uses "
        "--em-shrinkage as given (the default, and bit-identical to the historical "
        "behaviour). The others ESTIMATE alpha analytically from the pre-training tasks "
        "and use it DIRECTLY -- unlike --iw-nu-mode, which inverted alpha into an "
        "Inverse-Wishart nu that S31 showed cannot express it: Psi is scaled by nu, so "
        "the realised intensity is floored at 0.970 regardless. 'oas_whitened' measures "
        "alpha against the kernel target actually in force (S30.7), reusing "
        "--iw-target-ridge.",
    )
    p.add_argument(
        "--iw-target-ridge",
        type=float,
        default=0.05,
        help="Ridge tau for --iw-nu-mode oas_whitened: the target is "
        "(1-tau)*K_hat + tau*I on a unit-mean-diagonal K_hat. NOT optional in practice. "
        "A Gram matrix has cond ~1e7, so raw K^(-1/2) has eigenvalues ~1e3 and the "
        "whitened sample is dominated by directions carrying no signal; S30.7 measured "
        "pure whitening (tau=0) at +2947%% Frobenius error against spherical, and "
        "tau=0.05 at -21.2%%. tau=1 recovers plain OAS exactly.",
    )
    p.add_argument(
        "--em-likelihood-noise",
        type=float,
        default=1e-2,
        help="Noise VARIANCE used in EM pre-training's E-step, which computes each task's "
        "posterior at the inducing points as K(K+sigma^2 I)^-1 y. Was hardcoded at 1e-2 "
        "with no way to change it and never fitted; S27.35 scanned it and found the "
        "optimum near sigma^2 = 1-3, i.e. 2-3 orders of magnitude higher, worth +23.5 "
        "percent rank correlation and -78 percent RMSE at n=5. On unit-variance standardized data 1e-2 "
        "means sigma=0.1. Default kept at 1e-2 so existing results reproduce exactly; "
        "raise it deliberately. NOTE S27.35's caveat: a large value may be switching the "
        "empirical Sigma off rather than improving it -- check em_frozen against "
        "pretrained_gp_frozen before concluding otherwise.",
    )
    p.add_argument(
        "--prior-cache",
        type=str,
        default=None,
        help="Directory for content-addressed pre-trained priors. Pre-training dominates "
        "cost (~61 min of HyperBO per LCBench cell at 25000 iters) and most OFAT factors "
        "do not change it. The key derives from the DATA plus the fit hyperparameters and "
        "includes --pretrain-seed, so repeated fits under different seeds are RETAINED "
        "SEPARATELY rather than overwriting -- which makes the variance of the trained "
        "model itself measurable, and lets BO loops be re-run post hoc against a frozen "
        "prior. A miss is always correct, so the cache is safe to delete.",
    )
    p.add_argument(
        "--prior-cache-mode",
        type=str,
        default="readwrite",
        choices=["off", "read", "write", "readwrite"],
        help="`read` reproduces a previous run without polluting the cache; `write` "
        "always refits and overwrites; `off` disables caching entirely.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="em_frozen,random",
        help=(
            "comma-separated subset of em_frozen,em_finetuned,"
            "pretrained_gp_frozen,pretrained_gp_tuned,vanilla_gp,"
            "hyperbo_frozen,hyperbo_adapt,pacoh_frozen,pacoh_adapt,random"
        ),
    )
    p.add_argument("--ood-split", action="store_true")
    p.add_argument(
        "--benchmark",
        type=str,
        default="lcbench",
        choices=["lcbench", "pd1", "pd1_full", "pd1_probe", "pd1_loo"],
        help="Benchmark suite: lcbench (7D, 2000 shared configs); pd1 (HyperBO's own "
        "benchmark, matched-config subset); pd1_full (PD1 per-task unmatched pools "
        "for matched-input-free methods only).",
    )
    p.add_argument(
        "--per-task-standardize",
        action="store_true",
        help="Standardize each task's objective by ITS OWN mean/std (paper-faithful for "
        "heterogeneous suites like PD1) instead of one global scale from the pretrain "
        "pool. Global scaling puts heterogeneous tasks on very different scales and "
        "miscalibrates the transferred posteriors.",
    )
    p.add_argument(
        "--pd1-best-metric",
        action="store_true",
        help="PD1 objective = BEST (min) validation error over the training curve (what "
        "the HyperBO paper optimizes) instead of the last-epoch error (which is bimodal: "
        "late-diverging trials get a ~0.9 error and wreck GP calibration).",
    )
    p.add_argument(
        "--pd1-group",
        type=str,
        default="256",
        help="Batch-size subgroup for pd1_loo (matched-config intersection within a "
        "homogeneous subgroup is much larger than across all 24 tasks; bs=256 gives "
        "~73 configs across 10 tasks).",
    )
    p.add_argument(
        "--output-warp",
        type=str,
        default="none",
        choices=["none", "gaussian", "winsor", "neglog"],
        help="Per-task output transform before fitting (pd1_full only): none (plain "
        "standardization) or gaussian (rank/copula warp to N(0,1), taming heavy-tailed "
        "objectives like PD1's divergence tail -- the HyperBO-style transform).",
    )
    p.add_argument(
        "--em-shrinkage",
        type=float,
        default=0.0,
        help="EM M-step covariance shrinkage alpha: Sigma <- (1-a)*Sigma_ML + a*s*B, a "
        "trace-matched blend toward the base-kernel gram. The EM covariance is rank "
        "<=(K-1) from K pre-training tasks; on a shared on-grid pool the Nystrom "
        "residual vanishes, so without shrinkage the remaining directions carry only "
        "--cond-noise of variance. Computed at pre-training time from the historical "
        "corpus, so unlike the conditioning-time additive base it needs no target data.",
    )
    p.add_argument(
        "--cond-noise",
        type=float,
        default=COND_NOISE,
        help="Frozen conditioning likelihood noise (VARIANCE units -- "
        "GaussianLikelihood.noise is a variance, so 1e-3 means sigma~0.032) for the "
        "frozen-prior "
        "surrogates. An isotropic floor; contrast with --em-shrinkage, which adds "
        "structured off-span variance.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the config subsample and the pretrain/eval dataset split. Change "
        "it to draw a dev split disjoint from the reported test datasets.",
    )
    p.add_argument(
        "--pd1-source",
        type=str,
        default="full",
        choices=["lite", "full"],
        help="PD1 data source. full (DEFAULT) = the full public release (400 matched "
        "configs across 23 tasks, CC-BY 4.0, Wang et al. 2021) read from --pd1-root; "
        "lite = the reduced PD1Lite internal object storage mirror (~50 matched configs), which is "
        "Meta-internal and unavailable outside Meta. The default was `lite` until "
        "2026-09-08; it was changed because every committed sweep script passes "
        "`--pd1-source full` explicitly and S12.5 showed the reduced pool -- not the "
        "method -- was the binding constraint on reproducing the published PD1 result. "
        "See PD1_DATA.md.",
    )
    p.add_argument(
        "--pd1-candidate-pool",
        type=str,
        default="matched",
        choices=["matched", "full"],
        help='BO candidate set for the held-out PD1 task. "matched" is the 400-config '
        "shared intersection, which saturates within 50 iterations so every method "
        'returns the same configuration; "full" searches that task\'s own ~1959 '
        "configurations, which is what Wang et al. do. EM reaches the off-grid "
        "candidates via shift interpolation from the matched inducing points.",
    )
    p.add_argument(
        "--pd1-pretrain-pool",
        type=str,
        default="matched",
        choices=["matched", "full"],
        help="Pre-training data for the NON-EM meta-learners. Wang et al. train H-NLL / "
        "FSBO / ABLR / MIMO on matched + unmatched data (~1600 points/task) and restrict "
        "only H-EKL to matching inputs; we had been giving them 400. EM always keeps the "
        "matched grid, which its inducing points require.",
    )
    p.add_argument(
        "--em-base-mean",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="Parametric base mean m(.) fitted by the shared SumMLL. EM interpolates as "
        "mu(X) = m(X) + W @ delta_mu, so with a ConstantMean the extrapolated mean is "
        "FLAT away from the inducing grid -- a plausible handicap off-grid. Only "
        "'constant' was ever used before 2026-08-17. Orthogonal to --em-mean, which "
        "blends this base mean toward HyperBO's learned one.",
    )
    p.add_argument(
        "--iw-nu",
        type=float,
        default=None,
        help="Inverse-Wishart prior degrees of freedom for the EM M-step. A PRINCIPLED "
        "alternative to --em-shrinkage's ad-hoc trace-matched blend, which S27.9 found "
        "does not help on PD1. Never explored.",
    )
    p.add_argument(
        "--use-mean-prior",
        action="store_true",
        help="Enable EM's prior on the mean. Never explored.",
    )
    p.add_argument(
        "--use-covar-prior",
        action="store_true",
        help="Enable EM's prior on the covariance. Never explored.",
    )
    p.add_argument(
        "--em-init-mode",
        type=str,
        default="kernel",
        choices=["kernel", "naive"],
        help="EM initialization. 'kernel' seeds from the canonical kernel (the only "
        "mode ever run); 'naive' does not. Never explored.",
    )
    p.add_argument(
        "--em-mean",
        type=str,
        default="empirical",
        choices=["empirical", "hyperbo", "blend"],
        help="EM's parametric base mean m(.), which it interpolates off-grid as "
        "mu(X) = m(X) + W @ delta_mu. Every em_additive_* variant to date transferred "
        "HyperBO's COVARIANCE only and left m(.) as EM's ConstantMean; S27.15 argues "
        "that is why the hybrid wins on-grid but falls short off-grid. 'hyperbo' "
        "replaces m(.) with HyperBO's learned linear mean, 'blend' mixes them by "
        "--em-mean-weight (replacement is the degenerate beta=1 case). Raises rather "
        "than falling back if no HyperBO prior is available.",
    )
    p.add_argument(
        "--em-mean-weight",
        type=float,
        default=0.5,
        help="beta for --em-mean blend: (1-beta)*m_em + beta*m_hyperbo.",
    )
    p.add_argument(
        "--em-canonical-pool",
        type=str,
        default="inherit",
        choices=["inherit", "full"],
        help="Data the EM canonical / shrinkage-target kernel is fitted on. This is a "
        "SEPARATE decision from --pd1-pretrain-pool, which sets the baselines' data "
        "budget. A shared-GP MLL has no matched-input requirement, so restricting this "
        "kernel to the 400-config matched grid is a pure handicap: it is the kernel "
        "driving the Nystrom map that carries the empirical covariance OFF-grid. "
        "'full' fits it on the richest per-task pools available regardless of "
        "--pd1-pretrain-pool; 'inherit' (default) keeps the historical behaviour so "
        "existing results remain reproducible. Raises rather than falling back if the "
        "full pools are not loaded.",
    )
    p.add_argument(
        "--pd1-root",
        type=str,
        default="",
        help="Directory holding the extracted full PD1 release; defaults to "
        "pd1_full_data.DEFAULT_PD1_ROOT.",
    )
    p.add_argument(
        "--hyperbo-iters",
        type=int,
        default=0,
        help="HyperBO pre-training steps; 0 = use --meta-iters. Decoupled per method so "
        "each baseline can be given its own converged budget: a shared --meta-iters "
        "forces every meta-learner onto whichever budget the slowest one can afford, "
        "which under-trains the others. The PD1 paper uses 50000 Adam steps.",
    )
    p.add_argument(
        "--pacoh-iters", type=int, default=0, help="PACOH steps; 0 = --meta-iters."
    )
    p.add_argument(
        "--ablr-iters", type=int, default=0, help="ABLR steps; 0 = --meta-iters."
    )
    p.add_argument(
        "--em-canonical",
        type=str,
        default="sumll",
        choices=["sumll", "hyperbo"],
        help="Canonical/interpolation kernel for the EM prior. 'sumll' (default) "
        "fits an ARD Matern by summed marginal likelihood. 'hyperbo' reuses "
        "HyperBO's PRE-TRAINED deep kernel, which requires pre-training HyperBO "
        "first and so costs its full budget even if HyperBO is not evaluated.",
    )
    p.add_argument(
        "--deep-kernel",
        type=lambda s: tuple(int(x) for x in s.split(",")) if s else None,
        default=None,
        help="Hidden dims for a learned MLP embedding under the canonical "
        "kernel, e.g. 32,32. Default keeps the plain ARD Matern.",
    )
    p.add_argument(
        "--ekl-optimizer",
        type=str,
        default="lbfgs",
        choices=["adam", "lbfgs"],
        help="Optimizer for HyperBO's EKL objective. Wang et al. (2021) use "
        "L-BFGS for EKL and reserve Adam for NLL, noting the low-rank Matern "
        "NLL problem that motivated Adam does not arise for EKL. Optimizing EKL "
        "with Adam under-trains it, which is why our earlier 7D EKL ablation "
        "found EKL worse than NLL.",
    )
    p.add_argument("--ekl-lbfgs-iters", type=int, default=100)
    p.add_argument(
        "--pretrain-seed",
        type=int,
        default=0,
        help="Seed for meta-learner pre-training, independent of --split-seed. "
        "One prior is shared by every BO run, so comparisons between neural "
        "meta-learners have an effective sample size of ONE prior; vary this to "
        "measure prior-draw variance.",
    )
    p.add_argument(
        "--hyperbo-zero-mean",
        action="store_true",
        help="Use a zero prior mean instead of the learned linear mean W.phi(x), "
        "which turns HyperBO into FSBO (Wistuba & Grabocka 2021). Wang et al. "
        "report H-NLL beating FSBO precisely because the learned mean helps in "
        "early BO, so reproducing that ordering is a soundness check on our "
        "HyperBO: if the zero-mean variant WINS here, our mean is mis-trained.",
    )
    p.add_argument(
        "--ablr-feat-dim",
        type=int,
        default=50,
        help="Dimension of the ABLR basis, i.e. the width of the Bayesian linear "
        "head. Never previously swept.",
    )
    p.add_argument(
        "--ablr-hidden",
        type=lambda s: tuple(int(x) for x in s.split(",")) if s else None,
        default=None,
        help="MLP hidden dims for the ABLR feature map; defaults to --hyperbo-hidden.",
    )
    p.add_argument(
        "--ablr-per-task-precision",
        action="store_true",
        help="Give each pre-training task its own (alpha, beta) precisions, as "
        "Perrone et al. (2018) prescribe, instead of sharing one pair across "
        "heterogeneous tasks.",
    )
    p.add_argument(
        "--loo-shard",
        type=str,
        default="0/1",
        help="Run shard i of n of a leave-one-out sweep (round-robin over folds), so "
        "independent folds can be run as parallel processes and concatenated.",
    )
    p.add_argument("--out", type=str, default="/tmp/bo_results.json")
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Cap intra-op torch threads (0 = leave torch's default). Set this when "
        "running several sweeps concurrently so they do not oversubscribe the box.",
    )
    return p


def main() -> None:
    global EM_SHRINKAGE, COND_NOISE, DEEP_KERNEL, EM_CANONICAL
    global HYPERBO_BASE_KERNEL
    global PD1_SOURCE, PD1_ROOT
    p = build_parser()
    args = p.parse_args()

    EM_SHRINKAGE = args.em_shrinkage
    COND_NOISE = args.cond_noise
    DEEP_KERNEL = args.deep_kernel
    EM_CANONICAL = args.em_canonical
    PD1_SOURCE = args.pd1_source
    if args.pd1_root:
        PD1_ROOT = args.pd1_root
    else:
        from _scratch_bo.pd1_full_data import DEFAULT_PD1_ROOT

        PD1_ROOT = DEFAULT_PD1_ROOT

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.smoke:
        (
            args.n_configs,
            args.n_iters,
            args.n_eval,
            args.n_seeds,
            args.n_em,
            args.n_pretrain,
        ) = (
            80,
            15,
            2,
            2,
            20,
            12,
        )
        # keep the deep-kernel meta-learners cheap in smoke mode
        args.meta_iters = min(args.meta_iters, 50)
        args.pacoh_particles = min(args.pacoh_particles, 4)
    methods = args.methods.split(",")

    # _gaussian_rank_warp maps values through their RANK within the tensor it is given,
    # so it is pool-dependent. In arm C the matched grid and the per-task full pools are
    # warped separately, which puts EM's inducing values and the candidates on two
    # different monotone scales -- precisely at the off-grid points arm C exists to
    # test. winsor/neglog are pointwise and unaffected (D4).
    # The defect is 'the rank warp is applied to more than one pool', which happens
    # whenever the per-task pools are loaded at all (candidate OR pretrain OR
    # canonical pool = full) and on pd1_full, which warps eval pools and the matched
    # subset separately. Keying on pd1_candidate_pool alone missed both (round-4 R4).
    _multi_pool = (
        args.pd1_candidate_pool == "full"
        or args.pd1_pretrain_pool == "full"
        or args.em_canonical_pool == "full"
        or args.benchmark == "pd1_full"
    )
    if args.output_warp == "gaussian" and _multi_pool:
        raise ValueError(
            "--output-warp gaussian is rank-based and therefore pool-dependent, so it "
            "cannot be combined with --pd1-candidate-pool full: the matched grid and "
            "the full per-task pools would be warped against different reference sets. "
            "Use --output-warp neglog or winsor for arm C, or restrict to "
            "--pd1-candidate-pool matched."
        )

    # --iw-nu is only consulted inside `if use_covar_prior:` in pretrain_em_prior, so
    # passing it alone is a silent no-op -- verified as occurrence #7 of this project's
    # recurring failure mode. Note also that valid values satisfy nu > M-1, where M is
    # the number of inducing points (400 on PD1's matched grid), so small values raise.
    if args.iw_nu is not None and not args.use_covar_prior:
        raise ValueError(
            "--iw-nu sets the Inverse-Wishart prior's degrees of freedom, which is only "
            "read when the covariance prior is enabled. Pass --use-covar-prior too, or "
            "drop --iw-nu. Refusing to accept a flag that would do nothing. Valid "
            "values are > M-1 where M is the inducing-point count."
        )

    # PD1-only flags are read by the PD1 code paths alone. Passing them with the default
    # lcbench benchmark silently does nothing, which is how the canon_matrix and
    # hb_hybrid "PD1" cells (S26.4) ran LCBench under PD1 filenames for four cells each.
    # Detection is on argv rather than on the parsed value, because every one of these
    # flags has a non-None default that a value check cannot distinguish from disuse.
    if not args.benchmark.startswith("pd1"):
        pd1_only = (
            "--pd1-candidate-pool",
            "--pd1-pretrain-pool",
            "--pd1-group",
            "--pd1-root",
            "--pd1-source",
            "--em-canonical-pool",
        )
        passed = [
            f for f in pd1_only if any(a.split("=")[0] == f for a in sys.argv[1:])
        ]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} require a PD1 benchmark, but "
                f"--benchmark={args.benchmark}. These flags are ignored on the LCBench "
                "path, so this run would silently be plain LCBench."
            )

    if args.benchmark == "pd1_probe":
        _pd1_probe(args)
        return

    if args.benchmark == "pd1_loo":
        run_pd1_matched_loo(args, methods)
        return

    if args.benchmark == "pd1_full":
        run_pd1_full(args, methods)
        return

    if args.benchmark == "pd1":
        print("Loading PD1 datasets (matched-config subset) ...", flush=True)
        Xn, Y_all, names = load_pd1_pool()
        print(
            f"PD1: {len(names)} datasets, {Xn.shape[0]} matched configs, "
            f"d={Xn.shape[1]}",
            flush=True,
        )
    else:
        names = list(DATASET_NAMES)
        print(f"Loading {len(names)} LCBench datasets ...", flush=True)
        Xn, Y_all = load_pool(names)

    # Cap the config pool to the available shared grid (PD1's matched set is small).
    if args.n_configs > Xn.shape[0]:
        print(
            f"Capping --n-configs {args.n_configs} -> {Xn.shape[0]} (pool size)",
            flush=True,
        )
        args.n_configs = Xn.shape[0]

    # Deterministic split: subsample shared configs, then choose pretrain/eval datasets.
    torch.manual_seed(args.split_seed)
    _perm = torch.randperm(Xn.shape[0])
    cfg = _perm[: args.n_configs]  # configs used to PRE-TRAIN the prior
    if args.novel_config_split:
        if 2 * args.n_configs > Xn.shape[0]:
            raise ValueError("novel-config-split needs 2*n_configs <= pool size")
        cfg_bo = _perm[args.n_configs : 2 * args.n_configs]  # held-out BO configs
    else:
        cfg_bo = cfg  # standard: BO runs on the same pool the prior was trained on
    X_pool = Xn[cfg]  # pretrain pool (n_configs, d)
    Yc = [Y[cfg] for Y in Y_all]  # pretrain targets per dataset
    X_bo = Xn[cfg_bo]  # BO candidate pool
    Yc_bo = [Y[cfg_bo] for Y in Y_all]  # BO targets per dataset

    if args.ood_split:
        # Group datasets by the SHAPE of their config->accuracy response (row-
        # standardized), then pretrain on one group and evaluate on the MOST
        # DISSIMILAR group, so the empirical prior is a poor fit at eval time.
        R = torch.stack(
            [
                (Yc[i].squeeze(-1) - Yc[i].mean()) / Yc[i].std().clamp_min(1e-8)
                for i in range(len(names))
            ]
        )  # (n_ds, n_configs)
        C = (R @ R.T) / R.shape[1]  # ~Pearson correlation between datasets
        n_ds = C.shape[0]
        off = C.clone()
        off.fill_diagonal_(2.0)
        flat = int(off.argmin().item())
        i_star, j_star = flat // n_ds, flat % n_ds  # least-correlated anchor pair
        d = C[:, i_star] - C[:, j_star]  # high => i_star-like, low => j_star-like
        order = torch.argsort(d, descending=True).tolist()
        pre_ix = order[: args.n_pretrain]  # most i_star-like -> pretrain
        eval_ix = [k for k in reversed(order) if k not in set(pre_ix)][
            : args.n_eval
        ]  # most j_star-like (dissimilar) -> eval

        def _mc(g1, g2):
            v = [C[a, b].item() for a in g1 for b in g2 if a != b]
            return sum(v) / len(v) if v else float("nan")

        print(
            f"OOD split: pretrain n={len(pre_ix)}, eval n={len(eval_ix)} | "
            f"within-pretrain corr={_mc(pre_ix, pre_ix):.3f}, "
            f"within-eval corr={_mc(eval_ix, eval_ix):.3f}, "
            f"CROSS corr={_mc(pre_ix, eval_ix):.3f}",
            flush=True,
        )
    else:
        dperm = torch.randperm(len(names))
        eval_ix = dperm[: args.n_eval].tolist()
        pre_ix = dperm[args.n_eval : args.n_eval + args.n_pretrain].tolist()

    Ypool = torch.cat([Yc[i] for i in pre_ix])
    ymean, ystd = Ypool.mean(), Ypool.std().clamp_min(1e-8)
    pretrain_ds = [
        ExperimentDataset(X=X_pool, Y=(Yc[i] - ymean) / ystd) for i in pre_ix
    ]
    print(
        f"Pretrain on {len(pre_ix)} datasets; evaluate on {len(eval_ix)}: "
        f"{', '.join(names[i] for i in eval_ix)}",
        flush=True,
    )

    # Re-seed before pre-training so the prior can be resampled independently of the
    # dataset split. Neural meta-learners (HyperBO/PACOH/ABLR) train ONE prior that is
    # then reused by every BO run, so per-run statistics say nothing about how much of a
    # method's score is prior-draw luck; varying this seed is the only way to measure it.
    # EM is unaffected -- its pre-training is deterministic given the data.
    torch.manual_seed(args.pretrain_seed)

    print(
        "Fitting shared SumMLL kernel "
        f"(canonical={'hyperbo-pretrained' if args.em_canonical == 'hyperbo' else ('deep ' + str(DEEP_KERNEL) if DEEP_KERNEL else 'ARD Matern')}) "
        "+ pre-training EM prior ...",
        flush=True,
    )
    t0 = time.time()
    hyperbo_prior_early = None
    _hb_early_stats: dict = {}
    # --em-mean needs a HyperBO prior, and on this path the real one is trained ~30
    # lines below, so requesting it without --em-canonical hyperbo used to raise with a
    # message whose advice ("include a hyperbo_* method") does not work here (D11).
    if args.em_canonical == "hyperbo" or args.em_mean != "empirical":
        # EM's prior is normally built before HyperBO; reusing HyperBO's kernel inverts
        # that dependency, so pre-train it here and hand the same object to the hyperbo_*
        # methods below rather than paying 25k steps twice.
        print("Pre-training HyperBO first (its kernel seeds EM) ...", flush=True)
        torch.manual_seed(args.pretrain_seed * 1009 + 11)
        hyperbo_prior_early = prior_cache.cached(
            args.prior_cache,
            args.prior_cache_mode,
            "hyperbo",
            pretrain_ds,
            _hyperbo_cache_params(args, X_pool.shape[-1]),
            lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_hyperbo(
                datasets=_ds,
                input_dim=_dim,
                hidden_dims=args.hyperbo_hidden,
                use_linear_mean=not args.hyperbo_zero_mean,
                loss_type=args.hyperbo_loss,
                ekl_optimizer=args.ekl_optimizer,
                ekl_lbfgs_iterations=args.ekl_lbfgs_iters,
                num_iterations=args.hyperbo_iters or args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
                task_batch_size=args.meta_task_batch,
            ),
            stats=_hb_early_stats,
        )
        HYPERBO_BASE_KERNEL = hyperbo_kernel_from_prior(hyperbo_prior_early)
    mean_s, covar_s = build_shared_kernel(
        pretrain_ds, deep_hidden=DEEP_KERNEL, base_mean=args.em_base_mean
    )
    if hyperbo_prior_early is not None:
        covar_s = hyperbo_kernel_from_prior(hyperbo_prior_early)
    # --em-mean was applied only on the pd1_loo path, so the mean-transfer factor was
    # silently ignored on LCBench (review, integration #1). Applied here, where the
    # prior actually exists -- an earlier attempt at this fix landed in _pd1_probe and
    # referenced a name that is not defined there.
    # Keep the PRE-BLEND mean for the non-EM baselines: rebinding mean_s here used to
    # hand HyperBO's meta-learned mean to pretrained_gp_frozen as well, turning the
    # control into a second transfer method (D1).
    mean_baseline = mean_s
    mean_s = _apply_em_mean(mean_s, args, hyperbo_prior_early)
    em_prior = pretrain_em_prior(
        datasets=pretrain_ds,
        mean_module=mean_s,
        covar_module=covar_s,
        likelihood_noise=torch.tensor(args.em_likelihood_noise, dtype=torch.double),
        covariance_shrinkage=_resolve_em_shrinkage(args, pretrain_ds, covar_s),
        num_em_iterations=args.n_em,
        enable_interpolation=True,
        iw_nu=_resolve_iw_nu(args, pretrain_ds, covar_s),
        use_mean_prior=args.use_mean_prior,
        use_covar_prior=args.use_covar_prior,
        init_mode=args.em_init_mode,
    )
    # t0 was started before the optional "HyperBO first" block above, whose kernel seeds
    # EM. Charging that fit to EM would make --em-canonical hyperbo look ~100x more
    # expensive than it is and would double-count it against HyperBO (S30.1).
    em_pretrain_s = (time.time() - t0) - float(_hb_early_stats.get("compute_s") or 0.0)
    print(f"  pretraining done in {em_pretrain_s:.1f}s", flush=True)

    # One-time pre-training cost per meta-learner (the efficiency axis of the
    # perf-vs-cost Pareto). 0.0 when the method is not requested.
    hyperbo_pretrain_s = 0.0
    pacoh_pretrain_s = 0.0
    ablr_pretrain_s = 0.0

    # Meta-learning baselines (HyperBO / PACOH). Pre-trained once on the SAME
    # standardized pretrain datasets as the EM prior, for a fair comparison. Only
    # trained when a corresponding method is requested (they are the expensive part).
    hyperbo_prior = None
    pacoh_prior = None
    # Per-kind cache accounting for this cell; see the LOO path for the rationale.
    pcache_hits = {"em": 0, "hyperbo": 0, "pacoh": 0, "ablr": 0}
    pcache_calls = {"em": 0, "hyperbo": 0, "pacoh": 0, "ablr": 0}
    if any(m.startswith("hyperbo") for m in methods):
        t1 = time.time()
        print("Pre-training HyperBO deep-kernel prior ...", flush=True)
        # Re-seed per method so this baseline's init does not depend on how many
        # RNG draws upstream code consumed. Without this, changing the canonical
        # kernel (which builds an MLP) silently reinitializes every neural
        # baseline, moving arms that should be bit-identical controls.
        torch.manual_seed(args.pretrain_seed * 1009 + 11)
        if hyperbo_prior_early is not None:
            # Paid for already when its kernel seeded the EM prior.
            hyperbo_prior = hyperbo_prior_early
            _hb_main_stats: dict = {
                "hit": bool(_hb_early_stats.get("hit")),
                # The fit was genuinely paid for, just earlier in this cell. Attribute
                # the real cost to HyperBO here and subtract it from em_pretrain_s,
                # rather than reporting HyperBO as free and EM as expensive.
                "compute_s": float(_hb_early_stats.get("compute_s") or 0.0),
            }
        else:
            _hb_main_stats = {}
            hyperbo_prior = prior_cache.cached(
                args.prior_cache,
                args.prior_cache_mode,
                "hyperbo",
                pretrain_ds,
                _hyperbo_cache_params(args, X_pool.shape[-1]),
                lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_hyperbo(
                    datasets=_ds,
                    input_dim=_dim,
                    hidden_dims=args.hyperbo_hidden,
                    use_linear_mean=not args.hyperbo_zero_mean,
                    loss_type=args.hyperbo_loss,
                    ekl_optimizer=args.ekl_optimizer,
                    ekl_lbfgs_iterations=args.ekl_lbfgs_iters,
                    num_iterations=args.hyperbo_iters or args.meta_iters,
                    learning_rate=args.meta_lr,
                    subsample_size=args.meta_subsample,
                    task_batch_size=args.meta_task_batch,
                ),
                stats=_hb_main_stats,
            )
        if hyperbo_prior is not None:
            HYPERBO_BASE_KERNEL = hyperbo_kernel_from_prior(hyperbo_prior)
        print(f"  HyperBO pretraining done in {time.time() - t1:.1f}s", flush=True)
        # Only real fitting time, so a warm cache cannot masquerade as a free model.
        hyperbo_pretrain_s = float(_hb_main_stats.get("compute_s") or 0.0)
        pcache_hits["hyperbo"] += int(bool(_hb_main_stats.get("hit")))
        pcache_calls["hyperbo"] += 1
    if any(m.startswith("pacoh") for m in methods):
        t1 = time.time()
        print(
            f"Pre-training PACOH-GP prior (K={args.pacoh_particles} particles) ...",
            flush=True,
        )
        # Re-seed per method so this baseline's init does not depend on how many
        # RNG draws upstream code consumed. Without this, changing the canonical
        # kernel (which builds an MLP) silently reinitializes every neural
        # baseline, moving arms that should be bit-identical controls.
        torch.manual_seed(args.pretrain_seed * 1009 + 13)
        pacoh_prior = pretrain_pacoh_gp(
            datasets=pretrain_ds,
            input_dim=X_pool.shape[-1],
            config=PACOHGPConfig(
                num_particles=args.pacoh_particles,
                hidden_dims=args.pacoh_hidden or args.hyperbo_hidden,
                hyper_prior_std=args.pacoh_hyper_prior_std,
                svgd_length_scale=args.pacoh_svgd_lengthscale,
            ),
            num_iterations=args.pacoh_iters or args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
        )
        print(f"  PACOH pretraining done in {time.time() - t1:.1f}s", flush=True)
        pacoh_pretrain_s = time.time() - t1

    # ABLR transfer-BO baseline (Perrone et al.): shared feature map meta-trained on the
    # same standardized pretrain datasets. Only trained when the method is requested.
    ablr_prior = None
    if any(m.startswith("ablr") for m in methods):
        t1 = time.time()
        print("Pre-training ABLR shared feature map ...", flush=True)
        # Re-seed per method so this baseline's init does not depend on how many
        # RNG draws upstream code consumed. Without this, changing the canonical
        # kernel (which builds an MLP) silently reinitializes every neural
        # baseline, moving arms that should be bit-identical controls.
        torch.manual_seed(args.pretrain_seed * 1009 + 17)
        ablr_prior = pretrain_ablr(
            datasets=pretrain_ds,
            input_dim=X_pool.shape[-1],
            hidden=args.ablr_hidden or tuple(args.hyperbo_hidden),
            feat_dim=args.ablr_feat_dim,
            num_iterations=args.ablr_iters or args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
            per_task_precision=args.ablr_per_task_precision,
        )
        ablr_pretrain_s = time.time() - t1
        print(f"  ABLR pretraining done in {ablr_pretrain_s:.1f}s", flush=True)

    def surrogate_fn(method, X, Y):
        return make_surrogate(
            method,
            X,
            Y,
            em_prior,
            mean_s,
            covar_s,
            hyperbo_prior,
            pacoh_prior,
            ablr_prior,
            mean_baseline=mean_baseline,
        )

    # results[method] = list over (dataset,seed) of best-so-far trajectories (raw acc)
    results: dict[str, list[list[float]]] = {m: [] for m in methods}
    calib_records: dict[str, list] = {m: [] for m in methods}
    run_ds: list[int] = []  # eval-dataset index per run (for dataset-clustered ranks)
    # Total online BO wall-time per method (surrogate build + posterior + any
    # per-step refit) summed over all runs -- the amortized inference cost.
    method_bo_s: dict[str, float] = {m: 0.0 for m in methods}
    pool_max_raw: list[float] = []

    _t_start = time.time()
    _n_runs_total = len(eval_ix) * args.n_seeds
    _run_i = 0
    for ei in eval_ix:
        _set_cur_ds(names[ei])
        dY = (Yc_bo[ei] - ymean) / ystd  # standardized BO targets (n_configs, 1)
        raw = Yc_bo[ei].squeeze(-1)  # raw accuracy on the BO pool (n_configs,)
        for seed in range(args.n_seeds):
            # The initial design must vary with the PRIOR, not only with the seed.
            # Otherwise at --n-seeds 1 every prior shares one initial design, the
            # per-prior noise e_p becomes a single shared e, and the between-prior
            # spread estimates only sigma_B^2 -- understating Var by sigma_W^2,
            # which S27.24 measured as the LARGER component for most methods. The
            # replication axis has to be independent to be worth anything.
            _bo_seed = 1000 * ei + seed + 100003 * args.pretrain_seed
            g = torch.Generator().manual_seed(_bo_seed)
            init_idx = torch.randperm(args.n_configs, generator=g)[
                : args.n_init
            ].tolist()
            pool_max_raw.append(raw.max().item())
            run_ds.append(int(ei))
            for m in methods:
                gm = torch.Generator().manual_seed(7 * _bo_seed + 13)
                _t_bo = time.time()
                traj_std = run_bo(
                    m,
                    X_bo,
                    dY,
                    init_idx,
                    args.n_iters,
                    gm,
                    surrogate_fn,
                    acquisition=args.acquisition,
                    ucb_beta=args.ucb_beta,
                    obs_noise=args.obs_noise,
                    batch_q=args.batch_q,
                    batch_mode=args.batch_mode,
                    n_fantasies=args.n_fantasies,
                    calib_sink=calib_records[m],
                )
                method_bo_s[m] += time.time() - _t_bo
                traj_raw = [v * ystd.item() + ymean.item() for v in traj_std]
                results[m].append(traj_raw)
            _run_i += 1
            print(
                f"    progress {_run_i}/{_n_runs_total} "
                f"({names[ei]} seed {seed}) | {time.time() - _t_start:.0f}s elapsed",
                flush=True,
            )
        print(f"  done dataset {names[ei]}", flush=True)

    # Aggregate: mean best-so-far and mean simple regret vs #evals.
    pool_max_mean = sum(pool_max_raw) / len(pool_max_raw)
    # Trajectory length is n_iters+1 normally, or n_iters when n_init=0 (no step-0
    # initial-design entry); derive it from the actual results.
    n_pts = len(results[methods[0]][0])
    summary = {}
    for m in methods:
        T = torch.tensor(results[m])  # (runs, n_pts)
        mean_best = T.mean(0)
        # paired simple regret to each run's pool max
        pm = torch.tensor(pool_max_raw).unsqueeze(1)
        regret_all = pm - T  # (runs, n_pts)
        regret = regret_all.mean(0)
        regret_sem = regret_all.std(0) / (T.shape[0] ** 0.5)
        summary[m] = {
            "mean_best": mean_best.tolist(),
            "mean_regret": regret.tolist(),
            "regret_sem": regret_sem.tolist(),
        }

    # Per-run final simple regret (raw units) per method, tagged by eval dataset, so
    # downstream analysis can compute dataset-clustered average ranks / win-rates.
    per_run = {
        "eval_dataset_idx": run_ds,
        "pool_max_raw": pool_max_raw,
        "final_regret": {
            m: [pool_max_raw[r] - results[m][r][-1] for r in range(len(pool_max_raw))]
            for m in methods
        },
        # Full per-run best-so-far trajectories (raw units), so downstream analysis can
        # compute dataset-clustered rank/CD at ANY horizon and regret-AUC (not just final).
        "traj_raw": {m: results[m] for m in methods},
    }

    # 1-step-ahead predictive calibration per method (mean NLL + 95% coverage of the true
    # target under the surrogate posterior at acquired points; standardized units).
    calibration = {}
    for m in methods:
        recs = calib_records[m]
        if recs:
            nlls = [r[0] for r in recs]
            cov = [r[1] for r in recs]
            calibration[m] = {
                "mean_nll": sum(nlls) / len(nlls),
                "coverage95": sum(cov) / len(cov),
                "n_points": len(recs),
            }

    out = {
        "config": vars(args),
        "eval_datasets": [names[i] for i in eval_ix],
        "n_runs": len(pool_max_raw),
        "pool_max_mean": pool_max_mean,
        "init_evals": args.n_init,
        "summary": summary,
        "per_run": per_run,
        "calibration": calibration,
        "timings": {
            # one-time pre-training cost (efficiency axis). These are REAL fitting
            # seconds: a cache hit contributes 0 and is recorded in prior_cache_hits
            # instead, so a warm cache cannot be read as a free model (S30.1).
            "em_pretrain_s": em_pretrain_s,
            "hyperbo_pretrain_s": hyperbo_pretrain_s,
            "pacoh_pretrain_s": pacoh_pretrain_s,
            "ablr_pretrain_s": ablr_pretrain_s,
            "prior_cache_hits": dict(pcache_hits),
            "prior_cache_calls": dict(pcache_calls),
            "meta_iters": args.meta_iters,
            # total online BO wall-time per method over all runs
            "method_bo_s": method_bo_s,
            "n_runs": len(pool_max_raw),
        },
    }
    atomic_write_json(args.out, out)

    # Console report
    checkpoints = sorted({0, 5, 10, 20, 30, args.n_iters})
    checkpoints = [c for c in checkpoints if c < n_pts]
    print(
        f"\n=== BO on {args.benchmark.upper()} "
        f"(finite pool of {args.n_configs} configs) ==="
    )
    print(
        f"{len(eval_ix)} eval datasets x {args.n_seeds} seeds = "
        f"{len(pool_max_raw)} runs"
    )
    print(
        f"init design: {args.n_init} random evals | "
        f"pool max accuracy (mean): {pool_max_mean:.3f}\n"
    )
    print("Mean best accuracy found vs #BO evaluations (higher=better):")
    hdr = f"{'method':<22}" + "".join(f"@{c:<7}" for c in checkpoints)
    print(hdr)
    for m in methods:
        mb = summary[m]["mean_best"]
        print(f"{m:<22}" + "".join(f"{mb[c]:<8.3f}" for c in checkpoints))
    print("\nMean simple regret +/- s.e.m. vs #BO evaluations (lower=better):")
    print(hdr)
    for m in methods:
        rg = summary[m]["mean_regret"]
        se = summary[m]["regret_sem"]
        print(
            f"{m:<22}" + "".join(f"{rg[c]:.2f}\u00b1{se[c]:.2f} " for c in checkpoints)
        )

    # Paired regret differences vs em_finetuned (same (dataset, seed) init per run,
    # so the dominant between-run variance cancels -> a far tighter test of small
    # edges).
    if "em_finetuned" in results:
        pm = torch.tensor(pool_max_raw).unsqueeze(1)
        base_reg = pm - torch.tensor(results["em_finetuned"])  # (runs, n_pts)
        print(
            "\nPaired regret difference (method - em_finetuned); "
            "negative => better than em_finetuned:"
        )
        print(hdr)
        for m in methods:
            if m == "em_finetuned":
                continue
            d = (pm - torch.tensor(results[m])) - base_reg  # (runs, n_pts)
            dm = d.mean(0)
            dsem = d.std(0) / (d.shape[0] ** 0.5)
            print(
                f"{m:<22}"
                + "".join(f"{dm[c]:+.3f}\u00b1{dsem[c]:.3f} " for c in checkpoints)
            )

        for cmp_m in ("em_additive_freshbase", "em_additive_3way"):
            if cmp_m not in results:
                continue
            pm1 = torch.tensor(pool_max_raw)
            ra = pm1 - torch.tensor(results[cmp_m])[:, -1]  # final regret per run
            rf = pm1 - torch.tensor(results["em_finetuned"])[:, -1]
            dd = ra - rf
            print(
                f"\nPer-eval-dataset paired diff ({cmp_m} - em_finetuned) @final "
                "(negative => additive better):"
            )
            for di, ei in enumerate(eval_ix):
                seg = dd[di * args.n_seeds : (di + 1) * args.n_seeds]
                print(
                    f"  {names[ei]:<18} mean={seg.mean():+.3f} "
                    f"sem={seg.std() / (len(seg) ** 0.5):.3f}"
                )
    if _HP_LOG:
        import collections as _c
        import statistics as _st

        agg = _c.defaultdict(list)
        for meth, ds, ls, os_ in _HP_LOG:
            agg[(meth, ds)].append((ls, os_))
        print(
            "\nFresh-base hyperparameters by (method, dataset) "
            "[min_ls near 0.01 floor => overfitting]:"
        )
        for meth, ds in sorted(agg):
            lss = [x[0] for x in agg[(meth, ds)]]
            oss = [x[1] for x in agg[(meth, ds)]]
            print(
                f"  {meth:<22} {ds:<20} min_ls={_st.mean(lss):.3f} "
                f"outputscale={_st.mean(oss):.2f}"
            )
    # Timing / efficiency summary (perf-vs-cost Pareto inputs).
    n_runs = len(pool_max_raw)
    print(
        f"\n=== Timing (meta_iters={args.meta_iters}) ===\n"
        f"pretrain: em={em_pretrain_s:.1f}s  hyperbo={hyperbo_pretrain_s:.1f}s  "
        f"pacoh={pacoh_pretrain_s:.1f}s  ablr={ablr_pretrain_s:.1f}s"
    )
    print(f"{'method':<22}{'total_bo_s':>12}{'per_run_ms':>12}{'final_regret':>14}")
    for m in methods:
        tot = method_bo_s[m]
        per = 1000.0 * tot / max(n_runs, 1)
        fr = summary[m]["mean_regret"][-1]
        print(f"{m:<22}{tot:>12.1f}{per:>12.1f}{fr:>14.3f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
