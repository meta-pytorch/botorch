#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Surrogate diagnostics for the empirical-GP BO study, across ALL methods.

Runs on the same LCBench setup as ``bo_experiment.py`` and answers "why does a method
win or lose the BO loop?" separately from the loop itself:

1. **Shared SumMLL kernel hyperparameters** (lengthscales, outputscale, mean).
2. **Prior-mean informativeness** -- Spearman correlation of each surrogate's PRIOR mean
   (no observations at all) with the true per-config accuracy. This is the zero-shot
   quantity the ``n_init=0`` study measures indirectly; here it is measured directly.
   Constant-mean models (``vanilla_gp``, ``pretrained_gp_frozen``, and ABLR whose BLR
   head has a zero prior mean) are structurally uninformative and score ~0.
3. **Ranking quality and calibration vs #observations** -- after ``n_obs`` random
   observations, on the *unobserved* pool: Spearman correlation of the predictive mean
   with the truth (the signal LogEI actually exploits), mean predictive sigma, RMSE,
   the calibration ratio sigma/RMSE (<<1 overconfident), Gaussian NLL and 95% coverage.
   Sweeping ``--n-obs-grid`` traces how fast each surrogate's ranking sharpens with
   data, which is the mechanism behind the early-regime separation.

Results are written to JSON so the tables in EXPERIMENTAL_RESULTS.md are reproducible.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from _scratch_bo import bo_experiment as _boexp, prior_cache
from _scratch_bo.bo_experiment import (
    _posterior_moments,
    build_shared_kernel,
    hyperbo_kernel_from_prior,
    load_pool,
    make_surrogate,
    pretrain_ablr,
)
from _scratch_bo.lcbench_io import LCBENCH_DATASET_NAMES
from botorch.models.empirical_gps import (
    PACOHGPConfig,
    pretrain_em_prior,
    pretrain_hyperbo,
    pretrain_pacoh_gp,
)
from botorch.models.empirical_gps.utils import ExperimentDataset

torch.set_default_dtype(torch.float64)

DEFAULT_METHODS = (
    "em_frozen,em_finetuned,em_noisefit,hyperbo_frozen,hyperbo_adapt,"
    "pacoh_frozen,ablr,pretrained_gp_frozen,vanilla_gp"
)


def _diag_hyperbo_params(args, input_dim: int) -> dict:
    """EFFECTIVE HyperBO fit parameters on THIS harness.

    Deliberately not `bo_experiment._hyperbo_cache_params`. That function reports
    `args.hyperbo_loss` and `args.hyperbo_iters`, but this harness hardcodes
    `loss_type="NLL"` and passes `args.meta_iters`, and it does not define the
    `--hyperbo-*` or `--ekl-*` flags at all. Keying a fit under a description it does
    not satisfy is precisely the wrong-prior hit the cache exists to prevent, and it
    would be invisible: the run would simply use someone else's model.

    Values for arguments this harness never passes are the callee's own defaults, so
    the dict describes what was actually fitted. Keys mirror the other harness so that
    a genuinely identical fit CAN share an entry rather than being duplicated.
    """
    return {
        "input_dim": int(input_dim),
        "hidden_dims": str(args.hyperbo_hidden),
        "use_linear_mean": True,  # pretrain_hyperbo default; never overridden here
        "loss_type": "NLL",  # hardcoded at every call site in this file
        "ekl_optimizer": "adam",  # callee default; inert under NLL
        "ekl_lbfgs_iterations": 100,  # callee default; inert under NLL
        "num_iterations": args.meta_iters,
        "learning_rate": args.meta_lr,
        "subsample_size": args.meta_subsample,
        "task_batch_size": args.meta_task_batch,
        "pretrain_seed": args.pretrain_seed,
    }


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    if a.norm() < 1e-9 or b.norm() < 1e-9:
        return 0.0  # a constant predictor carries no ranking information
    return float((a @ b) / denom)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.reshape(-1).std() < 1e-9:
        return 0.0
    ra = a.reshape(-1).argsort().argsort().double()
    rb = b.reshape(-1).argsort().argsort().double()
    return pearson(ra, rb)


def prior_moments(model, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PRIOR mean/sigma at candidates (no conditioning), for any surrogate type."""
    if hasattr(model, "forward"):
        with torch.no_grad():
            prior = model.forward(cand)
        mean = prior.mean
        if mean.ndim > 1:  # PACOH: batched over K particles -> mixture prior mean
            mean = mean.mean(0)
        var = prior.variance
        if var.ndim > 1:
            var = var.mean(0)
        return mean.reshape(-1), var.clamp_min(1e-12).sqrt().reshape(-1)
    # ABLR and anything else without a GP prior: fall back to the posterior of the
    # (essentially empty) conditioning set, which is its prior for our purposes.
    with torch.no_grad():
        return _posterior_moments(model, cand)


def _mean(rows, i: int) -> float:
    """Mean of column i over per-eval-dataset rows."""
    return sum(r[i] for r in rows) / max(len(rows), 1)


def _load_benchmark(args) -> tuple[list[str], torch.Tensor, list[torch.Tensor]]:
    """Load a suite as (names, X_pool, per-task Y).

    Everything downstream in ``main`` is benchmark-agnostic, so this is the only place
    that needs to know which suite is being run.
    """
    if args.benchmark == "pd1_loo":
        # PD1_SOURCE is module-level state in bo_experiment and is read inside
        # load_pd1_pool, so it must be set BEFORE the call, not passed to it.
        _boexp.PD1_SOURCE = args.pd1_source
        if not _boexp.PD1_ROOT:
            from _scratch_bo.pd1_full_data import DEFAULT_PD1_ROOT

            _boexp.PD1_ROOT = DEFAULT_PD1_ROOT
        print(f"Loading PD1 (source={args.pd1_source}, matched pool) ...", flush=True)
        Xn, Y_all, names = _boexp.load_pd1_pool()
        print(
            f"  PD1: {len(names)} tasks, {Xn.shape[0]} matched configs, "
            f"d={Xn.shape[1]}",
            flush=True,
        )
        return list(names), Xn, Y_all

    names = list(LCBENCH_DATASET_NAMES)
    print("Loading LCBench ...", flush=True)
    Xn, Y_all = load_pool(names)
    return names, Xn, Y_all


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Split out of main() so tests can validate flag
    strings without executing anything."""
    p = argparse.ArgumentParser()
    p.add_argument("--n-configs", type=int, default=200)
    p.add_argument("--n-pretrain", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--n-em", type=int, default=50)
    p.add_argument(
        "--n-obs-grid",
        type=str,
        default="5,10,20,50",
        help="comma-separated observation counts to evaluate ranking/calibration at",
    )
    p.add_argument("--n-obs", type=int, default=None, help="alias for a single point")
    p.add_argument("--meta-iters", type=int, default=2000)
    p.add_argument("--meta-lr", type=float, default=1e-3)
    p.add_argument("--meta-subsample", type=int, default=64)
    p.add_argument("--meta-task-batch", type=int, default=5)
    p.add_argument("--pacoh-particles", type=int, default=10)
    p.add_argument(
        "--hyperbo-hidden",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(32, 32),
    )
    p.add_argument(
        "--em-canonical",
        type=str,
        default="sumll",
        choices=["sumll", "hyperbo"],
        help="Canonical/interpolation kernel for the EM prior. "
        "'hyperbo' reuses HyperBO's pre-trained deep kernel; only "
        "meaningful off-grid (--n-inducing < --n-configs).",
    )
    p.add_argument(
        "--deep-kernel",
        type=lambda s: tuple(int(x) for x in s.split(",")) if s else None,
        default=None,
        help="Hidden dims for a learned MLP embedding under the canonical kernel, e.g. "
        "32,32. Default (None) keeps the plain ARD Matern. The canonical kernel sets "
        "the predictive-variance SHAPE, which diagnostics identified as the residual "
        "gap to HyperBO.",
    )
    p.add_argument(
        "--hetero-frac",
        type=float,
        default=1.0,
        help="Fraction of the pool each historical task observes, with a DIFFERENT "
        "random subset per task. 1.0 (default) is the usual dense shared grid. Values "
        "< 1 create the sparse, irregular, per-task observation regime the "
        "continuous-domain EM model is designed for and that neither LCBench nor PD1 "
        "otherwise tests. Requires --n-inducing so a shared reference set exists.",
    )
    p.add_argument(
        "--n-inducing",
        type=int,
        default=0,
        help="Reference/inducing points for the EM prior; 0 = use the whole "
        "pool (on-grid, Nystrom residual identically zero). Set < --n-configs "
        "to evaluate OFF-grid, which restores the full-rank Lambda term.",
    )
    p.add_argument(
        "--em-shrinkage",
        type=float,
        default=0.0,
        help="EM M-step covariance shrinkage alpha",
    )
    p.add_argument(
        "--cond-noise",
        type=float,
        default=1e-3,
        help="Frozen conditioning-likelihood noise (std, standardized units). The 1e-3 "
        "default treats targets as essentially noiseless; if the task has real "
        "observation noise this collapses posterior variance, showing up as "
        "overconfidence (sigma/RMSE << 1) rather than as poor RMSE.",
    )
    p.add_argument(
        "--em-likelihood-noise",
        type=float,
        default=1e-2,
        help="Noise VARIANCE for EM pre-training's E-step. Mirrors bo_experiment so the "
        "same cell can run on BO and regression. Default 1e-2 reproduces every existing "
        "result; S27.35 puts the optimum near 1-3.",
    )
    p.add_argument(
        "--em-base-mean",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="Parametric base mean m(.) fitted by the shared SumMLL. Mirrors "
        "bo_experiment so the same OFAT cell can run on BO and regression.",
    )
    p.add_argument(
        "--em-mean",
        type=str,
        default="empirical",
        choices=["empirical", "hyperbo", "blend"],
        help="Blend EM's base mean toward HyperBO's learned mean. Mirrors "
        "bo_experiment. Requires a HyperBO prior, i.e. --em-canonical hyperbo here.",
    )
    p.add_argument(
        "--em-mean-weight",
        type=float,
        default=0.5,
        help="beta for --em-mean blend: (1-beta)*m_em + beta*m_hyperbo.",
    )
    p.add_argument(
        "--iw-nu",
        type=float,
        default=None,
        help="Inverse-Wishart prior dof for the EM M-step. Only read when "
        "--use-covar-prior is set; valid values are > M-1.",
    )
    p.add_argument(
        "--use-mean-prior",
        action="store_true",
        help="Enable EM's prior on the mean.",
    )
    p.add_argument(
        "--use-covar-prior",
        action="store_true",
        help="Enable EM's Inverse-Wishart prior on the covariance.",
    )
    p.add_argument(
        "--em-init-mode",
        type=str,
        default="kernel",
        choices=["kernel", "naive"],
        help="EM initialisation mode.",
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default="lcbench",
        choices=["lcbench", "pd1_loo"],
        help="Which suite to run regression/calibration on. 'pd1_loo' uses PD1's "
        "matched-config pool, giving the PD1 regression cell that EXPERIMENT_PLAN "
        "stage 4.1 needs to ask whether regression performance predicts BO "
        "performance on BOTH benchmarks rather than only LCBench.",
    )
    p.add_argument(
        "--pd1-source",
        type=str,
        default="full",
        choices=["lite", "full"],
        help="PD1 release to read when --benchmark pd1_loo. 'full' is the 23-task, "
        "400-matched-config public release.",
    )
    p.add_argument(
        "--pretrain-seed",
        type=int,
        default=0,
        help="Which pre-training prior to draw. Mirrors bo_experiment so a single OFAT "
        "cell can be replicated across priors on BOTH the BO and regression harnesses. "
        "Without it the --priors expansion passed an unrecognised flag to bo_diagnose "
        "and every regression cell crash-looped forever (round-7 R7-1).",
    )
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument(
        "--prior-cache",
        type=str,
        default=None,
        help="Directory for content-addressed pre-trained priors, shared with "
        "bo_experiment. Mirrors that flag exactly so a single OFAT cell can reuse one "
        "prior on BOTH harnesses. Beyond the speedup this is what makes hyperbo_frozen "
        "a real control on this harness: served from cache it is byte-identical across "
        "configs by construction, instead of merely expected to be (S28.6, S30.2).",
    )
    p.add_argument(
        "--prior-cache-mode",
        type=str,
        default="readwrite",
        choices=["off", "read", "write", "readwrite"],
        help="off disables the cache; read reproduces a previous run without writing.",
    )
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", type=str, default="/tmp/bo_diagnose.json")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    _boexp.COND_NOISE = args.cond_noise
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    obs_grid = (
        [args.n_obs]
        if args.n_obs is not None
        else [int(v) for v in args.n_obs_grid.split(",") if v.strip()]
    )

    # Same guard as bo_experiment: --iw-nu is only read inside `if use_covar_prior:`,
    # so accepting it alone would be a silent no-op (S27.16, no-op #7).
    if args.iw_nu is not None and not args.use_covar_prior:
        raise ValueError(
            "--iw-nu is only read when --use-covar-prior is set. Pass both, or drop "
            "--iw-nu. Refusing to accept a flag that would do nothing."
        )

    names, Xn, Y_all = _load_benchmark(args)

    # PD1 has 23 tasks against LCBench's 35, and only ~400 matched configs against
    # LCBench's 2000, so the LCBench defaults over-request on PD1. Clamp explicitly and
    # say so, rather than letting torch.randperm silently return a short split.
    n_tasks = len(names)
    if args.n_eval + args.n_pretrain > n_tasks:
        n_eval = min(args.n_eval, max(1, n_tasks // 4))
        n_pretrain = n_tasks - n_eval
        print(
            f"  clamped split to this suite: n_eval {args.n_eval}->{n_eval}, "
            f"n_pretrain {args.n_pretrain}->{n_pretrain} ({n_tasks} tasks available)",
            flush=True,
        )
        args.n_eval, args.n_pretrain = n_eval, n_pretrain
    if args.n_configs > Xn.shape[0]:
        print(
            f"  clamped n_configs {args.n_configs}->{Xn.shape[0]} (pool size)",
            flush=True,
        )
        args.n_configs = Xn.shape[0]

    # Same deterministic split convention as bo_experiment.main.
    # The split stays fixed at 42 so every prior sees the SAME tasks; only the
    # pre-training draw varies. Otherwise --pretrain-seed would confound "different
    # prior" with "different split" and the between-prior variance would absorb both.
    torch.manual_seed(42)
    cfg = torch.randperm(Xn.shape[0])[: args.n_configs]
    dperm = torch.randperm(len(names))
    eval_ix = dperm[: args.n_eval].tolist()
    pre_ix = dperm[args.n_eval : args.n_eval + args.n_pretrain].tolist()

    X_pool = Xn[cfg]
    Yc = [Y[cfg] for Y in Y_all]
    Ypool = torch.cat([Yc[i] for i in pre_ix])
    ymean, ystd = Ypool.mean(), Ypool.std().clamp_min(1e-8)
    # Reference/inducing set Z. When n_inducing < n_configs the EM prior is anchored on a
    # STRICT SUBSET of the pool, so evaluation points are off-grid and the interpolated
    # covariance keeps its Nystrom residual
    #     Sigma(X) = Lambda(X) + W Sigma_Z W^T,  Lambda(X) = K(X,X) - W K(Z,X).
    # Lambda is full rank and is what makes the continuous-domain EGP a universal
    # approximator. With Z == pool (the previous behaviour, and the default here for
    # backwards compatibility) every test point satisfies X in Z, so Lambda == 0 exactly
    # and the model collapses to the rank-(K-1) empirical covariance -- a hard ceiling
    # that no amount of target data can lift, and the likely reason EM's advantage
    # disappears as n_train grows. The paper itself specifies Z as a random subset of the
    # historical inputs, so tying them was also a fidelity bug.
    n_ind = args.n_inducing if args.n_inducing > 0 else args.n_configs
    n_ind = min(n_ind, args.n_configs)
    ind_ix = torch.randperm(args.n_configs, generator=torch.Generator().manual_seed(7))[
        :n_ind
    ]
    # NOTE: the reference set is passed separately via `inducing_points`; every method
    # still OBSERVES all n_configs points per task. An earlier version subsetted the
    # datasets themselves, which silently cut the pre-training data for every method
    # (HyperBO's NLL blew up to 12.3 at M=30) and confounded the off-grid question with
    # a data reduction.
    X_ind = X_pool[ind_ix]
    if args.hetero_frac < 1.0:
        # HETEROGENEOUS OBSERVATIONS: every historical task is observed at its OWN random
        # subset of the pool, so no two tasks share a grid. This is the regime the
        # continuous-domain EM construction is actually built for -- sparse, irregular,
        # per-task observation locations resolved onto a shared reference set by the
        # E-step -- and it is the regime NEITHER LCBench nor PD1 tests by default, since
        # both are dense shared grids where a parametric deep kernel is strongest and the
        # empirical prior's distinctive machinery is idle. Evaluation points are also
        # off-grid here, so the Nystrom residual Lambda is non-zero and the canonical
        # kernel does real work (on-grid it is provably a no-op).
        k_obs = max(4, int(round(args.hetero_frac * args.n_configs)))
        pretrain_ds = []
        for i in pre_ix:
            gi = torch.Generator().manual_seed(20011 + 97 * i)
            sub = torch.randperm(args.n_configs, generator=gi)[:k_obs]
            pretrain_ds.append(
                ExperimentDataset(X=X_pool[sub], Y=((Yc[i] - ymean) / ystd)[sub])
            )
        print(
            f"  heterogeneous: {k_obs}/{args.n_configs} configs per task, "
            "a different subset for each (no shared grid)",
            flush=True,
        )
    else:
        pretrain_ds = [
            ExperimentDataset(X=X_pool, Y=(Yc[i] - ymean) / ystd) for i in pre_ix
        ]
    print(
        f"  inducing set: {n_ind} of {args.n_configs} pool configs "
        f"({'ON-grid, Lambda==0' if n_ind == args.n_configs else 'OFF-grid, Lambda>0'})",
        flush=True,
    )

    # Re-seed with the prior index so --pretrain-seed actually draws a DIFFERENT prior.
    # Setting the flag without this would have been a silent no-op: it would appear in
    # the recorded config while every prior produced identical numbers, which is the
    # worst possible outcome for a replication axis (round-7).
    torch.manual_seed(args.pretrain_seed * 100003 + 7)
    print(
        f"Fitting shared kernel + EM prior (pretrain_seed={args.pretrain_seed}) ...",
        flush=True,
    )
    out: dict = {"config": vars(args), "eval_datasets": [names[i] for i in eval_ix]}
    hyperbo_prior_early = None
    # --em-mean needs a HyperBO prior too, not just --em-canonical hyperbo. Gating on
    # the canonical flag alone made every --em-mean run on this harness raise, which is
    # what would have aborted six OFAT cells on both regression regimes (round-4 R5/R6).
    if args.em_canonical == "hyperbo" or args.em_mean != "empirical":
        # Inverts the usual order (EM first): HyperBO's pre-trained kernel becomes
        # EM's canonical/interpolation kernel. Reused below so 25k steps are paid
        # once. Expected to be a NO-OP on-grid, where Lambda == 0 and W == I.
        print("Pre-training HyperBO first (its kernel seeds EM) ...", flush=True)
        # Re-seed per baseline so its init does not depend on how many RNG draws
        # upstream code consumed. Without this, configs that build extra
        # parameterised modules (--em-base-mean linear, --deep-kernel) shift the
        # stream and silently move `hyperbo_frozen`, which is supposed to be a
        # bit-identical control across configs (§27.38).
        torch.manual_seed(args.pretrain_seed * 1009 + 11)
        hyperbo_prior_early = prior_cache.cached(
            args.prior_cache,
            args.prior_cache_mode,
            "hyperbo",
            pretrain_ds,
            _diag_hyperbo_params(args, int(X_pool.shape[-1])),
            lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_hyperbo(
                datasets=_ds,
                input_dim=_dim,
                hidden_dims=args.hyperbo_hidden,
                loss_type="NLL",
                num_iterations=args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
                task_batch_size=args.meta_task_batch,
            ),
        )
    mean_s, covar_s = build_shared_kernel(
        pretrain_ds, deep_hidden=args.deep_kernel, base_mean=args.em_base_mean
    )
    if hyperbo_prior_early is not None:
        covar_s = hyperbo_kernel_from_prior(hyperbo_prior_early)
    # Mean transfer, mirroring bo_experiment. hyperbo_prior_early is the only HyperBO
    # prior available at this point, so --em-mean requires --em-canonical hyperbo here;
    # _apply_em_mean raises rather than silently falling back if it is absent.
    mean_s = _boexp._apply_em_mean(mean_s, args, hyperbo_prior_early)
    em_prior = pretrain_em_prior(
        datasets=pretrain_ds,
        mean_module=mean_s,
        covar_module=covar_s,
        likelihood_noise=torch.tensor(args.em_likelihood_noise, dtype=torch.double),
        covariance_shrinkage=args.em_shrinkage,
        num_em_iterations=args.n_em,
        enable_interpolation=True,
        inducing_points=X_ind if n_ind < args.n_configs else None,
        iw_nu=args.iw_nu,
        use_mean_prior=args.use_mean_prior,
        use_covar_prior=args.use_covar_prior,
        init_mode=args.em_init_mode,
    )

    # ---- 0. EM prior covariance spectrum ---------------------------------------
    # The M-step covariance is a sum of K rank-1 task deviations plus the E-step
    # posterior term, so with COMPLETE pre-training data it is essentially rank K-1 in
    # an n_configs-dimensional space. On a shared on-grid pool the Nystrom residual
    # vanishes (W ~ I), so whatever variance is missing here is missing at the BO
    # candidates too -- which is the structural cause of frozen-EM overconfidence.
    Sigma = em_prior.Sigma_inducing.detach()
    evals = torch.linalg.eigvalsh(Sigma).flip(0).clamp_min(0)
    tr = float(evals.sum())
    cum = torch.cumsum(evals, 0) / max(tr, 1e-300)
    eff_rank = float(
        torch.exp(
            -(
                (evals / max(tr, 1e-300)).clamp_min(1e-300)
                * (evals / max(tr, 1e-300)).clamp_min(1e-300).log()
            ).sum()
        )
    )  # participation ratio (exp of spectral entropy)
    n_above = int((evals > 1e-8 * evals[0]).sum())
    out["em_covariance_spectrum"] = {
        "shrinkage": args.em_shrinkage,
        "dim": int(Sigma.shape[-1]),
        "n_pretrain_tasks": len(pre_ix),
        "trace": tr,
        "effective_rank_entropy": eff_rank,
        "n_eigs_above_1e-8_rel": n_above,
        "frac_trace_top22": float(cum[min(21, len(cum) - 1)]),
        "top_eigs": [float(v) for v in evals[:10]],
        "median_eig": float(evals[len(evals) // 2]),
        "min_eig": float(evals[-1]),
    }
    print(f"\n=== EM prior covariance spectrum (shrinkage={args.em_shrinkage}) ===")
    print(f"  dim={Sigma.shape[-1]}  pretrain tasks K={len(pre_ix)}  trace={tr:.3f}")
    print(f"  effective rank (spectral entropy) : {eff_rank:.1f}")
    print(f"  #eigenvalues > 1e-8 x lambda_max  : {n_above}")
    print(
        f"  fraction of trace in top 22 modes : {float(cum[min(21, len(cum) - 1)]):.4f}"
    )
    print(
        f"  lambda_max={float(evals[0]):.4g}  median={float(evals[len(evals) // 2]):.4g}"
        f"  min={float(evals[-1]):.4g}"
    )

    hyperbo_prior = pacoh_prior = ablr_prior = None

    def _publish_hyperbo_base(prior) -> None:
        """Expose a pre-trained HyperBO kernel to the em_additive_hyperbo* variants.

        make_surrogate reads bo_experiment.HYPERBO_BASE_KERNEL at call time; without this
        those methods raise. Kept separate from EM_CANONICAL on purpose -- the canonical
        kernel drives interpolation, the shrinkage target and EM init, whereas this is the
        additive door at conditioning time.
        """
        if prior is not None:
            _boexp.HYPERBO_BASE_KERNEL = _boexp.hyperbo_kernel_from_prior(prior)

    if hyperbo_prior_early is not None:
        _publish_hyperbo_base(hyperbo_prior_early)
    if any(m.startswith("hyperbo") for m in methods):
        t = time.time()
        print("Pre-training HyperBO ...", flush=True)
        # Re-seed per baseline so its init does not depend on how many RNG draws
        # upstream code consumed. Without this, configs that build extra
        # parameterised modules (--em-base-mean linear, --deep-kernel) shift the
        # stream and silently move `hyperbo_frozen`, which is supposed to be a
        # bit-identical control across configs (§27.38).
        torch.manual_seed(args.pretrain_seed * 1009 + 11)
        if hyperbo_prior_early is not None:
            hyperbo_prior = hyperbo_prior_early
        else:
            hyperbo_prior = prior_cache.cached(
                args.prior_cache,
                args.prior_cache_mode,
                "hyperbo",
                pretrain_ds,
                _diag_hyperbo_params(args, int(X_pool.shape[-1])),
                lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_hyperbo(
                    datasets=_ds,
                    input_dim=_dim,
                    hidden_dims=args.hyperbo_hidden,
                    loss_type="NLL",
                    num_iterations=args.meta_iters,
                    learning_rate=args.meta_lr,
                    subsample_size=args.meta_subsample,
                    task_batch_size=args.meta_task_batch,
                ),
            )
        _publish_hyperbo_base(hyperbo_prior)
        print(f"  done in {time.time() - t:.1f}s", flush=True)
    if any(m.startswith("pacoh") for m in methods):
        t = time.time()
        print("Pre-training PACOH-GP ...", flush=True)
        # Re-seed per baseline so its init does not depend on how many RNG draws
        # upstream code consumed. Without this, configs that build extra
        # parameterised modules (--em-base-mean linear, --deep-kernel) shift the
        # stream and silently move `hyperbo_frozen`, which is supposed to be a
        # bit-identical control across configs (§27.38).
        torch.manual_seed(args.pretrain_seed * 1009 + 13)
        _pacoh_params = {
            "input_dim": int(X_pool.shape[-1]),
            "num_particles": args.pacoh_particles,
            "hidden_dims": str(args.hyperbo_hidden),
            "num_iterations": args.meta_iters,
            "learning_rate": args.meta_lr,
            "subsample_size": args.meta_subsample,
            "task_batch_size": args.meta_task_batch,
            "pretrain_seed": args.pretrain_seed,
        }
        pacoh_prior = prior_cache.cached(
            args.prior_cache,
            args.prior_cache_mode,
            "pacoh",
            pretrain_ds,
            _pacoh_params,
            lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_pacoh_gp(
                datasets=_ds,
                input_dim=_dim,
                config=PACOHGPConfig(
                    num_particles=args.pacoh_particles,
                    hidden_dims=args.hyperbo_hidden,
                ),
                num_iterations=args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
                task_batch_size=args.meta_task_batch,
            ),
        )
        print(f"  done in {time.time() - t:.1f}s", flush=True)
    if "ablr" in methods:
        t = time.time()
        print("Pre-training ABLR ...", flush=True)
        # Re-seed per baseline so its init does not depend on how many RNG draws
        # upstream code consumed. Without this, configs that build extra
        # parameterised modules (--em-base-mean linear, --deep-kernel) shift the
        # stream and silently move `hyperbo_frozen`, which is supposed to be a
        # bit-identical control across configs (§27.38).
        torch.manual_seed(args.pretrain_seed * 1009 + 17)
        _ablr_params = {
            "input_dim": int(X_pool.shape[-1]),
            "hidden": str(args.hyperbo_hidden),
            "num_iterations": args.meta_iters,
            "learning_rate": args.meta_lr,
            "subsample_size": args.meta_subsample,
            "pretrain_seed": args.pretrain_seed,
        }
        ablr_prior = prior_cache.cached(
            args.prior_cache,
            args.prior_cache_mode,
            "ablr",
            pretrain_ds,
            _ablr_params,
            lambda _ds=pretrain_ds, _dim=int(X_pool.shape[-1]): pretrain_ablr(
                datasets=_ds,
                input_dim=_dim,
                hidden=args.hyperbo_hidden,
                num_iterations=args.meta_iters,
                learning_rate=args.meta_lr,
                subsample_size=args.meta_subsample,
            ),
        )
        print(f"  done in {time.time() - t:.1f}s", flush=True)

    def build(method, X, Y):
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
        )

    def _find_lengthscale(kern):
        """Walk down to the first kernel that actually carries a lengthscale.

        Kernel nesting differs by canonical choice: a plain ScaleKernel(Matern) exposes
        the Matern at .base_kernel, but HyperBODeepKernel's .base_kernel is itself a
        ScaleKernel whose lengthscale is None (the Matern is one level deeper). This is
        a reporting field only, so a miss must not abort the run.
        """
        seen = set()
        stack = [kern]
        while stack:
            k = stack.pop()
            if k is None or id(k) in seen:
                continue
            seen.add(id(k))
            val = getattr(k, "lengthscale", None)
            if val is not None:
                return val
            for attr in ("base_kernel", "base", "kernels"):
                sub = getattr(k, attr, None)
                if sub is None:
                    continue
                stack.extend(sub if isinstance(sub, (list, tuple)) else [sub])
        return None

    ls_t = _find_lengthscale(covar_s)
    out["shared_kernel"] = {
        "lengthscales": (
            [round(float(x), 4) for x in ls_t.detach().reshape(-1)]
            if ls_t is not None
            else None
        ),
        "outputscale": (
            float(covar_s.outputscale)
            if getattr(covar_s, "outputscale", None) is not None
            else None
        ),
        # Only a ConstantMean has `.constant`; under --em-base-mean linear or
        # --em-mean this attribute does not exist (review, integration #5).
        "mean_constant": (
            float(mean_s.constant)
            if getattr(mean_s, "constant", None) is not None
            else None
        ),
        "mean_type": type(mean_s).__name__,
    }
    print("\n=== Shared canonical kernel (normalized inputs, standardized targets) ===")
    sk = out["shared_kernel"]
    print(f"lengthscales : {sk['lengthscales']}")
    print(f"outputscale  : {sk['outputscale']}  (target var ~1.0)")
    print(
        "mean constant: "
        + (
            f"{sk['mean_constant']:.4f}"
            if sk["mean_constant"] is not None
            else f"n/a ({sk.get('mean_type', 'non-constant mean')})"
        )
    )

    # ---- 2. prior-mean informativeness, per method -----------------------------
    print("\n=== Prior-mean informativeness (no observations at all) ===")
    print(
        "  spearman  : rank correlation of the PRIOR mean with the truth (whole pool)"
    )
    print("  top1_reg  : regret of the argmax-of-prior-mean pick (std units) -- the")
    print("              zero-shot BO pick, which depends on the mean's EXTREMES")
    print("  top10_prec: overlap of the prior's top-10 with the true top-10")
    print(
        "  constant-mean models (vanilla, pretrained_gp, ABLR's BLR head) score ~0.\n"
    )
    dummy_X = X_pool[:1]
    dummy_Y = torch.zeros(1, 1, dtype=X_pool.dtype)
    prior_rows: dict[str, list[tuple[float, float, float]]] = {m: [] for m in methods}
    for m in methods:
        try:
            model = build(m, dummy_X, dummy_Y)
            pmean, _ = prior_moments(model, X_pool)
        except Exception as e:  # a method that cannot be built without data
            print(f"  {m:<22} unavailable ({type(e).__name__})", flush=True)
            continue
        for ei in eval_ix:
            dY = ((Yc[ei] - ymean) / ystd).reshape(-1)
            sp = spearman(pmean, dY)
            top1 = float(dY.max() - dY[int(pmean.argmax())])
            k = min(10, dY.numel())
            pred_top = set(torch.topk(pmean, k).indices.tolist())
            true_top = set(torch.topk(dY, k).indices.tolist())
            prior_rows[m].append((sp, top1, len(pred_top & true_top) / k))
    out["prior_mean"] = {}
    print(f"{'method':<24}{'spearman':>10}{'top1_reg':>11}{'top10_prec':>12}")
    for m in methods:
        vals = prior_rows[m]
        if not vals:
            continue
        sp, t1, tp = (sum(v[j] for v in vals) / len(vals) for j in range(3))
        out["prior_mean"][m] = {
            "spearman": sp,
            "top1_regret_std": t1,
            "top10_precision": tp,
            "per_dataset_spearman": dict(
                zip([names[i] for i in eval_ix], [v[0] for v in vals])
            ),
        }
        print(f"{m:<24}{sp:>+10.3f}{t1:>11.3f}{tp:>12.2f}")

    # ---- 3. ranking quality + calibration vs #observations ----------------------
    out["by_n_obs"] = {}
    for n_obs in obs_grid:
        # Every metric here is scored on the HELD-OUT pool, so a grid point that
        # consumes the whole pool has nothing to score and is skipped rather than
        # producing an empty (and float-dtyped) index tensor.
        if n_obs >= args.n_configs:
            print(
                f"\n=== skipping n_obs={n_obs}: pool is only {args.n_configs} "
                "configs, no held-out points ==="
            )
            continue
        print(f"\n=== After {n_obs} random observations (unobserved pool) ===")
        print(
            f"{'method':<24}{'rank_corr':>10}{'sigma':>9}{'RMSE':>9}"
            f"{'calib':>8}{'NLL':>10}{'cov95':>8}"
        )
        agg: dict[str, list[tuple[float, ...]]] = {m: [] for m in methods}
        for ei in eval_ix:
            dY = ((Yc[ei] - ymean) / ystd).reshape(-1)
            # The prior index MUST enter the observation subset. EM pre-training is
            # deterministic, so without this term every EM method returns bit-identical
            # output across all 9 priors, between-prior variance is exactly 0, and every
            # paired t is 0/0 -- a dead replication axis that silently fabricates
            # significance. bo_experiment.py:3240 keys its initial design on the
            # prior for
            # this reason; this harness never received the fix (§28.6).
            g = torch.Generator().manual_seed(1000 * ei + 100003 * args.pretrain_seed)
            obs = torch.randperm(args.n_configs, generator=g)[:n_obs].tolist()
            rest = torch.tensor(
                [i for i in range(args.n_configs) if i not in set(obs)],
                dtype=torch.long,
            )
            Xo, Yo = X_pool[obs], dY[obs].unsqueeze(-1)
            true = dY[rest]
            for m in methods:
                try:
                    model = build(m, Xo, Yo)
                    with torch.no_grad():
                        pmean, psig = _posterior_moments(model, X_pool[rest])
                        # ALSO record the noise-inclusive predictive. A GP's
                        # fitted sigma^2 on a deterministic benchmark is not
                        # measurement noise -- it is misspecification the kernel
                        # could not explain, and that structure is present at TEST
                        # points too. The model's own belief about a new observable
                        # is N(mu, k** + sigma^2), so scoring only N(mu, k**) scores
                        # a distribution the model does not claim describes the data
                        # (S27.33). Which one is "right" depends on whether the
                        # question is "how well is f known" (latent) or "how well is
                        # the next observation predicted" (noise-inclusive), so
                        # record BOTH rather than baking the choice into the corpus.
                        try:
                            pn = model.posterior(X_pool[rest], observation_noise=True)
                            # Mirror _posterior_moments: PACOH returns a
                            # GaussianMixturePosterior whose .variance keeps the
                            # per-particle MCMC dim, so .reshape(-1) yields
                            # n_particles * n_points (10 * 195 = 1950) against a
                            # 195-element pmean. The latent path marginalizes here and
                            # this one did not, so every PD1/LCBench regression cell
                            # that reached PACOH died at the nll_y division -- outside
                            # the handler below, so it took the whole run with it.
                            if hasattr(pn, "mixture_variance"):
                                var_y = pn.mixture_variance
                            else:
                                var_y = pn.variance
                            psig_y = var_y.clamp_min(1e-12).sqrt().reshape(-1)
                        except Exception as exc_y:  # noqa: BLE001
                            # Report rather than swallow. When this fires, nll_obs
                            # silently FALLS BACK to the latent nll below, so the
                            # corpus records a "noise-inclusive" number that is really
                            # the latent one and is indistinguishable from a genuine
                            # equality. ABLRSurrogate.posterior() takes no
                            # observation_noise kwarg, so ablr hits this on every cell.
                            print(
                                f"    WARNING: {m} noise-inclusive predictive "
                                f"unavailable, nll_obs will equal nll: "
                                f"{type(exc_y).__name__}: {exc_y}"
                            )
                            psig_y = None
                except Exception as exc:  # noqa: BLE001
                    # Silently dropping a method made it vanish from the results table
                    # and averaged its competitors over a different subset of eval
                    # tasks, so the rows were not comparable (D6).
                    print(
                        f"    WARNING: {m} failed on this eval task: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                rc = spearman(pmean, true)
                rmse = float(((pmean - true) ** 2).mean().sqrt())
                sig = float(psig.mean())
                z = (true - pmean) / psig.clamp_min(1e-9)
                nll = float(
                    (
                        0.5 * math.log(2 * math.pi)
                        + psig.clamp_min(1e-9).log()
                        + 0.5 * z**2
                    ).mean()
                )
                cov = float((z.abs() <= 1.96).double().mean())
                # Variance recalibration: rescale the predictive sd by a single optimal
                # factor tau and recompute NLL. tau^2 = mean(z^2) is the ML scale, so
                # nll_recal isolates how much of the NLL gap is pure variance
                # miscalibration (which a post-hoc scaling could fix) from how much is
                # structural -- a wrong predictive MEAN, which no rescaling can repair.
                tau2 = float((z**2).mean().clamp_min(1e-12))
                nll_recal = float(
                    (
                        0.5 * math.log(2 * math.pi)
                        + psig.clamp_min(1e-9).log()
                        + 0.5 * math.log(tau2)
                        + 0.5 * z**2 / tau2
                    ).mean()
                )
                # Same metrics under the noise-inclusive predictive, so the corpus can
                # be read either way without a re-run.
                if psig_y is not None:
                    zy = (true - pmean) / psig_y.clamp_min(1e-9)
                    nll_y = float(
                        (
                            0.5 * math.log(2 * math.pi)
                            + psig_y.clamp_min(1e-9).log()
                            + 0.5 * zy**2
                        ).mean()
                    )
                    sig_y = float(psig_y.mean())
                    cov_y = float((zy.abs() <= 1.96).double().mean())
                else:
                    nll_y, sig_y, cov_y = nll, sig, cov
                agg[m].append(
                    (
                        rc,
                        sig,
                        rmse,
                        nll,
                        cov,
                        nll_recal,
                        math.sqrt(tau2),
                        nll_y,
                        sig_y,
                        cov_y,
                    )
                )
        rows = {}
        for m in methods:
            if not agg[m]:
                continue
            k = len(agg[m])
            rc, sig, rmse, nll, cov, nll_recal, tau = (
                sum(x[j] for x in agg[m]) / k for j in range(7)
            )
            rows[m] = {
                "rank_corr": rc,
                "mean_sigma": sig,
                "rmse": rmse,
                "calib_ratio": sig / max(rmse, 1e-9),
                "nll": nll,
                "coverage95": cov,
                "nll_recalibrated": nll_recal,
                "tau": tau,
                # Per-eval-dataset values so downstream analysis can compute a
                # dataset-clustered SEM and a paired test. Runs on the same dataset
                # share a historical corpus and are NOT independent, so a pooled
                # standard error over runs would overstate significance.
                "per_dataset_nll": [x[3] for x in agg[m]],
                "per_dataset_rmse": [x[2] for x in agg[m]],
                # Noise-inclusive counterparts (S27.33). `nll`/`mean_sigma`/
                # `coverage95` above use the LATENT posterior; these use
                # N(mu, k** + sigma^2), the model's own predictive for an observable.
                "nll_obs": _mean(agg[m], 7),
                "mean_sigma_obs": _mean(agg[m], 8),
                "coverage95_obs": _mean(agg[m], 9),
                "calib_ratio_obs": _mean(agg[m], 8) / max(_mean(agg[m], 2), 1e-9),
            }
            print(
                f"{m:<24}{rc:>+10.3f}{sig:>9.4f}{rmse:>9.4f}"
                f"{sig / max(rmse, 1e-9):>8.2f}{nll:>10.2f}{cov:>8.2f}"
            )
        out["by_n_obs"][str(n_obs)] = rows

    _boexp.atomic_write_json(args.out, out)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
