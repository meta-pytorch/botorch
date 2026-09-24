#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""1D learning-curve extrapolation on LCBench: empirical-GP vs HyperBO vs PACOH.

Setup
-----
Every LCBench (dataset, hyperparameter config) pair has a validation-accuracy learning
curve over epochs. We meta-learn a prior from a set of *complete* historical curves, then
for each held-out curve condition the surrogate on its first ``context_frac`` fraction of
epochs and extrapolate the remaining epochs. This is the 1D setting the paper used PACOH
for; here we refresh it against the open-sourced empirical-GP and the meta-learning
baselines on a common footing.

We report, on the held-out *tail* epochs, RMSE / NLL / 95%-coverage, aggregated across
eval curves with dataset-clustered s.e.m.

Methods
-------
- em_1d          : EmpiricalOneDimensionalGP (empirical mean+cov from historical curves).
- hyperbo_frozen : HyperBO deep-kernel prior (input_dim=1), frozen (paper-faithful).
- hyperbo_adapt  : HyperBO, per-curve Gaussian-noise re-fit only.
- pacoh_frozen   : PACOH-GP SVGD mixture (input_dim=1), frozen (paper-faithful).
- pacoh_adapt    : PACOH, per-curve Gaussian-noise re-fit only.
- vanilla_gp     : SingleTaskGP fit on the observed prefix only (no transfer).

All meta-learners are pre-trained on the SAME standardized historical curves for a fair
comparison, and all methods extrapolate the SAME held-out prefixes.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from _scratch_bo.lcbench_io import (
    LCBENCH_DATASET_NAMES as DATASET_NAMES,
    load_lcbench_data,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps import (
    EmpiricalOneDimensionalGP,
    HyperBOModel,
    PACOHGPConfig,
    PACOHGPModel,
    pretrain_hyperbo,
    pretrain_pacoh_gp,
)
from botorch.models.empirical_gps.utils import ExperimentDataset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

METRIC = "Train/val_accuracy"
COND_NOISE = 1e-3  # frozen conditioning-likelihood noise (std units)


def load_curves(
    names: list[str], n_curves: int, seed: int = 0
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the shared epoch grid (E,1) and per-dataset curve tensors (n_curves,E)."""
    grid = None
    per_ds: list[torch.Tensor] = []
    g = torch.Generator().manual_seed(seed)
    for nm in names:
        data = load_lcbench_data(nm, METRIC, dtype=torch.double)
        curves = data.metrics  # (num_configs, num_epochs)
        n_epochs = curves.shape[-1]
        if grid is None:
            grid = torch.linspace(0, 1, n_epochs, dtype=torch.double).unsqueeze(-1)
        idx = torch.randperm(curves.shape[0], generator=g)[:n_curves]
        per_ds.append(curves[idx])  # (n_curves, num_epochs)
    return grid, per_ds


def _pred_moments(model, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginal predictive (mean, variance) as 1D tensors.

    Works uniformly for a standard GPyTorchPosterior and the PACOH
    GaussianMixturePosterior (both expose the marginalized ``mean``/``variance``).
    """
    with torch.no_grad():
        post = model.posterior(X)
        # PACOH returns a GaussianMixturePosterior (per-particle .mean/.variance keep
        # the MCMC dim); use the marginalized mixture moments. Standard posteriors
        # fall through to .mean/.variance.
        if hasattr(post, "mixture_mean"):
            mean = post.mixture_mean.reshape(-1)
            var = post.mixture_variance.clamp_min(1e-12).reshape(-1)
        else:
            mean = post.mean.reshape(-1)
            var = post.variance.clamp_min(1e-12).reshape(-1)
    return mean, var


def extrapolation_metrics(
    mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor
) -> tuple[float, float, float]:
    """Held-out-tail RMSE, Gaussian NLL, and 95% central-interval coverage."""
    y = y.reshape(-1)
    rmse = float(((mean - y) ** 2).mean().sqrt())
    nll = float(0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var).mean())
    z = 1.959963984540054
    covered = ((y - mean).abs() <= z * var.sqrt()).double().mean()
    return rmse, nll, float(covered)


def make_surrogate(
    method: str,
    prefix_X: torch.Tensor,
    prefix_Y: torch.Tensor,
    hist_X: torch.Tensor,
    hist_Y: torch.Tensor,
    hyperbo_prior=None,
    pacoh_prior=None,
):
    """Build a 1D-curve surrogate conditioned on a prefix (standardized units)."""
    if method == "em_1d":
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        model = EmpiricalOneDimensionalGP(
            train_X=prefix_X,
            train_Y=prefix_Y,
            historical_X=hist_X,
            historical_Y=hist_Y.unsqueeze(-1),  # (num_curves, E, 1)
            likelihood=lik,
        )
        model.eval()
        return model

    if method in ("hyperbo_frozen", "hyperbo_adapt"):
        model = HyperBOModel.from_pretrained(
            hyperbo_prior, train_X=prefix_X, train_Y=prefix_Y, freeze_pretrained=True
        )
        if method == "hyperbo_adapt":
            model.likelihood.raw_noise.requires_grad_(True)
            model.train()
            model.likelihood.train()
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass
        model.eval()
        model.likelihood.eval()
        return model

    if method in ("pacoh_frozen", "pacoh_adapt"):
        model = PACOHGPModel.from_pretrained(
            pacoh_prior, train_X=prefix_X, train_Y=prefix_Y, freeze_pretrained=True
        )
        if method == "pacoh_adapt":
            model.likelihood.raw_noise.requires_grad_(True)
            model.train()
            Yt = prefix_Y.squeeze(-1)
            opt = torch.optim.Adam([model.likelihood.raw_noise], lr=0.1, foreach=True)
            try:
                for _ in range(50):
                    opt.zero_grad()
                    out = model.likelihood(model.forward(prefix_X))
                    loss = -out.log_prob(Yt).sum()
                    loss.backward()
                    opt.step()
            except Exception:
                pass
        model.eval()
        model.likelihood.eval()
        return model

    if method == "vanilla_gp":
        gp = SingleTaskGP(prefix_X, prefix_Y, outcome_transform=None)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        gp.eval()
        return gp

    raise ValueError(f"unknown method {method}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-curves", type=int, default=100, help="curves per dataset")
    p.add_argument("--n-eval", type=int, default=4, help="held-out eval datasets")
    p.add_argument("--n-pretrain", type=int, default=15, help="pretrain datasets")
    p.add_argument("--n-eval-curves", type=int, default=30, help="eval curves per ds")
    p.add_argument("--context-frac", type=float, default=0.3)
    p.add_argument(
        "--hyperbo-hidden",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(32, 32),
    )
    p.add_argument("--meta-iters", type=int, default=2000)
    p.add_argument("--meta-lr", type=float, default=1e-3)
    p.add_argument("--meta-subsample", type=int, default=None)
    p.add_argument("--meta-task-batch", type=int, default=None)
    p.add_argument("--pacoh-particles", type=int, default=10)
    p.add_argument("--pacoh-hyper-prior-std", type=float, default=1.0)
    p.add_argument("--pacoh-svgd-lengthscale", type=float, default=None)
    p.add_argument(
        "--hyperbo-loss",
        type=str,
        default="NLL",
        choices=["NLL", "EKL"],
        help="HyperBO pre-training objective. EKL is the paper's empirical-KL loss "
        "(valid here because all curves share the epoch grid); it is a training "
        "objective only and does not alter targets or the model's Gaussian nature.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="em_1d,hyperbo_frozen,hyperbo_adapt,pacoh_frozen,pacoh_adapt,vanilla_gp",
    )
    p.add_argument("--out", type=str, default="/tmp/curve_results.json")
    args = p.parse_args()

    if args.smoke:
        args.n_curves = 40
        args.n_eval = 2
        args.n_pretrain = 6
        args.n_eval_curves = 8
        args.meta_iters = min(args.meta_iters, 50)
        args.pacoh_particles = min(args.pacoh_particles, 4)
    methods = args.methods.split(",")

    names = list(DATASET_NAMES)
    print(f"Loading {len(names)} LCBench datasets ...", flush=True)
    grid, per_ds = load_curves(names, args.n_curves)
    n_epochs = grid.shape[0]

    torch.manual_seed(42)
    dperm = torch.randperm(len(names))
    eval_ix = dperm[: args.n_eval].tolist()
    pre_ix = dperm[args.n_eval : args.n_eval + args.n_pretrain].tolist()

    # Standardize with the pretrain-curve pool statistics.
    pool = torch.cat([per_ds[i].reshape(-1) for i in pre_ix])
    ymean, ystd = pool.mean(), pool.std().clamp_min(1e-8)

    hist_curves = torch.cat([per_ds[i] for i in pre_ix], dim=0)  # (C, E)
    hist_Y = (hist_curves - ymean) / ystd  # standardized (C, E)
    # Meta-learners consume one ExperimentDataset per historical curve.
    pretrain_ds = [
        ExperimentDataset(X=grid, Y=hist_Y[c].unsqueeze(-1))
        for c in range(hist_Y.shape[0])
    ]
    print(
        f"Pretrain on {len(pre_ix)} datasets ({hist_Y.shape[0]} curves); "
        f"evaluate on {len(eval_ix)}: {', '.join(names[i] for i in eval_ix)}",
        flush=True,
    )

    hyperbo_prior = None
    pacoh_prior = None
    hyperbo_pretrain_s = 0.0
    pacoh_pretrain_s = 0.0
    if any(m.startswith("hyperbo") for m in methods):
        t1 = time.time()
        print("Pre-training HyperBO deep-kernel prior (1D) ...", flush=True)
        hyperbo_prior = pretrain_hyperbo(
            datasets=pretrain_ds,
            input_dim=1,
            hidden_dims=args.hyperbo_hidden,
            loss_type=args.hyperbo_loss,
            num_iterations=args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
        )
        hyperbo_pretrain_s = time.time() - t1
        print(f"  done in {hyperbo_pretrain_s:.1f}s", flush=True)
    if any(m.startswith("pacoh") for m in methods):
        t1 = time.time()
        print(
            f"Pre-training PACOH-GP prior (1D, K={args.pacoh_particles}) ...",
            flush=True,
        )
        pacoh_prior = pretrain_pacoh_gp(
            datasets=pretrain_ds,
            input_dim=1,
            config=PACOHGPConfig(
                num_particles=args.pacoh_particles,
                hidden_dims=args.hyperbo_hidden,
                hyper_prior_std=args.pacoh_hyper_prior_std,
                svgd_length_scale=args.pacoh_svgd_lengthscale,
            ),
            num_iterations=args.meta_iters,
            learning_rate=args.meta_lr,
            subsample_size=args.meta_subsample,
            task_batch_size=args.meta_task_batch,
        )
        pacoh_pretrain_s = time.time() - t1
        print(f"  done in {pacoh_pretrain_s:.1f}s", flush=True)

    def surrogate_fn(method, pX, pY):
        return make_surrogate(method, pX, pY, grid, hist_Y, hyperbo_prior, pacoh_prior)

    k = max(1, int(round(args.context_frac * n_epochs)))
    prefix_X = grid[:k]  # (k, 1)
    tail_X = grid[k:]  # (E-k, 1)

    # metrics[method] = list over eval curves of (rmse, nll, coverage); parallel
    # cluster[method] tags each with its eval-dataset index for clustered s.e.m.
    metrics: dict[str, list[tuple[float, float, float]]] = {m: [] for m in methods}
    cluster: dict[str, list[int]] = {m: [] for m in methods}
    method_infer_s: dict[str, float] = {m: 0.0 for m in methods}

    t_start = time.time()
    for ei in eval_ix:
        ds_curves = (per_ds[ei] - ymean) / ystd  # (n_curves, E) standardized
        g = torch.Generator().manual_seed(1000 * ei)
        sel = torch.randperm(ds_curves.shape[0], generator=g)[: args.n_eval_curves]
        for c in sel.tolist():
            curve = ds_curves[c]  # (E,)
            pX, pY = prefix_X, curve[:k].unsqueeze(-1)  # (k,1)
            y_tail = curve[k:]
            for m in methods:
                _t_m = time.time()
                model = surrogate_fn(m, pX, pY)
                mean, var = _pred_moments(model, tail_X)
                method_infer_s[m] += time.time() - _t_m
                metrics[m].append(extrapolation_metrics(mean, var, y_tail))
                cluster[m].append(ei)
        print(f"  done dataset {names[ei]} | {time.time() - t_start:.0f}s", flush=True)

    # Aggregate with dataset-clustered s.e.m. (cluster = eval dataset).
    def clustered(vals: list[float], clu: list[int]) -> tuple[float, float]:
        by: dict[int, list[float]] = {}
        for v, c in zip(vals, clu):
            by.setdefault(c, []).append(v)
        per = [sum(x) / len(x) for x in by.values()]
        mean = sum(per) / len(per)
        if len(per) > 1:
            sd = (sum((x - mean) ** 2 for x in per) / (len(per) - 1)) ** 0.5
            sem = sd / (len(per) ** 0.5)
        else:
            sem = float("nan")
        return mean, sem

    summary = {}
    for m in methods:
        rmses = [t[0] for t in metrics[m]]
        nlls = [t[1] for t in metrics[m]]
        covs = [t[2] for t in metrics[m]]
        rmse_m, rmse_s = clustered(rmses, cluster[m])
        nll_m, nll_s = clustered(nlls, cluster[m])
        cov_m, cov_s = clustered(covs, cluster[m])
        summary[m] = {
            "rmse": rmse_m,
            "rmse_sem": rmse_s,
            "nll": nll_m,
            "nll_sem": nll_s,
            "coverage": cov_m,
            "coverage_sem": cov_s,
            "n_curves": len(rmses),
        }

    out = {
        "config": vars(args),
        "eval_datasets": [names[i] for i in eval_ix],
        "context_frac": args.context_frac,
        "summary": summary,
        "timings": {
            "hyperbo_pretrain_s": hyperbo_pretrain_s,
            "pacoh_pretrain_s": pacoh_pretrain_s,
            "meta_iters": args.meta_iters,
            "hyperbo_loss": args.hyperbo_loss,
            "method_infer_s": method_infer_s,
            "n_curves_evaluated": sum(len(v) for v in metrics.values())
            // max(len(methods), 1),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"\n=== 1D LCBench learning-curve extrapolation "
        f"(context={args.context_frac:.0%}, {n_epochs - k}/{n_epochs} tail epochs) ==="
    )
    print(f"{'method':<16}{'RMSE':>16}{'NLL':>16}{'95% cov':>16}")
    for m in methods:
        s = summary[m]
        print(
            f"{m:<16}{s['rmse']:.3f}\u00b1{s['rmse_sem']:.3f}".rjust(16)
            + f"{s['nll']:.3f}\u00b1{s['nll_sem']:.3f}".rjust(16)
            + f"{s['coverage']:.2f}\u00b1{s['coverage_sem']:.2f}".rjust(16)
        )
    print(
        f"\n=== Timing (meta_iters={args.meta_iters}, "
        f"hyperbo_loss={args.hyperbo_loss}) ===\n"
        f"pretrain: hyperbo={hyperbo_pretrain_s:.1f}s  pacoh={pacoh_pretrain_s:.1f}s"
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
