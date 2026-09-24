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
import json
import time
from typing import Callable

import torch
from _scratch_bo.lcbench_io import (
    LCBENCH_DATASET_NAMES as DATASET_NAMES,
    load_lcbench_data,
)
from botorch.acquisition.analytic import _log_ei_helper
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps import EMEmpiricalGaussianProcess, pretrain_em_prior
from botorch.models.empirical_gps.em_empirical_gp import build_shared_gp_model_list
from botorch.models.empirical_gps.utils import ExperimentDataset
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

METRIC = "Train/val_accuracy"
COND_NOISE = (
    1e-3  # frozen conditioning-likelihood noise (std units); obs are ~noiseless
)


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


def build_shared_kernel(pretrain_ds: list[ExperimentDataset]):
    mean = ConstantMean()
    covar = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=7,
            lengthscale_constraint=LogTransformedInterval(
                0.01, 100.0, initial_value=1.0
            ),
        ),
        outputscale_constraint=LogTransformedInterval(0.01, 100.0, initial_value=1.0),
    )
    _, mll = build_shared_gp_model_list(pretrain_ds, mean, covar)
    fit_gpytorch_mll(mll)
    return mean, covar


def make_surrogate(
    method: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    em_prior,
    mean_s,
    covar_s,
):
    """Build (and, where applicable, fit) a surrogate on observed (X, Y std-units)."""
    if method in ("em_frozen", "em_finetuned"):
        lik = GaussianLikelihood()
        lik.noise = torch.tensor(COND_NOISE, dtype=torch.double)
        # em_finetuned augments the empirical prior with a fresh, fully-trainable
        # additive base kernel (Sigma + K_base) fit on the observed data; em_frozen
        # uses the pure empirical prior (no base, no fit).
        base = (
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=7))
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
        model.eval()
        lik.eval()
        return model

    if method in ("pretrained_gp_frozen", "pretrained_gp_tuned"):
        # Reuse the shared SumMLL kernel (fitted lengthscales + outputscale).
        cc = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=7))
        with torch.no_grad():
            cc.raw_outputscale.data.copy_(covar_s.raw_outputscale.data)
            cc.base_kernel.raw_lengthscale.data.copy_(
                covar_s.base_kernel.raw_lengthscale.data
            )
        if method == "pretrained_gp_frozen":
            # Pure reuse of the pre-trained hyperparameters: also inherit the shared
            # ConstantMean and a fixed conditioning noise (matching em_frozen); NO fit.
            mm = ConstantMean()
            with torch.no_grad():
                mm.constant.data.copy_(mean_s.constant.data)
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
        cc = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=7))
        with torch.no_grad():
            cc.raw_outputscale.data.copy_(covar_s.raw_outputscale.data)
            cc.base_kernel.raw_lengthscale.data.copy_(
                covar_s.base_kernel.raw_lengthscale.data
            )
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

    raise ValueError(f"unknown method {method}")


def run_bo(
    method: str,
    pool_X: torch.Tensor,
    pool_Y: torch.Tensor,  # standardized (N,1)
    init_idx: list[int],
    n_iters: int,
    rng: torch.Generator,
    surrogate_fn: Callable,
) -> list[float]:
    """Return best-so-far trajectory (standardized units), length n_iters+1."""
    n = pool_X.shape[0]
    observed = list(init_idx)
    remaining = [i for i in range(n) if i not in set(observed)]
    best = pool_Y[observed].max().item()
    traj = [best]
    for _ in range(n_iters):
        if not remaining:
            traj.append(best)
            continue
        if method == "random":
            j = torch.randint(len(remaining), (1,), generator=rng).item()
            pick = remaining[j]
        else:
            model = surrogate_fn(method, pool_X[observed], pool_Y[observed])
            cand = pool_X[remaining]  # (M, 7)
            with torch.no_grad():
                # Latent posterior over the finite candidate set (2D input; the
                # empirical GP forward expects (n, d), not the (b, q, d) that
                # BoTorch acquisition functions pass). LogEI over a finite set is
                # argmax-identical to BoTorch's analytic LogEI.
                post = model.posterior(cand)
                mean = post.mean.reshape(-1)
                sigma = post.variance.clamp_min(1e-12).sqrt().reshape(-1)
                log_ei = _log_ei_helper((mean - best) / sigma) + sigma.log()
            pick = remaining[int(log_ei.argmax().item())]
        observed.append(pick)
        remaining.remove(pick)
        best = max(best, pool_Y[pick].item())
        traj.append(best)
    return traj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-configs", type=int, default=200)
    p.add_argument("--n-iters", type=int, default=40)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--n-em", type=int, default=50)
    p.add_argument("--n-pretrain", type=int, default=25)
    p.add_argument(
        "--methods",
        type=str,
        default="em_frozen,random",
        help=(
            "comma-separated subset of em_frozen,em_finetuned,"
            "pretrained_gp_frozen,pretrained_gp_tuned,vanilla_gp,random"
        ),
    )
    p.add_argument("--out", type=str, default="/tmp/bo_results.json")
    args = p.parse_args()

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
    methods = args.methods.split(",")

    names = list(DATASET_NAMES)
    print(f"Loading {len(names)} LCBench datasets ...", flush=True)
    Xn, Y_all = load_pool(names)

    # Deterministic split: subsample shared configs, hold out eval datasets.
    torch.manual_seed(42)
    cfg = torch.randperm(Xn.shape[0])[: args.n_configs]
    dperm = torch.randperm(len(names))
    eval_ix = dperm[: args.n_eval].tolist()
    pre_ix = dperm[args.n_eval : args.n_eval + args.n_pretrain].tolist()

    X_pool = Xn[cfg]  # (n_configs, 7)
    Yc = [Y[cfg] for Y in Y_all]  # per-dataset (n_configs, 1)
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

    print("Fitting shared SumMLL kernel + pre-training EM prior ...", flush=True)
    t0 = time.time()
    mean_s, covar_s = build_shared_kernel(pretrain_ds)
    em_prior = pretrain_em_prior(
        datasets=pretrain_ds,
        mean_module=mean_s,
        covar_module=covar_s,
        likelihood_noise=torch.tensor(1e-2, dtype=torch.double),
        num_em_iterations=args.n_em,
        enable_interpolation=True,
    )
    pretrain_time = time.time() - t0
    print(f"  pretraining done in {pretrain_time:.1f}s", flush=True)

    def surrogate_fn(method, X, Y):
        return make_surrogate(method, X, Y, em_prior, mean_s, covar_s)

    # results[method] = list over (dataset,seed) of best-so-far trajectories (raw acc)
    results: dict[str, list[list[float]]] = {m: [] for m in methods}
    pool_max_raw: list[float] = []
    timing: dict[str, float] = dict.fromkeys(methods, 0.0)

    for ei in eval_ix:
        dY = (Yc[ei] - ymean) / ystd  # standardized (n_configs, 1)
        raw = Yc[ei].squeeze(-1)  # raw accuracy (n_configs,)
        for seed in range(args.n_seeds):
            g = torch.Generator().manual_seed(1000 * ei + seed)
            init_idx = torch.randperm(args.n_configs, generator=g)[
                : args.n_init
            ].tolist()
            pool_max_raw.append(raw.max().item())
            for m in methods:
                gm = torch.Generator().manual_seed(7 * (1000 * ei + seed) + 13)
                _t0 = time.perf_counter()
                traj_std = run_bo(
                    m, X_pool, dY, init_idx, args.n_iters, gm, surrogate_fn
                )
                timing[m] += time.perf_counter() - _t0
                traj_raw = [v * ystd.item() + ymean.item() for v in traj_std]
                results[m].append(traj_raw)
        print(f"  done dataset {names[ei]}", flush=True)

    # Aggregate: mean best-so-far and mean simple regret vs #evals.
    pool_max_mean = sum(pool_max_raw) / len(pool_max_raw)
    n_pts = args.n_iters + 1
    summary = {}
    for m in methods:
        T = torch.tensor(results[m])  # (runs, n_pts)
        mean_best = T.mean(0)
        # paired simple regret to each run's pool max
        pm = torch.tensor(pool_max_raw).unsqueeze(1)
        regret = (pm - T).mean(0)
        summary[m] = {
            "mean_best": mean_best.tolist(),
            "mean_regret": regret.tolist(),
        }

    out = {
        "config": vars(args),
        "eval_datasets": [names[i] for i in eval_ix],
        "n_runs": len(pool_max_raw),
        "pool_max_mean": pool_max_mean,
        "init_evals": args.n_init,
        "summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Console report
    checkpoints = sorted({0, 5, 10, 20, 30, args.n_iters})
    checkpoints = [c for c in checkpoints if c < n_pts]
    print(f"\n=== BO on LCBench (finite pool of {args.n_configs} configs) ===")
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
    print("\nMean simple regret vs #BO evaluations (lower=better):")
    print(hdr)
    for m in methods:
        rg = summary[m]["mean_regret"]
        print(f"{m:<22}" + "".join(f"{rg[c]:<8.4f}" for c in checkpoints))
    n_evals = len(pool_max_raw) * args.n_iters
    print("\n=== Runtime breakdown ===")
    print(f"pretraining (shared SumMLL kernel + EM prior, ONCE): {pretrain_time:.1f}s")
    print(f"{'method':<22}{'total_s':<10}{'per_run_s':<11}{'per_eval_ms':<12}")
    for m in methods:
        tot = timing[m]
        pr = tot / max(len(pool_max_raw), 1)
        pe = 1000 * tot / max(n_evals, 1)
        print(f"{m:<22}{tot:<10.1f}{pr:<11.3f}{pe:<12.1f}")
    print(
        f"BO loops total: {sum(timing.values()):.1f}s "
        f"({len(pool_max_raw)} runs x {args.n_iters} iters)"
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
