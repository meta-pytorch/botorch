#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-task check: does an explicit empirical scale s1 help the additive combination?

Replicates the notebook's Section E EM multi-task GP study -- regularize the
rank-limited empirical cross-task covariance with a fitted ICM base kernel -- and
compares two ways of combining the empirical kernel with the (frozen) ICM base,
both fitting exactly one scale parameter + observation noise by marginal
likelihood (apples-to-apples):

  - additive:    K_emp + ScaleKernel(ICM)              ; fits {s2, noise}
  - additive_s1: ScaleKernel(K_emp) + ScaleKernel(ICM) ; fits {s1, s2, noise}
  (ICM lengthscales frozen throughout; only the outputscale magnitudes + noise are fit.)

Scores held-out-epoch RMSE/NLL on the target task across target-obs budgets,
averaged over seeds, with a paired (additive_s1 - additive) NLL diff (same fitted
ICM per seed).
"""

from __future__ import annotations

import argparse
import math

import torch
from _scratch_bo.lcbench_io import load_lcbench_data
from botorch.models.empirical_gps import MultiTaskEmpiricalOneDimensionalGP
from gpytorch.kernels import AdditiveKernel, Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)
METRIC = "Train/val_accuracy"


class ICMBaseKernel(Kernel):
    """Fitted ICM as a frozen base kernel over [epoch, task] inputs (mirrors the
    notebook)."""

    def __init__(self, B: torch.Tensor, lengthscale: float, outputscale: float) -> None:
        super().__init__()
        self.register_buffer("B", B)
        self.register_buffer(
            "log_ls", torch.log(torch.as_tensor(lengthscale, dtype=B.dtype))
        )
        self.register_buffer(
            "log_os", torch.log(torch.as_tensor(outputscale, dtype=B.dtype))
        )

    def forward(self, x1, x2, diag=False, **kwargs):
        le1 = x1[..., 0].clamp_min(1e-8).log()
        t1 = x1[..., 1].long()
        le2 = x2[..., 0].clamp_min(1e-8).log()
        t2 = x2[..., 1].long()
        ls2 = torch.exp(2.0 * self.log_ls)
        kee = torch.exp(self.log_os) * torch.exp(
            -0.5 * (le1.unsqueeze(-1) - le2.unsqueeze(-2)) ** 2 / ls2
        )
        btt = self.B[t1.unsqueeze(-1), t2.unsqueeze(-2)]
        k = btt * kee
        return k.diagonal(dim1=-2, dim2=-1) if diag else k


def rmse_nll(mu, gt, var):
    mu, gt = torch.as_tensor(mu), torch.as_tensor(gt)
    var = torch.as_tensor(var).clamp_min(1e-9)
    rmse = ((mu - gt) ** 2).mean().sqrt().item()
    nll = (0.5 * ((mu - gt) ** 2 / var + torch.log(2 * math.pi * var))).mean().item()
    return rmse, nll


def fit_icm(hist, tmean, elog, ntask, ne, n_iters):
    nh = hist[0].shape[0]
    hc = torch.stack(
        [torch.cat([hist[t][c] - tmean[t] for t in range(ntask)]) for c in range(nh)]
    )
    ll = torch.zeros(1, requires_grad=True)
    lo = torch.zeros(1, requires_grad=True)
    lr = torch.eye(ntask).clone().requires_grad_(True)
    lnz = torch.tensor([-2.0], requires_grad=True)
    dsq = (elog[:, None] - elog[None, :]) ** 2
    mm = ntask * ne
    eye = torch.eye(mm)
    opt = torch.optim.Adam([ll, lo, lr, lnz], lr=0.05)
    for _ in range(n_iters):
        opt.zero_grad()
        kee = lo.exp() * torch.exp(-0.5 * dsq / ll.exp() ** 2)
        low = torch.tril(lr)
        b = low @ low.T + 1e-4 * torch.eye(ntask)
        chol = torch.linalg.cholesky(torch.kron(b, kee) + lnz.exp() * eye + 1e-5 * eye)
        al = torch.cholesky_solve(hc.T, chol)
        (0.5 * (hc.T * al).sum() + nh * torch.log(torch.diag(chol)).sum()).backward()
        opt.step()
    with torch.no_grad():
        low = torch.tril(lr)
        b = low @ low.T + 1e-4 * torch.eye(ntask)
        return ICMBaseKernel(b.clone(), float(ll.exp()), float(lo.exp()))


def build_train(ci, n_target, tm, e_grid, donor_tasks, target_task, obs_e_idx):
    tx, ty = [], []
    for t in donor_tasks:
        for e in obs_e_idx:
            tx.append([e_grid[e].item(), float(t)])
            ty.append(tm[t][ci, e].item())
    for e in range(n_target):
        tx.append([e_grid[e].item(), float(target_task)])
        ty.append(tm[target_task][ci, e].item())
    return torch.tensor(tx), torch.tensor(ty).unsqueeze(-1)


def em_predict(tx, ty, hist, e_grid, task_names, target_task, base_kernel, mode, n_fit):
    common = {
        "train_X": tx,
        "train_Y": ty,
        "task_feature": -1,
        "historical_Xs": [e_grid.unsqueeze(-1) for _ in task_names],
        "historical_Ys": hist,
    }
    m = MultiTaskEmpiricalOneDimensionalGP(**common)
    emp = m.covar_module  # the (fixed) multi-task empirical kernel
    # ICM base wrapped in a ScaleKernel (trainable outputscale s2; lengthscales frozen).
    if mode == "additive":
        # K = K_empirical + s2 * K_ICM
        # (empirical magnitude FIXED; current OSS behavior)
        m.covar_module = AdditiveKernel(emp, ScaleKernel(base_kernel))
    elif mode == "additive_s1":
        # K = s1 * K_empirical + s2 * K_ICM  (also fit an empirical-kernel scale s1)
        m.covar_module = AdditiveKernel(ScaleKernel(emp), ScaleKernel(base_kernel))
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    m.train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.1)
    for _ in range(n_fit):
        opt.zero_grad()
        loss = -mll(m(tx), m.train_targets)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        q = torch.stack([e_grid, torch.full_like(e_grid, float(target_task))], dim=-1)
        p = m.posterior(q)
        return p.mean.squeeze(-1), p.variance.clamp_min(1e-9).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-hist", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=25)
    ap.add_argument("--n-fit", type=int, default=75)
    ap.add_argument("--icm-iters", type=int, default=200)
    args = ap.parse_args()

    task_names = ["vehicle", "segment", "jasmine"]
    donor_tasks = [0, 1]
    target_task = 2
    tm = [
        torch.as_tensor(load_lcbench_data(n, METRIC, dtype=torch.double).metrics)
        for n in task_names
    ]
    n_cfg = 200
    e_grid = torch.arange(1, tm[0].shape[1] + 1, dtype=torch.double)
    elog = torch.log(e_grid)
    ne = len(e_grid)
    ntask = len(task_names)
    obs_e_idx = list(range(0, ne, 5))
    n_targets = [0, 2, 5, 10, 20]
    modes = ["additive", "additive_s1"]
    rm = {mo: {nt: [] for nt in n_targets} for mo in modes}
    nl = {mo: {nt: [] for nt in n_targets} for mo in modes}

    for si in range(args.n_seeds):
        torch.manual_seed(1000 + si)
        perm = torch.randperm(n_cfg)
        hist = [t[perm[: args.n_hist]] for t in tm]
        tmean = torch.stack([hist[t].mean(0) for t in range(ntask)])
        evals = [int(perm[k]) for k in range(args.n_hist, args.n_hist + args.n_eval)]
        base_kernel = fit_icm(hist, tmean, elog, ntask, ne, args.icm_iters)
        for nt in n_targets:
            for ci in evals:
                gt = tm[target_task][ci]
                ev = slice(nt, None)
                tx, ty = build_train(
                    ci, nt, tm, e_grid, donor_tasks, target_task, obs_e_idx
                )
                for mo in modes:
                    mu, var = em_predict(
                        tx,
                        ty,
                        hist,
                        e_grid,
                        task_names,
                        target_task,
                        base_kernel,
                        mo,
                        args.n_fit,
                    )
                    r, n = rmse_nll(mu[ev], gt[ev], var[ev])
                    rm[mo][nt].append(r)
                    nl[mo][nt].append(n)
        print(f"  seed {si} done", flush=True)

    def agg(tbl, mo, nt):
        a = torch.tensor(tbl[mo][nt]).reshape(args.n_seeds, -1)
        sm = a.mean(1)
        sem = (
            (sm.std() / (args.n_seeds**0.5)).item()
            if args.n_seeds > 1
            else float("nan")
        )
        return sm.mean().item(), sem

    print(
        f"\nMulti-task EM GP additive vs additive_s1 "
        f"(target={task_names[target_task]}; "
        f"{args.n_seeds} seeds x {args.n_eval} configs, held-out-epoch metrics):"
    )
    for label, tbl in [("RMSE (lower=better)", rm), ("NLL (lower=better)", nl)]:
        print(f"\n{label}:")
        print(f"  {'method':<10}" + "".join(f"@{n:<10}" for n in n_targets))
        for mo in modes:
            print(
                f"  {mo:<10}"
                + "".join(
                    f"{agg(tbl, mo, n)[0]:5.2f}\u00b1{agg(tbl, mo, n)[1]:<4.2f} "
                    for n in n_targets
                )
            )
    print("\nPaired NLL diff (additive_s1 - additive); negative => s1 better/equal:")
    for nt in n_targets:
        d = (
            (torch.tensor(nl["additive_s1"][nt]) - torch.tensor(nl["additive"][nt]))
            .reshape(args.n_seeds, -1)
            .mean(1)
        )
        sem = (
            (d.std() / (args.n_seeds**0.5)).item() if args.n_seeds > 1 else float("nan")
        )
        print(f"  @{nt:<4} mean={d.mean().item():+.3f} sem={sem:.3f}")


if __name__ == "__main__":
    main()
