#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Do the multi-output and multi-task Empirical GPs actually buy anything?

The appendix describes both variants but shows no evidence they work. Each is
evaluated against the baseline it has to beat to justify its existence -- not
against a weak reference:

  multi-output : MultiOutputEmpiricalOneDimensionalGP, which models a DENSE
      cross-output covariance, versus INDEPENDENT EmpiricalOneDimensionalGPs, one
      per output, fit on the same historical curves. Both see identical data; the
      only difference is whether cross-output correlation is modeled. LCBench
      supplies genuinely correlated outputs (validation accuracy and validation
      cross-entropy for the same configuration over epochs).

  multi-task : MultiTaskEmpiricalOneDimensionalGP with the target task only
      partially observed and the remaining tasks fully observed, versus a
      SINGLE-TASK EmpiricalOneDimensionalGP fit on the target's observations
      alone. This isolates the value of cross-task transfer under partial
      observation, which is the setting the variant exists for.

Scoring: held-out progression points, RMSE and NLL, over E evaluation datasets x S
seeds. Because runs on the same dataset share a historical corpus they are not
independent, so the reported standard error is clustered by dataset, and the
multi-output/multi-task versus baseline comparison is PAIRED on the split (same
seed, same observed indices, same held-out targets).
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from _scratch_bo.lcbench_io import (
    LCBENCH_DATASET_NAMES as DATASET_NAMES,
    load_lcbench_data,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.empirical_gps import (
    EmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalGP,
    MultiTaskEmpiricalOneDimensionalGP,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

VAL_ACC = "Train/val_accuracy"
VAL_CE = "Train/val_cross_entropy"


def _curves(name: str, metric: str) -> torch.Tensor:
    """`(num_curves, num_epochs)` metric curves for one LCBench dataset."""
    d = load_lcbench_data(name, metric_name=metric)
    Y = d.Y if hasattr(d, "Y") else d.metrics
    return torch.as_tensor(Y, dtype=torch.double)


def _standardize(train: torch.Tensor, *others: torch.Tensor):
    """Standardize by the TRAINING statistics only; never peek at held-out data."""
    mu = train.mean()
    sd = train.std().clamp_min(1e-8)
    return ((train - mu) / sd, *[(o - mu) / sd for o in others]), (mu, sd)


def _gauss_nll(mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> float:
    var = var.clamp_min(1e-10)
    return float((0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)).mean())


def _score(post, y: torch.Tensor) -> dict[str, float]:
    """Accuracy AND calibration -- RMSE alone hides overconfidence, NLL alone conflates
    the two, and coverage/calib_ratio separate them: calib_ratio = mean sigma / RMSE is
    1.0 for a perfectly calibrated model, < 1 when overconfident."""
    mean = post.mean.reshape(y.shape)
    var = post.variance.reshape(y.shape).clamp_min(1e-12)
    sd = var.sqrt()
    rmse = float(((mean - y) ** 2).mean().sqrt())
    return {
        "rmse": rmse,
        "nll": _gauss_nll(mean, var, y),
        "coverage95": float(((y - mean).abs() <= 1.96 * sd).double().mean()),
        "calib_ratio": float(sd.mean()) / max(rmse, 1e-12),
        "mean_sigma": float(sd.mean()),
    }


def _make_base(d: int = 1):
    """Fresh ScaleKernel(Matern-5/2) to add to the empirical covariance.

    The joint empirical covariance has rank at most the number of historical curves,
    which for the multi-output model is spread over an (n*m)-dimensional space -- it is
    MORE rank-starved than the independent per-output models it is competing with. A
    full-rank additive base is the same repair that keeps em_finetuned's 7D NLL bounded
    where em_frozen's diverges.
    """
    from gpytorch.kernels import MaternKernel, ScaleKernel

    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))


def _fit(model) -> None:
    """Fit the base kernel + noise by marginal likelihood; keep the prior on failure."""
    try:
        model.train()
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    except Exception:  # noqa: BLE001 - a borderline fit must not abort the sweep
        pass
    model.eval()


# --------------------------------------------------------------------------- MO
def run_multioutput(args) -> list[dict]:
    """Joint vs independent modeling of two correlated LCBench metrics.

    Each arm is run with and without a fitted additive base kernel, because the joint
    covariance is rank-limited over an (n*m)-dimensional space and is therefore MORE
    rank-starved than the independent per-output models it competes with.
    """
    rows = []
    names = list(DATASET_NAMES)[: args.n_eval]
    arms = (("", False), ("+base", True)) if args.base_kernel else (("", False),)
    for ds in names:
        acc = _curves(ds, VAL_ACC)
        ce = _curves(ds, VAL_CE)
        n_curves = min(acc.shape[0], ce.shape[0], args.n_hist + args.n_target)
        acc, ce = acc[:n_curves], ce[:n_curves]
        n_ep = acc.shape[1]
        X = torch.linspace(0.0, 1.0, n_ep, dtype=torch.double).unsqueeze(-1)
        for seed in range(args.n_seeds):
            g = torch.Generator().manual_seed(1000 * (abs(hash(ds)) % 9973) + seed)
            perm = torch.randperm(n_curves, generator=g)
            hist_ix, tgt_ix = perm[: args.n_hist], perm[args.n_hist]
            HY = torch.stack([acc[hist_ix], ce[hist_ix]], dim=-1)
            tgt = torch.stack([acc[tgt_ix], ce[tgt_ix]], dim=-1)
            for n_obs in args.n_obs_grid:
                obs, held = torch.arange(n_obs), torch.arange(n_obs, n_ep)
                if held.numel() == 0:
                    continue
                (HYs, tgts), _ = _standardize(HY, tgt)
                trX, trY = X[obs], tgts[obs]
                teX, teY = X[held], tgts[held]
                for tag, use_base in arms:
                    try:
                        mo = MultiOutputEmpiricalOneDimensionalGP(
                            train_X=trX,
                            train_Y=trY,
                            historical_X=X,
                            historical_Y=HYs,
                            base_covar_module=_make_base() if use_base else None,
                        )
                        _fit(mo) if use_base else mo.eval()
                        with torch.no_grad():
                            sc = _score(mo.posterior(teX), teY)
                        rows.append(
                            dict(
                                study="mo",
                                dataset=ds,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"multioutput_egp{tag}",
                                **sc,
                            )
                        )
                    except Exception as e:  # noqa: BLE001 - record, keep sweeping
                        rows.append(
                            dict(
                                study="mo",
                                dataset=ds,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"multioutput_egp{tag}",
                                error=str(e)[:120],
                            )
                        )
                    try:
                        agg: dict[str, list[float]] = {}
                        for j in range(HYs.shape[-1]):
                            so = EmpiricalOneDimensionalGP(
                                train_X=trX,
                                train_Y=trY[:, j : j + 1],
                                historical_X=X,
                                historical_Y=HYs[..., j : j + 1],
                                base_covar_module=_make_base() if use_base else None,
                            )
                            _fit(so) if use_base else so.eval()
                            with torch.no_grad():
                                sc = _score(so.posterior(teX), teY[:, j : j + 1])
                            for k, v in sc.items():
                                agg.setdefault(k, []).append(v)
                        rows.append(
                            dict(
                                study="mo",
                                dataset=ds,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"independent_egp{tag}",
                                **{k: sum(v) / len(v) for k, v in agg.items()},
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        rows.append(
                            dict(
                                study="mo",
                                dataset=ds,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"independent_egp{tag}",
                                error=str(e)[:120],
                            )
                        )
    return rows


# --------------------------------------------------------------------------- MT
def run_multitask(args) -> list[dict]:
    """Cross-task transfer when the target task is only partially observed."""
    rows = []
    pool = list(DATASET_NAMES)[: args.n_eval + args.n_tasks]
    arms = (("", False), ("+base", True)) if args.base_kernel else (("", False),)
    for e in range(args.n_eval):
        tgt_name = pool[e]
        other = [q for q in pool if q != tgt_name][: args.n_tasks - 1]
        tgt_all = _curves(tgt_name, VAL_ACC)
        oth_all = [_curves(o, VAL_ACC) for o in other]
        n_ep = tgt_all.shape[1]
        X = torch.linspace(0.0, 1.0, n_ep, dtype=torch.double).unsqueeze(-1)
        K = min([args.n_hist, tgt_all.shape[0]] + [o.shape[0] for o in oth_all])
        for seed in range(args.n_seeds):
            g = torch.Generator().manual_seed(7919 * (e + 1) + seed)
            perm = torch.randperm(tgt_all.shape[0], generator=g)
            hist_ix, tgt_ix = perm[:K], perm[K]
            hist_Ys = [tgt_all[hist_ix]] + [o[:K] for o in oth_all]
            hist_Xs = [X for _ in hist_Ys]
            tgt = tgt_all[tgt_ix]
            for n_obs in args.n_obs_grid:
                obs, held = torch.arange(n_obs), torch.arange(n_obs, n_ep)
                if held.numel() == 0:
                    continue
                (hs, ts), _ = _standardize(torch.stack(hist_Ys), tgt)
                hist_std = [hs[i] for i in range(hs.shape[0])]
                zeros_tr = torch.zeros(n_obs, 1, dtype=torch.double)
                zeros_te = torch.zeros(held.numel(), 1, dtype=torch.double)
                trX = torch.cat([X[obs], zeros_tr], dim=-1)
                teX = torch.cat([X[held], zeros_te], dim=-1)
                trY, teY = ts[obs].unsqueeze(-1), ts[held].unsqueeze(-1)
                for tag, use_base in arms:
                    try:
                        mt = MultiTaskEmpiricalOneDimensionalGP(
                            train_X=trX,
                            train_Y=trY,
                            task_feature=-1,
                            historical_Xs=hist_Xs,
                            historical_Ys=hist_std,
                            base_covar_module=_make_base() if use_base else None,
                        )
                        _fit(mt) if use_base else mt.eval()
                        with torch.no_grad():
                            sc = _score(mt.posterior(teX), teY)
                        rows.append(
                            dict(
                                study="mt",
                                dataset=tgt_name,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"multitask_egp{tag}",
                                **sc,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001
                        rows.append(
                            dict(
                                study="mt",
                                dataset=tgt_name,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"multitask_egp{tag}",
                                error=str(ex)[:120],
                            )
                        )
                    try:
                        st = EmpiricalOneDimensionalGP(
                            train_X=X[obs],
                            train_Y=trY,
                            historical_X=X,
                            historical_Y=hist_std[0].unsqueeze(-1),
                            base_covar_module=_make_base() if use_base else None,
                        )
                        _fit(st) if use_base else st.eval()
                        with torch.no_grad():
                            sc = _score(st.posterior(X[held]), teY)
                        rows.append(
                            dict(
                                study="mt",
                                dataset=tgt_name,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"singletask_egp{tag}",
                                **sc,
                            )
                        )
                    except Exception as ex:  # noqa: BLE001
                        rows.append(
                            dict(
                                study="mt",
                                dataset=tgt_name,
                                seed=seed,
                                n_obs=n_obs,
                                method=f"singletask_egp{tag}",
                                error=str(ex)[:120],
                            )
                        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=str, default="both", choices=["mo", "mt", "both"])
    p.add_argument("--n-eval", type=int, default=6)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-hist", type=int, default=40)
    p.add_argument("--n-target", type=int, default=1)
    p.add_argument("--n-tasks", type=int, default=4)
    p.add_argument("--n-obs-grid", type=str, default="5,10,20")
    p.add_argument(
        "--base-kernel",
        action="store_true",
        help="Also run every arm with a fitted additive base kernel.",
    )
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", type=str, default="/tmp/mo_mt.json")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.n_obs_grid = [int(v) for v in args.n_obs_grid.split(",") if v.strip()]
    torch.manual_seed(0)

    rows: list[dict] = []
    if args.study in ("mo", "both"):
        print("multi-output study ...", flush=True)
        rows += run_multioutput(args)
    if args.study in ("mt", "both"):
        print("multi-task study ...", flush=True)
        rows += run_multitask(args)

    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    print(f"{len(ok)} scored rows, {len(bad)} errors")
    for b in bad[:3]:
        print("  ERROR:", b.get("method"), b["error"])
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
