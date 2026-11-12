# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare front generators on shared fitted noisy-data GP posterior means.

Contributor: @M-Colley. This is a surrogate-model diagnostic, not a human
study or a closed-loop BO comparison. True analytic outcomes are used only
for training data and separate audits of returned designs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import math
import statistics
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.optimize import optimize_with_nsgaii
from botorch_community.utils.multi_objective.fpbi import sample_model_fpbi
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

# Experiments are repository files rather than installed Python packages.
_SPEC = importlib.util.spec_from_file_location(
    "_gp_fpbi_shared_benchmark", Path(__file__).with_name("benchmark_fpbi.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_SHARED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SHARED
_SPEC.loader.exec_module(_SHARED)
Case, CASES = _SHARED.Case, _SHARED.CASES
checked_front, common_cap = _SHARED.checked_front, _SHARED.common_cap
EvaluationCounter = _SHARED.EvaluationCounter
GenerationFailure = _SHARED.GenerationFailure
json_write, write_csv = _SHARED.json_write, _SHARED.write_csv
matched_generations, quality_metrics = (
    _SHARED.matched_generations,
    _SHARED.quality_metrics,
)


class CountedGP(Model):
    """Count queried decision vectors while sharing one fitted GP."""

    def __init__(self, fitted: SingleTaskGP) -> None:
        """Wrap a fitted model without refitting it.

        Args:
            fitted: The shared, trained model for this case and seed.
        """
        super().__init__()
        self.fitted = fitted
        self.counter = EvaluationCounter()

    @property
    def num_outputs(self) -> int:
        return self.fitted.num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        return self.fitted.batch_shape

    def posterior(self, X: Tensor, **kwargs: Any) -> Any:
        self.counter.posterior_calls += 1
        self.counter.objective_calls += 1
        self.counter.point_evaluations += (
            X.numel() // self.fitted.train_inputs[0].shape[-1]
        )
        with torch.no_grad():
            return self.fitted.posterior(X, **kwargs)


def feasible_designs(
    case: Case, count: int, seed: int, *, anchors: bool = True
) -> Tensor:
    """Generate feasible designs, optionally anchoring training endpoints."""
    X = SobolEngine(case.dimensions, scramble=True, seed=seed).draw(count).double()
    if case.constrained:
        X = 0.2 + 0.6 * X
    if not anchors:
        return X
    if case.dimensions == 1:
        X[0, 0], X[1, 0] = (0.2, 0.8) if case.constrained else (0.0, 1.0)
    else:
        X[:3] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])
    return X


def fit_case(
    case: Case, seed: int, config: dict[str, Any]
) -> tuple[SingleTaskGP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = config["train_3d"] if case.objectives == 3 else config["train_2d"]
    X = feasible_designs(case, n, seed)
    truth = case.costs(X)
    low, high = case.cost_bounds()
    scale = high - low
    generator = torch.Generator().manual_seed(seed + 1000)
    noise = torch.randn(truth.shape, generator=generator, dtype=torch.double)
    Y = truth + config["noise_sd"] * scale * noise
    variance = (config["noise_sd"] * scale).square().clamp_min(1e-6).expand_as(Y)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SingleTaskGP(
            X, Y, train_Yvar=variance, outcome_transform=Standardize(case.objectives)
        )
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(model.likelihood, model),
            optimizer_kwargs={"options": {"maxiter": config["fit_maxiter"]}},
        )
    fit_seconds = time.perf_counter() - started
    model.eval()
    audit_X = feasible_designs(case, 512, seed + 4000, anchors=False)
    with torch.no_grad():
        posterior = model.posterior(audit_X)
        error = (posterior.mean - case.costs(audit_X)) / scale
        sd = posterior.variance.clamp_min(0).sqrt() / scale
    return model, {
        "training_points": n,
        "training_sha256": hashlib.sha256(
            X.numpy().tobytes() + Y.numpy().tobytes()
        ).hexdigest(),
        "fit_seconds": fit_seconds,
        "heldout_normalized_rmse": error.square().mean().sqrt().item(),
        "heldout_latent_95_coverage": (error.abs() <= 1.96 * sd).double().mean().item(),
        "heldout_mean_normalized_sd": sd.mean().item(),
        "fit_warnings": [str(w.message) for w in caught],
    }


def generate(
    case: Case,
    model: SingleTaskGP,
    seed: int,
    method: str,
    config: dict[str, Any],
    generations: int | None = None,
) -> dict[str, Any]:
    wrapper = CountedGP(model).eval()

    def operation() -> tuple:
        if method == "fpbi":
            options = (
                {"inequality_constraints": case.constraints} if case.constrained else {}
            )
            result = sample_model_fpbi(
                wrapper,
                case.bounds(),
                n_steps=(
                    config["steps_3d"] if case.objectives == 3 else config["steps_2d"]
                ),
                maximize=False,
                num_restarts=config["restarts"],
                seed=seed,
                maxiter=config["maxiter"],
                **options,
            )
            X, Y = result.X, result.Y
            diagnostics = [asdict(d) for d in result.diagnostics]
            failed_rays = int((~result.ray_succeeded).sum())
        else:
            constraints = None
            if case.constrained:
                constraints = [
                    (torch.tensor([0]), torch.tensor([1.0], dtype=torch.double), 0.2),
                    (torch.tensor([0]), torch.tensor([-1.0], dtype=torch.double), -0.8),
                ]
            acquisition = MultiOutputPosteriorMean(
                wrapper, weights=-torch.ones(case.objectives, dtype=torch.double)
            )
            X, benefit = optimize_with_nsgaii(
                acquisition,
                case.bounds(),
                case.objectives,
                q=None,
                population_size=config["population"],
                max_gen=generations,
                seed=seed,
                inequality_constraints=constraints,
                max_attempts=1,
            )
            Y, diagnostics, failed_rays = -benefit, [], None
        return X, Y, diagnostics, failed_rays

    (X, Y, diagnostics, failed_rays), elapsed, caught = _SHARED.measured_operation(
        operation, wrapper.counter
    )
    return {
        "X": X,
        "Y": Y,
        "generation_seconds": elapsed,
        **asdict(wrapper.counter),
        "diagnostics": diagnostics,
        "failed_rays": failed_rays,
        "warnings": caught,
    }


def audit_front(case: Case, model: SingleTaskGP, measured: dict, cap: int) -> dict:
    """Select/cap in predicted space, then evaluate true outcomes separately."""
    X, predicted, checked = checked_front(case, measured["X"], measured["Y"])
    X, predicted = common_cap(case, X, predicted, cap)
    truth = case.costs(X)
    TX, TY, true_checked = checked_front(case, X, truth)
    reference = case.reference_front()
    predicted_metrics = quality_metrics(case, predicted, reference)
    true_metrics = quality_metrics(case, TY, reference)
    with torch.no_grad():
        sd = model.posterior(X).variance.clamp_min(0).sqrt()
    scale = case.cost_bounds()[1] - case.cost_bounds()[0]
    return {
        "predicted_hv_ratio": predicted_metrics["hv_ratio"],
        "true_hv_ratio": true_metrics["hv_ratio"],
        "true_igd": true_metrics["igd"],
        "returned_points": len(X),
        "true_nondominated_points": len(TX),
        "audit_true_evaluations": len(X),
        "returned_mean_normalized_sd": (sd / scale).mean().item(),
        "returned_normalized_rmse": ((predicted - truth) / scale)
        .square()
        .mean()
        .sqrt()
        .item(),
        "max_violation": true_checked["max_violation"],
        "raw_rows": checked["raw_rows"],
        "X": X.tolist(),
        "predicted_Y": predicted.tolist(),
        "true_Y": truth.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[c.name for c in CASES],
        default=[c.name for c in CASES],
    )
    parser.add_argument("--train-2d", type=int, default=24)
    parser.add_argument("--train-3d", type=int, default=48)
    parser.add_argument("--noise-sd", type=float, default=0.02)
    parser.add_argument("--fit-maxiter", type=int, default=75)
    parser.add_argument("--steps-2d", type=int, default=5)
    parser.add_argument("--steps-3d", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()
    if (
        min(args.train_2d, args.train_3d) < 3
        or not math.isfinite(args.noise_sd)
        or args.noise_sd < 0
    ):
        parser.error("training counts must be >=3 and noise-sd nonnegative")
    for key in (
        "fit_maxiter",
        "steps_2d",
        "steps_3d",
        "restarts",
        "maxiter",
        "population",
        "generations",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(1)
    config = {k: v for k, v in vars(args).items() if k != "output_dir"}
    paths = [
        Path(__file__),
        Path(__file__).with_name("benchmark_fpbi.py"),
        Path(__file__).parents[3] / "botorch_community/utils/multi_objective/fpbi.py",
        Path(__file__).parents[3]
        / "botorch_community/utils/multi_objective/decision_making.py",
    ]
    protocol = {
        "config": config,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("botorch", "torch", "gpytorch", "numpy", "scipy", "pymoo")
        },
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[3])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in paths
        },
        "model": (
            "Shared SingleTaskGP, Standardize outcomes, known fixed observation "
            "variances, CPU double"
        ),
        "noise": (
            "Independent Gaussian, SD is a fraction of known analytic Pareto "
            "outcome ranges"
        ),
        "arms": (
            "FPBI; NSGA-II fixed generations; NSGA-II generations rounded to FPBI "
            "queried decision-vector count (cap 300)"
        ),
        "audit": (
            "Common output cap selected using posterior means; true analytic "
            "values and uncertainty audited outside generation timing/counters"
        ),
        "limitations": (
            "Four analytic cases, three seeds, one machine with concurrent work; "
            "no BO trajectory, human-benefit or calibrated-safety claim. Known "
            "analytic ranges are used for benchmark normalization/capping."
        ),
    }
    json_write(output / "protocol.json", protocol)
    rows, raw = [], []
    for case in [c for c in CASES if c.name in args.cases]:
        for seed in args.seeds:
            try:
                model, fit = fit_case(case, seed, config)
                fit_error = None
            except Exception as error:
                model, fit = None, {}
                fit_error = f"{type(error).__name__}: {error}"
            fpbi_points = None
            methods = ["fpbi", "nsga_fixed", "nsga_matched"]
            if seed % 2:
                methods[1:] = methods[1:][::-1]
            for method in methods:
                row = {
                    "case": case.name,
                    "seed": seed,
                    "method": method,
                    **fit,
                    "error": None,
                }
                try:
                    if fit_error is not None:
                        raise RuntimeError(f"Shared GP fit failed: {fit_error}")
                    if method == "nsga_matched" and fpbi_points is None:
                        raise RuntimeError(
                            "FPBI failed; no measured budget is available"
                        )
                    generations = (
                        matched_generations(fpbi_points, config["population"], 300)
                        if method == "nsga_matched"
                        else config["generations"]
                    )
                    measured = generate(case, model, seed, method, config, generations)
                    row.update(
                        {
                            k: v
                            for k, v in measured.items()
                            if k not in ("X", "Y", "diagnostics", "warnings")
                        }
                    )
                    if method == "fpbi":
                        fpbi_points = measured["point_evaluations"]
                    steps = (
                        config["steps_3d"]
                        if case.objectives == 3
                        else config["steps_2d"]
                    )
                    audit = audit_front(
                        case,
                        model,
                        measured,
                        case.objectives + steps ** (case.objectives - 1),
                    )
                    row.update(
                        {
                            k: v
                            for k, v in measured.items()
                            if k not in ("X", "Y", "diagnostics", "warnings")
                        }
                    )
                    row.update(
                        {
                            k: v
                            for k, v in audit.items()
                            if k not in ("X", "predicted_Y", "true_Y")
                        }
                    )
                    row["budget_ratio"] = (
                        measured["point_evaluations"] / fpbi_points
                        if fpbi_points is not None
                        else None
                    )
                    raw.append(
                        {
                            **row,
                            **audit,
                            "diagnostics": measured["diagnostics"],
                            "warnings": measured["warnings"],
                        }
                    )
                except Exception as error:
                    if isinstance(error, GenerationFailure):
                        row.update(error.accounting)
                    row["error"] = f"{type(error).__name__}: {error}"
                    raw.append(row.copy())
                rows.append({k: v for k, v in row.items() if k != "fit_warnings"})
                json_write(output / "runs.json", raw)
                print(
                    case.name,
                    seed,
                    method,
                    row.get("true_hv_ratio"),
                    row["error"],
                    flush=True,
                )
    write_csv(output / "runs.csv", rows)
    summary = []
    for case in args.cases:
        for method in ("fpbi", "nsga_fixed", "nsga_matched"):
            group = [r for r in rows if r["case"] == case and r["method"] == method]
            good = [r for r in group if r["error"] is None]
            record = {
                "case": case,
                "method": method,
                "successes": len(good),
                "failures": len(group) - len(good),
            }
            for key in (
                "true_hv_ratio",
                "predicted_hv_ratio",
                "true_igd",
                "generation_seconds",
                "point_evaluations",
                "posterior_calls",
                "returned_points",
                "heldout_normalized_rmse",
                "heldout_latent_95_coverage",
                "returned_mean_normalized_sd",
                "budget_ratio",
            ):
                values = [r[key] for r in good if r.get(key) is not None]
                record[key + "_median"] = statistics.median(values) if values else None
            summary.append(record)
    write_csv(output / "summary.csv", summary)
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest()
        == protocol["source_sha256"][str(p.relative_to(Path(__file__).parents[3]))]
        for p in paths
    ), "Sources changed during run"
    json_write(
        output / "completion.json",
        {
            "runs": len(rows),
            "failures": sum(r["error"] is not None for r in rows),
            "source_hashes_unchanged": True,
        },
    )


if __name__ == "__main__":
    main()
