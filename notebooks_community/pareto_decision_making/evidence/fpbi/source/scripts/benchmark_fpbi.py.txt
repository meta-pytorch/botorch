"""Small reproducible FPBI / upstream BoTorch NSGA-II comparison.

Run with the repository environment; pymoo is an optional benchmark dependency.
Writes its protocol before observing results, counts every objective-point
evaluation (including finite differences and diagnostics), and keeps raw fronts.
This is a deterministic-model front-generation experiment, not a BO experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import statistics
import subprocess
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean
from botorch.models.deterministic import GenericDeterministicModel
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from openbo.decision_making import sample_fpbi, sample_model_fpbi

UPSTREAM_DOC = "https://botorch.readthedocs.io/en/stable/utils.html"
DTYPE = torch.double
FEASIBILITY_TOL = 1e-6


@dataclass(frozen=True)
class Case:
    name: str
    dimensions: int
    objectives: int
    description: str
    constrained: bool = False

    def costs(self, X: Tensor) -> Tensor:
        """True analytic costs; also work with posterior t-batch shapes."""
        x = X[..., 0]
        if self.name == "concave_2d":
            return torch.stack((x.clamp_min(0).sqrt(), (1 - x).clamp_min(0).sqrt()), -1)
        if self.objectives == 2:
            return torch.stack((x.square(), (1 - x).square()), -1)
        a, b = X[..., 0] * math.pi / 2, X[..., 1] * math.pi / 2
        radius = 1 + (X[..., 2] - 0.5).square()
        return radius.unsqueeze(-1) * torch.stack(
            (a.cos() * b.cos(), a.cos() * b.sin(), a.sin()), -1
        )

    def constraints(self, X: Tensor) -> Tensor:
        """Identical feasible set for both optimizers, >=0 convention."""
        x = X[..., 0]
        if self.constrained:
            return torch.stack((x - 0.2, 0.8 - x), -1)
        return torch.empty((*X.shape[:-1], 0), dtype=X.dtype, device=X.device)

    def bounds(self) -> Tensor:
        return torch.stack(
            (torch.zeros(self.dimensions), torch.ones(self.dimensions))
        ).double()

    def cost_bounds(self) -> tuple[Tensor, Tensor]:
        low, high = (0.04, 0.64) if self.constrained else (0.0, 1.0)
        return torch.full((self.objectives,), low, dtype=DTYPE), torch.full(
            (self.objectives,), high, dtype=DTYPE
        )

    def normalize(self, Y: Tensor) -> Tensor:
        lo, hi = self.cost_bounds()
        return (Y - lo) / (hi - lo)

    def reference_front(self, n: int = 4097) -> Tensor:
        if self.objectives == 2:
            a, b = (0.2, 0.8) if self.constrained else (0.0, 1.0)
            return self.costs(torch.linspace(a, b, n, dtype=DTYPE).unsqueeze(-1))
        # Absolute isotropic normals give uniform surface directions in the
        # positive octant. Fixed reference seed is independent of run seeds.
        rng = np.random.default_rng(1729)
        directions = np.abs(rng.normal(size=(n, 3)))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return torch.cat((torch.from_numpy(directions), torch.eye(3, dtype=DTYPE)))

    def reference_hv(self) -> float:
        """Exact continuous-front HV in known normalized cost coordinates."""
        if self.objectives == 3:
            return 1.1**3 - math.pi / 6  # unit positive-octant sphere
        if self.name == "concave_2d":
            return 1.1**2 - math.pi / 4  # unit positive-quadrant circle
        a, b = (0.2, 0.8) if self.constrained else (0.0, 1.0)
        span = b * b - a * a

        def primitive(x: float) -> float:
            return (1 - a * a) * x * x - (4 / 3) * x**3 + 0.5 * x**4

        area = (primitive(b) - primitive(a)) / span**2
        return 1.1**2 - area


CASES = (
    Case("convex_2d", 1, 2, "Convex front (x^2, (1-x)^2), x in [0,1]."),
    Case("concave_2d", 1, 2, "Nonconvex objective image: (sqrt(x), sqrt(1-x))."),
    Case("constrained_2d", 1, 2, "Quadratic front with 0.2 <= x <= 0.8.", True),
    Case(
        "sphere_3d",
        3,
        3,
        "DTLZ2-style unit positive-octant sphere; radial nuisance variable.",
    ),
)


@dataclass
class EvaluationCounter:
    objective_calls: int = 0
    point_evaluations: int = 0
    posterior_calls: int = 0


class GenerationFailure(RuntimeError):
    """Retain observed effort even if an optimizer cannot return a front."""

    def __init__(self, error: Exception, accounting: dict[str, Any]) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.accounting = accounting


def measured_operation(operation: Callable[[], Any], counter: EvaluationCounter):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = operation()
        except Exception as error:
            raise GenerationFailure(
                error,
                {
                    "generation_seconds": time.perf_counter() - started,
                    **asdict(counter),
                    "warnings": [str(w.message) for w in caught],
                },
            ) from error
    return result, time.perf_counter() - started, [str(w.message) for w in caught]


class CountedModel(GenericDeterministicModel):
    """A real BoTorch deterministic model with explicit work counters."""

    def __init__(self, case: Case) -> None:
        self.counter = EvaluationCounter()

        def evaluate(X: Tensor) -> Tensor:
            self.counter.objective_calls += 1
            self.counter.point_evaluations += X.numel() // case.dimensions
            return case.costs(X)

        super().__init__(f=evaluate, num_outputs=case.objectives)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        self.counter.posterior_calls += 1
        return super().posterior(X, *args, **kwargs)


def checked_front(
    case: Case, X: Tensor, Y: Tensor
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Check feasibility and common tolerance-based objective deduplication."""
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("front arrays must have matching two-dimensional rows")
    if len(X) == 0:
        return (
            X,
            Y,
            {"raw_rows": 0, "feasible_rows": 0, "front_rows": 0, "max_violation": None},
        )
    bounds = case.bounds()
    constraints = case.constraints(X)
    violation = torch.maximum(
        (bounds[0] - X).clamp_min(0).amax(-1), (X - bounds[1]).clamp_min(0).amax(-1)
    )
    if constraints.numel():
        violation = torch.maximum(violation, (-constraints).clamp_min(0).amax(-1))
    finite = torch.isfinite(X).all(-1) & torch.isfinite(Y).all(-1)
    feasible = finite & (violation <= FEASIBILITY_TOL)
    FX, FY = X[feasible], Y[feasible]
    keep = is_non_dominated(-FY, deduplicate=True)
    chosen: list[int] = []
    Z = case.normalize(FY)
    for index in torch.where(keep)[0].tolist():
        if not any(
            torch.allclose(Z[index], Z[j], rtol=0, atol=FEASIBILITY_TOL) for j in chosen
        ):
            chosen.append(index)
    indices = torch.tensor(chosen, dtype=torch.long)
    return (
        FX[indices],
        FY[indices],
        {
            "raw_rows": len(X),
            "feasible_rows": int(feasible.sum()),
            "front_rows": len(indices),
            "max_violation": float(violation.max()) if finite.all() else None,
        },
    )


def common_cap(case: Case, X: Tensor, Y: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    """Same deterministic greedy HV cap for every method, outside timing."""
    if len(Y) <= cap:
        return X, Y
    benefits = -case.normalize(Y)
    hv = Hypervolume(-torch.full((case.objectives,), 1.1, dtype=DTYPE))
    chosen: list[int] = []
    remaining = list(range(len(Y)))
    for _ in range(cap):
        # Stable first-index ties, no randomized pruning of a front.
        best = max(remaining, key=lambda i: hv.compute(benefits[chosen + [i]]))
        chosen.append(best)
        remaining.remove(best)
    indices = torch.tensor(chosen, dtype=torch.long)
    return X[indices], Y[indices]


def quality_metrics(
    case: Case, Y: Tensor, reference: Tensor
) -> dict[str, float | None]:
    if len(Y) == 0:
        return {"hv": 0.0, "hv_ratio": 0.0, "igd": None}
    Z = case.normalize(Y)
    hv = Hypervolume(-torch.full((case.objectives,), 1.1, dtype=DTYPE)).compute(-Z)
    # IGD is the mean reference-to-nearest-approximation Euclidean distance.
    igd = torch.cdist(case.normalize(reference), Z).amin(-1).mean().item()
    return {"hv": hv, "hv_ratio": hv / case.reference_hv(), "igd": igd}


def matched_generations(evaluations: int, population: int, limit: int) -> int:
    """Generation-granular nearest budget; measured discrepancy is reported."""
    return min(limit, max(1, math.floor(evaluations / population + 0.5)))


def run_fpbi(case: Case, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    model = CountedModel(case)
    bounds = case.bounds()
    options = {
        "n_steps": config["steps_3d"] if case.objectives == 3 else config["steps_2d"],
        "maximize": False,
        "num_restarts": config["restarts"],
        "seed": seed,
        "maxiter": config["maxiter"],
        "feasibility_tol": FEASIBILITY_TOL,
    }
    if case.constrained:
        options["inequality_constraints"] = case.constraints

    def operation():
        if config["fpbi_api"] == "model":
            return sample_model_fpbi(model, bounds, **options)
        return sample_fpbi(model.forward, bounds, **options)

    result, elapsed, caught = measured_operation(operation, model.counter)
    diagnostics = [asdict(d) for d in result.diagnostics]
    return dict(
        X=result.X,
        Y=result.Y,
        generation_seconds=elapsed,
        **asdict(model.counter),
        diagnostics=diagnostics,
        solver_attempts=len(diagnostics),
        rejected_attempts=sum(not d["accepted"] for d in diagnostics),
        failed_rays=int((~result.ray_succeeded).sum()),
        nominal_rays=result.ray_succeeded.numel(),
        warnings=caught,
        basis_fallback=bool(result.geometry.used_fallback_basis),
    )


def run_nsga(
    case: Case, seed: int, generations: int, config: dict[str, Any]
) -> dict[str, Any]:
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

    model = CountedModel(case)
    acquisition = MultiOutputPosteriorMean(
        model, weights=-torch.ones(case.objectives, dtype=DTYPE)
    )
    constraints = None
    if case.constrained:
        constraints = [
            (torch.tensor([0]), torch.tensor([1.0], dtype=DTYPE), 0.2),
            (torch.tensor([0]), torch.tensor([-1.0], dtype=DTYPE), -0.8),
        ]

    def operation():
        return optimize_with_nsgaii(
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

    (X, benefit), elapsed, caught = measured_operation(operation, model.counter)
    return dict(
        X=X,
        Y=-benefit,
        generation_seconds=elapsed,
        **asdict(model.counter),
        diagnostics=[],
        solver_attempts=None,
        rejected_attempts=None,
        failed_rays=None,
        nominal_rays=None,
        basis_fallback=None,
        generations=generations,
        warnings=caught,
    )


def json_write(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        *sorted((repo / "src/openbo/decision_making").glob("*.py")),
    ]
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "botorch", "gpytorch", "numpy", "scipy", "pymoo")
        },
        "torch_threads": torch.get_num_threads(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines(),
        "source_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "upstream_signature": str(inspect.signature(optimize_with_nsgaii)),
        "upstream_documentation": UPSTREAM_DOC,
        "config": config,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for case in dict.fromkeys(row["case"] for row in rows):
        for method in ("fpbi", "nsga_output_cap", "nsga_eval_matched"):
            group = [r for r in rows if r["case"] == case and r["method"] == method]
            if not group:
                continue
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "runs": len(group),
                "failures": sum(r["error"] is not None for r in group),
            }
            for name in (
                "generation_seconds",
                "point_evaluations",
                "posterior_calls",
                "front_rows",
                "metric_rows",
                "hv_ratio",
                "igd",
                "max_violation",
                "failed_rays",
                "budget_ratio",
            ):
                values = [r[name] for r in group if r.get(name) is not None]
                item[name + "_median"] = statistics.median(values) if values else None
                item[name + "_min"] = min(values) if values else None
                item[name + "_max"] = max(values) if values else None
            summaries.append(item)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, names)
        writer.writeheader()
        writer.writerows(rows)


def report(
    output: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    lines = [
        "# FPBI comparison: small CPU deterministic-model experiment",
        "",
        (
            "Both methods optimize the same analytic costs through real BoTorch deterministic models. "
            "This measures front generation and Python/Torch evaluation overhead; it is not a noisy BO study."
        ),
        "",
        (
            "Settings were saved in protocol.json before the measured runs. "
            "No settings were selected based on the measured outcomes."
        ),
        "",
        "| Case | Method | HV / exact HV ↑ | Scaled IGD ↓ | Seconds | Evaluated points | Front / cap | Failed runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.5g}"

    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case"],
                    row["method"],
                    number(row["hv_ratio_median"]),
                    number(row["igd_median"]),
                    number(row["generation_seconds_median"]),
                    number(row["point_evaluations_median"]),
                    number(row["metric_rows_median"]),
                    str(row["failures"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Table values are medians across the preregistered seeds. "
                "summary.csv includes the observed min/max; these small samples do not support significance claims."
            ),
            "",
            "## Budgets and accounting",
            "",
            (
                f"- FPBI uses {config['restarts']} starts per extreme and adds its extrema to ray starts; "
                "finite-difference and repeated diagnostic objective evaluations are all counted."
            ),
            (
                f"- nsga_output_cap uses population {config['population']} for {config['output_generations']} "
                "generations. Both methods receive the same maximum output size m + n_steps**(m-1), "
                "then the same deterministic greedy hypervolume cap. This matches a cap, not total work or actual retained cardinality."
            ),
            (
                "- nsga_eval_matched rounds the measured FPBI point count to the nearest population-sized "
                f"generation, bounded at {config['max_budget_generations']} generations. Actual counts and budget_ratio "
                "(NSGA-II / FPBI) are recorded; duplicate rejection may change population evaluation counts."
            ),
            (
                "- An evaluated point is one decision vector in a model evaluation. A batched call with 32 "
                "vectors counts as 32 points and one posterior call. These are model calls, not expensive observed experiments."
            ),
            (
                "- Generation timing excludes imports, warmup, metric/reference work, common output capping, serialization and plotting. "
                "It includes each method's internal filtering and optimizer setup. One CPU Torch thread is used."
            ),
            (
                "- FPBI uses local SLSQP; its per-attempt and per-ray diagnostics are preserved. NSGA-II has one "
                "attempt and generation-based termination; its returned-front feasibility is independently checked, "
                "but the upstream wrapper supplies no per-generation convergence certificate."
            ),
            "",
            "## Metrics and reference",
            "",
            (
                "All metrics use each case's known theoretical Pareto objective ranges, not a method's inferred "
                "normalization. Hypervolume uses normalized cost reference (1.1,...,1.1) and exact continuous-front "
                "HV (analytic integrals or quarter-circle/octant-sphere volume). IGD uses 4,097 fixed front samples "
                "(plus the three sphere axes); two-dimensional samples are uniform in x, sphere samples use normalized "
                "absolute standard normals with reference seed 1729. IGD is reference-to-approximation distance, "
                "so its averaging measure is explicitly this finite reference set."
            ),
            "",
            (
                "Metrics admit only finite points feasible within 1e-6, filter dominance, and deduplicate normalized "
                "outcomes within 1e-6. raw_fronts retain uncapped optimizer outputs; each run also records uncapped quality."
            ),
            "",
            "## Limits",
            "",
            (
                "Four low-dimensional analytic cases, three seeds by default, CPU double precision, and exact "
                "deterministic posterior means are a diagnostic comparison, not evidence of global superiority. "
                "Cheap vectorized objectives favor population batching in wall time; real GP posterior cost and parallel "
                "hardware can change that result. Local extrema, nonconvexity, and FPBI rays with zero direction "
                "components can affect coverage or cause infeasible rays. Output cap and objective-point matching "
                "answer different questions and must be read alongside actual effort and failure counts."
            ),
            "",
            (
                "Reproduce using the command in protocol.json. Install the optional comparator with "
                "`uv pip install --python .venv/Scripts/python.exe pymoo`; no package runtime dependency was added."
            ),
            "",
            (
                f"Upstream API: [BoTorch documentation]({UPSTREAM_DOC}). "
                "Installed versions, actual signature, source hashes, Git state and machine details are in provenance.json."
            ),
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plots(output: Path, summary: list[dict[str, Any]], cases: tuple[Case, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("fpbi", "nsga_output_cap", "nsga_eval_matched")
    labels = ("FPBI", "NSGA-II: output cap", "NSGA-II: eval matched")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, key, title, logarithmic in zip(
        axes.flat,
        ("hv_ratio", "igd", "generation_seconds", "point_evaluations"),
        (
            "HV / exact continuous-front HV (higher)",
            "Scaled IGD (lower)",
            "Generation seconds",
            "Objective points evaluated",
        ),
        (False, True, True, True),
        strict=True,
    ):
        for j, method in enumerate(methods):
            values = [
                next(
                    r
                    for r in summary
                    if r["case"] == case.name and r["method"] == method
                )
                for case in cases
            ]
            ys = np.array([r[key + "_median"] or np.nan for r in values])
            lo = np.array([r[key + "_min"] or np.nan for r in values])
            hi = np.array([r[key + "_max"] or np.nan for r in values])
            axis.bar(
                np.arange(len(cases)) + (j - 1) * 0.25,
                ys,
                width=0.24,
                yerr=np.stack((ys - lo, hi - ys)),
                label=labels[j],
                capsize=3,
            )
        axis.set_xticks(range(len(cases)), [c.name.replace("_", "\n") for c in cases])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("FPBI and upstream BoTorch NSGA-II — medians and observed ranges")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--cases",
        choices=[c.name for c in CASES],
        nargs="+",
        default=[c.name for c in CASES],
    )
    parser.add_argument("--steps-2d", type=int, default=5)
    parser.add_argument("--steps-3d", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--output-generations", type=int, default=20)
    parser.add_argument("--max-budget-generations", type=int, default=300)
    parser.add_argument("--fpbi-api", choices=["model", "direct"], default="model")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    for name in (
        "steps_2d",
        "steps_3d",
        "restarts",
        "maxiter",
        "population",
        "output_generations",
        "max_budget_generations",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    torch.set_num_threads(1)
    cases = tuple(c for c in CASES if c.name in args.cases)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runs.json").exists():
        raise FileExistsError(
            "Use a new output directory to preserve previous measured runs"
        )
    config = vars(args).copy()
    config["output_dir"] = str(output)
    config["feasibility_tol"] = FEASIBILITY_TOL
    config["dtype"] = "torch.float64"
    config["device"] = "cpu"
    json_write(
        output / "protocol.json",
        {
            "preregistered_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "cases": [asdict(c) for c in cases],
            "command": ".venv\\Scripts\\python.exe scripts/benchmark_fpbi.py "
            + " ".join(
                f'--{key.replace("_", "-")} "{value}"'
                if key == "output_dir"
                else f"--{key.replace('_', '-')} " + " ".join(map(str, value))
                if isinstance(value, list)
                else f"--{key.replace('_', '-')} {value}"
                for key, value in vars(args).items()
                if not isinstance(value, bool)
            ),
            "budget_policy": "same output cap plus separately rounded FPBI objective-point-matched generations",
            "reference": "known objective ranges; exact continuous-front HV; fixed 4097-point reference for IGD",
        },
    )
    initial_provenance = provenance(config)
    json_write(output / "provenance.json", initial_provenance)
    # Warm all implementations before measuring. Model construction is excluded
    # consistently; optimizer setup and its internal filtering remain timed.
    warm_config = {
        **config,
        "steps_2d": 2,
        "restarts": 1,
        "maxiter": 50,
        "population": 8,
    }
    run_fpbi(CASES[0], 987, warm_config)
    run_nsga(CASES[0], 987, 2, warm_config)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for case in cases:
        reference = case.reference_front()
        steps = config["steps_3d"] if case.objectives == 3 else config["steps_2d"]
        cap = case.objectives + steps ** (case.objectives - 1)
        for seed in args.seeds:
            fpbi_evaluations = None
            nsga_order = (
                ("nsga_output_cap", "nsga_eval_matched")
                if seed % 2 == 0
                else ("nsga_eval_matched", "nsga_output_cap")
            )
            for method in ("fpbi", *nsga_order):
                torch.manual_seed(seed)
                row: dict[str, Any] = {
                    "case": case.name,
                    "seed": seed,
                    "method": method,
                    "cap": cap,
                    "error": None,
                }
                try:
                    if method == "fpbi":
                        measured = run_fpbi(case, seed, config)
                        fpbi_evaluations = measured["point_evaluations"]
                    else:
                        if method == "nsga_eval_matched" and fpbi_evaluations is None:
                            raise RuntimeError(
                                "FPBI failed; its point count is unavailable for budget matching"
                            )
                        generations = (
                            config["output_generations"]
                            if method == "nsga_output_cap"
                            else matched_generations(
                                fpbi_evaluations,
                                config["population"],
                                config["max_budget_generations"],
                            )
                        )
                        measured = run_nsga(case, seed, generations, config)
                    X, Y, checked = checked_front(case, measured["X"], measured["Y"])
                    uncapped = quality_metrics(case, Y, reference)
                    CX, CY = common_cap(case, X, Y, cap)
                    metrics = quality_metrics(case, CY, reference)
                    row.update(
                        {
                            key: value
                            for key, value in measured.items()
                            if key not in ("X", "Y", "diagnostics", "warnings")
                        }
                    )
                    row.update(checked)
                    row.update(metrics)
                    row.update(
                        {
                            "metric_rows": len(CY),
                            "budget_ratio": measured["point_evaluations"]
                            / fpbi_evaluations
                            if fpbi_evaluations
                            else None,
                            "uncapped_hv_ratio": uncapped["hv_ratio"],
                            "uncapped_igd": uncapped["igd"],
                        }
                    )
                    raw.append(
                        dict(
                            **row,
                            X=measured["X"].tolist(),
                            Y=measured["Y"].tolist(),
                            metric_X=CX.tolist(),
                            metric_Y=CY.tolist(),
                            diagnostics=measured["diagnostics"],
                            warnings=measured["warnings"],
                        )
                    )
                except Exception as error:  # noqa: BLE001 - record each failed run and continue the protocol.
                    if isinstance(error, GenerationFailure):
                        row.update(
                            {
                                k: v
                                for k, v in error.accounting.items()
                                if k != "warnings"
                            }
                        )
                    row["error"] = f"{type(error).__name__}: {error}"
                    raw.append(row.copy())
                rows.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
                json_write(output / "runs.json", raw)
    summary = summarize(rows)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summary)
    report(output, rows, summary, config)
    if not args.skip_plots:
        plots(output, summary, cases)
    final_provenance = provenance(config)
    final_provenance["source_unchanged_during_run"] = (
        initial_provenance["source_sha256"] == final_provenance["source_sha256"]
    )
    json_write(output / "provenance_end.json", final_provenance)
    print(f"Saved {len(rows)} measured runs to {output}", flush=True)


if __name__ == "__main__":
    main()
