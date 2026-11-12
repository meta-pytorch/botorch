"""Focus point boundary intersection, Schmitt et al. (2022), equations 16--27."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from openbo.decision_making._selection import _pareto, _validate_y

Objective = Callable[[Tensor], Tensor]


class _Posterior(Protocol):
    @property
    def mean(self) -> Tensor: ...


class _PosteriorModel(Protocol):
    def posterior(self, X: Tensor) -> _Posterior: ...


@dataclass(frozen=True)
class FPBIGeometry:
    """Normalized cost-space geometry; tensor fields preserve dtype/device.

    ``basis`` has direction vectors as columns. Zero columns pad flat axes.
    ``anchors`` uses the paper's asymmetric grid, equation 25, verbatim.
    ``utopia`` and ``nadir`` are in oriented *cost* coordinates (``-Y`` when
    maximizing). ``focus`` is always in normalized cost coordinates.
    """

    utopia: Tensor
    nadir: Tensor
    normalized_extremes: Tensor
    active_objectives: Tensor
    farthest_pair: Tensor
    center: Tensor
    focus: Tensor
    direction: Tensor
    basis: Tensor
    anchors: Tensor
    degenerate: Tensor
    used_fallback_basis: Tensor
    n_steps: int
    maximize: bool


@dataclass(frozen=True)
class SolverDiagnostic:
    """One SLSQP attempt, never silently accepted after solver failure."""

    kind: str
    index: int
    restart: int
    solver_success: bool
    accepted: bool
    status: int
    message: str
    max_violation: float
    kappa: float | None = None


@dataclass(frozen=True)
class FPBIResult:
    """Feasible, nondominated unique samples and every solver attempt."""

    X: Tensor
    Y: Tensor
    extreme_X: Tensor
    extreme_Y: Tensor
    geometry: FPBIGeometry
    diagnostics: tuple[SolverDiagnostic, ...]
    ray_succeeded: Tensor


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _cross(vectors: list[np.ndarray]) -> np.ndarray:
    """Generalized cross product, equation 23 (cofactor convention)."""
    A = np.stack(vectors)
    return np.asarray(
        [
            (-1.0) ** j * np.linalg.det(np.delete(A, j, axis=1))
            for j in range(A.shape[1])
        ]
    )


def _geometry_one(costs: np.ndarray, focus: np.ndarray, n_steps: int) -> dict:
    m = costs.shape[-1]
    # Tied scalar extrema can contain dominated rows. They must not distort
    # range normalization; equal rows remain for stable farthest-pair ties.
    le = np.all(costs[:, None] <= costs[None, :], axis=-1)
    strict = np.any(costs[:, None] < costs[None, :], axis=-1)
    earlier = np.arange(len(costs))[:, None] < np.arange(len(costs))[None, :]
    dominated = np.any(le & (strict | earlier), axis=0)
    front = costs[~dominated]
    lo, hi = front.min(0), front.max(0)
    with np.errstate(over="ignore"):
        raw_span = hi - lo
    if not np.isfinite(raw_span).all():
        raise ValueError("Extreme objective ranges overflow; rescale the objectives before FPBI.")
    scale = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.0)
    lo_s, hi_s = lo / scale, hi / scale
    span = hi_s - lo_s
    active = span > 0
    Z = np.where(active, (costs / scale - lo_s) / np.where(active, span, 1.0), 0.0)
    indices = np.flatnonzero(~dominated)
    distances = np.linalg.norm(Z[indices, None] - Z[None, indices], axis=-1)
    ia, ib = np.unravel_index(np.argmax(distances), distances.shape)
    a, b = int(indices[ia]), int(indices[ib])
    length = float(distances[ia, ib])
    center = (Z[a] + Z[b]) / 2
    ns = focus - center
    if np.any(focus[~active] != 0):
        raise ValueError("focus must be zero along flat objectives.")
    basis = np.zeros((m, max(m - 1, 0)))
    fallback = False
    active_idx = np.flatnonzero(active)
    k = len(active_idx)
    if length > 0 and k > 1:
        E = Z[:, active_idx]
        s = ns[active_idx]
        directions = [(E[b] - E[a]) / length]
        remainder = [i for i in indices if i not in (a, b)]
        aux = [E[i] - center[active_idx] for i in remainder]
        # Rank-deficient/correlated extrema need not provide enough independent
        # auxiliaries. The deterministic completion below is a library extension.
        for ell in range(1, k - 1):
            tail = aux[ell:] if len(aux) >= k - 2 else []
            rows = [s, *directions, *tail]
            candidate = _cross(rows) if len(rows) == k - 1 else np.zeros(k)
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-12:
                fallback = True
                # Complete perpendicular to ns and existing plane directions,
                # retaining the paper's d1 even when ns is not normal to it.
                best, best_norm = np.zeros(k), 0.0
                orthogonal: list[np.ndarray] = []
                for v in [*directions, s]:
                    v = v.copy()
                    for q in orthogonal:
                        v -= np.dot(v, q) * q
                    if np.linalg.norm(v) > 1e-12:
                        orthogonal.append(v / np.linalg.norm(v))
                for e in np.eye(k):
                    v = e.copy()
                    for q in orthogonal:
                        v -= np.dot(v, q) * q
                    vnorm = float(np.linalg.norm(v))
                    if vnorm > best_norm:
                        best, best_norm = v, vnorm
                candidate, norm = best, best_norm
            if norm <= 1e-12:
                raise ValueError("Cannot construct an FPBI sampling plane.")
            directions.append(candidate / norm)
        basis[np.ix_(active_idx, np.arange(k - 1))] = np.stack(directions, axis=1)
    gamma = (np.arange(1, n_steps + 1) - n_steps / 2) * length / n_steps
    coefficients = np.asarray(list(product(gamma, repeat=max(m - 1, 0))))
    coefficients = (
        coefficients.reshape(-1, max(m - 1, 0)) if m > 1 else np.zeros((1, 0))
    )
    anchors = center + coefficients @ basis.T
    return {
        "utopia": lo,
        "nadir": hi,
        "normalized_extremes": Z,
        "active_objectives": active,
        "farthest_pair": np.asarray([a, b]),
        "center": center,
        "focus": focus,
        "direction": ns,
        "basis": basis,
        "anchors": anchors,
        "degenerate": np.asarray(length == 0),
        "used_fallback_basis": np.asarray(fallback),
    }


def fpbi_geometry(
    extreme_Y: Tensor,
    *,
    n_steps: int = 10,
    focus: Tensor | None = None,
    maximize: bool = True,
    max_rays: int = 10000,
) -> FPBIGeometry:
    """Build the paper's FPBI plane/grid from individual extreme outcomes.

    ``extreme_Y`` has shape ``(..., m, m)``; row i is the outcome from
    optimizing objective i. ``focus`` broadcasts to ``(..., m)`` in normalized
    cost space, independently of decision preferences. The default is Utopia
    (zero). Constant axes map to zero; deficient bases use a deterministic
    orthogonal completion, reported in ``used_fallback_basis``.
    """
    _validate_y(extreme_Y)
    if extreme_Y.shape[-2] != extreme_Y.shape[-1]:
        raise ValueError("extreme_Y must have shape (..., m, m).")
    if extreme_Y.shape[-1] < 2:
        raise ValueError("FPBI requires at least two objectives.")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    n_steps, max_rays = (
        _positive_int(n_steps, "n_steps"),
        _positive_int(max_rays, "max_rays"),
    )
    m = extreme_Y.shape[-1]
    if n_steps ** (m - 1) > max_rays:
        raise ValueError("FPBI grid exceeds max_rays; lower n_steps or raise max_rays.")
    f = (
        torch.zeros_like(extreme_Y[..., 0, :])
        if focus is None
        else torch.as_tensor(focus, dtype=extreme_Y.dtype, device=extreme_Y.device)
    )
    try:
        f = torch.broadcast_to(f, extreme_Y.shape[:-2] + (m,))
    except RuntimeError as exc:
        raise ValueError("focus must broadcast to (..., m).") from exc
    if not torch.isfinite(f).all():
        raise ValueError("focus must be finite.")
    C = (-extreme_Y if maximize else extreme_Y).detach().cpu().double().numpy()
    fs = f.detach().cpu().double().numpy().reshape(-1, m)
    items = [_geometry_one(c, p, n_steps) for c, p in zip(C.reshape(-1, m, m), fs, strict=True)]
    fields = {}
    for name, value in items[0].items():
        dtype = (
            torch.bool
            if value.dtype == np.bool_
            else (torch.long if name == "farthest_pair" else extreme_Y.dtype)
        )
        arr = np.stack([item[name] for item in items])
        fields[name] = torch.as_tensor(
            arr, dtype=dtype, device=extreme_Y.device
        ).reshape(extreme_Y.shape[:-2] + value.shape)
    return FPBIGeometry(**fields, n_steps=n_steps, maximize=maximize)


def sample_fpbi(
    objective: Objective,
    bounds: Tensor,
    *,
    n_steps: int = 10,
    focus: Tensor | None = None,
    maximize: bool = True,
    inequality_constraints: Objective | None = None,
    equality_constraints: Objective | None = None,
    initial_X: Tensor | None = None,
    num_restarts: int = 4,
    seed: int = 0,
    maxiter: int = 300,
    feasibility_tol: float = 1e-6,
    max_rays: int = 10000,
) -> FPBIResult:
    """Optimize individual extrema, then solve every FPBI subproblem by SLSQP.

    ``objective(X)`` accepts ``(n,d)`` and returns ``(n,m)`` tensors. Bounds
    have BoTorch shape ``(2,d)``. Optional nonlinear constraint callables
    return finite tensors; inequalities follow SciPy/BoTorch's ``>=0``
    convention and equalities are zero. Bounds and constraints apply to both
    the extreme solves and ray solves. ``kappa`` is unrestricted as in eq. 27.

    The solver is local and uses finite differences on normalized inputs.
    Only successful, independently feasibility-checked solves are retained.
    An unsolved individual extreme raises; failed rays are exposed in
    diagnostics and ``ray_succeeded``. Returned tensors use bounds' dtype and
    device; SciPy optimization itself runs on the CPU without autograd.
    """
    if not isinstance(bounds, Tensor) or not bounds.is_floating_point():
        raise TypeError("bounds must be a floating point torch.Tensor.")
    if bounds.dtype not in (torch.float32, torch.float64):
        raise TypeError("SLSQP sampling requires float32 or float64 bounds.")
    if bounds.ndim != 2 or bounds.shape[0] != 2 or not bounds.shape[1]:
        raise ValueError("bounds must have nonempty shape (2, d).")
    if not torch.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("bounds must be finite and have upper > lower.")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    for value, name in [
        (num_restarts, "num_restarts"),
        (maxiter, "maxiter"),
        (n_steps, "n_steps"),
        (max_rays, "max_rays"),
    ]:
        _positive_int(value, name)
    if (
        isinstance(feasibility_tol, bool)
        or not math.isfinite(feasibility_tol)
        or feasibility_tol <= 0
    ):
        raise ValueError("feasibility_tol must be finite and positive.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    d = bounds.shape[1]
    low, width = bounds[0], bounds[1] - bounds[0]
    if not torch.isfinite(width).all():
        raise ValueError("bounds' widths must be finite.")

    def tensor_x(u: np.ndarray) -> Tensor:
        return (
            low
            + torch.as_tensor(
                np.asarray(u).copy(), dtype=bounds.dtype, device=bounds.device
            )
            * width
        )

    expected_m = None

    def values(u: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = objective(tensor_x(u).reshape(1, d))
        if (
            not isinstance(y, Tensor)
            or not y.is_floating_point()
            or y.ndim != 2
            or y.shape[0] != 1
            or y.shape[1] < 2
        ):
            raise ValueError("objective must return a tensor of shape (n, m).")
        if not torch.isfinite(y).all():
            raise ValueError("objective returned nonfinite values.")
        if expected_m is not None and y.shape[1] != expected_m:
            raise ValueError("objective changed its number of outputs.")
        return y.detach().cpu().double().numpy()[0]

    def constraints(fn: Objective | None, u: np.ndarray) -> np.ndarray:
        if fn is None:
            return np.zeros(0)
        with torch.no_grad():
            v = torch.as_tensor(fn(tensor_x(u).reshape(1, d)))
        if not torch.isfinite(v).all():
            raise ValueError("constraint callable returned nonfinite values.")
        return v.detach().cpu().double().numpy().reshape(-1)

    starts = (
        torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
        .draw(num_restarts, dtype=torch.double)
        .numpy()
    )
    starts[0] = 0.5
    if initial_X is not None:
        init = torch.as_tensor(initial_X, dtype=bounds.dtype, device=bounds.device)
        if init.ndim != 2 or init.shape[-1] != d or not len(init):
            raise ValueError("initial_X must have nonempty shape (n, d).")
        if (
            not torch.isfinite(init).all()
            or (init < bounds[0]).any()
            or (init > bounds[1]).any()
        ):
            raise ValueError("initial_X must be finite and lie inside bounds.")
        starts = np.concatenate(
            [((init - low) / width).detach().cpu().double().numpy(), starts]
        )
    first = values(starts[0])
    m = len(first)
    expected_m = m
    if n_steps ** (m - 1) > max_rays:
        raise ValueError("FPBI grid exceeds max_rays.")
    sign = -1.0 if maximize else 1.0
    diagnostics: list[SolverDiagnostic] = []
    # SLSQP's default double-precision probes vanish after casting to float32;
    # its stopping tolerance must likewise respect the evaluator's precision.
    eps = math.sqrt(torch.finfo(bounds.dtype).eps)
    options = {
        "maxiter": maxiter,
        "ftol": max(min(feasibility_tol / 10, 1e-9), torch.finfo(bounds.dtype).eps * 2),
        "eps": eps,
    }

    def base_constraints(with_kappa: bool = False) -> list[dict]:
        get_u = (lambda v: v[:-1]) if with_kappa else (lambda v: v)
        result = []
        if inequality_constraints is not None:
            result.append(
                {
                    "type": "ineq",
                    "fun": lambda v: constraints(inequality_constraints, get_u(v)),
                }
            )
        if equality_constraints is not None:
            result.append(
                {
                    "type": "eq",
                    "fun": lambda v: constraints(equality_constraints, get_u(v)),
                }
            )
        return result

    def violation(u: np.ndarray, ray_residual: np.ndarray | None = None) -> float:
        terms = [0.0, float(np.max(-u)), float(np.max(u - 1))]
        iv = constraints(inequality_constraints, u)
        ev = constraints(equality_constraints, u)
        if len(iv):
            terms.append(float(np.max(-iv)))
        if len(ev):
            terms.append(float(np.max(np.abs(ev))))
        if ray_residual is not None:
            terms.append(float(np.max(-ray_residual)))
        return max(terms)

    extreme_u, extreme_y = [], []
    for i in range(m):
        best = None
        for restart, u0 in enumerate(starts):
            opt = minimize(
                lambda u, index=i: sign * values(u)[index],
                u0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * d,
                constraints=base_constraints(),
                options=options,
            )
            viol = violation(opt.x) if np.isfinite(opt.x).all() else float("inf")
            accepted = bool(
                opt.success and viol <= feasibility_tol and math.isfinite(opt.fun)
            )
            diagnostics.append(
                SolverDiagnostic(
                    "extreme",
                    i,
                    restart,
                    bool(opt.success),
                    accepted,
                    int(opt.status),
                    str(opt.message),
                    viol,
                )
            )
            if accepted and (best is None or opt.fun < best.fun):
                best = opt
        if best is None:
            raise RuntimeError(
                f"No successful feasible solve for extreme objective {i}."
            )
        extreme_u.append(best.x)
        extreme_y.append(values(best.x))
    extreme_U = np.asarray(extreme_u)
    EY = torch.as_tensor(
        np.asarray(extreme_y), dtype=bounds.dtype, device=bounds.device
    )
    geometry = fpbi_geometry(
        EY, n_steps=n_steps, focus=focus, maximize=maximize, max_rays=max_rays
    )
    lo = geometry.utopia.detach().cpu().double().numpy()
    hi = geometry.nadir.detach().cpu().double().numpy()
    scale = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.0)
    lo_s = lo / scale
    span = hi / scale - lo_s
    active = span > 0

    def normalized(u: np.ndarray) -> np.ndarray:
        return np.where(
            active, (sign * values(u) / scale - lo_s) / np.where(active, span, 1.0), 0.0
        )

    ns = geometry.direction.detach().cpu().double().numpy()
    anchors = geometry.anchors.detach().cpu().double().numpy()
    all_u, all_y = list(extreme_U), list(np.asarray(extreme_y))
    ray_ok = np.zeros(len(anchors), dtype=bool)
    if not bool(geometry.degenerate):
        if not np.any(ns < 0):
            raise ValueError(
                "focus gives no negative search-direction component; kappa is unbounded."
            )
        seen: dict[tuple[float, ...], bool] = {}
        for i, anchor in enumerate(anchors):
            key = tuple(anchor)
            if key in seen:
                ray_ok[i] = seen[key]
                continue

            def ray_residual(v: np.ndarray, anchor: np.ndarray = anchor) -> np.ndarray:
                return anchor + v[-1] * ns - normalized(v[:-1])

            best = None
            for restart, u0 in enumerate(np.concatenate([extreme_U, starts])):
                # A negative initial kappa is legitimate and sometimes needed.
                diff = anchor - normalized(u0)
                k0 = float(np.min(diff[ns < 0] / -ns[ns < 0]))
                v0 = np.append(u0, min(k0, 0.0))
                cons = base_constraints(True) + [{"type": "ineq", "fun": ray_residual}]
                opt = minimize(
                    lambda v: -v[-1],
                    v0,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * d + [(None, None)],
                    constraints=cons,
                    options=options,
                )
                finite = bool(np.isfinite(opt.x).all() and math.isfinite(opt.fun))
                viol = (
                    violation(opt.x[:-1], ray_residual(opt.x))
                    if finite
                    else float("inf")
                )
                accepted = bool(opt.success and viol <= feasibility_tol and finite)
                diagnostics.append(
                    SolverDiagnostic(
                        "ray",
                        i,
                        restart,
                        bool(opt.success),
                        accepted,
                        int(opt.status),
                        str(opt.message),
                        viol,
                        float(opt.x[-1]) if finite else None,
                    )
                )
                if accepted and (best is None or opt.fun < best.fun):
                    best = opt
            seen[key] = ray_ok[i] = best is not None
            if best is not None:
                all_u.append(best.x[:-1])
                all_y.append(values(best.x[:-1]))
    Y = torch.as_tensor(np.asarray(all_y), dtype=bounds.dtype, device=bounds.device)
    U = torch.as_tensor(np.asarray(all_u), dtype=bounds.dtype, device=bounds.device)
    keep = _pareto(sign * Y)
    # Objective-space duplicate removal is deterministic and scale normalized.
    chosen: list[int] = []
    scale_t = torch.maximum(geometry.utopia.abs(), geometry.nadir.abs()).clamp_min(1)
    span_t = geometry.nadir / scale_t - geometry.utopia / scale_t
    Z = (sign * Y / scale_t - geometry.utopia / scale_t) / torch.where(
        span_t > 0, span_t, torch.ones_like(span_t)
    )
    for i in torch.where(keep)[0].tolist():
        if not any(
            torch.allclose(Z[i], Z[j], rtol=0.0, atol=feasibility_tol) for j in chosen
        ):
            chosen.append(i)
    chosen_tensor = torch.tensor(chosen, device=bounds.device, dtype=torch.long)
    return FPBIResult(
        X=low + U[chosen_tensor] * width,
        Y=Y[chosen_tensor],
        extreme_X=tensor_x(extreme_U),
        extreme_Y=EY,
        geometry=geometry,
        diagnostics=tuple(diagnostics),
        ray_succeeded=torch.as_tensor(ray_ok, device=bounds.device),
    )


def sample_model_fpbi(
    model: _PosteriorModel,
    bounds: Tensor,
    *,
    model_batch_index: tuple[int, ...] | None = None,
    **kwargs,
) -> FPBIResult:
    """Sample a BoTorch-compatible model's posterior *mean* objective front.

    Works with any ``model.posterior(X).mean`` tensor interface, including
    botorch_rs. This optimizes the mean, not posterior samples or an acquisition
    function. For batched models explicitly identify one ``model_batch_index``.
    """

    def objective(X: Tensor) -> Tensor:
        mean = model.posterior(X).mean
        if model_batch_index is not None:
            mean = mean[model_batch_index]
        if mean.ndim != 2:
            raise ValueError(
                "Batched posterior: provide model_batch_index to select one problem."
            )
        return mean

    return sample_fpbi(objective, bounds, **kwargs)
