#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for MC and qMC sampling.

References

.. [Trikalinos2014polytope]
    T. A. Trikalinos and G. van Valkenhoef. Efficient sampling from uniform
    density n-polytopes. Technical report, Brown University, 2014.
"""

from __future__ import annotations

import warnings

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy

import torch

from botorch.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    InfeasibilityError,
)
from botorch.exceptions.warnings import UserInputWarning
from botorch.sampling.qmc import NormalQMCEngine

from botorch.utils.transforms import normalize, standardize, unnormalize
from scipy.spatial import Delaunay, HalfspaceIntersection
from torch import LongTensor, Tensor
from torch.distributions import Normal
from torch.quasirandom import SobolEngine


if TYPE_CHECKING:
    from botorch.models.deterministic import (  # pragma: no cover
        GenericDeterministicModel,
    )


@contextmanager
def manual_seed(seed: int | None = None) -> Generator[None, None, None]:
    r"""Contextmanager for manual setting the torch.random seed.

    Args:
        seed: The seed to set the random number generator to.

    Returns:
        Generator

    Example:
        >>> with manual_seed(1234):
        >>>     X = torch.rand(3)
    """
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def draw_sobol_samples(
    bounds: Tensor,
    n: int,
    q: int,
    batch_shape: Iterable[int] | torch.Size | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples. As a best practice, use powers of 2.
        q: The size of each q-batch.
        batch_shape: The batch shape of the samples. If given, returns samples
            of shape `n x batch_shape x q x d`, where each batch is an
            `n x q x d`-dim tensor of qMC samples.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x batch_shape x q x d`-dim tensor of qMC samples from the box
        defined by bounds.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        >>> samples = draw_sobol_samples(bounds, 16, 2)
    """
    batch_shape = batch_shape or torch.Size()
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    d = bounds.shape[-1]
    sobol_engine = SobolEngine(q * d, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(batch_size * n, dtype=bounds.dtype)
    samples_raw = samples_raw.view(*batch_shape, n, q, d).to(device=bounds.device)
    if batch_shape != torch.Size():
        samples_raw = samples_raw.permute(-3, *range(len(batch_shape)), -2, -1)
    return unnormalize(samples_raw, bounds, update_constant_bounds=False)


def draw_sobol_normal_samples(
    d: int,
    n: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int | None = None,
) -> Tensor:
    r"""Draw qMC samples from a multi-variate standard normal N(0, I_d).

    A primary use-case for this functionality is to compute an QMC average
    of f(X) over X where each element of X is drawn N(0, 1).

    Args:
        d: The dimension of the normal distribution.
        n: The number of samples to return. As a best practice, use powers of 2.
        device: The torch device.
        dtype:  The torch dtype.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A tensor of qMC standard normal samples with dimension `n x d` with device
        and dtype specified by the input.

    Example:
        >>> samples = draw_sobol_normal_samples(2, 16)
    """
    normal_qmc_engine = NormalQMCEngine(d=d, seed=seed, inv_transform=True)
    samples = normal_qmc_engine.draw(n, dtype=dtype)
    return samples.to(device=device)


def sample_hypersphere(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Sample uniformly from a unit d-sphere.

    Args:
        d: The dimension of the hypersphere.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype:  The torch dtype.

    Returns:
        An  `n x d` tensor of uniform samples from from the d-hypersphere.

    Example:
        >>> sample_hypersphere(d=5, n=10)
    """
    if d == 1:
        with manual_seed(seed=seed):
            rnd = torch.randint(0, 2, (n, 1), device=device, dtype=dtype)
        return 2 * rnd - 1
    if qmc:
        rnd = draw_sobol_normal_samples(d=d, n=n, device=device, dtype=dtype, seed=seed)
    else:
        with manual_seed(seed=seed):
            rnd = torch.randn(n, d, device=device, dtype=dtype)
    samples = rnd / torch.linalg.norm(rnd, dim=-1, keepdim=True)
    return samples


def sample_simplex(
    d: int,
    n: int = 1,
    qmc: bool = False,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    r"""Sample uniformly from a d-simplex.

    Args:
        d: The dimension of the simplex.
        n: The number of samples to return.
        qmc: If True, use QMC Sobol sampling (instead of i.i.d. uniform).
        seed: If provided, use as a seed for the RNG.
        device: The torch device.
        dtype: The torch dtype.

    Returns:
        An `n x d` tensor of uniform samples from from the d-simplex.

    Example:
        >>> sample_simplex(d=3, n=10)
    """
    if d == 1:
        return torch.ones(n, 1, device=device, dtype=dtype)
    if qmc:
        sobol_engine = SobolEngine(d - 1, scramble=True, seed=seed)
        rnd = sobol_engine.draw(n, dtype=dtype)
    else:
        with manual_seed(seed=seed):
            rnd = torch.rand(n, d - 1, dtype=dtype)
    srnd, _ = torch.sort(rnd, dim=-1)
    zeros = torch.zeros(n, 1, dtype=dtype)
    ones = torch.ones(n, 1, dtype=dtype)
    srnd = torch.cat([zeros, srnd, ones], dim=-1)
    if device is not None:
        srnd = srnd.to(device)
    return srnd[..., 1:] - srnd[..., :-1]


def sample_polytope(
    A: Tensor,
    b: Tensor,
    x0: Tensor,
    n: int = 10000,
    n0: int = 100,
    n_thinning: int = 1,
    seed: int | None = None,
) -> Tensor:
    r"""
    Hit and run sampler from uniform sampling points from a polytope,
    described via inequality constraints A*x<=b.

    Args:
        A: A `m x d`-dim Tensor describing inequality constraints
            so that all samples satisfy `Ax <= b`.
        b: A `m`-dim Tensor describing the inequality constraints
            so that all samples satisfy `Ax <= b`.
        x0: A `d`-dim Tensor representing a starting point of the chain
            satisfying the constraints.
        n: The number of resulting samples kept in the output.
        n0: The number of burn-in samples. The chain will produce
            n+n0 samples but the first n0 samples are not saved.
        n_thinning: The amount of thinnning. This function will return every
            `n_thinning`-th sample from the chain (after burn-in).
        seed: The seed for the sampler. If omitted, use a random seed.

    Returns:
        (n, d) dim Tensor containing the resulting samples.
    """
    # Check that starting point satisfies the constraints.
    if not ((slack := A @ x0 - b) <= 0).all():
        raise InfeasibilityError(
            f"Starting point does not satisfy the constraints. Inputs: {A=},"
            f"{b=}, {x0=}, A@x0-b={slack}."
        )
    # Remove rows where all elements of A are 0. This avoids nan and infs later.
    # A may have zero rows in it when this is called from PolytopeSampler
    # with equality constraints (which are absorbed into A & b).
    non_zero_rows = torch.any(A != 0, dim=-1)
    A = A[non_zero_rows]
    b = b[non_zero_rows]

    n_tot = n0 + n * n_thinning
    seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()
    with manual_seed(seed=seed):
        rands = torch.rand(n_tot, dtype=A.dtype, device=A.device)

    # Sample uniformly from unit hypersphere in d dims.
    # Increment seed by +1 to avoid correlation with step size, see #2156 for details.
    Rs = sample_hypersphere(
        d=x0.shape[0], n=n_tot, dtype=A.dtype, device=A.device, seed=seed + 1
    ).unsqueeze(-1)

    # Use batch operations for matrix multiplication.
    ARs = (A @ Rs).squeeze(-1)
    out = torch.empty(n, A.size(-1), dtype=A.dtype, device=A.device)
    x = x0.clone()
    large_constant = torch.finfo().max
    for i, (ar, r, rnd) in enumerate(zip(ARs, Rs, rands)):
        # Given x, the next point in the chain is x+alpha*r.
        # It must satisfy A(x+alpha*r)<=b, which implies A*alpha*r<=b-Ax,
        # so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        # If x is at the boundary, b - Ax = 0. If ar > 0, then we must
        # have alpha <= 0. If ar < 0, we must have alpha >= 0.
        # ar == 0 is an unlikely event that provides no signal.
        # b - A @ x is always >= 0, clamping for numerical tolerances.
        w = (b - A @ x).squeeze().clamp(min=0.0) / ar
        # Find upper bound for alpha. If there are no constraints on
        # the upper bound of alpha, set it to a large value.
        pos = w > 0
        alpha_max = w[pos].min().item() if pos.any() else large_constant
        # Find lower bound for alpha.
        neg = w < 0
        alpha_min = w[neg].max().item() if neg.any() else -large_constant
        # Handle the boundary case.
        if (w_eq_0 := (w == 0)).any():
            # If ar > 0 at the boundary, alpha <= 0.
            if w_eq_0.logical_and(ar > 0).any():
                alpha_max = min(alpha_max, 0.0)
            # If ar < 0 at the boundary, alpha >= 0.
            if w_eq_0.logical_and(ar < 0).any():
                alpha_min = max(alpha_min, 0.0)
        # alpha ~ Uniform[alpha_min, alpha_max]
        alpha = alpha_min + rnd * (alpha_max - alpha_min)
        x = x + alpha * r
        if (k := i - n0) >= 0:  # save samples after burn-in period
            idx, rem = divmod(k, n_thinning)
            if rem == 0:
                out[idx] = x.squeeze()
    return out


def batched_multinomial(
    weights: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: torch.Generator | None = None,
    out: Tensor | None = None,
) -> LongTensor:
    r"""Sample from multinomial with an arbitrary number of batch dimensions.

    Args:
        weights: A `batch_shape x num_categories` tensor of weights. For each batch
            index `i, j, ...`, this functions samples from a multinomial with `input`
            `weights[i, j, ..., :]`. Note that the weights need not sum to one, but must
            be non-negative, finite and have a non-zero sum.
        num_samples: The number of samples to draw for each batch index. Must be smaller
            than `num_categories` if `replacement=False`.
        replacement: If True, samples are drawn with replacement.
        generator: A a pseudorandom number generator for sampling.
        out: The output tensor (optional). If provided, must be of size
            `batch_shape x num_samples`.

    Returns:
        A `batch_shape x num_samples` tensor of samples.

    This is a thin wrapper around `torch.multinomial` that allows weight (`input`)
    tensors with an arbitrary number of batch dimensions (`torch.multinomial` only
    allows a single batch dimension). The calling signature is the same as for
    `torch.multinomial`.

    Example:
        >>> weights = torch.rand(2, 3, 10)
        >>> samples = batched_multinomial(weights, 4)  # shape is 2 x 3 x 4
    """
    batch_shape, n_categories = weights.shape[:-1], weights.size(-1)
    flat_samples = torch.multinomial(
        input=weights.view(-1, n_categories),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
        out=None if out is None else out.view(-1, num_samples),
    )
    return flat_samples.view(*batch_shape, num_samples)


def _convert_bounds_to_inequality_constraints(bounds: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert bounds into inequality constraints of the form Ax <= b.

    Args:
        bounds: A `2 x d`-dim tensor of bounds

    Returns:
        A two-element tuple containing
            - A: A `2d x d`-dim tensor of coefficients
            - b: A `2d x 1`-dim tensor containing the right hand side
    """
    d = bounds.shape[-1]
    eye = torch.eye(d, dtype=bounds.dtype, device=bounds.device)
    lower, upper = bounds
    lower_finite, upper_finite = bounds.isfinite()
    A = torch.cat([-eye[lower_finite], eye[upper_finite]], dim=0)
    b = torch.cat([-lower[lower_finite], upper[upper_finite]], dim=0).unsqueeze(-1)
    return A, b


def find_interior_point(
    A: npt.NDArray,
    b: npt.NDArray,
    A_eq: npt.NDArray | None = None,
    b_eq: npt.NDArray | None = None,
) -> npt.NDArray:
    r"""Find an interior point of a polytope via linear programming.

    Args:
        A: A `n_ineq x d`-dim numpy array containing the coefficients of the
            constraint inequalities.
        b: A `n_ineq x 1`-dim numpy array containing the right hand sides of
            the constraint inequalities.
        A_eq: A `n_eq x d`-dim numpy array containing the coefficients of the
            constraint equalities.
        b_eq: A `n_eq x 1`-dim numpy array containing the right hand sides of
            the constraint equalities.

    Returns:
        A `d`-dim numpy array containing an interior point of the polytope.
        This function will raise a ValueError if there is no such point.

    This method solves the following Linear Program:

        min -s subject to A @ x <= b - 2 * s, s >= 0, A_eq @ x = b_eq

    In case the polytope is unbounded, then it will also constrain the slack
    variable `s` to `s<=1`.
    """
    # augment inequality constraints: A @ (x, s) <= b
    d = A.shape[-1]
    ncon = A.shape[-2] + 1
    c = np.zeros(d + 1)
    c[-1] = -1
    b_ub = np.zeros(ncon)
    b_ub[:-1] = b.reshape(-1)
    A_ub = np.zeros((ncon, d + 1))
    A_ub[:-1, :-1] = A
    A_ub[:-1, -1] = 2.0
    A_ub[-1, -1] = -1.0

    result = scipy.optimize.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(None, None),
        method="highs",
    )

    if result.status == 3:
        # problem is unbounded - to find a bounded solution we constrain the
        # slack variable `s` to `s <= 1.0`.
        A_s = np.concatenate([np.zeros((1, d)), np.ones((1, 1))], axis=-1)
        A_ub = np.concatenate([A_ub, A_s], axis=0)
        b_ub = np.concatenate([b_ub, np.ones(1)], axis=-1)
        result = scipy.optimize.linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(None, None),
            method="highs",
        )

    if result.status == 2:
        raise InfeasibilityError(
            "No feasible point found. Constraint polytope appears empty. "
            + "Check your constraints."
        )
    elif result.status > 0:
        raise ValueError(
            "Problem checking constraint specification. "
            + f"linprog status: {result.message}"
        )
    # the x in the result is really (x, s)
    return result.x[:-1]


class PolytopeSampler(ABC):
    """Base class for samplers that sample points from a polytope."""

    def __init__(
        self,
        inequality_constraints: tuple[Tensor, Tensor] | None = None,
        equality_constraints: tuple[Tensor, Tensor] | None = None,
        bounds: Tensor | None = None,
        interior_point: Tensor | None = None,
    ) -> None:
        r"""
        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded above (below).
            interior_point: A `d x 1`-dim Tensor presenting a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.
        """
        if inequality_constraints is None:
            if bounds is None:
                raise BotorchError(
                    "PolytopeSampler requires either inequality constraints or bounds."
                )
            A = torch.empty(
                0, bounds.shape[-1], dtype=bounds.dtype, device=bounds.device
            )
            b = torch.empty(0, 1, dtype=bounds.dtype, device=bounds.device)
        else:
            A, b = inequality_constraints
        if bounds is not None:
            # add inequality constraints for bounds
            # TODO: make sure there are not deduplicate constraints
            A2, b2 = _convert_bounds_to_inequality_constraints(bounds=bounds)
            A = torch.cat([A, A2], dim=0)
            b = torch.cat([b, b2], dim=0)
        self.A = A
        self.b = b
        self.equality_constraints = equality_constraints

        if equality_constraints is not None:
            self.C, self.d = equality_constraints
            U, S, Vh = torch.linalg.svd(self.C)
            r = torch.nonzero(S).size(0)  # rank of matrix C
            self.nullC = Vh[r:, :].transpose(-1, -2)  # orthonormal null space of C,
            # satisfying # C @ nullC = 0 and nullC.T @ nullC = I
            # using the change of variables x=x0+nullC*y,
            # sample y satisfies A*nullC*y<=b-A*x0.
            # the linear constraint is automatically satisfied as x0 satisfies it.
        else:
            self.C = None
            self.d = None
            self.nullC = torch.eye(
                self.A.size(-1), dtype=self.A.dtype, device=self.A.device
            )

        self.new_A = self.A @ self.nullC  # doesn't depend on the initial point

        # initial point for the original, not transformed, problem
        if interior_point is not None:
            if self.feasible(interior_point):
                self.x0 = interior_point
            else:
                raise InfeasibilityError("The given input point is not feasible.")
        else:
            self.x0 = self.find_interior_point()

    def feasible(self, x: Tensor) -> bool:
        r"""Check whether a point is contained in the polytope.

        Args:
            x: A `d x 1`-dim Tensor.

        Returns:
            True if `x` is contained inside the polytope (incl. its boundary),
            False otherwise.
        """
        ineq = (self.A @ x - self.b <= 0).all()
        if self.equality_constraints is not None:
            eq = (self.C @ x - self.d == 0).all()
            return ineq & eq
        return ineq

    def find_interior_point(self) -> Tensor:
        r"""Find an interior point of the polytope.

        Returns:
            A `d x 1`-dim Tensor representing a point contained in the polytope.
            This function will raise a ValueError if there is no such point.
        """
        if self.equality_constraints:
            # equality constraints: A_eq * (x, s) = b_eq
            A_eq = np.zeros((self.C.size(0), self.C.size(-1) + 1))
            A_eq[:, :-1] = self.C.cpu().numpy()
            b_eq = self.d.cpu().numpy()
        else:
            A_eq = None
            b_eq = None
        x0 = find_interior_point(
            A=self.A.cpu().numpy(), b=self.b.cpu().numpy(), A_eq=A_eq, b_eq=b_eq
        )
        return torch.from_numpy(x0).to(self.A).unsqueeze(-1)

    # -------- Abstract methods to be implemented by subclasses -------- #

    @abstractmethod
    def draw(self, n: int = 1) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        pass  # pragma: no cover


class HitAndRunPolytopeSampler(PolytopeSampler):
    r"""A sampler for sampling from a polyope using a hit-and-run algorithm."""

    def __init__(
        self,
        inequality_constraints: tuple[Tensor, Tensor] | None = None,
        equality_constraints: tuple[Tensor, Tensor] | None = None,
        bounds: Tensor | None = None,
        interior_point: Tensor | None = None,
        n_burnin: int = 200,
        n_thinning: int = 20,
        seed: int | None = None,
    ) -> None:
        r"""A sampler for sampling from a polyope using a hit-and-run algorithm.

        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded from above (below). If
                omitted, no bounds (in addition to the above constraints) are applied.
            interior_point: A `d x 1`-dim Tensor representing a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.
            n_burnin: The number of burn in samples. The sampler will discard
                `n_burnin` samples before returning the first sample.
            n_thinning: The amount of thinning. The sampler will return every
                `n_thinning` sample (after burn-in). This may need to be increased
                for sets of constraints that are difficult to satisfy (i.e. in which
                case the volume of the constraint polytope is small relative to that
                of its bounding box).
            seed: The random seed.
        """
        if inequality_constraints is None and bounds is None:
            raise BotorchError(
                "HitAndRunPolytopeSampler requires either inequality constraints "
                "or bounds."
            )
        # Normalize constraints to avoid the following issue:
        # https://github.com/meta-pytorch/botorch/issues/1225
        offset, scale = None, None
        if inequality_constraints or equality_constraints:
            if bounds is None:
                warnings.warn(
                    "HitAndRunPolytopeSampler did not receive `bounds`, which can "
                    "lead to non-uniform sampling if the parameter ranges are very "
                    "different (see "
                    "https://github.com/meta-pytorch/botorch/issues/1225).",
                    UserInputWarning,
                    stacklevel=3,
                )
            else:
                if inequality_constraints:
                    inequality_constraints = normalize_dense_linear_constraints(
                        bounds=bounds, constraints=inequality_constraints
                    )
                if equality_constraints:
                    equality_constraints = normalize_dense_linear_constraints(
                        bounds=bounds, constraints=equality_constraints
                    )
                lower, upper = bounds
                offset = lower
                scale = upper - lower
                if interior_point is not None:
                    # If provided, we also need to normalize the interior point
                    interior_point = (interior_point - offset[:, None]) / scale[:, None]
                bounds = torch.zeros_like(bounds)
                bounds[1, :] = 1.0

        super().__init__(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds,
            interior_point=interior_point,
        )
        self.n_burnin: int = n_burnin
        self.n_thinning: int = n_thinning
        self.num_samples_generated: int = 0
        self._seed: int | None = seed
        self._offset: Tensor | None = offset
        self._scale: Tensor | None = scale

    def draw(self, n: int = 1) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        # There are two layers of normalization. In the outer layer, the space
        # has been normalized to the unit cube. In the inner layer, we remove
        # any equality constraints and sample on the subspace defined by those
        # equality constraints, with an additional shift to normalize the interior
        # point to the origin. Below, after sampling in that inner layer, we have
        # to reverse both layers of normalization.
        transformed_samples = sample_polytope(
            # Run this on the cpu since there is a lot of looping going on
            A=self.new_A.cpu(),
            b=(self.b - self.A @ self.x0).cpu(),
            x0=torch.zeros(
                (self.nullC.size(1), 1), dtype=self.A.dtype, device=torch.device("cpu")
            ),
            n=n,
            n0=self.n_burnin if self.num_samples_generated == 0 else 0,
            n_thinning=self.n_thinning,
            seed=self._seed,
        ).to(self.b)
        # Update the seed for the next call in a deterministic fashion
        if self._seed is not None:
            self._seed += n
        # Unnormalize the inner layer
        init_shift = self.x0.transpose(-1, -2)
        samples = init_shift + transformed_samples @ self.nullC.transpose(-1, -2)
        # Keep the last element as the beginning of the next chain
        self.x0 = samples[-1].reshape(-1, 1)
        # Unnormalize the outer layer
        if self._scale is not None:
            samples = self._offset + self._scale * samples
        self.num_samples_generated += n
        return samples


class DelaunayPolytopeSampler(PolytopeSampler):
    r"""A polytope sampler using Delaunay triangulation.

    This sampler first enumerates the vertices of the constraint polytope and
    then uses a Delaunay triangulation to tesselate its convex hull.

    The sampling happens in two stages:
    1. First, we sample from the set of hypertriangles generated by the
    Delaunay triangulation (i.e. which hyper-triangle to draw the sample
    from) with probabilities proportional to the triangle volumes.
    2. Then, we sample uniformly from the chosen hypertriangle by sampling
    uniformly from the unit simplex of the appropriate dimension, and
    then computing the convex combination of the vertices of the
    hypertriangle according to that draw from the simplex.

    The best reference (not exactly the same, but functionally equivalent) is
    [Trikalinos2014polytope]_. A simple R implementation is available at
    https://github.com/gertvv/tesselample.
    """

    def __init__(
        self,
        inequality_constraints: tuple[Tensor, Tensor] | None = None,
        equality_constraints: tuple[Tensor, Tensor] | None = None,
        bounds: Tensor | None = None,
        interior_point: Tensor | None = None,
    ) -> None:
        r"""Initialize DelaunayPolytopeSampler.

        Args:
            inequality_constraints: Tensors `(A, b)` describing inequality
                constraints `A @ x <= b`, where `A` is a `n_ineq_con x d`-dim
                Tensor and `b` is a `n_ineq_con x 1`-dim Tensor, with `n_ineq_con`
                the number of inequalities and `d` the dimension of the sample space.
            equality_constraints: Tensors `(C, d)` describing the equality constraints
                `C @ x = d`, where `C` is a `n_eq_con x d`-dim Tensor and `d` is a
                `n_eq_con x 1`-dim Tensor with `n_eq_con` the number of equalities.
            bounds: A `2 x d`-dim tensor of box bounds, where `inf` (`-inf`) means
                that the respective dimension is unbounded from above (below).
            interior_point: A `d x 1`-dim Tensor representing a point in the
                (relative) interior of the polytope. If omitted, determined
                automatically by solving a Linear Program.

        Warning: The vertex enumeration performed in this algorithm can become
        extremely costly if there are a large number of inequalities. Similarly,
        the triangulation can get very expensive in high dimensions. Only use
        this algorithm for moderate dimensions / moderately complex constraint sets.
        An alternative is the `HitAndRunPolytopeSampler`.
        """
        super().__init__(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds,
            interior_point=interior_point,
        )
        # shift coordinate system to be anchored at x0
        new_b = self.b - self.A @ self.x0
        if self.new_A.shape[-1] < 2:
            # if the polytope is in dim 1 (i.e. a line segment) Qhull won't work
            tshlds = new_b / self.new_A
            neg = self.new_A < 0
            self.y_min = tshlds[neg].max()
            self.y_max = tshlds[~neg].min()
            self.dim = 1
        else:
            # Qhull expects inputs of the form A @ x + b <= 0, so we need to negate here
            halfspaces = torch.cat([self.new_A, -new_b], dim=-1).cpu().numpy()
            vertices = HalfspaceIntersection(
                halfspaces=halfspaces, interior_point=np.zeros(self.new_A.shape[-1])
            ).intersections
            self.dim = vertices.shape[-1]
            try:
                delaunay = Delaunay(vertices)
            except ValueError as e:
                if "Points cannot contain NaN" in str(e):
                    raise ValueError("Polytope is unbounded.")
                raise e  # pragma: no cover
            polytopes = torch.from_numpy(
                np.array([delaunay.points[s] for s in delaunay.simplices]),
            ).to(self.A)
            volumes = torch.stack([torch.det(p[1:] - p[0]).abs() for p in polytopes])
            self._polytopes = polytopes
            self._p = volumes / volumes.sum()

    def draw(self, n: int = 1, seed: int | None = None) -> Tensor:
        r"""Draw samples from the polytope.

        Args:
            n: The number of samples.
            seed: The random seed.

        Returns:
            A `n x d` Tensor of samples from the polytope.
        """
        if self.dim == 1:
            with manual_seed(seed):
                e = torch.rand(n, 1, device=self.new_A.device, dtype=self.new_A.dtype)
            transformed_samples = self.y_min + (self.y_max - self.y_min) * e
        else:
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.A.device)
                generator.manual_seed(seed)
            index_rvs = torch.multinomial(
                self._p,
                num_samples=n,
                replacement=True,
                generator=generator,
            )
            simplex_rvs = sample_simplex(
                d=self.dim + 1, n=n, seed=seed, device=self.A.device, dtype=self.A.dtype
            )
            transformed_samples = torch.stack(
                [rv @ self._polytopes[idx] for rv, idx in zip(simplex_rvs, index_rvs)]
            )
        init_shift = self.x0.transpose(-1, -2)
        samples = init_shift + transformed_samples @ self.nullC.transpose(-1, -2)
        return samples


def normalize_sparse_linear_constraints(
    bounds: Tensor, constraints: list[tuple[Tensor, Tensor, float]]
) -> list[tuple[Tensor, Tensor, float]]:
    r"""Normalize sparse linear constraints to the unit cube.

    Args:
        bounds: A `2 x d`-dim tensor containing the box bounds.
        constraints: A list of tuples (`indices`, `coefficients`, `rhs`), with
            `indices` and `coefficients` one-dimensional tensors and `rhs` a
            scalar, where each tuple encodes an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs` or
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
    """
    new_constraints = []
    for index, coefficient, rhs in constraints:
        if index.ndim != 1:
            raise ValueError(
                "`indices` must be a one-dimensional tensor. This method does not "
                "support the kind of 'inter-point constraints' that are supported by "
                "`optimize_acqf()`. To achieve this behavior, you need define the "
                "problem on the joint space over `q` points and impose use constraints,"
                "see https://github.com/meta-pytorch/botorch/issues/2468#issuecomment-2287706461"  # noqa: E501
            )
        lower, upper = bounds[:, index]
        s = upper - lower
        new_constraints.append(
            (index, s * coefficient, (rhs - torch.dot(coefficient, lower)).item())
        )
    return new_constraints


def normalize_dense_linear_constraints(
    bounds: Tensor,
    constraints: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    r"""Normalize dense linear constraints to the unit cube.

    Args:
        bounds: A `2 x d`-dim tensor containing the box bounds.
        constraints: A tensor tuple `(A, b)` describing constraints
            `A @ x (<)= b`, where `A` is a `n_con x d`-dim Tensor and
            `b` is a `n_con x 1`-dim Tensor, with `n_con` the number of
            constraints and `d` the dimension of the sample space.

    Returns:
        A tensor tuple `(A_nlz, b_nlz)` of normalized constraints.
    """
    lower, upper = bounds
    A, b = constraints
    A_nlz = (upper - lower) * A
    b_nlz = b - (A @ lower).unsqueeze(-1)
    return A_nlz, b_nlz


def get_polytope_samples(
    n: int,
    bounds: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    seed: int | None = None,
    n_burnin: int = 10_000,
    n_thinning: int = 32,
) -> Tensor:
    r"""Sample from polytope defined by box bounds and (in)equality constraints.

    This uses a hit-and-run Markov chain sampler.

    NOTE: Much of the functionality of this method has been moved into
    `HitAndRunPolytopeSampler`. If you want to repeatedly draw samples, you should
    use `HitAndRunPolytopeSampler` directly in order to avoid repeatedly running
    a burn-in of the chain. To do so, you need to convert the sparse constraint
    format that `get_polytope_samples` expects to the dense constraint format that
    `HitAndRunPolytopeSampler` expects. This can be done via the
    `sparse_to_dense_constraints` method (but remember to adjust the constraint
    from the `Ax >= b` format expecxted here to the `Ax <= b` format expected by
    `PolytopeSampler` by multiplying both `A` and `b` by -1.)

    NOTE: This method does not support the kind of "inter-point constraints" that
    are supported by `optimize_acqf()`. To achieve this behavior, you need define the
    problem on the joint space over `q` points and impose use constraints, see:
    https://github.com/meta-pytorch/botorch/issues/2468#issuecomment-2287706461

    Args:
        n: The number of samples.
        bounds: A `2 x d`-dim tensor containing the box bounds.
        inequality_constraints: A list of tuples (`indices`, `coefficients`, `rhs`),
            with `indices` and `coefficients` one-dimensional tensors and `rhs` a
            scalar, where each tuple encodes an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality_constraints: A list of tuples (`indices`, `coefficients`, `rhs`),
            with `indices` and `coefficients` one-dimensional tensors and `rhs` a
            scalar, where each tuple encodes an equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        seed: The random seed.
        n_burnin: The number of burn-in samples for the Markov chain sampler.
        n_thinning: The amount of thinnning. This function will return every
            `n_thinning`-th sample from the chain (after burn-in).

    Returns:
        A `n x d`-dim tensor of samples.
    """
    if inequality_constraints:
        A, b = sparse_to_dense_constraints(
            d=bounds.shape[-1],
            constraints=inequality_constraints,
        )
        # Note that the inequality constraints are of the form Ax >= b,
        # but PolytopeSampler expects inequality constraints of the
        # form Ax <= b, so we multiply by -1 below.
        dense_inequality_constraints = (-A, -b)
    else:
        dense_inequality_constraints = None
    if equality_constraints:
        dense_equality_constraints = sparse_to_dense_constraints(
            d=bounds.shape[-1], constraints=equality_constraints
        )
    else:
        dense_equality_constraints = None
    polytope_sampler = HitAndRunPolytopeSampler(
        bounds=bounds,
        inequality_constraints=dense_inequality_constraints,
        equality_constraints=dense_equality_constraints,
        n_burnin=n_burnin,
        n_thinning=n_thinning,
        seed=seed,
    )
    return polytope_sampler.draw(n=n)


def sparse_to_dense_constraints(
    d: int,
    constraints: list[tuple[Tensor, Tensor, float]],
) -> tuple[Tensor, Tensor]:
    r"""Convert parameter constraints from a sparse format into a dense format.

    This method converts sparse triples of the form (indices, coefficients, rhs)
    to constraints of the form Ax >= b or Ax = b.

    Args:
        d: The input dimension.
        constraints: A list of tuples (`indices`, `coefficients`, `rhs`),
            with `indices` and `coefficients` one-dimensional tensors and `rhs` a
            scalar, where each tuple encodes an (in)equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs` or
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A two-element tuple containing:
            - A: A `n_constraints x d`-dim tensor of coefficients.
            - b: A `n_constraints x 1`-dim tensor of right hand sides.
    """
    _t = constraints[0][1]
    A = torch.zeros(len(constraints), d, dtype=_t.dtype, device=_t.device)
    b = torch.zeros(len(constraints), 1, dtype=_t.dtype, device=_t.device)
    for i, (indices, coefficients, rhs) in enumerate(constraints):
        A[i, indices.long()] = coefficients
        b[i] = rhs
    return A, b


# This is only used in get_optimal_samples, which in turn is only used in the input
# constructors of
# 1) qJointEntropySearch,
# 2) qSelfCorrectingBayesianOptimization, and
# 3) qTestSetInformationGain.
def optimize_posterior_samples(
    paths: GenericDeterministicModel,
    bounds: Tensor,
    raw_samples: int = 2048,
    num_restarts: int = 4,
    sample_transform: Callable[[Tensor], Tensor] | None = None,
    return_transformed: bool = False,
    suggested_points: Tensor | None = None,
    options: dict | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Cheaply maximizes posterior samples by random querying followed by
    gradient-based optimization using SciPy's L-BFGS-B routine.

    Args:
        paths: Random Fourier Feature-based sample paths from the GP
        bounds: The bounds on the search space.
        raw_samples: The number of samples with which to query the samples initially.
            Raw samples are cheap to evaluate, so this should ideally be set much higher
            than num_restarts.
        num_restarts: The number of points selected for gradient-based optimization.
            Should be set low relative to the number of raw samples for time-efficiency.
        sample_transform: A callable transform of the sample outputs (e.g.
            MCAcquisitionObjective or ScalarizedPosteriorTransform.evaluate) used to
            negate the objective or otherwise transform the output.
        return_transformed: A boolean indicating whether to return the transformed
            or non-transformed samples.
        suggested_points: Tensor of suggested input locations that are high-valued.
            These are more densely evaluated during the sampling phase of optimization.
        options: Options for generation of initial candidates, passed to
            gen_batch_initial_conditions.

    Returns:
        A two-element tuple containing:
            - X_opt: A `num_optima x [batch_size] x d`-dim tensor of optimal inputs x*.
            - f_opt: A `num_optima x [batch_size] x m`-dim, optionally
                `num_optima x [batch_size] x 1`-dim,  tensor of optimal outputs f*.
    """
    options = {} if options is None else options

    def path_func(x) -> Tensor:
        res = paths(x)
        if sample_transform:
            res = sample_transform(res)

        return res.squeeze(-1)

    # queries all samples on all candidates - output shape
    # raw_samples x num_optima x num_models
    frac_random = 1 if suggested_points is None else options.get("frac_random", 0.9)
    candidate_set = draw_sobol_samples(
        bounds=bounds, n=round(raw_samples * frac_random), q=1
    ).squeeze(-2)
    if frac_random < 1:
        perturbed_suggestions = sample_truncated_normal_perturbations(
            X=suggested_points,
            n_discrete_points=round(raw_samples * (1 - frac_random)),
            sigma=options.get("sample_around_best_sigma", 1e-2),
            bounds=bounds,
        )
        candidate_set = torch.cat((candidate_set, perturbed_suggestions))

    candidate_queries = path_func(candidate_set)
    idx = boltzmann_sample(
        function_values=candidate_queries.unsqueeze(-1),
        num_samples=num_restarts,
        eta=options.get("eta", 2.0),
        replacement=False,
    )
    ics = candidate_set[idx, :]

    # to avoid circular import, the import occurs here
    from botorch.generation.gen import gen_candidates_scipy

    X_top_k, f_top_k = gen_candidates_scipy(
        ics,
        path_func,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        use_parallel_mode=False,
    )
    f_opt, arg_opt = f_top_k.max(dim=-1, keepdim=True)

    # For each sample (and possibly for every model in the batch of models), this
    # retrieves the argmax. We flatten, pick out the indices and then reshape to
    # the original batch shapes (so instead of pickig out the argmax of a
    # (num_optima, num_models, num_restarts, D)-shaped Tensor along the num_restarts
    # dim, we pick it out of (num_optima * num_models, num_restarts, D)
    final_shape = candidate_queries.shape[:-1]
    X_opt = X_top_k.reshape(final_shape.numel(), num_restarts, -1)[
        torch.arange(final_shape.numel()), arg_opt.flatten()
    ].reshape(*final_shape, -1)

    # if we return transformed, we do not need to pass the samples through paths
    # paths a second time but rather just return the transformed optimal values
    if return_transformed:
        return X_opt, f_opt

    f_opt = paths(X_opt.unsqueeze(-2)).squeeze(-2)
    return X_opt, f_opt


def boltzmann_sample(
    function_values: Tensor,
    num_samples: int,
    eta: float,
    replacement: bool = False,
    temp_decrease: float = 0.5,
):
    """
    Perform Boltzmann sampling from a set of function values, weighted by the
    exponentiated difference between function values and their standardized mean.

    Args:
        function_values: A [batch_shape] x N  tensor of function values.
        num_samples: The number of samples (restarts) to draw.
        eta: The Boltzmann temperature, controls the sharpness of the weighting. If the
            temperature is too high, causing NaN values, the eta parameter is
            succesively decreased by 'temp_decrease'.
        replacement: If True, samples are drawn with replacement, allowing duplicates.
        temp_decrease: The rate at which temperature decreases in case of inf weights.

        Returns:
        A [batch_shape] x num_samples tensor of indices of sampled positions.
    """
    norm_weights = standardize(function_values)
    weights = torch.exp(eta * norm_weights)
    while torch.isinf(weights).any():
        eta *= temp_decrease
        weights = torch.exp(eta * norm_weights)

    # squeeze in case of m = 1 (mono-output provided as batch_size x N x 1)
    return batched_multinomial(
        weights=weights.squeeze(-1), num_samples=num_samples, replacement=replacement
    )


def sample_truncated_normal_perturbations(
    X: Tensor,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
    qmc: bool = True,
) -> Tensor:
    r"""Sample points around `X`.

    Sample perturbed points around `X` such that the added perturbations
    are sampled from N(0, sigma^2 I) and truncated to be within [0,1]^d.

    Args:
        X: A `n x d`-dim tensor starting points.
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the points.
        bounds: A `2 x d`-dim tensor containing the bounds.
        qmc: A boolean indicating whether to use qmc.

    Returns:
        A `n_discrete_points x d`-dim tensor containing the sampled points.
    """
    X = normalize(X, bounds=bounds)
    d = X.shape[1]
    # sample points from N(X_center, sigma^2 I), truncated to be within
    # [0, 1]^d.
    if X.shape[0] > 1:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X = X[rand_indices]
    if qmc:
        std_bounds = torch.zeros(2, d, dtype=X.dtype, device=X.device)
        std_bounds[1] = 1
        u = draw_sobol_samples(bounds=std_bounds, n=n_discrete_points, q=1).squeeze(1)
    else:
        u = torch.rand((n_discrete_points, d), dtype=X.dtype, device=X.device)
    # compute bounds to sample from
    a = -X
    b = 1 - X
    # compute z-score of bounds
    alpha = a / sigma
    beta = b / sigma
    normal = Normal(0, 1)
    cdf_alpha = normal.cdf(alpha)
    # use inverse transform
    perturbation = normal.icdf(cdf_alpha + u * (normal.cdf(beta) - cdf_alpha)) * sigma
    # add perturbation and clip points that are still outside
    perturbed_X = (X + perturbation).clamp(0.0, 1.0)
    return unnormalize(perturbed_X, bounds=bounds)


def sample_perturbed_subset_dims(
    X: Tensor,
    bounds: Tensor,
    n_discrete_points: int,
    sigma: float = 1e-1,
    qmc: bool = True,
    prob_perturb: float | None = None,
) -> Tensor:
    r"""Sample around `X` by perturbing a subset of the dimensions.

    By default, dimensions are perturbed with probability equal to
    `min(20 / d, 1)`. As shown in [Regis]_, perturbing a small number
    of dimensions can be beneificial. The perturbations are sampled
    from N(0, sigma^2 I) and truncated to be within [0,1]^d.

    Args:
        X: A `n x d`-dim tensor starting points. `X`
            must be normalized to be within `[0, 1]^d`.
        bounds: The bounds to sample perturbed values from
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the points.
        qmc: A boolean indicating whether to use qmc.
        prob_perturb: The probability of perturbing each dimension. If omitted,
            defaults to `min(20 / d, 1)`.

    Returns:
        A `n_discrete_points x d`-dim tensor containing the sampled points.

    """
    if bounds.ndim != 2:
        raise BotorchTensorDimensionError("bounds must be a `2 x d`-dim tensor.")
    elif X.ndim != 2:
        raise BotorchTensorDimensionError("X must be a `n x d`-dim tensor.")
    d = bounds.shape[-1]
    if prob_perturb is None:
        # Only perturb a subset of the features
        prob_perturb = min(20.0 / d, 1.0)

    if X.shape[0] == 1:
        X_cand = X.repeat(n_discrete_points, 1)
    else:
        rand_indices = torch.randint(X.shape[0], (n_discrete_points,), device=X.device)
        X_cand = X[rand_indices]
    pert = sample_truncated_normal_perturbations(
        X=X_cand,
        n_discrete_points=n_discrete_points,
        sigma=sigma,
        bounds=bounds,
        qmc=qmc,
    )

    # find cases where we are not perturbing any dimensions
    mask = (
        torch.rand(
            n_discrete_points,
            d,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        <= prob_perturb
    )
    ind = (~mask).all(dim=-1).nonzero()
    # perturb `n_perturb` of the dimensions
    n_perturb = ceil(d * prob_perturb)
    perturb_mask = torch.zeros(d, dtype=mask.dtype, device=mask.device)
    perturb_mask[:n_perturb].fill_(1)
    # TODO: use batched `torch.randperm` when available:
    # https://github.com/pytorch/pytorch/issues/42502
    for idx in ind:
        mask[idx] = perturb_mask[torch.randperm(d, device=bounds.device)]
    # Create candidate points
    X_cand[mask] = pert[mask]
    return X_cand
