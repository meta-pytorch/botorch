#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import itertools
import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.initializers import initialize_q_batch
from botorch.optim.optimize import (
    _optimize_acqf,
    _validate_sequential_inputs,
    OptimizeAcqfInputs,
)
from botorch.optim.parameter_constraints import evaluate_feasibility
from botorch.optim.utils.acquisition_utils import fix_features, get_X_baseline
from botorch.utils.sampling import (
    draw_sobol_samples,
    HitAndRunPolytopeSampler,
    sparse_to_dense_constraints,
)
from botorch.utils.transforms import unnormalize
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor
from torch.quasirandom import SobolEngine

# Default values.
# NOTE: When changing a default, update the corresponding value in the docstrings.
STD_CONT_PERTURBATION = 0.1
RAW_SAMPLES = 1024  # Number of candidates from which to select starting points.
NUM_RESTARTS = 20  # Number of restarts of optimizer with different starting points.
MAX_BATCH_SIZE = 2048  # Maximum batch size.
MAX_ITER_ALTER = 64  # Maximum number of alternating iterations.
MAX_ITER_DISCRETE = 4  # Maximum number of discrete iterations.
MAX_ITER_CONT = 8  # Maximum number of continuous iterations.
# Maximum number of discrete values for a discrete dimension.
# If there are more values for a dimension, we will use continuous
# relaxation to optimize it.
MAX_DISCRETE_VALUES = 20
# Maximum number of iterations for optimizing the continuous relaxation
# during initialization
MAX_ITER_INIT = 100
CONVERGENCE_TOL = 1e-8  # Optimizer convergence tolerance.
DUPLICATE_TOL = 1e-6  # Tolerance for deduplicating initial candidates.
STOP_AFTER_SHARE_CONVERGED = 1.0  # We optimize multiple configurations at once
# in ``optimize_acqf_mixed_alternating``. This option controls, whether to stop
# optimizing after the given share has converged.
# Convergence is defined as the improvements of one discrete, followed by a scalar
# optimization yield less than ``CONVERGENCE_TOL`` improvements.

SUPPORTED_OPTIONS = {
    "initialization_strategy",
    "tol",
    "maxiter_alternating",
    "maxiter_discrete",
    "maxiter_continuous",
    "maxiter_init",
    "max_discrete_values",
    "num_spray_points",
    "std_cont_perturbation",
    "batch_limit",
    "init_batch_limit",
}
SUPPORTED_INITIALIZATION = {"continuous_relaxation", "equally_spaced", "random"}


def _setup_continuous_relaxation(
    discrete_dims: dict[int, list[float]],
    max_discrete_values: int,
    post_processing_func: Callable[[Tensor], Tensor] | None,
) -> tuple[list[int], Callable[[Tensor], Tensor] | None]:
    r"""Update ``discrete_dims`` and ``post_processing_func`` to use
    continuous relaxation for discrete dimensions that have more than
    ``max_discrete_values`` values. These dimensions are removed from
    ``discrete_dims`` and ``post_processing_func`` is updated to round
    them to the nearest integer.
    """

    dims_to_relax, dims_to_keep = {}, {}
    for index, values in discrete_dims.items():
        if len(values) > max_discrete_values:
            dims_to_relax[index] = values
        else:
            dims_to_keep[index] = values

    if len(dims_to_relax) == 0:
        return discrete_dims, post_processing_func

    def new_post_processing_func(X: Tensor) -> Tensor:
        r"""Round the relaxed dimensions to the nearest integer and apply the original
        ``post_processing_func``."""

        X = round_discrete_dims(X=X, discrete_dims=dims_to_relax)
        if post_processing_func is not None:
            X = post_processing_func(X)
        return X

    return dims_to_keep, new_post_processing_func


def _filter_infeasible(
    X: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None,
) -> Tensor:
    r"""Filters infeasible points from a set of points.

    NOTE: This function only supports intra-point constraints. This is validated
        in ``optimize_acqf_mixed_alternating``, so we do not repeat the
        validation in here.

    Args:
        X: A tensor of points of shape ``n x d``.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``. ``indices`` and
            ``coefficients`` should be torch tensors. See the docstring of
            ``make_scipy_linear_constraints`` for an example.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) == rhs``. ``indices`` and
            ``coefficients`` should be torch tensors. Example:
            ``[(torch.tensor([1, 3]), torch.tensor([1.0, 0.5]), -0.1)]``

    Returns:
        The tensor ``X`` with infeasible points removed.
    """
    # X is reshaped to [n, 1, d] in order to be able to apply
    # ``evaluate_feasibility`` which operates on the batch level
    is_feasible = evaluate_feasibility(
        X=X.unsqueeze(-2),
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=None,
    )
    return X[is_feasible]


def get_nearest_neighbors(
    current_x: Tensor,
    bounds: Tensor,
    discrete_dims: dict[int, list[float]],
) -> Tensor:
    r"""Generate all 1-Manhattan distance neighbors of a given input. The neighbors
    are generated for the discrete dimensions only.

    NOTE: This assumes that ``current_x`` is detached and uses in-place operations,
    which are known to be incompatible with autograd.

    Args:
        current_x: The design to find the neighbors of. A tensor of shape ``d``.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column of ``X``.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.

    Returns:
        A tensor of shape ``num_neighbors x d``, denoting all unique 1-Manhattan
        distance neighbors.
    """
    # we first transform the discrete values in the current_x to its index in the
    # values of the discrete_dims, so that we can easily increase/decrease
    # the discrete dimensions by one.
    # Map discrete values in current_x to their indices in discrete_dims
    discrete_indices = []

    t_discrete_dims = torch.tensor(
        list(discrete_dims.keys()), dtype=torch.long, device=current_x.device
    )

    for dim, values in discrete_dims.items():
        val = current_x[dim].item()
        # Find the index of the closest value in values
        idx = min(range(len(values)), key=lambda i: abs(values[i] - val))
        discrete_indices.append(idx)
    current_x_int = current_x.clone()
    current_x_int[t_discrete_dims] = torch.tensor(
        discrete_indices, dtype=current_x.dtype, device=current_x.device
    )
    lower_clamp = bounds[0].clone()
    lower_clamp[t_discrete_dims] = torch.tensor(
        [0 for _ in discrete_dims.values()],
        dtype=lower_clamp.dtype,
        device=lower_clamp.device,
    )
    upper_clamp = bounds[1].clone()
    upper_clamp[t_discrete_dims] = torch.tensor(
        [len(values) - 1 for values in discrete_dims.values()],
        dtype=upper_clamp.dtype,
        device=upper_clamp.device,
    )
    num_discrete = len(discrete_dims)
    diag_ones = torch.eye(num_discrete, dtype=current_x.dtype, device=current_x.device)
    # Neighbors obtained by increasing a discrete dimension by one.
    plus_neighbors = current_x_int.repeat(num_discrete, 1)
    plus_neighbors[:, t_discrete_dims] += diag_ones
    plus_neighbors.clamp_(max=upper_clamp)
    # Neighbors obtained by decreasing a discrete dimension by one.
    minus_neighbors = current_x_int.repeat(num_discrete, 1)
    minus_neighbors[:, t_discrete_dims] -= diag_ones
    minus_neighbors.clamp_(min=lower_clamp)
    unique_neighbors = torch.cat([minus_neighbors, plus_neighbors], dim=0).unique(dim=0)
    # Also remove current_x if it is in unique_neighbors.
    unique_neighbors = unique_neighbors[
        ~(unique_neighbors == current_x_int).all(dim=-1)
    ]

    # Transform unique_neighbors back to the original non-integer space
    for dim, values in discrete_dims.items():
        t_values = torch.tensor(
            values, device=unique_neighbors.device, dtype=unique_neighbors.dtype
        )
        idx = unique_neighbors[:, dim].long()
        unique_neighbors[:, dim] = t_values[idx]

    return unique_neighbors


def get_categorical_neighbors(
    current_x: Tensor,
    cat_dims: dict[int, list[float]],
    max_num_cat_values: int = MAX_DISCRETE_VALUES,
) -> Tensor:
    r"""Generate all 1-Hamming distance neighbors of a given input. The neighbors
    are generated for the categorical dimensions only.

    We assume that all categorical values are equidistant. If the number of values
    is greater than ``max_num_cat_values``, we sample uniformly from the
    possible values for that dimension.

    NOTE: This assumes that ``current_x`` is detached and uses in-place operations,
    which are known to be incompatible with autograd.

    Args:
        current_x: The design to find the neighbors of. A tensor of shape ``d``.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        max_num_cat_values: Maximum number of values for a categorical parameter,
            beyond which values are uniformly sampled.

    Returns:
        A tensor of shape ``num_neighbors x d``, denoting up to ``max_num_cat_values``
        unique 1-Hamming distance neighbors for each categorical dimension.
    """

    # Neighbors are generated by considering all possible values for each
    # categorical dimension, one at a time.
    def _get_cat_values(dim: int) -> Sequence[int]:
        r"""Get a sequence of up to ``max_num_cat_values`` values that a categorical
        feature may take."""
        current_value = current_x[dim]
        if len(cat_dims[dim]) <= max_num_cat_values:
            return cat_dims[dim]
        else:
            return random.sample(
                [v for v in cat_dims[dim] if v != current_value], k=max_num_cat_values
            )

    new_cat_values_dict = {dim: _get_cat_values(dim) for dim in cat_dims.keys()}
    new_cat_values_lst = list(
        itertools.chain.from_iterable(new_cat_values_dict.values())
    )
    new_cat_values = torch.tensor(
        new_cat_values_lst, device=current_x.device, dtype=current_x.dtype
    )

    new_cat_idcs = torch.cat(
        tuple(
            torch.full(
                (min(len(values), max_num_cat_values),), dim, device=current_x.device
            )
            for dim, values in new_cat_values_dict.items()
        )
    )

    neighbors = current_x.repeat(len(new_cat_values), 1)
    # Assign the new values to their corresponding columns.
    neighbors.scatter_(1, new_cat_idcs.view(-1, 1), new_cat_values.view(-1, 1))

    unique_neighbors = neighbors.unique(dim=0)
    # Also remove current_x if it is in unique_neighbors.
    unique_neighbors = unique_neighbors[~(unique_neighbors == current_x).all(dim=-1)]
    return unique_neighbors


def get_spray_points(
    X_baseline: Tensor,
    cont_dims: Tensor,
    discrete_dims: dict[int, list[float]],
    cat_dims: dict[int, list[float]],
    bounds: Tensor,
    num_spray_points: int,
    std_cont_perturbation: float = STD_CONT_PERTURBATION,
) -> Tensor:
    r"""Generate spray points by perturbing the Pareto optimal points.

    Given the points on the Pareto frontier, we create perturbations (spray points)
    by adding Gaussian perturbation to the continuous parameters and 1-Manhattan
    distance neighbors of the discrete (binary and integer) parameters.

    Args:
        X_baseline: Tensor of best acquired points across BO run.
        cont_dims: Indices of continuous parameters/input dimensions.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column of ``X``.
        num_spray_points: Number of spray points to return.
        std_cont_perturbation: standard deviation of Normal perturbations of
            continuous dimensions. Default is STD_CONT_PERTURBATION = 0.2.

    Returns:
        A (num_spray_points x d)-dim tensor of perturbed points.
    """
    dim = bounds.shape[-1]
    device, dtype = X_baseline.device, X_baseline.dtype
    perturb_nbors = torch.zeros(0, dim, device=device, dtype=dtype)
    t_discrete_dims = torch.tensor(
        list(discrete_dims.keys()), dtype=torch.long, device=device
    )
    t_cat_dims = torch.tensor(list(cat_dims.keys()), dtype=torch.long, device=device)
    for x in X_baseline:
        if len(discrete_dims) > 0:
            discrete_perturbs = get_nearest_neighbors(
                current_x=x, bounds=bounds, discrete_dims=discrete_dims
            )
            discrete_perturbs = discrete_perturbs[
                torch.randint(
                    len(discrete_perturbs), (num_spray_points,), device=device
                )
            ]
        if len(cat_dims) > 0:
            cat_perturbs = get_categorical_neighbors(current_x=x, cat_dims=cat_dims)
            cat_perturbs = cat_perturbs[
                torch.randint(len(cat_perturbs), (num_spray_points,), device=device)
            ]

        cont_perturbs = x[cont_dims] + std_cont_perturbation * torch.randn(
            num_spray_points, len(cont_dims), device=device, dtype=dtype
        )
        cont_perturbs = cont_perturbs.clamp_(
            min=bounds[0, cont_dims], max=bounds[1, cont_dims]
        )
        nbds = torch.zeros(num_spray_points, dim, device=device, dtype=dtype)
        if len(discrete_dims) > 0:
            nbds[..., t_discrete_dims] = discrete_perturbs[..., t_discrete_dims]
        if len(cat_dims) > 0:
            nbds[..., t_cat_dims] = cat_perturbs[..., t_cat_dims]

        nbds[..., cont_dims] = cont_perturbs
        perturb_nbors = torch.cat([perturb_nbors, nbds], dim=0)
    return perturb_nbors


def sample_feasible_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: dict[int, list[float]],
    cat_dims: dict[int, list[float]],
    num_points: int,
) -> Tensor:
    r"""Sample feasible points from the optimization domain.

    Feasibility is determined according to the discrete dimensions taking
    integer values and the inequality constraints being satisfied.

    If there are no inequality constraints, Sobol is used to generate the base points.
    Otherwise, we use the polytope sampler to generate the base points. The base points
    are then rounded to the nearest integer values for the discrete dimensions, and
    the infeasible points are filtered out (in case rounding leads to infeasibility).

    This method will do 10 attempts to generate ``num_points`` feasible points, and
    return the points generated so far. If no points are generated, it will error out.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        num_points: The number of points to sample.

    Returns:
        A tensor of shape ``num_points x d`` containing the sampled points.
    """
    bounds = opt_inputs.bounds
    all_points = torch.empty(
        0, bounds.shape[-1], device=bounds.device, dtype=bounds.dtype
    )
    if (
        opt_inputs.inequality_constraints is None
        and opt_inputs.equality_constraints is None
    ):
        # Generate base points using Sobol.
        sobol_engine = SobolEngine(dimension=bounds.shape[-1], scramble=True)

        def generator(n: int) -> Tensor:
            samples = sobol_engine.draw(n=n, dtype=bounds.dtype).to(bounds.device)
            return unnormalize(X=samples, bounds=bounds)

    else:
        # Generate base points using polytope sampler.
        # Since we may generate many times, we initialize the sampler with burn-in
        # to reduce the start-up cost for subsequent calls.
        if opt_inputs.inequality_constraints is not None:
            A, b = sparse_to_dense_constraints(
                d=bounds.shape[-1], constraints=opt_inputs.inequality_constraints
            )
            ineqs = (-A, -b)
        else:
            ineqs = None
        if opt_inputs.equality_constraints is not None:
            A_eq, b_eq = sparse_to_dense_constraints(
                d=bounds.shape[-1], constraints=opt_inputs.equality_constraints
            )
            eqs = (A_eq, b_eq)
        else:
            eqs = None
        polytope_sampler = HitAndRunPolytopeSampler(
            bounds=bounds, inequality_constraints=ineqs, equality_constraints=eqs
        )

        def generator(n: int) -> Tensor:
            return polytope_sampler.draw(n=n)

    for _ in range(10):
        num_remaining = num_points - len(all_points)
        if num_remaining <= 0:
            break
        # Generate twice as many, since we're likely to filter out some points.
        base_points = generator(n=num_remaining * 2)
        # Round the discrete dimensions to the nearest integer.
        base_points = round_discrete_dims(X=base_points, discrete_dims=discrete_dims)
        base_points = round_discrete_dims(X=base_points, discrete_dims=cat_dims)
        # Fix the fixed features.
        base_points = fix_features(
            X=base_points,
            fixed_features=opt_inputs.fixed_features,
            replace_current_value=True,
        )
        # Filter out infeasible points.
        feasible_points = _filter_infeasible(
            X=base_points,
            inequality_constraints=opt_inputs.inequality_constraints,
            equality_constraints=opt_inputs.equality_constraints,
        )
        all_points = torch.cat([all_points, feasible_points], dim=0)

    if len(all_points) == 0:
        raise CandidateGenerationError(
            "Could not generate any feasible starting points for mixed optimizer."
        )
    return all_points[:num_points]


def round_discrete_dims(X: Tensor, discrete_dims: dict[int, list[float]]) -> Tensor:
    """Round the discrete dimensions of a tensor to the nearest allowed values.

    Args:
        X: A tensor of shape ``n x d``, where ``d`` is the number of dimensions.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.

    Returns:
        A tensor of the same shape as ``X``, with discrete dimensions rounded to
        the nearest allowed values.
    """
    X = X.clone()
    for dim, values in discrete_dims.items():
        allowed = torch.tensor(values, device=X.device, dtype=X.dtype)
        diffs = torch.abs(X[..., dim].unsqueeze(-1) - allowed)
        idx = torch.argmin(diffs, dim=-1)
        X[..., dim] = allowed[idx]
    return X


def generate_starting_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: dict[int, list[float]],
    cat_dims: dict[int, list[float]],
    cont_dims: Tensor,
) -> tuple[Tensor, Tensor]:
    """Generate initial starting points for the alternating optimization.

    This method attempts to generate the initial points using the specified
    options and completes any missing points using ``sample_feasible_points``.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes ``acq_function``, ``bounds``, ``num_restarts``,
            ``raw_samples``, ``options``, ``fixed_features`` and constraints
            from ``opt_inputs``.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        cont_dims: A tensor of indices corresponding to continuous parameters.

    Returns:
        A tuple of two tensors: a (num_restarts x d)-dim tensor of starting points
        and a (num_restarts)-dim tensor of their respective acquisition values.
        In rare cases, this method may return fewer than ``num_restarts`` points.
    """
    bounds = opt_inputs.bounds
    binary_dims = []
    for dim, values in discrete_dims.items():
        if len(values) == 2:
            binary_dims.append(dim)
    num_binary = len(binary_dims)
    num_discrete = len(discrete_dims) - num_binary
    num_restarts = opt_inputs.num_restarts
    raw_samples = none_throws(opt_inputs.raw_samples)

    options = opt_inputs.options or {}
    initialization_strategy = options.get(
        "initialization_strategy",
        (
            "equally_spaced"
            if num_discrete == 0 and num_binary >= 2
            else "continuous_relaxation"
        ),
    )
    if initialization_strategy not in SUPPORTED_INITIALIZATION:
        raise UnsupportedError(  # pragma: no cover
            f"Unsupported initialization strategy: {initialization_strategy}."
            f"Supported strategies are: {SUPPORTED_INITIALIZATION}."
        )

    # Initialize ``x_init_candts`` here so that it's always defined as a tensor.
    x_init_candts = torch.empty(
        0, bounds.shape[-1], device=bounds.device, dtype=bounds.dtype
    )
    if initialization_strategy == "continuous_relaxation":
        try:
            # Optimize the acquisition function with continuous relaxation.
            updated_opt_inputs = dataclasses.replace(
                opt_inputs,
                q=1,
                return_best_only=False,
                options={
                    "maxiter": options.get("maxiter_init", MAX_ITER_INIT),
                    "batch_limit": options.get("batch_limit", MAX_BATCH_SIZE),
                    "init_batch_limit": options.get("init_batch_limit", MAX_BATCH_SIZE),
                },
                post_processing_func=None,
            )
            x_init_candts, _ = _optimize_acqf(opt_inputs=updated_opt_inputs)
            x_init_candts = x_init_candts.squeeze(-2).detach()
        except Exception as e:
            warnings.warn(
                "Failed to initialize using continuous relaxation. Using "
                "`sample_feasible_points` for initialization. Original error "
                f"message: {e}",
                OptimizationWarning,
                stacklevel=2,
            )

    if len(x_init_candts) == 0:
        # Generate Sobol points as a fallback for ``continuous_relaxation`` and for
        # further refinement in ``equally_spaced`` strategy.
        x_init_candts = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1)
        x_init_candts = x_init_candts.squeeze(-2)

    if initialization_strategy == "equally_spaced":
        if num_discrete > 0:
            raise ValueError(  # pragma: no cover
                "Equally spaced initialization is not supported with non-binary "
                "discrete variables."
            )
        # Picking initial points by equally spaced number of features/binary inputs.
        k = torch.randint(
            low=0,
            high=num_binary,
            size=(raw_samples,),
            dtype=torch.int64,
            device=bounds.device,
        )
        x_init_candts[:, binary_dims] = bounds[0, binary_dims]
        binary_dims_t = torch.as_tensor(binary_dims, device=bounds.device)
        for i, xi in enumerate(x_init_candts):
            rand_binary_dims = binary_dims_t[
                torch.randperm(num_binary, device=xi.device)[: k[i]]
            ]
            x_init_candts[i, rand_binary_dims] = bounds[1, rand_binary_dims]

    num_spray_points = assert_is_instance(
        options.get("num_spray_points", 20 if num_discrete == 0 else 0), int
    )
    if (
        num_spray_points > 0
        and (X_baseline := get_X_baseline(acq_function=opt_inputs.acq_function))
        is not None
    ):
        perturb_nbors = get_spray_points(
            X_baseline=X_baseline,
            cont_dims=cont_dims,
            discrete_dims=discrete_dims,
            cat_dims=cat_dims,
            bounds=bounds,
            num_spray_points=num_spray_points,
            std_cont_perturbation=assert_is_instance(
                options.get("std_cont_perturbation", STD_CONT_PERTURBATION), float
            ),
        )
        x_init_candts = torch.cat([x_init_candts, perturb_nbors], dim=0)

    # For each discrete dimension, map to the nearest allowed value
    x_init_candts = round_discrete_dims(X=x_init_candts, discrete_dims=discrete_dims)
    x_init_candts = round_discrete_dims(X=x_init_candts, discrete_dims=cat_dims)
    x_init_candts = fix_features(
        X=x_init_candts,
        fixed_features=opt_inputs.fixed_features,
        replace_current_value=True,
    )
    x_init_candts = _filter_infeasible(
        X=x_init_candts,
        inequality_constraints=opt_inputs.inequality_constraints,
        equality_constraints=opt_inputs.equality_constraints,
    )

    # If there are fewer than ``num_restarts`` feasible points, attempt to
    # generate more.
    if len(x_init_candts) < num_restarts:
        new_x_init = sample_feasible_points(
            opt_inputs=opt_inputs,
            discrete_dims=discrete_dims,
            cat_dims=cat_dims,
            num_points=num_restarts - len(x_init_candts),
        )
        x_init_candts = torch.cat([x_init_candts, new_x_init], dim=0)

    with torch.no_grad():
        acq_vals = torch.cat(
            [
                opt_inputs.acq_function(X_.unsqueeze(-2))
                for X_ in x_init_candts.split(
                    options.get("init_batch_limit", MAX_BATCH_SIZE)
                )
            ]
        )
    if len(x_init_candts) > num_restarts:
        # If there are more than ``num_restarts`` feasible points, select a diverse
        # set of initializers using Boltzmann sampling.
        x_init_candts, acq_vals = initialize_q_batch(
            X=x_init_candts, acq_vals=acq_vals, n=num_restarts
        )

    return x_init_candts, acq_vals


def discrete_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: dict[int, list[float]],
    cat_dims: dict[int, list[float]],
    current_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Discrete nearest neighbour search.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes ``acq_function``, ``bounds``, ``options``
            and constraints from ``opt_inputs``.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        current_x: Batch of starting points. A tensor of shape ``b x d``.

    Returns:
        A tuple of two tensors: a (b, d)-dim tensor of optimized point
            and a scalar tensor of correspondins acquisition value.
    """
    with torch.no_grad():
        current_acqvals = opt_inputs.acq_function(current_x.unsqueeze(1))
    options = opt_inputs.options or {}
    maxiter_discrete = options.get("maxiter_discrete", MAX_ITER_DISCRETE)
    done = torch.zeros(len(current_x), dtype=torch.bool)
    for _ in range(assert_is_instance(maxiter_discrete, int)):
        # we don't batch this, as the number of x_neighbors can be different
        # for each entry (as duplicates are removed), and the most expensive
        # op is the acq_function, which is batched
        # TODO finding the set of neighbors currently is done sequentially
        # for one item in the batch after the other
        x_neighbors_list = [None for _ in done]
        for i in range(len(done)):
            # don't do anything if we are already done
            if done[i]:
                continue

            neighbors = []

            # if we have discrete_dims look for neighbors by stepping +1/-1
            if len(discrete_dims) > 0:
                x_neighbors_discrete = get_nearest_neighbors(
                    current_x=current_x[i].detach(),
                    bounds=opt_inputs.bounds,
                    discrete_dims=discrete_dims,
                )
                x_neighbors_discrete = _filter_infeasible(
                    X=x_neighbors_discrete,
                    inequality_constraints=opt_inputs.inequality_constraints,
                    equality_constraints=opt_inputs.equality_constraints,
                )
                neighbors.append(x_neighbors_discrete)

            # if we have cat_dims look for neighbors by changing the cat's
            if len(cat_dims) > 0:
                x_neighbors_cat = get_categorical_neighbors(
                    current_x=current_x[i].detach(),
                    cat_dims=cat_dims,
                )
                x_neighbors_cat = _filter_infeasible(
                    X=x_neighbors_cat,
                    inequality_constraints=opt_inputs.inequality_constraints,
                    equality_constraints=opt_inputs.equality_constraints,
                )
                neighbors.append(x_neighbors_cat)

            x_neighbors = torch.cat(neighbors, dim=0)
            if x_neighbors.numel() == 0:
                # If the i'th point has no neighbors, we mark it as done
                done[i] = True  # pragma: no cover
            x_neighbors_list[i] = x_neighbors

        # Exit if all batches converged or have no valid neighbors left.
        if done.all():
            break

        all_x_neighbors = torch.cat(
            [
                x_neighbors
                for x_neighbors in x_neighbors_list
                if x_neighbors is not None
            ],
            dim=0,
        )  # shape: (sum(#neihbors of the items in the batch), d)
        with torch.no_grad():
            # This is the most expensive call in this function.
            # The reason that ``discrete_step`` uses a batched x
            # rather than looping over each batch within x is so that
            # we can batch this call. This leads to an overall speedup
            # even though the above and below for loops cannot
            # be sped up by batching.
            acq_vals = torch.cat(
                [
                    opt_inputs.acq_function(X_.unsqueeze(-2))
                    for X_ in all_x_neighbors.split(
                        options.get("init_batch_limit", MAX_BATCH_SIZE)
                    )
                ]
            )
        offset = 0
        for i in range(len(done)):
            if done[i]:
                continue

            # We index into all_x_neighbors in the following convoluted way,
            # as it is a flattened version of x_neighbors_list with the
            # None entries removed. That is why we do not increase offset if done[i].
            width = len(x_neighbors_list[i])
            x_neighbors = all_x_neighbors[offset : offset + width]
            max_acq, argmax = acq_vals[offset : offset + width].max(dim=0)
            improvement = max_acq - current_acqvals[i]
            if improvement > 0:
                current_acqvals[i], current_x[i] = (
                    max_acq,
                    x_neighbors[argmax],
                )
            if improvement <= options.get("tol", CONVERGENCE_TOL):
                done[i] = True

            offset += width

    return current_x, current_acqvals


def continuous_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Tensor,
    cat_dims: Tensor,
    current_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Continuous search using L-BFGS-B through optimize_acqf.

    Args:
        opt_inputs: Common set of arguments for acquisition optimization.
            This function utilizes ``acq_function``, ``bounds``, ``options``,
            ``fixed_features`` and constraints from ``opt_inputs``.
            ``opt_inputs.return_best_only`` should be ``False``.
        discrete_dims: A dictionary mapping indices of discrete dimensions
            to a list of allowed values for that dimension.
        cat_dims: A tensor of indices corresponding to categorical parameters.
        current_x: Starting point. A tensor of shape ``b x d``.

    Returns:
        A tuple of two tensors: a (b x d)-dim tensor of optimized points
            and a (b)-dim tensor of acquisition values.
    """

    if opt_inputs.return_best_only:
        raise UnsupportedError(
            "`continuous_step` does not support `return_best_only=True`."
        )

    d = current_x.shape[-1]
    options = opt_inputs.options or {}
    non_cont_dims = torch.cat((discrete_dims, cat_dims), dim=0)

    if len(non_cont_dims) == d:  # nothing continuous to optimize
        with torch.no_grad():
            return current_x, opt_inputs.acq_function(current_x.unsqueeze(1))

    updated_opt_inputs = dataclasses.replace(
        opt_inputs,
        q=1,
        raw_samples=None,
        # unsqueeze to add the q dimension
        batch_initial_conditions=current_x.unsqueeze(1),
        fixed_features={
            **{d: current_x[:, d] for d in non_cont_dims.tolist()},
            **(opt_inputs.fixed_features or {}),
        },
        options={
            "maxiter": options.get("maxiter_continuous", MAX_ITER_CONT),
            "tol": options.get("tol", CONVERGENCE_TOL),
            "batch_limit": options.get("batch_limit", MAX_BATCH_SIZE),
            "max_optimization_problem_aggregation_size": 1,
        },
    )
    best_X, best_acq_values = _optimize_acqf(opt_inputs=updated_opt_inputs)
    return best_X.view_as(current_x), best_acq_values


def optimize_acqf_mixed_alternating(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    discrete_dims: Mapping[int, Sequence[float]] | None = None,
    cat_dims: Mapping[int, Sequence[float]] | None = None,
    options: dict[str, Any] | None = None,
    q: int = 1,
    raw_samples: int = RAW_SAMPLES,
    num_restarts: int = NUM_RESTARTS,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    sequential: bool = True,
    fixed_features: dict[int, float] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""
    Optimizes acquisition function over mixed integer, categorical, and continuous
    input spaces. Multiple random restarting starting points are picked by evaluating
    a large set of initial candidates. From each starting point, alternating
    discrete/categorical local search and continuous optimization via (L-BFGS)
    is performed for a fixed number of iterations.

    The discrete dimensions that have more than
    ``options.get("max_discrete_values", MAX_DISCRETE_VALUES)`` values will
    be optimized using continuous relaxation.
    The categorical dimensions that have more than ``MAX_DISCRETE_VALUES`` values
    will be optimized by selecting random subsamples of the possible values.

    Args:
        acq_function: BoTorch Acquisition function.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column of ``X``.
        discrete_dims: A dictionary mapping indices of discrete and binary
            dimensions to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions
            to a list of allowed values for that dimension.
        options: Dictionary specifying optimization options. Supports the following:
        - "initialization_strategy": Strategy used to generate the initial candidates.
            "random", "continuous_relaxation" or "equally_spaced" (linspace style).
        - "tol": The algorithm terminates if the absolute improvement in acquisition
            value of one iteration is smaller than this number.
        - "maxiter_alternating": Number of alternating steps. Defaults to 64.
        - "maxiter_discrete": Maximum number of iterations in each discrete step.
            Defaults to 4.
        - "maxiter_continuous": Maximum number of iterations in each continuous step.
            Defaults to 8.
        - "max_discrete_values": Maximum number of values for a discrete dimension
            to be optimized using discrete step / local search. The discrete dimensions
            with more values will be optimized using continuous relaxation.
        - "num_spray_points": Number of spray points (around ``X_baseline``) to add to
            the points generated by the initialization strategy. Defaults to 20 if
            all discrete variables are binary and to 0 otherwise.
        - "std_cont_perturbation": Standard deviation of the normal perturbations of
            the continuous variables used to generate the spray points.
            Defaults to 0.1.
        - "batch_limit": The maximum batch size for jointly evaluating candidates
            during optimization.
        - "init_batch_limit": The maximum batch size for jointly evaluating candidates
            during initialization. During initialization, candidates are evaluated
            in a ``no_grad`` context, which reduces memory usage. As a result,
            ``init_batch_limit`` can be set to a larger value than ``batch_limit``.
            Defaults to ``batch_limit``, if given.
        q: Number of candidates.
        raw_samples: Number of initial candidates used to select starting points from.
            Defaults to 1024.
        num_restarts: Number of random restarts. Defaults to 20.
        post_processing_func: A function that post-processes an optimization result
            appropriately (i.e., according to ``round-trip`` transformations).
        sequential: Whether to use joint or sequential optimization across q-batch.
            This currently only supports sequential optimization.
        fixed_features: A map ``{feature_index: value}`` for features that
            should be fixed to a particular value during generation.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``. ``indices`` and
            ``coefficients`` should be torch tensors. See the docstring of
            ``make_scipy_linear_constraints`` for an example.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) == rhs``. ``indices`` and
            ``coefficients`` should be torch tensors. Example:
            ``[(torch.tensor([1, 3]), torch.tensor([1.0, 0.5]), -0.1)]`` Equality
            constraints can only be used with continuous degrees of freedom.

    Returns:
        A tuple of two tensors: a (q x d)-dim tensor of optimized points
            and a (q)-dim tensor of their respective acquisition values.
    """

    if sequential is False:  # pragma: no cover
        raise NotImplementedError(
            "`optimize_acqf_mixed_alternating` only supports sequential optimization."
        )

    cat_dims = cat_dims or {}
    discrete_dims = discrete_dims or {}

    # sort the values in discrete dims in ascending order
    discrete_dims = {dim: sorted(values) for dim, values in discrete_dims.items()}

    # sort the categorical dims in ascending order
    cat_dims = {dim: sorted(values) for dim, values in cat_dims.items()}

    for dim, values in discrete_dims.items():
        lower_bnd, upper_bnd = bounds[:, dim].tolist()
        lower, upper = values[0], values[-1]
        if lower != lower_bnd:
            raise ValueError(
                f"Discrete dimension {dim} must start at the lower bound "
                f"{lower_bnd} but starts at {lower}."
            )
        if upper != upper_bnd:
            raise ValueError(
                f"Discrete dimension {dim} must end at the upper bound "
                f"{upper_bnd} but end at {upper}."
            )

    fixed_features = fixed_features or {}
    options = options or {}
    if options.get("max_optimization_problem_aggregation_size", 1) != 1:
        raise UnsupportedError(
            "optimize_acqf_mixed_alternating does not support "
            "max_optimization_problem_aggregation_size != 1. "
            "You might leave this option empty, though."
        )
    options.setdefault("batch_limit", MAX_BATCH_SIZE)
    options.setdefault("init_batch_limit", options["batch_limit"])
    if not (keys := set(options.keys())).issubset(SUPPORTED_OPTIONS):
        unsupported_keys = keys.difference(SUPPORTED_OPTIONS)
        raise UnsupportedError(
            f"Received an unsupported option {unsupported_keys}. {SUPPORTED_OPTIONS=}."
        )

    if equality_constraints is not None:
        for indices, _, __ in equality_constraints:
            # Raise an error if any index in indices is in discrete_dims or cat_dims
            if any(idx in discrete_dims or idx in cat_dims for idx in indices.tolist()):
                raise ValueError(
                    "Equality constraints can only be used with continuous degrees "
                    "of freedom."
                )

    # Update discrete dims and post processing functions to account for any
    # dimensions that should be using continuous relaxation.
    discrete_dims, post_processing_func = _setup_continuous_relaxation(
        discrete_dims=discrete_dims,
        max_discrete_values=assert_is_instance(
            options.get("max_discrete_values", MAX_DISCRETE_VALUES), int
        ),
        post_processing_func=post_processing_func,
    )

    opt_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=None,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=None,
        return_best_only=False,  # We don't want to perform the cont. optimization
        # step and only return best, but this function itself only returns best
        gen_candidates=gen_candidates_scipy,
        sequential=sequential,  # only relevant if all dims are cont.
    )
    if sequential:
        # Sequential optimization requires return_best_only to be True
        # But we turn it off here, as we "manually" perform the seq.
        # conditioning in the loop below
        _validate_sequential_inputs(
            opt_inputs=dataclasses.replace(opt_inputs, return_best_only=True)
        )

    base_X_pending = acq_function.X_pending if q > 1 else None
    dim = bounds.shape[-1]
    tkwargs: dict[str, Any] = {"device": bounds.device, "dtype": bounds.dtype}
    # Remove fixed features from dims, so they don't get optimized.
    discrete_dims = {
        dim: values
        for dim, values in discrete_dims.items()
        if dim not in fixed_features
    }
    cat_dims = {
        dim: values for dim, values in cat_dims.items() if dim not in fixed_features
    }
    non_cont_dims = [*discrete_dims.keys(), *cat_dims.keys()]
    if len(non_cont_dims) == 0:
        # If the problem is fully continuous, fall back to standard optimization.
        return _optimize_acqf(
            opt_inputs=dataclasses.replace(
                opt_inputs,
                return_best_only=True,
            )
        )
    if not (
        isinstance(non_cont_dims, list)
        and len(set(non_cont_dims)) == len(non_cont_dims)
        and min(non_cont_dims) >= 0
        and max(non_cont_dims) <= dim - 1
    ):
        raise ValueError(
            "`discrete_dims` and `cat_dims` must be dictionaries with unique, disjoint "
            "integers as keys between 0 and num_dims - 1."
        )
    discrete_dims_t = torch.tensor(
        list(discrete_dims.keys()), dtype=torch.long, device=tkwargs["device"]
    )
    cat_dims_t = torch.tensor(
        list(cat_dims.keys()), dtype=torch.long, device=tkwargs["device"]
    )
    non_cont_dims = torch.tensor(
        non_cont_dims, dtype=torch.long, device=tkwargs["device"]
    )
    cont_dims = complement_indices_like(indices=non_cont_dims, d=dim)
    # Fixed features are all in cont_dims. Remove them, so they don't get optimized.
    ff_idcs = torch.tensor(
        list(fixed_features.keys()), dtype=torch.long, device=tkwargs["device"]
    )
    cont_dims = cont_dims[(cont_dims.unsqueeze(-1) != ff_idcs).all(dim=-1)]
    candidates = torch.empty(0, dim, **tkwargs)
    for _q in range(q):
        # Generate starting points.
        best_X, best_acq_val = generate_starting_points(
            opt_inputs=opt_inputs,
            discrete_dims=discrete_dims,
            cat_dims=cat_dims,
            cont_dims=cont_dims,
        )

        done = torch.zeros(len(best_X), dtype=torch.bool, device=tkwargs["device"])
        for _step in range(options.get("maxiter_alternating", MAX_ITER_ALTER)):
            starting_acq_val = best_acq_val.clone()
            best_X[~done], best_acq_val[~done] = discrete_step(
                opt_inputs=opt_inputs,
                discrete_dims=discrete_dims,
                cat_dims=cat_dims,
                current_x=best_X[~done],
            )

            best_X[~done], best_acq_val[~done] = continuous_step(
                opt_inputs=opt_inputs,
                discrete_dims=discrete_dims_t,
                cat_dims=cat_dims_t,
                current_x=best_X[~done],
            )

            improvement = best_acq_val - starting_acq_val
            done_now = improvement < options.get("tol", CONVERGENCE_TOL)
            done = done | done_now
            if done.float().mean() >= STOP_AFTER_SHARE_CONVERGED:
                break

        new_candidate = best_X[torch.argmax(best_acq_val)].unsqueeze(0)
        candidates = torch.cat([candidates, new_candidate], dim=-2)
        # Update pending points to include the new candidate.
        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
    if q > 1:
        acq_function.set_X_pending(base_X_pending)

    if post_processing_func is not None:
        candidates = post_processing_func(candidates)

    with torch.no_grad():
        acq_value = acq_function(candidates)  # compute joint acquisition value
    return candidates, acq_value


def complement_indices_like(indices: Tensor, d: int) -> Tensor:
    r"""Computes a tensor of complement indices: {range(d) \\ indices}.
    Same as complement_indices but returns an integer tensor like indices.
    """
    return torch.tensor(
        complement_indices(indices.tolist(), d),
        device=indices.device,
        dtype=indices.dtype,
    )


def complement_indices(indices: list[int], d: int) -> list[int]:
    r"""Computes a list of complement indices: {range(d) \\ indices}.

    Args:
        indices: a list of integers.
        d: an integer dimension in which to compute the complement.

    Returns:
        A list of integer indices.
    """
    return sorted(set(range(d)).difference(indices))
