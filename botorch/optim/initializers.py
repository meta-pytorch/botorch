#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [Regis]
    R. G. Regis, C. A. Shoemaker. Combining radial basis function
    surrogates and dynamic coordinate search in high-dimensional
    expensive black-box optimization, Engineering Optimization, 2013.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Optional, Union

import torch
from botorch.acquisition import analytic, monte_carlo, multi_objective
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    _get_value_function,
    qKnowledgeGradient,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.acquisition.utils import isinstance_af
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    BotorchWarning,
    SamplingWarning,
)
from botorch.models.model import Model
from botorch.optim.utils import fix_features, get_X_baseline
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import (
    boltzmann_sample,
    draw_sobol_samples,
    get_polytope_samples,
    manual_seed,
    sample_perturbed_subset_dims,
    sample_truncated_normal_perturbations,
)
from botorch.utils.transforms import unnormalize
from torch import Tensor
from torch.quasirandom import SobolEngine

TGenInitialConditions = Callable[
    [
        # reasoning behind this annotation: contravariance
        qKnowledgeGradient,
        Tensor,
        int,
        int,
        int,
        Optional[dict[int, float]],
        Optional[dict[str, Union[bool, float, int]]],
        Optional[list[tuple[Tensor, Tensor, float]]],
        Optional[list[tuple[Tensor, Tensor, float]]],
    ],
    Optional[Tensor],
]


def transform_constraints(
    constraints: list[tuple[Tensor, Tensor, float]] | None, q: int, d: int
) -> list[tuple[Tensor, Tensor, float]] | None:
    r"""Transform constraints to sample from a d*q-dimensional space instead of a
    d-dimensional state.

    This function assumes that constraints are the same for each input batch,
    and broadcasts the constraints accordingly to the input batch shape.

    Args:
        constraints: A list of tuples (indices, coefficients, rhs), with each tuple
            encoding an (in-)equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) (>)= rhs`.
            If `indices` is a 2-d Tensor, this supports specifying constraints across
            the points in the `q`-batch (inter-point constraints). If `None`, this
            function is a nullop and simply returns `None`.
        q: Size of the `q`-batch.
        d: Dimensionality of the problem.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of transformed constraints, if
        there are constraints. Returns `None` otherwise.
    """
    if constraints is None:
        return None
    transformed = []
    for constraint in constraints:
        if len(constraint[0].shape) == 1:
            transformed += transform_intra_point_constraint(constraint, d, q)
        else:
            transformed.append(transform_inter_point_constraint(constraint, d))
    return transformed


def transform_intra_point_constraint(
    constraint: tuple[Tensor, Tensor, float], d: int, q: int
) -> list[tuple[Tensor, Tensor, float]]:
    r"""Transforms an intra-point/pointwise constraint from
    d-dimensional space to a d*q-dimesional space.

    Args:
        constraints: A list of tuples (indices, coefficients, rhs), with each tuple
            encoding an (in-)equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) (>)= rhs`. Here `indices` must
            be one-dimensional, and the constraint is applied to all points within the
            `q`-batch.
        d: Dimensionality of the problem.

    Raises:
        ValueError: If indices in the constraints are larger than the
            dimensionality d of the problem.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of transformed constraints.
    """
    indices, coefficients, rhs = constraint
    if indices.max() >= d:
        raise ValueError(
            f"Constraint indices cannot exceed the problem dimension {d=}."
        )
    return [
        (
            torch.tensor(
                [i * d + j for j in indices], dtype=torch.int64, device=indices.device
            ),
            coefficients,
            rhs,
        )
        for i in range(q)
    ]


def transform_inter_point_constraint(
    constraint: tuple[Tensor, Tensor, float], d: int
) -> tuple[Tensor, Tensor, float]:
    r"""Transforms an inter-point constraint from
    d-dimensional space to a d*q dimesional space.

    Args:
        constraints: A list of tuples (indices, coefficients, rhs), with each tuple
            encoding an (in-)equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) (>)= rhs`. `indices` must be a
            2-d Tensor, where in each row `indices[i] = (k_i, l_i)` the first index
            `k_i` corresponds to the `k_i`-th element of the `q`-batch and the second
            index `l_i` corresponds to the `l_i`-th feature of that element.

    Raises:
        ValueError: If indices in the constraints are larger than the
            dimensionality d of the problem.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: Transformed constraint.
    """
    indices, coefficients, rhs = constraint
    if indices[:, 1].max() >= d:
        raise ValueError(
            f"Constraint indices cannot exceed the problem dimension {d=}."
        )
    return (
        torch.tensor(
            [r[0] * d + r[1] for r in indices], dtype=torch.int64, device=indices.device
        ),
        coefficients,
        rhs,
    )


def sample_q_batches_from_polytope(
    n: int,
    q: int,
    bounds: Tensor,
    n_burnin: int,
    n_thinning: int,
    seed: int | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> Tensor:
    r"""Samples `n` q-baches from a polytope of dimension `d`.

    Args:
        n: Number of q-batches to sample.
        q: Number of samples per q-batch
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        n_burnin: The number of burn-in samples for the Markov chain sampler.
        n_thinning: The amount of thinning. The sampler will return every
            `n_thinning` sample (after burn-in).
        seed: The random seed.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `n x q x d`-dim tensor of samples.
    """

    # check if inter-point constraints are present
    inter_point = any(
        len(indices.shape) > 1
        for constraints in (inequality_constraints or [], equality_constraints or [])
        for indices, _, _ in constraints
    )

    if inter_point:
        samples = get_polytope_samples(
            n=n,
            bounds=torch.hstack([bounds for _ in range(q)]),
            inequality_constraints=transform_constraints(
                constraints=inequality_constraints, q=q, d=bounds.shape[1]
            ),
            equality_constraints=transform_constraints(
                constraints=equality_constraints, q=q, d=bounds.shape[1]
            ),
            seed=seed,
            n_burnin=n_burnin,
            n_thinning=n_thinning * q,
        )
    else:
        samples = get_polytope_samples(
            n=n * q,
            bounds=bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            seed=seed,
            n_burnin=n_burnin,
            n_thinning=n_thinning,
        )
    return samples.view(n, q, -1).cpu()


def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, bool | float | int] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    generator: Callable[[int, int, int | None], Tensor] | None = None,
    fixed_X_fantasies: Tensor | None = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimization.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic. Note: if `sample_around_best` is True (the default is False),
            then `2 * raw_samples` samples are used.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch_topn`, `initialize_q_batch_nonneg`, and
            `initialize_q_batch`. If `options` contains a `topn=True` then
            `initialize_q_batch_topn` will be used. Else if `options` contains a
            `nonnegative=True` entry, then `acq_function` is assumed to be
            non-negative (useful when using custom acquisition functions).
            `initialize_q_batch` will be used otherwise. In addition, an
            "init_batch_limit" option can be passed to specify the batch limit
            for the initialization. This is useful for avoiding memory limits
            when computing the batch posterior over raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        generator: Callable for generating samples that are then further
            processed. It receives `n`, `q` and `seed` as arguments
            and returns a tensor of shape `n x q x d`.
        fixed_X_fantasies: A fixed set of fantasy points to concatenate to
            the `q` candidates being initialized along the `-2` dimension. The
            shape should be `num_pseudo_points x d`. E.g., this should be
            `num_fantasies x d` for KG and `num_fantasies*num_pareto x d`
            for HVKG.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    if bounds.isinf().any():
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    options = options or {}
    sample_around_best = options.get("sample_around_best", False)
    if sample_around_best and equality_constraints:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present."
        )
    if sample_around_best and generator:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when custom "
            "generator is be used."
        )
    seed: int | None = options.get("seed")
    batch_limit: int | None = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds_cpu = bounds.cpu()

    if options.get("topn"):
        init_func = initialize_q_batch_topn
        init_func_opts = ["sorted", "largest"]
    elif options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        init_func_opts = ["alpha", "eta"]
    else:
        init_func = initialize_q_batch
        init_func_opts = ["eta"]

    for opt in init_func_opts:
        # default value of "largest" to "acq_function.maximize" if it exists
        if opt == "largest" and hasattr(acq_function, "maximize"):
            init_kwargs[opt] = acq_function.maximize

        if opt in options:
            init_kwargs[opt] = options.get(opt)

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM:
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
            stacklevel=3,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if generator is not None:
                X_rnd = generator(n, q, seed)
            # check if no constraints are provided
            elif not (inequality_constraints or equality_constraints):
                if effective_dim <= SobolEngine.MAXDIM:
                    X_rnd = draw_sobol_samples(bounds=bounds_cpu, n=n, q=q, seed=seed)
                else:
                    with manual_seed(seed):
                        # load on cpu
                        X_rnd_nlzd = torch.rand(
                            n, q, bounds_cpu.shape[-1], dtype=bounds.dtype
                        )
                    X_rnd = unnormalize(
                        X_rnd_nlzd, bounds_cpu, update_constant_bounds=False
                    )
            else:
                X_rnd = sample_q_batches_from_polytope(
                    n=n,
                    q=q,
                    bounds=bounds,
                    n_burnin=options.get("n_burnin", 10000),
                    n_thinning=options.get("n_thinning", 32),
                    seed=seed,
                    equality_constraints=equality_constraints,
                    inequality_constraints=inequality_constraints,
                )

            # sample additional points around best
            if sample_around_best:
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                    prob_perturb=options.get("sample_around_best_prob_perturb"),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )
            # Keep X on CPU for consistency & to limit GPU memory usage.
            X_rnd = fix_features(
                X_rnd, fixed_features=fixed_features, replace_current_value=True
            ).cpu()

            # Append the fixed fantasies to the randomly generated points
            if fixed_X_fantasies is not None:
                if (d_f := fixed_X_fantasies.shape[-1]) != (d_r := X_rnd.shape[-1]):
                    raise BotorchTensorDimensionError(
                        "`fixed_X_fantasies` and `bounds` must both have the same "
                        f"trailing dimension `d`, but have {d_f} and {d_r}, "
                        "respectively."
                    )
                X_rnd = torch.cat(
                    [
                        X_rnd,
                        fixed_X_fantasies.cpu()
                        .unsqueeze(0)
                        .expand(X_rnd.shape[0], *fixed_X_fantasies.shape),
                    ],
                    dim=-2,
                )

            # Evaluate the acquisition function on `X_rnd` using `batch_limit`
            # sized chunks.
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                # Evaluate the acquisition function on `X_rnd` using `batch_limit`
                # sized chunks.
                acq_vals = torch.cat(
                    [
                        acq_function(x_.to(device=device)).cpu()
                        for x_ in X_rnd.split(split_size=batch_limit, dim=0)
                    ],
                    dim=0,
                )

            # Downselect the initial conditions based on the acquisition function values
            batch_initial_conditions, _ = init_func(
                X=X_rnd, acq_vals=acq_vals, n=num_restarts, **init_kwargs
            )
            batch_initial_conditions = batch_initial_conditions.to(device=device)

            # Return the initial conditions if no warnings were raised
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions

            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd

    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
        stacklevel=2,
    )
    return batch_initial_conditions


def gen_one_shot_kg_initial_conditions(
    acq_function: qKnowledgeGradient,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, bool | float | int] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> Tensor | None:
    r"""Generate a batch of smart initializations for qKnowledgeGradient.

    This function generates initial conditions for optimizing one-shot KG using
    the maximizer of the posterior objective. Intutively, the maximizer of the
    fantasized posterior will often be close to a maximizer of the current
    posterior. This function uses that fact to generate the initial conditions
    for the fantasy points. Specifically, a fraction of `1 - frac_random` (see
    options) is generated by sampling from the set of maximizers of the
    posterior objective (obtained via random restart optimization) according to
    a softmax transformation of their respective values. This means that this
    initialization strategy internally solves an acquisition function
    maximization problem. The remaining `frac_random` fantasy points as well as
    all `q` candidate points are chosen according to the standard initialization
    strategy in `gen_batch_initial_conditions`.

    Args:
        acq_function: The qHypervolumeKnowledgeGradient instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q' x d` tensor that can be used as initial conditions
        for `optimize_acqf()`. Here `q' = q + num_fantasies` is the total number
        of points (candidate points plus fantasy points).

    Example:
        >>> qHVKG = qHypervolumeKnowledgeGradient(model, ref_point=num_fantasies=64)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_one_shot_hvkg_initial_conditions(
        >>>     qHVKG, bounds, q=3, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    options = options or {}
    frac_random: float = options.get("frac_random", 0.1)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )
    q_aug = acq_function.get_augmented_q_batch_size(q=q)

    # Check for inter-point (in)equality constraints
    for constraints, type_of_constraints in [
        (inequality_constraints, "inequality"),
        (equality_constraints, "equality"),
    ]:
        if constraints is not None and len(constraints) > 0:
            for indices, _, _ in constraints:
                if len(indices.shape) > 1:
                    raise NotImplementedError(
                        "Indices must be one-dimensional "
                        "in gen_one_shot_kg_initial_conditions. "
                        f"Received indices {indices} "
                        f"for {type_of_constraints} constrained."
                    )

    # TODO: Avoid unnecessary computation by not generating all candidates
    ics = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q_aug,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features=fixed_features,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    # compute maximizer of the value function
    value_function = _get_value_function(
        model=acq_function.model,
        objective=acq_function.objective,
        posterior_transform=acq_function.posterior_transform,
        sampler=acq_function.inner_sampler,
        project=getattr(acq_function, "project", None),
    )
    from botorch.optim.optimize import optimize_acqf

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        fixed_features=fixed_features,
        return_best_only=False,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    # sampling from the optimizers
    n_value = int((1 - frac_random) * (q_aug - q))  # number of non-random ICs
    idx = boltzmann_sample(
        function_values=fantasy_vals,
        num_samples=num_restarts * n_value,
        eta=options.get("eta", 2.0),
        replacement=True,
    )
    # set the respective initial conditions to the sampled optimizers
    ics[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)
    return ics


def gen_one_shot_hvkg_initial_conditions(
    acq_function: qHypervolumeKnowledgeGradient,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, bool | float | int] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> Tensor | None:
    r"""Generate a batch of smart initializations for qHypervolumeKnowledgeGradient.

    This function generates initial conditions for optimizing one-shot HVKG using
    the hypervolume maximizing set (of fixed size) under the posterior mean.
    Intutively, the hypervolume maximizing set of the fantasized posterior mean
    will often be close to a hypervolume maximizing set under the current posterior
    mean. This function uses that fact to generate the initial conditions
    for the fantasy points. Specifically, a fraction of `1 - frac_random` (see
    options) of the restarts are generated by learning the hypervolume maximizing sets
    under the current posterior mean, where each hypervolume maximizing set is
    obtained from maximizing the hypervolume from a different starting point. Given
    a hypervolume maximizing set, the `q` candidate points are selected using to the
    standard initialization strategy in `gen_batch_initial_conditions`, with the fixed
    hypervolume maximizing set. The remaining `frac_random` restarts fantasy points
    as well as all `q` candidate points are chosen according to the standard
    initialization strategy in `gen_batch_initial_conditions`.

    Args:
        acq_function: The qKnowledgeGradient instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).
        inequality constraints: Optionally, list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. Each
            tensor of indices must be one-dimensional, since inter-point
            constraints are not supported here.
        equality constraints: Optionally, a list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q' x d` tensor that can be used as initial conditions
        for `optimize_acqf()`. Here `q' = q + num_fantasies` is the total number
        of points (candidate points plus fantasy points).

    Example:
        >>> qHVKG = qHypervolumeKnowledgeGradient(model, ref_point)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_one_shot_hvkg_initial_conditions(
        >>>     qHVKG, bounds, q=3, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    from botorch.optim.optimize import optimize_acqf

    options = options or {}
    frac_random: float = options.get("frac_random", 0.1)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )

    value_function = _get_hv_value_function(
        model=acq_function.model,
        ref_point=acq_function.ref_point,
        objective=acq_function.objective,
        sampler=acq_function.inner_sampler,
        use_posterior_mean=acq_function.use_posterior_mean,
    )

    is_mf_hvkg = isinstance(acq_function, qMultiFidelityHypervolumeKnowledgeGradient)
    if is_mf_hvkg:
        dim = bounds.shape[-1]
        fidelity_dims, fidelity_targets = zip(*acq_function.target_fidelities.items())
        value_function = FixedFeatureAcquisitionFunction(
            acq_function=value_function,
            d=dim,
            columns=fidelity_dims,
            values=fidelity_targets,
        )

        non_fidelity_dims = list(set(range(dim)) - set(fidelity_dims))

    num_optim_restarts = int(round(num_restarts * (1 - frac_random)))
    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds[:, non_fidelity_dims] if is_mf_hvkg else bounds,
        q=acq_function.num_pareto,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        fixed_features=fixed_features,
        return_best_only=False,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        sequential=False,
    )
    # sampling from the optimizers
    if num_optim_restarts > 0:
        idx = boltzmann_sample(
            function_values=fantasy_vals,
            num_samples=num_optim_restarts * acq_function.num_fantasies,
            eta=options.get("eta", 2.0),
            replacement=True,
        )

        optim_ics = fantasy_cands[idx]
        if is_mf_hvkg:
            # add fixed features
            optim_ics = value_function._construct_X_full(optim_ics)
        optim_ics = optim_ics.reshape(
            num_optim_restarts, acq_function.num_pseudo_points, bounds.shape[-1]
        )

    # get random initial conditions
    num_random_restarts = num_restarts - num_optim_restarts
    if num_random_restarts > 0:
        q_aug = acq_function.get_augmented_q_batch_size(q=q)
        base_ics = gen_batch_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q_aug,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

        if num_optim_restarts > 0:
            probs = torch.full(
                (num_restarts,),
                1.0 / num_restarts,
                dtype=optim_ics.dtype,
                device=optim_ics.device,
            )
            optim_idxr = probs.multinomial(
                num_samples=num_optim_restarts, replacement=False
            )
            base_ics[optim_idxr, q:] = optim_ics
    else:
        # optim_ics is num_restarts x num_pseudo_points x d
        # add padding so that base_ics is num_restarts x q+num_pseudo_points x d
        q_padding = torch.zeros(
            optim_ics.shape[0],
            q,
            optim_ics.shape[-1],
            dtype=optim_ics.dtype,
            device=optim_ics.device,
        )
        base_ics = torch.cat([q_padding, optim_ics], dim=-2)

    if num_optim_restarts > 0:
        all_ics = []
        if num_random_restarts > 0:
            optim_idcs = optim_idxr.view(-1).tolist()
        else:
            optim_idcs = list(range(num_restarts))
        for i in list(range(num_restarts)):
            if i in optim_idcs:
                # optimize the q points,
                # given fixed, optimized fantasy designs
                ics = gen_batch_initial_conditions(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=1,
                    raw_samples=raw_samples,
                    fixed_features=fixed_features,
                    options=options,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    fixed_X_fantasies=base_ics[i, q:],
                )
            else:
                # ics are all randomly sampled
                ics = base_ics[i : i + 1]
            all_ics.append(ics)
        return torch.cat(all_ics, dim=0)

    return base_ics


def gen_value_function_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    current_model: Model,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, bool | float | int] | None = None,
) -> Tensor:
    r"""Generate a batch of smart initializations for optimizing
    the value function of qKnowledgeGradient.

    This function generates initial conditions for optimizing the inner problem of
    KG, i.e. its value function, using the maximizer of the posterior objective.
    Intutively, the maximizer of the fantasized posterior will often be close to a
    maximizer of the current posterior. This function uses that fact to generate the
    initital conditions for the fantasy points. Specifically, a fraction of `1 -
    frac_random` (see options) of raw samples is generated by sampling from the set of
    maximizers of the posterior objective (obtained via random restart optimization)
    according to a softmax transformation of their respective values. This means that
    this initialization strategy internally solves an acquisition function
    maximization problem. The remaining raw samples are generated using
    `draw_sobol_samples`. All raw samples are then evaluated, and the initial
    conditions are selected according to the standard initialization strategy in
    'initialize_q_batch' individually for each inner problem.

    Args:
        acq_function: The value function instance to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of
            task features.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        current_model: The model of the KG acquisition function that was used to
            generate the fantasy model of the value function.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. These contain all
            settings for the standard heuristic initialization from
            `gen_batch_initial_conditions`. In addition, they contain
            `frac_random` (the fraction of fully random fantasy points),
            `num_inner_restarts` and `raw_inner_samples` (the number of random
            restarts and raw samples for solving the posterior objective
            maximization problem, respectively) and `eta` (temperature parameter
            for sampling heuristic from posterior objective maximizers).

    Returns:
        A `num_restarts x batch_shape x q x d` tensor that can be used as initial
        conditions for `optimize_acqf()`. Here `batch_shape` is the batch shape
        of value function model.

    Example:
        >>> fant_X = torch.rand(5, 1, 2)
        >>> fantasy_model = model.fantasize(fant_X, SobolQMCNormalSampler(16))
        >>> value_function = PosteriorMean(fantasy_model)
        >>> bounds = torch.tensor([[0., 0.], [1., 1.]])
        >>> Xinit = gen_value_function_initial_conditions(
        >>>     value_function, bounds, num_restarts=10, raw_samples=512,
        >>>     options={"frac_random": 0.25},
        >>> )
    """
    options = options or {}
    seed: int | None = options.get("seed")
    frac_random: float = options.get("frac_random", 0.6)
    if not 0 < frac_random < 1:
        raise ValueError(
            f"frac_random must take on values in (0,1). Value: {frac_random}"
        )

    # compute maximizer of the current value function
    value_function = _get_value_function(
        model=current_model,
        objective=getattr(acq_function, "objective", None),
        posterior_transform=acq_function.posterior_transform,
        sampler=getattr(acq_function, "sampler", None),
        project=getattr(acq_function, "project", None),
    )
    from botorch.optim.optimize import optimize_acqf

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=options.get("num_inner_restarts", 20),
        raw_samples=options.get("raw_inner_samples", 1024),
        fixed_features=fixed_features,
        return_best_only=False,
        options={
            k: v
            for k, v in options.items()
            if k
            not in ("frac_random", "num_inner_restarts", "raw_inner_samples", "eta")
        },
    )

    batch_shape = acq_function.model.batch_shape
    # sampling from the optimizers
    n_value = int((1 - frac_random) * raw_samples)  # number of non-random ICs
    if n_value > 0:
        idx = boltzmann_sample(
            function_values=fantasy_vals.expand(*batch_shape, -1),
            num_samples=n_value,
            eta=options.get("eta", 2.0),
            replacement=True,
        ).permute(-1, *range(len(batch_shape)))
        resampled = fantasy_cands[idx]
    else:
        resampled = torch.empty(
            0,
            *batch_shape,
            1,
            bounds.shape[-1],
            dtype=fantasy_cands.dtype,
            device=fantasy_cands.device,
        )
    # add qMC samples
    randomized = draw_sobol_samples(
        bounds=bounds, n=raw_samples - n_value, q=1, batch_shape=batch_shape, seed=seed
    ).to(resampled)
    # full set of raw samples
    X_rnd = torch.cat([resampled, randomized], dim=0)
    X_rnd = fix_features(
        X_rnd, fixed_features=fixed_features, replace_current_value=True
    )

    # evaluate the raw samples
    with torch.no_grad():
        acq_vals = acq_function(X_rnd)

    # select the restart points using the heuristic
    X_init, _ = initialize_q_batch(
        X=X_rnd, acq_vals=acq_vals, n=num_restarts, eta=options.get("eta", 2.0)
    )
    return X_init


def initialize_q_batch(
    X: Tensor, acq_vals: Tensor, n: int, eta: float = 1.0
) -> tuple[Tensor, Tensor]:
    r"""Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from `X` (without replacement) with probability
    proportional to `exp(eta * Z)`, where
    `Z = (acq_vals - mean(acq_vals)) / std(acq_vals)`
    and `eta` is a temperature parameter.

    When using an acquisiton function that is non-negative and possibly zero
    over large areas of the feature space (e.g. qEI), you should use
    `initialize_q_batch_nonneg` instead.

    Args:
        X: A `b x batch_shape x q x d` tensor of `b` - `batch_shape` samples of
            `q`-batches from a d`-dim feature space. Typically, these are generated
            using qMC sampling.
        acq_vals: A tensor of `b x batch_shape` outcomes associated with the samples.
            Typically, this is the value of the batch acquisition function to be
            maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.

    Returns:
        - An `n x batch_shape x q x d` tensor of `n` - `batch_shape` `q`-batch initial
          conditions, where each batch of `n x q x d` samples is selected independently.
        - An `n x batch_shape` tensor of the corresponding acquisition values.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> X_rnd = torch.rand(500, 3, 6)
        >>> X_init, acq_init = initialize_q_batch(X=X_rnd, acq_vals=qUCB(X_rnd), n=10)
    """
    n_samples = X.shape[0]
    batch_shape = X.shape[1:-2] or torch.Size()
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X, acq_vals

    Ystd = acq_vals.std(dim=0)
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
            stacklevel=3,
        )
        idcs = torch.randperm(n=n_samples, device=X.device)[:n]
        return X[idcs], acq_vals[idcs]

    max_val, max_idx = torch.max(acq_vals, dim=0)
    idcs = boltzmann_sample(
        acq_vals.permute(*range(1, len(batch_shape) + 1), 0),
        num_samples=n,
        eta=eta,
    ).permute(-1, *range(len(batch_shape)))

    # make sure we get the maximum
    if batch_shape == torch.Size():
        if max_idx not in idcs:
            idcs[-1] = max_idx
    else:
        has_max = (max_idx == idcs).any(dim=0)
        idcs[-1, ~has_max] = max_idx[~has_max]

    if batch_shape == torch.Size():
        return X[idcs], acq_vals[idcs]
    else:
        X_select = X.gather(
            dim=0, index=idcs.view(*idcs.shape, 1, 1).expand(n, *X.shape[1:])
        )
        acq_select = acq_vals.gather(dim=0, index=idcs)
        return X_select, acq_select


def initialize_q_batch_nonneg(
    X: Tensor, acq_vals: Tensor, n: int, eta: float = 1.0, alpha: float = 1e-4
) -> tuple[Tensor, Tensor]:
    r"""Heuristic for selecting initial conditions for non-neg. acquisition functions.

    This function is similar to `initialize_q_batch`, but designed specifically
    for acquisition functions that are non-negative and possibly zero over
    large areas of the feature space (e.g. qEI). All samples for which
    `acq_vals < alpha * max(acq_vals)` will be ignored (assuming that `acq_vals`
    contains at least one positive value).

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        acq_vals: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.
        eta: Temperature parameter for weighting samples.
        alpha: The threshold (as a fraction of the maximum observed value) under
            which to ignore samples. All input samples for which
            `Y < alpha * max(Y)` will be ignored.

    Returns:
        - An `n x q x d` tensor of `n` `q`-batch initial conditions.
        - An `n` tensor of the corresponding acquisition values.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> X_rnd = torch.rand(500, 3, 6)
        >>> X_init, acq_init = initialize_q_batch_nonneg(
        ...     X=X_rnd, acq_vals=qEI(X_rnd), n=10
        ... )
    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    elif n == n_samples:
        return X, acq_vals

    max_val, max_idx = torch.max(acq_vals, dim=0)
    if torch.any(max_val <= 0):
        warnings.warn(
            "All acquisition values for raw sampled points are nonpositive, so "
            "initial conditions are being selected randomly.",
            BadInitialCandidatesWarning,
            stacklevel=3,
        )
        idcs = torch.randperm(n=n_samples, device=X.device)[:n]
        return X[idcs], acq_vals[idcs]

    # make sure there are at least `n` points with positive acquisition values
    pos = acq_vals > 0
    num_pos = pos.sum().item()
    if num_pos < n:
        # select all positive points and then fill remaining quota with randomly
        # selected points
        remaining_indices = (~pos).nonzero(as_tuple=False).view(-1)
        rand_indices = torch.randperm(
            remaining_indices.shape[0], device=acq_vals.device
        )
        sampled_remaining_indices = remaining_indices[rand_indices[: n - num_pos]]
        pos[sampled_remaining_indices] = 1
        return X[pos], acq_vals[pos]
    # select points within alpha of max_val, iteratively decreasing alpha by a
    # factor of 10 as necessary
    alpha_pos = acq_vals >= alpha * max_val
    while alpha_pos.sum() < n:
        alpha = 0.1 * alpha
        alpha_pos = acq_vals >= alpha * max_val
    alpha_pos_idcs = torch.arange(len(acq_vals), device=acq_vals.device)[alpha_pos]
    weights = torch.exp(eta * (acq_vals[alpha_pos] / max_val - 1))
    idcs = alpha_pos_idcs[torch.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs], acq_vals[idcs]


def initialize_q_batch_topn(
    X: Tensor, acq_vals: Tensor, n: int, largest: bool = True, sorted: bool = True
) -> tuple[Tensor, Tensor]:
    r"""Take the top `n` initial conditions for candidate generation.

    Args:
        X: A `b x q x d` tensor of `b` samples of `q`-batches from a `d`-dim.
            feature space. Typically, these are generated using qMC.
        acq_vals: A tensor of `b` outcomes associated with the samples. Typically, this
            is the value of the batch acquisition function to be maximized.
        n: The number of initial condition to be generated. Must be less than `b`.

    Returns:
        - An `n x q x d` tensor of `n` `q`-batch initial conditions.
        - An `n` tensor of the corresponding acquisition values.

    Example:
        >>> # To get `n=10` starting points of q-batch size `q=3`
        >>> # for model with `d=6`:
        >>> qUCB = qUpperConfidenceBound(model, beta=0.1)
        >>> X_rnd = torch.rand(500, 3, 6)
        >>> X_init, acq_init = initialize_q_batch_topn(
        ...     X=X_rnd, acq_vals=qUCB(X_rnd), n=10
        ... )

    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X, acq_vals

    Ystd = acq_vals.std(dim=0)
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
            stacklevel=3,
        )
        idcs = torch.randperm(n=n_samples, device=X.device)[:n]
        return X[idcs], acq_vals[idcs]

    topk_out, topk_idcs = acq_vals.topk(n, largest=largest, sorted=sorted)
    return X[topk_idcs], topk_out


def sample_points_around_best(
    acq_function: AcquisitionFunction,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
    best_pct: float = 5.0,
    subset_sigma: float = 1e-1,
    prob_perturb: float | None = None,
) -> Tensor | None:
    r"""Find best points and sample nearby points.

    Args:
        acq_function: The acquisition function.
        n_discrete_points: The number of points to sample.
        sigma: The standard deviation of the additive gaussian noise for
            perturbing the best points.
        bounds: A `2 x d`-dim tensor containing the bounds.
        best_pct: The percentage of best points to perturb.
        subset_sigma: The standard deviation of the additive gaussian
            noise for perturbing a subset of dimensions of the best points.
        prob_perturb: The probability of perturbing each dimension.

    Returns:
        An optional `n_discrete_points x d`-dim tensor containing the
            sampled points. This is None if no baseline points are found.
    """
    X = get_X_baseline(acq_function=acq_function)
    if X is None:
        return
    with torch.no_grad():
        try:
            posterior = acq_function.model.posterior(X)
        except AttributeError:
            warnings.warn(
                "Failed to sample around previous best points.",
                BotorchWarning,
                stacklevel=3,
            )
            return
        mean = posterior.mean
        while mean.ndim > 2:
            # take average over batch dims
            mean = mean.mean(dim=0)
        try:
            f_pred = acq_function.objective(mean)
        # Some acquisition functions do not have an objective
        # and for some acquisition functions the objective is None
        except (AttributeError, TypeError):
            f_pred = mean
        if hasattr(acq_function, "maximize"):
            # make sure that the optimiztaion direction is set properly
            if not acq_function.maximize:
                f_pred = -f_pred
        try:
            # handle constraints for EHVI-based acquisition functions
            constraints = acq_function.constraints
            if constraints is not None:
                neg_violation = -torch.stack(
                    [c(mean).clamp_min(0.0) for c in constraints], dim=-1
                ).sum(dim=-1)
                feas = neg_violation == 0
                if feas.any():
                    f_pred[~feas] = float("-inf")
                else:
                    # set objective equal to negative violation
                    f_pred = neg_violation
        except AttributeError:
            pass
        if f_pred.ndim == mean.ndim and f_pred.shape[-1] > 1:
            # multi-objective
            # find pareto set
            is_pareto = is_non_dominated(f_pred)
            best_X = X[is_pareto]
        else:
            if f_pred.shape[-1] == 1:
                f_pred = f_pred.squeeze(-1)
            n_best = max(1, round(X.shape[0] * best_pct / 100))
            # the view() is to ensure that best_idcs is not a scalar tensor
            best_idcs = torch.topk(f_pred, n_best).indices.view(-1)
            best_X = X[best_idcs]
    use_perturbed_sampling = best_X.shape[-1] >= 20 or prob_perturb is not None
    n_trunc_normal_points = (
        n_discrete_points // 2 if use_perturbed_sampling else n_discrete_points
    )
    perturbed_X = sample_truncated_normal_perturbations(
        X=best_X,
        n_discrete_points=n_trunc_normal_points,
        sigma=sigma,
        bounds=bounds,
    )
    if use_perturbed_sampling:
        perturbed_subset_dims_X = sample_perturbed_subset_dims(
            X=best_X,
            bounds=bounds,
            # ensure that we return n_discrete_points
            n_discrete_points=n_discrete_points - n_trunc_normal_points,
            sigma=sigma,
            prob_perturb=prob_perturb,
        )
        perturbed_X = torch.cat([perturbed_X, perturbed_subset_dims_X], dim=0)
        # shuffle points
        perm = torch.randperm(perturbed_X.shape[0], device=X.device)
        perturbed_X = perturbed_X[perm]
    return perturbed_X


def is_nonnegative(acq_function: AcquisitionFunction) -> bool:
    r"""Determine whether a given acquisition function is non-negative.

    Args:
        acq_function: The `AcquisitionFunction` instance.

    Returns:
        True if `acq_function` is non-negative, False if not, or if the behavior
        is unknown (for custom acquisition functions).

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.1)
        >>> is_nonnegative(qEI)  # returns True
    """
    return isinstance_af(
        acq_function,
        (
            analytic.ExpectedImprovement,
            analytic.ConstrainedExpectedImprovement,
            analytic.ProbabilityOfImprovement,
            analytic.NoisyExpectedImprovement,
            monte_carlo.qExpectedImprovement,
            monte_carlo.qNoisyExpectedImprovement,
            monte_carlo.qProbabilityOfImprovement,
            multi_objective.analytic.ExpectedHypervolumeImprovement,
            multi_objective.monte_carlo.qExpectedHypervolumeImprovement,
            multi_objective.monte_carlo.qNoisyExpectedHypervolumeImprovement,
        ),
    )
