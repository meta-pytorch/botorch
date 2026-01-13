#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for multi-objective acquisition functions.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from math import ceil
from typing import Any

import torch
from botorch.acquisition import monte_carlo  # noqa F401
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.utils import _prune_inferior_shared_processing
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from pyre_extensions import assert_is_instance
from torch import Tensor


def get_default_partitioning_alpha(num_objectives: int) -> float:
    r"""Determines an approximation level based on the number of objectives.

    If ``alpha`` is 0, FastNondominatedPartitioning should be used. Otherwise,
    an approximate NondominatedPartitioning should be used with approximation
    level ``alpha``.

    Args:
        num_objectives: the number of objectives.

    Returns:
        The approximation level ``alpha``.
    """
    if num_objectives <= 4:
        return 0.0
    elif num_objectives > 6:
        warnings.warn(
            "EHVI works best for less than 7 objectives.",
            BotorchWarning,
            stacklevel=3,
        )
    return 10 ** (-2 if num_objectives >= 6 else -3)


def prune_inferior_points_multi_objective(
    model: Model,
    X: Tensor,
    ref_point: Tensor,
    objective: MCMultiOutputObjective | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    marginalize_dim: int | None = None,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be pareto optimal.

    Given a model, an objective, and an input tensor ``X``, this function returns
    the subset of points in ``X`` that have some probability of being pareto
    optimal, better than the reference point, and feasible. This function uses
    sampling to estimate the probabilities, the higher the number of points ``n``
    in ``X`` the higher the number of samples ``num_samples`` should be to obtain
    accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape ``n x d``. Batched inputs are currently not
            supported.
        ref_point: The reference point.
        objective: The objective under which to evaluate the posterior.
        constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply
            feasibility.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            ``0 < max_frac <= 1``. Ensures that the number of elements in the
            returned tensor does not exceed ``ceil(max_frac * n)``.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A ``n' x d`` with subset of points in ``X``, where

            n' = min(N_nz, ceil(max_frac * n))

        with ``N_nz`` the number of points in ``X`` that have non-zero (empirical,
        under ``num_samples`` samples) probability of being pareto optimal.
    """
    max_points, obj_vals, infeas = _prune_inferior_shared_processing(
        model=model,
        X=X,
        is_moo=True,
        objective=objective,
        constraints=constraints,
        num_samples=num_samples,
        max_frac=max_frac,
        marginalize_dim=marginalize_dim,
    )
    if infeas.any():
        obj_vals[infeas] = ref_point
    pareto_mask = is_non_dominated(obj_vals, deduplicate=False) & (
        obj_vals > ref_point
    ).all(dim=-1)
    probs = pareto_mask.to(dtype=X.dtype).mean(dim=0)
    idcs = probs.nonzero().view(-1)
    if idcs.shape[0] > max_points:
        counts, order_idcs = torch.sort(probs, stable=True, descending=True)
        idcs = order_idcs[:max_points]
    effective_n_w = obj_vals.shape[-2] // X.shape[-2]
    idcs = (idcs / effective_n_w).long().unique()
    return X[idcs]


def compute_sample_box_decomposition(
    pareto_fronts: Tensor,
    partitioning: type[BoxDecomposition] = DominatedPartitioning,
    maximize: bool = True,
    num_constraints: int = 0,
) -> Tensor:
    r"""Computes the box decomposition associated with some sampled optimal
    objectives. This also supports the single-objective and constrained optimization
    setting. An objective ``y`` is feasible if ``y <= 0``.

    To take advantage of batch computations, we pad the hypercell bounds with a
    ``2 x (M + K)``-dim Tensor of zeros ``[0, 0]``.

    Args:
        pareto_fronts: A ``num_pareto_samples x num_pareto_points x M`` dim Tensor
            containing the sampled optimal set of objectives.
        partitioning: A ``BoxDecomposition`` module that is used to obtain the
            hyper-rectangle bounds for integration. In the unconstrained case, this
            gives the partition of the dominated space. In the constrained case, this
            gives the partition of the feasible dominated space union the infeasible
            space.
        maximize: If true, the box-decomposition is computed assuming maximization.
        num_constraints: The number of constraints ``K``.

    Returns:
        A ``num_pareto_samples x 2 x J x (M + K)``-dim Tensor containing the bounds for
        the hyper-rectangles. The number ``J`` is the smallest number of boxes needed
        to partition all the Pareto samples.
    """
    tkwargs: dict[str, Any] = {
        "dtype": pareto_fronts.dtype,
        "device": pareto_fronts.device,
    }
    # We will later compute ``norm.log_prob(NEG_INF)``, this is ``-inf``
    # if ``NEG_INF`` is too small.
    NEG_INF = -1e10

    if pareto_fronts.ndim != 3:
        raise UnsupportedError(
            "Currently this only supports Pareto fronts of the shape "
            "`num_pareto_samples x num_pareto_points x num_objectives`."
        )

    num_pareto_samples = pareto_fronts.shape[0]
    M = pareto_fronts.shape[-1]
    K = num_constraints
    ref_point = torch.ones(M, **tkwargs) * NEG_INF
    weight = 1.0 if maximize else -1.0

    if M == 1:
        # Only consider a Pareto front with one element.
        extreme_values = assert_is_instance(
            weight * torch.max(weight * pareto_fronts, dim=-2).values, Tensor
        )
        ref_point = weight * ref_point.expand(extreme_values.shape)

        if maximize:
            hypercell_bounds = torch.stack(
                [ref_point, extreme_values], dim=-2
            ).unsqueeze(-1)
        else:
            hypercell_bounds = torch.stack(
                [extreme_values, ref_point], dim=-2
            ).unsqueeze(-1)
    else:
        bd_list = []
        for i in range(num_pareto_samples):
            bd_list = bd_list + [
                partitioning(ref_point=ref_point, Y=weight * pareto_fronts[i, :, :])
            ]

        # ``num_pareto_samples x 2 x J x (M + K)``
        hypercell_bounds = (
            BoxDecompositionList(*bd_list).get_hypercell_bounds().movedim(0, 1)
        )

        # If minimizing, then the bounds should be negated and flipped
        if not maximize:
            hypercell_bounds = weight * torch.flip(hypercell_bounds, dims=[1])

    # Add an extra box for the inequality constraint.
    if K > 0:
        # ``num_pareto_samples x 2 x (J - 1) x K``
        feasible_boxes = torch.zeros(hypercell_bounds.shape[:-1] + (K,), **tkwargs)

        feasible_boxes[..., 0, :, :] = NEG_INF
        # ``num_pareto_samples x 2 x (J - 1) x (M + K)``
        hypercell_bounds = torch.cat([hypercell_bounds, feasible_boxes], dim=-1)

        # ``num_pareto_samples x 2 x 1 x (M + K)``
        infeasible_box = torch.zeros(
            hypercell_bounds.shape[:-2] + (1, M + K), **tkwargs
        )
        infeasible_box[..., 1, :, M:] = -NEG_INF
        infeasible_box[..., 0, :, 0:M] = NEG_INF
        infeasible_box[..., 1, :, 0:M] = -NEG_INF

        # ``num_pareto_samples x 2 x J x (M + K)``
        hypercell_bounds = torch.cat([hypercell_bounds, infeasible_box], dim=-2)

    # ``num_pareto_samples x 2 x J x (M + K)``
    return hypercell_bounds


def random_search_optimizer(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_points: int,
    maximize: bool,
    pop_size: int = 1024,
    max_tries: int = 10,
) -> tuple[Tensor, Tensor]:
    r"""Optimize a function via random search.

    Args:
        model: The model.
        bounds: A ``2 x d``-dim Tensor containing the input bounds.
        num_points: The number of optimal points to be outputted.
        maximize: If true, we consider a maximization problem.
        pop_size: The number of function evaluations per try.
        max_tries: The maximum number of tries.

    Returns:
        A two-element tuple containing

        - A ``num_points x d``-dim Tensor containing the collection of optimal inputs.
        - A ``num_points x M``-dim Tensor containing the collection of optimal
            objectives.
    """
    tkwargs: dict[str, Any] = {"dtype": bounds.dtype, "device": bounds.device}
    weight = 1.0 if maximize else -1.0
    optimal_inputs = torch.tensor([], **tkwargs)
    optimal_outputs = torch.tensor([], **tkwargs)
    num_tries = 0
    num_found = 0
    ratio = 2
    while ratio > 1 and num_tries < max_tries:
        X = draw_sobol_samples(bounds=bounds, n=pop_size, q=1).squeeze(-2)
        Y = model.posterior(X).mean
        X_aug = torch.cat([optimal_inputs, X], dim=0)
        Y_aug = torch.cat([optimal_outputs, Y], dim=0)
        pareto_mask = is_non_dominated(weight * Y_aug)
        optimal_inputs = X_aug[pareto_mask]
        optimal_outputs = Y_aug[pareto_mask]
        num_found = len(optimal_inputs)
        ratio = ceil(num_points / num_found)
        num_tries = num_tries + 1
    # If maximum number of retries exceeded throw out a runtime error.
    if ratio > 1:
        error_text = f"Only found {num_found} optimal points instead of {num_points}."
        raise RuntimeError(error_text)
    else:
        return optimal_inputs[:num_points], optimal_outputs[:num_points]


def sample_optimal_points(
    model: Model,
    bounds: Tensor,
    num_samples: int,
    num_points: int,
    optimizer: Callable[
        [GenericDeterministicModel, Tensor, int, bool, Any], tuple[Tensor, Tensor]
    ] = random_search_optimizer,
    maximize: bool = True,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute a collection of optimal inputs and outputs from samples of a Gaussian
    Process (GP).

    Steps:
    (1) The samples are generated using random Fourier features (RFFs).
    (2) The samples are optimized sequentially using an optimizer.

    TODO: We can generalize the GP sampling step to accommodate for other sampling
        strategies rather than restricting to RFFs e.g. decoupled sampling.

    TODO: Currently this defaults to random search optimization, might want to
        explore some other alternatives.

    Args:
        model: The model. This does not support models which include fantasy
            observations.
        bounds: A ``2 x d``-dim Tensor containing the input bounds.
        num_samples: The number of GP samples.
        num_points: The number of optimal points to be outputted.
        optimizer: A callable that solves the deterministic optimization problem.
        maximize: If true, we consider a maximization problem.
        optimizer_kwargs: The additional arguments for the optimizer.

    Returns:
        A two-element tuple containing

        - A ``num_samples x num_points x d``-dim Tensor containing the collection of
            optimal inputs.
        - A ``num_samples x num_points x M``-dim Tensor containing the collection of
            optimal objectives.
    """
    tkwargs: dict[str, Any] = {"dtype": bounds.dtype, "device": bounds.device}
    M = model.num_outputs
    d = bounds.shape[-1]
    if M == 1:
        if num_points > 1:
            raise UnsupportedError(
                "For single-objective optimization `num_points` should be 1."
            )
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    pareto_sets = torch.zeros((num_samples, num_points, d), **tkwargs)
    pareto_fronts = torch.zeros((num_samples, num_points, M), **tkwargs)
    for i in range(num_samples):
        sample_i = get_matheron_path_model(model=model)
        ps_i, pf_i = optimizer(
            model=sample_i,
            bounds=bounds,
            num_points=num_points,
            maximize=maximize,
            **optimizer_kwargs,
        )
        pareto_sets[i, ...] = ps_i
        pareto_fronts[i, ...] = pf_i

    return pareto_sets, pareto_fronts
