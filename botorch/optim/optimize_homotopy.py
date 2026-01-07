# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

from collections.abc import Callable

from typing import Any

import torch
from botorch.acquisition import AcquisitionFunction

from botorch.generation.gen import TGenCandidates
from botorch.optim.homotopy import Homotopy
from botorch.optim.initializers import TGenInitialConditions
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from torch import Tensor


def prune_candidates(
    candidates: Tensor, acq_values: Tensor, prune_tolerance: float
) -> Tensor:
    r"""Prune candidates based on their distance to other candidates.

    Args:
        candidates: An `n x d` tensor of candidates.
        acq_values: An `n` tensor of candidate values.
        prune_tolerance: The minimum distance to prune candidates.

    Returns:
        An `m x d` tensor of pruned candidates.
    """
    if candidates.ndim != 2:
        raise ValueError("`candidates` must be of size `n x d`.")
    if acq_values.ndim != 1 or len(acq_values) != candidates.shape[0]:
        raise ValueError("`acq_values` must be of size `n`.")
    if prune_tolerance < 0:
        raise ValueError("`prune_tolerance` must be >= 0.")
    sorted_inds = acq_values.argsort(descending=True)
    candidates = candidates[sorted_inds]

    candidates_new = candidates[:1, :]
    for i in range(1, candidates.shape[0]):
        if (
            torch.cdist(candidates[i : i + 1, :], candidates_new).min()
            > prune_tolerance
        ):
            candidates_new = torch.cat(
                [candidates_new, candidates[i : i + 1, :]], dim=-2
            )
    return candidates_new


def optimize_acqf_homotopy(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    homotopy: Homotopy,
    prune_tolerance: float = 1e-4,
    raw_samples: int | None = None,
    options: dict[str, bool | float | int | str] | None = None,
    final_options: dict[str, bool | float | int | str] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
    fixed_features: dict[int, float] | None = None,
    fixed_features_list: list[dict[int, float]] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    gen_candidates: TGenCandidates | None = None,
    *,
    ic_generator: TGenInitialConditions | None = None,
    timeout_sec: float | None = None,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`
            (if inequality_constraints is provided, these bounds can be -inf and
            +inf, respectively).
        q: The number of candidates.
        homotopy: Homotopy object that will make the necessary modifications to the
            problem when calling `step()`.
        prune_tolerance: The minimum distance to prune candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation in the initial step of the homotopy.
        final_options: Options for candidate generation in the final step of
            the homotopy.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example. When q=1, or when
            applying the same constraint to each candidate in the batch
            (intra-point constraint), `indices` should be a 1-d tensor.
            For inter-point constraints, in which the constraint is applied to the
            whole batch of candidates, `indices` must be a 2-d tensor, where
            in each row `indices[i] =(k_i, l_i)` the first index `k_i` corresponds
            to the `k_i`-th element of the `q`-batch and the second index `l_i`
            corresponds to the `l_i`-th feature of that element.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`. See the docstring of
            `make_scipy_linear_constraints` for an example.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver. You need to pass in `batch_initial_conditions` in this case.
            Using non-linear inequality constraints also requires that `batch_limit`
            is set to 1, which will be done automatically if not specified in
            `options`.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        fixed_features_list: A list of maps `{feature_index: value}`. The i-th
            item represents the fixed_feature for the i-th optimization. If
            `fixed_features_list` is provided, `optimize_acqf_mixed` is invoked.
            All indices (`feature_index`) should be non-negative.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        gen_candidates: A callable for generating candidates (and their associated
            acquisition values) given a tensor of initial conditions and an
            acquisition function. Other common inputs include lower and upper bounds
            and a dictionary of options, but refer to the documentation of specific
            generation functions (e.g gen_candidates_scipy and gen_candidates_torch)
            for method-specific inputs. Default: `gen_candidates_scipy`
        ic_generator: Function for generating initial conditions. Not needed when
            `batch_initial_conditions` are provided. Defaults to
            `gen_one_shot_kg_initial_conditions` for `qKnowledgeGradient` acquisition
            functions and `gen_batch_initial_conditions` otherwise. Must be specified
            for nonlinear inequality constraints.
        timeout_sec: Max amount of time optimization can run for.
        retry_on_optimization_warning: Whether to retry candidate generation with a new
            set of initial conditions when it fails with an `OptimizationWarning`.
        ic_gen_kwargs: Additional keyword arguments passed to function specified by
            `ic_generator`
    """
    if fixed_features and fixed_features_list:
        raise ValueError(
            "Either `fixed_feature` or `fixed_features_list` can be provided, not both."
        )

    if fixed_features:
        message = (
            "The `fixed_features` argument is deprecated, "
            "use `fixed_features_list` instead."
        )
        warnings.warn(
            message,
            DeprecationWarning,
            stacklevel=2,
        )

    shared_optimize_acqf_kwargs = {
        "num_restarts": num_restarts,
        "inequality_constraints": inequality_constraints,
        "equality_constraints": equality_constraints,
        "nonlinear_inequality_constraints": nonlinear_inequality_constraints,
        "return_best_only": False,  # False to make n_restarts persist through homotopy.
        "gen_candidates": gen_candidates,
        "ic_generator": ic_generator,
        "timeout_sec": timeout_sec,
        "retry_on_optimization_warning": retry_on_optimization_warning,
        **ic_gen_kwargs,
    }

    if fixed_features_list and len(fixed_features_list) > 1:
        optimization_fn = optimize_acqf_mixed
        fixed_features_kwargs = {"fixed_features_list": fixed_features_list}
    else:
        optimization_fn = optimize_acqf
        fixed_features_kwargs = {
            "fixed_features": fixed_features_list[0]
            if fixed_features_list
            else fixed_features
        }

    candidate_list, acq_value_list = [], []
    if q > 1:
        base_X_pending = acq_function.X_pending

    for _ in range(q):
        candidates = batch_initial_conditions
        q_raw_samples = raw_samples
        homotopy.restart()

        while not homotopy.should_stop:
            candidates, acq_values = optimization_fn(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                options=options,
                batch_initial_conditions=candidates,
                raw_samples=q_raw_samples,
                **fixed_features_kwargs,
                **shared_optimize_acqf_kwargs,
            )

            homotopy.step()

            # Set raw_samples to None such that pruned restarts are not repopulated
            # at each step in the homotopy.
            q_raw_samples = None

            # Prune candidates
            candidates = prune_candidates(
                candidates=candidates.squeeze(1),
                acq_values=acq_values,
                prune_tolerance=prune_tolerance,
            ).unsqueeze(1)

        # Optimize one more time with the final options
        candidates, acq_values = optimization_fn(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            options=final_options,
            raw_samples=q_raw_samples,
            batch_initial_conditions=candidates,
            **fixed_features_kwargs,
            **shared_optimize_acqf_kwargs,
        )

        # Post-process the candidates and grab the best candidate
        if post_processing_func is not None:
            candidates = post_processing_func(candidates)
            acq_values = acq_function(candidates)

        best = torch.argmax(acq_values.view(-1), dim=0)
        candidate, acq_value = candidates[best], acq_values[best]

        # Keep the new candidate and update the pending points
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        selected_candidates = torch.cat(candidate_list, dim=-2)

        if q > 1:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, selected_candidates], dim=-2)
                if base_X_pending is not None
                else selected_candidates
            )

    if q > 1:  # Reset acq_function to previous X_pending state
        acq_function.set_X_pending(base_X_pending)

    homotopy.reset()  # Reset the homotopy parameters

    return selected_candidates, torch.stack(acq_value_list)
