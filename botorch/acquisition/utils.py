#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for acquisition functions.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    UnsupportedError,
)
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.fully_bayesian import MCMC_DIM
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.objective import (
    compute_feasibility_indicator,
    compute_smoothed_feasibility_indicator,
)
from botorch.utils.sampling import optimize_posterior_samples
from botorch.utils.transforms import is_ensemble, normalize_indices
from gpytorch.models import GP
from pyre_extensions import none_throws
from torch import Tensor


def get_acquisition_function(*args, **kwargs) -> None:
    raise DeprecationError(
        "`get_acquisition_function` has been moved to `botorch.acquisition.factory`."
    )


def repeat_to_match_aug_dim(target_tensor: Tensor, reference_tensor: Tensor) -> Tensor:
    """Repeat target_tensor until it has the same first dimension as reference_tensor
    This works regardless of the batch shapes and q.
    This is useful as we sometimes modify sample shapes such as in LearnedObjective.

    Args:
        target_tensor: A `sample_size x batch_shape x q x m`-dim Tensor
        reference_tensor: A `(augmented_sample * sample_size) x batch_shape x q`-dim
            Tensor. `augmented_sample` could be 1.

    Returns:
        The content of `target_tensor` potentially repeated so that its first dimension
        matches that of `reference_tensor`.
        The shape will be `(augmented_sample * sample_size) x batch_shape x q x m`.

    Examples:
        >>> import torch
        >>> target_tensor = torch.arange(3).repeat(2, 1).T
        >>> target_tensor
        tensor([[0, 0],
                [1, 1],
                [2, 2]])
        >>> repeat_to_match_aug_dim(target_tensor, torch.zeros(6))
        tensor([[0, 0],
                [1, 1],
                [2, 2],
                [0, 0],
                [1, 1],
                [2, 2]])
    """
    augmented_sample_num, remainder = divmod(
        reference_tensor.shape[0], target_tensor.shape[0]
    )
    if remainder != 0:
        raise ValueError(
            "The first dimension of reference_tensor must "
            "be a multiple of target_tensor's."
        )

    # using repeat here as obj might be constructed as
    # obj.reshape(-1, *samples.shape[2:]) where the first 2 dimensions are
    # of shape `augmented_samples x sample_shape`.
    repeat_size = (augmented_sample_num,) + (1,) * (target_tensor.ndim - 1)
    return target_tensor.repeat(*repeat_size)


def compute_best_feasible_objective(
    samples: Tensor,
    obj: Tensor,
    constraints: list[Callable[[Tensor], Tensor]] | None,
    model: Model | None = None,
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    X_baseline: Tensor | None = None,
    infeasible_obj: Tensor | None = None,
) -> Tensor:
    """Computes the largest `obj` value that is feasible under the `constraints`. If
    `constraints` is None, returns the best unconstrained objective value.

    When no feasible observations exist and `infeasible_obj` is not `None`, returns
    `infeasible_obj` (potentially reshaped). When no feasible observations exist and
    `infeasible_obj` is `None`, uses `model`, `objective`, `posterior_transform`, and
    `X_baseline` to infer and return an `infeasible_obj` `M` s.t. `M < min_x f(x)`.

    Args:
        samples: `(sample_shape) x batch_shape x q x m`-dim posterior samples.
        obj: A `(sample_shape) x batch_shape x q`-dim Tensor of MC objective values.
        constraints: A list of constraint callables which map posterior samples to
            a scalar. The associated constraint is considered satisfied if this
            scalar is less than zero.
        model: A Model, only required when there are no feasible observations.
        objective: An MCAcquisitionObjective, only optionally used when there are no
            feasible observations.
        posterior_transform: A PosteriorTransform, only optionally used when there are
            no feasible observations.
        X_baseline: A `batch_shape x d`-dim Tensor of baseline points, only required
            when there are no feasible observations.
        infeasible_obj: A Tensor to be returned when no feasible points exist.

    Returns:
        A `(sample_shape) x batch_shape`-dim Tensor of best feasible objectives.
    """
    if constraints is None:  # unconstrained case
        # we don't need to differentiate through X_baseline for now, so taking
        # the regular max over the n points to get best_f is fine
        with torch.no_grad():
            return obj.amax(dim=-1, keepdim=False)

    is_feasible = compute_feasibility_indicator(
        constraints=constraints, samples=samples
    )  # sample_shape x batch_shape x q

    if is_feasible.any(dim=-1).all():
        infeasible_value = -torch.inf

    elif infeasible_obj is not None:
        infeasible_value = infeasible_obj.item()

    else:
        if model is None:
            raise ValueError(
                "Must specify `model` when no feasible observation exists."
            )
        if X_baseline is None:
            raise ValueError(
                "Must specify `X_baseline` when no feasible observation exists."
            )
        warnings.warn(
            "When all training points are infeasible, it is better to use "
            "q(Log)ProbabilityOfFeasibility.",
            BotorchWarning,
            stacklevel=2,
        )

        infeasible_value = _estimate_objective_lower_bound(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            X=X_baseline,
        ).item()

    is_feasible = repeat_to_match_aug_dim(
        target_tensor=is_feasible, reference_tensor=obj
    )
    obj = torch.where(is_feasible, obj, infeasible_value)
    with torch.no_grad():
        return obj.amax(dim=-1, keepdim=False)


def _estimate_objective_lower_bound(
    model: Model,
    objective: MCAcquisitionObjective | None,
    posterior_transform: PosteriorTransform | None,
    X: Tensor,
) -> Tensor:
    """Estimates a lower bound on the objective values by evaluating the at uniformly
    random points in the bounding box of `X`, returning the 6-sigma lower bound of the
    computed statistics.

    Args:
        model: A fitted model.
        objective: An MCAcquisitionObjective with `m` outputs.
        posterior_transform: A PosteriorTransform.
        X: A `n x d`-dim Tensor of design points from which to draw convex combinations.

    Returns:
        A `m`-dimensional Tensor of lower bounds of the objectives.
    """
    # we do not have access to `bounds` here, so we infer the bounding box
    # from data, expanding by 10% in each direction
    X_lb = X.min(dim=-2, keepdim=True).values
    X_ub = X.max(dim=-2, keepdim=True).values
    X_range = X_ub - X_lb
    X_padding = 0.1 * X_range
    uniform_samples = torch.rand(
        *X.shape[:-2], 32, X.shape[-1], dtype=X.dtype, device=X.device
    )
    X_samples = X_lb - X_padding + uniform_samples * (X_range + 2 * X_padding)
    # infeasible cost M is such that -M < min_x f(x), thus
    # 0 < min_x f(x) - (-M), so we should take -M as a lower
    # bound on the best feasible objective
    return -get_infeasible_cost(
        X=X_samples,
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
    )


def get_infeasible_cost(
    X: Tensor,
    model: Model,
    objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    posterior_transform: PosteriorTransform | None = None,
) -> Tensor:
    r"""Get infeasible cost for a model and objective.

    For each outcome, computes an infeasible cost `M` such that
    `-M < min_x f(x)` almost always, so that feasible points are preferred.

    Args:
        X: A `n x d` Tensor of `n` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted botorch model with `m` outcomes.
        objective: The objective with which to evaluate the model output.
        posterior_transform: A PosteriorTransform (optional).

    Returns:
        An `m`-dim tensor of infeasible cost values.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> objective = lambda Y: Y[..., -1] ** 2
        >>> M = get_infeasible_cost(train_X, model, obj)
    """
    if objective is None:

        def objective(Y: Tensor, X: Tensor | None = None):
            return Y.squeeze(-1)

    posterior = model.posterior(X, posterior_transform=posterior_transform)
    # We check both the upper and lower bound of the posterior, since the objective
    # may be increasing or decreasing. For objectives that are neither (eg. absolute
    # distance from a target), this should still provide a good bound.
    six_stdv = 6 * posterior.variance.clamp_min(0).sqrt()
    lb = torch.stack(
        [
            objective(posterior.mean - six_stdv, X=X),
            objective(posterior.mean + six_stdv, X=X),
        ],
        dim=0,
    )
    lb = lb.min(dim=0).values

    if lb.ndim < posterior.mean.ndim:
        lb = lb.unsqueeze(-1)
    # Take outcome-wise min. Looping in to handle batched models.
    while lb.dim() > 1:
        lb = lb.min(dim=-2).values
    return -(lb.clamp_max(0.0))


def _prune_inferior_shared_processing(
    model: Model,
    X: Tensor,
    is_moo: bool,
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: MCSampler | None = None,
    marginalize_dim: int | None = None,
) -> tuple[int, Tensor, Tensor]:
    r"""Shared data processing for `prune_inferior_points` and
    `prune_inferior_points_multi_objective`.

    Returns:
        - max_points: The maximum number of points to keep.
        - obj_vals: The objective values of the points in `X`.
        - infeas: A boolean tensor indicating feasibility of `X`.
    """
    func_name = (
        "prune_inferior_points_multi_objective" if is_moo else "prune_inferior_points"
    )
    if marginalize_dim is None and is_ensemble(model):
        marginalize_dim = MCMC_DIM

    if X.ndim > 2:
        raise UnsupportedError(
            f"Batched inputs `X` are currently unsupported by `{func_name}`"
        )
    if X.size(-2) == 0:
        raise ValueError("X must have at least one point.")
    if max_frac <= 0 or max_frac > 1.0:
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    max_points = math.ceil(max_frac * X.size(-2))
    with torch.no_grad():
        posterior = model.posterior(X=X, posterior_transform=posterior_transform)
    if sampler is None:
        sampler = get_sampler(
            posterior=posterior, sample_shape=torch.Size([num_samples])
        )
    samples = sampler(posterior)
    if objective is not None:
        obj_vals = objective(samples=samples, X=X)
    elif is_moo:
        obj_vals = samples
    else:
        obj_vals = samples.squeeze(-1)
    if obj_vals.ndim > (2 + is_moo):
        if obj_vals.ndim == (3 + is_moo) and marginalize_dim is not None:
            if marginalize_dim < 0:
                # Update `marginalize_dim` to be positive while accounting for
                # removal of output dimension in SOO.
                marginalize_dim = (not is_moo) + none_throws(
                    normalize_indices([marginalize_dim], d=obj_vals.ndim)
                )[0]
            obj_vals = obj_vals.mean(dim=marginalize_dim)
        else:
            raise UnsupportedError(
                "Models with multiple batch dims are currently unsupported by "
                f"`{func_name}`."
            )
    infeas = ~compute_feasibility_indicator(
        constraints=constraints,
        samples=samples,
        marginalize_dim=marginalize_dim,
    )

    return max_points, obj_vals, infeas


def prune_inferior_points(
    model: Model,
    X: Tensor,
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: MCSampler | None = None,
    marginalize_dim: int | None = None,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be the best point.

    Given a model, an objective, and an input tensor `X`, this function returns
    the subset of points in `X` that have some probability of being the best
    point under the objective. This function uses sampling to estimate the
    probabilities, the higher the number of points `n` in `X` the higher the
    number of samples `num_samples` should be to obtain accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape `n x d`. Batched inputs are currently not
            supported.
        objective: The objective under which to evaluate the posterior.
        posterior_transform: A PosteriorTransform (optional).
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are satisfied if `constraint(samples) < 0`.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.
        sampler: If provided, will use this customized sampler instead of
            automatically constructing one with `num_samples`.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being the best point.
    """
    max_points, obj_vals, infeas = _prune_inferior_shared_processing(
        model=model,
        X=X,
        is_moo=False,
        objective=objective,
        posterior_transform=posterior_transform,
        constraints=constraints,
        num_samples=num_samples,
        max_frac=max_frac,
        sampler=sampler,
        marginalize_dim=marginalize_dim,
    )
    if infeas.all():
        # if no points are feasible, keep the point closest to being feasible
        with torch.no_grad():
            posterior = model.posterior(X=X, posterior_transform=posterior_transform)
        if sampler is None:
            sampler = get_sampler(
                posterior=posterior, sample_shape=torch.Size([num_samples])
            )
        samples = sampler(posterior)
        # use the probability of feasibility as the objective for computing best points
        obj_vals = compute_smoothed_feasibility_indicator(
            constraints=constraints,
            samples=samples,
            eta=1e-3,
            log=True,
        )

    elif infeas.any():
        # set infeasible points to worse than worst objective across all samples
        # Use clone() here to avoid deprecated `index_put_` on an expanded tensor
        obj_vals = obj_vals.clone()
        obj_vals[infeas] = obj_vals.min() - 1

    is_best = torch.argmax(obj_vals, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)

    if len(idcs) > max_points:
        counts, order_idcs = torch.sort(counts, stable=True, descending=True)
        idcs = order_idcs[:max_points]

    return X[idcs]


def project_to_target_fidelity(
    X: Tensor,
    target_fidelities: dict[int, float] | None = None,
    d: int | None = None,
) -> Tensor:
    r"""Project `X` onto the target set of fidelities.

    This function assumes that the set of feasible fidelities is a box, so
    projecting here just means setting each fidelity parameter to its target
    value. If X does not contain the fidelity dimensions, this will insert
    them and set them to their target values.

    Args:
        X: A `batch_shape x q x (d or d-d_f)`-dim Tensor of with `q` `d` or
            `d-d_f`-dim design points for each t-batch, where d_f is the
            number of fidelity dimensions. If the argument `d` is not provided,
            `X` must include the fidelity dimensions and have a trailing`X` must
            include the fidelity dimensions and have a trailing
        target_fidelities: A dictionary mapping a subset of columns of `X` (the
            fidelity parameters) to their respective target fidelity value. If
            omitted, assumes that the last column of X is the fidelity parameter
            with a target value of 1.0.
        d: The total dimension `d`.

    Return:
        A `batch_shape x q x d`-dim Tensor `X_proj` with fidelity parameters
            projected to the provided fidelity values.
    """
    if target_fidelities is None:
        target_fidelities = {-1: 1.0}
    if d is None:
        # assume X contains the fidelity dimensions
        d = X.shape[-1]
    # normalize to positive indices
    tfs = {k if k >= 0 else d + k: v for k, v in target_fidelities.items()}
    ones = torch.ones(*X.shape[:-1], device=X.device, dtype=X.dtype)
    if X.shape[-1] == d:
        # X contains fidelity dimensions
        # here we're looping through the feature dimension of X - this could be
        # slow for large `d`, we should optimize this for that case
        X_proj = torch.stack(
            [X[..., i] if i not in tfs else tfs[i] * ones for i in range(d)], dim=-1
        )
    elif X.shape[-1] == d - len(target_fidelities):
        # need to insert fidelity dimensions
        cols = []
        X_idx = 0
        for i in range(d):
            if i not in tfs:
                cols.append(X[..., X_idx])
                X_idx += 1
            else:
                cols.append(tfs[i] * ones)
        X_proj = torch.stack(cols, dim=-1)
    else:
        raise BotorchTensorDimensionError(
            "X must have a last dimension with size `d` or `d-d_f`,"
            f" but got {X.shape[-1]}."
        )

    return X_proj


def expand_trace_observations(
    X: Tensor, fidelity_dims: list[int] | None = None, num_trace_obs: int = 0
) -> Tensor:
    r"""Expand `X` with trace observations.

    Expand a tensor of inputs with "trace observations" that are obtained during
    the evaluation of the candidate set. This is used in multi-fidelity
    optimization. It can be though of as augmenting the `q`-batch with additional
    points that are the expected trace observations.

    Let `f_i` be the `i`-th fidelity parameter. Then this functions assumes that
    for each element of the q-batch, besides the fidelity `f_i`, we will observe
    additonal fidelities `f_i1, ..., f_iK`, where `K = num_trace_obs`, during
    evaluation of the candidate set `X`. Specifically, this function assumes
    that `f_ij = (K-j) / (num_trace_obs + 1) * f_i` for all `i`. That is, the
    expansion is performed in parallel for all fidelities (it does not expand
    out all possible combinations).

    Args:
        X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design points
            (incl. the fidelity parameters) for each t-batch.
        fidelity_dims: The indices of the fidelity parameters. If omitted,
            assumes that the last column of X contains the fidelity parameters.
        num_trace_obs: The number of trace observations to use.

    Return:
        A `batch_shape x (q + num_trace_obs x q) x d` Tensor `X_expanded` that
            expands `X` with trace observations.
    """
    if num_trace_obs == 0:  # No need to expand if we don't use trace observations
        return X

    if fidelity_dims is None:
        fidelity_dims = [-1]

    # The general strategy in the following is to expand `X` to the desired
    # shape, and then multiply it (point-wise) with a tensor of scaling factors
    reps = [1] * (X.ndim - 2) + [1 + num_trace_obs, 1]
    X_expanded = X.repeat(*reps)  # batch_shape x (q + num_trace_obs x q) x d
    scale_fac = torch.ones_like(X_expanded)
    s_pad = 1 / (num_trace_obs + 1)
    # tensor of  num_trace_obs scaling factors equally space between 1-s_pad and s_pad
    sf = torch.linspace(1 - s_pad, s_pad, num_trace_obs, device=X.device, dtype=X.dtype)
    # repeat each element q times
    q = X.size(-2)
    sf = torch.repeat_interleave(sf, q)  # num_trace_obs * q
    # now expand this to num_trace_obs x q x num_fidelities
    sf = sf.unsqueeze(-1).expand(X_expanded.size(-2) - q, len(fidelity_dims))
    # change relevant entries of the scaling tensor
    scale_fac[..., q:, fidelity_dims] = sf
    return scale_fac * X_expanded


def project_to_sample_points(X: Tensor, sample_points: Tensor) -> Tensor:
    r"""Augment `X` with sample points at which to take weighted average.

    Args:
        X: A `batch_shape x 1 x d`-dim Tensor of with one d`-dim design points
            for each t-batch.
        sample_points: `p x d'`-dim Tensor (`d' < d`) of `d'`-dim sample points at
            which to compute the expectation. The `d'`-dims refer to the trailing
            columns of X.
    Returns:
        A `batch_shape x p x d` Tensor where the q-batch includes the `p` sample points.
    """
    batch_shape = X.shape[:-2]
    p, d_prime = sample_points.shape
    X_new = X.repeat(*(1 for _ in batch_shape), p, 1)  # batch_shape x p x d
    X_new[..., -d_prime:] = sample_points
    return X_new


def get_optimal_samples(
    model: GP,
    bounds: Tensor,
    num_optima: int,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    posterior_transform: ScalarizedPosteriorTransform | None = None,
    objective: MCAcquisitionObjective | None = None,
    return_transformed: bool = False,
    options: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Draws sample paths from the posterior and maximizes the samples using GD.

    Args:
        model: The model from which samples are drawn.
        bounds: Bounds of the search space. If the model inputs are
            normalized, the bounds should be normalized as well.
        num_optima: The number of paths to be drawn and optimized.
        raw_samples: The number of candidates randomly sample.
            Defaults to 1024.
        num_restarts: The number of candidates to do gradient-based
            optimization on. Defaults to 20.
        posterior_transform: A ScalarizedPosteriorTransform (may e.g. be used to
            scalarize multi-output models or negate the objective).
        objective: An MCAcquisitionObjective, used to negate the objective or otherwise
            transform sample outputs. Cannot be combined with `posterior_transform`.
        return_transformed: If True, return the transformed samples.
        options: Options for generation of initial candidates, passed to
            gen_batch_initial_conditions.
    Returns:
        The optimal input locations and corresponding outputs, x* and f*.

    """
    if posterior_transform and not isinstance(
        posterior_transform, ScalarizedPosteriorTransform
    ):
        raise ValueError(
            "Only the ScalarizedPosteriorTransform is supported for "
            "get_optimal_samples."
        )
    if posterior_transform and objective:
        raise ValueError(
            "Only one of `posterior_transform` and `objective` can be specified."
        )

    if posterior_transform:
        sample_transform = posterior_transform.evaluate
    elif objective:
        sample_transform = objective
    else:
        sample_transform = None

    paths = get_matheron_path_model(
        model=model,
        sample_shape=torch.Size([num_optima]),
        ensemble_as_batch=True,
    )
    suggested_points = prune_inferior_points(
        model=model,
        X=model.train_inputs[0],
        posterior_transform=posterior_transform,
        objective=objective,
    )
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=bounds,
        raw_samples=raw_samples,
        num_restarts=num_restarts,
        sample_transform=sample_transform,
        return_transformed=return_transformed,
        suggested_points=suggested_points,
        options=options,
    )
    return optimal_inputs, optimal_outputs
