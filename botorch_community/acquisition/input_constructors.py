#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.

Contributors:
    hvarfner (bayesian_active_learning, scorebo)
    SaiAakash (rei)
    sahilKirangi (113878) (rei input constructor)
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import torch
from botorch.acquisition.input_constructors import (
    acqf_input_constructor,
    get_best_f_analytic,
)
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import get_optimal_samples
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch_community.acquisition.discretized import (
    DiscretizedExpectedImprovement,
    DiscretizedNoisyExpectedImprovement,
    DiscretizedProbabilityOfImprovement,
)
from botorch_community.acquisition.rei import (
    LogRegionalExpectedImprovement,
    qLogRegionalExpectedImprovement,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization
from torch import Tensor


@acqf_input_constructor(
    DiscretizedExpectedImprovement, DiscretizedProbabilityOfImprovement
)
def construct_inputs_best_f(
    model: Model,
    training_data: SupervisedDataset | dict[Hashable, SupervisedDataset],
    posterior_transform: PosteriorTransform | None = None,
    best_f: float | Tensor | None = None,
) -> dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring ``best_f``.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for ``best_f``.
        best_f: Threshold above (or below) which improvement is defined.
        posterior_transform: The posterior transform to be used in the
            acquisition function.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
            posterior_transform=posterior_transform,
        )

    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "best_f": best_f,
    }


@acqf_input_constructor(DiscretizedNoisyExpectedImprovement)
def construct_inputs_noisy(
    model: Model,
    posterior_transform: PosteriorTransform | None = None,
    X_pending: Tensor | None = None,
) -> dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring ``best_f``.

    Args:
        model: The model to be used in the acquisition function.
        best_f: Threshold above (or below) which improvement is defined.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: Points already tried, but not yet included in the
            training data.


    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
    }


@acqf_input_constructor(
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
)
def construct_inputs_BAL(
    model: Model,
    X_pending: Tensor | None = None,
):
    inputs = {
        "model": model,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(qStatisticalDistanceActiveLearning)
def construct_inputs_SAL(
    model: Model,
    distance_metric: str = "hellinger",
    X_pending: Tensor | None = None,
):
    inputs = {
        "model": model,
        "distance_metric": distance_metric,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(
    LogRegionalExpectedImprovement,
    qLogRegionalExpectedImprovement,
)
def construct_inputs_rei(
    model: Model,
    training_data: SupervisedDataset | dict[Hashable, SupervisedDataset],
    X_dev: Tensor,
    posterior_transform: PosteriorTransform | None = None,
    best_f: float | Tensor | None = None,
    length: float = 0.8,
    bounds: Tensor | None = None,
) -> dict[str, Any]:
    r"""Construct kwargs for Regional Expected Improvement.

    Supports both the analytic ``LogRegionalExpectedImprovement`` and the
    Monte Carlo ``qLogRegionalExpectedImprovement``.  The returned dict
    covers the shared constructor arguments; MC-specific parameters
    (e.g. ``sampler``, ``X_pending``) can be added by the caller.

    Args:
        model: A fitted single-outcome model.
        training_data: Dataset(s) used to train the model.
            Used to infer ``best_f`` when it is not provided explicitly.
        X_dev: A ``n x d``-dim Tensor of ``n`` pre-drawn sample points in
            ``[0, 1]^d`` representing relative positions within a trust
            region.  Larger ``n`` gives a more accurate region average at
            the cost of extra model evaluations.
        posterior_transform: A ``PosteriorTransform``.  Required when using
            a multi-output model so the posterior can be collapsed to a
            single scalar output before computing improvement.
        best_f: The incumbent (best observed) function value.  Improvement
            is measured relative to this threshold.  Defaults to the
            maximum observed ``Y`` in ``training_data``.
        length: Side length of the cubic trust region centred on the
            candidate point ``X``.  Must be in ``(0, 1]`` when the design
            space is normalised to ``[0, 1]^d``.  Defaults to ``0.8``.
        bounds: A ``2 x d``-dim Tensor of lower (row 0) and upper (row 1)
            bounds for each input dimension.  Used to clamp the trust
            region so it never extends outside the feasible space.
            Defaults to the unit hypercube ``[0, 1]^d``.

    Returns:
        A dict mapping kwarg names of the acquisition function constructor
        to their values.
    """
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
            posterior_transform=posterior_transform,
        )
    return {
        "model": model,
        "best_f": best_f,
        "X_dev": X_dev,
        "posterior_transform": posterior_transform,
        "length": length,
        "bounds": bounds,
    }


@acqf_input_constructor(qSelfCorrectingBayesianOptimization)
def construct_inputs_SCoreBO(
    model: Model,
    bounds: list[tuple[float, float]],
    num_optima: int = 8,
    posterior_transform: ScalarizedPosteriorTransform | None = None,
    distance_metric: str = "hellinger",
    X_pending: Tensor | None = None,
):
    dtype = model.train_targets.dtype
    # the number of optima are per model
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=model,
        bounds=torch.as_tensor(bounds, dtype=dtype).T,
        num_optima=num_optima,
        posterior_transform=posterior_transform,
        return_transformed=True,
    )
    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "distance_metric": distance_metric,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
    }
    return inputs
