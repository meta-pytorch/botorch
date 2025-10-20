#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.

Contributor: hvarfner (bayesian_active_learning, scorebo)
"""

from __future__ import annotations

from typing import Any, Hashable, List, Optional, Tuple

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
    DiscretizedProbabilityOfImprovement,
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
    r"""Construct kwargs for the acquisition functions requiring `best_f`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for `best_f`.
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


@acqf_input_constructor(
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
)
def construct_inputs_BAL(
    model: Model,
    X_pending: Optional[Tensor] = None,
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
    X_pending: Optional[Tensor] = None,
):
    inputs = {
        "model": model,
        "distance_metric": distance_metric,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(qSelfCorrectingBayesianOptimization)
def construct_inputs_SCoreBO(
    model: Model,
    bounds: List[Tuple[float, float]],
    num_optima: int = 8,
    posterior_transform: Optional[ScalarizedPosteriorTransform] = None,
    distance_metric: str = "hellinger",
    X_pending: Optional[Tensor] = None,
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
