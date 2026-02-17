#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for predictive entropy search (PES). The code utilizes the
implementation designed for the multi-objective batch setting.

NOTE: The PES acquisition might not be differentiable. As a result, we recommend
optimizing the acquisition function using finite differences.

"""

from __future__ import annotations

from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class qPredictiveEntropySearch(qMultiObjectivePredictiveEntropySearch):
    r"""The acquisition function for Predictive Entropy Search.

    This acquisition function approximates the mutual information between the
    observation at a candidate point ``X`` and the optimal set of inputs using
    expectation propagation (EP).

    NOTES:
    (i) The expectation propagation procedure can potentially fail due to the unstable
    EP updates. This is however unlikely to happen in the single-objective setting
    because we have much fewer EP factors. The jitter added in the training phase
    (``ep_jitter``) and testing phase (``test_jitter``) can be increased to prevent
    these failures from happening. More details in the description of
    ``qMultiObjectivePredictiveEntropySearch``.

    (ii) The estimated acquisition value could be negative.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        max_ep_iterations: int = 250,
        ep_jitter: float = 1e-4,
        test_jitter: float = 1e-4,
        threshold: float = 1e-2,
    ) -> None:
        r"""Predictive entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A ``num_samples x d``-dim tensor containing the sampled
                optimal inputs of dimension ``d``. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            maximize: If true, we consider a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            max_ep_iterations: The maximum number of expectation propagation
                iterations. (The minimum number of iterations is set at 3.)
            ep_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update during the training
                phase.
            test_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update in the testing
                phase.
            threshold: The convergence threshold for expectation propagation. This
                assesses the relative change in the mean and covariance. We default
                to one percent change i.e. ``threshold = 1e-2``.
        """
        super().__init__(
            model=model,
            pareto_sets=optimal_inputs.unsqueeze(-2),
            maximize=maximize,
            X_pending=X_pending,
            max_ep_iterations=max_ep_iterations,
            ep_jitter=ep_jitter,
            test_jitter=test_jitter,
            threshold=threshold,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qPredictiveEntropySearch on the candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of Predictive Entropy Search values at the
            given design points ``X``.
        """
        return self._compute_information_gain(X)
