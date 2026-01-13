#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Cost models to be used with multi-fidelity optimization.

Cost models are useful for defining known cost functions when the cost of an evaluation
is heterogeneous in fidelity. For a full worked example, see the
``tutorial <https://botorch.org/docs/tutorials/multi_fidelity_bo>``_ on continuous
multi-fidelity Bayesian Optimization.
"""

from __future__ import annotations

import torch
from botorch.models.deterministic import DeterministicModel
from torch import Tensor


class AffineFidelityCostModel(DeterministicModel):
    r"""Deterministic, affine cost model operating on fidelity parameters.

    For each (q-batch) element of a candidate set ``X``, this module computes a
    cost of the form

        cost = fixed_cost + sum_j weights[j] * X[fidelity_dims[j]]

    For a full worked example, see the
    ``tutorial <https://botorch.org/docs/tutorials/multi_fidelity_bo>``_ on continuous
    multi-fidelity Bayesian Optimization.

    Example:
        >>> from botorch.models import AffineFidelityCostModel
        >>> from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        >>> cost_model = AffineFidelityCostModel(
        >>>    fidelity_weights={6: 1.0}, fixed_cost=5.0
        >>> )
        >>> cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    """

    def __init__(
        self,
        fidelity_weights: dict[int, float] | None = None,
        fixed_cost: float = 0.01,
    ) -> None:
        r"""
        Args:
            fidelity_weights: A dictionary mapping a subset of columns of ``X``
                (the fidelity parameters) to its associated weight in the
                affine cost expression. If omitted, assumes that the last
                column of ``X`` is the fidelity parameter with a weight of 1.0.
            fixed_cost: The fixed cost of running a single candidate point (i.e.
                an element of a q-batch).
        """
        if fidelity_weights is None:
            fidelity_weights = {-1: 1.0}
        super().__init__()
        self.fidelity_dims = sorted(fidelity_weights)
        self.fixed_cost = fixed_cost
        weights = torch.tensor([fidelity_weights[i] for i in self.fidelity_dims])
        self.register_buffer("weights", weights)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost on a candidate set X.

        Computes a cost of the form

            cost = fixed_cost + sum_j weights[j] * X[fidelity_dims[j]]

        for each element of the q-batch

        Args:
            X: A ``batch_shape x q x d'``-dim tensor of candidate points.

        Returns:
            A ``batch_shape x q x 1``-dim tensor of costs.
        """
        # TODO: Consider different aggregation (i.e. max) across q-batch
        lin_cost = torch.einsum(
            "...f,f", X[..., self.fidelity_dims], self.weights.to(X)
        )
        return self.fixed_cost + lin_cost.unsqueeze(-1)


class FixedCostModel(DeterministicModel):
    r"""Deterministic, fixed cost model.

    For each (q-batch) element of a candidate set ``X``, this module computes a
    fixed cost per objective.
    """

    def __init__(
        self,
        fixed_cost: Tensor,
    ) -> None:
        r"""
        Args:
            fixed_cost: A ``m``-dim tensor containing the fixed cost of evaluating each
                objective.
        """
        super().__init__()
        self.register_buffer("fixed_cost", fixed_cost)
        self._num_outputs = fixed_cost.shape[-1]

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost on a candidate set X.

        Computes the fixed cost of evaluating each objective for each element
        of the q-batch.

        Args:
            X: A ``batch_shape x q x d'``-dim tensor of candidate points.

        Returns:
            A ``batch_shape x q x m``-dim tensor of costs.
        """
        view_shape = [1] * (X.ndim - 1) + [self._num_outputs]
        expand_shape = X.shape[:-1] + torch.Size([self._num_outputs])
        return self.fixed_cost.view(view_shape).expand(expand_shape)
