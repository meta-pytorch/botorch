#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-Fidelity Acquisition Functions for Multi-objective Bayesian optimization.

References

.. [Irshad2021MOMF]
    F. Irshad, S. Karsch, and A. Döpp. Expected hypervolume improvement for
    simultaneous multi-objective and multi-fidelity optimization.
    arXiv preprint arXiv:2112.13901, 2021.

"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class MOMF(qExpectedHypervolumeImprovement):
    def __init__(
        self,
        model: Model,
        ref_point: list[float] | Tensor,
        partitioning: NondominatedPartitioning,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        X_pending: Tensor | None = None,
        cost_call: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""MOMF acquisition function supporting m>=2 outcomes.
        The model needs to have train_obj that has a fidelity
        objective appended to its end.
        In the following example we consider a 2-D output space
        but the ref_point is 3D because of fidelity objective.

        See [Irshad2021MOMF]_ for details.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0, 0.0]
            >>> cost_func = lambda X: 5 + X[..., -1]
            >>> momf = MOMF(model, ref_point, partitioning, cost_func)
            >>> momf_val = momf(test_X)

        Args:
            model: A fitted model. There are two default assumptions in the training
                data. ``train_X`` should have fidelity parameter ``s`` as the last
                dimension of the input and ``train_Y`` contains a trust objective as
                its last dimension.
            ref_point: A list or tensor with ``m+1`` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                The '+1' takes care of the trust objective appended to ``train_Y``.
                This is a reference point for the objective values (i.e. after
                applying``objective`` to the samples).
            partitioning: A ``NondominatedPartitioning`` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using ``get_sampler``.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to ``IdentityMCMultiOutputObjective()``.
            constraints: A list of callables, each mapping a Tensor of dimension
                ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                ``sample_shape x batch-shape x q``, where negative values imply
                feasibility. The acquisition function will compute expected feasible
                hypervolume.
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design
                points that have points that have been submitted for function
                evaluation but have not yet been evaluated. Concatenated into ``X``
                upon forward call. Copied and set to have no gradient.
            cost_call: A callable cost function mapping a Tensor of dimension
                ``batch_shape x q x d`` to a cost Tensor of dimension
                ``batch_shape x q x m``. Defaults to an AffineCostModel with
                ``C(s) = 1 + s``.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same eta is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                eta value.
        """

        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )

        if cost_call is None:
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 1.0}, fixed_cost=1.0
            )
        else:
            cost_model = GenericDeterministicModel(cost_call)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        self.cost_aware_utility = cost_aware_utility

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        hv_gain = self._compute_qehvi(samples=samples, X=X)
        cost_weighted_qehvi = self.cost_aware_utility(X=X, deltas=hv_gain)
        return cost_weighted_qehvi
