#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod

import torch
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.models.model import Model
from botorch.utils import apply_constraints
from botorch.utils.transforms import normalize_indices
from torch import Tensor


class MCMultiOutputObjective(MCAcquisitionObjective):
    r"""Abstract base class for MC multi-output objectives.

    Args:
        _is_mo: A boolean denoting whether the objectives are multi-output.
    """

    _is_mo: bool = True

    @abstractmethod
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the multi-output objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim Tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim Tensors of inputs.

        Returns:
            A ``sample_shape x batch_shape x q x m'``-dim Tensor of objective
            values with ``m'`` the output dimension. This assumes maximization
            in each output dimension).

        This method is usually not called directly, but via the objectives.

        Example:
            >>> # ``__call__`` method:
            >>> samples = sampler(posterior)
            >>> outcomes = multi_obj(samples)
        """
        pass  # pragma: no cover


class GenericMCMultiOutputObjective(GenericMCObjective, MCMultiOutputObjective):
    r"""Multi-output objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.
    """

    pass


class IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Trivial objective that returns the unaltered samples.

    Example:
        >>> identity_objective = IdentityMCMultiOutputObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def __init__(
        self, outcomes: list[int] | None = None, num_outcomes: int | None = None
    ) -> None:
        r"""Initialize Objective.

        Args:
            outcomes: A list of the ``m'`` indices that the weights should be
                applied to.
            num_outcomes: The total number of outcomes ``m``
        """
        super().__init__()
        if outcomes is not None:
            if len(outcomes) < 2:
                raise BotorchTensorDimensionError(
                    "Must specify at least two outcomes for MOO."
                )
            if any(i < 0 for i in outcomes):
                if num_outcomes is None:
                    raise BotorchError(
                        "num_outcomes is required if any outcomes are less than 0."
                    )
                outcomes = normalize_indices(outcomes, num_outcomes)
            self.register_buffer("outcomes", torch.tensor(outcomes, dtype=torch.long))

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if hasattr(self, "outcomes"):
            return samples.index_select(-1, self.outcomes.to(device=samples.device))
        return samples


class WeightedMCMultiOutputObjective(IdentityMCMultiOutputObjective):
    r"""Objective that reweights samples by given weights vector.

    Example:
        >>> weights = torch.tensor([1.0, -1.0])
        >>> weighted_objective = WeightedMCMultiOutputObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = weighted_objective(samples)
    """

    def __init__(
        self,
        weights: Tensor,
        outcomes: list[int] | None = None,
        num_outcomes: int | None = None,
    ) -> None:
        r"""Initialize Objective.

        Args:
            weights: ``m'``-dim tensor of outcome weights.
            outcomes: A list of the ``m'`` indices that the weights should be
                applied to.
            num_outcomes: the total number of outcomes ``m``
        """
        super().__init__(outcomes=outcomes, num_outcomes=num_outcomes)
        if weights.ndim != 1:
            raise BotorchTensorDimensionError(
                f"weights must be an 1-D tensor, but got {weights.shape}."
            )
        elif outcomes is not None and weights.shape[0] != len(outcomes):
            raise BotorchTensorDimensionError(
                "weights must contain the same number of elements as outcomes, "
                f"but got {weights.numel()} weights and {len(outcomes)} outcomes."
            )
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        samples = super().forward(samples=samples)
        return samples * self.weights.to(samples)


class FeasibilityWeightedMCMultiOutputObjective(MCMultiOutputObjective):
    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        constraint_idcs: list[int],
        objective: MCMultiOutputObjective | None = None,
    ) -> None:
        r"""Construct a feasibility-weighted objective.

        This applies feasibility weighting before calculating the objective value.
        Defaults to identity if no constraints or objective is present.

        NOTE: By passing in a single-output ``MCAcquisitionObjective`` as the
        ``objective``, this can be used as a single-output
        ``MCAcquisitionObjective`` as well.

        Args:
            model: A fitted Model.
            X_baseline: An ``n x d``-dim tensor of points already observed.
            constraint_idcs: The outcome indices of the constraints. Constraints are
                handled by weighting the samples according to a sigmoid approximation
                of feasibility. A positive constraint outcome implies feasibility.
            objective: An optional objective to apply after feasibility-weighting
                the samples.
        """
        super().__init__()
        num_outputs = model.num_outputs
        # Get the non-negative indices.
        constraint_idcs = [
            num_outputs + idx if idx < 0 else idx for idx in constraint_idcs
        ]
        if len(constraint_idcs) != len(set(constraint_idcs)):
            raise ValueError("Received duplicate entries for `constraint_idcs`.")
        # Extract the indices for objective outcomes.
        objective_idcs = [i for i in range(num_outputs) if i not in constraint_idcs]
        if len(constraint_idcs) > 0:
            # Import locally to avoid circular import.
            from botorch.acquisition.utils import get_infeasible_cost

            inf_cost = get_infeasible_cost(
                X=X_baseline, model=model, objective=lambda y, X: y
            )[objective_idcs]

            def apply_feasibility_weights(Y: Tensor, X: Tensor | None = None) -> Tensor:
                return apply_constraints(
                    obj=Y[..., objective_idcs],
                    constraints=[lambda Y, i=i: -Y[..., i] for i in constraint_idcs],
                    samples=Y,
                    # This ensures that the dtype/device is set properly.
                    infeasible_cost=inf_cost.to(Y),
                )

            self.apply_feasibility_weights = apply_feasibility_weights
        else:
            self.apply_feasibility_weights = lambda Y: Y
        if objective is None:
            self.objective = lambda Y, X: Y
        else:
            self.objective = objective
            self._verify_output_shape = objective._verify_output_shape

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return self.objective(self.apply_feasibility_weights(samples), X=X)
