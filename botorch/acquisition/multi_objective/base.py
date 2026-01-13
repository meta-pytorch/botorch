#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Base classes for multi-objective acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.base import MCSampler
from torch import Tensor


class MultiObjectiveAnalyticAcquisitionFunction(AcquisitionFunction):
    r"""Abstract base class for Multi-Objective batch acquisition functions."""

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Constructor for the MultiObjectiveAnalyticAcquisitionFunction base class.

        Args:
            model: A fitted model.
            posterior_transform: A PosteriorTransform (optional).
        """
        super().__init__(model=model)
        if posterior_transform is None or isinstance(
            posterior_transform, PosteriorTransform
        ):
            self.posterior_transform = posterior_transform
        else:
            raise UnsupportedError(
                "Only a posterior_transform of type PosteriorTransform is "
                "supported for Multi-Objective analytic acquisition functions."
            )

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a ``batch_shape x 1 x d`` X Tensor of t-batches with ``1``
        ``d``-dim design point each, and returns a Tensor with shape
        ``batch_shape'``, where ``batch_shape'`` is the broadcasted batch shape of
        model and input ``X``.
        """
        pass  # pragma: no cover

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


class MultiObjectiveMCAcquisitionFunction(AcquisitionFunction, MCSamplerMixin, ABC):
    r"""Abstract base class for Multi-Objective batch acquisition functions.

    NOTE: This does not inherit from ``MCAcquisitionFunction`` to avoid circular
    imports.

    Args:
        _default_sample_shape: The ``sample_shape`` for the default sampler.
    """

    _default_sample_shape = torch.Size([128])

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""Constructor for the ``MultiObjectiveMCAcquisitionFunction`` base class.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using ``get_sampler``.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See ``ForkedRNGSampler`` and ``StochasticSampler`` as examples.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to ``IdentityMCMultiOutputObjective()``.
            constraints: A list of callables, each mapping a Tensor of dimension
                ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                ``sample_shape x batch-shape x q``, where negative values imply
                feasibility.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same eta is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                eta value.
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        if objective is None:
            objective = IdentityMCMultiOutputObjective()
        elif not isinstance(objective, MCMultiOutputObjective):
            raise UnsupportedError(
                "Only objectives of type MCMultiOutputObjective are supported for "
                "Multi-Objective MC acquisition functions."
            )
        if (
            hasattr(model, "input_transform")
            and isinstance(model.input_transform, InputPerturbation)
            and constraints is not None
        ):
            raise UnsupportedError(
                "Constraints are not supported with input perturbations, due to"
                "sample q-batch shape being different than that of the inputs."
                "Use a composite objective that applies feasibility weighting to"
                "samples before calculating the risk measure."
            )
        self.add_module("objective", objective)
        self.constraints = constraints
        if constraints:
            if type(eta) is not Tensor:
                eta = torch.full((len(constraints),), eta)
            self.register_buffer("eta", eta)
        self.X_pending = None
        if X_pending is not None:
            self.set_X_pending(X_pending)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a ``batch_shape x q x d`` X Tensor of t-batches with ``q``
        ``d``-dim design points each, and returns a Tensor with shape
        ``batch_shape'``, where ``batch_shape'`` is the broadcasted batch shape of
        model and input ``X``. Should utilize the result of ``set_X_pending`` as
        needed to account for pending function evaluations.
        """
        pass  # pragma: no cover
