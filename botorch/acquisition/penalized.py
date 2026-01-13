#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Modules to add regularization to acquisition functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import UnsupportedError
from torch import Tensor


class L2Penalty(torch.nn.Module):
    r"""L2 penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor):
        r"""Initializing L2 regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        regularization_term = (
            torch.linalg.norm((X - self.init_point), ord=2, dim=-1).max(dim=-1).values
            ** 2
        )
        return regularization_term


class L1Penalty(torch.nn.Module):
    r"""L1 penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor):
        r"""Initializing L1 regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        regularization_term = (
            torch.linalg.norm((X - self.init_point), ord=1, dim=-1).max(dim=-1).values
        )
        return regularization_term


class GaussianPenalty(torch.nn.Module):
    r"""Gaussian penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor, sigma: float):
        r"""Initializing Gaussian regularization.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
            sigma: The parameter used in gaussian function.
        """
        super().__init__()
        self.init_point = init_point
        self.sigma = sigma

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        sq_diff = torch.linalg.norm((X - self.init_point), ord=2, dim=-1) ** 2
        pdf = torch.exp(sq_diff / 2 / self.sigma**2)
        regularization_term = pdf.max(dim=-1).values
        return regularization_term


class GroupLassoPenalty(torch.nn.Module):
    r"""Group lasso penalty class to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, init_point: Tensor, groups: list[list[int]]):
        r"""Initializing Group-Lasso regularization.

        Args:
            init_point: The "1 x dim" reference point against which we want
                to regularize.
            groups: Groups of indices used in group lasso.
        """
        super().__init__()
        self.init_point = init_point
        self.groups = groups

    def forward(self, X: Tensor) -> Tensor:
        r"""
        X should be batch_shape x 1 x dim tensor. Evaluation for q-batch is not
        implemented yet.
        """
        if X.shape[-2] != 1:
            raise NotImplementedError(
                "group-lasso has not been implemented for q>1 yet."
            )

        regularization_term = group_lasso_regularizer(
            X=X.squeeze(-2) - self.init_point, groups=self.groups
        )
        return regularization_term


def narrow_gaussian(X: Tensor, a: Tensor) -> Tensor:
    return torch.exp(-0.5 * (X / a) ** 2)


def nnz_approx(X: Tensor, target_point: Tensor, a: Tensor) -> Tensor:
    r"""Differentiable relaxation of ||X - target_point||_0

    Args:
        X: An ``n x d`` tensor of inputs.
        target_point: A tensor of size ``n`` corresponding to the target point.
        a: A scalar tensor that controls the differentiable relaxation.
    """
    d = X.shape[-1]
    if d != target_point.shape[-1]:
        raise ValueError("X and target_point have different shapes.")
    return d - narrow_gaussian(X - target_point, a).sum(dim=-1, keepdim=True)


class L0Approximation(torch.nn.Module):
    r"""Differentiable relaxation of the L0 norm using a Gaussian basis function."""

    def __init__(self, target_point: Tensor, a: float = 1.0, **tkwargs: Any) -> None:
        r"""Initializing L0 penalty with differentiable relaxation.

        Args:
            target_point: A tensor corresponding to the target point.
            a: A hyperparameter that controls the differentiable relaxation.
        """
        super().__init__()
        self.target_point = target_point
        # hyperparameter to control the differentiable relaxation in L0 norm function.
        self.register_buffer("a", torch.tensor(a, **tkwargs))

    def __call__(self, X: Tensor) -> Tensor:
        return nnz_approx(X=X, target_point=self.target_point, a=self.a)


class L0PenaltyApprox(L0Approximation):
    r"""Differentiable relaxation of the L0 norm to be added to any arbitrary
    acquisition function to construct a PenalizedAcquisitionFunction."""

    def __init__(self, target_point: Tensor, a: float = 1.0, **tkwargs: Any) -> None:
        r"""Initializing L0 penalty with differentiable relaxation.

        Args:
            target_point: A tensor corresponding to the target point.
            a: A hyperparameter that controls the differentiable relaxation.
        """
        super().__init__(target_point=target_point, a=a, **tkwargs)

    def __call__(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.
        Returns:
            A tensor of size "batch_shape" representing the acqfn for each q-batch.
        """
        return super().__call__(X=X).squeeze(dim=-1).min(dim=-1).values


class PenalizedAcquisitionFunction(AcquisitionFunction):
    r"""Single-outcome acquisition function regularized by the given penalty.

    The usage is similar to:
        raw_acqf = NoisyExpectedImprovement(...)
        penalty = GroupLassoPenalty(...)
        acqf = PenalizedAcquisitionFunction(raw_acqf, penalty)
    """

    def __init__(
        self,
        raw_acqf: AcquisitionFunction,
        penalty_func: torch.nn.Module,
        regularization_parameter: float,
    ) -> None:
        r"""Initializing Group-Lasso regularization.

        Args:
            raw_acqf: The raw acquisition function that is going to be regularized.
            penalty_func: The regularization function.
            regularization_parameter: Regularization parameter used in optimization.
        """
        super().__init__(model=raw_acqf.model)
        self.raw_acqf = raw_acqf
        self.penalty_func = penalty_func
        self.regularization_parameter = regularization_parameter

    def forward(self, X: Tensor) -> Tensor:
        raw_value = self.raw_acqf(X=X)
        penalty_term = self.penalty_func(X)
        return raw_value - self.regularization_parameter * penalty_term

    @property
    def X_pending(self) -> Tensor | None:
        return self.raw_acqf.X_pending

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if not isinstance(self.raw_acqf, AnalyticAcquisitionFunction):
            self.raw_acqf.set_X_pending(X_pending=X_pending)
        else:
            raise UnsupportedError(
                "The raw acquisition function is Analytic and does not account "
                "for X_pending yet."
            )


def group_lasso_regularizer(X: Tensor, groups: list[list[int]]) -> Tensor:
    r"""Computes the group lasso regularization function for the given point.

    Args:
        X: A bxd tensor representing the points to evaluate the regularization at.
        groups: List of indices of different groups.

    Returns:
        Computed group lasso norm of at the given points.
    """
    return torch.sum(
        torch.stack(
            [
                math.sqrt(len(g)) * torch.linalg.norm(X[..., g], ord=2, dim=-1)
                for g in groups
            ],
            dim=-1,
        ),
        dim=-1,
    )


class L1PenaltyObjective(torch.nn.Module):
    r"""
    L1 penalty objective class. An instance of this class can be added to any
    arbitrary objective to construct a PenalizedMCObjective.
    """

    def __init__(self, init_point: Tensor):
        r"""Initializing L1 penalty objective.

        Args:
            init_point: The "1 x dim" reference point against which
                we want to regularize.
        """
        super().__init__()
        self.init_point = init_point

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.

        Returns:
            A "1 x batch_shape x q" tensor representing the penalty for each point.
            The first dimension corresponds to the dimension of MC samples.
        """
        return torch.linalg.norm((X - self.init_point), ord=1, dim=-1).unsqueeze(dim=0)


class PenalizedMCObjective(GenericMCObjective):
    r"""Penalized MC objective.

    Allows to construct a penalized MC-objective by adding a penalty term to
    the original objective.

        mc_acq(X) = objective(X) + penalty_objective(X)

    Note: PenalizedMCObjective allows adding penalty at the MCObjective level,
    different from the AcquisitionFunction level in PenalizedAcquisitionFunction.

    Example:
        >>> regularization_parameter = 0.01
        >>> init_point = torch.zeros(3) # assume data dim is 3
        >>> objective = lambda Y, X: torch.sqrt(Y).sum(dim=-1)
        >>> l1_penalty_objective = L1PenaltyObjective(init_point=init_point)
        >>> l1_penalized_objective = PenalizedMCObjective(
                objective, l1_penalty_objective, regularization_parameter
            )
        >>> samples = sampler(posterior)
                objective, l1_penalty_objective, regularization_parameter
    """

    def __init__(
        self,
        objective: Callable[[Tensor, Tensor | None], Tensor],
        penalty_objective: torch.nn.Module,
        regularization_parameter: float,
        expand_dim: int | None = None,
    ) -> None:
        r"""Penalized MC objective.

        Args:
            objective: A callable ``f(samples, X)`` mapping a
                ``sample_shape x batch-shape x q x m``-dim Tensor ``samples`` and
                an optional ``batch-shape x q x d``-dim Tensor ``X`` to a
                ``sample_shape x batch-shape x q``-dim Tensor of objective values.
            penalty_objective: A torch.nn.Module ``f(X)`` that takes in a
                ``batch-shape x q x d``-dim Tensor ``X`` and outputs a
                ``1 x batch-shape x q``-dim Tensor of penalty objective values.
            regularization_parameter: weight of the penalty (regularization) term
            expand_dim: dim to expand penalty_objective to match with objective when
                fully bayesian model is used. If None, no expansion is performed.
        """
        super().__init__(objective=objective)
        self.penalty_objective = penalty_objective
        self.regularization_parameter = regularization_parameter
        self.expand_dim = expand_dim

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the penalized objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim Tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of objective values
            with penalty added for each point.
        """
        obj = super().forward(samples=samples, X=X)
        penalty_obj = self.penalty_objective(X)
        # when fully bayesian model is used, we pass unmarginalize_dim to match the
        # shape between obj ``sample_shape x batch-shape x mcmc_samples x q`` and
        # penalty_obj ``1 x batch-shape x q``
        if self.expand_dim is not None:
            # reshape penalty_obj to match the dim
            penalty_obj = penalty_obj.unsqueeze(self.expand_dim)
        # this happens when samples is a ``q x m``-dim tensor and X is a ``q x d``-dim
        # tensor; obj returned from GenericMCObjective is a ``q``-dim tensor and
        # penalty_obj is a ``1 x q``-dim tensor.
        if obj.ndim == 1:
            assert penalty_obj.shape == torch.Size([1, samples.shape[-2]])
            penalty_obj = penalty_obj.squeeze(dim=0)
        return obj - self.regularization_parameter * penalty_obj


class L0PenaltyApproxObjective(L0Approximation):
    r"""Differentiable relaxation of the L0 norm penalty objective class.
    An instance of this class can be added to any arbitrary objective to
    construct a PenalizedMCObjective.
    """

    def __init__(self, target_point: Tensor, a: float = 1.0, **tkwargs: Any) -> None:
        r"""Initializing L0 penalty with differentiable relaxation.

        Args:
            target_point: A tensor corresponding to the target point.
            a: A hyperparameter that controls the differentiable relaxation.
        """
        super().__init__(target_point=target_point, a=a, **tkwargs)

    def __call__(self, X: Tensor) -> Tensor:
        r"""
        Args:
            X: A "batch_shape x q x dim" representing the points to be evaluated.
        Returns:
            A "1 x batch_shape x q" tensor representing the penalty for each point.
            The first dimension corresponds to the dimension of MC samples.
        """
        return super().__call__(X=X).squeeze(dim=-1).unsqueeze(dim=0)
