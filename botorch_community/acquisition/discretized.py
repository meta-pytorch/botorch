#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all botorch acquisition functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)

from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from torch import Tensor


class DiscretizedAcquistionFunction(AcquisitionFunction, ABC):
    r"""DiscretizedAcquistionFunction is an abstract base class for acquisition
    functions that are defined on discrete distributions. It wraps a model and
    implements a forward method that computes the acquisition function value at
    a given set of points.
    This class can be subclassed to define acquisiton functions for Riemann-
    distributed posteriors.
    The acquisition function must have the form $$acq(x) = \int p(y|x) ag(x)$$,
    where $$ag$$ is defined differently for each acquisition function.
    The ag_integrate method, which computes the integral of ag between two points, must
    be implemented by subclasses to define the specific acquisition functions.
    """

    def __init__(self, model: Model, posterior_transform: PosteriorTransform) -> None:
        r"""
        Initialize the DiscretizedAcquistionFunction

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
            posterior_transform: A ScalarizedPosteriorTransform that can only
                indicate minimization or maximization of the objective.
        """
        super().__init__(model=model)
        self.maximize = True
        if posterior_transform is not None:
            unsupported_error_message = (
                "Only scalarized posterior transforms with a"
                "single objective and 0.0 offset are supported."
            )
            if (
                not isinstance(posterior_transform, ScalarizedPosteriorTransform)
                or (posterior_transform.offset != 0.0)
                or len(posterior_transform.weights) != 1
                or posterior_transform.weights[0] not in [-1.0, 1.0]
            ):
                raise UnsupportedError(unsupported_error_message)

            self.maximize = posterior_transform.weights[0] == 1.0

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of the acquisition function at the given
            design points `X`.
        """
        discrete_posterior = self.model.posterior(X)
        if not self.maximize:
            discrete_posterior.borders = -torch.flip(discrete_posterior.borders, [0])
            discrete_posterior.probabilities = torch.flip(
                discrete_posterior.probabilities, [-1]
            )

        result = discrete_posterior.integrate(self.ag_integrate)
        # remove q dimension
        return result.squeeze(-1)

    @abstractmethod
    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        This function calculates the integral that computes the acquisition function
        without the posterior factor from lower_bound to upper_bound.
        That is, our acquisition function is assumed to have the form
        \int ag(x) * p(x) dx,
        and this function calculates \int_{lower_bound}^{upper_bound} ag(x) dx.
        The `integrate` method of the posterior (`BoundedRiemannPosterior`)
        then computes the final acquisition value.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        pass  # pragma: no cover


class DiscretizedExpectedImprovement(DiscretizedAcquistionFunction):
    r"""DiscretizedExpectedImprovement is an acquisition function that
    computes the expected improvement over the current best observed value
    for a Riemann distribution.
    """

    def __init__(
        self,
        model: Model,
        best_f: Tensor,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""
        Initialize the DiscretizedExpectedImprovement

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
            best_f: A tensor representing the current best observed value.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        As Expected improvement has ag(y) = (y - best_f).clamp(min=0), and
        is defined as \int ag(y) * p(y) dy, we can calculate the integral
        of ag(y) like so:
        We just calculate ag(y) at beginning and end, and since the function has
        a gradient of 1 or 0, we can just take the average of the two.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        best_f = self.best_f.to(lower_bound)

        # Case 1: best_f >= upper_bound, entire interval gives 0 improvement
        case1_mask = best_f >= upper_bound

        # Case 2: best_f <= lower_bound, entire interval gives improvement
        case2_mask = best_f <= lower_bound

        # Case 3: lower_bound < best_f < upper_bound, partial improvement
        case3_mask = ~(case1_mask | case2_mask)

        # Initialize result tensor
        result = torch.zeros_like(lower_bound)

        # Case 1: result is already 0

        # Case 2: integral = (
        #    ((upper_bound + lower_bound)/2 - best_f)
        #     * (upper_bound - lower_bound)
        # )
        if case2_mask.any():
            bucket_width = upper_bound - lower_bound
            bucket_center = (upper_bound + lower_bound) / 2
            result = torch.where(
                case2_mask, (bucket_center - best_f) * bucket_width, result
            )

        # Case 3: integral = (upper_bound - best_f)²/2
        if case3_mask.any():
            result = torch.where(case3_mask, (upper_bound - best_f).pow(2) / 2, result)

        return result.clamp_min(0)


class DiscretizedProbabilityOfImprovement(DiscretizedAcquistionFunction):
    r"""DiscretizedProbabilityOfImprovement is an acquisition function that
    computes the probability of improvement over the current best observed value
    for a Riemann distribution.
    """

    def __init__(
        self,
        model: Model,
        best_f: Tensor,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""
        Initialize the DiscretizedProbabilityOfImprovement

        Args:
            model: A fitted model that is used to compute the posterior
                distribution over the outcomes of interest.
                The model should be a `PFNModel`.
            best_f: A tensor representing the current best observed value.
        """

        super().__init__(model, posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def ag_integrate(self, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        r"""
        PI is defined as \int ag(y) * p(y) dy, where ag(y) = I(y - best_f)
        and I being the indicator function.

        So all we need to do is calculate the portion between the bounds
        that is larger than best_f.
        We do this by comparing how much higher the upper bound is than best_f,
        compared to the size of the bucket.
        Then we clamp at one if best_f is below lower_bound and at zero if
        best_f is above upper_bound.

        Args:
            lower_bound: lower bound of integral
            upper_bound: upper bound of integral

        Returns:
            A `(b)`-dim Tensor of acquisition function derivatives at the given
            design points `X`.
        """
        best_f = self.best_f.to(lower_bound)
        # two separate clamps needed below, as one is a tensor and one a scalar
        return (
            (upper_bound - best_f).clamp(min=0.0).clamp(max=upper_bound - lower_bound)
        )
