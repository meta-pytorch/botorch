#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class StoppingCriterion(Protocol):
    r"""Protocol for evaluating optimization convergence.

    Stopping criteria are implemented as objects rather than functions, so that they
    can keep track of past function values between optimization steps.
    """

    def __call__(self, fvals: Tensor) -> bool:
        r"""Evaluate the stopping criterion.

        Args:
            fvals: tensor containing function values for the current iteration. If
                `fvals` contains more than one element, then the stopping criterion is
                evaluated element-wise and True is returned if the stopping criterion is
                true for all elements.

        Returns:
            Stopping indicator (if True, stop the optimization).
        """
        ...  # pragma: no cover

    def reset(self) -> None:
        r"""Reset the stopping criterion to its initial state.

        This method should be called before starting a new optimization run
        to ensure that any internal state from previous runs is cleared.
        """
        ...  # pragma: no cover


class ExpMAStoppingCriterion:
    r"""Exponential moving average stopping criterion.

    Computes an exponentially weighted moving average over window length `n_window`
    and checks whether the relative decrease in this moving average between steps
    is less than a provided tolerance level. That is, in iteration `i`, it computes

        v[i,j] := fvals[i - n_window + j] * w[j]

    for all `j = 0, ..., n_window`, where `w[j] = exp(-eta * (1 - j / n_window))`.
    Letting `ma[i] := sum_j(v[i,j])`, the criterion evaluates to `True` whenever

        (ma[i-1] - ma[i]) / abs(ma[i-1]) < rel_tol (if minimize=True)
        (ma[i] - ma[i-1]) / abs(ma[i-1]) < rel_tol (if minimize=False)
    """

    def __init__(
        self,
        maxiter: int = 10000,
        minimize: bool = True,
        n_window: int = 10,
        eta: float = 1.0,
        rel_tol: float = 1e-5,
    ) -> None:
        r"""Exponential moving average stopping criterion.

        Args:
            maxiter: Maximum number of iterations.
            minimize: If True, assume minimization.
            n_window: The size of the exponential moving average window.
            eta: The exponential decay factor in the weights.
            rel_tol: Relative tolerance for termination.
        """
        self.maxiter = maxiter
        self.minimize = minimize
        self.n_window = n_window
        self.rel_tol = rel_tol
        self.iter = 0
        weights = torch.exp(torch.linspace(-eta, 0, self.n_window))
        self.weights = weights / weights.sum()
        self._prev_fvals = None

    def __call__(self, fvals: Tensor) -> bool:
        r"""Evaluate the stopping criterion.

        Args:
            fvals: tensor containing function values for the current iteration. If
                `fvals` contains more than one element, then the stopping criterion is
                evaluated element-wise and True is returned if the stopping criterion is
                true for all elements.

        TODO: add support for utilizing gradient information

        Returns:
            Stopping indicator (if True, stop the optimziation).
        """
        self.iter += 1
        if self.iter == self.maxiter:
            return True

        if self._prev_fvals is None:
            self._prev_fvals = fvals.unsqueeze(0)
        else:
            self._prev_fvals = torch.cat(
                [self._prev_fvals[-self.n_window :], fvals.unsqueeze(0)]
            )

        if self._prev_fvals.size(0) < self.n_window + 1:
            return False

        weights = self.weights
        weights = weights.to(fvals)
        if self._prev_fvals.ndim > 1:
            weights = weights.unsqueeze(-1)

        # TODO: Update the exp moving average efficiently
        prev_ma = (self._prev_fvals[:-1] * weights).sum(dim=0)
        ma = (self._prev_fvals[1:] * weights).sum(dim=0)
        # TODO: Handle approx. zero losses (normalize by min/max loss range)
        rel_delta = (prev_ma - ma) / prev_ma.abs()

        if not self.minimize:
            rel_delta = -rel_delta
        if torch.max(rel_delta) < self.rel_tol:
            return True

        return False

    def reset(self) -> None:
        r"""Reset the stopping criterion to its initial state.

        Resets the iteration counter and clears any stored function values.
        """
        self.iter = 0
        self._prev_fvals = None
