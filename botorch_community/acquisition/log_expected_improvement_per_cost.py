#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Log Expected Improvement with evaluation cost (analytic).

References

.. [Ament2023logei]
    S. Ament, S. Daulton, D. Eriksson, M. Balandat, E. Bakshy.
    Unexpected Improvements to Expected Improvement for Bayesian 
    Optimization. Advances in Neural Information Processing Systems,
    36, 2023.

.. [Xie2025costaware]
    Q. Xie, L. Cai, A. Terenin, P. I. Frazier, Z. Scully
    Cost-Aware Stopping for Bayesian Optimization.
    arXiv:2507.12453, 2025.

Contributor: wgst
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.analytic import (
    _log_ei_helper,
    _scaled_improvement,
    AnalyticAcquisitionFunction,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from torch import Tensor


class LogExpectedImprovementPerCost(AnalyticAcquisitionFunction):
    r"""Single-outcome Log Expected Improvement with evaluation cost (analytic).

    Computes the log expected improvement adjusted for the cost of evaluating
    the candidate point:

    ``LogEIC(x) = LogEI(x; best_f) - alpha * log(c(x)),``

    where ``LogEI`` is the log expected improvement [Ament2023logei]_ and
    ``c(x)`` is the evaluation cost at ``x``.  The argmax of ``LogEIC``
    is the cost-adjusted most promising candidate to evaluate next.

    This acquisition function underlies the cost-aware stopping rule of
    [Xie2025costaware]_: stop when ``max_x LogEIC(x) + log(lambda)`` is
    non-positive, i.e. no candidate's expected improvement exceeds its cost
    scaled by the exchange rate ``lambda``.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> # Cost proportional to first input (e.g. reaction time)
        >>> cost = lambda X: 1.0 + 3.0 * X[..., 0]
        >>> LogEIC = LogExpectedImprovementPerCost(
        ...     model, best_f=0.2, cost_callable=cost
        ... )
        >>> leic = LogEIC(test_X)
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        cost_callable: Callable[[Tensor], Tensor],
        alpha: float = 1.0,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Log Expected Improvement with evaluation cost.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a ``b``-dim Tensor (batch mode)
                representing the best function value observed so far
                (assumed noiseless).
            cost_callable: A callable ``c(X: Tensor[..., d]) -> Tensor[...]``
                that returns the strictly positive evaluation cost at each
                candidate point ``X``.  Supports spatially varying costs
                (e.g. ``lambda X: 1.0 + 3.0 * X[..., 0]``).  Must broadcast
                over the leading batch dimensions of ``X``.
            alpha: Cost exponent in ``c(x)^alpha``.  ``1.0`` (default) matches
                the primary formulation of [Xie2025costaware]_.  Values less
                than 1 reduce the influence of cost.
            posterior_transform: A PosteriorTransform. If using a multi-output
                model, a PosteriorTransform that transforms the multi-output
                posterior into a single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.cost_callable = cost_callable
        self.alpha = alpha
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Log Expected Improvement with cost on the candidate set X.

        Args:
            X: A ``(b1 x ... bk) x 1 x d``-dim batched tensor of
                ``d``-dim design points.

        Returns:
            A ``(b1 x ... bk)``-dim tensor of LogEIC values.
        """
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        log_ei = (_log_ei_helper(u) + sigma.log()).squeeze(-1)

        # X has shape (..., 1, d); squeeze the q-dim before passing to cost_callable.
        costs = self.cost_callable(X.squeeze(-2))
        if not (costs > 0).all():
            raise ValueError(
                "cost_callable must return strictly positive values; "
                "got non-positive cost(s)."
            )
        log_cost = costs.clamp(min=1e-12).log()
        return log_ei - self.alpha * log_cost
