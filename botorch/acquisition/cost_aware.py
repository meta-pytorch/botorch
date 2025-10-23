#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Cost functions for cost-aware acquisition functions, e.g. multi-fidelity KG.
To be used in a context where there is an objective/cost tradeoff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from botorch.acquisition.objective import (
    GenericMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
)
from botorch.models.deterministic import DeterministicModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.base import MCSampler
from pyre_extensions import none_throws
from torch import Tensor
from torch.nn import Module


class CostAwareUtility(Module, ABC):
    """Abstract base class for cost-aware utilities."""

    @abstractmethod
    def forward(
        self, X: Tensor, deltas: Tensor, sampler: MCSampler | None = None
    ) -> Tensor:
        r"""Evaluate the cost-aware utility on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.
            sampler: A sampler used for sampling from the posterior of the cost
                model. Some subclasses ignore this argument.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-transformed utilities.
        """


class GenericCostAwareUtility(CostAwareUtility):
    r"""Generic cost-aware utility wrapping a callable."""

    def __init__(self, cost: Callable[[Tensor, Tensor], Tensor]) -> None:
        r"""Generic cost-aware utility wrapping a callable.

        Args:
            cost: A callable mapping a `batch_shape x q x d'`-dim candidate set
                to a `batch_shape`-dim tensor of costs
        """
        super().__init__()
        self._cost_callable: Callable[[Tensor, Tensor], Tensor] = cost

    def forward(
        self, X: Tensor, deltas: Tensor, sampler: MCSampler | None = None
    ) -> Tensor:
        r"""Evaluate the cost function on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d'`-dim Tensor of with `q` `d`-dim design
                points for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.
            sampler: Ignored.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-weighted utilities.
        """
        return self._cost_callable(X, deltas)


class InverseCostWeightedUtility(CostAwareUtility):
    r"""A cost-aware utility using inverse cost weighting based on a model.

    Computes the cost-aware utility by inverse-weighting samples
    `U = (u_1, ..., u_N)` of the increase in utility. If `use_mean=True`, this
    uses the posterior mean `mean_cost` of the cost model, i.e.
    `weighted utility = mean(U) / mean_cost`. If `use_mean=False`, it uses
    samples `C = (c_1, ..., c_N)` from the posterior of the cost model and
    performs the inverse weighting on the sample level:
    `weighted utility = mean(u_1 / c_1, ..., u_N / c_N)`.

    Where values in (u_1, ..., u_N) are negative, or for mean(U) < 0, the
    weighted utility is instead calculated via scaling by the cost, i.e. if
    `use_mean=True`: `weighted_utility = mean(U) * mean_cost` and if
    `use_mean=False`:
    `weighted utility = mean(u_1 * c_1, u_2 / c_2, u_3 * c_3, ..., u_N / c_N)`,
    depending on whether (`u_*` >= 0), as with `u_2` and `u_N` in this case, or
    (`u_*` < 0) as with `u_1` and `u_3`.

    The cost is additive across multiple elements of a q-batch.
    """

    def __init__(
        self,
        cost_model: DeterministicModel | GPyTorchModel,
        use_mean: bool = True,
        cost_objective: MCAcquisitionObjective | None = None,
        log: bool = False,
    ) -> None:
        r"""Cost-aware utility that weights increase in utility by inverse cost.
        For negative increases in utility, the utility is instead scaled by the
        cost. See the class description for more information.

        Args:
            cost_model: A model of the cost of evaluating a candidate
                set `X`, where `X` are the same features as in the model for the
                acquisition function this is to be used with. If no cost_objective
                is specified, the outputs are required to be non-negative.
            use_mean: If True, use the posterior mean, otherwise use posterior
                samples from the cost model.
            cost_objective: If specified, transform the posterior mean / the
                posterior samples from the cost model. This can be used e.g. to
                un-transform predictions/samples of a cost model fit on the
                log-transformed cost (often done to ensure non-negativity). If the
                cost model is multi-output, then by default this will sum the cost
                across outputs. NOTE: ``cost_objective`` must output
                strictly positive values; forward will raise a ``ValueError`` otherwise.
            min_cost: A value used to clamp the cost samples so that they are not
                too close to zero, which may cause numerical issues.
        Returns:
            The inverse-cost-weighted utility.
        """
        super().__init__()
        if cost_objective is None:
            if cost_model.num_outputs == 1:
                cost_objective = IdentityMCObjective()
            else:
                # sum over outputs
                cost_objective = GenericMCObjective(lambda Y, X: Y.sum(dim=-1))

        self.cost_model = cost_model
        self.cost_objective: MCAcquisitionObjective = cost_objective
        self._use_mean = use_mean
        self._log = log

    def forward(
        self,
        X: Tensor,
        deltas: Tensor,
        sampler: MCSampler | None = None,
        X_evaluation_mask: Tensor | None = None,
    ) -> Tensor:
        r"""Evaluate the cost function on the candidates and improvements. Note
        that negative values of `deltas` are instead scaled by the cost, and not
        inverse-weighted. See the class description for more information.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.
            sampler: A sampler used for sampling from the posterior of the cost
                model (required if `use_mean=False`, ignored if `use_mean=True`).
            X_evaluation_mask: A `q x m`-dim boolean tensor indicating which
                outcomes should be evaluated for each design in the batch.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-weighted utilities.
        """
        if not self._use_mean and sampler is None:
            raise RuntimeError("Must provide `sampler` if `use_mean=False`")
        if X_evaluation_mask is not None:
            # TODO: support different evaluation masks for each X. This requires
            # either passing evaluation_mask to `cost_model.posterior`
            # or assuming that evaluating `cost_model.posterior(X)` on all
            # `q` points and then only selecting the costs for relevant points
            # does not change the cost function for each point. This would not be
            # true for instance if the incremental cost of evaluating an additional
            # point decreased as the number of points increased.
            if not all(
                torch.equal(X_evaluation_mask[0], X_evaluation_mask[i])
                for i in range(1, X_evaluation_mask.shape[0])
            ):
                raise NotImplementedError(
                    "Currently, all candidates must be evaluated on the same outputs."
                )
            output_indices = X_evaluation_mask[0].nonzero().view(-1).tolist()
        else:
            output_indices = None
        cost_posterior = self.cost_model.posterior(X, output_indices=output_indices)
        if self._use_mean:
            cost = cost_posterior.mean  # batch_shape x q x m'
        else:
            # This will be of shape num_fantasies x batch_shape x q x m'
            cost = none_throws(sampler)(cost_posterior)
        cost = self.cost_objective(cost)

        # Ensure that costs are positive
        if not torch.all(cost > 0.0):
            raise ValueError(
                "Costs must be strictly positive. Consider clamping cost_objective."
            )

        # sum costs along q-batch
        cost = cost.sum(dim=-1)

        # compute and return the ratio on the sample level - If `use_mean=True`
        # this operation involves broadcasting the cost across fantasies.
        if self._log:
            # if _log is True then input deltas are in log space
            # so original deltas cannot be <= 0
            return deltas - torch.log(cost)
        else:
            # We multiply by the cost if the deltas are <= 0, see discussion #2914
            return torch.where(deltas > 0, deltas / cost, deltas * cost)
