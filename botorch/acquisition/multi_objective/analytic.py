#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions for Multi-objective Bayesian optimization.

References

.. [Yang2019]
    Yang, K., Emmerich, M., Deutz, A. et al. Efficient computation of expected
    hypervolume improvement using box decomposition algorithms. J Glob Optim 75,
    3–34 (2019)

"""

from __future__ import annotations

from itertools import product

import torch
from botorch.acquisition.multi_objective.base import (
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Normal


class ExpectedHypervolumeImprovement(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        ref_point: list[float],
        partitioning: NondominatedPartitioning,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Expected Hypervolume Improvement supporting m>=2 outcomes.

        This computes EHVI using the algorithm from [Yang2019]_, but
        additionally computes gradients via auto-differentiation as proposed by
        [Daulton2020qehvi]_.

        Note: this is currently inefficient in two ways due to the binary partitioning
        algorithm that we use for the box decomposition:

            - We have more boxes in our decomposition
            - If we used a box decomposition that used ``inf`` as the upper bound for
                the last dimension *in all hypercells*, then we could reduce the number
                of terms we need to compute from 2^m to 2^(m-1). [Yang2019]_ do this
                by using DKLV17 and LKF17 for the box decomposition.

        TODO: Use DKLV17 and LKF17 for the box decomposition as in [Yang2019]_ for
        greater efficiency.

        TODO: Add support for outcome constraints.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> EHVI = ExpectedHypervolumeImprovement(model, ref_point, partitioning)
            >>> ehvi = EHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list with ``m`` elements representing the reference point
                (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the outcome values (i.e., after
                applying ``posterior_transform`` if provided).
            partitioning: A ``NondominatedPartitioning`` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles.
            posterior_transform: A ``PosteriorTransform``.
        """
        # TODO: we could refactor this __init__ logic into a
        # HypervolumeAcquisitionFunction Mixin
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        better_than_ref = (partitioning.pareto_Y > ref_point).all(dim=1)
        if not better_than_ref.any() and partitioning.pareto_Y.shape[0] > 0:
            raise ValueError(
                "At least one pareto point must be better than the reference point."
            )
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("ref_point", ref_point)
        self.partitioning = partitioning
        cell_bounds = self.partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        # create indexing tensor of shape ``2^m x m``
        self._cross_product_indices = torch.tensor(
            list(product(*[[0, 1] for _ in range(ref_point.shape[0])])),
            dtype=torch.long,
            device=ref_point.device,
        )
        self.normal = Normal(0, 1)

    def psi(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""Compute Psi function.

        For each cell i and outcome k:

            Psi(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            sigma_k * PDF((upper_{i,k} - mu_k) / sigma_k) + (
            mu_k - lower_{i,k}
            ) * (1 - CDF(upper_{i,k} - mu_k) / sigma_k)
            )

        See Equation 19 in [Yang2019]_ for more details.

        Args:
            lower: A ``num_cells x m``-dim tensor of lower cell bounds
            upper: A ``num_cells x m``-dim tensor of upper cell bounds
            mu: A ``batch_shape x 1 x m``-dim tensor of means
            sigma: A ``batch_shape x 1 x m``-dim tensor of standard
                deviations (clamped).

        Returns:
            A ``batch_shape x num_cells x m``-dim tensor of values.
        """
        u = (upper - mu) / sigma
        return sigma * self.normal.log_prob(u).exp() + (mu - lower) * (
            1 - self.normal.cdf(u)
        )

    def nu(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        r"""Compute Nu function.

        For each cell i and outcome k:

            nu(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            upper_{i,k} - lower_{i,k}
            ) * (1 - CDF((upper_{i,k} - mu_k) / sigma_k))

        See Equation 25 in [Yang2019]_ for more details.

        Args:
            lower: A ``num_cells x m``-dim tensor of lower cell bounds
            upper: A ``num_cells x m``-dim tensor of upper cell bounds
            mu: A ``batch_shape x 1 x m``-dim tensor of means
            sigma: A ``batch_shape x 1 x m``-dim tensor of standard
                deviations (clamped).

        Returns:
            A ``batch_shape x num_cells x m``-dim tensor of values.
        """
        return (upper - lower) * (1 - self.normal.cdf((upper - mu) / sigma))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X, posterior_transform=self.posterior_transform
        )
        mu = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        # clamp here, since upper_bounds will contain ``inf``s, which
        # are not differentiable
        cell_upper_bounds = self.cell_upper_bounds.clamp_max(
            1e10 if X.dtype == torch.double else 1e8
        )
        # Compute psi(lower_i, upper_i, mu_i, sigma_i) for i=0, ... m-2
        psi_lu = self.psi(
            lower=self.cell_lower_bounds, upper=cell_upper_bounds, mu=mu, sigma=sigma
        )
        # Compute psi(lower_m, lower_m, mu_m, sigma_m)
        psi_ll = self.psi(
            lower=self.cell_lower_bounds,
            upper=self.cell_lower_bounds,
            mu=mu,
            sigma=sigma,
        )
        # Compute nu(lower_m, upper_m, mu_m, sigma_m)
        nu = self.nu(
            lower=self.cell_lower_bounds, upper=cell_upper_bounds, mu=mu, sigma=sigma
        )
        # compute the difference psi_ll - psi_lu
        psi_diff = psi_ll - psi_lu

        # this is batch_shape x num_cells x 2 x (m-1)
        stacked_factors = torch.stack([psi_diff, nu], dim=-2)

        # Take the cross product of psi_diff and nu across all outcomes
        # e.g. for m = 2
        # for each batch and cell, compute
        # [psi_diff_0, psi_diff_1]
        # [nu_0, psi_diff_1]
        # [psi_diff_0, nu_1]
        # [nu_0, nu_1]
        # this tensor has shape: ``batch_shape x num_cells x 2^m x m``
        all_factors_up_to_last = stacked_factors.gather(
            dim=-2,
            index=self._cross_product_indices.expand(
                stacked_factors.shape[:-2] + self._cross_product_indices.shape
            ),
        )
        # compute product for all 2^m terms,
        # sum across all terms and hypercells
        return all_factors_up_to_last.prod(dim=-1).sum(dim=-1).sum(dim=-1)
