#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Algorithms for partitioning the dominated space into hyperrectangles."""

from __future__ import annotations

from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    FastPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    compute_dominated_hypercell_bounds_2d,
    get_partition_bounds,
)
from torch import Tensor


class DominatedPartitioning(FastPartitioning):
    r"""Partition dominated space into axis-aligned hyperrectangles.

    This uses the Algorithm 1 from [Lacour17]_.

    Example:
        >>> bd = DominatedPartitioning(ref_point, Y)
    """

    def _partition_space_2d(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This direct method works for ``m=2`` outcomes.
        """
        cell_bounds = compute_dominated_hypercell_bounds_2d(
            # flip self.pareto_Y because it is sorted in decreasing order (since
            # self._pareto_Y was sorted in increasing order and we multiplied by -1)
            pareto_Y_sorted=self.pareto_Y.flip(-2),
            ref_point=self.ref_point,
        )
        self.hypercell_bounds = cell_bounds

    def _get_partitioning(self) -> None:
        r"""Get the bounds of each hypercell in the decomposition."""
        minimization_cell_bounds = get_partition_bounds(
            Z=self._Z, U=self._U, ref_point=self._neg_ref_point.view(-1)
        )
        cell_bounds = -minimization_cell_bounds.flip(0)
        self.hypercell_bounds = cell_bounds

    def _compute_hypervolume_if_y_has_data(self) -> Tensor:
        r"""Compute hypervolume that is dominated by the Pareto Frontier.

        Returns:
            A ``(batch_shape)``-dim tensor containing the hypervolume dominated by
                each Pareto frontier.
        """
        return (
            (self.hypercell_bounds[1] - self.hypercell_bounds[0])
            .prod(dim=-1)
            .sum(dim=-1)
        )

    def _get_single_cell(self) -> None:
        r"""Set the partitioning to be a single cell in the case of no Pareto points."""
        # Set lower and upper bounds to be the reference point to define an empty cell
        cell_bounds = self.ref_point.expand(
            2, *self._neg_pareto_Y.shape[:-2], 1, self.num_outcomes
        ).clone()
        self.hypercell_bounds = cell_bounds
