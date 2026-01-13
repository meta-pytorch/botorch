#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Box decomposition container."""

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from torch import Tensor
from torch.nn import Module, ModuleList


class BoxDecompositionList(Module):
    r"""A list of box decompositions."""

    def __init__(self, *box_decompositions: BoxDecomposition) -> None:
        r"""Initialize the box decomposition list.

        Args:
            *box_decompositions: An variable number of box decompositions

        Example:
            >>> bd1 = FastNondominatedPartitioning(ref_point, Y=Y1)
            >>> bd2 = FastNondominatedPartitioning(ref_point, Y=Y2)
            >>> bd = BoxDecompositionList(bd1, bd2)
        """
        super().__init__()
        self.box_decompositions = ModuleList(box_decompositions)

    @property
    def pareto_Y(self) -> list[Tensor]:
        r"""This returns the non-dominated set.

        Note: Internally, we store the negative pareto set (minimization).

        Returns:
            A list where the ith element is the ``n_pareto_i x m``-dim tensor
                of pareto optimal outcomes for each box_decomposition ``i``.
        """
        return [p.pareto_Y for p in self.box_decompositions]

    @property
    def ref_point(self) -> Tensor:
        r"""Get the reference point.

        Note: Internally, we store the negative reference point (minimization).

        Returns:
            A ``n_box_decompositions x m``-dim tensor of outcomes.
        """
        return torch.stack([p.ref_point for p in self.box_decompositions], dim=0)

    def get_hypercell_bounds(self) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x n_box_decompositions x num_cells x num_outcomes``-dim tensor
                containing the lower and upper vertices bounding each hypercell.
        """
        bounds_list = []
        max_num_cells = 0
        for p in self.box_decompositions:
            bounds = p.get_hypercell_bounds()
            max_num_cells = max(max_num_cells, bounds.shape[-2])
            bounds_list.append(bounds)
        # pad the decomposition with empty cells so that all
        # decompositions have the same number of cells
        for i, bounds in enumerate(bounds_list):
            num_missing = max_num_cells - bounds.shape[-2]
            if num_missing > 0:
                padding = torch.zeros(
                    2,
                    num_missing,
                    bounds.shape[-1],
                    dtype=bounds.dtype,
                    device=bounds.device,
                )
                bounds_list[i] = torch.cat(
                    [
                        bounds,
                        padding,
                    ],
                    dim=-2,
                )

        return torch.stack(bounds_list, dim=-3)

    def update(self, Y: list[Tensor] | Tensor) -> None:
        r"""Update the partitioning.

        Args:
            Y: A ``n_box_decompositions x n x num_outcomes``-dim tensor or a list
                where the ith  element contains the new points for
                box_decomposition ``i``.
        """
        if (
            torch.is_tensor(Y)
            and Y.ndim != 3
            and Y.shape[0] != len(self.box_decompositions)
        ) or (isinstance(Y, list) and len(Y) != len(self.box_decompositions)):
            raise BotorchTensorDimensionError(
                "BoxDecompositionList.update requires either a batched tensor Y, "
                "with one batch per box decomposition or a list of tensors with "
                "one element per box decomposition."
            )
        for i, p in enumerate(self.box_decompositions):
            p.update(Y[i])

    def compute_hypervolume(self) -> Tensor:
        r"""Compute hypervolume that is dominated by the Pareto Froniter.

        Returns:
            A ``(batch_shape)``-dim tensor containing the hypervolume dominated by
                each Pareto frontier.
        """
        return torch.stack(
            [p.compute_hypervolume() for p in self.box_decompositions], dim=0
        )
