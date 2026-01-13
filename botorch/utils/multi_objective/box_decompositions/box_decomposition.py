#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Box decomposition algorithms.

References

.. [Lacour17]
    R. Lacour, K. Klamroth, C. Fonseca. A box decomposition algorithm to
    compute the hypervolume indicator. Computers & Operations Research,
    Volume 79, 2017.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.multi_objective.box_decompositions.utils import (
    _expand_ref_point,
    _pad_batch_pareto_frontier,
    update_local_upper_bounds_incremental,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor
from torch.nn import Module


class BoxDecomposition(Module, ABC):
    r"""An abstract class for box decompositions.

    Note: Internally, we store the negative reference point (minimization).
    """

    def __init__(self, ref_point: Tensor, sort: bool, Y: Tensor | None = None) -> None:
        """Initialize BoxDecomposition.

        Args:
            ref_point: A ``m``-dim tensor containing the reference point.
            sort: A boolean indicating whether to sort the Pareto frontier.
            Y: A ``(batch_shape) x n x m``-dim tensor of outcomes.
        """
        super().__init__()
        self._neg_ref_point = -ref_point
        self.sort = torch.tensor(sort, dtype=torch.bool)
        self.num_outcomes = ref_point.shape[-1]
        self.register_buffer("hypercell_bounds", None)

        if Y is not None:
            if Y.isnan().any():
                raise ValueError(
                    "NaN inputs are not supported. Got Y with "
                    f"{Y.isnan().sum()} NaN values."
                )
            self._neg_Y = -Y
            self._validate_inputs()
            self._neg_pareto_Y = self._compute_pareto_Y()
            self.partition_space()
        else:
            self._neg_Y = None
            self._neg_pareto_Y = None

    @property
    def pareto_Y(self) -> Tensor:
        r"""This returns the non-dominated set.

        Returns:
            A ``n_pareto x m``-dim tensor of outcomes.
        """
        if self._neg_pareto_Y is not None:
            return -self._neg_pareto_Y
        raise BotorchError("pareto_Y has not been initialized")

    @property
    def ref_point(self) -> Tensor:
        r"""Get the reference point.

        Returns:
            A ``m``-dim tensor of outcomes.
        """
        return -self._neg_ref_point

    @property
    def Y(self) -> Tensor:
        r"""Get the raw outcomes.

        Returns:
            A ``n x m``-dim tensor of outcomes.
        """
        if self._neg_Y is not None:
            return -self._neg_Y
        raise BotorchError("Y data has not been initialized")

    def _compute_pareto_Y(self) -> Tensor:
        if self._neg_Y is None:
            raise BotorchError("Y data has not been initialized")
        # is_non_dominated assumes maximization
        if self._neg_Y.shape[-2] == 0:
            return self._neg_Y
        # assumes maximization
        pareto_Y = -_pad_batch_pareto_frontier(
            Y=self.Y,
            ref_point=_expand_ref_point(
                ref_point=self.ref_point, batch_shape=self.batch_shape
            ),
        )
        if not self.sort:
            return pareto_Y
        # sort by first objective
        if len(self.batch_shape) > 0:
            pareto_Y = pareto_Y.gather(
                index=torch.argsort(pareto_Y[..., :1], dim=-2).expand(pareto_Y.shape),
                dim=-2,
            )
        else:
            pareto_Y = pareto_Y[torch.argsort(pareto_Y[:, 0])]
        return pareto_Y

    def _reset_pareto_Y(self) -> bool:
        r"""Update the non-dominated front.

        Returns:
            A boolean indicating whether the Pareto frontier has changed.
        """
        pareto_Y = self._compute_pareto_Y()

        if (self._neg_pareto_Y is None) or not torch.equal(
            pareto_Y, self._neg_pareto_Y
        ):
            self._neg_pareto_Y = pareto_Y
            return True
        return False

    def partition_space(self) -> None:
        r"""Compute box decomposition."""
        if self.num_outcomes == 2:
            try:
                self._partition_space_2d()
            except NotImplementedError:
                self._partition_space()
        else:
            self._partition_space()

    def _partition_space_2d(self) -> None:
        r"""Compute box decomposition for 2 objectives."""
        raise NotImplementedError

    @abstractmethod
    def _partition_space(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This method supports an arbitrary number of outcomes, but is
        less efficient than ``partition_space_2d`` for the 2-outcome case.
        """

    @abstractmethod
    def get_hypercell_bounds(self) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x num_cells x num_outcomes``-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """

    def _update_neg_Y(self, Y: Tensor) -> bool:
        r"""Update the set of outcomes.

        Returns:
            A boolean indicating if _neg_Y was initialized.
        """
        if Y.isnan().any():
            raise ValueError(
                "NaN inputs are not supported. Got Y with "
                f"{Y.isnan().sum()} NaN values."
            )
        # multiply by -1, since internally we minimize.
        if self._neg_Y is not None:
            self._neg_Y = torch.cat([self._neg_Y, -Y], dim=-2)
            return False
        self._neg_Y = -Y
        return True

    def update(self, Y: Tensor) -> None:
        r"""Update non-dominated front and decomposition.

        By default, the partitioning is recomputed. Subclasses can override
        this functionality.

        Args:
            Y: A ``(batch_shape) x n x m``-dim tensor of new, incremental outcomes.
        """
        self._update_neg_Y(Y=Y)
        self.reset()

    def _validate_inputs(self) -> None:
        self.batch_shape = self.Y.shape[:-2]
        self.num_outcomes = self.Y.shape[-1]
        if len(self.batch_shape) > 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports a single "
                f"batch dimension, but got {len(self.batch_shape)} "
                "batch dimensions."
            )
        elif len(self.batch_shape) > 0 and self.num_outcomes > 2:
            raise NotImplementedError(
                f"{type(self).__name__} only supports a batched box "
                f"decompositions in the 2-objective setting."
            )

    def reset(self) -> None:
        r"""Reset non-dominated front and decomposition."""
        self._validate_inputs()
        is_new_pareto = self._reset_pareto_Y()
        # Update decomposition if the Pareto front changed
        if is_new_pareto:
            self.partition_space()

    @abstractmethod
    def _compute_hypervolume_if_y_has_data(self) -> Tensor:
        """Compute hypervolume for the case that there is data in self._neg_pareto_Y."""

    def compute_hypervolume(self) -> Tensor:
        r"""Compute hypervolume that is dominated by the Pareto Froniter.

        Returns:
            A ``(batch_shape)``-dim tensor containing the hypervolume dominated by
                each Pareto frontier.
        """
        if self._neg_pareto_Y is None:
            return torch.tensor(0.0)

        if self._neg_pareto_Y.shape[-2] == 0:
            return torch.zeros(
                self._neg_pareto_Y.shape[:-2],
                dtype=self._neg_pareto_Y.dtype,
                device=self._neg_pareto_Y.device,
            )
        return self._compute_hypervolume_if_y_has_data()


class FastPartitioning(BoxDecomposition, ABC):
    r"""A class for partitioning the (non-)dominated space into hyper-cells.

    Note: this assumes maximization. Internally, it multiplies outcomes by -1
    and performs the decomposition under minimization.

    This class is abstract to support to two applications of Alg 1 from
    [Lacour17]_: 1) partitioning the space that is dominated by the Pareto
    frontier and 2) partitioning the space that is not dominated by the
    Pareto frontier.
    """

    def __init__(
        self,
        ref_point: Tensor,
        Y: Tensor | None = None,
    ) -> None:
        """
        Args:
            ref_point: A ``m``-dim tensor containing the reference point.
            Y: A ``(batch_shape) x n x m``-dim tensor
        """
        super().__init__(ref_point=ref_point, Y=Y, sort=ref_point.shape[-1] == 2)

    def update(self, Y: Tensor) -> None:
        r"""Update non-dominated front and decomposition.

        Args:
            Y: A ``(batch_shape) x n x m``-dim tensor of new, incremental outcomes.
        """
        if self._update_neg_Y(Y=Y):
            self.reset()
        else:
            if self.num_outcomes == 2 or self._neg_pareto_Y.shape[-2] == 0:
                # If there are two objective, recompute the box decomposition
                # because the partitions can be computed analytically.
                # If the current pareto set has no points, recompute the box
                # decomposition.
                self.reset()
            else:
                # only include points that are better than the reference point
                better_than_ref = (Y > self.ref_point).all(dim=-1)
                Y = Y[better_than_ref]
                Y_all = torch.cat([self._neg_pareto_Y, -Y], dim=-2)
                pareto_mask = is_non_dominated(-Y_all)
                # determine the number of points in Y that are Pareto optimal
                num_new_pareto = pareto_mask[-Y.shape[-2] :].sum()
                self._neg_pareto_Y = Y_all[pareto_mask]
                if num_new_pareto > 0:
                    # update local upper bounds for the minimization problem
                    self._U, self._Z = update_local_upper_bounds_incremental(
                        # this assumes minimization
                        new_pareto_Y=self._neg_pareto_Y[-num_new_pareto:],
                        U=self._U,
                        Z=self._Z,
                    )
                    # use the negative local upper bounds as the new pareto
                    # frontier for the minimization problem and perform
                    # box decomposition on dominated space.
                    self._get_partitioning()

    @abstractmethod
    def _get_single_cell(self) -> None:
        r"""Set the partitioning to be a single cell in the case of no Pareto points.

        This method should set self.hypercell_bounds
        """
        pass  # pragma: no cover

    def partition_space(self) -> None:
        if self._neg_pareto_Y.shape[-2] == 0:
            self._get_single_cell()
        else:
            super().partition_space()

    def _partition_space(self):
        r"""Partition the non-dominated space into disjoint hypercells.

        This method supports an arbitrary number of outcomes, but is
        less efficient than ``partition_space_2d`` for the 2-outcome case.
        """
        if len(self.batch_shape) > 0:
            # this could be triggered when m=2 outcomes and
            # BoxDecomposition._partition_space_2d is not overridden.
            raise NotImplementedError(
                "_partition_space does not support batch dimensions."
            )
        # this assumes minimization
        # initialize local upper bounds
        self.register_buffer("_U", self._neg_ref_point.unsqueeze(-2).clone())
        # initialize defining points to be the dummy points \hat{z} that are
        # defined in Sec 2.1 in [Lacour17]_. Note that in [Lacour17]_, outcomes
        # are assumed to be between [0,1], so they used 0 rather than -inf.
        self._Z = torch.zeros(
            1,
            self.num_outcomes,
            self.num_outcomes,
            dtype=self.Y.dtype,
            device=self.Y.device,
        )
        for j in range(self.ref_point.shape[-1]):
            # use ref point for maximization as the ideal point for minimization.
            self._Z[0, j] = float("-inf")
            self._Z[0, j, j] = self._U[0, j]
        # incrementally update local upper bounds and defining points
        # for each new Pareto point
        self._U, self._Z = update_local_upper_bounds_incremental(
            new_pareto_Y=self._neg_pareto_Y,
            U=self._U,
            Z=self._Z,
        )
        self._get_partitioning()

    @abstractmethod
    def _get_partitioning(self) -> None:
        r"""Compute partitioning given local upper bounds for the minimization problem.

        This method should set self.hypercell_bounds
        """
        pass  # pragma: no cover

    def get_hypercell_bounds(self) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x (batch_shape) x num_cells x m``-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        return self.hypercell_bounds
