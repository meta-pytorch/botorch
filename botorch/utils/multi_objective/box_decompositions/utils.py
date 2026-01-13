#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for box decomposition algorithms."""

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Size, Tensor


def _expand_ref_point(ref_point: Tensor, batch_shape: Size) -> Tensor:
    r"""Expand reference point to the proper batch_shape.

    Args:
        ref_point: A ``(batch_shape) x m``-dim tensor containing the reference
            point.
        batch_shape: The batch shape.

    Returns:
        A ``batch_shape x m``-dim tensor containing the expanded reference point
    """
    if ref_point.shape[:-1] != batch_shape:
        if ref_point.ndim > 1:
            raise BotorchTensorDimensionError(
                "Expected ref_point to be a `batch_shape x m` or `m`-dim tensor, "
                f"but got {ref_point.shape}."
            )
        ref_point = ref_point.view(
            *(1 for _ in batch_shape), ref_point.shape[-1]
        ).expand(batch_shape + ref_point.shape[-1:])
    return ref_point


def _pad_batch_pareto_frontier(
    Y: Tensor,
    ref_point: Tensor,
    is_pareto: bool = False,
    feasibility_mask: Tensor | None = None,
) -> Tensor:
    r"""Get a batch Pareto frontier by padding the pareto frontier with repeated points.

    This assumes maximization.

    Args:
        Y: A ``(batch_shape) x n x m``-dim tensor of points
        ref_point: a ``(batch_shape) x m``-dim tensor containing the reference point
        is_pareto: a boolean indicating whether the points in Y are already
            non-dominated.
        feasibility_mask: A ``(batch_shape) x n``-dim tensor of booleans indicating
            whether each point is feasible.

    Returns:
        A ``(batch_shape) x max_num_pareto x m``-dim tensor of padded Pareto
            frontiers.
    """
    tkwargs = {"dtype": Y.dtype, "device": Y.device}
    ref_point = ref_point.unsqueeze(-2)
    batch_shape = Y.shape[:-2]
    if len(batch_shape) > 1:
        raise UnsupportedError(
            "_pad_batch_pareto_frontier only supports a single "
            f"batch dimension, but got {len(batch_shape)} "
            "batch dimensions."
        )
    if feasibility_mask is not None:
        # set infeasible points to be the reference point (corresponding to the batch)
        Y = torch.where(feasibility_mask.unsqueeze(-1), Y, ref_point)
    if not is_pareto:
        pareto_mask = is_non_dominated(Y)
    else:
        pareto_mask = torch.ones(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    better_than_ref = (Y > ref_point).all(dim=-1)
    # is_non_dominated assumes maximization
    # TODO: filter out points that are worse than the reference point first here
    pareto_mask = pareto_mask & better_than_ref
    if len(batch_shape) == 0:
        return Y[pareto_mask]
    # Note: in the batch case, the Pareto frontier is padded by repeating
    # a Pareto point. This ensures that the padded box-decomposition has
    # the same number of points, which enables fast batch operations.
    max_n_pareto = pareto_mask.sum(dim=-1).max().item()
    pareto_Y = torch.empty(*batch_shape, max_n_pareto, Y.shape[-1], **tkwargs)
    for i, pareto_i in enumerate(pareto_mask):
        pareto_i = Y[i, pareto_mask[i]]
        n_pareto = pareto_i.shape[0]
        if n_pareto > 0:
            pareto_Y[i, :n_pareto] = pareto_i
            # pad pareto_Y, so that all batches have the same size Pareto set
            pareto_Y[i, n_pareto:] = pareto_i[-1]
        else:
            # if there are no pareto points in this batch, use the reference
            # point
            pareto_Y[i, :] = ref_point[i]
    return pareto_Y


def compute_local_upper_bounds(
    U: Tensor, Z: Tensor, z: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Compute local upper bounds.

    Note: this assumes minimization.

    This uses the incremental algorithm (Alg. 1) from [Lacour17]_.

    Args:
        U: A ``n x m``-dim tensor containing the local upper bounds.
        Z: A ``n x m x m``-dim tensor containing the defining points.
        z: A ``m``-dim tensor containing the new point.

    Returns:
        2-element tuple containing:

        - A new ``n' x m``-dim tensor local upper bounds.
        - A ``n' x m x m``-dim tensor containing the defining points.
    """
    num_outcomes = U.shape[-1]
    z_dominates_U = (U > z).all(dim=-1)
    # Select upper bounds that are dominated by z.
    # These are the search zones that contain z.
    if not z_dominates_U.any():
        return U, Z
    A = U[z_dominates_U]
    A_Z = Z[z_dominates_U]
    P = []
    P_Z = []
    mask = torch.ones(num_outcomes, dtype=torch.bool, device=U.device)
    for j in range(num_outcomes):
        mask[j] = 0
        z_uj_max = A_Z[:, mask, j].max(dim=-1).values.view(-1)
        add_z = z[j] >= z_uj_max
        if add_z.any():
            u_j = A[add_z].clone()
            u_j[:, j] = z[j]
            P.append(u_j)
            A_Z_filtered = A_Z[add_z]
            Z_ku = A_Z_filtered[:, mask]
            lt_zj = Z_ku[..., j] <= z[j]
            P_uj = torch.zeros(
                u_j.shape[0], num_outcomes, num_outcomes, dtype=U.dtype, device=U.device
            )
            P_uj[:, mask] = Z_ku[lt_zj].view(P_uj.shape[0], num_outcomes - 1, -1)
            P_uj[:, ~mask] = z
            P_Z.append(P_uj)
        mask[j] = 1
    # filter out elements of U that are in A
    not_z_dominates_U = ~z_dominates_U
    U = U[not_z_dominates_U]
    # remaining indices
    Z = Z[not_z_dominates_U]
    if len(P) > 0:
        # add points from P_Z
        Z = torch.cat([Z, *P_Z], dim=0)
        # return elements in P or elements in (U that are not in A)
        U = torch.cat([U, *P], dim=-2)
    return U, Z


def get_partition_bounds(Z: Tensor, U: Tensor, ref_point: Tensor) -> Tensor:
    r"""Get the cell bounds given the local upper bounds and the defining points.

    This implements Equation 2 in [Lacour17]_.

    Args:
        Z: A ``n x m x m``-dim tensor containing the defining points. The first
            dimension corresponds to u_idx, the second dimension corresponds to j,
            and Z[u_idx, j] is the set of definining points Z^j(u) where
            u = U[u_idx].
        U: A ``n x m``-dim tensor containing the local upper bounds.
        ref_point: A ``m``-dim tensor containing the reference point.

    Returns:
        A ``2 x num_cells x m``-dim tensor containing the lower and upper vertices
            bounding each hypercell.
    """
    bounds = torch.empty(2, U.shape[0], U.shape[-1], dtype=U.dtype, device=U.device)
    for u_idx in range(U.shape[0]):
        # z_1^1(u)
        bounds[0, u_idx, 0] = Z[u_idx, 0, 0]
        # z_1^r(u)
        bounds[1, u_idx, 0] = ref_point[0]
        for j in range(1, U.shape[-1]):
            bounds[0, u_idx, j] = Z[u_idx, :j, j].max()
            bounds[1, u_idx, j] = U[u_idx, j]
    # remove empty partitions
    # Note: the equality will evaluate as True if the lower and upper bound
    # are both (-inf), which could happen if the reference point is -inf.
    empty = (bounds[1] <= bounds[0]).any(dim=-1)
    return bounds[:, ~empty]


def update_local_upper_bounds_incremental(
    new_pareto_Y: Tensor, U: Tensor, Z: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Update the current local upper with the new pareto points.

    This assumes minimization.

    Args:
        new_pareto_Y: A ``n x m``-dim tensor containing the new
            Pareto points.
        U: A ``n' x m``-dim tensor containing the local upper bounds.
        Z: A ``n x m x m``-dim tensor containing the defining points.

    Returns:
        2-element tuple containing:

        - A new ``n' x m``-dim tensor local upper bounds.
        - A ``n' x m x m``-dim tensor containing the defining points
    """
    for i in range(new_pareto_Y.shape[-2]):
        U, Z = compute_local_upper_bounds(U=U, Z=Z, z=new_pareto_Y[i])
    return U, Z


def compute_non_dominated_hypercell_bounds_2d(
    pareto_Y_sorted: Tensor, ref_point: Tensor
) -> Tensor:
    r"""Compute an axis-aligned partitioning of the non-dominated space for 2
    objectives.

    Args:
        pareto_Y_sorted: A ``(batch_shape) x n_pareto x 2``-dim tensor of pareto
            outcomes that are sorted by the 0th dimension in increasing order. All
            points must be better than the reference point.
        ref_point: A ``(batch_shape) x 2``-dim reference point.

    Returns:
        A ``2 x (batch_shape) x n_pareto + 1 x m``-dim tensor of cell bounds.
    """
    # add boundary point to each front
    # the boundary point is the extreme value in each outcome
    # (a single coordinate of reference point)
    batch_shape = pareto_Y_sorted.shape[:-2]
    if ref_point.ndim == pareto_Y_sorted.ndim - 1:
        expanded_boundary_point = ref_point.unsqueeze(-2)
    else:
        view_shape = torch.Size([1] * len(batch_shape)) + torch.Size([1, 2])
        expanded_shape = batch_shape + torch.Size([1, 2])
        expanded_boundary_point = ref_point.view(view_shape).expand(expanded_shape)

    # add the points (ref, y) and (x, ref) to the corresponding ends
    pareto_Y_sorted0, pareto_Y_sorted1 = torch.split(pareto_Y_sorted, 1, dim=-1)
    expanded_boundary_point0, expanded_boundary_point1 = torch.split(
        expanded_boundary_point, 1, dim=-1
    )
    left_end = torch.cat(
        [expanded_boundary_point0[..., :1, :], pareto_Y_sorted1[..., :1, :]], dim=-1
    )
    right_end = torch.cat(
        [pareto_Y_sorted0[..., -1:, :], expanded_boundary_point1[..., :1, :]], dim=-1
    )
    front = torch.cat([left_end, pareto_Y_sorted, right_end], dim=-2)
    # The top left corners of axis-aligned rectangles in dominated partitioning.
    # These are the bottom left corners of the non-dominated partitioning
    front0, front1 = torch.split(front, 1, dim=-1)
    bottom_lefts = torch.cat([front0[..., :-1, :], front1[..., 1:, :]], dim=-1)
    top_right_xs = torch.cat(
        [
            front0[..., 1:-1, :],
            torch.full(
                bottom_lefts.shape[:-2] + torch.Size([1, 1]),
                float("inf"),
                dtype=front.dtype,
                device=front.device,
            ),
        ],
        dim=-2,
    )
    top_rights = torch.cat(
        [
            top_right_xs,
            torch.full(
                bottom_lefts.shape[:-1] + torch.Size([1]),
                float("inf"),
                dtype=front.dtype,
                device=front.device,
            ),
        ],
        dim=-1,
    )
    return torch.stack([bottom_lefts, top_rights], dim=0)


def compute_dominated_hypercell_bounds_2d(
    pareto_Y_sorted: Tensor, ref_point: Tensor
) -> Tensor:
    r"""Compute an axis-aligned partitioning of the dominated space for 2-objectives.

    Args:
        pareto_Y_sorted: A ``(batch_shape) x n_pareto x 2``-dim tensor of pareto
            outcomes that are sorted by the 0th dimension in increasing order.
        ref_point: A ``2``-dim reference point.

    Returns:
        A ``2 x (batch_shape) x n_pareto x m``-dim tensor of cell bounds.
    """
    # add boundary point to each front
    # the boundary point is the extreme value in each outcome
    # (a single coordinate of reference point)
    batch_shape = pareto_Y_sorted.shape[:-2]
    if ref_point.ndim == pareto_Y_sorted.ndim - 1:
        expanded_boundary_point = ref_point.unsqueeze(-2)
    else:
        view_shape = torch.Size([1] * len(batch_shape)) + torch.Size([1, 2])
        expanded_shape = batch_shape + torch.Size([1, 2])
        expanded_boundary_point = ref_point.view(view_shape).expand(expanded_shape)

    # add the points (ref, y) and (x, ref) to the corresponding ends
    pareto_Y_sorted0, pareto_Y_sorted1 = torch.split(pareto_Y_sorted, 1, dim=-1)
    expanded_boundary_point0, expanded_boundary_point1 = torch.split(
        expanded_boundary_point, 1, dim=-1
    )
    left_end = torch.cat(
        [expanded_boundary_point0[..., :1, :], pareto_Y_sorted0[..., :1, :]], dim=-1
    )
    right_end = torch.cat(
        [pareto_Y_sorted1[..., :1, :], expanded_boundary_point1[..., :1, :]], dim=-1
    )
    front = torch.cat([left_end, pareto_Y_sorted, right_end], dim=-2)
    # compute hypervolume by summing rectangles from min_x -> max_x
    top_rights = front[..., 1:-1, :]
    bottom_lefts = torch.cat(
        [
            front[..., :-2, :1],
            expanded_boundary_point1.expand(*top_rights.shape[:-1], 1),
        ],
        dim=-1,
    )
    return torch.stack([bottom_lefts, top_rights], dim=0)
