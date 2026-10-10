#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for multitask modeling.
"""

from __future__ import annotations

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from linear_operator import to_linear_operator
from linear_operator.operators import BlockDiagLinearOperator


def separate_mtmvn(mvn: MultitaskMultivariateNormal) -> list[MultivariateNormal]:
    """
    Separate a MTMVN into a list of MVNs, where covariance across data within each task
    are preserved, while covariance across task are dropped.
    """
    # T150340766 Upstream as a class method on gpytorch MultitaskMultivariateNormal.
    full_covar = mvn.lazy_covariance_matrix
    num_data, num_tasks = mvn.mean.shape[-2:]
    # Independent tasks (e.g. from ``from_independent_mvns``) have a block-diagonal
    # covariance with one ``num_data x num_data`` block per task.
    is_block_diag = (
        not mvn._interleaved
        and isinstance(full_covar, BlockDiagLinearOperator)
        and full_covar.num_blocks == num_tasks
    )

    mvns = []
    for c in range(num_tasks):
        if is_block_diag:
            # Take the block of task c. Slicing a ``BlockDiagLinearOperator`` (see
            # below) can return wrong results, since ``BlockLinearOperator._getitem``
            # assumes that the blocks are interleaved.
            task_covar = full_covar.base_linear_op[..., c, :, :]
        elif mvn._interleaved:
            # For interleaved: task c data points are at positions
            # c, c+num_tasks, c+2*num_tasks, ...
            # Must use tensor indexing for strided access.
            task_indices = torch.arange(
                c, num_data * num_tasks, num_tasks, device=full_covar.device
            )
            task_covar = full_covar[..., task_indices, :]
            task_covar = task_covar[..., :, task_indices]
        else:
            # For non-interleaved: task c data points are at contiguous positions
            # c*num_data to (c+1)*num_data. Use slice-based indexing which
            # LinearOperator handles more efficiently than tensor indexing.
            start = c * num_data
            end = start + num_data
            task_covar = full_covar[..., start:end, start:end]

        mvns.append(
            MultivariateNormal(mvn.mean[..., c], to_linear_operator(task_covar))
        )
    return mvns
