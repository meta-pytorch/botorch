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


def separate_mtmvn(mvn: MultitaskMultivariateNormal) -> list[MultivariateNormal]:
    """
    Separate a MTMVN into a list of MVNs, where covariance across data within each task
    are preserved, while covariance across task are dropped.
    """
    # T150340766 Upstream as a class method on gpytorch MultitaskMultivariateNormal.
    full_covar = mvn.lazy_covariance_matrix
    num_data, num_tasks = mvn.mean.shape[-2:]

    mvns = []
    for c in range(num_tasks):
        # Compute indices for task c's data points
        if mvn._interleaved:
            # For interleaved: task c data points are at positions
            # c, c+num_tasks, c+2*num_tasks, ...
            task_indices = torch.arange(
                c, num_data * num_tasks, num_tasks, device=full_covar.device
            )
        else:
            # For non-interleaved: task c data points are at positions
            # c*num_data to (c+1)*num_data
            task_indices = torch.arange(
                c * num_data, (c + 1) * num_data, device=full_covar.device
            )

        # Extract covariance submatrix for task c
        task_covar = full_covar[..., task_indices, :]
        task_covar = task_covar[..., :, task_indices]

        mvns.append(
            MultivariateNormal(mvn.mean[..., c], to_linear_operator(task_covar))
        )
    return mvns
