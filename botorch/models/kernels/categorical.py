#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from gpytorch.kernels.kernel import Kernel
from torch import Tensor


class CategoricalKernel(Kernel):
    r"""A Kernel for categorical features.

    Computes ``exp(-dist(x1, x2) / lengthscale)``, where
    ``dist(x1, x2)`` is zero if ``x1 == x2`` and one if ``x1 != x2``.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        if diag:
            if x1.shape[-2] != x2.shape[-2]:
                raise BotorchTensorDimensionError(
                    "diag=True requires `x1` and `x2` to have the same number of "
                    f"points, but got {x1.shape[-2]} and {x2.shape[-2]}."
                )
            # Comparing rows directly avoids the `n1 x n2 x d` cross-product that would
            # otherwise be built and then discarded.
            dists = (x1 != x2) / self.lengthscale
            if last_dim_is_batch:
                return torch.exp(-dists.transpose(-1, -2))
            return torch.exp(-dists.mean(-1))

        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        dists = delta / self.lengthscale.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)
        return torch.exp(-dists)
