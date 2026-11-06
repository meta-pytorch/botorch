#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Randomized inducing point allocation methods.

Contributor: @Mamunur02
"""

from __future__ import annotations

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.utils.inducing_point_allocators import (
    InducingPointAllocator,
    QualityFunction,
    UnitQualityFunction,
)
from gpytorch.module import Module
from torch import Tensor


class RandomizedPivotedCholesky(InducingPointAllocator):
    r"""Sample inducing points using randomized pivoted Cholesky.

    At each iteration, sample an unselected input with probability proportional
    to its residual diagonal. This is an independently implemented version of
    the randomized Nyström algorithm described by Chen, Epperly, Tropp, and
    Webber [chen2022randnystrom]_. If the residual rank is numerically exhausted,
    remaining indices are sampled uniformly without replacement so the result
    always contains exactly ``num_inducing`` points.

    This implementation supports unbatched two dimensional inputs and float32
    or float64 data. It stops residual-weighted sampling when the residual trace
    is at most machine epsilon times the initial trace, then samples the remaining
    indices uniformly without replacement. Duplicate input coordinates are allowed,
    while selected row indices are always distinct. It stores an ``n x m`` factor
    and uses O(n m^2) arithmetic. Sampling is controlled by ``torch.manual_seed``.

    Example:
        >>> allocator = RandomizedPivotedCholesky()
        >>> model = SingleTaskVariationalGP(
        ...     train_X,
        ...     train_Y,
        ...     inducing_points=64,
        ...     learn_inducing_points=False,
        ...     inducing_point_allocator=allocator,
        ... )
        >>> model.init_inducing_points(replacement_X)

    References:
        .. [chen2022randnystrom] Y. Chen, E. Epperly, J. Tropp, and R. Webber,
            "Randomly pivoted Cholesky: Practical approximation of a kernel
            matrix with few entry evaluations," 2022.
            https://arxiv.org/abs/2207.06503
    """

    def _get_quality_function(self) -> QualityFunction:
        """Return the unit quality function for interface compatibility."""
        return UnitQualityFunction()

    @torch.no_grad()
    def allocate_inducing_points(
        self,
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> Tensor:
        """Choose inducing points by sequential residual weighted sampling.

        Args:
            inputs: Unbatched training inputs of shape ``n x d``.
            covar_module: A GPyTorch covariance module.
            num_inducing: Number of distinct input indices to select.
            input_batch_shape: Must be empty; batched allocation is unsupported.

        Returns:
            A detached tensor of shape ``num_inducing x d``.

        Raises:
            UnsupportedError: If inputs, batch shape, count, or covariance
                outputs are unsupported or invalid.
        """
        if inputs.ndim != 2:
            raise UnsupportedError("inputs must be a 2-dimensional tensor.")
        if input_batch_shape != torch.Size():
            raise UnsupportedError("input_batch_shape must be empty.")
        if inputs.dtype not in (torch.float32, torch.float64):
            raise UnsupportedError("inputs must have dtype float32 or float64.")
        n = inputs.shape[0]
        if isinstance(num_inducing, bool) or not isinstance(num_inducing, int):
            raise UnsupportedError("num_inducing must be an integer.")
        if not 1 <= num_inducing <= n:
            raise UnsupportedError("num_inducing must satisfy 1 <= m <= n.")

        covar_module = covar_module.to(inputs.device)
        diagonal = covar_module(inputs, diag=True).clone()
        if diagonal.shape != torch.Size([n]):
            raise UnsupportedError("covariance diagonal must have shape (n,).")
        if diagonal.dtype != inputs.dtype:
            raise UnsupportedError("covariance and inputs must have the same dtype.")
        if not torch.isfinite(diagonal).all() or (diagonal < 0).any():
            raise UnsupportedError(
                "covariance diagonal must be finite and nonnegative."
            )

        # Normalize before summation and factorization to avoid overflow for
        # large output scales. Scaling a PSD matrix does not change the pivots.
        scale = diagonal.max()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        residual = diagonal / scale
        initial_sum = residual.sum()
        factor = inputs.new_zeros((n, num_inducing))
        selected = torch.zeros(n, dtype=torch.bool, device=inputs.device)
        indices = torch.empty(num_inducing, dtype=torch.long, device=inputs.device)
        eps_threshold = torch.finfo(inputs.dtype).eps * initial_sum

        for j in range(num_inducing):
            total = residual.sum()
            if total <= eps_threshold:
                remaining = (~selected).nonzero(as_tuple=False).squeeze(-1)
                order = torch.randperm(remaining.numel(), device=inputs.device)
                tail = remaining[order]
                indices[j:] = tail[: num_inducing - j]
                break

            # multinomial accepts nonnegative unnormalized weights.
            pivot = torch.multinomial(residual, 1).squeeze(0)
            indices[j] = pivot
            selected[pivot] = True
            if j == num_inducing - 1:
                break

            column_op = covar_module(inputs, inputs[pivot : pivot + 1])
            column = column_op.to_dense().reshape(-1)
            if column.shape != torch.Size([n]) or not torch.isfinite(column).all():
                raise UnsupportedError("kernel columns must be finite and unbatched.")
            column = column / scale
            if j:
                column = column - factor[:, :j] @ factor[pivot, :j]
            pivot_value = residual[pivot]
            if not torch.isfinite(pivot_value) or pivot_value <= 0:
                raise UnsupportedError("sampled pivot must have positive residual.")
            factor[:, j] = column / pivot_value.sqrt()
            residual = (residual - factor[:, j].square()).clamp_min(0)
            residual[selected] = 0

        return inputs[indices].detach()
