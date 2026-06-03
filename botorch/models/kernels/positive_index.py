#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from gpytorch.constraints import GreaterThan, Interval, Positive
from gpytorch.kernels import IndexKernel, Kernel
from gpytorch.priors import Prior


class PositiveIndexKernel(IndexKernel):
    r"""
    A kernel for discrete indices with strictly positive correlations. This is
    enforced by a positivity constraint on the decomposed covariance matrix.

    Similar to IndexKernel but ensures all off-diagonal correlations are positive
    by using a Cholesky-like parameterization with positive elements.

    .. math::
        k(i, j) = \frac{(LL^T)_{i,j}}{(LL^T)_{t,t}}

    where L is a lower triangular matrix with positive elements and t is the
    target_task_index.
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 1,
        task_prior: Prior | None = None,
        normalize_covar_matrix: bool = False,
        var_constraint: Interval | None = None,
        target_task_index: int = 0,
        unit_scale_for_target: bool = True,
        **kwargs,
    ):
        r"""A kernel for discrete indices with strictly positive correlations.

        Args:
            num_tasks (int): Total number of indices.
            rank (int): Rank of the covariance matrix parameterization.
            task_prior (Prior, optional): Prior for the covariance matrix.
            normalize_covar_matrix (bool): Whether to normalize the covariance matrix.
            target_task_index (int): Index of the task whose diagonal element should be
                normalized to 1. Defaults to 0 (first task).
            unit_scale_for_target (bool): Whether to ensure the target task's has unit
                outputscale.
            **kwargs: Additional arguments passed to IndexKernel.
        """
        if rank > num_tasks:
            raise RuntimeError(
                "Cannot create a task covariance matrix larger than the number of tasks"
            )
        if not (0 <= target_task_index < num_tasks):
            raise ValueError(
                f"target_task_index must be between 0 and {num_tasks - 1}, "
                f"got {target_task_index}"
            )
        Kernel.__init__(self, **kwargs)

        if var_constraint is None:
            var_constraint = Positive()

        self.register_parameter(
            name="raw_var",
            parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)),
        )
        self.register_constraint("raw_var", var_constraint)
        # delete covar factor from parameters
        self.normalize_covar_matrix = normalize_covar_matrix
        self.num_tasks = num_tasks
        self.target_task_index = target_task_index
        self.register_parameter(
            name="raw_covar_factor",
            parameter=torch.nn.Parameter(
                torch.rand(*self.batch_shape, num_tasks, rank)
            ),
        )
        self.unit_scale_for_target = unit_scale_for_target
        if task_prior is not None:
            if not isinstance(task_prior, Prior):
                raise TypeError(
                    f"Expected gpytorch.priors.Prior but got "
                    f"{type(task_prior).__name__}"
                )
            self.register_prior(
                "IndexKernelPrior",
                task_prior,
                lambda m: m._lower_triangle_corr,
                lambda m, v: m._set_lower_triangle_corr(v),
            )
        self.register_constraint("raw_covar_factor", GreaterThan(0.0))

    def _covar_factor_params(self, m):
        return m.covar_factor

    def _covar_factor_closure(self, m, v):
        m._set_covar_factor(v)

    @property
    def covar_factor(self):
        return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)

    @covar_factor.setter
    def covar_factor(self, value):
        self._set_covar_factor(value)

    def _set_covar_factor(self, value):
        # This must be a tensor
        self.initialize(
            raw_covar_factor=self.raw_covar_factor_constraint.inverse_transform(value)
        )

    @property
    def _lower_triangle_corr(self):
        lower_row, lower_col = torch.tril_indices(
            self.num_tasks, self.num_tasks, offset=-1
        )
        covar = self.covar_matrix
        norm_factor = covar.diagonal(dim1=-1, dim2=-2).sqrt()
        corr = covar / (norm_factor.unsqueeze(-1) * norm_factor.unsqueeze(-2))
        low_tri = corr[..., lower_row, lower_col]

        return low_tri

    def _set_lower_triangle_corr(self, value):
        """Set covar_factor to produce the given lower-triangle correlations.

        Assembles a symmetric correlation matrix from the lower-triangle values,
        then projects it to the nearest positive-definite correlation matrix via
        eigenvalue clamping before Cholesky decomposition. This guarantees the
        setter never fails even when independently sampled correlation values
        do not form a PD matrix.

        Args:
            value: Tensor of lower-triangle correlation values.
        """
        n = self.num_tasks
        eps = 1e-6
        n_lower = n * (n - 1) // 2
        lower_row, lower_col = torch.tril_indices(n, n, offset=-1)
        # Expand under-batched input (e.g. scalar from sample_from_prior)
        if value.dim() == 0 or (value.dim() == 1 and value.shape[0] != n_lower):
            value = value.unsqueeze(-1).expand(*self.batch_shape, n_lower)
        elif value.shape[:-1] != self.batch_shape:
            value = value.expand(*self.batch_shape, n_lower)
        batch_shape = value.shape[:-1]
        corr = (
            torch.eye(n, dtype=value.dtype, device=value.device)
            .expand(*batch_shape, n, n)
            .clone()
        )
        corr[..., lower_row, lower_col] = value.clamp(0.0, 1.0)
        corr[..., lower_col, lower_row] = value.clamp(0.0, 1.0)
        # Project to nearest PD correlation matrix via eigenvalue clamping
        eigvals, eigvecs = torch.linalg.eigh(corr)
        eigvals = eigvals.clamp(min=eps)
        corr = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        # Re-normalize diagonals to 1
        d = corr.diagonal(dim1=-1, dim2=-2).sqrt()
        corr = corr / (d.unsqueeze(-1) * d.unsqueeze(-2))
        chol = torch.linalg.cholesky(corr)
        rank = self.raw_covar_factor.shape[-1]
        self._set_covar_factor(chol[..., :, :rank].clamp(min=eps))

    def _eval_covar_matrix(self):
        cf = self.covar_factor
        covar = cf @ cf.transpose(-1, -2) + torch.diag_embed(self.var)
        # Normalize by the target task's diagonal element
        if self.unit_scale_for_target:
            norm_factor = covar[..., self.target_task_index, self.target_task_index]
            covar = covar / norm_factor.unsqueeze(-1).unsqueeze(-1)
        return covar

    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
