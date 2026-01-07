#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchError
from botorch.posteriors.base_samples import _reshape_base_samples_non_interleaved
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from linear_operator.operators import BlockDiagLinearOperator, LinearOperator

from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NanError
from torch import Tensor


def extract_batch_covar(mt_mvn: MultitaskMultivariateNormal) -> LinearOperator:
    r"""Extract a batched independent covariance matrix from an MTMVN.

    Args:
        mt_mvn: A multi-task multivariate normal with a block diagonal
            covariance matrix.

    Returns:
        A lazy covariance matrix consisting of a batch of the blocks of
            the diagonal of the MultitaskMultivariateNormal.

    """
    lazy_covar = mt_mvn.lazy_covariance_matrix
    if not isinstance(lazy_covar, BlockDiagLinearOperator):
        raise BotorchError(
            f"Expected BlockDiagLinearOperator, but got {type(lazy_covar)}."
        )
    return lazy_covar.base_linear_op


def _reshape_base_samples(
    base_samples: Tensor, sample_shape: torch.Size, posterior: GPyTorchPosterior
) -> Tensor:
    r"""Manipulate shape of base_samples to match `MultivariateNormal.rsample`.

    This ensure that base_samples are used in the same way as in
    gpytorch.distributions.MultivariateNormal. For CBD, it is important to ensure
    that the same base samples are used for the in-sample points here and in the
    cached box decompositions.

    Args:
        base_samples: The base samples.
        sample_shape: The sample shape.
        posterior: The joint posterior is over (X_baseline, X).

    Returns:
        Reshaped and expanded base samples.
    """
    mvn = posterior.distribution
    loc = mvn.loc
    peshape = posterior._extended_shape()
    base_samples = base_samples.view(
        sample_shape + torch.Size([1] * (loc.ndim - 1)) + peshape[-2:]
    ).expand(sample_shape + loc.shape[:-1] + peshape[-2:])
    if posterior._is_mt:
        base_samples = _reshape_base_samples_non_interleaved(
            mvn=posterior.distribution,
            base_samples=base_samples,
            sample_shape=sample_shape,
        )
    base_samples = base_samples.reshape(
        -1, *loc.shape[:-1], mvn.lazy_covariance_matrix.shape[-1]
    )
    base_samples = base_samples.permute(*range(1, loc.dim() + 1), 0)
    return base_samples.reshape(
        *peshape[:-2],
        peshape[-1],
        peshape[-2],
        *sample_shape,
    )


def sample_cached_cholesky(
    posterior: GPyTorchPosterior,
    baseline_L: Tensor,
    q: int,
    base_samples: Tensor,
    sample_shape: torch.Size,
    max_tries: int = 6,
) -> Tensor:
    r"""Get posterior samples at the `q` new points from the joint multi-output
    posterior.

    Args:
        posterior: The joint posterior is over (X_baseline, X).
        baseline_L: The baseline lower triangular cholesky factor.
        q: The number of new points in X.
        base_samples: The base samples.
        sample_shape: The sample shape.
        max_tries: The number of tries for computing the Cholesky
            decomposition with increasing jitter.


    Returns:
        A `sample_shape x batch_shape x q x m`-dim tensor of posterior
            samples at the new points.
    """
    # compute bottom left covariance block
    mvn = posterior.distribution
    lazy_covar = (
        extract_batch_covar(mt_mvn=mvn)
        if isinstance(mvn, MultitaskMultivariateNormal)
        else mvn.lazy_covariance_matrix
    )
    # Get the `q` new rows of the batched covariance matrix
    bottom_rows = lazy_covar[..., -q:, :].to_dense()
    # The covariance in block form is:
    # [K(X_baseline, X_baseline), K(X_baseline, X)]
    # [K(X, X_baseline), K(X, X)]
    # bl := K(X, X_baseline)
    # br := K(X, X)
    # Get bottom right block of new covariance
    bl, br = bottom_rows.split([bottom_rows.shape[-1] - q, q], dim=-1)
    # Solve Ax = b
    # where A = K(X_baseline, X_baseline) and b = K(X, X_baseline)^T
    # and bl_chol := x^T
    # bl_chol is the new `(batch_shape) x q x n`-dim bottom left block
    # of the cholesky decomposition
    bl_chol = torch.linalg.solve_triangular(
        baseline_L, bl.transpose(-2, -1), upper=False
    ).transpose(-2, -1)

    # Compute the new bottom right block of the Cholesky
    # decomposition via:
    # Cholesky(K(X, X) - bl_chol @ bl_chol^T)
    br_to_chol = br - bl_chol @ bl_chol.transpose(-2, -1)
    # TODO: technically we should make sure that we add a
    # consistent nugget to the cached covariance and the new block
    br_chol = psd_safe_cholesky(br_to_chol, max_tries=max_tries)
    # Create a `(batch_shape) x q x (n+q)`-dim tensor containing the
    # `q` new bottom rows of the Cholesky decomposition
    new_Lq = torch.cat([bl_chol, br_chol], dim=-1)
    mean = posterior.distribution.mean
    base_samples = _reshape_base_samples(
        base_samples=base_samples,
        sample_shape=sample_shape,
        posterior=posterior,
    )
    if not isinstance(posterior.distribution, MultitaskMultivariateNormal):
        # add output dim
        mean = mean.unsqueeze(-1)
        # add batch dim corresponding to output dim
        new_Lq = new_Lq.unsqueeze(-3)
    new_mean = mean[..., -q:, :]
    res = (
        new_Lq.matmul(base_samples)
        .add(new_mean.transpose(-1, -2).unsqueeze(-1))
        .permute(-1, *range(posterior.distribution.loc.dim() - 1), -2, -3)
        .contiguous()
    )
    contains_nans = torch.isnan(res).any()
    contains_infs = torch.isinf(res).any()
    if contains_nans or contains_infs:
        suffix_args = []
        if contains_nans:
            suffix_args.append("nans")
        if contains_infs:
            suffix_args.append("infs")
        suffix = " and ".join(suffix_args)
        raise NanError(f"Samples contain {suffix}.")
    return res
