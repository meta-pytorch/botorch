#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import wraps

import torch
from torch import Tensor


def lognorm_to_norm(mu: Tensor, Cov: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance of a MVN from those of the associated log-MVN

    If `Y` is log-normal with mean mu_ln and covariance Cov_ln, then
    `X ~ N(mu_n, Cov_n)` with

        Cov_n_{ij} = log(1 + Cov_ln_{ij} / (mu_ln_{i} * mu_n_{j}))
        mu_n_{i} = log(mu_ln_{i}) - 0.5 * log(1 + Cov_ln_{ii} / mu_ln_{i}**2)

    Args:
        mu: A `batch_shape x n` mean vector of the log-Normal distribution.
        Cov: A `batch_shape x n x n` covariance matrix of the log-Normal
            distribution.

    Returns:
        A two-tuple containing:

        - The `batch_shape x n` mean vector of the Normal distribution
        - The `batch_shape x n x n` covariance matrix of the Normal distribution
    """
    Cov_n = torch.log1p(Cov / (mu.unsqueeze(-1) * mu.unsqueeze(-2)))
    mu_n = torch.log(mu) - 0.5 * torch.diagonal(Cov_n, dim1=-1, dim2=-2)
    return mu_n, Cov_n


def norm_to_lognorm(mu: Tensor, Cov: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance of a log-MVN from its MVN sufficient statistics

    If `X ~ N(mu, Cov)` and `Y = exp(X)`, then `Y` is log-normal with

        mu_ln_{i} = exp(mu_{i} + 0.5 * Cov_{ii})
        Cov_ln_{ij} = exp(mu_{i} + mu_{j} + 0.5 * (Cov_{ii} + Cov_{jj})) *
        (exp(Cov_{ij}) - 1)

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        Cov: A `batch_shape x n x n` covariance matrix of the Normal distribution.

    Returns:
        A two-tuple containing:

        - The `batch_shape x n` mean vector of the log-Normal distribution.
        - The `batch_shape x n x n` covariance matrix of the log-Normal
            distribution.
    """
    diag = torch.diagonal(Cov, dim1=-1, dim2=-2)
    b = mu + 0.5 * diag
    mu_ln = torch.exp(b)
    Cov_ln = torch.special.expm1(Cov) * torch.exp(b.unsqueeze(-1) + b.unsqueeze(-2))
    return mu_ln, Cov_ln


def norm_to_lognorm_mean(mu: Tensor, var: Tensor) -> Tensor:
    """Compute mean of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` mean vector of the log-Normal distribution.
    """
    return torch.exp(mu + 0.5 * var)


def norm_to_lognorm_variance(mu: Tensor, var: Tensor) -> Tensor:
    """Compute variance of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` variance vector of the log-Normal distribution.
    """
    b = mu + 0.5 * var
    return torch.special.expm1(var) * torch.exp(2 * b)


def expand_and_copy_tensor(X: Tensor, batch_shape: torch.Size) -> Tensor:
    r"""Expand and copy X according to batch_shape.

    Args:
        X: A `input_batch_shape x n x d`-dim tensor of inputs.
        batch_shape: The new batch shape.

    Returns:
        A `new_batch_shape x n x d`-dim tensor of inputs, where `new_batch_shape`
        is `input_batch_shape` against `batch_shape`.
    """
    try:
        batch_shape = torch.broadcast_shapes(X.shape[:-2], batch_shape)
    except RuntimeError:
        raise RuntimeError(
            f"Provided batch shape ({batch_shape}) and input batch shape "
            f"({X.shape[:-2]}) are not broadcastable."
        )
    expand_shape = batch_shape + X.shape[-2:]
    return X.expand(expand_shape).clone()


def subset_transform(transform):
    r"""Decorator of an input transform function to separate out indexing logic."""

    @wraps(transform)
    def f(self, X: Tensor, **kwargs) -> Tensor:
        if not hasattr(self, "indices") or self.indices is None:
            return transform(self, X, **kwargs)
        has_shape = hasattr(self, "batch_shape")
        Y = expand_and_copy_tensor(X, self.batch_shape) if has_shape else X.clone()
        Y[..., self.indices] = transform(self, X[..., self.indices], **kwargs)
        return Y

    return f


def interaction_features(X: Tensor) -> Tensor:
    """Computes the interaction features between the inputs.

    Args:
        X: A `batch_shape x q x d`-dim tensor of inputs.
        indices: The input dimensions to generate interaction features for.

    Returns:
        A `n x q x 1 x (d * (d-1) / 2))`-dim tensor of interaction features.
    """
    dim = X.shape[-1]
    row_idcs, col_idcs = torch.triu_indices(dim, dim, offset=1)
    return (X.unsqueeze(-1) @ X.unsqueeze(-2))[..., row_idcs, col_idcs].unsqueeze(-2)


def nanstd(X: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Computes the standard deviation of the input, ignoring NaNs.

    Args:
        X: A `batch_shape x n x d`-dim tensor of inputs.
        dim: The dimension along which to compute the standard deviation.
        keepdim: If True, the dimension along which the standard deviation is
            compute is kept.
    """
    n = (~torch.isnan(X)).sum(dim=dim, keepdim=keepdim)
    return (
        (X - X.nanmean(dim=dim, keepdim=True)).pow(2).nanmean(dim=dim, keepdim=keepdim)
        * n
        / (n - 1)
    ).sqrt()


def kumaraswamy_warp(X: Tensor, c0: Tensor, c1: Tensor, eps: float = 1e-8) -> Tensor:
    """Warp inputs through a Kumaraswamy CDF.

    This assumes that X is contained within the unit cube. This first
    normalizes inputs to [eps, 1-eps]^d (to ensure that no values are 0 or 1)
    and then applies passes those inputs through a Kumaraswamy CDF.

    Args:
        X: A `batch_shape x n x d`-dim tensor of inputs.
        c0: A `d`-dim tensor of the concentration0 parameter for the
            Kumaraswamy distribution.
        c1: A `d`-dim tensor of the concentration1 parameter for the
            Kumaraswamy distribution.
        eps: A small value that is used to ensure inputs are not 0 or 1.

    Returns:
        A `batch_shape x n x d`-dim tensor of warped inputs.
    """
    X_range = 1 - 2 * eps
    X = torch.clamp(X * X_range + eps, eps, 1.0 - eps)
    return 1 - torch.pow((1 - torch.pow(X, c1)), c0)


def inv_kumaraswamy_warp(
    X: Tensor, c0: Tensor, c1: Tensor, eps: float = 1e-8
) -> Tensor:
    """Map warped inputs through an inverse Kumaraswamy CDF.

    This takes warped inputs (X) and transforms those via an inverse
    Kumaraswamy CDF. This then unnormalizes the inputs using bounds of
    [eps, 1-eps]^d and ensures that the values are within [0, 1]^d.

    Args:
        X: A `batch_shape x n x d`-dim tensor of inputs.
        c0: A `d`-dim tensor of the concentration0 parameter for the
            Kumaraswamy distribution.
        c1: A `d`-dim tensor of the concentration1 parameter for the
            Kumaraswamy distribution.
        eps: A small value that is used to ensure inputs are not 0 or 1.

    Returns:
        A `batch_shape x n x d`-dim tensor of untransformed inputs.
    """
    X_range = 1 - 2 * eps
    # unnormalize from [eps, 1-eps] to [0,1]
    untf_X = (1 - (1 - X).pow(1 / c0)).pow(1 / c1)
    return ((untf_X - eps) / X_range).clamp(0.0, 1.0)
