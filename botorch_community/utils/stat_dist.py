#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for computing statistical distances between Gaussian distributions.

Contributor: hvarfner
"""

import torch
from torch import Tensor


def mvn_kl_divergence(
    p_mean: Tensor, q_mean: Tensor, p_covar: Tensor, q_covar: Tensor
) -> Tensor:
    """Computes the KL-divergence between two multivariate normal distributions.
    Args:
        p_mean: A `batch_shape x dist_shape x 1`-dim Tensor of means of
            the first distribution.
        q_mean: A `batch_shape x dist_shape x dist_shape`-dim Tensor of
            covariances of the first distribution,  where the covariances
            are (optionally) in the q-batch dim.
        p_mean: A `batch_shape x dist_shape x 1`-dim Tensor of means of
            the second distribution.
        q_mean: A `batch_shape x dist_shape x dist_shape`-dim Tensor of
            covariances of the second distribution where the covariances
            are (optionally) in the q-batch dim.
    Returns:
        A tensor of shape `batch_shape x dist_shape` denoting the KL-divergence
        between the multivariate Gaussian distributions p and q.
    """
    p_inv_covar = torch.inverse(p_covar)
    mean_diff = p_mean - q_mean
    kl_first_term = torch.diagonal(
        torch.matmul(p_inv_covar, q_covar), dim1=-2, dim2=-1
    ).sum(-1, keepdim=True)
    kl_second_term = torch.matmul(
        torch.matmul(mean_diff.transpose(-2, -1), p_inv_covar), mean_diff
    ).squeeze(-1)
    kl_third_term = (torch.logdet(p_covar) - torch.logdet(q_covar)).unsqueeze(-1)
    return 0.5 * (kl_first_term + kl_second_term + kl_third_term - p_mean.shape[-2])


def mvn_hellinger_distance(
    p_mean: Tensor, q_mean: Tensor, p_covar: Tensor, q_covar: Tensor
) -> Tensor:
    """Computes the (2)-Hellinger distance between two multivariate normal
        distributions.
    Args:
        p_mean: A `batch_shape x dist_shape x 1`-dim Tensor of means of
            the first distribution.
        q_mean: A `batch_shape x dist_shape x dist_shape`-dim Tensor of
            covariances of the first distribution,  where the covariances
            are (optionally) in the q-batch dim.
        p_mean: A `batch_shape x dist_shape x 1`-dim Tensor of means of
            the second distribution.
        q_mean: A `batch_shape x dist_shape x dist_shape`-dim Tensor of
            covariances of the second distribution where the covariances
            are (optionally) in the q-batch dim.
    Returns:
        A tensor of shape `batch_shape x dist_shape` denoting the KL-divergence
        between the multivariate Gaussian distributions p and q.
    """
    p_logdet = torch.logdet(p_covar).unsqueeze(-1)
    q_logdet = torch.logdet(q_covar).unsqueeze(-1)
    avg_covar = (p_covar + q_covar) / 2

    # we need to re-use the cholesky decomp, so we compute it once here
    L_avg = torch.linalg.cholesky(avg_covar)
    L_inv = torch.inverse(L_avg)

    # removes one dimension, which needs to be recouped
    pq_logdet = (
        torch.pow(torch.diagonal(L_avg, dim1=-2, dim2=-1), 2)
        .prod(dim=-1, keepdim=True)
        .log()
    )
    base_logterm = 0.25 * (p_logdet + q_logdet) - 0.5 * pq_logdet

    mean_diff = p_mean - q_mean
    L_mean_diff = torch.matmul(L_inv, mean_diff)
    exp_logterm = -0.125 * torch.matmul(L_mean_diff.transpose(-2, -1), L_mean_diff)
    sq_hdist = 1 - (base_logterm + exp_logterm.squeeze(-1)).exp()
    return sq_hdist.clamp_min(0.0).sqrt()
