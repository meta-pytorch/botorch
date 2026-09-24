#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stein Variational Gradient Descent (SVGD) utilities.

This module provides a general-purpose SVGD implementation for approximating
target distributions as a set of particles. It is used by PACOH-GP to
approximate the PAC-optimal hyper-posterior Q* over GP prior parameters.

References:
    .. [Liu2016svgd]
        Q. Liu and D. Wang.
        Stein Variational Gradient Descent: A General Purpose Bayesian
        Inference Algorithm. NeurIPS, 2016.

    .. [Rothfuss2021pacoh]
        J. Rothfuss, V. Fortuin, M. Josifoski, and A. Krause.
        PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees.
        ICML, 2021.
"""

from __future__ import annotations

import torch
from torch import Tensor


def svgd_kernel(
    particles: Tensor,
    length_scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the RBF kernel matrix and its gradients for SVGD.

    K[k,k'] = exp(-||φ_k - φ_k'||² / (2ℓ²))

    Args:
        particles: `K x D` tensor of particle parameter vectors.
        length_scale: RBF kernel length scale ℓ. If None, uses the median
            heuristic: ℓ² = median(pairwise distances²) / (2 * ln(K)).

    Returns:
        Tuple of (kernel_matrix, kernel_grads):
        - kernel_matrix: `K x K` RBF kernel matrix.
        - kernel_grads: `K x K x D` tensor where kernel_grads[k, k', :] =
          ∇_{φ_k} K(φ_k, φ_k').
    """
    K = particles.shape[0]

    # Pairwise squared distances: ||φ_k - φ_k'||²
    # Shape: (K, K)
    diffs = particles.unsqueeze(0) - particles.unsqueeze(1)  # (K, K, D)
    sq_dists = (diffs**2).sum(dim=-1)  # (K, K)

    # Median heuristic for length scale
    if length_scale is None:
        positive_dists = sq_dists[sq_dists > 0]
        if positive_dists.numel() == 0:
            # K=1 or all particles identical: use a default length scale.
            # The kernel matrix will be all ones, so h doesn't matter.
            h = 1.0
        else:
            median_sq_dist = torch.median(positive_dists)
            h = median_sq_dist / (
                2.0 * max(torch.log(torch.tensor(float(K))).item(), 1.0)
            )
    else:
        h = length_scale**2

    # RBF kernel matrix
    kernel_matrix = torch.exp(-sq_dists / (2.0 * h + 1e-8))  # (K, K)

    # Gradient of kernel w.r.t. first argument:
    # ∇_{φ_k} K(φ_k, φ_k') = K(φ_k, φ_k') * (φ_k' - φ_k) / h
    # diffs[k, k'] = φ_k' - φ_k, so this is just K * diffs / h
    kernel_grads = kernel_matrix.unsqueeze(-1) * diffs / (h + 1e-8)  # (K, K, D)

    return kernel_matrix, kernel_grads


def svgd_update(
    particles: Tensor,
    score_grads: Tensor,
    kernel_fn: type[svgd_kernel] | None = None,
    step_size: float = 1e-3,
    length_scale: float | None = None,
) -> Tensor:
    """Execute one SVGD update step.

    Updates particles according to the SVGD rule (Liu & Wang, 2016):
        φ ← φ + (η/K) * (K @ ∇_φ ln Q* + ∇_φ K)

    The first term (attractive) drives particles toward high-density regions
    of the target distribution. The second term (repulsive) maintains
    diversity by pushing particles apart.

    Args:
        particles: `K x D` tensor of current particle parameter vectors.
        score_grads: `K x D` tensor of pre-computed score gradients
            ∇_φ ln Q*(φ_k) for each particle.
        kernel_fn: Callable that computes the SVGD kernel. If None, uses
            the default `svgd_kernel` function.
        step_size: Learning rate η for the update.
        length_scale: Length scale for the RBF kernel. If None, uses
            median heuristic.

    Returns:
        `K x D` tensor of updated particle parameter vectors.
    """
    K = particles.shape[0]

    # Compute kernel matrix and gradients
    if kernel_fn is None:
        kernel_matrix, kernel_grads = svgd_kernel(particles, length_scale)
    else:
        kernel_matrix, kernel_grads = kernel_fn(particles, length_scale)

    # SVGD update direction:
    # φ_update[k] = (1/K) * Σ_{k'} [K(φ_k', φ_k) * ∇_{φ_k'} ln Q*(φ_k')
    #                                + ∇_{φ_k'} K(φ_k', φ_k)]
    #
    # Attractive term: K @ score_grads / K  (K x K) @ (K x D) -> (K x D)
    attractive = kernel_matrix @ score_grads / K

    # Repulsive term: sum over k' of ∇_{φ_k'} K(φ_k', φ_k) / K
    # kernel_grads has shape (K, K, D) with grads[k', k, :] = ∇_{φ_k'} K(φ_k', φ_k)
    # We want sum over k' for each k, so sum over dim 0
    repulsive = kernel_grads.sum(dim=0) / K  # (K, D)

    # Apply update
    updated_particles = particles + step_size * (attractive + repulsive)

    return updated_particles
