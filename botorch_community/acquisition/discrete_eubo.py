#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Vectorized analytic Expected Utility of the Best Option (EUBO) pair selector
for discrete candidate pools (e.g., retrieval-augmented or catalog-based
preferential Bayesian optimization).

References:

.. [Lin2022preference]
    Z. J. Lin, R. Astudillo, P. I. Frazier, and E. Bakshy. Preference
    Exploration for Efficient Bayesian Optimization with Multiple Outcomes.
    International Conference on Artificial Intelligence and Statistics
    (AISTATS), 2022.

.. [Astudillo2023qeubo]
    R. Astudillo, Z. J. Lin, E. Bakshy, and P. I. Frazier. qEUBO: A
    Decision-Theoretic Acquisition Function for Preferential Bayesian
    Optimization. AISTATS, 2023.

Contributor: M-Colley
"""

from __future__ import annotations

import math

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from torch import Tensor

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def select_discrete_eubo_pair(
    pref_model: Model,
    candidates: Tensor,
    compared_pairs: Tensor | None = None,
    sigma_jitter: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Select the optimal candidate pair ``(i*, j*)`` from a discrete pool via EUBO.

    Evaluates the joint preference posterior over all ``n`` discrete ``candidates``
    in a single pass and computes exact closed-form ``E[max(u_i, u_j)]`` across all
    ``n x n`` pairs without continuous multi-start L-BFGS optimization.

    Args:
        pref_model: A preference ``Model`` (such as ``PairwiseGP`` or
            ``DualPairwiseGP``) mapping ``(n, d)`` candidates to scalar utilities.
        candidates: A ``n x d`` tensor of discrete candidate points with ``n >= 2``.
        compared_pairs: Optional ``m x 2`` integer tensor of already-queried index
            pairs ``[i, j]`` to exclude (symmetrically) from selection.
        sigma_jitter: Minimum variance floor on ``Var(u_i - u_j)`` for stability.

    Returns:
        A 3-tuple ``(best_pair, best_eubo, eubo_matrix)`` where:
        - ``best_pair``: LongTensor of shape ``(2,)`` containing ``[idx_a, idx_b]``
          ordered such that ``mean[idx_a] >= mean[idx_b]``.
        - ``best_eubo``: Scalar tensor with ``E[max(u_{idx_a}, u_{idx_b})]``.
        - ``eubo_matrix``: Full symmetric ``(n, n)`` EUBO tensor (prior to masking).
    """
    if candidates.ndim != 2 or candidates.shape[0] < 2:
        raise UnsupportedError(
            "`candidates` must have shape `(n, d)` with at least `n >= 2` items."
        )

    n = candidates.shape[0]
    with torch.no_grad():
        posterior = pref_model.posterior(candidates)
        mean = posterior.mean.squeeze(-1)  # (n,)
        cov = posterior.covariance_matrix  # (n, n)
        diag_var = torch.diagonal(cov, dim1=-2, dim2=-1)

        delta = mean.unsqueeze(1) - mean.unsqueeze(0)  # (n, n)
        diff_var = (diag_var.unsqueeze(1) + diag_var.unsqueeze(0) - 2.0 * cov).clamp(
            min=sigma_jitter
        )
        sigma = torch.sqrt(diff_var)
        z = delta / sigma

        cdf = 0.5 * (1.0 + torch.erf(z * INV_SQRT_2))
        pdf = INV_SQRT_2PI * torch.exp(-0.5 * z * z)
        eubo_matrix = mean.unsqueeze(0) + delta * cdf + sigma * pdf

        # Mask diagonal (self-duels) and upper/lower duplicate symmetries
        scored = eubo_matrix.clone()
        tri_mask = torch.tril(
            torch.ones((n, n), dtype=torch.bool, device=candidates.device)
        )
        scored.masked_fill_(tri_mask, float("-inf"))

        if compared_pairs is not None and compared_pairs.numel() > 0:
            if compared_pairs.ndim != 2 or compared_pairs.shape[1] != 2:
                raise UnsupportedError("`compared_pairs` must have shape `(m, 2)`.")
            pairs = compared_pairs.long().to(candidates.device)
            lo = torch.minimum(pairs[:, 0], pairs[:, 1])
            hi = torch.maximum(pairs[:, 0], pairs[:, 1])
            valid = (lo >= 0) & (hi < n)
            scored[lo[valid], hi[valid]] = float("-inf")

        flat_idx = int(torch.argmax(scored).item())
        best_val = scored.view(-1)[flat_idx]
        if not torch.isfinite(best_val):
            raise UnsupportedError(
                "No uncompared candidate pairs remain in `candidates`."
            )

        idx_i = flat_idx // n
        idx_j = flat_idx % n
        if mean[idx_i] >= mean[idx_j]:
            best_pair = torch.tensor(
                [idx_i, idx_j], dtype=torch.long, device=candidates.device
            )
        else:
            best_pair = torch.tensor(
                [idx_j, idx_i], dtype=torch.long, device=candidates.device
            )

        return best_pair, best_val, eubo_matrix
