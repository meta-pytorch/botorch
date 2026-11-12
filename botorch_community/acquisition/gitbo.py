#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gradient-Informed Bayesian Optimization (GIT-BO).

GIT-BO [yu2025gitbo]_ is a high-dimensional Bayesian optimization strategy
that aligns the search with a low-dimensional *active subspace*
[constantine2014active]_ estimated from the surrogate model's own
predictive-mean input gradients. Each iteration:

1. A large discrete candidate set is scored with a quantile-based UCB
   acquisition (the posterior quantile at level ``quantile``; for a Gaussian
   posterior this equals the classic UCB ``mu + sqrt(beta) * sigma`` with
   ``beta = z_quantile**2``).
2. During the same forward pass, the Jacobian ``G`` of the posterior mean
   with respect to the candidate inputs is obtained with a single
   ``torch.autograd.grad`` call.
3. On the next iteration, the empirical second-moment matrix
   ``H = G^T G / N`` is eigendecomposed and new candidates are sampled
   inside the span of the top eigenvectors (the gradient-informed
   subspace), centered at the mean of the evaluated points.

The first iteration (no gradient information yet) falls back to quasi-random
Sobol candidates over the full design space.

The implementation is surrogate-agnostic: any BoTorch ``Model`` whose
posterior mean is differentiable with respect to the test inputs can be used,
including ``SingleTaskGP`` and the community ``PFNModel``
(``botorch_community.models.prior_fitted_network``). The scoring is
deliberately implemented as plain functions rather than an
``AcquisitionFunction`` subclass: GIT-BO evaluates a fixed discrete candidate
set pointwise (it is not optimized with ``optimize_acqf``), and the
input-gradient Jacobian is a second output that the ``forward(X) -> Tensor``
acquisition interface cannot express.

References

.. [yu2025gitbo]
    R. T.-Y. Yu, C. Picard, F. Ahmed. GIT-BO: High-Dimensional Bayesian
    Optimization with Tabular Foundation Models. International Conference
    on Learning Representations, 2026. arXiv:2505.20685.
.. [constantine2014active]
    P. G. Constantine, E. Dow, Q. Wang. Active Subspace Methods in Theory
    and Practice: Applications to Kriging Surfaces. SIAM Journal on
    Scientific Computing, 2014.

Contributor: rosenyu304
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor


class GITBOStepResult(NamedTuple):
    """Result of one GIT-BO iteration.

    Attributes:
        candidate: A ``1 x d`` tensor with the candidate maximizing the
            acquisition, to be evaluated next.
        acq_values: A ``num_candidates``-dim tensor of quantile-UCB scores.
        candidate_set: A ``num_candidates x d`` tensor of the scored points.
        gradients: A ``num_candidates x d`` tensor with the posterior-mean
            input gradients. Pass this to the next ``gitbo_step`` call to
            construct the gradient-informed subspace.
        subspace: A ``d x r`` tensor of subspace basis vectors used to sample
            this step's candidates, or ``None`` on the Sobol (first) step.
        eigenvalues: A ``d``-dim tensor of eigenvalues of ``H = G^T G / N``
            in descending order, or ``None`` on the Sobol (first) step.
    """

    candidate: Tensor
    acq_values: Tensor
    candidate_set: Tensor
    gradients: Tensor
    subspace: Tensor | None
    eigenvalues: Tensor | None


def quantile_ucb(
    model: Model,
    X: Tensor,
    quantile: float = 0.975,
    compute_mean_gradients: bool = True,
    eval_in_q_batch: bool = False,
    batch_limit: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    r"""Score candidates with a posterior-quantile UCB acquisition.

    The score of a candidate ``x`` is the ``quantile``-level quantile of the
    posterior over ``f(x)``. For a Gaussian posterior this is exactly the
    classic UCB ``mu + sqrt(beta) * sigma`` with ``beta = z_quantile**2``
    (e.g. ``quantile=0.975`` corresponds to ``beta ~= 3.84``). Posteriors
    that implement ``quantile`` (e.g. ``GPyTorchPosterior``) or ``icdf``
    (e.g. ``BoundedRiemannPosterior``) are supported.

    As a side product, the Jacobian of the posterior mean with respect to
    ``X`` is computed with a single ``torch.autograd.grad`` call. This is
    valid because each candidate is evaluated as an independent test point,
    so the gradient of the summed means recovers the per-candidate rows.

    Args:
        model: A fitted single-outcome model. Its posterior mean must be
            differentiable with respect to the test inputs if
            ``compute_mean_gradients=True``.
        X: A ``num_candidates x d`` tensor of candidate points.
        quantile: The quantile level in (0, 1) used as the UCB score.
        compute_mean_gradients: If ``True``, also return the
            ``num_candidates x d`` posterior-mean input gradients.
        eval_in_q_batch: If ``True``, evaluate the posterior on ``X``
            directly so all candidates form one q-batch. This is the
            memory-efficient layout for PFN models, which then encode the
            training context only once. If ``False``, the posterior is
            evaluated on ``X.unsqueeze(-2)`` (``num_candidates x 1 x d``),
            the appropriate layout for GP models, where it yields
            independent ``q=1`` marginal posteriors.
        batch_limit: If given, candidates are pushed through the posterior
            in chunks of at most ``batch_limit`` points to bound memory.

    Returns:
        A two-element tuple containing:

        - A ``num_candidates``-dim tensor of acquisition scores (detached).
        - A ``num_candidates x d`` tensor of posterior-mean gradients, or
          ``None`` if ``compute_mean_gradients=False``.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be `num_candidates x d`, got shape {X.shape}.")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
    num_candidates = X.shape[0]
    chunk_size = num_candidates if batch_limit is None else batch_limit
    if chunk_size < 1:
        raise ValueError(f"batch_limit must be positive, got {batch_limit}.")

    all_scores, all_grads = [], []
    for X_chunk in X.split(chunk_size):
        X_eval = X_chunk.detach().clone().requires_grad_(compute_mean_gradients)
        with torch.enable_grad() if compute_mean_gradients else torch.no_grad():
            posterior = model.posterior(
                X_eval if eval_in_q_batch else X_eval.unsqueeze(-2)
            )
            mean = posterior.mean
            if compute_mean_gradients:
                grad = torch.autograd.grad(mean.sum(), X_eval)[0]
                all_grads.append(grad.detach())
        with torch.no_grad():
            all_scores.append(_posterior_quantile(posterior, quantile).reshape(-1))
    scores = torch.cat(all_scores)
    gradients = torch.cat(all_grads) if compute_mean_gradients else None
    return scores, gradients


def _posterior_quantile(posterior, quantile: float) -> Tensor:
    """Evaluate the marginal posterior quantile, via `quantile` or `icdf`."""
    q = torch.tensor(quantile, dtype=posterior.dtype, device=posterior.device)
    try:
        return posterior.quantile(q)
    except (NotImplementedError, AttributeError):
        return posterior.icdf(quantile)


def compute_active_subspace(
    gradients: Tensor,
    rank: int | float = 15,
) -> tuple[Tensor, Tensor]:
    r"""Compute the gradient-informed (active) subspace.

    Forms the empirical second-moment matrix ``H = G^T G / N`` of the
    posterior-mean gradients ``G`` and eigendecomposes it. The
    eigendecomposition is performed in double precision for numerical
    stability and the results are cast back to the input dtype.

    Args:
        gradients: An ``N x d`` tensor of posterior-mean input gradients.
        rank: If an integer ``>= 1``, the fixed subspace dimension (clamped
            to ``d``). If a float in ``(0, 1)``, the smallest rank whose
            cumulative eigenvalue ratio reaches that fraction (recomputed on
            every call). Non-integer values ``>= 1`` are truncated to
            integers.

    Returns:
        A two-element tuple containing:

        - A ``d x r`` tensor whose columns are the top eigenvectors of ``H``.
        - A ``d``-dim tensor of all eigenvalues of ``H`` in descending order.
    """
    if gradients.dim() != 2:
        raise ValueError(
            f"gradients must be an `N x d` tensor, got shape {gradients.shape}."
        )
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    d = gradients.shape[-1]
    G = gradients.to(dtype=torch.double)
    H = G.transpose(-2, -1) @ G / G.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(H)  # ascending
    eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(-1)
    if rank < 1:  # percent-variance mode
        total = eigenvalues.sum()
        if total <= 0:
            r = 1
        else:
            cumulative_ratio = eigenvalues.cumsum(-1) / total
            r = int((cumulative_ratio < rank).sum().item()) + 1
    else:
        r = int(rank)
    r = min(r, d)
    return (
        eigenvectors[:, :r].to(dtype=gradients.dtype),
        eigenvalues.to(dtype=gradients.dtype),
    )


def sample_subspace_candidates(
    subspace: Tensor,
    origin: Tensor,
    bounds: Tensor,
    num_candidates: int = 5000,
    scale: float = 0.2,
) -> Tensor:
    r"""Sample candidate points inside a low-dimensional subspace.

    Candidates are ``x = origin + alpha @ subspace^T`` with subspace
    coordinates ``alpha ~ Uniform(-scale, scale)^r``, and are then clamped
    elementwise into ``bounds`` (out-of-box samples are projected onto the
    box faces).

    Args:
        subspace: A ``d x r`` tensor of subspace basis vectors.
        origin: A ``d``-dim tensor with the subspace origin. GIT-BO uses the
            mean of all evaluated points.
        bounds: A ``2 x d`` tensor of lower and upper box bounds.
        num_candidates: The number of candidates to sample.
        scale: The half-width of the uniform sampling box in subspace
            coordinates. Since the basis vectors have unit norm, the maximal
            displacement from the origin is ``scale * sqrt(r)``.

    Returns:
        A ``num_candidates x d`` tensor of candidate points.
    """
    r = subspace.shape[-1]
    alpha = (
        torch.rand(num_candidates, r, dtype=subspace.dtype, device=subspace.device) * 2
        - 1
    ) * scale
    X = origin.unsqueeze(0) + alpha @ subspace.transpose(-2, -1)
    return torch.clamp(X, bounds[0], bounds[1])


def gitbo_step(
    model: Model,
    train_X: Tensor,
    gradients: Tensor | None,
    bounds: Tensor,
    num_candidates: int = 5000,
    rank: int | float = 15,
    scale: float = 0.2,
    quantile: float = 0.975,
    eval_in_q_batch: bool = False,
    batch_limit: int | None = None,
) -> GITBOStepResult:
    r"""Run one GIT-BO candidate-generation and scoring step.

    If ``gradients`` is ``None`` (first iteration) or numerically zero, the
    candidate set is drawn with scrambled Sobol sampling over ``bounds``.
    Otherwise candidates are sampled in the gradient-informed subspace of
    the previous step's gradients, centered at ``train_X.mean(0)``.

    This function is stateless: the caller owns the training data and the
    surrogate (fit or condition the model on all data before each call — the
    model itself is not refit here) and carries ``result.gradients`` into
    the next call, so the subspace always derives from the previous
    iteration's gradients, as in [yu2025gitbo]_.

    Args:
        model: A fitted single-outcome model over the current training data.
        train_X: An ``n x d`` tensor of evaluated points; its mean is the
            subspace origin.
        gradients: The ``gradients`` field of the previous step's result, or
            ``None`` to draw Sobol candidates.
        bounds: A ``2 x d`` tensor of box bounds for the design space.
        num_candidates: The number of candidates to generate and score.
        rank: Subspace rank; see ``compute_active_subspace``.
        scale: Subspace sampling half-width; see
            ``sample_subspace_candidates``.
        quantile: Quantile level of the UCB score; see ``quantile_ucb``.
        eval_in_q_batch: Posterior batching layout; see ``quantile_ucb``.
            Use ``True`` for PFN models and ``False`` for GP models.
        batch_limit: Optional chunk size for posterior evaluation; see
            ``quantile_ucb``.

    Returns:
        A ``GITBOStepResult`` with the argmax candidate, the scores, the
        candidate set, the new gradients, and the subspace used (if any).
    """
    if bounds.dim() != 2 or bounds.shape[0] != 2:
        raise ValueError(f"bounds must be a `2 x d` tensor, got shape {bounds.shape}.")
    subspace = eigenvalues = None
    if gradients is None or not gradients.norm().item() > 0:
        candidate_set = draw_sobol_samples(
            bounds=bounds, n=num_candidates, q=1
        ).squeeze(-2)
    else:
        subspace, eigenvalues = compute_active_subspace(gradients, rank=rank)
        candidate_set = sample_subspace_candidates(
            subspace=subspace,
            origin=train_X.mean(dim=0),
            bounds=bounds,
            num_candidates=num_candidates,
            scale=scale,
        )
    acq_values, new_gradients = quantile_ucb(
        model=model,
        X=candidate_set,
        quantile=quantile,
        compute_mean_gradients=True,
        eval_in_q_batch=eval_in_q_batch,
        batch_limit=batch_limit,
    )
    candidate = candidate_set[acq_values.argmax()].unsqueeze(0)
    return GITBOStepResult(
        candidate=candidate,
        acq_values=acq_values,
        candidate_set=candidate_set,
        gradients=new_gradients,
        subspace=subspace,
        eigenvalues=eigenvalues,
    )
