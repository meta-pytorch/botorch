# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Principled selection of the EM pre-training noise, and alternatives that avoid it.

§27.40 established that `likelihood_noise` is not a noise level: the E-step keeps
eigendirection lambda with weight lambda/(lambda+sigma^2), so sigma^2 chooses an
EFFECTIVE
RANK for `Sigma_ind`, whose true rank is <= T-1 for T pre-training tasks. Grid-searching
it
on a benchmark where the answer is visible is not a method; this module implements ones
that are.

Four approaches, in increasing order of how much they change the model.

1. LOO-TASK MARGINAL LIKELIHOOD  (`select_loo_marginal`)
   Hold out each pre-training task in turn, fit the prior on the rest, score the
   held-out
   task's data under it, and pick the sigma^2 with the best mean score. Uses only
   pre-training data, so it is legitimate on a new problem where no test set exists.
   This
   is the direct automation of what the grid does by hand.

2. LEDOIT-WOLF / OAS ANALYTIC SHRINKAGE  (`ledoit_wolf_alpha`, `oas_alpha`, `rblw_alpha`)

   PRIMARY REFERENCE for the OAS and RBLW estimators:
     Chen, Wiesel, Eldar & Hero, "Shrinkage Algorithms for MMSE Covariance Estimation",
     IEEE Trans. Signal Processing 58(10), 2010.  https://arxiv.org/abs/0907.4698
   Ledoit & Wolf, "A well-conditioned estimator for large-dimensional covariance
     matrices", J. Multivariate Analysis 88(2), 2004, is the earlier distribution-free
     estimator that OAS improves on under Gaussianity.
   The textbook estimator for a rank-deficient sample covariance. Shrinks the empirical
   Sigma toward a scaled identity with an intensity chosen in CLOSED FORM to minimise
   expected Frobenius risk -- no tuning, no held-out data. This is the correct tool for
   the
   job sigma^2 has been doing by accident, and it separates the two concerns: sigma^2
   goes
   back to meaning observation noise, and shrinkage handles regularisation.

3. EFFECTIVE-RANK MATCHING  (`sigma_for_effective_rank`)
   Solve for the sigma^2 whose retention profile keeps about T-1 directions, i.e. the
   sample rank. Cheap, interpretable, and it makes the §27.40 prediction concrete: the
   value should fall as T grows.

4. MARCHENKO-PASTUR EDGE  (`mp_threshold`)
   Random-matrix theory gives the largest eigenvalue attributable to pure noise when
   estimating a K x K covariance from T samples. Directions below that edge carry no
   signal. This yields a threshold with no free parameters at all.

The comparison harness scores each selector against the grid optimum, which is the
quantity they are trying to recover WITHOUT looking at held-out data.
"""

from __future__ import annotations

import math

import torch


def ledoit_wolf_alpha(X: torch.Tensor) -> tuple[float, float]:
    """Closed-form shrinkage intensity toward a scaled identity.

    ``X`` is (T, K): one row per pre-training task, one column per inducing point.
    Returns ``(alpha, mu)`` for ``Sigma_shrunk = (1-alpha) * S + alpha * mu * I``.

    Ledoit & Wolf (2004). alpha minimises E||Sigma_shrunk - Sigma_true||_F^2 and needs
    no
    held-out data, which is exactly the regime here: T is small, K is large, and S is
    singular by construction.
    """
    T, K = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    S = (Xc.T @ Xc) / max(T, 1)
    mu = float(S.diagonal().mean())

    # d^2 = ||S - mu I||_F^2 : how far the sample covariance is from the target.
    d2 = float(((S - mu * torch.eye(K, dtype=S.dtype)) ** 2).sum())
    # b_bar^2 : the variance of S itself, i.e. how much of d^2 is estimation noise.
    b2 = 0.0
    for t in range(T):
        xt = Xc[t : t + 1]
        b2 += float(((xt.T @ xt - S) ** 2).sum())
    b2 /= max(T * T, 1)
    b2 = min(b2, d2)  # Ledoit-Wolf truncation
    alpha = b2 / d2 if d2 > 0 else 1.0
    return float(min(max(alpha, 0.0), 1.0)), mu


def oas_alpha(X: torch.Tensor) -> tuple[float, float]:
    """Oracle Approximating Shrinkage -- Chen et al. 2010, eq. (23).

    https://arxiv.org/abs/0907.4698. Verified against the paper:
        rho = [(1 - 2/p)tr(S^2) + tr(S)^2] / [(n + 1 - 2/p)(tr(S^2) - tr(S)^2/p)]
    with p the dimension and n the sample count, clipped to [0, 1].

    Same shape as Ledoit-Wolf but with an intensity tuned for Gaussian data, which is
    usually better when T is very small -- the regime we are in.
    """
    T, K = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    S = (Xc.T @ Xc) / max(T, 1)
    mu = float(S.diagonal().mean())
    tr_S2 = float((S @ S).diagonal().sum())
    tr2_S = float(S.diagonal().sum()) ** 2
    num = (1.0 - 2.0 / K) * tr_S2 + tr2_S
    den = (T + 1.0 - 2.0 / K) * (tr_S2 - tr2_S / K)
    alpha = 1.0 if den <= 0 else num / den
    return float(min(max(alpha, 0.0), 1.0)), mu


def sigma_for_effective_rank(eigvals: torch.Tensor, target_rank: float) -> float:
    """Smallest sigma^2 whose retention profile keeps about ``target_rank`` directions.

    Effective rank is sum_i lambda_i / (lambda_i + sigma^2), a continuous relaxation of
    "how many directions survive the E-step". Bisection because the profile is monotone.
    """
    ev = eigvals.clamp_min(0).double()
    if target_rank >= len(ev):
        return 1e-8

    def eff(s2: float) -> float:
        return float((ev / (ev + s2)).sum())

    lo, hi = 1e-8, float(ev.max()) * 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eff(mid) > target_rank:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def mp_threshold(eigvals: torch.Tensor, n_tasks: int, n_dims: int) -> float:
    """Marchenko-Pastur upper edge: the largest eigenvalue explainable by pure noise.

    For a K x K covariance from T samples with per-direction variance s2, sample
    eigenvalues of a pure-noise matrix fall below s2 * (1 + sqrt(K/T))^2. Anything under
    that edge carries no signal, so it is a parameter-free rank threshold.
    """
    ev = eigvals.clamp_min(0).double()
    q = n_dims / max(n_tasks, 1)
    bulk = float(ev.median())  # robust stand-in for the noise level
    return bulk * (1.0 + math.sqrt(q)) ** 2


def effective_rank(eigvals: torch.Tensor, s2: float) -> float:
    ev = eigvals.clamp_min(0).double()
    return float((ev / (ev + s2)).sum())


def estimate_iw_nu(X: torch.Tensor, method: str = "oas") -> tuple[float, float]:
    """Degrees of freedom for an Inverse-Wishart prior, chosen by empirical Bayes.

    For ``Sigma ~ IW(Psi, nu)`` the posterior mean is
    ``(T*S + Psi) / (T + nu - K - 1)``, i.e. linear shrinkage toward ``Psi/(nu-K-1)``
    with intensity ``alpha = (nu-K-1)/(T+nu-K-1)``. Inverting that gives

        nu = (K+1) + alpha*T/(1-alpha)

    so picking ``alpha`` with OAS or Ledoit-Wolf is exactly empirical-Bayes selection of
    ``nu`` -- which turns the hand-set ``--iw-nu`` into an estimated quantity without
    adding a new regulariser.

    §27.46 measured OAS as the better of the two at the sample sizes we actually have
    (T = 6-25); the two converge as T grows. Returns ``(nu, alpha)``.
    """
    T, K = X.shape
    alpha, _mu = oas_alpha(X) if method == "oas" else ledoit_wolf_alpha(X)
    # alpha -> 1 means "trust the prior entirely", which is nu -> infinity. Cap so the
    # caller gets a usable float rather than an inf that poisons the M-step.
    alpha = min(alpha, 0.999)
    nu = (K + 1.0) + alpha * T / (1.0 - alpha)
    return float(nu), float(alpha)


def rblw_alpha(X: torch.Tensor) -> tuple[float, float]:
    """Rao-Blackwell Ledoit-Wolf shrinkage -- Chen et al. 2010, eq. (17).

    https://arxiv.org/abs/0907.4698. The Rao-Blackwellised version of Ledoit-Wolf: under
    Gaussianity it dominates plain LW in mean-squared error, and unlike OAS it is a
    closed form rather than the fixed point of an iteration, so it cannot oscillate.

        rho = [ (n-2)/n * tr(S^2) + tr(S)^2 ] / [ (n+2) * ( tr(S^2) - tr(S)^2/p ) ]

    NOT YET BENCHMARKED here. S27.46 compared LW and OAS only; RBLW sits between them in
    the paper and is cheap to evaluate, so it belongs in the next sweep rather than in a
    recommendation.
    """
    T, K = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    S = (Xc.T @ Xc) / max(T, 1)
    mu = float(S.diagonal().mean())
    tr_S2 = float((S @ S).diagonal().sum())
    tr2_S = float(S.diagonal().sum()) ** 2
    num = ((T - 2.0) / T) * tr_S2 + tr2_S
    den = (T + 2.0) * (tr_S2 - tr2_S / K)
    alpha = 1.0 if den <= 0 else num / den
    return float(min(max(alpha, 0.0), 1.0)), mu


def _inv_sqrt(F: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """F^(-1/2) via symmetric eigendecomposition, floored to stay finite."""
    F = 0.5 * (F + F.T)
    w, V = torch.linalg.eigh(F.double())
    w = w.clamp_min(eps * float(w.max().clamp_min(eps)))
    return (V * w.rsqrt()) @ V.T


def estimate_iw_nu_whitened(
    X: torch.Tensor, F: torch.Tensor, ridge: float = 0.05
) -> tuple[float, float]:
    """``estimate_iw_nu`` with alpha measured against the target the EM ACTUALLY uses.

    THE INCONSISTENCY THIS FIXES. ``_compute_prior_scale_matrix`` sets
    ``Psi = (nu + N + 1) * Sigma_init``, and with the default ``init_mode="kernel"``
    ``Sigma_init`` is ``K(Z, Z)`` -- so the Inverse-Wishart target is ALREADY the
    parametric kernel, not a sphere. But ``oas_alpha`` is Chen et al. eq. (23), derived
    for the spherical target and hard-coding it via ``mu = S.diagonal().mean()``. So the
    shrinkage INTENSITY was being chosen for one target and applied to another.

    Whitening by F puts the estimator in coordinates where the target IS the identity, so
    the published formula applies verbatim -- reusing ``oas_alpha`` rather than
    re-deriving it, which is the entire justification for doing it this way.

    RIDGE IS MANDATORY. S30.7 measured pure whitening at +2947% Frobenius error against
    the spherical baseline: a Gram matrix has cond ~1e7, so F^(-1/2) has eigenvalues ~1e3
    and the whitened sample is dominated by directions where F carries no signal. With
    ``F_tau = (1-tau) F_hat + tau I`` on a unit-mean-diagonal F_hat, the same simulation
    gave -21.2% at tau=0.05. tau=1 recovers plain OAS exactly, so the ridge is also the
    knob that makes this strictly a generalisation.
    """
    T, K = X.shape
    if F.shape != (K, K):
        raise ValueError(
            f"whitened OAS needs the target on the SAME points as the data: "
            f"X is {T}x{K} so F must be {K}x{K}, got {tuple(F.shape)}"
        )
    F = F.double()
    F = F / float(F.diagonal().mean().clamp_min(1e-300))
    F = (1.0 - ridge) * F + ridge * torch.eye(K, dtype=F.dtype, device=F.device)
    Xc = X.double() - X.double().mean(dim=0, keepdim=True)
    alpha, _mu = oas_alpha(Xc @ _inv_sqrt(F))
    alpha = min(alpha, 0.999)
    nu = (K + 1.0) + alpha * T / (1.0 - alpha)
    return float(nu), float(alpha)
