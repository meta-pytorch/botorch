#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental coordinate-ascent and KL-fit utilities for the EM-based GP.

These components build on the EM model
(:class:`botorch.models.empirical_gps.em_empirical_gp.EMEmpiricalGaussianProcess`)
and provide alternative routines for jointly fitting the kernel hyperparameters
together with the EM-estimated prior.

.. note::
    These utilities are **experimental**: their API may change and they have not
    been validated as thoroughly as the core EM model. Prefer the standard
    ``pretrain_em_prior`` / ``EMEmpiricalGaussianProcess`` workflow for production.

Contents:

- ``coordinate_ascent_em``: alternate EM updates with MLL-based kernel optimization.
- ``KLPriorFitMLL`` / ``fit_kernel_to_em_prior``: fit kernel hyperparameters by
  KL minimization against the EM-estimated prior (an empirical-Bayes heuristic).
- ``kl_divergence_mvn``: KL divergence between two multivariate normals. This is a
  raw-tensor convenience equivalent to the ``MultivariateNormal`` KL registered with
  ``torch.distributions.kl_divergence``, but computed directly from (mean, covariance)
  pairs via Cholesky factorization.
"""

from __future__ import annotations

import copy

import torch
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps.em_empirical_gp import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    EMPriorContainer,
    pretrain_em_prior,
)
from botorch.models.empirical_gps.utils import ExperimentDataset
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ExactGP
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor


# =============================================================================
# Gaussian KL Divergence
# =============================================================================


def kl_divergence_mvn(
    mu_p: Tensor,
    Sigma_p: Tensor,
    mu_q: Tensor,
    Sigma_q: Tensor,
) -> Tensor:
    """Compute KL(N(mu_p, Sigma_p) || N(mu_q, Sigma_q)).

    Uses Cholesky-based computation for numerical stability.

    Args:
        mu_p: (M,) mean of p.
        Sigma_p: (M, M) covariance of p.
        mu_q: (M,) mean of q.
        Sigma_q: (M, M) covariance of q.

    Returns:
        Scalar KL divergence.
    """
    M = mu_p.shape[0]
    L_q = psd_safe_cholesky(Sigma_q)
    log_det_q = 2.0 * L_q.diagonal().log().sum()
    L_p = psd_safe_cholesky(Sigma_p)
    log_det_p = 2.0 * L_p.diagonal().log().sum()

    # tr(Sigma_q^{-1} Sigma_p)
    Sigma_q_inv_Sigma_p = torch.cholesky_solve(Sigma_p, L_q)
    trace_term = Sigma_q_inv_Sigma_p.diagonal().sum()

    # (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p)
    diff = mu_q - mu_p
    quad_term = diff @ torch.cholesky_solve(diff.unsqueeze(-1), L_q).squeeze(-1)

    return 0.5 * (log_det_q - log_det_p + trace_term + quad_term - M)


# =============================================================================
# Coordinate Ascent for Joint Kernel + EM Optimization
# =============================================================================


def coordinate_ascent_em(
    datasets: list[ExperimentDataset],
    mean_module: Mean,
    covar_module: Kernel,
    likelihood_noise: Tensor | float | None = None,
    num_em_iterations: int = 16,
    n_coord_ascent_iterations: int = 1,
    inducing_points: Tensor | None = None,
    optimize_inducing_points: bool = False,
    use_mean_prior: bool = False,
    use_covar_prior: bool = False,
    iw_nu: float | None = None,
    em_convergence_tol: float | None = 1e-6,
    enable_interpolation: bool = True,
    init_mode: str = "kernel",
    mll_max_iter: int = 50,
    mll_dataset_subsample: int | None = None,
    mll_use_adam: bool | None = None,
    mll_adam_lr: float = 0.01,
    mll_adam_steps: int = 200,
    container_history: list[EMPriorContainer] | None = None,
) -> EMPriorContainer:
    """Coordinate ascent: alternate EM updates with MLL-based kernel optimization.

    This function implements the following loop:

    1. Run EM with current kernel params → EMPriorContainer
    2. Optimize kernel hyperparameters (and optionally inducing locations)
       via the observed-data MLL with detached EM estimates
    3. Re-run EM with updated kernel → new EMPriorContainer
    4. Repeat for n_coord_ascent_iterations

    The kernel params are updated in-place on the shared mean_module/covar_module,
    so each MLL optimization starts from the previous iteration's params.

    Args:
        datasets: List of ExperimentDataset objects.
        mean_module: Parametric mean module (modified in-place).
        covar_module: Parametric covariance module (modified in-place).
        likelihood_noise: Noise variance for observations.
        num_em_iterations: Max EM iterations per coordinate ascent step.
        n_coord_ascent_iterations: Number of outer coordinate ascent iterations.
        inducing_points: Optional (M, d) inducing point locations.
        optimize_inducing_points: If True, make inducing points learnable.
        use_mean_prior: If True, use kernel prior on μ.
        use_covar_prior: If True, use Inverse-Wishart prior on Σ.
        iw_nu: Degrees of freedom for IW prior.
        em_convergence_tol: Convergence tolerance for EM early stopping.
        enable_interpolation: If True, enable shift interpolation.
        init_mode: Initialization mode for EM algorithm.
        mll_max_iter: Maximum L-BFGS iterations for kernel optimization.
        mll_dataset_subsample: If set, randomly subsample this many datasets
            per MLL evaluation for faster optimization. None uses all datasets.
        mll_use_adam: If True, use Adam optimizer (via fit_gpytorch_mll_torch)
            instead of L-BFGS. If None (default), auto-selects: Adam when
            optimize_inducing_points=True (700+ params), L-BFGS otherwise.
        mll_adam_lr: Learning rate for Adam optimizer (default: 0.01).
        mll_adam_steps: Number of Adam steps (default: 200).
        container_history: Optional list to record the optimization trajectory.
            When provided, a deep copy of the container after the initial EM and
            after each coordinate-ascent cycle is appended, i.e.
            ``[after_em_0, after_mll1_em1, ...]``. Pass ``None`` (default) to skip
            recording, in which case no copies are made.

    Returns:
        The final EMPriorContainer after coordinate ascent. If
        ``container_history`` was provided, it is populated in place with the
        per-step trajectory (its last element equals the returned container).
    """
    # Local imports to avoid a botorch.fit <-> botorch.models circular import.
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.fit import fit_gpytorch_mll_torch

    # Auto-select optimizer: Adam for inducing point optimization (many params),
    # L-BFGS for kernel-only optimization (few params)
    if mll_use_adam is None:
        mll_use_adam = optimize_inducing_points

    # Step 0: Initial EM run with current kernel params
    em_prior = pretrain_em_prior(
        datasets=datasets,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood_noise=likelihood_noise,
        inducing_points=inducing_points,
        num_em_iterations=num_em_iterations,
        use_mean_prior=use_mean_prior,
        use_covar_prior=use_covar_prior,
        iw_nu=iw_nu,
        em_convergence_tol=em_convergence_tol,
        enable_interpolation=enable_interpolation,
        init_mode=init_mode,
    )

    # Record the baseline EM container if a history collector was provided.
    # IMPORTANT: We must deep-copy containers because EMPriorContainer stores
    # live references to mean_module/covar_module. Without copying, subsequent
    # coord ascent iterations that modify these modules in-place (via
    # fit_gpytorch_mll) would corrupt all previously recorded containers.
    if container_history is not None:
        container_history.append(copy.deepcopy(em_prior))

    # Adam optimizer factory for inducing-point optimization. Defined once here
    # (it is loop-invariant) rather than as a closure inside the loop below.
    def _adam_optimizer(parameters, **kwargs):
        return torch.optim.Adam(parameters, lr=mll_adam_lr, foreach=True)

    # Coordinate ascent iterations
    for _ in range(n_coord_ascent_iterations):
        # Create model from the EM prior with pretrained=True (skips re-running EM)
        # and freeze_pretrained=False (so kernel params are optimizable)
        dummy_X = em_prior.X_inducing[:1]
        dummy_Y = torch.zeros(1, 1, dtype=dummy_X.dtype, device=dummy_X.device)

        noise_constraint = GreaterThan(1e-16, transform=None)
        mll_likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        if isinstance(likelihood_noise, Tensor):
            mll_likelihood.noise = likelihood_noise.detach()
        elif likelihood_noise is not None:
            mll_likelihood.noise = torch.tensor(
                likelihood_noise, dtype=dummy_X.dtype, device=dummy_X.device
            )
        mll_likelihood.noise_covar.raw_noise.requires_grad_(False)

        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=dummy_X,
            train_Y=dummy_Y,
            likelihood=mll_likelihood,
            freeze_pretrained=False,
            learnable_inducing_points=optimize_inducing_points,
        )

        # Optionally subsample datasets for faster MLL evaluation.
        # Use the full EM prior (mu, Sigma from all K datasets) but evaluate
        # the MLL on only a random subset of datasets.
        if mll_dataset_subsample is not None and mll_dataset_subsample < len(datasets):
            perm = torch.randperm(len(datasets))[:mll_dataset_subsample]
            model.datasets = [datasets[i] for i in perm]

        # Wrap in MLL and optimize kernel (+ optionally inducing locations)
        mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)

        if mll_use_adam:
            # Use Adam — better for high-dimensional optimization (inducing points)
            # Disable the default stopping criterion to rely only on step_limit
            fit_gpytorch_mll_torch(
                mll,
                step_limit=mll_adam_steps,
                optimizer=_adam_optimizer,
                stopping_criterion=None,
            )
        else:
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": mll_max_iter}}
            )

        # Extract optimized inducing locations if they were learnable
        updated_inducing = None
        if optimize_inducing_points:
            updated_inducing = model._X_inducing.detach().clone()

        # Re-run EM with optimized kernel (and optionally new inducing locations)
        # on the FULL dataset set (not the subsample)
        em_prior = pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood_noise=likelihood_noise,
            inducing_points=updated_inducing
            if updated_inducing is not None
            else inducing_points,
            num_em_iterations=num_em_iterations,
            use_mean_prior=use_mean_prior,
            use_covar_prior=use_covar_prior,
            iw_nu=iw_nu,
            em_convergence_tol=em_convergence_tol,
            enable_interpolation=enable_interpolation,
            init_mode=init_mode,
        )

        if container_history is not None:
            container_history.append(copy.deepcopy(em_prior))

    return em_prior


# =============================================================================
# KL Prior Fit for Kernel Optimization
# =============================================================================


class KLPriorFitMLL(MarginalLogLikelihood):
    """MLL-compatible wrapper for KL prior fit — use with fit_gpytorch_mll.

    Fits kernel hyperparameters by minimizing KL(N(mu_em, Sigma_em) || N(m_phi, K_phi))
    where m_phi and K_phi are the parametric mean and kernel evaluated at reference
    points Z. This is NOT the correct EM M-step for kernel parameters (since the
    prior log p(theta_k | mu, Sigma) doesn't depend on phi when mu, Sigma are free
    parameters). However, it IS a valid empirical Bayes heuristic that fits the
    kernel covariance structure to match the EM-estimated prior at reference points.

    Can be used as:
    1. Cheap initialization for coordinate ascent (before observed-data MLL refinement)
    2. Standalone alternative if it performs well numerically
    """

    def __init__(
        self,
        likelihood: Likelihood,
        model: ExactGP,
        X_ref: Tensor,
        mu_em: Tensor,
        Sigma_em: Tensor,
    ) -> None:
        """Initialize the KL-prior-fit MLL. See the class docstring for usage."""
        super().__init__(likelihood, model)
        self.register_buffer("_X_ref", X_ref)
        self.register_buffer("_mu_em", mu_em.detach())
        self.register_buffer("_Sigma_em", Sigma_em.detach())

    def forward(
        self,
        function_dist: MultivariateNormal,
        target: Tensor,
    ) -> Tensor:
        """Compute negative KL divergence (fit_gpytorch_mll maximizes this)."""
        K = self.model.covar_module(self._X_ref, self._X_ref).to_dense()
        m = self.model.mean_module(self._X_ref)
        if m.dim() > 1:
            m = m.squeeze(-1)
        return -kl_divergence_mvn(self._mu_em, self._Sigma_em, m, K)


def fit_kernel_to_em_prior(
    mean_module: Mean,
    covar_module: Kernel,
    X_ref: Tensor,
    mu_em: Tensor,
    Sigma_em: Tensor,
) -> None:
    """Fit kernel hyperparameters by KL minimization against EM-estimated prior.

    Creates a dummy SingleTaskGP sharing the provided mean/covar modules,
    wraps the KL objective as a MarginalLogLikelihood, and uses fit_gpytorch_mll
    (L-BFGS) to optimize.

    Args:
        mean_module: Parametric mean module (parameters will be updated in-place).
        covar_module: Parametric covariance module (parameters will be updated).
        X_ref: (M, d) reference point locations.
        mu_em: (M,) EM-converged mean at reference points.
        Sigma_em: (M, M) EM-converged covariance at reference points.
    """
    # Local import to avoid a botorch.fit <-> botorch.models circular import.
    from botorch.fit import fit_gpytorch_mll

    dummy_gp = SingleTaskGP(
        train_X=X_ref[:1],
        train_Y=torch.zeros(1, 1, dtype=X_ref.dtype, device=X_ref.device),
        mean_module=mean_module,
        covar_module=covar_module,
        outcome_transform=None,
    )
    kl_mll = KLPriorFitMLL(dummy_gp.likelihood, dummy_gp, X_ref, mu_em, Sigma_em)
    fit_gpytorch_mll(kl_mll)
