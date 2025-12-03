#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Cross-validation utilities using batch evaluation mode.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor


class CVFolds(NamedTuple):
    train_X: Tensor
    test_X: Tensor
    train_Y: Tensor
    test_Y: Tensor
    train_Yvar: Tensor | None = None
    test_Yvar: Tensor | None = None


class CVResults(NamedTuple):
    model: GPyTorchModel
    posterior: GPyTorchPosterior
    observed_Y: Tensor
    observed_Yvar: Tensor | None = None


class LOOCVResults(NamedTuple):
    """Results from efficient Leave-One-Out cross-validation.

    This named tuple contains the LOO predictive means and variances,
    along with the observed Y values and optionally the observation noise.
    The ``mean`` and ``variance`` fields have shape ``n x m`` or
    ``batch_shape x n x m``.
    """

    mean: Tensor
    variance: Tensor
    observed_Y: Tensor
    observed_Yvar: Tensor | None = None
    model: GPyTorchModel | None = None


class EnsembleLOOCVResults(NamedTuple):
    """Results from efficient Leave-One-Out cross-validation for ensemble models.

    For ensemble models like fully Bayesian GPs, the LOO predictions from each
    ensemble member form a Gaussian mixture. This class contains both the
    per-member results (``per_model_mean``, ``per_model_variance``) and the
    aggregated mixture statistics (``mean``, ``variance``).

    The aggregated ``mean`` and ``variance`` have shape ``n x m``, while the
    per-member results have shape ``num_models x n x m``.
    """

    mean: Tensor
    variance: Tensor
    observed_Y: Tensor
    observed_Yvar: Tensor | None = None
    model: GPyTorchModel | None = None
    per_model_mean: Tensor | None = None
    per_model_variance: Tensor | None = None


def gen_loo_cv_folds(
    train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None
) -> CVFolds:
    r"""Generate LOO CV folds w.r.t. to `n`.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x (m)` or `batch_shape x n x (m)` (batch mode) tensor of
            training observations.
        train_Yvar: An `n x (m)` or `batch_shape x n x (m)` (batch mode) tensor
            of observed measurement noise.

    Returns:
        CVFolds NamedTuple with the following fields:

        - train_X: A `n x (n-1) x d` or `batch_shape x n x (n-1) x d` tensor of
          training features.
        - test_X: A `n x 1 x d` or `batch_shape x n x 1 x d` tensor of test features.
        - train_Y: A `n x (n-1) x m` or `batch_shape x n x (n-1) x m` tensor of
          training observations.
        - test_Y: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of test
          observations.
        - train_Yvar: A `n x (n-1) x m` or `batch_shape x n x (n-1) x m` tensor
          of observed measurement noise.
        - test_Yvar: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observed
          measurement noise.

    Example:
        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
        >>> cv_folds.train_X.shape
        torch.Size([10, 9, 1])
    """
    masks = torch.eye(train_X.shape[-2], dtype=torch.uint8, device=train_X.device)
    masks = masks.to(dtype=torch.bool)
    if train_Y.dim() < train_X.dim():
        # add output dimension
        train_Y = train_Y.unsqueeze(-1)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.unsqueeze(-1)
    train_X_cv = torch.cat(
        [train_X[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_X_cv = torch.cat([train_X[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3)
    train_Y_cv = torch.cat(
        [train_Y[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
    )
    test_Y_cv = torch.cat([train_Y[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3)
    if train_Yvar is None:
        train_Yvar_cv = None
        test_Yvar_cv = None
    else:
        train_Yvar_cv = torch.cat(
            [train_Yvar[..., ~m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
        test_Yvar_cv = torch.cat(
            [train_Yvar[..., m, :].unsqueeze(dim=-3) for m in masks], dim=-3
        )
    return CVFolds(
        train_X=train_X_cv,
        test_X=test_X_cv,
        train_Y=train_Y_cv,
        test_Y=test_Y_cv,
        train_Yvar=train_Yvar_cv,
        test_Yvar=test_Yvar_cv,
    )


def batch_cross_validation(
    model_cls: type[GPyTorchModel],
    mll_cls: type[MarginalLogLikelihood],
    cv_folds: CVFolds,
    fit_args: dict[str, Any] | None = None,
    observation_noise: bool = False,
    model_init_kwargs: dict[str, Any] | None = None,
) -> CVResults:
    r"""Perform cross validation by using GPyTorch batch mode.

    WARNING: This function is currently very memory inefficient; use it only
        for problems of small size.

    Args:
        model_cls: A GPyTorchModel class. This class must initialize the likelihood
            internally. Note: Multi-task GPs are not currently supported.
        mll_cls: A MarginalLogLikelihood class.
        cv_folds: A CVFolds tuple.
        fit_args: Arguments passed along to fit_gpytorch_mll.
        model_init_kwargs: Keyword arguments passed to the model constructor.

    Returns:
        A CVResults tuple with the following fields

        - model: GPyTorchModel for batched cross validation
        - posterior: GPyTorchPosterior where the mean has shape `n x 1 x m` or
          `batch_shape x n x 1 x m`
        - observed_Y: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observations.
        - observed_Yvar: A `n x 1 x m` or `batch_shape x n x 1 x m` tensor of observed
          measurement noise.

    Example:
        >>> import torch
        >>> from botorch.cross_validation import (
        ...     batch_cross_validation, gen_loo_cv_folds
        ... )
        >>>
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.models.transforms.input import Normalize
        >>> from botorch.models.transforms.outcome import Standardize
        >>> from gpytorch.mlls import ExactMarginalLogLikelihood

        >>> train_X = torch.rand(10, 1)
        >>> train_Y = torch.rand_like(train_X)
        >>> cv_folds = gen_loo_cv_folds(train_X, train_Y)
        >>> input_transform = Normalize(d=train_X.shape[-1])
        >>>
        >>> cv_results = batch_cross_validation(
        ...    model_cls=SingleTaskGP,
        ...    mll_cls=ExactMarginalLogLikelihood,
        ...    cv_folds=cv_folds,
        ...    model_init_kwargs={
        ...        "input_transform": input_transform,
        ...    },
        ... )
    """
    if issubclass(model_cls, MultiTaskGP):
        raise UnsupportedError(
            "Multi-task GPs are not currently supported by `batch_cross_validation`."
        )
    model_init_kws = model_init_kwargs if model_init_kwargs is not None else {}
    if cv_folds.train_Yvar is not None:
        model_init_kws["train_Yvar"] = cv_folds.train_Yvar
    model_cv = model_cls(
        train_X=cv_folds.train_X,
        train_Y=cv_folds.train_Y,
        **model_init_kws,
    )
    mll_cv = mll_cls(model_cv.likelihood, model_cv)
    mll_cv.to(cv_folds.train_X)

    fit_args = fit_args or {}
    mll_cv = fit_gpytorch_mll(mll_cv, **fit_args)

    # Evaluate on the hold-out set in batch mode
    with torch.no_grad():
        posterior = model_cv.posterior(
            cv_folds.test_X, observation_noise=observation_noise
        )

    return CVResults(
        model=model_cv,
        posterior=posterior,
        observed_Y=cv_folds.test_Y,
        observed_Yvar=cv_folds.test_Yvar,
    )


def efficient_loo_cv(
    model: GPyTorchModel,
) -> LOOCVResults:
    r"""Compute efficient Leave-One-Out cross-validation for a GP model.

    This function leverages a well-known linear algebraic identity to compute
    all LOO predictive distributions in O(n^3) time, compared to the naive
    approach which requires O(n^4) time (O(n^3) per fold for n folds).

    The efficient LOO formulas for GPs are:

    .. math::

        \mu_{LOO,i} = y_i - \frac{[K^{-1}y]_i}{[K^{-1}]_{ii}}

        \sigma^2_{LOO,i} = \frac{1}{[K^{-1}]_{ii}}

    where K is the covariance matrix including observation noise.

    NOTE: This function assumes the model has already been fitted and that the
    model's `forward` method returns a `MultivariateNormal` distribution.

    Args:
        model: A fitted GPyTorchModel whose `forward` method returns a
            `MultivariateNormal` distribution.

    Returns:
        LOOCVResults: A named tuple containing:
            - mean: The LOO predictive means with shape `n x m` or
              `batch_shape x n x m`.
            - variance: The LOO predictive variances with shape `n x m` or
              `batch_shape x n x m`.
            - observed_Y: The observed Y values with shape `n x m` or
              `batch_shape x n x m`.
            - observed_Yvar: The observed noise variances (if provided) with shape
              `n x m` or `batch_shape x n x m`.
            - model: The fitted GP model.

    Example:
        >>> import torch
        >>> from botorch.cross_validation import efficient_loo_cv
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.fit import fit_gpytorch_mll
        >>> from gpytorch.mlls import ExactMarginalLogLikelihood
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
        >>> loo_results = efficient_loo_cv(model)
        >>> loo_results.mean.shape
        torch.Size([20, 1])
    """
    # Save the original training mode state
    was_training = model.training

    # The model must be in eval mode for proper covariance computation
    model.eval()

    # Get training data - model should have train_inputs attribute
    if not hasattr(model, "train_inputs") or model.train_inputs is None:
        raise UnsupportedError(
            "Model must have train_inputs attribute for efficient LOO CV."
        )
    if not hasattr(model, "train_targets") or model.train_targets is None:
        raise UnsupportedError(
            "Model must have train_targets attribute for efficient LOO CV."
        )

    train_X = model.train_inputs[0]  # Shape: n x d or batch_shape x n x d
    train_Y = model.train_targets  # Shape: n or batch_shape x n (batched outputs)

    n = train_X.shape[-2]

    # Compute the prior distribution using the model's forward method
    with torch.no_grad():
        # Apply input transform if present, since it is not evaluated in
        # forward in eval mode
        if hasattr(model, "input_transform") and model.input_transform is not None:
            train_X_transformed = model.input_transform(train_X)
        else:
            train_X_transformed = train_X

        # Get the prior distribution from the model's forward method
        # The model should be in training mode to get the prior (not posterior)
        model.train()
        prior_dist = model.forward(train_X_transformed)
        if was_training:
            model.train()
        else:
            model.eval()

        # Check that we got a MultivariateNormal
        if not isinstance(prior_dist, MultivariateNormal):
            raise UnsupportedError(
                f"Model's forward method must return a MultivariateNormal, "
                f"got {type(prior_dist).__name__}."
            )

        # Extract mean from the prior
        mean = prior_dist.mean  # n or batch_shape x n

        # Add observation noise to the diagonal via the likelihood
        # The likelihood adds noise: K_noisy = K + sigma^2 * I
        noisy_mvn = model.likelihood(prior_dist)
        K_noisy = noisy_mvn.covariance_matrix  # This is K + noise * I

        # Compute K_inv using Cholesky decomposition for numerical stability
        # L L^T = K_noisy, so K_inv = L^{-T} L^{-1}
        L = torch.linalg.cholesky(K_noisy)

        # K_inv_y = K^{-1} y, computed via L L^T x = y => x = L^{-T} L^{-1} y
        # train_Y has shape batch_shape x n or n
        # mean has shape batch_shape x n or n
        # Residuals: y - mean
        residuals = train_Y - mean  # batch_shape x n or n

        # Add output dimension for cholesky_solve (needs ... x n x m)
        residuals = residuals.unsqueeze(-1)  # batch_shape x n x 1 or n x 1

        # Solve K_noisy @ alpha = residuals for alpha (this is K^{-1} @ residuals)
        # Using cholesky_solve: solves A @ X = B given cholesky(A)
        K_inv_residuals = torch.cholesky_solve(residuals, L)  # batch_shape x n x 1

        # Compute diagonal of K^{-1}
        # K_inv = L^{-T} @ L^{-1}, so K_inv_diag[i] = sum_j (L^{-1}[j,i])^2
        # First compute L^{-1} by solving L @ X = I
        identity = torch.eye(n, dtype=L.dtype, device=L.device)
        if L.dim() > 2:
            # Batch case: expand identity to match batch shape
            identity = identity.expand(*L.shape[:-2], n, n)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)  # L^{-1}
        K_inv_diag = (L_inv**2).sum(dim=-2)  # Sum over rows: batch_shape x n or n

        # Expand K_inv_diag to match the output dimensions
        K_inv_diag = K_inv_diag.unsqueeze(-1)  # batch_shape x n x 1 or n x 1

        # Compute LOO predictions using the efficient formulas:
        # mu_loo_i = y_i - (K_inv @ (y - mean))_i / K_inv_ii
        # var_loo_i = 1 / K_inv_ii
        train_Y_expanded = train_Y.unsqueeze(-1)  # batch_shape x n x 1 or n x 1
        loo_mean = train_Y_expanded - K_inv_residuals / K_inv_diag
        loo_variance = 1.0 / K_inv_diag

    # Get observed Yvar if available (for fixed noise models)
    observed_Yvar = None
    if hasattr(model, "likelihood") and isinstance(
        model.likelihood, FixedNoiseGaussianLikelihood
    ):
        observed_Yvar = model.likelihood.noise.unsqueeze(-1)

    return LOOCVResults(
        mean=loo_mean,
        variance=loo_variance,
        observed_Y=train_Y_expanded,
        observed_Yvar=observed_Yvar,
        model=model,
    )


def ensemble_loo_cv(
    model: GPyTorchModel,
) -> EnsembleLOOCVResults:
    r"""Compute efficient LOO cross-validation for ensemble models.

    This function computes Leave-One-Out cross-validation for ensemble models
    like `SaasFullyBayesianSingleTaskGP`. For these models, the `forward` method
    returns a `MultivariateNormal` with a batch dimension containing statistics
    for all models in the ensemble.

    The LOO predictions from each ensemble member form a Gaussian mixture.
    This function computes both the per-member LOO results and the aggregated
    mixture mean and variance for each left-out point using the law of total
    variance:

    .. math::

        \mu_{mix} = \frac{1}{K} \sum_{k=1}^{K} \mu_k

        \sigma^2_{mix} = \frac{1}{K} \sum_{k=1}^{K} \sigma^2_k +
            \frac{1}{K} \sum_{k=1}^{K} \mu_k^2 - \mu_{mix}^2

    where K is the number of ensemble members.

    NOTE: This function returns both aggregated mixture statistics (`mean` and
    `variance`) and per-ensemble-member statistics (`per_model_mean` and
    `per_model_variance`). If per-member LOO predictions are preferred over
    the aggregated Gaussian mixture statistics (e.g., for downstream analysis
    or alternative aggregation schemes), the per-member results can be used
    directly from the returned `EnsembleLOOCVResults`.

    NOTE: This function assumes the model has already been fitted (e.g., using
    `fit_fully_bayesian_model_nuts`) and that the model is an ensemble model
    with `_is_ensemble = True`.

    Args:
        model: A fitted ensemble GPyTorchModel (e.g., SaasFullyBayesianSingleTaskGP)
            whose `forward` method returns a `MultivariateNormal` distribution
            with a batch dimension for ensemble members.

    Returns:
        EnsembleLOOCVResults: A named tuple containing:
            - mean: The LOO predictive mixture means with shape `n x m`.
            - variance: The LOO predictive mixture variances with shape `n x m`.
            - observed_Y: The observed Y values with shape `n x m`.
            - observed_Yvar: The observed noise variances (if provided).
            - model: The fitted ensemble GP model.
            - per_model_mean: The per-ensemble-member LOO means with shape
              `num_models x n x m`.
            - per_model_variance: The per-ensemble-member LOO variances with shape
              `num_models x n x m`.

    Example:
        >>> import torch
        >>> from botorch.cross_validation import ensemble_loo_cv
        >>> from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        >>> from botorch.models.fully_bayesian import fit_fully_bayesian_model_nuts
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)
        >>> model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(model, warmup_steps=64, num_samples=32)
        >>> loo_results = ensemble_loo_cv(model)
        >>> loo_results.mean.shape  # Aggregated mixture mean
        torch.Size([20, 1])
        >>> loo_results.per_model_mean.shape  # Per-ensemble-member means
        torch.Size([32, 20, 1])
    """
    # Check that this is an ensemble model
    if not getattr(model, "_is_ensemble", False):
        raise UnsupportedError(
            "ensemble_loo_cv requires an ensemble model (with _is_ensemble=True). "
            f"Got model of type {type(model).__name__}. "
            "For non-ensemble models, use efficient_loo_cv instead."
        )

    # Get the per-ensemble-member LOO results using the efficient implementation
    loo_results = efficient_loo_cv(model)

    # The per-member results have shape (num_models x n x m)
    per_model_mean = loo_results.mean  # num_models x n x m
    per_model_variance = loo_results.variance  # num_models x n x m

    # Check that we have a batch dimension for the ensemble
    if per_model_mean.dim() < 3:
        raise UnsupportedError(
            "Expected ensemble model to produce batched LOO results with shape "
            f"(num_models x n x m), but got shape {per_model_mean.shape}."
        )

    # Compute the Gaussian mixture statistics using the law of total variance
    # For a mixture of K Gaussians with equal weights:
    # mu_mix = (1/K) * sum(mu_k)
    # var_mix = (1/K) * sum(var_k) + (1/K) * sum(mu_k^2) - mu_mix^2
    #         = E[var] + E[mu^2] - E[mu]^2
    #         = E[var] + Var[mu]

    # Mean of the mixture: average over ensemble dimension (dim=0)
    mixture_mean = per_model_mean.mean(dim=0)  # n x m

    # Variance of the mixture using law of total variance:
    # Var(Y) = E[Var(Y|K)] + Var(E[Y|K])
    # where K indexes the ensemble member
    mean_of_variances = per_model_variance.mean(dim=0)  # E[Var(Y|K)]
    variance_of_means = per_model_mean.var(dim=0)  # Var(E[Y|K])
    mixture_variance = mean_of_variances + variance_of_means  # n x m

    # Get observed Y - should be the same for all ensemble members
    # Take from the first ensemble member
    observed_Y = loo_results.observed_Y
    if observed_Y.dim() >= 3:
        observed_Y = observed_Y[0]  # Remove ensemble dimension

    # Handle observed Yvar if present
    observed_Yvar = loo_results.observed_Yvar
    if observed_Yvar is not None and observed_Yvar.dim() >= 3:
        observed_Yvar = observed_Yvar[0]  # Remove ensemble dimension

    return EnsembleLOOCVResults(
        mean=mixture_mean,
        variance=mixture_variance,
        observed_Y=observed_Y,
        observed_Yvar=observed_Yvar,
        model=model,
        per_model_mean=per_model_mean,
        per_model_variance=per_model_variance,
    )
