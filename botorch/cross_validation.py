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
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from linear_operator.operators import DiagLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor


class CVFolds(NamedTuple):
    train_X: Tensor
    test_X: Tensor
    train_Y: Tensor
    test_Y: Tensor
    train_Yvar: Tensor | None = None
    test_Yvar: Tensor | None = None


class CVResults(NamedTuple):
    """Results from cross-validation.

    This named tuple contains the cross-validation predictions and observed values.
    For both ``batch_cross_validation`` and ``efficient_loo_cv``, the ``posterior``
    field contains the predictive distribution with mean and variance accessible
    via ``posterior.mean`` and ``posterior.variance``.

    For ``batch_cross_validation``, the posterior has shape ``n x 1 x m`` where n
    is the number of folds, 1 is the single held-out point per fold, and m is the
    number of outputs.

    For ``efficient_loo_cv``, the posterior has the same shape structure to maintain
    consistency, though the underlying distribution is constructed from the
    efficient LOO formulas rather than from separate model fits.
    """

    model: GPyTorchModel
    posterior: GPyTorchPosterior
    observed_Y: Tensor
    observed_Yvar: Tensor | None = None


def gen_loo_cv_folds(
    train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None
) -> CVFolds:
    r"""Generate LOO CV folds w.r.t. to ``n``.

    Args:
        train_X: A ``n x d`` or ``batch_shape x n x d`` (batch mode) tensor of training
            features.
        train_Y: A ``n x (m)`` or ``batch_shape x n x (m)`` (batch mode) tensor of
            training observations.
        train_Yvar: An ``n x (m)`` or ``batch_shape x n x (m)`` (batch mode) tensor
            of observed measurement noise.

    Returns:
        CVFolds NamedTuple with the following fields:

        - train_X: A ``n x (n-1) x d`` or ``batch_shape x n x (n-1) x d`` tensor of
          training features.
        - test_X: A ``n x 1 x d`` or ``batch_shape x n x 1 x d`` tensor of test
          features.
        - train_Y: A ``n x (n-1) x m`` or ``batch_shape x n x (n-1) x m`` tensor of
          training observations.
        - test_Y: A ``n x 1 x m`` or ``batch_shape x n x 1 x m`` tensor of test
          observations.
        - train_Yvar: A ``n x (n-1) x m`` or ``batch_shape x n x (n-1) x m`` tensor
          of observed measurement noise.
        - test_Yvar: A ``n x 1 x m`` or ``batch_shape x n x 1 x m`` tensor of observed
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
        cv_folds: A CVFolds tuple. For LOO-CV with n training points, the leading
            dimension of size n represents the n folds (batch dimension), e.g.,
            ``cv_folds.train_X`` has shape ``n x (n-1) x d`` and ``cv_folds.test_X``
            has shape ``n x 1 x d``. This batch structure enables fitting n
            independent GPs simultaneously.
        fit_args: Arguments passed along to fit_gpytorch_mll.
        model_init_kwargs: Keyword arguments passed to the model constructor.

    Returns:
        A CVResults tuple with the following fields

        - model: GPyTorchModel for batched cross validation
        - posterior: GPyTorchPosterior where the mean has shape ``n x 1 x m`` or
          ``batch_shape x n x 1 x m``
        - observed_Y: A ``n x 1 x m`` or ``batch_shape x n x 1 x m`` tensor of
          observations.
        - observed_Yvar: A ``n x 1 x m`` or ``batch_shape x n x 1 x m`` tensor
          of observed measurement noise.

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


def loo_cv(model: GPyTorchModel, observation_noise: bool = True) -> CVResults:
    r"""Compute efficient Leave-One-Out cross-validation for a GP model.

    This is a high-level convenience function that automatically dispatches to
    the appropriate LOO CV implementation based on the model type:

    - For ensemble models (``_is_ensemble=True``): Uses ``ensemble_loo_cv`` which
      returns a ``GaussianMixturePosterior`` with both per-member and mixture
      statistics.
    - For standard GP models: Uses ``efficient_loo_cv`` which returns a
      ``GPyTorchPosterior`` with the LOO predictive distributions.

    Both implementations use efficient O(n³) matrix algebra rather than the
    naive O(n⁴) approach of refitting models for each fold.

    NOTE: This function does not refit the model to each LOO fold. The model
    hyperparameters are kept fixed, providing a fast approximation to full
    LOO CV. For models where hyperparameter changes are significant, consider
    using ``batch_cross_validation`` instead.

    Args:
        model: A fitted GPyTorchModel. The model type determines which LOO CV
            implementation is used.
        observation_noise: If True (default), return the posterior
            predictive variance (including observation noise). If False,
            return the posterior variance of the latent function (excluding
            observation noise). The posterior variance is computed by
            subtracting the observation noise from the posterior predictive
            variance.

    Returns:
        CVResults: A named tuple containing:
            - model: The fitted GP model.
            - posterior: The LOO predictive distributions. For ensemble models,
              this is a ``GaussianMixturePosterior``; otherwise, it's a
              ``GPyTorchPosterior``.
            - observed_Y: The observed Y values.
            - observed_Yvar: The observed noise variances (if applicable).

    Example:
        >>> import torch
        >>> from botorch.cross_validation import loo_cv
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.fit import fit_gpytorch_mll
        >>> from gpytorch.mlls import ExactMarginalLogLikelihood
        >>>
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
        >>> loo_results = loo_cv(model)
        >>> loo_results.posterior.mean.shape
        torch.Size([20, 1, 1])

    See Also:
        - ``efficient_loo_cv``: Direct access to the standard GP implementation.
        - ``ensemble_loo_cv``: Direct access to the ensemble model implementation.
        - ``batch_cross_validation``: Full LOO CV with model refitting.
    """
    if getattr(model, "_is_ensemble", False):
        return ensemble_loo_cv(model, observation_noise=observation_noise)
    else:
        return efficient_loo_cv(model, observation_noise=observation_noise)


def efficient_loo_cv(
    model: GPyTorchModel,
    observation_noise: bool = True,
) -> CVResults:
    r"""Compute efficient Leave-One-Out cross-validation for a GP model.

    NOTE: This function does not refit the model to each LOO fold, in contrast to
    batch_cross_validation. This is a memory- and compute-efficient way to compute LOO,
    but it does not account for potential changes in the model parameters due to the
    removal of a single observation. This is typically ok in cases with a lot of data,
    but can result in substantial differences (typically over-estimating performance)
    in the low data regime.

    This function leverages a well-known linear algebraic identity to compute
    all LOO predictive distributions in O(n^3) time, compared to the naive
    approach which requires O(n^4) time (O(n^3) per fold for n folds).

    The efficient LOO formulas for GPs are:

    .. math::

        \mu_{LOO,i} = y_i - \frac{[K^{-1}(y - \mu)]_i}{[K^{-1}]_{ii}}

        \sigma^2_{LOO,i} = \frac{1}{[K^{-1}]_{ii}}

    where K is the covariance matrix including observation noise. This gives
    the posterior predictive variance (including noise). To get the posterior
    variance (excluding noise), we subtract the observation noise:

    .. math::

        \sigma^2_{posterior,i} = \sigma^2_{LOO,i} - \sigma^2_{noise}

    NOTE: This function assumes the model has already been fitted and that the
    model's ``forward`` method returns a ``MultivariateNormal`` distribution.

    Args:
        model: A fitted GPyTorchModel whose ``forward`` method returns a
            ``MultivariateNormal`` distribution.
        observation_noise: If True (default), return the posterior
            predictive variance (including observation noise). If False,
            return the posterior variance of the latent function (excluding
            observation noise).

    Returns:
        CVResults: A named tuple containing:
            - model: The fitted GP model.
            - posterior: A GPyTorchPosterior with the LOO predictive distributions.
              The posterior mean and variance have shape ``n x 1 x m`` or
              ``batch_shape x n x 1 x m``, matching the structure of
              ``batch_cross_validation`` (n folds, 1 held-out point per fold,
              m outputs). The underlying distribution has diagonal covariance
              since LOO predictions at different held-out points are computed
              independently.
            - observed_Y: The observed Y values with shape ``n x 1 x m`` or
              ``batch_shape x n x 1 x m``.
            - observed_Yvar: The observed noise variances (if provided) with shape
              ``n x 1 x m`` or ``batch_shape x n x 1 x m``.

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
        >>> loo_results.posterior.mean.shape
        torch.Size([20, 1, 1])
    """
    # Compute raw LOO predictions
    loo_mean, loo_variance, train_Y = _compute_loo_predictions(
        model, observation_noise=observation_noise
    )

    # Get the number of outputs
    num_outputs = model.num_outputs

    # Build the posterior from raw LOO predictions
    posterior = _build_loo_posterior(
        loo_mean=loo_mean, loo_variance=loo_variance, num_outputs=num_outputs
    )

    # Reshape observed data to LOO CV output format: n x 1 x m
    observed_Y = _reshape_to_loo_cv_format(train_Y, num_outputs)

    # Get observed Yvar if available (for fixed noise models)
    observed_Yvar = None
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        observed_Yvar = _reshape_to_loo_cv_format(model.likelihood.noise, num_outputs)

    return CVResults(
        model=model,
        posterior=posterior,
        observed_Y=observed_Y,
        observed_Yvar=observed_Yvar,
    )


def _subtract_observation_noise(model: GPyTorchModel, loo_variance: Tensor) -> Tensor:
    r"""Subtract observation noise from LOO variance to get posterior variance.

    The efficient LOO formula computes the posterior predictive variance, which
    includes observation noise. To get the posterior variance of the latent
    function (without noise), we subtract the observation noise variance.

    This implementation uses the likelihood's ``forward`` method to extract noise
    variances in a general way. The ``forward`` method takes function samples and
    returns a distribution where the variance represents the observation noise.

    .. math::

        \sigma^2_{posterior,i} = \sigma^2_{LOO,i} - \sigma^2_{noise}

    Args:
        model: The GP model with a likelihood containing the noise variance.
        loo_variance: The LOO posterior predictive variance with shape
            ``... x n x 1``.

    Returns:
        The posterior variance (without noise) with the same shape.
    """
    likelihood = model.likelihood

    # Use the likelihood's forward method to extract noise variances.
    # By passing zeros as function samples, the returned distribution's
    # variance gives us the observation noise at each point.
    noise_shape = loo_variance.shape[:-1]  # ... x n
    zeros = torch.zeros(
        noise_shape, dtype=loo_variance.dtype, device=loo_variance.device
    )

    # Some likelihoods (e.g., SparseOutlierGaussianLikelihood) require training
    # inputs to be passed to correctly compute the noise. We pass the model's
    # train_inputs if available.
    train_inputs = getattr(model, "train_inputs", None)

    # Call forward to get the observation noise distribution.
    # We pass train_inputs as a positional argument so it flows through *params
    # to the noise model, which is compatible with both standard Noise classes
    # (that use *params) and SparseOutlierNoise (that uses X as the first arg).
    noise_dist = likelihood.forward(zeros, train_inputs)

    # Extract noise variance and reshape to match loo_variance
    noise = noise_dist.variance.unsqueeze(-1)  # ... x n x 1

    loo_variance = loo_variance - noise

    # Clamp to ensure non-negative variance
    return loo_variance.clamp(min=0.0)


def _compute_loo_predictions(
    model: GPyTorchModel,
    observation_noise: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute raw LOO predictions (means and variances) for a GP model.

    This is an internal helper that computes the leave-one-out predictive means
    and variances using efficient matrix algebra. The formulas are:

    .. math::

        \mu_{LOO,i} = y_i - \frac{[K^{-1}(y - \mu)]_i}{[K^{-1}]_{ii}}

        \sigma^2_{LOO,i} = \frac{1}{[K^{-1}]_{ii}}

    where K is the covariance matrix including observation noise and μ is the
    prior mean. This gives the posterior predictive variance (including noise).
    To get the posterior variance (excluding noise), we subtract the observation
    noise variance.

    Args:
        model: A fitted GPyTorchModel in eval mode whose ``forward`` method returns
            a ``MultivariateNormal`` distribution.
        observation_noise: If True (default), return the posterior
            predictive variance (including observation noise). If False,
            return the posterior variance of the latent function (excluding
            observation noise).

    Returns:
        A tuple of (loo_mean, loo_variance, train_Y) where:
        - loo_mean: LOO predictive means with shape ``... x n x 1``
        - loo_variance: LOO predictive variances with shape ``... x n x 1``
        - train_Y: The training targets from the model

    Raises:
        UnsupportedError: If the model doesn't have required attributes or
            the forward method doesn't return a MultivariateNormal.
    """
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

    # Check for models with auxiliary inputs (e.g., auxiliary experiment data)
    # In such models, train_inputs[0] is a tuple of tensors rather than a single tensor
    if isinstance(train_X, tuple):
        raise UnsupportedError(
            "Efficient LOO CV is not supported for models with auxiliary inputs. "
            "train_inputs[0] is a tuple of tensors, indicating auxiliary data."
        )

    train_Y = model.train_targets  # Shape: n or batch_shape x n (batched outputs)

    n = train_X.shape[-2]
    prior_dist = model.forward(train_X)

    # Check that we got a MultivariateNormal
    if not isinstance(prior_dist, MultivariateNormal):
        raise UnsupportedError(
            f"Model's forward method must return a MultivariateNormal, "
            f"got {type(prior_dist).__name__}."
        )

    # Extract mean from the prior
    # Shape: n for single-output, or m x n for batched multi-output
    mean = prior_dist.mean

    # Add observation noise to the diagonal via the likelihood
    # The likelihood adds noise: K_noisy = K + sigma^2 * I
    # Some likelihoods (e.g., SparseOutlierGaussianLikelihood) require training
    # inputs to be passed to correctly apply the noise model. We pass them as
    # a positional argument for compatibility with both standard likelihoods
    # and SparseOutlierGaussianLikelihood.
    train_inputs = model.train_inputs
    noisy_mvn = model.likelihood(prior_dist, train_inputs)

    # Get the covariance matrix - use lazy representation for potential caching
    K_noisy = noisy_mvn.lazy_covariance_matrix.to_dense()

    # Compute Cholesky decomposition (adds jitter if needed)
    L = psd_safe_cholesky(K_noisy)

    # Compute K^{-1}(y - mean) via Cholesky solve
    # Shape: ... x n x 1 where ... is batch_shape (includes m for multi-output)
    residuals = (train_Y - mean).unsqueeze(-1)
    K_inv_residuals = torch.cholesky_solve(residuals, L)

    # Compute diagonal of K^{-1}
    # K_inv = L^{-T} @ L^{-1}, so K_inv_diag[i] = sum_j (L^{-1}[j,i])^2
    identity = torch.eye(n, dtype=L.dtype, device=L.device)
    if L.dim() > 2:
        identity = identity.expand(*L.shape[:-2], n, n)
    L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
    K_inv_diag = (L_inv**2).sum(dim=-2)  # ... x n

    # Compute LOO predictions using the efficient formulas:
    # sigma2_loo_i = 1 / [K^{-1}]_{ii}
    # mu_loo_i = y_i - [K^{-1}(y - mean)]_i / [K^{-1}]_{ii}
    # K_inv_diag has shape ... x n, so after unsqueeze(-1) we get ... x n x 1
    # (the last dim is 1 because each GP is single-output).
    loo_variance = (1.0 / K_inv_diag).unsqueeze(-1)  # ... x n x 1
    loo_mean = train_Y.unsqueeze(-1) - K_inv_residuals * loo_variance  # ... x n x 1

    # If we want the posterior (noiseless) variance, subtract the noise
    if not observation_noise:
        loo_variance = _subtract_observation_noise(model, loo_variance)

    return loo_mean, loo_variance, train_Y


def _build_loo_posterior(
    loo_mean: Tensor,
    loo_variance: Tensor,
    num_outputs: int,
) -> GPyTorchPosterior:
    r"""Build a GPyTorchPosterior from raw LOO predictions.

    Args:
        loo_mean: LOO means with shape ``... x m x n x 1`` (multi-output) or
            ``... x n x 1`` (single-output), where ``...`` is optional batch_shape.
        loo_variance: LOO variances with same shape as loo_mean.
        num_outputs: Number of outputs (m). 1 for single-output models.

    Returns:
        A GPyTorchPosterior with shape ``... x n x 1 x m``.
    """
    # Reshape tensors to final shape: ... x n x 1 x m
    if num_outputs > 1:
        # Multi-output: ... x m x n x 1 -> ... x n x 1 x m
        # The m dimension is at position -3, move it to position -1
        loo_mean = loo_mean.movedim(-3, -1)
        loo_variance = loo_variance.movedim(-3, -1)
    else:
        # Single-output: ... x n x 1 -> ... x n x 1 x 1
        loo_mean = loo_mean.unsqueeze(-1)
        loo_variance = loo_variance.unsqueeze(-1)

    # Create distribution: for multi-output use MTMVN, for single-output use MVN.
    # Both require mean shape ... x n x q (where q=1) and diagonal covariance.
    # We squeeze the m dimension to get ... x n x 1 for the MVN mean, then
    # iterate over outputs to create independent MVNs.
    mvns = [
        MultivariateNormal(
            mean=loo_mean[..., t],
            covariance_matrix=DiagLinearOperator(loo_variance[..., t]),
        )
        for t in range(num_outputs)
    ]

    if num_outputs > 1:
        mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
    else:
        mvn = mvns[0]

    return GPyTorchPosterior(distribution=mvn)


def _reshape_to_loo_cv_format(tensor: Tensor, num_outputs: int) -> Tensor:
    r"""Reshape a tensor to the standard LOO CV output format: ``n x 1 x m``.

    This helper converts tensors with internal model format (which varies by
    number of outputs) to the consistent output format used by LOO CV results.

    Args:
        tensor: Input tensor with shape:
            - Single-output: ``n`` (1D)
            - Multi-output: ``m x n`` (2D)
        num_outputs: Number of outputs (m). 1 for single-output models.

    Returns:
        Reshaped tensor with shape ``n x 1 x m``.
    """
    if num_outputs > 1:
        # Multi-output: m x n -> n x m -> n x 1 x m
        return tensor.movedim(-2, -1).unsqueeze(-2)
    else:
        # Single-output: n -> n x 1 -> n x 1 x 1
        return tensor.unsqueeze(-1).unsqueeze(-1)


def ensemble_loo_cv(
    model: GPyTorchModel,
    observation_noise: bool = True,
) -> CVResults:
    r"""Compute efficient LOO cross-validation for ensemble models.

    This function computes Leave-One-Out cross-validation for ensemble models
    like ``SaasFullyBayesianSingleTaskGP``. For these models, the ``forward`` method
    returns a ``MultivariateNormal`` with a batch dimension containing statistics
    for all models in the ensemble.

    The LOO predictions from each ensemble member form a Gaussian mixture.
    This function returns a ``CVResults`` with a ``GaussianMixturePosterior`` that
    provides both per-member statistics (via ``posterior.mean`` and
    ``posterior.variance``) and aggregated mixture statistics (via
    ``posterior.mixture_mean`` and ``posterior.mixture_variance``).

    The mixture statistics are computed using the law of total variance:

    .. math::

        \mu_{mix} = \frac{1}{K} \sum_{k=1}^{K} \mu_k

        \sigma^2_{mix} = \frac{1}{K} \sum_{k=1}^{K} \sigma^2_k +
            \frac{1}{K} \sum_{k=1}^{K} \mu_k^2 - \mu_{mix}^2

    where K is the number of ensemble members.

    NOTE: This function assumes the model has already been fitted (e.g., using
    ``fit_fully_bayesian_model_nuts``) and that the model is an ensemble model
    with ``_is_ensemble = True``.

    Args:
        model: An ensemble GPyTorchModel (e.g., SaasFullyBayesianSingleTaskGP)
            whose ``forward`` method returns a ``MultivariateNormal`` distribution
            with a batch dimension for ensemble members.
        observation_noise: If True (default), return the posterior
            predictive variance (including observation noise). If False,
            return the posterior variance of the latent function (excluding
            observation noise).

    Returns:
        CVResults: A named tuple containing:
            - model: The fitted ensemble GP model.
            - posterior: A ``GaussianMixturePosterior`` with per-member shape
              ``n x num_models x 1 x 1``. Access per-member statistics via
              ``posterior.mean`` and ``posterior.variance``, and mixture
              statistics via ``posterior.mixture_mean`` and
              ``posterior.mixture_variance``.
            - observed_Y: The observed Y values with shape ``n x 1 x 1``.
            - observed_Yvar: The observed noise variances (if provided).

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
        >>> loo_results.posterior.mean.shape  # Per-member means
        torch.Size([20, 32, 1, 1])
        >>> loo_results.posterior.mixture_mean.shape  # Aggregated mixture mean
        torch.Size([20, 1, 1])
    """
    # Check that this is an ensemble model
    if not getattr(model, "_is_ensemble", False):
        raise UnsupportedError(
            "ensemble_loo_cv requires an ensemble model (with _is_ensemble=True). "
            f"Got model of type {type(model).__name__}. "
            "For non-ensemble models, use efficient_loo_cv instead."
        )

    # Compute raw LOO predictions
    # For ensemble models, shapes are: num_models x n x 1
    loo_mean, loo_variance, train_Y = _compute_loo_predictions(
        model, observation_noise=observation_noise
    )

    # Validate that we have the expected batch dimension for ensemble
    if loo_mean.dim() < 3:
        raise UnsupportedError(
            "Expected ensemble model to produce batched LOO results with shape "
            f"(batch_shape x num_models x n x 1), but got shape {loo_mean.shape}."
        )

    # Get the number of outputs
    num_outputs = getattr(model, "_num_outputs", 1)

    # Build the GaussianMixturePosterior
    posterior = _build_ensemble_loo_posterior(
        loo_mean=loo_mean, loo_variance=loo_variance, num_outputs=num_outputs
    )

    # Extract observed data (first ensemble member) and reshape to LOO CV format
    observed_Y, observed_Yvar = _get_ensemble_observed_data(
        model=model, train_Y=train_Y, num_outputs=num_outputs
    )

    return CVResults(
        model=model,
        posterior=posterior,
        observed_Y=observed_Y,
        observed_Yvar=observed_Yvar,
    )


def _build_ensemble_loo_posterior(
    loo_mean: Tensor,
    loo_variance: Tensor,
    num_outputs: int,
) -> GaussianMixturePosterior:
    r"""Build a GaussianMixturePosterior from raw ensemble LOO predictions.

    This function takes raw LOO means and variances from an ensemble model
    (computed by ``_compute_loo_predictions``) and packages them into a
    GaussianMixturePosterior that provides both per-member and mixture statistics.

    Args:
        loo_mean: LOO means with shape ``batch_shape x num_models x n x 1``
            (single-output) or ``batch_shape x num_models x m x n x 1``
            (multi-output).
        loo_variance: LOO variances with same shape as loo_mean.
        num_outputs: Number of outputs (m). 1 for single-output models.

    Returns:
        GaussianMixturePosterior with shape ``batch_shape x n x num_models x 1 x m``.
        The num_models dimension is at MCMC_DIM=-3.
    """
    # Normalize shapes: add m=1 dimension for single-output to match multi-output
    if num_outputs == 1:
        # Single-output: ... x num_models x n x 1 -> ... x num_models x 1 x n x 1
        loo_mean = loo_mean.unsqueeze(-3)
        loo_variance = loo_variance.unsqueeze(-3)

    # Now both cases have shape: ... x num_models x m x n x 1
    # Transform to target shape: ... x n x num_models x 1 x m
    # 1. squeeze(-1): ... x num_models x m x n
    # 2. movedim(-1, -3): ... x n x num_models x m (move n before num_models)
    # 3. unsqueeze(-2): ... x n x num_models x 1 x m
    loo_mean = loo_mean.squeeze(-1).movedim(-1, -3).unsqueeze(-2)
    loo_variance = loo_variance.squeeze(-1).movedim(-1, -3).unsqueeze(-2)

    # Create distribution: iterate over outputs to create independent MVNs
    # After indexing with [..., t], shape is: batch_shape x n x num_models x 1
    mvns = [
        MultivariateNormal(
            mean=loo_mean[..., t],
            covariance_matrix=DiagLinearOperator(loo_variance[..., t]),
        )
        for t in range(num_outputs)
    ]

    if num_outputs > 1:
        mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
    else:
        mvn = mvns[0]

    return GaussianMixturePosterior(distribution=mvn)


def _get_ensemble_observed_data(
    model: GPyTorchModel,
    train_Y: Tensor,
    num_outputs: int,
) -> tuple[Tensor, Tensor | None]:
    r"""Extract observed data from an ensemble model for LOO CV.

    Extracts the first ensemble member's training targets and observation noise,
    verifies all members share the same data, and reshapes to LOO CV format.

    Args:
        model: The ensemble GP model.
        train_Y: Training targets with shape ``... x num_models x n`` (single-output)
            or ``... x num_models x m x n`` (multi-output).
        num_outputs: Number of outputs (m).

    Returns:
        (observed_Y, observed_Yvar) with shape ``... x n x 1 x m``.

    Raises:
        UnsupportedError: If ensemble members have different training data.
    """
    # num_models is at dim -2 for single-output, -3 for multi-output
    num_models_dim = -2 if num_outputs == 1 else -3

    # Verify all ensemble members share the same training data
    _verify_ensemble_data_consistency(train_Y, num_models_dim, "train_Y")

    # Extract first ensemble member's data (they're all the same)
    train_Y_first = train_Y.select(num_models_dim, 0)
    observed_Y = _reshape_to_loo_cv_format(train_Y_first, num_outputs)

    # Get observed Yvar if available (for fixed noise models)
    observed_Yvar = None
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        noise = model.likelihood.noise
        # Noise has the same shape structure as train_Y
        # Verify consistency and extract first member
        if noise.dim() > 1:
            _verify_ensemble_data_consistency(
                noise, num_models_dim, "observation noise"
            )
            noise = noise.select(num_models_dim, 0)
        observed_Yvar = _reshape_to_loo_cv_format(noise, num_outputs)

    return observed_Y, observed_Yvar


def _verify_ensemble_data_consistency(
    tensor: Tensor,
    num_models_dim: int,
    tensor_name: str,
) -> None:
    r"""Verify all ensemble members have identical data along ``num_models_dim``.

    Args:
        tensor: Data tensor with a num_models dimension.
        num_models_dim: Dimension index for num_models (typically -2 or -3).
        tensor_name: Name for error messages (e.g., "train_Y").

    Raises:
        UnsupportedError: If data differs across ensemble members.
    """
    num_models = tensor.shape[num_models_dim]
    if num_models <= 1:
        return

    first_member = tensor.select(num_models_dim, 0)
    first_expanded = first_member.unsqueeze(num_models_dim).expand_as(tensor)

    if not torch.allclose(tensor, first_expanded):
        raise UnsupportedError(
            f"Ensemble members have different {tensor_name}. "
            "ensemble_loo_cv only supports ensembles where all members share the "
            "same training data (e.g., fully Bayesian models with MCMC samples). "
            "For ensembles with different data per member, cross-validate each "
            "member individually using efficient_loo_cv."
        )
