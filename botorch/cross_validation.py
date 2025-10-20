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
