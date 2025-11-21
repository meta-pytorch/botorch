#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract class for acquisition functions leveraging a cached Cholesky
decomposition of the posterior covariance over f(X_baseline).
"""

from __future__ import annotations

import warnings

import torch
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.higher_order_gp import HigherOrderGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.utils.low_rank import extract_batch_covar, sample_cached_cholesky
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from linear_operator.utils.errors import NanError, NotPSDError
from torch import Tensor


def supports_cache_root(model: Model) -> bool:
    r"""Checks if a model supports the cache_root functionality.
    The two criteria are that the model is not multi-task and the model
    produces a GPyTorchPosterior.
    """
    if isinstance(model, ModelListGP):
        return all(supports_cache_root(m) for m in model.models)
    # Multi task models and non-GPyTorch models are not supported.
    if isinstance(
        model, (MultiTaskGP, KroneckerMultiTaskGP, HigherOrderGP)
    ) or not isinstance(model, GPyTorchModel):
        return False
    # Models that return a TransformedPosterior are not supported.
    if hasattr(model, "outcome_transform") and (not model.outcome_transform._is_linear):
        return False
    return True


def _get_cache_root_not_supported_message(model_cls: type) -> str:
    msg = (
        "`cache_root` is only supported for GPyTorchModels that "
        "are not MultiTask models and don't produce a "
        f"TransformedPosterior. Got a model of type {model_cls}. Setting "
        "`cache_root = False`."
    )
    return msg


class CachedCholeskyMCSamplerMixin(MCSamplerMixin):
    r"""Abstract Mixin class for acquisition functions using a cached Cholesky.

    Specifically, this is for acquisition functions that require sampling from
    the posterior P(f(X_baseline, X) | D). The Cholesky of the posterior
    covariance over f(X_baseline) is cached.
    """

    def __init__(
        self,
        model: Model,
        cache_root: bool | None = None,
        sampler: MCSampler | None = None,
    ) -> None:
        r"""Set class attributes and perform compatibility checks.

        Args:
            model: A model.
            cache_root: A boolean indicating whether to cache the Cholesky.
                If None, will be set to True if the model supports it and False
                otherwise.
            sampler: An optional MCSampler object.
        """
        MCSamplerMixin.__init__(self, sampler=sampler)
        if cache_root is None:
            cache_root = supports_cache_root(model)
        elif cache_root and not supports_cache_root(model):
            warnings.warn(
                _get_cache_root_not_supported_message(type(model)),
                RuntimeWarning,
                stacklevel=3,
            )
            cache_root = False
        self._cache_root = cache_root

    def _compute_root_decomposition(
        self,
        posterior: Posterior,
    ) -> Tensor:
        r"""Cache Cholesky of the posterior covariance over f(X_baseline).

        Because `LinearOperator.root_decomposition` is decorated with LinearOperator's
        @cached decorator, this function is doing a lot implicitly:

        1) Check if a root decomposition has already been cached to `lazy_covar`.
          Note that it will not have been if `posterior.mvn` is a
          `MultitaskMultivariateNormal`, since we construct `lazy_covar` in that
          case.
        2) If the root decomposition has not been found in the cache, compute it.
        3) Write it to the cache of `lazy_covar`. Note that this will become
          inaccessible if `posterior.mvn` is a `MultitaskMultivariateNormal`,
          since in that case `lazy_covar`'s scope is only this function.

        Args:
            posterior: The posterior over f(X_baseline).
        """
        if isinstance(posterior.distribution, MultitaskMultivariateNormal):
            lazy_covar = extract_batch_covar(posterior.distribution)
        else:
            lazy_covar = posterior.distribution.lazy_covariance_matrix
        lazy_covar_root = lazy_covar.root_decomposition()
        return lazy_covar_root.root.to_dense()

    def _get_f_X_samples(self, posterior: GPyTorchPosterior, q_in: int) -> Tensor:
        r"""Get posterior samples at the `q_in` new points from the joint posterior.

        Args:
            posterior: The joint posterior is over (X_baseline, X).
            q_in: The number of new points in the posterior. See `_set_sampler` for
                more information.

        Returns:
            A `sample_shape x batch_shape x q x m`-dim tensor of posterior
                samples at the new points.
        """
        # Technically we should make sure that we add a consistent nugget to the
        # cached covariance (and box decompositions) and the new block.
        # But recomputing box decompositions every time the jitter changes would
        # be quite slow.
        if self._cache_root and hasattr(self, "_baseline_L"):
            try:
                return sample_cached_cholesky(
                    posterior=posterior,
                    baseline_L=self._baseline_L,
                    q=q_in,
                    base_samples=self.sampler.base_samples,
                    sample_shape=self.sampler.sample_shape,
                )
            except (NanError, NotPSDError):
                warnings.warn(
                    "Low-rank cholesky updates failed due NaNs or due to an "
                    "ill-conditioned covariance matrix. "
                    "Falling back to standard sampling.",
                    BotorchWarning,
                    stacklevel=3,
                )

        # TODO: improve efficiency for multi-task models
        samples = self.get_posterior_samples(posterior)
        if isinstance(self.model, HigherOrderGP):
            # Select the correct q-batch dimension for HOGP.
            q_dim = -self.model._num_dimensions
            q_idcs = (
                torch.arange(-q_in, 0, device=samples.device) + samples.shape[q_dim]
            )
            return samples.index_select(q_dim, q_idcs)
        else:
            return samples[..., -q_in:, :]

    def _set_sampler(
        self,
        q_in: int,
        posterior: Posterior,
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        Args:
            q_in: The effective input batch size. This is typically equal to the
                q-batch size of `X`. However, if using a one-to-many input transform,
                e.g., `InputPerturbation` with `n_w` perturbations, the posterior will
                have `n_w` points on the q-batch for each point on the q-batch of `X`.
                In which case, `q_in = q * n_w` is used.
            posterior: The posterior.
        """
        if self.q_in != q_in and self.base_sampler is not None:
            self.sampler._update_base_samples(
                posterior=posterior, base_sampler=self.base_sampler
            )
            self.q_in = q_in
