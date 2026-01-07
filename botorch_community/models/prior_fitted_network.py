#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This module defines the botorch model for PFNs (`PFNModel`) and it
provides handy helpers to download pretrained, public PFNs
with `download_model` and model paths with `ModelPaths`.
For the latter to work `pfns4bo` must be installed.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.logging import logger
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import match_batch_shape
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from botorch_community.posteriors.riemann import (
    BoundedRiemannPosterior,
    MultivariateRiemannPosterior,
)
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from pfns.train import MainConfig  # @manual=//pytorch/PFNs:PFNs
from torch import Tensor
from torch.nn import Module


def get_styles(
    model: Module, hps: dict | None, batch_size: int, device: str
) -> dict[str, Tensor]:
    if hps is None or (model.style_encoder is None and model.y_style_encoder is None):
        return {}
    style_kwargs = {}
    if model.style_encoder is not None:
        hps_subset = {
            k: v for k, v in hps.items() if k in model.style_encoder[0].hyperparameters
        }
        style = (
            model.style_encoder[0]
            .hyperparameters_dict_to_tensor(hps_subset)
            .repeat(batch_size, 1)
            .to(device)
            .float()
        )  # shape (batch_size, num_styles)
        style_kwargs["style"] = style

    if model.y_style_encoder is not None:
        hps_subset = {
            k: v
            for k, v in hps.items()
            if k in model.y_style_encoder[0].hyperparameters
        }
        y_style = (
            model.y_style_encoder[0]
            .hyperparameters_dict_to_tensor(hps_subset)
            .repeat(batch_size, 1)
            .to(device)
            .float()
        )  # shape (batch_size, num_styles)
        style_kwargs["y_style"] = y_style
    return style_kwargs


class PFNModel(Model):
    """Prior-data Fitted Network"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        model: Module | None = None,
        checkpoint_url: str = ModelPaths.pfns4bo_hebo,
        train_Yvar: Tensor | None = None,
        batch_first: bool = False,
        constant_model_kwargs: dict[str, Any] | None = None,
        input_transform: InputTransform | None = None,
        load_training_checkpoint: bool = False,
        style_hyperparameters: dict[str, Any] | None = None,
        style: Tensor
        | None = None,  # should have shape (num_styles,) or (num_features, num_styles)
    ) -> None:
        """Initialize a PFNModel.

        Either a pre-trained PFN model can be provided via the model kwarg,
        or a checkpoint_url can be provided from which the model will be
        downloaded. This defaults to the pfns4bo_hebo model.

        Loading the model does an unsafe "weights_only=False" load, so
        it is essential that checkpoint_url be a trusted source.

        Args:
            train_X: A `n x d` tensor of training features.
            train_Y: A `n x 1` tensor of training observations.
            model: A pre-trained PFN model with the following
                forward(train_X, train_Y, X) -> logit predictions of shape
                `n x b x c` where c is the number of discrete buckets
                borders: A `c+1`-dim tensor of bucket borders.
            checkpoint_url: The string URL of the PFN model to download and load.
                Will be ignored if model is provided.
            train_Yvar: Observed variance of train_Y. Currently ignored.
            batch_first: Whether the batch dimension is the first dimension of
                the input tensors. This is needed to support different PFN
                models. For batch-first x has shape `batch x seq_len x features`
                and for non-batch-first it has shape `seq_len x batch x features`.
            constant_model_kwargs: A dictionary of model kwargs that
                will be passed to the model in each forward pass.
            input_transform: A Botorch input transform.
            load_training_checkpoint: Whether to load a training checkpoint as
                produced by the PFNs training code, see github.com/automl/PFNs.
            style_hyperparameters: A dictionary of hyperparameters to be passed
                to the style and the y-style encoders. It is useful when training
                models with `hyperparameter_sampling` prior and its style
                encoder. One simply supplies the dict with the unnormalized
                hyperparameters, e.g., {"noise_std": 0.1}. Omitted values are
                treated as unknown and the value will build a Bayesian average
                for these, if `hyperparameter_sampling_skip_style_prob` > 0
                during pre-training.
            style: A tensor of style values to be passed to the model. These
                are raw style values of shape (num_styles,), which will then
                be extended as needed.

        """
        super().__init__()
        if model is None:
            model = download_model(
                model_path=checkpoint_url,
            )

        if load_training_checkpoint:
            # the model is not an actual model, but a training checkpoint
            # make a model out of it
            checkpoint = model
            config = MainConfig.from_dict(checkpoint["config"])
            model = config.model.create_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        if train_Yvar is not None:
            logger.debug("train_Yvar provided but ignored for PFNModel.")

        if train_Y.dim() != 2:
            raise UnsupportedError("train_Y must be 2-dimensional.")

        if train_X.dim() != 2:
            raise UnsupportedError("train_X must be 2-dimensional.")

        if train_Y.shape[-1] > 1:
            raise UnsupportedError("Only 1 target allowed for PFNModel.")

        if train_X.shape[0] != train_Y.shape[0]:
            raise UnsupportedError(
                "train_X and train_Y must have the same number of rows."
            )

        with torch.no_grad():
            self.transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self.train_X = train_X  # shape: (n, d)
        self.train_Y = train_Y  # shape: (n, 1)
        # Downstream botorch tooling expects a likelihood to be specified,
        # so here we use a FixedNoiseGaussianLikelihood that is unused.
        if train_Yvar is None:
            train_Yvar = torch.zeros_like(train_Y)
        self.train_Yvar = train_Yvar  # shape: (n, 1)
        self.likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
        self.pfn = model.to(device=train_X.device)
        self.batch_first = batch_first
        self.constant_model_kwargs = constant_model_kwargs or {}
        self.style_hyperparameters = style_hyperparameters
        self.style = style
        if input_transform is not None:
            self.input_transform = input_transform
        self._compute_styles()

    def _compute_styles(self):
        """
        Can be used to compute styles to be used for PFN prediction based on
        training data.

        When implemented, will directly modify self.style_hyperparameters or
        self.style.
        """
        pass

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        negate_train_ys: bool = False,
    ) -> BoundedRiemannPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A b? x q? x d`-dim Tensor, where `d` is the dimension of the
                feature space.
            output_indices: **Currently not supported for PFNModel.**
            observation_noise: **Currently not supported for PFNModel**.
            posterior_transform: **Currently not supported for PFNModel**.
            negate_train_ys: Whether to negate the training Ys. This is useful
                for minimization.

        Returns:
            A `BoundedRiemannPosterior`, representing a batch of b? x q?`
            distributions.
        """
        self.pfn.eval()
        if output_indices is not None:
            raise UnsupportedError(
                "output_indices is not None. PFNModel should not "
                "be a multi-output model."
            )
        if observation_noise:
            logger.warning(
                "observation_noise is not supported for PFNModel and is being ignored."
            )
        if posterior_transform is not None:
            raise UnsupportedError("posterior_transform is not supported for PFNModel.")

        X, train_X, train_Y, orig_X_shape, styles = self._prepare_data(
            X, negate_train_ys=negate_train_ys
        )

        probabilities = self.pfn_predict(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            **self.constant_model_kwargs,
            **styles,
        )  # (b, q, num_buckets)
        probabilities = probabilities.view(
            *orig_X_shape[:-1], -1
        )  # (b?, q?, num_buckets)

        return BoundedRiemannPosterior(
            borders=self.borders,
            probabilities=probabilities,
        )

    def _prepare_data(
        self, X: Tensor, negate_train_ys: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, torch.Size, dict[str, Tensor]]:
        orig_X_shape = X.shape  # X has shape b? x q? x d
        if len(X.shape) > 3:
            raise UnsupportedError(f"X must be at most 3-d, got {X.shape}.")
        while len(X.shape) < 3:
            X = X.unsqueeze(0)

        X = self.transform_inputs(X)  # shape (b , q, d)

        train_X = match_batch_shape(self.transformed_X, X)  # shape (b, n, d)
        train_Y = match_batch_shape(self.train_Y, X)  # shape (b, n, 1)
        if negate_train_ys:
            assert self.train_Y.mean().abs() < 1e-4, "train_Y must be zero-centered."
            train_Y = -train_Y
        styles = self._get_styles(
            batch_size=X.shape[0],
        )  # shape (b, num_styles)
        return X, train_X, train_Y, orig_X_shape, styles

    def _get_styles(self, batch_size) -> dict[str, Tensor]:
        style_kwargs = get_styles(
            model=self.pfn,
            hps=self.style_hyperparameters,
            batch_size=batch_size,
            device=self.train_X.device,
        )
        if self.style is not None:
            assert style_kwargs == {}, (
                "Cannot provide both style and style_hyperparameters."
            )
            style_kwargs["style"] = (
                self.style[None]
                .repeat(batch_size, 1, 1)
                .to(self.train_X.device)
                .float()
            )
        return style_kwargs

    def pfn_predict(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        **forward_kwargs,
    ) -> Tensor:
        """
        Make a prediction using the PFN model on X given training data.

        Args:
            X: has shape (b, q, d)
            train_X: has shape (b, n, d)
            train_Y: has shape (b, n, 1)
            **forward_kwargs: whatever kwargs to pass to the PFN model

        Returns: probabilities (b, q, num_buckets) for Riemann posterior.
        """

        if not self.batch_first:
            X = X.transpose(0, 1)  # shape (q, b, d)
            train_X = train_X.transpose(0, 1)  # shape (n, b, d)
            train_Y = train_Y.transpose(0, 1)  # shape (n, b, 1)

        logits = self.pfn(
            x=train_X.float(),
            y=train_Y.float(),
            test_x=X.float(),
            **forward_kwargs,
        )
        if not self.batch_first:
            logits = logits.transpose(0, 1)  # shape (b, q, num_buckets)
        logits = logits.to(X.dtype)

        probabilities = logits.softmax(dim=-1)  # shape (b, q, num_buckets)
        return probabilities

    @property
    def borders(self):
        return self.pfn.criterion.borders.to(self.train_X.dtype)


class PFNModelWithPendingPoints(PFNModel):
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        pending_X: Optional[Tensor] = None,
        negate_train_ys: bool = False,
    ) -> BoundedRiemannPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            `self.transform_inputs(X)` after the `self.eval()` call and before
            any `model.forward` or `model.likelihood` calls.

        Args:
            X: A b? x q? x d`-dim Tensor, where `d` is the dimension of the
                feature space.
            output_indices: **Currently not supported for PFNModel.**
            observation_noise: **Currently not supported for PFNModel**.
            posterior_transform: **Currently not supported for PFNModel**.
            pending_X: A tensor of shape n'' x d, where n'' is the number of
                pending points, which are to be observed but the value is
                not yet known.
            negate_train_ys: Whether to negate the training Ys. This is useful
                for minimization.

        Returns:
            A `BoundedRiemannPosterior`, representing a batch of b? x q?`
            distributions.
        """
        self.pfn.eval()
        if output_indices is not None:
            raise UnsupportedError(
                "output_indices is not None. PFNModel should not "
                "be a multi-output model."
            )
        if observation_noise:
            logger.warning(
                "observation_noise is not supported for PFNModel and is being ignored."
            )
        if posterior_transform is not None:
            raise UnsupportedError("posterior_transform is not supported for PFNModel.")

        X, train_X, train_Y, orig_X_shape, styles = self._prepare_data(
            X, negate_train_ys=negate_train_ys
        )

        if pending_X is not None:
            assert pending_X.dim() == 2, "pending_X must be 2-dimensional."
            pending_X = pending_X[None].repeat(X.shape[0], 1, 1)  # shape (b, n', d)
            train_X = torch.cat([train_X, pending_X], dim=1)  # shape (b, n+n', d)
            train_Y = torch.cat(
                [
                    train_Y,
                    torch.full(
                        (train_Y.shape[0], pending_X.shape[1], 1),
                        torch.nan,
                        device=train_Y.device,
                    ),
                ],
                dim=1,
            )  # shape (b, n+n', 1)

        probabilities = self.pfn_predict(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            **self.constant_model_kwargs,
            **styles,
        )  # (b, q, num_buckets)
        probabilities = probabilities.view(
            *orig_X_shape[:-1], -1
        )  # (b?, q?, num_buckets)

        return BoundedRiemannPosterior(
            borders=self.borders,
            probabilities=probabilities,
        )


class MultivariatePFNModel(PFNModel):
    """A multivariate PFN model that returns a joint posterior over q batch inputs.

    For this to work correctly it is necessary that the underlying model return a
    posterior for the latent f, not the noisy observed y.
    """

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> Union[BoundedRiemannPosterior, MultivariateRiemannPosterior]:
        """Computes the posterior over model outputs at the provided points.

        Will produce a MultivariateRiemannPosterior that fits a joint structure
        over the q batch dimension of X. This will require an additional forward
        pass through the PFN model, and some approximation.

        If q = 1 or there is no q dimension, will return a BoundedRiemannPosterior
        and behave the same as PFNModel.

        Args:
            X: A b? x q? x d`-dim Tensor, where `d` is the dimension of the
                feature space.
            output_indices: **Currently not supported for PFNModel.**
            observation_noise: **Currently not supported for PFNModel**.
            posterior_transform: **Currently not supported for PFNModel**.

        Returns:
            A posterior representing a batch of b? x q? distributions.
        """
        marginals = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        if len(X.shape) == 1 or X.shape[-2] == 1:
            # No q dimension, or q=1
            return marginals
        X, train_X, train_Y, orig_X_shape, styles = self._prepare_data(X)
        # Estimate correlation structure, making another forward pass.
        R = self.estimate_correlations(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            styles=styles,
            marginals=marginals,
        )  # (b, q, q)
        R = R.view(*orig_X_shape[:-2], X.shape[-2], X.shape[-2])  # (b?, q, q)
        return MultivariateRiemannPosterior(
            borders=self.borders,
            probabilities=marginals.probabilities,
            correlation_matrix=R,
        )

    def estimate_correlations(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        styles: dict[str, Tensor],
        marginals: BoundedRiemannPosterior,
    ) -> Tensor:
        """
        Estimate a correlation matrix R across the q batch of points in X.
        Will do a forward pass through the PFN model with batch size O(q^2).

        For every x_q in [x_1, ..., x_Q]:
           1. Add x_q to train_X, with y_q the 90th percentile value for f(x_q)
           2. Evaluate p(f(x_i)) for all points.

        Uses bivariate normal conditioning formulae, and so will be approximate.

        Args:
            X: evaluation point, shape (b, q, d)
            train_X: Training X, shape (b, n, d)
            train_Y: Training Y, shape (b, n, 1)
            styles: dict from name to tensor shaped (b, ns) for any styles.
            marginals: A posterior object with marginal posteriors for f(X), but no
                correlation structure yet added. posterior.probabilities has
                shape (b?, q, num_buckets).

        Returns: A (b, q, q) correlation matrix
        """
        # Compute conditional distributions with a forward pass
        cond_mean, cond_val = self._compute_conditional_means(
            X=X,
            train_X=train_X,
            train_Y=train_Y,
            styles=styles,
            marginals=marginals,
        )
        # Get marginal moments
        var = marginals.variance.squeeze(-1)  # (b?, q)
        mean = marginals.mean.squeeze(-1)  # (b?, q)
        if len(var.shape) == 1:
            var = var.unsqueeze(0)  # (b, q)
            mean = mean.unsqueeze(0)  # (b, q)
        # Estimate covariances from conditional distributions
        cov = self._estimate_covariances(
            cond_mean=cond_mean,
            cond_val=cond_val,
            mean=mean,
            var=var,
        )
        # Convert to correlation matrix
        S = 1 / torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))  # (b, q)
        S = S.unsqueeze(-1).expand(cov.shape)  # (b, q, q)
        R = S * cov * S.transpose(-1, -2)  # (b, q, q)
        return R

    def _compute_conditional_means(
        self,
        X: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        styles: dict[str, Tensor],
        marginals: BoundedRiemannPosterior,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute conditional means between pairs of points in X.

        Conditioning is done with an additional forward pass through the model. The
        returned conditional mean will be of shape (b, q, q), with entry [b, i, j] the
        conditional mean of j given i set to the conditioning value.

        Args:
            X: evaluation point, shape (b, q, d)
            train_X: Training X, shape (b, n, d)
            train_Y: Training Y, shape (b, n, 1)
            styles: dict from name to tensor shaped (b, ns) for any styles.
            marginals: A posterior object with marginal posteriors for f(X), but no
                correlation structure yet added. posterior.probabilities has
                shape (b?, q, num_buckets).

        Returns: conditional means (b, q, q), and values used for conditioning (b, q).
        """
        b, q, d = X.shape
        n = train_X.shape[-2]
        post_shape = marginals.probabilities.shape[:-1]
        # Find the 90th percentile of each eval point.
        cond_val = marginals.icdf(
            torch.full(post_shape, 0.9, device=X.device, dtype=X.dtype).unsqueeze(0)
        )  # (1, b?, q, 1)
        cond_val = cond_val.view(b, q)  # (b, q)
        # Construct conditional training data.
        # train_X will have shape (b, q, n+1, d), to have a conditional observation
        # for each point. train_Y will have shape (b, q, n+1, 1).
        train_X = train_X.unsqueeze(1).expand(b, q, n, d)
        cond_X = X.unsqueeze(-2)  # (b, q, 1, d)
        train_X = torch.cat((train_X, cond_X), dim=-2)  # (b, q, n+1, d)
        train_Y = train_Y.unsqueeze(1).expand(b, q, n, 1)
        cond_Y = cond_val.unsqueeze(-1).unsqueeze(-1)  # (b, q, 1, 1)
        train_Y = torch.cat((train_Y, cond_Y), dim=-2)  # (b, q, n+1, 1)
        cond_styles = {}
        for name, style in styles.items():
            ns = style.shape[-1]
            cond_styles[name] = style.unsqueeze(-2).expand(b, q, ns).reshape(b * q, ns)
        # Construct eval points
        eval_X = X.unsqueeze(1).expand(b, q, q, d)
        # Squeeze everything into necessary 2 batch dims, and do PFN forward pass
        cond_probabilities = self.pfn_predict(
            X=eval_X.reshape(b * q, q, d),
            train_X=train_X.reshape(b * q, n + 1, d),
            train_Y=train_Y.reshape(b * q, n + 1, 1),
            **cond_styles,
        )  # (b * q, q, num_buckets)
        # Object for conditional posteriors
        cond_posterior = BoundedRiemannPosterior(
            borders=self.borders,
            probabilities=cond_probabilities,
        )
        # Get conditional means
        cond_mean = cond_posterior.mean.squeeze(-1)  # (b * q, q)
        cond_mean = cond_mean.unsqueeze(0).view(b, q, q)
        return cond_mean, cond_val

    def _estimate_covariances(
        self,
        cond_mean: Tensor,
        cond_val: Tensor,
        mean: Tensor,
        var: Tensor,
    ) -> Tensor:
        """
        Estimate covariances from conditional distributions.

        Part one: Compute noise variance implied by conditional distributions
        E[f_j | y_j=y] = E[f_j] + var[f_j]/(var[f_j] + noise_var) * (y - E[f_j])
        Let Z_jj = (E[f_j | y_j=y] - E[f_j]) / (y - E[f_j]).
        Note that Z is in (0, 1].
        Then, noise_var_j = var[f_j](1/Z_jj - 1).

        Part two: Compute covariances for all pairs
        E[f_j|y_i=y] = E[f_j]+cov[f_j, f_i]/(var[f_i] + noise_var_i) * (y - E[f_i])
        Let Z_ij = (E[f_j | y_i=y] - E[f_j]) / (y - E[f_i]).
        Then, cov[f_j, f_i] = Z * (var[f_i] + noise_var)

        Args:
            cond_mean: (b, q, q) means of dim -1 conditioned on dim -2
            cond_val: (b, q) conditioned y value.
            var: (b, q) marginal variances
            mean: (b, q) marginal means

        Returns: Covariance matrix
        """
        Z = (cond_mean - mean.unsqueeze(-2).expand(cond_mean.shape)) / (
            cond_val - mean
        ).unsqueeze(-1)  # (b, q, q)
        # Z[i, j] is for j cond. on i
        noise_var = torch.clamp(
            var * (1 / torch.diagonal(Z, dim1=-2, dim2=-1) - 1), min=1e-8
        )  # (b, q)
        cov = Z * (var + noise_var).unsqueeze(-1)  # (b, q, q)
        # Symmetrize
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        cov = self._map_psd(cov)
        return cov

    def _map_psd(self, A):
        """
        Map A (assumed symmetric) to the nearest PSD matrix.
        """
        if torch.linalg.eigvals(A).real.min() < 0:
            L, Q = torch.linalg.eigh(A)
            L = torch.clamp(L, min=1e-6)
            A = Q @ torch.diag_embed(L) @ Q.transpose(-1, -2)
        return A
