#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [lin2025scalable]
    J. A. Lin, S. Ament, M. Balandat, D. Eriksson, J. M. Hernández-Lobato, E. Bakshy.
    Scalable Gaussian Processes with Latent Kronecker Structure.
    International Conference on Machine Learning 2025.

.. [lin2024scaling]
    J. A. Lin, S. Ament, M. Balandat, E. Bakshy. Scaling Gaussian Processes
    for Learning Curve Prediction via Latent Kronecker Structure. NeurIPS 2024
    Bayesian Decision-making and Uncertainty Workshop.

.. [lin2023sampling]
    J. A. Lin, J. Antorán, s. Padhy, D. Janz, J. M. Hernández-Lobato, A. Terenin.
    Sampling from Gaussian Process Posterior using Stochastic Gradient Descent.
    Advances in Neural Information Processing Systems 2023.
"""

import contextlib
import warnings
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import FantasizeMixin, Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import Mean, ZeroMean

from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from linear_operator import settings
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    KroneckerProductLinearOperator,
    MaskedLinearOperator,
)
from torch import Tensor


class LatentKroneckerGP(GPyTorchModel, ExactGP, FantasizeMixin):
    r"""
    A multi-task GP model which uses Kronecker structure despite missing entries.

    Leverages pathwise conditioning and iterative linear system solvers to
    efficiently draw samples from the GP posterior. See [lin2024scaling]_
    and [lin2025scalable]_ for details.

    For more information about pathwise conditioning, see [wilson2021pathwise]_
    and [Maddox2021bohdo]_. Details about iterative linear system solvers for GPs
    with pathwise conditioning can be found in [lin2023sampling]_.

    NOTE: This model requires iterative methods for efficient posterior inference.
    To enable iterative methods, the `use_iterative_methods` helper function can be
    used as a context manager.

    Example:
        >>> model = LatentKroneckerGP(train_X, train_T, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> with model.use_iterative_methods():
        >>>     fit_gpytorch_mll(mll)
        >>>     samples = model.posterior(test_X, test_T).rsample()
    """

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        covar_module_X: Module | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_T: A `batch_shape x t x 1` tensor of training time steps.
            train_Y: A `batch_shape x n x t` tensor of training observations,
                corresponding to the Cartesian product of `train_X` and `train_T`.
            likelihood: A likelihood. If omitted, use a standard
                `GaussianLikelihood` with inferred homoskedastic noise level.
            mean_module_X: The mean function to be used for X.
                If omitted, a `ZeroMean` will be used.
            mean_module_T: The mean function to be used for T.
                If omitted, a `ZeroMean` will be used.
            covar_module_X: The module computing the covariance matrix of X.
                If omitted, a `MaternKernel` will be used.
            covar_module_T: The module computing the covariance matrix of T.
                If omitted, a `MaternKernel` wrapped in a `ScaleKernel` will be used.
            input_transform: An input transform that is applied to X.
            outcome_transform: An outcome transform that is applied to Y.
                Note that `.train()` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            # transform inputs here to check resulting shapes
            # actual transforms will be applied in forward() and posterior()
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        batch_shape, ard_num_dims = transformed_X.shape[:-2], transformed_X.shape[-1]
        self._num_outputs = train_Y.shape[-1]

        expected_shape = torch.Size([*batch_shape, self._num_outputs, 1])
        train_T = torch.broadcast_to(train_T, (*batch_shape, *train_T.shape[-2:]))
        if train_T.shape != expected_shape:
            raise BotorchTensorDimensionError(
                f"Expected train_T with shape {expected_shape} but got {train_T.shape}."
            )
        self.train_T = train_T

        mask_valid_batch = train_Y.isfinite()
        # flatten over batch_shape
        mask_valid_flat = mask_valid_batch.reshape(-1, *mask_valid_batch.shape[-2:])
        # check that all masks are equal across batch_shape
        if not torch.all((mask_valid_flat == mask_valid_flat[0]).all(dim=(-2, -1))):
            raise ValueError(
                "Pattern of missing values in train_Y must be equal across batch_shape."
            )

        self.mask_valid = mask_valid_flat[0].flatten()
        train_Y = train_Y.reshape(*batch_shape, -1)[..., self.mask_valid]

        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(m=1, batch_shape=batch_shape)
        if outcome_transform is not None:
            outcome_transform.train()
            # transform outputs once and keep the results
            train_Y, _ = outcome_transform(train_Y.unsqueeze(-1), X=transformed_X)
            train_Y = train_Y.squeeze(-1)

        if likelihood is None:
            likelihood = GaussianLikelihood(batch_shape=batch_shape)

        ExactGP.__init__(
            self,
            train_inputs=train_X,
            train_targets=train_Y,
            likelihood=likelihood,
        )

        if mean_module_X is None:
            mean_module_X = ZeroMean(batch_shape=batch_shape)
        self.mean_module_X: Module = mean_module_X

        if mean_module_T is None:
            mean_module_T = ZeroMean(batch_shape=batch_shape)
        self.mean_module_T: Module = mean_module_T

        if covar_module_X is None:
            covar_module_X = MaternKernel(
                ard_num_dims=ard_num_dims, batch_shape=batch_shape
            )

        if covar_module_T is None:
            covar_module_T = ScaleKernel(
                base_kernel=MaternKernel(ard_num_dims=1, batch_shape=batch_shape),
            )

        self.covar_module_X: Module = covar_module_X
        self.covar_module_T: Module = covar_module_T

        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self.to(train_X)

    def use_iterative_methods(
        self,
        tol: float = 0.01,
        max_iter: int = 10000,
        covar_root_decomposition: bool = False,
        log_prob: bool = True,
        solves: bool = True,
    ):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                settings.fast_computations(
                    covar_root_decomposition=covar_root_decomposition,
                    log_prob=log_prob,
                    solves=solves,
                )
            )
            stack.enter_context(settings.cg_tolerance(tol))
            stack.enter_context(settings.max_cg_iterations(max_iter))
            return stack.pop_all()

    def _get_mean(self, X: Tensor, T: Tensor, mask: Tensor | None = None) -> Tensor:
        mean_X = self.mean_module_X(X).unsqueeze(-1)
        mean_T = self.mean_module_T(T).unsqueeze(-1)
        mean = KroneckerProductLinearOperator(mean_X, mean_T).squeeze(-1)
        return mean[..., mask] if mask is not None else mean

    def forward(self, X: Tensor, T: Tensor | None = None) -> MultivariateNormal:
        r"""
        Computes the joint distribution at the given input locations.

        Args:
            X: A tensor of `X`-locations at which to compute the joint distribution.
            T: A tensor of `T`-locations at which to compute the joint distribution.
                If None, defaults to using `self.train_T`.

        Returns:
            MultivariateNormal: The joint distribution at the specified input locations.
        """
        if T is None:
            T = self.train_T

        if self.training:
            X = self.transform_inputs(X)
            mask = self.mask_valid
        else:
            num_outputs = X.shape[-2] * T.shape[-2]
            mask = torch.ones(num_outputs, dtype=torch.bool, device=X.device)
            mask[: self.mask_valid.shape[-1]] = self.mask_valid

        mean = self._get_mean(X, T, mask=mask)

        covar_X = self.covar_module_X(X)
        covar_T = self.covar_module_T(T)
        covar = KroneckerProductLinearOperator(covar_X, covar_T)
        covar = MaskedLinearOperator(covar, row_mask=mask, col_mask=mask)

        return MultivariateNormal(mean, covar)

    def posterior(
        self,
        X: Tensor,
        T: Tensor | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            T: A `(batch_shape) x t x 1`-dim Tensor of `T`-locations at which to
                compute the posterior. If None, defaults to using `self.train_T`.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`). It is
                assumed to be in the outcome-transformed space if an outcome
                transform is used.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        if posterior_transform is not None:
            raise NotImplementedError(
                "Posterior transforms currently not supported for "
                f"{self.__class__.__name__}"
            )
        if not isinstance(self.likelihood, GaussianLikelihood):
            raise NotImplementedError(
                "Only GaussianLikelihood currently supported for "
                f"{self.__class__.__name__}"
            )
        if observation_noise is not False:
            raise NotImplementedError(
                "Observation noise currently not supported for "
                f"{self.__class__.__name__}"
            )

        if T is None:
            T = self.train_T
        return LatentKroneckerGPPosterior(self, X, T)

    def _rsample_from_base_samples(
        self,
        X: Tensor,
        T: Tensor,
        base_samples: Tensor,
        observation_noise: bool | Tensor = False,
    ) -> Tensor:
        r"""Sample from the posterior distribution at the provided points `X`
        using Matheron's rule, requiring `n + 2 n_train` base samples.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly
            T: A `(batch_shape) x t x 1`-dim Tensor of `T`-locations at which to
                evaluate the posterior samples.
            base_samples: A Tensor of `N(0, I)` base samples of shape
                `sample_shape x base_sample_shape`, typically obtained from
                a `Sampler`. This is used for deterministic optimization.
        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        # toggle eval mode to switch the behavior of input / outcome transforms
        # this also implicitly applies the input transform to the train_inputs
        self.eval()
        X_train = self.train_inputs[0]
        X_test = self.transform_inputs(X)
        n_train_full = X_train.shape[-2] * self._num_outputs
        n_train = self.train_targets.shape[-1]
        n_test = X_test.shape[-2] * T.shape[-2]

        sample_shape = base_samples.shape[: -len(self.batch_shape) - 1]
        w_train, eps_base, w_test = torch.split(
            base_samples, [n_train_full, n_train, n_test], dim=-1
        )
        eps = torch.sqrt(self.likelihood.noise) * eps_base

        # calculate prior sample evaluated at training data
        K_train_train_X = self.covar_module_X(X_train)
        K_train_train_T = self.covar_module_T(self.train_T)
        K_train_train = KroneckerProductLinearOperator(K_train_train_X, K_train_train_T)

        L_train_train_X = K_train_train_X.cholesky(upper=False)
        L_train_train_T = K_train_train_T.cholesky(upper=False)
        L_train_train = KroneckerProductLinearOperator(L_train_train_X, L_train_train_T)

        m_train = self._get_mean(X_train, self.train_T, mask=self.mask_valid)
        f_prior_train = L_train_train @ w_train.unsqueeze(-1)
        f_prior_train = m_train + f_prior_train.squeeze(-1)[..., self.mask_valid]

        # assemble and solve pathwise conditioning linear system
        K_train_train_valid = MaskedLinearOperator(
            K_train_train, row_mask=self.mask_valid, col_mask=self.mask_valid
        )
        noise_covar = ConstantDiagLinearOperator(
            self.likelihood.noise
            * torch.ones(*self.batch_shape, 1, dtype=X.dtype, device=X.device),
            diag_shape=n_train,
        )
        H = K_train_train_valid + noise_covar

        v = self.train_targets - (f_prior_train + eps)
        # expand once here to avoid repeated expansion
        # by MaskedLinearOperator later
        H_inv_v = torch.zeros(
            *sample_shape,
            *self.batch_shape,
            n_train_full,
            dtype=X.dtype,
            device=X.device,
        )
        with self.use_iterative_methods():
            H_inv_v[..., self.mask_valid] = H.solve(v.unsqueeze(-1)).squeeze(-1)

        # calculate prior sample evaluated at test data via conditional sampling
        K_test_test_X = self.covar_module_X(X_test).evaluate_kernel()
        K_test_test_T = self.covar_module_T(T).evaluate_kernel()
        K_train_test_X = self.covar_module_X(X_train, X_test).evaluate_kernel()
        K_train_test_T = self.covar_module_T(self.train_T, T).evaluate_kernel()

        L_train_test_X = L_train_train_X.solve_triangular(
            K_train_test_X.tensor, upper=False
        )
        L_train_test_T = L_train_train_T.solve_triangular(
            K_train_test_T.tensor, upper=False
        )

        L_test_test_X = (
            K_test_test_X - L_train_test_X.transpose(-2, -1) @ L_train_test_X
        ).cholesky(upper=False)
        L_test_test_T = (
            K_test_test_T - L_train_test_T.transpose(-2, -1) @ L_train_test_T
        ).cholesky(upper=False)

        L_test_train = KroneckerProductLinearOperator(
            L_train_test_X.transpose(-2, -1), L_train_test_T.transpose(-2, -1)
        )
        L_test_test = KroneckerProductLinearOperator(L_test_test_X, L_test_test_T)

        # match dimensions for broadcasting
        broadcast_shape = L_test_train.shape[:-2]
        extra_batch_dims = len(broadcast_shape) - len(self.batch_shape)
        for _ in range(extra_batch_dims):
            w_train = w_train.unsqueeze(len(sample_shape))
            w_test = w_test.unsqueeze(len(sample_shape))
            H_inv_v = H_inv_v.unsqueeze(len(sample_shape))

        m_test = self._get_mean(X_test, T)
        f_prior_test = L_test_train @ w_train.unsqueeze(-1)
        f_prior_test = f_prior_test + L_test_test @ w_test.unsqueeze(-1)
        f_prior_test = m_test + f_prior_test.squeeze(-1)

        K_train_test = KroneckerProductLinearOperator(K_train_test_X, K_train_test_T)
        # no MaskedLinearOperator here because H_inv_v is already expanded
        samples = K_train_test.transpose(-2, -1) @ H_inv_v.unsqueeze(-1)
        samples = samples + f_prior_test.unsqueeze(-1)
        # reshape samples to separate X and T dimensions
        # samples.shape = (*sample_shape, *broadcast_shape, n_test_x * n_t, 1)
        samples = samples.reshape(*samples.shape[:-2], X_test.shape[-2], T.shape[-2])
        # samples.shape = (*sample_shape, *broadcast_shape, n_test_x, n_t)
        if hasattr(self, "outcome_transform") and self.outcome_transform is not None:
            samples, _ = self.outcome_transform.untransform(samples, X=X)
        return samples

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ) -> Model:
        raise NotImplementedError(
            f"Conditioning currently not supported for {self.__class__.__name__}"
        )

    @classmethod
    def construct_inputs(cls, training_data: SupervisedDataset) -> dict[str, Any]:
        """
        Constructs the input tensors for LatentKroneckerGP from a SupervisedDataset.

        This method processes the provided training data to extract and organize the
        features and targets into the required format for the LatentKroneckerGP model.
        It factorizes inputs from the product space into the factors X and T.
        The matching output Y values are assembled by mapping observed values to their
        corresponding positions and filling missing values with NaN.

        Args:
            training_data: A SupervisedDataset containing training inputs and outputs.

        Returns:
            A dictionary with keys `train_X`, `train_T`, and `train_Y`, where:
                - `train_X`: The unique feature values (excluding the T dimension).
                - `train_T`: The unique feature values of the T dimension.
                - `train_Y`: The outputs aligned with the Cartesian product of
                    `train_X` and `train_T`, with missing values filled as NaN.
        """
        model_inputs = super().construct_inputs(training_data=training_data)

        if "train_Yvar" in model_inputs:
            warnings.warn(
                "Ignoring Yvar values in provided training data, because "
                "they are currently not supported by LatentKroneckerGP.",
                InputDataWarning,
                stacklevel=2,
            )

        t_idx = training_data.feature_names.index("step")
        x_idx = [i for i in range(len(training_data.feature_names)) if i != t_idx]

        # Factorize product space into factors X and T by finding unique values
        train_X, x_idx = model_inputs["train_X"][..., x_idx].unique(
            sorted=True, return_inverse=True, dim=-2
        )
        train_T, t_idx = model_inputs["train_X"][..., [t_idx]].unique(
            sorted=True, return_inverse=True, dim=-2
        )

        # Initialize train_Y with NaN for the full Cartesian product
        batch_shape = train_X.shape[:-2]
        n_x = train_X.shape[-2]
        n_t = train_T.shape[-2]
        train_Y = torch.full(
            (*batch_shape, n_x * n_t, 1),
            torch.nan,
            dtype=model_inputs["train_Y"].dtype,
            device=model_inputs["train_Y"].device,
        )

        # Convert 2D indices to 1D indices
        y_idx = x_idx * n_t + t_idx
        # Map original observations to their positions in the Cartesian product
        train_Y[..., y_idx, :] = model_inputs["train_Y"]
        train_Y = train_Y.reshape(*batch_shape, n_x, n_t)

        return {"train_X": train_X, "train_T": train_T, "train_Y": train_Y}
