#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from contextlib import contextmanager
from typing import Iterator

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions import Distribution, MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from linear_operator.operators import LinearOperator
from torch import Tensor


class OrthogonalAdditiveGP(SingleTaskGP):
    """A Gaussian Process with Orthogonal Additive Kernel for interpretable modeling.

    This GP model uses an OrthogonalAdditiveKernel which decomposes the function into
    interpretable additive components: a bias term, first-order effects for each input
    dimension, and optionally second-order interaction terms.

    The model supports posterior inference of individual additive components when
    `infer_all_components=True` is passed to the `posterior` method.
    """

    # Class-level default for inference mode (avoids __init__ override)
    _infer_all_components: bool = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: OrthogonalAdditiveKernel | None = None,
        second_order: bool = False,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the OrthogonalAdditiveGP.

        Args:
            train_X: Training inputs (batch_shape x n x d) in [0, 1]^d.
            train_Y: Training outputs (batch_shape x n x 1).
            covar_module: An OrthogonalAdditiveKernel instance. If None, creates a
                default kernel with dim inferred from train_X.
            second_order: If True and covar_module is None, enables second-order
                interactions in the default kernel. Ignored if covar_module is provided.
            likelihood: Optional likelihood (defaults to GaussianLikelihood).
            mean_module: Optional mean module (defaults to ConstantMean).
            outcome_transform: Optional outcome transform.
            input_transform: Optional input transform.

        Raises:
            TypeError: If covar_module is provided but is not an
                OrthogonalAdditiveKernel.
        """
        if covar_module is None:
            covar_module = OrthogonalAdditiveKernel(
                base_kernel=RBFKernel(),
                dim=train_X.shape[-1],
                second_order=second_order,
                dtype=train_X.dtype,
                device=train_X.device,
            )
        elif not isinstance(covar_module, OrthogonalAdditiveKernel):
            raise TypeError(
                f"covar_module must be an OrthogonalAdditiveKernel, "
                f"got {type(covar_module).__name__}"
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    @contextmanager
    def _component_inference_context(
        self, infer_all_components: bool = False
    ) -> Iterator[None]:
        """Context manager that temporarily sets component inference mode.

        Args:
            infer_all_components: If True, enables per-component posterior inference.

        Yields:
            None. The context manager sets internal state that is checked by
            `_get_test_prior_mean_and_covariances` to dispatch to the appropriate
            covariance computation.
        """
        prev_state = self._infer_all_components
        self._infer_all_components = infer_all_components
        try:
            yield
        finally:
            self._infer_all_components = prev_state

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform=None,
        infer_all_components: bool = False,
    ) -> GPyTorchPosterior:
        """Posterior inference of the additive Gaussian process.

        Args:
            X: The input tensor of shape (batch_shape x n x d).
            output_indices: Not supported for this model.
            observation_noise: Whether to add observation noise to the posterior.
            posterior_transform: Optional posterior transform.
            infer_all_components: If True, returns a posterior with a batch
                dimension corresponding to each additive component (bias, first-order
                effects, and optionally second-order interactions). The number of
                components is 1 + d (first-order only) or 1 + d + d*(d-1)/2
                (with second-order interactions).

        Returns:
            The posterior distribution at X.
        """
        # Use context manager to set inference mode, then delegate to GPyTorch
        with self._component_inference_context(infer_all_components):
            return super().posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

    def _get_test_prior_mean_and_covariances(
        self,
        train_inputs: list[Tensor],
        test_inputs: list[Tensor],
        **kwargs,
    ) -> tuple[
        Tensor,
        LinearOperator,
        LinearOperator,
        torch.Size,
        torch.Size,
        type[Distribution],
    ]:
        """Dispatches to appropriate covariance computation based on inference mode.

        This method is called by GPyTorch's ExactGP.__call__ during posterior
        computation. When `_infer_all_components` is True (set via the context
        manager), it returns per-component covariances with an extra batch dimension.
        Otherwise, it uses the standard kernel forward pass which sums over components.
        """
        if self._infer_all_components:
            return self._get_test_prior_mean_and_covariances_per_component(
                train_inputs, test_inputs, **kwargs
            )
        return super()._get_test_prior_mean_and_covariances(
            train_inputs=train_inputs, test_inputs=test_inputs, **kwargs
        )

    def _get_test_prior_mean_and_covariances_per_component(
        self,
        train_inputs: list[Tensor],
        test_inputs: list[Tensor],
        **kwargs,
    ) -> tuple[
        Tensor,
        LinearOperator,
        LinearOperator,
        torch.Size,
        torch.Size,
        type[Distribution],
    ]:
        """Computes mean and covariances with a batch dimension for each component.

        This enables posterior inference of individual additive components by returning
        covariance matrices with an extra leading batch dimension for each component.

        Returns:
            A tuple containing:
            - test_mean: The mean evaluated on the test set (batch_shape x n_test)
            - test_test_covar: Covariance between test points
                (num_components x batch_shape x n_test x n_test)
            - test_train_covar: Covariance between test and train points
                (num_components x batch_shape x n_test x n_train)
            - batch_shape: The batch shape of the model
            - test_shape: Shape (n_test,)
            - posterior_class: The class of the posterior to be instantiated
        """
        if len(train_inputs) != 1 or len(test_inputs) != 1:
            raise ValueError(
                "OrthogonalAdditiveGP expects a single input X, but received "
                f"{len(train_inputs)=}, and {len(test_inputs)=}."
            )

        X_train = train_inputs[0]
        X_test = test_inputs[0]

        # Batch shape includes the component dimension as the leading dimension
        # This is needed so GPyTorch correctly reshapes the predictive mean
        num_components = self.covar_module.num_components
        batch_shape = torch.Size([num_components]) + X_train.shape[:-2]

        # Get component-wise covariances using _non_reduced_forward
        # Shape: (num_components x batch_shape x n_test x n_test)
        test_test_covar = self.covar_module._non_reduced_forward(X_test, X_test)
        # Shape: (num_components x batch_shape x n_test x n_train)
        test_train_covar = self.covar_module._non_reduced_forward(X_test, X_train)

        # Prior mean: Only the bias component (index 0) should have the prior mean.
        # All other components represent deviations from the mean, so their prior
        # mean should be zero. This ensures that when we sum over all components,
        # we get the correct total posterior mean (prior mean added once).
        n_test = X_test.shape[-2]
        # Create a (num_components, n_test) tensor of zeros
        test_mean = torch.zeros(
            num_components, n_test, dtype=X_test.dtype, device=X_test.device
        )
        # Set the bias component's mean to the actual prior mean
        test_mean[0, :] = self.mean_module(X_test)
        test_shape = torch.Size([n_test])

        return (
            test_mean,
            test_test_covar,
            test_train_covar,
            batch_shape,
            test_shape,
            MultivariateNormal,
        )

    @property
    def component_indices(self) -> dict[str, Tensor]:
        """Returns component indices from the OrthogonalAdditiveKernel."""
        return self.covar_module.component_indices

    def evaluate_first_order_on_grid(
        self,
        grid_1d: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        r"""Evaluate first-order component posteriors on 1D grids.

        Uses diagonal test inputs with the existing GPyTorch posterior
        infrastructure. Each first-order component is evaluated on its
        own independent 1D grid.

        Args:
            grid_1d: 1D tensor of m points in [0, 1].

        Returns:
            Tuple of:
            - bias: (mean, variance) - scalar values
            - first_order: ((d, m) means, (d, m) variances) on 1D grids

        Example:
            >>> grid = torch.linspace(0, 1, 50)
            >>> (bias_mean, bias_var), (fo_mean, fo_var) = \\
            ...     model.evaluate_first_order_on_grid(grid)
            >>> # fo_mean[i, :] is component i's posterior mean on the 1D grid
        """
        self.eval()
        m = len(grid_1d)
        d = self.covar_module.dim

        # Diagonal test inputs: X[k, :] = [t_k, t_k, ..., t_k]
        # Each first-order component i sees its own 1D grid on dimension i
        X_diag = grid_1d.unsqueeze(-1).expand(m, d)

        # Use existing posterior with all-components mode
        posterior = self.posterior(X_diag, infer_all_components=True)

        # Squeeze output dimension (last dim) since this is single-output
        mean = posterior.mean.squeeze(-1)  # (num_components, m)
        variance = posterior.variance.squeeze(-1)  # (num_components, m)

        # Extract bias (component 0) - should be constant across grid
        bias_mean = mean[0, :].mean()
        bias_var = variance[0, :].mean()

        # Extract first-order (components 1 to d)
        first_order_means = mean[1 : d + 1, :]  # (d, m)
        first_order_vars = variance[1 : d + 1, :]  # (d, m)

        return (bias_mean, bias_var), (first_order_means, first_order_vars)

    @property
    def num_components(self) -> int:
        """Total number of additive components (bias + first-order [+ second-order])."""
        return self.covar_module.num_components

    def get_component_index(
        self,
        component_type: str,
        dim_index: int | tuple[int, int] | None = None,
    ) -> int:
        """Returns the component index for a given component type and dimension.

        See OrthogonalAdditiveKernel.get_component_index for details.
        """
        return self.covar_module.get_component_index(component_type, dim_index)
