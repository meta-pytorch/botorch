#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [burt2020svgp]
    David R. Burt and Carl Edward Rasmussen and Mark van der Wilk,
    Convergence of Sparse Variational Inference in Gaussian Process Regression,
    Journal of Machine Learning Research, 2020,
    http://jmlr.org/papers/v21/19-1015.html.

.. [hensman2013svgp]
    James Hensman and Nicolo Fusi and Neil D. Lawrence, Gaussian Processes
    for Big Data, Proceedings of the 29th Conference on Uncertainty in
    Artificial Intelligence, 2013, https://arxiv.org/abs/1309.6835.

.. [moss2023ipa]
    Henry B. Moss and Sebastian W. Ober and Victor Picheny,
    Inducing Point Allocation for Sparse Gaussian Processes
    in High-Throughput Bayesian Optimization, Proceedings of
    the 25th International Conference on Artificial Intelligence
    and Statistics, 2023, https://arxiv.org/pdf/2301.10123.pdf.

"""

from __future__ import annotations

import copy
import warnings
from typing import Self

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.warnings import UserInputWarning
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction,
    InducingPointAllocator,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor
from torch.nn import Module


TRANSFORM_WARNING = (
    "Using an {ttype} transform with `SingleTaskVariationalGP`. If this "
    "model is trained in minibatches, a {ttype} transform with learnable "
    "parameters would update its parameters for each minibatch, which is "
    "undesirable. If you do intend to train in minibatches, we recommend "
    "you not use a {ttype} transform and instead pre-transform your whole "
    "data set before fitting the model."
)


class ApproximateGPyTorchModel(GPyTorchModel):
    r"""
    Botorch wrapper class for various (variational) approximate GP models in
    GPyTorch.

    This can either include stochastic variational GPs (SVGPs) or
    variational implementations of weight space approximate GPs.
    """

    def __init__(
        self,
        model: ApproximateGP | None = None,
        likelihood: Likelihood | None = None,
        num_outputs: int = 1,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Args:
            model: Instance of gpytorch.approximate GP models. If omitted,
                constructs a ``_SingleTaskVariationalGP``.
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a ``GaussianLikelihood`` (if ``num_outputs=1``) or a
                ``MultitaskGaussianLikelihood``(if ``num_outputs>1``).
            num_outputs: Number of outputs expected for the GP model.
            args: Optional positional arguments passed to the
                ``_SingleTaskVariationalGP`` constructor if no model is provided.
            kwargs: Optional keyword arguments passed to the
                ``_SingleTaskVariationalGP`` constructor if no model is provided.
        """
        super().__init__()

        self.model = (
            _SingleTaskVariationalGP(*args, num_outputs=num_outputs, **kwargs)
            if model is None
            else model
        )

        if likelihood is None:
            if num_outputs == 1:
                self.likelihood = GaussianLikelihood()
            else:
                self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self.likelihood = likelihood
        self._desired_num_outputs = num_outputs

    @property
    def num_outputs(self):
        return self._desired_num_outputs

    def eval(self) -> Self:
        r"""Puts the model in ``eval`` mode."""
        return Module.eval(self)

    def train(self, mode: bool = True) -> Self:
        r"""Put the model in ``train`` mode.

        Args:
            mode: A boolean denoting whether to put in ``train`` or ``eval`` mode.
                If ``False``, model is put in ``eval`` mode.
        """
        return Module.train(self, mode=mode)

    def posterior(
        self,
        X,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(  # pragma: no cover
                f"{self.__class__.__name__}.posterior does not support output indices."
            )
        self.eval()  # make sure model is in eval mode

        # input transforms are applied at ``posterior`` in ``eval`` mode, and at
        # ``model.forward()`` at the training time
        X = self.transform_inputs(X)

        # check for the multi-batch case for multi-outputs b/c this will throw
        # warnings
        X_ndim = X.ndim
        if self.num_outputs > 1 and X_ndim > 2:
            X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), self.num_outputs, 1, 1)
        dist = self.model(X)
        if observation_noise:
            dist = self.likelihood(dist)

        posterior = GPyTorchPosterior(distribution=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior=posterior, X=X)
        return posterior

    def forward(self, X) -> MultivariateNormal:
        if self.training:
            X = self.transform_inputs(X)
        return self.model(X)


class _SingleTaskVariationalGP(ApproximateGP):
    """
    Base class wrapper for a stochastic variational Gaussian Process (SVGP)
    model [hensman2013svgp]_.

    Uses by default pivoted Cholesky initialization for allocating inducing points,
    however, custom inducing point allocators can be provided.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor | None = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Not used.
            num_outputs: Number of output responses per input.
            covar_module: Kernel function. If omitted, uses an ``RBFKernel``.
            mean_module: Mean of GP model. If omitted, uses a ``ConstantMean``.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
            inducing_point_allocator: The ``InducingPointAllocator`` used to
                initialize the inducing point locations. If omitted,
                uses ``GreedyVarianceReduction``.
        """
        # We use the model subclass wrapper to deal with input / outcome transforms.
        # The number of outputs will be correct here due to the check in
        # SingleTaskVariationalGP.
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

        if covar_module is None:
            covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=train_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            ).to(train_X)

        if inducing_point_allocator is None:
            inducing_point_allocator = GreedyVarianceReduction()

        # initialize inducing points if they are not given
        if not isinstance(inducing_points, Tensor):
            if inducing_points is None:
                # number of inducing points is 25% the number of data points
                # as a heuristic
                inducing_points = int(0.25 * train_X.shape[-2])

            inducing_points = inducing_point_allocator.allocate_inducing_points(
                inputs=train_X,
                covar_module=covar_module,
                num_inducing=inducing_points,
                input_batch_shape=input_batch_shape,
            )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2],
                batch_shape=self._aug_batch_shape,
            )

        variational_strategy_instance = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            variational_strategy_instance = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy_instance,
                num_tasks=num_outputs,
                task_dim=-1,
            )
        super().__init__(variational_strategy=variational_strategy_instance)

        self.mean_module = (
            ConstantMean(batch_shape=self._aug_batch_shape).to(train_X)
            if mean_module is None
            else mean_module
        )

        self.covar_module = covar_module

    def forward(self, X) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        latent_dist = MultivariateNormal(mean_x, covar_x)
        return latent_dist


class SingleTaskVariationalGP(ApproximateGPyTorchModel):
    r"""A single-task variational GP model following [hensman2013svgp]_.

    By default, the inducing points are initialized though the
    ``GreedyVarianceReduction`` of [burt2020svgp]_, which is known to be
    effective for building globally accurate models. However, custom
    inducing point allocators designed for specific down-stream tasks can also be
    provided (see [moss2023ipa]_ for details), e.g. ``GreedyImprovementReduction``
    when the goal is to build a model suitable for standard BO.

    A single-task variational GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently. However, batches of multi-output models
    are not supported at this time, if you need to use those, please use a
    ModelListGP.

    Use this model if you have a lot of data or if your responses are non-Gaussian.

    To train this model, you should use gpytorch.mlls.VariationalELBO and not
    the exact marginal log likelihood.

    Example:
        >>> import torch
        >>> from botorch.models import SingleTaskVariationalGP
        >>> from gpytorch.mlls import VariationalELBO
        >>>
        >>> train_X = torch.rand(20, 2)
        >>> model = SingleTaskVariationalGP(train_X)
        >>> mll = VariationalELBO(
        >>>     model.likelihood, model.model, num_data=train_X.shape[-2]
        >>> )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor | None = None,
        likelihood: Likelihood | None = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: Training inputs (due to the ability of the SVGP to sub-sample
                this does not have to be all of the training inputs).
            train_Y: Training targets (optional).
            likelihood: Instance of a GPyTorch likelihood. If omitted, uses a
                either a ``GaussianLikelihood`` (if ``num_outputs=1``) or a
                ``MultitaskGaussianLikelihood``(if ``num_outputs>1``).
            num_outputs: Number of output responses per input (default: 1).
            learn_inducing_points: If True, the inducing point locations are learned
                jointly with the other model parameters.
            covar_module: Kernel function. If omitted, uses an ``RBFKernel``.
            mean_module: Mean of GP model. If omitted, uses a ``ConstantMean``.
            variational_distribution: Type of variational distribution to use
                (default: CholeskyVariationalDistribution), the properties of the
                variational distribution will encourage scalability or ease of
                optimization.
            variational_strategy: Type of variational strategy to use (default:
                VariationalStrategy). The default setting uses "whitening" of the
                variational distribution to make training easier.
            inducing_points: The number or specific locations of the inducing points.
            inducing_point_allocator: The ``InducingPointAllocator`` used to
                initialize the inducing point locations. If omitted,
                uses ``GreedyVarianceReduction``.
            outcome_transform: An outcome transform that is applied to the training
                data during instantiation and to the posterior during inference.
                NOTE: If this model is trained in minibatches, an outcome transform
                with learnable parameters (such as ``Standardize``) would update its
                parameters for each minibatch, which is undesirable. If you do intend
                to train in minibatches, we recommend you not use an outcome transform
                and instead pre-transform your whole data set before fitting the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
                NOTE: If this model is trained in minibatches, an input transform
                with learnable parameters (such as ``Normalize``) would update its
                parameters for each minibatch, which is undesirable. If you do intend
                to train in minibatches, we recommend you not use an input transform
                and instead pre-transform your whole data set before fitting the model.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                warnings.warn(
                    TRANSFORM_WARNING.format(ttype="outcome"),
                    UserInputWarning,
                    stacklevel=3,
                )
                outcome_transform.train()
                train_Y, _ = outcome_transform(train_Y, X=transformed_X)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(
                train_X=transformed_X,
                train_Y=train_Y,
                check_nans_only=covar_module is not None,
            )
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            if num_outputs == 1:
                likelihood = get_gaussian_likelihood_with_lognormal_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        if learn_inducing_points and (inducing_point_allocator is not None):
            warnings.warn(
                "After all the effort of specifying an inducing point allocator, "
                "you probably want to stop the inducing point locations "
                "being further optimized during the model fit. If so "
                "then set `learn_inducing_points` to False.",
                UserWarning,
                stacklevel=3,
            )

        if inducing_point_allocator is None:
            self._inducing_point_allocator = GreedyVarianceReduction()
        else:
            self._inducing_point_allocator = inducing_point_allocator

        model = _SingleTaskVariationalGP(
            train_X=transformed_X,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=self._inducing_point_allocator,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            warnings.warn(
                TRANSFORM_WARNING.format(ttype="input"),
                UserInputWarning,
                stacklevel=3,
            )
            self.input_transform = input_transform

        # for model fitting utilities
        # TODO: make this a flag?
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective. For a model with ``m``
        outputs, a ``test_batch_shape x q x d``-shaped input ``X`` to the ``posterior``
        method returns a Posterior object over an output of shape
        ``broadcast(test_batch_shape, model.batch_shape) x q x m``.
        """
        return self._input_batch_shape

    def init_inducing_points(
        self,
        inputs: Tensor,
    ) -> Tensor:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to ``inputs`` through the model's inducing point allocation strategy.
        The variational distribution and variational strategy caches are reset.

        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.

        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        var_strat = self.model.variational_strategy
        clear_cache_hook(var_strat)
        if hasattr(var_strat, "base_variational_strategy"):
            var_strat = var_strat.base_variational_strategy
            clear_cache_hook(var_strat)

        with torch.no_grad():
            num_inducing = var_strat.inducing_points.size(-2)
            inducing_points = self._inducing_point_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=self.model.covar_module,
                num_inducing=num_inducing,
                input_batch_shape=self._input_batch_shape,
            )
            var_strat.inducing_points.copy_(inducing_points)
            var_strat.variational_params_initialized.fill_(0)

        return inducing_points
