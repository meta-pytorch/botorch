# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from botorch.utils.constraints import LogTransformedInterval
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.constraints import Interval
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior, HalfCauchyPrior, NormalPrior
from torch import Tensor
from torch.distributions.half_cauchy import HalfCauchy
from torch.nn import Parameter


EPS = 1e-8


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: Tensor | float | None = None):
        """Instantiates a new helper object.

        Args:
            tau: Value of the global shrinkage parameter. If `None`, the tau will be
                a free parameter and inferred from the data.
                Tau can be a tensor for batched models, like
                `EnsembleMapSaasSingleTaskGP`, where each batch has a different
                sparsity prior. If tau is a tensor, it must have shape `batch_shape`.
        """
        self._tau = torch.as_tensor(tau) if tau is not None else None

    def tau(self, m: Kernel) -> Tensor:
        """The global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a lengthscale.

        Returns:
            The global shrinkage parameter of the SAAS prior.
        """
        return (
            self._tau.to(m.lengthscale)
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscale_prior_param_or_closure(self, m: Kernel) -> Tensor:
        """Closure to compute the scaled inverse lengthscale parameter (`tau / l^2`)
        to which the SAAS prior is applied.

        Args:
            m: A kernel object equipped with a lengthscale.

        Returns:
            The scaled inverse lengthscale parameter.
        """
        tau = self.tau(m)
        return tau.view(*tau.shape, 1, 1) / (m.lengthscale**2)

    def inv_lengthscale_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        """Closure to set the inverse lengthscale prior parameter.

        Args:
            m: A kernel object equipped with a lengthscale.
            value: The value of the scaled inverse lengthscale parameter, (`tau / l^2`),
                used to recover and set the lengthscale of the kernel.
        """
        # Lengthscale is batch x m x 1 x d, update tau to avoid unwanted broadcasting.
        tau = self.tau(m)
        tau = tau.view(*tau.shape, 1, 1)
        lb = m.raw_lengthscale_constraint.lower_bound.to(tau)
        ub = m.raw_lengthscale_constraint.upper_bound.to(tau)
        m._set_lengthscale((tau / value.to(tau)).sqrt().clamp(lb + EPS, ub - EPS))

    def tau_prior_param_or_closure(self, m: Kernel) -> Tensor:
        """Closure to compute the global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a `raw_tau` parameter.

        Returns:
            The transformed global shrinkage parameter `tau`.
        """
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        """Closure to set the global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a `raw_tau` parameter.
            value: The value of the global shrinkage parameter.
        """
        lb = m.raw_tau_constraint.lower_bound.to(m.raw_tau)
        ub = m.raw_tau_constraint.upper_bound.to(m.raw_tau)
        m.raw_tau.data = m.raw_tau_constraint.inverse_transform(
            value.to(m.raw_tau).clamp(lb + EPS, ub - EPS)
        )


def add_saas_prior(
    base_kernel: Kernel,
    tau: float | None = None,
    log_scale: bool = True,
) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscale^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior

    Args:
        base_kernel: Base kernel that has a lengthscale and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If `None`, infer the global
            shrinkage parameter.
        log_scale: Set to `True` if the lengthscale and tau should be optimized on
            a log-scale without any domain rescaling. That is, we will learn
            `raw_lengthscale := log(lengthscale)` and this hyperparameter needs to
            satisfy the corresponding bound constraints. Setting this to `True` will
            generally improve the numerical stability, but requires an optimizer that
            can handle bound constraints, e.g., L-BFGS-B.

    Returns:
        Base kernel with SAAS priors added.
    """
    if not base_kernel.has_lengthscale:
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if hasattr(base_kernel, "lengthscale_prior"):
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")
    tkwargs = {"device": base_kernel.device, "dtype": base_kernel.dtype}

    batch_shape = base_kernel.raw_lengthscale.shape[:-2]
    IntervalClass = LogTransformedInterval if log_scale else Interval
    base_kernel.register_constraint(
        param_name="raw_lengthscale",
        constraint=IntervalClass(0.01, 1e4, initial_value=1),
        replace=True,
    )
    prior_helper = SaasPriorHelper(tau=tau)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.register_parameter(
            name="raw_tau",
            parameter=Parameter(torch.full(batch_shape, 0.1, **tkwargs)),
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=IntervalClass(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )
    # Place a HC(1) prior on tau / lengthscale^2
    base_kernel.register_prior(
        name="inv_lengthscale_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscale_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscale_prior_setting_closure,
    )
    return base_kernel


def get_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    input_transform: InputTransform | None = None,
    outcome_transform: OutcomeTransform | None = None,
    tau: float | None = None,
) -> SingleTaskGP:
    """Helper method for creating an unfitted MAP SAAS model.

    Args:
        train_X: Tensor of shape `n x d` with training inputs.
        train_Y: Tensor of shape `n x 1` with training targets.
        train_Yvar: Optional tensor of shape `n x 1` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transforms.
        tau: Fixed value of the global shrinkage tau. If None, the model
            places a HC(0.1) prior on tau and infers it.

    Returns:
        A SingleTaskGP with a Matern kernel and a SAAS prior.
    """
    # TODO: Shape checks
    _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
        train_X=train_X, train_Y=train_Y
    )
    mean_module = get_mean_module_with_normal_prior(batch_shape=aug_batch_shape)
    if input_transform is not None:
        with torch.no_grad():
            transformed_X = input_transform(train_X)
        ard_num_dims = transformed_X.shape[-1]
    else:
        ard_num_dims = train_X.shape[-1]
    base_kernel = MaternKernel(
        nu=2.5, ard_num_dims=ard_num_dims, batch_shape=aug_batch_shape
    )
    # NOTE: need to call `to` to set device and dtype before calling `add_saas_prior`,
    # since the SAAS prior contains tensors that are not parameters of the model, and
    # therefore not automatically moved to the correct device with a `to` call on the
    # model.
    base_kernel.to(train_X)
    add_saas_prior(base_kernel=base_kernel, tau=tau)
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
        batch_shape=aug_batch_shape,
    )
    if train_Yvar is None:
        likelihood = get_gaussian_likelihood_with_gamma_prior(
            batch_shape=aug_batch_shape
        )
    else:
        likelihood = None
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )
    model.to(train_X)
    return model


def get_mean_module_with_normal_prior(
    batch_shape: torch.Size | None = None,
) -> ConstantMean:
    """Return constant mean with a N(0, 1) prior constrained to [-10, 10].

    This prior assumes the outputs (targets) have been standardized to have zero mean
    and unit variance.

    Args:
        batch_shape: Optional batch shape for the constant-mean module.

    Returns:
        ConstantMean module.
    """
    return ConstantMean(
        constant_prior=NormalPrior(loc=0.0, scale=1.0),
        constant_constraint=Interval(
            -10,
            10,
            initial_value=0,
            transform=None,
        ),
        batch_shape=batch_shape or torch.Size(),
    )


def get_gaussian_likelihood_with_gamma_prior(batch_shape: torch.Size | None = None):
    """Return Gaussian likelihood with a Gamma(0.9, 10) prior.

    This prior prefers small noise, but also has heavy tails.

    Args:
        batch_shape: Batch shape for the likelihood.

    Returns:
        GaussianLikelihood with Gamma(0.9, 10) prior constrained to [1e-4, 0.1].
    """
    return GaussianLikelihood(
        noise_prior=GammaPrior(0.9, 10.0),
        noise_constraint=LogTransformedInterval(1e-4, 1, initial_value=1e-2),
        batch_shape=batch_shape or torch.Size(),
    )


def get_additive_map_saas_covar_module(
    ard_num_dims: int,
    num_taus: int = 4,
    active_dims: tuple[int, ...] | None = None,
    batch_shape: torch.Size | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
):
    """Return an additive map SAAS covar module.

    The constructed kernel is an additive kernel with `num_taus` terms. Each term is a
    scaled Matern kernel with a SAAS prior and a tau sampled from a HalfCauchy(0, 1)
    distrbution.

    Args:
        ard_num_dims: The number of inputs dimensions.
        num_taus: The number of taus to use (4 if omitted).
        active_dims: Active dims for the covar module. The kernel will be evaluated
            only using these columns of the input tensor.
        batch_shape: Batch shape for the covar module.

    Returns:
        An additive MAP SAAS covar module.
    """
    batch_shape = batch_shape or torch.Size()
    kernels = []
    for _ in range(num_taus):
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
        ).to(dtype=dtype, device=device)
        add_saas_prior(base_kernel=base_kernel, tau=HalfCauchy(0.1).sample(batch_shape))
        scaled_kernel = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=batch_shape,
        )
        kernels.append(scaled_kernel)
    return AdditiveKernel(*kernels)


class AdditiveMapSaasSingleTaskGP(SingleTaskGP):
    """An additive MAP SAAS single-task GP.

    This is a maximum-a-posteriori (MAP) version of sparse axis-aligned subspace BO
    (SAASBO), see `SaasFullyBayesianSingleTaskGP` for more details. SAASBO is a
    high-dimensional Bayesian optimization approach that uses approximate fully
    Bayesian inference via NUTS to learn the model hyperparameters. This works very
    well, but is very computationally expensive which limits the use of SAASBO to a
    small (~100) number of trials. Two of the main benefits with SAASBO are:

    (1) A sparse prior on the inverse lengthscales that avoid overfitting.
    (2) The ability to sample several (~16) sets of hyperparameters from the
        posterior that we can average over when computing the acquisition
        function (ensembling).

    The goal of this Additive MAP SAAS model is to retain the main benefits of the SAAS
    model while significantly speeding up the time to fit the model. We achieve this by
    creating an additive kernel where each kernel in the sum is a Matern-5/2 kernel
    with a SAAS prior and a separate outputscale. The sparsity level for each kernel
    is sampled from an HC(0.1) distribution leading to a mix of sparsity levels (as is
    often the case for the fully Bayesian SAAS model). We learn all the hyperparameters
    using MAP inference which is significantly faster than using NUTS.

    While we often find that the original SAAS model with NUTS performs better, the
    additive MAP SAAS model can be several orders of magnitude faster to fit, which
    makes it applicable to problems with potentially thousands of trials.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        num_taus: int = 4,
    ) -> None:
        """Instantiates an AdditiveMapSaasSingleTaskGP.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform.
            input_transform: An optional input transform.
            num_taus: The number of taus to use (4 if omitted).
        """
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        mean_module = get_mean_module_with_normal_prior(
            batch_shape=self._aug_batch_shape
        )
        likelihood = (
            get_gaussian_likelihood_with_lognormal_prior(
                batch_shape=self._aug_batch_shape
            )
            if train_Yvar is None
            else None
        )
        if input_transform is not None:
            with torch.no_grad():
                transformed_X = input_transform(train_X)
            ard_num_dims = transformed_X.shape[-1]
        else:
            ard_num_dims = train_X.shape[-1]

        covar_module = get_additive_map_saas_covar_module(
            ard_num_dims=ard_num_dims,
            num_taus=num_taus,
            batch_shape=self._aug_batch_shape,
            # Need to pass dtype and device at initialization of the covar_module
            # because its priors contain tensors, and prior are currently not moved
            # to the correct device/dtype when callling `to` on the model.
            dtype=train_X.dtype,
            device=train_X.device,
        )

        SingleTaskGP.__init__(
            self=self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        # Make sure that all buffers and parameters have the correct device and dtype
        self.to(dtype=train_X.dtype, device=train_X.device)


class EnsembleMapSaasSingleTaskGP(SingleTaskGP):
    _is_ensemble = True

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        num_taus: int = 4,
        taus: Tensor | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Instantiates an ``EnsembleMapSaasSingleTaskGP``, which is a batched
        ensemble of ``SingleTaskGP``s with the Matern-5/2 kernel and a SAAS prior.
        The model is intended to be trained with ``ExactMarginalLogLikelihood`` and
        ``fit_gpytorch_mll``. Under the hood, the model is equivalent to a
        multi-output ``BatchedMultiOutputGPyTorchModel``, but it produces a
        ``MixtureGaussiaPosterior``, which leads to ensembling of the model outputs.

        Args:
            train_X: An `n x d` tensor of training features.
            train_Y: An `n x 1` tensor of training observations.
            train_Yvar: An optional `n x 1` tensor of observed measurement noise.
            num_taus: The number of taus to use (4 if omitted). Each tau is
                a sparsity parameter for the corresponding kernel in the ensemble.
            taus: An optional tensor of shape `num_taus` containing the taus to use.
                If omitted, the taus are sampled from a HalfCauchy(0.1) distribution.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform. Note that `.train()` will
                be called on the outcome transform during instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        if taus is None:
            taus = HalfCauchy(torch.tensor(0.1)).sample([num_taus]).to(train_X)
        elif taus.shape != torch.Size([num_taus]):
            raise ValueError(
                f"Expected taus to be of shape {[num_taus]}. Got {taus.shape=}."
            )
        if train_Y.shape[-1] != 1:
            raise UnsupportedError(
                f"EnsembleMapSAASGP only supports single-output. Got {train_Y.shape=}."
            )
        if train_X.ndim != 2:
            raise UnsupportedError(
                f"EnsembleMapSAASGP only supports 2D inputs. Got {train_X.ndim=}."
            )
        # Add batch dimension for ensemble.
        train_X = train_X.repeat(num_taus, 1, 1)
        train_Y = train_Y.repeat(num_taus, 1, 1)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.repeat(num_taus, 1, 1)
        # Construct the sub-modules.
        if input_transform is not None:
            with torch.no_grad():
                transformed_X = input_transform(train_X)
            ard_num_dims = transformed_X.shape[-1]
        else:
            ard_num_dims = train_X.shape[-1]
        batch_shape = train_X.shape[:-2]  # This is torch.Size([num_taus]).
        mean_module = get_mean_module_with_normal_prior(batch_shape=batch_shape)
        base_kernel = MaternKernel(
            nu=2.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape
        )
        # NOTE: need to call `to` to set device and dtype before calling
        # `add_saas_prior`, since the SAAS prior contains tensors that are not
        # parameters of the model, and therefore not automatically moved to the
        # correct device with a `to` call on the model.
        base_kernel.to(train_X)
        add_saas_prior(base_kernel=base_kernel, tau=taus)
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_constraint=LogTransformedInterval(1e-2, 1e4, initial_value=10),
            batch_shape=batch_shape,
        )
        if train_Yvar is None:
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=batch_shape
            )
        else:
            likelihood = None

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GaussianMixturePosterior` object. Includes observation noise
                if specified.
        """
        posterior = super().posterior(
            X=X.unsqueeze(MCMC_DIM),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        return GaussianMixturePosterior(distribution=posterior.distribution)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        *,
        num_taus: int = 4,
    ) -> dict[str, BotorchContainer | Tensor]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`.

        Args:
            training_data: A `SupervisedDataset` containing the training data.
            num_taus: Number of taus to use in the ensemble (4 if omitted).
        """
        base_inputs = super().construct_inputs(training_data=training_data)
        return {**base_inputs, "num_taus": num_taus}
