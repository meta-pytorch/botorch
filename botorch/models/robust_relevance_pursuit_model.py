# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file contains a readily usable implementation of the robust Gaussian process
model of [Ament2024pursuit]_, leveraging the Relevance Pursuit algorithm.

In particular, this file contains a `RobustRelevancePursuitMixin` class, and a concrete
implementation of a `SingleTaskGP` model, `RobustRelevancePursuitSingleTaskGP`, which
has the same API as a standard `SingleTaskGP` model, but automatically instantiates the
robust likelihood `SparseOutlierGaussianLikelihood` and dispatches the relevance pursuit
algorithm during model fitting via `fit_gpytorch_mll`.

Even though a standard `SingleTaskGP` model is expressive enough to implement the robust
model by changing the likelihood, its optimization is more complex. So the main reason
for the `RobustRelevancePursuitMixin` class is to hide this complexity by using multiple
dispatch of `fit_gpytorch_mll`, which needs to do two distinct operations in the context
of the robust model:

(1) It needs to toggle the relevance pursuit discrete optimization algorithm that
    changes the support, and as a sub-task,
(2) it needs to still carry out the numerical optimization of the hyper-parameters given
    a fixed support, but still with a `SparseOutlierGaussianLikelihood`. Since the types
    of the marginal likelihood (`MarginalLogLikelihood`) and the likelihood
    (`SparseOutlierGaussianLikelihood`) are the same in both calls, the only way we can
    leverage the multiple dispatch mechanism is the model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import FitGPyTorchMLL
from botorch.models import SingleTaskGP
from botorch.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch.models.model import Model
from botorch.models.relevance_pursuit import (
    backward_relevance_pursuit,
    get_posterior_over_support,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    Likelihood,
)
from gpytorch.means.mean import Mean
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.module import Module
from torch import Tensor

# default fractions of outliers to consider during relevance pursuit
FRACTIONS_OF_OUTLIERS = [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
]


class RobustRelevancePursuitMixin(ABC):
    """A Mixin class for robust relevance pursuit models, which wraps a base likelihood
    with a `SparseOutlierGaussianLikelihood` to detect outliers, and calls the
    relevance pursuit algorithm during model fitting via `fit_gpytorch_mll`.

    This is distinct from the `RelevancePursuitMixin` class, which is a Mixin class to
    equip a specific module (the likelihood, in the case of the robust model) with the
    relevance pursuit algorithms.
    """

    def __init__(
        self,
        base_likelihood: GaussianLikelihood | FixedNoiseGaussianLikelihood,
        dim: int,
        prior_mean_of_support: float | None = None,
        convex_parameterization: bool = True,
        cache_model_trace: bool = False,
    ) -> None:
        """Initializes a robust relevance pursuit model, which wraps a base likelihood
        with a `SparseOutlierGaussianLikelihood` to detect outliers, and calls the
        relevance pursuit algorithm during model fitting via `fit_gpytorch_mll`.

        For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

        Args:
            base_likelihood: The base likelihood that will be wrapped by a
                `SparseOutlierGaussianLikelihood` to detect outliers.
            dim: The number of training data points, i.e. the maximum dimensionality
                of the support set of the likelihood.
            prior_mean_of_support: The mean value for the default exponential prior
                distribution over the support size.
            convex_parameterization: If True, use a convex parameterization of the
                sparse noise model. See `SparseOutlierGaussianLikelihood` for details.
            cache_model_trace: If True, cache the model trace during relevance pursuit.
        """
        self.likelihood = SparseOutlierGaussianLikelihood(
            base_noise=base_likelihood.noise_covar,
            dim=dim,
            convex_parameterization=convex_parameterization,
        )
        self.bmc_support_sizes: Tensor | None = None
        self.bmc_probabilities: Tensor | None = None
        self.cache_model_trace = cache_model_trace
        self.model_trace: list[SingleTaskGP] | None = None
        self.prior_mean_of_support: float = (
            int(0.2 * dim) if prior_mean_of_support is None else prior_mean_of_support
        )

    @abstractmethod
    def to_standard_model(self) -> Model:
        """Converts this `RobustRelevancePursuitMixin` to an equivalent standard model
        with the same robust likelihood and hyper-parameters. This leaves the model
        structure and predictions unchanged, but leads `fit_gpytorch_mll`'s dispatch to
        *numerically* optimize the hyper-parameters of the model with a fixed support
        set, as opposed to dispatching to the discrete optimization via the relevance
        pursuit algorithm.

        Returns:
            A standard model.
        """

    def load_standard_model(self, standard_model: Model) -> RobustRelevancePursuitMixin:
        """Loads the state dict of a model into the `RobustRelevancePursuitMixin`.

        Args:
            standard_model: A standard model with the same parameter structure and
                likelihood as the `RobustRelevancePursuitMixin` model.

        Returns:
            The `RobustRelevancePursuitMixin` with the standard model's state dict.
        """
        # need special case for the likelihood because raw_rho's shape changes
        # throughout the optimization
        self.likelihood = standard_model.likelihood
        # overwrite state_dict in place
        self.load_state_dict(standard_model.state_dict())
        return self


class RobustRelevancePursuitSingleTaskGP(SingleTaskGP, RobustRelevancePursuitMixin):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        r"""A robust single-task GP model that toggles the relevance pursuit algorithm
            during model fitting via `fit_gpytorch_mll`.

        For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A base likelihood that will be wrapped by a
                `SparseOutlierGaussianLikelihood` to detect outliers. If omitted,
                use a standard `GaussianLikelihood` with inferred noise level if
                `train_Yvar` is None, and a `FixedNoiseGaussianLikelihood` with the
                given noise observations if `train_Yvar` is not None.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, uses an `RBFKernel`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                forward pass.
            convex_parameterization: If True, use a convex parameterization of the
                sparse noise model. See `SparseOutlierGaussianLikelihood` for details.
            prior_mean_of_support: The mean value for the default exponential prior
                distribution over the support size.
            cache_model_trace: If True, cache the model trace during relevance pursuit.

        Example:
            >>> m = RobustRelevancePursuitSingleTaskGP(train_X=X, train_Y=Y)
            >>> mll = ExactMarginalLogLikelihood(model=m, likelihood=m.likelihood)
            >>> mll = fit_gpytorch_mll(mll)
        """
        self._original_X = train_X
        self._original_Y = train_Y
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
        # After canonical GP is instantiated, modify the likelihood
        RobustRelevancePursuitMixin.__init__(
            self,
            base_likelihood=self.likelihood,
            dim=train_X.shape[-2],
            prior_mean_of_support=prior_mean_of_support,
            convex_parameterization=convex_parameterization,
            cache_model_trace=cache_model_trace,
        )

    def to_standard_model(self) -> Model:
        """Returns a standard SingleTaskGP with the same parameters as this model.
        This is used to avoid recursion through the fit_gpytorch_mll dispatch."""
        # don't need to put model into training mode to access the untransformed inputs,
        # since we cached the original train_inputs
        is_training = self.training
        model = SingleTaskGP(
            train_X=self._original_X,
            train_Y=self._original_Y,
            train_Yvar=None,  # not needed because likelihood is already instantiated
            likelihood=self.likelihood,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
            outcome_transform=getattr(self, "outcome_transform", None),
            input_transform=getattr(self, "input_transform", None),
        )
        if not is_training:
            model.eval()
        return model


@FitGPyTorchMLL.register(
    MarginalLogLikelihood,
    SparseOutlierGaussianLikelihood,
    RobustRelevancePursuitMixin,
)
def _fit_rrp(
    mll: MarginalLogLikelihood,
    _: type[SparseOutlierGaussianLikelihood],
    __: type[RobustRelevancePursuitMixin],
    *,
    numbers_of_outliers: list[int] | None = None,
    fractions_of_outliers: list[float] | None = None,
    timeout_sec: float | None = None,
    relevance_pursuit_optimizer: Callable = backward_relevance_pursuit,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    # fit_gpytorch_mll kwargs
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
) -> MarginalLogLikelihood:
    """Fits a RobustRelevancePursuitGP model using the given marginal likelihood.

    For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

    Args:
        mll: The marginal likelihood to fit.
        _: A likelihood, only directly used for dispatching.
        _: A model, only directly used for dispatching.
        numbers_of_outliers: An optional list of numbers of outliers to consider during
            relevance pursuit. By default, the algorithm falls back to a default list
            of fractions of outliers, see below.
        fractions_of_outliers: An optional list of fractions of outliers to consider if
            numbers_of_outliers is None. By default, the algorithm uses
            `[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]`.
        relevance_pursuit_optimizer: The relevance pursuit optimizer to use. By default,
            uses `backward_relevance_pursuit`, which is generally the most powerful
            algorithm for challenging problems with a wide range of outliers. The
            `forward_relevance_pursuit` algorithm can be efficient when the number of
            outliers is relatively small.
        reset_parameters: If True, we will reset the sparse parameters of the model
            after each iteration of the relevance pursuit algorithm.
        reset_dense_parameters: If True, we will reset the dense parameters of the model
            after each iteration of the relevance pursuit algorithm.
        closure: A closure to use to compute the loss and the gradients, see docstring
            of `fit_gpytorch_mll` for details.
        optimizer: The numerical optimizer, see docstring of `fit_gpytorch_mll`.
        closure_kwargs: Additional arguments to pass to the `closure` function.
        optimizer_kwargs: Additional arguments to pass to `fit_gpytorch_mll`.

    Returns:
        The fitted marginal likelihood.
    """
    sparse_module = SparseOutlierNoise._from_model(mll.model)
    n = sparse_module.dim  # equal to the number of training data points

    if numbers_of_outliers is None:
        if fractions_of_outliers is None:
            fractions_of_outliers = FRACTIONS_OF_OUTLIERS

        # list from which BMC chooses
        numbers_of_outliers = [int(p * n) for p in fractions_of_outliers]

    optimizer_kwargs_: dict[str, Any] = (
        {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    )
    if timeout_sec is not None:
        optimizer_kwargs_["timeout_sec"] = timeout_sec / len(numbers_of_outliers)

    # Need to convert model to avoid recursion through fit_gpytorch_mll dispatch, since
    # relevance pursuit expects to call the base fit_gpytorch_mll.
    original_model = mll.model  # Robust Relevance Pursuit Model
    mll.model = original_model.to_standard_model()
    sparse_module = SparseOutlierNoise._from_model(mll.model)
    sparse_module, model_trace = relevance_pursuit_optimizer(
        sparse_module=sparse_module,
        mll=mll,
        sparsity_levels=numbers_of_outliers,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        record_model_trace=True,
        # These are the args of the canonical mll fit routine
        closure=closure,
        optimizer=optimizer,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=optimizer_kwargs_,
    )

    # Bayesian model comparison
    bmc_support_sizes, bmc_probabilities = get_posterior_over_support(
        SparseOutlierNoise,
        model_trace,
        prior_mean_of_support=original_model.prior_mean_of_support,
    )
    map_index = torch.argmax(bmc_probabilities)
    map_model = model_trace[map_index]  # choosing model with highest BMC score
    # overwrite mll.model with chosen model
    mll.model = original_model  # first restore original model pointer
    mll.model.load_standard_model(map_model)
    # Store the bmc results
    mll.model.bmc_support_sizes = bmc_support_sizes
    mll.model.bmc_probabilities = bmc_probabilities
    if mll.model.cache_model_trace:
        mll.model.model_trace = model_trace
    return mll


@FitGPyTorchMLL.register(
    _ApproximateMarginalLogLikelihood,
    SparseOutlierGaussianLikelihood,
    RobustRelevancePursuitMixin,
)
def _fit_rrp_approximate_mll(
    mll: _ApproximateMarginalLogLikelihood,
    _: type[SparseOutlierGaussianLikelihood],
    __: type[RobustRelevancePursuitMixin],
    **kwargs: Any,
) -> None:
    raise UnsupportedError(
        "Relevance Pursuit does not yet support approximate inference. "
    )
