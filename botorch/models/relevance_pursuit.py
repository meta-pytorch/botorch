# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Relevance Pursuit model structure and optimization routines for the sparse optimization
of Gaussian process hyper-parameters, see [Ament2024pursuit]_ for details.

References

.. [Ament2024pursuit]
    S. Ament, E. Santorella, D. Eriksson, B. Letham, M. Balandat, and E. Bakshy.
    Robust Gaussian Processes via Relevance Pursuit. Advances in Neural Information
    Processing Systems 37, 2024. Arxiv: https://arxiv.org/abs/2410.24222.
"""

from __future__ import annotations

import math

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from functools import partial
from typing import Any, cast, Optional
from warnings import warn

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor
from torch.nn.parameter import Parameter

MLL_ITER = 10_000  # let's take convergence seriously
MLL_TOL = 1e-8
RESET_PARAMETERS = True
RESET_DENSE_PARAMETERS = False


class RelevancePursuitMixin(ABC):
    """Mixin class to convert between the sparse and dense representations of the
    relevance pursuit modules' sparse parameters, as well as to compute the generalized
    support acquisition and support deletion criteria.
    """

    dim: int  # the total number of features
    _support: list[int]  # indices of the features in the support, subset of range(dim)

    def __init__(
        self,
        dim: int,
        support: list[int] | None,
    ) -> None:
        """Constructor for the RelevancePursuitMixin class.

        For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

        Args:
            dim: The total number of features.
            support: The indices of the features in the support, subset of range(dim).
        """

        self.dim = dim
        self._support = support if support is not None else []
        # Assumption: sparse_parameter is initialized in sparse representation
        self._is_sparse = True
        self._expansion_modifier = None
        self._contraction_modifier = None

    @property
    @abstractmethod
    def sparse_parameter(self) -> Parameter:
        """The sparse parameter, required to have a single indexing dimension."""
        pass  # pragma: no cover

    @abstractmethod
    def set_sparse_parameter(self, value: Parameter) -> None:
        """Sets the sparse parameter.

        NOTE: We can't use the property setter @sparse_parameter.setter because of
        the special way PyTorch treats Parameter types, including custom setters that
        bypass the @property setters before the latter are called.
        """
        pass  # pragma: no cover

    @staticmethod
    def _from_model(model: Model) -> RelevancePursuitMixin:
        """Retrieves a RelevancePursuitMixin from a model."""
        raise NotImplementedError  # pragma: no cover

    @property
    def is_sparse(self) -> bool:
        # Do we need to differentiate between a full support sparse representation and
        # a full support dense representation? The order the of the indices could be
        # different, unless we keep them sorted.
        return self._is_sparse

    @property
    def support(self) -> list[int]:
        """The indices of the active parameters."""
        return self._support

    @property
    def is_active(self) -> Tensor:
        """A Boolean Tensor of length `dim`, indicating which of the `dim` indices of
        `self.sparse_parameter` are in the support, i.e. active."""
        is_active = [(i in self.support) for i in range(self.dim)]
        return torch.tensor(
            is_active, dtype=torch.bool, device=self.sparse_parameter.device
        )

    @property
    def inactive_indices(self) -> Tensor:
        """An integral Tensor of length `dim - len(support)`, indicating which of the
        indices of `self.sparse_parameter` are not in the support, i.e. inactive."""
        device = self.sparse_parameter.device
        return torch.arange(self.dim, device=device)[~self.is_active]

    def to_sparse(self) -> RelevancePursuitMixin:
        """Converts the sparse parameter to its sparse (< dim) representation.

        Returns:
            The current object in its sparse representation.
        """
        if not self.is_sparse:
            self.set_sparse_parameter(
                torch.nn.Parameter(self.sparse_parameter[self.support])
            )
            self._is_sparse = True
        return self

    def to_dense(self) -> RelevancePursuitMixin:
        """Converts the sparse parameter to its dense, length-`dim` representation.

        Returns:
            The current object in its dense representation.
        """
        if self.is_sparse:
            dtype = self.sparse_parameter.dtype
            device = self.sparse_parameter.device
            zero = torch.tensor(
                0.0,
                dtype=dtype,
                device=device,
            )
            dense_parameter = [
                (
                    self.sparse_parameter[self.support.index(i)]
                    if i in self.support
                    else zero
                )
                for i in range(self.dim)
            ]
            dense_parameter = torch.tensor(dense_parameter, dtype=dtype, device=device)
            self.set_sparse_parameter(torch.nn.Parameter(dense_parameter))
            self._is_sparse = False
        return self

    def expand_support(self, indices: list[int]) -> RelevancePursuitMixin:
        """Expands the support by a number of indices.

        Args:
            indices: A list of indices of `self.sparse_parameter` to add to the support.

        Returns:
            The current object, updated with the expanded support.
        """
        for i in indices:
            if i in self.support:
                raise ValueError(f"Feature {i} already in the support.")

        self.support.extend(indices)
        # we need to add the parameter in the sparse representation
        if self.is_sparse:
            self.set_sparse_parameter(
                torch.nn.Parameter(
                    torch.cat(
                        (
                            self.sparse_parameter,
                            torch.zeros(len(indices)).to(self.sparse_parameter),
                        )
                    )
                )
            )
        return self

    def contract_support(self, indices: list[int]) -> RelevancePursuitMixin:
        """Contracts the support by a number of indices.

        Args:
            indices: A list of indices of `self.sparse_parameter` to remove from
                the support.

        Returns:
            The current object, updated with the contracted support.
        """
        # indices into the sparse representation of features to *keep*
        sparse_indices = list(range(len(self.support)))
        original_support = copy(self.support)
        for i in indices:
            if i not in self.support:
                raise ValueError(f"Feature {i} is not in support.")
            sparse_indices.remove(original_support.index(i))
            self.support.remove(i)

        # we need to remove the parameter in the sparse representation
        if self.is_sparse:
            self.set_sparse_parameter(Parameter(self.sparse_parameter[sparse_indices]))
        else:
            requires_grad = self.sparse_parameter.requires_grad
            self.sparse_parameter.requires_grad_(False)
            self.sparse_parameter[indices] = 0.0
            self.sparse_parameter.requires_grad_(requires_grad)  # restore
        return self

    # support initialization helpers
    def full_support(self) -> RelevancePursuitMixin:
        """Initializes the RelevancePursuitMixin with a full, size-`dim` support.

        Returns:
            The current object with full support in the dense representation.
        """
        self.expand_support([i for i in range(self.dim) if i not in self.support])
        self.to_dense()  # no reason to be sparse with full support
        return self

    def remove_support(self) -> RelevancePursuitMixin:
        """Initializes the RelevancePursuitMixin with an empty, size-zero support.

        Returns:
            The current object with empty support, representation unchanged.
        """
        self._support = []
        requires_grad = self.sparse_parameter.requires_grad
        if self.is_sparse:
            self.set_sparse_parameter(
                torch.nn.Parameter(torch.tensor([]).to(self.sparse_parameter))
            )
        else:
            self.sparse_parameter.requires_grad_(False)
            self.sparse_parameter[:] = 0.0
        self.sparse_parameter.requires_grad_(requires_grad)
        return self

    # the following two methods are the only ones that are specific to the marginal
    # likelihood optimization problem
    def support_expansion(
        self,
        mll: ExactMarginalLogLikelihood,
        n: int = 1,
        modifier: Callable[[Tensor], Tensor] | None = None,
    ) -> bool:
        """Computes the indices of the elements that maximize the gradient of the sparse
        parameter and that are not already in the support, and subsequently expands the
        support to include the elements if their gradient is positive.

        Args:
            mll: The marginal likelihood, containing the model to optimize.
                NOTE: Virtually all of the rest of the code is not specific to the
                marginal likelihood optimization, so we could generalize this to work
                with any objective.
            n: The maximum number of elements to select. NOTE: The actual number of
                elements that are added could be fewer if there are fewer than `n`
                elements with a positive gradient.
            modifier: A function that modifies the gradient of the inactive elements
                before computing the support expansion criterion. This can be used
                to select the maximum gradient magnitude for real-valued elements
                whose gradients are not non-negative, using modifier = torch.abs.

        Returns:
            True if the support was expanded, False otherwise.
        """
        # can't expand if the support is already full, or if n is non-positive
        if len(self.support) == self.dim or n <= 0:
            return False

        g = self.expansion_objective(mll)

        modifier = modifier if modifier is not None else self._expansion_modifier
        if modifier is not None:
            g = modifier(g)

        # support is already removed from consideration
        # gradient of the support parameters is not necessarily zero,
        # even for a converged solution in the presence of constraints.
        # NOTE: these indices are relative to self.inactive_indices.
        indices = g.argsort(descending=True)[:n]
        indices = indices[g[indices] > 0]
        if indices.numel() == 0:  # no indices with positive gradient
            return False
        self.expand_support(self.inactive_indices[indices].tolist())
        return True

    def expansion_objective(self, mll: ExactMarginalLogLikelihood) -> Tensor:
        """Computes an objective value for all the inactive parameters, i.e.
        self.sparse_parameter[~self.is_active] since we can't add already active
        parameters to the support. This value will be used to select the parameters.

        Args:
            mll: The marginal likelihood, containing the model to optimize.

        Returns:
            The expansion objective value for all the inactive parameters.
        """
        return self._sparse_parameter_gradient(mll)

    def _sparse_parameter_gradient(self, mll: ExactMarginalLogLikelihood) -> Tensor:
        """Computes the gradient of the marginal likelihood with respect to the
        sparse parameter.

        Args:
            mll: The marginal likelihood, containing the model to optimize.

        Returns:
            The gradient of the marginal likelihood with respect to the inactive
            sparse parameters.
        """
        # evaluate gradient of the sparse parameter
        is_sparse = self.is_sparse  # in order to restore the original representation
        self.to_dense()  # need the parameter in its dense parameterization

        requires_grad = self.sparse_parameter.requires_grad
        self.sparse_parameter.requires_grad_(True)
        if self.sparse_parameter.grad is not None:
            self.sparse_parameter.grad.zero_()
        mll.train()  # NOTE: this changes model.train_inputs
        model = mll.model
        X, Y = model.train_inputs[0], model.train_targets
        cast(
            Tensor,
            mll(
                mll.model(X),
                Y,
                *(model.transform_inputs(X=t_in) for t_in in model.train_inputs),
            ),
        ).backward()  # evaluation
        self.sparse_parameter.requires_grad_(requires_grad)

        g = self.sparse_parameter.grad
        if g is None:
            raise ValueError(
                "The gradient of the sparse_parameter is None, most likely "
                "because the passed marginal likelihood is not a function of the "
                "sparse_parameter."
            )

        if is_sparse:
            self.to_sparse()

        return g[~self.is_active]  # only need the inactive parameters

    def support_contraction(
        self,
        mll: ExactMarginalLogLikelihood,
        n: int = 1,
        modifier: Callable[[Tensor], Tensor] | None = None,
    ) -> bool:
        """Computes the indices of the elements with the smallest magnitude,
        and subsequently contracts the support by exluding the elements.

        Args:
            mll: The marginal likelihood, containing the model to optimize.
                NOTE: Virtually all of the rest of the code is not specific to the
                marginal likelihood optimization, so we could generalize this to work
                with any objective.
            n: The number of elements to select for removal.
            modifier: A function that modifies the parameter values before computing
                the support contraction criterion.

        Returns:
            True if the support was expanded, False otherwise.
        """
        # can't expand if the support is already empty, or if n is non-positive
        if len(self.support) == 0 or n <= 0:
            return False

        is_sparse = self.is_sparse
        self.to_sparse()
        x = self.sparse_parameter

        modifier = modifier if modifier is not None else self._contraction_modifier
        if modifier is not None:
            x = modifier(x)

        # for non-negative parameters, could break ties at zero
        # based on derivative
        sparse_indices = x.argsort(descending=False)[:n]
        indices = [self.support[i] for i in sparse_indices]
        self.contract_support(indices)
        if not is_sparse:
            self.to_dense()
        return True

    def optimize_mll(
        self,
        mll: ExactMarginalLogLikelihood,
        model_trace: list[Model] | None = None,
        reset_parameters: bool = RESET_PARAMETERS,
        reset_dense_parameters: bool = RESET_DENSE_PARAMETERS,
        # fit_gpytorch_mll kwargs
        closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
        optimizer: Callable | None = None,
        closure_kwargs: dict[str, Any] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        """Optimizes the marginal likelihood.

        Args:
            mll: The marginal likelihood, containing the model to optimize.
            model_trace: If not None, a list to which a deepcopy of the model state is
                appended. NOTE This operation is *in place*.
            reset_parameters: If True, initializes the sparse parameter to the all-zeros
                vector before every marginal likelihood optimization step. If False, the
                optimization is warm-started with the previous iteration's parameters.
            reset_dense_parameters: If True, re-initializes the dense parameters, e.g.
                other GP hyper-parameters that are *not* part of the Relevance Pursuit
                module, to the initial values provided by their associated constraints.
            closure: A closure to use to compute the loss and the gradients, see
                docstring of `fit_gpytorch_mll` for details.
            optimizer: The numerical optimizer, see docstring of `fit_gpytorch_mll`.
            closure_kwargs: Additional arguments to pass to the `closure` function.
            optimizer_kwargs: A dictionary of keyword arguments for the optimizer.

        Returns:
            The marginal likelihood after optimization.
        """
        if reset_parameters:
            # this might be beneficial because the parameters can
            # end up at a constraint boundary, which can anecdotally make
            # it more difficult to move the newly added parameters.
            with torch.no_grad():
                self.sparse_parameter.zero_()

        if reset_dense_parameters:
            initialize_dense_parameters(mll.model)

        # move to sparse representation for optimization
        # NOTE: this function should never force the dense representation, because some
        # models might never need it, and it would be inefficient.
        self.to_sparse()
        mll = fit_gpytorch_mll(
            mll,
            optimizer_kwargs=optimizer_kwargs,
            closure=closure,
            optimizer=optimizer,
            closure_kwargs=closure_kwargs,
        )
        if model_trace is not None:
            # need to record the full model here, rather than just the sparse parameter
            # since other hyper-parameters are co-adapted to the sparse parameter.
            model_trace.append(deepcopy(mll.model))
        return mll


# Optimization Algorithms
def forward_relevance_pursuit(
    sparse_module: RelevancePursuitMixin,
    mll: ExactMarginalLogLikelihood,
    sparsity_levels: list[int] | None = None,
    reset_parameters: bool = RESET_PARAMETERS,
    reset_dense_parameters: bool = RESET_DENSE_PARAMETERS,
    record_model_trace: bool = True,
    initial_support: list[int] | None = None,
    # fit_gpytorch_mll kwargs
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[RelevancePursuitMixin, Optional[list[Model]]]:
    """Forward Relevance Pursuit.

    NOTE: For the robust `SparseOutlierNoise` model of [Ament2024pursuit]_, the forward
    algorithm is generally faster than the backward algorithm, particularly when the
    maximum sparsity level is small, but it leads to less robust results when the number
    of outliers is large.

    For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

    Example:
        >>> base_noise = HomoskedasticNoise(
        >>>    noise_constraint=NonTransformedInterval(
        >>>        1e-5, 1e-1, initial_value=1e-3
        >>>    )
        >>> )
        >>> likelihood = SparseOutlierGaussianLikelihood(
        >>>    base_noise=base_noise,
        >>>    dim=X.shape[0],
        >>> )
        >>> model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> # NOTE: `likelihood.noise_covar` is the `RelevancePursuitMixin`
        >>> sparse_module = likelihood.noise_covar
        >>> sparse_module, model_trace = forward_relevance_pursuit(sparse_module, mll)

    Args:
        sparse_module: The relevance pursuit module.
        mll: The marginal likelihood, containing the model to optimize.
        sparsity_levels: The sparsity levels to expand the support to.
        reset_parameters: If true, initializes the sparse parameter to the all zeros
            after each iteration.
        reset_dense_parameters: If true, re-initializes the dense parameters, e.g.
            other GP hyper-parameters that are *not* part of the Relevance Pursuit
            module, to the initial values provided by their associated constraints.
        record_model_trace: If true, records the model state after every iteration.
        initial_support: The support with which to initialize the sparse module. By
            default, the support is initialized to the empty set.
        closure: A closure to use to compute the loss and the gradients, see docstring
            of `fit_gpytorch_mll` for details.
        optimizer: The numerical optimizer, see docstring of `fit_gpytorch_mll`.
        closure_kwargs: Additional arguments to pass to the `closure` function.
        optimizer_kwargs: A dictionary of keyword arguments to pass to the optimizer.
            By default, initializes the "options" sub-dictionary with `maxiter` and
            `ftol`, `gtol` values, unless specified.

    Returns:
        The relevance pursuit module after forward relevance pursuit optimization, and
        a list of models with different supports that were optimized.
    """
    sparse_module.remove_support()
    if initial_support is not None:
        sparse_module.expand_support(initial_support)

    if sparsity_levels is None:
        sparsity_levels = list(range(len(sparse_module.support), sparse_module.dim + 1))

    # since this is the forward algorithm, potential sparsity levels
    # must be in increasing order and unique.
    sparsity_levels = list(set(sparsity_levels))
    sparsity_levels.sort(reverse=False)

    optimizer_kwargs = _initialize_optimizer_kwargs(optimizer_kwargs)

    model_trace = [] if record_model_trace else None

    optimize_mll = partial(
        sparse_module.optimize_mll,
        model_trace=model_trace,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        # These are the args of the canonical mll fit routine
        closure=closure,
        optimizer=optimizer,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )

    # if sparsity levels contains the initial support, remove it
    if sparsity_levels[0] == len(sparse_module.support):
        sparsity_levels.pop(0)

    optimize_mll(mll)  # initial optimization

    for sparsity in sparsity_levels:
        support_size = len(sparse_module.support)
        num_expand = sparsity - support_size
        expanded = sparse_module.support_expansion(mll=mll, n=num_expand)
        if not expanded:  # stationary support
            warn(
                "Terminating optimization because the expansion from sparsity "
                f"{support_size} to {sparsity} was unsuccessful, usually due to "
                "reaching a stationary point of the marginal likelihood.",
                Warning,
                stacklevel=2,
            )
            break

        optimize_mll(mll)  # re-optimize support

    return sparse_module, model_trace


def backward_relevance_pursuit(
    sparse_module: RelevancePursuitMixin,
    mll: ExactMarginalLogLikelihood,
    sparsity_levels: list[int] | None = None,
    reset_parameters: bool = RESET_PARAMETERS,
    reset_dense_parameters: bool = RESET_DENSE_PARAMETERS,
    record_model_trace: bool = True,
    initial_support: list[int] | None = None,
    # fit_gpytorch_mll kwargs
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[RelevancePursuitMixin, Optional[list[Model]]]:
    """Backward Relevance Pursuit.

    NOTE: For the robust `SparseOutlierNoise` model of [Ament2024pursuit]_, the backward
    algorithm generally leads to more robust results than the forward algorithm,
    especially when the number of outliers is large, but is more expensive unless the
    support is contracted by more than one in each iteration.

    For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

    Example:
        >>> base_noise = HomoskedasticNoise(
        >>>    noise_constraint=NonTransformedInterval(
        >>>        1e-5, 1e-1, initial_value=1e-3
        >>>    )
        >>> )
        >>> likelihood = SparseOutlierGaussianLikelihood(
        >>>    base_noise=base_noise,
        >>>    dim=X.shape[0],
        >>> )
        >>> model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> # NOTE: `likelihood.noise_covar` is the `RelevancePursuitMixin`
        >>> sparse_module = likelihood.noise_covar
        >>> sparse_module, model_trace = backward_relevance_pursuit(sparse_module, mll)

    Args:
        sparse_module: The relevance pursuit module.
        mll: The marginal likelihood, containing the model to optimize.
        sparsity_levels: The sparsity levels to expand the support to.
        reset_parameters: If true, initializes the sparse parameter to the all zeros
            after each iteration.
        reset_dense_parameters: If true, re-initializes the dense parameters, e.g.
            other GP hyper-parameters that are *not* part of the Relevance Pursuit
            module, to the initial values provided by their associated constraints.
        record_model_trace: If true, records the model state after every iteration.
        initial_support: The support with which to initialize the sparse module. By
            default, the support is initialized to the full set.
        closure: A closure to use to compute the loss and the gradients, see docstring
            of `fit_gpytorch_mll` for details.
        optimizer: The numerical optimizer, see docstring of `fit_gpytorch_mll`.
        closure_kwargs: Additional arguments to pass to the `closure` function.
        optimizer_kwargs: A dictionary of keyword arguments to pass to the optimizer.
            By default, initializes the "options" sub-dictionary with `maxiter` and
            `ftol`, `gtol` values, unless specified.

    Returns:
        The relevance pursuit module after forward relevance pursuit optimization, and
        a list of models with different supports that were optimized.
    """
    if initial_support is not None:
        sparse_module.remove_support()
        sparse_module.expand_support(initial_support)
    else:
        sparse_module.full_support()

    if sparsity_levels is None:
        sparsity_levels = list(range(len(sparse_module.support) + 1))

    # since this is the backward algorithm, potential sparsity levels
    # must be in decreasing order, unique, and less than the initial support.
    sparsity_levels = list(set(sparsity_levels))
    sparsity_levels.sort(reverse=True)

    optimizer_kwargs = _initialize_optimizer_kwargs(optimizer_kwargs)

    model_trace = [] if record_model_trace else None

    def optimize_mll(mll):
        return sparse_module.optimize_mll(
            mll=mll,
            model_trace=model_trace,
            reset_parameters=reset_parameters,
            reset_dense_parameters=reset_dense_parameters,
            # These are the args of the canonical mll fit routine
            closure=closure,
            optimizer=optimizer,
            closure_kwargs=closure_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

    # if sparsity levels contains the initial support, remove it
    if sparsity_levels[0] == len(sparse_module.support):
        sparsity_levels.pop(0)

    optimize_mll(mll)  # initial optimization

    for sparsity in sparsity_levels:
        support_size = len(sparse_module.support)
        num_contract = support_size - sparsity
        contracted = sparse_module.support_contraction(mll=mll, n=num_contract)
        if not contracted:  # stationary support
            warn(
                "Terminating optimization because the contraction from sparsity "
                f"{support_size} to {sparsity} was unsuccessful.",
                Warning,
                stacklevel=2,
            )
            break

        optimize_mll(mll)  # re-optimize support

    return sparse_module, model_trace


# Bayesian Model Comparison
def get_posterior_over_support(
    rp_class: type[RelevancePursuitMixin],
    model_trace: list[Model],
    log_support_prior: Callable[[Tensor], Tensor] | None = None,
    prior_mean_of_support: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Computes the posterior distribution over a list of models.
    Assumes we are storing both likelihood and GP model in the model_trace.

    Example:
        >>> likelihood = SparseOutlierGaussianLikelihood(
        >>>    base_noise=base_noise,
        >>>    dim=X.shape[0],
        >>> )
        >>> model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> # NOTE: `likelihood.noise_covar` is the `RelevancePursuitMixin`
        >>> sparse_module = likelihood.noise_covar
        >>> sparse_module, model_trace = backward_relevance_pursuit(sparse_module, mll)
        >>> # NOTE: SparseOutlierNoise is the type of `sparse_module`
        >>> support_size, bmc_probabilities = get_posterior_over_support(
        >>>    SparseOutlierNoise, model_trace, prior_mean_of_support=2.0
        >>> )

    Args:
        rp_class: The relevance pursuit class to use for computing the support size.
            This is used to get the RelevancePursuitMixin from the Model via the static
            method `_from_model`. We could generalize this and let the user pass this
            getter instead.
        model_trace: A list of models with different support sizes, usually generated
            with relevance_pursuit.
        log_support_prior: Callable that computes the log prior probability of a
            support size. If None, uses a default exponential prior with a mean
            specified by `prior_mean_of_support`.
        prior_mean_of_support: A mean value for the default exponential prior
            distribution over the support size. Ignored if `log_support_prior`
            is passed.

    Returns:
        A tensor of posterior marginal likelihoods, one for each model in the trace.
    """
    if log_support_prior is None:
        if prior_mean_of_support is None:
            raise ValueError(
                "`log_support_prior` and `prior_mean_of_support` cannot both be None."
            )
        log_support_prior = partial(_exp_log_pdf, mean=prior_mean_of_support)

    log_support_prior = cast(Callable[[Tensor], Tensor], log_support_prior)

    def log_prior(
        model: Model,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        sparse_module = rp_class._from_model(model)
        num_support = torch.tensor(
            len(sparse_module.support), dtype=dtype, device=device
        )
        return num_support, log_support_prior(num_support)  # pyre-ignore[29]

    log_mll_trace = []
    log_prior_trace = []
    support_size_trace = []
    for model in model_trace:
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll.train()
        X, Y = mll.model.train_inputs[0], mll.model.train_targets
        F = mll.model(X)
        TX = mll.model.transform_inputs(X) if mll.model.training else X
        mll_i = cast(Tensor, mll(F, Y, TX))
        log_mll_trace.append(mll_i)
        support_size, log_prior_i = log_prior(
            model,
            dtype=mll_i.dtype,
            device=mll_i.device,
        )
        support_size_trace.append(support_size)
        log_prior_trace.append(log_prior_i)

    log_mll_trace = torch.stack(log_mll_trace)
    log_prior_trace = torch.stack(log_prior_trace)
    support_size_trace = torch.stack(support_size_trace)

    unnormalized_posterior_trace = log_mll_trace + log_prior_trace
    evidence = unnormalized_posterior_trace.logsumexp(dim=-1)
    posterior_probabilities = (unnormalized_posterior_trace - evidence).exp()
    return support_size_trace, posterior_probabilities


def _exp_log_pdf(x: Tensor, mean: Tensor) -> Tensor:
    """Compute the exponential log probability density.

    Args:
        x: A tensor of values.
        mean: A tensor of means.

    Returns:
        A tensor of log probabilities.
    """
    return -x / mean - math.log(mean)


def initialize_dense_parameters(model: Model) -> tuple[Model, dict[str, Any]]:
    """Sets the dense parameters of a model to their initial values. Infers initial
    values from the constraints, if no initial values are provided. If a parameter
    does not have a constraint, it is initialized to zero.

    Args:
        model: The model to initialize.

    Returns:
        The re-initialized model, and a dictionary of initial values.
    """
    constraints = dict(model.named_constraints())
    parameters = dict(model.named_parameters())
    initial_values = {
        n: getattr(constraints.get(n + "_constraint", None), "_initial_value", None)
        for n in parameters
    }
    lower_bounds = {
        n: getattr(
            constraints.get(n + "_constraint", None),
            "lower_bound",
            torch.tensor(-torch.inf),
        )
        for n in parameters
    }
    upper_bounds = {
        n: getattr(
            constraints.get(n + "_constraint", None),
            "upper_bound",
            torch.tensor(torch.inf),
        )
        for n in parameters
    }
    for name, value in initial_values.items():
        initial_values[name] = _get_initial_value(
            value=value,
            lower=lower_bounds[name],
            upper=upper_bounds[name],
        )

    # the initial values need to be converted to the transformed space
    for n, v in initial_values.items():
        c = constraints.get(n + "_constraint", None)
        # convert the constraint into the latent space
        if c is not None:
            initial_values[n] = c.inverse_transform(v)
    model.initialize(**initial_values)
    parameters = dict(model.named_parameters())
    return model, initial_values


def _get_initial_value(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    # if no initial value is provided, or the initial value is outside the bounds,
    # use a rule-based initialization.
    if value is None or not ((lower <= value) and (value <= upper)):
        if upper.isinf():
            value = 0.0 if lower.isinf() else lower + 1
        elif lower.isinf():  # implies u[n] is finite
            value = upper - 1
        else:  # both are finite
            # generally keep the value close to the lower bound in this case,
            # since many parameters (e.g. lengthscales) exhibit vanishing gradients
            # for large values.
            value = lower + torch.minimum(
                torch.ones_like(lower),
                (upper - lower) / 2,
            )
    return torch.as_tensor(value, dtype=lower.dtype, device=lower.device)


def _initialize_optimizer_kwargs(
    optimizer_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Initializes the optimizer kwargs with default values if they are not provided.

    Args:
        optimizer_kwargs: The optimizer kwargs to initialize.

    Returns:
        The initialized optimizer kwargs.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if optimizer_kwargs.get("options") is None:
        optimizer_kwargs["options"] = {}

    options = optimizer_kwargs["options"]
    if "maxiter" not in options:
        options.update({"maxiter": MLL_ITER})

    if ("ftol" not in options) and ("gtol" not in options):
        options.update({"ftol": MLL_TOL})
        options.update({"gtol": MLL_TOL})

    return optimizer_kwargs
