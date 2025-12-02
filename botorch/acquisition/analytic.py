#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.constants import get_constants_like
from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    compute_log_prob_feas_from_bounds,
    log_ndtr as log_Phi,
    log_phi,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp, logmeanexp
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor
from torch.nn.functional import pad

# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2


class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    """Base class for analytic acquisition functions."""

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
        allow_multi_output: bool = False,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
              allow_multi_output: If False, requires a posterior_transform if a
                multi-output model is passed.
        """
        super().__init__(model=model)
        if posterior_transform is None:
            if not allow_multi_output and model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> tuple[Tensor, Tensor | None]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors of shape `(b1 x ... x bk) x m` containing the first and
            second moments of the model posterior, where `m` is the number of outputs.
            Returns `None` instead of the second tensor if `compute_sigma` is False.
        """
        self.to(X)  # ensures buffers / parameters are on the same device and dtype
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2)  # remove q-batch dimension
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma


class LogProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Log Probability of Improvement.

    Logarithm of the probability of improvement over the current best observed value,
    computed using the analytic formula under a Normal posterior distribution. Only
    supports the case of q=1. Requires the posterior to be Gaussian. The model must be
    single-outcome.

    The logarithm of the probability of improvement is numerically better behaved
    than the original function, which can lead to significantly improved optimization
    of the acquisition function. This is analogous to the common practice of optimizing
    the *log* likelihood of a probabilistic model - rather the likelihood - for the
    sake of maximium likelihood estimation.

    `logPI(x) = log(P(y >= best_f)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> LogPI = LogProbabilityOfImprovement(model, best_f=0.2)
        >>> log_pi = LogPI(test_X)
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Log Probability of Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Log Probability of Improvement values at
            the given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)  # `(b1 x ... bk) x 1`
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return log_Phi(u.squeeze(-1))


class ProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Probability of Improvement.

    Probability of improvement over the current best observed value, computed
    using the analytic formula under a Normal posterior distribution. Only
    supports the case of q=1. Requires the posterior to be Gaussian. The model
    must be single-outcome.

    `PI(x) = P(y >= best_f), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PI = ProbabilityOfImprovement(model, best_f=0.2)
        >>> pi = PI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Probability of Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)  # `(b1 x ... bk) x 1`
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return Phi(u.squeeze(-1))


class qAnalyticProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Approximate, single-outcome batch Probability of Improvement using MVNXPB.

    This implementation uses MVNXPB, a bivariate conditioning algorithm for
    approximating P(a <= Y <= b) for multivariate normal Y.
    See [Trinh2015bivariate]_. This (analytic) approximate q-PI is given by
    `approx-qPI(X) = P(max Y >= best_f) = 1 - P(Y < best_f), Y ~ f(X),
    X = (x_1,...,x_q)`, where `P(Y < best_f)` is estimated using MVNXPB.
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        """qPI using an analytic approximation.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate approximate qPI on the candidate set X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each

        Returns:
            A `batch_shape`-dim Tensor of approximate Probability of Improvement values
            at the given design points `X`, where `batch_shape'` is the broadcasted
            batch shape of model and input `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )

        covariance = posterior.distribution.covariance_matrix
        bounds = pad(
            (self.best_f.unsqueeze(-1) - posterior.distribution.mean).unsqueeze(-1),
            pad=(1, 0) if self.maximize else (0, 1),
            value=-float("inf") if self.maximize else float("inf"),
        )
        # 1 - P(no improvement over best_f)
        solver = MVNXPB(covariance_matrix=covariance, bounds=bounds)
        return -solver.solve().expm1()


class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(f(x) - best_f, 0)),`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)

    NOTE: It is strongly recommended to use LogExpectedImprovement instead of regular
    EI, as it can lead to substantially improved BO performance through improved
    numerics. See https://arxiv.org/abs/2310.20708 for details.
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)  # `(b1 x ... bk) x 1`
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return (sigma * _ei_helper(u)).squeeze(-1)


class LogExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Log Expected Improvement (analytic).

    Computes the logarithm of the classic Expected Improvement acquisition function, in
    a numerically robust manner. In particular, the implementation takes special care
    to avoid numerical issues in the computation of the acquisition value and its
    gradient in regions where improvement is predicted to be virtually impossible.

    See [Ament2023logei]_ for details. Formally,

    `LogEI(x) = log(E(max(f(x) - best_f, 0))),`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> LogEI = LogExpectedImprovement(model, best_f=0.2)
        >>> ei = LogEI(test_X)
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        r"""Logarithm of single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate logarithm of Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of the logarithm of the Expected Improvement
            values at the given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)  # `(b1 x ... bk) x 1`
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return (_log_ei_helper(u) + sigma.log()).squeeze(-1)


class ConstrainedAnalyticAcquisitionFunctionMixin(ABC):
    r"""Base class for constrained analytic acquisition functions."""

    def __init__(
        self,
        constraints: dict[int, tuple[float | None, float | None]],
    ) -> None:
        r"""Analytic Log Probability of Feasibility.

        Args:
            model: A fitted single- or multi-output model.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None).
        """
        self.constraints = constraints
        self._preprocess_constraint_bounds(constraints=constraints)

    @abstractmethod
    def register_buffer(self, name: str, value: Tensor) -> None:
        """Add a buffer that can be accessed by `self.name` and stores the Tensor
        `value`, usually provided by derivatives that also inherit from `nn.Module`.
        """

    def _preprocess_constraint_bounds(
        self,
        constraints: dict[int, tuple[float | None, float | None]],
    ) -> None:
        r"""Set up constraint bounds.

        Args:
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        con_lower, con_lower_inds = [], []
        con_upper, con_upper_inds = [], []
        con_both, con_both_inds = [], []
        con_indices = list(constraints.keys())
        if len(con_indices) == 0:
            raise ValueError("There must be at least one constraint.")
        # CEI, LogCEI have an objective index, but LogPOF does not.
        if hasattr(self, "objective_index") and self.objective_index in con_indices:
            raise ValueError(
                "Output corresponding to objective should not be a constraint."
            )
        for k in con_indices:
            if constraints[k][0] is not None and constraints[k][1] is not None:
                if constraints[k][1] <= constraints[k][0]:
                    raise ValueError("Upper bound is less than the lower bound.")
                con_both_inds.append(k)
                con_both.append([constraints[k][0], constraints[k][1]])
            elif constraints[k][0] is not None:
                con_lower_inds.append(k)
                con_lower.append(constraints[k][0])
            elif constraints[k][1] is not None:
                con_upper_inds.append(k)
                con_upper.append(constraints[k][1])

        for name, value in [
            ("con_lower_inds", con_lower_inds),
            ("con_upper_inds", con_upper_inds),
            ("con_both_inds", con_both_inds),
            ("con_both", con_both),
            ("con_lower", con_lower),
            ("con_upper", con_upper),
        ]:
            # tensor-based indexing is much faster than list-based advanced indexing
            self.register_buffer(name, tensor=torch.as_tensor(value))

    def _compute_log_prob_feas(
        self,
        means: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        r"""Compute logarithm of the feasibility probability for each batch of X.

        Args:
            means: A `(b) x m`-dim Tensor of means.
            sigmas: A `(b) x m`-dim Tensor of standard deviations.

        Returns:
            A `(b)`-dim tensor of log feasibility probabilities

        Note: This function does case-work for upper bound, lower bound, and both-sided
        bounds. Another way to do it would be to use 'inf' and -'inf' for the
        one-sided bounds and use the logic for the both-sided case. But this
        causes an issue with autograd since we get 0 * inf.
        """
        return compute_log_prob_feas_from_bounds(
            con_lower_inds=self.con_lower_inds,
            con_upper_inds=self.con_upper_inds,
            con_both_inds=self.con_both_inds,
            con_lower=self.con_lower,
            con_upper=self.con_upper,
            con_both=self.con_both,
            means=means,
            sigmas=sigmas,
        )


class LogConstrainedExpectedImprovement(
    AnalyticAcquisitionFunction, ConstrainedAnalyticAcquisitionFunctionMixin
):
    r"""Log Constrained Expected Improvement (feasibility-weighted).

    Computes the logarithm of the analytic expected improvement for a Normal posterior
    distribution weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports non-batch mode (i.e. `q=1`). The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    See [Ament2023logei]_ for details. Formally,

    `LogConstrainedEI(x) = log(EI(x)) + Sum_i log(P(y_i \in [lower_i, upper_i]))`,

    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        # example where the 0th output has a non-negativity constraint and
        # the 1st output is the objective
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> LogCEI = LogConstrainedExpectedImprovement(model, 0.2, 1, constraints)
        >>> cei = LogCEI(test_X)
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        objective_index: int,
        constraints: dict[int, tuple[float | None, float | None]],
        maximize: bool = True,
    ) -> None:
        r"""Analytic Log Constrained Expected Improvement.

        Args:
            model: A fitted single- or multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, allow_multi_output=True)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.register_buffer("best_f", torch.as_tensor(best_f))
        ConstrainedAnalyticAcquisitionFunctionMixin.__init__(self, constraints)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Log Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Log Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        log_ei = _log_ei_helper(u) + sigma_obj.log()
        log_prob_feas = self._compute_log_prob_feas(means=means, sigmas=sigmas)
        return log_ei + log_prob_feas


class LogProbabilityOfFeasibility(
    AnalyticAcquisitionFunction, ConstrainedAnalyticAcquisitionFunctionMixin
):
    r"""Log Probability of Feasbility.

    Computes the logarithm of the analytic probability of feasibility for a Normal
    posterior distribution weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports non-batch mode (i.e. `q=1`). The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    See [Ament2023logei]_ for details. Formally,

    `LogPF(x) = Sum_i log(P(y_i \in [lower_i, upper_i]))`,

    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        # example where the 0th output has a non-negativity constraint
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> LogPOF = LogProbabilityOfFeasibility(model, constraints)
        >>> cei = LogPF(test_X)
    """

    _log: bool = True

    def __init__(
        self, model: Model, constraints: dict[int, tuple[float | None, float | None]]
    ) -> None:
        r"""Analytic Log Probability of Feasibility.

        Args:
            model: A fitted single- or multi-output model.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
        """
        super().__init__(model=model, allow_multi_output=True)
        self.posterior_transform = None
        ConstrainedAnalyticAcquisitionFunctionMixin.__init__(self, constraints)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Log Probability of Feasibility on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Log Probability of Feasibility values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        return self._compute_log_prob_feas(means=means, sigmas=sigmas)


class ConstrainedExpectedImprovement(
    AnalyticAcquisitionFunction, ConstrainedAnalyticAcquisitionFunctionMixin
):
    r"""Constrained Expected Improvement (feasibility-weighted).

    Computes the analytic expected improvement for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports non-batch mode (i.e. `q=1`). The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        # example where the 0th output has a non-negativity constraint and
        # 1st output is the objective
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> cEI = ConstrainedExpectedImprovement(model, 0.2, 1, constraints)
        >>> cei = cEI(test_X)

    NOTE: It is strongly recommended to use LogConstrainedExpectedImprovement instead
    of regular CEI, as it can lead to substantially improved BO performance through
    improved numerics. See https://arxiv.org/abs/2310.20708 for details.
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        objective_index: int,
        constraints: dict[int, tuple[float | None, float | None]],
        maximize: bool = True,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted single- or multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model=model, allow_multi_output=True)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.register_buffer("best_f", torch.as_tensor(best_f))
        ConstrainedAnalyticAcquisitionFunctionMixin.__init__(self, constraints)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        ei = sigma_obj * _ei_helper(u)
        log_prob_feas = self._compute_log_prob_feas(means=means, sigmas=sigmas)
        return ei.mul(log_prob_feas.exp())


class LogNoisyExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Log Noisy Expected Improvement (via fantasies).

    This computes Log Noisy Expected Improvement by averaging over the Expected
    Improvement values of a number of fantasy models. Only supports the case
    `q=1`. Assumes that the posterior distribution of the model is Gaussian.
    The model must be single-outcome.

    See [Ament2023logei]_ for details. Formally,

    `LogNEI(x) = log(E(max(y - max Y_base), 0))), (y, Y_base) ~ f((x, X_base))`,

    where `X_base` are previously observed points.

    Note: This acquisition function currently relies on using a SingleTaskGP
    with known observation noise. In other words, `train_Yvar` must be passed
    to the model. (required for noiseless fantasies).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> LogNEI = LogNoisyExpectedImprovement(model, train_X)
        >>> nei = LogNEI(test_X)
    """

    _log: bool = True

    def __init__(
        self,
        model: GPyTorchModel,
        X_observed: Tensor,
        num_fantasies: int = 20,
        maximize: bool = True,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Single-outcome Noisy Log Expected Improvement (via fantasies).

        Args:
            model: A fitted single-outcome model. Only `SingleTaskGP` models with
                known observation noise are currently supported.
            X_observed: A `n x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
        """
        _check_noisy_ei_model(model=model)
        # Sample fantasies.
        from botorch.sampling.normal import SobolQMCNormalSampler

        # Drop gradients from model.posterior if X_observed does not require gradients
        # as otherwise, gradients of the GP's kernel's hyper-parameters are tracked
        # through the rsample_from_base_sample method of GPyTorchPosterior. These
        # gradients are usually only required w.r.t. the marginal likelihood.
        with nullcontext() if X_observed.requires_grad else torch.no_grad():
            posterior = model.posterior(X=X_observed)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        Y_fantasized = sampler(posterior).squeeze(-1)
        batch_X_observed = X_observed.expand(num_fantasies, *X_observed.shape)
        # The fantasy model will operate in batch mode
        fantasy_model = _get_noiseless_fantasy_model(
            model=model, batch_X_observed=batch_X_observed, Y_fantasized=Y_fantasized
        )
        super().__init__(model=fantasy_model, posterior_transform=posterior_transform)
        best_f, _ = Y_fantasized.max(dim=-1) if maximize else Y_fantasized.min(dim=-1)
        self.best_f, self.maximize = best_f, maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate logarithm of the mean Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `b1 x ... bk`-dim tensor of Log Noisy Expected Improvement values at
            the given design points `X`.
        """
        # add batch dimension for broadcasting to fantasy models
        # (b1 x ... x bk) x num_fantasies x 1
        mean, sigma = self._mean_and_sigma(X.unsqueeze(-3))
        mean, sigma = mean.squeeze(-1), sigma.squeeze(-1)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        log_ei = _log_ei_helper(u) + sigma.log()
        # this is mathematically - though not numerically - equivalent to log(mean(ei))
        return logmeanexp(log_ei, dim=-1)


class NoisyExpectedImprovement(ExpectedImprovement):
    r"""Single-outcome Noisy Expected Improvement (via fantasies).

    This computes Noisy Expected Improvement by averaging over the Expected
    Improvement values of a number of fantasy models. Only supports the case
    `q=1`. Assumes that the posterior distribution of the model is Gaussian.
    The model must be single-outcome.

    `NEI(x) = E(max(y - max Y_baseline), 0)), (y, Y_baseline) ~ f((x, X_baseline))`,
    where `X_baseline` are previously observed points.

    Note: This acquisition function currently relies on using a SingleTaskGP
    with known observation noise. In other words, `train_Yvar` must be passed
    to the model. (required for noiseless fantasies).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
        >>> NEI = NoisyExpectedImprovement(model, train_X)
        >>> nei = NEI(test_X)

    NOTE: It is strongly recommended to use LogNoisyExpectedImprovement instead
    of regular NEI, as it can lead to substantially improved BO performance through
    improved numerics. See https://arxiv.org/abs/2310.20708 for details.
    """

    def __init__(
        self,
        model: GPyTorchModel,
        X_observed: Tensor,
        num_fantasies: int = 20,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Noisy Expected Improvement (via fantasies).

        Args:
            model: A fitted single-outcome model. Only `SingleTaskGP` models with
                known observation noise are currently supported.
            X_observed: A `n x d` Tensor of observed points that are likely to
                be the best observed points so far.
            num_fantasies: The number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance).
            maximize: If True, consider the problem a maximization problem.
        """
        _check_noisy_ei_model(model=model)
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        # Sample fantasies.
        from botorch.sampling.normal import SobolQMCNormalSampler

        # Drop gradients from model.posterior if X_observed does not require gradients
        # as otherwise, gradients of the GP's kernel's hyper-parameters are tracked
        # through the rsample_from_base_sample method of GPyTorchPosterior. These
        # gradients are usually only required w.r.t. the marginal likelihood.
        with nullcontext() if X_observed.requires_grad else torch.no_grad():
            posterior = model.posterior(X=X_observed)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        Y_fantasized = sampler(posterior).squeeze(-1)
        batch_X_observed = X_observed.expand(num_fantasies, *X_observed.shape)
        # The fantasy model will operate in batch mode
        fantasy_model = _get_noiseless_fantasy_model(
            model=model, batch_X_observed=batch_X_observed, Y_fantasized=Y_fantasized
        )
        best_f, _ = Y_fantasized.max(dim=-1) if maximize else Y_fantasized.min(dim=-1)
        super().__init__(model=fantasy_model, best_f=best_f, maximize=maximize)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Noisy Expected Improvement values at
            the given design points `X`.
        """
        # add batch dimension for broadcasting to fantasy models
        # (b1 x ... x bk) x num_fantasies x 1
        mean, sigma = self._mean_and_sigma(X.unsqueeze(-3))  # (b1 x ... x bk) x m1 x 1
        mean, sigma = mean.squeeze(-1), sigma.squeeze(-1)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return (sigma * _ei_helper(u)).mean(dim=-1)


class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)  # (b1 x ... x bk) x 1
        ucb = (mean if self.maximize else -mean) + self.beta.sqrt() * sigma
        return ucb.squeeze(-1)


class PosteriorMean(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Posterior Mean.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                actually returns -1 * minimum of the posterior mean.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Posterior Mean values at the
            given design points `X`.
        """
        mean, _ = self._mean_and_sigma(X, compute_sigma=False)  # (b1 x ... x bk) x 1
        if not self.maximize:
            mean = -mean
        return mean.squeeze(-1)


class ScalarizedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Scalarized Posterior Mean.

    This acquisition function returns a scalarized (across the q-batch)
    posterior mean given a vector of weights.
    """

    def __init__(
        self,
        model: Model,
        weights: Tensor,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Scalarized Posterior Mean.

        Args:
            model: A fitted single-outcome model.
            weights: A tensor of shape `q` for scalarization. In order to minimize
                the scalarized posterior mean, pass -weights.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("weights", weights)

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the scalarized posterior mean on the candidate set X.

        Args:
            X: A `(b1 x ... x bk) x q x d`-dim Tensor of `(b1 x ... x bk)`
                t-batches of `d`-dim design points each.

        Returns:
            A `(b1 x ... x bk)`-dim Tensor of Posterior Mean values at the given
            design points `X`.
        """
        # (b1 x ... x bk) x q x 1
        mean, _ = self._mean_and_sigma(X, compute_sigma=False)
        return mean.squeeze(-1) @ self.weights


class PosteriorStandardDeviation(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Standard Deviation.

    An acquisition function for pure exploration.
    Only supports the case of q=1. Requires the model's posterior to have
    `mean` and `variance` properties. The model must be either single-outcome
    or combined with a `posterior_transform` to produce a single-output posterior.

    Example:
        >>> import torch
        >>> from botorch.models.gp_regression import SingleTaskGP
        >>> from botorch.models.transforms.input import Normalize
        >>> from botorch.models.transforms.outcome import Standardize
        >>>
        >>> # Set up a model
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> model = SingleTaskGP(
        ...     train_X, train_Y, outcome_transform=Standardize(m=1),
        ...     input_transform=Normalize(d=2),
        ... )
        >>> # Now set up the acquisition function
        >>> PSTD = PosteriorStandardDeviation(model)
        >>> test_X = torch.zeros((1, 2), dtype=torch.float64)
        >>> std = PSTD(test_X)
        >>> std.item()
        0.16341639895667773
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Posterior Mean.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior standard deviation is negated.
                As a consequence,
                `optimize_acqf(PosteriorStandardDeviation(gp, maximize=False))`
                actually returns -1 * minimum of the posterior standard deviation.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior standard deviation on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Posterior Mean values at the
            given design points `X`.
        """
        _, std = self._mean_and_sigma(X)
        if not self.maximize:
            std = -std
        return std.view(X.shape[:-2])


# --------------- Helper functions for analytic acquisition functions. ---------------


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)


def _log_ei_helper(u: Tensor) -> Tensor:
    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == torch.float32 or u.dtype == torch.float64):
        raise TypeError(
            f"LogExpectedImprovement only supports torch.float32 and torch.float64 "
            f"dtypes, but received {u.dtype=}."
        )
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
    log_ei_upper = _ei_helper(u_upper).log()

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    # To this end, a second branch is necessary, depending on whether or not u is
    # smaller than approximately the negative inverse square root of the machine
    # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
    # as w approaches zero from below, even though the relative contribution to log_ei
    # vanishes in machine precision at that point.
    neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

    # mask u for to avoid NaNs in gradients in first and second branch
    u_lower = u.masked_fill(u > bound, bound)
    u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)

    # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
    # large negative numbers, and
    # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
    # The second special case is technically only required for single precision numbers
    # but does "the right thing" regardless.
    log_ei_lower = log_phi(u) + (
        torch.where(
            u > neg_inv_sqrt_eps,
            log1mexp(w),
            # The contribution of the next term relative to log_phi vanishes when
            # w_lower << eps but captures the leading order of the log1mexp term.
            -2 * u_lower.abs().log(),
        )
    )
    return torch.where(u > bound, log_ei_upper, log_ei_lower)


def _log_abs_u_Phi_div_phi(u: Tensor) -> Tensor:
    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.

    NOTE: In single precision arithmetic, the function becomes numerically unstable for
    u < -1e3. For this reason, a second branch in _log_ei_helper is necessary to handle
    this regime, where this function approaches -abs(u)^-2 asymptotically.

    The implementation is based on the following implementation of the logarithm of
    the scaled complementary error function (i.e. erfcx). Since we only require the
    positive branch for _log_ei_helper, _log_abs_u_Phi_div_phi does not have a branch,
    but is only valid for u < 0 (so that _neg_inv_sqrt2 * u > 0).

        def logerfcx(x: Tensor) -> Tensor:
            return torch.where(
                x < 0,
                torch.erfc(x.masked_fill(x > 0, 0)).log() + x**2,
                torch.special.erfcx(x.masked_fill(x < 0, 0)).log(),
        )

    Further, it is important for numerical accuracy to move u.abs() into the
    logarithm, rather than adding u.abs().log() to logerfcx. This is the reason
    for the rather complex name of this function: _log_abs_u_Phi_div_phi.
    """
    # get_constants_like allocates tensors with the appropriate dtype and device and
    # caches the result, which improves efficiency.
    a, b = get_constants_like(values=(_neg_inv_sqrt2, _log_sqrt_pi_div_2), ref=u)
    return torch.log(torch.special.erfcx(a * u) * u.abs()) + b


def _check_noisy_ei_model(model: GPyTorchModel) -> None:
    message = (
        "Only single-output `SingleTaskGP` models with known observation noise "
        "are currently supported for fantasy-based NEI & LogNEI."
    )
    if not isinstance(model, SingleTaskGP):
        raise UnsupportedError(f"{message} Model is not a `SingleTaskGP`.")
    if not isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        raise UnsupportedError(
            f"{message} Model likelihood is not a `FixedNoiseGaussianLikelihood`."
        )
    if model.num_outputs != 1:
        raise UnsupportedError(f"{message} Model has {model.num_outputs} outputs.")


def _get_noiseless_fantasy_model(
    model: SingleTaskGP, batch_X_observed: Tensor, Y_fantasized: Tensor
) -> SingleTaskGP:
    r"""Construct a fantasy model from a fitted model and provided fantasies.

    The fantasy model uses the hyperparameters from the original fitted model and
    assumes the fantasies are noiseless.

    Args:
        model: A fitted SingleTaskGP with known observation noise.
        batch_X_observed: A `b x n x d` tensor of inputs where `b` is the number of
            fantasies.
        Y_fantasized: A `b x n` tensor of fantasized targets where `b` is the number of
            fantasies.

    Returns:
        The fantasy model.
    """
    # initialize a copy of SingleTaskGP on the original training inputs
    # this makes SingleTaskGP a non-batch GP, so that the same hyperparameters
    # are used across all batches (by default, a GP with batched training data
    # uses independent hyperparameters for each batch).

    # We don't want to use the true `outcome_transform` and `input_transform` here
    # since the data being passed has already been transformed. We thus pass `None`
    # and will instead set them afterwards.
    fantasy_model = SingleTaskGP(
        train_X=model.train_inputs[0],
        train_Y=model.train_targets.unsqueeze(-1),
        train_Yvar=model.likelihood.noise_covar.noise.unsqueeze(-1),
        covar_module=deepcopy(model.covar_module),
        mean_module=deepcopy(model.mean_module),
        outcome_transform=None,
        input_transform=None,
    )

    Yvar = torch.full_like(Y_fantasized, 1e-7)

    # Set the outcome and input transforms of the fantasy model.
    # The transforms should already be in eval mode but just set them to be sure
    outcome_transform = getattr(model, "outcome_transform", None)
    if outcome_transform is not None:
        outcome_transform = deepcopy(outcome_transform).eval()
        fantasy_model.outcome_transform = outcome_transform
        # Need to transform the outcome just as in the SingleTaskGP constructor.
        # Need to unsqueeze for BoTorch and then squeeze again for GPyTorch.
        # Not transforming Yvar because 1e-7 is already close to 0 and it is a
        # relative, not absolute, value.
        Y_fantasized, _ = outcome_transform(
            Y_fantasized.unsqueeze(-1), Yvar.unsqueeze(-1), X=batch_X_observed
        )
        Y_fantasized = Y_fantasized.squeeze(-1)
    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None:
        fantasy_model.input_transform = deepcopy(input_transform).eval()

    # update training inputs/targets to be batch mode fantasies
    fantasy_model.set_train_data(
        inputs=batch_X_observed, targets=Y_fantasized, strict=False
    )
    # use noiseless fantasies
    fantasy_model.likelihood.noise_covar.noise = Yvar

    return fantasy_model
