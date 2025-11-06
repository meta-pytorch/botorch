# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Monte-Carlo variants of the LogEI family of improvements-based acquisition functions,
see [Ament2023logei]_ for details.

References

.. [Ament2023logei]
    S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy.
    Unexpected Improvements to Expected Improvement for Bayesian Optimization. Advances
    in Neural Information Processing Systems 36, 2023.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import TypeVar

import torch
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    prune_inferior_points,
)
from botorch.exceptions.errors import BotorchError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import (
    fatmax,
    log_fatplus,
    log_softplus,
    logmeanexp,
    smooth_amax,
)
from botorch.utils.transforms import match_batch_shape
from torch import Tensor

"""
NOTE: On the default temperature parameters:

tau_relu: It is generally important to set `tau_relu` to be very small, in particular,
smaller than the expected improvement value. Otherwise, the optimization can stagnate.
By setting `tau_relu=1e-6` by default, stagnation is exceedingly unlikely to occur due
to the smooth ReLU approximation for practical applications of BO.
IDEA: We could consider shrinking `tau_relu` with the progression of the optimization.

tau_max: This is only relevant for the batch (`q > 1`) case, and `tau_max=1e-2` is
sufficient to get a good approximation to the maximum improvement in the batch of
candidates. If `fat=False`, the smooth approximation to the maximum can saturate
numerically. It is therefore recommended to use `fat=True` when optimizing batches
of `q > 1` points.
"""
TAU_RELU = 1e-6
TAU_MAX = 1e-2
FloatOrTensor = TypeVar("FloatOrTensor", float, Tensor)


class LogImprovementMCAcquisitionFunction(SampleReducingMCAcquisitionFunction):
    r"""
    Abstract base class for Monte-Carlo-based batch LogEI acquisition functions.
    """

    _log: bool = True

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ) -> None:
        r"""Constructor of the base class for LogEI acquisition functions.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See `ForkedRNGSampler` and `StochasticSampler` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)constraint_indicator` for more details on this parameter.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the `max` operator over the `q` candidate points.
        """
        if isinstance(objective, ConstrainedMCObjective):
            raise BotorchError(
                "Log-Improvement should not be used with `ConstrainedMCObjective`."
                "Please pass the `constraints` directly to the constructor of the "
                "acquisition function."
            )
        q_reduction = partial(fatmax if fat else smooth_amax, tau=tau_max)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sample_reduction=logmeanexp,
            q_reduction=q_reduction,
            constraints=constraints,
            eta=eta,
            fat=fat,
        )
        self.tau_max = tau_max


class qLogExpectedImprovement(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Expected Improvement.

    This computes qLogEI by
    (1) sampling the joint posterior over q points,
    (2) evaluating the smoothed log improvement over the current best for each sample,
    (3) smoothly maximizing over q, and
    (4) averaging over the samples in log space.

    See [Ament2023logei]_ for details. Formally,

    `qLogEI(X) ~ log(qEI(X)) = log(E(max(max Y - best_f, 0)))`.

    where `Y ~ f(X)`, and `X = (x_1,...,x_q)`, .

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qLogEI = qLogExpectedImprovement(model, best_f, sampler)
        >>> qei = qLogEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
    ) -> None:
        r"""q-Log Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a scalar, or a `batch_shape`-dim tensor. In case of a batched model, the
                tensor can specify different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into `X` upon forward call. Copied and set to have no
                gradient.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)smoothed_constraint_indicator` for details.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max.
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            tau_max=check_tau(tau_max, name="tau_max"),
            fat=fat,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.tau_relu = check_tau(tau_relu, name="tau_relu")

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogExpectedImprovement on the candidate set `X`.

        Args:
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `mc_shape x batch_shape x q`-dim Tensor of expected improvement values.
        """
        li = _log_improvement(
            Y=obj,
            best_f=self.best_f,
            tau=self.tau_relu,
            fat=self._fat,
        )
        return li


class qLogNoisyExpectedImprovement(
    LogImprovementMCAcquisitionFunction, CachedCholeskyMCSamplerMixin
):
    r"""MC-based batch Log Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. A smooth approximation
    to the canonical improvement over previously observed points is computed
    for each sample and the logarithm of the average is returned.

    See [Ament2023logei]_ for details. Formally,

    `qLogNEI(X) ~ log(qNEI(X)) = Log E(max(max Y - max Y_baseline, 0))`,

    where `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`.

    For optimizing a batch of `q > 1` points using sequential greedy optimization,
    the incremental improvement from the latest point is computed and returned by
    default. I.e. the pending points are treated X_baseline. Often, the incremental
    EI is easier to optimize.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qLogNEI = qLogNoisyExpectedImprovement(model, train_X, sampler)
        >>> acqval = qLogNEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        prune_baseline: bool = True,
        cache_root: bool | None = None,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
        marginalize_dim: int | None = None,
        incremental: bool = True,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)smoothed_constraint_indicator` for details.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max.
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU.
            marginalize_dim: The dimension to marginalize over.
            incremental: Whether to compute incremental EI over the pending points
                or compute EI of the joint batch improvement (including pending
                points).

        TODO: similar to qNEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental q(Log)NEI from the new point. This would greatly increase
        efficiency for large batches.
        """
        # TODO: separate out baseline variables initialization and other functions
        # in qNEI to avoid duplication of both code and work at runtime.
        self.incremental = incremental

        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            # we set X_pending in init_baseline for incremental NEI
            X_pending=X_pending if not incremental else None,
            constraints=constraints,
            eta=eta,
            fat=fat,
            tau_max=tau_max,
        )
        self.tau_relu = tau_relu
        self.prune_baseline = prune_baseline
        self.marginalize_dim = marginalize_dim
        if incremental:
            self.X_pending = None  # required to initialize attribute for optimize_acqf
        self._init_baseline(
            model=model,
            X_baseline=X_baseline,
            # This is ignored in incremental=False
            X_pending=X_pending,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            cache_root=cache_root,
        )

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogNoisyExpectedImprovement per sample on the candidate set `X`.

        Args:
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of log noisy expected smoothed
            improvement values.
        """
        return _log_improvement(
            Y=obj,
            best_f=self.compute_best_f(obj),
            tau=self.tau_relu,
            fat=self._fat,
        )

    def _init_baseline(
        self,
        model: Model,
        X_baseline: Tensor,
        X_pending: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        cache_root: bool | None = None,
    ) -> None:
        CachedCholeskyMCSamplerMixin.__init__(
            self, model=model, cache_root=cache_root, sampler=sampler
        )
        if self.prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                marginalize_dim=self.marginalize_dim,
                constraints=self._constraints,
            )
        self.register_buffer("_X_baseline", X_baseline)
        # full_X_baseline is the set of points that should be considered as the
        # incumbent. For incremental EI, this contains the previously evaluated
        # points (X_baseline) and pending points (X_pending). For non-incremental
        # EI, this contains the  previously evaluated points (X_baseline).
        if X_pending is not None and self.incremental:
            full_X_baseline = torch.cat([X_baseline, X_pending], dim=-2)
        else:
            full_X_baseline = X_baseline
        self.register_buffer("_full_X_baseline", full_X_baseline)
        # registering buffers for _get_samples_and_objectives in the next `if` block
        self.register_buffer("baseline_samples", None)
        self.register_buffer("baseline_obj", None)
        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():  # this is _get_samples_and_objectives(X_baseline)
                posterior = self.model.posterior(
                    self.X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to `posterior.mvn` allows for reuse within
                #   this function, which may be helpful if the same root decomposition
                #   is produced by the calls to `self.base_sampler` and
                #   `self._cache_root_decomposition`.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                self.baseline_samples = self.get_posterior_samples(posterior)
                self.baseline_obj = self.objective(
                    self.baseline_samples, X=self.X_baseline
                )

            # We make a copy here because we will write an attribute `base_samples`
            # to `self.base_sampler.base_samples`, and we don't want to mutate
            # `self.sampler`.
            self.base_sampler = deepcopy(self.sampler)
            self.register_buffer(
                "_baseline_best_f",
                self._compute_best_feasible_objective(
                    samples=self.baseline_samples, obj=self.baseline_obj
                ),
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    @property
    def X_baseline(self) -> Tensor:
        """Returns the set of points that should be considered as the incumbent.

        For incremental EI, this contains the previously evaluated points
        (X_baseline) and pending points (X_pending). For non-incremental
        EI, this contains the  previously evaluated points (X_baseline).
        """
        return self._full_X_baseline

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Here pending points are concatenated with X_baseline and incremental
        NEI is computed.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if not self.incremental:
            return super().set_X_pending(X_pending=X_pending)
        if X_pending is None:
            if not hasattr(self, "_full_X_baseline") or (
                self._full_X_baseline.shape[-2] == self._X_baseline.shape[-2]
            ):
                return
            else:
                # reset pending points
                X_pending = None
        self._init_baseline(
            model=self.model,
            X_baseline=self._X_baseline,
            X_pending=X_pending,
            sampler=self.sampler,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            cache_root=self._cache_root,
        )

    def compute_best_f(self, obj: Tensor) -> Tensor:
        """Computes the best (feasible) noisy objective value.

        Args:
            obj: `sample_shape x batch_shape x q`-dim Tensor of objectives in forward.

        Returns:
            A `sample_shape x batch_shape`-dim Tensor of best feasible objectives.
        """
        if self._cache_root:
            val = self._baseline_best_f
        else:
            val = self._compute_best_feasible_objective(
                samples=self.baseline_samples, obj=self.baseline_obj
            )
        # ensuring shape, dtype, device compatibility with obj
        n_sample_dims = len(self.sample_shape)
        view_shape = torch.Size(
            [
                *val.shape[:n_sample_dims],  # sample dimensions
                *(1,) * (obj.ndim - val.ndim - 1),  # pad to match obj without `q`-dim
                *val.shape[n_sample_dims:],  # the rest
            ]
        )
        return val.view(view_shape).to(obj)  # obj.shape[:-1], i.e. without `q`-dim`

    def _get_samples_and_objectives(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A two-tuple `(samples, obj)`, where `samples` is a tensor of posterior
            samples with shape `sample_shape x batch_shape x q x m`, and `obj` is a
            tensor of MC objective values with shape `sample_shape x batch_shape x q`.
        """
        n_baseline, q = self.X_baseline.shape[-2], X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            obj_full = self.objective(samples_full, X=X_full)
            # Calculate the positive index for splitting the samples & objective values.
            split_dim = len(obj_full.shape) - 1
            # assigning baseline buffers so `best_f` can be computed in _sample_forward
            self.baseline_samples, samples = samples_full.split(
                [n_baseline, q], dim=split_dim
            )
            self.baseline_obj, obj = obj_full.split([n_baseline, q], dim=split_dim)
            return samples, obj

        # handle one-to-many input transforms
        n_plus_q = X_full.shape[-2]
        n_w = posterior._extended_shape()[-2] // n_plus_q
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        obj = self.objective(samples, X=X_full[..., -q:, :])
        return samples, obj

    def _compute_best_feasible_objective(self, samples: Tensor, obj: Tensor) -> Tensor:
        r"""Computes best feasible objective value from samples.

        Args:
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape`-dim Tensor of best feasible objectives.
        """
        return compute_best_feasible_objective(
            samples=samples,
            obj=obj,
            constraints=self._constraints,
            model=self.model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            X_baseline=self.X_baseline,
        )


class qLogProbabilityOfFeasibility(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch LogProbabilityOfFeasibility.

    This computes the log probability of feasibility by
    (1) sampling the joint posterior over q points
    (2) evaluating the feasibility of each sample
    (3) averaging over the sample and batch dimensions.

    `log_prob_feas(X) = log(P(f(X) <= 0)), where f(X) ~ GP`.
    """

    def __init__(
        self,
        model: Model,
        constraints: list[Callable[[Tensor], Tensor]],
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ):
        r"""Constructor of the batch log probability of feasibility acquisition
        function.

        Args:
            model: A fitted model.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See `ForkedRNGSampler` and `StochasticSampler` as examples.
            objective: Not used, kept for compatibility with interface.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)constraint_indicator` for more details on this parameter.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the `max` operator over the `q` candidate points.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            # theoretically we don't need an objective, but MCAcquisitionFunction will
            # throw an error if the model has multiple outputs and the objective is None
            objective=IdentityMCObjective(),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            fat=fat,
            tau_max=tau_max,
        )

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        """Compute the feasibility values at the MC-sample and batch (q) level.

        Args:
            X: A `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim
                design points each.

        Returns:
            A Tensor with shape `sample_sample x batch_shape x q`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        return compute_smoothed_feasibility_indicator(
            constraints=self._constraints,
            samples=self.get_posterior_samples(posterior),
            eta=self._eta,
            log=self._log,
            fat=self._fat,
        )

    def _sample_forward(self, obj: Tensor) -> Tensor:
        """Not necessary, since we are implementing `_non_reduced_forward` without it.
        Need to provide an implementation to avoid an abstract method error.
        """
        raise NotImplementedError("This should be dead code.")  # pragma: no cover


"""
###################################### utils ##########################################
"""


def _log_improvement(
    Y: Tensor,
    best_f: Tensor,
    tau: float | Tensor,
    fat: bool,
) -> Tensor:
    """Computes the logarithm of the softplus-smoothed improvement, i.e.
    `log_softplus(Y - best_f, beta=(1 / tau))`.
    Note that softplus is an approximation to the regular ReLU objective whose maximum
    pointwise approximation error is linear with respect to tau as tau goes to zero.

    Args:
        obj: `mc_samples x batch_shape x q`-dim Tensor of output samples.
        best_f: Best previously observed objective value(s), broadcastable with
            `mc_samples x batch_shape`-dim Tensor, i.e. `obj`'s dims without `q`.
        tau: Temperature parameter for smooth approximation of ReLU.
            as `tau -> 0`, maximum pointwise approximation error is linear w.r.t. `tau`.
        fat: Toggles the logarithmic / linear asymptotic behavior of the
            smooth approximation to ReLU.

    Returns:
        A `mc_samples x batch_shape x q`-dim Tensor of improvement values.
    """
    log_soft_clamp = log_fatplus if fat else log_softplus
    Z = Y - best_f.unsqueeze(-1).to(Y)
    return log_soft_clamp(Z, tau=tau)  # ~ ((Y - best_f) / Y_std).clamp(0)


def check_tau(tau: FloatOrTensor, name: str) -> FloatOrTensor:
    """Checks the validity of the tau arguments of the functions below, and returns
    `tau` if it is valid."""
    if isinstance(tau, Tensor) and tau.numel() != 1:
        raise ValueError(f"{name} is not a scalar: {tau.numel()=}.")
    if not (tau > 0):
        raise ValueError(f"{name} is non-positive: {tau=}.")
    return tau
