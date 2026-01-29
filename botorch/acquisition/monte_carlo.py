#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling. See [Rezende2014reparam]_, [Wilson2017reparam]_ and
[Balandat2020botorch]_.

References

.. [Rezende2014reparam]
    D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and
    approximate inference in deep generative models. ICML 2014.

.. [Wilson2017reparam]
    J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
    The reparameterization trick for acquisition functions. ArXiv 2017.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Protocol

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    prune_inferior_points,
    repeat_to_match_aug_dim,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import (
    concatenate_pending_points,
    is_ensemble,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class MCAcquisitionFunction(AcquisitionFunction, MCSamplerMixin, ABC):
    r"""
    Abstract base class for Monte-Carlo based batch acquisition functions.
    """

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""
        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated on the fly within the
                ``get_posterior_samples`` method using
                ``botorch.sampling.get_sampler``.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See ``ForkedRNGSampler`` and ``StochasticSampler`` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not posterior_transform.scalarize:
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if objective is None:
            objective = IdentityMCObjective()
        self.posterior_transform = posterior_transform
        self.objective: MCAcquisitionObjective = objective
        self.set_X_pending(X_pending)

    def _get_samples_and_objectives(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Computes posterior samples and objective values at input X.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of model inputs.

        Returns:
            A two-tuple ``(samples, obj)``, where ``samples`` is a tensor of posterior
            samples with shape ``sample_shape x batch_shape x q x m``, and ``obj`` is a
            tensor of MC objective values with shape ``sample_shape x batch_shape x q``.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples=samples, X=X)
        return samples, obj

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a ``batch_shape x q x d`` X Tensor of t-batches with ``q``
        ``d``-dim design points each, and returns a Tensor with shape
        ``batch_shape'``, where ``batch_shape'`` is the broadcasted batch shape of
        model and input ``X``. Should utilize the result of ``set_X_pending`` as
        needed to account for pending function evaluations.
        """
        pass  # pragma: no cover


class SampleReductionProtocol(Protocol):
    """For static type check of SampleReducingMCAcquisitionFunction's mc_reduction."""

    @staticmethod
    def __call__(X: Tensor, *, dim: torch.Size) -> Tensor:
        pass  # pragma: no cover


class SampleReducingMCAcquisitionFunction(MCAcquisitionFunction):
    r"""MC-based batch acquisition function that reduces across samples and implements
    a general treatment of outcome constraints.

    This class's ``forward`` computes the - possibly constrained - acquisition
    value by (1) computing the unconstrained utility for each MC sample using
    ``_sample_forward``, (2) weighing the utility values by the constraint
    indicator per MC sample, and (3) reducing (e.g. averaging) the weighted
    utility values over the MC dimension.

    NOTE: Do *NOT* override the ``forward`` method, unless you have thought
    about it well.

    ``forward`` is implemented generically to incorporate constraints in a
    principled way, and takes care of reducing over the Monte Carlo and batch
    dimensions via the ``sample_reduction`` and ``q_reduction`` arguments, which
    default to ``torch.mean`` and ``torch.max``, respectively.

    In order to implement a custom SampleReducingMCAcquisitionFunction, we only
    need to implement the ``_sample_forward(obj: Tensor) -> Tensor`` method,
    which maps objective samples to acquisition utility values without reducing
    the Monte Carlo and batch (i.e. q) dimensions (see details in the docstring
    of ``_sample_forward``).

    A note on design choices:

    The primary purpose of ``SampleReducingMCAcquisitionFunction``is to support
    outcome constraints. On the surface, designing a wrapper
    ``ConstrainedMCAcquisitionFunction`` could be an elegant solution to this
    end, but it would still require the acquisition functions to implement a
    ``_sample_forward`` method to weigh acquisition utilities at the sample
    level. Further, ``qNoisyExpectedImprovement`` is a special case that is
    hard to encompass in this pattern, since it requires the computation of the
    best *feasible* objective, which requires access to the constraint
    functions. However, if the constraints are stored in a wrapper class, they
    will be inaccessible to the forward pass. These problems are circumvented by
    the design of this class.
    """

    _log: bool = False  # whether the acquisition utilities are in log-space

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        sample_reduction: SampleReductionProtocol = torch.mean,
        q_reduction: SampleReductionProtocol = torch.amax,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        fat: list[bool | None] | bool = False,
    ):
        r"""Constructor of SampleReducingMCAcquisitionFunction.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given, a
                sampler is generated on the fly within the
                ``get_posterior_samples`` method using
                ``botorch.sampling.get_sampler``.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See ``ForkedRNGSampler`` and ``StochasticSampler`` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
                NOTE: ``ConstrainedMCObjective`` for outcome constraints is
                deprecated in favor of passing the ``constraints`` directly to this
                constructor.
            posterior_transform: A ``PosteriorTransform`` (optional).
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            sample_reduction: A callable that takes in a ``sample_shape x
                batch_shape`` Tensor of acquisition utility values, a keyword-argument
                ``dim`` that specifies the sample dimensions to reduce over, and
                returns a ``batch_shape``-dim Tensor of acquisition values.
            q_reduction: A callable that takes in a ``sample_shape x batch_shape
                x q`` Tensor of acquisition utility values, a keyword-argument ``dim``
                that specifies the q dimension to reduce over (i.e. -1), and returns
                a ``sample_shape x batch_shape``-dim Tensor of acquisition values.
            constraints: A list of constraint callables which map a Tensor of
                posterior samples of dimension ``sample_shape x batch-shape x q x
                m``-dim to a ``sample_shape x batch-shape x q``-dim Tensor. The
                associated constraints are considered satisfied if the output is less
                than zero. NOTE: Constraint-weighting is only compatible with
                non-negative acquistion utilities, e.g. all improvement-based
                acquisition functions.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. For more details, on this
                parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
            fat: Whether to apply a fat-tailed smooth approximation to the feasibility
                indicator or the canonical sigmoid approximation. For more details,
                on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
        """
        if constraints is not None and isinstance(objective, ConstrainedMCObjective):
            raise ValueError(
                "ConstrainedMCObjective as well as constraints passed to constructor."
                "Choose one or the other, preferably the latter."
            )
        # TODO: deprecate ConstrainedMCObjective
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        # Shall the need arise, sample_dim could be exposed in the constructor.
        sample_dim = tuple(range(len(self.sample_shape)))
        # the ensemble dimension is assumed to be the last dimension
        if is_ensemble(model):
            sample_dim = sample_dim + (-1,)
        self._sample_reduction = partial(sample_reduction, dim=sample_dim)
        self._q_reduction = partial(q_reduction, dim=-1)
        self._constraints = constraints
        self._eta = eta
        self._fat = fat

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Computes the acquisition value associated with the input ``X``. Weighs
        the acquisition utility values by smoothed constraint indicators if
        ``constraints`` was passed to the constructor of the class. Applies
        ``self.sample_reduction`` and ``self.q_reduction`` to reduce over the Monte
        Carlo and batch (q) dimensions.

        NOTE: Do *NOT* override the ``forward`` method for a custom acquisition
        function. Instead, implement the ``_sample_forward`` method. See the
        docstring of this class for details.

        Args:
            X: A ``batch_shape x q x d`` Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A Tensor with shape ``batch_shape'``, where ``batch_shape'`` is the
            broadcasted batch shape of model and input ``X``.
        """
        non_reduced_acqval = self._non_reduced_forward(X=X)
        return self._sample_reduction(self._q_reduction(non_reduced_acqval))

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        """Compute the constrained acquisition values at the MC-sample, q level.

        Args:
            X: A ``batch_shape x q x d`` Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A Tensor with shape ``sample_sample x batch_shape x q``.
        """
        samples, obj = self._get_samples_and_objectives(X)
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = self._sample_forward(obj)  # ``sample_sample x batch_shape x q``
        return self._apply_constraints(acqval=acqval, samples=samples)

    @abstractmethod
    def _sample_forward(self, obj: Tensor) -> Tensor:
        """Evaluates the acquisition utility per MC sample based on objective value obj.
        Should utilize the result of ``set_X_pending`` as needed to account for pending
        function evaluations.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition
                utility values.
        """
        pass  # pragma: no cover

    def _apply_constraints(self, acqval: Tensor, samples: Tensor) -> Tensor:
        """Multiplies the acquisition utility by constraint indicators.

        Args:
            acqval: ``sample_shape x batch_shape x q``-dim acquisition utility values.
            samples: ``sample_shape x batch_shape x q x m``-dim posterior samples.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition
                utility values multiplied by a smoothed constraint indicator
                per sample.
        """
        if self._constraints is not None:
            if not self._log and (acqval < 0).any():
                raise ValueError(
                    "Constraint-weighting requires unconstrained "
                    "acquisition values to be non-negative."
                )
            ind = compute_smoothed_feasibility_indicator(
                constraints=self._constraints,
                samples=samples,
                eta=self._eta,
                log=self._log,
                fat=self._fat,
            )
            acqval = acqval.add(ind) if self._log else acqval.mul(ind)
        return acqval


class qExpectedImprovement(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    ``qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)``

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)

    NOTE: It is strongly recommended to use ``qLogExpectedImprovement`` instead
    of regular ``qEI``, as it can lead to substantially improved BO performance through
    improved numerics. See https://arxiv.org/abs/2310.20708 for details.
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
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed
                noiseless). Can be a scalar, or a ``batch_shape``-dim tensor.
                In case of a batched model, the tensor can specify different
                values for each element of the batch.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to ``IdentityMCObjective()``.
                NOTE: ``ConstrainedMCObjective`` for outcome constraints is
                deprecated in favor of passing the ``constraints`` directly to
                this constructor.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
            constraints: A list of constraint callables which map a Tensor of
                posterior samples of dimension ``sample_shape x batch-shape x q
                x m``-dim to a ``sample_shape x batch-shape x q``-dim Tensor.
                The associated constraints are considered satisfied if the output
                is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the
                sigmoid approximation to the constraint indicators. For more
                details, on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
        """
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f))

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement per sample on the candidate set ``X``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of improvement
                utility values.
        """
        return (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)


class qNoisyExpectedImprovement(
    SampleReducingMCAcquisitionFunction, CachedCholeskyMCSamplerMixin
):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a ``best_f`` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the ``q`` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    ``qNEI(X) = E(max(max Y - max Y_baseline, 0))``, where
    ``(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)``

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)

    NOTE: It is strongly recommended to use ``qLogNoisyExpectedImprovement`` instead
    of regular ``qNEI``, as it can lead to substantially improved BO performance through
    improved numerics. See https://arxiv.org/abs/2310.20708 for details.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        prune_baseline: bool = True,
        cache_root: bool | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
        marginalize_dim: int | None = None,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A ``batch_shape x r x d``-dim Tensor of ``r`` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
                NOTE: ``ConstrainedMCObjective`` for outcome constraints is
                deprecated in favor of passing the ``constraints`` directly to
                this constructor.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into ``X`` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in ``X_baseline`` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                ``botorch.acquisition.utils.prune_inferior_points`` on ``X_baseline``
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over ``X_baseline`` and use low-rank updates.
            constraints: A list of constraint callables which map a Tensor of
                posterior samples of dimension ``sample_shape x batch-shape x q
                x m``-dim to a ``sample_shape x batch-shape x q``-dim Tensor.
                The associated constraints are considered satisfied if the output
                is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the
                sigmoid approximation to the constraint indicators. For more
                details, on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
            marginalize_dim: The dimension to marginalize over.

        TODO: similar to qNEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental qNEI from the new point. This would greatly increase
        efficiency for large batches.
        """
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        CachedCholeskyMCSamplerMixin.__init__(
            self, model=model, cache_root=cache_root, sampler=sampler
        )
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                constraints=self._constraints,
                marginalize_dim=marginalize_dim,
            )
        self.register_buffer("X_baseline", X_baseline)
        # registering buffers for _get_samples_and_objectives in the next ``if`` block
        self.register_buffer("baseline_samples", None)
        self.register_buffer("baseline_obj", None)
        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():  # this is _get_samples_and_objectives(X_baseline)
                posterior = self.model.posterior(
                    X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to ``posterior.mvn`` allows for reuse within
                #  this function, which may be helpful if the same root decomposition
                #  is produced by the calls to ``self.base_sampler`` and
                #  ``self._cache_root_decomposition``.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                baseline_samples = self.get_posterior_samples(posterior)
                baseline_obj = self.objective(baseline_samples, X=X_baseline)

            # We make a copy here because we will write an attribute ``base_samples``
            # to ``self.base_sampler.base_samples``, and we don't want to mutate
            # ``self.sampler``.
            self.base_sampler = deepcopy(self.sampler)
            self.baseline_samples = baseline_samples
            self.baseline_obj = baseline_obj
            self.register_buffer(
                "_baseline_best_f",
                self._compute_best_feasible_objective(
                    samples=baseline_samples, obj=baseline_obj
                ),  # ``sample_shape x batch_shape``-dim
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    def compute_best_f(self, obj: Tensor) -> Tensor:
        """Computes the best (feasible) noisy objective value.

        Args:
            obj: ``sample_shape x batch_shape x q``-dim Tensor of objectives in forward.

        Returns:
            A ``sample_shape x batch_shape``-dim Tensor of best feasible objectives.
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
                *(1,)
                * (obj.ndim - val.ndim - 1),  # pad to match obj, without ``q``-dim
                *val.shape[n_sample_dims:],  # the rest
            ]
        )
        return val.view(view_shape).to(obj)  # obj.shape[:-1], i.e. without ``q``-dim`

    def _sample_forward(self, obj: Tensor) -> Tensor:
        """Evaluate qNoisyExpectedImprovement per objective value in ``obj``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of noisy improvement values.
        """
        return (obj - self.compute_best_f(obj).unsqueeze(-1)).clamp_min(0)

    def _get_samples_and_objectives(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A ``batch_shape x q x d``-dim tensor of inputs.

        Returns:
            A two-tuple ``(samples, obj)``, where ``samples`` is a tensor of posterior
            samples with shape ``sample_shape x batch_shape x q x m``, and ``obj`` is a
            tensor of MC objective values with shape ``sample_shape x batch_shape x q``.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            samples = samples_full[..., -q:, :]
            obj_full = self.objective(samples_full, X=X_full)
            # assigning baseline buffers so ``best_f`` can be computed in
            # _sample_forward
            self.baseline_obj, obj = obj_full[..., :-q], obj_full[..., -q:]
            self.baseline_samples = samples_full[..., :-q, :]
        else:
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
            samples: ``sample_shape x batch_shape x q x m``-dim posterior samples.
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape``-dim Tensor of best feasible objectives.
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


class qProbabilityOfImprovement(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement.

    Estimates the probability of improvement over the current best observed
    value by sampling from the joint posterior distribution of the q-batch.
    MC-based estimates of a probability involves taking expectation of an
    indicator function; to support auto-differentiation, the indicator is
    replaced with a sigmoid function with temperature parameter ``tau``.

    ``qPI(X) = P(max Y >= best_f), Y ~ f(X), X = (x_1,...,x_q)``

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qPI = qProbabilityOfImprovement(model, best_f, sampler)
        >>> qpi = qPI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        tau: float = 1e-3,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        r"""q-Probability of Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed
                noiseless). Can be a ``batch_shape``-shaped tensor, which in
                case of a batched model specifies potentially different values
                for each element of the batch.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
                NOTE: ``ConstrainedMCObjective`` for outcome constraints is
                deprecated in favor of passing the ``constraints`` directly to
                this constructor.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
            constraints: A list of constraint callables which map posterior
                samples to a scalar. The associated constraint is considered
                satisfied if this scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the
                sigmoid approximation to the constraint indicators. For more
                details, on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        best_f = torch.as_tensor(best_f).unsqueeze(-1)  # adding batch dim
        self.register_buffer("best_f", best_f)
        self.register_buffer("tau", torch.as_tensor(tau))

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement per sample on the candidate set ``X``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of improvement
                indicators.
        """
        improvement = obj - self.best_f.to(obj)
        return torch.sigmoid(improvement / self.tau)


class qSimpleRegret(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Simple Regret.

    Samples from the joint posterior over the q-batch and computes the simple regret.

    ``qSR(X) = E(max Y), Y ~ f(X), X = (x_1,...,x_q)``

    Constraints should be provided as a ``ConstrainedMCObjective``.
    Passing ``constraints`` as an argument is not supported. This is because
    ``SampleReducingMCAcquisitionFunction`` computes the acquisition values on
    the sample level and then weights the sample-level acquisition values by a
    soft feasibility indicator. Hence, it expects non-log acquisition function
    values to be non-negative. ``qSimpleRegret`` acquisition values can be
    negative, so we instead use a ``ConstrainedMCObjective`` which applies
    constraints to the objectives (e.g. before computing the acquisition
    function) and shifts negative objective values using an infeasible cost to
    ensure non-negativity (before applying constraints and shifting them back).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qSR = qSimpleRegret(model, sampler)
        >>> qsr = qSR(test_X)
    """

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""q-Simple Regret.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A ``m x d``-dim Tensor of ``m`` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qSimpleRegret per sample on the candidate set ``X``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of simple regret values.
        """
        return obj


class qUpperConfidenceBound(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    ``qUCB = E(max(mu + |Y_tilde - mu|))``, where ``Y_tilde ~ N(mu, beta pi/2 Sigma)``
    and ``f(X)`` has distribution ``N(mu, Sigma)``.

    Constraints should be provided as a ``ConstrainedMCObjective``.
    Passing ``constraints`` as an argument is not supported. This is because
    ``SampleReducingMCAcquisitionFunction`` computes the acquisition values on
    the sample level and then weights the sample-level acquisition values by a
    soft feasibility indicator. Hence, it expects non-log acquisition function
    values to be non-negative. ``qUpperConfidenceBound`` acquisition values can
    be negative, so we instead use a ``ConstrainedMCObjective`` which applies
    constraints to the objectives (e.g. before computing the acquisition
    function) and shifts negative objective values using an infeasible cost to
    ensure non-negativity (before applying constraints and shifting them back).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design
                points that have points that have been submitted for function
                evaluation but have not yet been evaluated. Concatenated into X
                upon forward call. Copied and set to have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = self._get_beta_prime(beta=beta)

    def _get_beta_prime(self, beta: float) -> float:
        return math.sqrt(beta * math.pi / 2)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound per sample on the candidate set ``X``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return mean + self.beta_prime * (obj - mean).abs()


class qLowerConfidenceBound(qUpperConfidenceBound):
    r"""MC-based batched lower confidence bound.

    This acquisition function is useful for confident/risk-averse decision making.
    This acquisition function is intended to be maximized as with qUpperConfidenceBound,
    but the qLowerConfidenceBound will be pessimistic in the face of uncertainty and
    lead to conservative candidates.
    """

    def _get_beta_prime(self, beta: float) -> float:
        """Multiply beta prime by -1 to get the lower confidence bound."""
        return -super()._get_beta_prime(beta=beta)


class qPosteriorStandardDeviation(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Posterior Standard Deviation.

    An acquisition function for pure exploration.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qPSTD = qPosteriorStandardDeviation(model, sampler)
        >>> std = qPSTD(test_X)
    """

    def __init__(
        self,
        model: Model,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        r"""q-Posterior Standard Deviation.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. See
                ``MCAcquisitionFunction`` more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design
                points that have points that have been submitted for function
                evaluation but have not yet been evaluated. Concatenated into X
                upon forward call. Copied and set to have no gradient.
            constraints: A list of constraint callables which map a Tensor of
                posterior samples of dimension ``sample_shape x batch-shape x q
                x m``-dim to a ``sample_shape x batch-shape x q``-dim Tensor.
                The associated constraints are considered satisfied if the output
                is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the
                sigmoid approximation to the constraint indicators. For more
                details, on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        self._scale = math.sqrt(math.pi / 2)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qPosteriorStandardDeviation per sample on the candidate set ``X``.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return (obj - mean).abs() * self._scale
