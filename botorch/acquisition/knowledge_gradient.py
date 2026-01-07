#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch Knowledge Gradient (KG) via one-shot optimization as introduced in
[Balandat2020botorch]_. For broader discussion of KG see also [Frazier2008knowledge]_
and [Wu2016parallelkg]_.

.. [Balandat2020botorch]
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and
    E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
    Advances in Neural Information Processing Systems 33, 2020.

.. [Frazier2008knowledge]
    P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for
    sequential information collection. SIAM Journal on Control and Optimization,
    2008.

.. [Wu2016parallelkg]
    J. Wu and P. Frazier. The parallel knowledge gradient method for batch
    bayesian optimization. NIPS 2016.
"""

from __future__ import annotations

from collections.abc import Callable

from copy import deepcopy
from typing import Any

import torch
from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    MCSamplerMixin,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class qKnowledgeGradient(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""Batch Knowledge Gradient using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: int | None = 64,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        inner_sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
        current_value: Tensor | None = None,
    ) -> None:
        r"""q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None`, then the analytic posterior mean is used. Otherwise, the
                objective is MC-evaluated (using inner_sampler).
            posterior_transform: An optional PosteriorTransform. If given, this
                transforms the posterior before evaluation. If `objective is None`,
                then the analytic posterior mean of the transformed posterior is
                used. If `objective` is given, the `inner_sampler` is used to draw
                samples from the transformed posterior, which are then evaluated under
                the `objective`.
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
        """
        if sampler is None:
            if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization over X, X_fantasies
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        elif num_fantasies is not None:
            if sampler.sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        else:
            num_fantasies = sampler.sample_shape[0]
        super(MCAcquisitionFunction, self).__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        # if not explicitly specified, we use the posterior mean for linear objs
        if isinstance(objective, MCAcquisitionObjective) and inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        elif objective is not None and not isinstance(
            objective, MCAcquisitionObjective
        ):
            raise UnsupportedError(
                "Objectives that are not an `MCAcquisitionObjective` are not supported."
            )

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
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.set_X_pending(X_pending)
        self.X_pending: Tensor = self.X_pending
        self.inner_sampler = inner_sampler
        self.num_fantasies: int = num_fantasies
        self.current_value = current_value

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual,
            sampler=self.sampler,
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            sampler=self.inner_sampler,
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def evaluate(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X_actual` by
        solving the inner optimization problem.

        Args:
            X: A `b x q x d` Tensor with `b` t-batches of `q` design points
                each. Unlike `forward()`, this does not include solutions of the
                inner optimization problem.
            bounds: A `2 x d` tensor of lower and upper bounds for each column of
                the solutions to the inner problem.
            kwargs: Additional keyword arguments. This includes the options for
                optimization of the inner problem, i.e. `num_restarts`, `raw_samples`,
                an `options` dictionary to be passed on to the optimization helpers, and
                a `scipy_options` dictionary to be passed to `scipy.optimize.minimize`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X[b]` is averaged across the fantasy models.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X[b]`.
        """
        if hasattr(self, "expand"):
            X = self.expand(X)

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X,
            sampler=self.sampler,
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            sampler=self.inner_sampler,
            project=getattr(self, "project", None),
        )

        from botorch.generation.gen import gen_candidates_scipy

        # optimize the inner problem
        from botorch.optim.initializers import gen_value_function_initial_conditions

        initial_conditions = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=kwargs.get("num_restarts", 20),
            raw_samples=kwargs.get("raw_samples", 1024),
            current_model=self.model,
            options={**kwargs.get("options", {}), **kwargs.get("scipy_options", {})},
        )
        # initial_conditions shape: num_restarts x num_fantasies x n x q x d.

        _, values = gen_candidates_scipy(
            initial_conditions=initial_conditions,
            acquisition_function=value_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options=kwargs.get("scipy_options"),
            use_parallel_mode=False,
        )
        # get the maximizer for each batch
        values, _ = torch.max(values, dim=0)
        if self.current_value is not None:
            values = values - self.current_value
        # NOTE: using getattr to cover both no-attribute with qKG and None with qMFKG
        if getattr(self, "cost_aware_utility", None) is not None:
            values = self.cost_aware_utility(
                X=X, deltas=values, sampler=self.cost_sampler
            )
        # return average over the fantasy samples
        return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies, :]


class qMultiFidelityKnowledgeGradient(qKnowledgeGradient):
    r"""Batch Knowledge Gradient for multi-fidelity optimization.

    A version of `qKnowledgeGradient` that supports multi-fidelity optimization
    via a `CostAwareUtility` and the `project` and `expand` operators. If none
    of these are set, this acquisition function reduces to `qKnowledgeGradient`.
    Through `valfunc_cls` and `valfunc_argfac`, this can be changed into a custom
    multi-fidelity acquisition function (it is only KG if the terminal value is
    computed using a posterior mean).
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: int | None = 64,
        sampler: MCSampler | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        inner_sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
        current_value: Tensor | None = None,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] = lambda X: X,
        valfunc_cls: type[AcquisitionFunction] | None = None,
        valfunc_argfac: Callable[[Model], dict[str, Any]] | None = None,
    ) -> None:
        r"""Multi-Fidelity q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None`, then the analytic posterior mean is used. Otherwise, the
                objective is MC-evaluated (using inner_sampler).
            posterior_transform: An optional PosteriorTransform. If given, this
                transforms the posterior before evaluation. If `objective is None`,
                then the analytic posterior mean of the transformed posterior is
                used. If `objective` is given, the `inner_sampler` is used to draw
                samples from the transformed posterior, which are then evaluated under
                the `objective`.
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            project: A callable mapping a `batch_shape x q x d` tensor of design
                points to a tensor with shape `batch_shape x q_term x d` projected
                to the desired target set (e.g. the target fidelities in case of
                multi-fidelity optimization). For the basic case, `q_term = q`.
            expand: A callable mapping a `batch_shape x q x d` input tensor to
                a `batch_shape x (q + q_e)' x d`-dim output tensor, where the
                `q_e` additional points in each q-batch correspond to
                additional ("trace") observations.
            valfunc_cls: An acquisition function class to be used as the terminal
                value function.
            valfunc_argfac: An argument factory, i.e. callable that maps a `Model`
                to a dictionary of kwargs for the terminal value function (e.g.
                `best_f` for `ExpectedImprovement`).
        """
        if current_value is None and cost_aware_utility is not None:
            raise UnsupportedError(
                "Cost-aware KG requires current_value to be specified."
            )
        super().__init__(
            model=model,
            num_fantasies=num_fantasies,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            inner_sampler=inner_sampler,
            X_pending=X_pending,
            current_value=current_value,
        )
        self.cost_aware_utility = cost_aware_utility
        self.project = project
        self.expand = expand
        self._cost_sampler = None
        self.valfunc_cls = valfunc_cls
        self.valfunc_argfac = valfunc_argfac

    @property
    def cost_sampler(self):
        if self._cost_sampler is None:
            # Note: Using the deepcopy here is essential. Removing this poses a
            # problem if the base model and the cost model have a different number
            # of outputs or test points (this would be caused by expand), as this
            # would trigger re-sampling the base samples in the fantasy sampler.
            # By cloning the sampler here, the right thing will happen if the
            # the sizes are compatible, if they are not this will result in
            # samples being drawn using different base samples, but it will at
            # least avoid changing state of the fantasy sampler.
            self._cost_sampler = deepcopy(self.sampler)
        return self._cost_sampler

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiFidelityKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

                In addition, `X` may be augmented with fidelity parameters as
                part of thee `d`-dimension. Projecting fidelities to the target
                fidelity is handled by `project`.

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_eval = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )
        else:
            X_eval = X_actual

        # construct the fantasy model of shape `num_fantasies x b`
        # expand X (to potentially add trace observations)
        fantasy_model = self.model.fantasize(
            X=self.expand(X_eval),
            sampler=self.sampler,
        )
        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            sampler=self.inner_sampler,
            project=self.project,
            valfunc_cls=self.valfunc_cls,
            valfunc_argfac=self.valfunc_argfac,
        )

        # make sure to propagate gradients to the fantasy model train inputs
        # project the fantasy points
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        if self.cost_aware_utility is not None:
            values = self.cost_aware_utility(
                X=X_actual, deltas=values, sampler=self.cost_sampler
            )

        # return average over the fantasy samples
        return values.mean(dim=0)


class ProjectedAcquisitionFunction(AcquisitionFunction):
    r"""
    Defines a wrapper around  an `AcquisitionFunction` that incorporates the project
    operator. Typically used to handle value functions in look-ahead methods.
    """

    def __init__(
        self,
        base_value_function: AcquisitionFunction,
        project: Callable[[Tensor], Tensor],
    ) -> None:
        r"""
        Args:
            base_value_function: The wrapped `AcquisitionFunction`.
            project: A callable mapping a `batch_shape x q x d` tensor of design
                points to a tensor with shape `batch_shape x q_term x d` projected
                to the desired target set (e.g. the target fidelities in case of
                multi-fidelity optimization). For the basic case, `q_term = q`.
        """
        super().__init__(base_value_function.model)
        self.base_value_function = base_value_function
        self.project = project
        self.objective = getattr(base_value_function, "objective", None)
        self.posterior_transform = base_value_function.posterior_transform
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))


def _get_value_function(
    model: Model,
    objective: MCAcquisitionObjective | None = None,
    posterior_transform: PosteriorTransform | None = None,
    sampler: MCSampler | None = None,
    project: Callable[[Tensor], Tensor] | None = None,
    valfunc_cls: type[AcquisitionFunction] | None = None,
    valfunc_argfac: Callable[[Model], dict[str, Any]] | None = None,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function)."""
    if valfunc_cls is not None:
        common_kwargs: dict[str, Any] = {
            "model": model,
            "posterior_transform": posterior_transform,
        }
        if issubclass(valfunc_cls, MCAcquisitionFunction):
            common_kwargs["sampler"] = sampler
            common_kwargs["objective"] = objective
        kwargs = valfunc_argfac(model=model) if valfunc_argfac is not None else {}
        base_value_function = valfunc_cls(**common_kwargs, **kwargs)
    else:
        if objective is not None:
            base_value_function = qSimpleRegret(
                model=model,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
            )
        else:
            base_value_function = PosteriorMean(
                model=model, posterior_transform=posterior_transform
            )

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )


def _split_fantasy_points(X: Tensor, n_f: int) -> tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x num_fantasies x d, needs to be num_fantasies x b x 1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.
    # b x num_fantasies x d --> num_fantasies x b x d
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    # num_fantasies x b x 1 x d
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies
