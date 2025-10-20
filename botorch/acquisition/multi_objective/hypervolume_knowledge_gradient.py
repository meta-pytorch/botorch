# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
The hypervolume knowledge gradient acquisition function (HVKG).

References:

.. [Daulton2023hvkg]
    S. Daulton, M. Balandat, E. Bakshy. Hypervolume Knowledge Gradient: A
    Lookahead Approach for Multi-Objective Bayesian Optimization with Partial
    Information. Proceedings of the 40th International Conference on Machine
    Learning, 2023.
"""

import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from botorch import settings
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)

from botorch.acquisition.cost_aware import CostAwareUtility
from botorch.acquisition.decoupled import DecoupledAcquisitionFunction
from botorch.acquisition.knowledge_gradient import ProjectedAcquisitionFunction
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.exceptions.warnings import NumericsWarning
from botorch.models.deterministic import PosteriorMeanModel
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.safe_math import logdiffexp, logmeanexp
from botorch.utils.transforms import (
    average_over_ensemble_models,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class qHypervolumeKnowledgeGradient(
    DecoupledAcquisitionFunction,
    MultiObjectiveMCAcquisitionFunction,
    OneShotAcquisitionFunction,
):
    """Batch Hypervolume Knowledge Gradient using one-shot optimization.

    The hypervolume knowledge gradient seeks to maximize the difference in
    hypervolume of the hypervolume-maximizing set of a fixed size after
    conditioning the unknown observation(s) that would be recevied if X where
    evalauted. See [Daulton2023hvkg]_ for details.

    This computes the batch Hypervolume Knowledge Gradient using fantasies for
    the outer expectation and MC-sampling for the inner expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models (Note this is
    `N x N_pareto` optimal designs). For a fixed number of fantasies, all points
    in `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        num_fantasies: int = 8,
        num_pareto: int = 10,
        sampler: ListSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        inner_sampler: MCSampler | None = None,
        X_evaluation_mask: list[Tensor] | None = None,
        X_pending: Tensor | None = None,
        X_pending_evaluation_mask: Tensor | None = None,
        current_value: Tensor | None = None,
        use_posterior_mean: bool = True,
        cost_aware_utility: CostAwareUtility | None = None,
        log: bool = False,
    ) -> None:
        r"""q-Hypervolume Knowledge Gradient.

        Args:
            model: A fitted model. Must support fantasizing.
            ref_point: A `m`-dim tensor containing the reference point.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            num_pareto: The number of pareto optimal designs to consider.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified. The optimization performance
                does not seem particularly sensitive to the number of fantasies.
                As the number of fantasies increases, the estimation of the
                expectation over fantasies becomes more accurate, but the one-
                shot optimization problem gets harder as there are more "fantasy"
                designs that need to be optimized.
            objective: The objective under which the samples are evaluated. If
                `None`, then the analytic posterior mean is used. Otherwise, the
                objective is MC-evaluated (using inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None`.
            X_evaluation_mask: A `q x m`-dim tensor of booleans indicating which
                objective(s) each of the `q` points should be evaluated on.
            X_pending: A `n' x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            X_pending_evaluation_mask: A `n' x m`-dim tensor of booleans indicating
                which objective(s) each of the `n'` pending points are being
                evaluated on.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`. If pending points are used,
                this should be the current value under the fantasy model
                conditioned on the pending points so that the incremental KG value
                from the new candidates (not pending points) is used.
            use_posterior_mean: If true, optimize the hypervolume of the posterior
                mean, otherwise optimize the expected hypervolume. See
                [Daulton2023hvkg]_ for details.
            cost_aware_utility: A CostAwareUtility specifying the cost function for
                evaluating the `X` on the objectives indicated by `evaluation_mask`.
            log: If True, then returns the log of the HVKG value. If True, then it
                expects current_value to be in log-space and cost_aware_utility to
                output log utilities.
        """
        if sampler is None:
            # base samples should be fixed for joint optimization over X, X_fantasies
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
                for _ in range(model.num_outputs)
            ]
            sampler = ListSampler(*samplers)
        else:
            sample_shape = sampler.samplers[0].sample_shape
            if sample_shape != torch.Size([num_fantasies]):
                raise ValueError(
                    f"The sampler shape must match num_fantasies={num_fantasies}."
                )
        super().__init__(model=model, X_evaluation_mask=X_evaluation_mask)

        if inner_sampler is None:
            inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
        if current_value is None and cost_aware_utility is not None:
            raise UnsupportedError(
                "Cost-aware HVKG requires current_value to be specified."
            )
        self.register_buffer("ref_point", ref_point)
        self.sampler = sampler
        self.objective = objective
        self.set_X_pending(
            X_pending=X_pending, X_pending_evaluation_mask=X_pending_evaluation_mask
        )
        self.inner_sampler = inner_sampler
        self.num_fantasies = num_fantasies
        self.num_pareto = num_pareto
        self.num_pseudo_points = num_fantasies * num_pareto
        self.current_value = current_value
        self.use_posterior_mean = use_posterior_mean
        self.cost_aware_utility = cost_aware_utility
        self._cost_sampler = None

        self._log = log

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
        X_actual, X_fantasies = _split_hvkg_fantasy_points(
            X=X, n_f=self.num_fantasies, num_pareto=self.num_pareto
        )
        q = X_actual.shape[-2]

        # construct evaluation_mask
        evaluation_mask = self.construct_evaluation_mask(X=X_actual)
        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # Construct the fantasy model of shape `num_fantasies x b`
        # Note: For the decoupled, cost-aware (e.g. not async) setting, we
        # technically want to make sure to copy the base samples here, so
        # that the same fantasies are used for X_pending on the left and
        # right of the KG terms.
        fantasy_model = self.model.fantasize(
            X=X_actual,
            sampler=self.sampler,
            evaluation_mask=evaluation_mask,
        )

        # get the value function
        value_function = _get_hv_value_function(
            model=fantasy_model,
            ref_point=self.ref_point,
            objective=self.objective,
            sampler=self.inner_sampler,
            use_posterior_mean=self.use_posterior_mean,
            log=self._log,
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            # X_fantasies is num_pseudo_points x batch_shape x 1 x d
            # Reshape it into num_fantasies x batch_shape x num_pareto x d
            shape = torch.Size(
                [
                    self.num_fantasies,
                    *X_fantasies.shape[1:-2],
                    self.num_pareto,
                    X_fantasies.shape[-1],
                ]
            )
            values = value_function(X=X_fantasies.reshape(shape))  # num_fantasies x b

        if self.current_value is not None:
            if self._log:
                values = logdiffexp(self.current_value, values)
            else:
                values = values - self.current_value

        if self.cost_aware_utility is not None:
            if self._log:
                _check_log_utilities(self.cost_aware_utility)
            values = self.cost_aware_utility(
                # exclude pending points
                X=X_actual[..., :q, :],
                deltas=values,
                sampler=self.cost_sampler,
                X_evaluation_mask=self.X_evaluation_mask,
            )

        # return average over the fantasy samples
        if self._log:
            return logmeanexp(values, dim=0)
        else:
            return values.mean(dim=0)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_pseudo_points

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_pseudo_points, :]


class qMultiFidelityHypervolumeKnowledgeGradient(qHypervolumeKnowledgeGradient):
    r"""Batch Hypervolume Knowledge Gradient for multi-fidelity optimization.

    See [Daulton2023hvkg]_ for details.

    A version of `qHypervolumeKnowledgeGradient` that supports multi-fidelity
    optimization via a `CostAwareUtility` and the `project` and `expand`
    operators. If none of these are set, this acquisition function reduces to
    `qHypervolumeKnowledgeGradient`. Through `valfunc_cls` and `valfunc_argfac`,
    this can be changed into a custom multi-fidelity acquisition function.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        target_fidelities: dict[int, float],
        num_fantasies: int = 8,
        num_pareto: int = 10,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        inner_sampler: MCSampler | None = None,
        X_pending: Tensor | None = None,
        X_evaluation_mask: Tensor | None = None,
        X_pending_evaluation_mask: Tensor | None = None,
        current_value: Tensor | None = None,
        cost_aware_utility: CostAwareUtility | None = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        valfunc_cls: type[AcquisitionFunction] | None = None,
        valfunc_argfac: Callable[[Model], dict[str, Any]] | None = None,
        use_posterior_mean: bool = True,
        log: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Multi-Fidelity q-Knowledge Gradient (one-shot optimization).

        Args:
            model: A fitted model. Must support fantasizing.
            ref_point: A `m`-dim tensor containing the reference point.
            num_fantasies: The number of fantasy points to use. More fantasy
                points result in a better approximation, at the expense of
                memory and wall time. Unused if `sampler` is specified.
            num_pareto: The number of pareto optimal designs to consider.
            sampler: The sampler used to sample fantasy observations. Optional
                if `num_fantasies` is specified.
            objective: The objective under which the samples are evaluated. If
                `None`, then the analytic posterior mean is used. Otherwise, the
                objective is MC-evaluated (using inner_sampler).
            inner_sampler: The sampler used for inner sampling. Ignored if the
                objective is `None`.
            X_evaluation_mask: A `q x m`-dim tensor of booleans indicating which
                objective(s) each of the `q` points should be evaluated on.
            X_pending: A `n' x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.
            X_pending_evaluation_mask: A `n' x m`-dim tensor of booleans indicating
                which objective(s) each of the `n'` pending points are being
                evaluated on.
            current_value: The current value, i.e. the expected best objective
                given the observed points `D`. If omitted, forward will not
                return the actual KG value, but the expected best objective
                given the data set `D u X`. If pending points are used,
                this should be the current value under the fantasy model
                conditioned on the pending points so that the incremental KG value
                from the new candidates (not pending points) is used.
            use_posterior_mean: A boolean indicating whether to use the to optimize
                the hypervolume of the posterior mean or whether to optimize the
                expected hypervolume. See [Daulton2023hvkg]_ for details.
            cost_aware_utility: A CostAwareUtility specifying the cost function for
                evaluating the `X` on the objectives indicated by `evaluation_mask`.
            project: A callable mapping a `batch_shape x q x d` tensor of design
                points to a tensor with shape `batch_shape x q_term x d` projected
                to the desired target set (e.g. the target fidelities in case of
                multi-fidelity optimization). For the basic case, `q_term = q`.
            valfunc_cls: An acquisition function class to be used as the terminal
                value function.
            valfunc_argfac: An argument factory, i.e. callable that maps a `Model`
                to a dictionary of kwargs for the terminal value function (e.g.
                `best_f` for `ExpectedImprovement`).
            log: If True, then returns the log of the HVKG value. If True, then it
                expects current_value to be in log-space and cost_aware_utility to
                output log utilities.
        """

        super().__init__(
            model=model,
            ref_point=ref_point,
            num_fantasies=num_fantasies,
            num_pareto=num_pareto,
            sampler=sampler,
            objective=objective,
            inner_sampler=inner_sampler,
            X_evaluation_mask=X_evaluation_mask,
            X_pending=X_pending,
            X_pending_evaluation_mask=X_pending_evaluation_mask,
            current_value=current_value,
            use_posterior_mean=use_posterior_mean,
            cost_aware_utility=cost_aware_utility,
            log=log,
        )
        self.project = project
        if kwargs.get("expand") is not None:
            raise NotImplementedError(
                "Trace observations are not currently supported "
                "by `qMultiFidelityHypervolumeKnowledgeGradient`."
            )
        self.expand = lambda X: X
        self.valfunc_cls = valfunc_cls
        self.valfunc_argfac = valfunc_argfac
        self.target_fidelities = target_fidelities

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

                In addition, `X` may be augmented with fidelity parameteres as
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
        X_actual, X_fantasies = _split_hvkg_fantasy_points(
            X=X, n_f=self.num_fantasies, num_pareto=self.num_pareto
        )
        q = X_actual.shape[-2]

        # construct evaluation_mask
        evaluation_mask = self.construct_evaluation_mask(X=X_actual)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual,
            sampler=self.sampler,
            evaluation_mask=evaluation_mask,
        )
        # get the value function
        value_function = _get_hv_value_function(
            model=fantasy_model,
            ref_point=self.ref_point,
            objective=self.objective,
            sampler=self.inner_sampler,
            project=self.project,
            valfunc_cls=self.valfunc_cls,
            valfunc_argfac=self.valfunc_argfac,
            use_posterior_mean=self.use_posterior_mean,
            log=self._log,
        )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            # X_fantasies is num_pseudo_points  x batch_shape x 1 x d
            # Reshape it into num_fantasies x batch_shape x num_pareto x d
            shape = torch.Size(
                [
                    self.num_fantasies,
                    *X_fantasies.shape[1:-2],
                    self.num_pareto,
                    X_fantasies.shape[-1],
                ]
            )
            values = value_function(X=X_fantasies.reshape(shape))  # num_fantasies x b
        if self.current_value is not None:
            if self._log:
                # Assumes current value is in log-space
                values = logdiffexp(self.current_value, values)
            else:
                values = values - self.current_value

        if self.cost_aware_utility is not None:
            if self._log:
                _check_log_utilities(self.cost_aware_utility)
            values = self.cost_aware_utility(
                # exclude pending points
                X=X_actual[..., :q, :],
                deltas=values,
                sampler=self.cost_sampler,
                X_evaluation_mask=self.X_evaluation_mask,
            )

        # return average over the fantasy samples
        return logmeanexp(values, dim=0) if self._log else values.mean(dim=0)


def _get_hv_value_function(
    model: Model,
    ref_point: Tensor,
    objective: MCMultiOutputObjective | None = None,
    sampler: MCSampler | None = None,
    project: Callable[[Tensor], Tensor] | None = None,
    valfunc_cls: type[AcquisitionFunction] | None = None,
    valfunc_argfac: Callable[[Model], dict[str, Any]] | None = None,
    use_posterior_mean: bool = False,
    log: bool = False,
) -> AcquisitionFunction:
    r"""Construct value function (i.e. inner acquisition function).
    This is a method for computing hypervolume.
    """
    if use_posterior_mean:
        model = PosteriorMeanModel(model=model)
        sampler = StochasticSampler(sample_shape=torch.Size([1]))  # dummy sampler
    with warnings.catch_warnings():
        warnings.filterwarnings(
            message="qExpectedHypervolumeImprovement has known",
            action="ignore",
            category=NumericsWarning,
        )

        base_value_function_class = (
            qLogExpectedHypervolumeImprovement
            if log
            else qExpectedHypervolumeImprovement
        )
        base_value_function = base_value_function_class(
            model=model,
            ref_point=ref_point,
            partitioning=FastNondominatedPartitioning(
                ref_point=ref_point,
                Y=torch.empty(
                    (0, ref_point.shape[0]),
                    dtype=ref_point.dtype,
                    device=ref_point.device,
                ),
            ),  # create empty partitioning
            sampler=sampler,
            objective=objective,
        )
    # ProjectedAcquisitionFunction requires this
    base_value_function.posterior_transform = None

    if project is None:
        return base_value_function
    else:
        return ProjectedAcquisitionFunction(
            base_value_function=base_value_function,
            project=project,
        )


def _split_hvkg_fantasy_points(
    X: Tensor, n_f: int, num_pareto: int
) -> tuple[Tensor, Tensor]:
    r"""Split a one-shot HV-KGoptimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f*num_pareto) x d`-dim tensor of actual
            and fantasy points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x num_pareto x d`-dim tensor `X_fantasies` of
            fantasy points, where `X_fantasies[i, batch_idx]` is the i-th
            fantasy point associated with the batch indexed by `batch_idx`.
    """
    if n_f * num_pareto > X.size(-2):
        raise ValueError(
            f"`n_f*num_pareto` ({n_f * num_pareto}) must be less than"
            f" the `q`-batch dimension of `X` ({X.size(-2)})."
        )
    split_sizes = [X.size(-2) - n_f * num_pareto, n_f * num_pareto]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x n_f * num_pareto x d, needs to be n_f x b x num_pareto x d
    # reshape into num_fantasies x b x num_pareto x d
    new_shape = torch.Size(
        [n_f, *X_fantasies.shape[:-2], num_pareto, X_fantasies.shape[-1]]
    )
    X_fantasies = X_fantasies.reshape(new_shape)
    # n_f x b x num_pareto x d
    return X_actual, X_fantasies


def _check_log_utilities(cost_aware_utility: CostAwareUtility) -> None:
    if not hasattr(cost_aware_utility, "_log") or not cost_aware_utility._log:
        raise BotorchError(
            "Cost-aware HVKG has _log=True and requires cost_aware_utility "
            "to output log utilities."
        )
