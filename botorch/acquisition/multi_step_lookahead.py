#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A general implementation of multi-step look-ahead acquisition function with configurable
value functions. See [Jiang2020multistep]_.

.. [Jiang2020multistep]
    S. Jiang, D. R. Jiang, M. Balandat, B. Karrer, J. Gardner, and R. Garnett.
    Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees.
    In Advances in Neural Information Processing Systems 33, 2020.

"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, PosteriorMean
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.optim.initializers import initialize_q_batch
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    average_over_ensemble_models,
    match_batch_shape,
    t_batch_mode_transform,
    unnormalize,
)
from torch import Size, Tensor
from torch.distributions import Beta
from torch.nn import ModuleList


TAcqfArgConstructor = Callable[[Model, Tensor], dict[str, Any]]


class qMultiStepLookahead(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""MC-based batch Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        batch_sizes: list[int],
        num_fantasies: list[int] | None = None,
        samplers: list[MCSampler] | None = None,
        valfunc_cls: list[type[AcquisitionFunction] | None] | None = None,
        valfunc_argfacs: list[TAcqfArgConstructor | None] | None = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        inner_mc_samples: list[int] | None = None,
        X_pending: Tensor | None = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""q-Multi-Step Look-Ahead (one-shot optimization).

        Performs a ``k``-step lookahead by means of repeated fantasizing.

        Allows to specify the stage value functions by passing the respective class
        objects via the ``valfunc_cls`` list. Optionally,
        ``valfunc_argfacs`` takes a list of callables that generate
        additional kwargs for these constructors. By default,
        ``valfunc_cls`` will be chosen as ``[None, ..., None, PosteriorMean]``,
        which corresponds to the (parallel) multi-step KnowledgeGradient.
        If, in addition,
        ``k=1`` and ``q_1 = 1``, this reduces to the classic Knowledge Gradient.

        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!

        Args:
            model: A fitted model.
            batch_sizes: A list ``[q_1, ..., q_k]`` containing the batch sizes for the
                ``k`` look-ahead steps.
            num_fantasies: A list ``[f_1, ..., f_k]`` containing the number of fantasy
                points to use for the ``k`` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            valfunc_cls: A list of ``k + 1`` acquisition function classes
                to be used as the (stage + terminal) value functions. Each
                element (except for the last one) can be ``None``, in which case
                a zero stage value is assumed for the respective stage. If
                ``None``, this defaults to ``[None, ..., None, PosteriorMean]``
            valfunc_argfacs: A list of ``k + 1`` "argument factories",
                i.e. callables that map a ``Model`` and input tensor ``X`` to a
                dictionary of kwargs for the respective stage value function
                constructor (e.g. ``best_f`` for ``ExpectedImprovement``). If
                None, only the standard (``model``, ``sampler`` and ``objective``)
                kwargs will be used.
            objective: The objective under which the output is evaluated.
                If ``None``, use the model output (requires a single-output
                model or a posterior transform). Otherwise the objective is
                MC-evaluated (using ``inner_sampler``).
            posterior_transform: An optional PosteriorTransform. If given,
                this transforms the posterior before evaluation. If ``objective
                is None``, then the output of the transformed posterior is used.
                If ``objective`` is given, the ``inner_sampler`` is used to draw
                samples from the transformed posterior, which are then evaluated
                under the ``objective``.
            inner_mc_samples: A list ``[n_0, ..., n_k]`` containing the
                number of MC samples to be used for evaluating the stage value
                function. Ignored if the objective is ``None``.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have
                points that have been submitted for function evaluation but have
                not yet been evaluated. Concatenated into ``X`` upon forward
                call. Copied and set to have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        if objective is not None and not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "`qMultiStepLookahead` got a non-MC `objective`. This is not supported."
                " Use `posterior_transform` and `objective=None` instead."
            )

        super(MCAcquisitionFunction, self).__init__(model=model)
        self.batch_sizes = batch_sizes
        if not ((num_fantasies is None) ^ (samplers is None)):
            raise UnsupportedError(
                "qMultiStepLookahead requires exactly one of `num_fantasies` or "
                "`samplers` as arguments."
            )
        if samplers is None:
            # If collapse_fantasy_base_samples is False, the ``batch_range_override``
            # is set on the samplers during the forward call.
            samplers: list[MCSampler] = [
                SobolQMCNormalSampler(sample_shape=torch.Size([nf]))
                for nf in num_fantasies
            ]
        else:
            num_fantasies = [sampler.sample_shape[0] for sampler in samplers]
        self.num_fantasies = num_fantasies
        # By default do not use stage values and use PosteriorMean as terminal value
        # function (= multi-step KG)
        if valfunc_cls is None:
            valfunc_cls = [None for _ in num_fantasies] + [PosteriorMean]
        if inner_mc_samples is None:
            inner_mc_samples = [None] * (1 + len(num_fantasies))
        # TODO: Allow passing in inner samplers directly
        inner_samplers = _construct_inner_samplers(
            batch_sizes=batch_sizes,
            valfunc_cls=valfunc_cls,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
        )
        if valfunc_argfacs is None:
            valfunc_argfacs = [None] * (1 + len(batch_sizes))

        self.objective = objective
        self.posterior_transform = posterior_transform
        self.set_X_pending(X_pending)
        self.samplers = ModuleList(samplers)
        self.inner_samplers = ModuleList(inner_samplers)
        self._valfunc_cls = valfunc_cls
        self._valfunc_argfacs = valfunc_argfacs
        self._collapse_fantasy_base_samples = collapse_fantasy_base_samples

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for each
                batch, where ``q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``. Here ``q_i``
                is the number of candidates jointly considered in look-ahead step
                ``i``, and ``f_i`` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape ``batch_shape``.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        return _step(
            model=self.model,
            Xs=Xs,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            running_val=None,
        )

    @property
    def _num_auxiliary(self) -> int:
        r"""Number of auxiliary variables in the q-batch dimension.

        Returns:
             ``q_aux`` s.t. ``q + q_aux = augmented_q_batch_size``
        """
        return np.dot(self.batch_sizes, np.cumprod(self.num_fantasies)).item()

    def _set_samplers_batch_range(self, batch_shape: Size) -> None:
        r"""Set batch_range on samplers.

        Args:
            batch_shape: The batch shape of the input tensor ``X``.
        """
        tbatch_dim_start = -2 - len(batch_shape)
        for s in self.samplers:
            s.batch_range_override = (tbatch_dim_start, -2)

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions): ``q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``
        """
        return q + self._num_auxiliary

    def get_split_shapes(self, X: Tensor) -> tuple[Size, list[Size], list[int]]:
        r"""Get the split shapes from X.

        Args:
            X: A ``batch_shape x q_aug x d``-dim tensor including fantasy points.

        Returns:
            A 3-tuple ``(batch_shape, shapes, sizes)``, where
            ``shape[i] = f_i x .... x f_1 x batch_shape x q_i x d`` and
            ``size[i] = f_i * ... f_1 * q_i``.
        """
        batch_shape, (q_aug, d) = X.shape[:-2], X.shape[-2:]
        q = q_aug - self._num_auxiliary
        batch_sizes = [q] + self.batch_sizes
        # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
        shapes = [
            torch.Size(self.num_fantasies[:i][::-1] + [*batch_shape, q_i, d])
            for i, q_i in enumerate(batch_sizes)
        ]
        # Each X_i in the split X has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        sizes = [s[: (-2 - len(batch_shape))].numel() * s[-2] for s in shapes]
        return batch_shape, shapes, sizes

    def get_multi_step_tree_input_representation(self, X: Tensor) -> list[Tensor]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for each
                batch, where ``q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``. Here ``q_i``
                is the number of candidates jointly considered in look-ahead step
                ``i``, and ``f_i`` is respective number of fantasies.

        Returns:
            A list ``[X_j, ..., X_k]`` of tensors, where ``X_i`` has shape
            ``f_i x .... x f_1 x batch_shape x q_i x d``.

        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        # concatenate in pending points
        if self.X_pending is not None:
            X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)

        return [X0] + Xother

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for
                each batch, where ``q' = q + f_1 q_1 + f_2 f_1 q_2 + ...``.

        Returns:
            A ``batch_shape x q x d``-dim Tensor with ``q`` design points for
            each batch.
        """
        return X_full[..., : -self._num_auxiliary, :]

    def get_induced_fantasy_model(self, X: Tensor) -> Model:
        r"""Fantasy model induced by X.

        Args:
            X: A ``batch_shape x q' x d``-dim Tensor with ``q'`` design points for each
                batch, where ``q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...``. Here ``q_i``
                is the number of candidates jointly considered in look-ahead step
                ``i``, and ``f_i`` is respective number of fantasies.

        Returns:
            The fantasy model induced by X.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        return _get_induced_fantasy_model(
            model=self.model, Xs=Xs, samplers=self.samplers
        )


def _step(
    model: Model,
    Xs: list[Tensor],
    samplers: list[MCSampler | None],
    valfunc_cls: list[type[AcquisitionFunction] | None],
    valfunc_argfacs: list[TAcqfArgConstructor | None],
    inner_samplers: list[MCSampler | None],
    objective: MCAcquisitionObjective,
    posterior_transform: PosteriorTransform | None,
    running_val: Tensor | None = None,
    sample_weights: Tensor | None = None,
    step_index: int = 0,
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.

    Helper function computing the "value-to-go" of a multi-step lookahead scheme.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be
            possible to evaluate the model's posterior at ``Xs[0]``.
        Xs: A list ``[X_j, ..., X_k]`` of tensors, where ``X_i`` has shape
            ``f_i x .... x f_1 x batch_shape x q_i x d``.
        samplers: A list of ``k - j`` samplers, such that the number of
            samples of sampler ``i`` is ``f_i``. The last element of this list
            is considered the "inner sampler", which is used for evaluating
            the objective in case it is an MCAcquisitionObjective.
        valfunc_cls: A list of acquisition function class to be used as the
            (stage + terminal) value functions. Each element (except for the
            last one) can be ``None``, in which case a zero stage value is
            assumed for the respective stage.
        valfunc_argfacs: A list of callables that map a ``Model`` and input
            tensor ``X`` to a dictionary of kwargs for the respective stage
            value function constructor. If ``None``, only the standard
            ``model``, ``sampler`` and ``objective`` kwargs will be used.
        inner_samplers: A list of ``MCSampler`` objects, each to be used in the stage
            value function at the corresponding index.
        objective: The MCAcquisitionObjective under which the model output is evaluated.
        posterior_transform: A PosteriorTransform. Used to transform the posterior
            before sampling / evaluating the model output.
        running_val: As ``batch_shape``-dim tensor containing the current running value.
        sample_weights: A tensor of shape ``f_i x .... x f_1 x batch_shape`` when called
            in the ``i``-th step by which to weight the stage value samples. Used in
            conjunction with Gauss-Hermite integration or importance sampling. Assumed
            to be ``None`` in the initial step (when ``step_index=0``).
        step_index: The index of the look-ahead step. ``step_index=0`` indicates the
            initial step.

    Returns:
        A ``b``-dim tensor containing the multi-step value of the design ``X``.
    """
    X = Xs[0]
    if sample_weights is None:  # only happens in the initial step
        sample_weights = torch.ones(*X.shape[:-2], device=X.device, dtype=X.dtype)

    # compute stage value
    stage_val = _compute_stage_value(
        model=model,
        valfunc_cls=valfunc_cls[0],
        X=X,
        objective=objective,
        posterior_transform=posterior_transform,
        inner_sampler=inner_samplers[0],
        arg_fac=valfunc_argfacs[0],
    )
    if stage_val is not None:  # update running value
        # if not None, running_val has shape f_{i-1} x ... x f_1 x batch_shape
        # stage_val has shape f_i x ... x f_1 x batch_shape

        # this sum will add a dimension to running_val so that
        # updated running_val has shape f_i x ... x f_1 x batch_shape
        running_val = stage_val if running_val is None else running_val + stage_val

    # base case: no more fantasizing, return value
    if len(Xs) == 1:
        # compute weighted average over all leaf nodes of the tree
        batch_shape = running_val.shape[step_index:]
        # expand sample weights to make sure it is the same shape as running_val,
        # because we need to take a sum over sample weights for computing the
        # weighted average
        sample_weights = sample_weights.expand(running_val.shape)
        return (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    prop_grads = step_index > 0  # need to propagate gradients for steps > 0
    fantasy_model = model.fantasize(
        X=X, sampler=samplers[0], propagate_grads=prop_grads
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[0]
    )

    return _step(
        model=fantasy_model,
        Xs=Xs[1:],
        samplers=samplers[1:],
        valfunc_cls=valfunc_cls[1:],
        valfunc_argfacs=valfunc_argfacs[1:],
        inner_samplers=inner_samplers[1:],
        objective=objective,
        posterior_transform=posterior_transform,
        sample_weights=sample_weights,
        running_val=running_val,
        step_index=step_index + 1,
    )


def _compute_stage_value(
    model: Model,
    valfunc_cls: type[AcquisitionFunction] | None,
    X: Tensor,
    objective: MCAcquisitionObjective,
    posterior_transform: PosteriorTransform | None,
    inner_sampler: MCSampler | None = None,
    arg_fac: TAcqfArgConstructor | None = None,
) -> Tensor | None:
    r"""Compute the stage value of a multi-step look-ahead policy.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be
            possible to evaluate the model's posterior at ``Xs[0]``.
        valfunc_cls: The acquisition function class to be used as the stage
            value functions. If ``None``, a zero stage value is assumed
            (returns ``None``)
        X: A tensor with shape ``f_i x .... x f_1 x batch_shape x q_i x d``
            when called in the ``i``-th step.
        objective: The MCAcquisitionObjective under which the model output
            is evaluated.
        posterior_transform: A PosteriorTransform.
        inner_sampler: An ``MCSampler`` object to be used in the stage value
            function. Can be ``None`` for analytic acquisition functions or
            when using the default sampler of the acquisition function class.
        arg_fac: A callable mapping a ``Model`` and the input tensor ``X``
            to a dictionary of kwargs for the stage value function
            constructor. If ``None``, only the standard ``model``, ``sampler``
            and ``objective`` kwargs will be used.

    Returns:
        A ``f_i x ... x f_1 x batch_shape``-dim tensor of stage values, or ``None``
        (= zero stage value).
    """
    if valfunc_cls is None:
        return None
    common_kwargs: dict[str, Any] = {
        "model": model,
        "posterior_transform": posterior_transform,
    }
    if issubclass(valfunc_cls, MCAcquisitionFunction):
        common_kwargs["sampler"] = inner_sampler
        common_kwargs["objective"] = objective
    kwargs = arg_fac(model=model, X=X) if arg_fac is not None else {}
    stage_val_func = valfunc_cls(**common_kwargs, **kwargs)
    # shape of stage_val is f_i x ... x f_1 x batch_shape
    stage_val = stage_val_func(X=X)
    return stage_val


def _construct_sample_weights(
    prev_weights: Tensor, sampler: MCSampler
) -> Tensor | None:
    r"""Iteratively construct tensor of sample weights for multi-step look-ahead.

    Args:
        prev_weights: A ``f_i x .... x f_1 x batch_shape`` tensor of previous sample
            weights.
        sampler: A ``MCSampler`` that may have sample weights as the ``base_weights``
            attribute. If the sampler does not have a ``base_weights`` attribute,
            samples are weighted uniformly.

    Returns:
        A ``f_{i+1} x .... x f_1 x batch_shape`` tensor of sample weights for the next
        step.
    """
    new_weights = getattr(sampler, "base_weights", None)  # TODO: generalize this
    if new_weights is None:
        # uniform weights
        nf = sampler.sample_shape[0]
        new_weights = torch.ones(
            nf, device=prev_weights.device, dtype=prev_weights.dtype
        )
    # reshape new_weights to be f_{i+1} x 1 x ... x 1
    new_weights = new_weights.view(-1, *(1 for _ in prev_weights.shape))
    # normalize new_weights to sum to 1.0
    new_weights = new_weights / new_weights.sum()
    return new_weights * prev_weights


def _construct_inner_samplers(
    batch_sizes: list[int],
    valfunc_cls: list[type[AcquisitionFunction] | None],
    inner_mc_samples: list[int | None],
    objective: MCAcquisitionObjective | None = None,
) -> list[MCSampler | None]:
    r"""Check validity of inputs and construct inner samplers.

    Helper function to be used internally for constructing inner samplers.

    Args:
        batch_sizes: A list ``[q_1, ..., q_k]`` containing the batch sizes for the
            ``k`` look-ahead steps.
        valfunc_cls: A list of ``k + 1`` acquisition function classes to be used as the
            (stage + terminal) value functions. Each element (except for the last one)
            can be ``None``, in which case a zero stage value is assumed for the
            respective stage.
        inner_mc_samples: A list ``[n_0, ..., n_k]`` containing the number of MC
            samples to be used for evaluating the stage value function. Ignored if
            the objective is ``None``.
        objective: The objective under which the output is evaluated. If ``None``, use
            the model output (requires a single-output model or a posterior transform).
            Otherwise the objective is MC-evaluated (using ``inner_sampler``).

    Returns:
        A list with ``k + 1`` elements that are either ``MCSampler``s or `None.
    """
    inner_samplers = []
    for q, vfc, mcs in zip([None] + batch_sizes, valfunc_cls, inner_mc_samples):
        if vfc is None:
            inner_samplers.append(None)
        elif vfc == qMultiStepLookahead:
            raise UnsupportedError(
                "qMultiStepLookahead not supported as a value function "
                "(I see what you did there, nice try...)."
            )
        elif issubclass(vfc, AnalyticAcquisitionFunction):
            if objective is not None:
                raise UnsupportedError(
                    "Only PosteriorTransforms are supported for analytic value "
                    f"functions. Received a {objective.__class__.__name__}."
                )
            # At this point, we don't know the initial q-batch size here
            if q is not None and q > 1:
                raise UnsupportedError(
                    "Only batch sizes of q=1 are supported for analytic value "
                    "functions."
                )
            if q is not None and mcs is not None:
                warnings.warn(
                    "inner_mc_samples is ignored for analytic acquisition functions",
                    BotorchWarning,
                    stacklevel=3,
                )
            inner_samplers.append(None)
        else:
            inner_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([32 if mcs is None else mcs])
            )
            inner_samplers.append(inner_sampler)
    return inner_samplers


def _get_induced_fantasy_model(
    model: Model, Xs: list[Tensor], samplers: list[MCSampler | None]
) -> Model:
    r"""Recursive computation of the fantasy model induced by an input tree.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be
            possible to evaluate the model's posterior at ``Xs[0]``.
        Xs: A list ``[X_j, ..., X_k]`` of tensors, where ``X_i`` has shape
            ``f_i x .... x f_1 x batch_shape x q_i x d``.
        samplers: A list of ``k - j`` samplers, such that the number of
            samples of sampler ``i`` is ``f_i``. The last element of this list
            is considered the "inner sampler", which is used for evaluating
            the objective in case it is an MCAcquisitionObjective.

    Returns:
        A Model obtained by iteratively fantasizing over the input tree ``Xs``.
    """
    if len(Xs) == 1:
        return model
    else:
        fantasy_model = model.fantasize(
            X=Xs[0],
            sampler=samplers[0],
        )

        return _get_induced_fantasy_model(
            model=fantasy_model, Xs=Xs[1:], samplers=samplers[1:]
        )


def warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Tensor,
) -> Tensor:
    r"""Warm-start initialization for multi-step look-ahead acquisition functions.

    For now uses the same q' as in ``full_optimizer``. TODO: allow different ``q``.

    Args:
        acq_function: A qMultiStepLookahead acquisition function.
        bounds: A ``2 x d`` tensor of lower and upper bounds for each column
            of features.
        num_restarts: The number of starting points for multistart
            acquisition function optimization.
        raw_samples: The number of raw samples to consider in the
            initialization heuristic.
        full_optimizer: The full tree of optimizers of the previous iteration
            of shape ``batch_shape x q' x d``. Typically obtained by passing
            ``return_best_only=False`` and ``return_full_tree=True`` into
            ``optimize_acqf``.

    Returns:
        A ``num_restarts x q' x d`` tensor for initial points for optimization.

    This is a very simple initialization heuristic.
    TODO: Use the observed values to identify the fantasy sub-tree that is closest to
    the observed value.
    """
    batch_shape, shapes, sizes = acq_function.get_split_shapes(full_optimizer)
    Xopts = torch.split(full_optimizer, sizes, dim=-2)
    tkwargs = {"device": Xopts[0].device, "dtype": Xopts[0].dtype}

    B = Beta(torch.ones(1, **tkwargs), 3 * torch.ones(1, **tkwargs))

    def mixin_layer(X: Tensor, bounds: Tensor, eta: float) -> Tensor:
        perturbations = unnormalize(B.sample(X.shape).squeeze(-1), bounds)
        return (1 - eta) * X + eta * perturbations

    def make_init_tree(Xopts: list[Tensor], bounds: Tensor, etas: Tensor) -> Tensor:
        Xtrs = [mixin_layer(X=X, bounds=bounds, eta=eta) for eta, X in zip(etas, Xopts)]
        return torch.cat(Xtrs, dim=-2)

    def mixin_tree(T: Tensor, bounds: Tensor, alpha: float) -> Tensor:
        return (1 - alpha) * T + alpha * unnormalize(torch.rand_like(T), bounds)

    n_repeat = math.ceil(raw_samples / batch_shape[0])
    alphas = torch.linspace(0, 0.75, n_repeat, **tkwargs)
    etas = torch.linspace(0.1, 1.0, len(Xopts), **tkwargs)

    X_full = torch.cat(
        [
            mixin_tree(
                T=make_init_tree(Xopts=Xopts, bounds=bounds, etas=etas),
                bounds=bounds,
                alpha=alpha,
            )
            for alpha in alphas
        ],
        dim=0,
    )

    with torch.no_grad():
        acq_vals = acq_function(X_full)
    X_init, _ = initialize_q_batch(X=X_full, acq_vals=acq_vals, n=num_restarts, eta=1.0)
    return X_init[:raw_samples]


def make_best_f(model: Model, X: Tensor) -> dict[str, Any]:
    r"""Extract the best observed training input from the model."""
    return {"best_f": model.train_targets.max(dim=-1).values}
