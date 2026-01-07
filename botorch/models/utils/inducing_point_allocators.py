#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Functionality for allocating the inducing points of sparse Gaussian
process models.

References

.. [chen2018dpp]
    Laming Chen and Guoxin Zhang and Hanning Zhou, Fast greedy MAP inference
    for determinantal point process to improve recommendation diversity,
    Proceedings of the 32nd International Conference on Neural Information
    Processing Systems, 2018, https://arxiv.org/abs/1709.05135.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model

from botorch.utils.probability.utils import ndtr as Phi, phi
from gpytorch.module import Module
from linear_operator.operators import LinearOperator
from torch import Tensor

NEG_INF = torch.tensor(float("-inf"))


class InducingPointAllocator(ABC):
    r"""
    This class provides functionality to initialize the inducing point locations
    of an inducing point-based model, e.g. a `SingleTaskVariationalGP`.
    """

    @abstractmethod
    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the quality function required for this inducing point allocation strategy.

        Returns:
            A quality function.
        """

    def allocate_inducing_points(
        self,
        inputs: Tensor,
        covar_module: Module,
        num_inducing: int,
        input_batch_shape: torch.Size,
    ) -> Tensor:
        r"""
        Initialize the `num_inducing` inducing point locations according to a
        specific initialization strategy. todo say something about quality

        Args:
            inputs: A (\*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.

        Returns:
            A (\*batch_shape, m, d)-dim tensor of inducing point locations.
        """
        quality_function = self._get_quality_function()
        covar_module = covar_module.to(inputs.device)

        # We use 'no_grad' here because `inducing_points` are not
        # auto-differentiable with respect to the kernel hyper-parameters,
        # because `_pivoted_cholesky_init` does in-place operations.
        with torch.no_grad():
            # Evaluate lazily because this may only be needed to figure out what
            # case we are in
            possibly_lazy_kernel = covar_module(inputs)

        base_case = possibly_lazy_kernel.ndimension() == 2
        multi_task_case = (
            possibly_lazy_kernel.ndimension() == 3 and len(input_batch_shape) == 0
        )

        if base_case or multi_task_case:
            train_train_kernel = possibly_lazy_kernel.evaluate_kernel()

        if base_case:
            quality_scores = quality_function(inputs)
            inducing_points = _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=train_train_kernel,
                max_length=num_inducing,
                quality_scores=quality_scores,
            )
            return inducing_points

        if multi_task_case:
            input_element = inputs[0] if inputs.ndimension() == 3 else inputs
            kernel_element = train_train_kernel[0]
            quality_scores = quality_function(input_element)
            inducing_points = _pivoted_cholesky_init(
                train_inputs=input_element,
                kernel_matrix=kernel_element,
                max_length=num_inducing,
                quality_scores=quality_scores,
            )
            return inducing_points

        # batched input cases
        batched_inputs = (
            inputs.expand(*input_batch_shape, -1, -1)
            if inputs.ndimension() == 2
            else inputs
        )
        reshaped_inputs = batched_inputs.flatten(end_dim=-3)
        inducing_points = []
        for input_element in reshaped_inputs:
            # the extra kernel evals are a little wasteful but make it
            # easier to infer the task batch size
            # We use 'no_grad' here because `inducing_points` are not
            # auto-differentiable with respect to the kernel hyper-parameters,
            # because `_pivoted_cholesky_init` does in-place operations.
            with torch.no_grad():
                kernel_element = covar_module(input_element).evaluate_kernel()
            # handle extra task batch dimension
            kernel_element = (
                kernel_element[0]
                if kernel_element.ndimension() == 3
                else kernel_element
            )
            quality_scores = quality_function(input_element)
            inducing_points.append(
                _pivoted_cholesky_init(
                    train_inputs=input_element,
                    kernel_matrix=kernel_element,
                    max_length=num_inducing,
                    quality_scores=quality_scores,
                )
            )
        inducing_points = torch.stack(inducing_points).view(
            *input_batch_shape, num_inducing, -1
        )

        return inducing_points


class QualityFunction(ABC):
    """A function that scores inputs with respect
    to a specific criterion."""

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of quality scores for each input, of shape [n]
        """


class UnitQualityFunction(QualityFunction):
    """
    A function returning ones for each element. Using this quality function
    for inducing point allocation corresponds to allocating inducing points
    with the sole aim of minimizing predictive variance, i.e. the approach
    of [burt2020svgp]_.
    """

    @torch.no_grad()
    def __call__(self, inputs: Tensor) -> Tensor:  # [n, d]-> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of ones for each input, of shape [n]
        """
        return torch.ones([inputs.shape[0]], device=inputs.device, dtype=inputs.dtype)


class ExpectedImprovementQualityFunction(QualityFunction):
    """
    A function measuring the quality of input points as their expected
    improvement with respect to a conservative baseline. Expectations
    are according to the model from the previous BO step. See [moss2023ipa]_
    for details and justification.
    """

    def __init__(self, model: Model, maximize: bool):
        r"""
        Args:
            model: The model fitted during the previous BO step. For now, this
                must be a single task model (i.e. num_outputs=1).
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        if model.num_outputs != 1:
            raise NotImplementedError(
                "Multi-output models are currently not supported. "
            )
        self._model = model
        self._maximize = maximize

    @torch.no_grad()
    def __call__(self, inputs: Tensor) -> Tensor:  # [n, d] -> [n]
        """
        Args:
            inputs: inputs (of shape n x d)

        Returns:
            A tensor of quality scores for each input, of shape [n]
        """

        posterior = self._model.posterior(inputs)
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)

        best_f = torch.max(mean) if self._maximize else torch.min(mean)
        u = (mean - best_f) / sigma if self._maximize else -(mean - best_f) / sigma
        return sigma * (phi(u) + u * Phi(u))


class GreedyVarianceReduction(InducingPointAllocator):
    r"""
    The inducing point allocator proposed by [burt2020svgp]_, that
    greedily chooses inducing point locations with maximal (conditional)
    predictive variance.
    """

    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the unit quality function required for the greedy variance
        reduction inducing point allocation strategy.

        Returns:
            A quality function.
        """

        return UnitQualityFunction()


class GreedyImprovementReduction(InducingPointAllocator):
    r"""
    An inducing point allocator that greedily chooses inducing points with large
    predictive variance and that are in promising regions of the search
    space (according to the model form the previous BO step), see [moss2023ipa]_.
    """

    def __init__(self, model: Model, maximize: bool):
        r"""

        Args:
            model: The model fitted during the previous BO step.
            maximize: Set True if we are performing function maximization, else
                set False.
        """
        self._model = model
        self._maximize = maximize

    def _get_quality_function(
        self,
    ) -> QualityFunction:
        """
        Build the improvement-based quality function required for the greedy
        improvement reduction inducing point allocation strategy.

        Returns:
            A quality function.
        """

        return ExpectedImprovementQualityFunction(self._model, self._maximize)


def _pivoted_cholesky_init(
    train_inputs: Tensor,
    kernel_matrix: Tensor | LinearOperator,
    max_length: int,
    quality_scores: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    r"""
    A pivoted Cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, based on
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py but with a small
    modification to allow the underlying DPP to be defined through its diversity-quality
    decomposition,as discussed by [moss2023ipa]_. This method returns a greedy
    approximation of the MAP estimate of the specified DPP, i.e. its returns a
    set of points that are highly diverse (according to the provided kernel_matrix)
    and have high quality (according to the provided quality_scores).

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training inputs
        max_length: number of inducing points to initialize
        quality_scores: scores representing the quality of each candidate
            input (of shape [n])
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """

    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: ensure this works in batch mode, which it does not currently.

    # todo test for shape of quality function

    if quality_scores.shape[0] != train_inputs.shape[0]:
        raise ValueError(
            "_pivoted_cholesky_init requires a quality score for each of train_inputs"
        )

    if kernel_matrix.requires_grad:
        raise UnsupportedError(
            "`_pivoted_cholesky_init` does not support using a `kernel_matrix` "
            "with `requires_grad=True`."
        )

    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros(
        (max_length, item_size), device=kernel_matrix.device, dtype=kernel_matrix.dtype
    )
    di2s = kernel_matrix.diagonal()
    scores = di2s * torch.square(quality_scores)
    selected_item = torch.argmax(scores)
    selected_items = [selected_item]

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = NEG_INF
        scores = di2s * torch.square(quality_scores)
        selected_item = torch.argmax(scores)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points[:max_length, :]
