#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Discretization (rounding) functions for acquisition optimization.

References

.. [Daulton2022bopr]
    S. Daulton, X. Wan, D. Eriksson, M. Balandat, M. A. Osborne, E. Bakshy.
    Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic
    Reparameterization. Advances in Neural Information Processing Systems
    35, 2022.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn.functional import one_hot


def approximate_round(X: Tensor, tau: float = 1e-3) -> Tensor:
    r"""Differentiable approximate rounding function.

    This method is a piecewise approximation of a rounding function where
    each piece is a hyperbolic tangent function.

    Args:
        X: The tensor to round to the nearest integer (element-wise).
        tau: A temperature hyperparameter.

    Returns:
        The approximately rounded input tensor.
    """
    offset = X.floor()
    scaled_remainder = (X - offset - 0.5) / tau
    rounding_component = (torch.tanh(scaled_remainder) + 1) / 2
    return offset + rounding_component


class IdentitySTEFunction(Function):
    """Base class for functions using straight through gradient estimators.

    This class approximates the gradient with the identity function.
    """

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        r"""Use a straight-through estimator the gradient.

        This uses the identity function.

        Args:
            grad_output: A tensor of gradients.

        Returns:
            The provided tensor.
        """
        return grad_output


class RoundSTE(IdentitySTEFunction):
    r"""Round the input tensor and use a straight-through gradient estimator.

    [Daulton2022bopr]_ proposes using this in acquisition optimization.
    """

    @staticmethod
    def forward(ctx, X: Tensor) -> Tensor:
        r"""Round the input tensor element-wise.

        Args:
            X: The tensor to be rounded.

        Returns:
            A tensor where each element is rounded to the nearest integer.
        """
        return X.round()


class OneHotArgmaxSTE(IdentitySTEFunction):
    r"""Discretize a continuous relaxation of a one-hot encoded categorical.

    This returns a one-hot encoded categorical and use a straight-through
    gradient estimator via an identity function.

    [Daulton2022bopr]_ proposes using this in acquisition optimization.
    """

    @staticmethod
    def forward(ctx, X: Tensor) -> Tensor:
        r"""Discretize the input tensor.

        This applies a argmax along the last dimensions of the input tensor
        and one-hot encodes the result.

        Args:
            X: The tensor to be rounded.

        Returns:
            A tensor where each element is rounded to the nearest integer.
        """
        return one_hot(X.argmax(dim=-1), num_classes=X.shape[-1]).to(X)
