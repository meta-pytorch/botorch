#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Helpers for handling input or outcome constraints.
"""

from __future__ import annotations

import math

from collections.abc import Callable

from functools import partial

import torch
from gpytorch import settings
from gpytorch.constraints import Interval

from torch import Tensor


def get_outcome_constraint_transforms(
    outcome_constraints: tuple[Tensor, Tensor] | None,
) -> list[Callable[[Tensor], Tensor]] | None:
    r"""Create outcome constraint callables from outcome constraint tensors.

    Args:
        outcome_constraints: A tuple of `(A, b)`. For `k` outcome constraints
            and `m` outputs at `f(x)``, `A` is `k x m` and `b` is `k x 1` such
            that `A f(x) <= b`.

    Returns:
        A list of callables, each mapping a Tensor of size `b x q x m` to a
        tensor of size `b x q`, where `m` is the number of outputs of the model.
        Negative values imply feasibility. The callables support broadcasting
        (e.g. for calling on a tensor of shape `mc_samples x b x q x m`).

    Example:
        >>> # constrain `f(x)[0] <= 0`
        >>> A = torch.tensor([[1., 0.]])
        >>> b = torch.tensor([[0.]])
        >>> outcome_constraints = get_outcome_constraint_transforms((A, b))
    """
    if outcome_constraints is None:
        return None
    A, b = outcome_constraints

    def _oc(a: Tensor, rhs: Tensor, Y: Tensor) -> Tensor:
        r"""Evaluate constraints.

        Note: einsum multiples Y by a and sums over the `m`-dimension. Einsum
            is ~2x faster than using `(Y * a.view(1, 1, -1)).sum(dim-1)`.

        Args:
            a: `m`-dim tensor of weights for the outcomes
            rhs: Singleton tensor containing the outcome constraint value
            Y: `... x b x q x m` tensor of function values

        Returns:
            A `... x b x q`-dim tensor where negative values imply feasibility
        """
        lhs = torch.einsum("...m, m", [Y, a])
        return lhs - rhs

    return [partial(_oc, a, rhs) for a, rhs in zip(A, b)]


def get_monotonicity_constraints(
    d: int,
    descending: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Returns a system of linear inequalities `(A, b)` that generically encodes order
    constraints on the elements of a `d`-dimsensional space, i.e. `A @ x < b` implies
    `x[i] < x[i + 1]` for a `d`-dimensional vector `x`.

    Idea: Could encode `A` as sparse matrix, if it is supported well.

    Args:
        d: Dimensionality of the constraint space, i.e. number of monotonic parameters.
        descending: If True, forces the elements of a vector to be monotonically de-
            creasing and be monotonically increasing otherwise.
        dtype: The dtype of the returned Tensors.
        device: The device of the returned Tensors.

    Returns:
        A tuple of Tensors `(A, b)` representing the monotonicity constraint as a system
        of linear inequalities `A @ x < b`. `A` is `(d - 1) x d`-dimensional and `b` is
        `(d - 1) x 1`-dimensional.
    """
    A = torch.zeros(d - 1, d, dtype=dtype, device=device)
    idx = torch.arange(d - 1)
    A[idx, idx] = 1
    A[idx, idx + 1] = -1
    b = torch.zeros(d - 1, 1, dtype=dtype, device=device)
    if descending:
        A = -A
    return A, b


class NonTransformedInterval(Interval):
    """Modification of the GPyTorch interval class that does not apply transformations.

    This is generally useful, and it is a requirement for the sparse parameters of the
    Relevance Pursuit model [Ament2024pursuit]_, since it is not possible to achieve
    exact zeros with the sigmoid transformations that are applied by default in the
    GPyTorch Interval class. The variant implemented here does not apply transformations
    to the parameters, instead passing the bounds constraint to the scipy L-BFGS
    optimizer. This allows for the expression of exact zeros for sparse optimization
    algorithms.

    NOTE: On a high level, the cleanest solution for this would be to separate out the
    1) definition and book-keeping of parameter constraints on the one hand, and
    2) the re-parameterization of the variables with some monotonic transformation,
    since the two steps are orthogonal, but this would require refactoring GPyTorch.
    """

    def __init__(
        self,
        lower_bound: float | Tensor,
        upper_bound: float | Tensor,
        initial_value: float | Tensor | None = None,
    ):
        """Constructor of the NonTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=None,
            inv_transform=None,
            initial_value=initial_value,
        )

    def transform(self, tensor: Tensor) -> Tensor:
        return tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        return transformed_tensor


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. LogTransformedInterval skips this step to avoid
    numerical issues, and applies the log transform directly to the parameter values.
    GPyTorch automatically recognizes that the bound constraint have not been applied
    yet, and passes the bounds to the optimizer instead, which then optimizes
    log(parameter) under the constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        initial_value: float | None = None,
    ):
        """Constructor of the LogTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            (
                torch.tensor(initial_value).to(self.lower_bound)
                if initial_value is not None
                else None
            ),
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a "
                    "derived class like GreaterThan or LessThan instead."
                )

    def transform(self, tensor: Tensor) -> Tensor:
        """Transform the parameter using the exponential function.

        Args:
            tensor: Tensor of parameter values to transform.

        Returns:
            Tensor of transformed parameter values.
        """
        return self._transform(tensor)

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        """Untransform the parameter using the natural logarithm.

        Args:
            tensor: Tensor of parameter values to untransform.

        Returns:
            Tensor of untransformed parameter values.
        """
        return self._inv_transform(transformed_tensor)
