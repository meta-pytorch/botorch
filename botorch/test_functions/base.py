#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Base class for test functions for optimization benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Protocol

import torch
from botorch.exceptions.errors import InputDataError, UnsupportedError
from pyre_extensions import none_throws
from torch import Tensor
from torch.nn import Module


def validate_parameter_indices(
    dim: int,
    bounds: Tensor,
    continuous_inds: list[int],
    discrete_inds: list[int],
    categorical_inds: list[int],
) -> None:
    r"""Check that the parameter indices are valid.

    Args:
        dim: Number of search space dimensions.
        bounds: A `2 x d`-dim tensor of lower and upper bounds.
        continuous_inds: List of unique integers corresponding to continuous parameters.
        discrete_inds: List of unique integers corresponding to discrete parameters.
        categorical_inds: List of unique integers corresponding to categorical
            parameters.

    Raises:
        ValueError: If the parameter indices are invalid.
    """
    for inds in [continuous_inds, discrete_inds, categorical_inds]:
        if len(inds) == 0:
            continue  # Nothing to check
        if not (
            isinstance(inds, list)
            and all(isinstance(x, int) for x in inds)
            and len(set(inds)) == len(inds)
            and min(inds) >= 0
            and max(inds) <= dim - 1
        ):
            raise ValueError(
                "All parameter indices must be a list with unique integers between "
                f"0 and dim - 1. Got {inds=}."
            )
    # Let's make sure all parameters are covered.
    all_inds = continuous_inds + discrete_inds + categorical_inds
    if (len(all_inds) != dim) or (set(all_inds) != set(range(dim))):
        raise ValueError(
            f"All parameter indices must be present, got {dim=}, {continuous_inds=}, "
            f"{discrete_inds=}, {categorical_inds=}"
        )
    # Check the shape of the bounds
    if not bounds.shape == (2, dim):
        raise ValueError(
            f"Expected `bounds` to have shape `2 x d`. Got {bounds.shape=}, {dim=}."
        )
    # Check that the bounds are integer valued for discrete and categorical parameters.
    for inds in [discrete_inds, categorical_inds]:
        if len(inds) > 0 and (not (bounds[:, inds] == bounds[:, inds].round()).all()):
            raise ValueError(
                "Expected the lower and upper bounds of the discrete and categorical "
                "parameters to be integer-valued."
            )


def validate_inputs(
    X: Tensor,
    dim: int,
    bounds: Tensor,
    discrete_inds: list[int],
    categorical_inds: list[int],
) -> None:
    r"""Check that the inputs are valid.

    This method checks that the input tensor `X` has the correct shape, is within
    the bounds, and that the discrete and categorical parameters are integer-valued.

    Args:
        X: A `(batch_shape) x n x d`-dim tensor of point(s) at which to evaluate
        dim: Number of search space dimensions.
        bounds: A `2 x d`-dim tensor of lower and upper bounds.
        discrete_inds: List of unique integers corresponding to discrete parameters.
        categorical_inds: List of unique integers corresponding to categorical
            parameters.

    Raises:
        ValueError: If the parameter indices are invalid.
    """

    if not X.shape[-1] == dim:
        raise ValueError(
            "Expected `X` to have shape `(batch_shape) x n x d`. "
            f"Got {X.shape=} and {dim=}"
        )
    if not ((X >= bounds[0]).all() and (X <= bounds[1]).all()):
        raise ValueError("Expected `X` to be within the bounds of the test problem.")
    for inds in [discrete_inds, categorical_inds]:
        if not (X[..., inds] == X[..., inds].round()).all():
            raise ValueError(
                "Expected `X` to have integer values for the discrete and "
                "categorical parameters."
            )


class BaseTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: list[
        tuple[float, float]
    ]  # Bounds, must be integers for discrete/categorical parameters
    _check_grad_at_opt: bool = True
    continuous_inds: list[int] = []  # Float-valued range parameters (bounds inclusive)
    discrete_inds: list[int] = []  # Ordered integer parameters (bounds inclusive)
    categorical_inds: list[int] = []  # Unordered integer parameters (bounds inclusive)
    # Whether the problem is a minimization problem by default, with `negate=False`.
    _is_minimization_by_default: bool = True

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base constructor for test functions.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        if len(self._bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self._bounds)=}."
            )
        self.register_buffer(
            "bounds",
            torch.tensor(self._bounds, dtype=dtype).transpose(-1, -2),
        )
        validate_parameter_indices(
            dim=self.dim,
            bounds=self.bounds,
            continuous_inds=self.continuous_inds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `(batch_shape) x d`-dim tensor of point(s) at which to evaluate
                the function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
        """
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            _noise = torch.tensor(self.noise_std, device=X.device, dtype=X.dtype)
            f += _noise * torch.randn_like(f)
        if self.negate:
            f = -f
        return f

    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the function (w/o observation noise) on a set of points.

        Args:
            X: A `(batch_shape) x d`-dim tensor of point(s) at which to
                evaluate.

        Returns:
            A `batch_shape`-dim tensor.
        """
        validate_inputs(
            X=X,
            dim=self.dim,
            bounds=self.bounds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )
        return self._evaluate_true(X=X)

    @abstractmethod
    def _evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points.

        Args:
            X: A `(batch_shape) x d`-dim tensor of point(s) at which to
                evaluate.

        Returns:
            A `batch_shape`-dim tensor.
        """
        pass  # pragma: no cover

    @property
    def is_minimization_problem(self) -> bool:
        r"""Whether the problem is a minimization problem, after accounting
        for the `negate` option.
        """
        return (
            self._is_minimization_by_default
            if not self.negate
            else not self._is_minimization_by_default
        )


class ConstrainedBaseTestProblem(BaseTestProblem, ABC):
    r"""Base class for test functions with constraints.

    In addition to one or more objectives, a problem may have a number of outcome
    constraints of the form `c_i(x) >= 0` for `i=1, ..., n_c`.

    This base class provides common functionality for such problems.
    """

    num_constraints: int
    _check_grad_at_opt: bool = False
    constraint_noise_std: None | float | list[float] = None
    _worst_feasible_value: float | None = None

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint slack on a set of points.

        Constraints `i` is assumed to be feasible at `x` if the associated slack
        `c_i(x)` is positive. Zero slack means that the constraint is active. Negative
        slack means that the constraint is violated.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.
            noise: If `True`, add observation noise to the slack as specified by
                `noise_std`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        cons = self.evaluate_slack_true(X=X)
        if noise and self.constraint_noise_std is not None:
            _constraint_noise = torch.tensor(
                self.constraint_noise_std, device=X.device, dtype=X.dtype
            )
            cons += _constraint_noise * torch.randn_like(cons)
        return cons

    def is_feasible(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate whether the constraints are feasible on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraints.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim boolean tensor that is `True` iff all constraint
                slacks (potentially including observation noise) are positive.
        """
        return (self.evaluate_slack(X=X, noise=noise) >= 0.0).all(dim=-1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the constraint slack (w/o observation noise) on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        validate_inputs(
            X=X,
            dim=self.dim,
            bounds=self.bounds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )
        return self._evaluate_slack_true(X=X)

    @abstractmethod
    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the constraint slack (w/o observation noise) on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds
        """
        pass  # pragma: no cover

    @property
    def worst_feasible_value(self) -> float:
        r"""The worst feasible value of the objective function. This is useful when
        evaluating the performance of different optimization methods as this value
        can be assigned to all infeasible trials. This has the desirable property that
        any feasible trial has better performance than an infeasible trial.
        """
        if isinstance(self, MultiObjectiveTestProblem):
            return 0.0  # Can return 0.0 for MOO since this is the smallest possible HV
        elif self._worst_feasible_value is not None:
            return (
                -self._worst_feasible_value
                if self.negate
                else self._worst_feasible_value
            )
        raise NotImplementedError(
            f"Problem {self.__class__.__name__} does not specify the "
            "worst feasible value."
        )


class MultiObjectiveTestProblem(BaseTestProblem, ABC):
    r"""Base class for multi-objective test functions.

    TODO: add a pareto distance function that returns the distance
    between a provided point and the closest point on the true pareto front.
    """

    num_objectives: int
    _ref_point: list[float]
    _max_hv: float | None = None

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base constructor for multi-objective test functions.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective.
            negate: If True, negate the objectives.
        """
        if isinstance(noise_std, list) and len(noise_std) != len(self._ref_point):
            raise InputDataError(
                f"If specified as a list, length of noise_std ({len(noise_std)}) "
                f"must match the number of objectives ({len(self._ref_point)})"
            )
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        ref_point = torch.tensor(self._ref_point, dtype=dtype)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        if self._max_hv is not None:
            return self._max_hv
        else:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate `n` pareto optimal points."""
        raise NotImplementedError

    @property
    def is_minimization_problem(self) -> bool:
        raise UnsupportedError(
            "`is_minimization_problem` is not a valid property for "
            "multi-objective problems. "
        )


class SeedingMixin(ABC):
    _seeds: Iterator[int] | None
    _current_seed: int | None

    @property
    def has_seeds(self) -> bool:
        return self._seeds is not None

    def increment_seed(self) -> int:
        self._current_seed = next(none_throws(self._seeds))
        return none_throws(self._current_seed)

    @property
    def seed(self) -> int | None:
        return self._current_seed


# Outlier problems
class OutlierGenerator(Protocol):
    def __call__(self, problem: BaseTestProblem, X: Tensor, bounds: Tensor) -> Tensor:
        """Call signature for outlier generators for single-objective problems.

        Args:
            problem: The test problem.
            X: The input tensor.
            bounds: The bounds of the test problem.

        Returns:
            A tensor of outliers with shape X.shape[:-1] (1d if unbatched).
        """
        pass  # pragma: no cover


def constant_outlier_generator(
    problem: Any, X: Tensor, bounds: Any, constant: float
) -> Tensor:
    """
    Generates outliers that are all the same constant. To be used in conjunction with
    `partial` to fix the constant value and conform to the `OutlierGenerator` protocol.

    Example:
        >>> generator = partial(constant_outlier_generator, constant=1.0)

    Args:
        problem: Not used.
        X: The `batch_shape x n x d`-dim inputs. Also determines the number, dtype,
            and device of the returned tensor.
        bounds: Not used.
        constant: The constant value of the outliers.

    Returns:
        Tensor of shape `batch_shape x n` (1d if unbatched).
    """
    return torch.full(X.shape[:-1], constant, dtype=X.dtype, device=X.device)


class CorruptedTestProblem(BaseTestProblem, SeedingMixin):
    def __init__(
        self,
        base_test_problem: BaseTestProblem,
        outlier_generator: OutlierGenerator,
        outlier_fraction: float,
        bounds: list[tuple[float, float]] | None = None,
        seeds: Iterable[int] | None = None,
    ) -> None:
        """A problem with outliers.

        NOTE: Both noise_std and negate will be taken from the base test problem.

        Args:
            base_test_problem: The base function to be corrupted.
            outlier_generator: A function that generates outliers. It will be called
                with arguments `f`, `X` and `bounds`, where `f` is the
                `base_test_problem`, `X` is the
                argument passed to the `forward` method, and `bounds`
                are as here, and it returns the values of outliers.
            outlier_fraction: The fraction of outliers.
            bounds: The bounds of the function.
            seeds: The seeds to use for the outlier generator. If seeds are provided,
                the problem will iterate through the list of seeds, changing the seed
                with a call to `next(seeds)` with every `forward` call. If a list is
                provided, it will first be converted to an iterator.
        """
        self.dim: int = base_test_problem.dim
        self.continuous_inds = base_test_problem.continuous_inds
        self.discrete_inds = base_test_problem.discrete_inds
        self.categorical_inds = base_test_problem.categorical_inds
        self._bounds: list[tuple[float, float]] = (
            bounds if bounds is not None else base_test_problem._bounds
        )
        super().__init__(
            noise_std=base_test_problem.noise_std,
            negate=base_test_problem.negate,
        )
        self.base_test_problem = base_test_problem
        self.outlier_generator = outlier_generator
        self.outlier_fraction = outlier_fraction
        self._current_seed: int | None = None
        self._seeds: Iterator[int] | None = None if seeds is None else iter(seeds)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self.base_test_problem.evaluate_true(X)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
        Generate data at X and corrupt it, if noise is True.

        Args:
            X: The `batch_shape x n x d`-dim inputs.
            noise: Whether to corrupt the data.

        Returns:
            A `batch_shape x n`-dim tensor.
        """
        Y = super().forward(X, noise=noise)
        if noise:
            if self.has_seeds:
                self.increment_seed()
                torch.manual_seed(self.seed)
            corrupt = torch.rand(X.shape[:-1]) < self.outlier_fraction
            outliers = self.outlier_generator(
                problem=self.base_test_problem, X=X, bounds=self.bounds
            )
            Y = torch.where(corrupt, outliers, Y)
        return Y
