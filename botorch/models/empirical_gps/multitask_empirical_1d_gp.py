#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-task Empirical One-Dimensional GP model.

This module contains the MultiTaskEmpiricalOneDimensionalGP model which implements
a multi-task GP using the "long format" data representation where inputs include
a task feature column, following [lin2026empirical]_. Unlike the multi-output
version, this model supports partial observations where different tasks can have
different observed input locations.

The model supports heterogeneous historical data where each task can have:
- Different progression domains (historical_Xs[t])
- Different numbers of progression points
- But the same number of curves (required for cross-task covariance)

"""

from __future__ import annotations

import torch
from botorch.models.empirical_gps.empirical_1d_gp import BaseAugmentedEmpiricalKernel
from botorch.models.empirical_gps.utils import (
    center_curves,
    compute_sample_covariance,
    LinearInterpolation1D,
)
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.models import ExactGP
from torch import Tensor


def _extract_progression_and_task(
    x: Tensor, task_feature: int
) -> tuple[Tensor, Tensor]:
    """Extract progression values and task indices from long-format input.

    This function assumes d=1 (one-dimensional progression), so the input
    has exactly 2 columns: one for progression and one for task index.

    Args:
        x: `batch_shape x n x 2`-dim input tensor (progression + task feature).
        task_feature: Index of the task feature column (0 or 1).

    Returns:
        Tuple of (progression, task_idcs), each `batch_shape x n`.
    """
    task_feature_values = x[..., task_feature]
    # Task labels must be integer-valued. Casting to long truncates, so validate
    # before use rather than silently mapping e.g. 0.9 -> 0 or 1.5 -> 1. Accepted
    # values are rounded (not truncated) so a float like 1.9999999 maps to 2, not 1.
    if task_feature_values.is_floating_point():
        rounded = task_feature_values.round()
        if not torch.allclose(task_feature_values, rounded, atol=1e-6, rtol=0.0):
            bad = (
                task_feature_values[(task_feature_values - rounded).abs() > 1e-6]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Task labels must be integer-valued; got non-integer values {bad}."
            )
        task_idcs = rounded.long()
    else:
        task_idcs = task_feature_values.long()
    # For d=1, the non-task column is at index 0 if task_feature=1, else index 1
    progression_idx = 0 if task_feature != 0 else 1
    progression = x[..., progression_idx]
    return progression, task_idcs


def _validate_task_indices(task_idcs: Tensor, num_tasks: int) -> None:
    """Validate that all task indices are in [0, num_tasks).

    Defensive check — also validated by the model constructor, but guards
    against standalone mean/kernel usage with invalid inputs.

    Args:
        task_idcs: Integer tensor of task indices.
        num_tasks: Number of tasks (valid range is [0, num_tasks)).

    Raises:
        ValueError: If any task index is out of range.
    """
    invalid = (task_idcs < 0) | (task_idcs >= num_tasks)
    if invalid.any():
        bad = task_idcs[invalid].unique().tolist()
        raise ValueError(f"Task indices {bad} are out of range [0, {num_tasks}).")


def _validate_heterogeneous_historical_data(
    historical_Xs: list[Tensor],
    historical_Ys: list[Tensor],
) -> int:
    """Validate heterogeneous historical data and return num_curves.

    Args:
        historical_Xs: List of m tensors, where historical_Xs[t] is
            `num_progression_t x 1` for task t. Each task can have a
            different number of progression points and different domain.
        historical_Ys: List of m tensors, where historical_Ys[t] is
            `num_curves x num_progression_t` for task t. All tasks must
            have the same num_curves (curves are paired across tasks for
            cross-task covariance computation).

    Returns:
        num_curves: The number of curves (must be same for all tasks).

    Raises:
        ValueError: If validation fails.
    """
    if len(historical_Xs) != len(historical_Ys):
        raise ValueError(
            f"historical_Xs has {len(historical_Xs)} tasks but "
            f"historical_Ys has {len(historical_Ys)} tasks."
        )

    if len(historical_Ys) == 0:
        raise ValueError("historical_Ys cannot be empty.")

    # All tasks must have the same num_curves
    num_curves = historical_Ys[0].shape[0]
    for t, Y_t in enumerate(historical_Ys):
        if Y_t.ndim != 2:
            raise ValueError(
                f"historical_Ys[{t}] must be 2-dim (num_curves x num_progression), "
                f"got {Y_t.ndim}-dim."
            )
        if Y_t.shape[0] != num_curves:
            raise ValueError(
                f"All tasks must have the same number of curves. "
                f"Task 0 has {num_curves} curves, but task {t} has {Y_t.shape[0]}."
            )

    # Validate X shapes match Y shapes
    for t, (X_t, Y_t) in enumerate(zip(historical_Xs, historical_Ys)):
        if X_t.ndim != 2 or X_t.shape[-1] != 1:
            raise ValueError(
                f"historical_Xs[{t}] must be (num_progression_t x 1), got {X_t.shape}."
            )
        if X_t.shape[0] != Y_t.shape[1]:
            raise ValueError(
                f"historical_Xs[{t}] has {X_t.shape[0]} points but "
                f"historical_Ys[{t}] has {Y_t.shape[1]} points."
            )

    return num_curves


class MultiTaskEmpiricalOneDimensionalMean(Mean):
    """Mean module for multi-task empirical one-dimensional GP.

    This mean module handles inputs in "long format" where the last column
    contains task indices. It extracts the progression values and task indices,
    then returns the appropriate mean value for each input-task pair.

    Supports heterogeneous historical data where each task can have:
    - Different progression domains
    - Different numbers of progression points
    """

    def __init__(
        self,
        historical_Xs: list[Tensor],
        historical_Ys: list[Tensor],
        task_feature: int,
    ):
        """Instantiates a multi-task empirical one-dimensional mean function.

        Args:
            historical_Xs: List of m tensors, where historical_Xs[t] is
                `num_progression_t x 1` for task t. Each task can have a
                different number of progression points and different domain.
            historical_Ys: List of m tensors, where historical_Ys[t] is
                `num_curves x num_progression_t` for task t.
            task_feature: The index of the task feature in the input.

        Note:
            Assumes historical_Xs and historical_Ys have been validated by the
            caller (e.g., MultiTaskEmpiricalOneDimensionalGP).
        """
        super().__init__()
        self.task_feature = task_feature
        self.num_tasks = len(historical_Ys)

        # Create per-task interpolants for the mean
        self.interpolants = torch.nn.ModuleList()
        for t in range(self.num_tasks):
            X_t = historical_Xs[t]  # num_progression_t x 1
            Y_t = historical_Ys[t]  # num_curves x num_progression_t
            mean_t, _ = center_curves(Y_t, curve_dim=0)  # num_progression_t
            f_t = LinearInterpolation1D(X_t.squeeze(-1), mean_t)
            self.interpolants.append(f_t)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the mean function at x.

        Args:
            x: `batch_shape x n x (d+1)`-dim Tensor where the column at
                `task_feature` contains task indices.

        Returns:
            `batch_shape x n`-dim Tensor of mean values.
        """
        progression, task_idcs = _extract_progression_and_task(x, self.task_feature)

        # Handle case where there's no batch dimension
        squeeze_output = progression.ndim == 1
        if squeeze_output:
            progression = progression.unsqueeze(0)
            task_idcs = task_idcs.unsqueeze(0)

        # Validate task indices (defensive — also validated by the model
        # constructor, but guards against standalone mean module usage).
        _validate_task_indices(task_idcs, self.num_tasks)

        # Initialize output and process each task
        mean = torch.zeros_like(progression)
        for t in range(self.num_tasks):
            mask = task_idcs == t
            if mask.any():
                mean[mask] = self.interpolants[t](progression[mask])

        if squeeze_output:
            mean = mean.squeeze(0)

        if mean.isnan().any():
            # Guard against NaN in the computed mean. The most common cause is an
            # interpolant evaluated outside its training range (when bounds_error
            # is disabled), but NaN from any other source is also caught here.
            raise ValueError(
                "Mean contains NaN values, which typically indicates an "
                "interpolant was evaluated outside its training range."
            )
        return mean


class MultiTaskEmpiricalOneDimensionalKernel(Kernel):
    """Kernel for multi-task empirical one-dimensional GP.

    This kernel handles inputs in "long format" where the last column contains
    task indices. It combines the empirical one-dimensional kernel (for
    progression-based covariance) with task covariance from historical data.

    Supports heterogeneous historical data where each task can have:
    - Different progression domains
    - Different numbers of progression points
    - But the same number of curves (required for cross-task covariance)

    """

    def __init__(
        self,
        historical_Xs: list[Tensor],
        historical_Ys: list[Tensor],
        task_feature: int,
        num_curves: int,
        correction: int = 0,
    ) -> None:
        """Instantiates a multi-task empirical one-dimensional kernel.

        Args:
            historical_Xs: List of m tensors, where historical_Xs[t] is
                `num_progression_t x 1` for task t. Each task can have a
                different number of progression points and different domain.
            historical_Ys: List of m tensors, where historical_Ys[t] is
                `num_curves x num_progression_t` for task t.
            task_feature: The index of the task feature in the input.
            num_curves: Number of curves (must be same for all tasks).
            correction: Degrees of freedom correction for sample covariance.
                Default is 0 (ML estimate). Use 1 for Bessel correction.

        Note:
            Assumes historical_Xs and historical_Ys have been validated by the
            caller (e.g., MultiTaskEmpiricalOneDimensionalGP).
        """
        super().__init__()
        self.task_feature = task_feature
        self.num_tasks = len(historical_Ys)
        self.num_curves = num_curves
        self.correction = correction

        # Create per-task interpolants for centered curves
        self.interpolants = torch.nn.ModuleList()
        for t in range(self.num_tasks):
            X_t = historical_Xs[t]  # num_progression_t x 1
            Y_t = historical_Ys[t]  # num_curves x num_progression_t
            # num_curves x num_progression_t, centered across curves
            _, Y_centered_t = center_curves(Y_t, curve_dim=0)
            f_t = LinearInterpolation1D(X_t.squeeze(-1), Y_centered_t)
            self.interpolants.append(f_t)

    def _compute_basis(
        self,
        progression: Tensor,
        task_idcs: Tensor,
        batch_shape: torch.Size,
    ) -> Tensor:
        """Compute basis matrix by interpolating per-task historical curves.

        Boolean mask indexing flattens across all dims, producing a 1D tensor
        of matched values. The interpolant is a pure function (no
        batch-dependent state), so interpolating all matched values at once is
        equivalent to per-batch interpolation. The assignment back via the same
        mask restores the correct positions.

        Args:
            progression: ``batch_shape x n``-dim progression values.
            task_idcs: ``batch_shape x n``-dim integer task indices.
            batch_shape: Leading batch dimensions.

        Returns:
            ``batch_shape x n x num_curves``-dim basis matrix.
        """
        n = progression.shape[-1]
        U = torch.zeros(
            *batch_shape,
            n,
            self.num_curves,
            device=progression.device,
            dtype=progression.dtype,
        )
        _validate_task_indices(task_idcs, self.num_tasks)
        for t in range(self.num_tasks):
            mask = task_idcs == t
            if mask.any():
                U_t = self.interpolants[t](progression[mask])  # num_curves x k
                U[mask] = U_t.T  # k x num_curves
        return U

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **kwargs,
    ) -> Tensor:
        """Computes the kernel matrix k(x1, x2).

        Args:
            x1: `batch_shape x n1 x (d+1)`-dim Tensor with task indices.
            x2: `batch_shape x n2 x (d+1)`-dim Tensor with task indices.
            diag: If True, only returns the diagonal of the kernel matrix.

        Returns:
            A `batch_shape x n1 x n2`-dim Tensor of kernel values.
        """

        # Extract task indices and progression values
        prog_1, task_idcs_1 = _extract_progression_and_task(x1, self.task_feature)
        prog_2, task_idcs_2 = _extract_progression_and_task(x2, self.task_feature)

        # Handle case where there's no batch dimension
        squeeze_output = prog_1.ndim == 1
        if squeeze_output:
            prog_1, prog_2 = prog_1.unsqueeze(0), prog_2.unsqueeze(0)
            task_idcs_1, task_idcs_2 = (
                task_idcs_1.unsqueeze(0),
                task_idcs_2.unsqueeze(0),
            )

        batch_shape = prog_1.shape[:-1]

        # Compute basis matrices via per-task interpolation
        U1 = self._compute_basis(prog_1, task_idcs_1, batch_shape)
        U2 = U1 if x2 is x1 else self._compute_basis(prog_2, task_idcs_2, batch_shape)

        # compute_sample_covariance expects: ... x num_curves x n
        U1_t = U1.transpose(-2, -1)
        U2_t = None if x2 is x1 else U2.transpose(-2, -1)

        K = compute_sample_covariance(
            U1=U1_t,
            U2=U2_t,
            num_curves=self.num_curves,
            diag=diag,
            correction=self.correction,
        )

        # Remove the batch dimension we added if needed
        if squeeze_output:
            K = K.squeeze(0)

        return K


class MultiTaskEmpiricalOneDimensionalGP(ExactGP, MultiTaskGPyTorchModel):
    """Multi-task Empirical One-Dimensional GP model.

    This model implements a multi-task GP using the "long format" data
    representation where inputs include a task feature column. It uses
    historical one-dimensional curves to define both the mean function and the
    covariance structure.

    Unlike `MultiOutputEmpiricalOneDimensionalGP` which assumes all outputs
    are observed for all inputs, this model supports partial observations
    where different tasks can have different observed input locations.

    Supports heterogeneous historical data where each task can have:
    - Different progression domains
    - Different numbers of progression points
    - But the same number of curves (required for cross-task covariance)

    Example:
        >>> # Heterogeneous historical data: 2 tasks with different domains
        >>> historical_Xs = [
        ...     torch.linspace(0, 1, 100).unsqueeze(-1),   # Task 0: 100 points
        ...     torch.linspace(0.2, 0.8, 50).unsqueeze(-1),  # Task 1: 50 points
        ... ]
        >>> historical_Ys = [
        ...     torch.randn(50, 100),  # Task 0: 50 curves x 100 points
        ...     torch.randn(50, 50),   # Task 1: 50 curves x 50 points
        ... ]
        >>>
        >>> # Training data in long format: 20 observations with task indices
        >>> train_X = torch.cat([
        ...     torch.cat([torch.rand(10, 1), torch.zeros(10, 1)], dim=-1),
        ...     torch.cat([torch.rand(10, 1), torch.ones(10, 1)], dim=-1),
        ... ], dim=0)  # (20, 2)
        >>> train_Y = torch.randn(20, 1)
        >>>
        >>> model = MultiTaskEmpiricalOneDimensionalGP(
        ...     train_X=train_X,
        ...     train_Y=train_Y,
        ...     task_feature=-1,
        ...     historical_Xs=historical_Xs,
        ...     historical_Ys=historical_Ys,
        ... )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        historical_Xs: list[Tensor] | None = None,
        historical_Ys: list[Tensor] | None = None,
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        correction: int = 0,
        mean_module: MultiTaskEmpiricalOneDimensionalMean | None = None,
        covar_module: MultiTaskEmpiricalOneDimensionalKernel | None = None,
        base_covar_module: Kernel | None = None,
    ) -> None:
        """Instantiates a multi-task empirical one-dimensional GP model.

        Args:
            train_X: `n x (d + 1)`-dim Tensor of training inputs, where one
                column contains the task feature (specified by `task_feature`).
                The task indices in train_X can be a subset of the tasks
                defined in the historical data or pre-built modules.
            train_Y: `n x 1`-dim Tensor of training observations.
            task_feature: The index of the task feature column in train_X.
                Can be negative (e.g., -1 for last column).
            historical_Xs: List of m tensors, where historical_Xs[t] is
                `num_progression_t x 1` for task t. Each task can have a
                different number of progression points and different domain.
                Required when mean_module and covar_module are not provided.
            historical_Ys: List of m tensors, where historical_Ys[t] is
                `num_curves x num_progression_t` for task t. All tasks must
                have the same num_curves (curves are paired across tasks).
                Required when mean_module and covar_module are not provided.
            train_Yvar: `n x 1`-dim Tensor of observation noise variances.
                If None, a homoskedastic noise model is used.
            output_tasks: A list of task indices for which to compute model
                outputs. If None, uses all tasks from the historical data.
            correction: Degrees of freedom correction for sample covariance.
                Default is 0 (ML estimate). Use 1 for Bessel correction.
                Only used when building covar_module from historical data.
            mean_module: Optional pre-built mean module. If provided,
                historical_Xs/historical_Ys are not required for the mean.
                This enables reusing modules across multiple GP instances
                (e.g., conditioning on different experiments' observations).
            covar_module: Optional pre-built covariance module. Must be
                provided together with mean_module.
            base_covar_module: Optional base kernel added to the empirical kernel
                as ``K_empirical + K_base`` (see
                :class:`~botorch.models.empirical_gps.empirical_1d_gp.BaseAugmentedEmpiricalKernel`).
                Compose it with a task/index kernel (e.g.
                ``ProductKernel(MaternKernel(), IndexKernel(...))``) to add
                cross-task structure. Its parameters are fit by MLL per the
                ``requires_grad`` flags the caller sets; None (default) uses the
                pure empirical kernel.
        """
        # Determine mean/covar modules and num_tasks from either pre-built
        # modules or historical data
        has_modules = mean_module is not None or covar_module is not None
        has_historical = historical_Xs is not None and historical_Ys is not None

        if has_modules:
            if mean_module is None or covar_module is None:
                raise ValueError(
                    "mean_module and covar_module must be provided together, "
                    "or both omitted."
                )
            if mean_module.num_tasks != covar_module.num_tasks:
                raise ValueError(
                    f"mean_module.num_tasks ({mean_module.num_tasks}) != "
                    f"covar_module.num_tasks ({covar_module.num_tasks})."
                )
            if correction != 0:
                raise ValueError(
                    "correction is ignored when covar_module is pre-built. "
                    "Set correction when constructing the covar_module instead."
                )
            num_tasks_historical = mean_module.num_tasks
            num_curves = covar_module.num_curves
        elif has_historical:
            num_curves = _validate_heterogeneous_historical_data(
                historical_Xs, historical_Ys
            )
            num_tasks_historical = len(historical_Ys)
        else:
            raise ValueError(
                "Either (historical_Xs, historical_Ys) or "
                "(mean_module, covar_module) must be provided."
            )

        # Normalize task_feature to positive index
        d_plus_1 = train_X.shape[-1]
        if task_feature < 0:
            task_feature = task_feature + d_plus_1

        # Validate d=1 (this model only supports 1D progression)
        if d_plus_1 != 2:
            raise ValueError(
                f"MultiTaskEmpiricalOneDimensionalGP requires d=1 "
                f"(input should have 2 columns: progression + task), "
                f"got {d_plus_1} columns."
            )

        self._task_feature = task_feature
        self.num_non_task_features = d_plus_1 - 1

        # Get all tasks from training data — train_X tasks can be a
        # SUBSET of historical tasks (e.g., conditioning on only
        # downscaled observations to predict full-scale). Route through the
        # validated helper so labels are rounded/checked (not silently
        # truncated), consistent with the mean/kernel forward paths.
        _, task_idcs = _extract_progression_and_task(train_X, task_feature)
        all_tasks = task_idcs.unique(sorted=True).tolist()

        # num_tasks reflects ALL tasks defined in the prior, not just
        # the tasks observed in train_X
        self.num_tasks = num_tasks_historical

        # Validate that all observed task indices are valid
        if any(t < 0 or t >= num_tasks_historical for t in all_tasks):
            raise ValueError(
                f"Task indices in train_X must be in "
                f"[0, {num_tasks_historical - 1}], got {all_tasks}."
            )

        if output_tasks is None:
            # Default to ALL historical tasks, not just observed tasks,
            # so predictions cover the full set of tasks
            output_tasks = list(range(num_tasks_historical))
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        # Squeeze output dimension for ExactGP
        train_Y_squeezed = train_Y.squeeze(-1)

        # Set up likelihood
        if train_Yvar is not None:
            train_Yvar_squeezed = train_Yvar.squeeze(-1)
            likelihood = FixedNoiseGaussianLikelihood(
                noise=train_Yvar_squeezed, learn_additional_noise=False
            )
        else:
            likelihood = GaussianLikelihood()

        # Initialize ExactGP
        super().__init__(train_X, train_Y_squeezed, likelihood)

        # Ensure the likelihood matches the data device/dtype. A user-provided or
        # default-constructed GaussianLikelihood is created on CPU, which would
        # otherwise mismatch CUDA data in posterior()/forward().
        self.likelihood.to(device=train_X.device, dtype=train_X.dtype)

        # Store num_curves for reference
        self.num_curves = num_curves

        # Set up mean and covariance modules (either pre-built or from data)
        if has_modules:
            self.mean_module = mean_module
            self.covar_module = covar_module
        else:
            self.mean_module = MultiTaskEmpiricalOneDimensionalMean(
                historical_Xs=historical_Xs,
                historical_Ys=historical_Ys,
                task_feature=task_feature,
            )
            self.covar_module = MultiTaskEmpiricalOneDimensionalKernel(
                historical_Xs=historical_Xs,
                historical_Ys=historical_Ys,
                task_feature=task_feature,
                num_curves=num_curves,
                correction=correction,
            )

        # Optional additive base kernel (e.g. a pre-fit multi-task RBF x index
        # kernel): K = K_empirical + K_base.
        if base_covar_module is not None:
            self.covar_module = BaseAugmentedEmpiricalKernel(
                self.covar_module, base_covar_module
            )

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Computes the GP prior distribution at input locations x.

        Args:
            x: `batch_shape x n x (d+1)`-dim Tensor of input locations,
                including the task feature column.

        Returns:
            A MultivariateNormal distribution over the outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)

    def _split_inputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extracts features before task feature, task indices, and features after.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.

        Returns:
            3-element tuple of (features_before, task_indices, features_after).
        """
        # Overrides MultiTaskGPyTorchModel._split_inputs for base-class API
        # symmetry. This model's own forward/posterior use
        # _extract_progression_and_task (which rounds task labels), so this
        # override is reached only via the inherited base-class API.
        batch_shape = x.shape[:-2]
        task_idcs = x[..., self._task_feature].view(batch_shape + torch.Size([-1, 1]))
        task_idcs = self._map_tasks(task_idcs.round().long())  # round, not truncate
        return (
            x[..., : self._task_feature],
            task_idcs,
            x[..., (self._task_feature + 1) :],
        )

    def _map_tasks(self, task_values: Tensor) -> Tensor:
        """Identity mapping — required by MultiTaskGPyTorchModel.posterior."""
        return task_values

    @classmethod
    def from_homogeneous_data(
        cls,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        historical_X: Tensor,
        historical_Y: Tensor,
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        correction: int = 0,
    ) -> "MultiTaskEmpiricalOneDimensionalGP":
        """Construct model from homogeneous historical data (shared domain).

        This is a convenience method for when all tasks share the same
        progression domain. It converts the 3D tensor format to list format.

        Args:
            train_X: `n x (d + 1)`-dim Tensor of training inputs, where one
                column contains the task feature (specified by `task_feature`).
            train_Y: `n x 1`-dim Tensor of training observations.
            task_feature: The index of the task feature column in train_X.
                Can be negative (e.g., -1 for last column).
            historical_X: `num_progression x 1`-dim Tensor of historical
                progression values (shared across all tasks).
            historical_Y: `num_curves x num_progression x m`-dim Tensor of
                historical curves, where `m` is the number of tasks.
            train_Yvar: `n x 1`-dim Tensor of observation noise variances.
                If None, a homoskedastic noise model is used.
            output_tasks: A list of task indices for which to compute model
                outputs. If None, uses all tasks inferred from training data.
            correction: Degrees of freedom correction for sample covariance.
                Default is 0 (ML estimate). Use 1 for Bessel correction.

        Returns:
            A MultiTaskEmpiricalOneDimensionalGP model.

        Example:
            >>> # Historical data: 50 curves, 100 progression points, 2 tasks
            >>> historical_X = torch.linspace(0, 1, 100).unsqueeze(-1)
            >>> historical_Y = torch.randn(50, 100, 2)
            >>>
            >>> # Training data in long format
            >>> train_X = torch.cat([
            ...     torch.cat([torch.rand(10, 1), torch.zeros(10, 1)], dim=-1),
            ...     torch.cat([torch.rand(10, 1), torch.ones(10, 1)], dim=-1),
            ... ], dim=0)  # (20, 2)
            >>> train_Y = torch.randn(20, 1)
            >>>
            >>> model = MultiTaskEmpiricalOneDimensionalGP.from_homogeneous_data(
            ...     train_X=train_X,
            ...     train_Y=train_Y,
            ...     task_feature=-1,
            ...     historical_X=historical_X,
            ...     historical_Y=historical_Y,
            ... )
        """
        if historical_Y.ndim != 3:
            raise ValueError(
                f"historical_Y must be 3-dim (num_curves x num_progression x m), "
                f"got {historical_Y.ndim}-dim."
            )

        # Convert 3D tensor to list format
        m = historical_Y.shape[-1]
        # All tasks share the same domain tensor (read-only, not mutated).
        historical_Xs = [historical_X] * m
        historical_Ys = [historical_Y[..., t] for t in range(m)]

        return cls(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
            train_Yvar=train_Yvar,
            output_tasks=output_tasks,
            correction=correction,
        )

    @classmethod
    def from_wide_format(
        cls,
        train_X: Tensor,
        train_Y: Tensor,
        historical_Xs: list[Tensor],
        historical_Ys: list[Tensor],
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        correction: int = 0,
    ) -> "MultiTaskEmpiricalOneDimensionalGP":
        """Construct a MultiTaskEmpiricalOneDimensionalGP from wide-format data.

        This is a convenience method for constructing the model when all tasks
        are observed at all input locations (the "block design" case). It
        converts wide-format training data to long-format internally.

        Note: "Wide format" refers to the training data format, not the
        historical data format. Historical data uses the standard list format.

        Args:
            train_X: `n x d`-dim Tensor of training inputs (without task column).
            train_Y: `n x m`-dim Tensor of training observations, where `m` is
                the number of tasks.
            historical_Xs: List of m tensors, where historical_Xs[t] is
                `num_progression_t x 1` for task t.
            historical_Ys: List of m tensors, where historical_Ys[t] is
                `num_curves x num_progression_t` for task t.
            train_Yvar: `n x m`-dim Tensor of observation noise variances.
                If None, a homoskedastic noise model is used.
            output_tasks: A list of task indices for which to compute model
                outputs. If None, uses all tasks.
            correction: Degrees of freedom correction for sample covariance.
                Default is 0 (ML estimate). Use 1 for Bessel correction.

        Returns:
            A MultiTaskEmpiricalOneDimensionalGP model.

        Example:
            >>> # Heterogeneous historical data
            >>> historical_Xs = [
            ...     torch.linspace(0, 1, 100).unsqueeze(-1),
            ...     torch.linspace(0, 1, 80).unsqueeze(-1),
            ... ]
            >>> historical_Ys = [
            ...     torch.randn(50, 100),
            ...     torch.randn(50, 80),
            ... ]
            >>>
            >>> # Wide-format training data: 10 inputs, 2 tasks
            >>> train_X = torch.rand(10, 1)  # 10 x 1 (progression values only)
            >>> train_Y = torch.randn(10, 2)  # 10 x 2 (both tasks observed)
            >>>
            >>> model = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(
            ...     train_X=train_X,
            ...     train_Y=train_Y,
            ...     historical_Xs=historical_Xs,
            ...     historical_Ys=historical_Ys,
            ... )
        """
        n = train_X.shape[-2]
        m = train_Y.shape[-1]

        # Validate shapes
        if train_Y.shape[-2] != n:
            raise ValueError(
                f"train_X has {n} inputs but train_Y has {train_Y.shape[-2]} rows."
            )
        if train_Yvar is not None and train_Yvar.shape != train_Y.shape:
            raise ValueError(
                f"train_Yvar shape {train_Yvar.shape} must match "
                f"train_Y shape {train_Y.shape}."
            )
        if len(historical_Ys) != m:
            raise ValueError(
                f"train_Y has {m} tasks but historical_Ys has {len(historical_Ys)} "
                f"tasks."
            )

        # Convert to long format
        # Repeat each input m times (once for each task)
        # Ordering: (x_0, task_0), (x_0, task_1), ..., (x_0, task_{m-1}),
        #           (x_1, task_0), ...
        train_X_repeated = train_X.repeat_interleave(m, dim=-2)  # (n*m) x d

        # Create task indices: [0, 1, ..., m-1, 0, 1, ..., m-1, ...]
        task_indices = torch.arange(m, device=train_X.device, dtype=train_X.dtype)
        task_indices = task_indices.repeat(n).unsqueeze(-1)  # (n*m) x 1

        # Concatenate to form long-format X with task feature as last column
        train_X_long = torch.cat(
            [train_X_repeated, task_indices], dim=-1
        )  # (n*m) x (d+1)

        # Flatten Y: interleave tasks for each input
        # train_Y is n x m, we want (n*m) x 1 in order:
        # y_{0,0}, y_{0,1}, ..., y_{0,m-1}, y_{1,0}, ...
        train_Y_long = train_Y.reshape(-1, 1)  # (n*m) x 1

        # Flatten Yvar similarly if provided
        train_Yvar_long = None
        if train_Yvar is not None:
            train_Yvar_long = train_Yvar.reshape(-1, 1)  # (n*m) x 1

        return cls(
            train_X=train_X_long,
            train_Y=train_Y_long,
            task_feature=-1,  # Task feature is the last column
            historical_Xs=historical_Xs,
            historical_Ys=historical_Ys,
            train_Yvar=train_Yvar_long,
            output_tasks=output_tasks,
            correction=correction,
        )
