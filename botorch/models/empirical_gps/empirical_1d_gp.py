#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Empirical one-dimensional Gaussian Process models.

These models use a collection of historical one-dimensional curves to define an
empirical prior mean and covariance for a ``SingleTaskGP``. They support both
single-output and batch-independent multi-output modeling.

References

.. [lin2026empirical]
    J. A. Lin, S. Ament, L. C. Tiao, D. Eriksson, M. Balandat, and E. Bakshy.
    Empirical Gaussian Processes. International Conference on Machine Learning
    (ICML), 2026. https://arxiv.org/abs/2602.12082
"""

from __future__ import annotations

from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps.utils import (
    center_curves,
    compute_basis_matrix,
    compute_sample_covariance,
    compute_svd_basis_vectors,
    extract_slice_for_interp,
    instantiate_ard,
    LinearInterpolation1D,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import Mean
from torch import Tensor


# =============================================================================
# EmpiricalOneDimensionalGP Model
# =============================================================================


class EmpiricalOneDimensionalGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        historical_X: Tensor,
        historical_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        ard: bool = False,
    ) -> None:
        """Instantiates an empirical one-dimensional GP model.

        This GP uses historical one-dimensional curves to define the prior mean and
        covariance, following [lin2026empirical]_. Supports both single-output and
        multi-output with independent outputs (batch-independent GPs).

        Args:
            train_X: `batch_shape x n x 1`-dim Tensor of training inputs.
            train_Y: `batch_shape x n x m`-dim Tensor of training targets.
            historical_X: `num_progression x 1`-dim Tensor of progression values.
            historical_Y: `num_curves x num_progression x m`-dim Tensor of historical
                curves, where m is the number of outputs (m=1 for single-output).
            train_Yvar: `batch_shape x n x m`-dim Tensor of observation noise.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level if train_Yvar is None, and a
                FixedNoiseGaussianLikelihood with the given noise observations
                if train_Yvar is not None.
            input_transform: Input transform for the model. Not yet supported.
            outcome_transform: Outcome transform for the model. Not yet supported.
            mean_module: Optional custom mean module.
            covar_module: Optional custom covariance module.
            ard: Whether to use Automatic Relevance Determination on the basis.

        Raises:
            ValueError: If historical_Y is not 3-dimensional or if the number of
                outputs in historical_Y does not match train_Y.
            UnsupportedError: If input_transform or outcome_transform is provided.
        """
        # Check for unsupported transforms
        if input_transform is not None:
            raise UnsupportedError(
                "input_transform is not yet supported for EmpiricalOneDimensionalGP."
            )
        if outcome_transform is not None:
            raise UnsupportedError(
                "outcome_transform is not yet supported for EmpiricalOneDimensionalGP."
            )

        # Validate historical_Y is 3D
        if historical_Y.ndim != 3:
            raise ValueError(
                f"Expected historical_Y to be 3-dim (num_curves x num_progression x m),"
                f" got {historical_Y.ndim}-dim."
            )

        # Validate matching number of outputs
        num_outputs_train = train_Y.shape[-1]
        num_outputs_historical = historical_Y.shape[-1]

        if num_outputs_train != num_outputs_historical:
            raise ValueError(
                f"Number of outputs in train_Y ({num_outputs_train}) must match "
                f"historical_Y ({num_outputs_historical})."
            )

        if covar_module is None:
            covar_module = EmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                ard=ard,
            )
        elif not isinstance(covar_module, EmpiricalOneDimensionalKernel):
            raise ValueError(
                "covar_module must be an instance of EmpiricalOneDimensionalKernel."
            )
        elif ard != covar_module.ard:
            raise ValueError("`ard` argument must equal `covar_module.ard`.")

        if mean_module is None:
            mean_module = EmpiricalOneDimensionalMean(
                X_full=historical_X,
                Y_full=historical_Y,
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )


# =============================================================================
# Mean Module
# =============================================================================


class EmpiricalOneDimensionalMean(Mean):
    """Empirical one-dimensional mean function.

    Computes the mean by averaging historical curves and interpolating.
    """

    def __init__(
        self,
        X_full: Tensor,
        Y_full: Tensor,
    ):
        """Instantiates an empirical one-dimensional mean function.

        Args:
            X_full: `num_progression x 1`-dim Tensor of progression values.
            Y_full: `num_curves x num_progression x m`-dim Tensor of historical
                curves, where m is the number of outputs (m=1 for single-output).
        """
        if Y_full.ndim != 3:
            raise ValueError(
                f"Expected Y_full to be 3-dim (num_curves x num_progression x m), "
                f"got {Y_full.ndim}-dim."
            )

        super().__init__()
        self.X_full = X_full  # num_progression x 1

        self.num_outputs = Y_full.shape[-1]

        # Compute mean across curves:
        # num_curves x num_progression x m -> num_progression x m
        # Then transpose to m x num_progression for interpolation
        self.mean_full = Y_full.mean(dim=0).T  # m x num_progression

        self.f = LinearInterpolation1D(
            self.X_full.squeeze(-1),
            self.mean_full,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Computes the mean function at x.

        Args:
            x: Input locations. For single-output, a `batch_shape x n x d`-dim
                Tensor. For multi-output (after the SingleTaskGP transform), a
                `batch_shape x m x n x d`-dim Tensor whose m slices are identical
                (replicated by SingleTaskGP).

        Returns:
            A `batch_shape x n`-dim Tensor for single-output, or a
            `batch_shape x m x n`-dim Tensor for multi-output.
        """
        x_for_interp = extract_slice_for_interp(x, self.num_outputs)

        # Interpolate: result is m x batch_shape x n. Out-of-range inputs are
        # rejected by the interpolant itself (bounds_error=True).
        y = self.f(x_for_interp)

        # Rearrange: move m from position 0 to position -2 → batch_shape x m x n
        y = y.movedim(0, -2)

        # For single-output, squeeze the m=1 dimension
        if self.num_outputs == 1:
            y = y.squeeze(-2)

        return y


# =============================================================================
# Kernel Module
# =============================================================================


class EmpiricalOneDimensionalKernel(Kernel):
    r"""Empirical One-Dimensional Kernel.

    This kernel computes the empirical covariance of one-dimensional curves at given
    progression points by interpolating historical curve data.

    By default, when `num_curves > num_progression` and `ard=False`, the kernel
    uses SVD decomposition to accelerate computation. This reduces complexity
    from O(n1 * num_curves * n2) to O(n1 * r * n2) where r = min(num_curves,
    num_progression).
    """

    ard: bool = False

    def __init__(
        self,
        X_full: Tensor,
        Y_full: Tensor,
        ard: bool = False,
        curve_weights: Tensor | None = None,
        use_svd: bool | None = None,
        correction: int = 0,
    ) -> None:
        """Instantiates an empirical one-dimensional kernel.

        Args:
            X_full: `num_progression x 1`-dim Tensor of progression values.
            Y_full: `num_curves x num_progression x m`-dim Tensor of historical
                curves, where m is the number of outputs (m=1 for single-output).
            ard: Whether to use Automatic Relevance Determination (ARD).
            curve_weights: `num_curves`-dim Tensor of ARD weights.
            use_svd: Whether to use SVD acceleration. If None (default), SVD is
                used when num_curves > num_progression and ard=False. If True or
                False, directly toggles SVD on or off. Note: using ARD on the SVD
                basis implies a different prior than ARD on the original basis.
                This flag explicitly allows both approaches.
            correction: Degree of freedom correction to use for the computation of
                sample covariance, see `compute_sample_covariance` for details.
        """
        if Y_full.ndim != 3:
            raise ValueError(
                f"Expected Y_full to be 3-dim (num_curves x num_progression x m), "
                f"got {Y_full.ndim}-dim."
            )

        super().__init__()

        self.X_full = X_full

        self.num_curves = Y_full.shape[0]
        self.correction = correction
        self.num_outputs = Y_full.shape[-1]
        num_progression = Y_full.shape[1]

        # Center curves across the curve dimension (dim 0)
        _, Y_centered = center_curves(Y_full, curve_dim=0)

        # Reshape for interpolation: m x num_curves x num_progression
        Y_for_interp = Y_centered.movedim(-1, 0)

        # Apply SVD if requested (must be after reshaping for correct batched SVD)
        if use_svd is None:
            # Default: use SVD when num_curves > num_progression and ard is False
            self._use_svd = not ard and self.num_curves > num_progression
        else:
            self._use_svd = use_svd

        if self._use_svd:
            Y_for_interp = compute_svd_basis_vectors(Y_for_interp)
            # After SVD, curve dim becomes r = min(num_curves, num_progression)
            self._effective_num_curves = min(self.num_curves, num_progression)
        else:
            self._effective_num_curves = self.num_curves

        self.f = LinearInterpolation1D(self.X_full.squeeze(-1), Y_for_interp)

        self.Y_full = Y_full
        self.Y_full_centered = Y_centered

        if ard:
            # When using SVD + ARD, apply ARD weights to the SVD basis vectors
            # which have dimension r = min(num_curves, num_progression)
            instantiate_ard(
                obj=self,
                num_curves=self._effective_num_curves,
                curve_weights=curve_weights,
                dtype=Y_full.dtype,
                device=Y_full.device,
            )
        else:
            self.curve_weights = curve_weights
            self.ard = False

    @property
    def use_svd(self) -> bool:
        """A Boolean indicating whether the kernel uses the SVD efficiency technique."""
        return self._use_svd

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Computes the kernel matrix k(x1, x2).

        Args:
            x1: Input tensor. For single-output, `batch_shape x n1 x d`; for
                multi-output, `batch_shape x m x n1 x d` (after
                `multioutput_to_batch_mode_transform` has been applied).
            x2: Input tensor with the same shape structure as x1.
            diag: If True, only returns the diagonal.
            last_dim_is_batch: If True, treats the last dimension as batch.

        Returns:
            A `batch_shape x n1 x n2`-dim covariance matrix for single-output, or
            a `batch_shape x m x n1 x n2`-dim covariance matrix for multi-output.
            If diag=True, returns the diagonal with one fewer trailing dimension.
        """
        if last_dim_is_batch:
            raise NotImplementedError(
                "last_dim_is_batch=True not supported by EmpiricalOneDimensionalKernel."
            )

        # Prepare inputs for interpolation (extracts slice for multi-output)
        x1_for_interp = extract_slice_for_interp(x1, self.num_outputs)
        x2_for_interp = (
            x1_for_interp
            if x2 is x1
            else extract_slice_for_interp(x2, self.num_outputs)
        )

        # Compute basis matrices using shared helper
        Ux1 = compute_basis_matrix(
            f=self.f,
            x=x1_for_interp,
            num_outputs=self.num_outputs,
            curve_weights=self.curve_weights,
        )
        Ux2 = (
            Ux1
            if x2_for_interp is x1_for_interp
            else compute_basis_matrix(
                f=self.f,
                x=x2_for_interp,
                num_outputs=self.num_outputs,
                curve_weights=self.curve_weights,
            )
        )

        # Compute sample covariance
        K = compute_sample_covariance(
            U1=Ux1,
            U2=None if x2_for_interp is x1_for_interp else Ux2,
            num_curves=self.num_curves,
            diag=diag,
            correction=self.correction,
        )

        # Rearrange: move m from position 0 to position -3 → batch_shape x m x n1 x n2
        if diag:
            K = K.movedim(0, -2)  # batch_shape x m x n
        else:
            K = K.movedim(0, -3)  # batch_shape x m x n1 x n2

        # For single-output, squeeze the m=1 dimension
        if self.num_outputs == 1:
            if diag:
                K = K.squeeze(-2)
            else:
                K = K.squeeze(-3)

        return K
