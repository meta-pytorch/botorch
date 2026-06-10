#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utility functions for empirical one-dimensional Gaussian Processes."""

from __future__ import annotations

from typing import Callable

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.constraints import NonTransformedInterval
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


# =============================================================================
# Covariance Computation Helpers
# =============================================================================


def compute_svd_basis_vectors(A: Tensor) -> Tensor:
    """Compute SVD basis vectors for efficient Gram matrix computation.

    For a matrix A with shape (..., m, n) where m >> n, the Gram matrix A.T @ A
    can be computed more efficiently using the SVD basis vectors.

    Mathematical Background:
        The economy SVD gives A = U @ diag(S) @ Vh where:
        - U: (..., m, r) with orthonormal columns (U.T @ U = I_r)
        - S: (..., r) singular values
        - Vh: (..., r, n) with orthonormal rows
        - r = min(m, n)

        Since U.T @ U = I_r:
            A.T @ A = (U @ S @ Vh).T @ (U @ S @ Vh)
                    = Vh.T @ S @ U.T @ U @ S @ Vh
                    = Vh.T @ S^2 @ Vh
                    = (S @ Vh).T @ (S @ Vh)

        By using the (..., r, n) matrix S @ Vh instead of the (..., m, n) matrix A,
        we reduce complexity from O(k1 * m * k2) to O(k1 * r * k2) when
        computing products like A[..., idx1].T @ A[..., idx2].

    Example:
        >>> A = torch.randn(10000, 50)  # tall matrix with m >> n
        >>> B = compute_svd_basis_vectors(A)  # shape (50, 50)
        >>> # B.T @ B equals A.T @ A
        >>> A_batched = torch.randn(3, 10000, 50)  # batched tall matrices
        >>> B_batched = compute_svd_basis_vectors(A_batched)  # shape (3, 50, 50)

    Args:
        A: (..., m, n) tensor with arbitrary batch dimensions.

    Returns:
        (..., r, n) tensor where r = min(m, n), such that B.T @ B = A.T @ A.
    """
    # Economy SVD: A = U @ diag(S) @ Vh
    # U: (..., m, r), S: (..., r), Vh: (..., r, n) where r = min(m, n)
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Return S @ Vh, shape (..., r, n)
    return S.unsqueeze(-1) * Vh


def center_curves(Y: Tensor, curve_dim: int = -2) -> tuple[Tensor, Tensor]:
    """Center curves by subtracting the mean across the curve dimension.

    This is a common operation used by both mean and kernel modules.

    Args:
        Y: Tensor containing curves. The curve dimension contains different
            realizations to average over.
        curve_dim: Dimension containing the curves to average over.

    Returns:
        A tuple `(mean, centered)` where `mean` is the average across curves
        (with `curve_dim` removed) and `centered` is `Y` minus that mean,
        broadcast back to the shape of `Y`.
    """
    mean = Y.mean(dim=curve_dim, keepdim=True)
    centered = Y - mean
    return mean.squeeze(curve_dim), centered


def compute_sample_covariance(
    U1: Tensor,
    U2: Tensor | None,
    num_curves: int,
    diag: bool = False,
    correction: int = 0,
) -> Tensor:
    """Compute sample covariance from basis matrices.

    Computes: U1.T @ U2 / (num_curves - 1), or diag(U1.T @ U1) if diag=True.

    Args:
        U1: `... x num_curves x n1`-dim basis matrix.
        U2: `... x num_curves x n2`-dim basis matrix, or None to use U1.
        num_curves: Number of curves (for normalization).
        diag: If True, only compute the diagonal.
        correction: Degrees of freedom correction.

    Returns:
        Covariance matrix of shape `... x n1 x n2`, or diagonal `... x n1` if diag=True.
    """
    if U2 is None:
        U2 = U1

    if diag:
        K = (U1 * U2).sum(dim=-2)
    else:
        K = U1.transpose(-2, -1) @ U2

    if num_curves <= correction:
        raise ValueError(
            f"num_curves ({num_curves}) must be greater than correction ({correction})."
        )

    K = K / (num_curves - correction)
    return K


def extract_slice_for_interp(x: Tensor, num_outputs: int) -> Tensor:
    """Prepare input tensor for interpolation.

    For multi-output (num_outputs > 1), SingleTaskGP replicates X m times at
    dim -3. Since all slices are identical, we extract one slice to avoid
    broadcasting issues during interpolation.

    Also squeezes the trailing dimension if d=1, as required for 1D interpolation.

    Args:
        x: Input tensor. For single-output (m=1), `batch_shape x n x d`; for
            multi-output (m>1), `batch_shape x m x n x d` (replicated by
            SingleTaskGP).

    Returns:
        For d=1: `batch_shape x n` tensor suitable for 1D interpolation.
        For d>1: `batch_shape x n x d` tensor.
    """
    # For multi-output, extract one slice from the m dimension at position -3
    if num_outputs > 1 and x.ndim >= 3 and x.shape[-3] == num_outputs:
        x = x[..., 0, :, :]  # batch_shape x n x d

    # Squeeze trailing dimension if d=1 (for 1D interpolation)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)

    return x


def compute_basis_matrix(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    num_outputs: int,
    curve_weights: Tensor | None = None,
) -> Tensor:
    """Compute the basis matrix U(x) for covariance computation.

    Args:
        f: Interpolation function that takes x and returns interpolated values.
            Returns `m x num_curves x batch_shape x n` where m is num_outputs.
        x: `batch_shape x n`-dim Tensor of input locations (no trailing 1).
        num_outputs: Number of outputs (m). Always >= 1.
        curve_weights: Optional `num_curves`-dim Tensor of ARD weights.

    Returns:
        `m x batch_shape x num_curves x n`-dim Tensor.
    """
    # Interpolate: returns m x num_curves x batch_shape x n
    Ux = f(x)
    Ux = torch.as_tensor(Ux)

    # Move num_curves from position 1 to position -2: m x batch_shape x num_curves x n
    Ux = Ux.movedim(1, -2)

    # Apply ARD weights if present
    if curve_weights is not None:
        Ux = Ux * curve_weights.unsqueeze(-1)

    return Ux


def instantiate_ard(
    obj: Module,
    num_curves: int,
    curve_weights: Tensor | None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> None:
    """Instantiates the curve_weights parameter and constraint.

    Args:
        obj: The object to which to add the parameter and constraint.
        num_curves: Number of curves (or SVD basis vectors when using SVD).
        curve_weights: `num_curves`-dim Tensor of ARD weights. If None, initialized
            to ones.
        dtype: Data type for the curve_weights if created.
        device: Device for the curve_weights if created.
    """
    if curve_weights is None:
        curve_weights = Parameter(torch.ones(num_curves, dtype=dtype, device=device))
    obj.register_parameter("curve_weights", curve_weights)
    # IDEA: Could also apply a softmax so that we are taking a weighted average.
    obj.register_constraint(
        "curve_weights",
        NonTransformedInterval(lower_bound=0.0, upper_bound=torch.inf),
    )
    obj.ard = True


# =============================================================================
# Interpolation Utilities
# =============================================================================


class LinearInterpolation1D(Module):
    """PyTorch module for 1D linear interpolation with device-aware buffers.

    Stores interpolation knots and values as registered buffers, ensuring
    they move with `.to(device)` / `.cuda()` and are included in `state_dict()`.

    Similar to `scipy.interpolate.interp1d` with `kind="linear"`.

    Args:
        x: `n`-dim Tensor of observed input positions (knots).
        y: `batch_size x n`-dim Tensor of observed values at the knots.
        bounds_error: If True, raises a ValueError when x_new is beyond the
            bounds of the input data.
        fill_value: Value to use for points beyond the bounds of the input data.
        assume_sorted: If True, assumes that x is already sorted in ascending
            order. If False (default), x will be sorted and y reordered
            accordingly.

    Note:
        ``assume_sorted`` is a construction-time optimization flag and is not
        stored — after construction, ``_x`` is always sorted.

    Example:
        >>> x = torch.linspace(0, 1, 10)
        >>> y = torch.sin(x).unsqueeze(0)  # 1 x 10
        >>> interp = LinearInterpolation1D(x, y)
        >>> x_new = torch.tensor([0.25, 0.75])
        >>> y_new = interp(x_new)  # 1 x 2
    """

    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        bounds_error: bool = True,
        fill_value: float = torch.nan,
        assume_sorted: bool = False,
    ) -> None:
        """Initialize the interpolant. See the class docstring for argument details."""
        super().__init__()

        if x.ndim != 1:
            raise UnsupportedError(f"Expected x to be 1-dim, but got {x.shape}.")

        if not assume_sorted:
            x_ind = torch.argsort(x)
            x = x[x_ind]
            y = y[..., x_ind]

        self.register_buffer("_x", x)
        self.register_buffer("_y", y)
        self.register_buffer(
            "_bounds_error", torch.tensor(bounds_error, dtype=torch.bool)
        )
        self.register_buffer("_fill_value", torch.tensor(fill_value))

    def forward(self, x_new: Tensor) -> Tensor:
        """Interpolates y at x_new.

        Args:
            x_new: `new_batch_size x m`-dim Tensor of new inputs.

        Returns:
            y_new: `batch_size x new_batch_size x m`-dim Tensor of
                interpolated values.
        """
        return _interp1d_torch(
            x=self._x,
            y=self._y,
            x_new=x_new,
            bounds_error=self._bounds_error.item(),
            fill_value=self._fill_value.item(),
        )


def _interp1d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    x_new: torch.Tensor,
    bounds_error: bool | None = None,
    fill_value: float = torch.nan,
) -> torch.Tensor:
    """
    Torch implementation similar to scipy.interp1d. Supports batched evaluation.

    Args:
        x: ``n``-dim Tensor of observed inputs (sorted, 1D).
        y: ``batch_size x n``-dim Tensor of observed values at ``x``.
        x_new: ``new_batch_size x m``-dim Tensor of query locations.
        bounds_error: If True, raises ``ValueError`` when any value in
            ``x_new`` is outside ``[x.min(), x.max()]``.
        fill_value: Value used for out-of-bounds locations when
            ``bounds_error`` is False.

    Returns:
        ``batch_size x new_batch_size x m``-dim Tensor of interpolated values.
    """
    # dydx is the piecewise-linear slope within each interval of size len(x)-2
    dydx = y.diff(dim=-1) / x.diff(dim=-1)
    # searchsorted gives the point idx to be inserted before
    # -1 gives the interval idx, also the left-point idx to add dydx*dx to
    # Use contiguous tensors for searchsorted to avoid performance warning
    idx = torch.searchsorted(x.contiguous(), x_new.contiguous()) - 1
    # clamp to len(x)-2, as we never extrapolate beyond the last point, and set these
    # values to nan
    idx = torch.clamp(idx, 0, x.shape[-1] - 2)
    # relevant shift in location
    # need to expand the shape of x_new to match the shape of x
    x_expanded = x.expand(x_new.shape[:-1] + x.shape[-1:])
    dx = x_new - torch.gather(x_expanded, -1, idx)
    # add dydx*dx to get y_new
    y_new = y[..., idx] + dydx[..., idx] * dx
    x_min, x_max = x[..., [0, -1]]
    out_of_bounds = (x_new < x_min) | (x_new > x_max)
    if out_of_bounds.any() and bounds_error is not False:
        _interp1d_raise_out_of_bounds_error(x_new, x)

    return torch.where(out_of_bounds, fill_value, y_new)


def _interp1d_raise_out_of_bounds_error(
    x_new: Tensor,
    x: Tensor,
) -> None:
    x_new_min = x_new.min()
    if x_new_min < x.min():
        raise ValueError(
            f"A value ({x_new_min}) in x_new is below the interpolation "
            f"range's minimum value ({x.min()})."
        )
    x_new_max = x_new.max()
    if x_new_max > x.max():
        raise ValueError(
            f"A value ({x_new_max}) in x_new is above the interpolation "
            f"range's maximum value ({x.max()})."
        )
