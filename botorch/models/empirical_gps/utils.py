#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utility functions for empirical one-dimensional Gaussian Processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.constraints import NonTransformedInterval
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


# =============================================================================
# Covariance Computation Helpers
# =============================================================================


def compute_orthogonal_basis(A: Tensor, method: str = "svd") -> Tensor:
    """Compute an orthogonal basis for efficient Gram matrix computation.

    For a matrix A with shape (..., m, n) where m >> n, the Gram matrix A.T @ A
    can be computed more efficiently using a compact orthogonal basis.

    The economy SVD gives ``A = U @ diag(S) @ Vh``, where ``U`` is ``(..., m, r)`` with
    orthonormal columns (``U.T @ U = I_r``), ``S`` holds the ``r`` singular values,
    ``Vh`` is ``(..., r, n)`` with orthonormal rows, and ``r = min(m, n)``. Since
    ``U.T @ U = I_r``::

        A.T @ A = (U @ S @ Vh).T @ (U @ S @ Vh)
                = Vh.T @ S @ U.T @ U @ S @ Vh
                = Vh.T @ S^2 @ Vh
                = (S @ Vh).T @ (S @ Vh)

    By using the ``(..., r, n)`` matrix ``S @ Vh`` instead of the ``(..., m, n)`` matrix
    ``A``, we reduce complexity from ``O(k1 * m * k2)`` to ``O(k1 * r * k2)`` when
    computing products like ``A[..., idx1].T @ A[..., idx2]``.

    Two algorithms produce a basis with the same Gram matrix (``B.T @ B = A.T @ A``):

    - ``"svd"`` (default): economy SVD of ``A``. Numerically robust.
    - ``"eigh"``: eigendecomposition of the Gram matrix ``A.T @ A``. This avoids
      forming the ``(m, r)`` left singular vectors, so it is typically faster when
      ``m >> n``, at the cost of squaring the condition number (reduced accuracy for
      the smallest singular values).

    Example:
        >>> A = torch.randn(10000, 50)  # tall matrix with m >> n
        >>> B = compute_orthogonal_basis(A)  # shape (50, 50)
        >>> # B.T @ B equals A.T @ A
        >>> A_batched = torch.randn(3, 10000, 50)  # batched tall matrices
        >>> B_batched = compute_orthogonal_basis(A_batched)  # shape (3, 50, 50)

    Args:
        A: (..., m, n) tensor with arbitrary batch dimensions.
        method: Decomposition to use, either "svd" or "eigh".

    Returns:
        (..., r, n) tensor with r = min(m, n) such that
        B.T @ B = A.T @ A. Its rows are mutually orthogonal but not
        orthonormal (each is scaled by its singular value).
    """
    if method == "svd":
        # Economy SVD: A = U @ diag(S) @ Vh
        # U: (..., m, r), S: (..., r), Vh: (..., r, n) where r = min(m, n)
        _, S, Vh = torch.linalg.svd(A, full_matrices=False)
        # Return S @ Vh, shape (..., r, n)
        return S.unsqueeze(-1) * Vh
    if method == "eigh":
        # A.T @ A = V @ diag(lambda) @ V.T, so B = diag(sqrt(lambda)) @ V.T
        # satisfies B.T @ B = A.T @ A.
        gram = A.transpose(-2, -1) @ A
        eigvals, eigvecs = torch.linalg.eigh(gram)
        # eigh returns ascending eigenvalues; flip to descending (to match SVD)
        # and clamp tiny negative eigenvalues from round-off before the sqrt.
        eigvals = eigvals.flip(-1).clamp_min(0.0)
        eigvecs = eigvecs.flip(-1)
        B = eigvals.sqrt().unsqueeze(-1) * eigvecs.transpose(-2, -1)
        # Truncate to r = min(m, n) rows to match the economy-SVD shape.
        # When m < n the dropped rows have zero eigenvalues, so B.T @ B
        # is unchanged.
        r = min(A.shape[-2], A.shape[-1])
        return B[..., :r, :]
    raise ValueError(f"Unknown method: {method!r}. Must be 'svd' or 'eigh'.")


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

    Computes U1.T @ U2 / (num_curves - correction) (or its diagonal if
    diag=True). ``correction`` defaults to 0, i.e. division by ``num_curves``.

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
        num_curves: Number of curves (or orthogonal basis vectors when a
            reduced basis is used).
        curve_weights: `num_curves`-dim Tensor of ARD weights. If None, initialized
            to ones.
        dtype: Data type for the curve_weights if created.
        device: Device for the curve_weights if created.
    """
    if curve_weights is None:
        curve_weights = Parameter(torch.ones(num_curves, dtype=dtype, device=device))
    elif not isinstance(curve_weights, Parameter):
        # Docstrings advertise a plain Tensor; wrap it so register_parameter
        # (which rejects non-Parameter tensors) does not raise.
        curve_weights = Parameter(
            curve_weights, requires_grad=curve_weights.requires_grad
        )
    obj.register_parameter("curve_weights", curve_weights)
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


# =============================================================================
# Model Setup Helpers
# =============================================================================


def validate_historical_curves_3d(Y: Tensor, name: str = "Y_full") -> None:
    """Validate that a historical-curves tensor is 3-dimensional.

    Empirical one-dimensional GPs expect historical curves shaped as
    ``num_curves x num_progression x m`` (with ``m=1`` for single-output).

    Args:
        Y: The tensor to validate.
        name: Name used in the error message (e.g. ``"Y_full"`` or
            ``"historical_Y"``).

    Raises:
        ValueError: If ``Y`` is not 3-dimensional.
    """
    if Y.ndim != 3:
        raise ValueError(
            f"Expected {name} to be 3-dim (num_curves x num_progression x m), "
            f"got {Y.ndim}-dim."
        )


def validate_no_transforms(
    input_transform: InputTransform | None,
    outcome_transform: OutcomeTransform | None,
    model_name: str,
) -> None:
    """Raise if input/outcome transforms are provided.

    Empirical one-dimensional GP models do not yet support transforms.

    Args:
        input_transform: Input transform argument to validate.
        outcome_transform: Outcome transform argument to validate.
        model_name: Name of the model, used in the error messages.

    Raises:
        UnsupportedError: If either transform is not None.
    """
    if input_transform is not None:
        raise UnsupportedError(
            f"input_transform is not yet supported for {model_name}."
        )
    if outcome_transform is not None:
        raise UnsupportedError(
            f"outcome_transform is not yet supported for {model_name}."
        )


def build_mean_interpolant(
    X_full: Tensor,
    Y_full: Tensor,
) -> tuple[LinearInterpolation1D, int, Tensor]:
    """Build the interpolant for an empirical mean function.

    Averages the historical curves across the curve dimension and constructs a
    1D linear interpolant over the progression values.

    Args:
        X_full: `num_progression x 1`-dim Tensor of progression values.
        Y_full: `num_curves x num_progression x m`-dim Tensor of historical
            curves.

    Returns:
        A tuple ``(f, num_outputs, mean_full)`` where ``f`` is a
        ``LinearInterpolation1D`` over the per-output mean curves, ``num_outputs``
        is ``m``, and ``mean_full`` is the `m x num_progression`-dim mean.
    """
    num_outputs = Y_full.shape[-1]
    # num_curves x num_progression x m -> num_progression x m -> m x num_progression
    mean_full = Y_full.mean(dim=0).T
    f = LinearInterpolation1D(X_full.squeeze(-1), mean_full)
    return f, num_outputs, mean_full


def build_basis_interpolant(
    X_full: Tensor,
    Y_full: Tensor,
    *,
    ard: bool,
    use_svd: bool | None,
    vectorize_outputs: bool,
    method: str = "svd",
) -> tuple[LinearInterpolation1D, int, bool]:
    """Build the basis interpolant shared by empirical one-dimensional kernels.

    Centers the historical curves across the curve dimension, optionally applies
    an orthogonal-basis (SVD) compression, and constructs a 1D linear interpolant
    over the (per-output) basis curves.

    The ``vectorize_outputs`` flag captures the difference between the
    single-output and multi-output kernels:

    - ``vectorize_outputs=False`` (single-output): the SVD threshold is
      ``num_progression`` and the orthogonal basis is computed per output on
      the 3D ``m x num_curves x num_progression`` tensor.
    - ``vectorize_outputs=True`` (multi-output): the SVD threshold is
      ``num_progression * m`` and the orthogonal basis is computed on the
      vectorized 2D ``num_curves x (num_progression * m)`` tensor (to capture
      cross-output correlations) before reshaping back.

    Args:
        X_full: `num_progression x 1`-dim Tensor of progression values.
        Y_full: `num_curves x num_progression x m`-dim Tensor of historical
            curves.
        ard: Whether ARD is enabled. When True, SVD is disabled by default.
        use_svd: Whether to use SVD acceleration. If None, SVD is used when
            ``num_curves > threshold`` and ``ard`` is False.
        vectorize_outputs: See above. Selects the single- vs multi-output setup.
        method: Decomposition passed to ``compute_orthogonal_basis``.

    Returns:
        A tuple ``(f, effective_num_curves, use_svd)`` where ``f`` is a
        ``LinearInterpolation1D`` over the `m x effective_num_curves x
        num_progression` basis, ``effective_num_curves`` is the basis size after
        any SVD compression, and ``use_svd`` is the resolved SVD flag.
    """
    num_curves = Y_full.shape[-3]
    num_progression = Y_full.shape[-2]
    num_outputs = Y_full.shape[-1]

    # Center curves across the curve dimension
    _, Y_centered = center_curves(Y_full, curve_dim=-3)

    threshold = num_progression * num_outputs if vectorize_outputs else num_progression
    if use_svd is None:
        use_svd = not ard and num_curves > threshold

    if use_svd:
        if vectorize_outputs:
            # SVD on the vectorized basis to capture cross-output correlations.
            Y_vectorized = Y_centered.reshape(num_curves, num_progression * num_outputs)
            Y_svd = compute_orthogonal_basis(Y_vectorized, method=method)
            effective_num_curves = Y_svd.shape[0]
            # Reshape back to (r, num_progression, m) then move m to the front.
            Y_for_interp = Y_svd.reshape(
                effective_num_curves, num_progression, num_outputs
            ).movedim(-1, 0)
        else:
            # Per-output SVD on the 3D basis (batched over outputs).
            Y_for_interp = compute_orthogonal_basis(
                Y_centered.movedim(-1, 0), method=method
            )
            # r == min(num_curves, num_progression) (economy SVD), read from the
            # actual basis so this stays correct if the basis size ever changes.
            effective_num_curves = Y_for_interp.shape[-2]
    else:
        effective_num_curves = num_curves
        Y_for_interp = Y_centered.movedim(-1, 0)

    f = LinearInterpolation1D(X_full.squeeze(-1), Y_for_interp)
    return f, effective_num_curves, use_svd


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ExperimentDataset:
    """A single experiment dataset.

    Args:
        X: (n, d) Tensor of input locations.
        Y: (n, m) Tensor of target values.
        Yvar: Optional (n, m) Tensor of observation noise variances. Note: the
            EM routines (``pretrain_em_prior`` / ``EMEmpiricalGaussianProcess``)
            currently use a single shared scalar likelihood noise and do NOT
            consume per-dataset ``Yvar``; it is accepted for forward-compatibility
            and ignored by the EM path.
    """

    X: Tensor
    Y: Tensor
    Yvar: Tensor | None = None


@dataclass
class UniqueInputs:
    """Tracks unique input locations and per-experiment index mappings.

    Args:
        X_all: (N_unique, d) Tensor of unique input locations.
        experiment_indices: List of K tensors, where experiment_indices[i]
            is a (n_i,) tensor of indices into X_all for experiment i's inputs.
        forward_indices: (n_forward,) tensor of indices into X_all for
            the forward input locations. "Forward" refers to the input locations
            `X` passed to the `forward` method of empirical GP models, as opposed to
            the historical inputs contained in `datasets`.
    """

    X_all: Tensor
    experiment_indices: list[Tensor]
    forward_indices: Tensor


def build_unique_inputs(
    datasets: list[ExperimentDataset],
    X_forward: Tensor | None,
) -> UniqueInputs:
    """Build unique input locations with index mappings.

    Combines all experiment inputs and forward inputs, deduplicates them,
    and tracks which indices in X_all correspond to each experiment.

    **Important**: This uses `torch.unique` which performs EXACT equality matching.
    Near-duplicate points (differing by floating-point epsilon) will NOT be merged.

    Args:
        datasets: List of experiment datasets.
        X_forward: Forward input locations, or None if only computing on datasets.
            "Forward" refers to the input locations `X` passed to the `forward`
            method of general empirical GP models, as opposed to the historical inputs
            contained in `datasets`.

    Returns:
        UniqueInputs with X_all and index mappings.

    Raises:
        ValueError: If both datasets is empty and X_forward is None.
    """
    # Stack all inputs
    all_X_list = [d.X for d in datasets]
    if X_forward is not None:
        all_X_list.append(X_forward)

    if len(all_X_list) == 0:
        raise ValueError(
            "Cannot build unique inputs: datasets is empty and X_forward is None. "
            "At least one dataset or X_forward must be provided."
        )

    all_X = torch.cat(all_X_list, dim=0)  # (sum of n_i + n_forward, d)

    # Find unique rows using EXACT equality
    X_all, inverse_indices = torch.unique(all_X, dim=0, return_inverse=True)

    # Build index maps for each experiment
    offset = 0
    experiment_indices = []
    for d in datasets:
        n_i = d.X.shape[0]
        exp_indices = inverse_indices[offset : offset + n_i]
        experiment_indices.append(exp_indices)
        offset += n_i

    # Get indices for forward input (empty tensor if X_forward is None)
    if X_forward is not None:
        forward_indices = inverse_indices[offset:]
    else:
        forward_indices = torch.tensor([], dtype=torch.long, device=X_all.device)

    return UniqueInputs(
        X_all=X_all,
        experiment_indices=experiment_indices,
        forward_indices=forward_indices,
    )


# =============================================================================
# Matrix Utilities
# =============================================================================


def project_psd(A: Tensor, min_eigval: float = 0.0) -> Tensor:
    """Project a symmetric matrix to be positive semi-definite.

    Computes the eigendecomposition and clamps negative eigenvalues to min_eigval.
    This is the minimum-Frobenius-norm projection onto the PSD cone.

    Args:
        A: (N, N) symmetric matrix.
        min_eigval: Minimum eigenvalue to allow (default: 0).

    Returns:
        A_psd: (N, N) symmetric PSD matrix closest to A in Frobenius norm.
    """
    # Eigendecomposition (A should be symmetric, eigh is appropriate)
    eigvals, eigvecs = torch.linalg.eigh(A)

    # Clamp negative eigenvalues
    eigvals_clamped = torch.clamp(eigvals, min=min_eigval)

    # Reconstruct: A_psd = V @ diag(max(λ, 0)) @ V.T
    # Use scaled eigenvectors for efficient computation: (V * λ) @ V.T
    A_psd = (eigvecs * eigvals_clamped) @ eigvecs.T

    # Explicitly symmetrize to counteract numerical asymmetry from matmul
    return 0.5 * (A_psd + A_psd.T)


def trace_matched_shrinkage(
    cov: Tensor, target: Tensor, alpha: float | Tensor
) -> Tensor:
    r"""Blend a covariance toward a trace-matched structured target.

    Returns ``(1 - alpha) * cov + alpha * (tr(cov) / tr(target)) * target``. The
    target is trace-matched to ``cov`` (scaled so ``tr(scaled target) = tr(cov)``)
    so only its *correlation structure* is imposed, not its absolute scale. This
    is the shrinkage blend used by the EM M-step (``covariance_shrinkage``); it is
    equivalent to a trace-matched (empirical-Bayes) Inverse-Wishart-style shrinkage
    of the covariance toward ``target`` -- a MAP-EM update under an IW prior whose
    scale is re-matched to the data each step. (Conditioning-time augmentation
    instead uses an *additive* base kernel -- see ``BaseAugmentedEmpiricalKernel``.)

    Args:
        cov: ``(n, n)`` covariance to shrink.
        target: ``(n, n)`` structured shrinkage target (e.g. a base-kernel gram).
            Its trace is clamped to ``1e-12`` before division, so a (near-)zero-
            trace target degenerates gracefully to returning ``(1 - alpha) * cov``.
        alpha: Shrinkage intensity ``alpha in [0, 1]`` -- a Python float or a
            scalar Tensor. ``alpha = 0`` returns ``cov`` unchanged; ``alpha = 1``
            returns the trace-matched target.

    Returns:
        The ``(n, n)`` blended covariance, in ``cov``'s dtype/device.
    """
    if bool(((torch.as_tensor(alpha) < 0.0) | (torch.as_tensor(alpha) > 1.0)).any()):
        raise ValueError(f"Shrinkage intensity alpha must be in [0, 1]; got {alpha}.")
    tr_ratio = torch.trace(cov) / torch.trace(target).clamp_min(1e-12)
    return (1.0 - alpha) * cov + alpha * tr_ratio * target


# =============================================================================
# EM Prior -> Empirical Basis Curves
# =============================================================================


def em_prior_to_basis_curves(
    mu: Tensor,
    Sigma: Tensor,
    num_modes: int | None = None,
    correction: int = 0,
    eigenvalue_tol: float = 0.0,
) -> Tensor:
    """Synthesize empirical basis curves that reproduce an EM prior on a grid.

    Given an EM-learned prior mean ``mu`` and covariance ``Sigma`` defined on a
    grid of ``n`` progression points, this builds a small set of synthetic
    historical curves ``Y_full`` such that ``EmpiricalOneDimensionalGP`` (which
    forms a *centered* sample covariance of its curves) reproduces ``(mu, Sigma)``
    exactly at the grid points. This lets the cheaper empirical 1D GP act as a
    surrogate for the full EM prior.

    The construction is a Karhunen-Loeve expansion of ``Sigma`` encoded through a
    simplex so that the curves' cross-curve mean is exactly ``mu`` and their
    centered scatter is exactly ``Sigma``. Because the empirical kernel centers
    the curves (removing one degree of freedom), a rank-``r`` covariance requires
    ``r + 1`` curves -- not ``r``. With ``S`` an ``r x (r+1)`` matrix of
    orthonormal rows orthogonal to the all-ones vector (``S Sᵀ = I_r``,
    ``S 1 = 0``), the per-curve deviations are the columns of
    ``U Λ^{1/2} (√(r + 1 - correction)) S`` where ``Σ = U Λ Uᵀ``; the centered
    scatter is then ``(r + 1 - correction) Σ`` and the kernel's division by
    ``num_curves - correction`` recovers ``Sigma``.

    Args:
        mu: ``(n,)`` EM-learned mean at the grid points.
        Sigma: ``(n, n)`` EM-learned covariance at the grid points.
        num_modes: Cap on the number of leading eigenmodes retained; the
            effective rank is ``r = min(num_modes, #{eigenvalues >
            eigenvalue_tol})``. ``None`` (default) keeps every mode above
            ``eigenvalue_tol``, reproducing ``Sigma`` exactly. A smaller value
            yields a rank-``r`` approximation (fewer, cheaper curves).
        correction: Degrees-of-freedom correction matching the
            ``EmpiricalOneDimensionalGP`` / kernel ``correction`` the curves will
            be used with (the kernel divides by ``num_curves - correction``).
            Defaults to 0.
        eigenvalue_tol: Eigenvalues ``<= eigenvalue_tol`` are dropped as numerical
            noise / null space (default: 0.0).

    Returns:
        ``Y_full`` of shape ``(r + 1, n, 1)`` -- ``r + 1`` synthetic single-output
        curves on the grid -- suitable as ``historical_Y`` for
        ``EmpiricalOneDimensionalGP`` (paired with the grid as ``historical_X``).

    Raises:
        ValueError: If ``Sigma`` has no eigenvalues above ``eigenvalue_tol`` (the
            prior is degenerate, so no covariance structure can be encoded).
    """
    if (
        mu.dim() != 1
        or Sigma.dim() != 2
        or Sigma.shape[-1] != Sigma.shape[-2]
        or mu.shape[-1] != Sigma.shape[-1]
    ):
        raise ValueError(
            "Expected mu of shape (n,) and Sigma of shape (n, n) with matching n; "
            f"got mu {tuple(mu.shape)} and Sigma {tuple(Sigma.shape)}."
        )
    tkwargs = {"dtype": Sigma.dtype, "device": Sigma.device}
    # Match mu to Sigma's dtype/device so the mu + deviations sum below is safe.
    mu = mu.to(**tkwargs)

    # Eigendecomposition of the (symmetric PSD) covariance; eigh returns ascending.
    evals, evecs = torch.linalg.eigh(0.5 * (Sigma + Sigma.transpose(-1, -2)))
    # Sort descending so num_modes keeps the dominant variance directions.
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]

    if num_modes is not None:
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1 if specified; got {num_modes}.")
        evals = evals[:num_modes]
        evecs = evecs[:, :num_modes]

    keep = evals > eigenvalue_tol
    evals = evals[keep]
    evecs = evecs[:, keep]

    r = int(evals.numel())
    if r == 0:
        raise ValueError(
            "Sigma has no eigenvalues above eigenvalue_tol; the prior is "
            "degenerate and cannot be encoded as empirical basis curves."
        )

    num_curves = r + 1
    if correction >= num_curves:
        raise ValueError(
            f"correction ({correction}) must be < the number of synthesized "
            f"curves (r + 1 = {num_curves}); otherwise the curve scale "
            "sqrt(num_curves - correction) is undefined."
        )

    # S: (r, r + 1) with orthonormal rows orthogonal to the all-ones vector.
    # The complete QR of the ones vector gives an orthonormal basis whose first
    # column is proportional to ones; the remaining r columns span its complement.
    ones = torch.ones(num_curves, 1, **tkwargs)
    Q, _ = torch.linalg.qr(ones, mode="complete")  # (r+1, r+1)
    S = Q[:, 1:].transpose(-1, -2)  # (r, r+1): S Sᵀ = I_r, S 1 = 0

    scale = float(num_curves - correction) ** 0.5
    # Deviations: (n, r+1) = U Λ^{1/2} (scale · S)
    deviations = (evecs * evals.clamp_min(0.0).sqrt()) @ (scale * S)

    # Curves: mu + deviation, shaped (num_curves, n, 1) for single-output Y_full.
    Y_full = (mu.unsqueeze(-1) + deviations).transpose(-1, -2).unsqueeze(-1)
    return Y_full
