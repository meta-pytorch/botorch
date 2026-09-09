#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-output Empirical One-Dimensional GP model.

This module contains the MultiOutputEmpiricalOneDimensionalGP model which handles
multiple correlated outputs using empirical cross-output covariance from
historical one-dimensional curves, following [lin2026empirical]_. All outcomes are
assumed to be observed for all inputs.

For the multi-task version that supports partial observations (different tasks
observed at different inputs), see MultiTaskEmpiricalOneDimensionalGP in
multitask_empirical_1d_gp.py.

The `MultiOutputEmpiricalOneDimensionalGP` derives from `ExactGP` and `GPyTorchModel`,
leveraging GPyTorch's posterior inference machinery by vectorizing the multi-output
problem as a single-output problem with `(n*m)` observations.
"""

from __future__ import annotations

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.empirical_gps.empirical_1d_gp import (
    BaseAugmentedEmpiricalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch.models.empirical_gps.utils import (
    build_basis_interpolant,
    compute_basis_matrix,
    compute_sample_covariance,
    instantiate_ard,
    kronecker_factored_covariance,
    validate_historical_curves_3d,
    validate_no_transforms,
)
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.constraints import NonTransformedInterval
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means import Mean
from gpytorch.models import ExactGP
from linear_operator import to_linear_operator
from pyre_extensions import none_throws
from torch import Tensor


# =============================================================================
# Multi-Output Mean Module
# =============================================================================


class MultiOutputEmpiricalOneDimensionalMean(EmpiricalOneDimensionalMean):
    """Empirical learning curve mean function for multi-output models.

    Extends EmpiricalOneDimensionalMean to return a flattened output in interleaved
    format for use with the multi-output kernel.

    The output ordering is: (x_0, t_0), (x_0, t_1), ..., (x_1, t_0), ...
    where x_i are input locations and t_j are output indices.
    """

    def __init__(
        self,
        X_full: Tensor,
        Y_full: Tensor,
    ):
        """Instantiates a multi-output empirical learning curve mean function.

        Args:
            X_full: `num_progression x 1`-dim Tensor of progression values.
            Y_full: `num_curves x num_progression x m`-dim Tensor for multi-output.
        """
        super().__init__(X_full=X_full, Y_full=Y_full)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the flattened mean function at x.

        Args:
            x: `batch_shape x n x 1`-dim Tensor of input locations.

        Returns:
            `batch_shape x (n * m)`-dim Tensor in interleaved format.
        """
        # super().forward returns batch_shape x m x n for m > 1,
        # but batch_shape x n for m == 1 (squeezed)
        y = super().forward(x)

        # Handle single-output case where parent class squeezes m dimension
        if self.num_outputs == 1:
            # For m=1, y is batch_shape x n, already flattened correctly
            return y

        # Transpose to batch_shape x n x m for interleaving
        y = y.movedim(-2, -1)

        # Flatten to batch_shape x (n * m)
        # Creates ordering: (x_0, t_0), (x_0, t_1), ..., (x_1, t_0), ...
        return y.reshape(*y.shape[:-2], -1)


# =============================================================================
# Multi-Output Kernel (Cross-Output Covariance)
# =============================================================================


class MultiOutputEmpiricalOneDimensionalKernel(Kernel):
    """Empirical Learning Curve Kernel for multi-output models.

    This kernel computes the full cross-output covariance based on the empirical
    covariance of historical learning curves. It produces an `(n * m) x (n * m)`
    covariance matrix that captures both input-space covariance and cross-output
    correlations.

    For single-output models (without cross-output covariance), see
    `EmpiricalOneDimensionalKernel` in the main module.

    By default, when `num_curves > num_progression * m` and `ard=False`, the kernel
    uses SVD decomposition to accelerate computation. The SVD is applied to the
    vectorized bases of size `(num_curves, num_progression * m)` to capture
    cross-output correlations in the compressed representation.

    The optional ``learn_cross_output_shrinkage`` flag adds a single scalar
    ``rho in [0, 1]`` that scales the *cross-output* blocks of the covariance,
    leaving the within-output blocks untouched::

        K_rho = K * M,   M = kron(J_n, (1 - rho) I_m + rho J_m)

    ``rho = 1`` recovers the full joint covariance (the default, and exactly the
    behavior when the flag is off), while ``rho = 0`` decouples the outputs into
    ``m`` independent empirical GPs that still share an observation noise. Since
    ``(1 - rho) I_m + rho J_m`` has eigenvalues ``1 - rho`` and ``1 + (m - 1) rho``,
    both non-negative on ``[0, 1]``, ``M`` is PSD and ``K * M`` stays PSD by the
    Schur product theorem -- no eigenvalue repair is needed.

    This matters because the dense cross-output covariance is estimated over an
    ``(n * m)``-dimensional index set: with few historical curves it is badly
    rank-limited, and trusting it fully makes the posterior over-confident.
    Fitting ``rho`` by marginal likelihood lets the data decide how much
    cross-output structure to keep, instead of forcing a choice between the joint
    and independent models.
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
        learn_cross_output_shrinkage: bool = False,
        learn_covariance_ridge: bool = False,
        kronecker_factored: bool = False,
    ) -> None:
        """Instantiates a multi-output empirical learning curve kernel.

        Args:
            X_full: `num_progression x 1`-dim Tensor of progression values.
            Y_full: `num_curves x num_progression x m`-dim Tensor of historical
                learning curves for all `m` outputs.
            ard: Whether to use Automatic Relevance Determination (ARD) on the basis.
            curve_weights: `num_curves`-dim Tensor of ARD weights.
            use_svd: Whether to use SVD acceleration. If None (default), SVD is
                used when num_curves > num_progression * m and ard=False. If True
                or False, directly toggles SVD on or off. The SVD is applied to
                the vectorized bases of size (num_curves, num_progression * m) to
                preserve cross-output correlations.
            correction: Bessel-style denominator correction for the empirical
                covariance (divide by ``num_curves - correction``); defaults to 0
                (divide by ``num_curves``).
            learn_covariance_ridge: If True, register a positive scalar nugget
                added to the empirical covariance where inputs coincide. The
                empirical covariance is ``U^T U / N``, so its rank is capped by
                the number of historical curves: with 25 curves on a 50-dim
                index set, 26 directions have *exactly zero* prior variance and
                the model is infinitely confident in them. A ridge shifts every
                eigenvalue up, lifting the null space while barely perturbing
                well-estimated directions. Note a scalar *multiplier* cannot do
                this -- scaling leaves a zero eigenvalue at zero.
            learn_cross_output_shrinkage: If True, register a scalar ``rho`` in
                ``[0, 1]`` (initialized to 1, i.e. a no-op) that scales the
                cross-output covariance blocks and is fit by marginal likelihood.
                Defaults to False, which leaves the kernel exactly unchanged.
        """
        super().__init__()
        validate_historical_curves_3d(Y_full)

        self.num_outputs = Y_full.shape[-1]
        self.num_curves = Y_full.shape[-3]
        if correction >= self.num_curves:
            raise ValueError(
                f"correction ({correction}) must be < num_curves ({self.num_curves})."
            )
        self.correction = correction

        # Center curves, optionally SVD-compress the vectorized cross-output
        # basis (to preserve cross-output correlations), and build the
        # interpolant over the m x effective_num_curves x num_progression basis.
        self.f, self._effective_num_curves, self._use_svd = build_basis_interpolant(
            X_full=X_full,
            Y_full=Y_full,
            ard=ard,
            use_svd=use_svd,
            vectorize_outputs=True,
        )

        if ard:
            # Apply ARD weights to the (possibly SVD-compressed) basis
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

        # rho is registered only on request so that existing models keep an
        # unchanged parameter set (and unchanged state_dict keys).
        if learn_cross_output_shrinkage:
            self.register_parameter(
                "rho",
                torch.nn.Parameter(
                    torch.ones(1, dtype=Y_full.dtype, device=Y_full.device)
                ),
            )
            # NonTransformedInterval (rather than a sigmoid Interval) so that the
            # endpoints are attainable: rho = 1 must exactly reproduce the
            # unshrunk kernel, and rho = 0 the fully independent one.
            self.register_constraint(
                "rho", NonTransformedInterval(0.0, 1.0, initial_value=1.0)
            )
        else:
            self.rho = None

        # Kronecker factorization of the empirical covariance. Estimated once
        # here, from the historical curves on their own grid; ``forward`` then
        # rebuilds the progression factor from the *interpolated* basis so the
        # kernel still evaluates at arbitrary progression values.
        self.kronecker_factored = kronecker_factored
        if kronecker_factored:
            _, sigma_output = kronecker_factored_covariance(Y_full)
            self.register_buffer("_sigma_output", sigma_output)
        else:
            self._sigma_output = None

        # The ridge is registered only on request, for the same reason as rho:
        # existing models must keep an identical parameter set.
        if learn_covariance_ridge:
            self.register_parameter(
                "raw_ridge",
                torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
            )
            # Softplus-style positivity with a small floor. Unlike rho the
            # endpoints do not need to be attainable -- ridge = 0 exactly is the
            # unridged kernel, which is already available by not setting the flag.
            self.register_constraint("raw_ridge", GreaterThan(0.0))
        else:
            self.raw_ridge = None

    @property
    def ridge(self) -> Tensor | None:
        """The ridge magnitude, or None when the flag is off."""
        if self.raw_ridge is None:
            return None
        return self.raw_ridge_constraint.transform(self.raw_ridge)

    def _kronecker_covariance(self, Ux1: Tensor, Ux2: Tensor, diag: bool) -> Tensor:
        r"""Separable covariance ``Sigma_progression (x) Sigma_output``.

        The unconstrained estimate ``U^T U / N`` has rank at most ``num_curves``,
        so below ``n * m`` historical curves it leaves directions with exactly
        zero prior variance -- the measured cause of the joint model's
        over-confidence, and the reason it degrades as ``m`` grows. Constraining
        the covariance to a Kronecker product drops the free parameters from
        ``(nm)(nm+1)/2`` to ``n(n+1)/2 + m(m+1)/2``, so the same curves support a
        far better conditioned estimate, and the benefit *grows* with ``m``.

        ``Sigma_output`` is estimated once at construction. The progression factor
        is rebuilt here from the interpolated basis, which is what lets the kernel
        evaluate away from the historical grid:

            Sigma_prog(x1, x2) = 1/(N m) sum_k U_k(x1)^T Sigma_output^-1 U_k(x2)

        and the result is the Kronecker product under the interleaved index
        convention ``flat = i * m + t``.
        """
        m = self.num_outputs
        sigma_output = none_throws(self._sigma_output).to(Ux1.dtype)
        eye_m = torch.eye(m, dtype=Ux1.dtype, device=Ux1.device)
        inv_output = torch.linalg.inv(sigma_output + 1e-8 * eye_m)

        # (num_curves, n, m) from the flattened interleaved basis.
        curves_1 = Ux1.reshape(*Ux1.shape[:-1], -1, m)
        curves_2 = Ux2.reshape(*Ux2.shape[:-1], -1, m)
        sigma_progression = torch.einsum(
            "...kia,ab,...kjb->...ij", curves_1, inv_output, curves_2
        ) / (self.num_curves * m)

        if diag:
            # Diagonal of kron(A, B) is outer(diag(A), diag(B)), flattened in the
            # same interleaved order.
            return (
                sigma_progression.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
                * sigma_output.diagonal()
            ).reshape(*sigma_progression.shape[:-2], -1)
        # kron under flat = i * m + t, i.e. K[i*m+a, j*m+b] = P[i,j] * S[a,b].
        # The intermediate has index order (i, a, j, b), so P must broadcast over
        # the two output axes and S over the two progression axes -- hence the
        # unsqueeze(-2) on S. Without it S aligns against a progression axis and
        # the shapes silently disagree.
        kron = sigma_progression.unsqueeze(-1).unsqueeze(-3) * sigma_output.unsqueeze(
            -2
        )
        return kron.reshape(
            *sigma_progression.shape[:-2],
            sigma_progression.shape[-2] * m,
            sigma_progression.shape[-1] * m,
        )

    def _apply_ridge(self, K: Tensor, x1: Tensor, x2: Tensor, diag: bool) -> Tensor:
        """Add a white-noise nugget ``lambda`` where the inputs coincide.

        **This is equivalent to observation noise for prediction**, and the
        equivalence has been measured rather than assumed. At a test point
        distinct from every training input, a nugget ``lambda`` and an
        observation noise ``sigma^2 = lambda`` give identical posterior means,
        and the nugget's *latent* variance equals the noise model's *observation*
        variance (agreement to ~1e-9 on both). Choose this over likelihood noise
        only when you want the term to be part of the latent process -- e.g. so
        it appears in ``posterior().variance`` without ``observation_noise=True``,
        or so it is shared between a training input and a prediction at that same
        input, which is the one place the two genuinely differ.

        It is *not* a way to get uncertainty that observation noise cannot
        provide.

        Added only where ``x1`` and ``x2`` refer to the same input, so it is a
        genuine white-noise process rather than a constant offset added to every
        entry (which would be a rank-one bias term, not a ridge).
        """
        ridge = self.ridge.to(K.dtype).squeeze(-1).squeeze(-1)
        if diag:
            return K + ridge
        m = self.num_outputs
        # Same flat index requires both the same progression point and the same
        # output, under the interleaved convention flat = i * m + t.
        # x1/x2 arrive already squeezed to (batch..., n), so the comparison
        # broadcasts straight to (batch..., n1, n2).
        same_point = x1.unsqueeze(-1) == x2.unsqueeze(-2)
        same_point = same_point.repeat_interleave(m, dim=-2).repeat_interleave(
            m, dim=-1
        )
        outputs_1 = torch.arange(K.shape[-2], device=K.device) % m
        outputs_2 = torch.arange(K.shape[-1], device=K.device) % m
        same_output = outputs_1.unsqueeze(-1) == outputs_2.unsqueeze(-2)
        return K + ridge * (same_point & same_output).to(K.dtype)

    def _apply_cross_output_shrinkage(self, K: Tensor, n1: int, n2: int) -> Tensor:
        """Scale the cross-output entries of ``K`` by ``rho``.

        ``K`` is indexed by the interleaved flat index ``i * m + t``, so two
        entries belong to the same output exactly when their indices agree mod
        ``m``. Only entries with differing outputs are scaled.
        """
        m = self.num_outputs
        outputs_1 = torch.arange(n1 * m, device=K.device) % m
        outputs_2 = torch.arange(n2 * m, device=K.device) % m
        same_output = outputs_1.unsqueeze(-1) == outputs_2.unsqueeze(-2)
        rho = self.rho.to(K.dtype)
        return torch.where(same_output, K, rho * K)

    @property
    def use_svd(self) -> bool:
        """A Boolean indicating whether the kernel uses the SVD technique."""
        return self._use_svd

    def _compute_flattened_basis_matrix(self, x: Tensor) -> Tensor:
        """Compute the flattened basis matrix U(x) for multi-output covariance.

        Args:
            x: `batch_shape x n`-dim Tensor of input locations (no trailing 1).

        Returns:
            `batch_shape x num_curves x (n * m)`-dim basis matrix where the
            last dimension interleaves inputs and outputs as:
            (x_0, t_0), (x_0, t_1), ..., (x_0, t_{m-1}), (x_1, t_0), ...
        """
        # Use shared helper for interpolation and ARD weighting
        # Returns m x batch_shape x num_curves x n
        Ux = compute_basis_matrix(
            f=self.f,
            x=x,
            num_outputs=self.num_outputs,
            curve_weights=self.curve_weights,
        )

        # Rearrange for interleaving:
        # From: m x batch_shape x num_curves x n
        # To: batch_shape x num_curves x n x m
        Ux = Ux.movedim(0, -1)  # batch_shape x num_curves x n x m

        # Flatten last two dims: batch_shape x num_curves x (n * m)
        # Creates ordering: (x_0, t_0), (x_0, t_1), ..., (x_1, t_0), ...
        Ux = Ux.reshape(*Ux.shape[:-2], -1)

        return Ux

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Computes the kernel matrix k(x1, x2).

        Args:
            x1: `batch_shape x n1 x 1`-dim Tensor.
            x2: `batch_shape x n2 x 1`-dim Tensor.
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported for multi-output kernel.

        Returns:
            `batch_shape x (n1 * m) x (n2 * m)`-dim covariance matrix.
            If diag=True, returns `batch_shape x (n1 * m)`-dim diagonal.
        """
        if last_dim_is_batch:
            raise UnsupportedError(
                "last_dim_is_batch is not supported for "
                "MultiOutputEmpiricalOneDimensionalKernel."
            )

        # Capture input identity BEFORE squeezing: squeeze(-1) returns a fresh
        # view, so testing ``x2 is x1`` after rebinding x1 would always be False
        # and defeat the symmetric fast-path (recomputing Ux2 needlessly).
        same_inputs = x2 is x1
        x1 = x1.squeeze(-1)  # batch_shape x n1
        x2 = x1 if same_inputs else x2.squeeze(-1)

        # Compute flattened basis matrices: batch_shape x num_curves x (n * m)
        Ux1 = self._compute_flattened_basis_matrix(x1)
        Ux2 = Ux1 if x2 is x1 else self._compute_flattened_basis_matrix(x2)

        # Compute sample covariance
        # Always use original num_curves for normalization
        if self.kronecker_factored:
            K = self._kronecker_covariance(Ux1, Ux2, diag=diag)
        else:
            K = compute_sample_covariance(
                U1=Ux1,
                U2=None if x2 is x1 else Ux2,
                num_curves=self.num_curves,
                diag=diag,
                correction=self.correction,
            )

        # The diag path only touches same-index (hence same-output) entries, so
        # it is rho-invariant by construction and must not be rescaled.
        if self.rho is not None and not diag:
            K = self._apply_cross_output_shrinkage(K, x1.shape[-1], x2.shape[-1])

        if self.ridge is not None:
            K = self._apply_ridge(K, x1, x2, diag=diag)

        return K


# =============================================================================
# Vectorized Multi-Output GP using ExactGP and GPyTorchModel (Default)
# =============================================================================


class _VectorizedMeanWrapper(Mean):
    """Wrapper that adapts MultiOutputEmpiricalOneDimensionalMean for expanded input.

    The underlying mean module expects input of shape `n x 1` and returns `n*m`.
    This wrapper handles expanded input of shape `(n*m) x 1` by extracting unique
    values (every m-th element) before passing to the underlying mean.
    """

    def __init__(
        self, base_mean: MultiOutputEmpiricalOneDimensionalMean, num_outputs: int
    ) -> None:
        super().__init__()
        self.base_mean = base_mean
        self._num_outputs = num_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Compute mean at expanded input locations.

        Args:
            x: `batch_shape x (n*m) x 1`-dim Tensor of expanded input locations.

        Returns:
            `batch_shape x (n*m)`-dim Tensor of mean values.
        """
        # Extract unique inputs: every m-th element
        x_unique = x[..., :: self._num_outputs, :]  # batch_shape x n x 1
        # Compute mean on unique inputs, returns batch_shape x (n*m)
        return self.base_mean(x_unique)


class _VectorizedKernelWrapper(Kernel):
    """Wrapper that adapts MultiOutputEmpiricalOneDimensionalKernel for expanded input.

    The underlying kernel expects input of shape `n x 1` and returns `(n*m) x (n*m)`.
    This wrapper handles expanded input of shape `(n*m) x 1` by extracting unique
    values (every m-th element) before passing to the underlying kernel.
    """

    def __init__(
        self, base_kernel: MultiOutputEmpiricalOneDimensionalKernel, num_outputs: int
    ) -> None:
        super().__init__()
        self.base_kernel = base_kernel
        self._num_outputs = num_outputs

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Compute kernel matrix at expanded input locations.

        Args:
            x1: `batch_shape x (n1*m) x 1`-dim Tensor of expanded input locations.
            x2: `batch_shape x (n2*m) x 1`-dim Tensor of expanded input locations.
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported.

        Returns:
            `batch_shape x (n1*m) x (n2*m)`-dim covariance matrix.
        """
        # Extract unique inputs: every m-th element
        x1_unique = x1[..., :: self._num_outputs, :]  # batch_shape x n1 x 1
        if x2 is x1:
            x2_unique = x1_unique
        else:
            x2_unique = x2[..., :: self._num_outputs, :]  # batch_shape x n2 x 1

        # Compute kernel on unique inputs
        return self.base_kernel.forward(
            x1_unique, x2_unique, diag=diag, last_dim_is_batch=last_dim_is_batch
        )


class MultiOutputEmpiricalOneDimensionalGP(ExactGP, GPyTorchModel):
    """Multi-output Empirical Learning Curve GP that leverages GPyTorch's inference.

    This is the default implementation that derives from `ExactGP` and
    `GPyTorchModel`, leveraging GPyTorch's posterior inference machinery.
    The multi-output problem is vectorized as a single-output problem with
    `(n*m)` observations, where the kernel captures cross-output correlations
    in a full `(n*m) x (n*m)` covariance matrix.

    The key insight is that by expanding `train_X` from `n x 1` to `(n*m) x 1`
    (repeating each input m times) and flattening `train_Y` from `n x m` to
    `(n*m) x 1`, we present consistent dimensions to GPyTorch. Wrapper modules
    extract unique values from the expanded input before computing mean/kernel.

    After computing the posterior, the result is reshaped back to multi-output
    format (`q x m`) and returned as a `MultitaskMultivariateNormal`.

    This implementation benefits from:
    - Leveraging GPyTorch's battle-tested posterior computation
    - Automatic caching of Cholesky decomposition in `prediction_strategy`
    - Better integration with BoTorch/GPyTorch features

    Note:
        Transforms (input_transform, outcome_transform) are not yet supported.

    Note:
        The dense cross-output covariance is not always the right trade. It is
        estimated over an ``(n*m)``-dimensional index set from a limited number of
        historical realizations, so it is *more* rank-limited than the ``m``
        independent :class:`EmpiricalOneDimensionalGP` models it competes with.
        Whether the joint model wins depends on how many historical realizations
        you have relative to ``n*m``, how strongly the outputs are actually
        correlated, and the observation budget -- in internal comparisons the
        ordering flipped with the protocol. Adding a fitted additive base kernel
        via ``base_covar_module`` narrows the gap where the joint model trails.

        Prefer this model when you need calibrated *cross-output* covariances,
        which independent models cannot provide at all, or when historical data is
        plentiful relative to ``n*m``. Otherwise benchmark it against independent
        per-output models on your own data rather than assuming either way.
    """

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
        base_covar_module: Kernel | None = None,
        ard: bool = False,
        learn_cross_output_shrinkage: bool = False,
        learn_covariance_ridge: bool = False,
        kronecker_factored: bool = False,
    ) -> None:
        """Instantiates a multi-output empirical learning curve GP model.

        Args:
            train_X: `batch_shape x n x 1`-dim Tensor of training inputs.
            train_Y: `batch_shape x n x m`-dim Tensor of training observations,
                where `m` is the number of outputs.
            historical_X: `num_progression x 1`-dim Tensor of historical progression
                values.
            historical_Y: `num_curves x num_progression x m`-dim Tensor of historical
                learning curves for all outputs.
            train_Yvar: `batch_shape x n x m`-dim Tensor of observation noise
                variances. If None, a homoskedastic noise model is used.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level if train_Yvar is None, and a
                FixedNoiseGaussianLikelihood with the given noise observations
                if train_Yvar is not None.
            input_transform: Input transform for the model. Not yet supported.
            outcome_transform: Outcome transform for the model. Not yet supported.
            mean_module: Optional custom mean module.
            covar_module: Optional custom covariance module.
            base_covar_module: Optional base kernel added to the empirical kernel
                as ``K_empirical + K_base`` (see
                :class:`~botorch.models.empirical_gps.empirical_1d_gp.BaseAugmentedEmpiricalKernel`).
                Note it operates on the expanded ``(n*m, 1)`` progression inputs
                only (no output/index column), so it adds the *same* covariance
                across all outputs and cannot encode output-specific or
                cross-output structure. Its parameters are fit by MLL per the
                caller's ``requires_grad`` flags; None (default) uses the pure
                empirical kernel.
            ard: Whether to use Automatic Relevance Determination on the basis.
            learn_cross_output_shrinkage: If True, fit a scalar ``rho`` in
                ``[0, 1]`` that scales the cross-output covariance blocks by
                marginal likelihood. ``rho = 1`` (the initial value, and the
                behavior when this is False) keeps the full joint covariance;
                ``rho = 0`` decouples the outputs. Useful when the number of
                historical curves is small relative to ``num_progression * m``,
                where the empirical cross-output covariance is rank-limited and
                the joint posterior becomes over-confident. Ignored -- and
                required to agree -- if ``covar_module`` is supplied.

        Raises:
            ValueError: If historical_Y is not 3-dimensional.
            UnsupportedError: If input_transform or outcome_transform is provided.
        """
        # Check for unsupported transforms
        validate_no_transforms(
            input_transform,
            outcome_transform,
            "MultiOutputEmpiricalOneDimensionalGP",
        )

        validate_historical_curves_3d(historical_Y, name="historical_Y")

        n = train_X.shape[-2]
        m = historical_Y.shape[-1]
        self._true_num_outputs = m
        self._num_outputs = m
        batch_shape = train_Y.shape[:-2]

        # Expand X: repeat each x for all m outputs (interleaved)
        # [x1, x1, x1, x2, x2, x2, ...] for m=3
        # This ensures consistent dimensions for GPyTorch's prediction strategy
        train_X_expanded = train_X.repeat_interleave(m, dim=-2)  # (n*m) x 1

        # Flatten Y: interleave outputs
        # [y1_out1, y1_out2, y1_out3, y2_out1, ...]
        train_Y_flat = train_Y.reshape(*batch_shape, n * m)  # (n*m)

        # Flatten Yvar similarly if provided
        train_Yvar_flat = None
        if train_Yvar is not None:
            train_Yvar_flat = train_Yvar.reshape(*batch_shape, n * m)  # (n*m)

        # Set up base mean module if not provided
        if mean_module is None:
            base_mean = MultiOutputEmpiricalOneDimensionalMean(
                X_full=historical_X,
                Y_full=historical_Y,
            )
        else:
            base_mean = mean_module

        # Set up base covariance module if not provided
        if covar_module is None:
            base_kernel = MultiOutputEmpiricalOneDimensionalKernel(
                X_full=historical_X,
                Y_full=historical_Y,
                ard=ard,
                learn_cross_output_shrinkage=learn_cross_output_shrinkage,
                learn_covariance_ridge=learn_covariance_ridge,
                kronecker_factored=kronecker_factored,
            )
        elif not isinstance(covar_module, MultiOutputEmpiricalOneDimensionalKernel):
            raise ValueError(
                "covar_module must be an instance of "
                "MultiOutputEmpiricalOneDimensionalKernel."
            )
        elif ard != covar_module.ard:
            raise ValueError("`ard` argument must equal `covar_module.ard`.")
        elif learn_cross_output_shrinkage != (covar_module.rho is not None):
            raise ValueError(
                "`learn_cross_output_shrinkage` argument must match whether "
                "`covar_module` has a cross-output shrinkage parameter."
            )
        elif learn_covariance_ridge != (covar_module.raw_ridge is not None):
            raise ValueError(
                "`learn_covariance_ridge` argument must match whether "
                "`covar_module` has a covariance ridge parameter."
            )
        elif kronecker_factored != covar_module.kronecker_factored:
            raise ValueError(
                "`kronecker_factored` argument must match "
                "`covar_module.kronecker_factored`."
            )
        else:
            base_kernel = covar_module

        # Set up likelihood if not provided
        if likelihood is None:
            if train_Yvar_flat is not None:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=train_Yvar_flat, learn_additional_noise=False
                )
            else:
                likelihood = GaussianLikelihood()

        # Initialize ExactGP with the expanded train_X and flattened train_Y
        ExactGP.__init__(
            self,
            train_inputs=train_X_expanded,
            train_targets=train_Y_flat,
            likelihood=likelihood,
        )

        # Ensure the likelihood matches the data device/dtype. A user-provided or
        # default-constructed GaussianLikelihood is created on CPU, which would
        # otherwise mismatch CUDA data in posterior()/forward().
        self.likelihood.to(device=train_X.device, dtype=train_X.dtype)

        # Wrap mean and kernel to handle expanded input format
        # Wrappers extract unique values (every m-th element) before computing
        self.mean_module = _VectorizedMeanWrapper(base_mean, num_outputs=m)
        self.covar_module = _VectorizedKernelWrapper(base_kernel, num_outputs=m)

        # Optional additive base kernel that likewise operates on the expanded
        # (n*m, 1) inputs: K = K_empirical + K_base.
        if base_covar_module is not None:
            self.covar_module = BaseAugmentedEmpiricalKernel(
                self.covar_module, base_covar_module
            )

        # Store flattened observation noise for the posterior.
        self._train_Yvar_flat = train_Yvar_flat

        # Store base modules for access
        self._base_mean = base_mean
        self._base_kernel = base_kernel

    @property
    def num_outputs(self) -> int:
        return self._true_num_outputs

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Computes the GP prior distribution at input locations x.

        Args:
            x: `batch_shape x (n*m) x 1`-dim Tensor of expanded input locations.

        Returns:
            A MultivariateNormal distribution over `(n * m)`-dimensional outputs.
        """
        # The wrapped mean and kernel extract unique values before computing
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior:
        """Computes the posterior distribution at the given points.

        This method expands the test inputs, calls GPyTorchModel's posterior
        computation, and reshapes the result to multi-output format.

        Args:
            X: `q x 1`-dim Tensor of input locations.
            output_indices: Not supported; must be None (raises otherwise).
            observation_noise: If a bool, whether to add the model's observation
                noise to the posterior; requires either `train_Yvar` or a
                `GaussianLikelihood` (raises otherwise). If a Tensor, per-point
                noise variances broadcastable to the trailing `(q, m)` grid --
                e.g. a scalar, `(m,)` per-output, `(q, 1)` per-point, or a full
                `... x q x m` tensor.
            posterior_transform: Optional posterior transform.

        Returns:
            A GPyTorchPosterior over a MultitaskMultivariateNormal distribution
            with shape `batch_shape x q x m`.
        """
        self.eval()

        if output_indices is not None:
            raise UnsupportedError(
                "output_indices is not supported by "
                "MultiOutputEmpiricalOneDimensionalGP.posterior."
            )

        q = X.shape[-2]
        m = self._true_num_outputs

        # Expand test X: repeat for all outputs (interleaved)
        # This matches how we expanded train_X
        X_expanded = X.repeat_interleave(m, dim=-2)  # (q*m) x 1

        # Get posterior from parent class (GPyTorchModel)
        # Pass observation_noise=False here - we'll handle it manually to match
        # the custom implementation's behavior
        single_output_posterior = super().posterior(
            X=X_expanded,
            observation_noise=False,
            posterior_transform=None,  # Apply after reshaping
        )

        # Extract the MultivariateNormal
        mvn = single_output_posterior.distribution
        posterior_mean = mvn.mean
        posterior_cov = mvn.lazy_covariance_matrix

        # Add observation noise if requested. ``observation_noise`` may be a
        # bool (add the model's inferred homoskedastic noise) or a Tensor (add
        # the caller-supplied noise variance). Branch on the type explicitly:
        # ``if observation_noise:`` on a multi-element tensor raises an
        # ambiguous-boolean error, and on a scalar tensor it silently discards
        # the requested value.
        if isinstance(observation_noise, Tensor):
            noise = observation_noise.to(dtype=X.dtype, device=X.device)
            if noise.numel() == 1:
                noise_diag = noise.reshape(()).expand(q * m)
            else:
                # Broadcast to the trailing ``(q, m)`` grid, then flatten to the
                # interleaved ``(q*m)`` ordering. Broadcasting (rather than a bare
                # reshape) is what the docstring promises, and is what makes the
                # common shorthands work: ``(m,)`` for a per-output constant and
                # ``(q, 1)`` for a per-point value shared across outputs. A bare
                # reshape accepts neither -- it raises on any tensor whose element
                # count is not exactly ``q * m``.
                batch_shape = noise.shape[:-2] if noise.dim() >= 2 else ()
                noise_diag = noise.expand(*batch_shape, q, m).reshape(
                    *batch_shape, q * m
                )
            noise_cov = torch.diag_embed(noise_diag)
            posterior_cov = posterior_cov + to_linear_operator(noise_cov)
        elif observation_noise:
            if self._train_Yvar_flat is not None:
                avg_noise = self._train_Yvar_flat.mean().to(X)
            elif isinstance(self.likelihood, GaussianLikelihood):
                avg_noise = self.likelihood.noise.to(X)
            else:
                # Previously this fell through to ``avg_noise = 0.0``, silently
                # returning a noiseless posterior for a caller that explicitly
                # asked for observation noise -- the request was dropped with no
                # signal, which hides a genuine configuration error. Reachable by
                # passing a non-Gaussian likelihood (e.g. an explicitly
                # constructed FixedNoiseGaussianLikelihood) without ``train_Yvar``.
                raise UnsupportedError(
                    "observation_noise=True requires either `train_Yvar` or a "
                    "GaussianLikelihood, but this model has "
                    f"{type(self.likelihood).__name__} and no `train_Yvar`. Pass "
                    "`train_Yvar` at construction, or supply the noise explicitly "
                    "as a Tensor via `observation_noise`."
                )
            noise_eye = avg_noise * torch.eye(q * m, dtype=X.dtype, device=X.device)
            posterior_cov = posterior_cov + to_linear_operator(noise_eye)

        # Reshape mean from (q*m) to q x m
        mean_reshaped = posterior_mean.view(*posterior_mean.shape[:-1], q, m)

        # Create MultitaskMultivariateNormal with interleaved=True
        # (matches our data ordering)
        mtmvn = MultitaskMultivariateNormal(
            mean=mean_reshaped,
            covariance_matrix=posterior_cov,
            interleaved=True,
        )

        posterior = GPyTorchPosterior(distribution=mtmvn)

        if posterior_transform is not None:
            return posterior_transform(posterior)

        return posterior
