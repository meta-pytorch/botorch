# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy
import torch
from botorch.exceptions.errors import UnsupportedError
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.module import Module
from gpytorch.priors import Prior
from torch import nn, Tensor

_positivity_constraint = Positive()
SECOND_ORDER_PRIOR_ERROR_MSG = (
    "Second order interactions are disabled, but there is a prior on the second order "
    "coefficients. Please remove the second order prior or enable second order terms."
)


class OrthogonalAdditiveKernel(Kernel):
    r"""Orthogonal Additive Kernels (OAKs) were introduced in [Lu2022additive]_, though
    only for the case of Gaussian base kernels with a Gaussian input data distribution.

    The implementation here generalizes OAKs to arbitrary base kernels by using a
    Gauss-Legendre quadrature approximation to the required one-dimensional integrals
    involving the base kernels.

    .. [Lu2022additive]
        X. Lu, A. Boukouvalas, and J. Hensman. Additive Gaussian processes revisited.
        Proceedings of the 39th International Conference on Machine Learning. Jul 2022.
    """

    def __init__(
        self,
        base_kernel: Kernel,
        dim: int,
        quad_deg: int = 32,
        second_order: bool = False,
        batch_shape: torch.Size | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        coeff_constraint: Interval = _positivity_constraint,
        offset_prior: Prior | None = None,
        coeffs_1_prior: Prior | None = None,
        coeffs_2_prior: Prior | None = None,
    ):
        """
        Args:
            base_kernel: The kernel which to orthogonalize and evaluate in ``forward``.
            dim: Input dimensionality of the kernel.
            quad_deg: Number of integration nodes for orthogonalization.
            second_order: Toggles second order interactions. If true, both the time and
                space complexity of evaluating the kernel are quadratic in ``dim``.
            batch_shape: Optional batch shape for the kernel and its parameters.
            dtype: Initialization dtype for required Tensors.
            device: Initialization device for required Tensors.
            coeff_constraint: Constraint on the coefficients of the additive kernel.
            offset_prior: Prior on the offset coefficient. Should be prior with non-
                negative support.
            coeffs_1_prior: Prior on the parameter main effects. Should be prior with
                non-negative support.
            coeffs_2_prior: coeffs_1_prior: Prior on the parameter interactions. Should
                be prior with non-negative support.
        """
        super().__init__(batch_shape=batch_shape)
        self.base_kernel = base_kernel
        if not second_order and coeffs_2_prior is not None:
            raise AttributeError(SECOND_ORDER_PRIOR_ERROR_MSG)

        # integration nodes, weights for [0, 1]
        tkwargs = {"dtype": dtype, "device": device}
        z, w = leggauss(deg=quad_deg, a=0, b=1, **tkwargs)
        self.z = z.unsqueeze(-1).expand(quad_deg, dim)  # deg x dim
        self.w = w.unsqueeze(-1)
        self.register_parameter(
            name="raw_offset",
            parameter=nn.Parameter(torch.zeros(self.batch_shape, **tkwargs)),
        )
        log_d = math.log(dim)
        self.register_parameter(
            name="raw_coeffs_1",
            parameter=nn.Parameter(
                torch.zeros(*self.batch_shape, dim, **tkwargs) - log_d
            ),
        )
        self.register_parameter(
            name="raw_coeffs_2",
            parameter=(
                nn.Parameter(
                    torch.zeros(*self.batch_shape, int(dim * (dim - 1) / 2), **tkwargs)
                    - 2 * log_d
                )
                if second_order
                else None
            ),
        )
        if offset_prior is not None:
            self.register_prior(
                name="offset_prior",
                prior=offset_prior,
                param_or_closure=_offset_param,
                setting_closure=_offset_closure,
            )
        if coeffs_1_prior is not None:
            self.register_prior(
                name="coeffs_1_prior",
                prior=coeffs_1_prior,
                param_or_closure=_coeffs_1_param,
                setting_closure=_coeffs_1_closure,
            )
        if coeffs_2_prior is not None:
            self.register_prior(
                name="coeffs_2_prior",
                prior=coeffs_2_prior,
                param_or_closure=_coeffs_2_param,
                setting_closure=_coeffs_2_closure,
            )

        # for second order interactions, we only
        if second_order:
            self._rev_triu_indices = torch.tensor(
                _reverse_triu_indices(dim),
                device=device,
                dtype=int,
            )
            # zero tensor for construction of upper-triangular coefficient matrix
            self._quad_zero = torch.zeros(
                tuple(1 for _ in range(len(self.batch_shape) + 1)), **tkwargs
            ).expand(*self.batch_shape, 1)
        self.coeff_constraint = coeff_constraint
        self.dim = dim

    # =========================================================================
    # Helper methods for diag-dependent operations (reduces code duplication)
    # =========================================================================

    def _trailing_dims(self, diag: bool) -> tuple[None, ...]:
        """Returns trailing dims for broadcasting: (None,) or (None, None)."""
        return (None,) if diag else (None, None)

    def _triu_indices(
        self, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        """Upper triangular indices for second-order terms (i, j) where i < j."""
        return torch.triu_indices(self.dim, self.dim, offset=1, device=device)

    def _coeff_for_kernel_broadcast(self, coeff: Tensor, diag: bool) -> Tensor:
        """Expands coefficient for broadcasting with kernel matrices."""
        return coeff[(...,) + self._trailing_dims(diag)]

    def _bias_covariance(self, n1: int, n2: int, diag: bool) -> Tensor:
        """Creates constant (bias) covariance matrix from self.offset."""
        spatial_dims = (n1,) if diag else (n1, n2)
        return self.offset[(...,) + self._trailing_dims(diag)].expand(
            *self.batch_shape, *spatial_dims
        )

    def _slice_kernel_components(
        self, K_ortho: Tensor, indices: Tensor | int, diag: bool
    ) -> Tensor:
        """Slices kernel components along the component dimension."""
        slices = (..., indices) + (slice(None),) * (1 if diag else 2)
        return K_ortho[slices]

    @property
    def num_components(self) -> int:
        """Total number of additive components (bias + first-order [+ second-order])."""
        n = 1 + self.dim  # bias + first-order
        if self.raw_coeffs_2 is not None:
            n += self.dim * (self.dim - 1) // 2
        return n

    def k(self, x1: Tensor, x2: Tensor, diag: bool = False) -> Tensor:
        """Evaluates the kernel matrix base_kernel(x1, x2) on each input dimension
        independently.

        Args:
            x1: ``batch_shape x n1 x d``-dim Tensor in [0, 1]^dim.
            x2: ``batch_shape x n2 x d``-dim Tensor in [0, 1]^dim.
            diag: Whether to evaluate only the diagonal of the kernel matrix.

        Returns:
            A ``batch_shape x d x n1 x n2``-dim Tensor of kernel matrices, or a
            `batch_shape x d x n1`-dim Tensor of diagonal elements if `diag=True`.
        """
        # Reshape inputs to treat each dimension as a batch:
        # x1: batch_shape x n1 x d -> batch_shape x d x n1 x 1
        # x2: batch_shape x n2 x d -> batch_shape x d x n2 x 1
        x1_reshaped = x1.transpose(-1, -2).unsqueeze(-1)
        x2_reshaped = x2.transpose(-1, -2).unsqueeze(-1)
        return self.base_kernel(x1_reshaped, x2_reshaped, diag=diag).to_dense()

    @property
    def offset(self) -> Tensor:
        """Returns the ``batch_shape``-dim Tensor of zeroth-order coefficients."""
        return self.coeff_constraint.transform(self.raw_offset)

    @property
    def coeffs_1(self) -> Tensor:
        """Returns the ``batch_shape x d``-dim Tensor of first-order coefficients."""
        return self.coeff_constraint.transform(self.raw_coeffs_1)

    @property
    def coeffs_2(self) -> Tensor | None:
        """Returns the upper-triangular tensor of second-order coefficients.

        NOTE: We only keep track of the upper triangular part of raw second order
        coefficients since the effect of the lower triangular part is identical and
        exclude the diagonal, since it is associated with first-order effects only.
        While we could further exploit this structure in the forward pass, the
        associated indexing and temporary allocations make it significantly less
        efficient than the einsum-based implementation below.

        Returns:
            ``batch_shape x d x d``-dim Tensor of second-order coefficients.
        """
        if self.raw_coeffs_2 is not None:
            C2 = self.coeff_constraint.transform(self.raw_coeffs_2)
            C2 = torch.cat((C2, self._quad_zero), dim=-1)  # batch_shape x (d(d-1)/2+1)
            C2 = C2.index_select(-1, self._rev_triu_indices)
            return C2.reshape(*self.batch_shape, self.dim, self.dim)
        else:
            return None

    def _set_coeffs_1(self, value: Tensor) -> None:
        value = torch.as_tensor(value).to(self.raw_coeffs_1)
        value = value.expand(*self.batch_shape, self.dim)
        self.initialize(raw_coeffs_1=self.coeff_constraint.inverse_transform(value))

    def _set_coeffs_2(self, value: Tensor) -> None:
        value = torch.as_tensor(value).to(self.raw_coeffs_1)
        value = value.expand(*self.batch_shape, self.dim, self.dim)
        row_idcs, col_idcs = self._triu_indices()
        value = value[..., row_idcs, col_idcs].to(self.raw_coeffs_2)
        self.initialize(raw_coeffs_2=self.coeff_constraint.inverse_transform(value))

    def _set_offset(self, value: Tensor) -> None:
        value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.coeff_constraint.inverse_transform(value))

    @coeffs_1.setter
    def coeffs_1(self, value) -> None:
        self._set_coeffs_1(value)

    @coeffs_2.setter
    def coeffs_2(self, value) -> None:
        self._set_coeffs_2(value)

    @offset.setter
    def offset(self, value) -> None:
        self._set_offset(value)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Computes the kernel matrix k(x1, x2).

        Args:
            x1: ``batch_shape x n1 x d``-dim Tensor in [0, 1]^dim.
            x2: ``batch_shape x n2 x d``-dim Tensor in [0, 1]^dim.
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported by this kernel.

        Returns:
            A ``batch_shape x n1 x n2``-dim Tensor of kernel matrices.
        """
        if last_dim_is_batch:
            raise UnsupportedError(
                "OrthogonalAdditiveKernel does not support `last_dim_is_batch`."
            )
        K_ortho = self._orthogonal_base_kernels(
            x1, x2, diag=diag
        )  # batch_shape x d x n1 (x n2)

        # contracting over d, leading to ``batch_shape x n x n``-dim tensor, i.e.:
        #   K1 = torch.sum(self.coeffs_1[..., None, None] * K_ortho, dim=-3)
        non_diag_dim = [] if diag else [2]
        K1 = torch.einsum(
            self.coeffs_1,
            [..., 0],
            K_ortho,
            [..., 0, 1] + non_diag_dim,
            [..., 1] + non_diag_dim,
        )
        # adding the non-batch dimensions to offset
        K = K1 + self.offset[(...,) + self._trailing_dims(diag)]
        if self.coeffs_2 is not None:
            # Computing the tensor of second order interactions K2.
            # NOTE: K2 here is equivalent to:
            #   K2 = K_ortho.unsqueeze(-4) * K_ortho.unsqueeze(-3)  # d x d x n x n
            #   K2 = (self.coeffs_2[..., None, None] * K2).sum(dim=(-4, -3))
            # but avoids forming the ``batch_shape x d x d x n x n``-dim tensor
            # in memory.
            # Reducing over the dimensions with the O(d^2) quadratic terms:
            non_diag_dim = [] if diag else [3]
            K2 = torch.einsum(
                K_ortho,
                [..., 0, 2] + non_diag_dim,
                K_ortho,
                [..., 1, 2] + non_diag_dim,
                self.coeffs_2,
                [..., 0, 1],
                # i.e. contracting over the first two non-batch dims
                [..., 2] + non_diag_dim,
            )
            K = K + K2

        return K

    def _non_reduced_forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Computes the non-reduced kernel matrices for each additive component.

        Returns a stacked tensor of component kernel matrices that can be used for
        posterior inference of individual additive components. The components are:
        - 1 bias term (constant offset)
        - d first-order terms (one per input dimension)
        - d*(d-1)/2 second-order terms (upper triangular, excluding diagonal)

        Args:
            x1: `batch_shape x n1 x d`-dim Tensor in [0, 1]^dim.
            x2: `batch_shape x n2 x d`-dim Tensor in [0, 1]^dim.
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported by this kernel.

        Returns:
            A `batch_shape x num_components x n1 x n2`-dim Tensor of kernel matrices,
            where num_components = 1 + d (first-order only) or 1 + d + d*(d-1)/2
            (with second-order interactions).
        """
        if last_dim_is_batch:
            raise UnsupportedError(
                "OrthogonalAdditiveKernel does not support `last_dim_is_batch`."
            )

        K_ortho = self._orthogonal_base_kernels(
            x1, x2, diag=diag
        )  # batch_shape x d x n1 (x n2)
        n1, n2 = x1.shape[-2], x2.shape[-2]

        # First-order: batch_shape x d x n (x n)
        K1 = self._coeff_for_kernel_broadcast(self.coeffs_1, diag) * K_ortho

        # Bias: batch_shape x 1 x n (x n)
        component_dim = -2 if diag else -3
        K0 = self._bias_covariance(n1, n2, diag).unsqueeze(component_dim)

        component_tensors = [K0, K1]

        if self.raw_coeffs_2 is not None:
            row_idcs, col_idcs = self._triu_indices(x1.device)
            K_i = self._slice_kernel_components(K_ortho, row_idcs, diag)
            K_j = self._slice_kernel_components(K_ortho, col_idcs, diag)
            coeffs_2 = self.coeff_constraint.transform(self.raw_coeffs_2)
            K2 = self._coeff_for_kernel_broadcast(coeffs_2, diag) * (K_i * K_j)
            component_tensors.append(K2)

        return torch.cat(component_tensors, dim=component_dim)

    @property
    def component_indices(self) -> dict[str, Tensor]:
        """Returns mapping from component type to input dimension indices.

        This property helps users understand which batch index in the output of
        `_non_reduced_forward` corresponds to which additive component.

        Returns:
            A dict with keys:
            - 'bias': Tensor of shape (1,) with value 0 (the bias component index)
            - 'first_order': Tensor of shape (d,) with values 1, 2, ..., d
                (indices into batch dimension, mapping to input dimensions 0..d-1)
            - 'second_order': Tensor of shape (d*(d-1)/2, 2) with pairs (i, j)
                where i < j, representing interaction between input dims i and j
                (only present if second_order=True)
        """
        d = self.dim
        device = self.raw_offset.device

        result = {
            "bias": torch.tensor([0], device=device),
            "first_order": torch.arange(d, device=device),
        }

        if self.raw_coeffs_2 is not None:
            # Upper triangular indices (i, j) where i < j
            row_idcs, col_idcs = self._triu_indices(device)
            result["second_order"] = torch.stack([row_idcs, col_idcs], dim=-1)

        return result

    def get_component_index(
        self,
        component_type: str,
        dim_index: int | tuple[int, int] | None = None,
    ) -> int:
        """Returns the component index for a given component type and dimension.

        Args:
            component_type: One of "bias", "first_order", or "second_order"
            dim_index: For "first_order", the input dimension (0 to d-1).
                       For "second_order", a tuple (i, j) where i < j.
                       Not used for "bias".

        Returns:
            The integer index into the component batch dimension.

        Raises:
            ValueError: If component_type is unknown or dim_index is invalid.
            IndexError: If dim_index is out of range.
        """
        d = self.dim

        if component_type == "bias":
            return 0
        elif component_type == "first_order":
            if dim_index is None or not isinstance(dim_index, int):
                raise ValueError("dim_index must be an int for first_order")
            if dim_index < 0 or dim_index >= d:
                raise IndexError(f"dim_index {dim_index} out of range [0, {d - 1}]")
            return 1 + dim_index
        elif component_type == "second_order":
            if self.raw_coeffs_2 is None:
                raise ValueError("second_order components not enabled for this kernel")
            if (
                dim_index is None
                or not isinstance(dim_index, tuple)
                or len(dim_index) != 2
            ):
                raise ValueError("dim_index must be a tuple (i, j) for second_order")
            i, j = dim_index
            if i >= j:
                raise ValueError(f"For second_order, require i < j, got ({i}, {j})")
            if i < 0 or j >= d:
                raise IndexError(f"Invalid second_order index ({i}, {j}) for dim={d}")
            # Find the index in the upper triangular enumeration
            row_idcs, col_idcs = self._triu_indices()
            mask = (row_idcs == i) & (col_idcs == j)
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                raise IndexError(f"Invalid second_order index ({i}, {j})")
            return 1 + d + idx.item()
        else:
            raise ValueError(f"Unknown component_type: {component_type}")

    def _orthogonal_base_kernels(
        self, x1: Tensor, x2: Tensor, diag: bool = False
    ) -> Tensor:
        """Evaluates the set of ``d`` orthogonalized base kernels on (x1, x2).
        Note that even if the base kernel is positive, the orthogonalized versions
        can - and usually do - take negative values.

        Args:
            x1: ``batch_shape x n1 x d``-dim inputs to the kernel.
            x2: ``batch_shape x n2 x d``-dim inputs to the kernel.
            diag: Whether to evaluate only the diagonal of the kernel matrix.

        Returns:
            A ``batch_shape x d x n1 x n2``-dim Tensor, or a `batch_shape x d x n1`-dim
            Tensor if `diag=True`.
        """
        _check_hypercube(x1, "x1")
        if x1 is not x2:
            _check_hypercube(x2, "x2")
        Kx1x2 = self.k(x1, x2, diag=diag)  # batch_shape x d x n1 (x n2)
        # Overwriting allocated quadrature tensors with fitting dtype and device
        # self.z, self.w = self.z.to(x1), self.w.to(x1)
        # include normalization constant in weights
        # self.w: (q, 1), self.normalizer(): (d, 1, 1) -> w: (d, q, 1)
        w = self.w / self.normalizer().sqrt()
        # self.k(x1, self.z): batch_shape, d, n, q
        Skx1 = self.k(x1, self.z) @ w  # batch_shape, d, n, 1
        Skx2 = Skx1 if (x1 is x2) else self.k(x2, self.z) @ w
        correction = Skx1 @ Skx2.transpose(-2, -1) if not diag else Skx1.square()
        K_ortho = (Kx1x2 - correction).to_dense()  # batch_shape x d x n1 (x n2)
        return K_ortho

    def normalizer(self, eps: float = 1e-6) -> Tensor:
        """Integrates the ``d`` orthogonalized base kernels over ``[0, 1] x [0, 1]``.
        NOTE: If the module is in train mode, this needs to re-compute the normalizer
        each time because the underlying parameters might have changed.

        Args:
            eps: Minimum value constraint on the normalizers. Avoids division by zero.

        Returns:
            A ``(d, 1, 1)``-dim tensor of normalization constants.
        """
        if self.train() or getattr(self, "_normalizer", None) is None:
            # Computes w.T @ K @ w for each dimension d.
            w = self.w.squeeze(-1)  # (q, 1) -> (q,)
            normalizer = torch.einsum(
                w,
                [0],
                self.k(self.z, self.z),
                [2, 0, 1],
                w,
                [1],
                [2],
            ).clamp(eps)  # (d)
            self._normalizer = normalizer[..., None, None]
        return self._normalizer


def leggauss(
    deg: int,
    a: float = -1.0,
    b: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Computes Gauss-Legendre quadrature nodes and weights. Wraps
    ``numpy.polynomial.legendre.leggauss`` and returns Torch Tensors.

    Args:
        deg: Number of sample points and weights. Integrates poynomials of degree
            ``2 * deg + 1`` exactly.
        a, b: Lower and upper bound of integration domain.
        dtype: Desired floating point type of the return Tensors.
        device: Desired device type of the return Tensors.

    Returns:
        A tuple of Gauss-Legendre quadrature nodes and weights of length deg.
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    x, w = numpy.polynomial.legendre.leggauss(deg=deg)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    w = torch.as_tensor(w, dtype=dtype, device=device)
    if not (a == -1 and b == 1):  # need to normalize for different domain
        x = (b - a) * (x + 1) / 2 + a
        w = w * ((b - a) / 2)
    return x, w


def _check_hypercube(x: Tensor, name: str) -> None:
    """Raises a ``ValueError`` if an element ``x`` is not in [0, 1].

    Args:
        x: Tensor to be checked.
        name: Name of the Tensor for the error message.
    """
    tolerance = 1e-6
    if (x < -1 * tolerance).any() or (x > 1 + tolerance).any():
        raise ValueError(name + " is not in hypercube [0, 1]^d.")


def _reverse_triu_indices(d: int) -> list[int]:
    """Computes a list of indices which, upon indexing a ``d * (d - 1) / 2 + 1``-dim
    Tensor whose last element is zero, will lead to a vectorized representation of
    an upper-triangular matrix, whose diagonal is set to zero and whose super-diagonal
    elements are set to the ``d * (d - 1) / 2`` values in the original tensor.

    NOTE: This is a helper function for Orthogonal Additive Kernels, and allows the
    implementation to only register ``d * (d - 1) / 2`` parameters to model the second
    order interactions, instead of the full d^2 redundant terms.

    Args:
        d: Dimensionality that gives rise to the ``d * (d - 1) / 2`` quadratic terms.

    Returns:
        A list of integer indices in ``[0, d * (d - 1) / 2]``. See above for details.
    """
    indices = []
    j = 0
    d2 = int(d * (d - 1) / 2)
    for i in range(d):
        indices.extend(d2 for _ in range(i + 1))  # indexing zero (sub-diagonal)
        indices.extend(range(j, j + d - i - 1))  # indexing coeffs (super-diagonal)
        j += d - i - 1
    return indices


def _coeffs_1_param(m: Module) -> Tensor:
    return m.coeffs_1


def _coeffs_2_param(m: Module) -> Tensor:
    return m.coeffs_2


def _offset_param(m: Module) -> Tensor:
    return m.offset


def _coeffs_1_closure(m: Module, v: Tensor) -> Tensor:
    return m._set_coeffs_1(v)


def _coeffs_2_closure(m: Module, v: Tensor) -> Tensor:
    return m._set_coeffs_2(v)


def _offset_closure(m: Module, v: Tensor) -> Tensor:
    return m._set_offset(v)
