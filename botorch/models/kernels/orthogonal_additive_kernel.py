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
            base_kernel: The kernel which to orthogonalize and evaluate in `forward`.
            dim: Input dimensionality of the kernel.
            quad_deg: Number of integration nodes for orthogonalization.
            second_order: Toggles second order interactions. If true, both the time and
                space complexity of evaluating the kernel are quadratic in `dim`.
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

    def k(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Evaluates the kernel matrix base_kernel(x1, x2) on each input dimension
        independently.

        Args:
            x1: `batch_shape x n1 x d`-dim Tensor in [0, 1]^dim.
            x2: `batch_shape x n2 x d`-dim Tensor in [0, 1]^dim.

        Returns:
            A `batch_shape x d x n1 x n2`-dim Tensor of kernel matrices.
        """
        return self.base_kernel(x1, x2, last_dim_is_batch=True).to_dense()

    @property
    def offset(self) -> Tensor:
        """Returns the `batch_shape`-dim Tensor of zeroth-order coefficients."""
        return self.coeff_constraint.transform(self.raw_offset)

    @property
    def coeffs_1(self) -> Tensor:
        """Returns the `batch_shape x d`-dim Tensor of first-order coefficients."""
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
            `batch_shape x d x d`-dim Tensor of second-order coefficients.
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
        row_idcs, col_idcs = torch.triu_indices(self.dim, self.dim, offset=1)
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
            x1: `batch_shape x n1 x d`-dim Tensor in [0, 1]^dim.
            x2: `batch_shape x n2 x d`-dim Tensor in [0, 1]^dim.
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported by this kernel.

        Returns:
            A `batch_shape x n1 x n2`-dim Tensor of kernel matrices.
        """
        if last_dim_is_batch:
            raise UnsupportedError(
                "OrthogonalAdditiveKernel does not support `last_dim_is_batch`."
            )
        K_ortho = self._orthogonal_base_kernels(x1, x2)  # batch_shape x d x n1 x n2

        # contracting over d, leading to `batch_shape x n x n`-dim tensor, i.e.:
        #   K1 = torch.sum(self.coeffs_1[..., None, None] * K_ortho, dim=-3)
        K1 = torch.einsum(self.coeffs_1, [..., 0], K_ortho, [..., 0, 1, 2], [..., 1, 2])
        # adding the non-batch dimensions to offset
        K = K1 + self.offset[..., None, None]
        if self.coeffs_2 is not None:
            # Computing the tensor of second order interactions K2.
            # NOTE: K2 here is equivalent to:
            #   K2 = K_ortho.unsqueeze(-4) * K_ortho.unsqueeze(-3)  # d x d x n x n
            #   K2 = (self.coeffs_2[..., None, None] * K2).sum(dim=(-4, -3))
            # but avoids forming the `batch_shape x d x d x n x n`-dim tensor in memory.
            # Reducing over the dimensions with the O(d^2) quadratic terms:
            K2 = torch.einsum(
                K_ortho,
                [..., 0, 2, 3],
                K_ortho,
                [..., 1, 2, 3],
                self.coeffs_2,
                [..., 0, 1],
                [..., 2, 3],  # i.e. contracting over the first two non-batch dims
            )
            K = K + K2

        return K if not diag else K.diag()  # poor man's diag (TODO)

    def _orthogonal_base_kernels(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Evaluates the set of `d` orthogonalized base kernels on (x1, x2).
        Note that even if the base kernel is positive, the orthogonalized versions
        can - and usually do - take negative values.

        Args:
            x1: `batch_shape x n1 x d`-dim inputs to the kernel.
            x2: `batch_shape x n2 x d`-dim inputs to the kernel.

        Returns:
            A `batch_shape x d x n1 x n2`-dim Tensor.
        """
        _check_hypercube(x1, "x1")
        if x1 is not x2:
            _check_hypercube(x2, "x2")
        Kx1x2 = self.k(x1, x2)  # d x n x n
        # Overwriting allocated quadrature tensors with fitting dtype and device
        # self.z, self.w = self.z.to(x1), self.w.to(x1)
        # include normalization constant in weights
        w = self.w / self.normalizer().sqrt()
        Skx1 = self.k(x1, self.z) @ w  # batch_shape x d x n
        Skx2 = Skx1 if (x1 is x2) else self.k(x2, self.z) @ w  # d x n
        # this is a tensor of kernel matrices of orthogonal 1d kernels
        K_ortho = (Kx1x2 - Skx1 @ Skx2.transpose(-2, -1)).to_dense()  # d x n x n
        return K_ortho

    def normalizer(self, eps: float = 1e-6) -> Tensor:
        """Integrates the `d` orthogonalized base kernels over `[0, 1] x [0, 1]`.
        NOTE: If the module is in train mode, this needs to re-compute the normalizer
        each time because the underlying parameters might have changed.

        Args:
            eps: Minimum value constraint on the normalizers. Avoids division by zero.

        Returns:
            A `d`-dim tensor of normalization constants.
        """
        if self.train() or getattr(self, "_normalizer", None) is None:
            self._normalizer = (self.w.T @ self.k(self.z, self.z) @ self.w).clamp(eps)
        return self._normalizer


def leggauss(
    deg: int,
    a: float = -1.0,
    b: float = 1.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Computes Gauss-Legendre quadrature nodes and weights. Wraps
    `numpy.polynomial.legendre.leggauss` and returns Torch Tensors.

    Args:
        deg: Number of sample points and weights. Integrates poynomials of degree
            `2 * deg + 1` exactly.
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
    """Raises a `ValueError` if an element `x` is not in [0, 1].

    Args:
        x: Tensor to be checked.
        name: Name of the Tensor for the error message.
    """
    tolerance = 1e-6
    if (x < -1 * tolerance).any() or (x > 1 + tolerance).any():
        raise ValueError(name + " is not in hypercube [0, 1]^d.")


def _reverse_triu_indices(d: int) -> list[int]:
    """Computes a list of indices which, upon indexing a `d * (d - 1) / 2 + 1`-dim
    Tensor whose last element is zero, will lead to a vectorized representation of
    an upper-triangular matrix, whose diagonal is set to zero and whose super-diagonal
    elements are set to the `d * (d - 1) / 2` values in the original tensor.

    NOTE: This is a helper function for Orthogonal Additive Kernels, and allows the
    implementation to only register `d * (d - 1) / 2` parameters to model the second
    order interactions, instead of the full d^2 redundant terms.

    Args:
        d: Dimensionality that gives rise to the `d * (d - 1) / 2` quadratic terms.

    Returns:
        A list of integer indices in `[0, d * (d - 1) / 2]`. See above for details.
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
