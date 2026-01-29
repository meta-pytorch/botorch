#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Linear Elliptical Slice Sampler.

References

.. [Gessner2020]
    A. Gessner, O. Kanjilal, and P. Hennig. Integrals over gaussians under
    linear domain constraints. AISTATS 2020.

.. [Wu2024]
    K. Wu, and J. Gardner. A Fast, Robust Elliptical Slice Sampling Implementation for
    Linearly Truncated Multivariate Normal Distributions. arXiv:2407.10449. 2024.

This implementation is based (with multiple changes / optimiations) on
the following implementations based on the algorithm in [Gessner2020]_:
- https://github.com/alpiges/LinConGauss
- https://github.com/wjmaddox/pytorch_ess

In addition, the active intervals (from which the angle is sampled) are computed using
the improved algorithm described in [Wu2024]_:
https://github.com/kayween/linear-ess

The implementation here differentiates itself from the original implementations with:
1) Support for fixed feature equality constraints.
2) Support for non-standard Normal distributions.
3) Numerical stability improvements, especially relevant for high-dimensional cases.
4) Support multiple Markov chains running in parallel.
"""

from __future__ import annotations

import math

import torch
from botorch.utils.sampling import PolytopeSampler
from linear_operator.operators import DiagLinearOperator, LinearOperator
from torch import Tensor

_twopi = 2.0 * math.pi


class LinearEllipticalSliceSampler(PolytopeSampler):
    r"""Linear Elliptical Slice Sampler.

    Ideas:
    - Optimize computations if possible, potentially with torch.compile.
    - Extend fixed features constraint to general linear equality constraints.
    """

    def __init__(
        self,
        inequality_constraints: tuple[Tensor, Tensor] | None = None,
        bounds: Tensor | None = None,
        interior_point: Tensor | None = None,
        fixed_indices: list[int] | Tensor | None = None,
        mean: Tensor | None = None,
        covariance_matrix: Tensor | LinearOperator | None = None,
        covariance_root: Tensor | LinearOperator | None = None,
        check_feasibility: bool = False,
        burnin: int = 0,
        thinning: int = 0,
        num_chains: int = 1,
    ) -> None:
        r"""Initialize LinearEllipticalSliceSampler.

        Args:
            inequality_constraints: Tensors ``(A, b)`` describing inequality
                 constraints ``A @ x <= b``, where ``A`` is an
                 ``n_ineq_con x d``-dim Tensor and ``b`` is an
                 ``n_ineq_con x 1``-dim Tensor, with ``n_ineq_con`` the number of
                 inequalities and ``d`` the dimension of the sample space. If omitted,
                 must provide ``bounds`` instead.
            bounds: A ``2 x d``-dim tensor of box bounds. If omitted, must provide
                ``inequality_constraints`` instead.
            interior_point: A ``d x 1``-dim Tensor presenting a point in the (relative)
                interior of the polytope. If omitted, an interior point is determined
                automatically by solving a Linear Program. Note: It is crucial that
                the point lie in the interior of the feasible set (rather than on the
                boundary), otherwise the sampler will produce invalid samples.
            fixed_indices: Integer list or ``d``-dim Tensor representing the indices
                of dimensions that are constrained to be fixed to the values specified
                in the ``interior_point``, which is required to be passed in
                conjunction with ``fixed_indices``.
            mean: The ``d x 1``-dim mean of the MVN distribution (if omitted, use zero).
            covariance_matrix: The ``d x d``-dim covariance matrix of the MVN
                distribution (if omitted, use the identity).
            covariance_root: A ``d x d``-dim root of the covariance matrix such that
                covariance_root @ covariance_root.T = covariance_matrix. NOTE: This
                matrix is assumed to be lower triangular. covariance_root can only be
                passed in conjunction with fixed_indices if covariance_root is a
                DiagLinearOperator. Otherwise the factorization would need to be re-
                computed, as we need to solve in ``standardize``.
            check_feasibility: If True, raise an error if the sampling results in an
                infeasible sample. This creates some overhead and so is switched off
                by default.
            burnin: Number of samples to generate upon initialization to warm up the
                sampler.
            thinning: Number of samples to skip before returning a sample in ``draw``.
            num_chains: Number of Markov chains to run in parallel.

        This sampler samples from a multivariante Normal ``N(mean, covariance_matrix)``
        subject to linear domain constraints ``A x <= b`` (intersected with box bounds,
        if provided).
        """
        if interior_point is not None and interior_point.ndim == 1:
            interior_point = interior_point.unsqueeze(-1)

        if mean is not None and mean.ndim == 1:
            mean = mean.unsqueeze(-1)

        super().__init__(
            inequality_constraints=inequality_constraints,
            # TODO: Support equality constraints?
            interior_point=interior_point,
            bounds=bounds,
        )
        tkwargs = {"device": self.x0.device, "dtype": self.x0.dtype}
        if covariance_matrix is not None and covariance_root is not None:
            raise ValueError(
                "Provide either covariance_matrix or covariance_root, not both."
            )

        # can't unpack inequality constraints directly if bounds are passed
        A, b = self.A, self.b
        self._Az, self._bz = A, b
        self._is_fixed, self._not_fixed = None, None
        if fixed_indices is not None:
            mean, covariance_matrix, covariance_root = (
                self._fixed_features_initialization(
                    A=A,
                    b=b,
                    interior_point=interior_point,
                    fixed_indices=fixed_indices,
                    mean=mean,
                    covariance_matrix=covariance_matrix,
                    covariance_root=covariance_root,
                )
            )

        self._mean = mean
        # Have to delay factorization until after fixed features initialization.
        if covariance_matrix is not None:  # implies root is None
            covariance_root, info = torch.linalg.cholesky_ex(covariance_matrix)
            not_psd = torch.any(info)
            if not_psd:
                raise ValueError(
                    "Covariance matrix is not positive definite. "
                    "Currently only non-degenerate distributions are supported."
                )
        self._covariance_root = covariance_root

        # Rewrite the constraints as a system that constrains a standard Normal.
        self._standardization_initialization()

        # state of the sampler ("current point")
        self._x = self.x0.clone()
        self._z = self._transform(self._x)

        # Expand the shape to (d, num_chains) for running parallel Markov chains.
        if num_chains > 1:
            self._z = self._z.expand(-1, num_chains).clone()

        # We will need the following repeatedly, let's allocate them once
        self.zeros = torch.zeros((num_chains, 1), **tkwargs)
        self.ones = torch.ones((num_chains, 1), **tkwargs)
        self.indices_batch = torch.arange(
            num_chains, dtype=torch.int64, device=tkwargs["device"]
        )

        self.check_feasibility = check_feasibility
        self._lifetime_samples = 0
        if burnin > 0:
            self.thinning = 0
            self.draw(burnin)
        self.thinning = thinning

    def _fixed_features_initialization(
        self,
        A: Tensor,
        b: Tensor,
        interior_point: Tensor | None,
        fixed_indices: list[int] | Tensor,
        mean: Tensor | None,
        covariance_matrix: Tensor | None,
        covariance_root: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Modifies the constraint system (A, b) due to fixed indices and assigns
        the modified constraints system to ``self._Az``, ``self._bz``. NOTE: Needs to be
        called prior to ``self._standardization_initialization`` in the constructor.
        covariance_root and fixed_indices can both not be None only if covariance_root
        is a DiagLinearOperator. Otherwise, the covariance matrix would need to be
        refactorized.

        Returns:
            Tuple of ``mean`` and ``covariance_matrix`` tensors of the non-fixed
            dimensions.
        """
        if interior_point is None:
            raise ValueError(
                "If `fixed_indices` are provided, an interior point must also be "
                "provided in order to infer feasible values of the fixed features."
            )

        root_is_diag = isinstance(covariance_root, DiagLinearOperator)
        if covariance_root is not None and not root_is_diag:
            root_is_diag = (covariance_root.diag().diag() == covariance_root).all()
            if root_is_diag:  # convert the diagonal root to a DiagLinearOperator
                covariance_root = DiagLinearOperator(covariance_root.diagonal())
            else:  # otherwise, fail
                raise ValueError(
                    "Provide either covariance_root or fixed_indices, not both."
                )
        d = interior_point.shape[0]
        is_fixed, not_fixed = get_index_tensors(fixed_indices=fixed_indices, d=d)
        self._is_fixed = is_fixed
        self._not_fixed = not_fixed
        # Transforming constraint system to incorporate fixed features:
        # A @ x - b = (A[:, fixed] @ x[fixed] + A[:, not fixed] @ x[not fixed]) - b
        #           = A[:, not fixed] @ x[not fixed] - (b - A[:, fixed] @ x[fixed])
        #           = Az @ z - bz
        self._Az = A[:, not_fixed]
        self._bz = b - A[:, is_fixed] @ interior_point[is_fixed]
        if mean is not None:
            mean = mean[not_fixed]
        if covariance_matrix is not None:  # subselect active dimensions
            covariance_matrix = covariance_matrix[
                not_fixed.unsqueeze(-1), not_fixed.unsqueeze(0)
            ]
        if root_is_diag:  # in the special case of diagonal root, can subselect
            covariance_root = DiagLinearOperator(covariance_root.diagonal()[not_fixed])

        return mean, covariance_matrix, covariance_root

    def _standardization_initialization(self) -> None:
        """For non-standard mean and covariance, we're going to rewrite the problem as
        sampling from a standard normal distribution subject to modified constraints.
            A @ x - b = A @ (covar_root @ z + mean) - b
                      = (A @ covar_root) @ z - (b - A @ mean)
                      = _Az @ z - _bz
        NOTE: We need to standardize bz before Az in the following, because it relies
        on the untransformed Az. We can't simply use A instead because Az might have
        been subject to the fixed features transformation.
        """
        if self._mean is not None:
            self._bz = self._bz - self._Az @ self._mean
        if self._covariance_root is not None:
            self._Az = self._Az @ self._covariance_root

    @property
    def lifetime_samples(self) -> int:
        """The total number of samples generated by the sampler during its lifetime."""
        return self._lifetime_samples

    def draw(self, n: int = 1) -> Tensor:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A ``(n * num_chains) x d``-dim tensor of ``n * num_chains`` samples.
        """
        samples = []
        for _ in range(n):
            for _ in range(self.thinning):
                self.step()
            samples.append(self.step())
        return torch.cat(samples, dim=-1).transpose(-1, -2)

    def step(self) -> Tensor:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A ``d x num_chains``-dim tensor, where each column is a sample from a Markov
            chain.
        """
        nu = torch.randn_like(self._z)
        theta = self._draw_angle(nu=nu)

        self._z = z = self._get_cart_coords(nu=nu, theta=theta)
        self._x = x = self._untransform(z)

        self._lifetime_samples += 1
        if self.check_feasibility and (not self._is_feasible(self._x).all()):
            Axmb = self.A @ self._x - self.b
            violated_indices = Axmb > 0
            raise RuntimeError(
                "Sampling resulted in infeasible point. \n\t- Number "
                f"of violated constraints: {violated_indices.sum()}."
                f"\n\t- Magnitude of violations: {Axmb[violated_indices]}"
                "\n\t- If the error persists, please report this bug on GitHub."
            )
        return x

    def _draw_angle(self, nu: Tensor) -> Tensor:
        r"""Draw the rotation angle.

        Args:
            nu: A ``d x num_chains``-dim tensor (the "new" direction, drawn from
                N(0, I)).

        Returns:
            A ``num_chains``-dim Tensor containing the rotation angle (radians).
        """
        left, right = self._find_active_intersection_angles(nu)
        left, right = self._trim_intervals(left, right)

        # If left[i, j] <= right[i, j], then [left[i, j], right[i, j]] is an active
        # interval. On the other hand, if left[i, j] > right[i, j], then they are both
        # dummy variables and should be discarded. Thus, we clamp their difference so
        # that they do not contribute to the cumulative length.
        csum = right.sub(left).clamp(min=0.0).cumsum(dim=-1)

        u = csum[:, -1] * torch.rand(
            right.size(-2), dtype=right.dtype, device=right.device
        )

        # The returned index i satisfies csum[i - 1] < u <= csum[i]
        idx = torch.searchsorted(csum, u.unsqueeze(-1)).squeeze(-1)

        # Do a zero padding so that padded_csum[i] = csum[i - 1]
        padded_csum = torch.cat([self.zeros, csum], dim=-1)

        return u - padded_csum[self.indices_batch, idx] + left[self.indices_batch, idx]

    def _get_cart_coords(self, nu: Tensor, theta: Tensor) -> Tensor:
        r"""Determine location on the ellipse in Cartesian coordinates.

        Args:
            nu: A ``d x num_chains``-dim tensor (the "new" direction, drawn from
                N(0, I)).
            theta: A ``num_chains``-dim tensor of angles.

        Returns:
            A ``d x num_chains``-dim tensor of samples from the domain in Cartesian
            coordinates.
        """
        return self._z * torch.cos(theta) + nu * torch.sin(theta)

    def _trim_intervals(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        """Trim the intervals by a small positive constant. This encourages the Markov
        chain to stay in the interior of the domain.
        """
        gap = torch.clamp(right - left, min=0.0)
        eps = gap.mul(0.25).clamp(max=1e-6 if gap.dtype == torch.float32 else 1e-12)

        return left + eps, right - eps

    def _find_active_intersection_angles(self, nu: Tensor) -> tuple[Tensor, Tensor]:
        """Construct the active intersection angles.

        Args:
            nu: A ``d x num_chains``-dim tensor (the "new" direction, drawn from
                N(0, I)).

        Returns:
            A tuple (left, right) of two tensors of size ``num_chains x m`` representing
            the active intersection angles. For the i-th Markov chain and the j-th
            constraint, a pair of angles left[i, j] and right[i, j] is active if and
            only if left[i, j] <= right[i, j]. If left[i, j] > right[i, j], they are
            inactive and should be ignored.
        """
        alpha, beta = self._find_intersection_angles(nu)

        # It's easier to put ``num_chains`` as the first dimension,
        # because ``torch.searchsorted`` only supports searching in the last dimension
        alpha, beta = alpha.T, beta.T

        srted, indices = torch.sort(alpha, descending=False)
        cummax = beta[self.indices_batch.unsqueeze(-1), indices].cummax(dim=-1).values

        srted = torch.cat([srted, self.ones * 2 * math.pi], dim=-1)
        cummax = torch.cat([self.zeros, cummax], dim=-1)

        return cummax, srted

    def _find_intersection_angles(self, nu: Tensor) -> tuple[Tensor, Tensor]:
        """Compute all 2 * m intersections of the ellipse and the domain, where
        ``m = n_ineq_con`` is the number of inequality constraints defining the domain.
        If the i-th linear inequality constraint has no intersection with the ellipse,
        we will create two dummy intersection angles alpha_i = beta_i = 0.

        Args:
            nu: A ``d x num_chains``-dim tensor (the "new" direction, drawn from
                N(0, I)).

        Returns:
            A tuple of two tensors with the same size ``m x num_chains``. The first
            tensor represents the smaller intersection angles. The second tensor
            represents the larger intersection angles.
        """
        p = self._Az @ self._z
        q = self._Az @ nu

        radius = torch.sqrt(p**2 + q**2)

        ratio = self._bz / radius

        has_solution = ratio < 1.0

        arccos = torch.arccos(ratio)
        arccos[~has_solution] = 0.0
        arctan = torch.arctan2(q, p)

        theta1 = arctan + arccos
        theta2 = arctan - arccos

        # translate every angle to [0, 2 * pi]
        theta1 = theta1 + theta1.lt(0.0) * _twopi
        theta2 = theta2 + theta2.lt(0.0) * _twopi

        alpha = torch.minimum(theta1, theta2)
        beta = torch.maximum(theta1, theta2)

        return alpha, beta

    def _is_feasible(self, points: Tensor, transformed: bool = False) -> Tensor:
        r"""Returns a Boolean tensor indicating whether the ``points`` are feasible,
        i.e. they satisfy ``A @ points <= b``, where ``(A, b)`` are the tensors passed
        as the ``inequality_constraints`` to the constructor of the sampler.

        Args:
            points: A ``d x M``-dim tensor of points.
            transformed: Wether points are assumed to be transformed by a change of
                basis, which means feasibility should be computed based on the
                transformed constraint system (_Az, _bz), instead of (A, b).

        Returns:
            An ``M``-dim binary tensor where ``True`` indicates that the associated
            point is feasible.
        """
        A, b = (self._Az, self._bz) if transformed else (self.A, self.b)
        return (A @ points <= b).all(dim=0)

    def _transform(self, x: Tensor) -> Tensor:
        """Transforms the input so that it is equivalent to a standard Normal variable
        constrained with the modified system constraints (self._Az, self._bz).

        Args:
            x: The input tensor to be transformed, usually ``d x 1``-dimensional.

        Returns:
            A ``d x 1``-dimensional tensor of transformed values subject to the modified
            system of constraints.
        """
        if self._not_fixed is not None:
            x = x[self._not_fixed]
        return self._standardize(x)

    def _untransform(self, z: Tensor) -> Tensor:
        """The inverse transform of the ``_transform``, i.e. maps ``z`` back to the
        original space where it is subject to the original constraint system
        (self.A, self.b).

        Args:
            z: The transformed tensor to be un-transformed, usually
                ``d x 1``-dimensional.

        Returns:
            A ``d x 1``-dimensional tensor of un-transformed values subject to the
            original system of constraints.
        """
        if self._is_fixed is None:
            return self._unstandardize(z)
        else:
            x = self._x.clone()  # _x already contains the fixed values
            x[self._not_fixed] = self._unstandardize(z)
            return x

    def _standardize(self, x: Tensor) -> Tensor:
        """_transform helper standardizing the input ``x``, which is assumed to be a
        ``d x 1``-dim Tensor, or a ``len(self._not_fixed) x 1``-dim if there are fixed
        indices.
        """
        z = x
        if self._mean is not None:
            z = z - self._mean
        root = self._covariance_root
        if root is not None:
            z = torch.linalg.solve_triangular(root, z, upper=False)

        return z

    def _unstandardize(self, z: Tensor) -> Tensor:
        """_untransform helper un-standardizing the input ``z``, which is assumed to
        be a ``d x 1``-dim Tensor, or a ``len(self._not_fixed) x 1``-dim if there are
        fixed indices.
        """
        x = z
        if self._covariance_root is not None:
            x = self._covariance_root @ x
        if self._mean is not None:
            x = x + self._mean
        return x


def get_index_tensors(
    fixed_indices: list[int] | Tensor, d: int
) -> tuple[Tensor, Tensor]:
    """Converts ``fixed_indices`` to a ``d``-dim integral Tensor that is True at indices
    that are contained in ``fixed_indices`` and False otherwise.

    Args:
        fixed_indices: A list or Tensor of integer indices to fix.
        d: The dimensionality of the Tensors to be indexed.

    Returns:
        A Tuple of integral Tensors partitioning [1, d] into indices that are fixed
        (first tensor) and non-fixed (second tensor).
    """
    is_fixed = torch.as_tensor(fixed_indices)
    dtype, device = is_fixed.dtype, is_fixed.device
    dims = torch.arange(d, dtype=dtype, device=device)
    not_fixed = torch.tensor([i for i in dims if i not in is_fixed], dtype=dtype)
    return is_fixed, not_fixed
