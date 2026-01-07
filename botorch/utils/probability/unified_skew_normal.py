#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from inspect import getmembers

import torch
from botorch.utils.probability.linalg import augment_cholesky, block_matrix_concat
from botorch.utils.probability.mvnxpb import MVNXPB
from botorch.utils.probability.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)
from linear_operator.operators import LinearOperator
from linear_operator.utils.errors import NotPSDError
from torch import Tensor
from torch.distributions.multivariate_normal import Distribution, MultivariateNormal
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad


class UnifiedSkewNormal(Distribution):
    arg_constraints = {}

    def __init__(
        self,
        trunc: TruncatedMultivariateNormal,
        gauss: MultivariateNormal,
        cross_covariance_matrix: Tensor | LinearOperator,
        validate_args: bool | None = None,
    ):
        r"""Unified Skew Normal distribution of `Y | a < X < b` for jointly Gaussian
        random vectors `X ∈ R^m` and `Y ∈ R^n`.

        Batch shapes `trunc.batch_shape` and `gauss.batch_shape` must be broadcastable.
        Care should be taken when choosing `trunc.batch_shape`. When `trunc` is of lower
        batch dimensionality than `gauss`, the user should consider expanding `trunc` to
        hasten `UnifiedSkewNormal.log_prob`. In these cases, it is suggested that the
        user invoke `trunc.solver` before calling `trunc.expand` to avoid paying for
        multiple, identical solves.

        Args:
            trunc: Distribution of `Z = (X | a < X < b) ∈ R^m`.
            gauss: Distribution of `Y ∈ R^n`.
            cross_covariance_matrix: Cross-covariance `Cov(X, Y) ∈ R^{m x n}`.
            validate_args: Optional argument to super().__init__.
        """
        if len(trunc.event_shape) != len(gauss.event_shape):
            raise ValueError(
                f"{len(trunc.event_shape)}-dimensional `trunc` incompatible with"
                f"{len(gauss.event_shape)}-dimensional `gauss`."
            )
        # LinearOperator currently doesn't support torch.linalg.solve_triangular,
        # so for the time being, we cast the operator to dense here
        if isinstance(cross_covariance_matrix, LinearOperator):
            cross_covariance_matrix = cross_covariance_matrix.to_dense()
        try:
            batch_shape = torch.broadcast_shapes(trunc.batch_shape, gauss.batch_shape)
        except RuntimeError as e:
            raise ValueError("Incompatible batch shapes") from e

        super().__init__(
            batch_shape=batch_shape,
            event_shape=gauss.event_shape,
            validate_args=validate_args,
        )
        self.trunc = trunc
        self.gauss = gauss
        self.cross_covariance_matrix = cross_covariance_matrix
        if self._validate_args:
            try:
                # calling _orthogonalized_gauss first makes the following call
                # _orthogonalized_gauss.scale_tril which is used by self.rsample
                self._orthogonalized_gauss
                self.scale_tril
            except Exception as e:
                # error could be thrown by linalg.augment_cholesky (NotPSDError)
                # or torch.linalg.cholesky (with "positive-definite" in the message)
                if (
                    isinstance(e, NotPSDError)
                    or "positive-definite" in str(e)
                    or "PositiveDefinite" in str(e)
                ):
                    e = ValueError(
                        "UnifiedSkewNormal is only well-defined for positive definite"
                        " joint covariance matrices."
                    )
                raise e

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Computes the log probability `ln p(Y = value | a < X < b)`."""
        event_ndim = len(self.event_shape)
        if value.ndim < event_ndim or value.shape[-event_ndim:] != self.event_shape:
            raise ValueError(
                f"`value` with shape {value.shape} does not comply with the instance's"
                f"`event_shape` of {self.event_shape}."
            )

        # Iterate with a fixed batch size to keep memory overhead in check
        i = 0
        pre_shape = value.shape[: -len(self.event_shape) - len(self.batch_shape)]
        batch_size = self.batch_shape.numel()
        log_probs = torch.empty(
            pre_shape.numel() * batch_size, device=value.device, dtype=value.dtype
        )
        for batch in value.view(-1, *value.shape[len(pre_shape) :]):
            log_probs[i : i + batch_size] = self._log_prob(batch).view(-1)
            i += batch_size

        return log_probs.view(pre_shape + self.batch_shape)

    def _log_prob(self, value: Tensor) -> Tensor:
        r"""Computes the log probability `ln p(Y = value | a < X < b)`."""
        # Center by subtracting E[X | Y = value] from `bounds`.
        bounds = (
            self.trunc.bounds
            - self.trunc.loc.unsqueeze(-1)
            - self._iKyy_Kyx.transpose(-2, -1) @ (value - self.gauss.loc).unsqueeze(-1)
        )

        # Approximately solve for MVN CDF
        solver = MVNXPB(covariance_matrix=self._K_schur_Kyy, bounds=bounds)

        # p(Y = value | a < X < b) = P(a < X < b | Y = value)p(Y = value)/P(a < X < b)
        return solver.solve() + self.gauss.log_prob(value) - self.trunc.log_partition

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:  # noqa: B008
        r"""Draw samples from the Unified Skew Normal.

        Args:
            sample_shape: The shape of the samples.

        Returns:
            The (sample_shape x batch_shape x event_shape) tensor of samples.
        """
        residuals = self._orthogonalized_gauss.rsample(sample_shape=sample_shape)
        trunc_rvs = self.trunc.rsample(sample_shape=sample_shape) - self.trunc.loc
        cond_expectations = self.gauss.loc + trunc_rvs @ self._iKxx_Kxy
        return cond_expectations + residuals

    def expand(
        self, batch_shape: Sequence[int], _instance: UnifiedSkewNormal = None
    ) -> UnifiedSkewNormal:
        new = self._get_checked_instance(UnifiedSkewNormal, _instance)
        super(UnifiedSkewNormal, new).__init__(
            batch_shape=batch_shape, event_shape=self.event_shape, validate_args=False
        )

        new._validate_args = self._validate_args
        new.gauss = self.gauss.expand(batch_shape=batch_shape)
        new.trunc = self.trunc.expand(batch_shape=batch_shape)
        new.cross_covariance_matrix = self.cross_covariance_matrix.expand(
            batch_shape + self.cross_covariance_matrix.shape[-2:]
        )

        # Expand cached properties
        for name, _ in getmembers(
            UnifiedSkewNormal, lambda x: isinstance(x, lazy_property)
        ):
            if name not in self.__dict__:
                continue

            obj = getattr(self, name)
            if isinstance(obj, Tensor):
                base = obj if (obj._base is None) else obj._base
                new_obj = obj.expand(batch_shape + base.shape)
            elif isinstance(obj, Distribution):
                new_obj = obj.expand(batch_shape=batch_shape)
            else:
                raise TypeError(
                    f"Type {type(obj)} of UnifiedSkewNormal's lazy property "
                    f"{name} not supported."
                )

            setattr(new, name, new_obj)
        return new

    def __repr__(self) -> str:
        args_string = ", ".join(
            (
                f"trunc: {self.trunc}",
                f"gauss: {self.gauss}",
                f"cross_covariance_matrix: {self.cross_covariance_matrix.shape}",
            )
        )
        return self.__class__.__name__ + "(" + args_string + ")"

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        Kxx = self.trunc.covariance_matrix
        Kxy = self.cross_covariance_matrix
        Kyy = self.gauss.covariance_matrix
        return block_matrix_concat(blocks=([Kxx, Kxy], [Kxy.transpose(-2, -1), Kyy]))

    @lazy_property
    def scale_tril(self) -> Tensor:
        Lxx = self.trunc.scale_tril
        Lyx = self._iLxx_Kxy.transpose(-2, -1)
        if "_orthogonalized_gauss" in self.__dict__:
            n = Lyx.shape[-2]
            Lyy = self._orthogonalized_gauss.scale_tril
            return block_matrix_concat(blocks=([pad(Lxx, (0, n))], [Lyx, Lyy]))
        return augment_cholesky(Laa=Lxx, Lba=Lyx, Kbb=self.gauss.covariance_matrix)

    @lazy_property
    def _orthogonalized_gauss(self) -> MultivariateNormal:
        r"""Distribution of `Y ⊥ X = Y - E[Y | X]`, where `Y ~ gauss` and `X ~ untrunc`
        is the untruncated version of `Z ~ trunc`."""
        n = self.gauss.loc.shape[-1]
        parameters = {"loc": torch.zeros_like(self.gauss.loc)}
        if "scale_tril" in self.__dict__:
            parameters["scale_tril"] = self.scale_tril[..., -n:, -n:]
        else:
            beta = self._iLxx_Kxy
            parameters["covariance_matrix"] = (
                self.gauss.covariance_matrix - beta.transpose(-1, -2) @ beta
            )
        return MultivariateNormal(**parameters, validate_args=self._validate_args)

    @lazy_property
    def _iLyy_Kyx(self) -> Tensor:
        r"""Cov(Y, Y)^{-1/2}Cov(Y, X)`."""
        return torch.linalg.solve_triangular(
            self.gauss.scale_tril,
            self.cross_covariance_matrix.transpose(-1, -2),
            upper=False,
        )

    @lazy_property
    def _iKyy_Kyx(self) -> Tensor:
        r"""Cov(Y, Y)^{-1}Cov(Y, X)`."""
        return torch.linalg.solve_triangular(
            self.gauss.scale_tril.transpose(-1, -2),
            self._iLyy_Kyx,
            upper=True,
        )

    @lazy_property
    def _iLxx_Kxy(self) -> Tensor:
        r"""Cov(X, X)^{-1/2}Cov(X, Y)`."""
        return torch.linalg.solve_triangular(
            self.trunc.scale_tril, self.cross_covariance_matrix, upper=False
        )

    @lazy_property
    def _iKxx_Kxy(self) -> Tensor:
        r"""Cov(X, X)^{-1}Cov(X, Y)`."""
        return torch.linalg.solve_triangular(
            self.trunc.scale_tril.transpose(-1, -2),
            self._iLxx_Kxy,
            upper=True,
        )

    @lazy_property
    def _K_schur_Kyy(self) -> Tensor:
        r"""Cov(X, X) - Cov(X, Y)Cov(Y, Y)^{-1} Cov(Y, X)`."""
        beta = self._iLyy_Kyx
        return self.trunc.covariance_matrix - beta.transpose(-1, -2) @ beta
