#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Bivariate conditioning algorithm for approximating Gaussian probabilities,
see [Genz2016numerical]_ and [Trinh2015bivariate]_.

.. [Trinh2015bivariate]
    G. Trinh and A. Genz. Bivariate conditioning approximations for
    multivariate normal probabilities. Statistics and Computing, 2015.

.. [Genz2016numerical]
    A. Genz and G. Tring. Numerical Computation of Multivariate Normal Probabilities
    using Bivariate Conditioning. Monte Carlo and Quasi-Monte Carlo Methods, 2016.

.. [Gibson1994monte]
    GJ. Gibson, CA Galsbey, and DA Elston. Monte Carlo evaluation of multivariate normal
    integrals and sensitivity to variate ordering. Advances in Numerical Methods and
    Applications. 1994.
"""

from __future__ import annotations

from typing import Any, TypedDict
from warnings import warn

import torch
from botorch.utils.probability.bvn import bvn, bvnmom
from botorch.utils.probability.linalg import (
    augment_cholesky,
    block_matrix_concat,
    PivotedCholesky,
)
from botorch.utils.probability.utils import (
    case_dispatcher,
    ndtr as Phi,
    phi,
    STANDARDIZED_RANGE,
    swap_along_dim_,
)
from botorch.utils.safe_math import log as safe_log, mul as safe_mul
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import LongTensor, Tensor
from torch.nn.functional import pad


class mvnxpbState(TypedDict):
    step: int
    perm: LongTensor
    bounds: Tensor
    piv_chol: PivotedCholesky
    plug_ins: Tensor
    log_prob: Tensor
    log_prob_extra: Tensor | None


class MVNXPB:
    r"""An algorithm for approximating Gaussian probabilities ``P(X \in bounds)``, where
    ``X ~ N(0, covariance_matrix)``.
    """

    def __init__(self, covariance_matrix: Tensor, bounds: Tensor) -> None:
        r"""Initializes an MVNXPB instance.

        Args:
            covariance_matrix: Covariance matrices of shape ``batch_shape x [n, n]``.
            bounds: Tensor of lower and upper bounds, ``batch_shape x [n, 2]``. These
                bounds are standardized internally and clipped to STANDARDIZED_RANGE.
        """
        *batch_shape, _, n = covariance_matrix.shape
        device = covariance_matrix.device
        dtype = covariance_matrix.dtype
        perm = torch.arange(0, n, device=device).expand(*batch_shape, n).contiguous()

        # Standardize covariance matrices and bounds
        var = covariance_matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        std = var.sqrt()
        istd = var.rsqrt()
        matrix = istd * covariance_matrix * istd.transpose(-1, -2)

        # Clip first to avoid differentiating through ``istd * inf``
        bounds = istd * bounds.clip(*(std * lim for lim in STANDARDIZED_RANGE))

        # Initialize partial pivoted Cholesky
        piv_chol = PivotedCholesky(
            step=0,
            perm=perm.clone(),
            diag=std.squeeze(-1).clone(),
            tril=matrix.tril(),
        )
        self.step = 0
        self.perm = perm
        self.bounds = bounds
        self.piv_chol = piv_chol
        self.plug_ins = torch.full(
            batch_shape + [n], float("nan"), device=device, dtype=dtype
        )
        self.log_prob = torch.zeros(batch_shape, device=device, dtype=dtype)
        self.log_prob_extra: Tensor | None = None

    @classmethod
    def build(
        cls,
        step: int,
        perm: Tensor,
        bounds: Tensor,
        piv_chol: PivotedCholesky,
        plug_ins: Tensor,
        log_prob: Tensor,
        log_prob_extra: Tensor | None = None,
    ) -> MVNXPB:
        r"""Creates an MVNXPB instance from raw arguments. Unlike MVNXPB.__init__,
        this methods does not preprocess or copy terms.

        Args:
            step: Integer used to track the solver's progress.
            bounds: Tensor of lower and upper bounds, ``batch_shape x [n, 2]``.
            piv_chol: A PivotedCholesky instance for the system.
            plug_ins: Tensor of plug-in estimators used to update lower and upper bounds
                on random variables that have yet to be integrated out.
            log_prob: Tensor of log probabilities.
            log_prob_extra: Tensor of conditional log probabilities for the next random
                variable. Used when integrating over an odd number of random variables.
        """
        new = cls.__new__(cls)
        new.step = step
        new.perm = perm
        new.bounds = bounds
        new.piv_chol = piv_chol
        new.plug_ins = plug_ins
        new.log_prob = log_prob
        new.log_prob_extra = log_prob_extra
        return new

    def solve(self, num_steps: int | None = None, eps: float = 1e-10) -> Tensor:
        r"""Runs the MVNXPB solver instance for a fixed number of steps.

        Calculates a bivariate conditional approximation to P(X \in bounds), where
        X ~ N(0, Σ). For details, see [Genz2016numerical] or [Trinh2015bivariate]_.
        """
        if self.step > self.piv_chol.step:
            raise ValueError("Invalid state: solver ran ahead of matrix decomposition.")

        # Unpack some terms
        start = self.step
        bounds = self.bounds
        piv_chol = self.piv_chol
        L = piv_chol.tril
        y = self.plug_ins

        # Subtract marginal log probability of final term from previous result if
        # it did not fit in a block.
        ndim = y.shape[-1]
        if ndim > start and start % 2:
            self.log_prob = self.log_prob - self.log_prob_extra
            self.log_prob_extra = None

        # Iteratively compute bivariate conditional approximation
        num_steps = num_steps or ndim - start
        for i in range(start, start + num_steps):
            should_update_chol = self.step == piv_chol.step

            # Determine next pivot element
            if should_update_chol:
                pivot = self.select_pivot()
            else:  # pivot using order specified by precomputed pivoted Cholesky step
                mask = self.perm[..., i:] == piv_chol.perm[..., i : i + 1]
                pivot = i + torch.nonzero(mask, as_tuple=True)[-1]

            if pivot is not None and torch.any(pivot > i):
                self.pivot_(pivot=pivot)

            # Compute whitened bounds conditional on preceding plug-ins
            Lii = L[..., i, i].clone()
            if should_update_chol:
                Lii = Lii.clip(min=0).sqrt()  # conditional stddev
            inv_Lii = Lii.reciprocal()
            bounds_i = bounds[..., i, :].clone()
            if i != 0:
                bounds_i = bounds_i - torch.sum(
                    L[..., i, :i].clone() * y[..., :i].clone(), dim=-1, keepdim=True
                )
            lb, ub = (inv_Lii.unsqueeze(-1) * bounds_i).unbind(dim=-1)

            # Initialize ``i``-th plug-in value as univariate conditional expectation
            Phi_i = Phi(ub) - Phi(lb)
            small = Phi_i <= i * eps
            y[..., i] = case_dispatcher(  # used to select next pivot
                out=(phi(lb) - phi(ub)) / Phi_i,
                cases=(  # fallback cases for enhanced numerical stability
                    (
                        lambda small=small, lb=lb: small & (lb < -9),
                        lambda m, ub=ub: ub[m],
                    ),
                    (
                        lambda small=small, lb=lb: small & (lb > 9),
                        lambda m, lb=lb: lb[m],
                    ),
                    (
                        lambda small=small: small,
                        lambda m, lb=lb, ub=ub: 0.5 * (lb[m] + ub[m]),
                    ),
                ),
            )

            # Maybe finalize the current block
            if i and i % 2:
                h = i - 1
                blk = slice(h, i + 1)
                Lhh = L[..., h, h].clone()
                Lih = L[..., i, h].clone()

                std_i = (Lii.square() + Lih.square()).sqrt()
                istds = 1 / torch.stack([Lhh, std_i], -1)
                blk_bounds = bounds[..., blk, :].clone()
                if i > 1:
                    blk_bounds = blk_bounds - (
                        L[..., blk, : i - 1].clone() @ y[..., : i - 1, None].clone()
                    )

                blk_lower, blk_upper = (
                    pair.unbind(-1)  # pair of bounds for ``yh`` and ``yi``
                    for pair in safe_mul(istds.unsqueeze(-1), blk_bounds).unbind(-1)
                )
                blk_corr = Lhh * Lih * istds.prod(-1)
                blk_prob = bvn(blk_corr, *blk_lower, *blk_upper)
                zh, zi = bvnmom(blk_corr, *blk_lower, *blk_upper, p=blk_prob)

                # Replace 1D expectations with 2D ones ``L[blk, blk]^{-1} y[..., blk]``
                mask = blk_prob > 0
                y[..., h] = torch.where(mask, zh, 0)
                y[..., i] = torch.where(mask, inv_Lii * (std_i * zi - Lih * zh), 0)

                # Update running approximation to log probability
                self.log_prob = self.log_prob + safe_log(blk_prob)

            self.step += 1
            if should_update_chol:
                piv_chol.update_(eps=eps)

        # Factor in univariate probability if final term fell outside of a block.
        if self.step % 2:
            self.log_prob_extra = safe_log(Phi_i)
            self.log_prob = self.log_prob + self.log_prob_extra

        return self.log_prob

    def select_pivot(self) -> LongTensor | None:
        r"""GGE variable prioritization strategy from [Gibson1994monte]_.

        Returns the index of the random variable least likely to satisfy its bounds
        when conditioning on the previously integrated random variables ``X[:t - 1]``
        attaining the values of plug-in estimators ``y[:t - 1]``. Equivalently,
        ```
        argmin_{i = t, ..., n} P(X[i] \in bounds[i] | X[:t-1] = y[:t -1]),
        ```
        where ``t`` denotes the current step."""
        i = self.piv_chol.step
        L = self.piv_chol.tril
        bounds = self.bounds
        if i:
            bounds = bounds[..., i:, :] - L[..., i:, :i] @ self.plug_ins[..., :i, None]

        inv_stddev = torch.diagonal(L, dim1=-2, dim2=-1)[..., i:].clip(min=0).rsqrt()
        probs_1d = Phi(inv_stddev.unsqueeze(-1) * bounds).diff(dim=-1).squeeze(-1)
        return i + torch.argmin(probs_1d, dim=-1)

    def pivot_(self, pivot: LongTensor) -> None:
        r"""Swap random variables at ``pivot`` and ``step`` positions."""
        step = self.step
        if self.piv_chol.step == step:
            self.piv_chol.pivot_(pivot)
        elif self.step > self.piv_chol.step:
            raise ValueError

        for tnsr in (self.perm, self.bounds):
            swap_along_dim_(tnsr, i=self.step, j=pivot, dim=pivot.ndim)

    def __getitem__(self, key: Any) -> MVNXPB:
        return self.build(
            step=self.step,
            perm=self.perm[key],
            bounds=self.bounds[key],
            piv_chol=self.piv_chol[key],
            plug_ins=self.plug_ins[key],
            log_prob=self.log_prob[key],
            log_prob_extra=(
                None if self.log_prob_extra is None else self.log_prob_extra[key]
            ),
        )

    def concat(self, other: MVNXPB, dim: int) -> MVNXPB:
        if not isinstance(other, MVNXPB):
            raise TypeError(
                f"Expected `other` to be {type(self)} typed but was {type(other)}."
            )

        batch_ndim = self.log_prob.ndim
        if dim > batch_ndim or dim < -batch_ndim:
            raise ValueError(f"`dim={dim}` is not a valid batch dimension.")

        state_dict = self.asdict()
        for key, _other in other.asdict().items():
            _self = state_dict.get(key)
            if _self is None and _other is None:
                continue

            if type(_self) is not type(_other):
                raise TypeError(
                    f"Concatenation failed: `self.{key}` has type {type(_self)}, "
                    f"but `other.{key}` is of type {type(_self)}."
                )

            if isinstance(_self, PivotedCholesky):
                state_dict[key] = _self.concat(_other, dim=dim)
            elif isinstance(_self, Tensor):
                state_dict[key] = torch.concat((_self, _other), dim=dim)
            elif _self != _other:
                raise ValueError(
                    f"Concatenation failed: `self.{key}` does not equal `other.{key}`."
                )

        return self.build(**state_dict)

    def expand(self, *sizes: int) -> MVNXPB:
        state_dict = self.asdict()
        state_dict["piv_chol"] = state_dict["piv_chol"].expand(*sizes)
        for name, ndim in {
            "bounds": 2,
            "perm": 1,
            "plug_ins": 1,
            "log_prob": 0,
            "log_prob_extra": 0,
        }.items():
            src = state_dict[name]
            if isinstance(src, Tensor):
                state_dict[name] = src.expand(
                    sizes + src.shape[-ndim:] if ndim else sizes
                )
        return self.build(**state_dict)

    def augment(
        self,
        covariance_matrix: Tensor,
        bounds: Tensor,
        cross_covariance_matrix: Tensor,
        disable_pivoting: bool = False,
        jitter: float | None = None,
        max_tries: int | None = None,
    ) -> MVNXPB:
        r"""Augment an ``n``-dimensional MVNXPB instance to include ``m``
        additional random variables.
        """
        n = self.perm.shape[-1]
        m = covariance_matrix.shape[-1]
        if n != self.piv_chol.step:
            raise NotImplementedError(
                "Augmentation of incomplete solutions not implemented yet."
            )

        var = covariance_matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        std = var.sqrt()
        istd = var.rsqrt()
        Kmn = istd * cross_covariance_matrix
        if self.piv_chol.diag is None:
            diag = pad(std.squeeze(-1), (cross_covariance_matrix.shape[-1], 0), value=1)
        else:
            Kmn = Kmn * (1 / self.piv_chol.diag).unsqueeze(-2)
            diag = torch.concat([self.piv_chol.diag, std.squeeze(-1)], -1)

        # Augment partial pivoted Cholesky factor
        Kmm = istd * covariance_matrix * istd.transpose(-1, -2)
        Lnn = self.piv_chol.tril
        try:
            L = augment_cholesky(Laa=Lnn, Kba=Kmn, Kbb=Kmm, jitter=jitter)
        except NotPSDError:
            warn("Joint covariance matrix not positive definite, attempting recovery.")
            Knn = Lnn @ Lnn.transpose(-1, -2)
            Knm = Kmn.transpose(-1, -2)
            K = block_matrix_concat(blocks=((Knn, Knm), (Kmn, Kmm)))
            L = psd_safe_cholesky(K, jitter=jitter, max_tries=max_tries)

        if not disable_pivoting:
            Lmm = L[..., n:, n:].clone()
            L[..., n:, n:] = (Lmm @ Lmm.transpose(-2, -1)).tril()

        _bounds = istd * bounds.clip(*(std * lim for lim in STANDARDIZED_RANGE))
        _perm = torch.arange(n, n + m, dtype=self.perm.dtype, device=self.perm.device)
        _perm = _perm.expand(covariance_matrix.shape[:-2] + (m,))

        piv_chol = PivotedCholesky(
            step=n + m if disable_pivoting else n,
            tril=L.contiguous(),
            perm=torch.cat([self.piv_chol.perm, _perm], dim=-1).contiguous(),
            diag=diag,
        )

        return self.build(
            step=self.step,
            perm=torch.cat([self.perm, _perm], dim=-1),
            bounds=torch.cat([self.bounds, _bounds], dim=-2),
            piv_chol=piv_chol,
            plug_ins=pad(self.plug_ins, (0, m), value=float("nan")),
            log_prob=self.log_prob,
            log_prob_extra=self.log_prob_extra,
        )

    def detach(self) -> MVNXPB:
        state_dict = self.asdict()
        for key, obj in state_dict.items():
            if isinstance(obj, (PivotedCholesky, Tensor)):
                state_dict[key] = obj.detach()
        return self.build(**state_dict)

    def clone(self) -> MVNXPB:
        state_dict = self.asdict()
        for key, obj in state_dict.items():
            if isinstance(obj, (PivotedCholesky, Tensor)):
                state_dict[key] = obj.clone()
        return self.build(**state_dict)

    def asdict(self) -> mvnxpbState:
        return mvnxpbState(
            step=self.step,
            perm=self.perm,
            bounds=self.bounds,
            piv_chol=self.piv_chol,
            plug_ins=self.plug_ins,
            log_prob=self.log_prob,
            log_prob_extra=self.log_prob_extra,
        )
