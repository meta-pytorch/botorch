#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass, InitVar
from itertools import chain
from typing import Any

import torch
from botorch.utils.probability.utils import swap_along_dim_
from linear_operator.utils.errors import NotPSDError
from torch import LongTensor, Tensor
from torch.nn.functional import pad


def block_matrix_concat(blocks: Sequence[Sequence[Tensor]]) -> Tensor:
    rows = []
    shape = torch.broadcast_shapes(*(x.shape[:-2] for x in chain.from_iterable(blocks)))
    for tensors in blocks:
        parts = [x.expand(*shape, *x.shape[-2:]) for x in tensors]
        if len(parts) > 1:
            rows.append(torch.cat(parts, dim=-1))
        else:
            rows.extend(parts)
    return torch.concat(rows, dim=-2)


def augment_cholesky(
    Laa: Tensor,
    Kbb: Tensor,
    Kba: Tensor | None = None,
    Lba: Tensor | None = None,
    jitter: float | None = None,
) -> Tensor:
    r"""Computes the Cholesky factor of a block matrix `K = [[Kaa, Kab], [Kba, Kbb]]`
    based on a precomputed Cholesky factor `Kaa = Laa Laa^T`.

    Args:
        Laa: Cholesky factor of K's upper left block.
        Kbb: Lower-right block of K.
        Kba: Lower-left block of K.
        Lba: Precomputed solve `Kba Laa^{-T}`.
        jitter: Optional nugget to be added to the diagonal of Kbb.
    """
    if not (Kba is None) ^ (Lba is None):
        raise ValueError("One and only one of `Kba` or `Lba` must be provided.")

    if jitter is not None:
        Kbb = Kbb.clone()
        Kbb.diagonal(dim1=-2, dim2=-1).add_(jitter)

    if Lba is None:
        Lba = torch.linalg.solve_triangular(
            Laa.transpose(-2, -1), Kba, left=False, upper=True
        )

    Lbb, info = torch.linalg.cholesky_ex(Kbb - Lba @ Lba.transpose(-2, -1))
    if info.any():
        raise NotPSDError(
            "Schur complement of `K` with respect to `Kaa` not PSD for the given "
            "Cholesky factor `Laa`"
            f"{'.' if jitter is None else f' and nugget jitter={jitter}.'}"
        )

    n = Lbb.shape[-1]
    return block_matrix_concat(blocks=([pad(Laa, (0, n))], [Lba, Lbb]))


@dataclass
class PivotedCholesky:
    step: int
    tril: Tensor
    perm: LongTensor
    diag: Tensor | None = None
    validate_init: InitVar[bool] = True

    def __post_init__(self, validate_init: bool = True):
        if not validate_init:
            return

        if self.tril.shape[-2] != self.tril.shape[-1]:
            raise ValueError(
                f"Expected square matrices but `matrix` has shape `{self.tril.shape}`."
            )

        if self.perm.shape != self.tril.shape[:-1]:
            raise ValueError(
                f"`perm` of shape `{self.perm.shape}` incompatible with "
                f"`matrix` of shape `{self.tril.shape}`."
            )

        if self.diag is not None and self.diag.shape != self.tril.shape[:-1]:
            raise ValueError(
                f"`diag` of shape `{self.diag.shape}` incompatible with "
                f"`matrix` of shape `{self.tril.shape}`."
            )

    def __getitem__(self, key: Any) -> PivotedCholesky:
        return PivotedCholesky(
            step=self.step,
            tril=self.tril[key],
            perm=self.perm[key],
            diag=None if self.diag is None else self.diag[key],
        )

    def update_(self, eps: float = 1e-10) -> None:
        r"""Performs a single matrix decomposition step."""
        i = self.step
        L = self.tril
        Lii = self.tril[..., i, i].clone().clip(min=0).sqrt()

        # Finalize `i-th` row and column of Cholesky factor
        L[..., i, i] = Lii
        L[..., i, i + 1 :] = 0
        L[..., i + 1 :, i] = L[..., i + 1 :, i].clone() / Lii.unsqueeze(-1)

        # Update `tril(L[i + 1:, i + 1:])` to be the lower triangular part
        # of the Schur complement of `cov` with respect to `cov[:i, :i]`.
        rank1 = L[..., i + 1 :, i : i + 1].clone()
        rank1 = (rank1 * rank1.transpose(-1, -2)).tril()
        L[..., i + 1 :, i + 1 :] = L[..., i + 1 :, i + 1 :].clone() - rank1
        L[..., i:, i][Lii <= i * eps] = 0  # numerical stability clause
        self.step += 1

    def pivot_(self, pivot: LongTensor) -> None:
        *batch_shape, _, size = self.tril.shape
        if pivot.shape != tuple(batch_shape):
            raise ValueError("Argument `pivot` does to match with batch shape`.")

        # Perform basic swaps
        for key in ("perm", "diag"):
            tnsr = getattr(self, key, None)
            if tnsr is not None:
                swap_along_dim_(tnsr, i=self.step, j=pivot, dim=tnsr.ndim - 1)

        # Perform matrix swaps; prealloacte buffers for row/column linear indices
        size2 = size**2
        min_pivot = pivot.min()
        tkwargs = {"device": pivot.device, "dtype": pivot.dtype}
        buffer_col = torch.arange(size * (1 + min_pivot), size2, size, **tkwargs)
        buffer_row = torch.arange(0, max(self.step, pivot.max()), **tkwargs)
        head = buffer_row[: self.step]

        indices_v1 = []
        indices_v2 = []
        for i, piv in enumerate(pivot.view(-1, 1)):
            v1 = pad(piv, (1, 0), value=self.step).unsqueeze(-1)
            v2 = pad(piv, (0, 1), value=self.step).unsqueeze(-1)
            start = i * size2

            indices_v1.extend((start + v1 + size * v1).ravel())
            indices_v2.extend((start + v2 + size * v2).ravel())

            indices_v1.extend((start + size * v1 + head).ravel())
            indices_v2.extend((start + size * v2 + head).ravel())

            tail = buffer_col[piv - min_pivot :]
            indices_v1.extend((start + v1 + tail).ravel())
            indices_v2.extend((start + v2 + tail).ravel())

            interior = buffer_row[min(piv, self.step + 1) : piv]
            indices_v1.extend(start + size * interior + self.step)
            indices_v2.extend(start + size * piv + interior)

        swap_along_dim_(
            self.tril.view(-1),
            i=torch.as_tensor(indices_v1, **tkwargs),
            j=torch.as_tensor(indices_v2, **tkwargs),
            dim=0,
        )

    def expand(self, *sizes: int) -> PivotedCholesky:
        fields = {}
        for name, ndim in {"perm": 1, "diag": 1, "tril": 2}.items():
            src = getattr(self, name)
            if src is not None:
                fields[name] = src.expand(sizes + src.shape[-ndim:])
        return type(self)(step=self.step, **fields)

    def concat(self, other: PivotedCholesky, dim: int = 0) -> PivotedCholesky:
        if self.step != other.step:
            raise ValueError("Cannot conncatenate decompositions at different steps.")

        fields = {}
        for name in ("tril", "perm", "diag"):
            a = getattr(self, name)
            b = getattr(other, name)
            if type(a) is not type(b):
                raise NotImplementedError(f"Types of field {name} do not match.")

            if a is not None:
                fields[name] = torch.concat((a, b), dim=dim)

        return type(self)(step=self.step, **fields)

    def detach(self) -> PivotedCholesky:
        fields = {}
        for name in ("tril", "perm", "diag"):
            obj = getattr(self, name)
            if obj is not None:
                fields[name] = obj.detach()
        return type(self)(step=self.step, **fields)

    def clone(self) -> PivotedCholesky:
        fields = {}
        for name in ("tril", "perm", "diag"):
            obj = getattr(self, name)
            if obj is not None:
                fields[name] = obj.clone()
        return type(self)(step=self.step, **fields)
