#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Smooth spectral matrix functions with analytic, reusable higher derivatives."""

from typing import Any

import torch


def _symmetric(value: Any) -> Any:
    return (value + value.mT) * 0.5


class _SymmetricSylvester(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, coefficient: Any, rhs: Any) -> Any:
        eigenvalues, vectors = torch.linalg.eigh(coefficient)
        if bool((eigenvalues <= 0).any()):
            raise ValueError("The Sylvester coefficient must be positive definite.")
        transformed = vectors.mT @ rhs @ vectors
        result = (
            vectors
            @ (transformed / (eigenvalues.unsqueeze(-1) + eigenvalues.unsqueeze(-2)))
            @ vectors.mT
        )
        ctx.save_for_backward(coefficient, result)
        return result

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any, Any]:
        coefficient, result = ctx.saved_tensors
        # This same implicit rule is differentiable again. Saved outputs retain
        # their Function edge, so no eigenvector derivative is used at any order.
        dual = _SymmetricSylvester.apply(coefficient, _symmetric(gradient))
        return -(dual @ result + result @ dual), dual


class _SymmetricAbsolute(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Any) -> Any:
        eigenvalues, vectors = torch.linalg.eigh(value)
        tolerance = (
            torch.finfo(value.dtype).eps
            * eigenvalues.abs().max().clamp_min(1)
            * value.shape[-1]
        )
        if bool((eigenvalues.abs() <= tolerance).any()):
            raise ValueError(
                "PSD-floor derivatives are undefined at the clipping threshold."
            )
        result = (vectors * eigenvalues.abs().unsqueeze(-2)) @ vectors.mT
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any]:
        value, absolute = ctx.saved_tensors
        # |A|^2=A^2: |A| d|A| + d|A| |A| = A dA + dA A.
        dual = _SymmetricSylvester.apply(absolute, _symmetric(gradient))
        return (value @ dual + dual @ value,)


def smooth_psd_floor(value: Any, floor: float) -> Any:
    """Clip a mixed spectrum without differentiating its eigenvectors.

    Build the positive spectral projector at the midpoint of the clipping gap.
    Using the floor itself as the matrix-sign divider creates ill-conditioned
    Sylvester duals for ordinary zero eigenvalues and a tiny positive floor,
    followed by catastrophic cancellation in higher derivatives. A divider
    anywhere inside the gap gives exactly the same projector. Its choice can
    therefore be detached at every derivative order. Promote float32 internally
    to resolve tiny eigenspaces alongside ordinary covariance scales.
    """
    working = value.to(torch.double) if value.dtype == torch.float32 else value
    working = _symmetric(working)
    identity = torch.eye(value.shape[-1], device=value.device, dtype=working.dtype)
    with torch.no_grad():
        eigenvalues, vectors = torch.linalg.eigh(working)
        tolerance = (
            torch.finfo(working.dtype).eps
            * eigenvalues.abs().amax(-1, keepdim=True).clamp_min(1)
            * value.shape[-1]
        )
        if bool(((eigenvalues - floor).abs() <= tolerance).any()):
            raise ValueError(
                "PSD-floor derivatives are undefined at the clipping threshold."
            )
        below = eigenvalues < floor
        has_below, has_above = bool(below.any()), not bool(below.all())
        divider = None
        if has_below and has_above:
            lower = eigenvalues.masked_fill(~below, -torch.inf).amax()
            upper = eigenvalues.masked_fill(below, torch.inf).amin()
            divider = lower + (upper - lower) * 0.5
    if not has_below:
        return working.to(value.dtype)
    if not has_above:
        return (working * 0 + floor * identity).to(value.dtype)
    assert divider is not None
    # An orthogonal coordinate change can be held fixed at every derivative
    # order: F(A) = Q F(Q^T A Q) Q^T for every fixed Q. Use the detached
    # eigenbasis so small Sylvester duals are never mixed with large ordinary
    # modes in matrix products. The local primal is exactly diagonal in exact
    # arithmetic; remove only its roundoff residual while retaining the full
    # Q^T dA Q differential (including all off-diagonal directions).
    local = vectors.mT @ working @ vectors
    local = torch.diag_embed(eigenvalues) + (local - local.detach())
    shifted = local - divider * identity
    absolute = _SymmetricAbsolute.apply(shifted)
    projector = (identity + torch.linalg.solve(absolute, shifted)) * 0.5
    projected = floor * identity + _symmetric(projector @ (local - floor * identity))
    return (vectors @ projected @ vectors.mT).to(value.dtype)
