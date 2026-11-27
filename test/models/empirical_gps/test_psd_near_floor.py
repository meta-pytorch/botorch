#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext as no_host

import pytest
import torch
from botorch.models.empirical_gps._spectral import smooth_psd_floor
from botorch.models.empirical_gps.utils import project_psd

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def psd_projection(value, floor):
    return project_psd(value, min_eigval=floor)


def _stable_loewner_vjp(value, gradient, floor):
    """Independent divided differences; no absolute or Sylvester algebra."""
    symmetric = (value.double() + value.double().mT) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    positive = eigenvalues > floor
    left, right = positive.unsqueeze(-1), positive.unsqueeze(-2)
    gap = (eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)).abs()
    numerator = (
        torch.maximum(eigenvalues.unsqueeze(-1), eigenvalues.unsqueeze(-2)) - floor
    )
    mixed = left ^ right
    divided_difference = torch.where(
        left & right,
        1.0,
        torch.where(mixed, numerator / torch.where(mixed, gap, 1.0), 0.0),
    )
    symmetric_gradient = (gradient.double() + gradient.double().mT) * 0.5
    return (
        vectors
        @ (divided_difference * (vectors.mT @ symmetric_gradient @ vectors))
        @ vectors.mT
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("rotation", ["none", "hadamard", "random"])
@pytest.mark.parametrize("small", [0.0, -1e-8])
def test_tiny_floor_mixed_zero_positive_vjp_and_hvp_match_stable_oracle(
    device, dtype, rotation, small
):
    if rotation == "hadamard":
        basis = (
            torch.tensor(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                dtype=torch.double,
                device=device,
            )
            * 0.5
        )
    elif rotation == "random":
        generator = torch.Generator(device=device).manual_seed(41)
        basis, _ = torch.linalg.qr(
            torch.randn(4, 4, dtype=torch.double, device=device, generator=generator)
        )
    else:
        basis = torch.eye(4, dtype=torch.double, device=device)
    diagonal = basis.new_tensor([small, small, 1.0, 2.0])
    value = (basis @ torch.diag(diagonal) @ basis.mT).to(dtype).requires_grad_()
    gradient = value.new_tensor(
        [
            [0.7, 0.3, -0.5, 0.2],
            [0.3, -0.8, 0.4, 0.6],
            [-0.5, 0.4, 1.1, 0.9],
            [0.2, 0.6, 0.9, -0.2],
        ]
    )
    cross_block = basis.new_tensor(
        [
            [0.0, 0.0, 0.3, -0.4],
            [0.0, 0.0, 0.2, 0.5],
            [0.3, 0.2, 0.0, 0.0],
            [-0.4, 0.5, 0.0, 0.0],
        ]
    )
    direction = (basis @ cross_block @ basis.mT).to(dtype)
    floor, step = 1e-10, 1e-5
    plus = value.detach().double() + step * direction.double()
    minus = value.detach().double() - step * direction.double()
    # A generic 1e-5 perturbation would cross a 1e-10 floor and would not
    # approximate the local Hessian. Cross-block directions shift zero
    # eigenvalues negative only at O(step^2); explicitly check that neither
    # finite-difference endpoint changes clipping membership.
    positive = torch.linalg.eigvalsh(value.detach().double()) > floor
    assert torch.equal(torch.linalg.eigvalsh(plus) > floor, positive)
    assert torch.equal(torch.linalg.eigvalsh(minus) > floor, positive)
    expected = _stable_loewner_vjp(value.detach(), gradient, floor)
    numerical_hvp = (
        _stable_loewner_vjp(plus, gradient, floor)
        - _stable_loewner_vjp(minus, gradient, floor)
    ) / (2 * step)
    with no_host():
        output = smooth_psd_floor(value, floor)
        actual = torch.autograd.grad(
            (output * gradient).sum(), value, create_graph=True
        )[0]
        hvp = (
            torch.autograd.grad((actual * direction).sum(), value)[0]
            if actual.requires_grad
            else torch.zeros_like(value)
        )
    tolerance = 1e-7 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(
        actual.double(), expected, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        hvp.double(), numerical_hvp, atol=tolerance, rtol=tolerance
    )
    assert output.device == value.device and output.dtype == value.dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
def test_gap_projector_mixed_batch_and_single_side_graphs(device, dtype):
    diagonal = torch.tensor(
        [[-1.0, -1.0, -2.0, -2.0], [0.2, 0.2, 1.0, 2.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=dtype,
        device=device,
    )
    matrices = torch.diag_embed(diagonal).requires_grad_()
    gradient = torch.arange(1, 17, dtype=dtype, device=device).reshape(4, 4)
    output = smooth_psd_floor(matrices, 1e-10)
    torch.testing.assert_close(output, torch.diag_embed(diagonal.clamp_min(1e-10)))
    actual = torch.autograd.grad(
        (output * gradient).sum(), matrices, create_graph=True
    )[0]
    expected = _stable_loewner_vjp(matrices.detach(), gradient, 1e-10)
    torch.testing.assert_close(
        actual.double(),
        expected,
        atol=1e-6 if dtype == torch.float32 else 1e-12,
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
    )
    second = torch.autograd.grad(actual.square().sum(), matrices, create_graph=True)[0]
    third = torch.autograd.grad(second.square().sum(), matrices)[0]
    assert torch.isfinite(second).all() and torch.isfinite(third).all()
    for tensor in (
        matrices[:1].detach().requires_grad_(),
        matrices[1:2].detach().requires_grad_(),
    ):
        result = smooth_psd_floor(tensor, 1e-10)
        assert result.requires_grad
        derivative = torch.autograd.grad(result.square().sum(), tensor)[0]
        expected_derivative = (
            torch.zeros_like(tensor)
            if bool((tensor.detach().diagonal(dim1=-2, dim2=-1) < 1e-10).all())
            else 2 * tensor
        )
        torch.testing.assert_close(derivative, expected_derivative)


@pytest.mark.parametrize("device", DEVICES)
def test_gap_projector_retains_explicit_nondifferentiable_threshold_error(device):
    value = torch.diag(
        torch.tensor([0.1, -1.0, 2.0], dtype=torch.double, device=device)
    ).requires_grad_()
    with pytest.raises(ValueError, match="clipping threshold"):
        smooth_psd_floor(value, 0.1)


@pytest.mark.parametrize("device", DEVICES)
def test_public_projection_clamps_at_threshold(device):
    # Eigenvalues at the floor, e.g. from re-projecting a projected matrix, make
    # the smooth projector undefined. The public projection clamps instead.
    value = torch.diag(
        torch.tensor([0.1, -1.0, 2.0], dtype=torch.double, device=device)
    ).requires_grad_()
    result = psd_projection(value, 0.1)
    expected = torch.tensor([0.1, 0.1, 2.0], dtype=torch.double, device=device)
    torch.testing.assert_close(result.detach(), torch.diag(expected))
    result.sum().backward()
    assert bool(torch.isfinite(value.grad).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("differentiable", [False, True])
def test_float32_public_projection_classifies_promoted_matrix(
    device, sign, differentiable
):
    # Freeze the actual float32 input bits from an audited rotation. This
    # keeps the mixed spectrum stable across QR implementations/platforms.
    value = sign * torch.tensor(
        [
            [
                1.5666307210922241,
                -0.382068932056427,
                -0.03425976634025574,
                -0.4736507833003998,
            ],
            [
                -0.382068932056427,
                0.197843998670578,
                0.24863938987255096,
                0.35323911905288696,
            ],
            [
                -0.03425976634025574,
                0.24863938987255096,
                0.5523788928985596,
                0.5561136603355408,
            ],
            [
                -0.4736507833003998,
                0.35323911905288696,
                0.5561136603355408,
                0.6831463575363159,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    value.requires_grad_(differentiable)
    promoted = (value.detach().double() + value.detach().double().mT) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(promoted)
    floor = 1e-10
    assert bool((eigenvalues < floor).any()) and bool((eigenvalues > floor).any())
    expected = (
        (vectors * eigenvalues.clamp_min(floor).unsqueeze(-2)) @ vectors.mT
    ).float()
    gradient = value.new_tensor(
        [
            [0.7, 0.3, -0.5, 0.2],
            [0.3, -0.8, 0.4, 0.6],
            [-0.5, 0.4, 1.1, 0.9],
            [0.2, 0.6, 0.9, -0.2],
        ]
    )
    expected_vjp = _stable_loewner_vjp(promoted, gradient, floor)
    with no_host(), torch.set_grad_enabled(differentiable):
        output = psd_projection(value, floor)
        if differentiable:
            actual_vjp = torch.autograd.grad(
                (output * gradient).sum(), value, create_graph=True
            )[0]
            torch.testing.assert_close(
                actual_vjp.double(), expected_vjp, atol=2e-6, rtol=2e-6
            )
            second = torch.autograd.grad(
                (actual_vjp * gradient).sum(), value, create_graph=True
            )[0]
            third = torch.autograd.grad((second * gradient).sum(), value)[0]
            assert torch.isfinite(second).all() and torch.isfinite(third).all()
    torch.testing.assert_close(
        output, expected, atol=1e-7 if sign > 0 else 1e-12, rtol=2e-6
    )
    assert output.dtype == torch.float32 and output.device == value.device
