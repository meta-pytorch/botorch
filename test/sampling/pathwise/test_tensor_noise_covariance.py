#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product
from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise import gaussian_update
from botorch.sampling.pathwise.update_strategies import _gaussian_update_exact
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from linear_operator.operators import DenseLinearOperator
from torch import Tensor


class TestTensorNoiseCovariance(BotorchTestCase):
    @property
    def devices(self) -> list[str]:
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    @staticmethod
    def _rbf(x: Tensor, z: Tensor) -> Tensor:
        return (
            -0.5 * ((x.unsqueeze(-2) - z.unsqueeze(-3)) / 0.4).square().sum(-1)
        ).exp()

    def test_noise_override_values_and_derivatives(self) -> None:
        for device, dtype, provided_root in product(
            self.devices, (torch.float32, torch.float64), (False, True)
        ):
            with self.subTest(device=device, dtype=dtype, provided_root=provided_root):
                kernel = RBFKernel(batch_shape=torch.Size([2])).to(
                    device=device, dtype=dtype
                )
                kernel.lengthscale = torch.full_like(kernel.raw_lengthscale, 0.4)
                points = torch.tensor(
                    [[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]],
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                query = points.detach()[[0, 2]].clone().add_(0.03).requires_grad_()
                target = points.new_tensor(
                    [[0.3, -0.2, 0.6], [0.1, 0.4, -0.3]], requires_grad=True
                )
                samples = (
                    torch.linspace(-0.4, 0.7, 36, device=device, dtype=dtype)
                    .reshape(2, 3, 2, 3)
                    .requires_grad_()
                )
                noise = torch.linspace(
                    -0.5, 0.6, 36, device=device, dtype=dtype
                ).reshape_as(samples)
                covariance = (
                    0.08 * torch.eye(3, device=device, dtype=dtype)
                    + points.new_full((2, 3, 3), 0.02)
                ).requires_grad_()
                raw_root = (
                    torch.linalg.cholesky(self._rbf(points, points) + covariance)
                    .detach()
                    .requires_grad_()
                )
                scale_tril = raw_root.tril() if provided_root else None
                with patch.object(torch, "randn_like", return_value=noise):
                    actual = _gaussian_update_exact(
                        kernel,
                        points,
                        target,
                        samples,
                        noise_covariance=covariance,
                        scale_tril=scale_tril,
                    )(query)

                # Independent Gaussian conditioning on the SPD covariance and
                # lower-factor domains, without the pathwise implementation.
                symmetric_noise = (covariance + covariance.mT) / 2
                errors = (
                    target
                    - samples
                    - (
                        torch.linalg.cholesky(symmetric_noise) @ noise.unsqueeze(-1)
                    ).squeeze(-1)
                )
                if provided_root:
                    intermediate = torch.linalg.solve_triangular(
                        scale_tril, errors.unsqueeze(-1), upper=False
                    )
                    weight = torch.linalg.solve_triangular(
                        scale_tril.mT, intermediate, upper=True
                    )
                else:
                    weight = torch.linalg.solve(
                        self._rbf(points, points) + symmetric_noise,
                        errors.unsqueeze(-1),
                    )
                expected = (self._rbf(query, points) @ weight).squeeze(-1)
                tolerance = 2e-4 if dtype == torch.float32 else 2e-9
                self.assertAllClose(actual, expected, atol=tolerance, rtol=tolerance)
                leaves = (points, query, target, samples, covariance) + (
                    (raw_root,) if provided_root else ()
                )
                derivatives = [
                    torch.autograd.grad(value.square().sum(), leaves, create_graph=True)
                    for value in (actual, expected)
                ]
                for first, reference in zip(*derivatives, strict=True):
                    self.assertAllClose(
                        first, reference, atol=tolerance, rtol=tolerance
                    )
                if dtype == torch.float64:
                    # Kernel cdist double backward is not available in every
                    # supported Torch release. Check the complete noise/target/
                    # sample/factor algebra without requesting that derivative.
                    selected = tuple(range(2, len(leaves)))
                    hessians = [
                        torch.autograd.grad(
                            sum(derivative[index].sum() for index in selected),
                            tuple(leaves[index] for index in selected),
                        )
                        for derivative in derivatives
                    ]
                    for second, reference in zip(*hessians, strict=True):
                        self.assertAllClose(second, reference, atol=2e-8, rtol=2e-8)

    def test_tensor_and_lazy_noise_covariances_agree(self) -> None:
        for device, dtype in product(self.devices, (torch.float32, torch.float64)):
            with self.subTest(device=device, dtype=dtype):
                points = torch.linspace(
                    0.1, 0.9, 6, device=device, dtype=dtype
                ).reshape(3, 2)
                kernel = RBFKernel().to(points)
                covariance = 0.1 * torch.eye(
                    3, device=device, dtype=dtype
                ) + points.new_full((3, 3), 0.02)
                samples = points.new_tensor([[0.1, -0.2, 0.3], [0.4, 0.1, -0.5]])
                noise = torch.linspace(
                    -0.4, 0.7, 6, device=device, dtype=dtype
                ).reshape_as(samples)
                weights = []
                for supplied in (covariance, DenseLinearOperator(covariance)):
                    with patch.object(torch, "randn_like", return_value=noise):
                        weights.append(
                            _gaussian_update_exact(
                                kernel,
                                points,
                                points[:, 0],
                                samples,
                                noise_covariance=supplied,
                            ).weight
                        )
                self.assertAllClose(*weights, atol=2e-5, rtol=2e-5)

    def test_public_exact_gp_accepts_tensor_noise_covariance(self) -> None:
        for device, dtype, provided_root in product(
            self.devices, (torch.float32, torch.float64), (False, True)
        ):
            with self.subTest(device=device, dtype=dtype, provided_root=provided_root):
                x = torch.linspace(0.1, 0.9, 6, device=device, dtype=dtype).reshape(
                    3, 2
                )
                model = SingleTaskGP(
                    x, x[:, :1].sin(), covar_module=RBFKernel(), outcome_transform=None
                ).to(x)
                samples = x.new_tensor([[0.1, -0.2, 0.3], [0.4, 0.1, -0.5]])
                covariance = 0.1 * torch.eye(
                    3, device=device, dtype=dtype
                ) + x.new_full((3, 3), 0.02)
                noise = x.new_tensor([[0.3, -0.7, 0.1], [-0.2, 0.4, 0.5]])
                root = torch.linalg.cholesky(
                    model.covar_module(x).to_dense() + covariance
                )
                with patch.object(torch, "randn_like", return_value=noise):
                    path = gaussian_update(
                        model,
                        samples,
                        noise_covariance=covariance,
                        scale_tril=root if provided_root else None,
                    )
                errors = (
                    model.train_targets
                    - samples
                    - (torch.linalg.cholesky(covariance) @ noise.unsqueeze(-1)).squeeze(
                        -1
                    )
                )
                expected = torch.linalg.solve(
                    model.covar_module(x).to_dense() + covariance, errors.unsqueeze(-1)
                ).squeeze(-1)
                self.assertAllClose(path.weight, expected, atol=2e-5, rtol=2e-5)
