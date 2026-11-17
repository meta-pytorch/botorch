#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.transforms.outcome import Bilog
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal


class TestBilogDerivatives(BotorchTestCase):
    def _cases(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return itertools.product(devices, (torch.float, torch.double), (False, True))

    def test_origin_first_derivative(self):
        for device, dtype, inverse in self._cases():
            for shape in ((1, 1), (2, 3, 2)):
                with self.subTest(
                    device=device, dtype=dtype, inverse=inverse, shape=shape
                ):
                    Y = torch.zeros(
                        shape, dtype=dtype, device=device, requires_grad=True
                    )
                    transform = Bilog()
                    fn = transform.untransform if inverse else transform
                    actual = fn(Y)[0]
                    gradient = torch.autograd.grad(actual.sum(), Y)[0]
                    self.assertAllClose(actual, torch.zeros_like(Y), atol=0, rtol=0)
                    self.assertAllClose(gradient, torch.ones_like(Y), atol=0, rtol=0)
                    step = 1e-5 if dtype == torch.float else 1e-7
                    finite = (fn(Y.detach() + step)[0] - fn(Y.detach() - step)[0]) / (
                        2 * step
                    )
                    self.assertAllClose(gradient, finite, atol=0, rtol=step)

    def test_values_and_closed_form_first_derivatives(self):
        for device, dtype, inverse in self._cases():
            with self.subTest(device=device, dtype=dtype, inverse=inverse):
                large = 60.0 if inverse else 1000.0
                Y = torch.tensor(
                    [-large, -2.0, -1e-12, 0.0, 1e-12, 2.0, large],
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                ).unsqueeze(-1)
                transform = Bilog()
                actual = (transform.untransform if inverse else transform)(Y)[0]
                magnitude = Y.detach().abs()
                expected = Y.detach().sign() * (
                    magnitude.expm1() if inverse else magnitude.log1p()
                )
                gradient = torch.autograd.grad(actual.sum(), Y)[0]
                expected_gradient = (
                    magnitude.exp() if inverse else (1 + magnitude).reciprocal()
                )
                tolerance = 8 * torch.finfo(dtype).eps
                self.assertAllClose(actual, expected, atol=0, rtol=tolerance)
                self.assertAllClose(gradient, expected_gradient, atol=0, rtol=tolerance)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.isfinite(gradient).all())

    def test_partial_outputs_keep_origin_derivatives(self):
        for device, dtype, inverse in self._cases():
            with self.subTest(device=device, dtype=dtype, inverse=inverse):
                Y = torch.zeros(3, 2, dtype=dtype, device=device, requires_grad=True)
                transform = Bilog(outputs=[1])
                actual = (transform.untransform if inverse else transform)(Y)[0]
                gradient = torch.autograd.grad(actual.sum(), Y)[0]
                self.assertAllClose(gradient, torch.ones_like(Y), atol=0, rtol=0)

    def test_positive_branch_higher_derivative_convention_at_zero(self):
        # Bilog and its inverse are C1, not C2, at zero. Repeated autograd
        # follows the explicitly selected positive branch at that one point.
        for device, dtype, inverse in self._cases():
            with self.subTest(device=device, dtype=dtype, inverse=inverse):
                Y = torch.zeros(1, 1, dtype=dtype, device=device, requires_grad=True)
                transform = Bilog()
                actual = (transform.untransform if inverse else transform)(Y)[0]
                first = torch.autograd.grad(actual.sum(), Y, create_graph=True)[0]
                second = torch.autograd.grad(first.sum(), Y, create_graph=True)[0]
                third = torch.autograd.grad(second.sum(), Y)[0]
                self.assertAllClose(second, torch.full_like(Y, 1 if inverse else -1))
                self.assertAllClose(third, torch.full_like(Y, 1 if inverse else 2))

    def test_nonzero_first_second_third_derivatives(self):
        for device, dtype, inverse in self._cases():
            with self.subTest(device=device, dtype=dtype, inverse=inverse):
                Y = torch.tensor(
                    [-2.0, -0.4, 0.3, 2.0],
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                ).unsqueeze(-1)
                transform = Bilog()
                actual = (transform.untransform if inverse else transform)(Y)[0]
                first = torch.autograd.grad(actual.sum(), Y, create_graph=True)[0]
                second = torch.autograd.grad(first.sum(), Y, create_graph=True)[0]
                third = torch.autograd.grad(second.sum(), Y)[0]
                magnitude = Y.detach().abs()
                expected_first = (
                    magnitude.exp() if inverse else (1 + magnitude).reciprocal()
                )
                expected_second = Y.detach().sign() * (
                    magnitude.exp() if inverse else -(1 + magnitude).pow(-2)
                )
                expected_third = (
                    magnitude.exp() if inverse else 2 * (1 + magnitude).pow(-3)
                )
                for observed, expected in (
                    (first, expected_first),
                    (second, expected_second),
                    (third, expected_third),
                ):
                    self.assertAllClose(
                        observed, expected, atol=0, rtol=8 * torch.finfo(dtype).eps
                    )

    def test_posterior_origin_samples_keep_mean_derivative(self):
        for device, dtype in itertools.product(
            [torch.device("cpu")]
            + ([torch.device("cuda")] if torch.cuda.is_available() else []),
            (torch.float, torch.double),
        ):
            for n in (1, 3):
                with self.subTest(device=device, dtype=dtype, n=n):
                    mean = torch.zeros(
                        n, device=device, dtype=dtype, requires_grad=True
                    )
                    posterior = GPyTorchPosterior(
                        MultivariateNormal(
                            mean, torch.eye(n, device=device, dtype=dtype)
                        )
                    )
                    transformed = Bilog().untransform_posterior(posterior)
                    samples = transformed.rsample_from_base_samples(
                        torch.Size([2]), torch.zeros(2, n, device=device, dtype=dtype)
                    )
                    self.assertAllClose(
                        samples, torch.zeros_like(samples), atol=0, rtol=0
                    )
                    gradient = torch.autograd.grad(samples.sum(), mean)[0]
                    self.assertAllClose(
                        gradient, torch.full_like(mean, 2), atol=0, rtol=0
                    )
