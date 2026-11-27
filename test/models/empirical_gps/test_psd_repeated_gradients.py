#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.empirical_gps.utils import project_psd
from botorch.utils.testing import BotorchTestCase


class TestPSDRepeatedGradients(BotorchTestCase):
    def test_repeated_spectra_values_and_derivatives(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        spectra = ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 2.0, 2.0])
        for device, dtype, spectrum in itertools.product(
            devices, (torch.float, torch.double), spectra
        ):
            with self.subTest(device=device, dtype=dtype, spectrum=spectrum):
                values = torch.tensor(spectrum, device=device, dtype=dtype)
                matrix = torch.diag(values).requires_grad_()
                result = project_psd(matrix, 0.1)
                self.assertAllClose(result, torch.diag(values.clamp_min(0.1)))
                weights = torch.arange(
                    1, matrix.numel() + 1, device=device, dtype=dtype
                ).reshape_as(matrix)
                first = torch.autograd.grad(
                    (result.square() * weights).sum(), matrix, create_graph=True
                )[0]
                second = torch.autograd.grad(
                    first.square().sum(), matrix, create_graph=True
                )[0]
                third = torch.autograd.grad(second.square().sum(), matrix)[0]
                self.assertTrue(
                    all(torch.isfinite(value).all() for value in (first, second, third))
                )

    def test_rotated_mixed_spectrum_gradcheck_and_gradgradcheck(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                generator = torch.Generator(device=device).manual_seed(41)
                basis, _ = torch.linalg.qr(
                    torch.randn(
                        4, 4, device=device, dtype=torch.double, generator=generator
                    )
                )
                matrix = (
                    basis
                    @ torch.diag(basis.new_tensor([-1.0, -1.0, 2.0, 2.0]))
                    @ basis.mT
                ).requires_grad_()

                def function(value):
                    return project_psd(value, 0.1)

                self.assertTrue(
                    torch.autograd.gradcheck(function, (matrix,), atol=1e-5, rtol=2e-4)
                )
                self.assertTrue(
                    torch.autograd.gradgradcheck(
                        function, (matrix,), atol=2e-5, rtol=3e-4
                    )
                )
