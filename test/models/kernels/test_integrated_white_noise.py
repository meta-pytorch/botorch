#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for the integrated white noise kernel.
"""

import functools

import torch
from botorch.models.kernels.integrated_white_noise import IntegratedWhiteNoiseKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import Kernel, ScaleKernel


class NonStationary1DKernelTestMixin:
    """Mixin providing common test utilities for non-stationary 1D kernels."""

    def _get_tkwargs(self, dtype: torch.dtype) -> dict[str, object]:
        return {"device": self.device, "dtype": dtype}

    def _assert_kernel_basic_attributes(self, kernel: Kernel) -> None:
        self.assertFalse(kernel.is_stationary)
        self.assertFalse(kernel.has_lengthscale)

    def _assert_covariance_symmetry(
        self, kernel: Kernel, tkwargs: dict[str, object]
    ) -> None:
        x_rand = torch.rand(10, 1, **tkwargs)
        covar_sym = kernel(x_rand, x_rand).to_dense()
        self.assertAllClose(covar_sym, covar_sym.T)

    def _assert_positive_semi_definiteness(
        self, kernel: Kernel, tkwargs: dict[str, object], tol: float = 0.0
    ) -> None:
        x_lin = torch.linspace(0.1, 1.0, 10, **tkwargs).unsqueeze(-1)
        covar_psd = kernel(x_lin, x_lin).to_dense()
        eigenvalues = torch.linalg.eigvalsh(covar_psd)
        self.assertTrue(torch.all(eigenvalues >= -tol))

    def _assert_scale_kernel_works(
        self, kernel_factory, tkwargs: dict[str, object]
    ) -> None:
        x_scale = torch.rand(10, 1, **tkwargs)
        covar_module = ScaleKernel(kernel_factory())
        covar_scaled = covar_module(x_scale)
        self.assertEqual(covar_scaled.size(), torch.Size([10, 10]))

    def _assert_batch_dimensions_work(
        self, kernel: Kernel, tkwargs: dict[str, object]
    ) -> None:
        batch_x = torch.rand(2, 5, 1, **tkwargs)
        covar_batch = kernel(batch_x, batch_x)
        self.assertEqual(covar_batch.size(), torch.Size([2, 5, 5]))

    def _assert_different_sizes_work(
        self, kernel: Kernel, tkwargs: dict[str, object], n1: int, n2: int
    ) -> None:
        x1_diff = torch.rand(n1, 1, **tkwargs)
        x2_diff = torch.rand(n2, 1, **tkwargs)
        covar_diff = kernel(x1_diff, x2_diff).to_dense()
        self.assertEqual(covar_diff.size(), torch.Size([n1, n2]))

    def _assert_multi_dimensional_input_raises(
        self, kernel: Kernel, tkwargs: dict[str, object]
    ) -> None:
        x_multi_dim = torch.rand(5, 3, **tkwargs)
        with self.assertRaises(ValueError) as context:
            kernel(x_multi_dim, x_multi_dim).to_dense()
        self.assertIn("requires 1D inputs", str(context.exception))

    def _test_common_kernel_properties(
        self,
        kernel: Kernel,
        kernel_factory,
        tkwargs: dict[str, object],
        psd_tol: float = 0.0,
    ) -> None:
        """Test common properties shared by all non-stationary 1D kernels."""
        self._assert_kernel_basic_attributes(kernel)
        self._assert_covariance_symmetry(kernel, tkwargs)
        self._assert_positive_semi_definiteness(kernel, tkwargs, tol=psd_tol)
        self._assert_scale_kernel_works(kernel_factory, tkwargs)
        self._assert_batch_dimensions_work(kernel, tkwargs)
        self._assert_different_sizes_work(kernel, tkwargs, n1=2, n2=3)
        self._assert_multi_dimensional_input_raises(kernel, tkwargs)


class TestIntegratedWhiteNoiseKernel(NonStationary1DKernelTestMixin, BotorchTestCase):
    def test_orders(self) -> None:
        """Per-order closed-form values, diagonals, and common properties."""
        for dtype in [torch.float32, torch.float64]:
            for order in (1, 2, 3):
                with self.subTest(dtype=dtype, order=order):
                    self._test_order(dtype, order)

    def _test_order(self, dtype: torch.dtype, order: int) -> None:
        tkwargs = self._get_tkwargs(dtype)
        kernel = IntegratedWhiteNoiseKernel(order=order)
        factory = functools.partial(IntegratedWhiteNoiseKernel, order=order)
        # PSD becomes ill-conditioned for higher orders; allow a small tolerance.
        psd_tol = 0.0 if order == 1 else 1e-6
        self._test_common_kernel_properties(kernel, factory, tkwargs, psd_tol=psd_tol)

        x1 = torch.tensor([[1.0], [2.0], [3.0]], **tkwargs)
        x2 = torch.tensor([[0.5], [2.0], [4.0]], **tkwargs)
        x_diag = torch.tensor([[1.0], [2.0], [3.0], [4.0]], **tkwargs)

        if order == 1:
            # k(s, t) = min(s, t)
            expected = torch.tensor(
                [[0.5, 1.0, 1.0], [0.5, 2.0, 2.0], [0.5, 2.0, 3.0]], **tkwargs
            )
            expected_diag = torch.tensor([1.0, 2.0, 3.0, 4.0], **tkwargs)
        elif order == 2:
            # k(s, t) = min^2 * (3*max - min) / 6
            expected = torch.tensor(
                [
                    [0.625 / 6, 5.0 / 6, 11.0 / 6],
                    [1.375 / 6, 16.0 / 6, 40.0 / 6],
                    [2.125 / 6, 28.0 / 6, 81.0 / 6],
                ],
                **tkwargs,
            )
            expected_diag = x_diag.squeeze(-1) ** 3 / 3
        else:  # order == 3
            # k(s, t) = min^3 * (min^2 - 5*min*max + 10*max^2) / 120
            x1 = torch.tensor([[1.0], [2.0]], **tkwargs)
            x2 = torch.tensor([[1.0], [3.0]], **tkwargs)
            expected = torch.tensor(
                [[6.0 / 120, 76.0 / 120], [31.0 / 120, 8.0 * 64.0 / 120]],
                **tkwargs,
            )
            expected_diag = x_diag.squeeze(-1) ** 5 / 20

        self.assertAllClose(kernel(x1, x2).to_dense(), expected)
        self.assertAllClose(kernel(x_diag, x_diag, diag=True), expected_diag)

        # t = 0 gives zero covariance for every order.
        x_zero = torch.tensor([[0.0], [1.0]], **tkwargs)
        covar_zero = kernel(x_zero, x_zero).to_dense()
        self.assertAllClose(covar_zero[0], torch.zeros(2, **tkwargs))
        self.assertAllClose(covar_zero[:, 0], torch.zeros(2, **tkwargs))

    def test_invalid_order(self) -> None:
        """Order must be an integer >= 1."""
        for bad_order in (0, -1, 1.5, True):
            with self.subTest(order=bad_order):
                with self.assertRaises(ValueError) as context:
                    IntegratedWhiteNoiseKernel(order=bad_order)
                # Must not collide with the 1D-input validation message.
                self.assertNotIn("requires 1D inputs", str(context.exception))

    def test_higher_order_generalization(self) -> None:
        """The general formula extends beyond the three hardcoded orders."""
        tkwargs = self._get_tkwargs(torch.float64)
        kernel = IntegratedWhiteNoiseKernel(order=4)
        x = torch.linspace(0.1, 1.0, 10, **tkwargs).unsqueeze(-1)
        covar = kernel(x, x).to_dense()
        self.assertAllClose(covar, covar.T)
        eigenvalues = torch.linalg.eigvalsh(covar)
        # Order-4 Gram matrices are ill-conditioned; allow a looser float64 tol.
        self.assertTrue(torch.all(eigenvalues >= -1e-4))
        # Diagonal closed form: k(t, t) = t^7 / (3!^2 * 7) = t^7 / 252.
        diag = kernel(x, x, diag=True)
        self.assertAllClose(diag, x.squeeze(-1) ** 7 / 252)
