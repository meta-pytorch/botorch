#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.empirical_gps.utils import (
    build_unique_inputs,
    center_curves,
    compute_basis_matrix,
    compute_orthogonal_basis,
    compute_sample_covariance,
    ExperimentDataset,
    instantiate_ard,
    LinearInterpolation1D,
    project_psd,
    UniqueInputs,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import Kernel
from scipy.interpolate import interp1d
from torch import Tensor


class TestUtils(BotorchTestCase):
    def test_compute_orthogonal_basis(self) -> None:
        # Test that the orthogonal basis produces the same Gram matrix as original
        torch.manual_seed(42)  # Fix seed for reproducibility

        for dtype in [torch.float32, torch.float64]:
            # float32 SVD on CUDA (cuSOLVER) is less precise than CPU LAPACK;
            # use a looser float32 tolerance for the Gram reconstruction.
            tol = 1e-2 if dtype == torch.float32 else 1e-10

            # Test case 1: tall matrix (m >> n) - typical use case
            m, n = 1000, 50
            A = torch.randn(m, n, dtype=dtype, device=self.device)

            B = compute_orthogonal_basis(A)

            # B should have shape (r, n) where r = min(m, n) = n
            self.assertEqual(B.shape, (n, n))
            # B.T @ B should equal A.T @ A
            gram_original = A.T @ A
            gram_svd = B.T @ B
            self.assertAllClose(gram_original, gram_svd, atol=tol, rtol=tol)

            # Test case 2: wide matrix (m < n)
            m, n = 30, 100
            A = torch.randn(m, n, dtype=dtype, device=self.device)

            B = compute_orthogonal_basis(A)

            # B should have shape (r, n) where r = min(m, n) = m
            self.assertEqual(B.shape, (m, n))
            # B.T @ B should equal A.T @ A
            gram_original = A.T @ A
            gram_svd = B.T @ B
            self.assertAllClose(gram_original, gram_svd, atol=tol, rtol=tol)

            # Test case 3: square matrix
            m, n = 50, 50
            A = torch.randn(m, n, dtype=dtype, device=self.device)

            B = compute_orthogonal_basis(A)

            self.assertEqual(B.shape, (n, n))
            gram_original = A.T @ A
            gram_svd = B.T @ B
            self.assertAllClose(gram_original, gram_svd, atol=tol, rtol=tol)

    def test_compute_orthogonal_basis_eigh(self) -> None:
        # The eigh-based basis must yield the same Gram matrix as the SVD path.
        torch.manual_seed(42)
        for dtype in [torch.float32, torch.float64]:
            # float32 eigh on CUDA is less precise than CPU; loosen float32 tol.
            tol = 1e-2 if dtype == torch.float32 else 1e-8

            # Tall matrix (m >> n): the intended regime.
            m, n = 1000, 50
            A = torch.randn(m, n, dtype=dtype, device=self.device)
            B = compute_orthogonal_basis(A, method="eigh")
            self.assertEqual(B.shape, (n, n))
            self.assertAllClose(B.T @ B, A.T @ A, atol=tol, rtol=tol)

            # Matches the SVD path's Gram matrix.
            B_svd = compute_orthogonal_basis(A, method="svd")
            self.assertAllClose(B.T @ B, B_svd.T @ B_svd, atol=tol, rtol=tol)

            # Batched inputs.
            A_b = torch.randn(3, m, n, dtype=dtype, device=self.device)
            B_b = compute_orthogonal_basis(A_b, method="eigh")
            self.assertEqual(B_b.shape, (3, n, n))
            self.assertAllClose(
                B_b.transpose(-2, -1) @ B_b,
                A_b.transpose(-2, -1) @ A_b,
                atol=tol,
                rtol=tol,
            )

            # Wide matrix (m < n): eigh truncates to r = min(m, n) = m rows
            # (matching the economy SVD); dropped rows have zero eigenvalues,
            # so the Gram matrix is preserved.
            mw, nw = 30, 100
            Aw = torch.randn(mw, nw, dtype=dtype, device=self.device)
            Bw = compute_orthogonal_basis(Aw, method="eigh")
            self.assertEqual(Bw.shape, (mw, nw))
            self.assertAllClose(Bw.T @ Bw, Aw.T @ Aw, atol=tol, rtol=tol)

        # Unknown method raises.
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            compute_orthogonal_basis(A, method="bogus")

    def test_covariance_computation_helpers(self) -> None:
        """Test helper functions for covariance computation."""
        # Test center_curves with 2D tensor
        num_curves = 5
        num_progression = 10
        Y_2d = torch.randn(
            num_curves, num_progression, dtype=torch.float64, device=self.device
        )

        mean_2d, centered_2d = center_curves(Y_2d, curve_dim=-2)

        # Mean should have shape (num_progression,)
        self.assertEqual(mean_2d.shape, (num_progression,))
        # Centered should have same shape as original
        self.assertEqual(centered_2d.shape, Y_2d.shape)
        # Centered curves should have zero mean along curve_dim
        self.assertAllClose(
            centered_2d.mean(dim=-2),
            torch.zeros(num_progression, dtype=torch.float64, device=self.device),
            atol=1e-10,
        )
        # Mean + centered should recover original
        self.assertAllClose(mean_2d.unsqueeze(-2) + centered_2d, Y_2d)

        # Test center_curves with 3D tensor (multi-output)
        m = 3  # number of outputs
        Y_3d = torch.randn(
            num_curves, num_progression, m, dtype=torch.float64, device=self.device
        )

        mean_3d, centered_3d = center_curves(Y_3d, curve_dim=-3)

        # Mean should have shape (num_progression, m)
        self.assertEqual(mean_3d.shape, (num_progression, m))
        # Centered should have same shape as original
        self.assertEqual(centered_3d.shape, Y_3d.shape)
        # Centered curves should have zero mean along curve_dim
        self.assertAllClose(
            centered_3d.mean(dim=-3),
            torch.zeros(num_progression, m, dtype=torch.float64, device=self.device),
            atol=1e-10,
        )

        # Test compute_sample_covariance - full covariance
        n1, n2 = 4, 6
        U1 = torch.randn(num_curves, n1, dtype=torch.float64, device=self.device)
        U2 = torch.randn(num_curves, n2, dtype=torch.float64, device=self.device)

        K = compute_sample_covariance(U1, U2, num_curves=num_curves, diag=False)

        # Shape should be n1 x n2
        self.assertEqual(K.shape, (n1, n2))
        # Should equal U1.T @ U2 / num_curves
        expected_K = (U1.mT @ U2) / num_curves
        self.assertAllClose(K, expected_K)

        # Test compute_sample_covariance - symmetric case (U2=None)
        K_sym = compute_sample_covariance(U1, None, num_curves=num_curves, diag=False)

        self.assertEqual(K_sym.shape, (n1, n1))
        # Should be symmetric
        self.assertAllClose(K_sym, K_sym.T)

        # Test compute_sample_covariance - diagonal
        K_diag = compute_sample_covariance(U1, None, num_curves=num_curves, diag=True)

        self.assertEqual(K_diag.shape, (n1,))
        # Should equal diagonal of full covariance
        self.assertAllClose(K_diag, K_sym.diag())

        # Test compute_sample_covariance - with correction parameter
        for correction in (0, 1, 2):
            K_corrected = compute_sample_covariance(
                U1, U2, num_curves=num_curves, diag=False, correction=correction
            )
            # With correction=1, should divide by (num_curves - correction)
            expected_K_corrected = U1.mT @ U2 / (num_curves - correction)
            self.assertAllClose(K_corrected, expected_K_corrected)

        # Test compute_sample_covariance - error when num_curves <= correction
        with self.assertRaisesRegex(
            ValueError, "num_curves .* must be greater than correction"
        ):
            compute_sample_covariance(
                U1, U2, num_curves=num_curves, diag=False, correction=num_curves
            )

        # Test compute_basis_matrix - single output (m=1)
        x = torch.linspace(0, 1, 8, dtype=torch.float64, device=self.device)
        # For single-output, Y_for_interp should be m x num_curves x num_progression
        Y_for_interp = torch.randn(
            1, num_curves, num_progression, dtype=torch.float64, device=self.device
        )
        f = LinearInterpolation1D(
            torch.linspace(
                0, 1, num_progression, dtype=torch.float64, device=self.device
            ),
            Y_for_interp,
        )

        Ux = compute_basis_matrix(f=f, x=x, num_outputs=1, curve_weights=None)

        # Shape should be m x batch_shape x num_curves x n = 1 x num_curves x 8
        self.assertEqual(Ux.shape, (1, num_curves, 8))

        # Test compute_basis_matrix - with ARD weights
        curve_weights = torch.rand(num_curves, dtype=torch.float64, device=self.device)
        Ux_ard = compute_basis_matrix(
            f=f, x=x, num_outputs=1, curve_weights=curve_weights
        )

        # Should have applied weights
        self.assertEqual(Ux_ard.shape, (1, num_curves, 8))
        # Verify weights were applied correctly
        expected_Ux_ard = Ux * curve_weights.unsqueeze(-1)
        self.assertAllClose(Ux_ard, expected_Ux_ard)

        # Test instantiate_ard - creates curve_weights parameter and constraint
        class DummyModule(Kernel):
            ard: bool = False

        dummy = DummyModule()
        instantiate_ard(
            dummy,
            num_curves=num_curves,
            curve_weights=None,
            dtype=Y_2d.dtype,
            device=self.device,
        )

        self.assertTrue(dummy.ard)
        self.assertTrue(hasattr(dummy, "curve_weights"))
        self.assertEqual(dummy.curve_weights.shape, (num_curves,))
        # Initial weights should be ones
        self.assertAllClose(
            dummy.curve_weights,
            torch.ones(num_curves, dtype=torch.float64, device=self.device),
        )

        # Test instantiate_ard with 3D tensor
        dummy_3d = DummyModule()
        instantiate_ard(
            dummy_3d,
            num_curves=num_curves,
            curve_weights=None,
            dtype=Y_3d.dtype,
            device=self.device,
        )

        self.assertTrue(dummy_3d.ard)
        self.assertEqual(dummy_3d.curve_weights.shape, (num_curves,))

        # Test instantiate_ard with provided curve_weights
        dummy_custom = DummyModule()
        custom_weights = torch.nn.Parameter(
            torch.rand(num_curves, dtype=torch.float64, device=self.device)
        )
        instantiate_ard(
            dummy_custom, num_curves=num_curves, curve_weights=custom_weights
        )

        self.assertTrue(dummy_custom.ard)
        self.assertAllClose(dummy_custom.curve_weights, custom_weights)

    def test_linear_interpolation_1d(self) -> None:
        """Test LinearInterpolation1D: scipy correctness, batching, module features."""

        def wrapped_interp1d_scipy(
            x: Tensor, y: Tensor, bounds_error: bool | None = None
        ) -> Callable[[Tensor], Tensor]:
            x = x.cpu().numpy()
            y = y.cpu().numpy()

            def f(xnew: Tensor, x: np.ndarray = x, y: np.ndarray = y) -> Tensor:
                return torch.as_tensor(
                    interp1d(x, y, bounds_error=bounds_error)(xnew.cpu().numpy()),
                    device=xnew.device,
                    dtype=xnew.dtype,
                )

            return f

        # --- Correctness against scipy ---
        n = 5
        x = torch.rand(n, device=self.device)
        x = (x - x.min()) / (x.max() - x.min())
        y = torch.sin(x * 2 * math.pi)
        n2 = 2 * n
        xq = torch.linspace(0, 1, n2, device=self.device)
        yq_hat = LinearInterpolation1D(x, y)(xq)

        yq_hat_scipy = wrapped_interp1d_scipy(x, y)(xq)
        dtype = y.dtype
        tol = 1e-12 if dtype == torch.float64 else 1e-5
        self.assertAllClose(yq_hat_scipy, yq_hat, atol=tol)

        # --- Batched targets ---
        batch_size = (3, 7)
        y = torch.rand(*batch_size, n, device=self.device)
        yq_hat_scipy = wrapped_interp1d_scipy(x, y)(xq)
        yq_hat = LinearInterpolation1D(x, y)(xq)
        self.assertAllClose(yq_hat_scipy, yq_hat, atol=tol)

        # --- Batched inputs + batched targets ---
        x_batch_size = (4, 9)
        xq = torch.rand(*x_batch_size, n2, device=self.device)
        yq_hat_scipy = wrapped_interp1d_scipy(x, y)(xq)
        yq_hat = LinearInterpolation1D(x, y)(xq)
        self.assertEqual(yq_hat.shape, (*batch_size, *x_batch_size, n2))
        self.assertAllClose(yq_hat_scipy, yq_hat, atol=tol)

        # --- Batch train inputs not supported ---
        with self.assertRaisesRegex(
            UnsupportedError, "Expected x to be 1-dim, but got"
        ):
            LinearInterpolation1D(torch.rand(*batch_size, n, device=self.device), y)

        # --- Out of bounds errors (both directions) ---
        xq = torch.linspace(-1, 1, n2, device=self.device)
        with self.assertRaisesRegex(ValueError, "is below the interpolation range"):
            LinearInterpolation1D(x, y)(xq)

        xq = torch.linspace(0, 2, n2, device=self.device)
        with self.assertRaisesRegex(ValueError, "is above the interpolation range"):
            LinearInterpolation1D(x, y)(xq)

        # --- bounds_error=False fills with NaN (default fill_value) ---
        xq = torch.linspace(-1, 2, n2, device=self.device)
        yq_hat_scipy = wrapped_interp1d_scipy(x, y, bounds_error=False)(xq)
        yq_hat = LinearInterpolation1D(x, y, bounds_error=False)(xq)
        self.assertAllClose(yq_hat, yq_hat_scipy, atol=tol, equal_nan=True)

        # --- bounds_error=False with custom fill_value ---
        yq_filled = LinearInterpolation1D(x, y, bounds_error=False, fill_value=0.0)(xq)
        in_bounds = (xq >= x.min()) & (xq <= x.max())
        self.assertAllClose(
            yq_filled[..., ~in_bounds], torch.zeros_like(yq_filled[..., ~in_bounds])
        )
        self.assertAllClose(yq_filled[..., in_bounds], yq_hat[..., in_bounds], atol=tol)

        # --- Differentiability ---
        x = torch.rand(2, device=self.device)
        y = torch.randn(2, device=self.device)
        xq = x.mean().detach().requires_grad_(True)
        yq_hat = LinearInterpolation1D(x, y)(xq)
        yq_hat.backward()
        expected_grad = y.diff() / x.diff()
        self.assertAllClose(expected_grad.item(), xq.grad.item(), atol=1e-5)

        # --- Module features: buffers, state_dict, unsorted x ---
        n = 10
        x = torch.linspace(0, 1, n, device=self.device)
        y = torch.sin(x * 2 * math.pi).unsqueeze(0)  # 1 x n
        interp = LinearInterpolation1D(x, y)

        # Buffers should be registered
        buffer_names = {name for name, _ in interp.named_buffers()}
        self.assertIn("_x", buffer_names)
        self.assertIn("_y", buffer_names)

        # state_dict should contain the buffers
        sd = interp.state_dict()
        self.assertIn("_x", sd)
        self.assertIn("_y", sd)
        self.assertIn("_bounds_error", sd)
        self.assertIn("_fill_value", sd)
        self.assertAllClose(sd["_x"], x)
        self.assertAllClose(sd["_y"], y)

        # state_dict round-trip preserves predictions
        x_new = torch.tensor([0.25, 0.5, 0.75], device=self.device)
        y_original = interp(x_new)

        interp2 = LinearInterpolation1D(
            torch.zeros(n, device=self.device),
            torch.zeros(1, n, device=self.device),
        )
        interp2.load_state_dict(sd)
        self.assertAllClose(interp2(x_new), y_original)

        # Unsorted x should be sorted automatically
        perm = torch.randperm(n, device=self.device)
        interp_unsorted = LinearInterpolation1D(x[perm], y[..., perm])
        self.assertAllClose(interp_unsorted(x_new), y_original)

        # .to() moves buffers and preserves predictions (dtype transfer
        # works on CPU; device transfer exercises the same code path on GPU)
        x_knots = torch.linspace(0, 1, n)
        y_knots = torch.sin(x_knots * 2 * math.pi).unsqueeze(0)
        x_query = torch.tensor([0.25, 0.5, 0.75])

        interp_f32 = LinearInterpolation1D(x_knots, y_knots)
        y_f32 = interp_f32(x_query)

        interp_f64 = LinearInterpolation1D(x_knots, y_knots).to(dtype=torch.float64)
        for buf in interp_f64.buffers():
            if buf.is_floating_point():
                self.assertEqual(buf.dtype, torch.float64)
        y_f64 = interp_f64(x_query.to(torch.float64))
        self.assertAllClose(y_f64.float(), y_f32, atol=1e-6)

        interp_moved = LinearInterpolation1D(x_knots, y_knots).to(self.device)
        for buf in interp_moved.buffers():
            # Compare device type (self.device is e.g. 'cuda', buf.device 'cuda:0').
            self.assertEqual(buf.device.type, self.device.type)
        y_moved = interp_moved(x_query.to(self.device))
        self.assertAllClose(y_moved.cpu(), y_f32, atol=1e-6)

    def test_build_unique_inputs(self) -> None:
        # Test basic functionality with overlapping inputs
        X1 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        Y1 = torch.tensor([[1.0], [2.0], [3.0]])
        X2 = torch.tensor([[1.0, 1.0], [3.0, 3.0]])  # [1,1] overlaps with X1
        Y2 = torch.tensor([[4.0], [5.0]])

        datasets = [
            ExperimentDataset(X=X1, Y=Y1),
            ExperimentDataset(X=X2, Y=Y2),
        ]

        result = build_unique_inputs(datasets, X_forward=None)

        self.assertIsInstance(result, UniqueInputs)
        # Should have 4 unique points: [0,0], [1,1], [2,2], [3,3]
        expected_X_all = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.assertEqual(result.X_all.shape[0], 4)
        self.assertAllClose(result.X_all, expected_X_all, atol=0, rtol=0)
        self.assertEqual(len(result.experiment_indices), 2)
        self.assertEqual(len(result.experiment_indices[0]), 3)
        self.assertEqual(len(result.experiment_indices[1]), 2)
        # Forward indices should be empty
        self.assertEqual(len(result.forward_indices), 0)

        # Test with forward inputs
        X1 = torch.tensor([[0.0], [1.0], [2.0]])
        Y1 = torch.tensor([[1.0], [2.0], [3.0]])
        X_forward = torch.tensor([[1.0], [3.0]])  # [1.0] overlaps with X1

        datasets = [ExperimentDataset(X=X1, Y=Y1)]

        result = build_unique_inputs(datasets, X_forward=X_forward)

        # Should have 4 unique points: [0], [1], [2], [3]
        expected_X_all = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(result.X_all.shape[0], 4)
        self.assertAllClose(result.X_all, expected_X_all, atol=0, rtol=0)
        self.assertEqual(len(result.experiment_indices), 1)
        self.assertEqual(len(result.experiment_indices[0]), 3)
        self.assertEqual(len(result.forward_indices), 2)
        # Verify forward_indices point to the correct unique inputs
        # [1.0] -> index 1, [3.0] -> index 3
        self.assertAllClose(
            result.X_all[result.forward_indices], X_forward, atol=0, rtol=0
        )

        # Test with only forward inputs (empty datasets)
        X_forward = torch.tensor([[0.0], [1.0], [1.0], [2.0]])  # [1.0] duplicated

        result = build_unique_inputs(datasets=[], X_forward=X_forward)

        # Should have 3 unique points: [0], [1], [2]
        expected_X_all = torch.tensor([[0.0], [1.0], [2.0]])
        self.assertEqual(result.X_all.shape[0], 3)
        self.assertAllClose(result.X_all, expected_X_all, atol=0, rtol=0)
        self.assertEqual(len(result.experiment_indices), 0)
        self.assertEqual(len(result.forward_indices), 4)
        # Verify forward_indices correctly map back to original X_forward
        # [0.0] -> 0, [1.0] -> 1, [1.0] -> 1, [2.0] -> 2
        self.assertAllClose(
            result.X_all[result.forward_indices], X_forward, atol=0, rtol=0
        )

        # Test empty datasets and None X_forward raises ValueError
        with self.assertRaisesRegex(
            ValueError,
            "Cannot build unique inputs: datasets is empty and X_forward is None",
        ):
            build_unique_inputs(datasets=[], X_forward=None)

        # Test index mapping correctness
        X1 = torch.tensor([[0.0], [1.0]])
        Y1 = torch.tensor([[1.0], [2.0]])
        X2 = torch.tensor([[1.0], [2.0]])
        Y2 = torch.tensor([[3.0], [4.0]])

        datasets = [
            ExperimentDataset(X=X1, Y=Y1),
            ExperimentDataset(X=X2, Y=Y2),
        ]

        result = build_unique_inputs(datasets, X_forward=None)

        # Verify that indexing X_all with experiment_indices recovers original inputs
        for i, dataset in enumerate(datasets):
            recovered_X = result.X_all[result.experiment_indices[i]]
            self.assertAllClose(recovered_X, dataset.X, atol=0, rtol=0)

    def test_project_psd(self) -> None:
        eigval_tol = 1e-12
        # Test already PSD matrix (identity) is unchanged
        A = torch.eye(3, dtype=torch.float64)

        A_psd = project_psd(A)

        self.assertAllClose(A_psd, A)

        # Test matrix with negative eigenvalues gets projected to PSD
        eigvals = torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
        V = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        A = V @ torch.diag(eigvals) @ V.T
        min_eigval = eigval_tol
        A_psd = project_psd(A, min_eigval=min_eigval)

        # Result should be PSD (all eigenvalues >= 0)
        eigvals_result = torch.linalg.eigvalsh(A_psd)
        self.assertTrue((eigvals_result >= min_eigval - eigval_tol).all())
        # Result should be symmetric
        self.assertAllClose(A_psd, A_psd.T)

        # Test with min_eigval parameter
        eigvals = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
        V = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        A = V @ torch.diag(eigvals) @ V.T

        min_eigval = 0.1
        A_psd = project_psd(A, min_eigval=min_eigval)

        # All eigenvalues should be >= min_eigval
        eigvals_result = torch.linalg.eigvalsh(A_psd)
        self.assertTrue((eigvals_result >= min_eigval - eigval_tol).all())

        # Test symmetry preservation for random symmetric matrix
        A = torch.randn(5, 5, dtype=torch.float64)
        A = 0.5 * (A + A.T)

        A_psd = project_psd(A)

        self.assertAllClose(A_psd, A_psd.T)
