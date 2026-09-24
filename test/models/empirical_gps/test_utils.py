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
from botorch.models.empirical_gps.empirical_1d_gp import (
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch.models.empirical_gps.utils import (
    build_basis_interpolant,
    build_mean_interpolant,
    build_sliding_window_curves,
    build_unique_inputs,
    center_curves,
    compute_basis_matrix,
    compute_orthogonal_basis,
    compute_sample_covariance,
    em_prior_to_basis_curves,
    ExperimentDataset,
    filter_diverged_curves,
    instantiate_ard,
    kronecker_factored_covariance,
    LinearInterpolation1D,
    project_psd,
    trace_matched_shrinkage,
    UniqueInputs,
    validate_historical_curves_3d,
    validate_no_transforms,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
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

        # A plain (non-Parameter) Tensor is accepted, matching the docstring,
        # and wrapped into a Parameter rather than raising in register_parameter.
        dummy_plain = DummyModule()
        plain_weights = torch.rand(num_curves, dtype=torch.float64, device=self.device)
        instantiate_ard(dummy_plain, num_curves=num_curves, curve_weights=plain_weights)
        self.assertIsInstance(dummy_plain.curve_weights, torch.nn.Parameter)
        self.assertAllClose(dummy_plain.curve_weights.detach(), plain_weights)

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

    def test_trace_matched_shrinkage(self) -> None:
        tkwargs = {"dtype": torch.float64, "device": self.device}
        torch.manual_seed(0)
        # A random SPD covariance and a distinct SPD target.
        Q = torch.linalg.qr(torch.randn(4, 4, **tkwargs))[0]
        cov = Q @ torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5], **tkwargs)) @ Q.T
        target = torch.eye(4, **tkwargs) + 0.1 * torch.ones(4, 4, **tkwargs)

        # alpha = 0 returns cov unchanged.
        self.assertAllClose(trace_matched_shrinkage(cov, target, 0.0), cov)

        # alpha = 1 returns the trace-matched target (tr preserved).
        blended1 = trace_matched_shrinkage(cov, target, 1.0)
        self.assertAlmostEqual(
            float(torch.trace(blended1)), float(torch.trace(cov)), places=6
        )
        tr_ratio = torch.trace(cov) / torch.trace(target)
        self.assertAllClose(blended1, tr_ratio * target)

        # 0 < alpha < 1 matches the explicit convex blend and preserves the trace.
        alpha = 0.3
        expected = (1.0 - alpha) * cov + alpha * tr_ratio * target
        blended = trace_matched_shrinkage(cov, target, alpha)
        self.assertAllClose(blended, expected)
        self.assertAlmostEqual(
            float(torch.trace(blended)), float(torch.trace(cov)), places=6
        )

        # Result preserves dtype/device and stays symmetric.
        self.assertEqual(blended.dtype, cov.dtype)
        self.assertEqual(blended.device.type, cov.device.type)
        self.assertAllClose(blended, blended.T)

        # A tensor-valued alpha (as produced by the fittable models) works too.
        alpha_t = torch.tensor(0.3, **tkwargs)
        self.assertAllClose(trace_matched_shrinkage(cov, target, alpha_t), expected)

        # A (near-)zero-trace target degenerates gracefully via the internal trace
        # clamp: the result is (1 - alpha) * cov, with no NaN/inf.
        zero_target = torch.zeros(4, 4, **tkwargs)
        degraded = trace_matched_shrinkage(cov, zero_target, 0.3)
        self.assertTrue(torch.isfinite(degraded).all())
        self.assertAllClose(degraded, 0.7 * cov)

        # alpha outside [0, 1] is rejected, as a float and as a Tensor.
        for bad_alpha in (-0.1, 1.5):
            with self.assertRaisesRegex(ValueError, r"alpha must be in \[0, 1\]"):
                trace_matched_shrinkage(cov, target, bad_alpha)
            with self.assertRaisesRegex(ValueError, r"alpha must be in \[0, 1\]"):
                trace_matched_shrinkage(cov, target, torch.tensor(bad_alpha, **tkwargs))

    def test_validate_historical_curves_3d(self) -> None:
        Y3 = torch.randn(4, 5, 1, dtype=torch.float64, device=self.device)
        # Valid 3D tensors do not raise (default and custom name).
        validate_historical_curves_3d(Y3)
        validate_historical_curves_3d(Y3, name="historical_Y")

        # Non-3D tensors raise with the given name in the message.
        with self.assertRaisesRegex(ValueError, "Expected Y_full to be 3-dim"):
            validate_historical_curves_3d(Y3.squeeze(-1))
        with self.assertRaisesRegex(ValueError, "Expected historical_Y to be 3-dim"):
            validate_historical_curves_3d(Y3.squeeze(-1), name="historical_Y")

    def test_validate_no_transforms(self) -> None:
        # No transforms: does not raise.
        validate_no_transforms(None, None, "SomeModel")

        with self.assertRaisesRegex(
            UnsupportedError, "input_transform is not yet supported for SomeModel"
        ):
            validate_no_transforms(Normalize(d=1), None, "SomeModel")

        with self.assertRaisesRegex(
            UnsupportedError, "outcome_transform is not yet supported for SomeModel"
        ):
            validate_no_transforms(None, Standardize(m=1), "SomeModel")

    def test_build_mean_interpolant(self) -> None:
        num_curves, num_progression, m = 6, 7, 2
        X_full = torch.linspace(
            0, 1, num_progression, dtype=torch.float64, device=self.device
        ).unsqueeze(-1)
        Y_full = torch.randn(
            num_curves, num_progression, m, dtype=torch.float64, device=self.device
        )

        f, num_outputs, mean_full = build_mean_interpolant(X_full, Y_full)

        self.assertEqual(num_outputs, m)
        self.assertEqual(mean_full.shape, (m, num_progression))
        self.assertAllClose(mean_full, Y_full.mean(dim=0).T)
        self.assertIsInstance(f, LinearInterpolation1D)
        # Interpolating at the knots recovers the mean curves exactly.
        self.assertAllClose(f(X_full.squeeze(-1)), mean_full)

    def test_build_basis_interpolant(self) -> None:
        torch.manual_seed(0)
        num_curves, num_progression, m = 8, 5, 3
        X_full = torch.linspace(
            0, 1, num_progression, dtype=torch.float64, device=self.device
        ).unsqueeze(-1)
        Y_full = torch.randn(
            num_curves, num_progression, m, dtype=torch.float64, device=self.device
        )
        _, Y_centered = center_curves(Y_full, curve_dim=-3)
        Y_basis = Y_centered.movedim(-1, 0)  # m x num_curves x num_progression
        knots = X_full.squeeze(-1)

        # --- vectorize_outputs=False, ARD disables SVD ---
        f, eff, used = build_basis_interpolant(
            X_full,
            Y_full,
            ard=True,
            use_svd=None,
            vectorize_outputs=False,
            method="eigh",
        )
        self.assertFalse(used)
        self.assertEqual(eff, num_curves)
        self.assertAllClose(f(knots), Y_basis)

        # --- vectorize_outputs=False, default SVD (num_curves > num_progression) ---
        f, eff, used = build_basis_interpolant(
            X_full,
            Y_full,
            ard=False,
            use_svd=None,
            vectorize_outputs=False,
            method="eigh",
        )
        self.assertTrue(used)
        self.assertEqual(eff, min(num_curves, num_progression))
        basis = f(knots)  # m x r x num_progression
        self.assertEqual(basis.shape, (m, eff, num_progression))
        # Per-output covariance is preserved by the SVD compression.
        self.assertAllClose(
            basis.transpose(-2, -1) @ basis,
            Y_basis.transpose(-2, -1) @ Y_basis,
            atol=1e-8,
        )

        # --- vectorize_outputs=True, SVD off ---
        f, eff, used = build_basis_interpolant(
            X_full, Y_full, ard=False, use_svd=False, vectorize_outputs=True
        )
        self.assertFalse(used)
        self.assertEqual(eff, num_curves)
        self.assertAllClose(f(knots), Y_basis)

        # --- vectorize_outputs=True, SVD forced on (num_curves < num_progression*m) ---
        f, eff, used = build_basis_interpolant(
            X_full, Y_full, ard=False, use_svd=True, vectorize_outputs=True
        )
        self.assertTrue(used)
        self.assertEqual(eff, min(num_curves, num_progression * m))
        basis = f(knots)  # m x eff x num_progression
        self.assertEqual(basis.shape, (m, eff, num_progression))
        # Vectorized cross-output Gram matrix is preserved by the SVD compression.
        Y_svd = basis.movedim(0, -1).reshape(eff, num_progression * m)
        Y_vec = Y_centered.reshape(num_curves, num_progression * m)
        self.assertAllClose(Y_svd.T @ Y_svd, Y_vec.T @ Y_vec, atol=1e-8)

        # --- vectorize_outputs=True, default SVD (num_curves > num_progression*m) ---
        num_curves_big = 20
        Y_big = torch.randn(
            num_curves_big,
            num_progression,
            m,
            dtype=torch.float64,
            device=self.device,
        )
        f, eff, used = build_basis_interpolant(
            X_full, Y_big, ard=False, use_svd=None, vectorize_outputs=True
        )
        self.assertTrue(used)
        self.assertEqual(eff, min(num_curves_big, num_progression * m))


class TestEMPriorToBasisCurves(BotorchTestCase):
    """Verify em_prior_to_basis_curves reproduces an EM prior on the grid."""

    def _make_prior(self, n: int, tkwargs: dict) -> tuple[Tensor, Tensor, Tensor]:
        """A full-rank RBF prior (mu, Sigma) on a 1D grid."""
        X_full = torch.linspace(0.0, 1.0, n, **tkwargs).unsqueeze(-1)
        d2 = (X_full - X_full.transpose(-1, -2)) ** 2
        Sigma = torch.exp(-0.5 * d2 / 0.15**2) + 1e-4 * torch.eye(n, **tkwargs)
        mu = torch.sin(3.0 * X_full.squeeze(-1))
        return X_full, mu, Sigma

    def test_reproduces_mu_and_sigma_exactly(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        n = 12
        X_full, mu, Sigma = self._make_prior(n, tkwargs)

        Y_full = em_prior_to_basis_curves(mu, Sigma)
        # Full-rank Sigma (r = n) requires r + 1 = n + 1 curves.
        self.assertEqual(Y_full.shape, (n + 1, n, 1))

        mean_module = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_full)
        kernel = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)

        mean_grid = mean_module(X_full).reshape(-1)
        self.assertAllClose(mean_grid, mu, atol=1e-6)

        K = kernel(X_full, X_full).to_dense()
        self.assertEqual(K.shape, (n, n))
        self.assertAllClose(K, Sigma, atol=1e-5)

    def test_truncation_matches_low_rank_reconstruction(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        n, k = 12, 3
        X_full, mu, Sigma = self._make_prior(n, tkwargs)

        Y_full = em_prior_to_basis_curves(mu, Sigma, num_modes=k)
        self.assertEqual(Y_full.shape, (k + 1, n, 1))

        kernel = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)
        K = kernel(X_full, X_full).to_dense()

        # K should equal the rank-k eigen-reconstruction of Sigma.
        evals, evecs = torch.linalg.eigh(Sigma)
        order = torch.argsort(evals, descending=True)
        evals_k = evals[order][:k]
        evecs_k = evecs[:, order][:, :k]
        Sigma_k = (evecs_k * evals_k) @ evecs_k.transpose(-1, -2)
        self.assertAllClose(K, Sigma_k, atol=1e-5)

        # The mean is exact regardless of truncation.
        mean_module = EmpiricalOneDimensionalMean(X_full=X_full, Y_full=Y_full)
        self.assertAllClose(mean_module(X_full).reshape(-1), mu, atol=1e-6)

    def test_more_modes_reduce_covariance_error(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        n = 12
        X_full, mu, Sigma = self._make_prior(n, tkwargs)

        def cov_err(num_modes: int) -> float:
            Y_full = em_prior_to_basis_curves(mu, Sigma, num_modes=num_modes)
            kernel = EmpiricalOneDimensionalKernel(X_full=X_full, Y_full=Y_full)
            K = kernel(X_full, X_full).to_dense()
            return (K - Sigma).norm().item()

        self.assertGreater(cov_err(2), cov_err(5))

    def test_degenerate_prior_raises(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        n = 5
        mu = torch.zeros(n, **tkwargs)
        Sigma = torch.zeros(n, n, **tkwargs)
        with self.assertRaises(ValueError):
            em_prior_to_basis_curves(mu, Sigma)

    def test_correction_round_trips_and_validates(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        n = 10
        X_full, mu, Sigma = self._make_prior(n, tkwargs)
        # A non-default Bessel correction must round-trip when the kernel uses
        # the same correction (guards the sqrt(num_curves - correction) scale).
        Y_full = em_prior_to_basis_curves(mu, Sigma, correction=1)
        kernel = EmpiricalOneDimensionalKernel(
            X_full=X_full, Y_full=Y_full, correction=1
        )
        K = kernel(X_full, X_full).to_dense()
        self.assertAllClose(K, Sigma, atol=1e-5)
        # correction >= number of synthesized curves is rejected (would otherwise
        # take the square root of a negative number).
        with self.assertRaisesRegex(ValueError, "correction"):
            em_prior_to_basis_curves(mu, Sigma, correction=n + 5)

    def test_input_validation(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        _, mu, Sigma = self._make_prior(6, tkwargs)
        # mu/Sigma dimension mismatch is rejected (shape-assert guard).
        with self.assertRaisesRegex(ValueError, "matching n"):
            em_prior_to_basis_curves(torch.zeros(4, **tkwargs), torch.eye(3, **tkwargs))
        # A non-1D mu is rejected by the same guard.
        with self.assertRaisesRegex(ValueError, "Expected mu"):
            em_prior_to_basis_curves(mu.unsqueeze(-1), Sigma)
        # num_modes < 1 raises a clear error (not the degenerate-prior message).
        with self.assertRaisesRegex(ValueError, "num_modes"):
            em_prior_to_basis_curves(mu, Sigma, num_modes=0)


class TestBuildSlidingWindowCurves(BotorchTestCase):
    def test_shapes_and_values_single_output(self) -> None:
        series = torch.arange(10, dtype=torch.double)
        curves = build_sliding_window_curves(series, window_size=4, stride=2)
        # (10 - 4) // 2 + 1 == 4 windows
        self.assertEqual(curves.shape, torch.Size([4, 4, 1]))
        expected = torch.tensor(
            [[0.0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]],
            dtype=torch.double,
        ).unsqueeze(-1)
        self.assertAllClose(curves, expected)

    def test_default_stride_is_maximally_overlapping(self) -> None:
        series = torch.arange(6, dtype=torch.double)
        curves = build_sliding_window_curves(series, window_size=3)
        self.assertEqual(curves.shape, torch.Size([4, 3, 1]))

    def test_multi_output_preserves_output_dim(self) -> None:
        # Distinct per-output values so a transposed axis would be caught.
        series = torch.stack(
            [torch.arange(8, dtype=torch.double), 100 + torch.arange(8).double()],
            dim=-1,
        )
        curves = build_sliding_window_curves(series, window_size=3, stride=3)
        self.assertEqual(curves.shape, torch.Size([2, 3, 2]))
        self.assertAllClose(curves[0, :, 0], torch.tensor([0.0, 1.0, 2.0]).double())
        self.assertAllClose(curves[0, :, 1], torch.tensor([100.0, 101, 102]).double())

    def test_window_equal_to_series_length(self) -> None:
        series = torch.arange(5, dtype=torch.double)
        curves = build_sliding_window_curves(series, window_size=5)
        self.assertEqual(curves.shape, torch.Size([1, 5, 1]))

    def test_output_is_contiguous(self) -> None:
        # unfold returns a strided view; the helper must materialize it.
        curves = build_sliding_window_curves(torch.arange(9).double(), window_size=3)
        self.assertTrue(curves.is_contiguous())

    def test_dtype_and_device_preserved(self) -> None:
        series = torch.arange(6, dtype=torch.float32, device=self.device)
        curves = build_sliding_window_curves(series, window_size=2)
        self.assertEqual(curves.dtype, torch.float32)
        self.assertEqual(curves.device.type, self.device.type)

    def test_feeds_empirical_kernel(self) -> None:
        # The whole point of the helper: the result is a valid historical_Y.
        curves = build_sliding_window_curves(torch.randn(60).double(), window_size=10)
        grid = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
        kernel = EmpiricalOneDimensionalKernel(X_full=grid, Y_full=curves)
        mean = EmpiricalOneDimensionalMean(X_full=grid, Y_full=curves)
        self.assertEqual(kernel(grid, grid).to_dense().shape, torch.Size([10, 10]))
        self.assertEqual(mean(grid).shape, torch.Size([10]))

    def test_invalid_inputs(self) -> None:
        series = torch.arange(6, dtype=torch.double)
        with self.assertRaisesRegex(ValueError, "shape"):
            build_sliding_window_curves(series.reshape(1, 2, 3), window_size=2)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_sliding_window_curves(series, window_size=0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_sliding_window_curves(series, window_size=2, stride=0)
        with self.assertRaisesRegex(ValueError, "exceeds the series length"):
            build_sliding_window_curves(series, window_size=7)


class TestFilterDivergedCurves(BotorchTestCase):
    def _corpus(self) -> torch.Tensor:
        # 4 curves x 3 progression points x 1 output; curve 2 is fine, the
        # others exercise each rejection reason.
        return torch.tensor(
            [
                [1.0, 2.0, 3.0],  # ok
                [1.0, float("nan"), 3.0],  # nan
                [4.0, 5.0, 6.0],  # ok
                [1.0, 1e13, 3.0],  # blown up
            ],
            dtype=torch.double,
        ).unsqueeze(-1)

    def test_drops_nonfinite_and_blown_up(self) -> None:
        Y, keep = filter_diverged_curves(self._corpus())
        self.assertEqual(Y.shape, torch.Size([2, 3, 1]))
        self.assertEqual(keep.tolist(), [True, False, True, False])
        self.assertAllClose(Y[:, :, 0], torch.tensor([[1.0, 2, 3], [4, 5, 6]]).double())

    def test_infinity_is_dropped(self) -> None:
        Y = torch.tensor([[1.0, 2.0], [1.0, float("inf")]], dtype=torch.double)
        _, keep = filter_diverged_curves(Y)
        self.assertEqual(keep.tolist(), [True, False])

    def test_none_threshold_keeps_large_but_finite(self) -> None:
        Y, keep = filter_diverged_curves(self._corpus(), max_abs_value=None)
        # the 1e13 curve is finite, so only the NaN curve goes
        self.assertEqual(keep.tolist(), [True, False, True, True])
        self.assertEqual(Y.shape, torch.Size([3, 3, 1]))

    def test_threshold_is_respected(self) -> None:
        Y = torch.tensor([[1.0, 2.0], [1.0, 50.0]], dtype=torch.double)
        self.assertEqual(
            filter_diverged_curves(Y, max_abs_value=10.0)[1].tolist(), [True, False]
        )
        self.assertEqual(
            filter_diverged_curves(Y, max_abs_value=100.0)[1].tolist(), [True, True]
        )

    def test_negative_divergence(self) -> None:
        # the bound is on magnitude, so a large negative value is divergence too
        Y = torch.tensor([[1.0, 2.0], [1.0, -1e9]], dtype=torch.double)
        self.assertEqual(filter_diverged_curves(Y)[1].tolist(), [True, False])

    def test_2d_input_supported(self) -> None:
        Y, keep = filter_diverged_curves(torch.ones(3, 5, dtype=torch.double))
        self.assertEqual(Y.shape, torch.Size([3, 5]))
        self.assertTrue(bool(keep.all()))

    def test_multi_output_drops_whole_curve(self) -> None:
        # a curve is dropped if ANY output diverges: the outputs are modelled
        # jointly, so a half-valid curve is not usable.
        Y = torch.ones(2, 3, 2, dtype=torch.double)
        Y[1, 0, 1] = 1e13
        _, keep = filter_diverged_curves(Y)
        self.assertEqual(keep.tolist(), [True, False])

    def test_mask_aligns_parallel_arrays(self) -> None:
        # the documented use: drop the matching rows of a parameter matrix
        params = torch.arange(4, dtype=torch.double).unsqueeze(-1)
        _, keep = filter_diverged_curves(self._corpus())
        self.assertAllClose(params[keep].squeeze(-1), torch.tensor([0.0, 2.0]).double())

    def test_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected Y of shape"):
            filter_diverged_curves(torch.ones(5, dtype=torch.double))
        with self.assertRaisesRegex(ValueError, "All 2 curves were dropped"):
            filter_diverged_curves(torch.full((2, 3), 1e13, dtype=torch.double))


class TestKroneckerFactoredCovariance(BotorchTestCase):
    """Separable covariance estimation via the matrix-normal flip-flop MLE.

    The point is rank: the unconstrained sample covariance over an ``n*m`` index
    set has rank at most ``num_curves``, while the Kronecker form needs only
    ``n(n+1)/2 + m(m+1)/2`` parameters, so it stays well-conditioned with far
    fewer curves. The benefit grows with ``m``.
    """

    def _separable(self, n=6, m=3, num_curves=3000, seed=0):
        torch.manual_seed(seed)
        A = torch.randn(n, n, dtype=torch.double)
        sigma_p = A @ A.T / n
        B = torch.randn(m, m, dtype=torch.double)
        sigma_o = B @ B.T / m
        sigma_o = sigma_o * (m / sigma_o.diagonal().sum())
        full = torch.kron(sigma_p, sigma_o)
        chol = torch.linalg.cholesky(full + 1e-9 * torch.eye(n * m, dtype=torch.double))
        draws = (chol @ torch.randn(n * m, num_curves, dtype=torch.double)).T
        return draws.reshape(num_curves, n, m), sigma_p, sigma_o

    def test_recovers_a_separable_covariance(self) -> None:
        Y, sigma_p, sigma_o = self._separable()
        p_hat, o_hat = kronecker_factored_covariance(Y, num_iters=8)
        truth = torch.kron(sigma_p, sigma_o)
        estimate = torch.kron(p_hat, o_hat)
        self.assertLess(float((estimate - truth).norm() / truth.norm()), 0.1)

    def test_output_factor_is_trace_normalized(self) -> None:
        # The factorization is only identified up to scale; pinning the trace
        # keeps the iteration from drifting.
        Y, _, _ = self._separable()
        _, o_hat = kronecker_factored_covariance(Y, num_iters=5)
        self.assertAlmostEqual(float(o_hat.diagonal().sum()), o_hat.shape[-1], places=6)

    def test_beats_the_unconstrained_estimate_on_rank(self) -> None:
        # The entire motivation. 12 curves over a 6x3 = 18-dim index set: the
        # unconstrained sample covariance can reach rank 12 at best.
        torch.manual_seed(0)
        n, m, num_curves = 6, 3, 12
        Y = torch.randn(num_curves, n, m, dtype=torch.double)
        flat = (Y - Y.mean(0, keepdim=True)).reshape(num_curves, -1)
        unconstrained = flat.T @ flat / num_curves
        p_hat, o_hat = kronecker_factored_covariance(Y, num_iters=5)
        factored = torch.kron(p_hat, o_hat)

        def _rank(K: torch.Tensor) -> int:
            eig = torch.linalg.eigvalsh(0.5 * (K + K.transpose(-1, -2)))
            return int((eig > eig.max() * K.shape[-1] * torch.finfo(K.dtype).eps).sum())

        self.assertGreater(_rank(factored), _rank(unconstrained))

    def test_both_factors_are_symmetric_psd(self) -> None:
        Y, _, _ = self._separable(num_curves=500)
        for factor in kronecker_factored_covariance(Y, num_iters=5):
            self.assertAllClose(factor, factor.transpose(-1, -2))
            eig = torch.linalg.eigvalsh(factor)
            self.assertGreater(float(eig.min()), -1e-8)

    def test_shapes(self) -> None:
        Y = torch.randn(20, 7, 4, dtype=torch.double)
        p_hat, o_hat = kronecker_factored_covariance(Y, num_iters=3)
        self.assertEqual(p_hat.shape, torch.Size([7, 7]))
        self.assertEqual(o_hat.shape, torch.Size([4, 4]))

    def test_log_likelihood_is_non_decreasing(self) -> None:
        # The guarantee flip-flop actually provides. Each half-step is the MLE of
        # its factor given the other, so the Gaussian log-likelihood cannot
        # decrease. Note this is NOT the same as monotone Frobenius distance to
        # the true parameter -- at finite sample size the MLE is not the
        # Frobenius-closest estimate, and asserting that instead fails here.
        Y, _, _ = self._separable(num_curves=800)
        num_curves = Y.shape[0]
        centered = (Y - Y.mean(dim=0, keepdim=True)).reshape(num_curves, -1)
        sample = centered.T @ centered / num_curves

        log_likelihoods = []
        for iters in (1, 2, 3, 5, 10):
            p_hat, o_hat = kronecker_factored_covariance(Y, num_iters=iters)
            estimate = torch.kron(p_hat, o_hat)
            _, logdet = torch.linalg.slogdet(estimate)
            log_likelihoods.append(
                float(
                    -0.5 * (logdet + torch.trace(torch.linalg.solve(estimate, sample)))
                )
            )
        for earlier, later in zip(log_likelihoods, log_likelihoods[1:]):
            self.assertGreaterEqual(later, earlier - 1e-9)

    def test_converges_and_stays_put(self) -> None:
        Y, _, _ = self._separable(num_curves=800)
        p5, o5 = kronecker_factored_covariance(Y, num_iters=5)
        p50, o50 = kronecker_factored_covariance(Y, num_iters=50)
        self.assertAllClose(torch.kron(p5, o5), torch.kron(p50, o50), atol=1e-6)

    def test_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be"):
            kronecker_factored_covariance(torch.randn(10, 5, dtype=torch.double))
        with self.assertRaisesRegex(ValueError, "at least 2 curves"):
            kronecker_factored_covariance(torch.randn(1, 5, 2, dtype=torch.double))
