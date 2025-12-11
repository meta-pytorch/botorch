#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.vbll_helper import (
    DenseNormal,
    DenseNormalPrec,
    get_parameterization,
    LowRankNormal,
    Normal,
    tp,
)


def _build_loc_and_cholesky(shape, **tkwargs):
    """Small helper for loc and chol gen."""
    loc = torch.randn(shape, **tkwargs)
    batch_shape = shape[:-1]
    dim = shape[-1]
    L = torch.randn(*batch_shape, dim, dim, **tkwargs)
    L = torch.tril(L)
    diag_idx = torch.arange(dim, device=loc.device)
    L[..., diag_idx, diag_idx] = (
        torch.rand(*batch_shape, dim, **tkwargs) + 0.1
    )  # Ensure positive diagonals
    return loc, L


class VBLLHelperTestCase(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.tkwargs = {"dtype": torch.float64, "device": self.device}


class TestNormal(VBLLHelperTestCase):
    def test_initialization(self):
        # Test with scalar inputs
        loc = torch.tensor(0.0, **self.tkwargs)
        scale = torch.tensor(1.0, **self.tkwargs)
        dist = Normal(loc, scale)
        self.assertEqual(dist.loc, loc)
        self.assertEqual(dist.scale, scale)

        # Test with vector inputs
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        scale = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = Normal(loc, scale)
        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.scale == scale))

        # Test with batch dimensions
        loc = torch.randn(2, 3, **self.tkwargs)
        scale = torch.rand(2, 3, **self.tkwargs) + 0.1  # Ensure positive
        dist = Normal(loc, scale)
        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.scale == scale))

    def test_properties(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        scale = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = Normal(loc, scale)

        # Test mean property
        self.assertTrue(torch.all(dist.mean == loc))

        # Test var property
        expected_var = scale**2
        self.assertTrue(torch.all(dist.var == expected_var))

        # Test covariance_diagonal property
        self.assertTrue(torch.all(dist.covariance_diagonal == expected_var))

        # Test covariance property
        expected_cov = torch.diag_embed(expected_var)
        self.assertTrue(torch.all(dist.covariance == expected_cov))

        # Test precision property
        expected_prec = torch.diag_embed(1.0 / expected_var)
        self.assertTrue(torch.all(dist.precision == expected_prec))

        # Test logdet_covariance property
        expected_logdet = 2 * torch.log(scale).sum(-1)
        self.assertAllClose(dist.logdet_covariance, expected_logdet)

        # Test logdet_precision property
        expected_logdet_prec = -2 * torch.log(scale).sum(-1)
        self.assertAllClose(dist.logdet_precision, expected_logdet_prec)

        # Test trace_covariance property
        expected_trace = expected_var.sum(-1)
        self.assertAllClose(dist.trace_covariance, expected_trace)

        # Test trace_precision property
        expected_trace_prec = (1.0 / expected_var).sum(-1)
        self.assertAllClose(dist.trace_precision, expected_trace_prec)

        # Test chol_covariance property
        expected_chol = torch.diag_embed(scale)
        self.assertTrue(torch.all(dist.chol_covariance == expected_chol))

    def test_inner_products(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        scale = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = Normal(loc, scale)

        # Test vector for inner products
        b = torch.tensor([[1.0], [2.0], [3.0]], **self.tkwargs)

        # Test covariance_weighted_inner_prod
        expected_cov_prod = (dist.var.unsqueeze(-1) * (b**2)).sum(-2)
        actual_cov_prod = dist.covariance_weighted_inner_prod(b)
        self.assertAllClose(actual_cov_prod, expected_cov_prod.squeeze(-1))

        # Test precision_weighted_inner_prod
        expected_prec_prod = ((b**2) / dist.var.unsqueeze(-1)).sum(-2)
        actual_prec_prod = dist.precision_weighted_inner_prod(b)
        self.assertAllClose(actual_prec_prod, expected_prec_prod.squeeze(-1))

    def test_addition(self):
        # Test addition with another Normal
        loc1 = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        scale1 = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist1 = Normal(loc1, scale1)

        loc2 = torch.tensor([3.0, 2.0, 1.0], **self.tkwargs)
        scale2 = torch.tensor([3.0, 2.0, 1.0], **self.tkwargs)
        dist2 = Normal(loc2, scale2)

        result = dist1 + dist2

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc1 + loc2
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale (variance addition)
        expected_var = dist1.var + dist2.var
        expected_scale = torch.sqrt(torch.clip(expected_var, min=1e-12))
        self.assertAllClose(result.scale, expected_scale)

        # Test addition with a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        result = dist1 + tensor

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc1 + tensor
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale (should be unchanged)
        self.assertTrue(torch.all(result.scale == scale1))

        # Test invalid addition
        with self.assertRaises(NotImplementedError):
            _ = dist1 + 5.0

    def test_matmul(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        scale = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = Normal(loc, scale)

        # Test matrix multiplication with a vector
        vec = torch.tensor([[2.0], [3.0], [4.0]], **self.tkwargs)
        result = dist @ vec

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc @ vec
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale
        new_cov = dist.covariance_weighted_inner_prod(
            vec.unsqueeze(-3), reduce_dim=False
        )
        expected_scale = torch.sqrt(torch.clip(new_cov, min=1e-12))
        self.assertAllClose(result.scale, expected_scale)

    def test_squeeze(self):
        # Create a distribution with an extra dimension
        loc = torch.tensor([[0.0, 1.0, 2.0]], **self.tkwargs)
        scale = torch.tensor([[1.0, 2.0, 3.0]], **self.tkwargs)
        dist = Normal(loc, scale)

        # Squeeze the first dimension
        result = dist.squeeze(0)

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check dimensions
        self.assertEqual(result.loc.shape, torch.Size([3]))
        self.assertEqual(result.scale.shape, torch.Size([3]))

        # Check values
        self.assertTrue(torch.all(result.loc == loc.squeeze(0)))
        self.assertTrue(torch.all(result.scale == scale.squeeze(0)))


class TestDenseNormal(VBLLHelperTestCase):
    def test_initialization(self):
        # Test with vector input for loc and matrix for cholesky
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormal(loc, L)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.scale_tril == L))

        # Test with batch dimensions
        loc, L = _build_loc_and_cholesky((2, 3), **self.tkwargs)
        dist = DenseNormal(loc, L)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.scale_tril == L))

    def test_properties(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormal(loc, L)

        # Test mean property
        self.assertTrue(torch.all(dist.mean == loc))

        # Test chol_covariance property
        self.assertTrue(torch.all(dist.chol_covariance == L))

        # Test covariance property
        expected_cov = L @ tp(L)
        self.assertAllClose(dist.covariance, expected_cov)

        # Test logdet_covariance property
        expected_logdet = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        self.assertAllClose(dist.logdet_covariance, expected_logdet)

        # Test trace_covariance property
        expected_trace = (L**2).sum(-1).sum(-1)
        self.assertAllClose(dist.trace_covariance, expected_trace)

    def test_inner_products(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormal(loc, L)

        # Test vector for inner products
        b = torch.tensor([[1.0], [2.0], [3.0]], **self.tkwargs)

        # Test covariance_weighted_inner_prod
        expected_cov_prod = ((tp(L) @ b) ** 2).sum(-2)
        actual_cov_prod = dist.covariance_weighted_inner_prod(b)
        self.assertAllClose(actual_cov_prod, expected_cov_prod.squeeze(-1))

        # Test precision_weighted_inner_prod
        expected_prec_prod = (
            torch.linalg.solve_triangular(L, b, upper=False) ** 2
        ).sum(-2)
        actual_prec_prod = dist.precision_weighted_inner_prod(b)
        self.assertAllClose(actual_prec_prod, expected_prec_prod.squeeze(-1))

    def test_inverse_covariance(self):
        # Test the inverse_covariance property that uses solve_triangular
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormal(loc, L)

        # Compute expected inverse_covariance
        Eye = torch.eye(
            L.shape[-1],
            device=L.device,
            dtype=L.dtype,
        )
        W = torch.linalg.solve_triangular(L, Eye, upper=False)
        expected_inv_cov = tp(W) @ W

        # Compare with the property
        actual_inv_cov = dist.inverse_covariance
        self.assertAllClose(actual_inv_cov, expected_inv_cov)

        # verify that inverse_covariance * covariance ≈ identity
        cov = dist.covariance
        identity_approx = actual_inv_cov @ cov
        eye = torch.eye(3, dtype=torch.float64)
        self.assertAllClose(identity_approx, eye, rtol=1e-5, atol=1e-5)

    def test_matmul(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormal(loc, L)

        # Test matrix multiplication with a vector
        vec = torch.tensor([[2.0], [3.0], [4.0]], **self.tkwargs)
        result = dist @ vec

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc @ vec
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale
        new_cov = dist.covariance_weighted_inner_prod(
            vec.unsqueeze(-3), reduce_dim=False
        )
        expected_scale = torch.sqrt(torch.clip(new_cov, min=1e-12))
        self.assertAllClose(result.scale, expected_scale)

    def test_squeeze(self):
        # Create a distribution with an extra dimension
        loc, L = _build_loc_and_cholesky((1, 3), **self.tkwargs)
        dist = DenseNormal(loc, L)

        # Squeeze the first dimension
        result = dist.squeeze(0)

        # Check result type
        self.assertIsInstance(result, DenseNormal)

        # Check dimensions
        self.assertEqual(result.loc.shape, torch.Size([3]))
        self.assertEqual(result.scale_tril.shape, torch.Size([3, 3]))

        # Check values
        self.assertTrue(torch.all(result.loc == loc.squeeze(0)))
        self.assertTrue(torch.all(result.scale_tril == L.squeeze(0)))


class TestLowRankNormal(VBLLHelperTestCase):
    def test_initialization(self):
        # Test with vector input for loc and matrices for cov_factor and diag
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        cov_factor = torch.randn(
            3, 2, **self.tkwargs
        )  # 3-dim vector with rank 2 factor
        diag = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = LowRankNormal(loc, cov_factor, diag)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.cov_factor == cov_factor))
        self.assertTrue(torch.all(dist.cov_diag == diag))

        # Test with batch dimensions
        loc = torch.randn(2, 3, **self.tkwargs)
        cov_factor = torch.randn(2, 3, 2, **self.tkwargs)
        diag = torch.rand(2, 3, **self.tkwargs) + 0.1  # Ensure positive
        dist = LowRankNormal(loc, cov_factor, diag)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.cov_factor == cov_factor))
        self.assertTrue(torch.all(dist.cov_diag == diag))

    def test_properties(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        cov_factor = torch.randn(3, 2, **self.tkwargs)
        diag = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = LowRankNormal(loc, cov_factor, diag)

        # Test mean property
        self.assertTrue(torch.all(dist.mean == loc))

        # Test NotImplementedError for unimplemented properties
        with self.assertRaises(NotImplementedError):
            _ = dist.chol_covariance

        with self.assertRaises(NotImplementedError):
            _ = dist.inverse_covariance

        # Test logdet_covariance property
        term1 = torch.log(diag).sum(-1)
        arg1 = tp(cov_factor) @ (cov_factor / diag.unsqueeze(-1))
        term2 = torch.linalg.det(
            arg1 + torch.eye(arg1.shape[-1], dtype=torch.float64)
        ).log()
        expected_logdet = term1 + term2
        self.assertAllClose(dist.logdet_covariance, expected_logdet)

        # Test trace_covariance property
        expected_trace_diag = diag.sum(-1)
        expected_trace_lowrank = (cov_factor**2).sum(-1).sum(-1)
        expected_trace = expected_trace_diag + expected_trace_lowrank
        self.assertAllClose(dist.trace_covariance, expected_trace)

    def test_inner_products(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        cov_factor = torch.randn(3, 2, **self.tkwargs)
        diag = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = LowRankNormal(loc, cov_factor, diag)

        # Test vector for inner products
        b = torch.tensor([[1.0], [2.0], [3.0]], **self.tkwargs)

        # Test covariance_weighted_inner_prod
        diag_term = (diag.unsqueeze(-1) * (b**2)).sum(-2)
        factor_term = ((tp(cov_factor) @ b) ** 2).sum(-2)
        expected_cov_prod = diag_term + factor_term
        actual_cov_prod = dist.covariance_weighted_inner_prod(b)
        self.assertAllClose(actual_cov_prod, expected_cov_prod.squeeze(-1))

        # Test precision_weighted_inner_prod raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = dist.precision_weighted_inner_prod(b)

    def test_matmul(self):
        loc = torch.tensor([0.0, 1.0, 2.0], **self.tkwargs)
        cov_factor = torch.randn(3, 2, **self.tkwargs)
        diag = torch.tensor([1.0, 2.0, 3.0], **self.tkwargs)
        dist = LowRankNormal(loc, cov_factor, diag)

        # Test matrix multiplication with a vector
        vec = torch.tensor([[2.0], [3.0], [4.0]], **self.tkwargs)
        result = dist @ vec

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc @ vec
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale
        new_cov = dist.covariance_weighted_inner_prod(
            vec.unsqueeze(-3), reduce_dim=False
        )
        expected_scale = torch.sqrt(torch.clip(new_cov, min=1e-12))
        self.assertAllClose(result.scale, expected_scale)

    def test_squeeze(self):
        # Create a distribution with an extra dimension
        loc = torch.tensor([[0.0, 1.0, 2.0]], **self.tkwargs)
        cov_factor = torch.randn(1, 3, 2, **self.tkwargs)
        diag = torch.tensor([[1.0, 2.0, 3.0]], **self.tkwargs)
        dist = LowRankNormal(loc, cov_factor, diag)

        # Squeeze the first dimension
        result = dist.squeeze(0)

        # Check result type
        self.assertIsInstance(result, LowRankNormal)

        # Check dimensions
        self.assertEqual(result.loc.shape, torch.Size([3]))
        self.assertEqual(result.cov_factor.shape, torch.Size([3, 2]))
        self.assertEqual(result.cov_diag.shape, torch.Size([3]))

        # Check values
        self.assertTrue(torch.all(result.loc == loc.squeeze(0)))
        self.assertTrue(torch.all(result.cov_factor == cov_factor.squeeze(0)))
        self.assertTrue(torch.all(result.cov_diag == diag.squeeze(0)))


class TestDenseNormalPrec(VBLLHelperTestCase):
    def test_initialization(self):
        # Test with vector input for loc and matrix for cholesky
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.tril == L))

        # Test with batch dimensions
        loc, L = _build_loc_and_cholesky((2, 3), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        self.assertTrue(torch.all(dist.loc == loc))
        self.assertTrue(torch.all(dist.tril == L))

    def test_properties(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        # Test mean property
        self.assertTrue(torch.all(dist.mean == loc))

        # Test inverse_covariance property
        expected_prec = L @ tp(L)
        self.assertAllClose(dist.inverse_covariance, expected_prec)

        # Test NotImplementedError for chol_covariance
        with self.assertRaises(NotImplementedError):
            _ = dist.chol_covariance

        # Test logdet_covariance property
        expected_logdet = -2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        self.assertAllClose(dist.logdet_covariance, expected_logdet)

    def test_inner_products(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        # Test vector for inner products
        b = torch.tensor([[1.0], [2.0], [3.0]], **self.tkwargs)

        # Test covariance_weighted_inner_prod
        expected_cov_prod = (torch.linalg.solve(L, b) ** 2).sum(-2)
        actual_cov_prod = dist.covariance_weighted_inner_prod(b)
        self.assertAllClose(actual_cov_prod, expected_cov_prod.squeeze(-1))

        # Test precision_weighted_inner_prod
        expected_prec_prod = ((tp(L) @ b) ** 2).sum(-2)
        actual_prec_prod = dist.precision_weighted_inner_prod(b)
        self.assertAllClose(actual_prec_prod, expected_prec_prod.squeeze(-1))

    def test_covariance(self):
        # Test the covariance property that uses cholesky_inverse
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        # Compute expected covariance using cholesky_inverse
        expected_cov = torch.cholesky_inverse(L)

        # Compare with the property
        actual_cov = dist.covariance
        self.assertAllClose(actual_cov, expected_cov)

        # verify that covariance * precision_matrix ≈ identity
        precision = dist.inverse_covariance
        identity_approx = actual_cov @ precision
        eye = torch.eye(3, dtype=torch.float64)
        self.assertAllClose(identity_approx, eye, rtol=1e-5, atol=1e-5)

        # check that the covariance is symmetric
        self.assertAllClose(actual_cov, actual_cov.t())

    def test_matmul(self):
        loc, L = _build_loc_and_cholesky((3,), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        # Test matrix multiplication with a vector
        vec = torch.tensor([[2.0], [3.0], [4.0]], **self.tkwargs)
        result = dist @ vec

        # Check result type
        self.assertIsInstance(result, Normal)

        # Check mean
        expected_mean = loc @ vec
        self.assertTrue(torch.all(result.mean == expected_mean))

        # Check scale
        new_cov = dist.covariance_weighted_inner_prod(
            vec.unsqueeze(-3), reduce_dim=False
        )
        expected_scale = torch.sqrt(torch.clip(new_cov, min=1e-12))
        self.assertAllClose(result.scale, expected_scale)

    def test_squeeze(self):
        # Create a distribution with an extra dimension
        loc, L = _build_loc_and_cholesky((1, 3), **self.tkwargs)
        dist = DenseNormalPrec(loc, L)

        # Squeeze the first dimension
        result = dist.squeeze(0)

        # Check result type
        self.assertIsInstance(result, DenseNormalPrec)

        # Check dimensions
        self.assertEqual(result.loc.shape, torch.Size([3]))
        self.assertEqual(result.tril.shape, torch.Size([3, 3]))

        # Check values
        self.assertTrue(torch.all(result.loc == loc.squeeze(0)))
        self.assertTrue(torch.all(result.tril == L.squeeze(0)))


class TestGetParameterization(BotorchTestCase):
    def test_get_parameterization(self):
        # Test all valid parameterizations
        self.assertEqual(get_parameterization("dense"), DenseNormal)
        self.assertEqual(get_parameterization("dense_precision"), DenseNormalPrec)
        self.assertEqual(get_parameterization("diagonal"), Normal)
        self.assertEqual(get_parameterization("lowrank"), LowRankNormal)

        # Test invalid parameterization
        with self.assertRaises(ValueError):
            get_parameterization("invalid_param")
