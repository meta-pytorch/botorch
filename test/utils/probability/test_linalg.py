#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy

import torch
from botorch.utils.probability.linalg import augment_cholesky, PivotedCholesky
from botorch.utils.testing import BotorchTestCase


class TestPivotedCholesky(BotorchTestCase):
    def setUp(self):
        super().setUp()
        n = 5
        with torch.random.fork_rng():
            torch.random.manual_seed(0)

            matrix = torch.randn(2, n, n)
            matrix = matrix @ matrix.transpose(-1, -2)

            diag = matrix.diagonal(dim1=-2, dim2=-1).sqrt()
            idiag = diag.reciprocal().unsqueeze(-1)

            piv_chol = PivotedCholesky(
                step=0,
                tril=(idiag * matrix * idiag.transpose(-2, -1)).tril(),
                perm=torch.arange(n)[None].expand(len(matrix), n).contiguous(),
                diag=diag.clone(),
            )

        self.diag = diag
        self.matrix = matrix
        self.piv_chol = piv_chol

        self.piv_chol.update_()
        self.piv_chol.pivot_(torch.tensor([2, 3]))
        self.piv_chol.update_()

    def test_update_(self):
        # Construct permuted matrices A
        n = self.matrix.shape[-1]
        A = (1 / self.diag).unsqueeze(-1) * self.matrix * (1 / self.diag).unsqueeze(-2)
        A = A.gather(-1, self.piv_chol.perm.unsqueeze(-2).repeat(1, n, 1))
        A = A.gather(-2, self.piv_chol.perm.unsqueeze(-1).repeat(1, 1, n))

        # Test upper left block
        L = torch.linalg.cholesky(A[..., :2, :2])
        self.assertTrue(L.allclose(self.piv_chol.tril[..., :2, :2]))

        # Test lower left block
        beta = torch.linalg.solve_triangular(L, A[..., :2:, 2:], upper=False)
        self.assertTrue(
            beta.transpose(-1, -2).allclose(self.piv_chol.tril[..., 2:, :2])
        )

        # Test lower right block
        schur = A[..., 2:, 2:] - beta.transpose(-1, -2) @ beta
        self.assertTrue(schur.tril().allclose(self.piv_chol.tril[..., 2:, 2:]))

    def test_pivot_(self):
        piv_chol = deepcopy(self.piv_chol)
        self.assertEqual(piv_chol.perm.tolist(), [[0, 2, 1, 3, 4], [0, 3, 2, 1, 4]])

        piv_chol.pivot_(torch.tensor([2, 3]))
        self.assertEqual(piv_chol.perm.tolist(), [[0, 2, 1, 3, 4], [0, 3, 1, 2, 4]])
        self.assertTrue(piv_chol.tril[0].equal(self.piv_chol.tril[0]))

        error_msg = "Argument `pivot` does to match with batch shape`."
        with self.assertRaisesRegex(ValueError, error_msg):
            piv_chol.pivot_(torch.tensor([1, 2, 3]))

        A = self.piv_chol.tril[1]
        B = piv_chol.tril[1]
        self.assertTrue(A[2:4, :2].equal(B[2:4, :2].roll(1, 0)))
        self.assertTrue(A[4:, 2:4].equal(B[4:, 2:4].roll(1, 1)))

    def test_pivot_gradient(self):
        # ``pivot_`` permutes the entries of ``tril`` (by swapping pairs of
        # entries), so the gradient w.r.t. ``tril`` is the same permutation of the
        # gradient w.r.t. the output. This must also hold for batch members that do
        # not pivot.
        step = self.piv_chol.step
        perm = self.piv_chol.perm
        pivot = torch.tensor([3, step])  # the second member does not pivot
        tril = self.piv_chol.tril.clone().requires_grad_(True)
        piv_chol = PivotedCholesky(step=step, tril=tril.clone(), perm=perm.clone())
        piv_chol.pivot_(pivot)
        weights = torch.rand_like(tril)
        (grad,) = torch.autograd.grad((weights * piv_chol.tril).sum(), tril)
        expected = PivotedCholesky(step=step, tril=weights.clone(), perm=perm.clone())
        expected.pivot_(pivot)
        self.assertTrue(grad.equal(expected.tril))
        self.assertTrue(grad[1].equal(weights[1]))

    def test_concat(self):
        A = self.piv_chol.expand(2, 2)
        B = self.piv_chol.expand(1, 2)
        C = B.concat(B, dim=0)
        for key in ("tril", "perm", "diag"):
            self.assertTrue(getattr(A, key).equal(getattr(C, key)))

        B.step = A.step + 1
        error_msg = "Cannot conncatenate decompositions at different steps."
        with self.assertRaisesRegex(ValueError, error_msg):
            A.concat(B, dim=0)

        B.step = A.step
        B.perm = None
        error_msg = "Types of field perm do not match."
        with self.assertRaisesRegex(NotImplementedError, error_msg):
            A.concat(B, dim=0)

    def test_clone(self):
        self.piv_chol.diag.requires_grad_(True)
        try:
            other = self.piv_chol.clone()
            for key in ("tril", "perm", "diag"):
                a = getattr(self.piv_chol, key)
                b = getattr(other, key)
                self.assertTrue(a.equal(b))
                self.assertFalse(a is b)

            other.diag.sum().backward()
            self.assertTrue(self.piv_chol.diag.grad.eq(1).all())
        finally:
            self.piv_chol.diag.requires_grad_(False)

    def test_detach(self):
        self.piv_chol.diag.requires_grad_(True)
        try:
            other = self.piv_chol.detach()
            for key in ("tril", "perm", "diag"):
                a = getattr(self.piv_chol, key)
                b = getattr(other, key)
                self.assertTrue(a.equal(b))
                self.assertFalse(a is b)

            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.diag.sum().backward()

        finally:
            self.piv_chol.diag.requires_grad_(False)

    def test_expand(self):
        other = self.piv_chol.expand(3, 2)
        for key in ("tril", "perm", "diag"):
            a = getattr(self.piv_chol, key)
            b = getattr(other, key)
            self.assertEqual(b.shape[: -a.ndim], (3,))
            self.assertTrue(b._base is a)

    def test_augment(self):
        K = self.matrix
        n = K.shape[-1]
        m = n // 2
        Kaa = K[:, 0:m, 0:m]
        Laa = torch.linalg.cholesky(Kaa)
        Kbb = K[:, m:, m:]

        error_msg = "One and only one of `Kba` or `Lba` must be provided."
        with self.assertRaisesRegex(ValueError, error_msg):
            augment_cholesky(Laa, Kbb)

        Kba = K[:, m:, 0:m]
        L_augmented = augment_cholesky(Laa, Kbb, Kba)
        L = torch.linalg.cholesky(K)
        self.assertAllClose(L_augmented, L)

        # with jitter
        jitter = 3e-2
        Laa = torch.linalg.cholesky(Kaa + jitter * torch.eye(m).unsqueeze(0))
        L_augmented = augment_cholesky(Laa, Kbb, Kba, jitter=jitter)
        L = torch.linalg.cholesky(K + jitter * torch.eye(n).unsqueeze(0))
        self.assertAllClose(L_augmented, L)

    def test_errors(self):
        matrix = self.matrix
        diag = self.diag
        diag = matrix.diagonal(dim1=-2, dim2=-1).sqrt()
        idiag = diag.reciprocal().unsqueeze(-1)
        n = matrix.shape[-1]

        # testing with erroneous inputs
        wrong_matrix = matrix[..., 0]
        error_msg = "Expected square matrices but `matrix` has shape.*"
        with self.assertRaisesRegex(ValueError, error_msg):
            PivotedCholesky(
                step=0,
                tril=wrong_matrix,
                perm=torch.arange(n)[None].expand(len(matrix), n).contiguous(),
                diag=diag.clone(),
                validate_init=True,
            )

        wrong_perm = torch.arange(n)[None].expand(2 * len(matrix), n).contiguous()
        error_msg = "`perm` of shape .* incompatible with `matrix` of shape .*"
        with self.assertRaisesRegex(ValueError, error_msg):
            PivotedCholesky(
                step=0,
                tril=(idiag * matrix * idiag.transpose(-2, -1)).tril(),
                perm=wrong_perm,
                diag=diag.clone(),
            )

        wrong_diag = torch.ones(2 * len(diag))
        error_msg = "`diag` of shape .* incompatible with `matrix` of shape .*"
        with self.assertRaises(ValueError, msg=error_msg):
            PivotedCholesky(
                step=0,
                tril=(idiag * matrix * idiag.transpose(-2, -1)).tril(),
                perm=torch.arange(n)[None].expand(len(matrix), n).contiguous(),
                diag=wrong_diag,
            )

        # testing without validation, should pass,
        # even though input does not have correct shape
        piv_chol = PivotedCholesky(
            step=0,
            tril=matrix[..., 0],
            perm=torch.arange(n)[None].expand(len(matrix), n).contiguous(),
            diag=diag.clone(),
            validate_init=False,
        )
        self.assertTrue(isinstance(piv_chol, PivotedCholesky))
