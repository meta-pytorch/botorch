#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Callable
from itertools import product
from unittest import mock

import numpy as np
import numpy.typing as npt
import torch
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch.optim.parameter_constraints import (
    _arrayify,
    _generate_unfixed_lin_constraints,
    _generate_unfixed_nonlin_constraints,
    _make_linear_constraints,
    _make_nonlinear_constraints,
    evaluate_feasibility,
    get_constraint_tolerance,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
    nonlinear_constraint_is_feasible,
    project_to_equality_constraints,
    project_to_feasible_space_via_slsqp,
)
from botorch.utils.testing import BotorchTestCase
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeWarning
from scipy.optimize._minimize import standardize_constraints


class TestParameterConstraints(BotorchTestCase):
    def test_arrayify(self):
        for dtype in (torch.float, torch.double, torch.int, torch.long):
            t = torch.tensor([[1, 2], [3, 4]], device=self.device).type(dtype)
            t_np = _arrayify(t)
            self.assertIsInstance(t_np, np.ndarray)
            self.assertTrue(t_np.dtype == np.float64)

    def test_make_nonlinear_constraints(self):
        def nlc(x):
            return 4 - x.sum()

        def f_np_wrapper(x: npt.NDArray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            X = (
                torch.from_numpy(x)
                .to(self.device)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            loss = f(X).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            fval = loss.item()
            return fval, gradf

        shapeX = torch.Size((3, 2, 4))
        b, q, d = shapeX
        x = np.random.rand(shapeX.numel())
        # intra
        constraints = _make_nonlinear_constraints(
            f_np_wrapper=f_np_wrapper, nlc=nlc, is_intrapoint=True, shapeX=shapeX
        )
        self.assertEqual(len(constraints), b * q)
        self.assertTrue(
            all(set(c.keys()) == {"fun", "jac", "type"} for c in constraints)
        )
        self.assertTrue(all(c["type"] == "ineq" for c in constraints))
        self.assertAllClose(constraints[0]["fun"](x), 4.0 - x[:d].sum())
        self.assertAllClose(constraints[1]["fun"](x), 4.0 - x[d : 2 * d].sum())
        jac_exp = np.zeros(shapeX.numel())
        jac_exp[:4] = -1
        self.assertAllClose(constraints[0]["jac"](x), jac_exp)
        jac_exp = np.zeros(shapeX.numel())
        jac_exp[4:8] = -1
        self.assertAllClose(constraints[1]["jac"](x), jac_exp)
        # inter
        constraints = _make_nonlinear_constraints(
            f_np_wrapper=f_np_wrapper, nlc=nlc, is_intrapoint=False, shapeX=shapeX
        )
        self.assertEqual(len(constraints), 3)
        self.assertTrue(
            all(set(c.keys()) == {"fun", "jac", "type"} for c in constraints)
        )
        self.assertTrue(all(c["type"] == "ineq" for c in constraints))
        self.assertAllClose(constraints[0]["fun"](x), 4.0 - x[: q * d].sum())
        self.assertAllClose(constraints[1]["fun"](x), 4.0 - x[q * d : 2 * q * d].sum())
        jac_exp = np.zeros(shapeX.numel())
        jac_exp[: q * d] = -1.0
        self.assertAllClose(constraints[0]["jac"](x), jac_exp)
        jac_exp = np.zeros(shapeX.numel())
        jac_exp[q * d : 2 * q * d] = -1.0
        self.assertAllClose(constraints[1]["jac"](x), jac_exp)

    def test_make_scipy_nonlinear_inequality_constraints(self):
        def nlc(x):
            return 4 - x.sum()

        def f_np_wrapper(x: npt.NDArray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            X = (
                torch.from_numpy(x)
                .to(self.device)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            loss = f(X).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            fval = loss.item()
            return fval, gradf

        shapeX = torch.Size((3, 2, 4))
        b, q, _ = shapeX
        x = torch.ones(shapeX.numel(), device=self.device)

        with self.assertRaisesRegex(
            ValueError, f"A nonlinear constraint has to be a tuple, got {type(nlc)}."
        ):
            make_scipy_nonlinear_inequality_constraints([nlc], f_np_wrapper, x, shapeX)
        with self.assertRaisesRegex(
            ValueError,
            "A nonlinear constraint has to be a tuple of length 2, got length 1.",
        ):
            make_scipy_nonlinear_inequality_constraints(
                [(nlc,)], f_np_wrapper, x, shapeX
            )
        with self.assertRaisesRegex(
            ValueError,
            "`batch_initial_conditions` must satisfy the non-linear inequality "
            "constraints.",
        ):
            make_scipy_nonlinear_inequality_constraints(
                [(nlc, False)], f_np_wrapper, x, shapeX
            )
        # empty list
        res = make_scipy_nonlinear_inequality_constraints([], f_np_wrapper, x, shapeX)
        self.assertEqual(res, [])
        # only inter
        x = torch.zeros(shapeX.numel(), device=self.device)
        res = make_scipy_nonlinear_inequality_constraints(
            [(nlc, False)], f_np_wrapper, x, shapeX
        )
        self.assertEqual(len(res), b)
        # only intra
        res = make_scipy_nonlinear_inequality_constraints(
            [(nlc, True)], f_np_wrapper, x, shapeX
        )
        self.assertEqual(len(res), b * q)
        # intra and inter
        res = make_scipy_nonlinear_inequality_constraints(
            [(nlc, True), (nlc, False)], f_np_wrapper, x, shapeX
        )
        self.assertEqual(len(res), b * q + b)

    def test_make_linear_constraints(self):
        # equality constraints, 1d indices (intra-point) -> one row per (b, q) pair
        indices = torch.tensor([1, 2], dtype=torch.long, device=self.device)
        for dtype, shapeX in product(
            (torch.float, torch.double), (torch.Size([3, 2, 4]), torch.Size([2, 4]))
        ):
            b = shapeX[0] if len(shapeX) == 3 else 1
            q, d = shapeX[-2:]
            n = shapeX.numel()
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=self.device)
            block = _make_linear_constraints(
                indices=indices,
                coefficients=coefficients,
                rhs=1.0,
                shapeX=shapeX,
                eq=True,
            )
            # eq → lb == ub == rhs; row count = b * q
            self.assertEqual(block.n_rows, b * q)
            self.assertTrue(np.allclose(block.lb, 1.0))
            self.assertTrue(np.allclose(block.ub, 1.0))
            # Materialize and verify per-row semantics against legacy formula.
            A = sparse.coo_array(
                (block.vals, (block.rows, block.cols)),
                shape=(block.n_rows, n),
            ).toarray()
            x = np.random.rand(n)
            # First row is (i=0, j=0) → columns shifted by indices only.
            self.assertAlmostEqual(float(A[0] @ x), x[1] + 2 * x[2])
            # Last row is (i=b-1, j=q-1).
            offset = (b - 1) * q * d + (q - 1) * d
            self.assertAlmostEqual(float(A[-1] @ x), x[offset + 1] + 2 * x[offset + 2])

        # inequality constraints, 1d indices → lb=rhs, ub=+inf
        for shapeX in [torch.Size([1, 1, 2]), torch.Size([1, 2])]:
            block = _make_linear_constraints(
                indices=torch.tensor([1]),
                coefficients=torch.tensor([1.0]),
                rhs=1.0,
                shapeX=shapeX,
                eq=False,
            )
            self.assertEqual(block.n_rows, 1)
            self.assertTrue(np.allclose(block.lb, 1.0))
            self.assertTrue(np.all(np.isinf(block.ub)))

        # inter-point: 2d indices → one row per t-batch element
        indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.long, device=self.device)
        for dtype, shapeX in product(
            (torch.float, torch.double), (torch.Size([3, 2, 4]), torch.Size([2, 4]))
        ):
            b = shapeX[0] if len(shapeX) == 3 else 1
            q, d = shapeX[-2:]
            n = shapeX.numel()
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=self.device)
            block = _make_linear_constraints(
                indices=indices,
                coefficients=coefficients,
                rhs=1.0,
                shapeX=shapeX,
                eq=True,
            )
            self.assertEqual(block.n_rows, b)
            self.assertTrue(np.allclose(block.lb, 1.0))
            self.assertTrue(np.allclose(block.ub, 1.0))
            A = sparse.coo_array(
                (block.vals, (block.rows, block.cols)),
                shape=(block.n_rows, n),
            ).toarray()
            x = np.random.rand(n)
            for i in range(b):
                pos1 = i * (q * d) + 0 * d + 3  # q-row=0, feature=3
                pos2 = i * (q * d) + 1 * d + 2  # q-row=1, feature=2
                self.assertAlmostEqual(float(A[i] @ x), x[pos1] + 2 * x[pos2])

        # scalar tensor → ValueError
        with self.assertRaises(ValueError):
            _make_linear_constraints(
                indices=torch.tensor(0),
                coefficients=torch.tensor([1.0]),
                rhs=1.0,
                shapeX=torch.Size([1, 1, 2]),
                eq=False,
            )
        # shapeX dim < 2 → UnsupportedError
        with self.assertRaises(UnsupportedError):
            _make_linear_constraints(
                shapeX=torch.Size([2]),
                indices=indices,
                coefficients=coefficients,
                rhs=0.0,
            )

    def test_make_scipy_linear_constraints(self):
        for shapeX in [torch.Size([2, 1, 4]), torch.Size([1, 4])]:
            b = shapeX[0] if len(shapeX) == 3 else 1
            n = shapeX.numel()
            # Empty list when no constraints.
            res = make_scipy_linear_constraints(
                shapeX=shapeX, inequality_constraints=None, equality_constraints=None
            )
            self.assertEqual(res, [])

            indices = torch.tensor([0, 1], dtype=torch.long, device=self.device)
            coefficients = torch.tensor([1.5, -1.0], device=self.device)

            # Both inequality and equality constraints → two separate
            # LinearConstraints (ineq first, eq second). Splitting is what
            # silences scipy's OptimizeWarning on SLSQP conversion.
            lcs = make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
            self.assertEqual(len(lcs), 2)
            for lc in lcs:
                self.assertIsInstance(lc, LinearConstraint)
                self.assertTrue(sparse.issparse(lc.A))
                self.assertEqual(lc.A.shape, (b, n))
            ineq_lc, eq_lc = lcs
            # Inequality: lb=rhs, ub=+inf. Equality: lb=ub=rhs.
            self.assertTrue(np.allclose(ineq_lc.lb, 1.0))
            self.assertTrue(np.all(np.isinf(ineq_lc.ub)))
            self.assertTrue(np.allclose(eq_lc.lb, 1.0))
            self.assertTrue(np.allclose(eq_lc.ub, 1.0))

            # Round-trip via scipy's standardize_constraints → dicts. With
            # eq and ineq in separate LCs, scipy emits one dict per LC
            # (vectorized fun returning a length-b vector; vectorized jac
            # returning the dense A) and the OptimizeWarning we used to get
            # for "eq and ineq in the same constraint" no longer fires.
            x = np.random.rand(n)
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                old = standardize_constraints(lcs, x0=np.zeros(n), meth="slsqp")
            self.assertEqual({c["type"] for c in old}, {"ineq", "eq"})
            ineq_A = ineq_lc.A.toarray()
            eq_A = eq_lc.A.toarray()
            ineq_expected = ineq_A @ x - ineq_lc.lb
            eq_expected = eq_A @ x - eq_lc.lb
            for c in old:
                val = np.atleast_1d(np.asarray(c["fun"](x), dtype=float))
                jac = np.atleast_2d(np.asarray(c["jac"](x), dtype=float))
                if c["type"] == "ineq":
                    np.testing.assert_allclose(val, ineq_expected, atol=1e-10)
                    np.testing.assert_allclose(jac, ineq_A, atol=1e-10)
                else:
                    np.testing.assert_allclose(val, eq_expected, atol=1e-10)
                    np.testing.assert_allclose(jac, eq_A, atol=1e-10)

            # Inequality only → single LC, length-1 list.
            lcs = make_scipy_linear_constraints(
                shapeX=shapeX, inequality_constraints=[(indices, coefficients, 1.0)]
            )
            self.assertEqual(len(lcs), 1)
            self.assertEqual(lcs[0].A.shape, (b, n))
            self.assertTrue(np.all(np.isinf(lcs[0].ub)))
            # Equality only → single LC, length-1 list.
            lcs = make_scipy_linear_constraints(
                shapeX=shapeX, equality_constraints=[(indices, coefficients, 1.0)]
            )
            self.assertEqual(len(lcs), 1)
            self.assertEqual(lcs[0].A.shape, (b, n))
            self.assertTrue(np.allclose(lcs[0].lb, lcs[0].ub))

            # 2-dim (inter-point) indices: same two-LC split, b rows each.
            indices2 = indices.unsqueeze(0)
            lcs = make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices2, coefficients, 1.0)],
                equality_constraints=[(indices2, coefficients, 1.0)],
            )
            self.assertEqual(len(lcs), 2)
            for lc in lcs:
                self.assertEqual(lc.A.shape, (b, n))

    def test_make_scipy_linear_constraints_sparsity_intra_point(self):
        """Intra-point mixture: per-row column sets are pairwise disjoint
        (block-diagonal pattern) and nnz == b * q * len(indices)."""
        b, q, d = 4, 3, 5
        shapeX = torch.Size([b, q, d])
        indices = torch.tensor([0, 1, 2], dtype=torch.long, device=self.device)
        coefficients = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        (lc,) = make_scipy_linear_constraints(
            shapeX=shapeX,
            equality_constraints=[(indices, coefficients, 1.0)],
        )
        self.assertEqual(lc.A.shape, (b * q, b * q * d))
        self.assertEqual(lc.A.nnz, b * q * len(indices))
        dense = lc.A.toarray()
        col_sets = [set(np.flatnonzero(dense[row])) for row in range(b * q)]
        for r1 in range(b * q):
            self.assertEqual(len(col_sets[r1]), len(indices))
            for r2 in range(r1 + 1, b * q):
                self.assertTrue(col_sets[r1].isdisjoint(col_sets[r2]))

    def test_make_scipy_linear_constraints_inter_point_pattern(self):
        """Inter-point: scattered strided non-zeros, e.g. cols [0, d, 2d]."""
        b, q, d = 1, 3, 5
        shapeX = torch.Size([b, q, d])
        # one constraint touching feature 0 of each q-row
        indices = torch.tensor(
            [[0, 0], [1, 0], [2, 0]], dtype=torch.long, device=self.device
        )
        coefficients = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        (lc,) = make_scipy_linear_constraints(
            shapeX=shapeX,
            inequality_constraints=[(indices, coefficients, 0.0)],
        )
        self.assertEqual(lc.A.shape, (b, b * q * d))
        nz_cols = sorted(np.flatnonzero(lc.A.toarray()[0]).tolist())
        self.assertEqual(nz_cols, [0, d, 2 * d])

    def test_make_scipy_linear_constraints_mixed_intra_inter(self):
        """Stack an intra-point and an inter-point constraint into one
        LinearConstraint, verifying the COO row-offset arithmetic across
        blocks with mismatched n_rows (intra contributes b*q rows; inter
        contributes b)."""
        b, q, d = 3, 2, 4
        shapeX = torch.Size([b, q, d])
        n = shapeX.numel()
        intra_indices = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        inter_indices = torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.long, device=self.device
        )
        coeffs = torch.tensor([1.0, -1.0], device=self.device)
        (lc,) = make_scipy_linear_constraints(
            shapeX=shapeX,
            equality_constraints=[
                (intra_indices, coeffs, 0.5),  # broadcast → b*q = 6 rows
                (inter_indices, coeffs, 0.7),  # broadcast → b   = 3 rows
            ],
        )
        # 6 intra rows then 3 inter rows over n = b*q*d = 24 columns.
        self.assertEqual(lc.A.shape, (b * q + b, n))
        self.assertTrue(np.allclose(lc.lb[: b * q], 0.5))
        self.assertTrue(np.allclose(lc.ub[: b * q], 0.5))
        self.assertTrue(np.allclose(lc.lb[b * q :], 0.7))
        self.assertTrue(np.allclose(lc.ub[b * q :], 0.7))

        x = np.random.rand(n)
        A = lc.A.toarray()
        # Intra row 0 → (i=0, j=0); cols [0, 1].
        self.assertAlmostEqual(float(A[0] @ x), x[0] - x[1])
        # Intra row 1 → (i=0, j=1); cols offset by d = 4.
        self.assertAlmostEqual(float(A[1] @ x), x[4] - x[5])
        # Intra last row → (i=b-1, j=q-1); cols (b-1)*q*d + (q-1)*d + [0, 1].
        intra_last_off = (b - 1) * q * d + (q - 1) * d
        self.assertAlmostEqual(
            float(A[b * q - 1] @ x), x[intra_last_off] - x[intra_last_off + 1]
        )
        # Inter row 0 (== global row b*q) → i=0; cols [0*d+2, 1*d+3] = [2, 7].
        self.assertAlmostEqual(float(A[b * q] @ x), x[2] - x[7])
        # Inter last row → i=b-1; cols (b-1)*q*d + [0*d+2, 1*d+3] = [18, 23].
        inter_last_off = (b - 1) * q * d
        self.assertAlmostEqual(
            float(A[-1] @ x),
            x[inter_last_off + 2] - x[inter_last_off + d + 3],
        )

        # Round-trip via scipy's standardizer: one ineq dict would be absent
        # (we passed only equalities), so we expect one eq dict whose
        # vectorized fun(x) equals A @ x - lb element-wise.
        old = standardize_constraints([lc], x0=np.zeros(n), meth="slsqp")
        self.assertEqual({c["type"] for c in old}, {"eq"})
        eq_dict = next(c for c in old if c["type"] == "eq")
        np.testing.assert_allclose(
            np.atleast_1d(np.asarray(eq_dict["fun"](x), dtype=float)),
            A @ x - lc.lb,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.atleast_2d(np.asarray(eq_dict["jac"](x), dtype=float)),
            A,
            atol=1e-10,
        )

    def test_make_scipy_linear_constraints_unsupported(self):
        shapeX = torch.Size([2, 1, 4])
        coefficients = torch.tensor([1.5, -1.0], device=self.device)

        # test that >2-dim indices raises an UnsupportedError
        indices = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        indices = indices.unsqueeze(0).unsqueeze(0)
        with self.assertRaises(UnsupportedError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # test that out of bounds index raises an error
        indices = torch.tensor([0, 4], dtype=torch.long, device=self.device)
        with self.assertRaises(RuntimeError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # test that two-d index out-of-bounds raises an error
        # q out of bounds
        indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.long, device=self.device)
        with self.assertRaises(RuntimeError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
        # d out of bounds
        indices = torch.tensor([[0, 0], [0, 4]], dtype=torch.long, device=self.device)
        with self.assertRaises(RuntimeError):
            make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )

    def test_nonlinear_constraint_is_feasible(self):
        def nlc(x):
            return 4 - x.sum()

        self.assertTrue(
            nonlinear_constraint_is_feasible(
                nlc, True, torch.tensor([[[1.5, 1.5], [1.5, 1.5]]], device=self.device)
            )
        )
        self.assertFalse(
            nonlinear_constraint_is_feasible(
                nlc,
                True,
                torch.tensor(
                    [[[1.5, 1.5], [1.5, 1.5], [3.5, 1.5]]], device=self.device
                ),
            )
        )
        self.assertEqual(
            nonlinear_constraint_is_feasible(
                nlc,
                True,
                torch.tensor(
                    [[[1.5, 1.5], [1.5, 1.5]], [[1.5, 1.5], [1.5, 3.5]]],
                    device=self.device,
                ),
            ).tolist(),
            [True, False],
        )
        self.assertTrue(
            nonlinear_constraint_is_feasible(
                nlc, False, torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], device=self.device)
            )
        )
        self.assertTrue(
            nonlinear_constraint_is_feasible(
                nlc,
                False,
                torch.tensor(
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    device=self.device,
                ),
            ).all()
        )
        self.assertFalse(
            nonlinear_constraint_is_feasible(
                nlc, False, torch.tensor([[[1.5, 1.5], [1.5, 1.5]]], device=self.device)
            )
        )
        self.assertEqual(
            nonlinear_constraint_is_feasible(
                nlc,
                False,
                torch.tensor(
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.5, 1.5], [1.5, 1.5]]],
                    device=self.device,
                ),
            ).tolist(),
            [True, False],
        )

    def test_generate_unfixed_nonlin_constraints(self):
        def nlc1(x):
            return 4 - x.sum(dim=-1)

        def nlc2(x):
            return x[..., 0] - 1

        # first test with one constraint
        (new_nlc1,) = _generate_unfixed_nonlin_constraints(
            constraints=[(nlc1, True)], fixed_features={1: 2.0}, dimension=3
        )
        self.assertAllClose(
            nlc1(torch.tensor([[4.0, 2.0, 2.0]], device=self.device)),
            new_nlc1[0](torch.tensor([[4.0, 2.0]], device=self.device)),
        )
        # test with several constraints
        constraints = [(nlc1, True), (nlc2, True)]
        new_constraints = _generate_unfixed_nonlin_constraints(
            constraints=constraints, fixed_features={1: 2.0}, dimension=3
        )
        for nlc, new_nlc in zip(constraints, new_constraints):
            self.assertAllClose(
                nlc[0](torch.tensor([[4.0, 2.0, 2.0]], device=self.device)),
                new_nlc[0](torch.tensor([[4.0, 2.0]], device=self.device)),
            )
        # test with several constraints and two fixes
        constraints = [(nlc1, True), (nlc2, True)]
        new_constraints = _generate_unfixed_nonlin_constraints(
            constraints=constraints, fixed_features={1: 2.0, 2: 1.0}, dimension=3
        )
        for nlc, new_nlc in zip(constraints, new_constraints):
            self.assertAllClose(
                nlc[0](torch.tensor([[4.0, 2.0, 1.0]], device=self.device)),
                new_nlc[0](torch.tensor([[4.0]], device=self.device)),
            )

    def test_generate_unfixed_lin_constraints(self):
        # Case 1: some fixed features are in the indices
        indices = [
            torch.arange(4, device=self.device),
            torch.arange(2, -1, -1, device=self.device),
        ]
        coefficients = [
            torch.tensor([-0.1, 0.2, -0.3, 0.4], device=self.device),
            torch.tensor([-0.1, 0.3, -0.5], device=self.device),
        ]
        rhs = [0.5, 0.5]
        dimension = 4
        fixed_features = {1: 1, 3: 2}
        new_constraints = _generate_unfixed_lin_constraints(
            constraints=list(zip(indices, coefficients, rhs)),
            fixed_features=fixed_features,
            dimension=dimension,
            eq=False,
        )
        for i, (new_indices, new_coefficients, new_rhs) in enumerate(new_constraints):
            if i % 2 == 0:  # first list of indices is [0, 1, 2, 3]
                self.assertTrue(
                    torch.equal(new_indices, torch.arange(2, device=self.device))
                )
            else:  # second list of indices is [2, 1, 0]
                self.assertTrue(
                    torch.equal(
                        new_indices, torch.arange(1, -1, -1, device=self.device)
                    )
                )
            mask = [True] * indices[i].shape[0]
            subtract = 0
            for j, old_idx in enumerate(indices[i]):
                if old_idx.item() in fixed_features:
                    mask[j] = False
                    subtract += fixed_features[old_idx.item()] * coefficients[i][j]
            self.assertTrue(torch.equal(new_coefficients, coefficients[i][mask]))
            self.assertEqual(new_rhs, rhs[i] - subtract)

        # Case 2: none of fixed features are in the indices, but have to be renumbered
        indices = [
            torch.arange(2, 6, device=self.device),
            torch.arange(5, 2, -1, device=self.device),
        ]
        fixed_features = {0: -10, 1: 10}
        dimension = 6
        new_constraints = _generate_unfixed_lin_constraints(
            constraints=list(zip(indices, coefficients, rhs)),
            fixed_features=fixed_features,
            dimension=dimension,
            eq=False,
        )
        for i, (new_indices, new_coefficients, new_rhs) in enumerate(new_constraints):
            if i % 2 == 0:  # first list of indices is [2, 3, 4, 5]
                self.assertTrue(
                    torch.equal(new_indices, torch.arange(4, device=self.device))
                )
            else:  # second list of indices is [5, 4, 3]
                self.assertTrue(
                    torch.equal(new_indices, torch.arange(3, 0, -1, device=self.device))
                )

            self.assertTrue(torch.equal(new_coefficients, coefficients[i]))
            self.assertEqual(new_rhs, rhs[i])

        # Case 3: all fixed features are in the indices
        indices = [
            torch.arange(4, device=self.device),
            torch.arange(2, -1, -1, device=self.device),
        ]
        # Case 3a: problem is feasible
        dimension = 4
        fixed_features = {0: 2, 1: 1, 2: 1, 3: 2}
        for eq in [False, True]:
            new_constraints = _generate_unfixed_lin_constraints(
                constraints=[(indices[0], coefficients[0], rhs[0])],
                fixed_features=fixed_features,
                dimension=dimension,
                eq=eq,
            )
            self.assertEqual(new_constraints, [])
        # Case 3b: problem is infeasible
        for eq in [False, True]:
            prefix = "Ineq" if not eq else "Eq"
            with self.assertRaisesRegex(CandidateGenerationError, prefix):
                new_constraints = _generate_unfixed_lin_constraints(
                    constraints=[(indices[1], coefficients[1], rhs[1])],
                    fixed_features=fixed_features,
                    dimension=dimension,
                    eq=eq,
                )

    def test_evaluate_feasibility(self) -> None:
        # Check that the feasibility is evaluated correctly.
        X = torch.tensor(  # 3 x 2 x 3 -> leads to output of shape 3.
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 3.0]],
                [[2.0, 2.0, 1.0], [2.0, 2.0, 5.0]],
                [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            ],
            device=self.device,
        )
        # X[..., 2] * 4 >= 5.
        inequality_constraints = [
            (
                torch.tensor([2], device=self.device),
                torch.tensor([4], device=self.device),
                5.0,
            )
        ]
        # X[..., 0] + X[..., 1] == 4.
        equality_constraints = [
            (
                torch.tensor([0, 1], device=self.device),
                torch.ones(2, device=self.device),
                4.0,
            )
        ]

        # sum(X, dim=-1) < 5.
        def nlc1(x):
            return 5 - x.sum(dim=-1)

        # Only inequality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                inequality_constraints=inequality_constraints,
            ),
            torch.tensor([False, False, True], device=self.device),
        )
        # Only equality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                equality_constraints=equality_constraints,
            ),
            torch.tensor([False, True, False], device=self.device),
        )
        # Both inequality and equality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            ),
            torch.tensor([False, False, False], device=self.device),
        )
        # Nonlinear inequality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                nonlinear_inequality_constraints=[(nlc1, True)],
            ),
            torch.tensor([True, False, False], device=self.device),
        )
        # No constraints.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
            ),
            torch.ones(3, device=self.device, dtype=torch.bool),
        )

    def test_evaluate_feasibility_inter_point(self) -> None:
        # Check that inter-point constraints evaluate correctly.
        X = torch.tensor(  # 3 x 2 x 3 -> leads to output of shape 3.
            [
                [[1.0, 1.0, 1.0], [0.0, 1.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 1.0, 3.0]],
                [[2.0, 2.0, 1.0], [2.0, 2.0, 5.0]],
            ],
            dtype=torch.double,
            device=self.device,
        )
        linear_inter_cons = (  # X[..., 0, 0] - X[..., 1, 0] >= / == 0.
            torch.tensor([[0, 0], [1, 0]], device=self.device),
            torch.tensor([1.0, -1.0], device=self.device),
            0,
        )
        # Linear inequality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                inequality_constraints=[linear_inter_cons],
            ),
            torch.tensor([True, False, True], device=self.device),
        )
        # Linear equality.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                equality_constraints=[linear_inter_cons],
            ),
            torch.tensor([False, False, True], device=self.device),
        )
        # Linear equality with too high of a tolerance.
        self.assertAllClose(
            evaluate_feasibility(
                X=X,
                equality_constraints=[linear_inter_cons],
                tolerance=100,
            ),
            torch.tensor([True, True, True], device=self.device),
        )

        # Nonlinear inequality.
        def nlc1(x):  # X.sum(over q & d) >= 10.0
            return x.sum() - 10.0

        self.assertEqual(
            evaluate_feasibility(
                X=X,
                nonlinear_inequality_constraints=[(nlc1, False)],
            ).tolist(),
            [False, False, True],
        )
        # All together.
        self.assertEqual(
            evaluate_feasibility(
                X=X,
                inequality_constraints=[linear_inter_cons],
                equality_constraints=[linear_inter_cons],
                nonlinear_inequality_constraints=[(nlc1, False)],
            ).tolist(),
            [False, False, True],
        )

    def test_get_constraint_tolerance(self):
        self.assertEqual(get_constraint_tolerance(dtype=torch.double), 1e-8)
        self.assertEqual(get_constraint_tolerance(dtype=torch.float), 1e-6)
        self.assertEqual(get_constraint_tolerance(dtype=torch.half), 1e-4)
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            get_constraint_tolerance(dtype=torch.long)


class TestMakeScipyBounds(BotorchTestCase):
    def test_make_scipy_bounds(self):
        X = torch.zeros(3, 1, 2)
        # both None
        self.assertIsNone(make_scipy_bounds(X=X, lower_bounds=None, upper_bounds=None))
        # lower None
        upper_bounds = torch.ones(2)
        bounds = make_scipy_bounds(X=X, lower_bounds=None, upper_bounds=upper_bounds)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(
            np.all(np.equal(bounds.lb, np.full((3, 1, 2), float("-inf")).flatten()))
        )
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))
        # upper None
        lower_bounds = torch.zeros(2)
        bounds = make_scipy_bounds(X=X, lower_bounds=lower_bounds, upper_bounds=None)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(
            np.all(np.equal(bounds.ub, np.full((3, 1, 2), float("inf")).flatten()))
        )
        # floats
        bounds = make_scipy_bounds(X=X, lower_bounds=0.0, upper_bounds=1.0)
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))

        # 1-d tensors
        bounds = make_scipy_bounds(
            X=X, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )
        self.assertIsInstance(bounds, Bounds)
        self.assertTrue(np.all(np.equal(bounds.lb, np.zeros((3, 1, 2)).flatten())))
        self.assertTrue(np.all(np.equal(bounds.ub, np.ones((3, 1, 2)).flatten())))


class TestProjectToEqualityConstraints(BotorchTestCase):
    def test_project_to_equality_constraints(self):
        for dtype in (torch.float, torch.double):
            # Test 1: Single equality constraint x[0] + x[1] = 1.0
            X = torch.tensor(
                [[[0.6, 0.6]]], dtype=dtype, device=self.device
            )  # violates: sum = 1.2
            eq_constraints = [
                (
                    torch.tensor([0, 1], device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.0,
                )
            ]
            projected = project_to_equality_constraints(X, eq_constraints)
            # Check constraint is satisfied
            self.assertAlmostEqual(
                projected[..., 0].item() + projected[..., 1].item(), 1.0, places=6
            )
            # Check it's the closest point (equal correction to both dims)
            self.assertAlmostEqual(projected[0, 0, 0].item(), 0.5, places=6)
            self.assertAlmostEqual(projected[0, 0, 1].item(), 0.5, places=6)

            # Test 2: Already feasible point
            X_feasible = torch.tensor([[[0.3, 0.7]]], dtype=dtype, device=self.device)
            projected = project_to_equality_constraints(X_feasible, eq_constraints)
            self.assertAllClose(projected, X_feasible, atol=1e-6)

            # Test 3: Multiple constraints
            # x[0] + x[1] + x[2] = 1.0 and x[0] - x[1] = 0.0
            X3 = torch.tensor([[[0.5, 0.3, 0.4]]], dtype=dtype, device=self.device)
            eq_constraints_multi = [
                (
                    torch.tensor([0, 1, 2], device=self.device),
                    torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=self.device),
                    1.0,
                ),
                (
                    torch.tensor([0, 1], device=self.device),
                    torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                    0.0,
                ),
            ]
            projected = project_to_equality_constraints(X3, eq_constraints_multi)
            # Check both constraints
            p = projected[0, 0]
            self.assertAlmostEqual((p[0] + p[1] + p[2]).item(), 1.0, places=5)
            self.assertAlmostEqual((p[0] - p[1]).item(), 0.0, places=5)

            # Test 4: Batch of q-points
            X_batch = torch.tensor(
                [[[0.6, 0.6], [0.8, 0.8]]], dtype=dtype, device=self.device
            )
            projected = project_to_equality_constraints(X_batch, eq_constraints)
            for j in range(2):
                self.assertAlmostEqual(
                    (projected[0, j, 0] + projected[0, j, 1]).item(),
                    1.0,
                    places=6,
                )

    def test_project_to_equality_constraints_skips_inter_point(self):
        for dtype in (torch.float, torch.double):
            inter_constraint = (
                torch.tensor([[0, 0], [1, 1]], device=self.device),
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                0.0,
            )
            intra_constraint = (
                torch.tensor([0, 1], device=self.device),
                torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                1.0,
            )

            # Only inter-point constraints: X should be returned unchanged.
            X = torch.tensor([[[0.6, 0.6]]], dtype=dtype, device=self.device)
            result = project_to_equality_constraints(X, [inter_constraint])
            self.assertAllClose(result, X)

            # Mixed: inter-point constraint should be skipped, intra applied.
            projected = project_to_equality_constraints(
                X, [intra_constraint, inter_constraint]
            )
            self.assertAlmostEqual(
                (projected[0, 0, 0] + projected[0, 0, 1]).item(), 1.0, places=5
            )

    def test_project_to_equality_constraints_empty(self):
        X = torch.tensor([[[0.5, 0.5]]], device=self.device)
        # Empty list should return X unchanged
        result = project_to_equality_constraints(X, [])
        self.assertAllClose(result, X)


class TestProjectToFeasibleSpace(BotorchTestCase):
    def test_project_to_feasible_space_via_slsqp(self) -> None:
        """Test projecting points to feasible space via SLSQP optimization."""
        for dtype in (torch.float, torch.double):
            # Define bounds for a 3D space
            bounds = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=dtype, device=self.device
            )

            # Test case 1: No constraints
            X = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(X=X, bounds=bounds)
            self.assertAllClose(projected, X)

            # Test case 2: With inequality constraints
            # Constraint: x[0] + x[1] >= 1.5
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]

            # Point satisfies constraint
            X = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, inequality_constraints=inequality_constraints
            )

            self.assertAllClose(projected, X)

            # Point violates constraint
            X = torch.tensor([[0.5, 0.5, 1.0]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, inequality_constraints=inequality_constraints
            )
            # Verify constraint is satisfied: x[0] + x[1] >= 1.5
            self.assertGreaterEqual(projected[0, 0] + projected[0, 1], 1.5 - 1e-6)
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case 3: With equality constraints
            # Constraint: x[0] + x[1] = 1.5
            equality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]

            X = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, equality_constraints=equality_constraints
            )
            # Verify constraint is satisfied: x[0] + x[1] = 1.5
            self.assertAllClose(
                (projected[0, 0] + projected[0, 1]).item(), 1.5, atol=1e-6
            )
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case 4: Combined inequality and equality constraints
            # Inequality: x[2] >= 0.5
            # Equality: x[0] + x[1] = 2.0
            inequality_constraints = [
                (
                    torch.tensor([2], dtype=torch.long, device=self.device),
                    torch.tensor([1.0], dtype=dtype, device=self.device),
                    0.5,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    2.0,
                )
            ]

            X = torch.tensor([[0.8, 0.8, 0.2]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )
            # Verify constraints
            self.assertAllClose(
                (projected[0, 0] + projected[0, 1]).item(), 2.0, atol=1e-6
            )
            self.assertGreaterEqual(projected[0, 2], 0.5 - 1e-6)
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case 5: Batch processing
            X = torch.tensor(
                [[1.0, 1.0, 1.0], [0.5, 0.5, 0.1], [2.0, 1.8, 1.9]],
                dtype=dtype,
                device=self.device,
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )

            # Check that all batch elements satisfy constraints
            for i in range(3):
                self.assertAllClose(
                    (projected[i, 0] + projected[i, 1]).item(), 2.0, atol=1e-6
                )
                self.assertGreaterEqual(projected[i, 2], 0.5 - 1e-6)
                # Check bounds
                self.assertTrue(torch.all(projected[i] >= bounds[0] - 1e-6))
                self.assertTrue(torch.all(projected[i] <= bounds[1] + 1e-6))

            # Test case 6: Multi-dimensional batch
            X = torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]],
                    [[0.5, 0.5, 0.1], [2.0, 1.8, 1.9]],
                ],
                dtype=dtype,
                device=self.device,
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )
            self.assertEqual(projected.shape, X.shape)
            self.assertTrue(torch.all(projected >= bounds[0] - 1e-6))
            self.assertTrue(torch.all(projected <= bounds[1] + 1e-6))
            projected_2d = projected.view(-1, 3)
            for i in range(4):
                self.assertAllClose(
                    (projected_2d[i, 0] + projected_2d[i, 1]).item(), 2.0, atol=1e-6
                )
                self.assertGreaterEqual(projected_2d[i, 2], 0.5 - 1e-6)
                # Check bounds
                self.assertTrue(torch.all(projected_2d[i] >= bounds[0] - 1e-6))
                self.assertTrue(torch.all(projected_2d[i] <= bounds[1] + 1e-6))

    def test_project_to_feasible_space_via_slsqp_inter_point_constraints(
        self,
    ) -> None:
        """Test projecting points with inter-point inequality constraints."""
        for dtype in (torch.float, torch.double):
            tol = get_constraint_tolerance(dtype=dtype)
            # Define bounds for a 2D space with q=2 points
            bounds = torch.tensor(
                [[0.0, 0.0], [2.0, 2.0]], dtype=dtype, device=self.device
            )

            # Test case: Inter-point inequality constraint
            # Constraint: x[0, 0] - x[1, 0] >= 0.5 (first point's x0 >= second
            # point's x0 + 0.5)
            inequality_constraints = [
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], dtype=torch.long, device=self.device
                    ),
                    torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                    0.5,
                )
            ]

            # Case 1: Point satisfies inter-point constraint
            X = torch.tensor(
                [[[1.5, 1.0], [0.8, 1.0]]], dtype=dtype, device=self.device
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, inequality_constraints=inequality_constraints
            )
            # Should remain unchanged since constraint is satisfied: 1.5 - 0.8 =
            # 0.7 >= 0.5
            self.assertAllClose(projected, X)

            # Case 2: Point violates inter-point constraint
            X = torch.tensor(
                [[[1.0, 1.0], [0.8, 1.0]]], dtype=dtype, device=self.device
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, inequality_constraints=inequality_constraints
            )
            # Verify constraint is satisfied: x[0, 0] - x[1, 0] >= 0.5
            self.assertGreaterEqual(projected[0, 0, 0] - projected[0, 1, 0], 0.5 - tol)
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case: Inter-point equality constraint
            # Constraint: x[0, 1] + x[1, 1] = 2.0 (sum of y-coordinates equals 2.0)
            equality_constraints = [
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], dtype=torch.long, device=self.device
                    ),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    2.0,
                )
            ]

            X = torch.tensor(
                [[[1.2, 0.8], [0.6, 0.9]]], dtype=dtype, device=self.device
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X, bounds=bounds, equality_constraints=equality_constraints
            )
            # Verify constraint is satisfied: x[0, 1] + x[1, 1] = 2.0
            self.assertAllClose((projected[0, 0, 1] + projected[0, 1, 1]).item(), 2.0)
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case: Combined inter-point constraints
            # Inequality: x[0, 0] - x[1, 0] >= 0.3
            # Equality: x[0, 1] + x[1, 1] = 1.8
            inequality_constraints = [
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], dtype=torch.long, device=self.device
                    ),
                    torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                    0.3,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], dtype=torch.long, device=self.device
                    ),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.8,
                )
            ]

            X = torch.tensor(
                [[[0.9, 1.2], [0.8, 1.1]]], dtype=dtype, device=self.device
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )
            # Verify both constraints
            self.assertGreaterEqual(projected[0, 0, 0] - projected[0, 1, 0], 0.3 - tol)
            self.assertAllClose((projected[0, 0, 1] + projected[0, 1, 1]).item(), 1.8)
            self.assertTrue((bounds[0] <= projected).all())
            self.assertTrue((bounds[1] >= projected).all())

            # Test case: Batch processing with inter-point constraints
            X = torch.tensor(
                [
                    [[1.0, 0.8], [0.8, 0.9]],  # batch 1
                    [[0.5, 1.2], [0.4, 1.1]],  # batch 2
                ],
                dtype=dtype,
                device=self.device,
            )
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )

            # Check that all batch elements satisfy constraints
            for i in range(2):
                self.assertGreaterEqual(
                    projected[i, 0, 0] - projected[i, 1, 0], 0.3 - tol
                )
                self.assertAllClose(
                    (projected[i, 0, 1] + projected[i, 1, 1]).item(), 1.8
                )
                # Check bounds
                self.assertTrue(torch.all(projected[i] >= bounds[0]))
                self.assertTrue(torch.all(projected[i] <= bounds[1]))

    def test_project_to_feasible_space_via_slsqp_nonlinear(self) -> None:
        """Project slightly infeasible points onto nonlinear inequality constraints."""
        for dtype in (torch.float, torch.double):
            tol = get_constraint_tolerance(dtype=dtype)
            bounds = torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=self.device
            )

            def disk(x):
                return 1 - x.pow(2).sum(dim=-1)

            def half_plane(x):
                return x[..., 0] + x[..., 1] - 0.5

            nonlinear_inequality_constraints = [(disk, True), (half_plane, True)]

            # Outside the unit disk but within box bounds.
            X = torch.tensor([[0.85, 0.85]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            )
            for nlc, is_intrapoint in nonlinear_inequality_constraints:
                self.assertTrue(
                    nonlinear_constraint_is_feasible(
                        nlc,
                        is_intrapoint=is_intrapoint,
                        x=projected.unsqueeze(0),
                        tolerance=tol,
                    ).all()
                )
            self.assertTrue(torch.all(projected >= bounds[0] - tol))
            self.assertTrue(torch.all(projected <= bounds[1] + tol))

            # Nonlinear constraints together with linear inequalities.
            inequality_constraints = [
                (
                    torch.tensor([0], dtype=torch.long, device=self.device),
                    torch.tensor([1.0], dtype=dtype, device=self.device),
                    0.2,
                )
            ]
            X = torch.tensor([[0.85, 0.85]], dtype=dtype, device=self.device)
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                nonlinear_inequality_constraints=[(disk, True)],
            )
            self.assertGreaterEqual(projected[0, 0].item(), 0.2 - tol)
            self.assertTrue(
                nonlinear_constraint_is_feasible(
                    disk, is_intrapoint=True, x=projected.unsqueeze(0), tolerance=tol
                ).all()
            )

    @mock.patch(
        "botorch.optim.parameter_constraints.minimize",
        return_value=mock.Mock(success=False, message="failed reason"),
    )
    def test_project_to_feasible_space_via_slsqp_exception(self, _: mock.Mock) -> None:
        bounds = torch.tensor([[0.0, 0.0], [2.0, 2.0]], device=self.device)

        X = torch.tensor([[1.0, 1.0]], device=self.device)
        with self.assertRaisesRegex(
            CandidateGenerationError, "Optimization failed: failed reason"
        ):
            project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                equality_constraints=[
                    (
                        torch.tensor([0, 1], dtype=torch.long, device=self.device),
                        torch.tensor([1.0, 1.0], device=self.device),
                        2.0,
                    )
                ],
            )

    def test_project_to_feasible_space_with_scalar_fixed_features(self) -> None:
        """Test projection preserves scalar fixed_features values."""
        for dtype in (torch.float, torch.double):
            tol = get_constraint_tolerance(dtype=dtype)
            # Setup: 3D search space, bounds [[0, 0, 0], [2, 2, 2]]
            bounds = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=dtype, device=self.device
            )
            # Constraint: x[0] + x[1] >= 1.5
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]
            # Infeasible point X = [[0.3, 0.3, 1.0]] (0.6 < 1.5)
            X = torch.tensor([[0.3, 0.3, 1.0]], dtype=dtype, device=self.device)
            # fixed_features = {0: 0.3} (scalar)
            fixed_features: dict[int, float | torch.Tensor] = {0: 0.3}
            # Execute: project to feasible space with fixed_features
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
            )
            # Assert: x[0] remains at 0.3 (fixed)
            self.assertAllClose(
                projected[0, 0], torch.tensor(0.3, dtype=dtype, device=self.device)
            )
            # Assert: constraint is satisfied (x[0] + x[1] >= 1.5)
            self.assertGreaterEqual(
                (projected[0, 0] + projected[0, 1]).item(), 1.5 - tol
            )
            # Assert: bounds are respected
            self.assertTrue(torch.all(projected >= bounds[0] - tol))
            self.assertTrue(torch.all(projected <= bounds[1] + tol))

    def test_project_to_feasible_space_with_batched_fixed_features(self) -> None:
        """Test projection preserves batched (tensor) fixed_features values."""
        for dtype in (torch.float, torch.double):
            tol = get_constraint_tolerance(dtype=dtype)
            # Setup: 3D search space, bounds [[0, 0, 0], [2, 2, 2]]
            bounds = torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=dtype, device=self.device
            )
            # Constraint: x[0] + x[1] >= 1.5
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]
            # Batch of 3 infeasible points (all violate x[0] + x[1] >= 1.5)
            # X must be 3D: batch x q x d when using tensor fixed_features
            X = torch.tensor(
                [
                    [[0.2, 0.3, 1.0]],  # batch 0, q=1
                    [[0.4, 0.5, 0.5]],  # batch 1, q=1
                    [[0.1, 0.2, 1.5]],  # batch 2, q=1
                ],
                dtype=dtype,
                device=self.device,
            )  # Shape: [3, 1, 3]
            # fixed_features = {0: tensor([0.2, 0.4, 0.1])} (different per batch)
            fixed_values = torch.tensor(
                [0.2, 0.4, 0.1], dtype=dtype, device=self.device
            )
            fixed_features: dict[int, float | torch.Tensor] = {0: fixed_values}
            # Execute: project to feasible space with batched fixed_features
            projected = project_to_feasible_space_via_slsqp(
                X=X,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
            )
            # Assert: each batch element preserves its respective fixed value for x[0]
            self.assertAllClose(projected[:, 0, 0], fixed_values)
            # Assert: constraint is satisfied for each batch element
            for i in range(3):
                self.assertGreaterEqual(
                    (projected[i, 0, 0] + projected[i, 0, 1]).item(), 1.5 - tol
                )
            # Assert: bounds are respected
            self.assertTrue(torch.all(projected >= bounds[0] - tol))
            self.assertTrue(torch.all(projected <= bounds[1] + tol))
