#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    eval_lin_constraint,
    evaluate_feasibility,
    get_constraint_tolerance,
    lin_constraint_jac,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
    nonlinear_constraint_is_feasible,
    project_to_feasible_space_via_slsqp,
)
from botorch.utils.testing import BotorchTestCase
from scipy.optimize import Bounds


class TestParameterConstraints(BotorchTestCase):
    def test_arrayify(self):
        for dtype in (torch.float, torch.double, torch.int, torch.long):
            t = torch.tensor([[1, 2], [3, 4]], device=self.device).type(dtype)
            t_np = _arrayify(t)
            self.assertIsInstance(t_np, np.ndarray)
            self.assertTrue(t_np.dtype == np.float64)

    def test_eval_lin_constraint(self):
        res = eval_lin_constraint(
            flat_idxr=[0, 2],
            coeffs=np.array([1.0, -2.0]),
            rhs=0.5,
            x=np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(res, -5.5)

    def test_lin_constraint_jac(self):
        dummy_array = np.array([1.0])
        res = lin_constraint_jac(
            dummy_array, flat_idxr=[0, 2], coeffs=np.array([1.0, -2.0]), n=3
        )
        self.assertTrue(all(np.equal(res, np.array([1.0, 0.0, -2.0]))))

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
        # equality constraints, 1d indices
        indices = torch.tensor([1, 2], dtype=torch.long, device=self.device)
        for dtype, shapeX in product(
            (torch.float, torch.double), (torch.Size([3, 2, 4]), torch.Size([2, 4]))
        ):
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=self.device)
            constraints = _make_linear_constraints(
                indices=indices,
                coefficients=coefficients,
                rhs=1.0,
                shapeX=shapeX,
                eq=True,
            )
            self.assertTrue(
                all(set(c.keys()) == {"fun", "jac", "type"} for c in constraints)
            )
            self.assertTrue(all(c["type"] == "eq" for c in constraints))
            self.assertEqual(len(constraints), shapeX[:-1].numel())
            x = np.random.rand(shapeX.numel())
            self.assertEqual(constraints[0]["fun"](x), x[1] + 2 * x[2] - 1.0)
            jac_exp = np.zeros(shapeX.numel())
            jac_exp[[1, 2]] = [1, 2]
            self.assertTrue(np.allclose(constraints[0]["jac"](x), jac_exp))
            self.assertEqual(constraints[-1]["fun"](x), x[-3] + 2 * x[-2] - 1.0)
            jac_exp = np.zeros(shapeX.numel())
            jac_exp[[-3, -2]] = [1, 2]
            self.assertTrue(np.allclose(constraints[-1]["jac"](x), jac_exp))

        # inequality constraints, 1d indices
        for shapeX in [torch.Size([1, 1, 2]), torch.Size([1, 2])]:
            lcs = _make_linear_constraints(
                indices=torch.tensor([1]),
                coefficients=torch.tensor([1.0]),
                rhs=1.0,
                shapeX=shapeX,
                eq=False,
            )
            self.assertEqual(len(lcs), 1)
            self.assertEqual(lcs[0]["type"], "ineq")

        # constraint across q-batch (2d indics), equality constraint
        indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.long, device=self.device)

        for dtype, shapeX in product(
            (torch.float, torch.double), (torch.Size([3, 2, 4]), torch.Size([2, 4]))
        ):
            q, d = shapeX[-2:]
            b = 1 if len(shapeX) == 2 else shapeX[0]
            coefficients = torch.tensor([1.0, 2.0], dtype=dtype, device=self.device)
            constraints = _make_linear_constraints(
                indices=indices,
                coefficients=coefficients,
                rhs=1.0,
                shapeX=shapeX,
                eq=True,
            )
            self.assertTrue(
                all(set(c.keys()) == {"fun", "jac", "type"} for c in constraints)
            )
            self.assertTrue(all(c["type"] == "eq" for c in constraints))
            self.assertEqual(len(constraints), b)
            x = np.random.rand(shapeX.numel())
            offsets = [q * d, d]
            # rule is [i, j, k] is i * offset[0] + j * offset[1] + k
            for i in range(b):
                pos1 = i * offsets[0] + 3
                pos2 = i * offsets[0] + 1 * offsets[1] + 2
                self.assertEqual(constraints[i]["fun"](x), x[pos1] + 2 * x[pos2] - 1.0)
                jac_exp = np.zeros(shapeX.numel())
                jac_exp[[pos1, pos2]] = [1, 2]
                self.assertTrue(np.allclose(constraints[i]["jac"](x), jac_exp))
        # make sure error is raised for scalar tensors
        with self.assertRaises(ValueError):
            constraints = _make_linear_constraints(
                indices=torch.tensor(0),
                coefficients=torch.tensor([1.0]),
                rhs=1.0,
                shapeX=torch.Size([1, 1, 2]),
                eq=False,
            )
        # test that len(shapeX) < 2 raises an error
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
            res = make_scipy_linear_constraints(
                shapeX=shapeX, inequality_constraints=None, equality_constraints=None
            )
            self.assertEqual(res, [])
            indices = torch.tensor([0, 1], dtype=torch.long, device=self.device)
            coefficients = torch.tensor([1.5, -1.0], device=self.device)
            # both inequality and equality constraints
            cs = make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
            self.assertEqual(len(cs), 2 * b)
            self.assertTrue({c["type"] for c in cs} == {"ineq", "eq"})
            # inequality only
            cs = make_scipy_linear_constraints(
                shapeX=shapeX, inequality_constraints=[(indices, coefficients, 1.0)]
            )
            self.assertEqual(len(cs), b)
            self.assertTrue(all(c["type"] == "ineq" for c in cs))
            # equality only
            cs = make_scipy_linear_constraints(
                shapeX=shapeX, equality_constraints=[(indices, coefficients, 1.0)]
            )
            self.assertEqual(len(cs), b)
            self.assertTrue(all(c["type"] == "eq" for c in cs))

            # test that 2-dim indices work properly
            indices = indices.unsqueeze(0)
            cs = make_scipy_linear_constraints(
                shapeX=shapeX,
                inequality_constraints=[(indices, coefficients, 1.0)],
                equality_constraints=[(indices, coefficients, 1.0)],
            )
            self.assertEqual(len(cs), 2 * b)
            self.assertTrue({c["type"] for c in cs} == {"ineq", "eq"})

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

    @mock.patch(
        "botorch.optim.parameter_constraints.minimize",
        return_value=mock.Mock(success=False, message="failed reason"),
    )
    def test_project_to_feasible_space_via_slsqp_exception(self, _: mock.Mock) -> None:
        bounds = torch.tensor([[0.0, 0.0], [2.0, 2.0]], device=self.device)

        X = torch.tensor([[1.0, 1.0]], device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Optimization failed: failed reason"):
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
