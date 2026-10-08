#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.exceptions.errors import InputDataError, UnsupportedError
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.multi_objective import (
    BNH,
    BraninCurrin,
    C2DTLZ2,
    CarSideImpact,
    CONSTR,
    ConstrainedBraninCurrin,
    DH1,
    DH2,
    DH3,
    DH4,
    DiscBrake,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ7,
    GMM,
    MultiObjectiveTestProblem,
    MW7,
    OSY,
    Penicillin,
    SRN,
    ToyRobust,
    VehicleSafety,
    WeldedBeam,
    ZDT1,
    ZDT2,
    ZDT3,
)
from botorch.utils.testing import (
    BaseTestProblemTestCaseMixIn,
    BotorchTestCase,
    ConstrainedTestProblemTestCaseMixin,
    MultiObjectiveTestProblemTestCaseMixin,
)


def _hypervolume_2d(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    r"""Exact hypervolume of the 2-d outcomes ``Y`` (minimization) w.r.t. the
    reference point, computed with a sweep over the first objective."""
    Y = Y[(Y < ref_point).all(dim=-1)]
    Y = Y[Y[:, 0].argsort()]
    f_1 = Y[:, 1].cummin(dim=0).values
    widths = torch.cat([Y[1:, 0], ref_point[:1]]) - Y[:, 0]
    return (widths * (ref_point[1] - f_1)).sum().item()


class DummyMOProblem(MultiObjectiveTestProblem):
    _ref_point = [0.0, 0.0]
    _num_objectives = 2
    _bounds = [(0.0, 1.0)] * 2
    dim = 2
    continuous_inds = list(range(dim))

    def _evaluate_true(self, X):
        f_X = X + 2
        return -f_X if self.negate else f_X


class TestBaseTestMultiObjectiveProblem(BotorchTestCase):
    def test_base_mo_problem(self):
        for negate in (True, False):
            for noise_std in (None, 1.0, [1.0, 2.0]):
                f = DummyMOProblem(noise_std=noise_std, negate=negate)
                self.assertEqual(f.noise_std, noise_std)
                self.assertEqual(f.negate, negate)
                for dtype in (torch.float, torch.double):
                    f.to(dtype=dtype, device=self.device)
                    X = torch.rand(3, 2, dtype=dtype, device=self.device)
                    f_X = f.evaluate_true(X)
                    expected_f_X = -(X + 2) if negate else X + 2
                    self.assertTrue(torch.equal(f_X, expected_f_X))
                with self.assertRaises(NotImplementedError):
                    f.gen_pareto_front(1)
            with self.assertRaisesRegex(
                InputDataError, "must match the number of objectives"
            ):
                f = DummyMOProblem(noise_std=[1.0, 2.0, 3.0], negate=negate)
            with self.assertRaisesRegex(UnsupportedError, "is_minimization_problem"):
                f.is_minimization_problem


class TestBraninCurrin(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [BraninCurrin()]

    def test_init(self):
        for f in self.functions:
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 2)


class TestDH(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.dims = [2, 3, 4, 5]
        self.bounds = [
            [[0.0, -1], [1, 1]],
            [[0.0, -1, -1], [1, 1, 1]],
            [[0.0, 0, -1, -1], [1, 1, 1, 1]],
            [[0.0, -0.15, -1, -1, -1], [1, 1, 1, 1, 1]],
        ]
        self.expected = [
            [[0.0, 1.0], [1.0, 1.0 / 1.2 + 1.0]],
            [[0.0, 1.0], [1.0, 2.0 / 1.2 + 20.0]],
            [[0.0, 1.88731], [1.0, 1.9990726 * 100]],
            [[0.0, 1.88731], [1.0, 150.0]],
        ]

    @property
    def functions(self) -> list[BaseTestProblem]:
        return [DH1(dim=2), DH2(dim=3), DH3(dim=4), DH4(dim=5)]

    def test_init(self):
        for i, f in enumerate(self.functions):
            with self.assertRaises(ValueError):
                f.__class__(dim=1)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, self.dims[i])
            self.assertTrue(
                torch.equal(
                    f.bounds,
                    torch.tensor(
                        self.bounds[i], dtype=f.bounds.dtype, device=f.bounds.device
                    ),
                )
            )

    def test_function_values(self):
        for i, f in enumerate(self.functions):
            test_X = torch.zeros(2, self.dims[i], device=self.device)
            test_X[1] = 1.0
            f = f.to(device=self.device)
            actual = f(test_X)
            expected = torch.tensor(self.expected[i], device=self.device)
            self.assertAllClose(actual, expected)

    def test_dh4_max_hv(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        for dim in (3, 4, 5):
            f = DH4(dim=dim).to(**tkwargs)
            # Points on the Pareto set: h(x) is minimized at x_0 + x_1 = 0.849894,
            # and g(x) = 0 where h(x) >= 0, while g(x) is maximized where h(x) < 0
            # (i.e., for x_0 > 0.985335), which leads to negative values of f_1.
            x_0 = torch.cat(
                [
                    torch.linspace(0, 0.98, 100001, **tkwargs),
                    torch.linspace(0.98, 1, 20001, **tkwargs),
                ]
            )
            x_1 = (0.849894 - x_0).clamp(-0.15, 1)
            x_rest = (x_0 > 0.985335).to(x_0).unsqueeze(-1).expand(-1, dim - 2)
            Y = f.evaluate_true(torch.cat([x_0[:, None], x_1[:, None], x_rest], -1))
            self.assertLess(Y[:, 1].min().item(), -0.7 * (dim - 2))
            hv = _hypervolume_2d(Y, f.ref_point)
            self.assertLessEqual(hv, f.max_hv)
            self.assertGreater(hv, f.max_hv - 1e-5)


class TestDTLZ(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            DTLZ1(dim=5, num_objectives=2),
            DTLZ2(dim=5, num_objectives=2),
            DTLZ3(dim=5, num_objectives=2),
            DTLZ4(dim=5, num_objectives=2),
            DTLZ5(dim=5, num_objectives=2),
            DTLZ7(dim=5, num_objectives=2),
            DTLZ7(dim=5, num_objectives=2, noise_std=[0.1, 0.2]),
        ]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(ValueError):
                f.__class__(dim=1, num_objectives=2)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 5)
            self.assertEqual(f.k, 4)

    def test_gen_pareto_front(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                for negate in (True, False):
                    f.negate = negate
                    f = f.to(dtype=dtype, device=self.device)
                    if isinstance(f, (DTLZ5, DTLZ7)):
                        with self.assertRaises(NotImplementedError):
                            f.gen_pareto_front(n=1)
                    else:
                        pareto_f = f.gen_pareto_front(n=10)
                        if negate:
                            pareto_f *= -1
                        self.assertEqual(pareto_f.dtype, dtype)
                        self.assertEqual(pareto_f.device.type, self.device.type)
                        self.assertTrue((pareto_f > 0).all())
                        if isinstance(f, DTLZ1):
                            # assert is the hyperplane sum_i (f(x_i)) = 0.5
                            self.assertTrue(
                                torch.allclose(
                                    pareto_f.sum(dim=-1),
                                    torch.full(
                                        pareto_f.shape[0:1],
                                        0.5,
                                        dtype=dtype,
                                        device=self.device,
                                    ),
                                )
                            )
                        elif isinstance(f, (DTLZ2, DTLZ3, DTLZ4)):
                            # assert the points lie on the surface
                            # of the unit hypersphere
                            self.assertTrue(
                                torch.allclose(
                                    pareto_f.pow(2).sum(dim=-1),
                                    torch.ones(
                                        pareto_f.shape[0],
                                        dtype=dtype,
                                        device=self.device,
                                    ),
                                )
                            )

    def test_dtlz1_max_hv(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        f = DTLZ1(dim=5, num_objectives=2).to(**tkwargs)
        # The Pareto front is the segment f_0 + f_1 = 0.5 with f >= 0.
        self.assertEqual(f.max_hv, 400.0**2 - 0.5**2 / 2)
        self.assertEqual(DTLZ1(dim=5, num_objectives=3).max_hv, 400.0**3 - 0.5**3 / 6)
        x_0 = torch.linspace(0, 1, 10001, **tkwargs)
        X = torch.cat([x_0.unsqueeze(-1), torch.full((10001, 4), 0.5, **tkwargs)], -1)
        hv = _hypervolume_2d(f.evaluate_true(X), f.ref_point)
        self.assertLessEqual(hv, f.max_hv)
        self.assertGreater(hv, f.max_hv - 1e-4)

    def test_dtlz4(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # DTLZ4 is DTLZ2 with the position variables x_i mapped to x_i^100.
        X = torch.tensor([0.99, 0.3, 0.5, 0.5, 0.5], **tkwargs)
        theta = 0.99**100 * math.pi / 2
        g = (0.3 - 0.5) ** 2
        expected = torch.tensor([math.cos(theta), math.sin(theta)], **tkwargs)
        f = DTLZ4(dim=5).to(**tkwargs)
        self.assertAllClose(f.evaluate_true(X), (1 + g) * expected)
        for M in (2, 3):
            X = torch.rand(10, 6, **tkwargs)
            X_pow = torch.cat([X[:, : M - 1].pow(100), X[:, M - 1 :]], dim=-1)
            self.assertAllClose(
                DTLZ4(dim=6, num_objectives=M).to(**tkwargs).evaluate_true(X),
                DTLZ2(dim=6, num_objectives=M).to(**tkwargs).evaluate_true(X_pow),
            )


class TestGMM(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            GMM(num_objectives=4),
            GMM(num_objectives=4, noise_std=[0.0, 0.1, 0.2, 0.3]),
        ]

    def test_init(self):
        f = self.functions[0]
        with self.assertRaises(UnsupportedError):
            f.__class__(num_objectives=5)
        self.assertEqual(f.num_objectives, 4)
        self.assertEqual(f.dim, 2)

    def test_result(self):
        x = torch.tensor(
            [
                [[0.0342, 0.8055], [0.7844, 0.4831]],
                [[0.5236, 0.3158], [0.0992, 0.9873]],
                [[0.4693, 0.5792], [0.5357, 0.9451]],
            ],
            device=self.device,
        )
        expected_f_x = -torch.tensor(
            [
                [
                    [3.6357e-03, 5.9030e-03, 5.8958e-03, 1.0309e-04],
                    [1.6304e-02, 3.1430e-04, 4.7323e-04, 2.0691e-04],
                ],
                [
                    [1.2251e-01, 3.2309e-02, 3.7199e-02, 5.4211e-03],
                    [1.9378e-04, 1.5290e-03, 3.5051e-04, 3.6924e-07],
                ],
                [
                    [3.5550e-01, 5.9409e-02, 1.7352e-01, 8.5574e-02],
                    [3.2686e-02, 9.7298e-02, 7.2311e-02, 1.5613e-03],
                ],
            ],
            device=self.device,
        )
        f = self.functions[0]
        f.to(device=self.device)
        for dtype in (torch.float, torch.double):
            f.to(dtype=dtype)
            f_x = f(x.to(dtype=dtype))
            self.assertTrue(
                torch.allclose(f_x, expected_f_x.to(dtype=dtype), rtol=1e-4, atol=1e-4)
            )


class TestMW7(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            MW7(dim=3),
            MW7(dim=3, noise_std=[0.1, 0.2]),
            MW7(dim=3, constraint_noise_std=[0.05, 0.025]),
        ]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(ValueError):
                f.__class__(dim=1)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 3)


class TestZDT(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            ZDT1(dim=3, num_objectives=2),
            ZDT2(dim=3, num_objectives=2),
            ZDT3(dim=3, num_objectives=2),
            ZDT3(dim=3, num_objectives=2, noise_std=0.1),
            ZDT3(dim=3, num_objectives=2, noise_std=[0.1, 0.2]),
        ]

    def test_init(self):
        for f in self.functions:
            with self.assertRaises(NotImplementedError):
                f.__class__(dim=3, num_objectives=3)
            with self.assertRaises(NotImplementedError):
                f.__class__(dim=3, num_objectives=1)
            with self.assertRaises(ValueError):
                f.__class__(dim=1, num_objectives=2)
            self.assertEqual(f.num_objectives, 2)
            self.assertEqual(f.dim, 3)

    def test_gen_pareto_front(self):
        for dtype in (torch.float, torch.double):
            for f in self.functions:
                for negate in (True, False):
                    f.negate = negate
                    f = f.to(dtype=dtype, device=self.device)
                    pareto_f = f.gen_pareto_front(n=11)
                    if negate:
                        pareto_f *= -1
                    self.assertEqual(pareto_f.dtype, dtype)
                    self.assertEqual(pareto_f.device.type, self.device.type)
                    if isinstance(f, ZDT1):
                        self.assertTrue(
                            torch.equal(pareto_f[:, 1], 1 - pareto_f[:, 0].sqrt())
                        )
                    elif isinstance(f, ZDT2):
                        self.assertTrue(
                            torch.equal(pareto_f[:, 1], 1 - pareto_f[:, 0].pow(2))
                        )
                    elif isinstance(f, ZDT3):
                        f_0 = pareto_f[:, 0]
                        f_1 = pareto_f[:, 1]
                        # check f_0 is in the expected discontinuous part of the pareto
                        # front
                        self.assertTrue(
                            (
                                (f_0[:3] >= f._parts[0][0])
                                & (f_0[:3] <= f._parts[0][1])
                            ).all()
                        )
                        for i in range(0, 4):
                            f_0_i = f_0[3 + 2 * i : 3 + 2 * (i + 1)]
                            comparison = f_0_i > torch.tensor(
                                f._parts[i + 1], dtype=dtype, device=self.device
                            )
                            self.assertTrue((comparison[..., 0]).all())
                            self.assertTrue((~comparison[..., 1]).all())
                            self.assertTrue(
                                ((comparison[..., 0]) & (~comparison[..., 1])).all()
                            )
                        # check f_1
                        self.assertTrue(
                            torch.equal(
                                f_1,
                                1 - f_0.sqrt() - f_0 * torch.sin(10 * math.pi * f_0),
                            )
                        )


# ------------------ Unconstrained Multi-objective test problems ------------------ #


class TestCarSideImpact(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [CarSideImpact(), CarSideImpact(noise_std=[0.1, 0.2, 0.3, 0.4])]


class TestPenicillin(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [Penicillin(), Penicillin(noise_std=[0.1, 0.2, 0.3])]


class TestToyRobust(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [ToyRobust(), ToyRobust(noise_std=[0.1, 0.2])]


class TestVehicleSafety(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [VehicleSafety(), VehicleSafety(noise_std=[0.1, 0.2, 0.3])]


# ------------------ Constrained Multi-objective test problems ------------------ #


class TestBNH(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [BNH(), BNH(noise_std=[0.1, 0.2])]


class TestSRN(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [SRN(), SRN(noise_std=[0.1, 0.2])]

    def test_function_values(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        f = SRN().to(**tkwargs)
        # f_0 = 2 + (x_0 - 2)^2 + (x_1 - 1)^2, f_1 = 9 x_0 - (x_1 - 1)^2
        X = torch.tensor([1.0, 3.0], **tkwargs)
        self.assertAllClose(f.evaluate_true(X), torch.tensor([7.0, 5.0], **tkwargs))
        # c_0 = 225 - x_0^2 - x_1^2, c_1 = 3 x_1 - x_0 - 10
        X = torch.tensor([-2.5, 10.0], **tkwargs)
        self.assertAllClose(
            f.evaluate_slack_true(X), torch.tensor([118.75, 22.5], **tkwargs)
        )
        # The Pareto set is x_0 = -2.5, x_1 in [2.5, 14.79].
        X = torch.stack(
            [
                torch.full((11,), -2.5, **tkwargs),
                torch.linspace(2.5, 14.79, 11, **tkwargs),
            ],
            dim=-1,
        )
        self.assertTrue(f.is_feasible(X, noise=False).all())
        X = torch.tensor([-2.5, 15.0], **tkwargs)
        self.assertFalse(f.is_feasible(X, noise=False).item())


class TestCONSTR(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [CONSTR(), CONSTR(noise_std=[0.1, 0.2])]


class TestConstrainedBraninCurrin(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            ConstrainedBraninCurrin(),
            ConstrainedBraninCurrin(noise_std=[0.1, 0.2]),
            ConstrainedBraninCurrin(constraint_noise_std=0.1),
        ]


class TestC2DTLZ2(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [
            C2DTLZ2(dim=3, num_objectives=2),
            C2DTLZ2(dim=3, num_objectives=2, noise_std=0.1),
            C2DTLZ2(dim=3, num_objectives=2, noise_std=[0.1, 0.2]),
        ]

    def test_batch_shapes(self):
        f = C2DTLZ2(dim=3, num_objectives=2).to(device=self.device)
        X = torch.rand(2, 4, 3, device=self.device, dtype=torch.double)
        slack = f.evaluate_slack_true(X)
        self.assertEqual(slack.shape, torch.Size([2, 4, 1]))
        self.assertAllClose(slack[1, 2], f.evaluate_slack_true(X[1, 2]))
        self.assertEqual(f.evaluate_slack_true(X[0, 0]).shape, torch.Size([1]))

    def test_constraint(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        for M, r in ((2, 0.2), (3, 0.4), (4, 0.5)):
            f = C2DTLZ2(dim=M + 2, num_objectives=M).to(**tkwargs)
            self.assertEqual(f._r, r)
            X = torch.rand(100, M + 2, **tkwargs)
            F = f.evaluate_true(X)
            # feasible iff F is within distance r of e_i or of (1, ..., 1) / sqrt(M)
            centers = torch.cat(
                [torch.eye(M, **tkwargs), torch.full((1, M), M**-0.5, **tkwargs)]
            )
            min_dist = torch.cdist(F, centers).min(dim=-1).values
            self.assertAllClose(
                f.evaluate_slack_true(X).squeeze(-1), r**2 - min_dist.pow(2)
            )
        # A point on the Pareto front at distance ~0.25 > r from the center.
        f = C2DTLZ2(dim=3, num_objectives=2).to(**tkwargs)
        X = torch.tensor([0.5 + 0.5 / math.pi, 0.5, 0.5], **tkwargs)
        self.assertFalse(f.is_feasible(X, noise=False).item())

    def test_max_hv(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        f = C2DTLZ2(dim=3, num_objectives=2).to(**tkwargs)
        # feasible part of the Pareto front (x_1 = x_2 = 0.5)
        x_0 = torch.linspace(0, 1, 100001, **tkwargs)
        X = torch.cat([x_0.unsqueeze(-1), torch.full((100001, 2), 0.5, **tkwargs)], -1)
        X = X[f.is_feasible(X, noise=False)]
        hv = _hypervolume_2d(f.evaluate_true(X), f.ref_point)
        self.assertLessEqual(hv, f.max_hv)
        self.assertGreater(hv, f.max_hv - 1e-5)
        with self.assertRaises(NotImplementedError):
            C2DTLZ2(dim=4, num_objectives=3).max_hv


class TestDiscBrake(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [DiscBrake(), DiscBrake(noise_std=[0.1, 0.2])]


class TestWeldedBeam(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [WeldedBeam(), WeldedBeam(noise_std=[0.1, 0.2])]

    def test_feasibility(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        f = WeldedBeam().to(**tkwargs)
        # The minimum-cost design of [Deb1991] is feasible, while the smallest beam
        # violates the shear stress, bending stress and buckling constraints.
        X = torch.tensor([[0.2444, 6.2187, 8.2915, 0.2444], [1.0, 1.0, 10.0, 1.0]])
        self.assertTrue(f.is_feasible(X.to(**tkwargs), noise=False).all())
        slack = f.evaluate_slack_true(f.bounds[0])
        self.assertTrue((slack[[0, 1, 3]] < 0).all())
        self.assertFalse(f.is_feasible(f.bounds[0], noise=False).item())


class TestOSY(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    MultiObjectiveTestProblemTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    @property
    def functions(self) -> list[BaseTestProblem]:
        return [OSY(), OSY(noise_std=[0.1, 0.2])]
