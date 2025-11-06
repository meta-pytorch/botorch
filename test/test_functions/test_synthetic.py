#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import InputDataError
from botorch.test_functions.synthetic import (
    Ackley,
    AckleyMixed,
    Beale,
    Branin,
    Bukin,
    ConstrainedGramacy,
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    ConstrainedSyntheticTestFunction,
    Cosine8,
    DixonPrice,
    DropWave,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    KeaneBumpFunction,
    Labs,
    Levy,
    Michalewicz,
    Powell,
    PressureVessel,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    SpeedReducer,
    StyblinskiTang,
    SyntheticTestFunction,
    TensionCompressionString,
    ThreeHumpCamel,
    WeldedBeamSO,
)
from botorch.utils.testing import (
    BaseTestProblemTestCaseMixIn,
    BotorchTestCase,
    ConstrainedTestProblemTestCaseMixin,
    SyntheticTestFunctionTestCaseMixin,
)
from torch import Tensor


class DummySyntheticTestFunction(SyntheticTestFunction):
    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(-1, 1), (-1, 1)]
    _optimal_value = 0

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -X.pow(2).sum(dim=-1)


class DummySyntheticTestFunctionWithOptimizers(DummySyntheticTestFunction):
    _optimizers = [(0, 0)]


class TestCustomBounds(BotorchTestCase):
    functions_with_custom_bounds = [  # Function name and the default dimension.
        (Ackley, 2),
        (Beale, 2),
        (Branin, 2),
        (Bukin, 2),
        (Cosine8, 8),
        (DropWave, 2),
        (DixonPrice, 2),
        (EggHolder, 2),
        (Griewank, 2),
        (Hartmann, 6),
        (ConstrainedHartmann, 6),
        (HolderTable, 2),
        (Levy, 2),
        (Michalewicz, 2),
        (Powell, 4),
        (Rastrigin, 2),
        (Rosenbrock, 2),
        (Shekel, 4),
        (SixHumpCamel, 2),
        (StyblinskiTang, 2),
        (ThreeHumpCamel, 2),
    ]

    def test_custom_bounds(self):
        with self.assertRaisesRegex(
            InputDataError,
            "Expected the bounds to match the dimensionality of the domain. ",
        ):
            DummySyntheticTestFunctionWithOptimizers(bounds=[(0, 0)])

        with self.assertRaisesRegex(
            ValueError, "No global optimum found within custom bounds"
        ):
            DummySyntheticTestFunctionWithOptimizers(bounds=[(1, 2), (3, 4)])

        dummy = DummySyntheticTestFunctionWithOptimizers(bounds=[(-2, 2), (-3, 3)])
        self.assertEqual(dummy._bounds[0], (-2, 2))
        self.assertEqual(dummy._bounds[1], (-3, 3))
        self.assertAllClose(
            dummy.bounds,
            torch.tensor([[-2, -3], [2, 3]], dtype=torch.double),
        )

        # Test each function with custom bounds.
        for func_class, dim in self.functions_with_custom_bounds:
            bounds = [(-1e5, 1e5) for _ in range(dim)]
            bounds_tensor = torch.tensor(bounds, dtype=torch.double).T
            func = func_class(bounds=bounds)
            self.assertEqual(func._bounds, bounds)
            self.assertAllClose(func.bounds, bounds_tensor)


class DummyConstrainedSyntheticTestFunction(ConstrainedSyntheticTestFunction):
    dim = 2
    continuous_inds = list(range(dim))
    num_constraints = 1
    _bounds = [(-1, 1), (-1, 1)]
    _optimal_value = 0

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -X.pow(2).sum(dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        return -X.norm(dim=-1, keepdim=True) + 1


class TestConstraintNoise(BotorchTestCase):
    functions = [
        DummyConstrainedSyntheticTestFunction(),
        DummyConstrainedSyntheticTestFunction(constraint_noise_std=0.1),
        DummyConstrainedSyntheticTestFunction(constraint_noise_std=[0.1]),
    ]

    def test_constraint_noise_length_validation(self):
        with self.assertRaisesRegex(
            InputDataError, "must match the number of constraints"
        ):
            DummyConstrainedSyntheticTestFunction(constraint_noise_std=[0.1, 0.2])

    def test_worst_feasible_value_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DummyConstrainedSyntheticTestFunction().worst_feasible_value


class TestAckley(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Ackley(), Ackley(negate=True), Ackley(noise_std=0.1), Ackley(dim=3)]


class TestBeale(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Beale(), Beale(negate=True), Beale(noise_std=0.1)]


class TestBranin(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Branin(), Branin(negate=True), Branin(noise_std=0.1)]


class TestBukin(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Bukin(), Bukin(negate=True), Bukin(noise_std=0.1)]


class TestCosine8(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Cosine8(), Cosine8(negate=True), Cosine8(noise_std=0.1)]


class TestDropWave(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [DropWave(), DropWave(negate=True), DropWave(noise_std=0.1)]


class TestDixonPrice(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        DixonPrice(),
        DixonPrice(negate=True),
        DixonPrice(noise_std=0.1),
        DixonPrice(dim=3),
    ]


class TestEggHolder(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [EggHolder(), EggHolder(negate=True), EggHolder(noise_std=0.1)]


class TestGriewank(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Griewank(),
        Griewank(negate=True),
        Griewank(noise_std=0.1),
        Griewank(dim=4),
    ]


class TestHartmann(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Hartmann(),
        Hartmann(negate=True),
        Hartmann(noise_std=0.1),
        Hartmann(dim=3),
        Hartmann(dim=3, negate=True),
        Hartmann(dim=3, noise_std=0.1),
        Hartmann(dim=4),
        Hartmann(dim=4, negate=True),
        Hartmann(dim=4, noise_std=0.1),
    ]

    def test_dimension(self):
        with self.assertRaises(ValueError):
            Hartmann(dim=2)


class TestHolderTable(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [HolderTable(), HolderTable(negate=True), HolderTable(noise_std=0.1)]


class TestLevy(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Levy(),
        Levy(negate=True),
        Levy(noise_std=0.1),
        Levy(dim=3),
        Levy(dim=3, negate=True),
        Levy(dim=3, noise_std=0.1),
    ]


class TestMichalewicz(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Michalewicz(),
        Michalewicz(negate=True),
        Michalewicz(noise_std=0.1),
        Michalewicz(dim=5),
        Michalewicz(dim=5, negate=True),
        Michalewicz(dim=5, noise_std=0.1),
        Michalewicz(dim=10),
        Michalewicz(dim=10, negate=True),
        Michalewicz(dim=10, noise_std=0.1),
    ]


class TestPowell(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Powell(), Powell(negate=True), Powell(noise_std=0.1)]


class TestRastrigin(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Rastrigin(),
        Rastrigin(negate=True),
        Rastrigin(noise_std=0.1),
        Rastrigin(dim=3),
        Rastrigin(dim=3, negate=True),
        Rastrigin(dim=3, noise_std=0.1),
    ]


class TestRosenbrock(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Rosenbrock(),
        Rosenbrock(negate=True),
        Rosenbrock(noise_std=0.1),
        Rosenbrock(dim=3),
        Rosenbrock(dim=3, negate=True),
        Rosenbrock(dim=3, noise_std=0.1),
    ]


class TestShekel(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [Shekel(), Shekel(negate=True), Shekel(noise_std=0.1)]


class TestSixHumpCamel(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [SixHumpCamel(), SixHumpCamel(negate=True), SixHumpCamel(noise_std=0.1)]


class TestStyblinskiTang(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        StyblinskiTang(),
        StyblinskiTang(negate=True),
        StyblinskiTang(noise_std=0.1),
        StyblinskiTang(dim=3),
        StyblinskiTang(dim=3, negate=True),
        StyblinskiTang(dim=3, noise_std=0.1),
    ]


class TestThreeHumpCamel(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        ThreeHumpCamel(),
        ThreeHumpCamel(negate=True),
        ThreeHumpCamel(noise_std=0.1),
    ]


class TestLabs(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        Labs(),
        Labs(negate=True),
        Labs(noise_std=0.1),
    ]

    def test_labs_optimizers(self):
        for dim in [10, 20, 30, 40, 50, 60]:
            labs = Labs(dim=dim)
            self.assertAllClose(
                labs.optimal_value,
                labs.evaluate_true(labs.optimizers).item(),
                atol=1e-2,
            )


class TestAckleyMixed(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        AckleyMixed(dim=5),
        AckleyMixed(dim=5, negate=True, randomize_optimum=True),
        AckleyMixed(dim=5, noise_std=0.1),
    ]

    def test_dimension(self):
        with self.assertRaisesRegex(ValueError, "Expected dim > 3. Got dim=3."):
            AckleyMixed(dim=3)


# ------------------ Constrained synthetic test problems ------------------ #


class TestConstrainedGramacy(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
    SyntheticTestFunctionTestCaseMixin,
):
    functions = [
        ConstrainedGramacy(),
        ConstrainedGramacy(negate=True),
        ConstrainedGramacy(noise_std=0.1, negate=True),
        ConstrainedGramacy(noise_std=0.1, constraint_noise_std=[0.1, 0.2], negate=True),
    ]


class TestConstrainedHartmann(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    SyntheticTestFunctionTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    for dim in [3, 6]:
        functions = [
            ConstrainedHartmann(dim=dim, negate=True),
            ConstrainedHartmann(noise_std=0.1, dim=dim, negate=True),
            ConstrainedHartmann(
                noise_std=0.1, constraint_noise_std=0.2, dim=dim, negate=True
            ),
        ]


class TestConstrainedHartmannSmooth(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    SyntheticTestFunctionTestCaseMixin,
    ConstrainedTestProblemTestCaseMixin,
):
    for dim in [3, 6]:
        functions = [
            ConstrainedHartmannSmooth(dim=dim, negate=True),
            ConstrainedHartmannSmooth(
                dim=dim, noise_std=0.1, constraint_noise_std=0.2, negate=True
            ),
        ]


class TestPressureVessel(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
):
    functions = [
        PressureVessel(),
        PressureVessel(noise_std=0.1, constraint_noise_std=0.1, negate=True),
        PressureVessel(
            noise_std=0.1, constraint_noise_std=[0.1, 0.2, 0.1, 0.2], negate=True
        ),
    ]


class TestSpeedReducer(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
):
    functions = [
        SpeedReducer(),
        SpeedReducer(noise_std=0.1, constraint_noise_std=0.1, negate=True),
        SpeedReducer(noise_std=0.1, constraint_noise_std=[0.1] * 11, negate=True),
    ]


class TestTensionCompressionString(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
):
    functions = [
        TensionCompressionString(),
        TensionCompressionString(
            noise_std=0.1, constraint_noise_std=[0.1, 0.2, 0.3, 0.4]
        ),
    ]


class TestWeldedBeamSO(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
):
    functions = [
        WeldedBeamSO(),
        WeldedBeamSO(noise_std=0.1, constraint_noise_std=[0.2] * 6),
    ]


class TestKeaneBumpFunction(
    BotorchTestCase,
    BaseTestProblemTestCaseMixIn,
    ConstrainedTestProblemTestCaseMixin,
):
    functions = [
        KeaneBumpFunction(dim=2),
        KeaneBumpFunction(dim=4, noise_std=0.1, constraint_noise_std=[0.1, 0.2]),
    ]
