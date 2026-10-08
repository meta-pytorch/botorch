#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.test_functions.multi_fidelity import (
    AugmentedBranin,
    AugmentedHartmann,
    AugmentedRosenbrock,
    BoreholeMultiFidelity,
    WingWeightMultiFidelity,
)
from botorch.utils.testing import (
    BaseTestProblemTestCaseMixIn,
    BotorchTestCase,
    SyntheticTestFunctionTestCaseMixin,
)


class TestAugmentedBranin(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        AugmentedBranin(),
        AugmentedBranin(negate=True),
        AugmentedBranin(noise_std=0.1),
    ]


class TestAugmentedHartmann(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        AugmentedHartmann(),
        AugmentedHartmann(negate=True),
        AugmentedHartmann(noise_std=0.1),
    ]


class TestAugmentedRosenbrock(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [
        AugmentedRosenbrock(),
        AugmentedRosenbrock(negate=True),
        AugmentedRosenbrock(noise_std=0.1),
        AugmentedRosenbrock(dim=4),
        AugmentedRosenbrock(dim=4, negate=True),
        AugmentedRosenbrock(dim=4, noise_std=0.1),
    ]

    def test_min_dimension(self):
        # At least two design parameters are needed for a non-trivial function.
        for dim in (2, 3):
            with self.assertRaisesRegex(ValueError, "at least 4 dimensions"):
                AugmentedRosenbrock(dim=dim)
        f = AugmentedRosenbrock()
        self.assertEqual(f.dim, 4)
        X = torch.tensor([0.0, 0.0, 1.0, 1.0], device=self.device)
        self.assertAllClose(f.to(self.device).evaluate_true(X), torch.ones_like(X[0]))


class TestDiscreteMultiFidelity(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    functions = [WingWeightMultiFidelity(), BoreholeMultiFidelity()]

    def test_optimizer(self):
        pass

    def test_each_fidelity_and_cost(self):
        dtypes = (torch.float, torch.double)
        batch_shapes = (torch.Size(), torch.Size([2]), torch.Size([2, 3]))
        for dtype, batch_shape, f in product(dtypes, batch_shapes, self.functions):
            f.to(device=self.device, dtype=dtype)
            X = torch.rand(*batch_shape, f.dim, device=self.device, dtype=dtype)
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])
            X[..., -1] = X[..., -1].round()
            for fidelity in f.fidelities:
                if X.ndim == 1:
                    X[-1] = fidelity
                else:
                    # only change one fidelity value to test that the masking in
                    # evaluate_true still yields the expected shapes
                    X[..., 0, -1] = fidelity
                res_forward = f(X)
                res_evaluate_true = f.evaluate_true(X)
                res_cost = f.cost(X)
                for method, res in {
                    "forward": res_forward,
                    "evaluate_true": res_evaluate_true,
                    "cost": res_cost,
                }.items():
                    with self.subTest(
                        f"{dtype}_{batch_shape}_{f.__class__.__name__}_{method}"
                        f"_{fidelity}"
                    ):
                        self.assertEqual(res.dtype, dtype)
                        self.assertEqual(res.device.type, self.device.type)
                        tail_shape = torch.Size(
                            [f.num_objectives] if f.num_objectives > 1 else []
                        )
                        self.assertEqual(res.shape, batch_shape + tail_shape)

    def test_fidelity_is_categorical(self):
        for f in self.functions:
            self.assertEqual(f.categorical_inds, [f.dim - 1])
            self.assertEqual(f.continuous_inds, list(range(f.dim - 1)))
            X = (f.bounds[0] + f.bounds[1]) / 2
            X[-1] = 0.5
            with self.assertRaisesRegex(ValueError, "integer values"):
                f.evaluate_true(X)
