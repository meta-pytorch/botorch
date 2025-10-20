#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import product

import torch

from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.generation.utils import (
    _flip_sub_unique,
    _remove_fixed_features_from_optimization,
)
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction


class TestGenerationUtils(BotorchTestCase):
    def test_flip_sub_unique(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            x = torch.tensor([0.69, 0.75, 0.69, 0.21, 0.86, 0.21], **tkwargs)
            y = _flip_sub_unique(x=x, k=1)
            y_exp = torch.tensor([0.21], **tkwargs)
            self.assertAllClose(y, y_exp)
            y = _flip_sub_unique(x=x, k=3)
            y_exp = torch.tensor([0.21, 0.86, 0.69], **tkwargs)
            self.assertAllClose(y, y_exp)
            y = _flip_sub_unique(x=x, k=10)
            y_exp = torch.tensor([0.21, 0.86, 0.69, 0.75], **tkwargs)
            self.assertAllClose(y, y_exp)
        # long dtype
        tkwargs["dtype"] = torch.long
        x = torch.tensor([1, 6, 4, 3, 6, 3], **tkwargs)
        y = _flip_sub_unique(x=x, k=1)
        y_exp = torch.tensor([3], **tkwargs)
        self.assertAllClose(y, y_exp)
        y = _flip_sub_unique(x=x, k=3)
        y_exp = torch.tensor([3, 6, 4], **tkwargs)
        self.assertAllClose(y, y_exp)
        y = _flip_sub_unique(x=x, k=4)
        y_exp = torch.tensor([3, 6, 4, 1], **tkwargs)
        self.assertAllClose(y, y_exp)
        y = _flip_sub_unique(x=x, k=10)
        self.assertAllClose(y, y_exp)

    def test_remove_fixed_features_from_optimization(self):
        fixed_features = {1: 1.0, 3: -1.0}
        b, q, d = 7, 3, 5
        initial_conditions = torch.randn(b, q, d, device=self.device)
        tensor_lower_bounds = torch.randn(q, d, device=self.device)
        tensor_upper_bounds = tensor_lower_bounds + torch.rand(q, d, device=self.device)
        old_inequality_constraints = [
            (
                torch.arange(0, 5, 2, device=self.device),
                torch.rand(3, device=self.device),
                1.0,
            )
        ]
        old_equality_constraints = [
            (
                torch.arange(0, 3, 1, device=self.device),
                torch.rand(3, device=self.device),
                1.0,
            )
        ]
        acqf = MockAcquisitionFunction()

        def check_bounds_and_init(old_val, new_val):
            if isinstance(old_val, float):
                self.assertEqual(old_val, new_val)
            elif isinstance(old_val, torch.Tensor):
                mask = [(i not in fixed_features.keys()) for i in range(d)]
                self.assertTrue(torch.equal(old_val[..., mask], new_val))
            else:
                self.assertIsNone(new_val)

        def check_cons(old_cons, new_cons):
            if old_cons:  # we don't fixed all dimensions in this test
                new_dim = d - len(fixed_features)
                self.assertTrue(
                    torch.all((new_cons[0][0] <= new_dim) & (new_cons[0][0] >= 0))
                )
            else:
                self.assertEqual(old_cons, new_cons)

        def check_nlc(old_nlcs, new_nlcs):
            complete_data = torch.tensor(
                [[4.0, 1.0, 2.0, -1.0, 3.0]], device=self.device
            )
            reduced_data = torch.tensor([[4.0, 2.0, 3.0]], device=self.device)
            if old_nlcs:
                self.assertAllClose(
                    old_nlcs[0][0](complete_data),
                    new_nlcs[0][0](reduced_data),
                )
            else:
                self.assertEqual(old_nlcs, new_nlcs)

        def nlc(x):
            return x[..., 2]

        old_nlcs = [(nlc, True)]

        for (
            lower_bounds,
            upper_bounds,
            inequality_constraints,
            equality_constraints,
            nonlinear_inequality_constraints,
        ) in product(
            [None, -1.0, tensor_lower_bounds],
            [None, 1.0, tensor_upper_bounds],
            [None, old_inequality_constraints],
            [None, old_equality_constraints],
            [None, old_nlcs],
        ):
            _no_ff = _remove_fixed_features_from_optimization(
                fixed_features=fixed_features,
                acquisition_function=acqf,
                initial_conditions=initial_conditions,
                d=initial_conditions.shape[-1],
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            )
            self.assertIsInstance(
                _no_ff.acquisition_function, FixedFeatureAcquisitionFunction
            )
            check_bounds_and_init(initial_conditions, _no_ff.initial_conditions)
            check_bounds_and_init(lower_bounds, _no_ff.lower_bounds)
            check_bounds_and_init(upper_bounds, _no_ff.upper_bounds)
            check_cons(inequality_constraints, _no_ff.inequality_constraints)
            check_cons(equality_constraints, _no_ff.equality_constraints)
            check_nlc(
                nonlinear_inequality_constraints,
                _no_ff.nonlinear_inequality_constraints,
            )
