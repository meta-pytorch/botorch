#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from botorch.models.transforms.utils import nanstd


class TestNanstdAutograd(unittest.TestCase):
    def test_observed_values_match_std_with_finite_first_and_second_derivatives(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in (torch.double, torch.float):
                for dim in (0, -1):
                    for keepdim in (False, True):
                        with self.subTest(
                            device=device, dtype=dtype, dim=dim, keepdim=keepdim
                        ):
                            X = torch.tensor(
                                [
                                    [0.2, torch.nan, 0.8, 0.5],
                                    [0.1, 0.7, 0.9, torch.nan],
                                    [0.4, 0.5, torch.nan, 0.2],
                                ],
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            result = nanstd(X, dim=dim, keepdim=keepdim)
                            rows = X.transpose(0, 1) if dim == 0 else X
                            expected = torch.stack(
                                [row[~torch.isnan(row)].std() for row in rows]
                            )
                            if keepdim:
                                expected = expected.unsqueeze(dim)
                            tolerance = 2e-4 if dtype == torch.float else 2e-11
                            torch.testing.assert_close(
                                result, expected, atol=tolerance, rtol=tolerance
                            )
                            direction = torch.linspace(
                                0.1, 0.6, X.numel(), device=device, dtype=dtype
                            ).reshape_as(X)

                            def derivatives(values):
                                first = torch.autograd.grad(
                                    values.log().sum(), X, create_graph=True
                                )[0]
                                second = torch.autograd.grad(
                                    (first * direction).sum(), X
                                )[0]
                                return first, second

                            actual_derivatives, expected_derivatives = (
                                derivatives(result),
                                derivatives(expected),
                            )
                            for actual, reference in zip(
                                actual_derivatives, expected_derivatives, strict=True
                            ):
                                self.assertTrue(torch.isfinite(actual).all())
                                torch.testing.assert_close(
                                    actual, reference, atol=tolerance, rtol=tolerance
                                )

    def test_constant_and_undefined_slices(self):
        X = torch.tensor(
            [
                [0.5, 0.5, torch.nan],
                [torch.nan, torch.nan, torch.nan],
                [0.5, torch.nan, torch.nan],
            ],
            dtype=torch.double,
            requires_grad=True,
        )
        result = nanstd(X, dim=-1)
        torch.testing.assert_close(
            result, result.new_tensor([0.0, torch.nan, torch.nan]), equal_nan=True
        )
        first = torch.autograd.grad(result[:1].sum(), X, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), X)[0]
        torch.testing.assert_close(first, torch.zeros_like(X), atol=0, rtol=0)
        torch.testing.assert_close(second, torch.zeros_like(X), atol=0, rtol=0)

    def test_numerical_gradgrad_with_missing_entries(self):
        X = torch.tensor(
            [[0.2, torch.nan, 0.8, 0.5], [0.1, 0.7, torch.nan, 0.4]],
            dtype=torch.double,
            requires_grad=True,
        )
        self.assertTrue(torch.autograd.gradcheck(lambda x: nanstd(x, -1), (X,)))
        self.assertTrue(torch.autograd.gradgradcheck(lambda x: nanstd(x, -1), (X,)))


if __name__ == "__main__":
    unittest.main()
