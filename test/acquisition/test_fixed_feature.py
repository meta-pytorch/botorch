#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.fixed_feature import (
    FixedFeatureAcquisitionFunction,
    get_device_of_sequence,
    get_dtype_of_sequence,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction


class TestFixedFeatureAcquisitionFunction(BotorchTestCase):
    def test_fixed_features(self) -> None:
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = train_X.norm(dim=-1, keepdim=True)
        model = SingleTaskGP(train_X, train_Y).to(device=self.device).eval()
        for q in [1, 2]:
            qEI = qExpectedImprovement(model, best_f=0.0)

            # test single point
            test_X = torch.rand(q, 3, device=self.device)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[..., -1:]
            )
            qei = qEI(test_X)
            qei_ff = qEI_ff(test_X[..., :-1])
            self.assertAllClose(qei, qei_ff)

            # test list input with float and scalar tensor
            for value in [0.5, torch.tensor(0.5)]:
                qEI_ff = FixedFeatureAcquisitionFunction(
                    qEI, d=3, columns=[2], values=[value]
                )
                qei_ff = qEI_ff(test_X[..., :-1])
                test_X_clone = test_X.clone()
                test_X_clone[..., 2] = value
                qei = qEI(test_X_clone)
                self.assertAllClose(qei, qei_ff)

                # test list input with Tensor and float
                qEI_ff = FixedFeatureAcquisitionFunction(
                    qEI, d=3, columns=[0, 2], values=[test_X[..., [0]], value]
                )
                qei_ff = qEI_ff(test_X[..., [1]])
                self.assertAllClose(qei, qei_ff)

            # test t-batch with broadcasting and list of floats
            test_X = torch.rand(q, 3, device=self.device).expand(4, q, 3)
            qei = qEI(test_X)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[0, :, -1:]
            )
            qei_ff = qEI_ff(test_X[..., :-1])
            self.assertAllClose(qei, qei_ff)

            # test t-batch with broadcasting and list of floats and Tensor
            # test list input with float and scalar tensor
            for value in [0.5, torch.tensor(0.5)]:
                qEI_ff = FixedFeatureAcquisitionFunction(
                    qEI, d=3, columns=[0, 2], values=[test_X[0, :, [0]], value]
                )
                qei_ff = qEI_ff(test_X[..., [1]])
                test_X_clone = test_X.clone()
                test_X_clone[..., 2] = value
                qei = qEI(test_X_clone)
                self.assertAllClose(qei, qei_ff)

            # test X_pending
            X_pending = torch.rand(2, 3, device=self.device)
            qEI.set_X_pending(X_pending)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=test_X[..., -1:]
            )
            self.assertAllClose(qEI.X_pending, qEI_ff.X_pending)

            # test setting X_pending from qEI_ff
            # (set target value to be last dim of X_pending and check if the
            # constructed X_pending on qEI is the full X_pending)
            X_pending = torch.rand(2, 3, device=self.device)
            qEI.X_pending = None
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[2], values=X_pending[..., -1:]
            )
            qEI_ff.set_X_pending(X_pending[..., :-1])
            self.assertAllClose(qEI.X_pending, X_pending)
            # test setting to None
            qEI_ff.set_X_pending(None)
            self.assertIsNone(qEI_ff.X_pending)

        # test gradient
        test_X = torch.rand(1, 3, device=self.device, requires_grad=True)
        qei = qEI(test_X)
        qEI_ff = FixedFeatureAcquisitionFunction(
            qEI, d=3, columns=[2], values=test_X[..., [2]].detach()
        )
        test_X_ff = test_X[..., :-1].detach().clone().requires_grad_(True)
        qei_ff = qEI_ff(test_X_ff)
        self.assertAllClose(qei, qei_ff)
        qei.backward()
        qei_ff.backward()
        self.assertAllClose(test_X.grad[..., :-1], test_X_ff.grad)

        # test list input with float and scalar tensor
        for value in [0.5, torch.tensor(0.5)]:
            # computing with fixed features
            test_X_ff = test_X[..., [1]].detach().clone().requires_grad_(True)
            qEI_ff = FixedFeatureAcquisitionFunction(
                qEI, d=3, columns=[0, 2], values=[test_X[..., [0]].detach(), value]
            )
            qei_ff = qEI_ff(test_X_ff)
            qei_ff.backward()
            # computing ground truth
            test_X_clone = test_X.detach().clone()
            test_X_clone[..., 2] = value
            test_X_clone.requires_grad_(True)
            qei = qEI(test_X_clone)
            qei.backward()
            self.assertAllClose(test_X_clone.grad[..., [1]], test_X_ff.grad)

        # test error b/c of incompatible input shapes
        with self.assertRaises(ValueError):
            qEI_ff(test_X)

        # test error when there is no X_pending (analytic EI)
        test_X = torch.rand(q, 3, device=self.device)
        analytic_EI = ExpectedImprovement(model, best_f=0.0)
        EI_ff = FixedFeatureAcquisitionFunction(
            analytic_EI, d=3, columns=[2], values=test_X[..., -1:]
        )
        with self.assertRaises(ValueError):
            EI_ff.X_pending

    def test_values_dtypes(self) -> None:
        acqf = MockAcquisitionFunction()

        for input, d, expected_dtype in [
            (torch.tensor([0.0], dtype=torch.float32), 1, torch.float32),
            (torch.tensor([0.0], dtype=torch.float64), 1, torch.float64),
            (
                [
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float64),
                ],
                2,
                torch.float64,
            ),
            ([0.0], 1, torch.float64),
            ([torch.tensor(0.0, dtype=torch.float32), 0.0], 2, torch.float64),
        ]:
            with self.subTest(input=input, d=d, expected_dtype=expected_dtype):
                self.assertEqual(get_dtype_of_sequence(input), expected_dtype)
                ff = FixedFeatureAcquisitionFunction(
                    acqf, d=d, columns=[2], values=input
                )
                self.assertEqual(ff.values.dtype, expected_dtype)

    def test_values_devices(self) -> None:
        acqf = MockAcquisitionFunction()
        cpu = torch.device("cpu")
        cuda = torch.device("cuda")

        test_cases = [
            (torch.tensor([0.0], device=cpu), 1, cpu),
            ([0.0], 1, cpu),
            ([0.0, torch.tensor([0.0], device=cpu)], 2, cpu),
        ]

        # Can only properly test this when running CUDA tests
        if self.device == torch.cuda:
            test_cases = test_cases + [
                (torch.tensor([0.0], device=cuda), 1, cuda),
                (
                    [
                        torch.tensor([0.0], dtype=cpu),
                        torch.tensor([0.0], dtype=cuda),
                    ],
                    2,
                    cuda,
                ),
                ([0.0], 1, cpu),
                ([torch.tensor(0.0, dtype=cuda), 0.0], 2, cuda),
            ]

        for input, d, expected_device in test_cases:
            with self.subTest(input=input, d=d, expected_device=expected_device):
                self.assertEqual(get_device_of_sequence(input), expected_device)
                ff = FixedFeatureAcquisitionFunction(
                    acqf, d=d, columns=[2], values=input
                )
                self.assertEqual(ff.values.device, expected_device)
