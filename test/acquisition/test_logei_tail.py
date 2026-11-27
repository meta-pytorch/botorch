#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.analytic import _log_ei_helper, LogExpectedImprovement
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


# log(phi(u) + u*Phi(u)) and its first four derivatives, evaluated with
# 80-decimal-digit mpmath arithmetic, independently of the implementation.
TAIL_CONTROLS = (
    (
        -6.0,
        -22.578879392169796,
        6.309840773209832,
        -0.9550459440022557,
        0.012259238284847766,
        0.004714124246823661,
    ),
    (
        -30.0,
        -457.724653760598,
        30.06644615416242,
        -0.997799716275543,
        0.00014524164559645778,
        1.4334086535819955e-05,
    ),
    (
        -200.0,
        -20011.515648259738,
        200.00999925013122,
        -0.9999500112467199,
        4.99775098392241e-07,
        7.494378443276065e-09,
    ),
    (
        -1000.0,
        -500014.73445209116,
        1000.001999994,
        -0.9999980000179998,
        3.999928001259977e-09,
        1.1999640008819792e-11,
    ),
    (
        -10000.0,
        -50000019.33961931,
        10000.000199999993,
        -0.9999999800000018,
        3.999999280000126e-12,
        1.1999996400000882e-15,
    ),
    (
        -100000.0,
        -5000000023.94479,
        100000.00002,
        -0.9999999998,
        3.9999999928e-15,
        1.1999999964e-19,
    ),
    (
        -999999.0,
        -499999000029.05,
        999999.000002,
        -0.999999999998,
        4.0000119999519995e-18,
        1.2000047999759999e-23,
    ),
)


class TestLogEiTail(BotorchTestCase):
    def test_negative_tail_derivatives(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, dtype, control in itertools.product(
            devices, (torch.float, torch.double), TAIL_CONTROLS
        ):
            with self.subTest(device=device, dtype=dtype, point=control[0]):
                point = torch.tensor(
                    control[0], dtype=dtype, device=device, requires_grad=True
                )
                result = _log_ei_helper(point)
                tolerance = (
                    64 * torch.finfo(dtype).eps if dtype == torch.float else 2e-10
                )
                torch.testing.assert_close(
                    result, point.new_tensor(control[1]), rtol=tolerance, atol=0
                )
                for order in range(1, 5):
                    result = torch.autograd.grad(result, point, create_graph=True)[0]
                    torch.testing.assert_close(
                        result,
                        point.new_tensor(control[order + 1]),
                        rtol=tolerance,
                        atol=1e-30,
                    )

    def test_public_log_ei_mean_and_scale_hessian(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                point, _, gradient, hessian, *_ = TAIL_CONTROLS[4]
                mean = torch.tensor(
                    point * 0.1, dtype=torch.double, device=device, requires_grad=True
                )
                scale = mean.new_tensor(0.1, requires_grad=True)
                posterior = MockPosterior(
                    mean=mean.reshape(1, 1), variance=scale.square().reshape(1, 1)
                )
                acquisition = LogExpectedImprovement(MockModel(posterior), best_f=0.0)
                value = acquisition(mean.new_zeros(1, 1))
                first = torch.autograd.grad(
                    value.sum(), (mean, scale), create_graph=True
                )
                expected_first = (gradient / scale, (1 - point * gradient) / scale)
                for observed, reference in zip(first, expected_first):
                    torch.testing.assert_close(observed, reference, rtol=2e-12, atol=0)
                matrix = torch.stack(
                    [
                        torch.stack(
                            torch.autograd.grad(g, (mean, scale), retain_graph=True)
                        )
                        for g in first
                    ]
                )
                expected = (
                    mean.new_tensor(
                        [
                            [hessian, -gradient - point * hessian],
                            [
                                -gradient - point * hessian,
                                -1 + 2 * point * gradient + point * point * hessian,
                            ],
                        ]
                    )
                    / scale.square()
                )
                torch.testing.assert_close(matrix, expected, rtol=2e-10, atol=1e-8)
