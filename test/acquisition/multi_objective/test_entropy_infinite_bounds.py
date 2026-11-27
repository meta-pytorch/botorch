#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.acquisition.multi_objective.joint_entropy_search import (
    _compute_entropy_monte_carlo,
    _compute_entropy_noiseless,
    _compute_entropy_upper_bound,
)
from botorch.utils.testing import BotorchTestCase


class TestEntropyInfiniteBounds(BotorchTestCase):
    def test_full_space_gaussian_values_and_derivatives(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for estimator in ("0", "LB", "LB2", "MC"):
                with self.subTest(device=device, estimator=estimator):
                    mean = torch.tensor(
                        [[[[[0.2, 0.3]]]]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    variance = mean.new_tensor([[[[[0.7, 0.8]]]]], requires_grad=True)
                    noise = torch.full_like(variance, 0.1, requires_grad=True)
                    bounds = mean.new_tensor(
                        [[[[-torch.inf, -torch.inf]], [[torch.inf, torch.inf]]]]
                    )
                    if estimator == "0":
                        value = _compute_entropy_noiseless(
                            bounds, mean, variance, noise
                        )
                    elif estimator == "MC":
                        normals = torch.randn(
                            (8, *mean.shape),
                            dtype=mean.dtype,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(61),
                        )
                        samples = mean + (variance + noise).sqrt() * normals
                        log_probability = -0.5 * (
                            normals.square()
                            + (variance + noise).log()
                            + math.log(2 * math.pi)
                        ).sum(-1).squeeze(-1)
                        value = _compute_entropy_monte_carlo(
                            bounds, mean, variance, noise, samples, log_probability
                        )
                    else:
                        value = _compute_entropy_upper_bound(
                            bounds,
                            mean,
                            variance,
                            noise,
                            only_diagonal=estimator == "LB2",
                        )
                    if estimator == "MC":
                        expected = -log_probability.mean(0).mean(-2)
                    else:
                        jitter = 1e-6 if estimator == "LB" else 0.0
                        expected = (
                            math.log(2 * math.pi)
                            + 1
                            + 0.5
                            * (variance + noise + jitter)
                            .log()
                            .sum(-1)
                            .squeeze(-1)
                            .mean(-2)
                        )
                    self.assertAllClose(value, expected)
                    parameters = (mean, variance, noise)
                    first = torch.autograd.grad(
                        value.sum(), parameters, create_graph=True
                    )
                    second = torch.autograd.grad(
                        sum(g.square().sum() for g in first),
                        parameters,
                        allow_unused=True,
                    )
                    for gradient in (*first, *second):
                        self.assertTrue(
                            gradient is None or torch.isfinite(gradient).all()
                        )
                    expected_var_gradient = 0.5 / (
                        variance + noise + (1e-6 if estimator == "LB" else 0)
                    )
                    self.assertAllClose(first[1], expected_var_gradient)

    def test_noiseless_half_line_matches_truncated_normal(self):
        mean = torch.tensor([[[[[0.2]]]]], dtype=torch.double, requires_grad=True)
        variance = torch.full_like(mean, 0.7, requires_grad=True)
        bounds = mean.new_tensor([[[[-torch.inf]], [[0.6]]]])
        value = _compute_entropy_noiseless(
            bounds, mean, variance, torch.zeros_like(variance)
        )
        g = (0.6 - mean) / variance.sqrt()
        normal = torch.distributions.Normal(torch.zeros_like(g), torch.ones_like(g))
        cdf = normal.cdf(g)
        expected = (
            (
                0.5 * (1 + math.log(2 * math.pi) + variance.log())
                + cdf.log()
                - 0.5 * g * normal.log_prob(g).exp() / cdf
            )
            .squeeze(-1)
            .squeeze(-1)
            .mean(-2)
        )
        self.assertAllClose(value, expected)
        actual_grad = torch.autograd.grad(
            value.sum(), (mean, variance), create_graph=True
        )
        expected_grad = torch.autograd.grad(
            expected.sum(), (mean, variance), create_graph=True
        )
        for actual, reference in zip(actual_grad, expected_grad):
            self.assertAllClose(actual, reference)
