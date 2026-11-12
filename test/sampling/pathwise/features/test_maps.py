#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from botorch.sampling.pathwise.features import KernelEvaluationMap, KernelFeatureMap
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel
from torch import Size
from torch.nn import Parameter


class TestFeatureMaps(BotorchTestCase):
    def test_kernel_evaluation_map_points_follow_module_conversion(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for parameter in (False, True):
                with self.subTest(device=device, parameter=parameter):
                    points = torch.tensor(
                        [[0.1, 0.3], [0.6, 0.8]],
                        dtype=torch.float64,
                        device=device,
                        requires_grad=True,
                    )
                    if parameter:
                        points = Parameter(points)
                    kernel = MaternKernel(nu=2.5, ard_num_dims=2).to(points)
                    feature_map = KernelEvaluationMap(kernel, points)
                    self.assertIs(feature_map.points, points)
                    feature_map.float().cpu()
                    self.assertEqual(feature_map.points.dtype, torch.float32)
                    self.assertEqual(feature_map.points.device, torch.device("cpu"))
                    self.assertEqual(
                        "points" in dict(feature_map.named_parameters()), parameter
                    )
                    query = torch.tensor([[0.2, 0.5]], requires_grad=True)
                    result = feature_map(query).to_dense()
                    self.assertEqual(result.dtype, torch.float32)
                    gradients = torch.autograd.grad(
                        result.square().sum(), (query, feature_map.points)
                    )
                    self.assertTrue(all(g.isfinite().all() for g in gradients))

    def test_kernel_evaluation_map_points_state_dict_roundtrip(self):
        for parameter in (False, True):
            with self.subTest(parameter=parameter):
                points = torch.tensor([[0.1], [0.7]], device=self.device)
                if parameter:
                    points = Parameter(points)
                feature_map = KernelEvaluationMap(
                    MaternKernel(nu=2.5).to(points), points
                )
                replacement = KernelEvaluationMap(
                    MaternKernel(nu=2.5).to(points), torch.zeros_like(points)
                )
                self.assertIn("points", feature_map.state_dict())
                replacement.load_state_dict(feature_map.state_dict())
                self.assertAllClose(replacement.points, points)

    def test_kernel_evaluation_map(self):
        d = 3
        n_points = 7
        n_test = 5
        kernel = MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=Size([2]))
        kernel.to(device=self.device)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            kernel.lengthscale = 0.1 + 0.3 * torch.rand_like(kernel.lengthscale)

        with self.assertRaisesRegex(RuntimeError, "Shape mismatch"):
            KernelEvaluationMap(kernel=kernel, points=torch.rand(4, n_points, d))

        for dtype in (torch.float32, torch.float64):
            kernel.to(dtype=dtype)
            X0 = torch.rand(n_test, d, dtype=dtype, device=self.device)
            X1 = torch.rand(n_points, d, dtype=dtype, device=self.device)
            kernel_map = KernelEvaluationMap(kernel=kernel, points=X1)
            self.assertEqual(kernel_map.batch_shape, kernel.batch_shape)
            # num_outputs should be the number of points (shape[-2]),
            # since kernel(x, points) returns ... x q x n_points
            self.assertEqual(kernel_map.num_outputs, n_points)
            self.assertTrue(kernel_map(X0).to_dense().equal(kernel(X0, X1).to_dense()))

        with patch.object(
            kernel_map, "output_transform", new=lambda z: torch.concat([z, z], dim=-1)
        ):
            self.assertEqual(kernel_map.num_outputs, 2 * n_points)

    def test_kernel_feature_map(self):
        d = 2
        m = 3
        weight = torch.rand(m, d, device=self.device)
        bias = torch.rand(m, device=self.device)
        kernel = MaternKernel(nu=2.5, batch_shape=Size([3])).to(self.device)
        feature_map = KernelFeatureMap(
            kernel=kernel,
            weight=weight,
            bias=bias,
            input_transform=MagicMock(side_effect=lambda x: x),
            output_transform=MagicMock(side_effect=lambda z: z.exp()),
        )

        X = torch.rand(2, d, device=self.device)
        features = feature_map(X)
        feature_map.input_transform.assert_called_once_with(X)
        feature_map.output_transform.assert_called_once()
        self.assertTrue((X @ weight.transpose(-2, -1) + bias).exp().equal(features))

        # Test batch_shape and num_outputs
        self.assertIs(feature_map.batch_shape, kernel.batch_shape)
        self.assertEqual(feature_map.num_outputs, weight.shape[-2])
        with patch.object(feature_map, "output_transform", new=None):
            self.assertEqual(feature_map.num_outputs, weight.shape[-2])
