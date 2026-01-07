#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import ceil
from unittest.mock import patch

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features import generators
from botorch.sampling.pathwise.features.generators import gen_kernel_features
from botorch.sampling.pathwise.features.maps import FeatureMap
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from torch import Size, Tensor


class TestFeatureGenerators(BotorchTestCase):
    def setUp(self, seed: int = 0) -> None:
        super().setUp()

        self.kernels = []
        self.num_inputs = d = 2
        self.num_features = 4096
        for kernel in (
            MaternKernel(nu=0.5, batch_shape=Size([])),
            MaternKernel(nu=1.5, ard_num_dims=1, active_dims=[0]),
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=Size([2]))),
            ScaleKernel(
                RBFKernel(ard_num_dims=1, batch_shape=Size([2, 2])), active_dims=[1]
            ),
        ):
            kernel.to(
                dtype=torch.float32 if (seed % 2) else torch.float64, device=self.device
            )
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                kern = kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel
                kern.lengthscale = 0.1 + 0.2 * torch.rand_like(kern.lengthscale)
                seed += 1

            self.kernels.append(kernel)

    def test_gen_kernel_features(self):
        for seed, kernel in enumerate(self.kernels):
            with torch.random.fork_rng():
                torch.random.manual_seed(seed)
                feature_map = gen_kernel_features(
                    kernel=kernel,
                    num_inputs=self.num_inputs,
                    num_outputs=self.num_features,
                )

                n = 4
                m = ceil(n * kernel.batch_shape.numel() ** -0.5)
                for input_batch_shape in ((n**2,), (m, *kernel.batch_shape, m)):
                    X = torch.rand(
                        (*input_batch_shape, self.num_inputs),
                        device=kernel.device,
                        dtype=kernel.dtype,
                    )
                    self._test_gen_kernel_features(kernel, feature_map, X)

    def _test_gen_kernel_features(
        self, kernel: Kernel, feature_map: FeatureMap, X: Tensor, atol: float = 3.0
    ):
        with self.subTest("test_initialization"):
            self.assertEqual(feature_map.weight.dtype, kernel.dtype)
            self.assertEqual(feature_map.weight.device, kernel.device)
            self.assertEqual(
                feature_map.weight.shape[-1],
                (
                    self.num_inputs
                    if kernel.active_dims is None
                    else len(kernel.active_dims)
                ),
            )

        with self.subTest("test_covariance"):
            features = feature_map(X)
            test_shape = torch.broadcast_shapes(
                (*X.shape[:-1], self.num_features), kernel.batch_shape + (1, 1)
            )
            self.assertEqual(features.shape, test_shape)
            K0 = features @ features.transpose(-2, -1)
            K1 = kernel(X).to_dense()
            self.assertTrue(
                K0.allclose(K1, atol=atol * self.num_features**-0.5, rtol=0)
            )

        # Test passing the wrong dimensional shape to `weight_generator`
        with self.assertRaisesRegex(UnsupportedError, "2-dim"), patch.object(
            generators,
            "_gen_fourier_features",
            side_effect=lambda **kwargs: kwargs["weight_generator"](Size([])),
        ):
            gen_kernel_features(
                kernel=kernel,
                num_inputs=self.num_inputs,
                num_outputs=self.num_features,
            )

        # Test requesting an odd number of features
        with self.assertRaisesRegex(UnsupportedError, "Expected an even number"):
            gen_kernel_features(
                kernel=kernel, num_inputs=self.num_inputs, num_outputs=3
            )
