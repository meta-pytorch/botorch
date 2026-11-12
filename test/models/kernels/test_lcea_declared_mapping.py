#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.contextual_lcea import LCEAKernel
from botorch.utils.testing import BotorchTestCase


class TestLCEADeclaredMapping(BotorchTestCase):
    def test_covariance_uses_declared_context_feature_order(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        decompositions = [
            {"a": [0, 2], "b": [3, 1]},
            {"a": [2, 3], "b": [0, 1]},
            {"a": [1, 0], "b": [3, 2]},
        ]
        for device in devices:
            for decomposition in decompositions:
                with self.subTest(device=device, decomposition=decomposition):
                    kernel = (
                        LCEAKernel(
                            decomposition,
                            batch_shape=torch.Size(),
                            train_embedding=False,
                            embs_feature_dict={"a": [0.25], "b": [0.75]},
                            device=device,
                        )
                        .double()
                        .to(device)
                    )
                    kernel.base_kernel.lengthscale = torch.tensor(
                        [0.3, 0.7], dtype=torch.double, device=device
                    )
                    kernel.outputscale_list = torch.tensor(
                        [0.8, 1.6], dtype=torch.double, device=device
                    )
                    X = torch.tensor(
                        [[0.1, 0.8, 0.3, 0.6], [0.9, 0.2, 0.7, 0.4]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    Y = X.detach().flip(0) * 0.7
                    actual = kernel(X, Y).to_dense()
                    context = kernel._eval_context_covar()
                    expected = sum(
                        context[i, j]
                        * kernel.base_kernel(X[..., left], Y[..., right]).to_dense()
                        for i, left in enumerate(decomposition.values())
                        for j, right in enumerate(decomposition.values())
                    )
                    self.assertAllClose(actual, expected)
                    self.assertAllClose(
                        torch.autograd.grad(actual.sum(), X)[0],
                        torch.autograd.grad(expected.sum(), X)[0],
                    )

    def test_noncontiguous_categorical_labels(self):
        for labels in (
            {"a": [10, -3], "b": [20, 7]},
            {"a": [20, 7], "b": [10, -3]},
        ):
            with self.subTest(labels=labels):
                torch.manual_seed(17)
                kernel = LCEAKernel(
                    {"a": [0], "b": [1]},
                    batch_shape=torch.Size(),
                    cat_feature_dict=labels,
                    embs_dim_list=[2, 1],
                ).double()
                encoded = {
                    name: [int(row[0] == 20), int(row[1] == 7)]
                    for name, row in labels.items()
                }
                torch.manual_seed(17)
                canonical = LCEAKernel(
                    {"a": [0], "b": [1]},
                    batch_shape=torch.Size(),
                    cat_feature_dict=encoded,
                    embs_dim_list=[2, 1],
                ).double()
                self.assertAllClose(
                    kernel._task_embeddings(), canonical._task_embeddings()
                )
