#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class TestGIBBONQueryBatchAxes(BotorchTestCase):
    def test_multiple_query_batch_axes(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for pending in (False, True):
                with self.subTest(device=device, pending=pending):
                    X = torch.tensor(
                        [[0.04], [0.27], [0.56], [0.81]],
                        dtype=torch.double,
                        device=device,
                    )
                    model = SingleTaskGP(X, (5.3 * X).sin(), outcome_transform=None)
                    acq = qLowerBoundMaxValueEntropy(
                        model,
                        X,
                        use_gumbel=False,
                        X_pending=X.new_tensor([[0.91]]) if pending else None,
                    )
                    acq.posterior_max_values = X.new_tensor(
                        [[0.75], [1.0], [1.2], [1.4]]
                    )
                    query = X.new_tensor([[[[0.19]], [[0.43]]]]).repeat(3, 1, 1, 1)
                    query.requires_grad_()
                    actual = acq(query)
                    expected = torch.stack([acq(row) for row in query])
                    self.assertEqual(actual.shape, torch.Size([3, 2]))
                    self.assertAllClose(actual, expected)
                    actual_grad = torch.autograd.grad(
                        actual.sum(), query, create_graph=True
                    )[0]
                    expected_grad = torch.autograd.grad(
                        expected.sum(), query, create_graph=True
                    )[0]
                    self.assertAllClose(actual_grad, expected_grad)
                    self.assertAllClose(
                        torch.autograd.grad(actual_grad.sum(), query)[0],
                        torch.autograd.grad(expected_grad.sum(), query)[0],
                    )
