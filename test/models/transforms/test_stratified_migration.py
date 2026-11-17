#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.transforms.outcome import StratifiedStandardize
from botorch.utils.testing import BotorchTestCase


class TestStratifiedMigration(BotorchTestCase):
    def test_task_mapping_migrates_without_changing_persistent_state(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda", torch.cuda.current_device()))
        for device, dtype, tasks in itertools.product(
            devices, (torch.float, torch.double), ([0, 1], [0, 3])
        ):
            with self.subTest(device=device, dtype=dtype, tasks=tasks):
                original_dtype = torch.double if dtype == torch.float else torch.float
                transform = StratifiedStandardize(
                    -1, torch.tensor(tasks), dtype=original_dtype
                ).to(device=device, dtype=dtype)
                mapping = transform.strata_mapping
                self.assertEqual(mapping.device, device)
                self.assertEqual(
                    mapping.dtype, torch.long if tasks == [0, 1] else dtype
                )
                self.assertNotIn("strata_mapping", transform.state_dict())
                self.assertEqual(
                    set(transform.state_dict()),
                    {"means", "stdvs", "_stdvs_sq", "_is_trained"},
                )
                X = torch.tensor(
                    [
                        [0.0, tasks[0]],
                        [0.2, tasks[1]],
                        [0.4, tasks[0]],
                        [0.6, tasks[1]],
                    ],
                    dtype=dtype,
                    device=device,
                )
                Y = X.new_tensor([[0.1], [0.3], [0.7], [1.1]])
                with torch.no_grad():
                    transform(Y, X=X)
                transform.eval()
                query = X.new_tensor([[0.2], [0.5]], requires_grad=True)
                variance = X.new_tensor([[0.03], [0.04]], requires_grad=True)
                actual, actual_variance = transform(query, variance, X=X[:2])
                means = torch.stack((Y[::2].mean(), Y[1::2].mean())).unsqueeze(-1)
                stdvs = torch.stack((Y[::2].std(), Y[1::2].std())).unsqueeze(-1)
                expected, expected_variance = (
                    (query - means) / stdvs,
                    variance / stdvs.square(),
                )
                self.assertAllClose(actual, expected)
                self.assertAllClose(actual_variance, expected_variance)
                losses = (
                    actual.sin().sum() + actual_variance.square().sum(),
                    expected.sin().sum() + expected_variance.square().sum(),
                )
                gradients = [
                    torch.autograd.grad(loss, (query, variance), create_graph=True)
                    for loss in losses
                ]
                hessians = [
                    torch.autograd.grad(
                        sum(g.square().sum() for g in row), (query, variance)
                    )
                    for row in gradients
                ]
                for a, b in zip(*gradients):
                    self.assertAllClose(a, b)
                for a, b in zip(*hessians):
                    self.assertAllClose(a, b)
                state = transform.state_dict()
                restored = StratifiedStandardize(-1, torch.tensor(tasks)).to(
                    device=device, dtype=dtype
                )
                restored.load_state_dict(state)
                self.assertAllClose(restored.eval()(query, X=X[:2])[0], expected)
                transform.cpu()
                self.assertEqual(transform.strata_mapping.device.type, "cpu")
