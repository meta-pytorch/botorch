#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    _update_damping,
)
from botorch.utils.testing import BotorchTestCase


class TestPESFirstPivotDamping(BotorchTestCase):
    def test_failed_first_pivot_is_damped(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for batched in (False, True):
                with self.subTest(device=device, batched=batched):
                    previous = torch.eye(2, dtype=torch.double, device=device)
                    proposal = previous.clone()
                    proposal[0, 0] = -1
                    damping = torch.ones((), dtype=torch.double, device=device)
                    if batched:
                        previous = previous.expand(2, 2, 2)
                        proposal = torch.stack((proposal, torch.eye(2).to(proposal)))
                        damping = damping.expand(2)
                    jitter = 1e-4
                    updated, root = _update_damping(previous, proposal, damping, jitter)
                    expected_damping = (
                        torch.tensor([0.5, 1.0], device=device, dtype=torch.double)
                        if batched
                        else damping * 0.5
                    )
                    self.assertAllClose(updated, expected_damping)
                    df = updated[..., None, None]
                    expected = (1 - df) * previous + df * proposal
                    # The retry path adds jitter to every member of the batch.
                    expected = expected + torch.eye(2).to(expected) * jitter
                    self.assertAllClose(root @ root.mT, expected)
                    self.assertTrue((root.diagonal(dim1=-2, dim2=-1) > 0).all())

    def test_invalid_previous_first_pivot_is_rejected(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                previous = torch.diag(torch.tensor([-1.0, 1.0], device=device))
                proposal = torch.eye(2, device=device)
                with self.assertRaisesRegex(ValueError, "previous covariance"):
                    _update_damping(previous, proposal, previous.new_tensor(1), 1e-4)
