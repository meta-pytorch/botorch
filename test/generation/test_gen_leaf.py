# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.generation.gen import gen_candidates_torch
from botorch.utils.testing import BotorchTestCase


class TestGenCandidatesTorchAttachedStarts(BotorchTestCase):
    def test_attached_initial_conditions_and_repeated_stages(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for fixed in (None, {1: 0.4}):
                with self.subTest(device=device, fixed=fixed):
                    leaf = torch.tensor(
                        [[[0.1, 0.2]], [[0.8, 0.4]]],
                        dtype=torch.double,
                        device=device,
                        requires_grad=True,
                    )
                    scale = torch.nn.Parameter(leaf.new_tensor(2.0))

                    def acquisition(X):
                        return -scale * (X - 0.73).square().sum(dim=(-2, -1))

                    initial = leaf * 1.0
                    for _ in range(2):
                        candidate, values = gen_candidates_torch(
                            initial,
                            acquisition,
                            lower_bounds=0.0,
                            upper_bounds=1.0,
                            fixed_features=fixed,
                            options={
                                "optimizer_options": {"lr": 0.05},
                                "stopping_criterion_options": {
                                    "maxiter": 100,
                                    "n_window": 100,
                                },
                            },
                        )
                        self.assertAllClose(values, acquisition(candidate))
                        self.assertAllClose(
                            candidate[..., 0],
                            candidate.new_full(candidate.shape[:-1], 0.73),
                            atol=0.005,
                            rtol=0,
                        )
                        if fixed:
                            self.assertAllClose(
                                candidate[..., 1],
                                candidate.new_full(candidate.shape[:-1], 0.4),
                            )
                        self.assertIsNone(leaf.grad)
                        self.assertIsNone(scale.grad)
                        # Homotopy passes the graph-attached result of one
                        # optimization as the next stage's initial conditions.
                        initial = candidate
