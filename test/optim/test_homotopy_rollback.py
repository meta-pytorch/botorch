# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from botorch.optim.homotopy import FixedHomotopySchedule, Homotopy, HomotopyParameter
from botorch.optim.optimize_homotopy import optimize_acqf_homotopy
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction


class TestHomotopyFailureRollback(BotorchTestCase):
    def test_reset_callback_failure_preserves_primary_error(self):
        for primary_failure in (False, True):
            with self.subTest(primary_failure=primary_failure):
                parameter = torch.nn.Parameter(torch.tensor(1.7))
                original = parameter.item()

                def callback():
                    if parameter.item() == original:
                        raise RuntimeError("reset callback failure")

                homotopy = Homotopy(
                    [HomotopyParameter(parameter, FixedHomotopySchedule([1.0, 0.5]))],
                    callbacks=[callback],
                )
                acquisition = MockAcquisitionFunction()
                expected = (
                    "primary optimizer failure"
                    if primary_failure
                    else "reset callback failure"
                )
                with mock.patch(
                    "botorch.optim.optimize_homotopy.optimize_acqf",
                    side_effect=RuntimeError("primary optimizer failure")
                    if primary_failure
                    else None,
                    return_value=(torch.tensor([[[0.3]]]), torch.tensor([1.0])),
                ):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        optimize_acqf_homotopy(
                            acquisition,
                            bounds=torch.tensor([[0.0], [1.0]]),
                            q=1,
                            num_restarts=1,
                            raw_samples=4,
                            homotopy=homotopy,
                        )
                self.assertEqual(parameter.item(), original)

    def test_optimization_failure_restores_parameters_and_pending(self):
        for q in (1, 2):
            for with_pending in (False, True):
                with self.subTest(q=q, with_pending=with_pending):
                    parameter = torch.nn.Parameter(torch.tensor(1.7))
                    homotopy = Homotopy(
                        [
                            HomotopyParameter(
                                parameter, FixedHomotopySchedule([1.0, 0.5])
                            )
                        ]
                    )
                    acquisition = MockAcquisitionFunction()
                    pending = torch.tensor([[0.95]]) if with_pending else None
                    acquisition.set_X_pending(pending)
                    original = acquisition.X_pending
                    calls = 0

                    def optimize(**kwargs):
                        nonlocal calls
                        calls += 1
                        # Two schedule steps plus one final solve per candidate.
                        # For q=2, fail only after the first pending update.
                        if calls == (2 if q == 1 else 5):
                            raise RuntimeError("deliberate stage failure")
                        return torch.tensor([[[0.3]]]), torch.tensor([1.0])

                    with (
                        mock.patch(
                            "botorch.optim.optimize_homotopy.optimize_acqf",
                            side_effect=optimize,
                        ),
                        self.assertRaisesRegex(
                            RuntimeError, "deliberate stage failure"
                        ),
                    ):
                        optimize_acqf_homotopy(
                            acquisition,
                            bounds=torch.tensor([[0.0], [1.0]]),
                            q=q,
                            num_restarts=1,
                            raw_samples=4,
                            homotopy=homotopy,
                        )
                    if original is None:
                        self.assertIsNone(acquisition.X_pending)
                    else:
                        self.assertAllClose(acquisition.X_pending, original)
                    self.assertEqual(parameter.item(), torch.tensor(1.7).item())
