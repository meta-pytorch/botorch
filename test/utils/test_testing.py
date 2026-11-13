#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.test_functions.synthetic import ConstrainedGramacy
from botorch.utils.testing import (
    BotorchTestCase,
    MockModel,
    MockPosterior,
    sample_random_feasible,
    skip_if_import_error,
)


class TestMockPosterior(BotorchTestCase):
    def test_basic_logic(self) -> None:
        mp = MockPosterior()
        self.assertEqual(mp.device.type, "cpu")
        self.assertEqual(mp.dtype, torch.float32)
        self.assertEqual(mp._extended_shape(), torch.Size())
        self.assertEqual(
            MockPosterior(variance=torch.rand(2))._extended_shape(), torch.Size([2])
        )

    def test_passing_tensors(self) -> None:
        mean = torch.rand(2)
        variance = torch.eye(2)
        samples = torch.rand(1, 2)
        mp = MockPosterior(mean=mean, variance=variance, samples=samples)
        self.assertEqual(mp.device.type, "cpu")
        self.assertEqual(mp.dtype, torch.float32)
        self.assertTrue(torch.equal(mp.mean, mean))
        self.assertTrue(torch.equal(mp.variance, variance))
        self.assertTrue(torch.all(mp.rsample() == samples.unsqueeze(0)))
        self.assertTrue(
            torch.all(mp.rsample(torch.Size([2])) == samples.repeat(2, 1, 1))
        )

    def test_rsample_from_base_samples(self) -> None:
        mp = MockPosterior()
        with self.assertRaisesRegex(
            RuntimeError, "`sample_shape` disagrees with shape of `base_samples`."
        ):
            mp.rsample_from_base_samples(torch.zeros(2, 2), torch.zeros(3))


class TestMockModel(BotorchTestCase):
    def test_basic(self) -> None:
        mp = MockPosterior()
        mm = MockModel(mp)
        X = torch.empty(0)
        self.assertEqual(mm.posterior(X), mp)
        self.assertEqual(mm.num_outputs, 0)
        mm.state_dict()
        mm.load_state_dict()


class TestMisc(BotorchTestCase):
    def test_warning_filtering(self) -> None:
        with warnings.catch_warnings(record=True) as ws:
            # Model with unstandardized float data, which would typically raise
            # multiple warnings.
            SingleTaskGP(
                train_X=torch.rand(5, 2, dtype=torch.float) * 10,
                train_Y=torch.rand(5, 1, dtype=torch.float) * 10,
            )
        self.assertFalse(any(w.category == InputDataWarning for w in ws))

    def test_skip_if_import_error(self) -> None:
        error_msg = "import error"
        msg = (
            "Skipping test because module is not installed. Received the "
            f"following error: {error_msg}"
        )

        def func():
            raise ImportError(error_msg)

        safe_func = skip_if_import_error(func=func)
        with warnings.catch_warnings(record=True) as ws:
            safe_func()
        self.assertTrue(any(str(w.message) == msg for w in ws))

    def test_sample_random_feasible_ignores_constraint_noise(self) -> None:
        # With this much constraint noise, about half of the infeasible points
        # have a nonnegative noisy slack.
        f = ConstrainedGramacy(constraint_noise_std=100.0)
        torch.manual_seed(0)
        for dtype in (torch.float, torch.double):
            f.to(dtype=dtype)
            for _ in range(20):
                X = sample_random_feasible(f=f, dtype=dtype, device=self.device)
                self.assertEqual(X.dtype, dtype)
                self.assertTrue(f.is_feasible(X.unsqueeze(0), noise=False).item())
