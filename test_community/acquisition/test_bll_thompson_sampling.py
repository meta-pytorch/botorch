#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest.mock as mock

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.bll_thompson_sampling import BLLMaxPosteriorSampling
from botorch_community.models.vblls import VBLLModel


def _get_vbll_model(num_inputs: int = 2, num_hidden: int = 3, **tkwargs) -> VBLLModel:
    test_backbone = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, num_hidden),
    ).to(**tkwargs)
    model = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)
    return model


class TestBLLMaxPosteriorSampling(BotorchTestCase):
    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        tkwargs = {"device": self.device, "dtype": torch.float64}
        num_inputs = 2
        num_hidden = 3
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden, **tkwargs)
        sampler = BLLMaxPosteriorSampling(model=model)

        # Test default parameters
        self.assertEqual(sampler.model, model)
        self.assertEqual(sampler.num_restarts, 10)
        self.assertEqual(sampler.discrete_inputs, False)
        self.assertEqual(len(sampler.bounds), num_inputs)

        # Test with custom parameters
        bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], **tkwargs)
        num_restarts = 5
        sampler = BLLMaxPosteriorSampling(
            model=model, num_restarts=5, bounds=bounds, discrete_inputs=True
        )

        self.assertEqual(sampler.num_restarts, num_restarts)
        self.assertEqual(sampler.discrete_inputs, True)
        self.assertEqual(sampler.bounds[0], (0.0, 1.0))
        self.assertEqual(sampler.bounds[1], (0.0, 2.0))

    def test_initialization_errors(self) -> None:
        """Test error handling for invalid model types."""
        # Test with non-BLL model
        non_bll_model = torch.nn.Linear(2, 1)
        with self.assertRaises(ValueError):
            BLLMaxPosteriorSampling(model=non_bll_model)

    def test_multi_output_model_unsupported(self) -> None:
        """Test that multi-output models raise (no objective to select the max)."""
        multi_output_model = VBLLModel(
            in_features=2,
            hidden_features=3,
            num_layers=1,
            out_features=2,
            device=self.device,
        )
        for discrete_inputs in (False, True):
            with self.assertRaisesRegex(UnsupportedError, "single-output"):
                BLLMaxPosteriorSampling(
                    model=multi_output_model, discrete_inputs=discrete_inputs
                )

    def test_call_discrete_inputs(self) -> None:
        """Test __call__ method with discrete inputs."""
        tkwargs = {"device": self.device, "dtype": torch.float64}
        # Set random seed for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

        num_inputs = 2
        num_hidden = 3
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden, **tkwargs)
        sampler = BLLMaxPosteriorSampling(model=model, discrete_inputs=True)

        # Create candidate points
        X_cand = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
            **tkwargs,
        )

        # Get samples
        num_samples = 2
        X_next = sampler(X_cand=X_cand, num_samples=num_samples)

        # Check output shape and type
        self.assertEqual(X_next.shape, torch.Size([num_samples, num_inputs]))
        self.assertEqual(X_next.dtype, torch.float64)

    def test_call_continuous_inputs(self) -> None:
        """Test __call__ method with continuous inputs (optimization)."""
        torch.manual_seed(0)
        tkwargs = {"device": self.device, "dtype": torch.float64}
        num_inputs = 2
        num_hidden = 3
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden, **tkwargs)
        sampler = BLLMaxPosteriorSampling(model=model, num_restarts=3)

        # Get samples
        num_samples = 2
        X_next = sampler(num_samples=num_samples)

        # Check output shape and type
        self.assertEqual(X_next.shape, torch.Size([num_samples, num_inputs]))
        self.assertEqual(X_next.dtype, torch.float64)

        # Check that all samples are within default bounds [0, 1]
        self.assertTrue(torch.all(X_next >= 0))
        self.assertTrue(torch.all(X_next <= 1))

    def test_call_errors(self) -> None:
        """Test error handling in __call__ method."""
        num_inputs = 2
        num_hidden = 3
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden)

        # Test with discrete_inputs=True but no X_cand provided
        sampler = BLLMaxPosteriorSampling(model=model, discrete_inputs=True)
        with self.assertRaises(ValueError):
            sampler(num_samples=1)

        # Test with discrete_inputs=False but X_cand provided
        sampler = BLLMaxPosteriorSampling(model=model, discrete_inputs=False)
        X_cand = torch.rand(5, num_inputs, dtype=torch.float64)
        with self.assertRaises(ValueError):
            sampler(X_cand=X_cand, num_samples=1)

    def test_custom_bounds(self) -> None:
        """Test sampling with custom bounds."""
        tkwargs = {"device": self.device, "dtype": torch.float64}
        num_inputs = 3
        num_hidden = 4
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden, **tkwargs)
        bounds = torch.tensor([[-1.0, 0.0, 2.0], [2.0, 1.0, 5.0]], **tkwargs)
        sampler = BLLMaxPosteriorSampling(model=model, bounds=bounds)

        # Get samples
        num_samples = 4
        X_next = sampler(num_samples=num_samples)

        # Check output shape
        self.assertEqual(X_next.shape, torch.Size([num_samples, num_inputs]))

        # Check that all samples are within bounds
        for i in range(num_inputs):
            self.assertTrue(torch.all(X_next[:, i] >= bounds[0, i]))
            self.assertTrue(torch.all(X_next[:, i] <= bounds[1, i]))

    def test_scipy_optimization_failures(self) -> None:
        """Test error handling when all optimization attempts fail."""

        tkwargs = {"device": self.device, "dtype": torch.float64}
        num_inputs = 2
        num_hidden = 3
        model = _get_vbll_model(num_inputs=num_inputs, num_hidden=num_hidden, **tkwargs)
        sampler = BLLMaxPosteriorSampling(model=model, num_restarts=3)

        # Create a mock with success=False
        mock_result = mock.Mock()
        mock_result.success = False
        mock_result.x = np.zeros(num_inputs)
        mock_result.fun = 0.0

        # patch for scipy.optimize.minimize
        with mock.patch("scipy.optimize.minimize", return_value=mock_result):
            with self.assertRaises(RuntimeError):
                sampler(num_samples=1)
