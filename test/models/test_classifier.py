#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import torch
from botorch.models.classifier import (
    RandomForestClassifierModel,
    SVCClassifierModel,
    XGBoostClassifierModel,
)
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase


class TestClassifierModels(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Create training data for binary classification
        # Features: 20 samples, 3 dimensions
        self.X_train = torch.randn(20, 3, dtype=torch.float64)

        # Binary labels for feasibility (0 or 1)
        self.y_feasible = torch.randint(0, 2, (20, 1), dtype=torch.float64)

        # Create datasets
        self.feasibility_dataset = SupervisedDataset(
            X=self.X_train,
            Y=self.y_feasible,
            feature_names=["x0", "x1", "x2"],
            outcome_names=["is_feasible"],
        )

    def test_random_forest_construct_inputs(self) -> None:
        """Test RandomForestClassifierModel construction from training data."""
        model_inputs = RandomForestClassifierModel.construct_inputs(
            training_data=self.feasibility_dataset,
            n_estimators=50,
            max_depth=5,
        )

        # Check that the required keys are present
        self.assertIn("f", model_inputs)
        self.assertIn("num_outputs", model_inputs)
        self.assertEqual(model_inputs["num_outputs"], 1)

        # Test prediction
        X_test = torch.randn(5, 3, dtype=torch.float64)
        predictions = model_inputs["f"](X_test)

        # Check prediction shape
        self.assertEqual(predictions.shape, (5, 1))

        # Check predictions are probabilities (between 0 and 1)
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions <= 1).all())

    def test_svc_construct_inputs(self) -> None:
        """Test that SVCClassifierModel can be constructed from training data."""
        model_inputs = SVCClassifierModel.construct_inputs(
            training_data=self.feasibility_dataset,
            kernel="rbf",
            C=1.0,
        )

        self.assertIn("f", model_inputs)
        self.assertIn("num_outputs", model_inputs)
        self.assertEqual(model_inputs["num_outputs"], 1)

        # Test prediction
        X_test = torch.randn(5, 3, dtype=torch.float64)
        predictions = model_inputs["f"](X_test)

        self.assertEqual(predictions.shape, (5, 1))
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions <= 1).all())

    def test_xgboost_construct_inputs(self) -> None:
        """Test that XGBoostClassifierModel can be constructed from training data."""
        model_inputs = XGBoostClassifierModel.construct_inputs(
            training_data=self.feasibility_dataset,
            n_estimators=50,
            max_depth=4,
        )

        self.assertIn("f", model_inputs)
        self.assertIn("num_outputs", model_inputs)
        self.assertEqual(model_inputs["num_outputs"], 1)

        # Test prediction
        X_test = torch.randn(5, 3, dtype=torch.float64)
        predictions = model_inputs["f"](X_test)

        self.assertEqual(predictions.shape, (5, 1))
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions <= 1).all())

    def test_random_forest_model_instantiation(self) -> None:
        """Test creating a RandomForestClassifierModel instance."""
        model_inputs = RandomForestClassifierModel.construct_inputs(
            training_data=self.feasibility_dataset,
            n_estimators=100,
        )

        model = RandomForestClassifierModel(**model_inputs)

        # Test forward pass
        X_test = torch.randn(5, 3, dtype=torch.float64)
        predictions = model(X_test)

        self.assertEqual(predictions.shape, (5, 1))

        # Test posterior
        posterior = model.posterior(X_test)
        self.assertEqual(posterior.mean.shape, (5, 1))


class TestClassifierModelEdgeCases(BotorchTestCase):
    """Test edge cases and error handling for classifier models."""

    def test_random_forest_with_2d_labels(self) -> None:
        """Test that 2D labels are correctly flattened for sklearn."""
        X_train = torch.randn(20, 3, dtype=torch.float64)
        y_train = torch.randint(0, 2, (20, 1), dtype=torch.float64)  # 2D labels

        dataset = SupervisedDataset(
            X=X_train,
            Y=y_train,
            feature_names=["x0", "x1", "x2"],
            outcome_names=["is_feasible"],
        )

        # Should handle 2D labels without error
        model_inputs = RandomForestClassifierModel.construct_inputs(
            training_data=dataset
        )

        X_test = torch.randn(5, 3, dtype=torch.float64)
        predictions = model_inputs["f"](X_test)

        self.assertEqual(predictions.shape, (5, 1))

    def test_device_dtype_preservation(self) -> None:
        """Test that predictions preserve device and dtype of input."""
        X_train = torch.randn(20, 3, dtype=torch.float64)
        y_train = torch.randint(0, 2, (20, 1), dtype=torch.float64)

        dataset = SupervisedDataset(
            X=X_train,
            Y=y_train,
            feature_names=["x0", "x1", "x2"],
            outcome_names=["is_feasible"],
        )

        model_inputs = RandomForestClassifierModel.construct_inputs(
            training_data=dataset
        )

        # Test with float32
        X_test_f32 = torch.randn(5, 3, dtype=torch.float32)
        predictions_f32 = model_inputs["f"](X_test_f32)
        self.assertEqual(predictions_f32.dtype, torch.float32)

        # Test with float64
        X_test_f64 = torch.randn(5, 3, dtype=torch.float64)
        predictions_f64 = model_inputs["f"](X_test_f64)
        self.assertEqual(predictions_f64.dtype, torch.float64)
