#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.classifier import SoftKNNClassifierModel
from botorch.models.transforms.input import Normalize
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase


class TestSoftKNNClassifierModel(BotorchTestCase):
    def _make_data(self, n: int = 20, d: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        X = torch.rand(n, d, dtype=torch.float64)
        Y = torch.randint(0, 2, (n, 1), dtype=torch.float64)
        return X, Y

    def test_basic_construction(self) -> None:
        X, Y = self._make_data()
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        self.assertEqual(model.num_outputs, 1)
        self.assertIsNone(model.learned_sigma)

    def test_forward_shape_and_range(self) -> None:
        X, Y = self._make_data()
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        test_X = torch.rand(5, 3, dtype=torch.float64)
        out = model(test_X)
        self.assertEqual(out.shape, torch.Size([5, 1]))
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    def test_batched_input(self) -> None:
        X, Y = self._make_data()
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        test_X = torch.rand(2, 5, 3, dtype=torch.float64)
        out = model(test_X)
        self.assertEqual(out.shape, torch.Size([2, 5, 1]))

    def test_posterior(self) -> None:
        X, Y = self._make_data()
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        test_X = torch.rand(5, 3, dtype=torch.float64)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, EnsemblePosterior)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

    def test_learnable_sigma(self) -> None:
        X, Y = self._make_data()
        d = X.shape[-1]
        model = SoftKNNClassifierModel(
            train_X=X,
            train_Y=Y,
            learnable_sigma=True,
            sigma_epochs=10,
        )
        self.assertIsNotNone(model.learned_sigma)
        self.assertEqual(model.learned_sigma.shape, torch.Size([d]))
        # Forward should still work
        test_X = torch.rand(5, d, dtype=torch.float64)
        out = model(test_X)
        self.assertEqual(out.shape, torch.Size([5, 1]))
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    def test_construct_inputs(self) -> None:
        X, Y = self._make_data()
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=[f"x{i}" for i in range(X.shape[-1])],
            outcome_names=["y"],
        )
        inputs = SoftKNNClassifierModel.construct_inputs(
            training_data=dataset,
            sigma=0.5,
            learnable_sigma=True,
            sigma_epochs=50,
        )
        self.assertIn("train_X", inputs)
        self.assertIn("train_Y", inputs)
        self.assertEqual(inputs["sigma"], 0.5)
        self.assertTrue(inputs["learnable_sigma"])
        self.assertEqual(inputs["sigma_epochs"], 50)
        self.assertTrue(torch.equal(inputs["train_X"], X))
        self.assertTrue(torch.equal(inputs["train_Y"], Y))
        # Round-trip: construct model from inputs
        model = SoftKNNClassifierModel(**inputs)
        self.assertEqual(model.num_outputs, 1)

    def test_construct_inputs_does_not_mutate_kwargs(self) -> None:
        X, Y = self._make_data()
        dataset = SupervisedDataset(
            X=X,
            Y=Y,
            feature_names=[f"x{i}" for i in range(X.shape[-1])],
            outcome_names=["y"],
        )
        kwargs = {"sigma": 0.5, "extra_key": "value"}
        SoftKNNClassifierModel.construct_inputs(training_data=dataset, **kwargs)
        # kwargs should not be mutated
        self.assertIn("sigma", kwargs)
        self.assertIn("extra_key", kwargs)

    def test_input_transform(self) -> None:
        X, Y = self._make_data()
        d = X.shape[-1]
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        intf = Normalize(d=d, bounds=bounds)
        model = SoftKNNClassifierModel(
            train_X=X, train_Y=Y, sigma=0.3, input_transform=intf
        )
        test_X = torch.rand(5, d, dtype=torch.float64)
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

    def test_all_same_class(self) -> None:
        X = torch.rand(10, 3, dtype=torch.float64)
        Y = torch.ones(10, 1, dtype=torch.float64)
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        test_X = torch.rand(5, 3, dtype=torch.float64)
        out = model(test_X)
        # All class-1 training data → predictions should be ~1.0
        self.assertTrue((out > 0.99).all())

    def test_learnable_sigma_with_input_transform(self) -> None:
        X, Y = self._make_data()
        d = X.shape[-1]
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        intf = Normalize(d=d, bounds=bounds)
        model = SoftKNNClassifierModel(
            train_X=X,
            train_Y=Y,
            learnable_sigma=True,
            sigma_epochs=10,
            input_transform=intf,
        )
        self.assertIsNotNone(model.learned_sigma)
        self.assertEqual(model.learned_sigma.shape, torch.Size([d]))
        test_X = torch.rand(5, d, dtype=torch.float64)
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        self.assertTrue((posterior.mean >= 0).all())
        self.assertTrue((posterior.mean <= 1).all())

    def test_single_training_point(self) -> None:
        X = torch.rand(1, 3, dtype=torch.float64)
        Y = torch.tensor([[1.0]], dtype=torch.float64)
        model = SoftKNNClassifierModel(train_X=X, train_Y=Y, sigma=0.3)
        test_X = torch.rand(5, 3, dtype=torch.float64)
        out = model(test_X)
        self.assertEqual(out.shape, torch.Size([5, 1]))
        # Single class-1 point → predictions should be ~1.0
        self.assertTrue((out > 0.99).all())
