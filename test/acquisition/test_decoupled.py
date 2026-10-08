#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch.acquisition.decoupled import DecoupledAcquisitionFunction
from botorch.exceptions import BotorchTensorDimensionError, BotorchWarning
from botorch.logging import shape_to_str
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils.testing import BotorchTestCase


class DummyDecoupledAcquisitionFunction(DecoupledAcquisitionFunction):
    def forward(self, X):
        pass


class TestDecoupledAcquisitionFunction(BotorchTestCase):
    def test_decoupled_acquisition_function(self):
        msg = "Can't instantiate abstract class DecoupledAcquisitionFunction"
        with self.assertRaisesRegex(TypeError, msg):
            DecoupledAcquisitionFunction()
        # test raises error if model is not ModelList
        msg = "DummyDecoupledAcquisitionFunction requires using a ModelList."
        model = SingleTaskGP(
            torch.rand(1, 3, device=self.device), torch.rand(1, 2, device=self.device)
        )
        with self.assertRaisesRegex(ValueError, msg):
            DummyDecoupledAcquisitionFunction(model=model)
        m = SingleTaskGP(
            torch.rand(1, 3, device=self.device), torch.rand(1, 1, device=self.device)
        )
        model = ModelListGP(m, m)
        # basic test
        af = DummyDecoupledAcquisitionFunction(model=model)
        self.assertIs(af.model, model)
        self.assertIsNone(af.X_evaluation_mask)
        self.assertIsNone(af.X_pending)
        # test set X_evaluation_mask
        # test wrong number of outputs
        eval_mask = torch.randint(0, 2, (2, 3), device=self.device).bool()
        msg = (
            "Expected X_evaluation_mask to be `q x m`, but got shape"
            f" {shape_to_str(eval_mask.shape)}."
        )
        with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
            af.X_evaluation_mask = eval_mask
        # test more than 2 dimensions
        eval_mask.unsqueeze_(0)
        msg = (
            "Expected X_evaluation_mask to be `q x m`, but got shape"
            f" {shape_to_str(eval_mask.shape)}."
        )
        with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
            af.X_evaluation_mask = eval_mask

        # set eval_mask
        eval_mask = eval_mask[0, :, :2]
        af.X_evaluation_mask = eval_mask
        self.assertIs(af.X_evaluation_mask, eval_mask)

        # test set_X_pending
        X_pending = torch.rand(1, 1, device=self.device)
        msg = (
            "If `self.X_evaluation_mask` is not None, then "
            "`X_pending_evaluation_mask` must be provided."
        )
        with self.assertRaisesRegex(ValueError, msg):
            af.set_X_pending(X_pending=X_pending)
        af.X_evaluation_mask = None
        X_pending = X_pending.requires_grad_(True)
        with warnings.catch_warnings(record=True) as ws:
            af.set_X_pending(X_pending)
        self.assertEqual(af.X_pending, X_pending)
        self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)
        self.assertIsNone(af.X_evaluation_mask)

        # test setting X_pending with X_pending_evaluation_mask
        X_pending = torch.rand(3, 1, device=self.device)
        # test raises exception
        # wrong number of outputs, wrong number of dims, wrong number of rows
        for shape in ([3, 1], [1, 3, 2], [1, 2]):
            eval_mask = torch.randint(0, 2, shape, device=self.device).bool()
            msg = (
                f"Expected `X_pending_evaluation_mask` of shape `{X_pending.shape[0]} "
                f"x {model.num_outputs}`, but got "
                f"{shape_to_str(eval_mask.shape)}."
            )

            with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
                af.set_X_pending(
                    X_pending=X_pending, X_pending_evaluation_mask=eval_mask
                )
        eval_mask = torch.randint(0, 2, (3, 2), device=self.device).bool()
        af.set_X_pending(X_pending=X_pending, X_pending_evaluation_mask=eval_mask)
        self.assertTrue(torch.equal(af.X_pending, X_pending))
        self.assertIs(af.X_pending_evaluation_mask, eval_mask)

        # test construct_evaluation_mask
        # X_evaluation_mask is None
        X = torch.rand(4, 5, 2, device=self.device)
        X_eval_mask = af.construct_evaluation_mask(X=X)
        expected_eval_mask = torch.cat(
            [torch.ones(X.shape[1:], dtype=torch.bool, device=self.device), eval_mask],
            dim=0,
        )
        self.assertTrue(torch.equal(X_eval_mask, expected_eval_mask))
        # test X_evaluation_mask is not None
        # test wrong shape
        af.X_evaluation_mask = torch.zeros(1, 2, dtype=bool, device=self.device)
        msg = "Expected the -2 dimension of X and X_evaluation_mask to match."
        with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
            af.construct_evaluation_mask(X=X)
        af.X_evaluation_mask = torch.randint(0, 2, (5, 2), device=self.device).bool()
        X_eval_mask = af.construct_evaluation_mask(X=X)
        expected_eval_mask = torch.cat([af.X_evaluation_mask, eval_mask], dim=0)
        self.assertTrue(torch.equal(X_eval_mask, expected_eval_mask))

        # test setting X_pending as None
        af.set_X_pending(X_pending=None, X_pending_evaluation_mask=None)
        self.assertIsNone(af.X_pending)
        self.assertIsNone(af.X_pending_evaluation_mask)

        # test construct_evaluation_mask when X_pending is None
        self.assertTrue(
            torch.equal(af.construct_evaluation_mask(X=X), af.X_evaluation_mask)
        )

    def test_set_X_pending_without_mask(self):
        # Pending points set without a mask are evaluated on all outcomes, so the
        # mask of previously set pending points must not be kept.
        m = SingleTaskGP(
            torch.rand(1, 3, device=self.device), torch.rand(1, 1, device=self.device)
        )
        af = DummyDecoupledAcquisitionFunction(model=ModelListGP(m, m))
        af.set_X_pending(
            X_pending=torch.rand(1, 3, device=self.device),
            X_pending_evaluation_mask=torch.tensor([[True, False]], device=self.device),
        )
        X_pending = torch.rand(3, 3, device=self.device)
        af.set_X_pending(X_pending=X_pending)
        self.assertTrue(torch.equal(af.X_pending, X_pending))
        self.assertIsNone(af.X_pending_evaluation_mask)
        self.assertIsNone(
            af.construct_evaluation_mask(X=torch.rand(2, 3, device=self.device))
        )
