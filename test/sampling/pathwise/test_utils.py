#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise.utils import (
    get_input_transform,
    get_output_transform,
    get_train_inputs,
    get_train_targets,
    InverseLengthscaleTransform,
    OutcomeUntransformer,
)
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, ScaleKernel


class TestTransforms(BotorchTestCase):
    def test_inverse_lengthscale_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.float64}
        kernel = MaternKernel(nu=2.5, ard_num_dims=3).to(**tkwargs)
        with self.assertRaisesRegex(RuntimeError, "does not implement `lengthscale`"):
            InverseLengthscaleTransform(ScaleKernel(kernel))

        x = torch.rand(3, 3, **tkwargs)
        transform = InverseLengthscaleTransform(kernel)
        self.assertTrue(transform(x).equal(kernel.lengthscale.reciprocal() * x))

    def test_outcome_untransformer(self):
        for untransformer in (
            OutcomeUntransformer(transform=Standardize(m=1), num_outputs=1),
            OutcomeUntransformer(transform=Standardize(m=2), num_outputs=2),
        ):
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                y = torch.rand(untransformer.num_outputs, 4, device=self.device)
            x = untransformer.transform(y.T)[0].T
            self.assertTrue(y.allclose(untransformer(x)))


class TestGetters(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(5, 2)
            train_Y = torch.randn(5, 2)

        self.models = []
        for num_outputs in (1, 2):
            self.models.append(
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                )
            )

            self.models.append(
                SingleTaskVariationalGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

    def test_get_input_transform(self):
        for model in self.models:
            self.assertIs(get_input_transform(model), model.input_transform)

    def test_get_output_transform(self):
        for model in self.models:
            transform = get_output_transform(model)
            self.assertIsInstance(transform, OutcomeUntransformer)
            self.assertIs(transform.transform, model.outcome_transform)

    def test_get_train_inputs(self):
        for model in self.models:
            model.train()
            X = (
                model.model.train_inputs[0]
                if isinstance(model, SingleTaskVariationalGP)
                else model.train_inputs[0]
            )
            Z = model.input_transform(X)
            train_inputs = get_train_inputs(model, transformed=False)
            self.assertIsInstance(train_inputs, tuple)
            self.assertEqual(len(train_inputs), 1)

            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

            model.eval()
            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))
            with delattr_ctx(model, "input_transform"), patch.object(
                model, "_original_train_inputs", new=None
            ):
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=False)[0]))
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            input_list = get_train_inputs(model_list)
            self.assertIsInstance(input_list, list)
            self.assertEqual(len(input_list), len(self.models))
            for model, train_inputs in zip(model_list.models, input_list):
                for a, b in zip(train_inputs, get_train_inputs(model)):
                    self.assertTrue(a.equal(b))

    def test_get_train_targets(self):
        for model in self.models:
            model.train()
            if isinstance(model, SingleTaskVariationalGP):
                F = model.model.train_targets
                Y = model.outcome_transform.untransform(F)[0].squeeze(dim=0)
            else:
                F = model.train_targets
                Y = OutcomeUntransformer(model.outcome_transform, model.num_outputs)(F)

            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))

            model.eval()
            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))
            with delattr_ctx(model, "outcome_transform"):
                self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
                self.assertTrue(F.equal(get_train_targets(model, transformed=False)))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            target_list = get_train_targets(model_list)
            self.assertIsInstance(target_list, list)
            self.assertEqual(len(target_list), len(self.models))
            for model, Y in zip(self.models, target_list):
                self.assertTrue(Y.equal(get_train_targets(model)))
