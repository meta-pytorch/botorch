#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from unittest import mock
from unittest.mock import PropertyMock

import torch
from botorch.acquisition.objective import (
    IdentityMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.utils.test_helpers import get_fully_bayesian_model, get_model
from botorch.utils.testing import BotorchTestCase


class TestPathwiseThompsonSampling(BotorchTestCase):
    def _test_thompson_sampling_base(self, model: Model) -> None:
        acq = PathwiseThompsonSampling(
            model=model,
        )
        X_observed = model.train_inputs[0]
        input_dim = X_observed.shape[-1]
        test_X = torch.rand(4, 1, input_dim).to(X_observed)
        # re-draw samples and expect other output
        acq_pass = acq(test_X)
        self.assertTrue(acq_pass.shape == test_X.shape[:-2])

        acq_pass1 = acq(test_X)
        self.assertAllClose(acq_pass1, acq(test_X))
        acq.redraw(batch_size=acq.batch_size)
        acq_pass2 = acq(test_X)
        self.assertFalse(torch.allclose(acq_pass1, acq_pass2))

    def _test_thompson_sampling_multi_output(self, model: Model) -> None:
        # using multi-output model with a posterior transform
        with self.assertRaisesRegex(
            UnsupportedError,
            "Must specify an objective or a posterior transform when using ",
        ):
            PathwiseThompsonSampling(model=model)

        X_observed = model.train_inputs[0]
        input_dim = X_observed.shape[-1]
        tkwargs = {"device": self.device, "dtype": X_observed.dtype}
        test_X = torch.rand(4, 1, input_dim, **tkwargs)
        weigths = torch.ones(2, **tkwargs)
        posterior_transform = ScalarizedPosteriorTransform(weights=weigths)
        acqf = PathwiseThompsonSampling(
            model=model, posterior_transform=posterior_transform
        )
        self.assertIsInstance(acqf.objective, IdentityMCObjective)
        # testing that the acquisition function is deterministic and executes
        # with the posterior transform
        acq_val = acqf(test_X)
        acq_val_2 = acqf(test_X)
        self.assertAllClose(acq_val, acq_val_2)

        posterior_transform.scalarize = False
        with self.assertRaisesRegex(
            UnsupportedError, "posterior_transform must scalarize the output"
        ):
            PathwiseThompsonSampling(
                model=model, posterior_transform=posterior_transform
            )

    def _test_thompson_sampling_batch(self, model: Model) -> None:
        X_observed = model.train_inputs[0]
        input_dim = X_observed.shape[-1]
        batch_acq = PathwiseThompsonSampling(
            model=model,
        )
        self.assertEqual(batch_acq.batch_size, None)
        test_X = torch.rand(4, 5, input_dim).to(X_observed)
        batch_acq(test_X)
        self.assertEqual(batch_acq.batch_size, 5)
        test_X = torch.rand(4, 7, input_dim).to(X_observed)
        with self.assertRaisesRegex(
            ValueError,
            "The batch size of PathwiseThompsonSampling should not "
            "change during a forward pass - was 5, now 7. Please re-initialize "
            "the acquisition if you want to change the batch size.",
        ):
            batch_acq(test_X)

        batch_acq2 = PathwiseThompsonSampling(model)
        test_X = torch.rand(4, 7, 1, input_dim).to(X_observed)
        self.assertEqual(batch_acq2(test_X).shape, test_X.shape[:-2])

        batch_acq3 = PathwiseThompsonSampling(model)
        test_X = torch.rand(4, 7, 3, input_dim).to(X_observed)
        self.assertEqual(batch_acq3(test_X).shape, test_X.shape[:-2])

    def test_thompson_sampling_single_task(self):
        input_dim = 2
        for dtype, standardize_model in product(
            (torch.float32, torch.float64), (True, False)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(4, input_dim, **tkwargs)
            num_objectives = 1
            train_Y = 10 * torch.rand(4, num_objectives, **tkwargs)
            model = get_model(train_X, train_Y, standardize_model=standardize_model)
            self._test_thompson_sampling_base(model)
            self._test_thompson_sampling_batch(model)

            # multi-output model
            num_objectives = 2
            train_Y = 10 * torch.rand(4, num_objectives, **tkwargs)
            model = get_model(train_X, train_Y, standardize_model=standardize_model)
            self._test_thompson_sampling_multi_output(model)

    def test_thompson_sampling_fully_bayesian(self):
        input_dim = 2
        num_objectives = 1
        tkwargs = {"device": self.device, "dtype": torch.float64}
        train_X = torch.rand(4, input_dim, **tkwargs)
        train_Y = 10 * torch.rand(4, num_objectives, **tkwargs)
        fb_model = get_fully_bayesian_model(train_X, train_Y, num_models=3, **tkwargs)
        acqf = PathwiseThompsonSampling(model=fb_model)
        acqf_vals = acqf(train_X)
        acqf_vals_2 = acqf(train_X)
        self.assertAllClose(acqf_vals, acqf_vals_2)

        batch_shape = (2, 5)
        test_X = torch.randn(*batch_shape, *train_X.shape, **tkwargs)
        batched_output = acqf(test_X)
        self.assertEqual(batched_output.shape, batch_shape)
        batched_output_2 = acqf(test_X)
        self.assertAllClose(batched_output, batched_output_2)

        with mock.patch.object(
            type(acqf.model), "batch_shape", new_callable=PropertyMock
        ) as mock_batch_shape:
            mock_batch_shape.return_value = (2, 3)
            with self.assertRaisesRegex(
                NotImplementedError,
                "Ensemble models with more than one ensemble dimension",
            ):
                acqf.redraw(batch_size=2)
