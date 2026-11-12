#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from itertools import product

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.additive_gp import OrthogonalAdditiveGP
from botorch.models.transforms.outcome import Log
from botorch.utils.testing import BotorchTestCase


class TestAdditiveComponentLayout(BotorchTestCase):
    def test_nonlinear_outcome_has_no_additive_components(self):
        X = torch.rand(5, 2, dtype=torch.double)
        model = OrthogonalAdditiveGP(X, 1 + X[..., :1], outcome_transform=Log())
        self.assertTrue(torch.isfinite(model.posterior(X[:2]).mean).all())
        with self.assertRaisesRegex(UnsupportedError, "linear outcome"):
            model.posterior(X[:2], infer_all_components=True)
        self.assertFalse(model._infer_all_components)

    def test_component_reconstruction_and_batching(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        shapes = [((), ()), ((), (2,)), ((2,), ()), ((2,), (3, 2))]
        for device, second_order, (training_batch, query_batch) in product(
            devices, (False, True), shapes
        ):
            with self.subTest(
                device=device,
                second_order=second_order,
                training_batch=training_batch,
                query_batch=query_batch,
            ):
                torch.manual_seed(19)
                tkwargs = {"dtype": torch.double, "device": device}
                X = torch.rand(*training_batch, 5, 2, **tkwargs)
                Y = 11 + (X[..., :1] * 4).sin() + X[..., 1:].square()
                model = OrthogonalAdditiveGP(X, Y, second_order=second_order)
                standardized = deepcopy(model)
                del standardized.outcome_transform
                query = torch.rand(*query_batch, 3, 2, **tkwargs)
                full = model.posterior(query)
                components = model.posterior(query, infer_all_components=True)
                self.assertAllClose(components.mean.sum(0), full.mean)
                original_units = standardized.posterior(
                    query, infer_all_components=True
                )
                expected_mean = original_units.mean * model.outcome_transform.stdvs
                expected_mean[0] += model.outcome_transform.means
                self.assertAllClose(components.mean, expected_mean)
                self.assertAllClose(
                    components.distribution.covariance_matrix,
                    original_units.distribution.covariance_matrix
                    * model.outcome_transform._stdvs_sq,
                )
                transform = ScalarizedPosteriorTransform(
                    weights=torch.tensor([2.0], **tkwargs), offset=3.0
                )
                transformed = model.posterior(
                    query, infer_all_components=True, posterior_transform=transform
                )
                self.assertAllClose(transformed.mean, 2 * components.mean + 3)
