#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


class TestQJointEntropySearch(BotorchTestCase):
    def test_singleobj_joint_entropy_search(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        estimation_types = ("LB", "MC")

        num_objectives = 1
        for (
            dtype,
            estimation_type,
            use_model_list,
            standardize_model,
            condition_noiseless,
        ) in product(
            (torch.float, torch.double),
            estimation_types,
            (False, True),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(train_X, train_Y, standardize_model, use_model_list)

            num_samples = 20

            optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)
            optimal_outputs = torch.rand(num_samples, num_objectives, **tkwargs)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = qJointEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    estimation_type=estimation_type,
                    num_samples=64,
                    X_pending=X_pending,
                    condition_noiseless=condition_noiseless,
                )
                self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        acq = qJointEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=-torch.ones(1, **tkwargs)
            ),
        )
        self.assertTrue(torch.all(acq.optimal_output_values == -acq.optimal_outputs))
        acq_X = acq(test_Xs[j])
        self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qJointEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="NO_EST",
                num_samples=64,
                X_pending=X_pending,
                condition_noiseless=condition_noiseless,
            )
            acq_X = acq(test_Xs[j])

        # Support with fully bayesian models is not yet implemented. Thus, we
        # throw an error for now.
        fully_bayesian_model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        with self.assertRaises(NotImplementedError):
            acq = qJointEntropySearch(
                model=fully_bayesian_model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="LB",
            )

    def test_conditioning_with_input_transform(self):
        # ``condition_on_observations`` applies the input transform, so the
        # optimal inputs must not be transformed beforehand.
        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(0)
        train_X = 10 * torch.rand(8, 1, **tkwargs)
        bounds = torch.tensor([[0.0], [10.0]], **tkwargs)
        model = SingleTaskGP(
            train_X, torch.sin(train_X), input_transform=Normalize(d=1, bounds=bounds)
        )
        model.eval()
        optimal_inputs = torch.tensor([[2.0], [5.0]], **tkwargs)
        optimal_outputs = torch.tensor([[1.5], [3.0]], **tkwargs)
        for condition_noiseless, use_model_list in product(
            (True, False), (True, False)
        ):
            acq = qJointEntropySearch(
                model=ModelListGP(model) if use_model_list else model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                condition_noiseless=condition_noiseless,
            )
            conditional_model = acq.conditional_model
            if use_model_list:
                conditional_model = conditional_model.models[0]
            self.assertAllClose(
                conditional_model.train_inputs[0][..., -1, :], optimal_inputs / 10
            )
            if not use_model_list:
                self.assertAllClose(
                    conditional_model._original_train_inputs[..., -1, :],
                    optimal_inputs,
                )

    def test_negating_posterior_transform(self):
        # Maximizing ``-f`` through a negating posterior transform is equivalent
        # to maximizing ``g = -f`` with a model of ``-f``.
        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(0)
        train_X = torch.rand(8, 1, **tkwargs)
        train_Y = torch.sin(6 * train_X)
        model = SingleTaskGP(train_X, train_Y).eval()
        neg_model = SingleTaskGP(train_X, -train_Y).eval()
        optimal_inputs = torch.tensor([[0.8], [0.75]], **tkwargs)
        # Values of ``f`` at the optimal inputs (i.e. the minimizers of ``f``).
        optimal_outputs = torch.tensor([[-1.3], [-1.2]], **tkwargs)
        posterior_transform = ScalarizedPosteriorTransform(
            weights=-torch.ones(1, **tkwargs)
        )
        X = torch.rand(5, 1, 1, **tkwargs)
        for estimation_type in ("LB", "MC"):
            with self.subTest(estimation_type=estimation_type):
                torch.manual_seed(1)
                acq = qJointEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    posterior_transform=posterior_transform,
                    estimation_type=estimation_type,
                )
                torch.manual_seed(1)
                neg_acq = qJointEntropySearch(
                    model=neg_model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=-optimal_outputs,
                    estimation_type=estimation_type,
                )
                with torch.no_grad():
                    self.assertAllClose(acq(X), neg_acq(X))
