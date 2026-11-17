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

    def test_mc_independent_candidates_are_additive(self):
        torch.manual_seed(0)
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(6, 2, **tkwargs)
        model = SingleTaskGP(train_X, torch.sin(5 * train_X[:, :1]))
        optimal_inputs = torch.rand(4, 2, **tkwargs)
        with torch.no_grad():
            optimal_outputs = model.posterior(optimal_inputs).mean + 1
        acq = qJointEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            estimation_type="MC",
            num_samples=4096,
        )
        acq.sampler = SobolQMCNormalSampler(torch.Size([4096]), seed=0)
        # The distant point is independent of both the other candidate and optima.
        X = torch.tensor([[0.3, 0.6], [50.0, 50.0]], **tkwargs)
        with torch.no_grad():
            joint = acq(X)
            separate = acq(X[:1]) + acq(X[1:])
        self.assertLess((joint - separate).abs().item(), 0.02)
