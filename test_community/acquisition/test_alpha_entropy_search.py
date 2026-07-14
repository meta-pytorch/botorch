#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase

from botorch_community.acquisition.alpha_entropy_search import qAlphaEntropySearch


class TestQAlphaEntropySearch(BotorchTestCase):
    def test_singleobj_alpha_entropy_search(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        estimation_types = ("LB",)

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
            acq = qAlphaEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type=estimation_type,
                num_samples=64,
                X_pending=None,
                condition_noiseless=condition_noiseless,
            )
            self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

            test_Xs = [
                torch.rand(4, 1, input_dim, **tkwargs),
                # AES only supports q=1! No X_pending nor q>1 eval points evaluated
                torch.rand(4, 5, 1, input_dim, **tkwargs),
            ]

            for j in range(len(test_Xs)):
                acq_X = acq(test_Xs[j])
                # assess shape
                self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

        acq = qAlphaEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=-torch.ones(1, **tkwargs)
            ),
        )
        self.assertTrue(torch.all(acq.optimal_output_values == -acq.optimal_outputs))
        acq_X = acq(test_Xs[-1])
        self.assertTrue(acq_X.shape == test_Xs[-1].shape[:-2])

        with self.assertRaises(ValueError):
            acq = qAlphaEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="NO_EST",
                num_samples=64,
                X_pending=None,
                condition_noiseless=condition_noiseless,
            )
            acq_X = acq(test_Xs[-1])

        # Support with fully bayesian models is not yet implemented. Thus, we
        # throw an error for now.
        fully_bayesian_model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        with self.assertRaises(NotImplementedError):
            acq = qAlphaEntropySearch(
                model=fully_bayesian_model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="LB",
            )

    def test_alpha_edge_values(self):
        # alpha=0.0 and alpha=1.0 are undefined for Amari's alpha-divergence, so
        # the constructor clamps them to eps and 1 - eps, respectively. Here we check
        # that this clamping avoids the division by zero in 1 / (alpha * (1 - alpha))
        # and produces finite acquisition values.
        tkwargs = {"device": self.device, "dtype": torch.double}
        input_dim = 2
        train_X = torch.rand(4, input_dim, **tkwargs)
        train_Y = torch.rand(4, 1, **tkwargs)
        model = get_model(train_X, train_Y, False, False)

        num_samples = 20
        optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)
        optimal_outputs = torch.rand(num_samples, 1, **tkwargs)
        test_X = torch.rand(4, 1, input_dim, **tkwargs)

        for alpha in (0.0, 1.0):
            acq = qAlphaEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                alpha=alpha,
            )
            acq_X = acq(test_X)
            self.assertTrue(torch.isfinite(acq_X).all())

    def test_q_greater_than_one_error(self):
        # We explicitly enforce q=1
        tkwargs = {"device": self.device, "dtype": torch.float}
        input_dim = 2
        train_X = torch.rand(4, input_dim, **tkwargs)
        train_Y = torch.rand(4, 1, **tkwargs)
        model = get_model(train_X, train_Y, False, False)

        num_samples = 20
        optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)
        optimal_outputs = torch.rand(num_samples, 1, **tkwargs)

        acq = qAlphaEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
        )
        with self.assertRaises(AssertionError):
            acq(torch.rand(4, 3, input_dim, **tkwargs))  # q=3 should fail

    def test_x_pending_incompatible_with_q1(self):
        # X_pending is concatenated onto the q dimension of X before evaluation,
        # so any non-empty X_pending pushes q above 1, which AES does not support.
        tkwargs = {"device": self.device, "dtype": torch.float}
        input_dim = 2
        train_X = torch.rand(4, input_dim, **tkwargs)
        train_Y = torch.rand(4, 1, **tkwargs)
        model = get_model(train_X, train_Y, False, False)

        num_samples = 20
        optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)
        optimal_outputs = torch.rand(num_samples, 1, **tkwargs)

        acq = qAlphaEntropySearch(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            X_pending=torch.rand(2, input_dim, **tkwargs),
        )
        with self.assertRaises(AssertionError):
            acq(torch.rand(4, 1, input_dim, **tkwargs))
