#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


class TestQPredictiveEntropySearch(BotorchTestCase):
    def test_predictive_entropy_search(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        num_objectives = 1

        for (
            dtype,
            use_model_list,
            standardize_model,
            maximize,
        ) in product(
            (torch.float, torch.double),
            (False, True),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)
            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
            )
            num_samples = 20
            optimal_inputs = torch.rand(num_samples, input_dim, **tkwargs)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]
                acq = qPredictiveEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    maximize=maximize,
                    X_pending=X_pending,
                )

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

    def test_predictive_entropy_search_observation_noise(self):
        # The optimality conditions apply to the latent function, not to the noisy
        # observations: as the observation noise grows, an observation carries
        # (almost) no information about the optimal inputs.
        torch.manual_seed(0)
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(6, 2, **tkwargs)
        train_Y = torch.sin(5 * train_X[:, :1])
        model = get_model(train_X=train_X, train_Y=train_Y)
        model.likelihood.noise = 1e4
        with torch.no_grad():
            acq = qPredictiveEntropySearch(
                model=model, optimal_inputs=torch.rand(2, 2, **tkwargs)
            )
            acq_X = acq(torch.rand(4, 1, 2, **tkwargs))
        self.assertTrue((acq_X.abs() < 1e-2).all())
