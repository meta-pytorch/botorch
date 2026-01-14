#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


def dummy_sample_pareto_frontiers(model):
    m = model.models[0] if isinstance(model, ModelListGP) else model
    return torch.rand(
        3,
        4,
        model.num_outputs,
        dtype=m.train_inputs[0].dtype,
        device=m.train_inputs[0].device,
    )


class TestQLowerBoundMultiObjectiveMaxValueEntropySearch(BotorchTestCase):
    def _base_test_lb_moo_max_value_entropy_search(self, estimation_type):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}

        for dtype, num_objectives, use_model_list, standardize_model in product(
            (torch.float, torch.double),
            (1, 2, 3),
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
            pareto_fronts = dummy_sample_pareto_frontiers(model)
            hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for X_pending in X_pending_list:
                acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                    model=model,
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                    X_pending=X_pending,
                )
                self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for test_X in test_Xs:
                    acq_X = acq(test_X)
                    # assess shape
                    self.assertTrue(acq_X.shape == test_X.shape[:-2])

    def test_lb_moo_max_value_entropy_search_0(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="0")

    def test_lb_moo_max_value_entropy_search_LB(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="LB")

    def test_lb_moo_max_value_entropy_search_LB2(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="LB2")

    def test_lb_moo_max_value_entropy_search_MC(self):
        self._base_test_lb_moo_max_value_entropy_search(estimation_type="MC")
