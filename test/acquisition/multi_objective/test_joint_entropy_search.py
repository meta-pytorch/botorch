#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.multi_objective.joint_entropy_search import (
    LowerBoundMultiObjectiveEntropySearch,
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition

from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


def dummy_sample_pareto_sets(model, num_pareto_samples, num_pareto_points):
    m = model.models[0] if isinstance(model, ModelListGP) else model
    input_dim = m.train_inputs[0].shape[-1]
    tkwargs = {"dtype": m.train_inputs[0].dtype, "device": m.train_inputs[0].device}
    return torch.rand(
        num_pareto_samples,
        num_pareto_points,
        input_dim,
        **tkwargs,
    )


def dummy_sample_pareto_fronts(model, num_pareto_samples, num_pareto_points):
    m = model.models[0] if isinstance(model, ModelListGP) else model
    num_objectives = model.num_outputs
    tkwargs = {"dtype": m.train_inputs[0].dtype, "device": m.train_inputs[0].device}
    return torch.rand(
        num_pareto_samples,
        num_pareto_points,
        num_objectives,
        **tkwargs,
    )


class DummyLowerBoundMultiObjectiveEntropySearch(LowerBoundMultiObjectiveEntropySearch):
    def _compute_posterior_statistics(self, X):
        pass

    def _compute_monte_carlo_variables(self, posterior):
        pass

    def forward(self, X):
        pass


class TestLowerBoundMultiObjectiveEntropySearch(BotorchTestCase):
    def test_abstract_raises(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        estimation_types = ("0", "LB", "LB2", "MC", "Dummy")
        for (
            dtype,
            num_objectives,
            estimation_type,
            use_model_list,
            standardize_model,
        ) in product(
            (torch.float, torch.double),
            (1, 2, 3),
            estimation_types,
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            # test batched model
            train_X = torch.rand(4, 3, 2, **tkwargs)
            train_Y = torch.rand(4, 3, num_objectives, **tkwargs)
            model = SingleTaskGP(train_X, train_Y)
            num_pareto_samples = 3
            num_pareto_points = 1 if num_objectives == 1 else 4

            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )
            pareto_fronts = dummy_sample_pareto_fronts(
                model, num_pareto_samples, num_pareto_points
            )
            hypercell_bounds = torch.rand(
                num_pareto_samples, 2, 4, num_objectives, **tkwargs
            )

            with self.assertRaises(NotImplementedError):
                DummyLowerBoundMultiObjectiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    pareto_fronts=pareto_fronts,
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                )

            # test wrong Pareto shape and hypercell bounds
            train_X = torch.rand(1, 2, **tkwargs)
            train_Y = torch.rand(1, num_objectives, **tkwargs)
            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
            )

            num_pareto_samples = 3
            num_pareto_points = 4

            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )
            pareto_fronts = dummy_sample_pareto_fronts(
                model, num_pareto_samples, num_pareto_points
            )
            hypercell_bounds = torch.rand(
                num_pareto_samples, 2, 4, num_objectives, **tkwargs
            )

            with self.assertRaises(UnsupportedError):
                DummyLowerBoundMultiObjectiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets.unsqueeze(0),
                    pareto_fronts=pareto_fronts,
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                )
            with self.assertRaises(UnsupportedError):
                DummyLowerBoundMultiObjectiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    pareto_fronts=pareto_fronts.unsqueeze(0),
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                )
            with self.assertRaises(UnsupportedError):
                DummyLowerBoundMultiObjectiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    pareto_fronts=pareto_fronts,
                    hypercell_bounds=hypercell_bounds.unsqueeze(0),
                    estimation_type=estimation_type,
                    num_samples=64,
                )

            if estimation_type == "Dummy":
                with self.assertRaises(NotImplementedError):
                    DummyLowerBoundMultiObjectiveEntropySearch(
                        model=model,
                        pareto_sets=pareto_sets,
                        pareto_fronts=pareto_fronts,
                        hypercell_bounds=hypercell_bounds,
                        estimation_type=estimation_type,
                        num_samples=64,
                    )
            else:
                DummyLowerBoundMultiObjectiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    pareto_fronts=pareto_fronts,
                    hypercell_bounds=hypercell_bounds,
                    estimation_type=estimation_type,
                    num_samples=64,
                )


class TestQLowerBoundMultiObjectiveJointEntropySearch(BotorchTestCase):
    def _base_test_lb_moo_joint_entropy_search(self, estimation_type):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}

        for (
            dtype,
            num_objectives,
            use_model_list,
            standardize_model,
        ) in product(
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
            num_pareto_samples = 3
            num_pareto_points = 4

            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )
            pareto_fronts = dummy_sample_pareto_fronts(
                model, num_pareto_samples, num_pareto_points
            )
            hypercell_bounds = compute_sample_box_decomposition(pareto_fronts)

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for X_pending in X_pending_list:
                acq = qLowerBoundMultiObjectiveJointEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    pareto_fronts=pareto_fronts,
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

    def test_lb_moo_joint_entropy_search_0(self):
        self._base_test_lb_moo_joint_entropy_search(estimation_type="0")

    def test_lb_moo_joint_entropy_search_LB(self):
        self._base_test_lb_moo_joint_entropy_search(estimation_type="LB")

    def test_lb_moo_joint_entropy_search_LB2(self):
        self._base_test_lb_moo_joint_entropy_search(estimation_type="LB2")

    def test_lb_moo_joint_entropy_search_MC(self):
        self._base_test_lb_moo_joint_entropy_search(estimation_type="MC")
