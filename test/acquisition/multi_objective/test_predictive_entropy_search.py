#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    _safe_update_omega,
    _update_damping,
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.exceptions import UnsupportedError
from botorch.models.model_list_gp_regression import ModelListGP
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


class TestQMultiObjectivePredictiveEntropySearch(BotorchTestCase):
    def test_initialization_errors(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        standardize_model = False
        for (
            dtype,
            num_objectives,
            use_model_list,
        ) in product(
            (torch.float, torch.double),
            (1, 2, 3),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            # test batched model
            train_X = torch.rand(4, 3, 2, **tkwargs)
            train_Y = torch.rand(4, 3, num_objectives, **tkwargs)
            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
            )
            num_pareto_samples = 3
            if num_objectives > 1:
                num_pareto_points = 4
            else:
                num_pareto_points = 1

            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )

            # test batch model error
            with self.assertRaises(NotImplementedError):
                qMultiObjectivePredictiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                )

            # test wrong Pareto set shape
            train_X = torch.rand(1, 2, **tkwargs)
            train_Y = torch.rand(1, num_objectives, **tkwargs)
            model = get_model(
                train_X=train_X,
                train_Y=train_Y,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
            )
            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )

            with self.assertRaises(UnsupportedError):
                qMultiObjectivePredictiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets.unsqueeze(0),
                )

            with self.assertRaises(UnsupportedError):
                qMultiObjectivePredictiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets.unsqueeze(-1),
                )

    def test_moo_predictive_entropy_search(self, use_model_list=False, maximize=False):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}

        for (
            dtype,
            num_objectives,
            standardize_model,
        ) in product(
            (torch.float, torch.double),
            (1, 2, 3),
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
            num_pareto_points = 1 if num_objectives == 1 else 4

            pareto_sets = dummy_sample_pareto_sets(
                model, num_pareto_samples, num_pareto_points
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]
                acq = qMultiObjectivePredictiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets,
                    maximize=maximize,
                    X_pending=X_pending,
                )

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

    def test_moo_predictive_entropy_search_maximize(self):
        self.test_moo_predictive_entropy_search(maximize=True)

    def test_moo_predictive_entropy_search_model_list(self):
        self.test_moo_predictive_entropy_search(use_model_list=True)

    def test_moo_predictive_entropy_search_model_list_maximize(self):
        self.test_moo_predictive_entropy_search(use_model_list=True, maximize=True)

    def test_update_damping(self):
        # test error when old and new covariance are not positive semi-definite
        tkwargs = {"device": self.device}

        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            cov_old = torch.ones(1, 2, 2, **tkwargs)
            cov_new = torch.ones(1, 2, 2, **tkwargs)
            damping_factor = torch.ones(1, **tkwargs)
            jitter = 0.0

            with self.assertRaises(ValueError):
                _update_damping(
                    nat_cov=cov_old,
                    nat_cov_new=cov_new,
                    damping_factor=damping_factor,
                    jitter=jitter,
                )

    def test_safe_omega_update(self):
        tkwargs = {"device": self.device}
        # test exception when EP fails because the jitter is too small and omega
        # update skips. This naturally depends on the precision.
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            N = 1
            P = 3
            M = 2
            mean_f = torch.zeros(2, M, N + P, **tkwargs)
            cov_f = torch.ones(2, M, N + P, N + P, **tkwargs)
            omega_f_nat_mean = torch.zeros(2, M, N + P, P, 2, **tkwargs)
            omega_f_nat_cov = torch.zeros(2, M, N + P, P, 2, 2, **tkwargs)
            maximize = True
            jitter = 0.0

            # The inversion of a factor of ``cov_f`` will fail spit out a
            # ``torch._C._LinAlgError`` error.
            omega_f_nat_mean_new, omega_f_nat_cov_new = _safe_update_omega(
                mean_f=mean_f,
                cov_f=cov_f,
                omega_f_nat_mean=omega_f_nat_mean,
                omega_f_nat_cov=omega_f_nat_cov,
                N=N,
                P=P,
                M=M,
                maximize=maximize,
                jitter=jitter,
            )

            self.assertTrue(torch.equal(omega_f_nat_mean, omega_f_nat_mean_new))
            self.assertTrue(torch.equal(omega_f_nat_cov, omega_f_nat_cov_new))
