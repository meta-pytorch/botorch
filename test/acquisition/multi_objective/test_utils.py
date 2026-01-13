#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from itertools import product
from unittest import mock

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    get_default_partitioning_alpha,
    prune_inferior_points_multi_objective,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import get_matheron_path_model
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


class TestUtils(BotorchTestCase):
    def test_get_default_partitioning_alpha(self):
        for m in range(2, 9):
            expected_val = 0.0 if m < 5 else 10 ** (-2 if m >= 6 else -3)
            self.assertEqual(
                expected_val, get_default_partitioning_alpha(num_objectives=m)
            )
        # In ``BotorchTestCase.setUp`` warnings are filtered, so here we
        # remove the filter to ensure a warning is issued as expected.
        warnings.resetwarnings()
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(0.01, get_default_partitioning_alpha(num_objectives=7))
        self.assertEqual(len(ws), 1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor, X: Tensor | None) -> Tensor:
        return samples


class TestMultiObjectiveUtils(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = mock.MagicMock()
        self.objective = DummyMCMultiOutputObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.ref_point = [0.0, 0.0]
        self.Y = torch.tensor([[1.0, 2.0]])
        self.seed = 1

    def test_prune_inferior_points_multi_objective(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.rand(3, 2, **tkwargs)
            ref_point = torch.tensor([0.25, 0.25], **tkwargs)
            # the event shape is ``q x m`` = 3 x 2
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # test that a batched X raises errors
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X.expand(2, 3, 2), ref_point=ref_point
                )
            # test that a batched model raises errors (event shape is ``q x m`` = 3 x m)
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 2)))
            with self.assertRaises(UnsupportedError):
                prune_inferior_points_multi_objective(
                    model=mm2, X=X, ref_point=ref_point
                )
            # test that invalid max_frac is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points_multi_objective(
                    model=mm, X=X, max_frac=1.1, ref_point=ref_point
                )
            # test that invalid X is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points_multi_objective(
                    model=mm, X=torch.empty(0, 0), ref_point=ref_point
                )
            # test basic behaviour
            X_pruned = prune_inferior_points_multi_objective(
                model=mm, X=X, ref_point=ref_point
            )
            # test constraints
            objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])
            samples_constrained = torch.tensor(
                [[1.0, 2.0, -1.0], [2.0, 1.0, -1.0], [3.0, 4.0, 1.0]], **tkwargs
            )
            mm_constrained = MockModel(MockPosterior(samples=samples_constrained))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm_constrained,
                X=X,
                ref_point=ref_point,
                objective=objective,
                constraints=[lambda Y: Y[..., -1]],
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

            # test non-repeated samples (requires mocking out MockPosterior's rsample)
            samples = torch.tensor(
                [[[3.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [1.0]]],
                device=self.device,
                dtype=dtype,
            )
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X))
            # test max_frac limiting
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point, max_frac=2 / 3
                )
            self.assertTrue(
                torch.equal(
                    torch.sort(X_pruned, stable=True).values,
                    torch.sort(X[:2], stable=True).values,
                )
            )
            # test that zero-probability is in fact pruned
            samples[2, 0, 0] = 10
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points_multi_objective(
                    model=mm, X=X, ref_point=ref_point
                )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

            # test marginalize_dim and constraints
            samples = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], **tkwargs)
            samples = samples.unsqueeze(-3).expand(
                *samples.shape[:-2],
                2,
                *samples.shape[-2:],
            )
            mm = MockModel(MockPosterior(samples=samples))
            X_pruned = prune_inferior_points_multi_objective(
                model=mm,
                X=X,
                ref_point=ref_point,
                objective=objective,
                constraints=[lambda Y: Y[..., -1] - 3.0],
                marginalize_dim=-3,
            )
            self.assertTrue(torch.equal(X_pruned, X[:2]))

    def test_compute_sample_box_decomposition(self):
        tkwargs = {"device": self.device}
        for dtype, maximize in product((torch.float, torch.double), (True, False)):
            tkwargs["dtype"] = dtype

            # test error when inputting incorrect Pareto front
            X = torch.rand(4, 3, 2, 1, **tkwargs)
            with self.assertRaises(UnsupportedError):
                compute_sample_box_decomposition(pareto_fronts=X, maximize=maximize)

            # test single and multi-objective setting
            for num_objectives in (1, 5):
                X = torch.rand(4, 3, num_objectives, **tkwargs)
                bd1 = compute_sample_box_decomposition(
                    pareto_fronts=X, maximize=maximize
                )

                # assess shape
                self.assertTrue(bd1.ndim == 4)
                self.assertTrue(bd1.shape[-1] == num_objectives)
                self.assertTrue(bd1.shape[-3] == 2)
                if num_objectives == 1:
                    self.assertTrue(bd1.shape[-2] == 1)

                # assess whether upper bound is greater than lower bound
                self.assertTrue(torch.all(bd1[:, 1, ...] - bd1[:, 0, ...] >= 0))

                # test constrained setting
                num_constraints = 7
                bd2 = compute_sample_box_decomposition(
                    pareto_fronts=X,
                    maximize=maximize,
                    num_constraints=num_constraints,
                )

                # assess shape
                self.assertTrue(bd2.ndim == 4)
                self.assertTrue(bd2.shape[-1] == num_objectives + num_constraints)
                self.assertTrue(bd2.shape[-2] == bd1.shape[-2] + 1)
                self.assertTrue(bd2.shape[-3] == 2)

                # assess whether upper bound is greater than lower bound
                self.assertTrue(torch.all(bd2[:, 1, ...] - bd2[:, 0, ...] >= 0))

                # the constraint padding should not change the box-decomposition
                # if the box-decomposition procedure is not random
                self.assertTrue(torch.equal(bd1, bd2[..., 0:-1, 0:num_objectives]))

                # test with a specified optimum
                opt_X = 2.0 if maximize else -3.0

                X[:, 0, :] = opt_X
                bd3 = compute_sample_box_decomposition(
                    pareto_fronts=X, maximize=maximize
                )

                # check optimum
                if maximize:
                    self.assertTrue(torch.all(bd3[:, 1, ...] == opt_X))
                else:
                    self.assertTrue(torch.all(bd3[:, 0, ...] == opt_X))


def get_model(
    dtype,
    device,
    num_points,
    input_dim,
    num_objectives,
    use_model_list,
    standardize_model,
):
    torch.manual_seed(123)
    tkwargs = {"dtype": dtype, "device": device}
    train_X = torch.rand(num_points, input_dim, **tkwargs)
    train_Y = torch.rand(num_points, num_objectives, **tkwargs)

    if standardize_model:
        if use_model_list:
            outcome_transform = Standardize(m=1)
        else:
            outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None

    if use_model_list and num_objectives > 1:
        model = ModelListGP(
            *[
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],
                    outcome_transform=outcome_transform,
                )
                for i in range(num_objectives)
            ]
        )
    else:
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )

    return model.eval(), train_X, train_Y


class TestThompsonSampling(BotorchTestCase):
    def test_random_search_optimizer(self):
        torch.manual_seed(1)
        input_dim = 3
        num_initial = 5
        tkwargs = {"device": self.device}
        optimizer_kwargs = {"pop_size": 1000, "max_tries": 5}

        for (
            dtype,
            maximize,
            num_objectives,
            use_model_list,
            standardize_model,
        ) in product(
            (torch.float, torch.double),
            (True, False),
            (1, 2),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype
            num_points = num_objectives

            model, X, Y = get_model(
                num_points=num_initial,
                input_dim=input_dim,
                num_objectives=num_objectives,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
                **tkwargs,
            )

            model_sample = get_matheron_path_model(model=model)

            input_dim = X.shape[-1]
            # fake bounds
            bounds = torch.zeros((2, input_dim), **tkwargs)
            bounds[1] = 1.0

            pareto_set, pareto_front = random_search_optimizer(
                model=model_sample,
                bounds=bounds,
                num_points=num_points,
                maximize=maximize,
                **optimizer_kwargs,
            )

            # check shape
            self.assertTrue(pareto_set.ndim == 2)
            self.assertTrue(pareto_front.ndim == 2)
            self.assertTrue(pareto_set.shape[-1] == X.shape[-1])
            self.assertTrue(pareto_front.shape[-1] == Y.shape[-1])
            self.assertTrue(pareto_front.shape[-2] == pareto_set.shape[-2])
            num_optimal_points = pareto_front.shape[-2]

            # check if samples are non-dominated
            weight = 1.0 if maximize else -1.0
            count = torch.sum(is_non_dominated(Y=weight * pareto_front))
            self.assertTrue(count == num_optimal_points)

        # Ask for more optimal points than query evaluations
        with self.assertRaises(RuntimeError):
            random_search_optimizer(
                model=model_sample,
                bounds=bounds,
                num_points=20,
                maximize=maximize,
                max_tries=1,
                pop_size=10,
            )

    def test_sample_optimal_points(self):
        torch.manual_seed(1)
        input_dim = 3
        num_initial = 5
        tkwargs = {"device": self.device}
        optimizer_kwargs = {"pop_size": 100, "max_tries": 1}
        num_samples = 2
        num_points = 1

        for (
            dtype,
            maximize,
            num_objectives,
            opt_kwargs,
            use_model_list,
            standardize_model,
        ) in product(
            (torch.float, torch.double),
            (True, False),
            (1, 2),
            (optimizer_kwargs, None),
            (False, True),
            (False, True),
        ):
            tkwargs["dtype"] = dtype

            model, X, Y = get_model(
                num_points=num_initial,
                input_dim=input_dim,
                num_objectives=num_objectives,
                use_model_list=use_model_list,
                standardize_model=standardize_model,
                **tkwargs,
            )

            input_dim = X.shape[-1]
            bounds = torch.zeros((2, input_dim), **tkwargs)
            bounds[1] = 1.0

            # check the error when asking for too many optimal points
            if num_objectives == 1:
                with self.assertRaises(UnsupportedError):
                    sample_optimal_points(
                        model=model,
                        bounds=bounds,
                        num_samples=num_samples,
                        num_points=2,
                        maximize=maximize,
                        optimizer=random_search_optimizer,
                        optimizer_kwargs=opt_kwargs,
                    )

            pareto_sets, pareto_fronts = sample_optimal_points(
                model=model,
                bounds=bounds,
                num_samples=num_samples,
                num_points=num_points,
                maximize=maximize,
                optimizer=random_search_optimizer,
                optimizer_kwargs=opt_kwargs,
            )

            # check shape
            ps_desired_shape = torch.Size([num_samples, num_points, input_dim])
            pf_desired_shape = torch.Size([num_samples, num_points, num_objectives])

            self.assertTrue(pareto_sets.shape == ps_desired_shape)
            self.assertTrue(pareto_fronts.shape == pf_desired_shape)
