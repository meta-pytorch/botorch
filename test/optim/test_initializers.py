#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import ExitStack
from itertools import product
from random import random
from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning, SamplingWarning
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_hvkg_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    gen_value_function_initial_conditions,
    initialize_q_batch,
    initialize_q_batch_nonneg,
    initialize_q_batch_topn,
    sample_perturbed_subset_dims,
    sample_points_around_best,
    sample_q_batches_from_polytope,
    transform_constraints,
    transform_inter_point_constraint,
    transform_intra_point_constraint,
)
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.sampling import manual_seed, unnormalize
from botorch.utils.testing import (
    BotorchTestCase,
    get_max_violation_of_bounds,
    get_max_violation_of_constraints,
    MockAcquisitionFunction,
    MockModel,
    MockPosterior,
)


class TestBoundsAndConstraintCheckers(BotorchTestCase):
    def test_bounds_check(self) -> None:
        bounds = torch.tensor([[1, 2], [3, 4]], device=self.device)
        samples = torch.tensor([[2, 3], [2, 3.1]], device=self.device)[None, :, :]
        result = get_max_violation_of_bounds(samples, bounds)
        self.assertAlmostEqual(result, -0.9, delta=1e-6)

        samples = torch.tensor([[2, 3], [2, 4.1]], device=self.device)[None, :, :]
        result = get_max_violation_of_bounds(samples, bounds)
        self.assertAlmostEqual(result, 0.1, delta=1e-6)

    def test_constraint_check(self) -> None:
        constraints = [
            (
                torch.tensor([1], device=self.device),
                torch.tensor([1.0], device=self.device),
                3,
            )
        ]
        samples = torch.tensor([[2, 3], [2, 3.1]], device=self.device)[None, :, :]
        result = get_max_violation_of_constraints(samples, constraints, equality=True)
        self.assertAlmostEqual(result, 0.1, delta=1e-6)

        result = get_max_violation_of_constraints(samples, constraints, equality=False)
        self.assertAlmostEqual(result, 0.0, delta=1e-6)


class TestInitializeQBatch(BotorchTestCase):
    def test_initialize_q_batch_nonneg(self):
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            acq_vals = torch.rand(5, device=self.device, dtype=dtype)
            ics_X, ics_acq_vals = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics_X.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics_X.device, X.device)
            self.assertEqual(ics_X.dtype, X.dtype)
            self.assertEqual(ics_acq_vals.shape, torch.Size([2]))
            self.assertEqual(ics_acq_vals.device, acq_vals.device)
            self.assertEqual(ics_acq_vals.dtype, acq_vals.dtype)
            # ensure nothing happens if we want all samples
            ics_X, ics_acq_vals = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=5)
            self.assertTrue(torch.equal(X, ics_X))
            self.assertTrue(torch.equal(acq_vals, ics_acq_vals))
            # make sure things work with constant inputs
            acq_vals = torch.ones(5, device=self.device, dtype=dtype)
            ics, _ = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure raises correct warning
            acq_vals = torch.zeros(5, device=self.device, dtype=dtype)
            with self.assertWarns(BadInitialCandidatesWarning):
                ics, _ = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=10)
            # test less than `n` positive acquisition values
            acq_vals = torch.arange(5, device=self.device, dtype=dtype) - 3
            ics_X, ics_acq_vals = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics_X.shape, torch.Size([2, 3, 4]))
            # check that we chose the point with the positive acquisition value
            self.assertTrue((ics_acq_vals > 0).any())
            # test less than `n` alpha_pos values
            acq_vals = torch.arange(5, device=self.device, dtype=dtype)
            ics, _ = initialize_q_batch_nonneg(X=X, acq_vals=acq_vals, n=2, alpha=1.0)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)

    def test_initialize_q_batch(self):
        for dtype, batch_shape in (
            (torch.float, torch.Size()),
            (torch.double, [3, 2]),
            (torch.float, (2,)),
            (torch.double, torch.Size([2, 3, 4])),
            (torch.float, []),
            (torch.float, [100]),
        ):
            # basic test
            X = torch.rand(5, *batch_shape, 3, 4, device=self.device, dtype=dtype)
            acq_vals = torch.rand(5, *batch_shape, device=self.device, dtype=dtype)
            ics_X, ics_acq_vals = initialize_q_batch(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics_X.shape, torch.Size([2, *batch_shape, 3, 4]))
            self.assertEqual(ics_X.device, X.device)
            self.assertEqual(ics_X.dtype, X.dtype)
            self.assertEqual(ics_acq_vals.shape, torch.Size([2, *batch_shape]))
            self.assertEqual(ics_acq_vals.device, acq_vals.device)
            self.assertEqual(ics_acq_vals.dtype, acq_vals.dtype)
            # check that the max value for every batch is returned
            max_values = acq_vals.max(dim=0).values
            contains_max = (ics_acq_vals == max_values).any(dim=0)
            self.assertTrue(contains_max.all())

            # ensure nothing happens if we want all samples
            ics_X, ics_acq_vals = initialize_q_batch(X=X, acq_vals=acq_vals, n=5)
            self.assertTrue(torch.equal(X, ics_X))
            self.assertTrue(torch.equal(acq_vals, ics_acq_vals))

            # ensure raises correct warning
            acq_vals = torch.zeros(5, device=self.device, dtype=dtype)
            with self.assertWarns(BadInitialCandidatesWarning):
                ics, _ = initialize_q_batch(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics.shape, torch.Size([2, *batch_shape, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch(X=X, acq_vals=acq_vals, n=10)

    def test_initialize_q_batch_topn(self):
        for dtype in (torch.float, torch.double):
            # basic test
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            acq_vals = torch.rand(5, device=self.device, dtype=dtype)
            ics_X, ics_acq_vals = initialize_q_batch_topn(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics_X.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics_X.device, X.device)
            self.assertEqual(ics_X.dtype, X.dtype)
            self.assertEqual(ics_acq_vals.shape, torch.Size([2]))
            self.assertEqual(ics_acq_vals.device, acq_vals.device)
            self.assertEqual(ics_acq_vals.dtype, acq_vals.dtype)
            # ensure nothing happens if we want all samples
            ics_X, ics_acq_vals = initialize_q_batch_topn(X=X, acq_vals=acq_vals, n=5)
            self.assertTrue(torch.equal(X, ics_X))
            self.assertTrue(torch.equal(acq_vals, ics_acq_vals))
            # make sure things work with constant inputs
            acq_vals = torch.ones(5, device=self.device, dtype=dtype)
            ics, _ = initialize_q_batch_topn(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            self.assertEqual(ics.device, X.device)
            self.assertEqual(ics.dtype, X.dtype)
            # ensure raises correct warning
            acq_vals = torch.zeros(5, device=self.device, dtype=dtype)
            with self.assertWarns(BadInitialCandidatesWarning):
                ics, _ = initialize_q_batch_topn(X=X, acq_vals=acq_vals, n=2)
            self.assertEqual(ics.shape, torch.Size([2, 3, 4]))
            with self.assertRaises(RuntimeError):
                initialize_q_batch_topn(X=X, acq_vals=acq_vals, n=10)

    def test_initialize_q_batch_largeZ(self):
        for dtype in (torch.float, torch.double):
            # testing large eta*Z
            X = torch.rand(5, 3, 4, device=self.device, dtype=dtype)
            acq_vals = torch.tensor(
                [-1e12, 0, 0, 0, 1e12], device=self.device, dtype=dtype
            )
            ics, _ = initialize_q_batch(X=X, acq_vals=acq_vals, n=2, eta=100)
            self.assertEqual(ics.shape[0], 2)


class TestGenBatchInitialCandidates(BotorchTestCase):
    def test_gen_batch_initial_inf_bounds(self):
        bounds = torch.rand(2, 2)
        bounds[0, 1] = float("inf")
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Currently only finite values in `bounds` are supported for "
            r"generating initial conditions for optimization.",
        ):
            gen_batch_initial_conditions(
                acq_function=mock.Mock(),
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=2,
            )

    def test_gen_batch_initial_conditions(self):
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for (
            dtype,
            nonnegative,
            seed,
            init_batch_limit,
            ffs,
            sample_around_best,
        ) in (
            (torch.float, True, None, None, None, True),
            (torch.double, False, 1234, 1, {0: 0.5}, False),
            (torch.double, True, 1234, None, {0: 0.5}, True),
        ):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))
            with mock.patch.object(
                MockAcquisitionFunction,
                "__call__",
                wraps=mock_acqf.__call__,
            ) as mock_acqf_call, warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=mock_acqf,
                    bounds=bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=10,
                    fixed_features=ffs,
                    options={
                        "nonnegative": nonnegative,
                        "eta": 0.01,
                        "alpha": 0.1,
                        "seed": seed,
                        "init_batch_limit": init_batch_limit,
                        "sample_around_best": sample_around_best,
                    },
                )
            expected_shape = torch.Size([2, 1, 2])
            self.assertEqual(batch_initial_conditions.shape, expected_shape)
            self.assertEqual(batch_initial_conditions.device, bounds.device)
            self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
            self.assertLess(
                get_max_violation_of_bounds(batch_initial_conditions, bounds),
                1e-6,
            )
            batch_shape = (
                torch.Size([])
                if init_batch_limit is None
                else torch.Size([init_batch_limit])
            )
            raw_samps = mock_acqf_call.call_args[0][0]
            batch_shape = (
                torch.Size([20 if sample_around_best else 10])
                if init_batch_limit is None
                else torch.Size([init_batch_limit])
            )
            expected_raw_samps_shape = batch_shape + torch.Size([1, 2])
            self.assertEqual(raw_samps.shape, expected_raw_samps_shape)

            if ffs is not None:
                for idx, val in ffs.items():
                    self.assertTrue(
                        torch.all(batch_initial_conditions[..., idx] == val)
                    )

    def test_gen_batch_initial_conditions_topn(self):
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        mock_acqf.maximize = True  # Add maximize attribute
        for (
            dtype,
            topn,
            largest,
            is_sorted,
            seed,
            init_batch_limit,
            ffs,
            sample_around_best,
        ) in (
            (torch.float, True, True, True, None, None, None, True),
            (torch.double, False, False, False, 1234, 1, {0: 0.5}, False),
            (torch.float, True, None, True, 1234, None, None, False),
            (torch.double, False, True, False, None, 1, {0: 0.5}, True),
            (torch.float, True, False, False, 1234, None, {0: 0.5}, True),
            (torch.double, False, None, True, None, 1, None, False),
            (torch.float, True, True, False, 1234, 1, {0: 0.5}, True),
            (torch.double, False, False, True, None, None, None, False),
        ):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))
            with mock.patch.object(
                MockAcquisitionFunction,
                "__call__",
                wraps=mock_acqf.__call__,
            ) as mock_acqf_call, warnings.catch_warnings():
                warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
                options = {
                    "topn": topn,
                    "sorted": is_sorted,
                    "seed": seed,
                    "init_batch_limit": init_batch_limit,
                    "sample_around_best": sample_around_best,
                }
                if largest is not None:
                    options["largest"] = largest
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=mock_acqf,
                    bounds=bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=10,
                    fixed_features=ffs,
                    options=options,
                )
            expected_shape = torch.Size([2, 1, 2])
            self.assertEqual(batch_initial_conditions.shape, expected_shape)
            self.assertEqual(batch_initial_conditions.device, bounds.device)
            self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
            self.assertLess(
                get_max_violation_of_bounds(batch_initial_conditions, bounds),
                1e-6,
            )
            batch_shape = (
                torch.Size([])
                if init_batch_limit is None
                else torch.Size([init_batch_limit])
            )
            raw_samps = mock_acqf_call.call_args[0][0]
            batch_shape = (
                torch.Size([20 if sample_around_best else 10])
                if init_batch_limit is None
                else torch.Size([init_batch_limit])
            )
            expected_raw_samps_shape = batch_shape + torch.Size([1, 2])
            self.assertEqual(raw_samps.shape, expected_raw_samps_shape)

            if ffs is not None:
                for idx, val in ffs.items():
                    self.assertTrue(
                        torch.all(batch_initial_conditions[..., idx] == val)
                    )

    def test_gen_batch_initial_conditions_highdim(self):
        d = 2200  # 2200 * 10 (q) > 21201 (sobol max dim)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        ffs_map = {i: random() for i in range(0, d, 2)}
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for dtype, nonnegative, seed, ffs, sample_around_best in (
            (torch.float, True, None, None, True),
            (torch.double, False, 1234, ffs_map, False),
            (torch.double, True, 1234, ffs_map, True),
        ):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=MockAcquisitionFunction(),
                    bounds=bounds,
                    q=10,
                    num_restarts=1,
                    raw_samples=2,
                    fixed_features=ffs,
                    options={
                        "nonnegative": nonnegative,
                        "eta": 0.01,
                        "alpha": 0.1,
                        "seed": seed,
                        "sample_around_best": sample_around_best,
                    },
                )
                self.assertTrue(
                    any(issubclass(w.category, SamplingWarning) for w in ws)
                )
            expected_shape = torch.Size([1, 10, d])
            self.assertEqual(batch_initial_conditions.shape, expected_shape)
            self.assertEqual(batch_initial_conditions.device, bounds.device)
            self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
            self.assertLess(
                get_max_violation_of_bounds(batch_initial_conditions, bounds), 1e-6
            )
            if ffs is not None:
                for idx, val in ffs.items():
                    self.assertTrue(
                        torch.all(batch_initial_conditions[..., idx] == val)
                    )

    def test_gen_batch_initial_conditions_warning(self) -> None:
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            samples = torch.zeros(10, 1, 2, device=self.device, dtype=dtype)
            with self.assertWarnsRegex(
                expected_warning=BadInitialCandidatesWarning,
                expected_regex="Unable to find non-zero acquisition",
            ), mock.patch(
                "botorch.optim.initializers.draw_sobol_samples",
                return_value=samples,
            ):
                batch_initial_conditions = gen_batch_initial_conditions(
                    acq_function=MockAcquisitionFunction(),
                    bounds=bounds,
                    q=1,
                    num_restarts=2,
                    raw_samples=10,
                    options={"seed": 1234},
                )
            self.assertTrue(
                torch.equal(
                    batch_initial_conditions,
                    torch.zeros(2, 1, 2, device=self.device, dtype=dtype),
                )
            )

    def test_gen_batch_initial_conditions_transform_intra_point_constraint(self):
        for dtype in (torch.float, torch.double):
            constraint = (
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
                torch.tensor([-1, -1]).to(dtype=dtype, device=self.device),
                -1.0,
            )
            constraints = transform_intra_point_constraint(
                constraint=constraint, d=3, q=3
            )
            self.assertEqual(len(constraints), 3)
            self.assertAllClose(
                constraints[0][0],
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                constraints[1][0],
                torch.tensor([3, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                constraints[2][0],
                torch.tensor([6, 7], dtype=torch.int64, device=self.device),
            )
            for constraint in constraints:
                self.assertAllClose(
                    torch.tensor([-1, -1], dtype=dtype, device=self.device),
                    constraint[1],
                )
                self.assertEqual(constraint[2], -1.0)
            # test failure on invalid d
            constraint = (
                torch.tensor([[0, 3]], dtype=torch.int64, device=self.device),
                torch.tensor([-1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Constraint indices cannot exceed the problem dimension d=3.",
            ):
                transform_intra_point_constraint(constraint=constraint, d=3, q=2)

    def test_gen_batch_intial_conditions_transform_inter_point_constraint(self):
        for dtype in (torch.float, torch.double):
            constraint = (
                torch.tensor([[0, 1], [1, 1]], dtype=torch.int64, device=self.device),
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            transformed = transform_inter_point_constraint(constraint=constraint, d=3)
            self.assertAllClose(
                transformed[0],
                torch.tensor([1, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[1],
                torch.tensor([1.0, -1.0]).to(dtype=dtype, device=self.device),
            )
            self.assertEqual(constraint[2], 0.0)
            # test failure on invalid d
            constraint = (
                torch.tensor([[0, 1], [1, 3]], dtype=torch.int64, device=self.device),
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Constraint indices cannot exceed the problem dimension d=3.",
            ):
                transform_inter_point_constraint(constraint=constraint, d=3)

    def test_gen_batch_initial_conditions_transform_constraints(self):
        for dtype in (torch.float, torch.double):
            # test with None
            self.assertIsNone(transform_constraints(constraints=None, d=3, q=3))
            constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.int64, device=self.device),
                    torch.tensor([-1.0, -1.0], dtype=dtype, device=self.device),
                    -1.0,
                ),
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
                    0,
                ),
            ]
            transformed = transform_constraints(constraints=constraints, d=3, q=3)
            self.assertEqual(len(transformed), 4)
            self.assertAllClose(
                transformed[0][0],
                torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[1][0],
                torch.tensor([3, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[2][0],
                torch.tensor([6, 7], dtype=torch.int64, device=self.device),
            )
            for constraint in transformed[:3]:
                self.assertAllClose(
                    torch.tensor([-1, -1], dtype=dtype, device=self.device),
                    constraint[1],
                )
                self.assertEqual(constraint[2], -1.0)
            self.assertAllClose(
                transformed[-1][0],
                torch.tensor([1, 4], dtype=torch.int64, device=self.device),
            )
            self.assertAllClose(
                transformed[-1][1],
                torch.tensor([1.0, -1.0], dtype=dtype, device=self.device),
            )
            self.assertEqual(transformed[-1][2], 0.0)

    def test_gen_batch_initial_conditions_sample_q_batches_from_polytope(self):
        n = 5
        q = 2
        d = 3

        for dtype in (torch.float, torch.double):
            bounds = torch.tensor(
                [[0, 0, 0], [1, 1, 1]], device=self.device, dtype=dtype
            )
            inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, 1], device=self.device, dtype=dtype),
                    torch.tensor(-0.5, device=self.device, dtype=dtype),
                )
            ]
            inter_point_inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, 1], device=self.device, dtype=dtype),
                    torch.tensor(-0.4, device=self.device, dtype=dtype),
                ),
                (
                    torch.tensor(
                        [[0, 1], [1, 1]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1, 1], device=self.device, dtype=dtype),
                    torch.tensor(0.3, device=self.device, dtype=dtype),
                ),
            ]
            equality_constraints = [
                (
                    torch.tensor([0, 1, 2], device=self.device, dtype=torch.int64),
                    torch.tensor([1, 1, 1], device=self.device, dtype=dtype),
                    torch.tensor(1, device=self.device, dtype=dtype),
                )
            ]
            inter_point_equality_constraints = [
                (
                    torch.tensor([0, 1, 2], device=self.device, dtype=torch.int64),
                    torch.tensor([1, 1, 1], device=self.device, dtype=dtype),
                    torch.tensor(1, device=self.device, dtype=dtype),
                ),
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
            ]
            for equalities, inequalities in product(
                [None, equality_constraints, inter_point_equality_constraints],
                [None, inequality_constraints, inter_point_inequality_constraints],
            ):
                samples = sample_q_batches_from_polytope(
                    n=n,
                    q=q,
                    bounds=bounds,
                    n_burnin=20,
                    n_thinning=4,
                    seed=42,
                    inequality_constraints=inequalities,
                    equality_constraints=equalities,
                )
                self.assertEqual(samples.shape, torch.Size((n, q, d)))

                tol = 4e-7

                # samples are always on cpu
                def _to_self_device(
                    x: torch.Tensor | None,
                ) -> torch.Tensor | None:
                    return None if x is None else x.to(device=self.device)

                self.assertLess(
                    get_max_violation_of_bounds(_to_self_device(samples), bounds), tol
                )

                self.assertLess(
                    get_max_violation_of_constraints(
                        _to_self_device(samples), constraints=equalities, equality=True
                    ),
                    tol,
                )

                self.assertLess(
                    get_max_violation_of_constraints(
                        _to_self_device(samples),
                        constraints=inequalities,
                        equality=False,
                    ),
                    tol,
                )

    def test_gen_batch_initial_conditions_constraints(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            inequality_constraints = [
                (
                    torch.tensor([1], device=self.device, dtype=torch.int64),
                    torch.tensor([-4], device=self.device, dtype=dtype),
                    torch.tensor(-3, device=self.device, dtype=dtype),
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0], device=self.device, dtype=torch.int64),
                    torch.tensor([1], device=self.device, dtype=dtype),
                    torch.tensor(0.5, device=self.device, dtype=dtype),
                )
            ]
            for nonnegative, seed, init_batch_limit, ffs in product(
                [True, False], [None, 1234], [None, 1], [None, {0: 0.5}]
            ):
                mock_acqf = MockAcquisitionFunction()
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "__call__",
                    wraps=mock_acqf.__call__,
                ) as mock_acqf_call, warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", category=BadInitialCandidatesWarning
                    )
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=mock_acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=2,
                        raw_samples=10,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": init_batch_limit,
                            "n_burnin": 3,
                            "n_thinning": 2,
                        },
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                    )
                expected_shape = torch.Size([2, 1, 2])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
                self.assertLess(
                    get_max_violation_of_bounds(batch_initial_conditions, bounds),
                    1e-6,
                )
                self.assertLess(
                    get_max_violation_of_constraints(
                        batch_initial_conditions,
                        inequality_constraints,
                        equality=False,
                    ),
                    1e-6,
                )
                self.assertLess(
                    get_max_violation_of_constraints(
                        batch_initial_conditions,
                        equality_constraints,
                        equality=True,
                    ),
                    1e-6,
                )

                batch_shape = (
                    torch.Size([])
                    if init_batch_limit is None
                    else torch.Size([init_batch_limit])
                )
                raw_samps = mock_acqf_call.call_args[0][0]
                batch_shape = (
                    torch.Size([10])
                    if init_batch_limit is None
                    else torch.Size([init_batch_limit])
                )
                expected_raw_samps_shape = batch_shape + torch.Size([1, 2])
                self.assertEqual(raw_samps.shape, expected_raw_samps_shape)
                self.assertTrue((raw_samps[..., 0] == 0.5).all())
                self.assertTrue((-4 * raw_samps[..., 1] >= -3).all())
                if ffs is not None:
                    for idx, val in ffs.items():
                        self.assertTrue(
                            torch.all(batch_initial_conditions[..., idx] == val)
                        )

    def test_gen_batch_initial_conditions_interpoint_constraints(self):
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            inequality_constraints = [
                (
                    torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                    torch.tensor([-1, -1.0], device=self.device, dtype=dtype),
                    torch.tensor(-1.0, device=self.device, dtype=dtype),
                )
            ]
            equality_constraints = [
                (
                    torch.tensor(
                        [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
                (
                    torch.tensor(
                        [[0, 0], [2, 0]], device=self.device, dtype=torch.int64
                    ),
                    torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                    0,
                ),
            ]

            for nonnegative, seed in product([True, False], [None, 1234]):
                mock_acqf = MockAcquisitionFunction()
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "__call__",
                    wraps=mock_acqf.__call__,
                ):
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=mock_acqf,
                        bounds=bounds,
                        q=3,
                        num_restarts=2,
                        raw_samples=10,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": None,
                            "n_burnin": 3,
                            "n_thinning": 2,
                        },
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                    )
                expected_shape = torch.Size([2, 3, 2])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)

                self.assertTrue((batch_initial_conditions.sum(dim=-1) <= 1).all())

                self.assertAllClose(
                    batch_initial_conditions[0, 0, 0],
                    batch_initial_conditions[0, 1, 0],
                    batch_initial_conditions[0, 2, 0],
                    atol=1e-7,
                )

                self.assertAllClose(
                    batch_initial_conditions[1, 0, 0],
                    batch_initial_conditions[1, 1, 0],
                    batch_initial_conditions[1, 2, 0],
                )
                self.assertLess(
                    get_max_violation_of_constraints(
                        batch_initial_conditions,
                        inequality_constraints,
                        equality=False,
                    ),
                    1e-6,
                )

    def test_gen_batch_initial_conditions_generator(self):
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor(
                [[0, 0, 0], [1, 1, 1]], device=self.device, dtype=dtype
            )
            for nonnegative, seed, init_batch_limit, ffs in product(
                [True, False], [None, 1234], [None, 1], [None, {0: 0.5}]
            ):

                def generator(n: int, q: int, seed: int | None):
                    with manual_seed(seed):
                        X_rnd_nlzd = torch.rand(
                            n,
                            q,
                            bounds.shape[-1],
                            dtype=bounds.dtype,
                            device=self.device,
                        )
                        X_rnd = unnormalize(
                            X_rnd_nlzd, bounds, update_constant_bounds=False
                        )
                        X_rnd[..., -1] = 0.42
                        return X_rnd

                mock_acqf = MockAcquisitionFunction()
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "__call__",
                    wraps=mock_acqf.__call__,
                ), warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", category=BadInitialCandidatesWarning
                    )
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=mock_acqf,
                        bounds=bounds,
                        q=2,
                        num_restarts=4,
                        raw_samples=10,
                        generator=generator,
                        fixed_features=ffs,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": init_batch_limit,
                        },
                    )
                expected_shape = torch.Size([4, 2, 3])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
                self.assertTrue((batch_initial_conditions[..., -1] == 0.42).all())
                self.assertLess(
                    get_max_violation_of_bounds(batch_initial_conditions, bounds),
                    1e-6,
                )
                if ffs is not None:
                    for idx, val in ffs.items():
                        self.assertTrue(
                            torch.all(batch_initial_conditions[..., idx] == val)
                        )

    def test_error_generator_with_sample_around_best(self):
        tkwargs = {"device": self.device, "dtype": torch.double}

        def generator(n: int, q: int, seed: int | None):
            return torch.rand(n, q, 3).to(**tkwargs)

        with self.assertRaisesRegex(
            UnsupportedError,
            "Option 'sample_around_best' is not supported when custom "
            "generator is be used.",
        ):
            gen_batch_initial_conditions(
                MockAcquisitionFunction(),
                bounds=torch.tensor([[0, 0], [1, 1]], **tkwargs),
                q=1,
                num_restarts=1,
                raw_samples=1,
                generator=generator,
                options={"sample_around_best": True},
            )

    def test_error_equality_constraints_with_sample_around_best(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # this will give something that does not respect the constraints
        # TODO: it would be good to have a utils function to check if the
        # constraints are obeyed
        with self.assertRaisesRegex(
            UnsupportedError,
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present.",
        ):
            gen_batch_initial_conditions(
                MockAcquisitionFunction(),
                bounds=torch.tensor([[0, 0], [1, 1]], **tkwargs),
                q=1,
                num_restarts=1,
                raw_samples=1,
                equality_constraints=[
                    (
                        torch.tensor([0], **tkwargs),
                        torch.tensor([1], **tkwargs),
                        torch.tensor(0.5, **tkwargs),
                    )
                ],
                options={"sample_around_best": True},
            )

    def test_gen_batch_initial_conditions_fixed_X_fantasies(self):
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        mock_acqf = MockAcquisitionFunction()
        mock_acqf.objective = lambda y: y.squeeze(-1)
        for dtype in (torch.float, torch.double):
            bounds = bounds.to(device=self.device, dtype=dtype)
            mock_acqf.X_baseline = bounds  # for testing sample_around_best
            mock_acqf.model = MockModel(MockPosterior(mean=bounds[:, :1]))
            fixed_X_fantasies = torch.rand(3, 2, dtype=dtype, device=self.device)
            for nonnegative, seed, init_batch_limit, ffs, sample_around_best in product(
                [True, False], [None, 1234], [None, 1], [None, {0: 0.5}], [True, False]
            ):
                with mock.patch.object(
                    MockAcquisitionFunction,
                    "__call__",
                    wraps=mock_acqf.__call__,
                ) as mock_acqf_call:
                    batch_initial_conditions = gen_batch_initial_conditions(
                        acq_function=mock_acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=2,
                        raw_samples=10,
                        fixed_features=ffs,
                        options={
                            "nonnegative": nonnegative,
                            "eta": 0.01,
                            "alpha": 0.1,
                            "seed": seed,
                            "init_batch_limit": init_batch_limit,
                            "sample_around_best": sample_around_best,
                        },
                        fixed_X_fantasies=fixed_X_fantasies,
                    )
                expected_shape = torch.Size([2, 4, 2])
                self.assertEqual(batch_initial_conditions.shape, expected_shape)
                self.assertEqual(batch_initial_conditions.device, bounds.device)
                self.assertEqual(batch_initial_conditions.dtype, bounds.dtype)
                self.assertLess(
                    get_max_violation_of_bounds(batch_initial_conditions, bounds),
                    1e-6,
                )
                batch_shape = (
                    torch.Size([])
                    if init_batch_limit is None
                    else torch.Size([init_batch_limit])
                )
                raw_samps = mock_acqf_call.call_args[0][0]
                batch_shape = (
                    torch.Size([20 if sample_around_best else 10])
                    if init_batch_limit is None
                    else torch.Size([init_batch_limit])
                )
                expected_raw_samps_shape = batch_shape + torch.Size([4, 2])
                self.assertEqual(raw_samps.shape, expected_raw_samps_shape)

                if ffs is not None:
                    for idx, val in ffs.items():
                        self.assertTrue(
                            torch.all(batch_initial_conditions[..., 0, idx] == val)
                        )
                self.assertTrue(
                    torch.equal(
                        batch_initial_conditions[:, 1:],
                        fixed_X_fantasies.unsqueeze(0).expand(2, 3, 2),
                    )
                )

        # test wrong shape
        msg = (
            "`fixed_X_fantasies` and `bounds` must both have the same trailing"
            " dimension `d`, but have 3 and 2, respectively."
        )
        with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
            gen_batch_initial_conditions(
                acq_function=mock_acqf,
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=10,
                fixed_X_fantasies=torch.rand(3, 3, dtype=dtype, device=self.device),
            )


class TestGenOneShotKGInitialConditions(BotorchTestCase):
    def test_gen_one_shot_kg_initial_conditions(self):
        num_fantasies = 8
        num_restarts = 4
        raw_samples = 16
        for dtype in (torch.float, torch.double):
            mean = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean))
            mock_kg = qKnowledgeGradient(model=mm, num_fantasies=num_fantasies)
            bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
            # test option error
            with self.assertRaises(ValueError):
                gen_one_shot_kg_initial_conditions(
                    acq_function=mock_kg,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"frac_random": 2.0},
                )

            forbidden_indices = torch.tensor([[0, 1], [0, 1]])
            for constraint_type in [
                "inequality_constraints",
                "equality_constraints",
            ]:
                constraint_kwarg = {
                    constraint_type: [
                        (
                            forbidden_indices,
                            torch.ones(2).to(mean),
                            0.5,
                        )
                    ]
                }
                with self.subTest(
                    f"intra-point {constraint_type} not supported"
                ), self.assertRaisesRegex(
                    NotImplementedError,
                    "Indices must be one-dimensional "
                    "in gen_one_shot_kg_initial_conditions. "
                    "Received indices",
                ):
                    gen_one_shot_kg_initial_conditions(
                        acq_function=mock_kg,
                        bounds=bounds,
                        q=1,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        **constraint_kwarg,
                    )

            # test generation logic
            q = 2
            mock_random_ics = torch.rand(num_restarts, q + num_fantasies, 2)
            mock_fantasy_cands = torch.ones(20, 1, 2)
            mock_fantasy_vals = torch.randn(20)
            for constraint_kwargs in [
                {},
                {
                    "inequality_constraints": [
                        (torch.tensor([0, 1]), torch.ones(2).to(mean), 0.5)
                    ]
                },  # test that no error is raised
                {
                    "equality_constraints": [
                        (torch.tensor([0, 1]), torch.ones(2).to(mean), 0.5)
                    ]
                    # test that no error is raised
                },
            ]:
                with ExitStack() as es:
                    mock_gbics = es.enter_context(
                        mock.patch(
                            "botorch.optim.initializers.gen_batch_initial_conditions",
                            return_value=mock_random_ics,
                        )
                    )
                    mock_optacqf = es.enter_context(
                        mock.patch(
                            "botorch.optim.optimize.optimize_acqf",
                            return_value=(mock_fantasy_cands, mock_fantasy_vals),
                        )
                    )
                    with self.subTest(f"Main test with {constraint_kwargs}"):
                        ics = gen_one_shot_kg_initial_conditions(
                            acq_function=mock_kg,
                            bounds=bounds,
                            q=q,
                            num_restarts=num_restarts,
                            raw_samples=raw_samples,
                            **constraint_kwargs,
                        )
                        mock_gbics.assert_called_once()
                        mock_optacqf.assert_called_once()
                        n_value = int((1 - 0.1) * num_fantasies)
                        self.assertTrue(
                            torch.equal(
                                ics[..., :-n_value, :],
                                mock_random_ics[..., :-n_value, :],
                            )
                        )
                        self.assertTrue(torch.all(ics[..., -n_value:, :] == 1))


class TestGenOneShotHVKGInitialConditions(BotorchTestCase):
    def test_gen_one_shot_hvkg_initial_conditions(self):
        num_fantasies = 8
        num_restarts = 4
        raw_samples = 16
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            X = torch.rand(4, 2, **tkwargs)
            Y1 = torch.rand(4, 1, **tkwargs)
            Y2 = torch.rand(4, 1, **tkwargs)
            m1 = SingleTaskGP(X, Y1)
            m2 = SingleTaskGP(X, Y2)
            model = ModelListGP(m1, m2)
            for acqf_class in (
                qHypervolumeKnowledgeGradient,
                qMultiFidelityHypervolumeKnowledgeGradient,
            ):
                is_mf_kg = acqf_class is qMultiFidelityHypervolumeKnowledgeGradient
                mf_kwargs = {"target_fidelities": {1: 1.0}} if is_mf_kg else {}
                hvkg = acqf_class(
                    model=model,
                    ref_point=torch.zeros(2, **tkwargs),
                    num_fantasies=num_fantasies,
                    **mf_kwargs,
                )
                bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
                # test option error
                with self.assertRaises(ValueError):
                    gen_one_shot_hvkg_initial_conditions(
                        acq_function=hvkg,
                        bounds=bounds,
                        q=1,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options={"frac_random": 2.0},
                    )
                # test generation logic
                q = 2
                mock_fantasy_cands = torch.ones(20, 10, 2)
                mock_fantasy_vals = torch.randn(20)

                def mock_gen_ics(*args, **kwargs):
                    fixed_X_fantasies = kwargs.get("fixed_X_fantasies")
                    if fixed_X_fantasies is None:
                        return torch.rand(
                            kwargs["num_restarts"], q + hvkg.num_pseudo_points, 2
                        )
                    rand_candidates = torch.rand(
                        1,
                        q,
                        2,
                        dtype=fixed_X_fantasies.dtype,
                        device=fixed_X_fantasies.device,
                    )
                    return torch.cat(
                        [
                            rand_candidates,
                            fixed_X_fantasies.unsqueeze(0),
                        ],
                        dim=-2,
                    )

                for frac_random in (0.1, 0.5, 0.99):
                    with ExitStack() as es:
                        mock_gbics = es.enter_context(
                            mock.patch(
                                (
                                    "botorch.optim.initializers."
                                    "gen_batch_initial_conditions"
                                ),
                                wraps=mock_gen_ics,
                            )
                        )
                        mock_optacqf = es.enter_context(
                            mock.patch(
                                "botorch.optim.optimize.optimize_acqf",
                                return_value=(
                                    (
                                        mock_fantasy_cands[..., :1]
                                        if is_mf_kg
                                        else mock_fantasy_cands
                                    ),
                                    mock_fantasy_vals,
                                ),
                            )
                        )
                        ics = gen_one_shot_hvkg_initial_conditions(
                            acq_function=hvkg,
                            bounds=bounds,
                            q=q,
                            num_restarts=num_restarts,
                            raw_samples=raw_samples,
                            options={"frac_random": frac_random},
                        )
                        if frac_random == 0.5:
                            expected_call_count = 3
                        elif frac_random == 0.99:
                            expected_call_count = 1
                        else:
                            expected_call_count = 4
                        self.assertEqual(mock_gbics.call_count, expected_call_count)
                        mock_optacqf.assert_called_once()
                        n_value = int(round((1 - frac_random) * num_restarts))
                        # check that there are the expected number of optimized points
                        self.assertTrue(
                            (ics == 1).all(dim=-1).sum()
                            == n_value * hvkg.num_pseudo_points
                        )


class TestGenValueFunctionInitialConditions(BotorchTestCase):
    def test_gen_value_function_initial_conditions(self):
        num_fantasies = 2
        num_solutions = 3
        num_restarts = 4
        raw_samples = 5
        n_train = 6
        dim = 2
        dtype = torch.float
        # run a thorough test with dtype float
        train_X = torch.rand(n_train, dim, device=self.device, dtype=dtype)
        train_Y = torch.rand(n_train, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        fant_X = torch.rand(num_solutions, 1, dim, device=self.device, dtype=dtype)
        fantasy_model = model.fantasize(
            fant_X, IIDNormalSampler(sample_shape=torch.Size([num_fantasies]))
        )
        bounds = torch.tensor([[0, 0], [1, 1]], device=self.device, dtype=dtype)
        value_function = PosteriorMean(fantasy_model)
        # test option error
        with self.assertRaises(ValueError):
            gen_value_function_initial_conditions(
                acq_function=value_function,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                current_model=model,
                options={"frac_random": 2.0},
            )
        # test output shape
        ics = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            current_model=model,
        )
        self.assertEqual(
            ics.shape, torch.Size([num_restarts, num_fantasies, num_solutions, 1, dim])
        )
        # test bounds
        self.assertTrue(torch.all(ics >= bounds[0]))
        self.assertTrue(torch.all(ics <= bounds[1]))
        # test dtype
        self.assertEqual(dtype, ics.dtype)

        # minimal test cases for when all raw samples are random, with dtype double
        dtype = torch.double
        n_train = 2
        dim = 1
        num_solutions = 1
        train_X = torch.rand(n_train, dim, device=self.device, dtype=dtype)
        train_Y = torch.rand(n_train, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)
        fant_X = torch.rand(1, 1, dim, device=self.device, dtype=dtype)
        fantasy_model = model.fantasize(
            fant_X, IIDNormalSampler(sample_shape=torch.Size([num_fantasies]))
        )
        bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
        value_function = PosteriorMean(fantasy_model)
        ics = gen_value_function_initial_conditions(
            acq_function=value_function,
            bounds=bounds,
            num_restarts=1,
            raw_samples=1,
            current_model=model,
            options={"frac_random": 0.99},
        )
        self.assertEqual(
            ics.shape, torch.Size([1, num_fantasies, num_solutions, 1, dim])
        )
        # test bounds
        self.assertTrue(torch.all(ics >= bounds[0]))
        self.assertTrue(torch.all(ics <= bounds[1]))
        # test dtype
        self.assertEqual(dtype, ics.dtype)


class TestSampleAroundBest(BotorchTestCase):
    def test_sample_points_around_best(self):
        tkwargs = {"device": self.device}
        _bounds = torch.ones(2, 2)
        _bounds[1] = 2
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = _bounds.to(**tkwargs)
            X_train = 1 + torch.rand(20, 2, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, prune_baseline=False, cache_root=False
            )
            with mock.patch(
                "botorch.optim.initializers.sample_perturbed_subset_dims"
            ) as mock_subset_dims:
                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                mock_subset_dims.assert_not_called()
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())
            # test model that returns a batched mean
            model = MockModel(
                MockPosterior(
                    mean=(2 * X_train + 1).sum(dim=-1, keepdim=True).unsqueeze(0)
                )
            )
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, prune_baseline=False, cache_root=False
            )
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=4,
                sigma=1e-3,
                bounds=bounds,
            )
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())

            # test EI without X_baseline
            acqf = qExpectedImprovement(model, best_f=0.0)

            with warnings.catch_warnings(record=True) as w:
                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            # set train inputs
            model.train_inputs = (X_train,)
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=4,
                sigma=1e-3,
                bounds=bounds,
            )
            self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())

            # test an acquisition function that has no posterior_transform
            # and maximize=False
            pm = PosteriorMean(model, maximize=False)
            self.assertIsNone(pm.posterior_transform)
            self.assertFalse(pm.maximize)
            X_rnd = sample_points_around_best(
                acq_function=pm,
                n_discrete_points=4,
                sigma=0,
                bounds=bounds,
                best_pct=1e-8,  # ensures that we only use best value
            )
            idx = (-model.posterior(X_train).mean).argmax()
            self.assertTrue((X_rnd == X_train[idx : idx + 1]).all(dim=-1).all())

            # test acquisition function that has no model
            ff = FixedFeatureAcquisitionFunction(pm, d=2, columns=[0], values=[0])
            # set X_baseline for testing purposes
            ff.X_baseline = X_train
            with warnings.catch_warnings(record=True) as w:
                X_rnd = sample_points_around_best(
                    acq_function=ff,
                    n_discrete_points=4,
                    sigma=1e-3,
                    bounds=bounds,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, BotorchWarning))
                self.assertIsNone(X_rnd)

            # test constraints with NEHVI
            constraints = [lambda Y: Y[..., 0]]
            ref_point = torch.zeros(2, **tkwargs)
            # test cases when there are and are not any feasible points
            for any_feas in (True, False):
                Y_train = torch.stack(
                    [
                        (
                            torch.linspace(-0.5, 0.5, X_train.shape[0], **tkwargs)
                            if any_feas
                            else torch.ones(X_train.shape[0], **tkwargs)
                        ),
                        X_train.sum(dim=-1),
                    ],
                    dim=-1,
                )
                moo_model = MockModel(MockPosterior(mean=Y_train, samples=Y_train))
                acqf = qNoisyExpectedHypervolumeImprovement(
                    moo_model,
                    ref_point=ref_point,
                    X_baseline=X_train,
                    constraints=constraints,
                    cache_root=False,
                    sampler=IIDNormalSampler(sample_shape=torch.Size([2])),
                )
                X_rnd = sample_points_around_best(
                    acq_function=acqf,
                    n_discrete_points=4,
                    sigma=0.0,
                    bounds=bounds,
                )
                self.assertTrue(X_rnd.shape, torch.Size([4, 2]))
                # this should be true since sigma=0
                # and we should only be returning feasible points
                violation = constraints[0](Y_train)
                neg_violation = -violation.clamp_min(0.0)
                feas = neg_violation == 0
                eq_mask = (X_train.unsqueeze(1) == X_rnd.unsqueeze(0)).all(dim=-1)
                if feas.any():
                    # determine
                    # create n_train x n_rnd tensor of booleans
                    eq_mask = (X_train.unsqueeze(1) == X_rnd.unsqueeze(0)).all(dim=-1)
                    # check that all X_rnd correspond to feasible points
                    self.assertEqual(eq_mask[feas].sum(), 4)
                else:
                    idcs = torch.topk(neg_violation, k=2).indices
                    self.assertEqual(eq_mask[idcs].sum(), 4)
                self.assertTrue((X_rnd >= 1).all())
                self.assertTrue((X_rnd <= 2).all())
            # test that subset_dims is called if d>=20
            X_train = 1 + torch.rand(10, 20, **tkwargs)
            model = MockModel(
                MockPosterior(mean=(2 * X_train + 1).sum(dim=-1, keepdim=True))
            )
            bounds = torch.ones(2, 20, **tkwargs)
            bounds[1] = 2
            # test NEI with X_baseline
            acqf = qNoisyExpectedImprovement(
                model, X_baseline=X_train, prune_baseline=False, cache_root=False
            )
            with mock.patch(
                "botorch.optim.initializers.sample_perturbed_subset_dims",
                wraps=sample_perturbed_subset_dims,
            ) as mock_subset_dims:
                X_rnd = sample_points_around_best(
                    acq_function=acqf, n_discrete_points=5, sigma=1e-3, bounds=bounds
                )
            self.assertEqual(X_rnd.shape, torch.Size([5, 20]))
            self.assertTrue((X_rnd >= 1).all())
            self.assertTrue((X_rnd <= 2).all())
            mock_subset_dims.assert_called_once()
            # test tiny prob_perturb to make sure we perturb at least one dimension
            X_rnd = sample_points_around_best(
                acq_function=acqf,
                n_discrete_points=5,
                sigma=1e-3,
                bounds=bounds,
                prob_perturb=1e-8,
            )
            self.assertTrue(
                ((X_rnd.unsqueeze(0) == X_train.unsqueeze(1)).all(dim=-1)).sum() == 0
            )
