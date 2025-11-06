#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import warnings
from abc import ABC
from typing import Any
from unittest import mock

import numpy as np
import torch
from botorch.acquisition.objective import (
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.exceptions.warnings import UserInputWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.sampling import (
    _convert_bounds_to_inequality_constraints,
    batched_multinomial,
    boltzmann_sample,
    DelaunayPolytopeSampler,
    draw_sobol_samples,
    find_interior_point,
    get_polytope_samples,
    HitAndRunPolytopeSampler,
    manual_seed,
    normalize_sparse_linear_constraints,
    optimize_posterior_samples,
    PolytopeSampler,
    sample_hypersphere,
    sample_perturbed_subset_dims,
    sample_polytope,
    sample_simplex,
    sample_truncated_normal_perturbations,
    sparse_to_dense_constraints,
)
from botorch.utils.testing import BotorchTestCase


def _get_constraints(device: torch.device, dtype: torch.dtype):
    bounds = torch.zeros(2, 3, device=device, dtype=dtype)
    bounds[1] = 1.0
    A = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 4.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    b = torch.tensor([[0.0], [1.0], [0.0], [0.0], [1.0]], device=device, dtype=dtype)
    x0 = torch.tensor([[0.1], [0.1], [0.1]], device=device, dtype=dtype)
    return bounds, A, b, x0


class TestManualSeed(BotorchTestCase):
    def test_manual_seed(self):
        initial_state = torch.random.get_rng_state()
        with manual_seed():
            self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))
        with manual_seed(1234):
            self.assertFalse(torch.all(torch.random.get_rng_state() == initial_state))
        self.assertTrue(torch.all(torch.random.get_rng_state() == initial_state))


class TestSampleUtils(BotorchTestCase):
    def test_draw_sobol_samples(self):
        batch_shapes = [None, [3, 5], torch.Size([2]), (5, 3, 2, 3), []]
        for d, q, n, batch_shape, seed, dtype in itertools.product(
            (1, 3),
            (1, 2),
            (2, 5),
            batch_shapes,
            (None, 1234),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            bounds = torch.stack([torch.rand(d), 1 + torch.rand(d)]).to(**tkwargs)
            samples = draw_sobol_samples(
                bounds=bounds, n=n, q=q, batch_shape=batch_shape, seed=seed
            )
            batch_shape = batch_shape or torch.Size()
            self.assertEqual(samples.shape, torch.Size([n, *batch_shape, q, d]))
            self.assertTrue(torch.all(samples >= bounds[0]))
            self.assertTrue(torch.all(samples <= bounds[1]))
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)
            if seed is not None:
                # Check that seed reproduces the same samples.
                samples2 = draw_sobol_samples(
                    bounds=bounds, n=n, q=q, batch_shape=batch_shape, seed=seed
                )
                self.assertTrue(torch.equal(samples, samples2))

    def test_sample_simplex(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_simplex(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.all(samples >= 0))
            self.assertTrue(torch.all(samples <= 1))
            self.assertTrue(torch.max((samples.sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)
            if seed is not None:
                # Check that seed reproduces the same samples.
                samples2 = sample_simplex(
                    d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
                )
                self.assertTrue(torch.equal(samples, samples2))

    def test_sample_hypersphere(self):
        for d, n, qmc, seed, dtype in itertools.product(
            (1, 2, 3), (2, 5), (False, True), (None, 1234), (torch.float, torch.double)
        ):
            samples = sample_hypersphere(
                d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
            )
            self.assertEqual(samples.shape, torch.Size([n, d]))
            self.assertTrue(torch.max((samples.pow(2).sum(dim=-1) - 1).abs()) < 1e-5)
            self.assertEqual(samples.device.type, self.device.type)
            self.assertEqual(samples.dtype, dtype)
            if seed is not None:
                # Check that seed reproduces the same samples.
                samples2 = sample_hypersphere(
                    d=d, n=n, qmc=qmc, seed=seed, device=self.device, dtype=dtype
                )
                self.assertTrue(torch.equal(samples, samples2))

    def test_batched_multinomial(self):
        num_categories = 5
        num_samples = 4
        Trulse = (True, False)
        for batch_shape, dtype, replacement, use_gen, use_out in itertools.product(
            ([], [3], [2, 3]), (torch.float, torch.double), Trulse, Trulse, Trulse
        ):
            weights = torch.rand(*batch_shape, num_categories, dtype=dtype)
            out = None
            if use_out:
                out = torch.empty(*batch_shape, num_samples, dtype=torch.long)
            samples = batched_multinomial(
                weights,
                num_samples,
                replacement=replacement,
                generator=torch.Generator() if use_gen else None,
                out=out,
            )
            self.assertEqual(samples.shape, torch.Size([*batch_shape, num_samples]))
            if use_out:
                self.assertTrue(torch.equal(samples, out))
            if not replacement:
                for s in samples.view(-1, num_samples):
                    self.assertTrue(torch.unique(s).size(0), num_samples)

    def test_convert_bounds_to_inequality_constraints(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # test basic case with no indefinite bounds
            lower_bounds = torch.rand(3, **tkwargs)
            upper_bounds = torch.rand_like(lower_bounds) + lower_bounds
            bounds = torch.stack([lower_bounds, upper_bounds], dim=0)
            A, b = _convert_bounds_to_inequality_constraints(bounds=bounds)
            identity = torch.eye(3, **tkwargs)
            self.assertTrue(torch.equal(A[:3], -identity))
            self.assertTrue(torch.equal(A[3:], identity))
            self.assertTrue(torch.equal(b[:3], -bounds[:1].t()))
            self.assertTrue(torch.equal(b[3:], bounds[1:].t()))
            # test filtering of indefinite bounds
            inf = float("inf")
            bounds = torch.tensor(
                [[-3.0, -inf, -inf], [inf, 2.0, inf]],
                **tkwargs,
            )
            A, b = _convert_bounds_to_inequality_constraints(bounds=bounds)
            A_xpct = torch.tensor([[-1.0, -0.0, -0.0], [0.0, 1.0, 0.0]], **tkwargs)
            b_xpct = torch.tensor([[3.0], [2.0]], **tkwargs)
            self.assertTrue(torch.equal(A, A_xpct))
            self.assertTrue(torch.equal(b, b_xpct))

    def test_sparse_to_dense_constraints(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            inequality_constraints = [
                (
                    torch.tensor([3], **tkwargs),
                    torch.tensor([4], **tkwargs),
                    3,
                )
            ]
            (A, b) = sparse_to_dense_constraints(
                d=4, constraints=inequality_constraints
            )
            expected_A = torch.tensor([[0.0, 0.0, 0.0, 4.0]], **tkwargs)
            self.assertTrue(torch.equal(A, expected_A))
            expected_b = torch.tensor([[3.0]], **tkwargs)
            self.assertTrue(torch.equal(b, expected_b))

    def test_normalize_sparse_linear_constraints(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            constraints = [
                (
                    torch.tensor([1, 2, 0], dtype=torch.int64, device=self.device),
                    torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                    1.0,
                )
            ]
            bounds = torch.tensor(
                [[0.1, 0.3, 0.1, 30.0], [0.6, 0.7, 0.7, 700.0]], **tkwargs
            )
            new_constraints = normalize_sparse_linear_constraints(bounds, constraints)
            expected_coefficients = torch.tensor([0.4000, 0.6000, 0.5000], **tkwargs)
            self.assertTrue(
                torch.allclose(new_constraints[0][1], expected_coefficients)
            )
            expected_rhs = 0.5
            self.assertAlmostEqual(new_constraints[0][-1], expected_rhs)
            with self.assertRaisesRegex(
                ValueError, "`indices` must be a one-dimensional tensor."
            ):
                normalize_sparse_linear_constraints(
                    bounds,
                    [(torch.tensor([[1, 2], [3, 4]]), torch.tensor([1.0, 1.0]), 1.0)],
                )

    def test_normalize_sparse_linear_constraints_wrong_dtype(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                constraints = [
                    (
                        torch.ones(3, dtype=torch.float, device=self.device),
                        torch.ones(3, **tkwargs),
                        1.0,
                    )
                ]
                bounds = torch.zeros(2, 4, **tkwargs)
                msg = "tensors used as indices must be long, byte or bool tensors"
                with self.assertRaises(IndexError, msg=msg):
                    normalize_sparse_linear_constraints(bounds, constraints)

    def test_find_interior_point(self):
        # basic problem: 1 <= x_1 <= 2, 2 <= x_2 <= 3
        A = np.concatenate([np.eye(2), -np.eye(2)], axis=0)
        b = np.array([2.0, 3.0, -1.0, -2.0])
        x = find_interior_point(A=A, b=b)
        self.assertTrue(np.allclose(x, np.array([1.5, 2.5])))
        # problem w/ negatives variables: -2 <= x_1 <= -1, -3 <= x_2 <= -2
        b = np.array([-1.0, -2.0, 2.0, 3.0])
        x = find_interior_point(A=A, b=b)
        self.assertTrue(np.allclose(x, np.array([-1.5, -2.5])))
        # problem with bound on a single variable: x_1 <= 0
        A = np.array([[1.0, 0.0]])
        b = np.zeros(1)
        x = find_interior_point(A=A, b=b)
        self.assertLessEqual(x[0].item(), 0.0)
        # unbounded problem: x >= 3
        A = np.array([[-1.0]])
        b = np.array([-3.0])
        x = find_interior_point(A=A, b=b)
        self.assertAlmostEqual(x.item(), 5.0, places=4)

    def test_get_polytope_samples(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = torch.zeros(2, 4, **tkwargs)
            bounds[1] = 1
            inequality_constraints = [
                (
                    torch.tensor([3], dtype=torch.int64, device=self.device),
                    torch.tensor([-4], **tkwargs),
                    -3,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0], dtype=torch.int64, device=self.device),
                    torch.tensor([1], **tkwargs),
                    0.5,
                )
            ]
            A, b = sparse_to_dense_constraints(d=4, constraints=inequality_constraints)
            C, d = sparse_to_dense_constraints(d=4, constraints=equality_constraints)

            samps = get_polytope_samples(
                n=5,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                seed=0,
                n_burnin=2,
                n_thinning=3,
            )
            expected_samps = HitAndRunPolytopeSampler(
                bounds=bounds,
                inequality_constraints=(-A, -b),
                equality_constraints=(C, d),
                n_burnin=2,
                n_thinning=3,
                seed=0,
            ).draw(5)
            self.assertTrue(torch.equal(samps, expected_samps))

            # test no equality constraints
            samps = get_polytope_samples(
                n=5,
                bounds=bounds,
                inequality_constraints=inequality_constraints,
                seed=0,
                n_burnin=2,
                n_thinning=3,
            )
            expected_samps = HitAndRunPolytopeSampler(
                bounds=bounds,
                inequality_constraints=(-A, -b),
                n_burnin=2,
                n_thinning=3,
                seed=0,
            ).draw(5)
            self.assertTrue(torch.equal(samps, expected_samps))

            # test no inequality constraints
            samps = get_polytope_samples(
                n=5,
                bounds=bounds,
                equality_constraints=equality_constraints,
                seed=0,
                n_burnin=2,
                n_thinning=3,
            )
            expected_samps = HitAndRunPolytopeSampler(
                bounds=bounds,
                equality_constraints=(C, d),
                n_burnin=2,
                n_thinning=3,
                seed=0,
            ).draw(5)
        self.assertTrue(torch.equal(samps, expected_samps))

    def test_sample_polytope_infeasible(self) -> None:
        with self.assertRaisesRegex(ValueError, "Starting point does not satisfy"):
            sample_polytope(
                A=torch.tensor([[0.0, 0.0]]),
                b=torch.tensor([[-1.0]]),
                x0=torch.tensor([[0.0], [0.0]]),
            )

    def test_sample_polytope_boundary(self) -> None:
        # Check that sample_polytope does not get stuck at the boundary.
        # This replicates https://github.com/meta-pytorch/botorch/issues/2351.
        samples = sample_polytope(
            A=torch.tensor(
                [
                    [-1.0, -1.0],
                    [0.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            b=torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]),
            x0=torch.tensor([[0.0], [0.0]]),
        )
        self.assertFalse((samples == 0).all())


class PolytopeSamplerTestBase(ABC):
    sampler_class: type[PolytopeSampler]
    sampler_kwargs: dict[str, Any] = {}
    constructor_seed_kwarg: dict[str, int] = {}
    draw_seed_kwarg: dict[str, int] = {}

    def test_sample_polytope(self):
        for dtype in (torch.float, torch.double):
            bounds, A, b, x0 = _get_constraints(device=self.device, dtype=dtype)
            for interior_point in (x0, None):
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10)
                self.assertTrue(torch.all(A @ samples.t() - b <= 0).item())
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # make sure we can draw multiple samples
                more_samples = sampler.draw(n=5)
                self.assertTrue(torch.all(A @ more_samples.t() - b <= 0).item())
                self.assertTrue((more_samples <= bounds[1]).all())
                self.assertTrue((more_samples >= bounds[0]).all())
                # the samples should all be unique
                all_samples = torch.cat([samples, more_samples], dim=0)
                self.assertEqual(
                    len(all_samples), len(torch.unique(all_samples, dim=0))
                )

    def test_sample_polytope_with_seed(self):
        for dtype in (torch.float, torch.double):
            bounds, A, b, x0 = _get_constraints(device=self.device, dtype=dtype)
            for interior_point in (x0, None):
                sampler1 = self.sampler_class(
                    inequality_constraints=(A, b),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                    **self.constructor_seed_kwarg,
                )
                sampler2 = self.sampler_class(
                    inequality_constraints=(A, b),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                    **self.constructor_seed_kwarg,
                )
                samples1 = sampler1.draw(n=10, **self.draw_seed_kwarg)
                samples2 = sampler2.draw(n=10, **self.draw_seed_kwarg)
                self.assertTrue(torch.allclose(samples1, samples2))

    def test_sample_polytope_with_eq_constraints(self):
        for dtype in (torch.float, torch.double):
            bounds, A, b, x0 = _get_constraints(device=self.device, dtype=dtype)
            C = torch.tensor([[1.0, -1, 0.0]], device=self.device, dtype=dtype)
            d = torch.zeros(1, 1, device=self.device, dtype=dtype)

            for interior_point in (x0, None):
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10)
                self.assertTrue(torch.all(A @ samples.t() - b <= 0).item())
                self.assertLessEqual((C @ samples.t() - d).abs().sum().item(), 1e-6)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # test no inequality constraints
                sampler = self.sampler_class(
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10)
                self.assertLessEqual((C @ samples.t() - d).abs().sum().item(), 1e-6)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())
                # test no inequality constraints or bounds
                with self.assertRaises(BotorchError):
                    self.sampler_class(
                        equality_constraints=(C, d),
                        interior_point=interior_point,
                        **self.sampler_kwargs,
                    )

    def test_sample_polytope_1d(self):
        for dtype in (torch.float, torch.double):
            A = torch.tensor(
                [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], device=self.device, dtype=dtype
            )
            b = torch.tensor([[0.0], [0.0], [1.0]], device=self.device, dtype=dtype)
            C = torch.tensor([[1.0, -1.0]], device=self.device, dtype=dtype)
            x0 = torch.tensor([[0.1], [0.1]], device=self.device, dtype=dtype)
            C = torch.tensor([[1.0, -1.0]], device=self.device, dtype=dtype)
            d = torch.tensor([[0.0]], device=self.device, dtype=dtype)
            bounds = torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]], device=self.device, dtype=dtype
            )
            for interior_point in (x0, None):
                sampler = self.sampler_class(
                    inequality_constraints=(A, b),
                    equality_constraints=(C, d),
                    bounds=bounds,
                    interior_point=interior_point,
                    **self.sampler_kwargs,
                )
                samples = sampler.draw(n=10)
                self.assertTrue(torch.all(A @ samples.t() - b <= 0).item())
                self.assertLessEqual((C @ samples.t() - d).abs().sum().item(), 1e-6)
                self.assertTrue((samples <= bounds[1]).all())
                self.assertTrue((samples >= bounds[0]).all())

    def test_initial_point(self):
        for dtype in (torch.float, torch.double):
            A = torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 4.0, 0.0]],
                device=self.device,
                dtype=dtype,
            )
            b = torch.tensor([[0.0], [-1.0], [1.0]], device=self.device, dtype=dtype)
            x0 = torch.tensor([[0.1], [0.1], [0.1]], device=self.device, dtype=dtype)
            bounds = torch.zeros(2, 3, device=self.device, dtype=dtype)
            bounds[1] = 1.0

            # testing for infeasibility of the initial point and
            # infeasibility of the original LP (status 2 of the linprog output).
            for interior_point in (x0, None):
                with self.assertRaises(ValueError):
                    self.sampler_class(
                        inequality_constraints=(A, b),
                        bounds=bounds,
                        interior_point=interior_point,
                    )

            class Result:
                status = 1
                message = "mock status 1"

            # testing for only status 1 of the LP
            with mock.patch("scipy.optimize.linprog") as mock_linprog:
                mock_linprog.return_value = Result()
                with self.assertRaises(ValueError):
                    self.sampler_class(
                        inequality_constraints=(A, b),
                        bounds=bounds,
                    )


class TestHitAndRunPolytopeSampler(PolytopeSamplerTestBase, BotorchTestCase):
    sampler_class = HitAndRunPolytopeSampler
    sampler_kwargs = {"n_burnin": 2, "n_thinning": 2}
    constructor_seed_kwarg = {"seed": 33125612}

    def test_normalization_warning(self):
        _, A, b, x0 = _get_constraints(device=self.device, dtype=torch.double)
        with self.assertWarnsRegex(
            UserInputWarning, "HitAndRunPolytopeSampler did not receive `bounds`"
        ):
            HitAndRunPolytopeSampler(inequality_constraints=(A, b), interior_point=x0)


class TestDelaunayPolytopeSampler(PolytopeSamplerTestBase, BotorchTestCase):
    sampler_class = DelaunayPolytopeSampler
    draw_seed_kwarg = {"seed": 33125612}

    def test_sample_polytope_unbounded(self):
        A = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 4.0, 1.0]],
            device=self.device,
        )
        b = torch.tensor([[0.0], [0.0], [0.0], [1.0]], device=self.device)
        x0 = torch.tensor([[0.1], [0.1], [0.1]], device=self.device)
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sampler_class(
                    inequality_constraints=(A, b),
                    interior_point=x0,
                    **self.sampler_kwargs,
                )


class TestOptimizePosteriorSamples(BotorchTestCase):
    def test_optimize_posterior_samples(self):
        # Fix the random seed to prevent flaky failures.
        torch.manual_seed(1)
        dims = 2
        dtype = torch.float64
        eps = 1e-4
        for_testing_speed_kwargs = {"raw_samples": 128, "num_restarts": 4}
        nums_optima = (1, 7)
        batch_shapes = ((), (2,), (3, 2))
        posterior_transforms = (
            None,
            ScalarizedPosteriorTransform(weights=-torch.ones(1, dtype=dtype)),
        )
        for num_optima, batch_shape, posterior_transform in itertools.product(
            nums_optima, batch_shapes, posterior_transforms
        ):
            bounds = torch.tensor([[0, 1]] * dims, dtype=dtype).T
            X = torch.rand(*batch_shape, 4, dims, dtype=dtype)
            Y = torch.pow(X - 0.5, 2).sum(dim=-1, keepdim=True)

            # having a noiseless model all but guarantees that the found optima
            # will be better than the observations
            model = SingleTaskGP(X, Y, torch.full_like(Y, eps))
            model.covar_module.lengthscale = 0.5
            paths = get_matheron_path_model(
                model=model, sample_shape=torch.Size([num_optima])
            )
            X_opt, f_opt = optimize_posterior_samples(
                paths=paths,
                bounds=bounds,
                sample_transform=(
                    posterior_transform.evaluate if posterior_transform else None
                ),
                **for_testing_speed_kwargs,
            )

            correct_X_shape = (num_optima,) + batch_shape + (dims,)
            correct_f_shape = (num_optima,) + batch_shape + (1,)

            self.assertEqual(X_opt.shape, correct_X_shape)
            self.assertEqual(f_opt.shape, correct_f_shape)
            self.assertTrue(torch.all(X_opt >= bounds[0]))
            self.assertTrue(torch.all(X_opt <= bounds[1]))

            # Check that the all found optima are larger than the observations
            # This is not 100% deterministic, but just about.
            Y_queries = paths(X)
            # this is when we negate, so the values should be smaller
            if posterior_transform:
                self.assertTrue(torch.all(f_opt < Y_queries.min(dim=-2).values))

            # otherwise, larger
            else:
                self.assertTrue(torch.all(f_opt > Y_queries.max(dim=-2).values))

        obj = LinearMCObjective(weights=-torch.ones(1, dtype=dtype))
        X_opt, f_opt = optimize_posterior_samples(
            paths=paths,
            bounds=bounds,
            sample_transform=obj,
            **for_testing_speed_kwargs,
        )
        self.assertTrue(torch.all(f_opt < Y_queries.max(dim=-2).values))

    def test_optimize_posterior_samples_multi_objective(self):
        # Fix the random seed to prevent flaky failures.
        torch.manual_seed(1)
        dims = 2
        dtype = torch.float64
        eps = 1e-4
        for_testing_speed_kwargs = {"raw_samples": 128, "num_restarts": 4}
        num_optima = 5
        batch_shape = (3,)

        # test that multi-output models are supported if there is an appropriate
        # scalarization
        bounds = torch.tensor([[0, 1]] * dims, dtype=dtype).T
        X = torch.rand(*batch_shape, 4, dims, dtype=dtype)
        Y1 = torch.pow(X - 0.5, 2).sum(dim=-1, keepdim=True)
        Y2 = torch.cos(X * 3).sum(dim=-1, keepdim=True)
        Y = torch.cat([Y1, Y2], dim=-1)
        # having a noiseless model all but guarantees that the found optima
        # will be better than the observations
        model = SingleTaskGP(X, Y, torch.full_like(Y, eps))
        model.covar_module.lengthscale = 0.5
        posterior_transform = ScalarizedPosteriorTransform(
            weights=torch.ones(2, dtype=dtype)
        )
        paths = get_matheron_path_model(
            model=model,
            sample_shape=torch.Size([num_optima]),
        )
        X_opt, f_opt = optimize_posterior_samples(
            paths=paths,
            bounds=bounds,
            sample_transform=posterior_transform.evaluate,
            **for_testing_speed_kwargs,
        )

        correct_X_shape = (num_optima,) + batch_shape + (dims,)
        correct_f_shape = (num_optima,) + batch_shape + (2,)
        self.assertEqual(X_opt.shape, correct_X_shape)
        self.assertEqual(f_opt.shape, correct_f_shape)
        self.assertTrue(torch.all(X_opt >= bounds[0]))
        self.assertTrue(torch.all(X_opt <= bounds[1]))

        X_opt, f_opt = optimize_posterior_samples(
            paths=paths,
            bounds=bounds,
            sample_transform=posterior_transform.evaluate,
            return_transformed=True,
            **for_testing_speed_kwargs,
        )
        correct_f_shape = (num_optima,) + batch_shape + (1,)
        self.assertEqual(f_opt.shape, correct_f_shape)


class TestSampleTruncatedNormalPerturbations(BotorchTestCase):
    def test_sample_truncated_normal_perturbations(self):
        tkwargs = {"device": self.device}
        n_discrete_points = 5
        _bounds = torch.ones(2, 4)
        _bounds[1] = 2
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            bounds = _bounds.to(**tkwargs)
            for n_best in (1, 2):
                X = 1 + torch.rand(n_best, 4, **tkwargs)
                # basic test
                perturbed_X = sample_truncated_normal_perturbations(
                    X=X,
                    n_discrete_points=n_discrete_points,
                    sigma=4,
                    bounds=bounds,
                    qmc=False,
                )
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 4]))
                self.assertTrue((perturbed_X >= 1).all())
                self.assertTrue((perturbed_X <= 2).all())
                # test qmc
                with mock.patch(
                    "botorch.utils.sampling.draw_sobol_samples",
                    wraps=draw_sobol_samples,
                ) as mock_sobol:
                    perturbed_X = sample_truncated_normal_perturbations(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        sigma=4,
                        bounds=bounds,
                        qmc=True,
                    )
                    mock_sobol.assert_called_once()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 4]))
                self.assertTrue((perturbed_X >= 1).all())
                self.assertTrue((perturbed_X <= 2).all())


class TestSamplePerturbedSubsetDims(BotorchTestCase):
    def test_sample_perturbed_subset_dims(self):
        tkwargs = {"device": self.device}
        n_discrete_points = 5

        # test that errors are raised
        with self.assertRaises(BotorchTensorDimensionError):
            sample_perturbed_subset_dims(
                X=torch.zeros(1, 1),
                n_discrete_points=1,
                sigma=1e-3,
                bounds=torch.zeros(1, 2, 1),
            )
        with self.assertRaises(BotorchTensorDimensionError):
            sample_perturbed_subset_dims(
                X=torch.zeros(1, 1, 1),
                n_discrete_points=1,
                sigma=1e-3,
                bounds=torch.zeros(2, 1),
            )
        for dtype in (torch.float, torch.double):
            for n_best in (1, 2):
                tkwargs["dtype"] = dtype
                bounds = torch.zeros(2, 21, **tkwargs)
                bounds[1] = 1
                X = torch.rand(n_best, 21, **tkwargs)
                # basic test
                with mock.patch(
                    "botorch.utils.sampling.draw_sobol_samples",
                ) as mock_sobol:
                    perturbed_X = sample_perturbed_subset_dims(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        qmc=False,
                        sigma=1e-3,
                        bounds=bounds,
                    )
                    mock_sobol.assert_not_called()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 21]))
                self.assertTrue((perturbed_X >= 0).all())
                self.assertTrue((perturbed_X <= 1).all())
                # test qmc
                with mock.patch(
                    "botorch.utils.sampling.draw_sobol_samples",
                    wraps=draw_sobol_samples,
                ) as mock_sobol:
                    perturbed_X = sample_perturbed_subset_dims(
                        X=X,
                        n_discrete_points=n_discrete_points,
                        sigma=1e-3,
                        bounds=bounds,
                    )
                    mock_sobol.assert_called_once()
                self.assertEqual(perturbed_X.shape, torch.Size([n_discrete_points, 21]))
                self.assertTrue((perturbed_X >= 0).all())
                self.assertTrue((perturbed_X <= 1).all())
                # for each point in perturbed_X compute the number of
                # dimensions it has in common with each point in X
                # and take the maximum number
                max_equal_dims = (
                    (perturbed_X.unsqueeze(0) == X.unsqueeze(1))
                    .sum(dim=-1)
                    .max(dim=0)
                    .values
                )
                # check that at least one dimension is perturbed
                self.assertTrue((20 - max_equal_dims >= 1).all())


class TestBoltzmannSample(BotorchTestCase):
    def test_boltzmann_sample(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float32, torch.float64):
            tkwargs["dtype"] = dtype

        function_values = torch.tensor([1.0, 2.0, 3.0, 4.0], **tkwargs)
        num_samples = 2
        eta = 1.0
        result = boltzmann_sample(function_values, num_samples, eta)
        self.assertEqual(result.shape, (num_samples,))

        # test batch dimensions
        function_values = torch.tensor(
            [[-1.0, 2.0, -3.0], [-4.0, -3.0, 1.0]], **tkwargs
        )
        num_samples = 2
        eta = 1.0
        result = boltzmann_sample(function_values, num_samples, eta)
        self.assertEqual(result.shape, (function_values.shape[0], num_samples))

        function_values = torch.tensor([1.0, 2.0, 3.0, 4.0], **tkwargs)
        num_samples = 5
        eta = 0.1

        # With replacement (should succeed even if num_samples > len(function_values))
        result_with_replacement = boltzmann_sample(
            function_values, num_samples, eta, replacement=True
        )
        self.assertEqual(result_with_replacement.shape, (num_samples,))

        # Without replacement (should fail if num_samples > len(function_values))
        with self.assertRaises(RuntimeError):
            boltzmann_sample(function_values, num_samples, eta, replacement=False)

        function_values = torch.tensor([1.0, 2.0, 3.0, 4.0], **tkwargs)
        num_samples = 2
        large_eta = 1000.0
        result = boltzmann_sample(function_values, num_samples, large_eta)
        self.assertEqual(result.shape, (num_samples,))
