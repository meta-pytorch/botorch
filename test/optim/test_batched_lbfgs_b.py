#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from botorch.optim.batched_lbfgs_b import (
    fmin_l_bfgs_b_batched,
    translate_bounds_for_lbfgsb,
)

from botorch.utils.testing import BotorchTestCase


class TestBatchedLBFGSB(BotorchTestCase):
    @staticmethod
    def quadratic_toy_func(x, batch_indices=None):
        f = np.sum(x**2, axis=-1)
        g = 2 * x
        if batch_indices is not None:
            # Just to verify batch_indices is passed correctly
            assert isinstance(batch_indices, list)
            assert len(batch_indices) == len(x)
        return f, g

    def test_basic_optimization(self):
        """Test basic optimization functionality."""
        func = self.quadratic_toy_func
        # Initial points
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Run optimization
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))
        self.assertEqual(len(results), 3)

        # All solutions should be close to 0
        self.assertTrue(np.allclose(xs, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(fs, 0.0, atol=1e-5))

        # Check that all optimizations were successful
        for result in results:
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], 0)

    def test_with_bounds(self):
        """Test optimization with bounds."""

        # Simple quadratic function: f(x) = (x - 2)^2
        def func(x):
            f = np.sum((x - 2) ** 2, axis=1)
            g = 2 * (x - 2)
            return f, g

        # Initial points
        x0 = np.array([[-1.0], [0.0], [1.0]])

        # Bounds: 0 <= x <= 1
        bounds = [(0, 1)]

        # Run optimization
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0, bounds=bounds)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))

        # All solutions should be at the upper bound (1.0)
        # since the minimum of (x - 2)^2 in [0, 1] is at x = 1
        self.assertTrue(np.allclose(xs, 1.0, atol=1e-5))
        self.assertTrue(np.allclose(fs, 1.0, atol=1e-5))  # f(1) = (1 - 2)^2 = 1

    def test_multidimensional(self):
        """Test optimization with multidimensional inputs."""

        # Rosenbrock function: f(x,y) = 10*(y - x^2)^2 + (1 - x)^2
        def func(x):
            a, b = 1.0, 10.0
            f = b * (x[:, 1] - x[:, 0] ** 2) ** 2 + (a - x[:, 0]) ** 2
            g = np.zeros_like(x)
            g[:, 0] = -2 * b * x[:, 0] * (x[:, 1] - x[:, 0] ** 2) - 2 * (a - x[:, 0])
            g[:, 1] = 2 * b * (x[:, 1] - x[:, 0] ** 2)
            return f, g

        # Initial points
        x0 = np.array(
            [
                [-1.0, 1.0],
                [2.0, 2.0],
            ]
        )

        # Run optimization
        # This does an open-ended optimization, but the full takes
        # only requires 0.2s on a Macbook
        xs, fs, _results = fmin_l_bfgs_b_batched(
            func,
            x0,
            factr=0,
            pgtol=1e-10,
            # interestingly, the bounds are important
            # to get good behavior, otherwise the optimizer
            # does go far off (in the C version) and finishes
            # early because it went so far away that everything
            # looks totally flat in the outcomes
            bounds=[(-2, 2), (-2, 2)],
        )

        # Check results
        self.assertEqual(xs.shape, (2, 2))
        self.assertEqual(fs.shape, (2,))

        # All solutions should be close to (1, 1)
        self.assertTrue(
            np.allclose(xs, np.array([[1.0, 1.0], [1.0, 1.0]]), atol=1e-1),
            msg=f"xs: {xs}",
        )
        self.assertTrue(np.allclose(fs, 0.0, atol=1e-3), fs)

    def test_pass_batch_indices(self):
        """Test the pass_batch_indices parameter."""

        # Function that uses batch_indices
        func = self.quadratic_toy_func

        # Initial points
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Run optimization with pass_batch_indices=True
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0, pass_batch_indices=True)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))
        self.assertEqual(len(results), 3)

        # All solutions should be close to 0
        self.assertTrue(np.allclose(xs, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(fs, 0.0, atol=1e-5))

    def test_callback(self):
        """Test optimization with callback function."""

        # Simple quadratic function: f(x) = x^2
        func = self.quadratic_toy_func

        # Callback function to track iterations
        iterations = [0, 0, 0]

        def callback(_):
            # _ is a single point, not a batch
            iterations[0] += 1
            return False  # Don't stop optimization

        # Initial points
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Run optimization with callback
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0, callback=callback)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))
        self.assertEqual(len(results), 3)

        # All solutions should be close to 0
        self.assertTrue(np.allclose(xs, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(fs, 0.0, atol=1e-5))

        # Check that callback was called at least once
        self.assertGreater(iterations[0], 0)

        # Add subtest with a callback that halts after one step
        with self.subTest("Halting after one step"):

            def one_step_callback(_):
                raise StopIteration

            xs_one_step, fs_one_step, results_one_step = fmin_l_bfgs_b_batched(
                func, x0, callback=one_step_callback
            )

            # Check that optimization stopped after one step
            self.assertEqual(xs_one_step.shape, (3, 1))
            self.assertEqual(fs_one_step.shape, (3,))
            self.assertEqual(len(results_one_step), 3)
            for result in results_one_step:
                self.assertFalse(result["success"])
                self.assertEqual(result["nit"], 1)

    def test_maxiter(self):
        """Test the maxiter parameter."""

        # Simple quadratic function: f(x) = x^2
        func = self.quadratic_toy_func

        # Initial points
        x0 = np.array([[10.0], [20.0], [30.0]])

        # Run optimization with maxiter=1
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0, maxiter=1)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))
        self.assertEqual(len(results), 3)

        # Solutions should not be close to 0 due to limited iterations
        self.assertFalse(np.allclose(xs, 0.0, atol=1e-5))
        self.assertFalse(np.allclose(fs, 0.0, atol=1e-5))

        # Check that all optimizations were stopped due to maxiter
        for result in results:
            self.assertFalse(result["success"])
            self.assertEqual(result["status"], 1)
            self.assertEqual(result["nit"], 1)

    def test_ftol(self):
        """Test the ftol parameter."""

        # Simple quadratic function: f(x) = x^2
        func = self.quadratic_toy_func

        # Initial points
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Run optimization with very strict ftol
        xs, fs, results = fmin_l_bfgs_b_batched(func, x0, ftol=1e-15, factr=None)

        # Check results
        self.assertEqual(xs.shape, (3, 1))
        self.assertEqual(fs.shape, (3,))
        self.assertEqual(len(results), 3)

        # Solutions should be very close to 0 due to strict tolerance
        self.assertTrue(np.allclose(xs, 0.0, atol=1e-7))
        self.assertTrue(np.allclose(fs, 0.0, atol=1e-14))

    def test_factr_ftol_exclusive(self):
        """Test that factr and ftol cannot be used together."""

        # Simple quadratic function: f(x) = x^2
        func = self.quadratic_toy_func

        x0 = np.array([[1.0]])

        # Both factr and ftol provided should raise an error
        with self.assertRaises(AssertionError):
            fmin_l_bfgs_b_batched(func, x0, factr=1e7, ftol=1e-10)

    def test_invalid_bounds_shape(self):
        """Test optimization with bounds of the wrong shape."""

        func = self.quadratic_toy_func

        x0 = np.array([[1.0], [2.0], [3.0]])

        # Incorrect bounds shape: should be a list of tuples with
        # length equal to the number of dimensions
        bounds = [(0, 1), (0, 1)]  # Incorrect shape for 1D input

        with self.assertRaises(ValueError):
            fmin_l_bfgs_b_batched(func, x0, bounds=bounds)

    def test_invalid_bounds_order(self):
        """Test optimization with bounds of the wrong order."""

        func = self.quadratic_toy_func
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Bounds with wrong order: lower bound is greater than upper bound
        bounds = [(1, 0)]  # Incorrect order

        with self.assertRaises(ValueError):
            fmin_l_bfgs_b_batched(func, x0, bounds=bounds)

    def test_maxls_zero(self):
        """Test the maxls parameter with zero value."""

        # Simple quadratic function: f(x) = x^2
        func = self.quadratic_toy_func

        # Initial points
        x0 = np.array([[1.0], [2.0], [3.0]])

        # Run optimization with maxls=0, which should raise a ValueError
        with self.assertRaises(ValueError):
            fmin_l_bfgs_b_batched(func, x0, maxls=0)


class TestTranslateBoundsForLBFGSB(BotorchTestCase):
    def test_no_bounds(self):
        """Test translate_bounds_for_lbfgsb with no bounds."""

        lower_bounds = None
        upper_bounds = None
        num_features = 3
        expected_bounds = [(None, None), (None, None), (None, None)]
        self.assertEqual(
            translate_bounds_for_lbfgsb(lower_bounds, upper_bounds, num_features, 1),
            expected_bounds,
        )

    def test_scalar_bounds(self):
        """Test translate_bounds_for_lbfgsb with scalar bounds."""

        lower_bounds = 0.0
        upper_bounds = 1.0
        num_features = 3
        expected_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.assertEqual(
            translate_bounds_for_lbfgsb(lower_bounds, upper_bounds, num_features, 1),
            expected_bounds,
        )

    def test_array_bounds(self):
        """Test translate_bounds_for_lbfgsb with array bounds."""

        lower_bounds = [0.0, 0.1, 0.2]
        upper_bounds = [1.0, 1.1, 1.2]
        num_features = 3
        expected_bounds = [(0.0, 1.0), (0.1, 1.1), (0.2, 1.2)]
        self.assertAlmostEqual(
            translate_bounds_for_lbfgsb(lower_bounds, upper_bounds, num_features, 1),
            expected_bounds,
        )

    def test_tensor_bounds(self):
        """Test translate_bounds_for_lbfgsb with tensor bounds."""

        lower_bounds = torch.tensor([0.0, 0.1, 0.2])
        upper_bounds = torch.tensor([1.0, 1.1, 1.2])
        num_features = 3
        expected_bounds = [(0.0, 1.0), (0.1, 1.1), (0.2, 1.2)]
        self.assertAlmostEqual(
            translate_bounds_for_lbfgsb(lower_bounds, upper_bounds, num_features, 1),
            expected_bounds,
        )

    def test_q_equals_2(self):
        """Test translate_bounds_for_lbfgsb with q=2."""

        lower_bounds = [0.0, 0.1, 0.2]
        upper_bounds = [1.0, 1.1, 1.2]
        num_features = 3
        q = 2
        expected_bounds = [
            (0.0, 1.0),
            (0.1, 1.1),
            (0.2, 1.2),
            (0.0, 1.0),
            (0.1, 1.1),
            (0.2, 1.2),
        ]
        self.assertEqual(
            translate_bounds_for_lbfgsb(lower_bounds, upper_bounds, num_features, q),
            expected_bounds,
        )
