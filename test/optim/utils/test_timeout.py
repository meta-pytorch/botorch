#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings

import numpy as np
import numpy.typing as npt
from botorch.optim.utils.timeout import minimize_with_timeout
from botorch.utils.testing import BotorchTestCase
from scipy.optimize import minimize, OptimizeResult


class TestMinimizeWithTimeout(BotorchTestCase):
    def test_minimize_with_timeout(self):
        def f_and_g(x: npt.NDArray, sleep_sec: float = 0.0):
            time.sleep(sleep_sec)
            return x**2, 2 * x

        base_kwargs = {
            "fun": f_and_g,
            "x0": np.array([1.0]),
            "method": "L-BFGS-B",
            "jac": True,
            "bounds": [(-2.0, 2.0)],
        }

        with self.subTest("test w/o timeout"):
            res = minimize_with_timeout(**base_kwargs)
            self.assertTrue(res.success)
            self.assertAlmostEqual(res.fun, 0.0)
            self.assertAlmostEqual(res.x.item(), 0.0)
            self.assertEqual(res.nit, 2)  # quadratic approx. is exact

        with self.subTest("test w/ non-binding timeout"):
            res = minimize_with_timeout(**base_kwargs, timeout_sec=1.0)
            self.assertTrue(res.success)
            self.assertAlmostEqual(res.fun, 0.0)
            self.assertAlmostEqual(res.x.item(), 0.0)
            self.assertEqual(res.nit, 2)  # quadratic approx. is exact

        with self.subTest("test w/ binding timeout"):
            for timeout_sec in [0, 1e-4]:
                res = minimize_with_timeout(
                    **base_kwargs, args=(1e-2,), timeout_sec=timeout_sec
                )
                self.assertFalse(res.success)
                self.assertEqual(res.nit, 1)  # only one call to the callback is made
                self.assertIn("Optimization timed out", res.message)

        # set up callback with mutable object to verify callback execution
        check_set = set()

        def callback(x: npt.NDArray) -> None:
            check_set.add("foo")

        with self.subTest("test w/ callout argument and non-binding timeout"):
            res = minimize_with_timeout(
                **base_kwargs, callback=callback, timeout_sec=1.0
            )
            self.assertTrue(res.success)
            self.assertTrue("foo" in check_set)

        # set up callback for method ``trust-constr`` w/ different signature
        check_set.clear()
        self.assertFalse("foo" in check_set)

        def callback_trustconstr(x: npt.NDArray, state: OptimizeResult) -> bool:
            check_set.add("foo")
            return False

        with self.subTest("test `trust-constr` method w/ callback"):
            res = minimize_with_timeout(
                **{**base_kwargs, "method": "trust-constr"},
                callback=callback_trustconstr,
            )
            self.assertTrue(res.success)
            self.assertTrue("foo" in check_set)

        # reset check set
        check_set.clear()
        self.assertFalse("foo" in check_set)

        with self.subTest("test `trust-constr` method w/ callback and timeout"):
            res = minimize_with_timeout(
                **{**base_kwargs, "method": "trust-constr"},
                args=(1e-3,),
                callback=callback_trustconstr,
                timeout_sec=1e-4,
            )
            self.assertFalse(res.success)
            self.assertTrue("foo" in check_set)

        with self.subTest("verify error if passing callable for `method` w/ timeout"):
            with self.assertRaisesRegex(
                NotImplementedError, "Custom callable not supported"
            ):
                minimize_with_timeout(
                    **{**base_kwargs, "method": lambda *args, **kwargs: None},
                    callback=callback,
                    timeout_sec=1e-4,
                )

        with self.subTest("test w/ binding timeout and without gradient"):

            def f(x: npt.NDArray, sleep_sec: float = 0.0) -> float:
                time.sleep(sleep_sec)
                return float((x**2).sum())

            res = minimize_with_timeout(
                **{**base_kwargs, "fun": f, "jac": None},
                args=(1e-2,),
                timeout_sec=1e-4,
            )
            self.assertFalse(res.success)
            self.assertIn("Optimization timed out", res.message)
            self.assertEqual(res.fun, f(res.x))

    def test_minimize_with_timeout_warning_filters(self):
        # Warnings about methods that cannot handle bounds or constraints are raised
        # as errors within ``minimize_with_timeout``, without modifying the warning
        # filters outside of it.
        filters = list(warnings.filters)
        minimize_with_timeout(
            fun=lambda x: float((x**2).sum()), x0=np.array([1.0]), method="L-BFGS-B"
        )
        self.assertEqual(warnings.filters, filters)
        kwargs = {
            "fun": lambda x: float((x**2).sum()),
            "x0": np.array([1.0]),
            "method": "Nelder-Mead",
            "constraints": [{"type": "ineq", "fun": lambda x: x[0]}],
        }
        with self.assertRaisesRegex(RuntimeWarning, "cannot handle constraints"):
            minimize_with_timeout(**kwargs)
        self.assertEqual(warnings.filters, filters)
        with self.assertWarnsRegex(RuntimeWarning, "cannot handle constraints"):
            minimize(**kwargs)
