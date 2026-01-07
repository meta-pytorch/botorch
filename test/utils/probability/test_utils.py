#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools

import numpy as np

import torch
from botorch.utils.probability import ndtr, utils
from botorch.utils.probability.utils import (
    log_erfc,
    log_erfcx,
    log_ndtr,
    log_phi,
    log_prob_normal_in,
    percentile_of_score,
    phi,
    standard_normal_log_hazard,
)
from botorch.utils.testing import BotorchTestCase
from numpy.polynomial.legendre import leggauss as numpy_leggauss
from scipy.stats import percentileofscore as percentile_of_score_scipy


class TestProbabilityUtils(BotorchTestCase):
    def test_case_dispatcher(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand([32])

        # Test default case
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            default=lambda mask: 0,
        )
        self.assertTrue(output.eq(0).all())

        # Test randomized value assignments
        levels = 0.25, 0.5, 0.75
        cases = [  # switching cases
            (lambda level=level: values < level, lambda mask, i=i: i)
            for i, level in enumerate(levels)
        ]

        cases.append(  # dummy case whose predicate is always False
            (lambda: torch.full(values.shape, False), lambda mask: float("nan"))
        )

        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=cases,
            default=lambda mask: len(levels),
        )

        self.assertTrue(output.isfinite().all())
        active = torch.full(values.shape, True)
        for i, level in enumerate(levels):
            mask = active & (values < level)
            self.assertTrue(output[mask].eq(i).all())
            active[mask] = False
        self.assertTrue(~active.any() or output[active].eq(len(levels)).all())

        # testing mask.all() branch
        edge_cases = [
            (lambda: torch.full(values.shape, True), lambda mask: float("nan"))
        ]
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=edge_cases,
            default=lambda mask: len(levels),
        )

        # testing if not active.any() branch
        pred = torch.full(values.shape, True)
        pred[0] = False
        edge_cases = [
            (lambda: pred, lambda mask: False),
            (lambda: torch.full(values.shape, True), lambda mask: False),
        ]
        output = utils.case_dispatcher(
            out=torch.full_like(values, float("nan")),
            cases=edge_cases,
            default=lambda mask: len(levels),
        )

    def test_build_positional_indices(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand(3, 2, 5)

        for dim in (values.ndim, -values.ndim - 1):
            with self.assertRaisesRegex(ValueError, r"dim=(-?\d+) invalid for shape"):
                utils.build_positional_indices(shape=values.shape, dim=dim)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        self.assertEqual(start.shape, values.shape[:-1])
        self.assertTrue(start.remainder(values.shape[-1]).eq(0).all())

        max_values, max_indices = values.max(dim=-1)
        self.assertTrue(values.view(-1)[start + max_indices].equal(max_values))

    def test_leggaus(self):
        for a, b in zip(utils.leggauss(20, dtype=torch.float64), numpy_leggauss(20)):
            self.assertEqual(a.dtype, torch.float64)
            self.assertTrue((a.numpy() == b).all())

    def test_swap_along_dim_(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            values = torch.rand(3, 2, 5)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        min_values, i = values.min(dim=-1)
        max_values, j = values.max(dim=-1)
        out = utils.swap_along_dim_(values.clone(), i=i, j=j, dim=-1)

        # Verify that positions of minimum and maximum values were swapped
        for vec, min_val, min_idx, max_val, max_idx in zip(
            out.view(-1, values.shape[-1]),
            min_values.ravel(),
            i.ravel(),
            max_values.ravel(),
            j.ravel(),
        ):
            self.assertEqual(vec[min_idx], max_val)
            self.assertEqual(vec[max_idx], min_val)

        start = utils.build_positional_indices(shape=values.shape, dim=-2)
        i_lidx = (start + i).ravel()
        j_lidx = (start + j).ravel()

        # Test passing in a pre-allocated copy buffer
        temp = values.view(-1).clone()[i_lidx]
        buff = torch.empty_like(temp)
        out2 = utils.swap_along_dim_(values.clone(), i=i, j=j, dim=-1, buffer=buff)
        self.assertTrue(out.equal(out2))
        self.assertTrue(temp.equal(buff))

        # Test homogeneous swaps
        temp = utils.swap_along_dim_(values.clone(), i=0, j=2, dim=-1)
        self.assertTrue(values[..., 0].equal(temp[..., 2]))
        self.assertTrue(values[..., 2].equal(temp[..., 0]))

        # Test exception handling
        with self.assertRaisesRegex(ValueError, "Batch shapes of `i`"):
            utils.swap_along_dim_(values, i=i.unsqueeze(-1), j=j, dim=-1)

        with self.assertRaisesRegex(ValueError, "Batch shapes of `j`"):
            utils.swap_along_dim_(values, i=i, j=j.unsqueeze(-1), dim=-1)

        with self.assertRaisesRegex(ValueError, "at most 1-dimensional"):
            utils.swap_along_dim_(values.view(-1), i=i, j=j_lidx, dim=0)

        with self.assertRaisesRegex(ValueError, "at most 1-dimensional"):
            utils.swap_along_dim_(values.view(-1), i=i_lidx, j=j, dim=0)

    def test_gaussian_probabilities(self) -> None:
        # test passes for each possible seed
        torch.manual_seed(torch.randint(high=1000, size=(1,)))
        # testing Gaussian probability functions
        for dtype in (torch.float, torch.double):
            rtol = 1e-12 if dtype == torch.double else 1e-6
            atol = rtol
            n = 16
            x = 3 * torch.randn(n, device=self.device, dtype=dtype)
            # first, test consistency between regular and log versions
            self.assertAllClose(phi(x), log_phi(x).exp(), atol=atol, rtol=rtol)
            self.assertAllClose(ndtr(x), log_ndtr(x).exp(), atol=atol, rtol=rtol)

            # test correctness of log_erfc and log_erfcx
            for special_f, custom_log_f in zip(
                (torch.special.erfc, torch.special.erfcx), (log_erfc, log_erfcx)
            ):
                with self.subTest(custom_log_f.__name__):
                    # first, testing for moderate values
                    n = 16
                    x = torch.rand(n, dtype=dtype, device=self.device)
                    x = torch.cat((-x, x))
                    x.requires_grad = True
                    custom_log_fx = custom_log_f(x)
                    special_log_fx = special_f(x).log()
                    self.assertAllClose(
                        custom_log_fx, special_log_fx, atol=atol, rtol=rtol
                    )
                    # testing backward passes
                    custom_log_fx.sum().backward()
                    x_grad = x.grad
                    x.grad[:] = 0
                    special_log_fx.sum().backward()
                    special_x_grad = x.grad
                    self.assertAllClose(x_grad, special_x_grad, atol=atol, rtol=rtol)

                    # testing robustness of log_erfc for large inputs
                    # large positive numbers are difficult for a naive implementation
                    x = torch.tensor(
                        [1e100 if dtype == torch.float64 else 1e10],
                        dtype=dtype,
                        device=self.device,
                    )
                    x = torch.cat((-x, x))  # looking at both tails
                    x.requires_grad = True
                    custom_log_fx = custom_log_f(x)
                    self.assertAllClose(
                        custom_log_fx.exp(),
                        special_f(x),
                        atol=atol,
                        rtol=rtol,
                    )
                    self.assertFalse(custom_log_fx.isnan().any())
                    self.assertFalse(custom_log_fx.isinf().any())
                    # we can't just take the log of erfc because the tail will be -inf
                    self.assertTrue(special_f(x).log().isinf().any())
                    # testing that gradients are usable floats
                    custom_log_fx.sum().backward()
                    self.assertFalse(x.grad.isnan().any())
                    self.assertFalse(x.grad.isinf().any())

            # test limit behavior of log_ndtr
            digits = 100 if dtype == torch.float64 else 20
            # zero = torch.tensor([0], dtype=dtype, device=self.device)
            ten = torch.tensor(10, dtype=dtype, device=self.device)
            digits_tensor = torch.arange(0, digits, dtype=dtype, device=self.device)
            # large negative values
            x_large_neg = -(ten ** digits_tensor.flip(-1))  # goes from -1e100 to -1
            x_large_pos = ten**digits_tensor  # goes from 1 to 1e100
            x = torch.cat((x_large_neg, x_large_pos))
            x.requires_grad = True

            torch_log_ndtr_x = torch.special.log_ndtr(x)
            log_ndtr_x = log_ndtr(x)
            self.assertTrue(
                torch.allclose(log_ndtr_x, torch_log_ndtr_x, atol=atol, rtol=rtol)
            )

            # let's test gradients too
            # first, note that the standard implementation exhibits numerical problems:
            # 1) it contains -Inf for reasonable parameter ranges, and
            # 2) the gradient is not strictly increasing, even ignoring Infs, and
            # takes non-sensical values (i.e. ~4e-01 at x = -1e100 in single precision,
            # and similar for some large negative x in double precision).
            torch_log_ndtr_x = torch.special.log_ndtr(x)
            torch_log_ndtr_x.sum().backward()
            torch_grad = x.grad.clone()
            self.assertTrue(torch_grad.isinf().any())

            # in contrast, our implementation permits numerically accurate gradients
            # throughout the testest range:
            x.grad[:] = 0  # zero out gradient
            log_ndtr_x.sum().backward()
            grad = x.grad.clone()
            # it does not contain Infs or NaNs
            self.assertFalse(grad.isinf().any())
            self.assertFalse(grad.isnan().any())
            # gradients are non-negative everywhere (approach zero as x goes to inf)
            self.assertTrue((grad[:digits] > 0).all())
            self.assertTrue((grad[digits:] >= 0).all())
            # gradients are strictly decreasing for x < 0
            self.assertTrue((grad.diff()[:digits] < 0).all())
            self.assertTrue((grad.diff()[digits:] <= 0).all())

            n = 16
            # first test is easiest: a < 0 < b
            a = -5 / 2 * torch.rand(n, dtype=dtype, device=self.device) - 1 / 2
            b = 5 / 2 * torch.rand(n, dtype=dtype, device=self.device) + 1 / 2
            self.assertTrue(
                torch.allclose(
                    log_prob_normal_in(a, b).exp(),
                    ndtr(b) - ndtr(a),
                    atol=atol,
                    rtol=rtol,
                )
            )

            # 0 < a < b, uses the a < b < 0 under the hood
            a = ten ** digits_tensor[:-1]
            b = ten ** digits_tensor[-1]
            a.requires_grad, b.requires_grad = True, True
            log_prob = log_prob_normal_in(a, b)
            self.assertTrue((log_prob < 0).all())
            self.assertTrue((log_prob.diff() < 0).all())

            # test gradients
            log_prob.sum().backward()
            # checking that both gradients give non-Inf, non-NaN results everywhere
            self.assertFalse(a.grad.isinf().any())
            self.assertFalse(a.grad.isnan().any())
            self.assertFalse(b.grad.isinf().any())
            self.assertFalse(b.grad.isnan().any())
            # since the upper bound is satisfied, relevant gradients are in lower bound
            self.assertTrue((a.grad.diff() < 0).all())

            # testing error raising for invalid inputs
            a = torch.randn(3, 4, dtype=dtype, device=self.device)
            b = torch.randn(3, 4, dtype=dtype, device=self.device)
            a[2, 3] = b[2, 3]
            with self.assertRaisesRegex(
                ValueError,
                "Received input tensors a, b for which not all a < b.",
            ):
                log_prob_normal_in(a, b)

            # testing gaussian hazard function
            n = 16
            x = torch.rand(n, dtype=dtype, device=self.device)
            x = torch.cat((-x, x))
            log_hx = standard_normal_log_hazard(x)
            expected_log_hx = log_phi(x) - log_ndtr(-x)
            self.assertAllClose(
                expected_log_hx,
                log_hx,
                atol=1e-8 if dtype == torch.double else 1e-7,
            )  # correctness
            # NOTE: Could extend tests here similarly to log_erfc(x) tests above, but
            # since the hazard functions are built on log_erfcx, not urgent.

        float16_msg = (
            "only supports torch.float32 and torch.float64 dtypes, but received "
            "x.dtype=torch.float16."
        )
        with self.assertRaisesRegex(TypeError, expected_regex=float16_msg):
            log_erfc(torch.tensor(1.0, dtype=torch.float16, device=self.device))

        with self.assertRaisesRegex(TypeError, expected_regex=float16_msg):
            log_ndtr(torch.tensor(1.0, dtype=torch.float16, device=self.device))

    def test_percentile_of_score(self) -> None:
        # compare to scipy.stats.percentileofscore with default settings
        # `kind='rank'` and `nan_policy='propagate'`
        torch.manual_seed(12345)
        n = 10
        for (
            dtype,
            data_batch_shape,
            score_batch_shape,
            output_shape,
        ) in itertools.product(
            (torch.float, torch.double),
            ((), (1,), (2,), (2, 3)),
            ((), (1,), (2,), (2, 3)),
            ((), (1,), (2,), (2, 3)),
        ):
            # calculate shapes
            data_shape = data_batch_shape + (n,) + output_shape
            score_shape = score_batch_shape + (1,) + output_shape
            dim = -1 - len(output_shape)
            # generate data
            data = torch.rand(*data_shape, dtype=dtype, device=self.device)
            score = torch.rand(*score_shape, dtype=dtype, device=self.device)
            # insert random nans to test nan policy
            data[data < 0.01] = torch.nan
            score[score < 0.01] = torch.nan
            # calculate percentile ranks using torch
            try:
                perct_torch = percentile_of_score(data, score, dim=dim).cpu().numpy()
            except RuntimeError:
                # confirm RuntimeError is raised because shapes cannot be broadcasted
                with self.assertRaises(ValueError):
                    np.broadcast_shapes(data_batch_shape, score_batch_shape)
                continue
            # check shape
            broadcast_shape = np.broadcast_shapes(data_batch_shape, score_batch_shape)
            expected_perct_shape = broadcast_shape + output_shape
            self.assertEqual(perct_torch.shape, expected_perct_shape)
            # calculate percentile ranks using scipy.stats.percentileofscore
            # scipy.stats.percentileofscore does not support broadcasting
            # loop over batch and output shapes instead
            perct_scipy = np.zeros_like(perct_torch)
            data_scipy = np.broadcast_to(
                data.cpu().numpy(), broadcast_shape + (n,) + output_shape
            )
            score_scipy = np.broadcast_to(
                score.cpu().numpy(), broadcast_shape + (1,) + output_shape
            )
            broadcast_idx_prod = list(
                itertools.product(*[list(range(d)) for d in broadcast_shape])
            )
            output_idx_prod = list(
                itertools.product(*[list(range(d)) for d in output_shape])
            )
            for broadcast_idx in broadcast_idx_prod:
                for output_idx in output_idx_prod:
                    data_idx = broadcast_idx + (slice(None),) + output_idx
                    score_idx = broadcast_idx + (0,) + output_idx
                    perct_idx = broadcast_idx + output_idx
                    perct_scipy[perct_idx] = percentile_of_score_scipy(
                        data_scipy[data_idx], score_scipy[score_idx]
                    )
            self.assertTrue(np.array_equal(perct_torch, perct_scipy, equal_nan=True))
