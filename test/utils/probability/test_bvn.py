#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

from itertools import count
from typing import Any

import torch
from botorch.exceptions import UnsupportedError
from botorch.utils.probability.bvn import (
    _bvnu_polar,
    _bvnu_taylor,
    bvn,
    bvnmom,
    bvnu,
    Phi,
)
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


def run_gaussian_estimator(
    estimator: Callable[[Tensor], tuple[Tensor, Tensor | float | int]],
    sqrt_cov: Tensor,
    num_samples: int,
    batch_limit: int | None = None,
    seed: int | None = None,
) -> Tensor:
    if batch_limit is None:
        batch_limit = num_samples

    ndim = sqrt_cov.shape[-1]
    tkwargs = {"dtype": sqrt_cov.dtype, "device": sqrt_cov.device}
    counter = 0
    numerator = 0
    denominator = 0

    with torch.random.fork_rng():
        if seed:
            torch.random.manual_seed(seed)

        while counter < num_samples:
            batch_size = min(batch_limit, num_samples - counter)
            samples = torch.tensordot(
                torch.randn(batch_size, ndim, **tkwargs),
                sqrt_cov,
                dims=([1], [-1]),
            )

            batch_numerator, batch_denominator = estimator(samples)
            counter = counter + batch_size
            numerator = numerator + batch_numerator
            denominator = denominator + batch_denominator

    return numerator / denominator, denominator


class TestBVN(BotorchTestCase):
    def setUp(
        self,
        nprobs_per_coeff: int = 3,
        bound_range: tuple[float, float] = (-3.0, 3.0),
        mc_num_samples: int = 10000,
        mc_batch_limit: int = 1000,
        mc_atol_multiplier: float = 4.0,
        seed: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ):
        super().setUp()
        self.dtype = dtype
        self.seed_generator = count(seed)
        self.nprobs_per_coeff = nprobs_per_coeff
        self.mc_num_samples = mc_num_samples
        self.mc_batch_limit = mc_batch_limit
        self.mc_atol_multiplier = mc_atol_multiplier

        pos_coeffs = torch.cat(
            [
                torch.linspace(0, 1, 5, **self.tkwargs),
                torch.tensor([0.01, 0.05, 0.924, 0.925, 0.99], **self.tkwargs),
            ]
        )
        self.correlations = torch.cat([pos_coeffs, -pos_coeffs[1:]])

        with torch.random.fork_rng():
            torch.manual_seed(next(self.seed_generator))
            _lower = torch.rand(
                nprobs_per_coeff, len(self.correlations), 2, **self.tkwargs
            )
            _upper = _lower + (1 - _lower) * torch.rand_like(_lower)

        self.lower_bounds = bound_range[0] + (bound_range[1] - bound_range[0]) * _lower
        self.upper_bounds = bound_range[0] + (bound_range[1] - bound_range[0]) * _upper

        self.sqrt_covariances = torch.zeros(
            len(self.correlations), 2, 2, **self.tkwargs
        )
        self.sqrt_covariances[:, 0, 0] = 1
        self.sqrt_covariances[:, 1, 0] = self.correlations
        self.sqrt_covariances[:, 1, 1] = (1 - self.correlations**2) ** 0.5

    @property
    def tkwargs(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "device": self.device}

    @property
    def xl(self):
        return self.lower_bounds[..., 0]

    @property
    def xu(self):
        return self.upper_bounds[..., 0]

    @property
    def yl(self):
        return self.lower_bounds[..., 1]

    @property
    def yu(self):
        return self.upper_bounds[..., 1]

    def test_bvnu_polar(self) -> None:
        r"""Test special cases where bvnu admits closed-form solutions.

        Note: inf should not be passed to _bvnu as bounds, use big numbers instead.
        """
        use_polar = self.correlations.abs() < 0.925
        r = self.correlations[use_polar]
        xl = self.xl[..., use_polar]
        yl = self.yl[..., use_polar]
        with self.subTest(msg="exact_unconstrained"):
            prob = _bvnu_polar(r, torch.full_like(r, -1e16), torch.full_like(r, -1e16))
            self.assertAllClose(prob, torch.ones_like(prob))

        with self.subTest(msg="exact_marginal"):
            prob = _bvnu_polar(
                r.expand_as(yl),
                torch.full_like(xl, -1e16),
                yl,
            )
            test = Phi(-yl)  # same as: 1 - P(y < yl)
            self.assertAllClose(prob, test)

        with self.subTest(msg="exact_independent"):
            prob = _bvnu_polar(torch.zeros_like(xl), xl, yl)
            test = Phi(-xl) * Phi(-yl)
            self.assertAllClose(prob, test)

    def test_bvnu_taylor(self) -> None:
        r"""Test special cases where bvnu admits closed-form solutions.

        Note: inf should not be passed to _bvnu as bounds, use big numbers instead.
        """
        use_taylor = self.correlations.abs() >= 0.925
        r = self.correlations[use_taylor]
        xl = self.xl[..., use_taylor]
        yl = self.yl[..., use_taylor]
        with self.subTest(msg="exact_unconstrained"):
            prob = _bvnu_taylor(r, torch.full_like(r, -1e16), torch.full_like(r, -1e16))
            self.assertAllClose(prob, torch.ones_like(prob))

        with self.subTest(msg="exact_marginal"):
            prob = _bvnu_taylor(
                r.expand_as(yl),
                torch.full_like(xl, -1e16),
                yl,
            )
            test = Phi(-yl)  # same as: 1 - P(y < yl)
            self.assertAllClose(prob, test)

        with self.subTest(msg="exact_independent"):
            prob = _bvnu_polar(torch.zeros_like(xl), xl, yl)
            test = Phi(-xl) * Phi(-yl)
            self.assertAllClose(prob, test)

    def test_bvn(self):
        r"""Monte Carlo unit test for `bvn`."""
        r = self.correlations.repeat(self.nprobs_per_coeff, 1)
        solves = bvn(r, self.xl, self.yl, self.xu, self.yu)
        with self.assertRaisesRegex(UnsupportedError, "same shape"):
            bvn(r[..., :1], self.xl, self.yl, self.xu, self.yu)

        with self.assertRaisesRegex(UnsupportedError, "same shape"):
            bvnu(r[..., :1], r, r)

        def _estimator(samples):
            accept = torch.logical_and(
                (samples > self.lower_bounds.unsqueeze(1)).all(-1),
                (samples < self.upper_bounds.unsqueeze(1)).all(-1),
            )
            numerator = torch.count_nonzero(accept, dim=1).double()
            denominator = len(samples)
            return numerator, denominator

        estimates, _ = run_gaussian_estimator(
            estimator=_estimator,
            sqrt_cov=self.sqrt_covariances,
            num_samples=self.mc_num_samples,
            batch_limit=self.mc_batch_limit,
            seed=next(self.seed_generator),
        )

        atol = self.mc_atol_multiplier * (self.mc_num_samples**-0.5)
        self.assertAllClose(estimates, solves, rtol=0, atol=atol)

    def test_bvnmom(self):
        r"""Monte Carlo unit test for `bvn`."""
        r = self.correlations.repeat(self.nprobs_per_coeff, 1)
        Ex, Ey = bvnmom(r, self.xl, self.yl, self.xu, self.yu)
        with self.assertRaisesRegex(UnsupportedError, "same shape"):
            bvnmom(r[..., :1], self.xl, self.yl, self.xu, self.yu)

        def _estimator(samples):
            accept = torch.logical_and(
                (samples > self.lower_bounds.unsqueeze(1)).all(-1),
                (samples < self.upper_bounds.unsqueeze(1)).all(-1),
            )
            numerator = torch.einsum("snd,psn->pnd", samples, accept.to(samples.dtype))
            denominator = torch.count_nonzero(accept, dim=1).to(samples.dtype)
            return numerator, denominator.unsqueeze(-1)

        estimates, num_samples = run_gaussian_estimator(
            estimator=_estimator,
            sqrt_cov=self.sqrt_covariances,
            num_samples=self.mc_num_samples,
            batch_limit=self.mc_batch_limit,
            seed=next(self.seed_generator),
        )
        for n, ex, ey, _ex, _ey in zip(
            *map(torch.ravel, (num_samples.squeeze(-1), Ex, Ey, *estimates.unbind(-1)))
        ):
            if n:
                atol = self.mc_atol_multiplier * (n**-0.5)
                self.assertAllClose(ex, _ex, rtol=0, atol=atol)
                self.assertAllClose(ey, _ey, rtol=0, atol=atol)
