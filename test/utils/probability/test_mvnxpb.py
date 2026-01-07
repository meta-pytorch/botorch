#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence

from copy import deepcopy

from functools import partial
from itertools import count, product
from typing import Any
from unittest.mock import patch

import torch
from botorch.utils.probability.linalg import PivotedCholesky
from botorch.utils.probability.mvnxpb import MVNXPB
from botorch.utils.testing import BotorchTestCase
from linear_operator.utils.errors import NotPSDError
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


class TestMVNXPB(BotorchTestCase):
    def setUp(
        self,
        ndims: Sequence[int] = (4, 8),
        batch_shape: Sequence[int] = (4,),
        bound_range: tuple[float, float] = (-5.0, 5.0),
        mc_num_samples: int = 100000,
        mc_batch_limit: int = 10000,
        mc_atol_multiplier: float = 4.0,
        seed: int = 1,
        dtype: torch.dtype = torch.float64,
    ):
        super().setUp()
        self.dtype = dtype
        self.seed_generator = count(seed)
        self.mc_num_samples = mc_num_samples
        self.mc_batch_limit = mc_batch_limit
        self.mc_atol_multiplier = mc_atol_multiplier

        self.bounds = []
        self.sqrt_covariances = []
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            for n in ndims:
                self.bounds.append(self.gen_bounds(n, batch_shape, bound_range))
                self.sqrt_covariances.append(
                    self.gen_covariances(n, batch_shape, as_sqrt=True)
                )

        # Create a toy MVNXPB instance for API testing
        tril = torch.rand([4, 2, 3, 3], **self.tkwargs)
        diag = torch.rand([4, 2, 3], **self.tkwargs)
        perm = torch.stack([torch.randperm(3) for _ in range(8)])
        perm = perm.reshape(4, 2, 3).to(**self.tkwargs)
        self.toy_solver = MVNXPB.build(
            step=0,
            perm=perm.clone(),
            bounds=torch.rand(4, 2, 3, 2, **self.tkwargs).cumsum(dim=-1),
            piv_chol=PivotedCholesky(tril=tril, perm=perm, diag=diag, step=0),
            plug_ins=torch.randn(4, 2, 3, **self.tkwargs),
            log_prob=torch.rand(4, 2, **self.tkwargs),
        )

    def gen_covariances(
        self,
        ndim: int,
        batch_shape: Sequence[int] = (),
        as_sqrt: bool = False,
    ) -> Tensor:
        shape = tuple(batch_shape) + (ndim, ndim)
        eigvals = -torch.rand(shape[:-1], **self.tkwargs).log()  # exponential rvs
        orthmat = torch.linalg.svd(torch.randn(shape, **self.tkwargs)).U
        sqrt_covar = orthmat * torch.sqrt(eigvals).unsqueeze(-2)
        return sqrt_covar if as_sqrt else sqrt_covar @ sqrt_covar.transpose(-2, -1)

    def gen_bounds(
        self,
        ndim: int,
        batch_shape: Sequence[int] = (),
        bound_range: tuple[float, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        shape = tuple(batch_shape) + (ndim,)
        lower = torch.rand(shape, **self.tkwargs)
        upper = lower + (1 - lower) * torch.rand_like(lower)
        if bound_range is not None:
            lower = bound_range[0] + (bound_range[1] - bound_range[0]) * lower
            upper = bound_range[0] + (bound_range[1] - bound_range[0]) * upper

        return torch.stack([lower, upper], dim=-1)

    @property
    def tkwargs(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "device": self.device}

    def assertEqualMXNBPB(self, A: MVNXPB, B: MVNXPB):
        for key, a in A.asdict().items():
            b = getattr(B, key)
            if isinstance(a, PivotedCholesky):
                continue
            elif isinstance(a, torch.Tensor):
                self.assertTrue(a.allclose(b, equal_nan=True))
            else:
                self.assertEqual(a, b)

        for key in ("perm", "tril", "diag"):
            a = getattr(A.piv_chol, key)
            b = getattr(B.piv_chol, key)
            self.assertTrue(a.allclose(b, equal_nan=True))

    def test_solve(self):
        r"""Monte Carlo unit test for `solve`."""

        def _estimator(samples, bounds):
            accept = torch.logical_and(
                (samples > bounds[..., 0]).all(-1),
                (samples < bounds[..., 1]).all(-1),
            )
            numerator = torch.count_nonzero(accept, dim=0).double()
            denominator = len(samples)
            return numerator, denominator

        for sqrt_cov, bounds in zip(self.sqrt_covariances, self.bounds):
            estimates, _ = run_gaussian_estimator(
                estimator=partial(_estimator, bounds=bounds),
                sqrt_cov=sqrt_cov,
                num_samples=self.mc_num_samples,
                batch_limit=self.mc_batch_limit,
                seed=next(self.seed_generator),
            )

            cov = sqrt_cov @ sqrt_cov.transpose(-2, -1)
            solver = MVNXPB(cov, bounds)
            solver.solve()

            atol = self.mc_atol_multiplier * (self.mc_num_samples**-0.5)
            for est, prob in zip(estimates, solver.log_prob.exp()):
                if est == 0.0:
                    continue

                self.assertAllClose(est, prob, rtol=0, atol=atol)

    def test_solve_batch(self):
        ndim = 3
        batch_shape = (3, 4)
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            bounds = self.gen_bounds(ndim, batch_shape, bound_range=(-5.0, +5.0))
            sqrt_cov = self.gen_covariances(ndim, batch_shape, as_sqrt=True)

        cov = sqrt_cov @ sqrt_cov.mT

        batched_solver = MVNXPB(cov, bounds)
        batched_solver.solve()

        # solution for each individual batch element is the same as
        # that of the entire batch
        for idx in product(*map(range, batch_shape)):
            solver = MVNXPB(cov[tuple(idx)], bounds[tuple(idx)])
            solver.solve()
            self.assertAlmostEqual(
                batched_solver.log_prob[tuple(idx)].item(),
                solver.log_prob.item(),
            )

    def test_augment(self):
        r"""Test `augment`."""
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))

            # Pick a set of subproblems at random
            index = torch.randint(
                low=0,
                high=len(self.sqrt_covariances),
                size=(),
                device=self.device,
            )
            sqrt_cov = self.sqrt_covariances[index]
            cov = sqrt_cov @ sqrt_cov.transpose(-2, -1)
            bounds = self.bounds[index]

            # Partially solve for `N`-dimensional integral
            N = cov.shape[-1]
            n = torch.randint(low=1, high=N - 2, size=())
            full = MVNXPB(cov, bounds=bounds)
            full.solve(num_steps=n)

            # Compare with solver run using a pre-computed `piv_chol`
            _perm = torch.arange(0, N, device=self.device)
            other = MVNXPB.build(
                step=0,
                perm=_perm.expand(*cov.shape[:-2], N).clone(),
                bounds=cov.diagonal(dim1=-2, dim2=-1).rsqrt().unsqueeze(-1)
                * bounds.clone(),
                piv_chol=full.piv_chol,
                plug_ins=full.plug_ins,
                log_prob=torch.zeros_like(full.log_prob),
            )
            other.solve(num_steps=n)
            self.assertTrue(full.perm.equal(other.perm))
            self.assertTrue(full.bounds.allclose(other.bounds))
            self.assertTrue(full.log_prob.allclose(other.log_prob))

            # Reorder terms according according to `full.perm`
            perm = full.perm.detach().clone()
            _cov = cov.gather(-2, perm.unsqueeze(-1).repeat(1, 1, N))
            _cov = _cov.gather(-1, perm.unsqueeze(-2).repeat(1, N, 1))
            _istd = _cov.diagonal(dim1=-2, dim2=-1).rsqrt()
            _bounds = bounds.gather(-2, perm.unsqueeze(-1).repeat(1, 1, 2))

            # Solve for same `n`-dimensional integral as `full.solve(num_steps=n)`
            init = MVNXPB(_cov[..., :n, :n], _bounds[..., :n, :])
            init.solve()

            # Augment solver with adaptive pivoting disabled
            with patch.object(init.piv_chol, "diag", new=None):
                _corr = _istd[..., n:, None] * _cov[..., n:, :] * _istd[:, None, :]
                temp = init.augment(
                    covariance_matrix=_corr[..., n:],
                    cross_covariance_matrix=_corr[..., :n],
                    bounds=_istd[..., n:, None] * _bounds[..., n:, :],
                    disable_pivoting=True,
                )
            self.assertTrue(temp.piv_chol.diag[..., :n].eq(1).all())
            self.assertEqual(temp.step, n)
            self.assertEqual(temp.piv_chol.step, N)
            self.assertTrue(temp.piv_chol.perm[..., n:].eq(_perm[n:]).all())
            del temp

            # Augment solver again, this time with pivoting enabled
            augm = init.clone().augment(
                covariance_matrix=_cov[..., n:, n:],
                cross_covariance_matrix=_cov[..., n:, :n],
                bounds=_bounds[..., n:, :],
            )

            # Patch `perm` to account for different starting points
            augm_perm = augm.perm
            temp_perm = perm.gather(-1, augm_perm)
            self.assertTrue(augm_perm.equal(augm.piv_chol.perm))
            with patch.object(augm, "perm", new=temp_perm), patch.object(
                augm.piv_chol, "perm", new=temp_perm
            ):
                self.assertEqualMXNBPB(full, augm)

            # Run both solvers
            augm.piv_chol = full.piv_chol.clone()
            augm.piv_chol.perm = augm_perm.clone()
            full.solve()
            augm.solve()

            # Patch `perm` to account for different starting points
            augm_perm = augm.perm
            temp_perm = perm.gather(-1, augm_perm)
            self.assertTrue(augm_perm.equal(augm.piv_chol.perm))
            with patch.object(augm, "perm", new=temp_perm), patch.object(
                augm.piv_chol, "perm", new=temp_perm
            ):
                self.assertEqualMXNBPB(full, augm)

            # testing errors
            fake_init = deepcopy(init)
            fake_init.piv_chol.step = fake_init.perm.shape[-1] + 1
            error_msg = "Augmentation of incomplete solutions not implemented yet."
            with self.assertRaisesRegex(NotImplementedError, error_msg):
                augm = fake_init.augment(
                    covariance_matrix=_cov[..., n:, n:],
                    cross_covariance_matrix=_cov[..., n:, :n],
                    bounds=_bounds[..., n:, :],
                )

            # Testing that solver will try to recover if it encounters
            # a non-psd matrix, even if it ultimately fails in this case
            error_msg = (
                "Matrix not positive definite after repeatedly adding jitter up to.*"
            )
            with self.assertRaisesRegex(NotPSDError, error_msg):
                fake_cov = torch.ones_like(_cov[..., n:, n:])
                augm = init.augment(
                    covariance_matrix=fake_cov,
                    cross_covariance_matrix=_cov[..., n:, :n],
                    bounds=_bounds[..., n:, :],
                )

    def test_getitem(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(1)
            mask = torch.rand(self.toy_solver.log_prob.shape) > 0.5

        other = self.toy_solver[mask]
        for key, b in other.asdict().items():
            a = getattr(self.toy_solver, key)
            if isinstance(b, PivotedCholesky):
                continue
            elif isinstance(b, torch.Tensor):
                self.assertTrue(a[mask].equal(b))
            else:
                self.assertEqual(a, b)

        for key in ("perm", "tril", "diag"):
            a = getattr(self.toy_solver.piv_chol, key)[mask]
            b = getattr(other.piv_chol, key)
            self.assertTrue(a.equal(b))

        fake_solver = deepcopy(self.toy_solver)
        fake_solver.log_prob_extra = torch.tensor([-1])
        fake_solver_1 = fake_solver[:1]
        self.assertEqual(fake_solver_1.log_prob_extra, fake_solver.log_prob_extra[:1])

    def test_concat(self):
        split = len(self.toy_solver.log_prob) // 2
        A = self.toy_solver[:split]
        B = self.toy_solver[split:]
        other = A.concat(B, dim=0)
        self.assertEqualMXNBPB(self.toy_solver, other)

        # Test exception handling
        with patch.object(A, "step", new=A.step + 1), self.assertRaisesRegex(
            ValueError, "`self.step` does not equal `other.step`."
        ):
            A.concat(B, dim=0)

        with self.assertRaisesRegex(ValueError, "not a valid batch dimension"):
            A.concat(B, dim=9)

        with self.assertRaisesRegex(ValueError, "not a valid batch dimension"):
            A.concat(B, dim=-9)

        with patch.object(A, "plug_ins", new=None), self.assertRaisesRegex(
            TypeError, "Concatenation failed: `self.plug_ins` has type"
        ):
            A.concat(B, dim=0)

    def test_clone(self):
        self.toy_solver.bounds.requires_grad_(True)
        try:
            other = self.toy_solver.clone()
            self.assertEqualMXNBPB(self.toy_solver, other)
            for key, a in self.toy_solver.asdict().items():
                if a is None or isinstance(a, int):
                    continue
                b = getattr(other, key)
                self.assertFalse(a is b)

            other.bounds.sum().backward()
            self.assertTrue(self.toy_solver.bounds.grad.eq(1).all())
        finally:
            self.toy_solver.bounds.requires_grad_(False)

    def test_detach(self):
        self.toy_solver.bounds.requires_grad_(True)
        try:
            other = self.toy_solver.detach()
            self.assertEqualMXNBPB(self.toy_solver, other)
            for key, a in self.toy_solver.asdict().items():
                if a is None or isinstance(a, int):
                    continue
                b = getattr(other, key)
                self.assertFalse(a is b)

            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.bounds.sum().backward()
        finally:
            self.toy_solver.bounds.requires_grad_(False)

    def test_expand(self):
        other = self.toy_solver.expand(2, 4, 2)
        self.assertEqualMXNBPB(self.toy_solver, other[0])
        self.assertEqualMXNBPB(self.toy_solver, other[1])

    def test_asdict(self):
        for key, val in self.toy_solver.asdict().items():
            self.assertTrue(val is getattr(self.toy_solver, key))

    def test_build(self):
        other = MVNXPB.build(**self.toy_solver.asdict())
        self.assertEqualMXNBPB(self.toy_solver, other)

    def test_exceptions(self):
        # in solve
        fake_solver = deepcopy(self.toy_solver)
        fake_solver.step = fake_solver.piv_chol.step + 1
        error_msg = "Invalid state: solver ran ahead of matrix decomposition."
        with self.assertRaises(ValueError, msg=error_msg):
            fake_solver.solve()

        # in _pivot
        with self.assertRaises(ValueError):
            pivot = torch.LongTensor([-1])  # this will not be used before the raise
            fake_solver.pivot_(pivot)

        error_msg = f"Expected `other` to be {type(fake_solver)} typed but was.*"
        with self.assertRaisesRegex(TypeError, error_msg):
            fake_solver.concat(1, 1)
