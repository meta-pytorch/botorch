#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from itertools import count

import torch
from botorch.utils.probability.mvnxpb import MVNXPB
from botorch.utils.probability.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)
from botorch.utils.testing import BotorchTestCase
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.special import ndtri


class TestTruncatedMultivariateNormal(BotorchTestCase):
    def setUp(
        self,
        ndims: Sequence[tuple[int, int]] = (2, 4),
        lower_quantile_max: float = 0.9,  # if these get too far into the tail, naive
        upper_quantile_min: float = 0.1,  # MC methods will not produce any samples.
        num_log_probs: int = 4,
        seed: int = 1,
    ) -> None:
        super().setUp()
        self.seed_generator = count(seed)
        self.num_log_probs = num_log_probs

        tkwargs = {"dtype": torch.float64}
        self.distributions = []
        self.sqrt_covariances = []
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            for ndim in ndims:
                loc = torch.randn(ndim, **tkwargs)
                sqrt_covariance = self.gen_covariances(ndim, as_sqrt=True).to(**tkwargs)
                covariance_matrix = sqrt_covariance @ sqrt_covariance.transpose(-1, -2)
                std = covariance_matrix.diag().sqrt()

                lb = lower_quantile_max * torch.rand(ndim, **tkwargs)
                ub = lb.clip(min=upper_quantile_min)  # scratch variable
                ub = ub + (1 - ub) * torch.rand(ndim, **tkwargs)
                bounds = loc.unsqueeze(-1) + std.unsqueeze(-1) * ndtri(
                    torch.stack([lb, ub], dim=-1)
                )

                self.distributions.append(
                    TruncatedMultivariateNormal(
                        loc=loc,
                        covariance_matrix=covariance_matrix,
                        bounds=bounds,
                        validate_args=True,
                    )
                )
                self.sqrt_covariances.append(sqrt_covariance)

    def gen_covariances(
        self,
        ndim: int,
        batch_shape: Sequence[int] = (),
        as_sqrt: bool = False,
    ) -> Tensor:
        shape = tuple(batch_shape) + (ndim, ndim)
        eigvals = -torch.rand(shape[:-1]).log()  # exponential rvs
        orthmat = torch.linalg.svd(torch.randn(shape)).U
        sqrt_covar = orthmat * torch.sqrt(eigvals).unsqueeze(-2)
        return sqrt_covar if as_sqrt else sqrt_covar @ sqrt_covar.transpose(-2, -1)

    def test_init(self):
        trunc = next(iter(self.distributions))
        with self.assertRaisesRegex(SyntaxError, "Missing required argument `bounds`"):
            TruncatedMultivariateNormal(
                loc=trunc.loc, covariance_matrix=trunc.covariance_matrix
            )

        with self.assertRaisesRegex(ValueError, r"Expected bounds.shape\[-1\] to be 2"):
            TruncatedMultivariateNormal(
                loc=trunc.loc,
                covariance_matrix=trunc.covariance_matrix,
                bounds=torch.empty(trunc.covariance_matrix.shape[:-1] + (1,)),
            )

        with self.assertRaisesRegex(ValueError, "`bounds` must be strictly increasing"):
            TruncatedMultivariateNormal(
                loc=trunc.loc,
                covariance_matrix=trunc.covariance_matrix,
                bounds=trunc.bounds.roll(shifts=1, dims=-1),
            )

    def test_solver(self):
        for trunc in self.distributions:
            # Test that solver was setup properly
            solver = trunc.solver
            self.assertIsInstance(solver, MVNXPB)
            self.assertTrue(solver.perm.equal(solver.piv_chol.perm))
            self.assertEqual(solver.step, trunc.covariance_matrix.shape[-1])

            bounds = torch.gather(
                trunc.covariance_matrix.diag().rsqrt().unsqueeze(-1)
                * (trunc.bounds - trunc.loc.unsqueeze(-1)),
                dim=-2,
                index=solver.perm.unsqueeze(-1).expand(*trunc.bounds.shape),
            )
            self.assertTrue(solver.bounds.allclose(bounds))

            # Test that (permuted) covariance matrices match
            A = solver.piv_chol.diag.unsqueeze(-1) * solver.piv_chol.tril
            A = A @ A.transpose(-2, -1)

            n = A.shape[-1]
            B = trunc.covariance_matrix
            B = B.gather(-1, solver.perm.unsqueeze(-2).repeat(n, 1))
            B = B.gather(-2, solver.perm.unsqueeze(-1).repeat(1, n))
            self.assertTrue(A.allclose(B))

    def test_log_prob(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            for trunc in self.distributions:
                # Test generic values
                vals = trunc.rsample(sample_shape=torch.Size([self.num_log_probs]))
                test = MultivariateNormal.log_prob(trunc, vals) - trunc.log_partition
                self.assertTrue(test.equal(trunc.log_prob(vals)))

                # Test out of bounds
                m = trunc.bounds.shape[-2] // 2
                oob = torch.concat(
                    [trunc.bounds[..., :m, 0] - 1, trunc.bounds[..., m:, 1] + 1], dim=-1
                )
                self.assertTrue(trunc.log_prob(oob).eq(-float("inf")).all())

    def test_expand(self):
        trunc = next(iter(self.distributions))
        other = trunc.expand(torch.Size([2]))
        for key in ("loc", "covariance_matrix", "bounds", "log_partition"):
            a = getattr(trunc, key)
            self.assertTrue(all(a.allclose(b) for b in getattr(other, key).unbind()))
