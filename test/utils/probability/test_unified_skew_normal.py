#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from copy import deepcopy

from itertools import count

from typing import Any

import torch
from botorch.utils.probability.mvnxpb import MVNXPB
from botorch.utils.probability.truncated_multivariate_normal import (
    TruncatedMultivariateNormal,
)
from botorch.utils.probability.unified_skew_normal import UnifiedSkewNormal
from botorch.utils.testing import BotorchTestCase
from linear_operator.operators import DenseLinearOperator
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.special import ndtri


class TestUnifiedSkewNormal(BotorchTestCase):
    def setUp(
        self,
        ndims: Sequence[tuple[int, int]] = ((1, 1), (2, 3), (3, 2), (3, 3)),
        lower_quantile_max: float = 0.9,  # if these get too far into the tail, naive
        upper_quantile_min: float = 0.1,  # MC methods will not produce any samples.
        num_log_probs: int = 4,
        mc_num_samples: int = 100000,
        mc_num_rsamples: int = 1000,
        mc_atol_multiplier: float = 4.0,
        seed: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ):
        super().setUp()
        self.dtype = dtype
        self.seed_generator = count(seed)
        self.num_log_probs = num_log_probs
        self.mc_num_samples = mc_num_samples
        self.mc_num_rsamples = mc_num_rsamples
        self.mc_atol_multiplier = mc_atol_multiplier

        self.distributions = []
        self.sqrt_covariances = []
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            for ndim_x, ndim_y in ndims:
                ndim_xy = ndim_x + ndim_y
                sqrt_covariance = self.gen_covariances(ndim_xy, as_sqrt=True)
                covariance = sqrt_covariance @ sqrt_covariance.transpose(-1, -2)

                loc_x = torch.randn(ndim_x, **self.tkwargs)
                cov_x = covariance[:ndim_x, :ndim_x]
                std_x = cov_x.diag().sqrt()
                lb = lower_quantile_max * torch.rand(ndim_x, **self.tkwargs)
                ub = lb.clip(min=upper_quantile_min)  # scratch variable
                ub = ub + (1 - ub) * torch.rand(ndim_x, **self.tkwargs)
                bounds_x = loc_x.unsqueeze(-1) + std_x.unsqueeze(-1) * ndtri(
                    torch.stack([lb, ub], dim=-1)
                )

                xcov = covariance[:ndim_x, ndim_x:]
                trunc = TruncatedMultivariateNormal(
                    loc=loc_x,
                    covariance_matrix=cov_x,
                    bounds=bounds_x,
                    validate_args=True,
                )

                gauss = MultivariateNormal(
                    loc=torch.randn(ndim_y, **self.tkwargs),
                    covariance_matrix=covariance[ndim_x:, ndim_x:],
                )

                self.sqrt_covariances.append(sqrt_covariance)
                self.distributions.append(
                    UnifiedSkewNormal(
                        trunc=trunc, gauss=gauss, cross_covariance_matrix=xcov
                    )
                )

    @property
    def tkwargs(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "device": self.device}

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

    def test_log_prob(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))
            for usn in self.distributions:
                shape = torch.Size([self.num_log_probs])
                vals = usn.gauss.rsample(sample_shape=shape)

                # Manually compute log probabilities
                alpha = torch.cholesky_solve(
                    usn.cross_covariance_matrix.T, usn.gauss.scale_tril
                )
                loc_condx = usn.trunc.loc + (vals - usn.gauss.loc) @ alpha
                cov_condx = (
                    usn.trunc.covariance_matrix - usn.cross_covariance_matrix @ alpha
                )
                solver = MVNXPB(
                    covariance_matrix=cov_condx.repeat(self.num_log_probs, 1, 1),
                    bounds=usn.trunc.bounds - loc_condx.unsqueeze(-1),
                )
                log_probs = (
                    solver.solve() + usn.gauss.log_prob(vals) - usn.trunc.log_partition
                )

                # Compare with log probabilities returned by class
                self.assertTrue(log_probs.allclose(usn.log_prob(vals)))

                # checking error handling when incorrectly shaped value is passed
                wrong_vals = torch.cat((vals, vals), dim=-1)
                error_msg = ".*with shape.*does not comply with the instance.*"
                with self.assertRaisesRegex(ValueError, error_msg):
                    usn.log_prob(wrong_vals)

    def test_rsample(self):
        # TODO: Replace with e.g. two-sample test.
        with torch.random.fork_rng():
            torch.random.manual_seed(next(self.seed_generator))

            # Pick a USN distribution at random
            index = torch.randint(low=0, high=len(self.distributions), size=())
            usn = self.distributions[index]
            sqrt_covariance = self.sqrt_covariances[index]

            # Generate draws using `rsample`
            samples_y = usn.rsample(sample_shape=torch.Size([self.mc_num_rsamples]))
            means = samples_y.mean(0)
            covar = samples_y.T.cov()

            # Generate draws using rejection sampling
            ndim = sqrt_covariance.shape[-1]
            base_rvs = torch.randn(self.mc_num_samples, ndim, **self.tkwargs)
            _samples_x, _samples_y = (base_rvs @ sqrt_covariance.T).split(
                usn.trunc.event_shape + usn.gauss.event_shape, dim=-1
            )

            _accept = torch.logical_and(
                (_samples_x > usn.trunc.bounds[..., 0] - usn.trunc.loc).all(-1),
                (_samples_x < usn.trunc.bounds[..., 1] - usn.trunc.loc).all(-1),
            )

            _means = usn.gauss.loc + _samples_y[_accept].mean(0)
            _covar = _samples_y[_accept].T.cov()

            atol = self.mc_atol_multiplier * (
                _accept.count_nonzero() ** -0.5 + self.mc_num_rsamples**-0.5
            )

            self.assertAllClose(_means, means, rtol=0, atol=atol)
            self.assertAllClose(_covar, covar, rtol=0, atol=atol)

    def test_expand(self):
        usn = next(iter(self.distributions))
        # calling these lazy properties to cached them and
        #  hit associated branches in expand
        usn._orthogonalized_gauss
        usn.covariance_matrix

        other = usn.expand(torch.Size([2]))
        for key in ("loc", "covariance_matrix"):
            a = getattr(usn.gauss, key)
            self.assertTrue(all(a.equal(b) for b in getattr(other.gauss, key).unbind()))

        for key in ("loc", "covariance_matrix", "bounds", "log_partition"):
            a = getattr(usn.trunc, key)
            self.assertTrue(all(a.equal(b) for b in getattr(other.trunc, key).unbind()))

        for b in other.cross_covariance_matrix.unbind():
            self.assertTrue(usn.cross_covariance_matrix.equal(b))

        fake_usn = deepcopy(usn)
        fake_usn.covariance_matrix = -1
        error_msg = (
            f"Type {type(-1)} of UnifiedSkewNormal's lazy property "
            "covariance_matrix not supported.*"
        )
        with self.assertRaisesRegex(TypeError, error_msg):
            other = fake_usn.expand(torch.Size([2]))

    def test_validate_args(self):
        for d in self.distributions:
            error_msg = ".*is only well-defined for positive definite.*"
            with self.assertRaisesRegex(ValueError, error_msg):
                gauss = deepcopy(d.gauss)
                gauss.covariance_matrix *= -1
                UnifiedSkewNormal(d.trunc, gauss, d.cross_covariance_matrix)

            error_msg = ".*-dimensional `trunc` incompatible with.*-dimensional `gauss"
            with self.assertRaisesRegex(ValueError, error_msg):
                gauss = deepcopy(d.gauss)
                gauss._event_shape = (*gauss._event_shape, 1)
                UnifiedSkewNormal(d.trunc, gauss, d.cross_covariance_matrix)

            error_msg = "Incompatible batch shapes"
            with self.assertRaisesRegex(ValueError, error_msg):
                gauss = deepcopy(d.gauss)
                trunc = deepcopy(d.trunc)
                gauss._batch_shape = (*gauss._batch_shape, 2)
                trunc._batch_shape = (*trunc._batch_shape, 3)
                UnifiedSkewNormal(trunc, gauss, d.cross_covariance_matrix)

    def test_properties(self):
        orth = "_orthogonalized_gauss"
        scal = "scale_tril"
        for d in self.distributions:
            # testing calling orthogonalized_gauss and scale_tril
            usn = UnifiedSkewNormal(
                d.trunc, d.gauss, d.cross_covariance_matrix, validate_args=False
            )
            self.assertTrue(orth not in usn.__dict__)
            self.assertTrue(scal not in usn.__dict__)
            usn._orthogonalized_gauss
            self.assertTrue(orth in usn.__dict__)
            self.assertTrue(scal not in usn.__dict__)
            usn.scale_tril
            self.assertTrue(orth in usn.__dict__)
            self.assertTrue(scal in usn.__dict__)

            # testing calling orthogonalized_gauss and scale_tril in reverse order
            usn = UnifiedSkewNormal(
                d.trunc, d.gauss, d.cross_covariance_matrix, validate_args=False
            )
            usn.scale_tril
            self.assertTrue(orth not in usn.__dict__)
            self.assertTrue(scal in usn.__dict__)
            usn._orthogonalized_gauss
            self.assertTrue(orth in usn.__dict__)
            self.assertTrue(scal in usn.__dict__)

    def test_covariance_matrix(self):
        for d in self.distributions:
            cov = d.covariance_matrix
            self.assertTrue(isinstance(cov, Tensor))

            # testing for symmetry
            self.assertAllClose(cov, cov.mT)

            # testing for positive-definiteness
            ispd = False
            try:
                torch.linalg.cholesky(cov)
                ispd = True
            except RuntimeError:
                pass
            self.assertTrue(ispd)

            # checking that linear operator to tensor conversion
            # leads to same covariance matrix
            xcov_linop = DenseLinearOperator(d.cross_covariance_matrix)
            usn_linop = UnifiedSkewNormal(
                trunc=d.trunc, gauss=d.gauss, cross_covariance_matrix=xcov_linop
            )
            cov_linop = usn_linop.covariance_matrix
            self.assertTrue(isinstance(cov_linop, Tensor))
            self.assertAllClose(cov, cov_linop)

    def test_repr(self):
        for d in self.distributions:
            r = repr(d)
            self.assertTrue(f"trunc: {d.trunc}" in r)
            self.assertTrue(f"gauss: {d.gauss}" in r)
            self.assertTrue(
                f"cross_covariance_matrix: {d.cross_covariance_matrix.shape}" in r
            )
