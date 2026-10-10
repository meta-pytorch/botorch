#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from contextlib import ExitStack
from unittest import mock

import torch
from botorch.exceptions import BotorchTensorDimensionError
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.utils.testing import BotorchTestCase, get_test_posterior
from gpytorch import settings as gpt_settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import InterpolatedLinearOperator, to_linear_operator
from torch.distributions.normal import Normal

ROOT_DECOMP_PATH = (
    "linear_operator.operators.dense_linear_operator."
    "DenseLinearOperator._root_decomposition"
)


class TestGPyTorchPosterior(BotorchTestCase):
    def test_GPyTorchPosterior(self) -> None:
        for dtype in (torch.float, torch.double):
            n = 3
            mean = torch.rand(n, dtype=dtype, device=self.device)
            variance = 1 + torch.rand(n, dtype=dtype, device=self.device)
            covar = variance.diag()
            mvn = MultivariateNormal(mean, to_linear_operator(covar))
            posterior = GPyTorchPosterior(distribution=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior._extended_shape(), torch.Size([n, 1]))
            self.assertTrue(torch.equal(posterior.mean, mean.unsqueeze(-1)))
            self.assertTrue(torch.equal(posterior.variance, variance.unsqueeze(-1)))
            # rsample
            samples = posterior.rsample()
            self.assertEqual(samples.shape, torch.Size([1, n, 1]))
            for sample_shape in ([4], [4, 2]):
                samples = posterior.rsample(sample_shape=torch.Size(sample_shape))
                self.assertEqual(samples.shape, torch.Size(sample_shape + [n, 1]))
            # check enabling of approximate root decomposition
            with ExitStack() as es:
                mock_func = es.enter_context(
                    mock.patch(
                        ROOT_DECOMP_PATH, return_value=torch.linalg.cholesky(covar)
                    )
                )
                es.enter_context(gpt_settings.max_cholesky_size(0))
                es.enter_context(
                    gpt_settings.fast_computations(covar_root_decomposition=True)
                )
                # need to clear cache, cannot re-use previous objects
                mvn = MultivariateNormal(mean, to_linear_operator(covar))
                posterior = GPyTorchPosterior(distribution=mvn)
                posterior.rsample(sample_shape=torch.Size([4]))
                mock_func.assert_called_once()

            # rsample w/ base samples
            base_samples = torch.randn(4, 3, device=self.device, dtype=dtype)
            # incompatible shapes
            with self.assertRaisesRegex(RuntimeError, "sample_shape"):
                posterior.rsample_from_base_samples(
                    sample_shape=torch.Size([3]), base_samples=base_samples
                )
            # ensure consistent result
            for sample_shape in ([4], [4, 2]):
                base_samples = torch.randn(
                    *sample_shape, 3, device=self.device, dtype=dtype
                )
                samples = [
                    posterior.rsample_from_base_samples(
                        sample_shape=torch.Size(sample_shape), base_samples=base_samples
                    )
                    for _ in range(2)
                ]
                self.assertAllClose(*samples)
            # Quantile & Density.
            marginal = Normal(
                loc=mean.unsqueeze(-1), scale=variance.unsqueeze(-1).sqrt()
            )
            q_val = torch.rand(2, dtype=dtype, device=self.device)
            quantile = posterior.quantile(q_val)
            self.assertEqual(quantile.shape, posterior._extended_shape(torch.Size([2])))
            expected = torch.stack([marginal.icdf(q) for q in q_val], dim=0)
            self.assertAllClose(quantile, expected)
            density = posterior.density(q_val)
            self.assertEqual(density.shape, posterior._extended_shape(torch.Size([2])))
            expected = torch.stack([marginal.log_prob(q).exp() for q in q_val], dim=0)
            self.assertAllClose(density, expected)

    def test_GPyTorchPosterior_Multitask(self):
        for dtype in (torch.float, torch.double):
            mean = torch.rand(3, 2, dtype=dtype, device=self.device)
            variance = 1 + torch.rand(3, 2, dtype=dtype, device=self.device)
            covar = variance.view(-1).diag()
            mvn = MultitaskMultivariateNormal(mean, to_linear_operator(covar))
            posterior = GPyTorchPosterior(distribution=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior._extended_shape(), torch.Size([3, 2]))
            self.assertTrue(torch.equal(posterior.mean, mean))
            self.assertTrue(torch.equal(posterior.variance, variance))
            # rsample
            samples = posterior.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, 2]))
            # rsample w/ base samples
            base_samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
            samples_b1 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertAllClose(samples_b1, samples_b2)
            base_samples2 = torch.randn(4, 2, 3, 2, device=self.device, dtype=dtype)
            samples2_b1 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertAllClose(samples2_b1, samples2_b2)

    def test_degenerate_GPyTorchPosterior(self):
        for dtype, multi_task in (
            (torch.float, False),
            (torch.double, False),
            (torch.double, True),
        ):
            # singular covariance matrix
            degenerate_covar = torch.tensor(
                [[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=dtype, device=self.device
            )
            mean = torch.rand(3, dtype=dtype, device=self.device)
            mvn = MultivariateNormal(mean, to_linear_operator(degenerate_covar))
            mean_exp = mean.unsqueeze(-1)
            variance_exp = degenerate_covar.diag().unsqueeze(-1)
            if multi_task:
                expected_dim = 2
                mvn = MultitaskMultivariateNormal.from_independent_mvns([mvn, mvn])
                mean_exp = mean_exp.repeat(1, 2)
                variance_exp = variance_exp.repeat(1, 2)
                base_samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
                base_samples2 = torch.randn(4, 2, 3, 2, device=self.device, dtype=dtype)
            else:
                expected_dim = 1
                base_samples = torch.randn(4, 3, device=self.device, dtype=dtype)
                base_samples2 = torch.randn(4, 2, 3, device=self.device, dtype=dtype)
            posterior = GPyTorchPosterior(distribution=mvn)
            # basics
            self.assertEqual(posterior.device.type, self.device.type)
            self.assertTrue(posterior.dtype == dtype)
            self.assertEqual(posterior._extended_shape(), torch.Size([3, expected_dim]))
            self.assertTrue(torch.equal(posterior.mean, mean_exp))
            self.assertTrue(torch.equal(posterior.variance, variance_exp))

            # rsample
            with warnings.catch_warnings(record=True) as ws:
                # we check that the p.d. warning is emitted - this only
                # happens once per posterior, so we need to check only once
                samples = posterior.rsample(sample_shape=torch.Size([4]))
                self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in ws))
                self.assertTrue(any("not p.d" in str(w.message) for w in ws))
            self.assertEqual(samples.shape, torch.Size([4, 3, expected_dim]))
            samples2 = posterior.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2, 3, expected_dim]))
            # rsample w/ base samples
            samples_b1 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            samples_b2 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4]), base_samples=base_samples
            )
            self.assertAllClose(samples_b1, samples_b2)
            samples2_b1 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            samples2_b2 = posterior.rsample_from_base_samples(
                sample_shape=torch.Size([4, 2]), base_samples=base_samples2
            )
            self.assertAllClose(samples2_b1, samples2_b2)

    def test_scalarize_posterior(self):
        for batch_shape, m, lazy, dtype in itertools.product(
            ([], [3]), (1, 2), (False, True), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            offset = torch.rand(1).item()
            weights = torch.randn(m, **tkwargs)
            # Make sure the weights are not too small.
            while torch.any(weights.abs() < 0.1):
                weights = torch.randn(m, **tkwargs)
            # test q=1
            posterior = get_test_posterior(batch_shape, m=m, lazy=lazy, **tkwargs)
            mean, covar = (
                posterior.distribution.mean,
                posterior.distribution.covariance_matrix,
            )
            new_posterior = scalarize_posterior(posterior, weights, offset)
            exp_size = torch.Size(batch_shape + [1, 1])
            self.assertEqual(new_posterior.mean.shape, exp_size)
            new_mean_exp = offset + (mean @ weights).unsqueeze(-1)
            self.assertAllClose(new_posterior.mean, new_mean_exp)
            self.assertEqual(new_posterior.variance.shape, exp_size)
            new_covar_exp = ((covar @ weights) @ weights).unsqueeze(-1)
            self.assertTrue(
                torch.allclose(new_posterior.variance[..., -1], new_covar_exp)
            )
            # test q=2, interleaved
            q = 2
            posterior = get_test_posterior(
                batch_shape, q=q, m=m, lazy=lazy, interleaved=True, **tkwargs
            )
            mean, covar = (
                posterior.distribution.mean,
                posterior.distribution.covariance_matrix,
            )
            new_posterior = scalarize_posterior(posterior, weights, offset)
            exp_size = torch.Size(batch_shape + [q, 1])
            self.assertEqual(new_posterior.mean.shape, exp_size)
            new_mean_exp = offset + (mean @ weights).unsqueeze(-1)
            self.assertAllClose(new_posterior.mean, new_mean_exp)
            self.assertEqual(new_posterior.variance.shape, exp_size)
            new_covar = new_posterior.distribution.covariance_matrix
            if m == 1:
                self.assertAllClose(new_covar, weights**2 * covar)
            else:
                w = weights.unsqueeze(0)
                covar00_exp = (w * covar[..., :m, :m] * w.t()).sum(-1).sum(-1)
                self.assertAllClose(new_covar[..., 0, 0], covar00_exp)
                covarnn_exp = (w * covar[..., -m:, -m:] * w.t()).sum(-1).sum(-1)
                self.assertAllClose(new_covar[..., -1, -1], covarnn_exp)
            # test q=2, non-interleaved
            # test independent special case as well
            for independent in (False, True) if m > 1 else (False,):
                posterior = get_test_posterior(
                    batch_shape,
                    q=q,
                    m=m,
                    lazy=lazy,
                    interleaved=False,
                    independent=independent,
                    **tkwargs,
                )
                mean, covar = (
                    posterior.distribution.mean,
                    posterior.distribution.covariance_matrix,
                )
                new_posterior = scalarize_posterior(posterior, weights, offset)
                exp_size = torch.Size(batch_shape + [q, 1])
                self.assertEqual(new_posterior.mean.shape, exp_size)
                new_mean_exp = offset + (mean @ weights).unsqueeze(-1)
                self.assertAllClose(new_posterior.mean, new_mean_exp)
                self.assertEqual(new_posterior.variance.shape, exp_size)
                new_covar = new_posterior.distribution.covariance_matrix
                if m == 1:
                    self.assertAllClose(new_covar, weights**2 * covar)
                else:
                    # construct the indices manually
                    cs = list(itertools.combinations_with_replacement(range(m), 2))
                    idx_nlzd = torch.tensor(
                        list(set(cs + [tuple(i[::-1]) for i in cs])),
                        dtype=torch.long,
                        device=self.device,
                    )
                    w = weights[idx_nlzd[:, 0]] * weights[idx_nlzd[:, 1]]
                    idx = q * idx_nlzd
                    covar00_exp = (covar[..., idx[:, 0], idx[:, 1]] * w).sum(-1)
                    self.assertAllClose(new_covar[..., 0, 0], covar00_exp)
                    idx_ = q - 1 + idx
                    covarnn_exp = (covar[..., idx_[:, 0], idx_[:, 1]] * w).sum(-1)
                    self.assertAllClose(new_covar[..., -1, -1], covarnn_exp)

            # test errors
            with self.assertRaises(RuntimeError):
                scalarize_posterior(posterior, weights[:-1], offset)
            with self.assertRaises(BotorchTensorDimensionError):
                scalarize_posterior(posterior, weights.unsqueeze(0), offset)

    def test_scalarize_posterior_independent_interpolated(self) -> None:
        # The covariance blocks of independent outputs were previously indexed one by
        # one, which errored out for `InterpolatedLinearOperator` blocks (as produced
        # e.g. for the multi-task sub-models of a `ModelListGP`).
        for batch_shape, dtype in itertools.product(
            ([], [3]), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            q, m = 3, 2
            mvns = []
            for _ in range(m):
                a = torch.rand(*batch_shape, q, q, **tkwargs)
                covar = a @ a.transpose(-1, -2) + torch.eye(q, **tkwargs)
                mean = torch.rand(*batch_shape, q, **tkwargs)
                lazy_covar = InterpolatedLinearOperator(to_linear_operator(covar))
                mvns.append(MultivariateNormal(mean, lazy_covar))
            mtmvn = MultitaskMultivariateNormal.from_independent_mvns(mvns)
            posterior = GPyTorchPosterior(mtmvn)
            weights = torch.tensor([0.5, -2.0], **tkwargs)
            offset = 1.0
            new_posterior = scalarize_posterior(posterior, weights, offset)
            expected_mean = offset + sum(w * mvn.mean for w, mvn in zip(weights, mvns))
            expected_covar = sum(
                w**2 * mvn.covariance_matrix for w, mvn in zip(weights, mvns)
            )
            self.assertAllClose(new_posterior.mean.squeeze(-1), expected_mean)
            self.assertAllClose(new_posterior.covariance_matrix, expected_covar)
