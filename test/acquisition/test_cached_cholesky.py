#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import mock

import torch
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.higher_order_gp import HigherOrderGP
from botorch.models.model import Model, ModelList
from botorch.models.transforms.outcome import Log
from botorch.sampling.normal import IIDNormalSampler, MCSampler
from botorch.utils.low_rank import extract_batch_covar
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from linear_operator.utils.errors import NanError, NotPSDError

CHOLESKY_PATH = "linear_operator.operators._linear_operator.psd_safe_cholesky"
EXTRACT_BATCH_COVAR_PATH = "botorch.acquisition.cached_cholesky.extract_batch_covar"


class DummyCachedCholeskyAcqf(MCAcquisitionFunction, CachedCholeskyMCSamplerMixin):
    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective | None = None,
        sampler: MCSampler | None = None,
        cache_root: bool | None = None,
    ):
        """A dummy cached cholesky acquisition function."""
        MCAcquisitionFunction.__init__(self, model=model, objective=objective)
        CachedCholeskyMCSamplerMixin.__init__(
            self, model=model, sampler=sampler, cache_root=cache_root
        )

    def forward(self, X):
        return X


class TestCachedCholeskyMCSamplerMixin(BotorchTestCase):
    def test_init(self):
        mean = torch.zeros(1, 1)
        variance = torch.ones(1, 1)
        mm = MockModel(MockPosterior(mean=mean, variance=variance))
        # basic test w/ invalid model.
        sampler = IIDNormalSampler(sample_shape=torch.Size([1]))

        # Test None default with unsupported model - should disable caching
        acqf = DummyCachedCholeskyAcqf(model=mm, sampler=sampler)
        self.assertFalse(acqf._cache_root)
        # Test explicit True with unsupported model - should warn and disable
        with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
            acqf = DummyCachedCholeskyAcqf(model=mm, sampler=sampler, cache_root=True)
        self.assertFalse(acqf._cache_root)  # gets turned to False
        # Unsupported outcome transform.
        stgp = SingleTaskGP(
            torch.zeros(1, 1), torch.zeros(1, 1), outcome_transform=Log()
        )
        with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
            acqf = DummyCachedCholeskyAcqf(model=stgp, cache_root=True)
        self.assertFalse(acqf._cache_root)
        # ModelList is not supported.
        model_list = ModelList(SingleTaskGP(torch.zeros(1, 1), torch.zeros(1, 1)))
        with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
            acqf = DummyCachedCholeskyAcqf(model=model_list, cache_root=True)
        self.assertFalse(acqf._cache_root)

        # basic test w/ supported model.
        stgp = SingleTaskGP(torch.zeros(1, 1), torch.zeros(1, 1))
        # Test None default with supported model - should auto-enable caching
        acqf = DummyCachedCholeskyAcqf(model=stgp, sampler=sampler)
        self.assertTrue(acqf._cache_root)
        self.assertEqual(acqf.sampler, sampler)
        # Test explicit True with supported model
        acqf = DummyCachedCholeskyAcqf(model=stgp, sampler=sampler, cache_root=True)
        self.assertTrue(acqf._cache_root)
        self.assertEqual(acqf.sampler, sampler)

        # test the base_samples are set to None
        self.assertIsNone(acqf.sampler.base_samples)
        # test model that uses matheron's rule and sampler.batch_range != (0, -1)
        hogp = HigherOrderGP(torch.zeros(1, 1), torch.zeros(1, 1, 1)).eval()
        with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
            acqf = DummyCachedCholeskyAcqf(model=hogp, sampler=sampler, cache_root=True)
        self.assertFalse(acqf._cache_root)

        # test deterministic model
        model = GenericDeterministicModel(f=lambda X: X)
        with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
            acqf = DummyCachedCholeskyAcqf(
                model=model, sampler=sampler, cache_root=True
            )
        self.assertFalse(acqf._cache_root)

    def test_cache_root_decomposition(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs["dtype"] = dtype
                # test mt-mvn
                train_x = torch.rand(2, 1, **tkwargs)
                train_y = torch.rand(2, 2, **tkwargs)
                test_x = torch.rand(2, 1, **tkwargs)
                model = SingleTaskGP(train_x, train_y)
                sampler = IIDNormalSampler(sample_shape=torch.Size([1]))
                with torch.no_grad():
                    posterior = model.posterior(test_x)
                acqf = DummyCachedCholeskyAcqf(
                    model=model,
                    sampler=sampler,
                    objective=GenericMCObjective(lambda Y, _: Y[..., 0]),
                )
                baseline_L = torch.eye(2, **tkwargs)
                with mock.patch(
                    EXTRACT_BATCH_COVAR_PATH, wraps=extract_batch_covar
                ) as mock_extract_batch_covar:
                    with mock.patch(
                        CHOLESKY_PATH, return_value=baseline_L
                    ) as mock_cholesky:
                        baseline_L_acqf = acqf._compute_root_decomposition(
                            posterior=posterior
                        )
                        mock_extract_batch_covar.assert_called_once_with(
                            posterior.distribution
                        )
                        mock_cholesky.assert_called_once()
                # test mvn
                model = SingleTaskGP(train_x, train_y[:, :1])
                with torch.no_grad():
                    posterior = model.posterior(test_x)
                with mock.patch(EXTRACT_BATCH_COVAR_PATH) as mock_extract_batch_covar:
                    with mock.patch(
                        CHOLESKY_PATH, return_value=baseline_L
                    ) as mock_cholesky:
                        baseline_L_acqf = acqf._compute_root_decomposition(
                            posterior=posterior
                        )
                        mock_extract_batch_covar.assert_not_called()
                        mock_cholesky.assert_called_once()
                self.assertTrue(torch.equal(baseline_L_acqf, baseline_L))

    def test_get_f_X_samples(self):
        sample_cached_cholesky_path = (
            "botorch.acquisition.cached_cholesky.sample_cached_cholesky"
        )
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs["dtype"] = dtype
                mean = torch.zeros(5, 1, **tkwargs)
                variance = torch.ones(5, 1, **tkwargs)
                mm = MockModel(
                    MockPosterior(
                        mean=mean,
                        variance=variance,
                        samples=torch.rand(5, 1, **tkwargs),
                    )
                )
                # basic test
                sampler = IIDNormalSampler(sample_shape=torch.Size([1]))

                with self.assertWarnsRegex(RuntimeWarning, "cache_root"):
                    acqf = DummyCachedCholeskyAcqf(
                        model=mm, sampler=sampler, cache_root=True
                    )
                self.assertFalse(acqf._cache_root)
                acqf._cache_root = True
                q = 3
                baseline_L = torch.eye(5 - q, **tkwargs)
                acqf._baseline_L = baseline_L
                posterior = mm.posterior(torch.rand(5, 1, **tkwargs))
                # basic test
                rv = torch.rand(1, 5, 1, **tkwargs)
                with mock.patch(
                    sample_cached_cholesky_path, return_value=rv
                ) as mock_sample_cached_cholesky:
                    samples = acqf._get_f_X_samples(posterior=posterior, q_in=q)
                    mock_sample_cached_cholesky.assert_called_once_with(
                        posterior=posterior,
                        baseline_L=acqf._baseline_L,
                        q=q,
                        base_samples=acqf.sampler.base_samples,
                        sample_shape=acqf.sampler.sample_shape,
                    )
                self.assertTrue(torch.equal(rv, samples))

                # test fall back when sampling from cached cholesky fails
                for error_cls in (NanError, NotPSDError):
                    base_samples = torch.rand(1, 5, 1, **tkwargs)
                    acqf.sampler.base_samples = base_samples
                    acqf._baseline_L = baseline_L
                    with (
                        mock.patch(
                            sample_cached_cholesky_path,
                            side_effect=error_cls,
                        ) as mock_sample_cached_cholesky,
                        warnings.catch_warnings(record=True) as ws,
                    ):
                        samples = acqf._get_f_X_samples(posterior=posterior, q_in=q)
                    mock_sample_cached_cholesky.assert_called_once_with(
                        posterior=posterior,
                        baseline_L=acqf._baseline_L,
                        q=q,
                        base_samples=base_samples,
                        sample_shape=acqf.sampler.sample_shape,
                    )
                    self.assertTrue(issubclass(ws[0].category, BotorchWarning))
                    self.assertTrue(samples.shape, torch.Size([1, q, 1]))
                # test HOGP
                hogp = HigherOrderGP(torch.zeros(2, 1), torch.zeros(2, 1, 1)).eval()
                acqf = DummyCachedCholeskyAcqf(
                    model=hogp, sampler=sampler, cache_root=True
                )
                mock_samples = torch.rand(5, 1, 1, **tkwargs)
                posterior = MockPosterior(
                    mean=mean, variance=variance, samples=mock_samples
                )
                samples = acqf._get_f_X_samples(posterior=posterior, q_in=q)
                self.assertTrue(torch.equal(samples, mock_samples[2:].unsqueeze(0)))
