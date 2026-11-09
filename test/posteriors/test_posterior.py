#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from botorch.posteriors import GPyTorchPosterior, Posterior, PosteriorList
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import to_linear_operator


class NotSoAbstractPosterior(Posterior):
    @property
    def device(self):
        pass

    @property
    def dtype(self):
        pass

    def rsample(self, *args):
        pass


class TestPosterior(BotorchTestCase):
    def test_abstract_base_posterior(self):
        with self.assertRaises(TypeError):
            Posterior()

    def test_notimplemented_errors(self):
        posterior = NotSoAbstractPosterior()
        with self.assertRaisesRegex(AttributeError, "NotSoAbstractPosterior"):
            posterior.mean
        with self.assertRaisesRegex(AttributeError, "NotSoAbstractPosterior"):
            posterior.variance
        with self.assertRaisesRegex(
            NotImplementedError, "not implement `_extended_shape`"
        ):
            posterior._extended_shape()
        with self.assertRaisesRegex(NotImplementedError, "not implement `batch_range`"):
            posterior.batch_range


class TestPosteriorList(BotorchTestCase):
    def _make_gpytorch_posterior(self, shape, dtype):
        mean = torch.rand(*shape, dtype=dtype, device=self.device)
        variance = 1 + torch.rand(*shape, dtype=dtype, device=self.device)
        covar = torch.diag_embed(variance)
        mvn = MultivariateNormal(mean, to_linear_operator(covar))
        return GPyTorchPosterior(distribution=mvn)

    def _make_deterministic_posterior(self, shape, dtype):
        mean = torch.rand(*shape, 1, dtype=dtype, device=self.device)
        return EnsemblePosterior(values=mean.unsqueeze(0))

    def test_posterior_list(self):
        for dtype, use_deterministic in product(
            (torch.float, torch.double), (False, True)
        ):
            shape = torch.Size([3])
            make_posterior = (
                self._make_deterministic_posterior
                if use_deterministic
                else self._make_gpytorch_posterior
            )
            p_1 = make_posterior(shape, dtype)
            p_2 = make_posterior(shape, dtype)
            p = PosteriorList(p_1, p_2)
            with self.assertRaisesRegex(NotImplementedError, "base_sample_shape"):
                p.base_sample_shape
            self.assertEqual(p._extended_shape(), shape + torch.Size([2]))
            self.assertEqual(p.device.type, self.device.type)
            self.assertEqual(p.dtype, dtype)
            self.assertTrue(
                torch.equal(p.mean, torch.cat([p_1.mean, p_2.mean], dim=-1))
            )
            self.assertTrue(
                torch.equal(p.variance, torch.cat([p_1.variance, p_2.variance], dim=-1))
            )
            # Test sampling.
            sample_shape = torch.Size([4])
            samples = p.rsample(sample_shape=sample_shape)
            self.assertEqual(samples.shape, torch.Size([4, 3, 2]))

    def test_posterior_list_errors(self):
        shape_1 = torch.Size([3, 2])
        shape_2 = torch.Size([4, 1])
        p_1 = self._make_gpytorch_posterior(shape_1, torch.float)
        p_2 = self._make_gpytorch_posterior(shape_2, torch.double)
        p = PosteriorList(p_1, p_2)

        with self.assertRaisesRegex(NotImplementedError, "same `batch_shape`"):
            p._extended_shape()
        dtype_err_msg = "Multi-dtype posteriors are currently not supported."
        with self.assertRaisesRegex(NotImplementedError, dtype_err_msg) as e:
            p.dtype
        # the message lists the actual dtypes, not a literal "{dtypes}"
        self.assertIn("torch.float32", str(e.exception))
        self.assertIn("torch.float64", str(e.exception))
        device_err_msg = "Multi-device posteriors are currently not supported."
        p_2._device = None
        with self.assertRaisesRegex(NotImplementedError, device_err_msg):
            p.device
        with self.assertRaisesRegex(AttributeError, "`PosteriorList` does not define"):
            p.rate
