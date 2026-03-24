#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.utils.multitask import separate_mtmvn
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal


class TestSeparateMTMVN(BotorchTestCase):
    def test_separate_mtmvn(self) -> None:
        for interleaved in (True, False):
            for batch_shape in (torch.Size([]), torch.Size([3])):
                with self.subTest(interleaved=interleaved, batch_shape=batch_shape):
                    self._test_separate_mtmvn(
                        interleaved=interleaved, batch_shape=batch_shape
                    )

    def _test_separate_mtmvn(self, interleaved: bool, batch_shape: torch.Size) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            num_data = 10
            num_tasks = 4
            n = num_data * num_tasks
            mean = torch.rand(*batch_shape, num_data, num_tasks, **tkwargs)
            a = torch.rand(*batch_shape, n, n, **tkwargs)
            covar = a @ a.transpose(-1, -2) + torch.eye(n, **tkwargs)
            mvn = MultitaskMultivariateNormal(
                mean=mean, covariance_matrix=covar, interleaved=interleaved
            )
            mtmvn_list = separate_mtmvn(mvn)

            self.assertEqual(len(mtmvn_list), num_tasks)
            dense_covar = covar.to_dense() if hasattr(covar, "to_dense") else covar

            for c, mvn_c in enumerate(mtmvn_list):
                self.assertIsInstance(mvn_c, MultivariateNormal)
                self.assertEqual(mvn_c.mean.shape, (*batch_shape, num_data))
                self.assertTrue(torch.equal(mvn_c.mean, mean[..., c]))
                if interleaved:
                    idx = torch.arange(c, n, num_tasks)
                else:
                    idx = torch.arange(c * num_data, (c + 1) * num_data)
                expected_covar = dense_covar[..., idx, :][..., :, idx]
                self.assertAllClose(mvn_c.covariance_matrix, expected_covar, atol=1e-5)
