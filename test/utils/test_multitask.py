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
    def _test_separate_mtmvn(self, interleaved=False):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mean = torch.rand(2, 2, **tkwargs)
            a = torch.rand(4, 4, **tkwargs)
            covar = a @ a.transpose(-1, -2) + torch.eye(4, **tkwargs)
            mvn = MultitaskMultivariateNormal(
                mean=mean, covariance_matrix=covar, interleaved=interleaved
            )
            mtmvn_list = separate_mtmvn(mvn)

            mean_1 = mean[..., 0]
            mean_2 = mean[..., 1]
            if interleaved:
                covar_1 = covar[::2, ::2]
                covar_2 = covar[1::2, 1::2]
            else:
                covar_1 = covar[:2, :2]
                covar_2 = covar[2:, 2:]

            self.assertEqual(len(mtmvn_list), 2)
            for mvn_i, mean_i, covar_i in zip(
                mtmvn_list, (mean_1, mean_2), (covar_1, covar_2)
            ):
                self.assertIsInstance(mvn_i, MultivariateNormal)
                self.assertTrue(torch.equal(mvn_i.mean, mean_i))
                self.assertAllClose(mvn_i.covariance_matrix, covar_i)

    def test_separate_mtmvn_interleaved(self) -> None:
        self._test_separate_mtmvn(interleaved=True)

    def test_separate_mtmvn_not_interleaved(self) -> None:
        self._test_separate_mtmvn(interleaved=False)
