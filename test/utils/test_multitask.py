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

    def _test_separate_mtmvn_larger(self, interleaved: bool) -> None:
        """Test with larger data and more tasks to verify numerical correctness."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            num_data = 10
            num_tasks = 4
            n = num_data * num_tasks
            mean = torch.rand(num_data, num_tasks, **tkwargs)
            a = torch.rand(n, n, **tkwargs)
            covar = a @ a.transpose(-1, -2) + torch.eye(n, **tkwargs)
            mvn = MultitaskMultivariateNormal(
                mean=mean, covariance_matrix=covar, interleaved=interleaved
            )
            mtmvn_list = separate_mtmvn(mvn)

            self.assertEqual(len(mtmvn_list), num_tasks)
            dense_covar = covar.to_dense() if hasattr(covar, "to_dense") else covar

            for c, mvn_c in enumerate(mtmvn_list):
                self.assertIsInstance(mvn_c, MultivariateNormal)
                # Check mean
                self.assertTrue(torch.equal(mvn_c.mean, mean[..., c]))
                # Check covariance against direct indexing of the dense matrix
                if interleaved:
                    idx = torch.arange(c, n, num_tasks)
                else:
                    idx = torch.arange(c * num_data, (c + 1) * num_data)
                expected_covar = dense_covar[idx][:, idx]
                self.assertAllClose(
                    mvn_c.covariance_matrix, expected_covar, atol=1e-5
                )

    def test_separate_mtmvn_larger_interleaved(self) -> None:
        self._test_separate_mtmvn_larger(interleaved=True)

    def test_separate_mtmvn_larger_not_interleaved(self) -> None:
        self._test_separate_mtmvn_larger(interleaved=False)

    def _test_separate_mtmvn_batched(self, interleaved: bool) -> None:
        """Test with batch dimensions."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            batch_shape = torch.Size([3])
            num_data = 5
            num_tasks = 3
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
                self.assertAllClose(
                    mvn_c.covariance_matrix, expected_covar, atol=1e-5
                )

    def test_separate_mtmvn_batched_interleaved(self) -> None:
        self._test_separate_mtmvn_batched(interleaved=True)

    def test_separate_mtmvn_batched_not_interleaved(self) -> None:
        self._test_separate_mtmvn_batched(interleaved=False)
