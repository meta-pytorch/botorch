#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskVariationalGP
from botorch.utils.multitask import separate_mtmvn
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.settings import lazily_evaluate_kernels


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

    def test_separate_mtmvn_variational_gp_posterior(self) -> None:
        # The posterior of a multi-output variational GP is an interleaved MTMVN with
        # a ``BlockInterleavedLinearOperator`` covariance of lazily evaluated kernels.
        tkwargs = {"device": self.device, "dtype": torch.double}
        X = torch.rand(12, 2, **tkwargs)
        Y = torch.randn(12, 3, **tkwargs)
        model = SingleTaskVariationalGP(
            X, Y[:, :2], num_outputs=2, inducing_points=X[:6]
        ).to(**tkwargs)
        # Use different lengthscales for the two outputs.
        model.model.covar_module.lengthscale = torch.tensor(
            [[[0.2, 0.5]], [[1.5, 0.9]]], **tkwargs
        )
        model.eval()
        model_list = ModelListGP(model, SingleTaskGP(X, Y[:, 2:])).eval()
        for batch_shape in (torch.Size([]), torch.Size([3]), torch.Size([4, 1])):
            for q in (1, 2, 3):
                with self.subTest(batch_shape=batch_shape, q=q), torch.no_grad():
                    Xt = torch.rand(*batch_shape, q, 2, **tkwargs)
                    with lazily_evaluate_kernels(False):
                        expected = model.posterior(Xt).distribution.covariance_matrix
                    mvn = model.posterior(Xt).distribution
                    self.assertTrue(mvn._interleaved)
                    for c, mvn_c in enumerate(separate_mtmvn(mvn)):
                        covar_c = mvn_c.lazy_covariance_matrix
                        expected_c = expected[..., c::2, c::2]
                        self.assertAllClose(covar_c.to_dense(), expected_c)
                        # The variance of a ModelListGP posterior is computed like this.
                        self.assertAllClose(
                            covar_c.unsqueeze(-3).diagonal(dim1=-2, dim2=-1),
                            expected_c.diagonal(dim1=-2, dim2=-1).unsqueeze(-2),
                        )
                    variance = model_list.posterior(Xt).variance
                    self.assertAllClose(variance[..., :2], model.posterior(Xt).variance)
