#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.likelihoods.pairwise import (
    PairwiseLikelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.testing import BotorchTestCase
from torch import Tensor
from torch.distributions import Bernoulli


class TestPairwiseLikelihood(BotorchTestCase):
    def test_pairwise_likelihood(self):
        # Test subclassing
        class BadCustomLikelihood(PairwiseLikelihood):
            pass

        with self.assertRaises(TypeError):
            # Can't instantiate with abstract methods p
            BadCustomLikelihood()

        class OkayCustomLikelihood(PairwiseLikelihood):
            def p(self, utility: Tensor, D: Tensor) -> Tensor:
                return D.to(utility) @ utility.unsqueeze(-1)

        likelihood = OkayCustomLikelihood()
        with self.assertRaises(NotImplementedError):
            likelihood.negative_log_gradient_sum(
                utility=torch.rand(2), D=torch.rand(2, 2)
            )

        with self.assertRaises(NotImplementedError):
            likelihood.negative_log_hessian_sum(
                utility=torch.rand(2), D=torch.rand(2, 2)
            )

        # Test implemented PairwiseLikelihood subclasses
        for batch_shape, likelihood_cls in itertools.product(
            (torch.Size(), torch.Size([2])),
            (PairwiseLogitLikelihood, PairwiseProbitLikelihood),
        ):
            n_datapoints = 4
            n_comps = 3
            X_dim = 4
            train_X = torch.rand(*batch_shape, n_datapoints, X_dim, dtype=torch.double)
            train_Y = train_X.sum(dim=-1, keepdim=True)
            train_comp = torch.stack(
                [
                    torch.randperm(train_Y.shape[-2])[:2]
                    for _ in range(torch.tensor(batch_shape).prod().int() * n_comps)
                ]
            ).reshape(*batch_shape, -1, 2)

            model = PairwiseGP(datapoints=train_X, comparisons=train_comp).eval()
            model.posterior(train_X)
            utility = model.utility
            D = model.D
            likelihood = likelihood_cls()

            # test forward
            dist = likelihood(utility, D)
            self.assertTrue(isinstance(dist, Bernoulli))

            # test p
            probs = likelihood.p(utility=utility, D=D)
            self.assertTrue(probs.shape == torch.Size((*batch_shape, n_comps)))
            self.assertTrue((probs >= 0).all() and (probs <= 1).all())

            # test log p
            log_probs = likelihood.log_p(utility=utility, D=D)
            self.assertTrue(torch.allclose(torch.log(probs), log_probs))

            # test negative_log_gradient_sum
            grad_sum = likelihood.negative_log_gradient_sum(utility=utility, D=D)
            self.assertEqual(grad_sum.shape, torch.Size((*batch_shape, n_datapoints)))

            # test negative_log_hessian_sum
            hess_sum = likelihood.negative_log_hessian_sum(utility=utility, D=D)
            self.assertEqual(
                hess_sum.shape, torch.Size((*batch_shape, n_datapoints, n_datapoints))
            )

            # the gradient and hessian only depend on utility differences, also
            # for large utilities, for which exp(utility) overflows
            for shift in (800.0, -800.0):
                self.assertAllClose(
                    likelihood.negative_log_gradient_sum(utility=utility + shift, D=D),
                    grad_sum,
                )
                self.assertAllClose(
                    likelihood.negative_log_hessian_sum(utility=utility + shift, D=D),
                    hess_sum,
                )

    def test_logit_negative_log_gradient_sum(self):
        likelihood = PairwiseLogitLikelihood()
        # item 0 is preferred over item 1, and item 1 over item 2
        D = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=torch.double)
        # gradient of -log(sigmoid(f_w - f_l)) summed over the comparisons, at
        # f_0 - f_1 = 0.5 and f_1 - f_2 = -2
        s0, s1 = torch.sigmoid(torch.tensor([-0.5, 2.0], dtype=torch.double))
        expected = torch.stack([-s0, s0 - s1, s1])
        for offset in (0.0, 800.0, -800.0):
            utility = torch.tensor([0.5, 0.0, 2.0], dtype=torch.double) + offset
            grad = likelihood.negative_log_gradient_sum(utility=utility, D=D)
            self.assertFalse(grad.isnan().any())
            self.assertAllClose(grad, expected)
