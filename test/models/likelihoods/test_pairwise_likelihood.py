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
    def test_logit_zero_rows_and_large_common_shift(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, dtype in itertools.product(devices, (torch.float, torch.double)):
            with self.subTest(device=device, dtype=dtype):
                utility = torch.tensor(
                    [[0.2, 0.7], [0.4, -0.3]], device=device, dtype=dtype
                )
                D = utility.new_tensor([[[1, -1], [0, 0]], [[0, 0], [-1, 1]]])
                likelihood = PairwiseLogitLikelihood()
                expected_gradient = likelihood.negative_log_gradient_sum(utility, D)
                expected_hessian = likelihood.negative_log_hessian_sum(utility, D)
                self.assertTrue(torch.isfinite(expected_gradient).all())
                self.assertTrue(torch.isfinite(expected_hessian).all())
                shifted = (utility + 1000).requires_grad_()
                # The zero comparison contributes log(2), with no utility
                # information, and a common offset cannot change preferences.
                loss = torch.nn.functional.softplus(
                    -(D @ shifted.unsqueeze(-1)).squeeze(-1)
                ).sum()
                gradient = torch.autograd.grad(loss, shifted, create_graph=True)[0]
                actual = likelihood.negative_log_gradient_sum(shifted, D)
                self.assertAllClose(actual, gradient)
                self.assertAllClose(actual, expected_gradient, atol=3e-5)
                hessian = likelihood.negative_log_hessian_sum(shifted, D)
                for batch in range(2):
                    dense = torch.autograd.functional.hessian(
                        lambda value: torch.nn.functional.softplus(
                            -(D[batch] @ value)
                        ).sum(),
                        shifted[batch],
                    )
                    self.assertAllClose(hessian[batch], dense)
                self.assertAllClose(hessian, expected_hessian, atol=3e-5)

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
