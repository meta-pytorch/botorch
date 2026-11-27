#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch.utils.testing import BotorchTestCase


class TestHeterogeneousTargetConditioning(BotorchTestCase):
    def test_target_conditioning_matches_gaussian_law(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for batch_shape in ((), (2,)):
                with self.subTest(device=device, batch_shape=batch_shape):
                    torch.manual_seed(19)
                    tkwargs = {"dtype": torch.double, "device": device}
                    train_Xs = [torch.rand(n, 2, **tkwargs) for n in (4, 3)]
                    train_Ys = [X.sum(-1, keepdim=True).sin() for X in train_Xs]
                    model = HeterogeneousMTGP(
                        train_Xs=train_Xs,
                        train_Ys=train_Ys,
                        train_Yvars=[torch.full_like(Y, 0.03) for Y in train_Ys],
                        feature_indices=[[0, 1], [1, 2]],
                        full_feature_dim=3,
                        use_saas_prior=False,
                        outcome_transform=None,
                    )
                    new_X = torch.rand(*batch_shape, 2, 3, **tkwargs)
                    new_X[..., -1] = 0
                    new_Y = new_X[..., :1].sin()
                    noise = torch.full_like(new_Y, 0.04)
                    test_X = torch.rand(*batch_shape, 3, 3, **tkwargs)
                    test_X[..., -1] = 0
                    joint = model.posterior(torch.cat((new_X, test_X), -2))
                    covariance = joint.distribution.covariance_matrix
                    mean = joint.mean
                    observed = covariance[..., :2, :2] + torch.diag_embed(
                        noise.squeeze(-1)
                    )
                    cross = covariance[..., 2:, :2]
                    expected_mean = mean[..., 2:, :] + cross @ torch.linalg.solve(
                        observed, new_Y - mean[..., :2, :]
                    )
                    expected_covariance = covariance[..., 2:, 2:] - cross @ (
                        torch.linalg.solve(observed, cross.mT)
                    )
                    conditioned = model.condition_on_observations(
                        new_X, new_Y, noise=noise
                    )
                    actual = conditioned.posterior(test_X)
                    self.assertAllClose(actual.mean, expected_mean)
                    self.assertAllClose(
                        actual.distribution.covariance_matrix, expected_covariance
                    )
                    self.assertEqual(conditioned.train_inputs[0].shape[-1], 4)
