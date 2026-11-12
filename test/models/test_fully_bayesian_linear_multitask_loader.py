#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.fully_bayesian import LinearPyroModel
from botorch.models.fully_bayesian_multitask import (
    LatentFeatureMultiTaskPyroMixin,
    SaasFullyBayesianMultiTaskGP,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.means import MultitaskMean


class _MultitaskLinearPyroModel(LatentFeatureMultiTaskPyroMixin, LinearPyroModel):
    pass


class TestLinearMultitaskMCMCLoader(BotorchTestCase):
    def test_loaded_mean_covariance_transform_and_posterior(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, task_feature, warp, fixed in itertools.product(
            devices, range(3), (False, True), (False, True)
        ):
            with self.subTest(
                device=device, task_feature=task_feature, warp=warp, fixed=fixed
            ):
                base = torch.tensor(
                    [[0.2, 0.7], [0.6, 0.3], [0.8, 0.9], [0.4, 0.5]],
                    dtype=torch.double,
                    device=device,
                )
                labels = base.new_tensor([[0.0], [1.0], [0.0], [1.0]])
                X = torch.cat(
                    (base[:, :task_feature], labels, base[:, task_feature:]), -1
                )
                Y = base[:, :1].sin()
                pyro = _MultitaskLinearPyroModel(use_input_warping=warp)
                model = SaasFullyBayesianMultiTaskGP(
                    X,
                    Y,
                    task_feature=task_feature,
                    train_Yvar=torch.full_like(Y, 0.03) if fixed else None,
                    pyro_model=pyro,
                    rank=1,
                )
                samples = {
                    "mean": base.new_tensor([[0.1, -0.2], [0.3, 0.5], [-0.4, 0.2]]),
                    "weight_variance": base.new_tensor(
                        [[0.7, 1.1], [1.3, 0.6], [0.9, 1.4]]
                    ),
                    "latent_features": base.new_tensor(
                        [[[0.1], [0.8]], [[-0.2], [0.5]], [[0.4], [1.2]]]
                    ),
                    "task_lengthscale": base.new_tensor([[0.5], [0.8], [1.1]]),
                    "noise": base.new_tensor([0.04, 0.05, 0.06]),
                }
                if warp:
                    samples.update(
                        c0=base.new_full((3, 2), 0.8), c1=base.new_full((3, 2), 1.2)
                    )
                model.load_mcmc_samples(samples)
                self.assertIsInstance(model.mean_module, MultitaskMean)
                for task in range(2):
                    self.assertAllClose(
                        model.mean_module.base_means[task].constant,
                        samples["mean"][:, task],
                    )
                transformed = model.input_transform(X).expand(3, *X.shape)
                self.assertAllClose(
                    transformed[..., task_feature], labels.squeeze(-1).expand(3, -1)
                )
                columns = [i for i in range(3) if i != task_feature]
                centered = base.expand(3, *base.shape)
                if warp:
                    eps = pyro._eps
                    centered = centered.clamp(eps, 1 - eps)
                    centered = 1 - (1 - centered.pow(samples["c1"].unsqueeze(-2))).pow(
                        samples["c0"].unsqueeze(-2)
                    )
                centered = centered - 0.5
                self.assertAllClose(transformed[..., columns], centered)
                weighted = centered * samples["weight_variance"].unsqueeze(-2)
                spatial = weighted @ centered.transpose(-2, -1)
                latent = samples["latent_features"]
                distance = (
                    (
                        (latent[:, :, None] - latent[:, None, :])
                        / samples["task_lengthscale"][:, None, None]
                    )
                    .square()
                    .sum(-1)
                    .sqrt()
                )
                root5 = 5**0.5 * distance
                task_covariance = (1 + root5 + root5.square() / 3) * (-root5).exp()
                tasks = labels.squeeze(-1).long()
                expected_covariance = spatial * task_covariance[:, tasks][:, :, tasks]
                self.assertAllClose(
                    model.covar_module(transformed).to_dense(), expected_covariance
                )
                # The public posterior must consume the returned centering/warp
                # transform and the actual task means/covariance jointly.
                posterior = model.posterior(X)
                noise = (
                    base.new_full((3, 4), 0.03)
                    if fixed
                    else samples["noise"].unsqueeze(-1).expand(3, 4)
                )
                train_mean = samples["mean"][:, tasks]
                factor = expected_covariance + torch.diag_embed(noise)
                expected_mean = train_mean + (
                    expected_covariance
                    @ torch.linalg.solve(factor, Y.squeeze(-1) - train_mean).unsqueeze(
                        -1
                    )
                ).squeeze(-1)
                expected_posterior_covariance = (
                    expected_covariance
                    - expected_covariance
                    @ torch.linalg.solve(factor, expected_covariance)
                )
                self.assertAllClose(posterior.mean.squeeze(-1), expected_mean)
                self.assertAllClose(
                    posterior.distribution.covariance_matrix,
                    expected_posterior_covariance,
                )
