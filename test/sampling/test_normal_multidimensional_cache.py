#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


class TestNormalSamplerMultidimensionalCache(BotorchTestCase):
    def test_copy_baseline_samples(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, dtype, outputs, shape, sampler_cls in itertools.product(
            devices,
            (torch.float, torch.double),
            (1, 2),
            (torch.Size([3]), torch.Size([2, 3]), torch.Size([2, 1, 3])),
            (IIDNormalSampler, SobolQMCNormalSampler),
        ):
            with self.subTest(
                device=device,
                dtype=dtype,
                outputs=outputs,
                shape=shape,
                sampler=sampler_cls,
            ):

                def posterior(q, batch):
                    if outputs == 1:
                        distribution = MultivariateNormal(
                            torch.zeros(*batch, q, device=device, dtype=dtype),
                            torch.eye(q, device=device, dtype=dtype),
                        )
                    else:
                        distribution = MultitaskMultivariateNormal(
                            torch.zeros(*batch, q, outputs, device=device, dtype=dtype),
                            torch.eye(q * outputs, device=device, dtype=dtype),
                        )
                    return GPyTorchPosterior(distribution)

                base = sampler_cls(shape, seed=7)
                base(posterior(3, ()))
                sampler = sampler_cls(shape, seed=8)
                joint = posterior(5, (2,))
                sampler._construct_base_samples(joint)
                original = sampler.base_samples.clone()
                sampler._update_base_samples(joint, base)
                event = (3,) if outputs == 1 else (3, outputs)
                expected = base.base_samples.reshape(*shape, 1, *event)
                if outputs == 1:
                    self.assertAllClose(sampler.base_samples[..., :3], expected)
                    self.assertAllClose(
                        sampler.base_samples[..., 3:], original[..., 3:]
                    )
                else:
                    self.assertAllClose(sampler.base_samples[..., :3, :], expected)
                    self.assertAllClose(
                        sampler.base_samples[..., 3:, :], original[..., 3:, :]
                    )

    def test_cached_noisy_improvement(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, acquisition_cls, outputs, shape in itertools.product(
            devices,
            (qNoisyExpectedImprovement, qLogNoisyExpectedImprovement),
            (1, 2),
            (torch.Size([2, 3]), torch.Size([2, 1, 3])),
        ):
            with self.subTest(
                device=device, acquisition=acquisition_cls, outputs=outputs, shape=shape
            ):
                X = torch.linspace(
                    0, 1, 6, device=device, dtype=torch.double
                ).unsqueeze(-1)
                Y = X.sin() if outputs == 1 else torch.cat((X.sin(), X.cos()), -1)
                model = SingleTaskGP(X, Y, torch.full_like(Y, 0.02))

                def acquisition(sample_shape):
                    return acquisition_cls(
                        model,
                        X[:3],
                        objective=GenericMCObjective(lambda Y, X: Y.sum(-1)),
                        sampler=SobolQMCNormalSampler(sample_shape, seed=9),
                        cache_root=True,
                        prune_baseline=False,
                    )

                candidate = X.new_tensor([[[0.6]], [[0.8]]]).requires_grad_()
                value = acquisition(shape)(candidate)
                expected = acquisition(torch.Size([shape.numel()]))(candidate)
                self.assertEqual(value.shape, torch.Size([2]))
                self.assertTrue(torch.isfinite(value).all())
                self.assertAllClose(value, expected)
                self.assertTrue(
                    torch.isfinite(torch.autograd.grad(value.sum(), candidate)[0]).all()
                )
