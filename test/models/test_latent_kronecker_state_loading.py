#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from unittest import skipUnless

import torch
from botorch.models.latent_kronecker_gp import LatentKroneckerGP
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


class TestLatentKroneckerStateLoading(BotorchTestCase):
    def _data(self, device, batch, missing, t=2):
        X = torch.tensor([[0.1], [0.45], [0.9]], device=device, dtype=torch.double)
        T = X.new_tensor([[0.0], [1.0]])[:t]
        Y = 1.8 + (X * 2 + T.squeeze(-1) * 0.7).sin()
        if missing:
            Y[1, 0] = torch.nan
        if batch:
            X = X.expand(2, -1, -1).clone()
            Y = Y.expand(2, -1, -1).clone()
            Y[1] += 0.2
        return X, T, Y

    def _test_without_outcome(self, device):
        for batch, missing, t, keep_transforms, assign in itertools.product(
            (False, True), (False, True), (1, 2), (False, True), (False, True)
        ):
            with self.subTest(
                batch=batch,
                missing=missing,
                t=t,
                keep_transforms=keep_transforms,
                assign=assign,
            ):
                X, T, Y = self._data(device, batch, missing, t)
                model = LatentKroneckerGP(X, T, Y, outcome_transform=None)
                targets = model.train_targets.clone()
                model.load_state_dict(
                    copy.deepcopy(model.state_dict()),
                    keep_transforms=keep_transforms,
                    assign=assign,
                )
                self.assertAllClose(model.train_targets, targets)
                self.assertEqual(model.train_targets.shape, targets.shape)
                self.assertIs(model.train_T, model.train_inputs[1])
                self.assertEqual(model.train_T.shape[-2], t)

    def _test_with_outcome(self, device):
        for batch, missing, chain, fixed_noise, assign in itertools.product(
            (False, True), (False, True), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                batch=batch,
                missing=missing,
                chain=chain,
                fixed_noise=fixed_noise,
                assign=assign,
            ):
                X, T, Y = self._data(device, batch, missing)
                batch_shape = X.shape[:-2]
                mask = Y.reshape(-1, *Y.shape[-2:])[0].isfinite().flatten()
                targets = Y.flatten(-2)[..., mask]

                def make(values):
                    standardize = Standardize(1, batch_shape=batch_shape)
                    outcome = (
                        ChainedOutcomeTransform(log=Log(), standardize=standardize)
                        if chain
                        else standardize
                    )
                    likelihood = (
                        FixedNoiseGaussianLikelihood(
                            noise=targets.new_full(targets.shape, 0.05)
                        )
                        if fixed_noise
                        else None
                    )
                    return LatentKroneckerGP(
                        X, T, values, outcome_transform=outcome, likelihood=likelihood
                    ).to(X)

                source, model = make(Y), make(Y * 2.7 + 1.2)
                initial_targets, initial_noise = model._extract_targets_and_noise()
                raw_targets, raw_noise = model.outcome_transform.untransform(
                    initial_targets, initial_noise, X=X
                )
                expected_targets, expected_noise = source.outcome_transform.eval()(
                    raw_targets, raw_noise, X=X
                )
                model.load_state_dict(copy.deepcopy(source.state_dict()), assign=assign)
                self.assertEqual(model.train_targets.shape, targets.shape)
                self.assertAllClose(model.train_targets, expected_targets.squeeze(-1))
                if fixed_noise:
                    self.assertEqual(model.likelihood.noise.shape, targets.shape)
                    self.assertAllClose(
                        model.likelihood.noise, expected_noise.squeeze(-1)
                    )
                actual_raw = model.outcome_transform.untransform(
                    model.train_targets.unsqueeze(-1),
                    model.likelihood.noise.unsqueeze(-1) if fixed_noise else None,
                    X=X,
                )
                self.assertAllClose(actual_raw[0], raw_targets)
                if fixed_noise:
                    self.assertAllClose(actual_raw[1], raw_noise)

    def test_without_outcome(self):
        self._test_without_outcome(torch.device("cpu"))

    def test_with_outcome(self):
        self._test_with_outcome(torch.device("cpu"))

    @skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_without_outcome_cuda(self):
        self._test_without_outcome(torch.device("cuda"))

    @skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_with_outcome_cuda(self):
        self._test_with_outcome(torch.device("cuda"))
