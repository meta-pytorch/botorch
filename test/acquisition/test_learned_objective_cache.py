#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, LearnedObjective
from botorch.models import PairwiseGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from test.acquisition.test_cached_cholesky import DummyCachedCholeskyAcqf


class TestLearnedObjectiveNoisyImprovementCache(BotorchTestCase):
    def test_joint_preference_law_for_duplicate_baseline(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device, acquisition_cls in itertools.product(
            devices, (qNoisyExpectedImprovement, qLogNoisyExpectedImprovement)
        ):
            with self.subTest(device=device, acquisition=acquisition_cls):
                X = torch.tensor(
                    [[0.1, 0.2], [0.3, 0.8], [0.7, 0.4], [0.9, 0.7]],
                    dtype=torch.double,
                    device=device,
                )
                Y = torch.stack((X[:, 0].sin(), X[:, 1].square()), -1)
                outcome = SingleTaskGP(X, Y, torch.full_like(Y, 0.03))
                points = torch.rand(
                    7,
                    2,
                    generator=torch.Generator().manual_seed(82),
                    dtype=torch.double,
                ).to(device)
                comparisons = torch.tensor(
                    [[0, 1], [2, 3], [4, 5], [5, 6], [0, 4], [3, 6]], device=device
                )
                np.random.seed(83)
                preference = PairwiseGP(points, comparisons)
                values = []
                for cache in (False, None, True):
                    objective = LearnedObjective(
                        preference, sample_shape=torch.Size([16]), seed=84
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        acquisition = acquisition_cls(
                            outcome,
                            X[:2],
                            objective=objective,
                            cache_root=cache,
                            sampler=SobolQMCNormalSampler(torch.Size([32]), seed=85),
                            prune_baseline=False,
                        )
                    self.assertFalse(acquisition._cache_root)
                    warning = any(
                        "must be sampled jointly" in str(item.message)
                        for item in caught
                    )
                    self.assertEqual(warning, cache is True)
                    value = acquisition(X[:1])
                    values.append(value.detach())
                self.assertAllClose(values[0], values[1])
                self.assertAllClose(values[0], values[2])
                improvement = (
                    values[0].exp()
                    if acquisition_cls is qLogNoisyExpectedImprovement
                    else values[0]
                )
                self.assertLess(improvement.item(), 1e-3)

    def test_wrapped_learned_objective_and_deterministic_cache(self):
        X = torch.tensor([[0.2], [0.6]], dtype=torch.double)
        outcome = SingleTaskGP(X, X.sin())
        preference = PairwiseGP(X, torch.tensor([[0, 1]]))
        learned = LearnedObjective(preference)
        wrapped = GenericMCObjective(learned)
        self.assertFalse(
            DummyCachedCholeskyAcqf(outcome, objective=wrapped)._cache_root
        )
        deterministic = LearnedObjective(
            GenericDeterministicModel(lambda value: value.sum(-1, keepdim=True))
        )
        self.assertTrue(
            DummyCachedCholeskyAcqf(outcome, objective=deterministic)._cache_root
        )
