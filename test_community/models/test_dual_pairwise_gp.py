#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.dual_pairwise_gp import DualPairwiseGP


class TestDualPairwiseGP(BotorchTestCase):
    def test_prior_and_strict_preferences(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for likelihood_type in ("probit", "logit"):
                tkwargs = {"device": self.device, "dtype": dtype}
                X = torch.linspace(0.0, 1.0, 6, **tkwargs).unsqueeze(-1)

                # Cold-start prior model (no observations yet)
                prior_model = DualPairwiseGP(X, likelihood_type=likelihood_type)
                self.assertEqual(prior_model.num_outputs, 1)
                self.assertEqual(prior_model.batch_shape, torch.Size())
                prior_post = prior_model.posterior(X)
                self.assertEqual(prior_post.mean.shape, torch.Size([6, 1]))
                self.assertTrue(torch.allclose(prior_post.mean, torch.zeros_like(X)))

                # Strict comparisons favoring index 4 (x = 0.8)
                comps = torch.tensor([[4, 0], [4, 1], [4, 2], [4, 5]], dtype=torch.long)
                model = DualPairwiseGP(
                    X, comparisons=comps, likelihood_type=likelihood_type
                )
                post = model.posterior(X)
                means = post.mean.squeeze(-1)
                self.assertEqual(int(torch.argmax(means).item()), 4)

                # Batched query evaluation and PosteriorTransform support
                X_batch = torch.rand(2, 3, 1, **tkwargs)
                post_batch = model.posterior(
                    X_batch,
                    posterior_transform=ScalarizedPosteriorTransform(
                        weights=torch.ones(1, **tkwargs)
                    ),
                )
                self.assertEqual(post_batch.mean.shape, torch.Size([2, 3, 1]))

    def test_ties_and_rejected_pairs_and_explicit_objectives(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.float64}
        X = torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.51, 0.49], [1.0, 1.0]], **tkwargs)
        explicit_obj = torch.tensor(
            [[0.1, 0.2], [0.8, 0.9], [0.8, 0.88], [0.2, 0.1]], **tkwargs
        )
        comps = torch.tensor([[1, 0]], dtype=torch.long)
        ties = torch.tensor([[1, 2]], dtype=torch.long)
        rejected = torch.tensor([[0, 3]], dtype=torch.long)

        model = DualPairwiseGP(
            datapoints=X,
            comparisons=comps,
            ties=ties,
            rejected_pairs=rejected,
            explicit_objectives=explicit_obj,
        )
        post = model.posterior(X)
        u = post.mean.squeeze(-1).detach()
        self.assertLess(abs(float(u[1] - u[2])), 0.25)
        self.assertGreater(float(u[1]), float(u[0]))
        self.assertGreater(float(u[1]), float(u[3]))

        weights = model.inferred_objective_weights
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, torch.Size([2]))
        self.assertTrue(torch.all(weights > 0.0))

    def test_input_validation_errors(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.float64}
        X = torch.rand(4, 2, **tkwargs)
        with self.assertRaises(UnsupportedError):
            DualPairwiseGP(X[:1])
        with self.assertRaises(UnsupportedError):
            DualPairwiseGP(X, likelihood_type="invalid")
        with self.assertRaises(UnsupportedError):
            DualPairwiseGP(X, tie_noise_variance=0.0)
        with self.assertRaises(UnsupportedError):
            DualPairwiseGP(X, comparisons=torch.tensor([[1, 1]]))
        with self.assertRaises(UnsupportedError):
            DualPairwiseGP(X, comparisons=torch.tensor([[0, 9]]))
        model = DualPairwiseGP(X)
        with self.assertRaises(UnsupportedError):
            model.posterior(X, output_indices=[0])
