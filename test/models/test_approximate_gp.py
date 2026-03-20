#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.warnings import UserInputWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.approximate_gp import (
    _SingleTaskVariationalGP,
    ApproximateGPyTorchModel,
    SingleTaskVariationalGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Log
from botorch.models.utils.inducing_point_allocators import (
    GreedyImprovementReduction,
    GreedyVarianceReduction,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.posteriors.torch import TorchPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import (
    BetaLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)


class TestApproximateGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.train_X = torch.rand(10, 1, device=self.device)
        self.train_Y = torch.sin(self.train_X) + torch.randn_like(self.train_X) * 0.2

    def test_initialization(self):
        # test non batch case
        model = ApproximateGPyTorchModel(train_X=self.train_X, train_Y=self.train_Y)
        self.assertIsInstance(model.model, _SingleTaskVariationalGP)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)
        self.assertIsInstance(model.model.variational_strategy, VariationalStrategy)
        self.assertEqual(model.num_outputs, 1)

        # test batch case
        stacked_y = torch.cat((self.train_Y, self.train_Y), dim=-1)
        model = ApproximateGPyTorchModel(
            train_X=self.train_X, train_Y=stacked_y, num_outputs=2
        )
        self.assertIsInstance(model.model, _SingleTaskVariationalGP)
        self.assertIsInstance(model.likelihood, MultitaskGaussianLikelihood)
        self.assertIsInstance(
            model.model.variational_strategy, IndependentMultitaskVariationalStrategy
        )
        self.assertEqual(model.num_outputs, 2)


class TestSingleTaskVariationalGP(BotorchTestCase):
    def setUp(self):
        super().setUp()
        train_X = torch.rand(10, 1, device=self.device)
        train_y = torch.sin(train_X) + torch.randn_like(train_X) * 0.2

        self.model = SingleTaskVariationalGP(
            train_X=train_X, likelihood=GaussianLikelihood()
        ).to(self.device)

        mll = VariationalELBO(self.model.likelihood, self.model.model, num_data=10)
        loss = -mll(self.model.likelihood(self.model(train_X)), train_y).sum()
        loss.backward()

    def test_posterior(self):
        # basic test of checking that the posterior works as intended
        test_x = torch.rand(30, 1, device=self.device)
        posterior = self.model.posterior(test_x)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        posterior = self.model.posterior(test_x, observation_noise=True)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        # now loop through all possibilities
        train_X = torch.rand(3, 10, 1, device=self.device)
        train_Y = torch.randn(3, 10, 2, device=self.device)
        test_X = torch.rand(3, 5, 1, device=self.device)

        all_tests = {
            "non_batched": [train_X[0], train_Y[0, :, :1], test_X[0]],
            "non_batched_mo": [train_X[0], train_Y[0], test_X[0]],
            "batched": [train_X, train_Y[..., :1], test_X],
            # batched multi-output is not supported at this time
            # "batched_mo": [train_X, train_Y, test_X],
            "non_batched_to_batched": [train_X[0], train_Y[0], test_X],
        }

        for test_name, [tx, ty, test] in all_tests.items():
            with self.subTest(test_name=test_name):
                model = SingleTaskVariationalGP(tx, ty, inducing_points=tx)
                posterior = model.posterior(test)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                # test batch_shape property
                self.assertEqual(model.batch_shape, tx.shape[:-2])

        # Test that checks if posterior_transform is correctly applied
        [tx1, ty1, test1] = all_tests["non_batched_mo"]
        model1 = SingleTaskVariationalGP(tx1, ty1, inducing_points=tx1)
        posterior_transform = ScalarizedPosteriorTransform(
            weights=torch.tensor([1.0, 1.0], device=self.device)
        )
        posterior1 = model1.posterior(test1, posterior_transform=posterior_transform)
        self.assertIsInstance(posterior1, GPyTorchPosterior)
        self.assertEqual(posterior1.mean.shape[1], 1)

    def test_variational_setUp(self):
        for dtype in [torch.float, torch.double]:
            train_X = torch.rand(10, 1, device=self.device, dtype=dtype)
            train_y = torch.randn(10, 3, device=self.device, dtype=dtype)

            for ty, num_out in [[train_y, 3], [train_y, 1], [None, 3]]:
                batched_model = SingleTaskVariationalGP(
                    train_X,
                    train_Y=ty,
                    num_outputs=num_out,
                    learn_inducing_points=False,
                ).to(self.device)
                mll = VariationalELBO(
                    batched_model.likelihood, batched_model.model, num_data=10
                )

                with torch.enable_grad():
                    loss = -mll(
                        batched_model.likelihood(batched_model(train_X)), train_y
                    ).sum()
                    loss.backward()

                # ensure that inducing points do not require grad
                model_var_strat = batched_model.model.variational_strategy
                self.assertEqual(
                    model_var_strat.base_variational_strategy.inducing_points.grad,
                    None,
                )

                # but that the covariance does have a gradient
                self.assertIsNotNone(
                    batched_model.model.covar_module.raw_lengthscale.grad
                )

                # check that we always have three outputs
                self.assertEqual(batched_model._num_outputs, 3)
                self.assertIsInstance(
                    batched_model.likelihood, MultitaskGaussianLikelihood
                )

    def test_likelihood(self):
        self.assertIsInstance(self.model.likelihood, GaussianLikelihood)
        self.assertTrue(self.model._is_custom_likelihood, True)

    def test_initializations(self):
        train_X = torch.rand(15, 1, device=self.device)
        train_Y = torch.rand(15, 1, device=self.device)

        stacked_train_X = torch.cat((train_X, train_X), dim=0)
        for X, num_ind in [[train_X, 5], [stacked_train_X, 20], [stacked_train_X, 5]]:
            model = SingleTaskVariationalGP(train_X=X, inducing_points=num_ind)
            if num_ind == 5:
                self.assertLessEqual(
                    model.model.variational_strategy.inducing_points.shape,
                    torch.Size((5, 1)),
                )
            else:
                # should not have 20 inducing points when 15 singular dimensions
                # are passed
                self.assertLess(
                    model.model.variational_strategy.inducing_points.shape[-2], num_ind
                )

        test_X = torch.rand(5, 1, device=self.device)

        # test transforms
        for inp_trans, out_trans in itertools.product(
            [None, Normalize(d=1)], [None, Log()]
        ):
            model = SingleTaskVariationalGP(
                train_X=train_X,
                train_Y=train_Y,
                outcome_transform=out_trans,
                input_transform=inp_trans,
            )

            if inp_trans is not None:
                self.assertIsInstance(model.input_transform, Normalize)
            else:
                self.assertFalse(hasattr(model, "input_transform"))
            if out_trans is not None:
                self.assertIsInstance(model.outcome_transform, Log)

                posterior = model.posterior(test_X)
                self.assertIsInstance(posterior, TransformedPosterior)
            else:
                self.assertFalse(hasattr(model, "outcome_transform"))

        # test user warnings when using transforms
        with self.assertWarnsRegex(
            UserInputWarning,
            "Using an input transform with `SingleTaskVariationalGP`",
        ):
            SingleTaskVariationalGP(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
            )
        with self.assertWarnsRegex(
            UserInputWarning,
            "Using an outcome transform with `SingleTaskVariationalGP`",
        ):
            SingleTaskVariationalGP(
                train_X=train_X,
                train_Y=train_Y,
                outcome_transform=Log(),
            )

        # test default inducing point allocator
        self.assertIsInstance(model._inducing_point_allocator, GreedyVarianceReduction)

        # test that can specify an inducing point allocator
        for ipa in [
            GreedyVarianceReduction(),
            GreedyImprovementReduction(model, maximize=True),
        ]:
            model = SingleTaskVariationalGP(train_X, inducing_point_allocator=ipa)
            self.assertTrue(type(model._inducing_point_allocator), type(ipa))

        # test warning when learning on and custom IPA provided
        with self.assertWarnsRegex(
            UserWarning, r"set `learn_inducing_points` to False"
        ):
            SingleTaskVariationalGP(
                train_X,
                learn_inducing_points=True,
                inducing_point_allocator=GreedyVarianceReduction(),
            )

    def test_inducing_point_init(self):
        train_X_1 = torch.rand(15, 1, device=self.device)
        train_X_2 = torch.rand(15, 1, device=self.device)

        # single-task
        model_1 = SingleTaskVariationalGP(train_X=train_X_1, inducing_points=5)
        model_1.init_inducing_points(train_X_2)
        model_1_inducing = model_1.model.variational_strategy.inducing_points

        model_2 = SingleTaskVariationalGP(train_X=train_X_2, inducing_points=5)
        model_2_inducing = model_2.model.variational_strategy.inducing_points

        self.assertEqual(model_1_inducing.shape, (5, 1))
        self.assertEqual(model_2_inducing.shape, (5, 1))
        self.assertAllClose(model_1_inducing, model_2_inducing)

        # multi-task
        model_1 = SingleTaskVariationalGP(
            train_X=train_X_1, inducing_points=5, num_outputs=2
        )
        model_1.init_inducing_points(train_X_2)
        model_1_inducing = (
            model_1.model.variational_strategy.base_variational_strategy.inducing_points
        )

        model_2 = SingleTaskVariationalGP(
            train_X=train_X_2, inducing_points=5, num_outputs=2
        )
        model_2_inducing = (
            model_2.model.variational_strategy.base_variational_strategy.inducing_points
        )

        self.assertEqual(model_1_inducing.shape, (5, 1))
        self.assertEqual(model_2_inducing.shape, (5, 1))
        self.assertAllClose(model_1_inducing, model_2_inducing)

        # batched inputs
        train_X_1 = torch.rand(2, 15, 1, device=self.device)
        train_X_2 = torch.rand(2, 15, 1, device=self.device)
        train_Y = torch.rand(2, 15, 1, device=self.device)

        model_1 = SingleTaskVariationalGP(
            train_X=train_X_1, train_Y=train_Y, inducing_points=5
        )
        model_1.init_inducing_points(train_X_2)
        model_1_inducing = model_1.model.variational_strategy.inducing_points
        model_2 = SingleTaskVariationalGP(
            train_X=train_X_2, train_Y=train_Y, inducing_points=5
        )
        model_2_inducing = model_2.model.variational_strategy.inducing_points

        self.assertEqual(model_1_inducing.shape, (2, 5, 1))
        self.assertEqual(model_2_inducing.shape, (2, 5, 1))
        self.assertAllClose(model_1_inducing, model_2_inducing)

    def test_custom_inducing_point_init(self):
        train_X_0 = torch.rand(15, 1, device=self.device)
        train_X_1 = torch.rand(15, 1, device=self.device)
        train_X_2 = torch.rand(15, 1, device=self.device)
        train_X_3 = torch.rand(15, 1, device=self.device)

        model_from_previous_step = SingleTaskVariationalGP(
            train_X=train_X_0, inducing_points=5
        )

        model_1 = SingleTaskVariationalGP(
            train_X=train_X_1,
            inducing_points=5,
            inducing_point_allocator=GreedyImprovementReduction(
                model_from_previous_step, maximize=True
            ),
        )
        model_1.init_inducing_points(train_X_2)
        model_1_inducing = model_1.model.variational_strategy.inducing_points

        model_2 = SingleTaskVariationalGP(
            train_X=train_X_2,
            inducing_points=5,
            inducing_point_allocator=GreedyImprovementReduction(
                model_from_previous_step, maximize=True
            ),
        )
        model_2_inducing = model_2.model.variational_strategy.inducing_points

        model_3 = SingleTaskVariationalGP(
            train_X=train_X_3,
            inducing_points=5,
            inducing_point_allocator=GreedyImprovementReduction(
                model_from_previous_step, maximize=False
            ),
        )
        model_3.init_inducing_points(train_X_2)
        model_3_inducing = model_3.model.variational_strategy.inducing_points

        self.assertEqual(model_1_inducing.shape, (5, 1))
        self.assertEqual(model_2_inducing.shape, (5, 1))
        self.assertAllClose(model_1_inducing, model_2_inducing)
        self.assertFalse(model_1_inducing[0, 0] == model_3_inducing[0, 0])

    def test_non_gaussian_likelihood_posterior(self) -> None:
        """Test that non-Gaussian likelihoods return TorchPosterior."""
        train_X = torch.rand(10, 1, device=self.device)
        model = SingleTaskVariationalGP(
            train_X=train_X,
            likelihood=BetaLikelihood(),
        ).to(self.device)
        test_X = torch.rand(5, 1, device=self.device)

        # Without observation noise, the distribution is MVN (from the GP),
        # so it should return GPyTorchPosterior.
        posterior = model.posterior(test_X, observation_noise=False)
        self.assertIsInstance(posterior, GPyTorchPosterior)

        # With observation noise, the likelihood transforms the MVN into a
        # Beta distribution, so it should return TorchPosterior.
        posterior = model.posterior(test_X, observation_noise=True)
        self.assertIsInstance(posterior, TorchPosterior)
        self.assertNotIsInstance(posterior, GPyTorchPosterior)

        # Verify that sampling works with the TorchPosterior and that the
        # shape doesn't have a spurious trailing dim from GPyTorchPosterior.
        # GPyTorchPosterior.rsample would unsqueeze(-1), adding an extra dim.
        samples = posterior.rsample(sample_shape=torch.Size([2]))
        self.assertEqual(samples.shape, torch.Size([2, 10, 5]))

    def test_input_transform(self) -> None:
        train_X = torch.linspace(1, 3, 10, dtype=torch.double)[:, None]
        y = -3 * train_X + 5

        for input_transform in [None, Normalize(1)]:
            with self.subTest(input_transform=input_transform):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Input data is not contained"
                    )
                    model = SingleTaskVariationalGP(
                        train_X=train_X, train_Y=y, input_transform=input_transform
                    )
                mll = VariationalELBO(
                    model.likelihood, model.model, num_data=train_X.shape[-2]
                )
                fit_gpytorch_mll(mll)
                post = model.posterior(torch.tensor([[train_X.mean()]]))
                self.assertAllClose(post.mean[0][0], y.mean(), atol=1e-3, rtol=1e-3)
