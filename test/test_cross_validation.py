#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestFitBatchCrossValidation(BotorchTestCase):
    def test_single_task_batch_cv(self) -> None:
        n = 10
        for batch_shape, m, dtype, observe_noise in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, n=n, **tkwargs
            )
            if m == 1:
                train_Y = train_Y.squeeze(-1)
            train_Yvar = torch.full_like(train_Y, 0.01) if observe_noise else None

            cv_folds = gen_loo_cv_folds(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
            )
            with self.subTest(
                "gen_loo_cv_folds -- check shapes, device, and dtype",
                batch_shape=batch_shape,
                m=m,
                dtype=dtype,
                observe_noise=observe_noise,
            ):
                # check shapes
                expected_shape_train_X = batch_shape + torch.Size(
                    [n, n - 1, train_X.shape[-1]]
                )
                expected_shape_test_X = batch_shape + torch.Size(
                    [n, 1, train_X.shape[-1]]
                )
                self.assertEqual(cv_folds.train_X.shape, expected_shape_train_X)
                self.assertEqual(cv_folds.test_X.shape, expected_shape_test_X)

                expected_shape_train_Y = batch_shape + torch.Size([n, n - 1, m])
                expected_shape_test_Y = batch_shape + torch.Size([n, 1, m])

                self.assertEqual(cv_folds.train_Y.shape, expected_shape_train_Y)
                self.assertEqual(cv_folds.test_Y.shape, expected_shape_test_Y)
                if observe_noise:
                    self.assertEqual(cv_folds.train_Yvar.shape, expected_shape_train_Y)
                    self.assertEqual(cv_folds.test_Yvar.shape, expected_shape_test_Y)
                else:
                    self.assertIsNone(cv_folds.train_Yvar)
                    self.assertIsNone(cv_folds.test_Yvar)

                # check device and dtype
                self.assertEqual(cv_folds.train_X.device.type, self.device.type)
                self.assertIs(cv_folds.train_X.dtype, dtype)

            input_transform = Normalize(d=train_X.shape[-1])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                cv_results = batch_cross_validation(
                    model_cls=SingleTaskGP,
                    mll_cls=ExactMarginalLogLikelihood,
                    cv_folds=cv_folds,
                    fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
                    model_init_kwargs={
                        "input_transform": input_transform,
                    },
                )
            with self.subTest(
                "batch_cross_validation",
                batch_shape=batch_shape,
                m=m,
                dtype=dtype,
                observe_noise=observe_noise,
            ):
                expected_shape = batch_shape + torch.Size([n, 1, m])
                self.assertEqual(cv_results.posterior.mean.shape, expected_shape)
                self.assertEqual(cv_results.observed_Y.shape, expected_shape)
                if observe_noise:
                    self.assertEqual(cv_results.observed_Yvar.shape, expected_shape)
                else:
                    self.assertIsNone(cv_results.observed_Yvar)

                # check device and dtype
                self.assertEqual(
                    cv_results.posterior.mean.device.type, self.device.type
                )
                self.assertIs(cv_results.posterior.mean.dtype, dtype)

    def test_mtgp(self):
        train_X, train_Y = get_random_data(
            batch_shape=torch.Size(), m=1, n=3, device=self.device
        )
        cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y)
        with self.assertRaisesRegex(
            UnsupportedError, "Multi-task GPs are not currently supported."
        ):
            batch_cross_validation(
                model_cls=MultiTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                fit_args={"optimizer_kwargs": {"options": {"maxiter": 1}}},
            )
