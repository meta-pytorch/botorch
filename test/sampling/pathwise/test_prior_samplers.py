#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import product
from unittest.mock import MagicMock, patch

import torch
from botorch.models import ModelListGP, SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms.input import FilterFeatures, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    GeneralizedLinearPath,
    PathList,
)
from botorch.sampling.pathwise.utils import get_train_inputs
from botorch.utils.test_helpers import get_sample_moments, standardize_moments
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from torch import Size
from torch.nn.functional import pad
from torch.quasirandom import SobolEngine


class TestPriorSamplers(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.models = defaultdict(list)
        self.num_features = 1024

        seed = 0
        for kernel in (
            MaternKernel(nu=2.5, ard_num_dims=2, batch_shape=Size([])),
            ScaleKernel(RBFKernel(ard_num_dims=2, batch_shape=Size([2]))),
        ):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                tkwargs = {"device": self.device, "dtype": torch.float64}

                base = kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel
                base.lengthscale = 0.1 + 0.3 * torch.rand_like(base.lengthscale)
                kernel.to(**tkwargs)

                uppers = 1 + 9 * torch.rand(base.lengthscale.shape[-1], **tkwargs)
                bounds = pad(uppers.unsqueeze(0), (0, 0, 1, 0))

                X = uppers * torch.rand(4, base.lengthscale.shape[-1], **tkwargs)
                Y = 10 * kernel(X).cholesky() @ torch.randn(4, 1, **tkwargs)
                if kernel.batch_shape:
                    Y = Y.squeeze(-1).transpose(0, 1)  # n x m

                input_transform = Normalize(d=X.shape[-1], bounds=bounds)
                outcome_transform = Standardize(m=Y.shape[-1])

                # SingleTaskGP w/ inferred noise in eval mode
                self.models["inferred"].append(
                    SingleTaskGP(
                        train_X=X,
                        train_Y=Y,
                        covar_module=deepcopy(kernel),
                        input_transform=deepcopy(input_transform),
                        outcome_transform=deepcopy(outcome_transform),
                    )
                    .to(**tkwargs)
                    .eval()
                )

                # SingleTaskGP w/ observed noise in train mode
                self.models["observed"].append(
                    SingleTaskGP(
                        train_X=X,
                        train_Y=Y,
                        train_Yvar=0.01 * torch.rand_like(Y),
                        covar_module=kernel,
                        input_transform=input_transform,
                        outcome_transform=outcome_transform,
                    ).to(**tkwargs)
                )

                # SingleTaskVariationalGP in train mode
                # When batched, uses a multitask format which break the tests below
                if not kernel.batch_shape:
                    self.models["variational"].append(
                        SingleTaskVariationalGP(
                            train_X=X,
                            train_Y=Y,
                            covar_module=kernel,
                            input_transform=input_transform,
                            outcome_transform=outcome_transform,
                        ).to(**tkwargs)
                    )

            seed += 1

    def test_draw_kernel_feature_paths(self):
        for seed, models in enumerate(self.models.values()):
            for model, sample_shape in product(models, [Size([1024]), Size([2, 512])]):
                with torch.random.fork_rng():
                    torch.random.manual_seed(seed)
                    paths = draw_kernel_feature_paths(
                        model=model,
                        sample_shape=sample_shape,
                        num_features=self.num_features,
                    )
                    self.assertIsInstance(paths, GeneralizedLinearPath)
                    self._test_draw_kernel_feature_paths(model, paths, sample_shape)

        with self.subTest("test_model_list"):
            model_list = ModelListGP(
                self.models["inferred"][0], self.models["observed"][0]
            )
            path_list = draw_kernel_feature_paths(
                model=model_list,
                sample_shape=sample_shape,
                num_features=self.num_features,
            )
            (train_X,) = get_train_inputs(model_list.models[0], transformed=False)
            X = torch.zeros(
                4, train_X.shape[-1], dtype=train_X.dtype, device=self.device
            )
            sample_list = path_list(X)
            self.assertIsInstance(path_list, PathList)
            self.assertIsInstance(sample_list, list)
            self.assertEqual(len(sample_list), len(path_list.paths))

        with self.subTest("test_initialization"):
            model = self.models["inferred"][0]
            sample_shape = torch.Size([16])
            expected_weight_shape = (
                sample_shape + model.covar_module.batch_shape + (self.num_features,)
            )
            weight_generator = MagicMock(
                side_effect=lambda _: torch.rand(expected_weight_shape)
            )
            draw_kernel_feature_paths(
                model=model,
                sample_shape=sample_shape,
                num_features=self.num_features,
                weight_generator=weight_generator,
            )
            weight_generator.assert_called_once_with(expected_weight_shape)

    def test_draw_kernel_feature_paths_input_transforms(self):
        tkwargs = {"device": self.device, "dtype": torch.float64}
        X = 10 * torch.rand(8, 3, **tkwargs)
        Y = X.sum(dim=-1, keepdim=True)
        with self.subTest("dimension_changing_input_transform"):
            indices = torch.tensor([0, 2], device=self.device)
            model = SingleTaskGP(
                X, Y, input_transform=FilterFeatures(feature_indices=indices)
            ).eval()
            paths = draw_kernel_feature_paths(
                model=model, sample_shape=Size([4]), num_features=self.num_features
            )
            self.assertEqual(paths(X).shape, Size([4, 8]))

        with self.subTest("learned_input_transform_in_train_mode"):
            model = SingleTaskGP(X, Y, input_transform=Normalize(d=3))
            self.assertTrue(model.training)
            coefficient = model.input_transform.coefficient.clone()
            paths = draw_kernel_feature_paths(
                model=model, sample_shape=Size([4]), num_features=self.num_features
            )
            # Evaluating the paths at test inputs must not re-learn the bounds.
            samples = paths(X[:2] / 10)
            self.assertAllClose(model.input_transform.coefficient, coefficient)
            self.assertTrue(model.input_transform.training)
            model.eval()
            self.assertAllClose(paths(X[:2] / 10), samples)

        with self.subTest("num_features_above_sobol_maxdim"):
            with patch.object(SobolEngine, "MAXDIM", 16):
                paths = draw_kernel_feature_paths(
                    model=model, sample_shape=Size([4]), num_features=32
                )
            self.assertEqual(paths.weight.shape, Size([4, 32]))

    def _test_draw_kernel_feature_paths(self, model, paths, sample_shape, atol=3):
        (train_X,) = get_train_inputs(model, transformed=False)
        X = torch.rand(16, train_X.shape[-1], dtype=train_X.dtype, device=self.device)

        # Evaluate sample paths
        samples = paths(X)
        batch_shape = (
            model.model.covar_module.batch_shape
            if isinstance(model, SingleTaskVariationalGP)
            else model.covar_module.batch_shape
        )
        self.assertEqual(samples.shape, sample_shape + batch_shape + X.shape[-2:-1])

        # Calculate sample statistics
        sample_moments = get_sample_moments(samples, sample_shape)
        if hasattr(model, "outcome_transform"):
            # Do this instead of untransforming exact moments
            sample_moments = standardize_moments(
                model.outcome_transform, *sample_moments
            )

        # Compute prior distribution
        prior = model.forward(X if model.training else model.input_transform(X))
        exact_moments = (prior.loc, prior.covariance_matrix)

        # Compare moments
        tol = atol * (paths.weight.shape[-1] ** -0.5 + sample_shape.numel() ** -0.5)
        for exact, estimate in zip(exact_moments, sample_moments):
            self.assertTrue(exact.allclose(estimate, atol=tol, rtol=0))
