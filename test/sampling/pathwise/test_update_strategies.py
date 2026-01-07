#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import chain
from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    gaussian_update,
    GeneralizedLinearPath,
    KernelEvaluationMap,
)
from botorch.sampling.pathwise.utils import get_train_inputs, get_train_targets
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.models import ExactGP
from linear_operator.operators import ZeroLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Size
from torch.nn.functional import pad


class TestPathwiseUpdates(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.models = defaultdict(list)

        seed = 0
        for kernel in (
            RBFKernel(ard_num_dims=2),
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2, batch_shape=Size([2]))),
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

    def test_gaussian_updates(self):
        for seed, model in enumerate(chain.from_iterable(self.models.values())):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._test_gaussian_updates(model)

    def _test_gaussian_updates(self, model):
        sample_shape = torch.Size([3])

        # Extract exact conditions and precompute covariances
        if isinstance(model, SingleTaskVariationalGP):
            Z = model.model.variational_strategy.inducing_points
            X = (
                Z
                if model.input_transform is None
                else model.input_transform.untransform(Z)
            )
            U = torch.randn(len(Z), device=Z.device, dtype=Z.dtype)
            Kuu = Kmm = model.model.covar_module(Z)
            noise_values = None
        else:
            (X,) = get_train_inputs(model, transformed=False)
            (Z,) = get_train_inputs(model, transformed=True)
            U = get_train_targets(model, transformed=True)
            Kmm = model.forward(X if model.training else Z).lazy_covariance_matrix
            Kuu = Kmm + model.likelihood.noise_covar(shape=Z.shape[:-1])
            noise_values = torch.randn(
                *sample_shape, *U.shape, device=U.device, dtype=U.dtype
            )

        # Disable sampling of noise variables `e` used to obtain `y = f + e`
        with delattr_ctx(model, "outcome_transform"), patch.object(
            torch,
            "randn_like",
            return_value=noise_values,
        ):
            prior_paths = draw_kernel_feature_paths(model, sample_shape=sample_shape)
            sample_values = prior_paths(X)
            update_paths = gaussian_update(
                model=model,
                sample_values=sample_values,
                target_values=U,
            )

        # Test initialization
        self.assertIsInstance(update_paths, GeneralizedLinearPath)
        self.assertIsInstance(update_paths.feature_map, KernelEvaluationMap)
        self.assertTrue(update_paths.feature_map.points.equal(Z))
        self.assertIs(
            update_paths.feature_map.input_transform,
            getattr(model, "input_transform", None),
        )

        # Compare with manually computed update weights `Cov(y, y)^{-1} (y - f - e)`
        Luu = psd_safe_cholesky(Kuu.to_dense())
        errors = U - sample_values
        if noise_values is not None:
            errors -= (
                model.likelihood.noise_covar(shape=Z.shape[:-1]).cholesky()
                @ noise_values.unsqueeze(-1)
            ).squeeze(-1)
        weight = torch.cholesky_solve(errors.unsqueeze(-1), Luu).squeeze(-1)
        self.assertTrue(weight.allclose(update_paths.weight))

        # Compare with manually computed update values at test locations
        Z2 = torch.rand(16, Z.shape[-1], device=self.device, dtype=Z.dtype)
        X2 = (
            model.input_transform.untransform(Z2)
            if hasattr(model, "input_transform")
            else Z2
        )
        features = update_paths.feature_map(X2)
        expected_updates = (features @ update_paths.weight.unsqueeze(-1)).squeeze(-1)
        actual_updates = update_paths(X2)
        self.assertTrue(actual_updates.allclose(expected_updates))

        # Test passing `noise_covariance`
        m = Z.shape[-2]
        update_paths = gaussian_update(
            model=model,
            sample_values=sample_values,
            target_values=U,
            noise_covariance=ZeroLinearOperator(m, m, dtype=X.dtype),
        )
        Lmm = psd_safe_cholesky(Kmm.to_dense())
        errors = U - sample_values
        weight = torch.cholesky_solve(errors.unsqueeze(-1), Lmm).squeeze(-1)
        self.assertTrue(weight.allclose(update_paths.weight))

        if isinstance(model, SingleTaskVariationalGP):
            # Test passing non-zero `noise_covariance``
            with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                with self.assertRaisesRegex(NotImplementedError, "not yet supported"):
                    gaussian_update(
                        model=model,
                        sample_values=sample_values,
                        noise_covariance="foo",
                    )
        else:
            # Test exact models with non-Gaussian likelihoods
            with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                with self.assertRaises(NotImplementedError):
                    gaussian_update(model=model, sample_values=sample_values)

            with self.subTest("Exact models with `None` target_values"):
                assert isinstance(model, ExactGP)
                torch.manual_seed(0)
                path_none_target_values = gaussian_update(
                    model=model,
                    sample_values=sample_values,
                )
                torch.manual_seed(0)
                path_with_target_values = gaussian_update(
                    model=model,
                    sample_values=sample_values,
                    target_values=get_train_targets(model, transformed=True),
                )
                self.assertAllClose(
                    path_none_target_values.weight, path_with_target_values.weight
                )
