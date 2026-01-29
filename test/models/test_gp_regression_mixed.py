#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.test_helpers import get_pvar_expected
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch.kernels.kernel import AdditiveKernel, ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestMixedSingleTaskGP(BotorchTestCase):
    observed_noise = False

    def test_gp(self):
        d = 3
        bounds = torch.tensor([[-1.0] * d, [1.0] * d])
        for batch_shape, m, ncat, dtype, observed_noise, use_octf in (
            (torch.Size(), 1, 0, torch.float, False, False),
            (torch.Size(), 2, 1, torch.double, True, True),
            (torch.Size([2]), 2, 3, torch.double, False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            # The model by default uses a ``Standardize`` outcome transform, so
            # to test without that transform we need to explicitly pass in ``None``.
            outcome_transform_kwargs = {} if use_octf else {"outcome_transform": None}

            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, d=d, **tkwargs
            )
            cat_dims = list(range(ncat))
            ord_dims = sorted(set(range(d)) - set(cat_dims))
            # test correct indices
            if (ncat < 3) and (ncat > 0):
                MixedSingleTaskGP(
                    train_X,
                    train_Y,
                    cat_dims=cat_dims,
                    input_transform=Normalize(
                        d=d,
                        bounds=bounds.to(**tkwargs),
                        transform_on_train=True,
                        indices=ord_dims,
                    ),
                )

            if len(cat_dims) == 0:
                with self.assertRaises(ValueError):
                    MixedSingleTaskGP(train_X, train_Y, cat_dims=cat_dims)
                continue

            train_Yvar = torch.full_like(train_Y, 0.1) if observed_noise else None
            model = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                train_Yvar=train_Yvar,
                **outcome_transform_kwargs,
            )
            self.assertEqual(model._ignore_X_dims_scaling_check, cat_dims)
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            if ncat < 3:
                self.assertIsInstance(model.covar_module, AdditiveKernel)
                sum_kernel, prod_kernel = model.covar_module.kernels
                self.assertIsInstance(sum_kernel, ScaleKernel)
                self.assertIsInstance(sum_kernel.base_kernel, AdditiveKernel)
                self.assertIsInstance(prod_kernel, ScaleKernel)
                self.assertIsInstance(prod_kernel.base_kernel, ProductKernel)
                sum_cont_kernel, sum_cat_kernel = sum_kernel.base_kernel.kernels
                prod_cont_kernel, prod_cat_kernel = prod_kernel.base_kernel.kernels
                self.assertIsInstance(sum_cont_kernel, RBFKernel)
                self.assertIsInstance(sum_cat_kernel, ScaleKernel)
                self.assertIsInstance(sum_cat_kernel.base_kernel, CategoricalKernel)
                self.assertIsInstance(prod_cont_kernel, RBFKernel)
                self.assertIsInstance(prod_cat_kernel, CategoricalKernel)
            else:
                self.assertIsInstance(model.covar_module, ScaleKernel)
                self.assertIsInstance(model.covar_module.base_kernel, CategoricalKernel)
            if observed_noise:
                self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            else:
                self.assertIsInstance(model.likelihood, GaussianLikelihood)

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([4, d]), **tkwargs)
            expected_shape = batch_shape + torch.Size([4, m])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # test adding observation noise
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            pvar = posterior_pred.variance
            pvar_exp = get_pvar_expected(posterior=posterior, model=model, X=X, m=m)
            self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

            # test batch evaluation
            X = torch.rand(2, *batch_shape, 3, d, **tkwargs)
            expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, m])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            # test adding observation noise in batch mode
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            pvar = posterior_pred.variance
            pvar_exp = get_pvar_expected(posterior=posterior, model=model, X=X, m=m)
            self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

    def test_condition_on_observations__(self):
        d = 3
        for batch_shape, m, ncat, dtype, observed_noise in (
            (torch.Size(), 2, 1, torch.float, True),
            (torch.Size([2]), 1, 2, torch.double, False),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, d=d, **tkwargs
            )
            cat_dims = list(range(ncat))
            train_Yvar = torch.full_like(train_Y, 0.1) if observed_noise else None
            model = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                train_Yvar=train_Yvar,
            )

            # evaluate model
            model.posterior(torch.rand(torch.Size([4, d]), **tkwargs))
            # test condition_on_observations
            fant_shape = torch.Size([2])

            # fantasize at different input points
            X_fant, Y_fant = get_random_data(
                fant_shape + batch_shape, m=m, d=d, n=3, **tkwargs
            )
            additional_kwargs = (
                {"noise": torch.full_like(Y_fant, 0.1)} if observed_noise else {}
            )
            cm = model.condition_on_observations(X_fant, Y_fant, **additional_kwargs)
            # fantasize at same input points (check proper broadcasting)
            additional_kwargs = (
                {"noise": torch.full_like(Y_fant[0], 0.1)} if observed_noise else {}
            )
            cm_same_inputs = model.condition_on_observations(
                X_fant[0], Y_fant, **additional_kwargs
            )

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, d, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(batch_shape + torch.Size([4, d]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(fant_shape + batch_shape + torch.Size([4, d]), **tkwargs),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape, fant_shape + batch_shape + torch.Size([4, m])
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + batch_shape + torch.Size([4, m]),
                )

                # check that fantasies of batched model are correct
                if len(batch_shape) > 0 and test_X.dim() == 2:
                    model_kwargs_non_batch = {
                        "train_X": train_X[0],
                        "train_Y": train_Y[0],
                        "cat_dims": cat_dims,
                    }
                    if observed_noise:
                        model_kwargs_non_batch["train_Yvar"] = train_Yvar[0]
                    model_non_batch = type(model)(**model_kwargs_non_batch)
                    non_batch_shapes = {
                        key: val.shape
                        for key, val in model_non_batch.state_dict().items()
                    }
                    state_dict_non_batch = {}
                    for key, val in model.state_dict().items():
                        if key in non_batch_shapes:
                            expected_shape = non_batch_shapes[key]
                            if val.ndim > len(expected_shape):
                                state_dict_non_batch[key] = val[0]
                            else:
                                state_dict_non_batch[key] = val
                        else:
                            state_dict_non_batch[key] = val
                    model_non_batch.load_state_dict(state_dict_non_batch)
                    model_non_batch.eval()
                    model_non_batch.likelihood.eval()
                    model_non_batch.posterior(torch.rand(torch.Size([4, d]), **tkwargs))
                    additional_kwargs = (
                        {"noise": torch.full_like(Y_fant, 0.1)}
                        if observed_noise
                        else {}
                    )
                    cm_non_batch = model_non_batch.condition_on_observations(
                        X_fant[0][0], Y_fant[:, 0, :], **additional_kwargs
                    )
                    non_batch_posterior = cm_non_batch.posterior(test_X)
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.mean[:, 0, ...],
                            non_batch_posterior.mean,
                            atol=1e-3,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.distribution.covariance_matrix[
                                :, 0, :, :
                            ],
                            non_batch_posterior.distribution.covariance_matrix,
                            atol=1e-3,
                        )
                    )

    def test_fantasize(self):
        d = 3
        for batch_shape, m, ncat, dtype, observed_noise in (
            (torch.Size(), 2, 1, torch.float, True),
            (torch.Size([2]), 1, 2, torch.double, False),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, d=d, **tkwargs
            )
            train_Yvar = torch.full_like(train_Y, 0.1) if observed_noise else None
            cat_dims = list(range(ncat))
            model = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                train_Yvar=train_Yvar,
            )

            # fantasize
            X_f = torch.rand(torch.Size(batch_shape + torch.Size([4, d])), **tkwargs)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            self.assertIsInstance(fm, model.__class__)

    def test_subset_model(self):
        d, m = 3, 2
        for batch_shape, ncat, dtype in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(
                batch_shape=batch_shape, m=m, d=d, **tkwargs
            )
            cat_dims = list(range(ncat))
            model = MixedSingleTaskGP(train_X, train_Y, cat_dims=cat_dims)
            with self.assertRaises(NotImplementedError):
                model.subset_output([0])
            # TODO: Support subsetting MixedSingleTaskGP models
            # X = torch.rand(torch.Size(batch_shape + torch.Size([3, d])), **tkwargs)
            # p = model.posterior(X)
            # p_sub = subset_model.posterior(X)
            # self.assertTrue(
            #     torch.allclose(p_sub.mean, p.mean[..., [0]], atol=1e-4, rtol=1e-4)
            # )
            # self.assertTrue(
            #     torch.allclose(
            #         p_sub.variance, p.variance[..., [0]], atol=1e-4, rtol=1e-4
            #     )
            # )

    def test_construct_inputs(self):
        d = 3
        for batch_shape, ncat, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X, Y = get_random_data(batch_shape=batch_shape, m=1, d=d, **tkwargs)
            cat_dims = list(range(ncat))
            training_data = SupervisedDataset(
                X,
                Y,
                feature_names=[f"x{i}" for i in range(d)],
                outcome_names=["y"],
            )
            model_kwargs = MixedSingleTaskGP.construct_inputs(
                training_data, categorical_features=cat_dims
            )
            self.assertTrue(X.equal(model_kwargs["train_X"]))
            self.assertTrue(Y.equal(model_kwargs["train_Y"]))
            self.assertEqual(model_kwargs["cat_dims"], cat_dims)
            self.assertIsNone(model_kwargs["likelihood"])

        # With train_Yvar.
        training_data = SupervisedDataset(
            X,
            Y,
            Yvar=Y,
            feature_names=[f"x{i}" for i in range(d)],
            outcome_names=["y"],
        )
        model_kwargs = MixedSingleTaskGP.construct_inputs(
            training_data, categorical_features=cat_dims
        )
        self.assertTrue(X.equal(model_kwargs["train_X"]))
        self.assertTrue(Y.equal(model_kwargs["train_Y"]))
        self.assertTrue(Y.equal(model_kwargs["train_Yvar"]))
