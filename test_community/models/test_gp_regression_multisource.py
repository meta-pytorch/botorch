#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import warnings

import torch

from botorch import fit_gpytorch_mll
from botorch.exceptions import InputDataError, OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.test_helpers import get_pvar_expected
from botorch.utils.testing import BotorchTestCase, get_random_data
from botorch_community.models.gp_regression_multisource import (
    _get_reliable_observations,
    get_random_x_for_agp,
    SingleTaskAugmentedGP,
)
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior


def _get_random_data_with_source(batch_shape, n, d, n_source, q=1, **tkwargs):
    rep_shape = batch_shape + torch.Size([1, 1])
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    bounds[-1, -1] = n_source - 1
    train_x = (
        get_random_x_for_agp(n=n, bounds=bounds, q=q).repeat(rep_shape).to(**tkwargs)
    )
    train_y = torch.sin(train_x[..., :1] * (2 * math.pi)).to(**tkwargs)
    train_y = train_y + 0.2 * torch.randn(n, 1, **tkwargs).repeat(rep_shape)
    return train_x, train_y


class TestAugmentedSingleTaskGP(BotorchTestCase):
    def _get_model_and_data(
        self,
        batch_shape,
        n,
        d,
        n_source,
        train_Yvar=False,
        outcome_transform=None,
        input_transform=None,
        extra_model_kwargs=None,
        **tkwargs,
    ):
        extra_model_kwargs = extra_model_kwargs or {}
        train_X, train_Y = _get_random_data_with_source(
            batch_shape, n, d, n_source, **tkwargs
        )
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": torch.full_like(train_Y, 0.01) if train_Yvar else None,
            "outcome_transform": outcome_transform,
            "input_transform": input_transform,
            "covar_module": get_matern_kernel_with_gamma_prior(
                ard_num_dims=train_X.shape[-1] - 1
            ),
            "likelihood": (
                None if train_Yvar else get_gaussian_likelihood_with_gamma_prior()
            ),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizationWarning)
            model = SingleTaskAugmentedGP(**model_kwargs, **extra_model_kwargs)
        return model, model_kwargs

    def test_data_init(self):
        d = 5
        for n, n_source in itertools.product((0, 1, 5, 10, 100), (1, 2, 5)):
            bounds = torch.stack([torch.zeros(d), torch.ones(d)])
            bounds[-1, -1] = n_source - 1
            if n == 0:
                self.assertRaises(InputDataError, get_random_x_for_agp, n, bounds, 1)
            else:
                x = get_random_x_for_agp(n, bounds, q=1)
                self.assertIn(n_source - 1, x[..., -1])
                self.assertEqual(x.shape, (n, d))

    def test_init_error(self):
        n, d = 10, 5
        for n_source, batch_shape in itertools.product(
            (1, 2, 3), (torch.Size([]), torch.Size([2]))
        ):
            # Test initialization
            train_X, train_Y = _get_random_data_with_source(
                batch_shape=batch_shape, n=n, d=d, n_source=n_source
            )
            if n_source == 1:
                self.assertRaises(
                    InputDataError, SingleTaskAugmentedGP, train_X, train_Y
                )
                continue
            else:
                model = SingleTaskAugmentedGP(train_X, train_Y)
                self.assertIsInstance(model, SingleTaskAugmentedGP)

            # Test initialization with m = 0
            self.assertRaises(
                InputDataError, SingleTaskAugmentedGP, train_X, train_Y, m=0
            )

    def test_get_reliable_observation(self):
        x = torch.linspace(0, 5, 15).reshape(-1, 1)
        true_y = torch.sin(x).reshape(-1, 1)
        y = torch.cos(x).reshape(-1, 1)

        model0 = SingleTaskGP(
            x,
            true_y,
            covar_module=get_matern_kernel_with_gamma_prior(x.shape[-1]),
            likelihood=get_gaussian_likelihood_with_gamma_prior(),
            outcome_transform=None,
        )
        model1 = SingleTaskGP(
            x,
            y,
            covar_module=get_matern_kernel_with_gamma_prior(x.shape[-1]),
            likelihood=get_gaussian_likelihood_with_gamma_prior(),
            outcome_transform=None,
        )

        res = _get_reliable_observations(model0, model1, x)
        true_res = torch.cat([torch.arange(0, 5, 1), torch.arange(9, 15, 1)]).int()
        self.assertListEqual(res.tolist(), true_res.tolist())

    def test_gp(self):
        d = 5
        bounds = torch.stack((torch.full((d - 1,), -1), torch.ones(d - 1)))
        for batch_shape, dtype, use_octf, use_intf, train_Yvar in itertools.product(
            (torch.Size(), torch.Size([2])),
            (torch.float, torch.double),
            (False, True),
            (False, True),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1, batch_shape=torch.Size()) if use_octf else None
            intf = (
                Normalize(d=d - 1, bounds=bounds.to(**tkwargs), transform_on_train=True)
                if use_intf
                else None
            )
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                n=10,
                d=d,
                n_source=5,
                train_Yvar=train_Yvar,
                outcome_transform=octf,
                input_transform=intf,
                **tkwargs,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            matern_kernel = model.covar_module.base_kernel
            self.assertIsInstance(matern_kernel, MaternKernel)
            self.assertIsInstance(matern_kernel.lengthscale_prior, GammaPrior)
            if use_octf:
                self.assertIsInstance(model.outcome_transform, Standardize)
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)
                # permute output dim
                train_X, train_Y, _ = model._transform_tensor_args(
                    X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                )
                # check that the train inputs have been transformed and set
                # on the model for each source
                for s in train_X[..., -1].unique():
                    self.assertTrue(
                        torch.equal(
                            model.models[int(s)].train_inputs[0],
                            intf(train_X[train_X[..., -1] == s][..., :-1]),
                        )
                    )

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([3, d - 1]), **tkwargs)
            expected_shape = batch_shape + torch.Size([3, 1])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # test adding observation noise
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                pp_tf = model.posterior(X, observation_noise=True)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                self.assertAllClose(posterior_pred.variance, expected_var)
            else:
                pvar = posterior_pred.variance
                pvar_exp = get_pvar_expected(posterior, model, X, 1)
                self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

            # Tensor valued observation noise.
            obs_noise = torch.rand((*X.shape[:-1], 1), **tkwargs)
            posterior_pred = model.posterior(X, observation_noise=obs_noise)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            if use_octf:
                _, obs_noise = model.outcome_transform.untransform(obs_noise, obs_noise)
            self.assertAllClose(posterior_pred.variance, posterior.variance + obs_noise)

    def test_condition_on_observations(self):
        for dtype, use_octf in itertools.product(
            (torch.float, torch.double),
            (False, True),
        ):
            d = 5
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1) if use_octf else None
            model, model_kwargs = self._get_model_and_data(
                batch_shape=torch.Size([]),
                n=10,
                d=d,
                n_source=5,
                outcome_transform=octf,
                **tkwargs,
            )
            d = d - 1
            # evaluate model
            model.posterior(torch.rand(torch.Size([4, d]), **tkwargs))
            # test condition_on_observations
            fant_shape = torch.Size([2])
            # fantasize at different input points
            X_fant, Y_fant = get_random_data(
                batch_shape=fant_shape, m=1, d=d, n=3, **tkwargs
            )
            c_kwargs = (
                {"noise": torch.full_like(Y_fant, 0.01)}
                if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                else {}
            )
            cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
            # fantasize at same input points (check proper broadcasting)
            c_kwargs_same_inputs = (
                {"noise": torch.full_like(Y_fant[0], 0.01)}
                if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                else {}
            )
            cm_same_inputs = model.condition_on_observations(
                X_fant[0], Y_fant, **c_kwargs_same_inputs
            )

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, d, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(torch.Size([]) + torch.Size([4, d]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(fant_shape + torch.Size([]) + torch.Size([4, d]), **tkwargs),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape,
                    fant_shape + torch.Size([4, 1]),
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + torch.Size([4, 1]),
                )

    def test_fixed_noise_likelihood(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                n=10,
                d=5,
                n_source=5,
                train_Yvar=True,
                **tkwargs,
            )
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            likelihood_noise = model.likelihood.noise.contiguous().view(-1)
            train_Y_var = model_kwargs["train_Yvar"].contiguous().view(-1)
            self.assertTrue(
                torch.equal(
                    likelihood_noise,
                    train_Y_var[: len(likelihood_noise)],
                )
            )

    def test_fantasized_noise(self):
        for batch_shape, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (torch.float, torch.double),
            (False, True),
        ):
            d = 5
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1, batch_shape=torch.Size()) if use_octf else None
            model, _ = self._get_model_and_data(
                batch_shape=batch_shape,
                n=10,
                d=d,
                n_source=5,
                train_Yvar=True,
                outcome_transform=octf,
                **tkwargs,
            )
            # fantasize
            X_f = torch.rand(
                torch.Size(batch_shape + torch.Size([4, d - 1])), **tkwargs
            )
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            noise = model.likelihood.noise.unsqueeze(-1)
            avg_noise = noise.mean(dim=-2, keepdim=True)
            fm_noise = fm.likelihood.noise.unsqueeze(-1)

            self.assertTrue((fm_noise[..., -4:, :] == avg_noise).all())
            # pass tensor of noise
            # noise is assumed to be outcome transformed
            # batch shape x n' x m
            obs_noise = torch.full(
                X_f.shape[:-1] + torch.Size([1]), 0.1, dtype=dtype, device=self.device
            )
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=obs_noise)
            fm_noise = fm.likelihood.noise.unsqueeze(-1)
            self.assertTrue((fm_noise[..., -4:, :] == obs_noise).all())
            # test batch shape x 1 x m
            obs_noise = torch.full(
                X_f.shape[:-2] + torch.Size([1, 1]),
                0.1,
                dtype=dtype,
                device=self.device,
            )
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=obs_noise)
            fm_noise = fm.likelihood.noise.unsqueeze(-1)
            self.assertTrue(
                (
                    fm_noise[..., -4:, :]
                    == obs_noise.expand(X_f.shape[:-1] + torch.Size([1]))
                ).all()
            )
