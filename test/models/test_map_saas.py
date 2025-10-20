#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import pickle
from functools import partial
from itertools import product
from typing import Any
from unittest import mock

import torch
from botorch.exceptions import UnsupportedError
from botorch.fit import (
    fit_gpytorch_mll,
    get_fitted_map_saas_ensemble,
    get_fitted_map_saas_model,
)
from botorch.models import SingleTaskGP
from botorch.models.map_saas import (
    add_saas_prior,
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    get_additive_map_saas_covar_module,
    get_gaussian_likelihood_with_gamma_prior,
    get_mean_module_with_normal_prior,
)
from botorch.models.transforms.input import (
    AppendFeatures,
    FilterFeatures,
    Normalize,
    NumericToCategoricalEncoding,
)
from botorch.models.transforms.outcome import Standardize
from botorch.optim.utils import get_parameters_and_bounds, sample_all_priors
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.test_utils.mock import mock_optimize
from botorch.utils.constraints import LogTransformedInterval
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import Interval
from gpytorch.kernels import AdditiveKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, HalfCauchyPrior, NormalPrior
from torch import Tensor
from torch.nn.functional import one_hot


class TestMapSaas(BotorchTestCase):
    def _get_data(self, **tkwargs) -> tuple[Tensor, Tensor, Tensor]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = 1 + 2 * torch.rand(10, 3, **tkwargs)
            train_Y = torch.sin(train_X[:, :1])
            test_X = 1 + 2 * torch.rand(5, 3, **tkwargs)
        return train_X, train_Y, test_X

    def _get_data_hardcoded(self, **tkwargs) -> tuple[Tensor, Tensor, Tensor]:
        """This is equal to _get_data on CPU with a seed of 0, and is hard-coded here
        to ensure that the results are identical on GPUs, which have different RNGs.
        """
        train_X = torch.tensor(
            [
                [2.9401, 2.4156, 1.9188],
                [2.8415, 2.2900, 2.5823],
                [1.3572, 1.7022, 2.1627],
                [1.5765, 1.9057, 1.3536],
                [1.7105, 2.2438, 1.9637],
                [1.8816, 1.8146, 1.4109],
                [2.3301, 2.5697, 1.4207],
                [2.3535, 1.2195, 2.0475],
                [1.4520, 2.1165, 2.1751],
                [2.3639, 2.4908, 1.4553],
            ],
            **tkwargs,
        )
        train_Y = torch.tensor(
            [
                [0.2001],
                [0.2956],
                [0.9773],
                [1.0000],
                [0.9903],
                [0.9521],
                [0.7253],
                [0.7090],
                [0.9930],
                [0.7016],
            ],
            **tkwargs,
        )
        test_X = torch.tensor(
            [
                [2.6198, 2.2612, 1.3918],
                [1.3055, 1.9630, 2.8350],
                [2.1441, 1.9617, 2.5921],
                [1.7359, 1.5444, 1.0829],
                [2.4974, 1.6835, 2.0765],
            ],
            **tkwargs,
        )
        return train_X, train_Y, test_X

    def test_add_saas_prior(self) -> None:
        for dtype, infer_tau in itertools.product(
            [torch.float, torch.double], [True, False]
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, _, _ = self._get_data(**tkwargs)
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]).to(
                **tkwargs
            )
            tau = None if infer_tau else 0.1234
            add_saas_prior(base_kernel=base_kernel, tau=tau)
            pickle.loads(pickle.dumps(base_kernel))  # pickle and unpickle should work
            if not infer_tau:  # Make sure there is no raw_tau parameter
                self.assertFalse(hasattr(base_kernel, "raw_tau"))
            else:
                self.assertTrue(hasattr(base_kernel, "raw_tau"))
                self.assertIsInstance(base_kernel.tau_prior, HalfCauchyPrior)
                self.assertAlmostEqual(base_kernel.tau_prior.scale.item(), 0.1)
                # Make sure there is a constraint on tau
                self.assertIsInstance(base_kernel.raw_tau_constraint, Interval)
                self.assertEqual(base_kernel.raw_tau_constraint.lower_bound, 1e-3)
                self.assertEqual(base_kernel.raw_tau_constraint.upper_bound, 10.0)
            self.assertIsInstance(base_kernel.inv_lengthscale_prior, HalfCauchyPrior)
            self.assertAlmostEqual(base_kernel.inv_lengthscale_prior.scale.item(), 1.0)
            # Make sure we have specified a constraint on the lengthscale
            self.assertIsInstance(base_kernel.raw_lengthscale_constraint, Interval)
            self.assertEqual(base_kernel.raw_lengthscale_constraint.lower_bound, 1e-2)
            self.assertEqual(base_kernel.raw_lengthscale_constraint.upper_bound, 1e4)
            # Lengthscale closures
            _inv_lengthscale_prior = base_kernel._priors["inv_lengthscale_prior"]
            self.assertIsInstance(_inv_lengthscale_prior[0], HalfCauchyPrior)
            self.assertAlmostEqual(_inv_lengthscale_prior[0].scale.item(), 1.0)
            base_kernel.lengthscale = 0.5678
            true_value = (0.1 if infer_tau else tau) / (0.5678**2)  # tau / ell^2
            self.assertAllClose(
                _inv_lengthscale_prior[1](base_kernel),
                true_value * torch.ones(1, train_X.shape[-1], **tkwargs),
            )
            _inv_lengthscale_prior[2](base_kernel, torch.tensor(5.55, **tkwargs))
            true_value = math.sqrt((0.1 if infer_tau else tau) / 5.55)
            self.assertAllClose(
                base_kernel.lengthscale,
                true_value * torch.ones(1, train_X.shape[-1], **tkwargs),
            )
            if infer_tau:  # Global shrinkage closures
                _tau_prior = base_kernel._priors["tau_prior"]
                self.assertIsInstance(_tau_prior[0], HalfCauchyPrior)
                self.assertAlmostEqual(_tau_prior[0].scale.item(), 0.1)
                _tau_prior[2](base_kernel, torch.tensor(1.234, **tkwargs))
                self.assertAlmostEqual(
                    _tau_prior[1](base_kernel).item(), 1.234, delta=1e-6
                )

            with self.assertRaisesRegex(UnsupportedError, "must have lengthscale"):
                add_saas_prior(base_kernel=ScaleKernel(base_kernel))

            kernel_with_prior = MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ).to(**tkwargs)
            with self.assertRaisesRegex(
                UnsupportedError, "must not specify a lengthscale prior"
            ):
                add_saas_prior(base_kernel=kernel_with_prior)

    def test_get_saas_model(self) -> None:
        for infer_tau, infer_noise in itertools.product([True, False], [True, False]):
            tkwargs = {"device": self.device, "dtype": torch.double}
            train_X, train_Y, test_X = self._get_data_hardcoded(**tkwargs)

            lb, ub = train_X.min(dim=0).values, train_X.max(dim=0).values
            mu, sigma = train_Y.mean(), train_Y.std()
            d = train_X.shape[-1]
            tau = None if infer_tau else 0.1234
            train_Yvar = (
                None
                if infer_noise
                else 0.1 * torch.arange(len(train_X), **tkwargs).unsqueeze(-1)
            )
            # Fit with transforms
            model = get_fitted_map_saas_model(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                tau=tau,
            )
            posterior = model.posterior(test_X)
            pred_mean, pred_var = posterior.mean, posterior.variance
            # Make sure the lengthscales are reasonable
            self.assertTrue(
                (model.covar_module.base_kernel.lengthscale[:, 1:] > 1e2).all()
            )
            self.assertTrue(model.covar_module.base_kernel.lengthscale[:, 0] < 10)
            # Test with fitting without transforms and make sure predictions match
            model2 = get_fitted_map_saas_model(
                train_X=(train_X - lb) / (ub - lb),
                train_Y=(train_Y - mu) / sigma,
                train_Yvar=(
                    train_Yvar / (sigma**2) if train_Yvar is not None else train_Yvar
                ),
                tau=tau,
            )
            posterior2 = model2.posterior((test_X - lb) / (ub - lb))
            pred_mean2 = mu + sigma * posterior2.mean
            pred_var2 = (sigma**2) * posterior2.variance
            self.assertAllClose(pred_mean, pred_mean2)
            self.assertAllClose(pred_var, pred_var2)

            # testing optimizer_options: short optimization run with maxiter = 3
            fit_gpytorch_mll_mock = mock.Mock(wraps=fit_gpytorch_mll)
            with mock.patch(
                "botorch.fit.fit_gpytorch_mll",
                new=fit_gpytorch_mll_mock,
            ):
                maxiter = 3
                model_short = get_fitted_map_saas_model(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    input_transform=Normalize(d=d),
                    outcome_transform=Standardize(m=1),
                    tau=tau,
                    optimizer_kwargs={"options": {"maxiter": maxiter}},
                )
                kwargs = fit_gpytorch_mll_mock.call_args.kwargs
                # fit_gpytorch_mll has "option" kwarg, not "optimizer_options"
                self.assertEqual(
                    kwargs["optimizer_kwargs"]["options"]["maxiter"], maxiter
                )

            # Compute marginal likelihood after short run.
            # Putting the MLL in train model to silence warnings.
            mll_short = ExactMarginalLogLikelihood(
                model=model_short, likelihood=model_short.likelihood
            ).train()
            train_inputs = mll_short.model.train_inputs
            train_targets = mll_short.model.train_targets
            output = mll_short.model(*train_inputs)
            loss_short = -mll_short(output, train_targets).item()

            # Make sure the correct bounds are extracted
            _, bounds = get_parameters_and_bounds(mll_short)
            if infer_noise:
                self.assertAllClose(
                    bounds["likelihood.noise_covar.raw_noise"][0].item(), math.log(1e-4)
                )
                self.assertAllClose(
                    bounds["likelihood.noise_covar.raw_noise"][1].item(), math.log(1)
                )
            self.assertAllClose(
                bounds["model.mean_module.raw_constant"][0].item(), -10.0
            )
            self.assertAllClose(
                bounds["model.mean_module.raw_constant"][1].item(), 10.0
            )
            self.assertAllClose(
                bounds["model.covar_module.raw_outputscale"][0].item(), math.log(1e-2)
            )
            self.assertAllClose(
                bounds["model.covar_module.raw_outputscale"][1].item(), math.log(1e4)
            )
            self.assertAllClose(
                bounds["model.covar_module.base_kernel.raw_lengthscale"][0].item(),
                math.log(1e-2),
            )
            self.assertAllClose(
                bounds["model.covar_module.base_kernel.raw_lengthscale"][1].item(),
                math.log(1e4),
            )
            if infer_tau:
                self.assertAllClose(
                    bounds["model.covar_module.base_kernel.raw_tau"][0].item(),
                    math.log(1e-3),
                )
                self.assertAllClose(
                    bounds["model.covar_module.base_kernel.raw_tau"][1].item(),
                    math.log(10.0),
                )

            # compute marginal likelihood after standard run
            mll = ExactMarginalLogLikelihood(
                model=model, likelihood=model.likelihood
            ).train()
            # reusing train_inputs and train_targets, since the transforms are the same
            loss = -mll(model(*train_inputs), train_targets).item()
            # longer running optimization should have smaller loss than the shorter one
            self.assertTrue(loss < loss_short)

    def test_get_saas_ensemble(self) -> None:
        train_X, train_Y, _ = self._get_data_hardcoded(device=self.device)
        with self.assertWarnsRegex(
            DeprecationWarning, "EnsembleMapSaasSingleTaskGP"
        ), mock.patch("botorch.fit.fit_gpytorch_mll") as mock_fit:
            model = get_fitted_map_saas_ensemble(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=train_X.shape[-1]),
                outcome_transform=Standardize(m=1, batch_shape=torch.Size([4])),
                optimizer_kwargs={"options": {"maxiter": 3}},
            )
        self.assertEqual(
            mock_fit.call_args.kwargs["optimizer_kwargs"], {"options": {"maxiter": 3}}
        )
        self.assertIsInstance(model, EnsembleMapSaasSingleTaskGP)

    def test_input_transform_in_train(self) -> None:
        train_X, train_Y, test_X = self._get_data()
        # Use a transform that only works in eval mode.
        append_tf = AppendFeatures(feature_set=torch.randn_like(train_X)).eval()
        with mock.patch.object(SingleTaskGP, "_validate_tensor_args") as mock_validate:
            get_fitted_map_saas_model(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=append_tf,
                outcome_transform=Standardize(m=1),
            )
        call_X = mock_validate.call_args[1]["X"]
        self.assertTrue(torch.equal(call_X, train_X))

    def test_filterfeatures_input_transform(self) -> None:
        train_X, train_Y, test_X = self._get_data()
        idxs_to_filter = [0, 2]
        filter_feature_transforn = FilterFeatures(
            feature_indices=torch.tensor(idxs_to_filter)
        )
        # Use a transform that only works in eval mode.
        model = get_fitted_map_saas_model(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=filter_feature_transforn,
            outcome_transform=Standardize(m=1),
        )
        self.assertTrue(model.train_inputs[0].shape[-1] == len(idxs_to_filter))
        self.assertAllClose(model.train_inputs[0], train_X[:, idxs_to_filter])
        self.assertTrue(
            model.covar_module.base_kernel.lengthscale.shape[-1] == len(idxs_to_filter)
        )

    def test_batch_model_fitting(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = 1 + 2 * torch.rand(10, 3, **tkwargs)
        train_Y = torch.cat(
            (torch.sin(train_X[:, :1]), torch.cos(train_X[:, :1])), dim=-1
        )

        for tau in [0.1, None]:
            batch_model = get_fitted_map_saas_model(
                train_X=train_X, train_Y=train_Y, tau=tau
            )
            model_1 = get_fitted_map_saas_model(
                train_X=train_X, train_Y=train_Y[:, :1], tau=tau
            )
            model_2 = get_fitted_map_saas_model(
                train_X=train_X, train_Y=train_Y[:, 1:], tau=tau
            )
            # Check lengthscales
            self.assertEqual(
                batch_model.covar_module.base_kernel.lengthscale.shape,
                torch.Size([2, 1, 3]),
            )
            self.assertEqual(
                model_1.covar_module.base_kernel.lengthscale.shape,
                torch.Size([1, 3]),
            )
            self.assertEqual(
                model_2.covar_module.base_kernel.lengthscale.shape,
                torch.Size([1, 3]),
            )
            self.assertAllClose(
                batch_model.covar_module.base_kernel.lengthscale[0, :],
                model_1.covar_module.base_kernel.lengthscale,
                atol=1e-3,
            )
            self.assertAllClose(
                batch_model.covar_module.base_kernel.lengthscale[1, :],
                model_2.covar_module.base_kernel.lengthscale,
                atol=1e-3,
            )
            # Check the outputscale
            self.assertEqual(
                batch_model.covar_module.outputscale.shape, torch.Size([2])
            )
            self.assertEqual(model_1.covar_module.outputscale.shape, torch.Size([]))
            self.assertEqual(model_2.covar_module.outputscale.shape, torch.Size([]))
            self.assertAllClose(
                batch_model.covar_module.outputscale,
                torch.stack(
                    (
                        model_1.covar_module.outputscale,
                        model_2.covar_module.outputscale,
                    )
                ),
                atol=1e-3,
            )
            # Check the mean
            self.assertEqual(batch_model.mean_module.constant.shape, torch.Size([2]))
            self.assertEqual(model_1.mean_module.constant.shape, torch.Size([]))
            self.assertEqual(model_2.mean_module.constant.shape, torch.Size([]))
            self.assertAllClose(
                batch_model.mean_module.constant,
                torch.stack(
                    (model_1.mean_module.constant, model_2.mean_module.constant)
                ),
                atol=1e-3,
            )
            # Check noise
            self.assertEqual(batch_model.likelihood.noise.shape, torch.Size([2, 1]))
            self.assertEqual(model_1.likelihood.noise.shape, torch.Size([1]))
            self.assertEqual(model_2.likelihood.noise.shape, torch.Size([1]))
            self.assertAllClose(
                batch_model.likelihood.noise,
                torch.stack((model_1.likelihood.noise, model_2.likelihood.noise)),
                atol=1e-3,
            )
            # Check tau
            if tau is None:
                self.assertEqual(
                    batch_model.covar_module.base_kernel.raw_tau.shape, torch.Size([2])
                )
                self.assertEqual(
                    model_1.covar_module.base_kernel.raw_tau.shape, torch.Size([])
                )
                self.assertEqual(
                    model_2.covar_module.base_kernel.raw_tau.shape, torch.Size([])
                )
                self.assertAllClose(
                    batch_model.covar_module.base_kernel.raw_tau,
                    torch.stack(
                        (
                            model_1.covar_module.base_kernel.raw_tau,
                            model_2.covar_module.base_kernel.raw_tau,
                        )
                    ),
                    atol=1e-3,
                )

    @mock_optimize
    def test_emsemble_map_saas(self) -> None:
        train_X, train_Y, test_X = self._get_data(device=self.device)
        d = train_X.shape[-1]
        num_taus = 8
        for with_options in (False, True):
            if with_options:
                extra_inputs = {
                    "train_Yvar": 0.1 * torch.rand_like(train_Y),
                    "taus": torch.rand(num_taus).to(train_X),
                    "input_transform": Normalize(d=d),
                    "outcome_transform": None,
                }
            else:
                extra_inputs = {}
            model = EnsembleMapSaasSingleTaskGP(
                train_X=train_X, train_Y=train_Y, num_taus=num_taus, **extra_inputs
            )
            sample_all_priors(model)  # Checks that the prior is configured correctly.
            mll = ExactMarginalLogLikelihood(model=model, likelihood=model.likelihood)
            fit_gpytorch_mll(mll)
            self.assertIsInstance(model.covar_module, ScaleKernel)
            self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
            self.assertEqual(
                model.covar_module.base_kernel.lengthscale.shape,
                torch.Size([num_taus, 1, d]),
            )
            self.assertEqual(model.batch_shape, torch.Size([num_taus]))
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GaussianMixturePosterior)
            if with_options:
                self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
                self.assertIsInstance(model.input_transform, Normalize)
                self.assertFalse(hasattr(model, "outcome_transform"))
            else:
                self.assertIsInstance(model.likelihood, GaussianLikelihood)
                self.assertIsInstance(model.outcome_transform, Standardize)
                self.assertFalse(hasattr(model, "input_transform"))

    def test_ensemble_map_saas_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected taus to be of shape"):
            EnsembleMapSaasSingleTaskGP(
                train_X=torch.rand(5, 3),
                train_Y=torch.rand(5, 1),
                num_taus=3,
                taus=torch.rand(2),
            )
        with self.assertRaisesRegex(UnsupportedError, "only supports single-output"):
            EnsembleMapSaasSingleTaskGP(
                train_X=torch.rand(5, 3), train_Y=torch.rand(5, 2)
            )
        with self.assertRaisesRegex(UnsupportedError, "only supports 2D inputs"):
            EnsembleMapSaasSingleTaskGP(
                train_X=torch.rand(2, 5, 3), train_Y=torch.rand(2, 5, 1)
            )

    def test_ensemble_map_saas_construct_inputs(self) -> None:
        """Test the construct_inputs class method for EnsembleMapSaasSingleTaskGP."""

        train_X, train_Y, _ = self._get_data()
        training_data = SupervisedDataset(
            X=train_X, Y=train_Y, feature_names=["x1", "x2", "x3"], outcome_names=["y"]
        )

        # Test with default num_taus
        inputs_default = EnsembleMapSaasSingleTaskGP.construct_inputs(
            training_data=training_data
        )
        self.assertIn("num_taus", inputs_default)
        self.assertEqual(inputs_default["num_taus"], 4)
        self.assertIn("train_X", inputs_default)
        self.assertIn("train_Y", inputs_default)
        self.assertAllClose(inputs_default["train_X"], train_X)
        self.assertAllClose(inputs_default["train_Y"], train_Y)

        # Test with custom num_taus
        custom_num_taus = 6
        inputs_custom = EnsembleMapSaasSingleTaskGP.construct_inputs(
            training_data=training_data, num_taus=custom_num_taus
        )
        self.assertIn("num_taus", inputs_custom)
        self.assertEqual(inputs_custom["num_taus"], custom_num_taus)
        self.assertIn("train_X", inputs_custom)
        self.assertIn("train_Y", inputs_custom)
        self.assertAllClose(inputs_custom["train_X"], train_X)
        self.assertAllClose(inputs_custom["train_Y"], train_Y)

        # Test with train_Yvar in the dataset
        train_Yvar = 0.1 * torch.rand_like(train_Y)
        training_data_with_yvar = SupervisedDataset(
            X=train_X,
            Y=train_Y,
            Yvar=train_Yvar,
            feature_names=["x1", "x2", "x3"],
            outcome_names=["y"],
        )
        inputs_with_yvar = EnsembleMapSaasSingleTaskGP.construct_inputs(
            training_data=training_data_with_yvar, num_taus=3
        )
        self.assertIn("train_Yvar", inputs_with_yvar)
        self.assertAllClose(inputs_with_yvar["train_Yvar"], train_Yvar)
        self.assertEqual(inputs_with_yvar["num_taus"], 3)
        model_with_yvar = EnsembleMapSaasSingleTaskGP(**inputs_with_yvar)
        self.assertIsInstance(model_with_yvar, EnsembleMapSaasSingleTaskGP)
        self.assertEqual(model_with_yvar.batch_shape, torch.Size([3]))


class TestAdditiveMapSaasSingleTaskGP(BotorchTestCase):
    def _get_data_and_model(
        self,
        infer_noise: bool,
        m: int = 1,
        batch_shape: list[int] | None = None,
        **tkwargs,
    ):
        batch_shape = batch_shape or []
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(batch_shape + [10, 4], **tkwargs)
            train_Y = (
                torch.sin(train_X)
                .sum(dim=-1, keepdim=True)
                .repeat(*[1] * (train_X.ndim - 1), m)
            )
            train_Yvar = (
                None if infer_noise else torch.rand(batch_shape + [10, m], **tkwargs)
            )
            model = AdditiveMapSaasSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
            )
        return train_X, train_Y, train_Yvar, model

    def test_input_transform_dimensions(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # Create data
            X = torch.rand(12, 2, **tkwargs) * 2
            Y = 1 - (X - 0.5).norm(dim=-1, keepdim=True)
            Y += 0.1 * torch.rand_like(Y)
            # Add a categorical feature
            new_col = torch.randint(0, 3, (X.shape[0], 1), **tkwargs)
            X = torch.cat([X, new_col], dim=1)

            input_transform = NumericToCategoricalEncoding(
                dim=3,
                categorical_features={2: 3},
                encoders={2: partial(one_hot, num_classes=3)},
            )

            model = AdditiveMapSaasSingleTaskGP(
                train_X=X,
                train_Y=Y,
                input_transform=input_transform,
            )
            self.assertEqual(model.covar_module.kernels[0].base_kernel.ard_num_dims, 5)

    def test_construct_mean_module(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        for batch_shape in [None, torch.Size([5])]:
            mean_module = get_mean_module_with_normal_prior(batch_shape=batch_shape).to(
                **tkwargs
            )
            self.assertIsInstance(mean_module, ConstantMean)
            self.assertIsInstance(mean_module.mean_prior, NormalPrior)
            self.assertEqual(mean_module.mean_prior.loc, torch.zeros(1, **tkwargs))
            self.assertEqual(mean_module.mean_prior.scale, torch.ones(1, **tkwargs))
            self.assertEqual(
                mean_module.raw_constant.shape, batch_shape or torch.Size()
            )
            self.assertEqual(mean_module.raw_constant_constraint.lower_bound, -10.0)
            self.assertEqual(mean_module.raw_constant_constraint.upper_bound, 10.0)

    def test_construct_likelihood(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        for batch_shape in [None, torch.Size([5])]:
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=batch_shape
            ).to(**tkwargs)
            self.assertIsInstance(likelihood, GaussianLikelihood)
            self.assertIsInstance(likelihood.noise_covar.noise_prior, GammaPrior)
            self.assertAllClose(
                likelihood.noise_covar.noise_prior.concentration,
                torch.tensor(0.9, **tkwargs),
            )
            self.assertAllClose(
                likelihood.noise_covar.noise_prior.rate, torch.tensor(10, **tkwargs)
            )
            self.assertEqual(
                likelihood.noise_covar.raw_noise.shape,
                torch.Size([1]) if batch_shape is None else torch.Size([5, 1]),
            )
            self.assertIsInstance(
                likelihood.noise_covar.raw_noise_constraint, LogTransformedInterval
            )
            self.assertAllClose(
                likelihood.noise_covar.raw_noise_constraint.lower_bound.item(), 1e-4
            )
            self.assertAllClose(
                likelihood.noise_covar.raw_noise_constraint.upper_bound.item(), 1.0
            )

    def test_construct_covar_module(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        for batch_shape in [None, torch.Size([5])]:
            covar_module = get_additive_map_saas_covar_module(
                ard_num_dims=10, num_taus=4, batch_shape=batch_shape
            ).to(**tkwargs)
            self.assertIsInstance(covar_module, AdditiveKernel)
            self.assertEqual(len(covar_module.kernels), 4)
            for kernel in covar_module.kernels:
                self.assertIsInstance(kernel, ScaleKernel)
                self.assertIsInstance(kernel.base_kernel, MaternKernel)
                expected_shape = (
                    torch.Size([1, 10])
                    if batch_shape is None
                    else torch.Size([5, 1, 10])
                )
                self.assertEqual(kernel.base_kernel.lengthscale.shape, expected_shape)
                # Check for a SAAS prior
                self.assertFalse(hasattr(kernel.base_kernel, "raw_tau"))
                self.assertIsInstance(
                    kernel.base_kernel.inv_lengthscale_prior, HalfCauchyPrior
                )
                self.assertAlmostEqual(
                    kernel.base_kernel.inv_lengthscale_prior.scale.item(), 1.0
                )

    def test_fit_model(self) -> None:
        for infer_noise, m, batch_shape in (
            (True, 1, None),
            (False, 1, [5]),
            (True, 2, None),
            (False, 2, [3]),
            (True, 3, [5]),
        ):
            tkwargs = {"device": self.device, "dtype": torch.double}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                infer_noise=infer_noise, m=m, batch_shape=batch_shape, **tkwargs
            )
            expected_batch_shape = (
                torch.Size(batch_shape) if batch_shape else torch.Size()
            )
            expected_aug_batch_shape = expected_batch_shape + torch.Size(
                [m] if m > 1 else []
            )

            # Test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, AdditiveKernel)
            self.assertIsInstance(
                model.likelihood,
                GaussianLikelihood if infer_noise else FixedNoiseGaussianLikelihood,
            )
            expected_Y, expected_Yvar = model.outcome_transform(
                Y=train_Y, Yvar=train_Yvar
            )
            expected_X, expected_Y, expected_Yvar = model._transform_tensor_args(
                X=train_X, Y=expected_Y, Yvar=expected_Yvar
            )
            self.assertAllClose(expected_X, model.train_inputs[0])
            self.assertAllClose(expected_Y, model.train_targets)
            if not infer_noise:
                self.assertAllClose(model.likelihood.noise_covar.noise, expected_Yvar)

            # Fit a model
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 5}})
            self.assertEqual(model.batch_shape, expected_batch_shape)
            self.assertEqual(model._aug_batch_shape, expected_aug_batch_shape)

            # Predict on some test points
            test_X = torch.rand(13, train_X.shape[-1], **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            # Mean/variance
            expected_shape = (*expected_batch_shape, 13, m)
            mean, var = posterior.mean, posterior.variance
            self.assertEqual(mean.shape, torch.Size(expected_shape))
            self.assertEqual(var.shape, torch.Size(expected_shape))

            # Test AdditiveMapSaasSingleTaskGP constructor
            input_transform = Normalize(d=train_X.shape[-1])
            outcome_transform = Standardize(m=m, batch_shape=expected_batch_shape)
            with mock.patch.object(
                SingleTaskGP, "__init__", wraps=SingleTaskGP.__init__
            ) as mock_init:
                model = AdditiveMapSaasSingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform,
                    num_taus=3,
                )
                self.assertEqual(
                    input_transform, mock_init.call_args[1]["input_transform"]
                )
                self.assertEqual(
                    outcome_transform, mock_init.call_args[1]["outcome_transform"]
                )
                self.assertIsInstance(
                    mock_init.call_args[1]["covar_module"], AdditiveKernel
                )
                self.assertEqual(3, len(mock_init.call_args[1]["covar_module"].kernels))

    def test_sample_from_prior_additive_map_saas(self) -> None:
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        for batch, m in product((torch.Size([]), torch.Size([3])), (1, 2)):
            train_X = torch.rand(*batch, 10, 4, **tkwargs)
            train_Y = torch.rand(*batch, 10, m, **tkwargs)
            for _ in range(10):
                model = AdditiveMapSaasSingleTaskGP(train_X=train_X, train_Y=train_Y)
                sample_all_priors(model)
