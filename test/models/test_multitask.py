#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import warnings
from typing import Any

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import OptimizationWarning
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import (
    get_task_value_remapping,
    KroneckerMultiTaskGP,
    MultiTaskGP,
)
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.test_helpers import gen_multi_task_dataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import IndexKernel, MaternKernel, MultitaskKernel, RBFKernel
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    HadamardGaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.means.linear_mean import LinearMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior, SmoothedBoxPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.settings import max_cholesky_size, max_root_decomposition_size


def _gen_model_and_data(
    fixed_noise: bool,
    task_feature: int = 0,
    output_tasks: list[int] | None = None,
    task_values: list[int] | None = None,
    skip_task_features_in_datasets: bool = False,
    input_transform: InputTransform | None = None,
    outcome_transform: OutcomeTransform | None = None,
    **tkwargs,
):
    datasets, (train_X, train_Y, train_Yvar) = gen_multi_task_dataset(
        yvar=0.05 if fixed_noise else None,
        task_values=task_values,
        skip_task_features_in_datasets=skip_task_features_in_datasets,
        **tkwargs,
    )
    model = MultiTaskGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        task_feature=task_feature,
        output_tasks=output_tasks,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )
    return model.to(**tkwargs), datasets, (train_X, train_Y, train_Yvar)


def _gen_model_single_output(**tkwargs):
    _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
    model = MultiTaskGP(train_X, train_Y, task_feature=0, output_tasks=[1])
    return model.to(**tkwargs)


def _gen_fixed_prior_model(**tkwargs):
    _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
    sd_prior = GammaPrior(2.0, 0.15)
    sd_prior._event_shape = torch.Size([2])
    model = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=0,
        task_covar_prior=LKJCovariancePrior(2, 0.6, sd_prior),
    )
    return model.to(**tkwargs)


def _gen_given_covar_module_model(**tkwargs):
    _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
    model = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=0,
        covar_module=RBFKernel(lengthscale_prior=LogNormalPrior(0.0, 1.0)),
    )
    return model.to(**tkwargs)


def _gen_random_kronecker_mt_data(batch_shape=None, **tkwargs):
    batch_shape = batch_shape or torch.Size()
    train_X = (
        torch.linspace(0, 0.95, 10, **tkwargs).unsqueeze(-1).expand(*batch_shape, 10, 1)
    )
    train_X = train_X + 0.05 * torch.rand(*batch_shape, 10, 2, **tkwargs)
    train_y1 = (
        torch.sin(train_X[..., 0] * (2 * math.pi))
        + torch.randn_like(train_X[..., 0]) * 0.2
    )
    train_y2 = (
        torch.cos(train_X[..., 1] * (2 * math.pi))
        + torch.randn_like(train_X[..., 0]) * 0.2
    )
    train_Y = torch.stack([train_y1, train_y2], dim=-1)
    return train_X, train_Y


def _gen_kronecker_model_and_data(model_kwargs=None, batch_shape=None, **tkwargs):
    model_kwargs = model_kwargs or {}
    train_X, train_Y = _gen_random_kronecker_mt_data(batch_shape=batch_shape, **tkwargs)
    model = KroneckerMultiTaskGP(train_X, train_Y, **model_kwargs)
    return model.to(**tkwargs), train_X, train_Y


class TestMultiTaskGP(BotorchTestCase):
    def test_MultiTaskGP(self) -> None:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        for dtype, use_intf, use_octf, task_values, fixed_noise in zip(
            (torch.float, torch.double),
            (False, True),
            (False, True),
            (None, [0, 2]),
            (False, True),
            strict=True,
        ):
            tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=1) if use_octf else None

            intf = (
                Normalize(d=2, bounds=bounds.to(**tkwargs), transform_on_train=True)
                if use_intf
                else None
            )
            model, datasets, (train_X, train_Y, train_Yvar) = _gen_model_and_data(
                fixed_noise=fixed_noise,
                task_values=task_values,
                input_transform=intf,
                outcome_transform=octf,
                **tkwargs,
            )
            self.assertIsInstance(model, MultiTaskGP)
            self.assertEqual(model.num_outputs, 2)
            if fixed_noise:
                self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            else:
                self.assertIsInstance(model.likelihood, HadamardGaussianLikelihood)
                self.assertEqual(model.likelihood.noise.shape, torch.Size([2]))
                self.assertEqual(model.likelihood.task_feature_index, 0)
            data_covar_module, task_covar_module = model.covar_module.kernels
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(data_covar_module, RBFKernel)
            self.assertIsInstance(data_covar_module.lengthscale_prior, LogNormalPrior)
            self.assertIsInstance(task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(task_covar_module.covar_factor.shape[-1], model._rank)
            if task_values is None:
                self.assertEqual(model._task_mapper, None)
                self.assertEqual(model._expected_task_values, {0, 1})
            else:
                self.assertEqual(model._task_mapper.shape, torch.Size([3]))
                self.assertEqual(model._expected_task_values, set(task_values))
            if use_octf:
                self.assertIsInstance(model.outcome_transform, Standardize)
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test posterior
            batch_test_x = torch.rand(3, 2, 1, **tkwargs)
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(batch_test_x)
            posterior_y = model.posterior(batch_test_x, observation_noise=True)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultitaskMultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([3, 2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([3, 2, 2]))
            if fixed_noise:
                noise_covar = torch.diag(
                    torch.tensor([0.05, 0.1], **tkwargs).repeat_interleave(2)
                ).expand(3, 4, 4)
            else:
                noise_covar = torch.diag(
                    model.likelihood.noise_covar.noise.repeat_interleave(2)
                ).expand(3, 4, 4)
            expected_y_covar = posterior_f.covariance_matrix + noise_covar
            self.assertTrue(
                torch.allclose(posterior_y.covariance_matrix, expected_y_covar)
            )

            # check that training data has input transform applied
            # check that the train inputs have been transformed and set on the model
            if use_intf:
                self.assertTrue(
                    model.train_inputs[0].equal(model.input_transform(train_X))
                )

            # test that posterior w/ observation noise raises appropriate error
            with self.assertRaisesRegex(
                NotImplementedError,
                "Passing a tensor of observations is not supported by MultiTaskGP.",
            ):
                model.posterior(test_x, observation_noise=torch.rand(2, **tkwargs))

            # test posterior w/ single output index
            posterior_f = model.posterior(
                test_x, output_indices=[0], observation_noise=True
            )
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 1]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 1]))

            # test posterior w/ bad output index
            with self.assertRaises(ValueError):
                model.posterior(test_x, output_indices=[3])

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultitaskMultivariateNormal)

            # test posterior with X including the task features
            posterior_expected = model.posterior(test_x, output_indices=[0])
            test_x = torch.cat([torch.zeros_like(test_x), test_x], dim=-1)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultivariateNormal)
            self.assertAllClose(posterior_f.mean, posterior_expected.mean)
            self.assertAllClose(
                posterior_f.covariance_matrix, posterior_expected.covariance_matrix
            )

            # test task features in X and output_indices is not None.
            with self.assertRaisesRegex(ValueError, "`output_indices` must be None"):
                model.posterior(test_x, output_indices=[0, 1])

            # test invalid task feature in X.
            invalid_x = test_x.clone()
            invalid_x[0, 0, 0] = 3
            if task_values is None:
                msg = "task features in `X`"
            else:
                msg = (
                    r"Received invalid raw task values. Expected raw value to be in"
                    r" \{0, 2\}, but got unexpected task values:"
                    r" \{3\}."
                )
            with self.assertRaisesRegex(ValueError, msg):
                model.posterior(invalid_x)

            # test that unsupported batch shape MTGPs throw correct error
            with self.assertRaises(ValueError):
                MultiTaskGP(torch.rand(2, 2, 2), torch.rand(2, 2, 1), 0)

            # test that bad feature index throws correct error
            _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
            with self.assertRaises(ValueError):
                MultiTaskGP(train_X, train_Y, 2)

            # test that bad output task throws correct error
            with self.assertRaises(RuntimeError):
                MultiTaskGP(train_X, train_Y, 0, output_tasks=[2])

            # test outcome transform
            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                p_utf = model.posterior(test_x)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p_utf).variance
                self.assertAllClose(posterior_f.variance, expected_var)
            msg = "Subsetting outputs is not supported by `MultiTaskGPyTorchModel`."
            with self.assertRaisesRegex(UnsupportedError, msg):
                model.subset_output(idcs=[0])

            if task_values is not None:
                test_x = torch.rand(2, 1, **tkwargs)
                test_x_task = torch.zeros_like(test_x)
                test_x_task[1, 0] = 2.0
                test_x = torch.cat([test_x_task, test_x], dim=-1)
                expected_task_mapper = torch.tensor(
                    [0.0, 0.0, 1.0], dtype=dtype, device=self.device
                )
                self.assertTrue(torch.equal(model._task_mapper, expected_task_mapper))
                # Test making predictions for task without observations.
                # These should be equivalent to predictions for the output task.
                test_X_unobserved = torch.rand(1, 2, **tkwargs)
                test_X_unobserved[0, 1] = 1.0
                with torch.no_grad():
                    posterior_unobserved = model.posterior(X=test_X_unobserved)
                test_X_observed = torch.rand(1, 2, **tkwargs)
                test_X_observed[0, 1] = 0.0
                with torch.no_grad():
                    posterior_observed = model.posterior(X=test_X_unobserved)
                self.assertTrue(
                    torch.allclose(posterior_unobserved.mean, posterior_observed.mean)
                )
                self.assertTrue(
                    torch.allclose(
                        posterior_unobserved.variance, posterior_observed.variance
                    )
                )

                # test split inputs
                _, task_idcs, _ = model._split_inputs(test_x)
                self.assertTrue(
                    torch.equal(
                        task_idcs,
                        torch.tensor([[0.0], [1.0]], dtype=dtype, device=self.device),
                    )
                )
            else:
                self.assertIsNone(model._task_mapper)

    def test_MultiTaskGP_single_output(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
            model = _gen_model_single_output(**tkwargs)
            data_covar_module, task_covar_module = model.covar_module.kernels
            self.assertIsInstance(model, MultiTaskGP)
            self.assertEqual(model.num_outputs, 1)
            self.assertIsInstance(model.likelihood, HadamardGaussianLikelihood)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(data_covar_module, RBFKernel)
            self.assertIsInstance(data_covar_module.lengthscale_prior, LogNormalPrior)
            self.assertIsInstance(task_covar_module, IndexKernel)
            self.assertEqual(model._rank, 2)
            self.assertEqual(task_covar_module.covar_factor.shape[-1], model._rank)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test posterior
            test_x = torch.rand(2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultivariateNormal)

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 1, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultivariateNormal)

            # test posterior transform
            post_tf = ScalarizedPosteriorTransform(weights=torch.ones(1, **tkwargs))
            posterior_f_tf = model.posterior(test_x, posterior_transform=post_tf)
            self.assertTrue(torch.equal(posterior_f.mean, posterior_f_tf.mean))

    def test_MultiTaskGP_fixed_prior(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _gen_fixed_prior_model(**tkwargs)
            _, task_covar_module = model.covar_module.kernels
            self.assertIsInstance(model, MultiTaskGP)
            self.assertIsInstance(task_covar_module, IndexKernel)
            self.assertIsInstance(
                task_covar_module.IndexKernelPrior, LKJCovariancePrior
            )

    def test_MultiTaskGP_given_covar_module(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            model = _gen_given_covar_module_model(**tkwargs)
            self.assertIsInstance(model, MultiTaskGP)
            data_covar_module, task_covar_module = model.covar_module.kernels
            self.assertIsInstance(task_covar_module, IndexKernel)
            self.assertIsInstance(data_covar_module, RBFKernel)
            self.assertIsInstance(data_covar_module.lengthscale_prior, LogNormalPrior)
            self.assertAlmostEqual(data_covar_module.lengthscale_prior.loc, 0.0)
            self.assertAlmostEqual(data_covar_module.lengthscale_prior.scale, 1.0)

    def test_custom_mean_and_likelihood(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
        mean_module = LinearMean(input_size=train_X.shape[-1])
        likelihood = GaussianLikelihood(noise_prior=LogNormalPrior(0, 1))
        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=0,
            mean_module=mean_module,
            likelihood=likelihood,
        )
        self.assertIs(model.mean_module, mean_module)
        self.assertIs(model.likelihood, likelihood)

    def test_multiple_output_indices(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        for fixed_noise in (True, False):
            model, datasets, (train_X, train_Y, train_Yvar) = _gen_model_and_data(
                task_values=[0, 1, 2], fixed_noise=fixed_noise, **tkwargs
            )
            test_X = torch.rand(2, 1, **tkwargs)
            for observation_noise in (True, False):
                posterior = model.posterior(
                    test_X, output_indices=[0, 2], observation_noise=observation_noise
                )
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertIsInstance(posterior.distribution, MultivariateNormal)
                self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
                self.assertEqual(posterior.variance.shape, torch.Size([2, 2]))

    def test_all_tasks_input(self) -> None:
        _, (train_X, train_Y, _) = gen_multi_task_dataset(
            dtype=torch.double, device=self.device
        )
        # Invalid: does not contain all tasks.
        with self.assertRaisesRegex(
            UnsupportedError, "does not contain all the task features"
        ):
            MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=0, all_tasks=[0])
        # Contains extra tasks.
        model = MultiTaskGP(
            train_X=train_X, train_Y=train_Y, task_feature=0, all_tasks=[0, 1, 2, 3]
        )
        self.assertEqual(model.num_tasks, 2)
        # Check that IndexKernel knows of all tasks.
        self.assertEqual(model.covar_module.kernels[1].raw_var.shape[-1], 2)

    def test_MultiTaskGP_construct_inputs(self) -> None:
        for dtype, fixed_noise, skip_task_features_in_datasets in zip(
            (torch.float, torch.double), (True, False), (True, False), strict=True
        ):
            tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}
            task_feature = 0
            model, datasets, (train_X, train_Y, train_Yvar) = _gen_model_and_data(
                fixed_noise=fixed_noise,
                task_feature=task_feature,
                skip_task_features_in_datasets=skip_task_features_in_datasets,
                **tkwargs,
            )

            # Validate prior config.
            with self.assertRaisesRegex(
                ValueError, ".* only config for LKJ prior is supported"
            ):
                data_dict = model.construct_inputs(
                    datasets,
                    task_feature=task_feature,
                    prior_config={"use_LKJ_prior": False},
                )
            # Validate eta.
            with self.assertRaisesRegex(ValueError, "eta must be a real number"):
                data_dict = model.construct_inputs(
                    datasets,
                    task_feature=task_feature,
                    prior_config={"use_LKJ_prior": True, "eta": "not_number"},
                )
            # Test that presence of `prior` and `prior_config` kwargs at the
            # same time causes error.
            with self.assertRaisesRegex(ValueError, "Only one of"):
                data_dict = model.construct_inputs(
                    datasets,
                    task_feature=task_feature,
                    task_covar_prior=1,
                    prior_config={"use_LKJ_prior": True, "eta": "not_number"},
                )
            data_dict = model.construct_inputs(
                datasets,
                task_feature=task_feature,
                output_tasks=[0],
                prior_config={"use_LKJ_prior": True, "eta": 0.6},
            )
            self.assertEqual(data_dict["output_tasks"], [0])
            self.assertEqual(data_dict["task_feature"], task_feature)
            if skip_task_features_in_datasets:
                # In this case, the task feature is appended at the end.
                self.assertAllClose(data_dict["train_X"], train_X[..., [1, 0]])
                # all_tasks is inferred from data when task features are omitted.
                self.assertEqual(data_dict["all_tasks"], [0, 1])
            else:
                self.assertAllClose(data_dict["train_X"], train_X)
                # all_tasks is not inferred when task features are included.
                self.assertNotIn("all_tasks", data_dict)
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))
            if fixed_noise:
                self.assertAllClose(data_dict["train_Yvar"], train_Yvar)
            else:
                self.assertNotIn("train_Yvar", data_dict)
            self.assertIsInstance(data_dict["task_covar_prior"], LKJCovariancePrior)

    def test_validatation_of_task_values(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        _, (train_X, train_Y, _) = gen_multi_task_dataset(**tkwargs)
        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=0,
            output_tasks=[1],
            all_tasks=[0, 1, 2],
            validate_task_values=False,
        )
        self.assertTrue(
            torch.equal(model._task_mapper, torch.tensor([0, 1, 1], **tkwargs))
        )
        self.assertTrue(
            torch.equal(
                torch.tensor([1], **tkwargs),
                model._map_tasks(task_values=torch.tensor([2], **tkwargs)),
            )
        )
        model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=0,
            output_tasks=[1],
            all_tasks=[0, 1, 2],
            validate_task_values=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Received invalid raw task values. Expected raw value to be in"
            r" \{0, 1\}, but got unexpected task"
            r" values: \{2\}.",
        ):
            model._map_tasks(task_values=torch.tensor([2], **tkwargs))


class TestKroneckerMultiTaskGP(BotorchTestCase):
    def test_KroneckerMultiTaskGP_default(self) -> None:
        bounds = torch.tensor([[-1.0, 0.0], [1.0, 1.0]])

        for batch_shape, dtype, use_intf, use_octf in itertools.product(
            (torch.Size(),),  # torch.Size([3])), TODO: Fix and test batch mode
            (torch.float, torch.double),
            (False, True),
            (False, True),
        ):
            tkwargs: dict[str, Any] = {"device": self.device, "dtype": dtype}

            octf = Standardize(m=2) if use_octf else None

            intf = (
                Normalize(d=2, bounds=bounds.to(**tkwargs), transform_on_train=True)
                if use_intf
                else None
            )

            # initialization with default settings
            model, train_X, _ = _gen_kronecker_model_and_data(
                model_kwargs={"outcome_transform": octf, "input_transform": intf},
                batch_shape=batch_shape,
                **tkwargs,
            )
            self.assertIsInstance(model, KroneckerMultiTaskGP)
            self.assertEqual(model.num_outputs, 2)
            self.assertIsInstance(model.likelihood, MultitaskGaussianLikelihood)
            self.assertEqual(model.likelihood.rank, 0)
            self.assertIsInstance(model.mean_module, MultitaskMean)
            self.assertIsInstance(model.covar_module, MultitaskKernel)
            base_kernel = model.covar_module
            self.assertIsInstance(base_kernel.data_covar_module, RBFKernel)
            self.assertIsInstance(base_kernel.task_covar_module, IndexKernel)
            task_covar_prior = base_kernel.task_covar_module.IndexKernelPrior
            self.assertIsInstance(task_covar_prior, LKJCovariancePrior)
            self.assertEqual(task_covar_prior.correlation_prior.eta, 1.5)
            self.assertIsInstance(task_covar_prior.sd_prior, SmoothedBoxPrior)
            lengthscale_prior = base_kernel.data_covar_module.lengthscale_prior
            self.assertIsInstance(lengthscale_prior, LogNormalPrior)
            self.assertEqual(base_kernel.task_covar_module.covar_factor.shape[-1], 2)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test posterior
            test_x = torch.rand(2, 2, **tkwargs)
            posterior_f = model.posterior(test_x)
            if not use_octf:
                self.assertIsInstance(posterior_f, GPyTorchPosterior)
                self.assertIsInstance(
                    posterior_f.distribution, MultitaskMultivariateNormal
                )
            else:
                self.assertIsInstance(posterior_f, TransformedPosterior)
                self.assertIsInstance(
                    posterior_f._posterior.distribution, MultitaskMultivariateNormal
                )

            self.assertEqual(posterior_f.mean.shape, torch.Size([2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([2, 2]))

            if use_octf:
                # ensure un-transformation is applied
                tmp_tf = model.outcome_transform
                del model.outcome_transform
                p_tf = model.posterior(test_x)
                model.outcome_transform = tmp_tf
                expected_var = tmp_tf.untransform_posterior(p_tf).variance
                self.assertAllClose(posterior_f.variance, expected_var)
            else:
                # test observation noise
                # TODO: outcome transform + likelihood noise?
                posterior_noisy = model.posterior(test_x, observation_noise=True)
                self.assertTrue(
                    torch.allclose(
                        posterior_noisy.variance,
                        model.likelihood(posterior_f.distribution).variance,
                    )
                )

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 2, **tkwargs)
            posterior_f = model.posterior(test_x)
            if not use_octf:
                self.assertIsInstance(posterior_f, GPyTorchPosterior)
                self.assertIsInstance(
                    posterior_f.distribution, MultitaskMultivariateNormal
                )
            else:
                self.assertIsInstance(posterior_f, TransformedPosterior)
                self.assertIsInstance(
                    posterior_f._posterior.distribution, MultitaskMultivariateNormal
                )
            self.assertEqual(posterior_f.mean.shape, torch.Size([3, 2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([3, 2, 2]))

            # test that using a posterior transform throws error
            post_tf = ScalarizedPosteriorTransform(weights=torch.ones(2, **tkwargs))
            with self.assertRaises(NotImplementedError):
                model.posterior(test_x, posterior_transform=post_tf)

    def test_KroneckerMultiTaskGP_custom(self) -> None:
        for batch_shape, dtype in itertools.product(
            (torch.Size(),),  # torch.Size([3])), TODO: Fix and test batch mode
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            # initialization with custom settings
            likelihood = MultitaskGaussianLikelihood(
                num_tasks=2,
                rank=1,
                batch_shape=batch_shape,
            )
            data_covar_module = MaternKernel(
                nu=1.5,
                lengthscale_prior=GammaPrior(2.0, 4.0),
            )
            task_covar_prior = LKJCovariancePrior(
                n=2,
                eta=torch.tensor(0.5, **tkwargs),
                sd_prior=SmoothedBoxPrior(math.exp(-3), math.exp(2), 0.1),
            )
            model_kwargs = {
                "likelihood": likelihood,
                "data_covar_module": data_covar_module,
                "task_covar_prior": task_covar_prior,
                "rank": 1,
            }

            model, train_X, _ = _gen_kronecker_model_and_data(
                model_kwargs=model_kwargs, batch_shape=batch_shape, **tkwargs
            )
            self.assertIsInstance(model, KroneckerMultiTaskGP)
            self.assertEqual(model.num_outputs, 2)
            self.assertIsInstance(model.likelihood, MultitaskGaussianLikelihood)
            self.assertEqual(model.likelihood.rank, 1)
            self.assertIsInstance(model.mean_module, MultitaskMean)
            self.assertIsInstance(model.covar_module, MultitaskKernel)
            base_kernel = model.covar_module
            self.assertIsInstance(base_kernel.data_covar_module, MaternKernel)
            self.assertIsInstance(base_kernel.task_covar_module, IndexKernel)
            task_covar_prior = base_kernel.task_covar_module.IndexKernelPrior
            self.assertIsInstance(task_covar_prior, LKJCovariancePrior)
            self.assertEqual(task_covar_prior.correlation_prior.eta, 0.5)
            lengthscale_prior = base_kernel.data_covar_module.lengthscale_prior
            self.assertIsInstance(lengthscale_prior, GammaPrior)
            self.assertEqual(lengthscale_prior.concentration, 2.0)
            self.assertEqual(lengthscale_prior.rate, 4.0)
            self.assertEqual(base_kernel.task_covar_module.covar_factor.shape[-1], 1)

            # test model fitting
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                mll = fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test posterior
            max_cholesky_sizes = [1, 800]
            for max_cholesky in max_cholesky_sizes:
                model.train()
                test_x = torch.rand(2, 2, **tkwargs)
                # small root decomp to enforce zero padding
                with max_cholesky_size(max_cholesky), max_root_decomposition_size(3):
                    posterior_f = model.posterior(test_x)
                    self.assertIsInstance(posterior_f, GPyTorchPosterior)
                    self.assertIsInstance(
                        posterior_f.distribution, MultitaskMultivariateNormal
                    )
                    self.assertEqual(posterior_f.mean.shape, torch.Size([2, 2]))
                    self.assertEqual(posterior_f.variance.shape, torch.Size([2, 2]))

            # test observation noise
            posterior_noisy = model.posterior(test_x, observation_noise=True)
            self.assertTrue(
                torch.allclose(
                    posterior_noisy.variance,
                    model.likelihood(posterior_f.distribution).variance,
                )
            )

            # test posterior (batch eval)
            test_x = torch.rand(3, 2, 2, **tkwargs)
            posterior_f = model.posterior(test_x)
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultitaskMultivariateNormal)
            self.assertEqual(posterior_f.mean.shape, torch.Size([3, 2, 2]))
            self.assertEqual(posterior_f.variance.shape, torch.Size([3, 2, 2]))


class TestMultiTaskUtils(BotorchTestCase):
    def test_get_task_value_remapping(self) -> None:
        for dtype in (torch.float, torch.double):
            observed_task_values = torch.tensor(
                [1, 3], dtype=torch.long, device=self.device
            )
            expected_mapping = torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=self.device
            )
            all_task_values = torch.arange(5, dtype=torch.long, device=self.device)
            mapping = get_task_value_remapping(
                observed_task_values=observed_task_values,
                all_task_values=all_task_values,
                dtype=dtype,
                default_task_value=1.0,
            )
            self.assertTrue(torch.equal(mapping, expected_mapping))
            # test default_task_value that has not been observed
            # and default_task_value=None
            for default_task_value in (0.0, None):
                mapping = get_task_value_remapping(
                    observed_task_values=observed_task_values,
                    all_task_values=all_task_values,
                    dtype=dtype,
                    default_task_value=default_task_value,
                )
                self.assertTrue(torch.equal(mapping[[1, 3]], expected_mapping[[1, 3]]))
                self.assertTrue(torch.isnan(mapping[[0, 2, 4]]).all())

    def test_get_task_value_remapping_invalid_dtype(self) -> None:
        observed_task_values = torch.tensor([1, 3])
        all_task_values = observed_task_values
        for dtype in (torch.int32, torch.long, torch.bool):
            with self.assertRaisesRegex(
                ValueError,
                f"dtype must be torch.float or torch.double, but got {dtype}.",
            ):
                get_task_value_remapping(
                    observed_task_values=observed_task_values,
                    all_task_values=all_task_values,
                    dtype=dtype,
                    default_task_value=None,
                )
