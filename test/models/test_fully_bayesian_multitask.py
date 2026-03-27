#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools

import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelList, ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import (
    LinearPyroModel,
    matern52_kernel,
    MaternPyroModel,
    MCMC_DIM,
    MIN_INFERRED_NOISE_LEVEL,
    SaasPyroModel,
)
from botorch.models.fully_bayesian_multitask import (
    FullyBayesianMultiTaskGP,
    LatentFeatureMultiTaskPyroMixin,
    MultiTaskPyroMixin,
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GaussianMixturePosterior
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.test_helpers import gen_multi_task_dataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.means import MultitaskMean

EXPECTED_KEYS = [
    "mean_module.base_means.0.raw_constant",
    "mean_module.base_means.1.raw_constant",
    "covar_module.kernels.1.raw_var",
    "covar_module.kernels.1.active_dims",
    "covar_module.kernels.0.base_kernel.raw_lengthscale",
    "covar_module.kernels.0.base_kernel.raw_lengthscale_constraint.lower_bound",
    "covar_module.kernels.0.active_dims",
    "covar_module.kernels.1.raw_var_constraint.upper_bound",
    "covar_module.kernels.0.base_kernel.raw_lengthscale_constraint.upper_bound",
    "covar_module.kernels.0.raw_outputscale_constraint.lower_bound",
    "covar_module.kernels.1.covar_factor",
    "covar_module.kernels.0.raw_outputscale_constraint.upper_bound",
    "covar_module.kernels.1.raw_var_constraint.lower_bound",
    "covar_module.kernels.0.raw_outputscale",
]
EXPECTED_KEYS_NOISE = EXPECTED_KEYS + [
    "likelihood.noise_covar.raw_noise",
    "likelihood.noise_covar.raw_noise_constraint.lower_bound",
    "likelihood.noise_covar.raw_noise_constraint.upper_bound",
]


class TestFullyBayesianMultiTaskGP(BotorchTestCase):
    def _get_base_data(
        self,
        observed_task_values: list[int] | None = None,
        **tkwargs,
    ):
        """Generate standard training data for tests."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
        if observed_task_values is None:
            observed_task_values = [0, 1]
        task_indices = torch.cat(
            [torch.full((5, 1), observed_task_values[i], **tkwargs) for i in (0, 1)],
            dim=0,
        )
        self.num_tasks = 2
        train_X = torch.cat([train_X, task_indices], dim=1)
        train_Y = torch.sin(train_X[:, :1])
        train_Yvar = 0.5 * torch.arange(10, **tkwargs).unsqueeze(-1)
        return train_X, train_Y, train_Yvar

    def _get_data_and_model(
        self,
        task_rank: int | None = None,
        output_tasks: list[int] | None = None,
        infer_noise: bool = False,
        use_outcome_transform: bool = True,
        observed_task_values: list[int] | None = None,
        all_tasks: list[int] | None = None,
        validate_task_values: bool = True,
        **tkwargs,
    ):
        train_X, train_Y, train_Yvar = self._get_base_data(
            observed_task_values=observed_task_values, **tkwargs
        )
        model = SaasFullyBayesianMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None if infer_noise else train_Yvar,
            task_feature=4,
            all_tasks=all_tasks,
            output_tasks=output_tasks,
            rank=task_rank,
            outcome_transform=(
                Standardize(m=1, batch_shape=train_X.shape[:-2])
                if use_outcome_transform
                else None
            ),
            validate_task_values=validate_task_values,
        )
        return train_X, train_Y, train_Yvar, model

    def _get_unnormalized_data(self, infer_noise: bool = False, **tkwargs):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 4, **tkwargs)
            train_Y = torch.sin(train_X[:, :1])
            task_indices = torch.cat(
                [torch.zeros(5, 1, **tkwargs), torch.ones(5, 1, **tkwargs)], dim=0
            )
            train_X = torch.cat([5 + 5 * train_X, task_indices], dim=1)
            test_X = 5 + 5 * torch.rand(5, 4, **tkwargs)
            if infer_noise:
                train_Yvar = None
            else:
                train_Yvar = 0.1 * torch.arange(10, **tkwargs).unsqueeze(-1)
        return train_X, train_Y, train_Yvar, test_X

    def _get_unnormalized_condition_data(
        self, num_models: int, num_cond: int, dim: int, infer_noise: bool, **tkwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            cond_X = 5 + 5 * torch.rand(num_models, num_cond, dim, **tkwargs)
            cond_Y = 10 + torch.sin(cond_X[..., :1])
            cond_Yvar = (
                None if infer_noise else 0.1 * torch.ones(cond_Y.shape, **tkwargs)
            )
        # adding the task dimension
        cond_X = torch.cat(
            [cond_X, torch.zeros(num_models, num_cond, 1, **tkwargs)], dim=-1
        )
        return cond_X, cond_Y, cond_Yvar

    def _get_mcmc_samples(self, num_samples: int, dim: int, task_rank: int, **tkwargs):
        mcmc_samples = {
            "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
            "outputscale": torch.rand(num_samples, **tkwargs),
            "mean": torch.randn(num_samples, self.num_tasks, **tkwargs),
            "noise": torch.rand(num_samples, 1, **tkwargs),
            "task_lengthscale": torch.rand(num_samples, 1, task_rank, **tkwargs),
            "latent_features": torch.rand(
                num_samples, self.num_tasks, task_rank, **tkwargs
            ),
        }
        return mcmc_samples

    def test_raises(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, **tkwargs),
                train_Yvar=torch.rand(10, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(12, 1, **tkwargs),
                train_Yvar=torch.rand(12, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_X to have shape n x d and train_Y to have shape n x 1",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, 1, **tkwargs),
                task_feature=4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Expected train_Yvar to be None or have the same shape as train_Y",
        ):
            SaasFullyBayesianMultiTaskGP(
                train_X=torch.rand(10, 4, **tkwargs),
                train_Y=torch.randn(10, 1, **tkwargs),
                train_Yvar=torch.rand(10, **tkwargs),
                task_feature=4,
            )
        _, _, _, model = self._get_data_and_model(**tkwargs)
        sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
        with self.assertRaisesRegex(
            NotImplementedError, "Fantasize is not implemented!"
        ):
            model.fantasize(
                X=torch.cat(
                    [torch.rand(5, 4, **tkwargs), torch.ones(5, 1, **tkwargs)], dim=1
                ),
                sampler=sampler,
            )

        # Make sure an exception is raised if the model has not been fitted
        not_fitted_error_msg = (
            "Model has not been fitted. You need to call "
            "`fit_fully_bayesian_model_nuts` to fit the model."
        )
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.num_mcmc_samples
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.median_lengthscale
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.forward(torch.rand(1, 4, **tkwargs))
        with self.assertRaisesRegex(RuntimeError, not_fitted_error_msg):
            model.posterior(torch.rand(1, 4, **tkwargs))

    def test_fit_model(
        self,
        dtype: torch.dtype = torch.double,
        infer_noise: bool = False,
        task_rank: int = 1,
        use_outcome_transform: bool = False,
    ):
        tkwargs = {"device": self.device, "dtype": dtype}
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            infer_noise=infer_noise,
            task_rank=task_rank,
            use_outcome_transform=use_outcome_transform,
            **tkwargs,
        )
        n = train_X.shape[0]
        d = train_X.shape[1] - 1

        # Handle outcome transforms (if used)
        train_Y_tf, train_Yvar_tf = train_Y, train_Yvar
        if use_outcome_transform:
            train_Y_tf, train_Yvar_tf = model.outcome_transform(
                Y=train_Y, Yvar=train_Yvar
            )
        expected_mapped_task_values = torch.zeros(10, **tkwargs)
        expected_mapped_task_values[5:] = 1

        # Test init
        self.assertIsNone(model.mean_module)
        self.assertIsNone(model.covar_module)
        self.assertIsNone(model.likelihood)
        self.assertIsInstance(model.pyro_model, MultitaskSaasPyroModel)
        self.assertAllClose(train_X[:, :-1], model.pyro_model.train_X[:, :-1])
        self.assertAllClose(
            model.pyro_model.train_X[:, -1], expected_mapped_task_values
        )
        self.assertAllClose(train_Y_tf, model.pyro_model.train_Y)
        if infer_noise:
            self.assertIsNone(model.pyro_model.train_Yvar)
        else:
            self.assertAllClose(
                train_Yvar_tf.clamp(MIN_INFERRED_NOISE_LEVEL),
                model.pyro_model.train_Yvar,
            )

        # Fit a model and check that the hyperparameters have the correct shape
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        data_covar_module, task_covar_module = model.covar_module.kernels
        self.assertEqual(model.batch_shape, torch.Size([3]))
        self.assertIsInstance(model.mean_module, MultitaskMean)
        self.assertEqual(model.mean_module.num_tasks, self.num_tasks)
        for i in range(self.num_tasks):
            self.assertEqual(
                model.mean_module.base_means[i].raw_constant.shape,
                model.batch_shape,
            )
        self.assertIsInstance(data_covar_module, ScaleKernel)
        self.assertEqual(data_covar_module.outputscale.shape, model.batch_shape)
        self.assertIsInstance(data_covar_module.base_kernel, MaternKernel)
        self.assertEqual(
            data_covar_module.base_kernel.lengthscale.shape, torch.Size([3, 1, d])
        )
        if infer_noise:
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertEqual(
                model.likelihood.noise_covar.noise.shape, torch.Size([3, 1])
            )
        else:
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
        self.assertIsInstance(task_covar_module, IndexKernel)
        # Predict on some test points
        for batch_shape in [[5], [5, 2], [5, 2, 6]]:
            test_X = torch.rand(*batch_shape, d, **tkwargs)
            posterior = model.posterior(test_X)
            self.assertIsInstance(posterior, GaussianMixturePosterior)
            self.assertIsInstance(posterior, GaussianMixturePosterior)

            # Test with observation noise.
            # Add task index to have variability in added noise.
            task_idcs = torch.tensor(
                [[i % self.num_tasks] for i in range(batch_shape[-1])],
                device=self.device,
            )
            test_X_w_task = torch.cat(
                [test_X, task_idcs.expand(*batch_shape, 1)], dim=-1
            )
            noise_free_posterior = model.posterior(X=test_X_w_task)
            noisy_posterior = model.posterior(X=test_X_w_task, observation_noise=True)
            self.assertAllClose(noisy_posterior.mean, noise_free_posterior.mean)
            added_noise = noisy_posterior.variance - noise_free_posterior.variance
            self.assertTrue(torch.all(added_noise > 0.0))
            if infer_noise is False:
                # Check that correct noise was added.
                train_tasks = train_X[..., 4]
                mean_noise_by_task = torch.tensor(
                    [
                        train_Yvar[train_tasks == i].mean(dim=0)
                        for i in train_tasks.unique(sorted=True)
                    ],
                    device=self.device,
                )
                expected_noise = mean_noise_by_task[task_idcs]
                self.assertAllClose(
                    added_noise, expected_noise.expand_as(added_noise), atol=1e-4
                )

            # Mean/variance
            num_outputs = self.num_tasks
            expected_shape = (
                *batch_shape[: MCMC_DIM + 2],
                *model.batch_shape,
                *batch_shape[MCMC_DIM + 2 :],
                num_outputs,
            )
            expected_shape = torch.Size(expected_shape)
            mean, var = posterior.mean, posterior.variance
            self.assertEqual(mean.shape, expected_shape)
            self.assertEqual(var.shape, expected_shape)

            # Mixture mean/variance/median/quantiles
            mixture_mean = posterior.mixture_mean
            mixture_variance = posterior.mixture_variance
            quantile1 = posterior.quantile(value=torch.tensor(0.01))
            quantile2 = posterior.quantile(value=torch.tensor(0.99))

            # Marginalized mean/variance
            self.assertEqual(
                mixture_mean.shape, torch.Size(batch_shape + [num_outputs])
            )
            self.assertEqual(
                mixture_variance.shape, torch.Size(batch_shape + [num_outputs])
            )
            self.assertTrue(mixture_variance.min() > 0.0)
            self.assertEqual(quantile1.shape, torch.Size(batch_shape + [num_outputs]))
            self.assertEqual(quantile2.shape, torch.Size(batch_shape + [num_outputs]))
            self.assertTrue((quantile2 > quantile1).all())

            dist = torch.distributions.Normal(
                loc=posterior.mean, scale=posterior.variance.sqrt()
            )
            torch.allclose(
                dist.cdf(quantile1.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                0.05 * torch.ones(batch_shape + [1], **tkwargs),
            )
            torch.allclose(
                dist.cdf(quantile2.unsqueeze(MCMC_DIM)).mean(dim=MCMC_DIM),
                0.95 * torch.ones(batch_shape + [1], **tkwargs),
            )
            # Invalid quantile should raise
            for q in [-1.0, 0.0, 1.0, 1.3333]:
                with self.assertRaisesRegex(
                    ValueError, "value is expected to be in the range"
                ):
                    posterior.quantile(value=torch.tensor(q))

            # Test model lists with fully Bayesian models and mixed modeling
            deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
            for ModelListClass, models, expected_outputs in zip(
                [ModelList, ModelListGP],
                [[deterministic, model], [model, model]],
                [num_outputs + 1, num_outputs * 2],
            ):
                expected_shape = (
                    *batch_shape[: MCMC_DIM + 2],
                    *model.batch_shape,
                    *batch_shape[MCMC_DIM + 2 :],
                    expected_outputs,
                )
                expected_shape = torch.Size(expected_shape)
                model_list = ModelListClass(*models)
                posterior = model_list.posterior(test_X)
                mean, var = posterior.mean, posterior.variance
                self.assertEqual(mean.shape, expected_shape)
                self.assertEqual(var.shape, expected_shape)

        # Check properties
        median_lengthscale = model.median_lengthscale
        self.assertEqual(median_lengthscale.shape, torch.Size([d]))
        self.assertEqual(model.num_mcmc_samples, 3)

        # Check the keys in the state dict
        true_keys = EXPECTED_KEYS_NOISE if infer_noise else EXPECTED_KEYS
        extra_keys = []
        if use_outcome_transform:
            extra_keys = [
                "outcome_transform.stdvs",
                "outcome_transform._is_trained",
                "outcome_transform._stdvs_sq",
                "outcome_transform.means",
            ]
        self.assertEqual(set(model.state_dict().keys()), {*true_keys, *extra_keys})

        # Check that we can load the state dict.
        state_dict = model.state_dict()
        _, _, _, m_new = self._get_data_and_model(
            infer_noise=infer_noise,
            task_rank=task_rank,
            use_outcome_transform=use_outcome_transform,
            **tkwargs,
        )
        expected_state_dict = {}
        if use_outcome_transform:
            expected_state_dict.update(
                {
                    "outcome_transform." + k: v
                    for k, v in model.outcome_transform.state_dict().items()
                }
            )
        for k, v in m_new.state_dict().items():
            self.assertEqual(expected_state_dict[k], v)
        self.assertEqual(expected_state_dict.keys(), m_new.state_dict().keys())
        m_new.load_state_dict(state_dict)
        self.assertEqual(model.state_dict().keys(), m_new.state_dict().keys())
        for k in model.state_dict().keys():
            self.assertTrue((model.state_dict()[k] == m_new.state_dict()[k]).all())
        preds1, preds2 = model.posterior(test_X), m_new.posterior(test_X)
        self.assertTrue(torch.equal(preds1.mean, preds2.mean))
        self.assertTrue(torch.equal(preds1.variance, preds2.variance))

        # Make sure the model shapes are set correctly
        self.assertEqual(model.pyro_model.train_X.shape, torch.Size([n, d + 1]))
        self.assertAllClose(train_X[:, :-1], model.pyro_model.train_X[:, :-1])
        self.assertAllClose(
            model.pyro_model.train_X[:, -1], expected_mapped_task_values
        )

        # Put the model in eval mode with reset=True (reset should be ignored)
        trained_model = model.train(mode=False, reset=True)
        self.assertIs(trained_model, model)
        self.assertAllClose(train_X[:, :-1], model.pyro_model.train_X[:, :-1])
        self.assertAllClose(
            model.pyro_model.train_X[:, -1], expected_mapped_task_values
        )
        self.assertIsNotNone(model.mean_module)
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.likelihood)
        # Put the model in train mode, without resetting
        trained_model = model.train(reset=False)
        self.assertIs(trained_model, model)
        self.assertAllClose(train_X[:, :-1], model.pyro_model.train_X[:, :-1])
        self.assertAllClose(
            model.pyro_model.train_X[:, -1], expected_mapped_task_values
        )
        self.assertIsNotNone(model.mean_module)
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.likelihood)
        # Put the model in train mode, with resetting
        trained_model = model.train()
        self.assertIs(trained_model, model)
        self.assertAllClose(train_X[:, :-1], model.pyro_model.train_X[:, :-1])
        self.assertAllClose(
            model.pyro_model.train_X[:, -1], expected_mapped_task_values
        )
        self.assertIsNone(model.mean_module)
        self.assertIsNone(model.covar_module)
        self.assertIsNone(model.likelihood)

    def test_fit_model_float(self):
        self.test_fit_model(dtype=torch.float)

    def test_fit_model_infer_noise(self):
        self.test_fit_model(infer_noise=True, task_rank=2)

    def test_fit_model_with_outcome_transform(self):
        self.test_fit_model(use_outcome_transform=True)

    def test_fit_model_with_unobserved_tasks(self) -> None:
        """Test fitting and predicting when some tasks have no training data."""
        dtype = torch.double
        tkwargs = {"device": self.device, "dtype": dtype}
        # Tasks 0 and 2 observed; task 1 has no training data
        _, _, _, model = self._get_data_and_model(
            infer_noise=True,
            use_outcome_transform=True,
            output_tasks=[2],
            observed_task_values=[0, 2],
            all_tasks=[0, 1, 2],
            validate_task_values=False,
            **tkwargs,
        )
        # Contiguous all_tasks → no mapper needed
        self.assertIsNone(model._task_mapper)
        self.assertEqual(model.pyro_model.num_tasks, 3)

        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        self.assertIsNotNone(model.mean_module)

        # Predict for observed tasks
        test_X = torch.rand(3, 4, **tkwargs)
        posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GaussianMixturePosterior)
        # output_tasks=[2] → single output
        self.assertEqual(posterior.mean.shape[-1], 1)

        # Predict for the UNOBSERVED task (task 1)
        test_X_unobs = torch.cat(
            [torch.rand(3, 4, **tkwargs), torch.ones(3, 1, **tkwargs)], dim=-1
        )
        posterior_unobs = model.posterior(test_X_unobs)
        self.assertIsInstance(posterior_unobs, GaussianMixturePosterior)

    def test_transforms(self, infer_noise: bool = False):
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar, test_X = self._get_unnormalized_data(**tkwargs)
        n, d = train_X.shape
        normalize_indices = torch.tensor(
            list(range(train_X.shape[-1] - 1)), device=self.device
        )

        lb, ub = (
            train_X[:, normalize_indices].min(dim=0).values,
            train_X[:, normalize_indices].max(dim=0).values,
        )
        train_X_new = train_X.clone()
        train_X_new[..., normalize_indices] = (train_X[..., normalize_indices] - lb) / (
            ub - lb
        )
        # TODO: add testing of stratified standardization
        mu, sigma = train_Y.mean(), train_Y.std()

        # Fit without transforms
        with torch.random.fork_rng():
            torch.manual_seed(42)
            gp1 = SaasFullyBayesianMultiTaskGP(
                train_X=train_X_new,
                train_Y=(train_Y - mu) / sigma,
                train_Yvar=None if infer_noise else train_Yvar / sigma**2,
                task_feature=d - 1,
            )
            fit_fully_bayesian_model_nuts(
                gp1, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
        posterior1 = gp1.posterior((test_X - lb) / (ub - lb), output_indices=[0])
        pred_mean1 = mu + sigma * posterior1.mean
        pred_var1 = (sigma**2) * posterior1.variance

        # Fit with transforms
        with torch.random.fork_rng():
            torch.manual_seed(42)
            gp2 = SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=None if infer_noise else train_Yvar,
                task_feature=d - 1,
                input_transform=Normalize(
                    d=train_X.shape[-1], indices=normalize_indices
                ),
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                gp2, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
            )
            posterior2 = gp2.posterior(X=test_X, output_indices=[0])
            pred_mean2, pred_var2 = posterior2.mean, posterior2.variance

        self.assertAllClose(pred_mean1, pred_mean2)
        self.assertAllClose(pred_var1, pred_var2)

    def test_transforms_infer_noise(self):
        self.test_transforms(infer_noise=True)

    def test_acquisition_functions(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # Using a single output model here since we test with single objective acqfs.
        train_X, train_Y, train_Yvar, model = self._get_data_and_model(
            task_rank=1, output_tasks=[0], **tkwargs
        )
        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        for include_task_feature in [True, False]:
            if not include_task_feature:
                test_X = train_X[..., :-1]
            else:
                test_X = train_X

            deterministic = GenericDeterministicModel(f=lambda x: x[..., :1])
            list_gp = ModelListGP(model, model)
            mixed_list = ModelList(deterministic, model)
            simple_sampler = get_sampler(
                posterior=model.posterior(test_X),
                sample_shape=torch.Size([2]),
            )
            list_gp_sampler = get_sampler(
                posterior=list_gp.posterior(test_X), sample_shape=torch.Size([2])
            )
            mixed_list_sampler = get_sampler(
                posterior=mixed_list.posterior(test_X), sample_shape=torch.Size([2])
            )
            acquisition_functions = [
                LogExpectedImprovement(model=model, best_f=train_Y.max()),
                ProbabilityOfImprovement(model=model, best_f=train_Y.max()),
                PosteriorMean(model=model),
                UpperConfidenceBound(model=model, beta=4),
                qLogExpectedImprovement(
                    model=model, best_f=train_Y.max(), sampler=simple_sampler
                ),
                qLogNoisyExpectedImprovement(
                    model=model, X_baseline=test_X, sampler=simple_sampler
                ),
                qProbabilityOfImprovement(
                    model=model, best_f=train_Y.max(), sampler=simple_sampler
                ),
                qSimpleRegret(model=model, sampler=simple_sampler),
                qUpperConfidenceBound(model=model, beta=4, sampler=simple_sampler),
                qLogNoisyExpectedHypervolumeImprovement(
                    model=list_gp,
                    X_baseline=test_X,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=list_gp_sampler,
                ),
                qLogExpectedHypervolumeImprovement(
                    model=list_gp,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=list_gp_sampler,
                    partitioning=NondominatedPartitioning(
                        ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                    ),
                ),
                # qEHVI/qNEHVI with mixed models
                qLogNoisyExpectedHypervolumeImprovement(
                    model=mixed_list,
                    X_baseline=test_X,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=mixed_list_sampler,
                ),
                qLogExpectedHypervolumeImprovement(
                    model=mixed_list,
                    ref_point=torch.zeros(2, **tkwargs),
                    sampler=mixed_list_sampler,
                    partitioning=NondominatedPartitioning(
                        ref_point=torch.zeros(2, **tkwargs), Y=train_Y.repeat([1, 2])
                    ),
                ),
            ]

            for acqf in acquisition_functions:
                for batch_shape in [[2], [6, 2], [5, 6, 2]]:
                    test_X = torch.rand(*batch_shape, 1, 4, **tkwargs)
                    if include_task_feature:
                        test_X = torch.cat(
                            [test_X, torch.zeros_like(test_X[..., :1])], dim=-1
                        )
                    self.assertEqual(acqf(test_X).shape, torch.Size(batch_shape))

    def test_condition_on_observations(self) -> None:
        # The following conditioned data shapes should work (output describes):
        # training data shape after cond(batch shape in output is req. in gpytorch)
        # X: num_models x n x d, Y: num_models x n x d --> num_models x n x d
        # X: n x d, Y: n x d --> num_models x n x d
        # X: n x d, Y: num_models x n x d --> num_models x n x d
        num_models = 3
        num_cond = 2
        task_rank = 2
        for infer_noise, dtype in itertools.product(
            (True, False), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, _, _, model = self._get_data_and_model(
                task_rank=task_rank,
                infer_noise=infer_noise,
                **tkwargs,
            )
            num_dims = train_X.shape[1] - 1
            mcmc_samples = self._get_mcmc_samples(
                num_samples=3,
                dim=num_dims,
                task_rank=task_rank,
                **tkwargs,
            )
            model.load_mcmc_samples(mcmc_samples)

            num_train = train_X.shape[0]
            test_X = torch.rand(num_models, num_dims, **tkwargs)

            cond_X, cond_Y, cond_Yvar = self._get_unnormalized_condition_data(
                num_models=num_models,
                num_cond=num_cond,
                infer_noise=infer_noise,
                dim=num_dims,
                **tkwargs,
            )

            # need to forward pass before conditioning
            model.posterior(train_X)
            cond_model = model.condition_on_observations(
                cond_X, cond_Y, noise=cond_Yvar
            )
            posterior = cond_model.posterior(test_X)
            self.assertEqual(
                posterior.mean.shape, torch.Size([num_models, len(test_X), 2])
            )

            # since the data is not equal for the conditioned points, a batch size
            # is added to the training data
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims + 1]),
            )

            # the batch shape of the condition model is added during conditioning
            self.assertEqual(cond_model.batch_shape, torch.Size([num_models]))

            # condition on identical sets of data (i.e. one set) for all models
            # i.e, with no batch shape. This infers the batch shape.
            cond_X_nobatch, cond_Y_nobatch = cond_X[0], cond_Y[0]

            # conditioning without a batch size - the resulting conditioned model
            # will still have a batch size
            model.posterior(train_X)
            cond_model = model.condition_on_observations(
                cond_X_nobatch, cond_Y_nobatch, noise=cond_Yvar
            )
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims + 1]),
            )

            # With batch size only on Y.
            cond_model = model.condition_on_observations(
                cond_X_nobatch, cond_Y, noise=cond_Yvar
            )
            self.assertEqual(
                cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + num_cond, num_dims + 1]),
            )

            # test repeated conditioning
            repeat_cond_X = cond_X.clone()
            repeat_cond_X[..., 0:-1] += 2
            repeat_cond_model = cond_model.condition_on_observations(
                repeat_cond_X, cond_Y, noise=cond_Yvar
            )
            self.assertEqual(
                repeat_cond_model.train_inputs[0].shape,
                torch.Size([num_models, num_train + 2 * num_cond, num_dims + 1]),
            )

            # test repeated conditioning without a batch size
            repeat_cond_X_nobatch = cond_X_nobatch.clone()
            repeat_cond_X_nobatch[..., 0:-1] += 2
            repeat_cond_model2 = repeat_cond_model.condition_on_observations(
                repeat_cond_X_nobatch, cond_Y_nobatch, noise=cond_Yvar
            )
            self.assertEqual(
                repeat_cond_model2.train_inputs[0].shape,
                torch.Size([num_models, num_train + 3 * num_cond, num_dims + 1]),
            )

    def test_load_samples(self):
        for task_rank, dtype, use_outcome_transform in itertools.product(
            [1, 2], [torch.float, torch.double], (False, True)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y, train_Yvar, model = self._get_data_and_model(
                task_rank=task_rank,
                use_outcome_transform=use_outcome_transform,
                **tkwargs,
            )

            d = train_X.shape[1] - 1
            mcmc_samples = self._get_mcmc_samples(
                num_samples=3,
                dim=d,
                task_rank=task_rank,
                **tkwargs,
            )
            model.load_mcmc_samples(mcmc_samples)
            data_covar_module, task_covar_module = model.covar_module.kernels
            self.assertTrue(
                torch.allclose(
                    data_covar_module.base_kernel.lengthscale,
                    mcmc_samples["lengthscale"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    data_covar_module.outputscale,
                    mcmc_samples["outputscale"],
                )
            )
            self.assertIsInstance(model.mean_module, MultitaskMean)
            for i in range(self.num_tasks):
                self.assertTrue(
                    torch.allclose(
                        model.mean_module.base_means[i].constant.data,
                        mcmc_samples["mean"][:, i],
                    )
                )

            self.assertTrue(
                torch.allclose(
                    task_covar_module.covar_matrix.to_dense(),
                    matern52_kernel(
                        mcmc_samples["latent_features"],
                        mcmc_samples["task_lengthscale"],
                    ),
                )
            )
            # Handle outcome transforms (if used)
            train_Y_tf, train_Yvar_tf = train_Y, train_Yvar
            if use_outcome_transform:
                train_Y_tf, train_Yvar_tf = model.outcome_transform(
                    Y=train_Y, Yvar=train_Yvar
                )

            self.assertTrue(
                torch.allclose(
                    model.pyro_model.train_Y,
                    train_Y_tf,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.pyro_model.train_Yvar,
                    train_Yvar_tf.clamp(MIN_INFERRED_NOISE_LEVEL),
                )
            )

    def test_construct_inputs(self):
        for dtype, infer_noise in [(torch.float, False), (torch.double, True)]:
            tkwargs = {"device": self.device, "dtype": dtype}
            task_feature = 0
            datasets, (train_X, train_Y, train_Yvar) = gen_multi_task_dataset(
                yvar=None if infer_noise else 0.05, **tkwargs
            )

            model = SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                task_feature=task_feature,
            )

            data_dict = model.construct_inputs(
                datasets,
                task_feature=task_feature,
                rank=1,
            )
            self.assertTrue(torch.equal(data_dict["train_X"], train_X))
            self.assertTrue(torch.equal(data_dict["train_Y"], train_Y))
            if train_Yvar is not None:
                self.assertAllClose(data_dict["train_Yvar"], train_Yvar)
            else:
                self.assertNotIn("train_Yvar", data_dict)
            self.assertEqual(data_dict["task_feature"], task_feature)
            self.assertEqual(data_dict["rank"], 1)
            self.assertTrue("task_covar_prior" not in data_dict)

            task_feature = -1
            datasets, (train_X, train_Y, train_Yvar) = gen_multi_task_dataset(
                yvar=None if infer_noise else 0.05, **tkwargs
            )

            d = train_X.shape[1] - 1
            model = SaasFullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                task_feature=task_feature,
            )
            self.assertEqual(model._task_feature, d)
            self.assertEqual(model.pyro_model.task_feature, d)

    def test_constructor_validation_and_defaults(self):
        """Test FullyBayesianMultiTaskGP constructor validation and defaults."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_Y, train_Yvar = self._get_base_data(**tkwargs)

        with self.subTest("rejects_single_task_pyro_model"):
            with self.assertRaisesRegex(
                ValueError, "pyro_model must be a multi-task model"
            ):
                FullyBayesianMultiTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    task_feature=4,
                    pyro_model=MaternPyroModel(),
                )

        with self.subTest("accepts_explicit_multitask_pyro_model"):
            pyro_model = MultitaskSaasPyroModel()
            model = FullyBayesianMultiTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                task_feature=4,
                pyro_model=pyro_model,
            )
            self.assertIs(model.pyro_model, pyro_model)
            self.assertIsInstance(model.pyro_model, MultiTaskPyroMixin)

    def test_non_saas_mt_model_load_state_dict(self):
        """Test round-trip load_state_dict with a non-SAAS multi-task PyroModel."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        class MultitaskMaternPyroModel(
            LatentFeatureMultiTaskPyroMixin, MaternPyroModel
        ):
            pass

        train_X, train_Y, train_Yvar = self._get_base_data(**tkwargs)

        pyro_model = MultitaskMaternPyroModel()
        model = FullyBayesianMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=4,
            pyro_model=pyro_model,
        )
        self.assertIsInstance(model.pyro_model, MultiTaskPyroMixin)

        fit_fully_bayesian_model_nuts(
            model, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        state_dict = model.state_dict()
        test_X = torch.rand(3, 4, **tkwargs)
        preds1 = model.posterior(test_X)

        pyro_model2 = MultitaskMaternPyroModel()
        m_new = FullyBayesianMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=4,
            pyro_model=pyro_model2,
        )
        m_new.load_state_dict(state_dict)

        preds2 = m_new.posterior(test_X)
        self.assertTrue(torch.equal(preds1.mean, preds2.mean))
        self.assertTrue(torch.equal(preds1.variance, preds2.variance))


class TestPyroModelMultitaskMixin(BotorchTestCase):
    """Tests for the MultiTaskPyroMixin and LatentFeatureMultiTaskPyroMixin classes."""

    def _make_multitask_data(self, n: int = 10, d: int = 3, num_tasks: int = 2):
        tkwargs = {"dtype": torch.double, "device": self.device}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            X = torch.rand(n, d, **tkwargs)
        task_indices = torch.cat(
            [torch.full((n // num_tasks, 1), i, **tkwargs) for i in range(num_tasks)],
            dim=0,
        )
        train_X = torch.cat([X, task_indices], dim=-1)
        train_Y = torch.sin(train_X[:, :1])
        return train_X, train_Y, tkwargs

    def _make_mt_model(self, pyro_cls):
        """Create a multitask PyroModel by composing the mixin with pyro_cls."""

        class _MT(LatentFeatureMultiTaskPyroMixin, pyro_cls):
            pass

        return _MT()

    def test_set_inputs_multitask(self) -> None:
        """Test that set_inputs correctly handles multitask configuration."""
        train_X, train_Y, tkwargs = self._make_multitask_data()
        for pyro_cls in (MaternPyroModel, SaasPyroModel, LinearPyroModel):
            with self.subTest(pyro_cls=pyro_cls.__name__):
                model = self._make_mt_model(pyro_cls)
                model.set_inputs(
                    train_X=train_X,
                    train_Y=train_Y,
                    task_feature=-1,
                    task_rank=1,
                )
                self.assertEqual(model.task_feature, train_X.shape[-1] - 1)
                self.assertEqual(model.num_tasks, 2)
                self.assertEqual(model.task_rank, 1)
                # ard_num_dims should exclude the task column
                self.assertEqual(model.ard_num_dims, train_X.shape[-1] - 1)
                self.assertIsInstance(model, MultiTaskPyroMixin)

    def test_sample_mean_multitask(self) -> None:
        """Test that sample_mean returns (num_tasks,) vector in multitask mode."""
        train_X, train_Y, tkwargs = self._make_multitask_data()
        model = self._make_mt_model(MaternPyroModel)
        model.set_inputs(train_X=train_X, train_Y=train_Y, task_feature=-1)
        mean = model.sample_mean(**tkwargs)
        self.assertEqual(mean.shape, torch.Size([2]))

    def test_multitask_helpers(self) -> None:
        """Test _get_task_indices_and_base_idxr and _build_task_covar."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        model = self._make_mt_model(MaternPyroModel)
        model.set_inputs(train_X=train_X, train_Y=train_Y, task_feature=-1)

        task_indices, base_idxr = model._get_task_indices_and_base_idxr(**tkwargs)
        self.assertEqual(task_indices.shape, torch.Size([10]))
        # base_idxr should select all columns except the task column
        self.assertEqual(base_idxr.shape, torch.Size([3]))
        # The task feature is at index 3, so base_idxr = [0, 1, 2]
        # (indices before task_feature are unchanged)
        self.assertTrue(
            torch.equal(base_idxr, torch.tensor([0, 1, 2], device=self.device))
        )

        # Test _build_task_covar returns correct shapes
        task_covar, task_idx = model._build_task_covar(**tkwargs)
        self.assertEqual(task_covar.shape, torch.Size([10, 10]))

    def test_maybe_multitask_transform_multitask(self) -> None:
        """Test _maybe_multitask_transform modifies K and mean in multitask."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        model = self._make_mt_model(MaternPyroModel)
        model.set_inputs(train_X=train_X, train_Y=train_Y, task_feature=-1)
        K = torch.eye(10, **tkwargs)
        mean = torch.tensor([0.1, 0.2], **tkwargs)
        K_out, mean_out = model._maybe_multitask_transform(K, mean, **tkwargs)
        # K should be modified (element-wise product with task covariance)
        self.assertEqual(K_out.shape, torch.Size([10, 10]))
        # mean should be expanded by task indices
        self.assertEqual(mean_out.shape, torch.Size([10]))

    def test_multitask_sample(self) -> None:
        """Test that LatentFeatureMultiTaskPyroMixin + various PyroModels sample()."""
        for base_model_cls in (MaternPyroModel, SaasPyroModel, LinearPyroModel):
            with self.subTest(base_model=base_model_cls.__name__):
                train_X, train_Y, tkwargs = self._make_multitask_data(
                    n=10, d=3, num_tasks=2
                )
                model = self._make_mt_model(base_model_cls)
                model.set_inputs(
                    train_X=train_X,
                    train_Y=train_Y,
                    task_feature=-1,
                    task_rank=1,
                )
                # Verify mixin attributes are correctly set
                self.assertEqual(model.num_tasks, 2)
                self.assertEqual(model.task_rank, 1)
                self.assertEqual(model.task_feature, 3)  # last column (d=3 -> index 3)
                self.assertEqual(model.ard_num_dims, 3)  # d columns, excluding task
                # sample() should run without error
                model.sample()

    def _get_multitask_mcmc_samples(
        self,
        num_mcmc: int,
        num_tasks: int,
        d: int,
        task_rank: int,
        include_outputscale: bool = False,
    ) -> dict[str, torch.Tensor]:
        tkwargs = {"dtype": torch.double, "device": self.device}
        mcmc_samples = {
            "mean": torch.randn(num_mcmc, num_tasks, **tkwargs),
            "lengthscale": torch.rand(num_mcmc, 1, d, **tkwargs),
            "noise": torch.rand(num_mcmc, 1, **tkwargs),
            "task_lengthscale": torch.rand(num_mcmc, 1, task_rank, **tkwargs),
            "latent_features": torch.rand(num_mcmc, num_tasks, task_rank, **tkwargs),
        }
        if include_outputscale:
            mcmc_samples["outputscale"] = torch.rand(num_mcmc, **tkwargs)
        return mcmc_samples

    def test_matern_multitask_load_mcmc_samples(self) -> None:
        """Test loading MCMC samples produces correct module types for Matern."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        num_mcmc = 4
        task_rank = 2

        model = self._make_mt_model(MaternPyroModel)
        model.set_inputs(
            train_X=train_X, train_Y=train_Y, task_feature=-1, task_rank=task_rank
        )
        mcmc_samples = self._get_multitask_mcmc_samples(
            num_mcmc=num_mcmc,
            num_tasks=2,
            d=3,
            task_rank=task_rank,
        )
        mean_module, covar_module, likelihood, warp = model.load_mcmc_samples(
            mcmc_samples
        )
        # Mean should be MultitaskMean
        self.assertIsInstance(mean_module, MultitaskMean)
        self.assertEqual(mean_module.num_tasks, 2)
        # Covar should be product kernel (data * task)
        self.assertEqual(len(covar_module.kernels), 2)
        self.assertIsInstance(covar_module.kernels[1], IndexKernel)
        self.assertIsNone(warp)

    def test_saas_multitask_load_mcmc_samples(self) -> None:
        """Test loading MCMC samples for MT mixin + SaasPyroModel."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        num_mcmc = 4
        task_rank = 2

        model = self._make_mt_model(SaasPyroModel)
        model.set_inputs(
            train_X=train_X, train_Y=train_Y, task_feature=-1, task_rank=task_rank
        )
        mcmc_samples = self._get_multitask_mcmc_samples(
            num_mcmc=num_mcmc,
            num_tasks=2,
            d=3,
            task_rank=task_rank,
            include_outputscale=True,
        )
        mean_module, covar_module, likelihood, warp = model.load_mcmc_samples(
            mcmc_samples
        )
        self.assertIsInstance(mean_module, MultitaskMean)
        # data_covar should be ScaleKernel; product kernel has 2 sub-kernels
        data_covar, task_covar = covar_module.kernels
        self.assertIsInstance(data_covar, ScaleKernel)
        self.assertIsInstance(task_covar, IndexKernel)

    def test_build_multitask_covariance_task_covar_values(self) -> None:
        """Test that _build_multitask_covariance produces correct task covariance."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        num_mcmc = 3
        task_rank = 2

        model = self._make_mt_model(SaasPyroModel)
        model.set_inputs(
            train_X=train_X, train_Y=train_Y, task_feature=-1, task_rank=task_rank
        )
        mcmc_samples = self._get_multitask_mcmc_samples(
            num_mcmc=num_mcmc,
            num_tasks=2,
            d=3,
            task_rank=task_rank,
            include_outputscale=True,
        )
        mean_module, covar_module, _, _ = model.load_mcmc_samples(mcmc_samples)
        _, task_covar_module = covar_module.kernels
        # Verify the task covariance matches what matern52_kernel would produce
        expected_task_covar = matern52_kernel(
            mcmc_samples["latent_features"],
            mcmc_samples["task_lengthscale"],
        )
        self.assertAllClose(
            task_covar_module.covar_matrix.to_dense(),
            expected_task_covar,
        )

    def test_prepare_features_strips_task_column(self) -> None:
        """Test _prepare_features strips the task column in multitask mode."""
        train_X, train_Y, tkwargs = self._make_multitask_data(n=10, d=3, num_tasks=2)
        model = self._make_mt_model(MaternPyroModel)
        model.set_inputs(train_X=train_X, train_Y=train_Y, task_feature=-1)
        X_out = model._prepare_features(train_X, **tkwargs)
        # Should have one fewer column (task column stripped)
        self.assertEqual(X_out.shape, torch.Size([10, 3]))
        # Should contain only the base feature columns
        expected = train_X[:, :3]
        self.assertAllClose(X_out, expected)

    def test_get_dummy_mcmc_samples_multitask_mixin(self) -> None:
        """Test get_dummy_mcmc_samples for MT mixin + SaasPyroModel."""
        tkwargs = {"dtype": torch.double, "device": self.device}
        train_X, train_Y, _ = self._make_multitask_data(n=10, d=3, num_tasks=2)
        model = self._make_mt_model(SaasPyroModel)
        model.set_inputs(train_X=train_X, train_Y=train_Y, task_feature=-1, task_rank=2)
        samples = model.get_dummy_mcmc_samples(num_mcmc_samples=4, **tkwargs)
        # mean should be (S, num_tasks)
        self.assertEqual(samples["mean"].shape, torch.Size([4, 2]))
        # task-specific keys
        self.assertEqual(samples["task_lengthscale"].shape, torch.Size([4, 2]))
        self.assertEqual(samples["latent_features"].shape, torch.Size([4, 2, 2]))
        # base model keys inherited via super()
        self.assertIn("lengthscale", samples)
        self.assertIn("outputscale", samples)
        self.assertIn("noise", samples)
