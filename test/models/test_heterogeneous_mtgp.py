#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch.models.kernels.heterogeneous_multitask import MultiTaskConditionalKernel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import IndexKernel, ProductKernel
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood


class TestHeterogeneousMTGP(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.ds1 = SupervisedDataset(
            X=torch.cat([torch.rand(5, 3), torch.zeros(5, 1)], dim=-1),
            Y=torch.rand(5, 1),
            feature_names=["x1", "x2", "x3", "task"],
            outcome_names=["task0"],
        )
        self.ds2 = SupervisedDataset(
            X=torch.cat([torch.rand(3, 2), torch.ones(3, 1)], dim=-1),
            Y=torch.rand(3, 1),
            feature_names=["x1", "x2", "task"],
            outcome_names=["task1"],
        )
        self.ds3 = SupervisedDataset(
            X=torch.cat([torch.rand(2, 4), torch.full((2, 1), 2)], dim=-1),
            Y=torch.rand(2, 1),
            feature_names=["x1", "x2", "x4", "x5", "task"],
            outcome_names=["task2"],
        )
        self.ds4 = SupervisedDataset(
            X=torch.cat([torch.rand(3, 3), torch.ones(3, 1)], dim=-1),
            Y=torch.rand(3, 1),
            feature_names=["x1", "x2", "x3", "task"],
            outcome_names=["task1"],
        )
        self.ds5 = SupervisedDataset(
            X=torch.rand(0, 4),
            Y=torch.rand(0, 1),
            feature_names=["x1", "x2", "x3", "task"],
            outcome_names=["task0"],
        )
        self.mtds = MultiTaskDataset(
            datasets=[self.ds1, self.ds2, self.ds3],
            target_outcome_name="task0",
            task_feature_index=-1,
        )

    def test_input_constructor_exceptions(self) -> None:
        invalid_mtds = MultiTaskDataset(
            datasets=[self.ds1, self.ds2],
            target_outcome_name="task0",
            task_feature_index=0,
        )
        with self.assertRaisesRegex(NotImplementedError, "task_feature_index"):
            HeterogeneousMTGP.construct_inputs(training_data=invalid_mtds)
        with self.assertRaisesRegex(NotImplementedError, "task_feature"):
            HeterogeneousMTGP.construct_inputs(training_data=self.mtds, task_feature=0)
        with self.assertRaisesRegex(NotImplementedError, "output_tasks"):
            HeterogeneousMTGP.construct_inputs(
                training_data=self.mtds, output_tasks=[1]
            )

    def test_input_constructor(self) -> None:
        model_inputs = HeterogeneousMTGP.construct_inputs(training_data=self.mtds)
        self.assertTrue(
            all(
                torch.equal(x_in, x_out)
                for x_in, x_out in zip(
                    model_inputs["train_Xs"],
                    (self.ds1.X[:, :-1], self.ds2.X[:, :-1], self.ds3.X[:, :-1]),
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(y_in, y_out)
                for y_in, y_out in zip(
                    model_inputs["train_Ys"], (self.ds1.Y, self.ds2.Y, self.ds3.Y)
                )
            )
        )
        self.assertIsNone(model_inputs["train_Yvars"])
        self.assertEqual(
            model_inputs["feature_indices"], [[0, 1, 2], [0, 1], [0, 1, 3, 4]]
        )
        self.assertEqual(model_inputs["full_feature_dim"], 5)
        self.assertIsNone(model_inputs["rank"])

    def test_standard_heterogeneous_mtgp(self) -> None:
        # Construct the model (inferred noise: train_Yvars is None).
        model_inputs = HeterogeneousMTGP.construct_inputs(training_data=self.mtds)
        model = HeterogeneousMTGP(**model_inputs)
        self.assertEqual(model.train_inputs[0].shape, torch.Size([10, 6]))
        self.assertEqual(model._task_feature, 5)
        self.assertEqual(model._output_tasks, [0])
        self.assertEqual(model.num_tasks, 3)
        covar_module = model.covar_module
        data_covar_module, task_covar_module = covar_module.kernels
        self.assertIsInstance(covar_module, ProductKernel)
        self.assertIsInstance(data_covar_module, MultiTaskConditionalKernel)
        self.assertIsInstance(task_covar_module, IndexKernel)
        self.assertEqual(len(data_covar_module.kernels), 3)
        self.assertEqual(
            data_covar_module.binary_map, [[1, 1, 0], [1, 0, 0], [1, 0, 1]]
        )

        with self.subTest("inferred_noise_uses_hadamard_likelihood"):
            self.assertIsInstance(model.likelihood, HadamardGaussianLikelihood)
            # HadamardGaussianLikelihood should have per-task noise parameters.
            self.assertEqual(
                model.likelihood.noise_covar.noise.shape[-1], model.num_tasks
            )

        # Evaluate the posterior.
        with self.assertRaisesRegex(UnsupportedError, "output_indices"):
            model.posterior(self.ds1.X, output_indices=[0, 1])
        posterior = model.posterior(self.ds1.X)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.distribution, MultivariateNormal)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        posterior = model.posterior(self.ds1.X.repeat(3, 1, 1))
        self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))
        # Evaluate the posterior where X includes a task feature
        # Test where task feature is not the target task
        X_with_task = torch.cat(
            [
                self.ds1.X.clone()[:, :-1],
                torch.ones(
                    *self.ds1.X.shape[:-1],
                    1,
                    dtype=self.ds1.X.dtype,
                    device=self.ds1.X.device,
                ),
            ],
            dim=-1,
        )
        with self.assertRaisesRegex(
            UnsupportedError, "Posterior can only be called for the target task."
        ):
            model.posterior(X_with_task)
        # test with target task
        X_with_task[..., -1] = 0
        posterior = model.posterior(X_with_task)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertIsInstance(posterior.distribution, MultivariateNormal)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

    def test_fixed_noise_likelihood(self) -> None:
        datasets = [
            SupervisedDataset(
                X=ds.X,
                Y=ds.Y,
                Yvar=torch.full_like(ds.Y, 0.1 * (i + 1)),
                feature_names=ds.feature_names,
                outcome_names=ds.outcome_names,
            )
            for i, ds in enumerate([self.ds1, self.ds2, self.ds3])
        ]
        mtds = MultiTaskDataset(
            datasets=datasets, target_outcome_name="task0", task_feature_index=-1
        )
        model = HeterogeneousMTGP(
            **HeterogeneousMTGP.construct_inputs(training_data=mtds)
        )
        self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)

    def test_identical_search_space(self) -> None:
        # Check that the model works fine with identical search spaces.
        mtds = MultiTaskDataset(
            datasets=[self.ds1, self.ds4],
            target_outcome_name="task0",
            task_feature_index=-1,
        )
        model_inputs = HeterogeneousMTGP.construct_inputs(training_data=mtds)
        self.assertEqual(model_inputs["feature_indices"], [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(model_inputs["full_feature_dim"], 3)

        # Construct the model.
        model = HeterogeneousMTGP(**model_inputs)
        self.assertEqual(model.train_inputs[0].shape, torch.Size([8, 4]))
        data_covar_module = model.covar_module.kernels[0]
        self.assertEqual(len(data_covar_module.kernels), 1)
        # Evaluate the posterior.
        posterior = model.posterior(self.ds1.X)
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
        posterior = model.posterior(self.ds1.X.repeat(3, 1, 1))
        self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))

    def test_with_no_target_data(self) -> None:
        mtds = MultiTaskDataset(
            datasets=[self.ds5, self.ds2, self.ds3],
            target_outcome_name="task0",
            task_feature_index=-1,
        )

        # Check the output of the input constructor.
        model_inputs = HeterogeneousMTGP.construct_inputs(training_data=mtds)
        with self.subTest("Test model input constructor"):
            for x_in, x_out in zip(
                model_inputs["train_Xs"],
                (self.ds5.X[:, :-1], self.ds2.X[:, :-1], self.ds3.X[:, :-1]),
            ):
                self.assertTrue(torch.equal(x_in, x_out))
            for y_in, y_out in zip(
                model_inputs["train_Ys"], (self.ds5.Y, self.ds2.Y, self.ds3.Y)
            ):
                self.assertIs(y_in, y_out)
            self.assertIsNone(model_inputs["train_Yvars"])
            self.assertEqual(
                model_inputs["feature_indices"], [[0, 1, 2], [0, 1], [0, 1, 3, 4]]
            )
            self.assertEqual(model_inputs["full_feature_dim"], 5)
            self.assertIsNone(model_inputs["rank"])
            self.assertEqual(model_inputs["all_tasks"], [0, 1, 2])

        # Construct the model.
        model = HeterogeneousMTGP(**model_inputs, validate_task_values=False)
        with self.subTest("Check for model attributes"):
            self.assertEqual(model.train_inputs[0].shape, torch.Size([5, 6]))
            self.assertEqual(model._task_feature, 5)
            self.assertEqual(model._output_tasks, [0])
            self.assertEqual(model.num_tasks, 3)
            data_covar_module = model.covar_module.kernels[0]
            self.assertIsInstance(data_covar_module, MultiTaskConditionalKernel)
            self.assertEqual(len(data_covar_module.kernels), 3)
            self.assertEqual(
                data_covar_module.binary_map, [[1, 1, 0], [1, 0, 0], [1, 0, 1]]
            )

        with self.subTest("Test model evaluation"):
            # Evaluation with task 0 succeeds.
            model.forward(model.map_to_full_tensor(X=torch.zeros(5, 3), task_index=0))
            # Evaluation with task 2 -- requires all_tasks to be passed in to the model.
            model.forward(model.map_to_full_tensor(X=torch.zeros(5, 4), task_index=2))
            # Evaluate the posterior.
            posterior = model.posterior(torch.rand(5, 3))
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertIsInstance(posterior.mvn, MultivariateNormal)
            self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))
            posterior = model.posterior(torch.rand(3, 5, 3))
            self.assertEqual(posterior.mean.shape, torch.Size([3, 5, 1]))
