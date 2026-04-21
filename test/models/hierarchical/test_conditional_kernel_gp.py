#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import gpytorch
import torch
from botorch.models.hierarchical.conditional_kernel_gp import (
    _row_equal,
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood


# This is the search space of the Jenatton test function.
DIM = 9
HIERARCHICAL_DEPENDENCIES = {
    0: {0: [1, 7], 1: [2, 8]},
    1: {0: [3], 1: [4]},
    2: {0.0: [5], 1.0: [6]},
}


class TestHierarchicalConditionalKernel(BotorchTestCase):
    def test_kernel_creation(self):
        lst_eval_and_separate_hierarchical = [
            (False, False),  # blocks [[3], [4], [5], [6], [7], [8]]
            (False, True),  # same as above
            (True, False),  # blocks [[0], [1, 7], [2, 8], [3], [4], [5], [6]]
            (True, True),  # blocks [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        ]
        lst_num_kernels = [6, 6, 7, 9]

        # Make sure the number of sub-kernels is correct.
        for (
            eval_hierarchical_features,
            separate_hierarchical_features,
        ), num_kernels in zip(lst_eval_and_separate_hierarchical, lst_num_kernels):
            with self.subTest(
                eval_hierarchical_features=eval_hierarchical_features,
                separate_hierarchical_features=separate_hierarchical_features,
            ):
                covar_module = HierarchicalConditionalKernel(
                    dim=DIM,
                    hierarchical_dependencies=HIERARCHICAL_DEPENDENCIES,
                    eval_hierarchical_features=eval_hierarchical_features,
                    separate_hierarchical_features=separate_hierarchical_features,
                )
                self.assertEqual(len(covar_module.kernels), num_kernels)

    def test_parent_value_types(self):
        """Test that kernel works with integer, float, and mixed parent values."""
        test_cases = [
            (
                "integer_values",
                {0: {0: [1, 7], 1: [2, 8]}, 1: {0: [3], 1: [4]}, 2: {0: [5], 1: [6]}},
            ),
            (
                "float_values",
                {
                    0: {0.0: [1, 7], 1.0: [2, 8]},
                    1: {0.0: [3], 1.0: [4]},
                    2: {0.0: [5], 1.0: [6]},
                },
            ),
            (
                "mixed_values",
                {
                    0: {0: [1, 7], 1: [2, 8]},
                    1: {0.0: [3], 1.0: [4]},
                    2: {0.0: [5], 1.0: [6]},
                },
            ),
        ]

        for name, hierarchical_deps in test_cases:
            with self.subTest(name=name):
                covar_module = HierarchicalConditionalKernel(
                    dim=9,
                    hierarchical_dependencies=hierarchical_deps,
                    eval_hierarchical_features=False,
                    use_saas_prior=False,
                    use_outputscale=False,
                )

                # Verify kernel creation succeeded
                self.assertEqual(len(covar_module.kernels), 6)

                # Verify ancestor_values are always stored as float64
                for ancestor_vals in covar_module.ancestor_values:
                    if len(ancestor_vals) > 0:
                        self.assertEqual(ancestor_vals.dtype, torch.float64)

    def test_kernel_evaluation(self):
        torch.manual_seed(42)
        covar_module = HierarchicalConditionalKernel(
            dim=DIM,
            hierarchical_dependencies=HIERARCHICAL_DEPENDENCIES,
            eval_hierarchical_features=False,
            use_saas_prior=False,
            use_outputscale=False,
        )

        # Testing is easier if we fix all lengthscales to 1.
        for kernel in covar_module.kernels:
            kernel.lengthscale = 1.0

        # Create test data for different branches
        X00_ = torch.rand(3, DIM)  # (x1=0, x2=0) -> x4, r8 active
        X00_[..., 0], X00_[..., 1] = 0.0, 0.0
        X00_[..., 2] = torch.tensor([0.0, 0.5, 1.0])

        X01_ = torch.rand(3, DIM)  # (x1=0, x2=1) -> x5, r8 active
        X01_[..., 0], X01_[..., 1] = 0.0, 1.0
        X01_[..., 2] = torch.tensor([0.0, 0.5, 1.0])

        X1_0 = torch.rand(3, DIM)  # (x1=1, x3=0) -> x6, r9 active
        X1_0[..., 0], X1_0[..., 2] = 1.0, 0.0
        X1_0[..., 1] = torch.tensor([0.0, 0.5, 1.0])

        X1_1 = torch.rand(3, DIM)  # (x1=1, x3=1) -> x7, r9 active
        X1_1[..., 0], X1_1[..., 2] = 1.0, 1.0
        X1_1[..., 1] = torch.tensor([0.0, 0.5, 1.0])

        base_cls = covar_module.kernels[0].__class__
        isotropic_covar_module = base_cls()
        isotropic_covar_module.lengthscale = 1.0

        # Due to symmetry, we only check the kernel matrices between X00_ and the other
        # three tensors.
        # Both x4 and r8 are shared
        self.assertAllClose(
            covar_module(X00_, X00_).to_dense(),
            isotropic_covar_module(X00_[..., 3], X00_[..., 3]).to_dense()
            + isotropic_covar_module(X00_[..., 7], X00_[..., 7]).to_dense(),
        )

        # Only r8 is shared
        expected = isotropic_covar_module(X00_[..., 7], X01_[..., 7]).to_dense()
        self.assertAllClose(covar_module(X00_, X01_).to_dense(), expected)

        # Make sure it evaluates the diagonal correctly.
        self.assertAllClose(
            covar_module(X00_, X01_, diag=True).to_dense(), expected.diagonal()
        )

        # No shared common parameter
        self.assertAllClose(covar_module(X00_, X1_0).to_dense(), torch.zeros(3, 3))

        # No shared common parameter
        self.assertAllClose(covar_module(X00_, X1_1).to_dense(), torch.zeros(3, 3))

        # Make sure the kernel evaluates the diagonal correctly.
        self.assertAllClose(
            covar_module(X00_, X1_1, diag=True).to_dense(), torch.zeros(3)
        )

    def test_float_value_separation(self):
        """Test proximity validation for float parent values."""
        with self.assertRaisesRegex(ValueError, "closer than the default tolerance"):
            HierarchicalConditionalKernel(
                dim=3, hierarchical_dependencies={0: {0.0: [1], 1e-9: [2]}}
            )
        # Sufficient separation constructs fine
        covar_module = HierarchicalConditionalKernel(
            dim=5,
            hierarchical_dependencies={0: {0.0: [1], 0.001: [2], 0.5: [3], 1.0: [4]}},
            eval_hierarchical_features=False,
        )
        self.assertEqual(len(covar_module.kernels), 4)


class TestHierarchicalConditionalKernelGP(BotorchTestCase):
    def test_multi_output_and_batched(self):
        """Exercise multi-output (m > 1) and batched (batch_shape != ()) inputs."""
        hierarchical_deps = {
            0: {0: [1, 7], 1: [2, 8]},
            1: {0: [3], 1: [4]},
            2: {0: [5], 1: [6]},
        }
        hierarchical_vals = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
        ]
        single_batch_X = torch.cat(
            [torch.tensor(hierarchical_vals), torch.rand(4, 6)], dim=-1
        )

        for name, train_X, train_Y, expected_mean_shape in [
            (
                "multi_output_unbatched",
                single_batch_X,
                torch.randn(4, 3),
                (4, 3),
            ),
            (
                "batched_single_output",
                torch.stack([single_batch_X, single_batch_X]),
                torch.randn(2, 4, 1),
                (2, 4, 1),
            ),
        ]:
            with self.subTest(name=name):
                model = HierarchicalConditionalKernelGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    hierarchical_dependencies=hierarchical_deps,
                )
                posterior = model.posterior(train_X)
                self.assertEqual(posterior.mean.shape, expected_mean_shape)

    def test_model_creation(self):
        """Test GP model creation with integer and float hierarchical parameters."""
        cases = [
            (
                "integer_values",
                {0: {0: [1, 7], 1: [2, 8]}, 1: {0: [3], 1: [4]}, 2: {0: [5], 1: [6]}},
                [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]],
            ),
            (
                "float_values",
                {
                    0: {0.0: [1, 7], 1.0: [2, 8]},
                    1: {0.0: [3], 1.0: [4]},
                    2: {0.0: [5], 1.0: [6]},
                },
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            ),
        ]
        for name, hierarchical_deps, hierarchical_vals in cases:
            with self.subTest(name=name):
                train_X = torch.cat(
                    [torch.tensor(hierarchical_vals), torch.rand(4, 6)], dim=-1
                )
                train_Y = torch.randn(4, 1)
                model = HierarchicalConditionalKernelGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    hierarchical_dependencies=hierarchical_deps,
                )
                self.assertIsInstance(model.covar_module, HierarchicalConditionalKernel)
                posterior = model.posterior(train_X)
                self.assertEqual(posterior.covariance_matrix.shape, (4, 4))
                self.assertEqual(posterior.variance.shape, (4, 1))

                # Exercise construct_inputs against a SupervisedDataset that uses
                # these same tensors, and verify the returned kwargs build a
                # structurally identical model.
                dataset = SupervisedDataset(
                    X=train_X,
                    Y=train_Y,
                    feature_names=[f"x{i}" for i in range(train_X.shape[-1])],
                    outcome_names=["y"],
                )
                inputs = HierarchicalConditionalKernelGP.construct_inputs(
                    training_data=dataset,
                    hierarchical_dependencies=hierarchical_deps,
                    eval_hierarchical_features=False,
                    separate_hierarchical_features=False,
                    use_saas_prior=False,
                    use_outputscale=False,
                )
                self.assertAllClose(inputs["train_X"], train_X)
                self.assertAllClose(inputs["train_Y"], train_Y)
                self.assertEqual(inputs["hierarchical_dependencies"], hierarchical_deps)
                self.assertEqual(inputs["eval_hierarchical_features"], False)
                self.assertEqual(inputs["separate_hierarchical_features"], False)
                self.assertEqual(inputs["use_saas_prior"], False)
                self.assertEqual(inputs["use_outputscale"], False)
                # Make sure the model construction succeeds with these inputs.
                HierarchicalConditionalKernelGP(
                    **inputs, outcome_transform=Standardize(m=1)
                )

    def test_with_input_transforms(self) -> None:
        """Test model creation with input transforms, focusing on correct
        dependency transforms and kernel evaluations.
        """
        hierarchical_dependencies = {
            0: {0: [1], 1: [8], 2: [3, 9]},
            1: {-2: [4], 3: [5]},
            2: {0: [6], 1: [7]},
        }
        train_X = torch.cat(
            [torch.tensor([[0, -2, 0], [1, 3, 1], [2, 0, 0]]), torch.rand(3, 7)], dim=-1
        )
        bounds = torch.tensor(
            [
                [0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 3, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        model = HierarchicalConditionalKernelGP(
            train_X=train_X,
            train_Y=torch.randn(3, 1),
            hierarchical_dependencies=hierarchical_dependencies,
            input_transform=Normalize(d=10, bounds=bounds),
        )
        expected_dependencies = {
            0: {0.0: [1], 1.0 / 3.0: [8], 2.0 / 3.0: [3, 9]},
            1: {0.0: [4], 1.0: [5]},
            2: {0.0: [6], 1.0: [7]},
        }
        for idx, expected_value_to_deps in expected_dependencies.items():
            value_to_deps = model.covar_module.hierarchical_dependencies[idx]
            for (expected_value, expected_deps), (value, deps) in zip(
                expected_value_to_deps.items(), value_to_deps.items()
            ):
                # Use the same comparison used within the kernel.
                self.assertTrue(
                    _row_equal(torch.tensor(value), torch.tensor(expected_value))
                )
                self.assertEqual(deps, expected_deps)
        # Check that the kernel evaluates correctly.
        # Using X[..., 0] = 1.0 here since division by 3 causes floating point errors.
        X = train_X[1:2]
        with (
            mock.patch(
                "botorch.models.hierarchical.conditional_kernel_gp._row_equal",
                wraps=_row_equal,
            ) as mock_row_equal,
            gpytorch.settings.lazily_evaluate_kernels(False),
        ):
            model.forward(X)
        # Row equal should evaluate to following, twice for each value:
        expected_res = [
            True,  # base
            False,  # x0=0
            False,  # x0=2/3
            False,  # x0=0, x1=0
            False,  # x0=0, x1=1
            False,  # x2=0
            True,  # x2=1
            True,  # x0=1/3
        ]
        for i, call_args in enumerate(mock_row_equal.call_args_list):
            res = bool(_row_equal(*call_args.args, **call_args.kwargs))
            self.assertEqual(res, expected_res[i // 2])


class TestHierarchicalConditionalKernelMultiTaskGP(BotorchTestCase):
    def _get_multitask_data(
        self,
        n_per_task: int = 5,
        n_tasks: int = 2,
        dim_non_task: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate multitask training data with hierarchical structure."""
        torch.manual_seed(42)
        data_list = []
        for task in range(n_tasks):
            X_task = torch.rand(n_per_task, dim_non_task)
            X_task[:, 0] = torch.randint(0, 2, (n_per_task,)).float()
            task_col = torch.full((n_per_task, 1), float(task))
            data_list.append(torch.cat([X_task, task_col], dim=-1))
        train_X = torch.cat(data_list, dim=0)
        train_Y = torch.randn(n_tasks * n_per_task, 1)
        return train_X, train_Y

    def test_model(self):
        """Test model instantiation, forward pass, posterior, and various options."""
        hierarchical_deps = {0: {0: [1], 1: [2]}}
        train_X, train_Y = self._get_multitask_data(n_tasks=3)

        # Basic instantiation and structure checks
        model = HierarchicalConditionalKernelMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=-1,
            hierarchical_dependencies=hierarchical_deps,
        )
        self.assertEqual(model.num_tasks, 3)
        self.assertEqual(model.num_non_task_features, 4)
        self.assertEqual(model._num_outputs, 3)
        self.assertEqual(model._task_feature, 4)
        self.assertIsInstance(
            model.covar_module.kernels[0], HierarchicalConditionalKernel
        )

        # Forward pass
        output = model(train_X)
        self.assertEqual(output.mean.shape, (15,))
        self.assertEqual(output.covariance_matrix.shape, (15, 15))

        # Posterior
        # Only get posterior for the specified task.
        self.assertEqual(model.posterior(train_X[:3]).mean.shape, (3, 1))
        # Output for all tasks when evaluated without a task feature.
        self.assertEqual(model.posterior(train_X[:3, :-1]).mean.shape, (3, 3))

        # Test with fixed noise
        model_fixed = HierarchicalConditionalKernelMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=-1,
            hierarchical_dependencies=hierarchical_deps,
            train_Yvar=torch.full_like(train_Y, 0.01),
        )
        self.assertIsInstance(model_fixed.likelihood, FixedNoiseGaussianLikelihood)
        self.assertEqual(model_fixed.posterior(train_X[:2]).mean.shape, (2, 1))

        # Test with output_tasks and rank
        train_X_3, train_Y_3 = self._get_multitask_data(n_tasks=3)
        model_opts = HierarchicalConditionalKernelMultiTaskGP(
            train_X=train_X_3,
            train_Y=train_Y_3,
            task_feature=-1,
            hierarchical_dependencies=hierarchical_deps,
            output_tasks=[0, 2],
            rank=2,
        )
        self.assertEqual(model_opts._num_outputs, 2)
        self.assertEqual(model_opts._output_tasks, [0, 2])
        self.assertEqual(model_opts._rank, 2)
        # Should get output for all output tasks when evaluated without a task feature.
        self.assertEqual(model_opts.posterior(train_X[:2, :-1]).mean.shape, (2, 2))

    def test_construct_inputs(self):
        """Test construct_inputs class method with MultiTaskDataset."""
        hierarchical_deps = {0: {0: [1], 1: [2]}}
        torch.manual_seed(42)
        dim_non_task = 4
        n_per_task = 5
        train_X, train_Y = self._get_multitask_data(
            n_per_task=n_per_task, dim_non_task=dim_non_task
        )

        # Create individual datasets for each task (without task feature column)
        datasets = []
        for task in range(2):
            datasets.append(
                SupervisedDataset(
                    X=train_X[task * n_per_task : (task + 1) * n_per_task],
                    Y=train_Y[task * n_per_task : (task + 1) * n_per_task],
                    feature_names=[f"x{i}" for i in range(dim_non_task)] + ["task"],
                    outcome_names=[f"y{task}"],
                )
            )

        mt_dataset = MultiTaskDataset(
            datasets=datasets, target_outcome_name="y0", task_feature_index=-1
        )

        inputs = HierarchicalConditionalKernelMultiTaskGP.construct_inputs(
            training_data=mt_dataset,
            task_feature=-1,
            hierarchical_dependencies=hierarchical_deps,
            eval_hierarchical_features=False,
            use_saas_prior=True,
            rank=1,
        )
        self.assertAllClose(inputs["train_X"], train_X)
        self.assertAllClose(inputs["train_Y"], train_Y)
        self.assertEqual(inputs["task_feature"], -1)
        self.assertEqual(inputs["hierarchical_dependencies"], hierarchical_deps)
        self.assertEqual(inputs["eval_hierarchical_features"], False)
        self.assertEqual(inputs["use_saas_prior"], True)
        self.assertEqual(inputs["rank"], 1)
        # Make sure the model construction succeeds.
        HierarchicalConditionalKernelMultiTaskGP(**inputs)
