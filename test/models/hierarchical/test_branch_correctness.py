#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless

import torch
from botorch.models.hierarchical.conditional_kernel_gp import (
    _transform_hierarchical_dependencies,
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from botorch.models.hierarchical.utils import get_index_to_path
from botorch.models.transforms.input import Normalize
from botorch.utils.testing import BotorchTestCase


class TestHierarchicalBranchCorrectness(BotorchTestCase):
    @skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fitting_uses_training_values_without_mutating_transform(self):
        X = torch.tensor([[2.0, 0.2], [4.0, 0.8]], device="cuda", dtype=torch.double)
        transform = Normalize(d=2).to(X)
        original_bounds = transform.bounds.clone()
        self.assertEqual(
            _transform_hierarchical_dependencies(X, {0: {2.0: [1]}}, transform),
            {0: {0.0: [1]}},
        )
        self.assertAllClose(transform.bounds, original_bounds)
        model = HierarchicalConditionalKernelGP(
            X,
            X[..., 1:],
            {0: {2.0: [1]}},
            input_transform=transform,
            use_saas_prior=False,
            outcome_transform=None,
        )
        self.assertEqual(model.covar_module.hierarchical_dependencies, {0: {0.0: [1]}})
        self.assertTrue(torch.isfinite(model.posterior(X[:1]).mean).all())

    def test_fixed_transform_bounds_can_retain_a_graph(self):
        bounds = torch.tensor(
            [[2.0, 0.0], [4.0, 1.0]], dtype=torch.double, requires_grad=True
        )
        transform = Normalize(d=2, bounds=bounds * 1.0)
        X = bounds.new_tensor([[2.0, 0.2], [4.0, 0.8]])
        self.assertEqual(
            _transform_hierarchical_dependencies(X, {0: {2.0: [1]}}, transform),
            {0: {0.0: [1]}},
        )

    def test_distinct_parents_can_have_close_categories(self):
        kernel = HierarchicalConditionalKernel(
            dim=4,
            hierarchical_dependencies={0: {1.0: [1]}, 2: {1.000001: [3]}},
            use_saas_prior=False,
        )
        self.assertEqual(sorted(sum(kernel.partition, [])), list(range(4)))

    def test_fitted_normalize_defines_branch_values(self):
        X = torch.tensor([[2.0, 0.2], [4.0, 0.8], [2.0, 0.5]], dtype=torch.double)
        model = HierarchicalConditionalKernelGP(
            X,
            X[..., 1:],
            {0: {2.0: [1]}},
            input_transform=Normalize(d=2),
            use_saas_prior=False,
            outcome_transform=None,
        )
        self.assertEqual(model.covar_module.hierarchical_dependencies, {0: {0.0: [1]}})
        reference = HierarchicalConditionalKernel(
            dim=2,
            hierarchical_dependencies={0: {0.0: [1]}},
            use_saas_prior=False,
        ).double()
        reference.load_state_dict(model.covar_module.state_dict())
        transformed = model.transform_inputs(X)
        self.assertAllClose(
            model.covar_module(transformed).to_dense(),
            reference(transformed).to_dense(),
        )

    def test_multitask_dependency_columns(self):
        for task_feature in (0, 1, 2, -1):
            with self.subTest(task_feature=task_feature):
                basic = torch.tensor(
                    [[0.1, 2.0], [0.5, 4.0], [0.4, 2.0], [0.8, 4.0]], dtype=torch.double
                )
                task = basic.new_tensor([[0.0], [1.0], [0.0], [1.0]])
                index = task_feature % 3
                X = torch.cat((basic[:, :index], task, basic[:, index:]), -1)
                model = HierarchicalConditionalKernelMultiTaskGP(
                    train_X=X,
                    train_Y=basic[:, :1],
                    task_feature=task_feature,
                    hierarchical_dependencies={1: {2.0: [0]}},
                    input_transform=Normalize(
                        d=3, indices=[i for i in range(3) if i != index]
                    ),
                    use_saas_prior=False,
                    outcome_transform=None,
                )
                self.assertEqual(
                    model.covar_module.kernels[0].hierarchical_dependencies,
                    {1: {0.0: [0]}},
                )

    def test_empty_dependencies_with_normalize(self):
        X = torch.rand(4, 2, dtype=torch.double)
        model = HierarchicalConditionalKernelGP(
            X,
            X[..., :1],
            {},
            input_transform=Normalize(d=2),
            use_saas_prior=False,
        )
        self.assertTrue(torch.isfinite(model.posterior(X[:1]).mean).all())

    def test_closed_cycles_and_invalid_indices_are_rejected(self):
        for dim, dependencies in (
            (3, {0: {0: [1]}, 1: {0: [0]}}),
            (2, {0: {0: [0]}}),
            (2, {0: {0: [2]}}),
            (2, {-1: {0: [1]}}),
        ):
            with self.subTest(dim=dim, dependencies=dependencies):
                with self.assertRaises(ValueError):
                    get_index_to_path(dim, dependencies)

    def test_deep_forests_do_not_require_python_recursion(self):
        dim = 1100
        paths = get_index_to_path(dim, {i: {0: [i + 1]} for i in range(dim - 1)})
        self.assertEqual(len(paths[dim - 1]), dim - 1)
