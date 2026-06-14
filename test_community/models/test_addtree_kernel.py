# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tests for the AddTree kernel module."""

import math

import numpy as np
import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.addtree.encoding import encode
from botorch_community.models.addtree.kernels import (
    AddTreeDeltaKernel,
    build_addtree_kernel,
)
from botorch_community.models.addtree.space import AddTreeSpace
from gpytorch.kernels import RBFKernel


def _binary_with_root_continuous_spec() -> dict:
    """Tree with continuous params at every level (tests broad coverage)."""
    return {
        "name": "root",
        "continuous": [
            {"name": "a", "lo": 0.0, "hi": 1.0},
            {"name": "b", "lo": 0.0, "hi": 1.0},
        ],
        "choices": [
            {
                "name": "L1",
                "options": {
                    "left": {
                        "name": "L1_left",
                        "continuous": [
                            {"name": "x", "lo": 0.0, "hi": 1.0},
                            {"name": "y", "lo": 0.0, "hi": 1.0},
                        ],
                    },
                    "right": {
                        "name": "L1_right",
                        "continuous": [
                            {"name": "x", "lo": 0.0, "hi": 1.0},
                            {"name": "y", "lo": 0.0, "hi": 1.0},
                            {"name": "z", "lo": 0.0, "hi": 1.0},
                        ],
                    },
                },
            }
        ],
    }


class TestAddTreeDeltaKernel(BotorchTestCase):
    def test_basic_equality(self) -> None:
        kern = AddTreeDeltaKernel(active_dims=(0,))
        x1 = torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float64)
        K = kern(x1).to_dense()
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(K, expected))

    def test_diag(self) -> None:
        kern = AddTreeDeltaKernel(active_dims=(0,))
        x1 = torch.tensor([[0.0], [1.0], [-1.0]], dtype=torch.float64)
        K_diag = kern(x1, diag=True).to_dense()
        self.assertTrue(torch.allclose(K_diag, torch.ones(3, dtype=torch.float64)))

    def test_active_dims_slicing(self) -> None:
        kern = AddTreeDeltaKernel(active_dims=(1,))
        x1 = torch.tensor([[0.0, 5.0], [1.0, 5.0], [9.0, 6.0]], dtype=torch.float64)
        K = kern(x1).to_dense()
        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(K, expected))

    def test_no_learnable_parameters(self) -> None:
        kern = AddTreeDeltaKernel(active_dims=(0,))
        self.assertEqual(len(list(kern.parameters())), 0)


class TestBuildAddTreeKernel(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.set_default_dtype(torch.float64)
        self.space = AddTreeSpace.from_dict(_binary_with_root_continuous_spec())

    def test_kernel_shape_and_psd(self) -> None:
        kern = build_addtree_kernel(self.space)

        rng = np.random.default_rng(42)
        params_list = [
            {
                "L1": "left",
                "root.a": rng.random(),
                "root.b": rng.random(),
                "L1_left.x": rng.random(),
                "L1_left.y": rng.random(),
            },
            {
                "L1": "left",
                "root.a": rng.random(),
                "root.b": rng.random(),
                "L1_left.x": rng.random(),
                "L1_left.y": rng.random(),
            },
            {
                "L1": "right",
                "root.a": rng.random(),
                "root.b": rng.random(),
                "L1_right.x": rng.random(),
                "L1_right.y": rng.random(),
                "L1_right.z": rng.random(),
            },
            {
                "L1": "right",
                "root.a": rng.random(),
                "root.b": rng.random(),
                "L1_right.x": rng.random(),
                "L1_right.y": rng.random(),
                "L1_right.z": rng.random(),
            },
        ]
        X = torch.stack([encode(self.space, p) for p in params_list])
        K = kern(X).to_dense().detach()
        self.assertEqual(K.shape, (len(params_list), len(params_list)))
        self.assertTrue(torch.allclose(K, K.T, atol=1e-12))
        torch.linalg.cholesky(K + 1e-6 * torch.eye(len(params_list), dtype=K.dtype))

    def test_diagonal_dominates(self) -> None:
        # Same encoded point twice -> K[0,0] = outputscale * (number of nodes)
        # because every Delta is 1 and every RBF on identical data is 1.
        kern = build_addtree_kernel(self.space)
        p = {
            "L1": "left",
            "root.a": 0.5,
            "root.b": 0.6,
            "L1_left.x": 0.7,
            "L1_left.y": 0.8,
        }
        v = encode(self.space, p)
        X = torch.stack([v, v])
        K = kern(X).to_dense().detach()
        outputscale = kern.outputscale.item()
        # 3 BFS nodes: root, L1_left, L1_right; for two identical points
        # the kernel sums Delta*RBF for each node. For path "left", node
        # L1_right has its data slots == sentinel for both rows
        # and its delta is 1 (sentinel == sentinel), and its RBF returns
        # 1 (data is identical). So K[0,0] == outputscale * 3.
        self.assertAlmostEqual(K[0, 0].item(), 3.0 * outputscale, places=5)

    def test_path_separation(self) -> None:
        # Two different paths should have lower similarity than self-similarity.
        kern = build_addtree_kernel(self.space)
        p_left = {
            "L1": "left",
            "root.a": 0.5,
            "root.b": 0.6,
            "L1_left.x": 0.7,
            "L1_left.y": 0.8,
        }
        p_right = {
            "L1": "right",
            "root.a": 0.5,
            "root.b": 0.6,
            "L1_right.x": 0.7,
            "L1_right.y": 0.8,
            "L1_right.z": 0.9,
        }
        X = torch.stack([encode(self.space, p_left), encode(self.space, p_right)])
        K = kern(X).to_dense().detach()
        self.assertGreater(K[0, 0].item(), K[0, 1].item())

    def test_lengthscale_bounds_match_paper(self) -> None:
        # The metric-bounds from the paper map to plain-lengthscale
        # interval [exp(-3.5), exp(2)].
        kern = build_addtree_kernel(self.space)
        rbf_kernels = [m for m in kern.modules() if isinstance(m, RBFKernel)]
        self.assertGreater(len(rbf_kernels), 0)
        constraint = rbf_kernels[0].raw_lengthscale_constraint
        self.assertAlmostEqual(
            constraint.lower_bound.item(),
            math.exp(-3.5),
            places=5,
        )
        self.assertAlmostEqual(
            constraint.upper_bound.item(),
            math.exp(2.0),
            places=5,
        )
