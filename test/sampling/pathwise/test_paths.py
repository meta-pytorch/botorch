#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.paths import PathDict, PathList, SamplePath
from botorch.utils.testing import BotorchTestCase
from torch.nn import ModuleDict, ModuleList


class IdentityPath(SamplePath):
    ensemble_as_batch: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        self.ensemble_as_batch = ensemble_as_batch


class TestGenericPaths(BotorchTestCase):
    def test_path_dict(self):
        with self.assertRaisesRegex(UnsupportedError, "preceded by a `reducer`"):
            PathDict(output_transform="foo")

        A = IdentityPath()
        B = IdentityPath()

        # Test __init__
        module_dict = ModuleDict({"0": A, "1": B})
        path_dict = PathDict(paths={"0": A, "1": B})
        self.assertTrue(path_dict.paths is not module_dict)

        path_dict = PathDict(paths=module_dict)
        self.assertIs(path_dict.paths, module_dict)

        # Test __call__
        x = torch.rand(3, device=self.device)
        output = path_dict(x)
        self.assertIsInstance(output, dict)
        self.assertTrue(x.equal(output.pop("0")))
        self.assertTrue(x.equal(output.pop("1")))
        self.assertTrue(not output)

        path_dict.reducer = torch.stack
        output = path_dict(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2,) + x.shape)
        self.assertTrue(output.eq(x).all())

        A.set_ensemble_as_batch(True)
        self.assertTrue(A.ensemble_as_batch)

        A.set_ensemble_as_batch(False)
        self.assertFalse(A.ensemble_as_batch)

        # Test `dict`` methods
        self.assertEqual(len(path_dict), 2)
        for key, val, (key_0, val_0), (key_1, val_1), key_2 in zip(
            path_dict,
            path_dict.values(),
            path_dict.items(),
            path_dict.paths.items(),
            path_dict.keys(),
        ):
            self.assertEqual(1, len({key, key_0, key_1, key_2}))
            self.assertEqual(1, len({val, val_0, val_1, path_dict[key]}))

        path_dict["1"] = A  # test __setitem__
        self.assertIs(path_dict.paths["1"], A)

        del path_dict["1"]  # test __delitem__
        self.assertEqual(("0",), tuple(path_dict))

    def test_path_list(self):
        with self.assertRaisesRegex(UnsupportedError, "preceded by a `reducer`"):
            PathList(output_transform="foo")

        # Test __init__
        A = IdentityPath()
        B = IdentityPath()
        module_list = ModuleList((A, B))
        path_list = PathList(paths=list(module_list))
        self.assertTrue(path_list.paths is not module_list)

        path_list = PathList(paths=module_list)
        self.assertIs(path_list.paths, module_list)

        # Test __call__
        x = torch.rand(3, device=self.device)
        output = path_list(x)
        self.assertIsInstance(output, list)
        self.assertTrue(x.equal(output.pop()))
        self.assertTrue(x.equal(output.pop()))
        self.assertTrue(not output)

        path_list.reducer = torch.stack
        output = path_list(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2,) + x.shape)
        self.assertTrue(output.eq(x).all())

        # Test ``list`` methods
        self.assertEqual(len(path_list), 2)
        for key, (path, path_0) in enumerate(zip(path_list, path_list.paths)):
            self.assertEqual(1, len({path, path_0, path_list[key]}))

        path_list[1] = A  # test __setitem__
        self.assertIs(path_list.paths[1], A)

        del path_list[1]  # test __delitem__
        self.assertEqual((A,), tuple(path_list))

    def test_generalized_linear_path_multi_dim(self):
        """Test GeneralizedLinearPath with multi-dimensional feature maps."""
        import torch
        from botorch.sampling.pathwise.features import FeatureMap
        from botorch.sampling.pathwise.paths import GeneralizedLinearPath

        # Create a mock feature map with 2D output
        class Mock2DFeatureMap(FeatureMap):
            def __init__(self):
                super().__init__()
                self.raw_output_shape = torch.Size([4, 3])  # 2D output
                self.batch_shape = torch.Size([])
                self.input_transform = None
                self.output_transform = None

            def forward(self, x):
                # Return a 2D feature tensor
                batch_shape = x.shape[:-1]
                return torch.randn(*batch_shape, *self.raw_output_shape)

        # Create path with 2D features
        feature_map = Mock2DFeatureMap()

        weight = torch.randn(3)  # Weight should match last dimension of features
        path = GeneralizedLinearPath(feature_map=feature_map, weight=weight)

        # Test forward pass - this should trigger einsum
        x = torch.rand(5, 2)  # batch_size x input_dim
        output = path(x)

        # Output should be reduced to 1D (batch_size,)
        self.assertEqual(output.shape, (5,))

        # Test with bias module
        class MockBias(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0])

        bias_module = MockBias()
        path_with_bias = GeneralizedLinearPath(
            feature_map=feature_map, weight=weight, bias_module=bias_module
        )
        output_with_bias = path_with_bias(x)
        self.assertEqual(output_with_bias.shape, (5,))
