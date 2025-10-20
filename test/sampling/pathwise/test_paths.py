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
        with self.assertRaisesRegex(UnsupportedError, "must be preceded by a join"):
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

        path_dict.join = torch.stack
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
        with self.assertRaisesRegex(UnsupportedError, "must be preceded by a join"):
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

        path_list.join = torch.stack
        output = path_list(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2,) + x.shape)
        self.assertTrue(output.eq(x).all())

        # Test `list` methods
        self.assertEqual(len(path_list), 2)
        for key, (path, path_0) in enumerate(zip(path_list, path_list.paths)):
            self.assertEqual(1, len({path, path_0, path_list[key]}))

        path_list[1] = A  # test __setitem__
        self.assertIs(path_list.paths[1], A)

        del path_list[1]  # test __delitem__
        self.assertEqual((A,), tuple(path_list))
