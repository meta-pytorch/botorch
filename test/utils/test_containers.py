#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import patch

import torch
from botorch.utils.containers import BotorchContainer, DenseContainer, SliceContainer
from botorch.utils.testing import BotorchTestCase
from torch import Size


@dataclass
class BadContainer(BotorchContainer):
    def __call__(self) -> None:
        pass  # pragma: nocover

    def __eq__(self, other) -> bool:
        pass  # pragma: nocover

    @property
    def shape(self):
        pass  # pragma: nocover

    @property
    def device(self):
        pass  # pragma: nocover

    @property
    def dtype(self):
        pass  # pragma: nocover


class TestContainers(BotorchTestCase):
    def test_base(self):
        with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):
            BotorchContainer()

        with self.assertRaisesRegex(AttributeError, "Missing .* `event_shape`"):
            BadContainer()

        with patch.multiple(BotorchContainer, __abstractmethods__=set()):
            container = BotorchContainer()
            with self.assertRaises(NotImplementedError):
                container()
            with self.assertRaises(NotImplementedError):
                container.device
            with self.assertRaises(NotImplementedError):
                container.dtype
            with self.assertRaises(NotImplementedError):
                container.shape
            with self.assertRaises(NotImplementedError):
                container.__eq__(None)

    def test_dense(self):
        for values in (
            torch.rand(3, 2, dtype=torch.float16),
            torch.rand(5, 4, 3, 2, dtype=torch.float64),
        ):
            event_shape = values.shape[values.ndim // 2 :]

            # Test some invalid shapes
            with self.assertRaisesRegex(ValueError, "Shape .* incompatible"):
                X = DenseContainer(values=values, event_shape=Size([3]))

            with self.assertRaisesRegex(ValueError, "Shape .* incompatible"):
                X = DenseContainer(values=values, event_shape=torch.Size([2, 3]))

            # Test some basic propeties
            X = DenseContainer(values=values, event_shape=event_shape)
            self.assertEqual(X.device, values.device)
            self.assertEqual(X.dtype, values.dtype)

            # Test ``shape`` property
            self.assertEqual(X.shape, values.shape)

            # Test ``__eq__``
            self.assertEqual(X, DenseContainer(values=values, event_shape=event_shape))
            self.assertNotEqual(
                X,
                DenseContainer(values=torch.rand_like(values), event_shape=event_shape),
            )

            # Test ``__call__``
            self.assertTrue(X().equal(values))

            # Test ``clone``
            self.assertEqual(X.clone(), X)

    def test_slice(self):
        for arity in (2, 4):
            for vals in (
                torch.rand(8, 2, dtype=torch.float16),
                torch.rand(8, 3, 2, dtype=torch.float16),
            ):
                indices = torch.stack(
                    [torch.randperm(len(vals))[:arity] for _ in range(4)]
                )
                event_shape = (arity * vals.shape[1],) + vals.shape[2:]
                with self.assertRaisesRegex(ValueError, "Shapes .* incompatible"):
                    SliceContainer(
                        values=vals,
                        indices=indices,
                        event_shape=(10 * event_shape[0],) + event_shape[1:],
                    )

                # Test some basic propeties
                groups = SliceContainer(
                    values=vals, indices=indices, event_shape=event_shape
                )
                self.assertEqual(groups.device, vals.device)
                self.assertEqual(groups.dtype, vals.dtype)
                self.assertEqual(groups.shape, groups().shape)

                # Test ``__eq__``
                self.assertEqual(
                    groups,
                    SliceContainer(
                        values=vals, indices=indices, event_shape=event_shape
                    ),
                )
                self.assertNotEqual(
                    groups,
                    SliceContainer(
                        values=torch.rand_like(vals),
                        indices=indices,
                        event_shape=event_shape,
                    ),
                )

                # Test ``__call__``
                dense = groups()
                index = int(torch.randint(high=len(dense), size=()))
                other = torch.cat([vals[i] for i in indices[index]])
                self.assertTrue(dense[index].equal(other))
