#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import redirect_stdout
from inspect import getsource, getsourcefile
from io import StringIO
from itertools import product
from unittest.mock import patch

from botorch.utils.dispatcher import Dispatcher, MDNotImplementedError
from botorch.utils.testing import BotorchTestCase


def _helper_test_source(val):
    """Helper method for testing ``Dispatcher._source``."""
    ...  # pragma: nocover


class TestDispatcher(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.dispatcher = Dispatcher(name="test")

    def test_encoder(self):
        self.assertEqual((int, str, list), self.dispatcher.encode_args((1, "a", [])))
        with patch.object(self.dispatcher, "_encoder", str.upper):
            self.assertEqual(("A", "B"), self.dispatcher.encode_args(("a", "b")))

    def test_getitem(self):
        with patch.dict(self.dispatcher.funcs, {}):
            self.dispatcher.add(signature=(int, str), func=lambda *_: None)

            args = 0, "a"
            types = self.dispatcher.encode_args(args)
            with self.assertRaisesRegex(RuntimeError, "One of `args` or `types`"):
                self.dispatcher.__getitem__(args=None, types=None)

            with self.assertRaisesRegex(RuntimeError, "Only one of `args` or `types`"):
                self.dispatcher.__getitem__(args=args, types=types)

            self.assertEqual(
                self.dispatcher[args], self.dispatcher.__getitem__(args=args)
            )

            self.assertEqual(
                self.dispatcher[args], self.dispatcher.__getitem__(types=types)
            )

    def test_register(self):
        signature = (int, float), (int, float)
        with patch.dict(self.dispatcher.funcs, {}):

            @self.dispatcher.register(*signature)
            def _pow(a: int, b: int):
                return a**b

            for type_a, type_b in product(*signature):
                args = type_a(2), type_b(3)
                self.assertEqual(self.dispatcher[args], _pow)

                retval = self.dispatcher(*args)
                test_type = float if (type_a is float or type_b is float) else int
                self.assertIs(type(retval), test_type)
                self.assertEqual(retval, test_type(8))

    def test_notImplemented(self):
        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            self.dispatcher[0]

        with self.assertRaisesRegex(NotImplementedError, "Could not find signature"):
            self.dispatcher(0)

    def test_inheritance(self):
        IntSubclass = type("IntSubclass", (int,), {})
        with patch.dict(self.dispatcher.funcs, {}):
            self.dispatcher.add(signature=(int,), func=lambda val: -val)
            self.assertEqual(self.dispatcher(IntSubclass(1)), -1)

    def test_MDNotImplementedError(self):
        Parent = type("Parent", (int,), {})
        Child = type("Child", (Parent,), {})
        with patch.dict(self.dispatcher.funcs, {}):

            @self.dispatcher.register(Parent)
            def _method_parent(val) -> str:
                if val < 0:
                    raise MDNotImplementedError  # defer to nothing
                return "parent"

            @self.dispatcher.register(Child)
            def _method_child(val) -> str:
                if val % 2:
                    return "child"
                raise MDNotImplementedError  # defer to parent

            self.assertEqual(self.dispatcher(Child(1)), "child")
            self.assertEqual(self.dispatcher(Child(2)), "parent")
            self.assertEqual(self.dispatcher(Child(-1)), "child")
            with self.assertRaisesRegex(NotImplementedError, "none completed"):
                self.dispatcher(Child(-2))

    def test_help(self):
        with patch.dict(self.dispatcher.funcs, {}):

            @self.dispatcher.register(int)
            def _method(val) -> None:
                """docstring"""
                ...  # pragma: nocover

            self.assertEqual(self.dispatcher._help(0), "docstring")
            with redirect_stdout(StringIO()) as buffer:
                self.dispatcher.help(0)
            self.assertEqual(buffer.getvalue().rstrip(), "docstring")

    def test_source(self):
        source = (
            f"File: {getsourcefile(_helper_test_source)}"
            f"\n\n{getsource(_helper_test_source)}"
        )
        with patch.dict(self.dispatcher.funcs, {}):
            self.dispatcher.add(signature=(int,), func=_helper_test_source)
            self.assertEqual(self.dispatcher._source(0), source)
            with redirect_stdout(StringIO()) as buffer:
                self.dispatcher.source(0)

            # buffer.getvalue() has two newlines at the end, one due to ``print``
            self.assertEqual(buffer.getvalue()[:-1], source)
            with self.assertRaisesRegex(TypeError, "No function found"):
                self.dispatcher._source(0.5)
