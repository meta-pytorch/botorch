#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

from inspect import getsource, getsourcefile
from typing import Any

from multipledispatch.dispatcher import (
    Dispatcher as MDDispatcher,
    MDNotImplementedError,  # trivial subclass of NotImplementedError
    str_signature,
)


def type_bypassing_encoder(arg: Any) -> type:
    # Allow type variables to be passed as pre-encoded arguments
    return arg if isinstance(arg, type) else type(arg)


class Dispatcher(MDDispatcher):
    r"""Clearing house for multiple dispatch functionality. This class extends
    `<multipledispatch.Dispatcher>` by: (i) generalizing the argument encoding
    convention during method lookup, (ii) implementing `__getitem__` as a dedicated
    method lookup function.
    """

    def __init__(
        self,
        name: str,
        doc: str | None = None,
        encoder: Callable[Any, type] = type,
    ) -> None:
        """
        Args:
            name: A string identifier for the `Dispatcher` instance.
            doc: A docstring for the multiply dispatched method(s).
            encoder: A callable that individually transforms the arguments passed
                at runtime in order to construct the key used for method lookup as
                `tuple(map(encoder, args))`. Defaults to `type`.
        """
        super().__init__(name=name, doc=doc)
        self._encoder = encoder

    def __getitem__(
        self,
        args: Any | None = None,
        types: tuple[type] | None = None,
    ) -> Callable:
        r"""Method lookup.

        Args:
            args: A set of arguments that act as identifiers for a stored method.
            types: A tuple of types that encodes `args`.

        Returns:
            A callable corresponding to the given `args` or `types`.
        """
        if types is None:
            if args is None:
                raise RuntimeError("One of `args` or `types` must be provided.")
            types = self.encode_args(args)
        elif args is not None:
            raise RuntimeError("Only one of `args` or `types` may be provided.")

        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if not func:
                msg = f"{self.name}: <{', '.join(cls.__name__ for cls in types)}"
                raise NotImplementedError(f"Could not find signature for {msg}")
            self._cache[types] = func
        return func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""Multiply dispatches a call to a collection of methods.

        Args:
            args: A set of arguments that act as identifiers for a stored method.
            kwargs: Optional keyword arguments passed to the retrieved method.

        Returns:
            The result of evaluating `func(*args, **kwargs)`, where `func` is
            the function obtained via method lookup.
        """
        types = self.encode_args(args)
        func = self.__getitem__(types=types)
        try:
            return func(*args, **kwargs)
        except MDNotImplementedError:
            # Traverses registered methods in order, yields whenever a match is found
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first, same as self.__getitem__(types=types)
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass

            raise NotImplementedError(
                f"Matching functions for {self.name:s}: {str_signature(types):s} "
                "found, but none completed successfully"
            )

    def dispatch(self, *types: type) -> Callable:
        r"""Method lookup strategy. Checks for an exact match before traversing
        the set of registered methods according to the current ordering.

        Args:
            types: A tuple of types that gets compared with the signatures
                of registered methods to determine compatibility.

        Returns:
            The first method encountered with a matching signature.
        """
        if types in self.funcs:
            return self.funcs[types]
        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def encode_args(self, args: Any) -> tuple[type]:
        r"""Converts arguments into a tuple of types used during method lookup."""
        return tuple(map(self.encoder, args if isinstance(args, tuple) else (args,)))

    def _help(self, *args: Any) -> str:
        r"""Returns the retrieved method's docstring."""
        return self.dispatch(*self.encode_args(args)).__doc__

    def help(self, *args: Any, **kwargs: Any) -> None:
        r"""Prints the retrieved method's docstring."""
        print(self._help(*args))

    def _source(self, *args: Any) -> str:
        r"""Returns the retrieved method's source types as a string."""
        func = self.dispatch(*self.encode_args(args))
        if not func:
            raise TypeError("No function found")
        return f"File: {getsourcefile(func)}\n\n{getsource(func)}"

    def source(self, *args, **kwargs) -> None:
        r"""Prints the retrieved method's source types."""
        print(self._source(*args))

    @property
    def encoder(self) -> Callable[Any, type]:
        return self._encoder
