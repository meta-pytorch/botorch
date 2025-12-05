#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Type, TypeVar

T = TypeVar("T")  # generic type variable
NoneType = type(None)  # stop gap for the return of NoneType in 3.10


def cast(typ: Type[T], obj: Any, optional: bool = False) -> T:
    """Cast an object to a type, optionally allowing None.

    Args:
        typ: Type to cast to
        obj: Object to cast
        optional: Whether to allow None

    Returns:
        Cast object
    """
    if (optional and obj is None) or isinstance(obj, typ):
        return obj

    return typ(obj)


class _DefaultType(type):
    r"""
    Private class whose sole instance `DEFAULT` is a special indicator
    representing that a default value should be assigned to an argument.
    Typically used in cases where `None` is an allowed argument.
    """


DEFAULT = _DefaultType("DEFAULT", (), {})


class _MissingType(type):
    r"""
    Private class whose sole instance `MISSING` is a special indicator
    representing that an optional argument has not been passed. Typically used
    in cases where `None` is an allowed argument.
    """


MISSING = _MissingType("MISSING", (), {})
