#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class _DefaultType(type):
    r"""
    Private class whose sole instance ``DEFAULT`` is as a special indicator
    representing that a default value should be assigned to an argument.
    Typically used in cases where ``None`` is an allowed argument.
    """


DEFAULT = _DefaultType("DEFAULT", (), {})
