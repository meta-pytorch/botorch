#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator

from functools import lru_cache
from numbers import Number

import torch
from torch import Tensor


@lru_cache(maxsize=None)
def get_constants(
    values: Number | Iterator[Number],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, ...]:
    r"""Returns scalar-valued Tensors containing each of the given constants.
    Used to expedite tensor operations involving scalar arithmetic. Note that
    the returned Tensors should not be modified in-place."""
    if isinstance(values, Number):
        return torch.full((), values, dtype=dtype, device=device)

    return tuple(torch.full((), val, dtype=dtype, device=device) for val in values)


def get_constants_like(
    values: Number | Iterator[Number],
    ref: Tensor,
) -> Tensor | Iterator[Tensor]:
    return get_constants(values, device=ref.device, dtype=ref.dtype)
