#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for interfacing Numpy and Torch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


torch_to_numpy_dtype_dict = {
    torch.bool: bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def as_ndarray(
    values: Tensor, dtype: np.dtype | None = None, inplace: bool = True
) -> npt.NDArray:
    r"""Helper for going from torch.Tensor to numpy.ndarray.

    Args:
        values: Tensor to be converted to ndarray.
        dtype: Optional numpy.dtype for the converted tensor.
        inplace: Boolean indicating whether memory should be shared if possible.

    Returns:
        An ndarray with the same data as ``values``.
    """
    with torch.no_grad():
        out = values.cpu()  # maybe transfer to cpu

        # Determine whether or not to ``clone``
        if (
            # cond 1: are we not in ``inplace`` mode?
            not inplace
            # cond 2: did we already copy when calling ``cpu`` above?
            and out.device == values.device
            # cond 3: will we copy when calling ``astype`` below?
            and (dtype is None or out.dtype == torch_to_numpy_dtype_dict[dtype])
        ):
            out = out.clone()

        # Convert to ndarray and maybe cast to ``dtype``
        out = out.numpy()
        return out.astype(dtype, copy=False)


def get_bounds_as_ndarray(
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | Tensor | None, float | Tensor | None]],
) -> npt.NDArray | None:
    r"""Helper method for converting bounds into an ndarray.

    Args:
        parameters: A dictionary of parameters.
        bounds: A dictionary of (optional) lower and upper bounds.

    Returns:
        An ndarray of bounds.
    """
    inf = float("inf")
    full_size = sum(param.numel() for param in parameters.values())
    out = np.full((full_size, 2), (-inf, inf))
    index = 0
    for name, param in parameters.items():
        size = param.numel()
        if name in bounds:
            lower, upper = bounds[name]
            lower = -inf if lower is None else lower
            upper = inf if upper is None else upper
            if isinstance(lower, Tensor):
                lower = lower.cpu().numpy()
            if isinstance(upper, Tensor):
                upper = upper.cpu().numpy()
            out[index : index + size, 0] = lower
            out[index : index + size, 1] = upper
        index = index + size
    # If all bounds are +/- inf, return None.
    if np.isinf(out).all():
        out = None
    return out


def get_per_element_bounds(
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | None, float | None]],
    batch_shape: torch.Size,
) -> npt.NDArray | None:
    r"""Convert bounds to an ndarray for a single batch element's parameters.

    For batched models where all batch elements share the same parameter
    constraints, this extracts bounds for one element's worth of parameters.

    Args:
        parameters: A dictionary of batched parameter tensors, each with shape
            ``(*batch_shape, *trailing_shape)``.
        bounds: A dictionary of (optional) lower and upper bounds.
        batch_shape: The batch shape shared by all parameters.

    Returns:
        An ndarray of shape ``(per_element_size, 2)`` or None if all bounds
        are infinite.
    """
    inf = float("inf")
    batch_size = max(int(torch.Size(batch_shape).numel()), 1)
    per_element_size = sum(param.numel() // batch_size for param in parameters.values())
    out = np.full((per_element_size, 2), (-inf, inf))
    index = 0
    for name, param in parameters.items():
        size = param.numel() // batch_size
        if name in bounds:
            lower, upper = bounds[name]
            lower = -inf if lower is None else lower
            upper = inf if upper is None else upper
            out[index : index + size, 0] = lower
            out[index : index + size, 1] = upper
        index += size
    if np.isinf(out).all():
        return None
    return out
