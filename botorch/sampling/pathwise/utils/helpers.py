#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from sys import maxsize
from typing import Callable, Iterable, Iterator, Tuple, Type, TypeVar

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling.pathwise.utils.mixins import TransformedModuleMixin
from botorch.sampling.pathwise.utils.transforms import (
    ChainedTransform,
    OutcomeUntransformer,
    TensorTransform,
)
from botorch.utils.types import MISSING
from gpytorch import kernels
from gpytorch.kernels.kernel import Kernel
from linear_operator import LinearOperator
from torch import Size, Tensor

TKernel = TypeVar("TKernel", bound=Kernel)
INF_DIM_KERNELS: Tuple[Type[Kernel], ...] = (
    kernels.MaternKernel,
    kernels.RBFKernel,
    kernels.MultitaskKernel,
)


def kernel_instancecheck(
    kernel: Kernel,
    types: TKernel | Tuple[TKernel, ...],
    reducer: Callable[[Iterator[bool]], bool] = any,
    max_depth: int = maxsize,
) -> bool:
    """Check if a kernel is an instance of specified kernel type(s).

    Args:
        kernel: The kernel to check
        types: Single kernel type or tuple of kernel types to check against
        reducer: Function to reduce multiple boolean checks (default: any)
        max_depth: Maximum depth to search in kernel hierarchy

    Returns:
        bool: Whether kernel matches the specified type(s)
    """
    if isinstance(kernel, types):
        return True

    if max_depth == 0 or not isinstance(kernel, Kernel):
        return False

    return reducer(
        kernel_instancecheck(module, types, reducer, max_depth - 1)
        for module in kernel.modules()
        if module is not kernel and isinstance(module, Kernel)
    )


def is_finite_dimensional(kernel: Kernel, max_depth: int = maxsize) -> bool:
    """Check if a kernel has a finite-dimensional feature map.

    Args:
        kernel: The kernel to check
        max_depth: Maximum depth to search in kernel hierarchy

    Returns:
        bool: Whether kernel has finite-dimensional feature map
    """
    return not kernel_instancecheck(
        kernel, types=INF_DIM_KERNELS, reducer=any, max_depth=max_depth
    )


def sparse_block_diag(
    blocks: Iterable[Tensor],
    base_ndim: int = 2,
) -> Tensor:
    """Creates a sparse block diagonal tensor from a list of tensors.

    Args:
        blocks: Iterable of tensors to arrange diagonally
        base_ndim: Number of dimensions to treat as matrix dimensions

    Returns:
        Tensor: Sparse block diagonal tensor
    """
    device = next(iter(blocks)).device
    values = []
    indices = []
    shape = torch.zeros(base_ndim, 1, dtype=torch.long, device=device)
    batch_shapes = []

    for blk in blocks:
        batch_shapes.append(blk.shape[:-base_ndim])
        if isinstance(blk, LinearOperator):
            blk = blk.to_dense()

        _blk = (blk if blk.is_sparse else blk.to_sparse()).coalesce()
        values.append(_blk.values())

        idx = _blk.indices()
        idx[-base_ndim:, :] += shape
        indices.append(idx)
        for i, size in enumerate(blk.shape[-base_ndim:]):
            shape[i] += size

    return torch.sparse_coo_tensor(
        indices=torch.concat(indices, dim=-1),
        values=torch.concat(values),
        size=Size((*torch.broadcast_shapes(*batch_shapes), *shape.squeeze(-1))),
    )


def append_transform(
    module: TransformedModuleMixin,
    attr_name: str,
    transform: InputTransform | OutcomeTransform | TensorTransform,
) -> None:
    """Appends a transform to a module's transform chain.

    Args:
        module: Module to append transform to
        attr_name: Name of transform attribute
        transform: Transform to append
    """
    other = getattr(module, attr_name, None)
    if other is None:
        setattr(module, attr_name, transform)
    else:
        setattr(module, attr_name, ChainedTransform(other, transform))


def prepend_transform(
    module: TransformedModuleMixin,
    attr_name: str,
    transform: InputTransform | OutcomeTransform | TensorTransform,
) -> None:
    """Prepends a transform to a module's transform chain.

    Args:
        module: Module to prepend transform to
        attr_name: Name of transform attribute
        transform: Transform to prepend
    """
    other = getattr(module, attr_name, None)
    if other is None:
        setattr(module, attr_name, transform)
    else:
        setattr(module, attr_name, ChainedTransform(transform, other))


def untransform_shape(
    transform: TensorTransform | InputTransform | OutcomeTransform,
    shape: Size,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Size:
    """Gets the shape after applying an inverse transform.

    Args:
        transform: Transform to invert
        shape: Input shape
        device: Optional device for test tensor
        dtype: Optional dtype for test tensor

    Returns:
        Size: Shape after inverse transform
    """
    if transform is None:
        return shape

    test_case = torch.empty(shape, device=device, dtype=dtype)
    if isinstance(transform, OutcomeTransform):
        if not getattr(transform, "_is_trained", True):
            return shape
        result, _ = transform.untransform(test_case)
    elif isinstance(transform, InputTransform):
        result = transform.untransform(test_case)
    else:
        result = transform(test_case)

    return result.shape[-test_case.ndim :]


def get_kernel_num_inputs(
    kernel: Kernel,
    num_ambient_inputs: int | None = None,
    default: (int | None) | None = MISSING,
) -> int | None:
    if kernel.active_dims is not None:
        return len(kernel.active_dims)

    if kernel.ard_num_dims is not None:
        return kernel.ard_num_dims

    if num_ambient_inputs is None:
        if default is MISSING:
            raise ValueError(
                "`num_ambient_inputs` must be passed when `kernel.active_dims` and "
                "`kernel.ard_num_dims` are both None and no `default` has been defined."
            )
        return default
    return num_ambient_inputs


def get_input_transform(model: GPyTorchModel) -> InputTransform | None:
    r"""Returns a model's input_transform or None."""
    return getattr(model, "input_transform", None)


def get_output_transform(model: GPyTorchModel) -> OutcomeUntransformer | None:
    r"""Returns a wrapped version of a model's outcome_transform or None."""
    transform = getattr(model, "outcome_transform", None)
    if transform is None:
        return None

    return OutcomeUntransformer(transform=transform, num_outputs=model.num_outputs)
