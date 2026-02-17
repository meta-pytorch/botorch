#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.sampling.pathwise.utils import (
    TInputTransform,
    TOutputTransform,
    TransformedModuleMixin,
)
from gpytorch.kernels import Kernel
from linear_operator.operators import LinearOperator
from torch import Size, Tensor
from torch.nn import Module


class FeatureMap(TransformedModuleMixin, Module):
    num_outputs: int
    batch_shape: Size
    input_transform: TInputTransform | None
    output_transform: TOutputTransform | None


class KernelEvaluationMap(FeatureMap):
    r"""A feature map defined by centering a kernel at a set of points."""

    def __init__(
        self,
        kernel: Kernel,
        points: Tensor,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a KernelEvaluationMap instance:

        .. code-block:: text

            feature_map(x) = output_transform(kernel(input_transform(x), points)).

        Args:
            kernel: The kernel :math:``k`` used to define the feature map.
            points: A tensor passed as the kernel's second argument.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        try:
            torch.broadcast_shapes(points.shape[:-2], kernel.batch_shape)
        except RuntimeError:
            raise RuntimeError(
                f"Shape mismatch: {points.shape=}, but {kernel.batch_shape=}."
            )

        super().__init__()
        self.kernel = kernel
        self.points = points
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor) -> Tensor | LinearOperator:
        return self.kernel(x, self.points)

    @property
    def num_outputs(self) -> int:
        if self.output_transform is None:
            return self.points.shape[-2]

        canary = torch.empty(
            1, self.points.shape[-2], device=self.points.device, dtype=self.points.dtype
        )
        return self.output_transform(canary).shape[-1]

    @property
    def batch_shape(self) -> Size:
        return self.kernel.batch_shape


class KernelFeatureMap(FeatureMap):
    r"""Representation of a kernel :math:``k: \mathcal{X}^2 \to \mathbb{R}`` as an
    n-dimensional feature map :math:``\phi: \mathcal{X} \to \mathbb{R}^n`` satisfying:
    :math:``k(x, x') ≈ \phi(x)^\top \phi(x')``.
    """

    def __init__(
        self,
        kernel: Kernel,
        weight: Tensor,
        bias: Tensor | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a KernelFeatureMap instance:

        .. code-block:: text

            feature_map(x) = output_transform(input_transform(x)^{T} weight + bias).

        Args:
            kernel: The kernel :math:``k`` used to define the feature map.
            weight: A tensor of weights used to linearly combine the module's inputs.
            bias: A tensor of biases to be added to the linearly combined inputs.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
        """
        super().__init__()
        self.kernel = kernel
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.weight = weight
        self.bias = bias
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose(-2, -1)
        return out if self.bias is None else out + self.bias

    @property
    def num_outputs(self) -> int:
        if self.output_transform is None:
            return self.weight.shape[-2]

        canary = torch.empty(
            self.weight.shape[-2], device=self.weight.device, dtype=self.weight.dtype
        )
        return self.output_transform(canary).shape[-1]

    @property
    def batch_shape(self) -> Size:
        return self.kernel.batch_shape
