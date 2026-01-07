#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
.. [rahimi2007random]
    A. Rahimi and B. Recht. Random features for large-scale kernel machines.
    Advances in Neural Information Processing Systems 20 (2007).

.. [sutherland2015error]
    D. J. Sutherland and J. Schneider. On the error of random Fourier features.
    arXiv preprint arXiv:1506.02785 (2015).
"""

from __future__ import annotations

from collections.abc import Callable

from typing import Any

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features.maps import KernelFeatureMap
from botorch.sampling.pathwise.utils import (
    ChainedTransform,
    FeatureSelector,
    InverseLengthscaleTransform,
    OutputscaleTransform,
    SineCosineTransform,
)
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.sampling import draw_sobol_normal_samples
from gpytorch import kernels
from gpytorch.kernels.kernel import Kernel
from torch import Size, Tensor
from torch.distributions import Gamma

TKernelFeatureMapGenerator = Callable[[Kernel, int, int], KernelFeatureMap]
GenKernelFeatures = Dispatcher("gen_kernel_features")


def gen_kernel_features(
    kernel: kernels.Kernel,
    num_inputs: int,
    num_outputs: int,
    **kwargs: Any,
) -> KernelFeatureMap:
    r"""Generates a feature map :math:`\phi: \mathcal{X} \to \mathbb{R}^{n}` such that
    :math:`k(x, x') ≈ \phi(x)^{T} \phi(x')`. For stationary kernels :math:`k`, defaults
    to the method of random Fourier features. For more details, see [rahimi2007random]_
    and [sutherland2015error]_.

    Args:
        kernel: The kernel :math:`k` to be represented via a finite-dim basis.
        num_inputs: The number of input features.
        num_outputs: The number of kernel features.
    """
    return GenKernelFeatures(
        kernel,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        **kwargs,
    )


def _gen_fourier_features(
    kernel: kernels.Kernel,
    weight_generator: Callable[[Size], Tensor],
    num_inputs: int,
    num_outputs: int,
) -> KernelFeatureMap:
    r"""Generate a feature map :math:`\phi: \mathcal{X} \to \mathbb{R}^{2l}` that
    approximates a stationary kernel so that :math:`k(x, x') ≈ \phi(x)^\top \phi(x')`.

    Following [sutherland2015error]_, we represent complex exponentials by pairs of
    basis functions :math:`\phi_{i}(x) = \sin(x^\top w_{i})` and
    :math:`\phi_{i + l} = \cos(x^\top w_{i}).

    Args:
        kernel: A stationary kernel :math:`k(x, x') = k(x - x')`.
        weight_generator: A callable used to generate weight vectors :math:`w`.
        num_inputs: The number of input features.
        num_outputs: The number of Fourier features.
    """
    if num_outputs % 2:
        raise UnsupportedError(
            f"Expected an even number of output features, but received {num_outputs=}."
        )

    input_transform = InverseLengthscaleTransform(kernel)
    if kernel.active_dims is not None:
        num_inputs = len(kernel.active_dims)
        input_transform = ChainedTransform(
            input_transform, FeatureSelector(indices=kernel.active_dims)
        )

    weight = weight_generator(
        Size([kernel.batch_shape.numel() * num_outputs // 2, num_inputs])
    ).reshape(*kernel.batch_shape, num_outputs // 2, num_inputs)

    output_transform = SineCosineTransform(
        torch.tensor((2 / num_outputs) ** 0.5, device=kernel.device, dtype=kernel.dtype)
    )
    return KernelFeatureMap(
        kernel=kernel,
        weight=weight,
        input_transform=input_transform,
        output_transform=output_transform,
    )


@GenKernelFeatures.register(kernels.RBFKernel)
def _gen_kernel_features_rbf(
    kernel: kernels.RBFKernel,
    *,
    num_inputs: int,
    num_outputs: int,
) -> KernelFeatureMap:
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        return draw_sobol_normal_samples(
            n=n,
            d=d,
            device=kernel.lengthscale.device,
            dtype=kernel.lengthscale.dtype,
        )

    return _gen_fourier_features(
        kernel=kernel,
        weight_generator=_weight_generator,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )


@GenKernelFeatures.register(kernels.MaternKernel)
def _gen_kernel_features_matern(
    kernel: kernels.MaternKernel,
    *,
    num_inputs: int,
    num_outputs: int,
) -> KernelFeatureMap:
    def _weight_generator(shape: Size) -> Tensor:
        try:
            n, d = shape
        except ValueError:
            raise UnsupportedError(
                f"Expected `shape` to be 2-dimensional, but {len(shape)=}."
            )

        dtype = kernel.lengthscale.dtype
        device = kernel.lengthscale.device
        nu = torch.tensor(kernel.nu, device=device, dtype=dtype)
        normals = draw_sobol_normal_samples(n=n, d=d, device=device, dtype=dtype)
        return Gamma(nu, nu).rsample((n, 1)).rsqrt() * normals

    return _gen_fourier_features(
        kernel=kernel,
        weight_generator=_weight_generator,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )


@GenKernelFeatures.register(kernels.ScaleKernel)
def _gen_kernel_features_scale(
    kernel: kernels.ScaleKernel,
    *,
    num_inputs: int,
    num_outputs: int,
) -> KernelFeatureMap:
    active_dims = kernel.active_dims
    feature_map = gen_kernel_features(
        kernel.base_kernel,
        num_inputs=num_inputs if active_dims is None else len(active_dims),
        num_outputs=num_outputs,
    )

    if active_dims is not None and active_dims is not kernel.base_kernel.active_dims:
        feature_map.input_transform = ChainedTransform(
            feature_map.input_transform, FeatureSelector(indices=active_dims)
        )

    feature_map.output_transform = ChainedTransform(
        OutputscaleTransform(kernel), feature_map.output_transform
    )
    return feature_map
