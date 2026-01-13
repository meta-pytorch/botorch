#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from torch import Tensor


class InfiniteWidthBNNKernel(Kernel):
    r"""Infinite-width BNN kernel.

    Defines the GP kernel which is equivalent to performing exact Bayesian
    inference on a fully-connected deep neural network with ReLU activations
    and i.i.d. priors in the infinite-width limit.
    See [Cho2009kernel]_ and [Lee2018deep]_ for details.

    .. [Cho2009kernel]
        Y. Cho, and L. Saul. Kernel methods for deep learning.
        Advances in Neural Information Processing Systems 22. 2009.
    .. [Lee2018deep]
        J. Lee, Y. Bahri, R. Novak, S. Schoenholz, J. Pennington, and J. Dickstein.
        Deep Neural Networks as Gaussian Processes.
        International Conference on Learning Representations. 2018.
    """

    has_lengthscale = False

    def __init__(
        self,
        depth: int = 3,
        batch_shape: torch.Size | None = None,
        active_dims: tuple[int, ...] | None = None,
        acos_eps: float = 1e-7,
        device: torch.device | None = None,
    ) -> None:
        r"""
        Args:
            depth: Depth of neural network.
            batch_shape: This will set a separate weight/bias var for each batch.
                It should be :math:``B_1 \times \ldots \times B_k`` if
                :math:``\mathbf`` is a
                :math:``B_1 \times \ldots \times B_k \times N \times D`` tensor.
            param active_dims: Compute the covariance of only a few input dimensions.
                The ints corresponds to the indices of the dimensions.
            param acos_eps: A small positive value to restrict acos inputs to
                :math``[-1 + \epsilon, 1 - \epsilon]``
            param device: Device for parameters.
        """
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.depth = depth
        self.acos_eps = acos_eps

        self.register_parameter(
            "raw_weight_var",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1, device=device)),
        )
        self.register_constraint("raw_weight_var", Positive())

        self.register_parameter(
            "raw_bias_var",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1, device=device)),
        )
        self.register_constraint("raw_bias_var", Positive())

    @property
    def weight_var(self) -> Tensor:
        return self.raw_weight_var_constraint.transform(self.raw_weight_var)

    @weight_var.setter
    def weight_var(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight_var)
        self.initialize(
            raw_weight_var=self.raw_weight_var_constraint.inverse_transform(value)
        )

    @property
    def bias_var(self) -> Tensor:
        return self.raw_bias_var_constraint.transform(self.raw_bias_var)

    @bias_var.setter
    def bias_var(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias_var)
        self.initialize(
            raw_bias_var=self.raw_bias_var_constraint.inverse_transform(value)
        )

    def _initialize_var(self, x: Tensor) -> Tensor:
        """Computes the initial variance of x for layer 0"""
        return (
            self.weight_var * torch.sum(x * x, dim=-1, keepdim=True) / x.shape[-1]
            + self.bias_var
        )

    def _update_var(self, K: Tensor, x: Tensor) -> Tensor:
        """Computes the updated variance of x for next layer"""
        return self.weight_var * K / 2 + self.bias_var

    def k(self, x1: Tensor, x2: Tensor) -> Tensor:
        r"""
        For single-layer infinite-width neural networks with i.i.d. priors,
        the covariance between outputs can be computed by
        :math:``K^0(x, x')=\sigma_b^2+\sigma_w^2\frac{x \cdot x'}{d_\text{input}}``.

        For deeper networks, we can recursively define the covariance as
        :math:`K^l(x, x')=\sigma_b^2+\sigma_w^2
        F_\phi(K^{l-1}(x, x'), K^{l-1}(x, x), K^{l-1}(x', x'))`
        where :math:``F_\phi`` is a deterministic function based on the
        activation function :math:``\phi``.

        For ReLU activations, this yields the arc-cosine kernel, which can be computed
        analytically.

        Args:
            x1: ``batch_shape x n1 x d``-dim Tensor
            x2: ``batch_shape x n2 x d``-dim Tensor
        """
        K_12 = (
            self.weight_var * (x1.matmul(x2.transpose(-2, -1)) / x1.shape[-1])
            + self.bias_var
        )

        for layer in range(self.depth):
            if layer == 0:
                K_11 = self._initialize_var(x1)
                K_22 = self._initialize_var(x2)
            else:
                K_11 = self._update_var(K_11, x1)
                K_22 = self._update_var(K_22, x2)

            sqrt_term = torch.sqrt(K_11.matmul(K_22.transpose(-2, -1)))

            fraction = K_12 / sqrt_term
            fraction = torch.clamp(
                fraction, min=-1 + self.acos_eps, max=1 - self.acos_eps
            )

            theta = torch.acos(fraction)
            theta_term = torch.sin(theta) + (torch.pi - theta) * fraction

            K_12 = (
                self.weight_var / (2 * torch.pi) * sqrt_term * theta_term
                + self.bias_var
            )

        return K_12

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool | None = False,
        last_dim_is_batch: bool | None = False,
        **params,
    ) -> Tensor:
        """
        Args:
            x1: ``batch_shape x n1 x d``-dim Tensor
            x2: ``batch_shape x n2 x d``-dim Tensor
            diag: If True, only returns the diagonal of the kernel matrix.
            last_dim_is_batch: Not supported by this kernel.
        """
        if last_dim_is_batch:
            raise RuntimeError("last_dim_is_batch not supported by this kernel.")

        if diag:
            K = self._initialize_var(x1)
            for _ in range(self.depth):
                K = self._update_var(K, x1)
            return K.squeeze(-1)
        else:
            return self.k(x1, x2)
