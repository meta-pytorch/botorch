#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Factory functions for creating test fixtures."""

from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from torch import Tensor


def make_X(
    batch_shape: list[int] | None = None,
    q: int = 1,
    d: int = 2,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
) -> Tensor:
    """Create a random input tensor for testing.

    Args:
        batch_shape: The batch shape for the tensor. Defaults to [].
        q: The number of candidates. Defaults to 1.
        d: The dimension of the input space. Defaults to 2.
        dtype: The dtype of the tensor. Defaults to torch.double.
        device: The device for the tensor. Defaults to None (CPU).

    Returns:
        A tensor of shape (*batch_shape, q, d) with random values in [0, 1).
    """
    if batch_shape is None:
        batch_shape = []
    return torch.rand(*batch_shape, q, d, dtype=dtype, device=device)


def make_trained_gp(
    n_train: int = 5,
    d: int = 2,
    m: int = 1,
    dtype: torch.dtype = torch.double,
    device: torch.device | None = None,
    with_known_noise: bool = False,
) -> SingleTaskGP:
    """Create a SingleTaskGP with random training data for testing.

    Args:
        n_train: The number of training points. Defaults to 5.
        d: The dimension of the input space. Defaults to 2.
        m: The number of outputs. Defaults to 1.
        dtype: The dtype of the tensors. Defaults to torch.double.
        device: The device for the tensors. Defaults to None (CPU).
        with_known_noise: If True, include train_Yvar. Defaults to False.

    Returns:
        A SingleTaskGP fitted with random training data where train_X has shape
        (n_train, d) and train_Y has shape (n_train, m).
    """
    train_X = torch.rand(n_train, d, dtype=dtype, device=device)
    train_Y = torch.rand(n_train, m, dtype=dtype, device=device)
    if with_known_noise:
        train_Yvar = torch.full_like(train_Y, 0.25)
        return SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar)
    return SingleTaskGP(train_X, train_Y)
