#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import torch

from torch import Tensor


def round_nearest(
    X: Tensor, increment: float, bounds: tuple[float, float] | None
) -> Tensor:
    r"""Rounds the input tensor to the nearest multiple of `increment`.

    Args:
        X: The input to be rounded.
        increment: The increment to round to.
        bounds: An optional tuple of two floats representing the lower and upper
            bounds on `X`. If provided, this will round to the nearest multiple
            of `increment` that lies within the bounds.

    Returns:
        The rounded input.
    """
    X_round = torch.round(X / increment) * increment
    if bounds is not None:
        X_round = torch.where(X_round < bounds[0], X_round + increment, X_round)
        X_round = torch.where(X_round > bounds[1], X_round - increment, X_round)
    return X_round
