#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from torch import Tensor


def _reshape_base_samples_non_interleaved(
    mvn: MultitaskMultivariateNormal, base_samples: Tensor, sample_shape: torch.Size
) -> Tensor:
    r"""Reshape base samples to account for non-interleaved MT-MVNs.

    This method is important for making sure that the ``n``th base sample
    only effects the posterior sample for the ``p``th point if ``p >= n``.
    Without this reshaping, for M>=2, the posterior samples for all ``n``
    points would be affected.

    Args:
        mvn: A MultitaskMultivariateNormal distribution.
        base_samples: A ``sample_shape x ``batch_shape`` x n x m``-dim
            tensor of base_samples.
        sample_shape: The sample shape.

    Returns:
        A ``sample_shape x ``batch_shape`` x n x m``-dim tensor of
            base_samples suitable for a non-interleaved-multi-task
            or single-task covariance matrix.
    """
    if not mvn._interleaved:
        new_shape = sample_shape + mvn._output_shape[:-2] + mvn._output_shape[:-3:-1]
        base_samples = (
            base_samples.transpose(-1, -2)
            .view(new_shape)
            .reshape(sample_shape + mvn.loc.shape)
            .view(base_samples.shape)
        )
    return base_samples
