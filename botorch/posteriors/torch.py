#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Posterior module to be used with PyTorch distributions.
"""

from __future__ import annotations

from typing import Any

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor
from torch.distributions.distribution import Distribution


class TorchPosterior(Posterior):
    r"""A posterior based on a PyTorch Distribution.

    NOTE: For any attribute that is not explicitly defined on the Posterior level, this
    returns the corresponding attribute of the distribution. This allows easy access
    to the distribution attributes, without having to expose them on the Posterior.
    """

    def __init__(self, distribution: Distribution) -> None:
        r"""A posterior based on a PyTorch Distribution.

        Args:
            distribution: A PyTorch Distribution object.
        """
        self.distribution = distribution
        # Get the device and dtype from distribution attributes.
        for attr in vars(distribution).values():
            if isinstance(attr, Tensor):
                self._device = attr.device
                self._dtype = attr.dtype
                break

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        This is generally used with a sampler that produces the base samples.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        return self.distribution.rsample(sample_shape=sample_shape)

    @property
    def device(self) -> torch.device:
        r"""The torch device of the distribution."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self._dtype

    def __getattr__(self, name: str) -> Any:
        r"""A catch-all for attributes not defined on the posterior level.

        Returns the attributes of the distribution instead.
        """
        return getattr(self.distribution, name)

    def __getstate__(self) -> dict[str, Any]:
        r"""A minimal utility to support pickle protocol.

        Pickle uses ``__get/setstate__`` to serialize / deserialize the objects.
        Since we define ``__getattr__`` above, it takes precedence over these
        methods, and we end up in an infinite loop unless we also define
        ``__getstate__`` and ``__setstate__``.
        """
        return self.__dict__

    def __setstate__(self, d: dict[str, Any]) -> None:
        r"""A minimal utility to support pickle protocol."""
        self.__dict__ = d

    def quantile(self, value: Tensor) -> Tensor:
        r"""Compute quantiles of the distribution.

        For multi-variate distributions, this may return the quantiles of
        the marginal distributions.
        """
        if value.numel() > 1:
            return torch.stack([self.quantile(v) for v in value], dim=0)
        return self.icdf(value)

    def density(self, value: Tensor) -> Tensor:
        r"""The probability density (or mass if discrete) of the distribution.

        For multi-variate distributions, this may return the density of
        the marginal distributions.
        """
        if value.numel() > 1:
            return torch.stack([self.density(v) for v in value], dim=0)
        return self.log_prob(value).exp()

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the distribution with
        the given ``sample_shape``.
        """
        return self.distribution._extended_shape(sample_shape=sample_shape)
