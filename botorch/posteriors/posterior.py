#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract base module for all botorch posteriors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Posterior(ABC):
    """Abstract base class for botorch posteriors."""

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: The base samples, obtained from the appropriate sampler.
                This is a tensor of shape ``sample_shape x base_sample_shape``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `rsample_from_base_samples`."
        )  # pragma: no cover

    @abstractmethod
    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        pass  # pragma: no cover

    def sample(self, sample_shape: torch.Size | None = None) -> Tensor:
        r"""Sample from the posterior without gradients.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    @property
    @abstractmethod
    def device(self) -> torch.device:
        r"""The torch device of the distribution."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        pass  # pragma: no cover

    def quantile(self, value: Tensor) -> Tensor:
        r"""Compute quantiles of the distribution.

        For multi-variate distributions, this may return the quantiles of
        the marginal distributions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `quantile` method."
        )  # pragma: no cover

    def density(self, value: Tensor) -> Tensor:
        r"""The probability density (or mass) of the distribution.

        For multi-variate distributions, this may return the density of
        the marginal distributions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `density` method."
        )  # pragma: no cover

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given ``sample_shape``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_extended_shape`."
        )

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The base shape of the base samples expected in ``rsample``.

        Informs the sampler to produce base samples of shape
        ``sample_shape x base_sample_shape``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `base_sample_shape`."
        )

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        ``base_sample_shape``. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `batch_range`."
        )
