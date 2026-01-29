#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The base class for sampler modules to be used with MC-evaluated acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.exceptions.errors import InputDataError
from botorch.posteriors import Posterior
from torch import Tensor
from torch.nn import Module


KWARGS_DEPRECATED_MSG = (
    "The {} argument of `MCSampler`s has been deprecated and will raise an "
    "error in a future version."
)
KWARG_ERR_MSG = (
    "`MCSampler`s no longer support the `{}` argument. "
    "Consider using `{}` for similar functionality."
)


class MCSampler(Module, ABC):
    r"""Abstract base class for Samplers.

    Subclasses must implement the ``forward`` method.

    Example:
        This method is usually not called directly, but via the sampler's
        ``__call__`` method:
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        sample_shape: torch.Size,
        seed: int | None = None,
    ) -> None:
        r"""Abstract base class for samplers.

        Args:
            sample_shape: The ``sample_shape`` of the samples to generate. The full
                shape of the samples is given by
                ``posterior._extended_shape(sample_shape)``.
            seed: An optional seed to use for sampling.
        """
        super().__init__()
        if not isinstance(sample_shape, torch.Size):
            raise InputDataError(
                "Expected `sample_shape` to be a `torch.Size` object, "
                f"got {sample_shape}."
            )
        self.sample_shape = sample_shape
        self.seed = (
            seed if seed is not None else int(torch.randint(0, 1000000, (1,)).item())
        )
        self.register_buffer("base_samples", None)

    @abstractmethod
    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        pass  # pragma no cover

    def _get_batch_range(self, posterior: Posterior) -> tuple[int, int]:
        r"""Get the t-batch range of the posterior with an optional override.

        In rare cases, e.g., in ``qMultiStepLookahead``, we may want to override the
        ``batch_range`` of the posterior. If this behavior is desired, one can set
        ``batch_range_override`` attribute on the samplers.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The t-batch range to use for collapsing the base samples.
        """
        if hasattr(self, "batch_range_override"):
            return self.batch_range_override
        return posterior.batch_range

    def _get_collapsed_shape(self, posterior: Posterior) -> torch.Size:
        r"""Get the shape of the base samples with the t-batches collapsed.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The collapsed shape of the base samples expected by the posterior. The
            t-batch dimensions of the base samples are collapsed to size 1. This is
            useful to prevent sampling variance across t-batches.
        """
        base_sample_shape = posterior.base_sample_shape
        batch_start, batch_end = self._get_batch_range(posterior)
        base_sample_shape = (
            base_sample_shape[:batch_start]
            + torch.Size([1 for _ in base_sample_shape[batch_start:batch_end]])
            + base_sample_shape[batch_end:]
        )
        return self.sample_shape + base_sample_shape

    def _get_extended_base_sample_shape(self, posterior: Posterior) -> torch.Size:
        r"""Get the shape of the base samples expected by the posterior.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The extended shape of the base samples expected by the posterior.
        """
        return self.sample_shape + posterior.base_sample_shape

    def _update_base_samples(
        self, posterior: Posterior, base_sampler: MCSampler
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        This is used in CachedCholeskyAcquisitionFunctions to ensure consistency.

        Args:
            posterior: The posterior for which the base samples are constructed.
            base_sampler: The base sampler to retrieve the base samples from.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `_update_base_samples`."
        )

    def _instance_check(self, base_sampler):
        r"""Check that ``base_sampler`` is an instance of ``self.__class__``."""
        if not isinstance(base_sampler, self.__class__):
            raise RuntimeError(
                "Expected `base_sampler` to be an instance of "
                f"{self.__class__.__name__}. Got {base_sampler}."
            )
