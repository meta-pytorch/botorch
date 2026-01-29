#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler modules producing N(0,1) samples, to be used with MC-evaluated
acquisition functions and Gaussian posteriors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.posteriors.higher_order import HigherOrderGPPosterior
from botorch.posteriors.multitask import MultitaskGPPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling.base import MCSampler
from botorch.utils.sampling import draw_sobol_normal_samples, manual_seed
from torch import Tensor
from torch.quasirandom import SobolEngine


class NormalMCSampler(MCSampler, ABC):
    r"""Base class for samplers producing (possibly QMC) N(0,1) samples.

    Subclasses must implement the ``_construct_base_samples`` method.
    """

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        self._construct_base_samples(posterior=posterior)
        samples = posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape,
            base_samples=self.base_samples.expand(
                self._get_extended_base_sample_shape(posterior=posterior)
            ),
        )
        return samples

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Generate base samples (if necessary).

        This function will generate a new set of base samples and register the
        ``base_samples`` buffer if one of the following is true:

        - the MCSampler has no ``base_samples`` attribute.
        - the output of ``_get_collapsed_shape`` does not agree with the shape of
            ``self.base_samples``.

        Args:
            posterior: The Posterior for which to generate base samples.
        """
        pass  # pragma: no cover

    def _update_base_samples(
        self, posterior: Posterior, base_sampler: MCSampler
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.

        This is used in CachedCholeskyAcquisitionFunctions to ensure consistency.

        Args:
            posterior: The posterior for which the base samples are constructed.
            base_sampler: The base sampler to retrieve the base samples from.
        """
        self._instance_check(base_sampler=base_sampler)
        self._construct_base_samples(posterior=posterior)
        if base_sampler.base_samples is not None:
            current_base_samples = base_sampler.base_samples.detach().clone()
            # This is the # of non-``sample_shape`` dimensions.
            base_ndims = current_base_samples.dim() - 1
            # Unsqueeze as many dimensions as needed to match target_shape.
            target_shape = self._get_collapsed_shape(posterior=posterior)
            view_shape = (
                self.sample_shape
                + torch.Size([1] * (len(target_shape) - current_base_samples.dim()))
                + current_base_samples.shape[-base_ndims:]
            )
            expanded_shape = (
                target_shape[:-base_ndims] + current_base_samples.shape[-base_ndims:]
            )
            # Use stored base samples:
            # Use all base_samples from the current sampler
            # this includes the base_samples from the base_sampler
            # and any base_samples for the new points in the sampler.
            # For example, when using sequential greedy candidate generation
            # then generate the new candidate point using last (-1) base_sample
            # in sampler. This copies that base sample.
            expanded_samples = current_base_samples.view(view_shape).expand(
                expanded_shape
            )
            non_transformed_posterior = (
                posterior._posterior
                if isinstance(posterior, TransformedPosterior)
                else posterior
            )
            if isinstance(
                non_transformed_posterior,
                (HigherOrderGPPosterior, MultitaskGPPosterior),
            ):
                n_train_samples = current_base_samples.shape[-1] // 2
                # The train base samples.
                self.base_samples[..., :n_train_samples] = expanded_samples[
                    ..., :n_train_samples
                ]
                # The train noise base samples.
                self.base_samples[..., -n_train_samples:] = expanded_samples[
                    ..., -n_train_samples:
                ]
            else:
                batch_shape = non_transformed_posterior.batch_shape
                single_output = (
                    len(posterior.base_sample_shape) - len(batch_shape)
                ) == 1
                if single_output:
                    self.base_samples[..., : current_base_samples.shape[-1]] = (
                        expanded_samples
                    )
                else:
                    self.base_samples[..., : current_base_samples.shape[-2], :] = (
                        expanded_samples
                    )


class IIDNormalSampler(NormalMCSampler):
    r"""Sampler for MC base samples using iid N(0,1) samples.

    Example:
        >>> sampler = IIDNormalSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Generate iid ``N(0,1)`` base samples (if necessary).

        This function will generate a new set of base samples and set the
        ``base_samples`` buffer if one of the following is true:

        - the MCSampler has no ``base_samples`` attribute.
        - the output of ``_get_collapsed_shape`` does not agree with the shape of
            ``self.base_samples``.

        Args:
            posterior: The Posterior for which to generate base samples.
        """
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            with manual_seed(seed=self.seed):
                base_samples = torch.randn(
                    target_shape, device=posterior.device, dtype=posterior.dtype
                )
            self.register_buffer("base_samples", base_samples)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)


class SobolQMCNormalSampler(NormalMCSampler):
    r"""Sampler for quasi-MC N(0,1) base samples using Sobol sequences.

    Example:
        >>> sampler = SobolQMCNormalSampler(torch.Size([1024]), seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def _construct_base_samples(self, posterior: Posterior) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        ``base_samples`` buffer if one of the following is true:

        - the MCSampler has no ``base_samples`` attribute.
        - the output of ``_get_collapsed_shape`` does not agree with the shape of
            ``self.base_samples``.

        Args:
            posterior: The Posterior for which to generate base samples.
        """
        target_shape = self._get_collapsed_shape(posterior=posterior)
        if self.base_samples is None or self.base_samples.shape != target_shape:
            base_collapsed_shape = target_shape[len(self.sample_shape) :]
            output_dim = base_collapsed_shape.numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(
                    "SobolQMCSampler only supports dimensions "
                    f"`q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}"
                )
            base_samples = draw_sobol_normal_samples(
                d=output_dim,
                n=self.sample_shape.numel(),
                device=posterior.device,
                dtype=posterior.dtype,
                seed=self.seed,
            )
            base_samples = base_samples.view(target_shape)
            self.register_buffer("base_samples", base_samples)
        self.to(device=posterior.device, dtype=posterior.dtype)
