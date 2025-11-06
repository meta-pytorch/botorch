#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.logging import logger
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.posteriors.torch import TorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import (
    IIDNormalSampler,
    NormalMCSampler,
    SobolQMCNormalSampler,
)
from botorch.utils.dispatcher import Dispatcher
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Distribution
from torch.quasirandom import SobolEngine


def _posterior_to_distribution_encoder(
    posterior: Posterior,
) -> type[Distribution] | type[Posterior]:
    r"""An encoder returning the type of the distribution for `TorchPosterior`
    and the type of the posterior for the rest.
    """
    if isinstance(posterior, TorchPosterior):
        return type(posterior.distribution)
    return type(posterior)


GetSampler = Dispatcher("get_sampler", encoder=_posterior_to_distribution_encoder)


def get_sampler(
    posterior: TorchPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the sampler for the given posterior.

    The sampler can be used as `sampler(posterior)` to produce samples
    suitable for use in acquisition function optimization via SAA.

    Args:
        posterior: A `Posterior` to get the sampler for.
        sample_shape: The sample shape of the samples produced by the
            given sampler. The full shape of the resulting samples is
            given by `posterior._extended_shape(sample_shape)`.
        seed: Seed used to initialize sampler.

    Returns:
        The `MCSampler` object for the given posterior.
    """
    return GetSampler(posterior, sample_shape=sample_shape, seed=seed)


@GetSampler.register(MultivariateNormal)
def _get_sampler_mvn(
    posterior: GPyTorchPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> NormalMCSampler:
    r"""The Sobol normal sampler for the `MultivariateNormal` posterior.

    If the output dim is too large, falls back to `IIDNormalSampler`.
    """
    sampler = SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)
    collapsed_shape = sampler._get_collapsed_shape(posterior=posterior)
    base_collapsed_shape = collapsed_shape[len(sample_shape) :]
    if base_collapsed_shape.numel() > SobolEngine.MAXDIM:
        logger.warning(
            f"Output dim {base_collapsed_shape.numel()} is too large for the "
            "Sobol engine. Using IIDNormalSampler instead."
        )
        sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
    return sampler


@GetSampler.register(TransformedPosterior)
def _get_sampler_derived(
    posterior: TransformedPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the sampler for the underlying posterior."""
    return get_sampler(
        posterior=posterior._posterior,
        sample_shape=sample_shape,
        seed=seed,
    )


@GetSampler.register(PosteriorList)
def _get_sampler_list(
    posterior: PosteriorList, sample_shape: torch.Size, *, seed: int | None = None
) -> MCSampler:
    r"""Get the `ListSampler` with the appropriate list of samplers.

    NOTE: Does not dispatch to Sobol sampling for normal posteriors due to
    correlations between samplers. Instead uses `IIDNormalSampler`. See the following
    issue for details: https://github.com/meta-pytorch/botorch/issues/2658
    """
    samplers = []
    for p in posterior.posteriors:
        sampler = get_sampler(posterior=p, sample_shape=sample_shape, seed=seed)
        if isinstance(sampler, SobolQMCNormalSampler):
            sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
        samplers.append(sampler)

    return ListSampler(*samplers)


@GetSampler.register(EnsemblePosterior)
def _get_sampler_ensemble(
    posterior: EnsemblePosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> MCSampler:
    r"""Get the `IndexSampler` for the `EnsemblePosterior`."""
    return IndexSampler(sample_shape=sample_shape, seed=seed)


@GetSampler.register(object)
def _not_found_error(
    posterior: Posterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> None:
    raise NotImplementedError(
        f"A registered `MCSampler` for posterior {posterior} is not found. You can "
        "implement and register one using `@GetSampler.register`."
    )
