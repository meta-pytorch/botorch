#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all botorch acquisition functions."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import torch
from botorch.exceptions import BotorchWarning
from botorch.models.model import Model, ModelDict
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from torch import Tensor
from torch.nn import Module


class AcquisitionFunction(Module, ABC):
    r"""Abstract base class for acquisition functions.

    Please note that if your acquisition requires a backwards call,
    you will need to wrap the backwards call inside of an enable_grad
    context to be able to optimize the acquisition. See #1164.
    """

    _log: bool = False  # whether the acquisition utilities are in log-space

    def __init__(self, model: Model) -> None:
        r"""Constructor for the AcquisitionFunction base class.

        Args:
            model: A fitted model.
        """
        super().__init__()
        self.model: Model = model

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: ``n x d`` Tensor with ``n`` ``d``-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    BotorchWarning,
                    stacklevel=2,
                )
            self.X_pending = X_pending.detach().clone()
        else:
            self.X_pending = X_pending

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A ``(b) x q x d``-dim Tensor of ``(b)`` t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A ``(b)``-dim Tensor of acquisition function values at the given
            design points ``X``.
        """
        pass  # pragma: no cover


class OneShotAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Abstract base class for acquisition functions using one-shot optimization
    """

    @abstractmethod
    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimization.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimization (including variables
            parameterizing the fantasy solutions).
        """
        pass  # pragma: no cover

    @abstractmethod
    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""Extract the candidates from a full "one-shot" parameterization.

        Args:
            X_full: A ``b x q_aug x d``-dim Tensor with ``b`` t-batches of ``q_aug``
                design points each.

        Returns:
            A ``b x q x d``-dim Tensor with ``b`` t-batches of ``q`` design points each.
        """
        pass  # pragma: no cover


class MCSamplerMixin(ABC):
    r"""A mix-in for adding sampler functionality into an acquisition function class.

    Attributes:
        _default_sample_shape: The ``sample_shape`` for the default sampler.
    """

    _default_sample_shape = torch.Size([512])

    def __init__(self, sampler: MCSampler | None = None) -> None:
        r"""Register the sampler on the acquisition function.

        Args:
            sampler: The sampler used to draw base samples for MC-based acquisition
                functions. If ``None``, a sampler is generated on the fly within
                the ``get_posterior_samples`` method using ``get_sampler``.
        """
        self.sampler = sampler

    def get_posterior_samples(self, posterior: Posterior) -> Tensor:
        r"""Sample from the posterior using the sampler.

        Args:
            posterior: The posterior to sample from.
        """
        if self.sampler is None:
            self.sampler = get_sampler(
                posterior=posterior, sample_shape=self._default_sample_shape
            )
        return self.sampler(posterior=posterior)

    @property
    def sample_shape(self) -> torch.Size:
        return (
            self.sampler.sample_shape
            if self.sampler is not None
            else self._default_sample_shape
        )


class MultiModelAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Abstract base class for acquisition functions that require
    multiple types of models.

    The intended use case for these acquisition functions are those
    where we have multiple models, each serving a distinct purpose.
    As an example, we can have a "regression" model that predicts
    one or more outcomes, and a "classification" model that predicts
    the probabilty that a given parameterization is feasible. The
    multi-model acquisition function can then weight the acquisition
    value computed with the "regression" model with the feasibility
    value predicted by the "classification" model to produce the
    composite acquisition value.

    This is currently only a placeholder to help with some development
    in Ax. We plan to add some acquisition functions utilizing multiple
    models in the future.
    """

    def __init__(self, model_dict: ModelDict) -> None:
        r"""Constructor for the MultiModelAcquisitionFunction base class.

        Args:
            model_dict: A ModelDict mapping labels to models.
        """
        super(AcquisitionFunction, self).__init__()
        self.model_dict: ModelDict = model_dict
