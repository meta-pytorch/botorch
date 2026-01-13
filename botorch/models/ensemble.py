#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Ensemble Models: Simple wrappers that allow the usage of ensembles
via the BoTorch Model and Posterior APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


class EnsembleModel(Model, ABC):
    """Abstract base class for ensemble models."""

    def __init__(self, weights: Tensor | None = None):
        """Initialize the ensemble model.

        Args:
            weights: Optional weights for the ensemble members.
                If None, the model weights will default to uniform in the
                corresponding mixture posterior.
        """
        super().__init__()
        # buffer ``weights`` is generally a name occupied by another module,
        # so we have to be more specific here
        self.ensemble_weights = weights

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the (ensemble) model output at X.

        Args:
            X: A ``batch_shape x n x d``-dim input tensor ``X``.

        Returns:
            A ``batch_shape x s x n x m``-dimensional output tensor where
            ``s`` is the size of the ensemble.
        """
        pass  # pragma: no cover

    def _forward(self, X: Tensor) -> Tensor:
        return self.forward(X=X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> EnsemblePosterior:
        r"""Compute the ensemble posterior at X.

        Args:
            X: A ``batch_shape x q x d``-dim input tensor ``X``.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior. If omitted, computes the posterior
                over all model outputs.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            An ``EnsemblePosterior`` object, representing ``batch_shape`` joint
            posteriors over ``n`` points and the outputs selected by ``output_indices``.
        """
        # Apply the input transforms in ``eval`` mode.
        self.eval()
        X = self.transform_inputs(X)
        # Note: we use a Tensor instance check so that ``observation_noise = True``
        # just gets ignored. This avoids having to do a bunch of case distinctions
        # when using a ModelList.
        if isinstance(kwargs.get("observation_noise"), Tensor):
            # TODO: Consider returning an MVN here instead
            raise UnsupportedError("Ensemble models do not support observation noise.")
        values = self._forward(X)
        # NOTE: The ``outcome_transform`` ``untransform``s the predictions rather
        # than the ``posterior`` (as is done in GP models). This is more general
        # since it works even if the transform doesn't support
        # ``untransform_posterior``.
        if hasattr(self, "outcome_transform"):
            values, _ = self.outcome_transform.untransform(values, X=X)
        if output_indices is not None:
            values = values[..., output_indices]
        posterior = EnsemblePosterior(values=values, weights=self.ensemble_weights)
        if posterior_transform is not None:
            return posterior_transform(posterior=posterior, X=X)
        else:
            return posterior
