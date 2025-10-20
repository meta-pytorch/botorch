#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for multi-output acquisition functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from torch import Tensor


class MultiOutputAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Abstract base class for multi-output acquisition functions.

    These are intended to be optimized with a multi-objective optimizer (e.g.
    NSGA-II).
    """

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """

    def set_X_pending(self, X_pending: Tensor | None) -> None:
        r"""Set the pending points.

        Args:
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation
                (but may not yet have been evaluated).
        """
        raise UnsupportedError(
            "X_pending is not supported for multi-output acquisition functions."
        )


class MultiOutputPosteriorMean(MultiOutputAcquisitionFunction):
    def __init__(self, model: Model, weights: Tensor | None = None) -> None:
        r"""Constructor for the MultiPosteriorMean.

        Maximization of all outputs is assumed by default. Minimizing outputs can
        be achieved by setting the corresponding weights to negative.

        Args:
            acqfs: A list of `m` acquisition functions.
            weights: A one-dimensional tensor with `m` elements representing the
                weights on the outputs.
        """
        super().__init__(model=model)
        if self.model.num_outputs < 2:
            raise NotImplementedError(
                "MultiPosteriorMean only supports multi-output models."
            )
        # TODO: this could be done via a posterior transform
        if weights is not None and weights.shape[0] != self.model.num_outputs:
            raise ValueError(
                f"weights must have {self.model.num_outputs} elements, but got"
                f" {weights.shape[0]}."
            )
        self.register_buffer("weights", weights)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        mean = self.model.posterior(X).mean.squeeze(-2)
        if self.weights is not None:
            return mean * self.weights
        return mean


class MultiOutputAcquisitionFunctionWrapper(MultiOutputAcquisitionFunction):
    r"""Multi-output wrapper around single-output acquisition functions."""

    def __init__(self, acqfs: list[AcquisitionFunction]) -> None:
        r"""Constructor for the AcquisitionFunction base class.

        Args:
            acqfs: A list of `m` acquisition functions.
        """
        # We could set the model to be an ensemble model consistent of the
        # model used in each acqf
        super().__init__(model=acqfs[0].model)
        self.acqfs: list[AcquisitionFunction] = acqfs

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on the candidate set X.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b) x m`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        return torch.stack([acqf(X) for acqf in self.acqfs], dim=-1)
