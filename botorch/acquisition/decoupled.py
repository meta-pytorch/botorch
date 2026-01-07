#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for decoupled acquisition functions."""

from __future__ import annotations

import warnings
from abc import ABC

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import BotorchWarning
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.logging import shape_to_str

from botorch.models.model import ModelList
from torch import Tensor


class DecoupledAcquisitionFunction(AcquisitionFunction, ABC):
    """
    Abstract base class for decoupled acquisition functions.
    A decoupled acquisition function where one may intend to
    evaluate a design on only a subset of the outcomes.
    Typically this would be handled by fantasizing, where one
    would fantasize as to what the partial observation would
    be if one were to evaluate a design on the subset of
    outcomes (e.g. you only fantasize at those outcomes). The
    `X_evaluation_mask` specifies which outcomes should be
    evaluated for each design.  `X_evaluation_mask` is `q x m`,
    where there are q design points in the batch and m outcomes.
    In the asynchronous case, where there are n' pending points,
    we need to track which outcomes each pending point should be
    evaluated on. In this case, we concatenate
    `X_pending_evaluation_mask` with `X_evaluation_mask` to obtain
    the full evaluation_mask.


    This abstract class handles generating and updating an evaluation mask,
    which is a boolean tensor indicating which outcomes a given design is
    being evaluated on. The evaluation mask has shape `(n' + q) x m`, where
    n' is the number of pending points and the q represents the new
    candidates to be generated.

    If `X(_pending)_evaluation_mas`k is None, it is assumed that `X(_pending)`
    will be evaluated on all outcomes.
    """

    def __init__(
        self, model: ModelList, X_evaluation_mask: Tensor | None = None, **kwargs
    ) -> None:
        r"""Initialize.

        Args:
            model: A model
            X_evaluation_mask: A `q x m`-dim boolean tensor
                indicating which outcomes the decoupled acquisition
                function should generate new candidates for.
        """
        if not isinstance(model, ModelList):
            raise ValueError(f"{self.__class__.__name__} requires using a ModelList.")
        super().__init__(model=model, **kwargs)
        self.num_outputs = model.num_outputs
        self.X_evaluation_mask = X_evaluation_mask
        self.X_pending_evaluation_mask = None
        self.X_pending = None

    @property
    def X_evaluation_mask(self) -> Tensor | None:
        r"""Get the evaluation indices for the new candidate."""
        return self._X_evaluation_mask

    @X_evaluation_mask.setter
    def X_evaluation_mask(self, X_evaluation_mask: Tensor | None = None) -> None:
        r"""Set the evaluation indices for the new candidate."""
        if X_evaluation_mask is not None:
            # TODO: Add batch support
            if (
                X_evaluation_mask.ndim != 2
                or X_evaluation_mask.shape[-1] != self.num_outputs
            ):
                raise BotorchTensorDimensionError(
                    "Expected X_evaluation_mask to be `q x m`, but got shape"
                    f" {shape_to_str(X_evaluation_mask.shape)}."
                )
        self._X_evaluation_mask = X_evaluation_mask

    def set_X_pending(
        self,
        X_pending: Tensor | None = None,
        X_pending_evaluation_mask: Tensor | None = None,
    ) -> None:
        r"""Informs the AF about pending design points for different outcomes.

        Args:
            X_pending: A `n' x d` Tensor with `n'` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
            X_pending_evaluation_mask: A `n' x m`-dim tensor of booleans indicating
                for which outputs the pending point is being evaluated on. If
                `X_pending_evaluation_mask` is `None`, it is assumed that
                `X_pending` will be evaluated on all outcomes.
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
            if X_pending_evaluation_mask is not None:
                if (
                    X_pending_evaluation_mask.ndim != 2
                    or X_pending_evaluation_mask.shape[0] != X_pending.shape[0]
                    or X_pending_evaluation_mask.shape[1] != self.num_outputs
                ):
                    raise BotorchTensorDimensionError(
                        f"Expected `X_pending_evaluation_mask` of shape "
                        f"`{X_pending.shape[0]} x {self.num_outputs}`, but "
                        f"got {shape_to_str(X_pending_evaluation_mask.shape)}."
                    )
                self.X_pending_evaluation_mask = X_pending_evaluation_mask
            elif self.X_evaluation_mask is not None:
                raise ValueError(
                    "If `self.X_evaluation_mask` is not None, then "
                    "`X_pending_evaluation_mask` must be provided."
                )

        else:
            self.X_pending = X_pending
            self.X_pending_evaluation_mask = X_pending_evaluation_mask

    def construct_evaluation_mask(self, X: Tensor) -> Tensor | None:
        r"""Construct the boolean evaluation mask for X and X_pending

        Args:
            X: A `batch_shape x n x d`-dim tensor of designs.

        Returns:
            A `n + n' x m`-dim tensor of booleans indicating
            which outputs should be evaluated.
        """
        if self.X_pending_evaluation_mask is not None:
            X_evaluation_mask = self.X_evaluation_mask
            if X_evaluation_mask is None:
                # evaluate all objectives for X
                X_evaluation_mask = torch.ones(
                    X.shape[-2], self.num_outputs, dtype=torch.bool, device=X.device
                )
            elif X_evaluation_mask.shape[0] != X.shape[-2]:
                raise BotorchTensorDimensionError(
                    "Expected the -2 dimension of X and X_evaluation_mask to match."
                )
            # construct mask for X
            return torch.cat(
                [X_evaluation_mask, self.X_pending_evaluation_mask], dim=-2
            )
        return self.X_evaluation_mask
