#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Prior-Guided Acquisition Functions

References

.. [Hvarfner2022]
    C. Hvarfner, D. Stoll, A. Souza, M. Lindauer, F. Hutter, L. Nardi. PiBO:
    Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization.
    ICLR 2022.
"""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction
from botorch.exceptions.errors import BotorchError
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor

from torch.nn import Module


class PriorGuidedAcquisitionFunction(AcquisitionFunction):
    r"""Class for weighting acquisition functions by a prior distribution.

    Supports MC and batch acquisition functions via
    SampleReducingAcquisitionFunction.

    See [Hvarfner2022]_ for details.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        prior_module: Module,
        log: bool = False,
        prior_exponent: float = 1.0,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""Initialize the prior-guided acquisition function.

        Args:
            acq_function: The base acquisition function.
            prior_module: A Module that computes the probability
                (or log probability) for the provided inputs.
                `prior_module.forward` should take a `batch_shape x q`-dim
                tensor of inputs and return a `batch_shape x q`-dim tensor
                of probabilities.
            log: A boolean that should be true if the acquisition function emits a
                log-transformed value and the prior module emits a log probability.
            prior_exponent: The exponent applied to the prior. This can be used
                for example  to decay the effect the prior over time as in
                [Hvarfner2022]_.
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
                Note: X_pending should be provided as an argument to or set on
                `PriorGuidedAcquisitionFunction`, but not set on the underlying
                acquisition function.
        """
        super().__init__(model=acq_function.model)
        if getattr(acq_function, "X_pending", None) is not None:
            raise BotorchError(
                "X_pending is set on acq_function, but should be set on "
                "`PriorGuidedAcquisitionFunction`."
            )
        self.acq_func = acq_function
        self.prior_module = prior_module
        self._log = log
        self._prior_exponent = prior_exponent
        self._is_sample_reducing_af = isinstance(
            acq_function, SampleReducingMCAcquisitionFunction
        )
        self.set_X_pending(X_pending=X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the acquisition function weighted by the prior."""
        # batch_shape x q
        prior = self.prior_module(X)
        if self._is_sample_reducing_af:
            # sample_shape x batch_shape x q
            af_val = self.acq_func._non_reduced_forward(X)
        else:
            if prior.shape[-1] > 1:
                raise NotImplementedError(
                    "q-batches with q>1 are only supported using "
                    "SampleReducingMCAcquisitionFunction."
                )
            # batch_shape x q
            af_val = self.acq_func(X).unsqueeze(-1)
        if self._log:
            weighted_af_val = af_val + prior * self._prior_exponent
        else:
            weighted_af_val = af_val * prior.pow(self._prior_exponent)
        if self._is_sample_reducing_af:
            return self.acq_func._sample_reduction(
                self.acq_func._q_reduction(weighted_af_val)
            )
        return weighted_af_val.squeeze(-1)  # squeeze q-dim
