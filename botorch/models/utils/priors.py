#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Optional

import torch
from gpytorch.priors import Prior
from gpytorch.priors.utils import BUFFERED_PREFIX
from torch import Tensor
from torch.distributions import Beta
from torch.nn import Module as TModule


class BetaPrior(Prior, Beta):
    """Beta Prior parameterized by concentration1 (alpha) and concentration0 (beta).

    pdf(x) = x^(alpha - 1) * (1 - x)^(beta - 1) / B(alpha, beta)

    where alpha > 0 and beta > 0 are the concentration parameters.
    Supported on [0, 1], useful as a prior on correlation parameters.
    """

    # Beta.concentration1/concentration0 are @property descriptors (they
    # delegate to an internal Dirichlet), so _bufferize_attributes cannot
    # delattr them.  We store separate buffers with the BUFFERED_PREFIX and
    # sync them back after load_state_dict.
    _PARAM_NAMES = ("concentration1", "concentration0")

    def __init__(
        self,
        concentration1: float,
        concentration0: float,
        validate_args: bool = False,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Initialize BetaPrior.

        Args:
            concentration1: Alpha (first concentration) parameter.
            concentration0: Beta (second concentration) parameter.
            validate_args: Whether to validate input arguments.
            transform: Optional transform to apply before computing log_prob.
        """
        TModule.__init__(self)
        Beta.__init__(
            self,
            concentration1=concentration1,
            concentration0=concentration0,
            validate_args=validate_args,
        )
        for attr in self._PARAM_NAMES:
            self.register_buffer(
                f"{BUFFERED_PREFIX}{attr}", getattr(self, attr).clone()
            )
        self._transform = transform

    def _update_dirichlet_concentration(self) -> None:
        """
        Sync buffered values back into the underlying Dirichlet distribution.
        """
        c1 = getattr(self, f"{BUFFERED_PREFIX}concentration1")
        c0 = getattr(self, f"{BUFFERED_PREFIX}concentration0")
        self._dirichlet.concentration = torch.stack([c1, c0], dim=-1)

    def _apply(self, fn: Callable):
        to_return = super()._apply(fn)
        self._update_dirichlet_concentration()
        return to_return

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._update_dirichlet_concentration()

    def expand(self, batch_shape: torch.Size) -> "BetaPrior":
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(
            self.concentration1.expand(batch_shape),
            self.concentration0.expand(batch_shape),
            validate_args=self._validate_args,
            transform=self._transform,
        )
