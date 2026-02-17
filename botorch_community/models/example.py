#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains a simple modification of SingleTaskGP as an example.

References:

.. [Example2024paper]
    S. Cakmak, D.Eriksson, M. Balandat, E. Bakshy. Example Paper.
    Proceedings of the Example Conference, 2024.

Contributor: saitcakmak
"""

from typing import Optional

from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch import Tensor


class ExampleModel(SingleTaskGP):
    def __init__(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ) -> None:
        r"""Initialize the example model from [Example2024paper]_.

        Args:
            train_X: A ``batch_shape x n x d`` tensor of training features.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            train_Yvar: An optional ``batch_shape x n x m`` tensor of observed
                measurement noise.
        """
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1])),
        )
