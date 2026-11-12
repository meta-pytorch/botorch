#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Latent Information Gain Acquisition Function for Neural Process Models.

References:

.. [Wu2023arxiv]
   Wu, D., Niu, R., Chinazzi, M., Vespignani, A., Ma, Y.-A., & Yu, R. (2023).
   Deep Bayesian Active Learning for Accelerating Stochastic Simulation.
   arXiv preprint arXiv:2106.02770. Retrieved from https://arxiv.org/abs/2106.02770

Contributor: eibarolle
"""

from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch_community.models.np_regression import NeuralProcessModel
from torch import Tensor
# reference: https://arxiv.org/abs/2106.02770


class LatentInformationGain(AcquisitionFunction):
    def __init__(
        self,
        model: NeuralProcessModel,
        num_samples: int = 10,
        min_std: float = 0.01,
        scaler: float = 0.5,
    ) -> None:
        """
        Latent Information Gain (LIG) Acquisition Function.
        Estimates the expected KL divergence between the latent distribution of a
        neural process given the context and the candidate points (with outcomes
        predicted by the decoder) and the one given the context points only.

        Args:
            model: The NeuralProcessModel to be used.
            num_samples: Int showing the # of samples for calculation, defaults to 10.
            min_std: Float representing the minimum possible standardized std,
                defaults to 0.01.
            scaler: Float scaling the std, defaults to 0.5.
        """
        if not isinstance(model, NeuralProcessModel):
            raise UnsupportedError(
                "LatentInformationGain requires a NeuralProcessModel, got a "
                f"{type(model).__name__}."
            )
        super().__init__(model)
        self.model = model
        self.num_samples = num_samples
        self.min_std = min_std
        self.scaler = scaler

    def forward(self, candidate_x: Tensor) -> Tensor:
        """
        Conduct the Latent Information Gain acquisition function for the inputs.

        Args:
            candidate_x: Candidate input points, as a Tensor. Ideally in the shape
                (N, q, D).

        Returns:
            torch.Tensor: The LIG scores of computed KLDs, in the shape (N, q).
        """
        device = candidate_x.device
        candidate_x = candidate_x.to(device)
        N, q, D = candidate_x.shape
        kl = torch.zeros(N, device=device, dtype=candidate_x.dtype)

        # The encoder and decoder operate on input-transformed points.
        train_X = self.model.transform_inputs(self.model.train_X)
        candidate_x = self.model.transform_inputs(candidate_x)
        x_c, y_c, _, _ = self.model.random_split_context_target(
            train_X, self.model.train_Y, self.model.n_context
        )
        # NOTE: The latent parameters are kept local, so that evaluating the
        # acquisition function does not change the state of the model.
        z_params_context = self.model.data_to_z_params(x_c, y_c)

        for i in range(N):
            x_i = candidate_x[i]
            kl_i = 0.0

            for _ in range(self.num_samples):
                sample_z = self.model.sample_z(*z_params_context)
                if sample_z.dim() == 1:
                    sample_z = sample_z.unsqueeze(0)

                y_pred = self.model.decoder(x_i, sample_z)

                combined_x = torch.cat([x_c, x_i], dim=0)
                combined_y = torch.cat([y_c, y_pred], dim=0)

                z_params_all = self.model.data_to_z_params(combined_x, combined_y)
                kl_sample = self.model.KLD_gaussian(
                    self.min_std,
                    self.scaler,
                    z_params_all=z_params_all,
                    z_params_context=z_params_context,
                )
                kl_i += kl_sample

            kl[i] = kl_i / self.num_samples

        return kl
