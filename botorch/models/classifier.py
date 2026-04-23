#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Classifier-based models for constraint boundaries and deterministic feasibility.

These models wrap classifiers as BoTorch deterministic models,
enabling them to be used for modeling binary constraints, feasibility, and other
discontinuous outputs where traditional GP models fail due to smoothness assumptions.
"""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.input import InputTransform
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


class SoftKNNClassifierModel(GenericDeterministicModel):
    """
    Soft K-Nearest Neighbors classifier wrapped as a BoTorch deterministic model.

    This model uses Gaussian kernel weighting to compute soft class probabilities.
    Supports both fixed scalar sigma and learnable per-dimension sigma trained via
    leave-one-out (LOO) cross-validation.

    Example:
        >>> from botorch.models.classifier import SoftKNNClassifierModel
        >>> from botorch.utils.datasets import SupervisedDataset
        >>> import torch
        >>>
        >>> X = torch.randn(100, 5)
        >>> y = torch.randint(0, 2, (100, 1), dtype=torch.float64)
        >>> dataset = SupervisedDataset(X=X, Y=y)
        >>>
        >>> # Fixed sigma
        >>> model_inputs = SoftKNNClassifierModel.construct_inputs(
        ...     training_data=dataset,
        ...     sigma=0.3
        ... )
        >>> model = SoftKNNClassifierModel(**model_inputs)
        >>>
        >>> # Learnable per-dimension sigma
        >>> model_inputs = SoftKNNClassifierModel.construct_inputs(
        ...     training_data=dataset,
        ...     learnable_sigma=True,
        ...     sigma_epochs=100
        ... )
        >>> model = SoftKNNClassifierModel(**model_inputs)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        sigma: float = 0.1,
        learnable_sigma: bool = False,
        sigma_lr: float = 0.1,
        sigma_epochs: int = 100,
        input_transform: InputTransform | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SoftKNNClassifierModel.

        Args:
            train_X: Training features tensor of shape (n, d).
            train_Y: Training labels tensor of shape (n,) or (n, 1), binary (0 or 1).
            sigma: Initial Gaussian kernel bandwidth (default: 0.1).
            learnable_sigma: If True, learn per-dimension sigma via LOO
                cross-validation (default: False).
            sigma_lr: Learning rate for sigma optimization (default: 0.1).
            sigma_epochs: Training epochs for sigma (default: 100).
            input_transform: Optional InputTransform applied to both training
                and test inputs before distance computation.
            **kwargs: Additional arguments (ignored).
        """
        # Ensure train_Y is 1D
        train_Y = train_Y.view(-1)

        # Apply input transform to training data if provided
        # This ensures train_X_t is in the same space as test inputs
        # (which are transformed via Model.transform_inputs in posterior())
        if input_transform is not None:
            train_X_t = input_transform(train_X)
        else:
            train_X_t = train_X

        # Learn or use fixed sigma
        learned_sigma_tensor: Tensor | None = None
        if learnable_sigma:
            # Learn per-dimension sigma via LOO cross-validation
            d = train_X_t.shape[-1]
            log_sigma = torch.nn.Parameter(
                torch.full(
                    (d,),
                    torch.log(torch.tensor(sigma, dtype=train_X_t.dtype)),
                    device=train_X_t.device,
                    dtype=train_X_t.dtype,
                )
            )

            optimizer = torch.optim.Adam([log_sigma], lr=sigma_lr, foreach=True)
            N = train_X_t.shape[0]
            train_Y_float = train_Y.to(dtype=train_X_t.dtype)

            for _ in range(sigma_epochs):
                optimizer.zero_grad()
                sigma_vec = log_sigma.exp()  # [d]

                # Pairwise distances with per-dim sigma: sum((x_i - x_j)^2 / sigma_j^2)
                diffs = train_X_t.unsqueeze(1) - train_X_t.unsqueeze(0)  # [N, N, d]
                dists = torch.sum((diffs**2) / (sigma_vec**2), dim=2)  # [N, N]

                # LOO: exclude self (diagonal)
                mask = ~torch.eye(N, dtype=torch.bool, device=train_X_t.device)
                weights = torch.exp(-dists / 2) * mask

                weighted_class1 = torch.sum(
                    weights * (train_Y_float == 1.0).to(dtype=train_X_t.dtype), dim=1
                )
                total_weights = torch.sum(weights, dim=1)
                prob_class1 = weighted_class1 / (total_weights + 1e-12)

                # Binary cross-entropy loss
                eps = 1e-7
                prob_class1_clamped = prob_class1.clamp(eps, 1 - eps)
                loss = -torch.mean(
                    train_Y_float * torch.log(prob_class1_clamped)
                    + (1 - train_Y_float) * torch.log(1 - prob_class1_clamped)
                )
                loss.backward()
                optimizer.step()

            # Detach learned sigma for inference
            sigma_final: Tensor | float = log_sigma.exp().detach()  # [d]
            learned_sigma_tensor = sigma_final
        else:
            sigma_final = sigma  # scalar

        # Create prediction closure with transformed training data
        def predict_proba_fn(X: Tensor) -> Tensor:
            original_shape = X.shape[:-1]
            # Already transformed via Model.transform_inputs if set
            X_flat = X.reshape(-1, X.shape[-1])

            diffs = X_flat.unsqueeze(1) - train_X_t.to(X_flat).unsqueeze(0)

            if isinstance(sigma_final, Tensor):
                # Per-dimension sigma
                dists = torch.sum((diffs**2) / (sigma_final.to(X_flat) ** 2), dim=2)
                weights = torch.exp(-dists / 2)
            else:
                # Scalar sigma
                dists = torch.sum(diffs**2, dim=2)
                weights = torch.exp(-dists / (2 * sigma_final**2))

            mask_class1 = train_Y.to(X_flat) == 1.0
            mask_class1 = mask_class1.to(dtype=X_flat.dtype)

            weighted_class1 = torch.matmul(weights, mask_class1)
            total_weights = torch.sum(weights, dim=1)
            probs_flat = weighted_class1 / (total_weights + 1e-12)

            return probs_flat.reshape(*original_shape, 1)

        # Initialize parent with the prediction function
        super().__init__(f=predict_proba_fn, num_outputs=1)

        # Register input_transform as a submodule so posterior() applies it
        if input_transform is not None:
            self.input_transform = input_transform

        # Expose learned sigma (if any) for inspection
        self.learned_sigma = learned_sigma_tensor

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Construct inputs for SoftKNNClassifierModel from training data.

        This method extracts training data and parameters that will be passed
        to __init__, where the input_transform is applied and the prediction
        closure is created. This ensures compatibility with Ax's model bridge,
        which adds input_transform after calling construct_inputs.

        Args:
            training_data: SupervisedDataset with X (features) and Y (labels).
            sigma: Initial Gaussian kernel bandwidth (default: 0.1).
            learnable_sigma: If True, learn per-dimension sigma via LOO
                cross-validation (default: False).
            sigma_lr: Learning rate for sigma optimization (default: 0.1).
            sigma_epochs: Training epochs for sigma (default: 100).
            input_transform: Optional InputTransform applied to both training
                and test inputs before distance computation.

        Returns:
            Dictionary with training data and model parameters.
        """
        return {
            "train_X": training_data.X.detach().clone(),
            "train_Y": training_data.Y.detach().clone(),
            "sigma": kwargs.get("sigma", 0.1),
            "learnable_sigma": kwargs.get("learnable_sigma", False),
            "sigma_lr": kwargs.get("sigma_lr", 0.1),
            "sigma_epochs": kwargs.get("sigma_epochs", 100),
            "input_transform": kwargs.get("input_transform", None),
        }
