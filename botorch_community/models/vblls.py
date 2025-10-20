#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains an implemenation of a Variational Bayesian Last Layer (VBLL) model
that can be used within BoTorch for Bayesian optimization.

References:

[1] P. Brunzema, M. Jordahn, J. Willes, S. Trimpe, J. Snoek, J. Harrison.
    Bayesian Optimization via Contrinual Variational Last Layer Training.
    International Conference on Learning Representations, 2025.

Contributor: brunzema
"""

from __future__ import annotations

import torch
import torch.nn as nn

from botorch.logging import logger
from botorch.posteriors import Posterior
from botorch_community.models.blls import AbstractBLLModel

from botorch_community.models.vbll_helper import DenseNormal, Normal, Regression
from botorch_community.posteriors.bll_posterior import BLLPosterior

from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class SampleModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        sampled_params: Tensor,
        num_inputs: int,
        num_outputs: int,
    ):
        """Network for posterior samples of BLL models.

        Args:
            backbone: Backbone of the original model BLL model
            sampled_params: Sampled parameters from the posterior distribution
                of the last layer.
            num_inputs: Number of inputs to the backbone.
            num_outputs: Number of outputs
        """
        super().__init__()
        self.backbone = backbone
        self.sampled_params = sampled_params
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the sample network.

        Args:
            x: Input as `batch_size x d`-dim Tensor.

        Returns:
            Output as `(sample_shape), batch_size, output_dim`-dim Tensor.
        """
        x = self.backbone(x)

        if self.sampled_params.dim() == 2:
            return (self.sampled_params @ x[..., None]).squeeze(-1)

        x_expanded = x.unsqueeze(0).expand(self.sampled_params.shape[0], -1, -1)
        return x_expanded @ self.sampled_params.transpose(-1, -2)


class VBLLNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = None,
        hidden_features: int = 64,
        out_features: int = 1,
        num_layers: int = 3,
        parameterization: str = "dense",
        cov_rank: int | None = None,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.01,
        clamp_noise_init: bool = True,
        kl_scale: float = 1.0,
        backbone: nn.Module | None = None,
        activation: nn.Module = nn.ELU(),
        device=None,
    ):
        """
        A model with a Variational Bayesian Linear Last (VBLL) layer.

        Args:
            in_features: Number of input features. Defaults to 2.
            hidden_features: Number of hidden units per layer. Defaults to 50.
            out_features: Number of output features. Defaults to 1.
            num_layers: Number of hidden layers in the MLP. Defaults to 3.
            parameterization: Parameterization of the posterior covariance of the last
                layer. Supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}.
            prior_scale: Scaling factor for the prior distribution in the Bayesian last
                layer. Defaults to 1.0.
            wishart_scale: Scaling factor for the Wishart prior in the Bayesian last
                layer. Defaults to 0.01.
            kl_scale: Weighting factor for the Kullback-Leibler (KL) divergence term in
                the loss. Defaults to 1.0.
            backbone: A predefined feature extractor to be used before the VBLL layer.
                If None, a default MLP structure is used. Defaults to None.
            activation: Activation function applied between hidden layers.
                Defaults to `nn.ELU()`.
            device: The device on which the model is stored. If None, the device is
                inferred based on availability. Defaults to None.

        Notes:
            - If a `backbone` module is provided, it is applied before the variational
            last layer. If not, we use a default MLP structure.
        """
        super().__init__()
        self.num_outputs = out_features

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.activation = activation
        self.kl_scale = kl_scale

        if backbone is None:
            if in_features is None:
                raise ValueError(
                    "Please specify the input dimension in the constructor."
                )

            hidden_layers = [
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features),
                    self.activation,
                )
                for _ in range(num_layers)
            ]
            self.backbone = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                self.activation,
                *hidden_layers,
            ).to(dtype=torch.float64, device=self.device)
            self.num_inputs = in_features

        else:
            self.backbone = backbone

            # Try to infer input size if backbone is a Sequential and starts with Linear
            if isinstance(backbone, nn.Sequential) and isinstance(
                backbone[0], nn.Linear
            ):
                self.num_inputs = backbone[0].in_features
            elif in_features is not None:
                self.num_inputs = in_features
            else:
                raise ValueError(
                    "Cannot infer input dimension from provided backbone. "
                    "Please specify `in_features` explicitly."
                )

        # could be changed to other vbll regression layers
        self.head = Regression(
            hidden_features,
            out_features,
            regularization_weight=1.0,  # will be adjusted dynamically
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            parameterization=parameterization,
            cov_rank=cov_rank,
            clamp_noise_init=clamp_noise_init,
        ).to(dtype=torch.float64, device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the VBLL network.

        Args:
            x: Input as `batch_shape x in_features`-dim Tensor.

        Returns:
            Output as `batch_shape x out_features`-dim Tensor.
        """
        x = self.backbone(x)
        return self.head(x)

    def sample_posterior_function(
        self, sample_shape: torch.Size | None = None
    ) -> nn.Module:
        """
        Samples a posterior function by drawing parameters from the model's learned
        distribution.

        Args:
            sample_shape: The desired shape for the sampled parameters. If None, a
                single sample is drawn. Defaults to None.

        Returns:
            A nn.Module that takes an input tensor `x` and returns the corresponding
            model output tensor. The function applies the backbone transformation
            and computes the final output using the sampled parameters.

        Notes:
            - If `sample_shape` is provided, multiple samples are drawn, and the
            function will return a batched output where the first dimension corresponds
            to different samples.
        """
        sample_shape = sample_shape or torch.Size()
        sampled_params = self.head.W().rsample(sample_shape).to(self.device)
        return SampleModel(
            backbone=self.backbone,
            sampled_params=sampled_params,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
        )


def _get_optimizer(
    optimizer_class: Optimizer,
    model_parameters: list,
    lr: float = 1e-3,
    **kwargs,
) -> Optimizer:
    """
    Creates and returns an optimizer.

    Args:
        optimizer_class: The optimizer class (e.g., torch.optim.AdamW).
        model_parameters: Parameters to be optimized.
        lr: Learning rate.
        **kwargs: Additional arguments to be passed to the optimizer.

    Returns:
        The initialized optimizer.
    """
    return optimizer_class(model_parameters, lr=lr, **kwargs)


class VBLLModel(AbstractBLLModel):
    def __init__(self, *args, **kwargs):
        """BoTorch compatible VBLL model."""
        super().__init__()
        self.model = VBLLNetwork(*args, **kwargs)

    @property
    def backbone(self):
        return self.model.backbone

    def sample(self, sample_shape: torch.Size | None = None) -> nn.Module:
        """Create posterior sample networks of the VBLL model. Note that posterior
        samples, we first sample from the posterior distribution of the last layer and
        then create a generalized linear model to get the posterior samples.

        Args:
            sample_shape: Number of samples to draw from the posterior distribution.
                If None, a single sample is drawn. Defaults to None.

        Returns:
            A nn.Module that takes an input as `batch_size x num_inputs`-dim Tensor and
            returns a `(sample_shape), batch_size, output_dim`-dim Tensor.
        """
        return self.model.sample_posterior_function(sample_shape)

    def __call__(self, X: Tensor) -> Normal | DenseNormal:
        """Forward pass through the VBLL model.

        Args:
            X: Input as `batch_size x num_inputs`-dim Tensor.

        Returns:
            Normal distribution with `batch_size x num_outputs` as the mean and
            `batch_size x num_outputs` as variance.
        """
        return self.model(X).predictive

    def fit(
        self,
        train_X: Tensor,
        train_y: Tensor,
        val_X: Tensor | None = None,
        val_y: Tensor | None = None,
        optimization_settings: dict | None = None,
        initialization_params: dict | None = None,
    ):
        """
        Fits the model to the given training data.

        Args:
            train_X: The input training data, expected to be a Tensor of shape
                `num_samples x d`.

            train_y: The target values for training, expected to be a Tensor of
                shape `num_samples x num_outputs`.

            val_X: The optional input validation data, expected to be a Tensor of shape
                `num_val_samples x d`.

            val_y: The optional target values for validation, expected to be a Tensor of
                shape `num_val_samples x num_outputs`.

            optimization_settings: A dict containing optimization-related settings.
                If a key is missing, default values will be used. Available settings:
                    - "num_epochs" (int, default=100): The maximum number of epochs.
                    - "patience" (int, default=10): Epochs before early stopping.
                    - "freeze_backbone" (bool, default=False): If True, the backbone of
                    the model is frozen.
                    - "batch_size" (int, default=32): Batch size for the training.
                    - "optimizer" (torch.optim.Optimizer, default=torch.optim.AdamW):
                    Optimizer for training.
                    - "wd" (float, default=1e-4): Weight decay (L2 regularization)
                    coefficient.
                    - "clip_val" (float, default=1.0): Gradient clipping threshold.

            initialization_params: A dictionary containing the initial parameters of the
                model for feature reuse. If None, the optimization will start from from
                the random initialization in the __init__ method.

        Returns:
            The function trains the model in place and does not return a value.
        """

        if (val_X is None) != (val_y is None):
            missing = "val_X" if val_X is None else "val_y"
            provided = "val_y" if val_X is None else "val_X"

            raise ValueError(
                f"Validation error: {missing} is None while {provided} is provided. "
                "Either both validation inputs (val_X and val_y) must be provided, or"
                "neither."
            )

        # Default settings
        default_opt_settings = {
            "num_epochs": 10_000,
            "freeze_backbone": False,
            "patience": 100,
            "batch_size": 32,
            "optimizer_class": torch.optim.AdamW,
            "lr": 1e-3,
            "wd": 1e-4,
            "clip_val": 1.0,
            "optimizer_kwargs": {},  # Optimizer-specific args (e.g., betas for Adam)
        }

        # Merge defaults with provided settings
        optimization_settings = {
            **default_opt_settings,
            **(optimization_settings or {}),
        }

        # Make dataloader based on train_X, train_y
        device = self.model.device
        dataset = [[train_X[i], train_y[i]] for i, _ in enumerate(train_X)]

        dataloader = DataLoader(
            dataset, shuffle=True, batch_size=optimization_settings["batch_size"]
        )

        if initialization_params is not None:
            self.model.load_state_dict(initialization_params)

        self.model.to(device)
        self.set_reg_weight(self.model.kl_scale / len(train_y))
        param_list = [
            {
                "params": self.model.head.parameters(),
                "weight_decay": 0.0,
            },
        ]

        # freeze backbone
        if not optimization_settings["freeze_backbone"]:
            param_list.append(
                {
                    "params": self.model.backbone.parameters(),
                    "weight_decay": optimization_settings["wd"],
                }
            )

        # Extract settings
        optimizer_class = optimization_settings["optimizer_class"]
        optimizer_kwargs = optimization_settings.get("optimizer_kwargs", {})

        # Initialize optimizer using helper function
        optimizer = _get_optimizer(
            optimizer_class=optimizer_class,
            model_parameters=param_list,
            lr=optimization_settings["lr"],
            **optimizer_kwargs,
        )

        best_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False
        best_model_state = None  # To store the best model parameters

        self.model.train()

        for epoch in range(1, optimization_settings["num_epochs"] + 1):
            # early stopping
            if early_stop:
                break

            running_loss = []

            for train_step, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = out.train_loss_fn(y)  # vbll layer will calculate the loss
                loss.backward()

                if optimization_settings["clip_val"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), optimization_settings["clip_val"]
                    )

                optimizer.step()
                running_loss.append(loss.item())

            # Calculate average loss over the epoch
            avg_train_loss = sum(running_loss[-len(dataloader) :]) / len(dataloader)

            # If validation data is provided, compute validation loss
            if val_X is not None and val_y is not None:
                self.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    out = self.model(x)
                    val_loss = out.val_loss_fn(val_y)

                self.model.train()  # Set model back to training mode

                # Use validation loss for early stopping
                current_loss = val_loss.item()
            else:
                # If no validation data, use training loss
                current_loss = avg_train_loss

            # Early stopping logic
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= optimization_settings["patience"]:
                early_stop = True

        # load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Early stopping at epoch {epoch} with loss {best_loss}.")

    def set_reg_weight(self, new_weight: float):
        self.model.head.regularization_weight = new_weight

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise=None,
        posterior_transform=None,
    ) -> Posterior:
        if X.dim() > 3:
            raise ValueError(f"Input must have at most 3 dimensions, got {X.dim()}.")

        # Determine if the input is batched
        batched = X.dim() == 3

        if not batched:
            N, D = X.shape
            B = 1
        else:
            B, N, D = X.shape
            X = X.reshape(B * N, D)

        posterior = self.model(X).predictive

        # Extract mean and variance
        mean = posterior.mean.squeeze(dim=-1)
        variance = posterior.variance.squeeze(dim=-1)
        cov = torch.diag_embed(variance)

        K = self.num_outputs
        mean = mean.reshape(B, N * K)

        # Cov must be `(B, N*K, N*K)`
        cov = cov.reshape(B, N, K, B, N, K)
        cov = torch.einsum("bnkbrl->bnkrl", cov)  # (B, N, K, N, K)
        cov = cov.reshape(B, N * K, N * K)

        # Remove fake batch dimension if not batched
        if not batched:
            mean = mean.squeeze(0)
            cov = cov.squeeze(0)

        # pass as MultivariateNormal to BLLPosterior
        distribution = MultivariateNormal(mean, cov)
        return BLLPosterior(
            model=self, distribution=distribution, X=X, output_dim=self.num_outputs
        )

    def __str__(self) -> str:
        return self.model.__str__()
