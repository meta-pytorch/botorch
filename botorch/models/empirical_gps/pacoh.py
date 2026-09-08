#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""PACOH-GP: PAC-Optimal Meta-Learning with Gaussian Process Base Learners.

This module implements the PACOH-GP method from Rothfuss et al. (2021),
extending the HyperBO framework with:
1. Meta-level KL regularization via a Gaussian hyper-prior
2. SVGD to approximate the PAC-optimal hyper-posterior as K particles
3. Mixture-of-GPs prediction by averaging GP posteriors across particles

The implementation reuses HyperBO's MLPFeatureExtractor, HyperBOLinearMean,
HyperBODeepKernel, and HyperBOModel for maximum code sharing.

References:
    .. [Rothfuss2021pacoh]
        J. Rothfuss, V. Fortuin, M. Josifoski, and A. Krause.
        PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees.
        ICML, 2021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gpytorch
import torch
from botorch.models.empirical_gps.hyperbo import (
    HyperBODeepKernel,
    HyperBOLinearMean,
    MLPFeatureExtractor,
)
from botorch.models.empirical_gps.svgd import svgd_update
from botorch.models.empirical_gps.utils import ExperimentDataset
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PACOHGPConfig:
    """Configuration for PACOH-GP meta-learning.

    Args:
        num_particles: Number of SVGD particles (K) to approximate the
            hyper-posterior.
        hyper_prior_std: Standard deviation σ_P of the Gaussian hyper-prior
            P = N(0, σ²_P I) over GP prior parameters φ.
        svgd_length_scale: Length scale ℓ for the SVGD RBF kernel. If None,
            uses the median heuristic.
        lambda_coeff: Meta-level scaling coefficient λ. If None, defaults to
            n (number of tasks).
        beta_coeff: Task-level scaling coefficient β. If None, defaults to
            m (dataset size).
        hidden_dims: MLP hidden layer sizes for the feature extractor.
        use_linear_mean: Whether to use a linear mean function on features.
    """

    num_particles: int = 10
    hyper_prior_std: float = 1.0
    svgd_length_scale: float | None = None
    lambda_coeff: float | None = None
    beta_coeff: float | None = None
    # The PACOH paper (Appendix C) uses 4 layers of 32 neurons with tanh.
    # Use (8,) for fast experimentation or to match the HyperBO paper.
    hidden_dims: tuple[int, ...] = (32, 32, 32, 32)
    use_linear_mean: bool = True


# =============================================================================
# Container for Pre-Trained PACOH State
# =============================================================================


@dataclass(frozen=True)
class PACOHPriorContainer:
    """Frozen container for pre-trained PACOH-GP state.

    Stores K particle state dicts, one per SVGD particle.
    Analogous to HyperBOPriorContainer but for multiple particles.

    Args:
        particle_states: List of K frozen state dicts, one per particle.
        config: PACOH-GP configuration used during training.
        input_dim: Input dimension of the training data.
        num_datasets: Number of datasets used for meta-training.
        final_loss: Final loss value at the end of training (for diagnostics).
    """

    particle_states: list[dict[str, Tensor]] = field(repr=False)
    config: PACOHGPConfig
    input_dim: int
    num_datasets: int = 0
    final_loss: float | None = None

    @property
    def num_particles(self) -> int:
        return len(self.particle_states)

    def save(self, path: str) -> None:
        """Save the container to disk."""
        torch.save(
            {
                "particle_states": self.particle_states,
                "config": self.config,
                "input_dim": self.input_dim,
                "num_datasets": self.num_datasets,
                "final_loss": self.final_loss,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PACOHPriorContainer":
        """Load a container from disk."""
        data = torch.load(path, weights_only=False)
        return cls(
            particle_states=data["particle_states"],
            config=data["config"],
            input_dim=data["input_dim"],
            num_datasets=data.get("num_datasets", 0),
            final_loss=data.get("final_loss"),
        )


# =============================================================================
# PACOH-GP Pre-Training
# =============================================================================


def _compute_batched_pacoh_score(
    batched_model: "PACOHGPModel",
    datasets: list[ExperimentDataset],
    hyper_prior_std: float,
    lambda_coeff: float,
    beta_coeff: float,
    n_tasks: int,
    subsample_size: int | None = None,
) -> Tensor:
    """Compute the PACOH score for all K particles simultaneously.

    Uses the batched PACOHGPModel (batch_shape=[K]) to compute all K
    marginal log-likelihoods in a single forward pass per dataset.

    Score_k = ln P(φ_k) + λ/(nβ+λ) · Σᵢ ln Z(Sᵢ, P_φ_k)

    Args:
        batched_model: PACOHGPModel with batch_shape=[K].
        datasets: List of datasets (or mini-batch).
        hyper_prior_std: σ_P for hyper-prior.
        lambda_coeff: λ scaling coefficient.
        beta_coeff: β scaling coefficient.
        n_tasks: Total number of tasks.
        subsample_size: Optional per-dataset subsampling.

    Returns:
        Scalar total score (summed over particles) for .backward().
    """
    device = next(batched_model.parameters()).device
    dtype = next(batched_model.parameters()).dtype

    # 1. Compute batched MLL across all datasets and particles
    total_mll = torch.tensor(0.0, device=device, dtype=dtype)
    for dataset in datasets:
        X = dataset.X
        Y = dataset.Y
        if Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)

        if subsample_size is not None and X.shape[0] > subsample_size:
            indices = torch.randperm(X.shape[0], device=X.device)[:subsample_size]
            X = X[indices]
            Y = Y[indices]

        # Batched forward: returns MultivariateNormal with batch_shape [K]
        output = batched_model.forward(X)
        output = batched_model.likelihood(output)
        # log_prob returns shape [K] — one per particle
        log_probs = output.log_prob(Y)  # [K]
        total_mll = total_mll + log_probs.sum()

    # Scale: undo per-point normalization (to match paper's unnormalized sum)
    # and scale for mini-batching
    n_batch = len(datasets)
    if n_batch < n_tasks:
        total_mll = total_mll * (n_tasks / n_batch)

    # 2. Compute hyper-prior log probability for all particles
    # Sum of -0.5 * ||φ_k||² / σ²_P over all K particles
    hyper_prior_logprob = torch.tensor(0.0, device=device, dtype=dtype)
    for p in batched_model.parameters():
        hyper_prior_logprob = hyper_prior_logprob - 0.5 * (p**2).sum() / (
            hyper_prior_std**2
        )

    # 3. Assemble total PACOH score (summed over particles)
    weight = lambda_coeff / (n_tasks * beta_coeff + lambda_coeff)
    total_score = hyper_prior_logprob + weight * total_mll

    return total_score


def pretrain_pacoh_gp(
    datasets: list[ExperimentDataset],
    input_dim: int,
    config: PACOHGPConfig | None = None,
    num_iterations: int = 1000,
    learning_rate: float = 1e-3,
    subsample_size: int | None = None,
    task_batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.double,
    verbose: bool = False,
) -> PACOHPriorContainer:
    """Pre-train PACOH-GP on historical datasets using SVGD.

    Implements Algorithm 2 from the PACOH paper: initializes K particles
    from the hyper-prior, then iteratively updates them via SVGD to
    approximate the PAC-optimal hyper-posterior Q*.

    Args:
        datasets: List of ExperimentDataset for meta-training.
        input_dim: Dimension of input features.
        config: PACOH-GP configuration. If None, uses defaults.
        num_iterations: Number of SVGD iterations.
        learning_rate: Step size η for SVGD updates.
        subsample_size: Optional per-dataset subsampling.
        task_batch_size: Number of tasks to sample per iteration for
            mini-batching. If None, uses all tasks.
        device: Device to train on.
        dtype: Data type.
        verbose: If True, print training progress.

    Returns:
        PACOHPriorContainer with K frozen particle states.
    """
    if len(datasets) == 0:
        raise ValueError("datasets cannot be empty")

    if config is None:
        config = PACOHGPConfig()

    if config.hyper_prior_std < 0.05:
        import warnings

        warnings.warn(
            f"hyper_prior_std={config.hyper_prior_std} is very small. "
            "This initializes MLP weights near zero, which can cause "
            "degenerate kernel matrices and Cholesky failures. "
            "Consider hyper_prior_std >= 0.1.",
            stacklevel=2,
        )

    K = config.num_particles
    n_tasks = len(datasets)

    if device is None:
        device = datasets[0].X.device

    # Default λ = n, β = average dataset size
    lambda_coeff = (
        config.lambda_coeff if config.lambda_coeff is not None else float(n_tasks)
    )
    avg_m = sum(d.X.shape[0] for d in datasets) / n_tasks
    beta_coeff = config.beta_coeff if config.beta_coeff is not None else avg_m

    # Create dummy training data for model initialization
    dummy_X = torch.zeros(1, input_dim, device=device, dtype=dtype)
    dummy_Y = torch.zeros(1, device=device, dtype=dtype)

    # Create batched training model with batch_shape=[K]
    # A single forward pass computes MLL for all K particles simultaneously
    batched_model = PACOHGPModel(
        train_X=dummy_X,
        train_Y=dummy_Y,
        num_particles=K,
        hidden_dims=config.hidden_dims,
        use_linear_mean=config.use_linear_mean,
    )
    batched_model.to(device=device, dtype=dtype)
    batched_model.train()

    # Get parameter dimension (per particle)
    D = sum(p.numel() for p in batched_model.parameters()) // K

    # Initialize K particles from hyper-prior N(0, σ²_P I) and load into model
    particles = torch.randn(K, D, device=device, dtype=dtype) * config.hyper_prior_std
    _load_particles_from_vectors(batched_model, particles, K)

    # SVGD training loop
    final_loss = None
    for iteration in range(num_iterations):
        # Mini-batch tasks if requested
        if task_batch_size is not None and task_batch_size < n_tasks:
            task_indices = torch.randperm(n_tasks)[:task_batch_size]
            task_batch = [datasets[i] for i in task_indices]
        else:
            task_batch = datasets

        # Enable gradients on all model parameters
        for p in batched_model.parameters():
            p.requires_grad_(True)
        batched_model.zero_grad()

        # Compute batched PACOH score (single forward pass for all K particles)
        total_score = _compute_batched_pacoh_score(
            batched_model=batched_model,
            datasets=task_batch,
            hyper_prior_std=config.hyper_prior_std,
            lambda_coeff=lambda_coeff,
            beta_coeff=beta_coeff,
            n_tasks=n_tasks,
            subsample_size=subsample_size,
        )
        total_score.backward()
        final_loss = -total_score.item() / K

        # Extract per-particle score gradients from batched parameters
        # Each batched parameter has shape [K, ...], so grad[k, ...] is
        # the score gradient for particle k
        score_grads = _extract_per_particle_grads(batched_model, K, D)

        # Extract current particle positions from batched parameters
        particles = _extract_particle_vectors(batched_model, K, D)

        # SVGD update
        particles = svgd_update(
            particles=particles.detach(),
            score_grads=score_grads,
            step_size=learning_rate,
            length_scale=config.svgd_length_scale,
        )

        # Load updated particles back into the batched model
        _load_particles_from_vectors(batched_model, particles, K)

        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {final_loss:.4f}")

    # Extract K frozen state dicts from the batched model
    particle_states = _extract_particle_state_dicts(batched_model, K)

    return PACOHPriorContainer(
        particle_states=particle_states,
        config=config,
        input_dim=input_dim,
        num_datasets=n_tasks,
        final_loss=final_loss,
    )


# =============================================================================
# Particle Vector Helpers
# =============================================================================


def _load_particles_from_vectors(
    model: "PACOHGPModel", particles: Tensor, K: int
) -> None:
    """Load K flat particle vectors into a batched model's parameters.

    Each batched parameter has shape [K, ...]. For each parameter, we
    reshape the corresponding slice of the particle vector to match.
    """
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            per_particle = p.numel() // K
            for k in range(K):
                p.data[k].copy_(
                    particles[k, offset : offset + per_particle].reshape(
                        p.data[k].shape
                    )
                )
            offset += per_particle


def _extract_particle_vectors(model: "PACOHGPModel", K: int, D: int) -> Tensor:
    """Extract K flat particle vectors from batched model parameters."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    particles = torch.zeros(K, D, device=device, dtype=dtype)
    offset = 0
    for p in model.parameters():
        per_particle = p.numel() // K
        for k in range(K):
            particles[k, offset : offset + per_particle] = p.data[k].reshape(-1)
        offset += per_particle
    return particles


def _extract_per_particle_grads(model: "PACOHGPModel", K: int, D: int) -> Tensor:
    """Extract per-particle score gradients from batched parameter .grad.

    Each batched parameter has grad shape [K, ...]. We flatten each
    particle's grad slice into the K x D score_grads matrix.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    score_grads = torch.zeros(K, D, device=device, dtype=dtype)
    offset = 0
    for p in model.parameters():
        per_particle = p.numel() // K
        grad = p.grad if p.grad is not None else torch.zeros_like(p)
        for k in range(K):
            score_grads[k, offset : offset + per_particle] = grad[k].reshape(-1)
        offset += per_particle
    return score_grads


def _extract_particle_state_dicts(
    model: "PACOHGPModel", K: int
) -> list[dict[str, Tensor]]:
    """Extract K individual state dicts from a batched model.

    For each batched parameter with shape [K, ...], extracts K
    individual tensors with shape [...].
    """
    particle_states = []
    for k in range(K):
        state_dict = {}
        for name, param in model.named_parameters():
            state_dict[name] = param.data[k].clone()
        particle_states.append(state_dict)
    return particle_states


# =============================================================================
# PACOH-GP Prediction Model (Batched GP)
# =============================================================================


class PACOHGPModel(ExactGP, GPyTorchModel):
    """PACOH-GP prediction model with K particles as a batched GP.

    Follows the SaasFullyBayesianSingleTaskGP pattern: creates all GP
    components with batch_shape=[K], then wraps predictions in
    GaussianMixturePosterior for mixture-of-GPs inference.

    Args:
        train_X: `n x d` tensor of training inputs.
        train_Y: `n` tensor of training targets.
        num_particles: Number of particles K.
        hidden_dims: MLP hidden layer sizes.
        use_linear_mean: Whether to use linear mean on features.
        likelihood: Gaussian likelihood with batch_shape=[K].
    """

    _num_outputs: int = 1
    _is_ensemble = True

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_particles: int,
        hidden_dims: tuple[int, ...] = (8,),
        use_linear_mean: bool = True,
        likelihood: GaussianLikelihood | None = None,
    ) -> None:
        if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)

        batch_shape = torch.Size([num_particles])

        if likelihood is None:
            likelihood = GaussianLikelihood(batch_shape=batch_shape)

        super().__init__(train_X, train_Y, likelihood)

        input_dim = train_X.shape[-1]
        self.num_particles = num_particles

        # Shared feature extractor with batch_shape
        self.feature_extractor = MLPFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            batch_shape=batch_shape,
        )

        # Mean module
        if use_linear_mean:
            self.mean_module = HyperBOLinearMean(
                feature_extractor=self.feature_extractor,
                output_dim=1,
                batch_shape=batch_shape,
            )
        else:
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)

        # Covariance module (deep kernel with batched base kernel)
        base_kernel = MaternKernel(nu=2.5, batch_shape=batch_shape)
        self.covar_module = HyperBODeepKernel(
            feature_extractor=self.feature_extractor,
            base_kernel=ScaleKernel(base_kernel, batch_shape=batch_shape),
        )

        self.hidden_dims = hidden_dims
        self.use_linear_mean = use_linear_mean
        self.to(train_X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Compute batched GP prior at inputs X.

        Args:
            X: `... x n x d` tensor of inputs.

        Returns:
            MultivariateNormal with batch_shape [K, ...].
        """
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        """Compute mixture posterior by averaging over K particles.

        Follows the SaasFullyBayesianSingleTaskGP pattern: unsqueezes X
        at MCMC_DIM to broadcast over particle batch, then wraps in
        GaussianMixturePosterior.

        Args:
            X: `... x q x d` tensor of test inputs.
            output_indices: Output indices (unused, for API compatibility).
            observation_noise: Whether to include observation noise.
            posterior_transform: Optional posterior transform.
            **kwargs: Additional keyword arguments.

        Returns:
            GaussianMixturePosterior with mixture_mean and mixture_variance.
        """
        # Unsqueeze at MCMC_DIM to broadcast over particle batch
        X_expanded = X.unsqueeze(MCMC_DIM)
        posterior = super().posterior(
            X=X_expanded,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        return GaussianMixturePosterior(distribution=posterior.distribution)

    def load_particles(self, particle_states: list[dict[str, Tensor]]) -> None:
        """Load K particle state dicts into the batched model.

        For each parameter key, stacks K individual tensors (e.g., [out, in])
        into a single batched tensor (e.g., [K, out, in]).

        Args:
            particle_states: List of K state dicts, one per particle.
        """
        K = len(particle_states)
        if K != self.num_particles:
            raise ValueError(f"Expected {self.num_particles} particle states, got {K}")

        # Stack all particle states into batched tensors
        batched_state = {}
        for key in particle_states[0]:
            stacked = torch.stack([ps[key] for ps in particle_states], dim=0)
            batched_state[key] = stacked

        # Load into the model - need to match the model's state dict keys
        model_state = self.state_dict()
        for key, value in batched_state.items():
            if key in model_state:
                model_state[key] = value

        self.load_state_dict(model_state, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        pacoh_prior: PACOHPriorContainer,
        train_X: Tensor,
        train_Y: Tensor,
        freeze_pretrained: bool = True,
    ) -> "PACOHGPModel":
        """Create a PACOHGPModel from a pre-trained container.

        Args:
            pacoh_prior: Pre-trained PACOHPriorContainer with K particles.
            train_X: Training inputs for GP conditioning.
            train_Y: Training targets for GP conditioning.
            freeze_pretrained: If True, freeze all pre-trained parameters.

        Returns:
            PACOHGPModel ready for posterior inference.
        """
        if train_X.shape[-1] != pacoh_prior.input_dim:
            raise ValueError(
                f"train_X feature dimension ({train_X.shape[-1]}) must match "
                f"pre-trained input_dim ({pacoh_prior.input_dim})"
            )

        config = pacoh_prior.config

        model = cls(
            train_X=train_X,
            train_Y=train_Y,
            num_particles=pacoh_prior.num_particles,
            hidden_dims=config.hidden_dims,
            use_linear_mean=config.use_linear_mean,
        )

        model.load_particles(pacoh_prior.particle_states)

        if freeze_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        return model
