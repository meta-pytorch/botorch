#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""HyperBO: Pre-trained Gaussian Processes for Bayesian Optimization.

This module implements PyTorch/BoTorch versions of the EKL and NLL-based HyperBO
models from the paper "Pre-trained Gaussian Processes for Bayesian Optimization"
(Wang et al., arXiv:2109.08215).

The implementation provides:
- MLPFeatureExtractor: Neural network for extracting features from inputs
- HyperBOLinearMean: Linear mean function on MLP features
- HyperBODeepKernel: Deep kernel applying a base kernel to MLP features
- HyperBOModel: Complete GP model with neural network-based priors
- HyperBONLL: Negative log-likelihood loss for pre-training
- HyperBOEKL: Empirical KL divergence loss for pre-training

References:
    .. [Wang2021hyperbo]
        Z. Wang, G. E. Dahl, K. Swersky, C. Lee, Z. Nado, J. Gilmer,
        J. Snoek, and Z. Ghahramani.
        Pre-trained Gaussian Processes for Bayesian Optimization.
        arXiv preprint arXiv:2109.08215. 2021.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gpytorch
import torch
from botorch.models.empirical_gps.utils import BatchedLinear, ExperimentDataset
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import Prior
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import nn, Tensor


# =============================================================================
# Pre-trained Prior Container
# =============================================================================


@dataclass(frozen=True)
class HyperBOPriorContainer:
    """Container for pre-trained HyperBO prior state.

    This container holds the neural network weights and kernel hyperparameters
    learned during pre-training. All state dict tensors have requires_grad=False.

    The container is designed to be:
    - Immutable after creation (frozen=True)
    - Serializable (can be saved/loaded for caching)
    - Complete (contains everything needed to create a model)

    Attributes:
        input_dim: Input dimension for the feature extractor.
        hidden_dims: MLP hidden layer dimensions.
        use_linear_mean: Whether linear mean was used.
        feature_extractor_state: State dict for MLPFeatureExtractor.
        mean_module_state: State dict for mean module (or None if ZeroMean).
        covar_module_state: State dict for HyperBODeepKernel.
        likelihood_state: State dict for GaussianLikelihood.
        num_datasets: Number of datasets used for pre-training.
        loss_type: "NLL" or "EKL" (which loss was used).
        final_loss: Final loss value after training (optional).
    """

    # Configuration
    input_dim: int
    hidden_dims: tuple[int, ...]
    use_linear_mean: bool

    # Pre-trained State (all frozen state dicts)
    feature_extractor_state: dict[str, Tensor]
    mean_module_state: dict[str, Tensor] | None
    covar_module_state: dict[str, Tensor]
    likelihood_state: dict[str, Tensor]

    # Training metadata
    num_datasets: int
    loss_type: str
    final_loss: float | None = None

    def save(self, path: str) -> None:
        """Save the container to disk.

        Args:
            path: File path to save the container.
        """
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "HyperBOPriorContainer":
        """Load a container from disk.

        Args:
            path: File path to load the container from.

        Returns:
            Loaded HyperBOPriorContainer.
        """
        return torch.load(path, weights_only=False)


def _freeze_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Detach and freeze all tensors in a state dict.

    Args:
        state_dict: Dictionary mapping parameter names to tensors.

    Returns:
        New dictionary with all tensors detached and requires_grad=False.
    """
    return {k: v.detach().clone().requires_grad_(False) for k, v in state_dict.items()}


def _extract_frozen_state(
    model: "HyperBOModel",
    num_datasets: int,
    loss_type: str,
    final_loss: float | None = None,
) -> HyperBOPriorContainer:
    """Extract frozen state from a trained HyperBOModel.

    All tensors in the returned container have requires_grad=False.

    Args:
        model: Trained HyperBOModel to extract state from.
        num_datasets: Number of datasets used for pre-training.
        loss_type: Loss type used ("NLL" or "EKL").
        final_loss: Final loss value after training (optional).

    Returns:
        HyperBOPriorContainer with frozen state dicts.
    """
    return HyperBOPriorContainer(
        input_dim=model.feature_extractor.input_dim,
        hidden_dims=model.hidden_dims,
        use_linear_mean=model.use_linear_mean,
        feature_extractor_state=_freeze_state_dict(
            model.feature_extractor.state_dict()
        ),
        mean_module_state=(
            _freeze_state_dict(model.mean_module.state_dict())
            if model.use_linear_mean
            else None
        ),
        covar_module_state=_freeze_state_dict(model.covar_module.state_dict()),
        likelihood_state=_freeze_state_dict(model.likelihood.state_dict()),
        num_datasets=num_datasets,
        loss_type=loss_type,
        final_loss=final_loss,
    )


def pretrain_hyperbo(
    datasets: list[ExperimentDataset],
    input_dim: int,
    hidden_dims: tuple[int, ...] = (8,),
    use_linear_mean: bool = True,
    loss_type: Literal["NLL", "EKL"] = "NLL",
    num_iterations: int = 1000,
    learning_rate: float = 1e-3,
    ekl_optimizer: Literal["adam", "lbfgs"] = "adam",
    ekl_lbfgs_iterations: int = 100,
    subsample_size: int | None = None,
    task_batch_size: int | None = None,
    lengthscale_prior: Prior | None = None,
    outputscale_prior: Prior | None = None,
    noise_prior: Prior | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.double,
    verbose: bool = False,
) -> HyperBOPriorContainer:
    """Pre-train HyperBO on historical datasets.

    This function handles the complete pre-training workflow:
    1. Create a temporary HyperBOModel for training
    2. Create the appropriate loss function (NLL or EKL)
    3. Run optimization loop
    4. Extract and freeze all learned parameters
    5. Return a container with frozen state

    Args:
        datasets: List of ExperimentDataset for pre-training.
        input_dim: Dimension of input features.
        hidden_dims: MLP hidden layer sizes.
        use_linear_mean: Whether to use linear mean function.
        loss_type: "NLL" for negative log-likelihood, "EKL" for empirical KL.
        num_iterations: Number of optimization iterations (gradient steps).
        learning_rate: Learning rate for Adam optimizer.
        ekl_optimizer: Optimizer for the EKL objective, "adam" or "lbfgs".
            Wang et al. (2021) use L-BFGS for EKL and Adam only for NLL.
        ekl_lbfgs_iterations: L-BFGS iterations when ekl_optimizer="lbfgs".
        subsample_size: Optional per-dataset datapoint subsampling for NLL loss.
        task_batch_size: Optional number of tasks to sample per iteration for
            mini-batched optimization over tasks (NLL only). The loss is rescaled
            to produce an unbiased gradient estimate. If None, uses all tasks.
        lengthscale_prior: Optional prior on kernel lengthscale.
        outputscale_prior: Optional prior on kernel outputscale.
        noise_prior: Optional prior on likelihood noise.
        device: Device to train on.
        dtype: Data type (recommend torch.double).
        verbose: If True, print training progress.

    Returns:
        HyperBOPriorContainer with frozen pre-trained state.

    Example:
        >>> prior = pretrain_hyperbo(
        ...     datasets=historical_datasets,
        ...     input_dim=5,
        ...     hidden_dims=(8,),
        ...     loss_type="NLL",
        ...     num_iterations=500,
        ... )
        >>> prior.save("hyperbo_prior.pt")
        >>>
        >>> # Later, create models for different test sets
        >>> for train_X, train_Y in test_sets:
        ...     model = HyperBOModel.from_pretrained(
        ...         prior, train_X, train_Y
        ...     )
    """
    if len(datasets) == 0:
        raise ValueError("datasets cannot be empty")

    # Determine device and dtype from first dataset if not specified
    if device is None:
        device = datasets[0].X.device
    if dtype is None:
        dtype = datasets[0].X.dtype

    # Create dummy training data for model initialization
    # The actual training uses the datasets via the loss function
    dummy_X = torch.zeros(1, input_dim, device=device, dtype=dtype)
    dummy_Y = torch.zeros(1, device=device, dtype=dtype)

    # Create model
    model = HyperBOModel(
        train_X=dummy_X,
        train_Y=dummy_Y,
        hidden_dims=hidden_dims,
        use_linear_mean=use_linear_mean,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        noise_prior=noise_prior,
    )
    model.to(device=device, dtype=dtype)
    model.train()

    # Create loss function
    if loss_type == "NLL":
        mll = HyperBONLL(
            likelihood=model.likelihood,
            model=model,
            datasets=datasets,
        )
    elif loss_type == "EKL":
        # EKL requires matching inputs across all datasets
        if not validate_matching_inputs(datasets):
            raise ValueError(
                "EKL loss requires all datasets to have the same input locations. "
                "Use 'NLL' loss for datasets with different input locations."
            )
        mll = HyperBOEKL(
            likelihood=model.likelihood,
            model=model,
            datasets=datasets,
            matching_inputs=datasets[0].X,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'NLL' or 'EKL'.")

    # Setup optimizer. EKL is optimized with L-BFGS by default in the reference
    # implementation (Wang et al. 2021 use 100 L-BFGS iterations); they note Adam was
    # needed for NLL only because a Matern NLL over many datapoints becomes numerically
    # low-rank, "This problem did not seem to occur for EKL because of much fewer
    # matching datapoints." Optimizing EKL with Adam therefore under-trains it.
    use_lbfgs = loss_type == "EKL" and ekl_optimizer == "lbfgs"
    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=ekl_lbfgs_iterations,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=True)

    # Training loop
    final_loss = None
    if use_lbfgs:

        def _closure() -> Tensor:
            optimizer.zero_grad()
            loss = -mll.compute_loss()
            loss.backward()
            return loss

        # LBFGS runs ekl_lbfgs_iterations inner steps per .step() call.
        optimizer.step(_closure)
        with torch.no_grad():
            final_loss = float(-mll.compute_loss())
        if verbose:
            print(f"EKL/L-BFGS ({ekl_lbfgs_iterations} iters), loss: {final_loss:.4f}")
    else:
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Compute loss (NLL/EKL returns log prob, so negate for minimization)
            if loss_type == "NLL":
                loss = -mll.compute_loss(
                    subsample_size=subsample_size,
                    task_batch_size=task_batch_size,
                )
            else:
                # EKL: use compute_loss method which calls model.forward() directly
                # to avoid GPyTorch's ExactGP training input check
                loss = -mll.compute_loss()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            if verbose and (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1}/{num_iterations}, "
                    f"Loss: {final_loss:.4f}"
                )

    # Extract and return frozen state
    return _extract_frozen_state(
        model=model,
        num_datasets=len(datasets),
        loss_type=loss_type,
        final_loss=final_loss,
    )


# =============================================================================
# Phase 1: Neural Network Feature Extractor
# =============================================================================


class MLPFeatureExtractor(nn.Module):
    """Multi-layer perceptron for extracting features from inputs.

    Used by both the mean function (linear head on features) and
    kernel (Matérn on features).

    Architecture options:
    - Original paper (Section 7.2): Single hidden layer with 8 nodes, tanh activation
    - Extended version: 2 hidden layers of size (32, 32) with tanh activation

    Use `hidden_dims=(8,)` to match the original paper exactly.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: Tuple of hidden layer sizes. Use `(8,)` for original paper,
            `(32, 32)` for extended. Default is `(8,)` to match the paper.
        batch_shape: Leading batch dimensions for batched evaluation. When empty
            (default), uses standard nn.Linear. When non-empty (e.g., [K] for K
            PACOH particles), uses BatchedLinear for parallel evaluation.
        activation: Activation function class. Default is `nn.Tanh`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (8,),
        batch_shape: torch.Size = torch.Size(),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]

        # Build MLP layers
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if len(batch_shape) > 0:
                layers.append(
                    BatchedLinear(prev_dim, hidden_dim, batch_shape=batch_shape)
                )
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from inputs.

        Args:
            x: `... x input_dim` tensor of inputs.

        Returns:
            `... x output_dim` tensor of feature embeddings.
        """
        return self.mlp(x)


# =============================================================================
# Phase 2: HyperBO Mean Module
# =============================================================================


class HyperBOLinearMean(Mean):
    """Linear mean function on neural network features.

    μ(x) = W @ φ(x) + b where φ is the MLP feature extractor.

    Args:
        feature_extractor: MLP feature extractor (shared with kernel).
        output_dim: Dimension of the mean output (typically 1).
        batch_shape: Leading batch dimensions for batched evaluation. When empty
            (default), uses standard nn.Linear. When non-empty (e.g., [K] for K
            PACOH particles), uses BatchedLinear for parallel evaluation.
        use_bias: Whether to include a bias term.
    """

    def __init__(
        self,
        feature_extractor: MLPFeatureExtractor,
        output_dim: int = 1,
        batch_shape: torch.Size = torch.Size(),
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        if len(batch_shape) > 0:
            self.linear: nn.Linear | BatchedLinear = BatchedLinear(
                feature_extractor.output_dim,
                output_dim,
                batch_shape=batch_shape,
                bias=use_bias,
            )
        else:
            self.linear = nn.Linear(
                feature_extractor.output_dim, output_dim, bias=use_bias
            )

    def forward(self, x: Tensor) -> Tensor:
        """Compute mean values.

        Args:
            x: `... x input_dim` tensor of inputs.

        Returns:
            `...` tensor of mean values (output_dim squeezed if 1).
        """
        features = self.feature_extractor(x)
        mean = self.linear(features)
        # Squeeze the last dimension if output_dim is 1
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        return mean


# =============================================================================
# Phase 3: HyperBO Deep Kernel
# =============================================================================


class HyperBODeepKernel(Kernel):
    """Deep kernel that applies a base kernel to neural network features.

    k(x, x') = k_base(φ(x), φ(x')) where φ is the MLP feature extractor.
    Typically uses Matérn-5/2 as the base kernel (as in the paper).

    Args:
        feature_extractor: MLP feature extractor (shared with mean).
        base_kernel: Base kernel to apply to features. If None, uses
            ScaleKernel(MaternKernel(nu=2.5)) with ARD.
    """

    has_lengthscale = False  # The base kernel handles lengthscale

    def __init__(
        self,
        feature_extractor: MLPFeatureExtractor,
        base_kernel: Kernel | None = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

        if base_kernel is None:
            # Default: ScaleKernel(MaternKernel) with ARD
            base_kernel = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=feature_extractor.output_dim,
                )
            )
        self.base_kernel = base_kernel

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Compute kernel matrix.

        Args:
            x1: `... x n1 x input_dim` tensor of inputs.
            x2: `... x n2 x input_dim` tensor of inputs.
            diag: If True, return only diagonal elements.
            last_dim_is_batch: If True, last dimension is batch.

        Returns:
            Kernel matrix of shape `... x n1 x n2` or `... x n` if diag=True.
        """
        # Extract features
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)

        # Apply base kernel to features
        return self.base_kernel(
            features1,
            features2,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **kwargs,
        )


# =============================================================================
# Phase 4: HyperBO Model Class
# =============================================================================


class HyperBOModel(ExactGP, GPyTorchModel):
    """HyperBO Gaussian Process model with neural network-based priors.

    The model uses:
    - Mean function: Linear projection of MLP features (or zero mean)
    - Kernel: Matérn-5/2 on MLP features (deep kernel)
    - Likelihood: Gaussian with learnable noise variance

    Compatible with both EKL and NLL loss functions for pre-training.

    IMPORTANT: After pre-training, call `freeze_pretrained_parameters()` to
    freeze the neural network weights. The original HyperBO paper never updates
    kernel hyperparameters during BO after pre-training.

    Args:
        train_X: `n x d` tensor of training inputs.
        train_Y: `n x 1` or `n` tensor of training targets.
        hidden_dims: Tuple of hidden layer sizes for MLP. Use `(8,)` for
            original paper, `(32, 32)` for extended. Default is `(8,)`.
        use_linear_mean: If True, use linear mean on features. If False, use
            zero mean.
        likelihood: Gaussian likelihood. If None, creates one.
        lengthscale_prior: Optional prior on kernel lengthscale.
        outputscale_prior: Optional prior on kernel outputscale.
        noise_prior: Optional prior on likelihood noise.
        mean_prior: Optional prior on mean function weights.
    """

    _num_outputs: int = 1

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        hidden_dims: tuple[int, ...] = (8,),
        use_linear_mean: bool = True,
        likelihood: GaussianLikelihood | None = None,
        lengthscale_prior: Prior | None = None,
        outputscale_prior: Prior | None = None,
        noise_prior: Prior | None = None,
        mean_prior: Prior | None = None,
    ) -> None:
        # Squeeze Y if needed
        if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)

        # Create likelihood if not provided
        if likelihood is None:
            likelihood = GaussianLikelihood(noise_prior=noise_prior)

        super().__init__(train_X, train_Y, likelihood)

        # Get input dimension
        input_dim = train_X.shape[-1]

        # Create shared feature extractor
        self.feature_extractor = MLPFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        )

        # Create mean module
        if use_linear_mean:
            self.mean_module = HyperBOLinearMean(
                feature_extractor=self.feature_extractor,
                output_dim=1,
            )
        else:
            # Zero mean
            self.mean_module = gpytorch.means.ZeroMean()

        # Create covariance module (deep kernel)
        # Note: We use an isotropic Matérn kernel (single lengthscale) rather than
        # ARD lengthscales because the MLP feature extractor can already learn
        # per-feature scaling. Using ARD would be redundant since the MLP's last
        # layer weights and ARD lengthscales can absorb each other, leading to
        # overparameterization.
        base_kernel = MaternKernel(
            nu=2.5,
            lengthscale_prior=lengthscale_prior,
        )
        self.covar_module = HyperBODeepKernel(
            feature_extractor=self.feature_extractor,
            base_kernel=ScaleKernel(
                base_kernel,
                outputscale_prior=outputscale_prior,
            ),
        )

        # Store configuration
        self.hidden_dims = hidden_dims
        self.use_linear_mean = use_linear_mean

        # Move to same device/dtype as training data
        self.to(train_X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Compute GP prior at inputs X.

        Args:
            X: `n x d` tensor of inputs.

        Returns:
            MultivariateNormal distribution.
        """
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    def freeze_pretrained_parameters(self) -> None:
        """Freeze all pre-trained parameters after fitting.

        The original HyperBO paper freezes kernel hyperparameters during BO.
        Call this method after pre-training to match the paper's behavior.

        Note: Likelihood noise is NOT frozen by default, allowing for
        task-specific noise adaptation.
        """
        # Freeze feature extractor (MLP weights)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze mean module parameters
        for param in self.mean_module.parameters():
            param.requires_grad = False

        # Freeze covariance module parameters (kernel hyperparameters)
        for param in self.covar_module.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters (for fine-tuning if needed)."""
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(
        cls,
        hyperbo_prior: HyperBOPriorContainer,
        train_X: Tensor,
        train_Y: Tensor,
        freeze_pretrained: bool = True,
    ) -> "HyperBOModel":
        """Create a HyperBOModel from a pre-trained prior.

        This method:
        1. Creates a new model with matching architecture
        2. Loads pre-trained weights from the container
        3. Optionally freezes all pre-trained parameters

        Args:
            hyperbo_prior: Pre-trained HyperBOPriorContainer.
            train_X: Training inputs for GP conditioning.
            train_Y: Training targets for GP conditioning.
            freeze_pretrained: If True (default), freeze all pre-trained
                parameters. Set to False for fine-tuning.

        Returns:
            HyperBOModel initialized with pre-trained weights.

        Example:
            >>> prior = HyperBOPriorContainer.load("hyperbo_prior.pt")
            >>> model = HyperBOModel.from_pretrained(prior, train_X, train_Y)
            >>> # model.forward() uses pre-trained features/kernel
            >>> posterior = model.posterior(test_X)
        """
        # Validate dimensions
        if train_X.shape[-1] != hyperbo_prior.input_dim:
            raise ValueError(
                f"train_X feature dimension ({train_X.shape[-1]}) must match "
                f"pre-trained input_dim ({hyperbo_prior.input_dim})"
            )

        # Create model with matching architecture
        model = cls(
            train_X=train_X,
            train_Y=train_Y,
            hidden_dims=hyperbo_prior.hidden_dims,
            use_linear_mean=hyperbo_prior.use_linear_mean,
        )

        # Load pre-trained state dicts
        model.feature_extractor.load_state_dict(hyperbo_prior.feature_extractor_state)
        if hyperbo_prior.mean_module_state is not None:
            model.mean_module.load_state_dict(hyperbo_prior.mean_module_state)
        model.covar_module.load_state_dict(hyperbo_prior.covar_module_state)
        model.likelihood.load_state_dict(hyperbo_prior.likelihood_state)

        # Freeze parameters if requested
        if freeze_pretrained:
            model.freeze_pretrained_parameters()

        return model


# =============================================================================
# Phase 5: NLL Loss Function
# =============================================================================


class HyperBONLL(MarginalLogLikelihood):
    """Negative Log-Likelihood loss for HyperBO pre-training.

    From paper Eq. 13:
        L(μ, k∘σ²) ≈ -1/N Σᵢ log p(D_fᵢ | μ, k∘σ²)

    where each log marginal likelihood is computed as (Eq. 14):
        log p(D_fᵢ | μ, k, σ²) = -1/2 * [(yᵢ - μᵢ)ᵀ Σᵢ⁻¹ (yᵢ - μᵢ)
                                        + log|Σᵢ| + Mᵢ log(2π)]

    This is the more flexible objective that works with different input
    locations across tasks.

    Args:
        likelihood: Gaussian likelihood.
        model: HyperBOModel instance.
        datasets: List of ExperimentDataset instances for pre-training.
    """

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        model: HyperBOModel,
        datasets: list[ExperimentDataset],
    ) -> None:
        super().__init__(likelihood, model)
        self.datasets = datasets
        self._num_data = sum(d.X.shape[0] for d in datasets)

    def forward(self, function_dist: MultivariateNormal, target: Tensor) -> Tensor:
        """Compute the sum of log marginal likelihoods across all datasets.

        Note: This method computes the MLL for each dataset separately,
        ignoring the function_dist and target arguments (which are for the
        model's train_X/train_Y). This allows compatibility with fit_gpytorch_mll.

        Args:
            function_dist: Function distribution (unused, for API compatibility).
            target: Target values (unused, for API compatibility).

        Returns:
            Sum of log marginal likelihoods normalized by total data points.
        """
        return self.compute_loss()

    def compute_loss(
        self,
        subsample_size: int | None = None,
        task_batch_size: int | None = None,
    ) -> Tensor:
        """Compute the sum of log marginal likelihoods across all datasets.

        This is the simplified API that doesn't require dummy arguments.

        Args:
            subsample_size: If provided, randomly subsample this many points
                from each dataset for stochastic optimization (as described in
                the HyperBO paper, Section 7.2). If None, uses all data points
                (full batch). This enables memory-efficient training on large
                datasets.
            task_batch_size: If provided, randomly sample this many tasks
                (datasets) per call for mini-batched optimization. The loss is
                rescaled by n_total / n_batch to produce an unbiased gradient
                estimate. If None, uses all tasks (full batch).

        Returns:
            Sum of log marginal likelihoods normalized by total data points.
        """
        # Get device and dtype from model parameters
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        total_log_prob = torch.tensor(0.0, device=device, dtype=dtype)

        # Mini-batch over tasks if requested
        n_total_tasks = len(self.datasets)
        if task_batch_size is not None and task_batch_size < n_total_tasks:
            task_indices = torch.randperm(n_total_tasks)[:task_batch_size]
            task_batch = [self.datasets[i] for i in task_indices]
        else:
            task_batch = self.datasets

        for dataset in task_batch:
            X = dataset.X
            Y = dataset.Y
            if Y.ndim == 2 and Y.shape[-1] == 1:
                Y = Y.squeeze(-1)

            # Stochastic subsampling per dataset (HyperBO paper Section 7.2)
            if subsample_size is not None and X.shape[0] > subsample_size:
                indices = torch.randperm(X.shape[0], device=X.device)[:subsample_size]
                X = X[indices]
                Y = Y[indices]

            # Compute GP prior distribution at dataset inputs directly via forward
            # to avoid GPyTorch's ExactGP training input checks
            output = self.model.forward(X)

            # Add likelihood noise
            output = self.likelihood(output)

            # Compute log probability
            log_prob = output.log_prob(Y)
            total_log_prob = total_log_prob + log_prob

        # Rescale for task mini-batching to get unbiased gradient estimate
        n_batch = len(task_batch)
        if n_batch < n_total_tasks:
            total_log_prob = total_log_prob * (n_total_tasks / n_batch)

        # Normalize by total number of data points (original, not subsampled)
        total_log_prob = total_log_prob / self._num_data

        # Add log probs of hyperparameter priors
        total_log_prob = total_log_prob + self._add_prior_log_probs()

        return total_log_prob

    def _add_prior_log_probs(self) -> Tensor:
        """Add log probabilities from hyperparameter priors."""
        prior_log_prob = torch.tensor(
            0.0,
            dtype=self.model.train_inputs[0].dtype,
            device=self.model.train_inputs[0].device,
        )
        for _, module, prior, closure, _ in self.model.named_priors():
            prior_log_prob = prior_log_prob + prior.log_prob(closure(module)).sum()
        return prior_log_prob


# =============================================================================
# Phase 6: EKL Loss Function
# =============================================================================


class HyperBOEKL(MarginalLogLikelihood):
    """Empirical KL Divergence loss for HyperBO pre-training.

    From paper Section 5.2:
        D_KL(N(μ̃, Σ̃), N(μ(x), k∘σ²(x)))

    where μ̃, Σ̃ are MLE estimates from matching-input data:
        μ̃ = (1/N) Y @ 1_N
        Σ̃ = (1/N) (Y - μ̃ @ 1_N^T) @ (Y - μ̃ @ 1_N^T)^T

    For degenerate Σ̃ (rank < M), uses SVD to project onto support.

    This objective is used when all tasks share the same input locations.

    Args:
        likelihood: Gaussian likelihood.
        model: HyperBOModel instance.
        datasets: List of ExperimentDataset instances for pre-training.
            All datasets must have observations at the same input locations.
        matching_inputs: The shared input locations across all datasets.
    """

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        model: HyperBOModel,
        datasets: list[ExperimentDataset],
        matching_inputs: Tensor,
    ) -> None:
        super().__init__(likelihood, model)
        self.datasets = datasets
        self.matching_inputs = matching_inputs
        self._num_tasks = len(datasets)

        # Validate matching inputs
        if not validate_matching_inputs(datasets):
            raise ValueError(
                "All datasets must have the same input locations for EKL loss."
            )

        # Precompute MLE estimates
        self._mu_tilde, self._Sigma_tilde = compute_mle_estimates(datasets)

    def forward(self, function_dist: MultivariateNormal, target: Tensor) -> Tensor:
        """Compute the negative EKL divergence.

        Since we want to minimize KL, and fit_gpytorch_mll maximizes the MLL,
        we return -KL.

        Args:
            function_dist: Function distribution (unused, for API compatibility).
            target: Target values (unused, for API compatibility).

        Returns:
            Negative KL divergence (to be maximized).
        """
        return self.compute_loss()

    def compute_loss(self) -> Tensor:
        """Compute the negative EKL divergence.

        This is the simplified API that doesn't require dummy arguments
        and uses model.forward() directly to avoid GPyTorch's ExactGP
        training input check.

        Returns:
            Negative KL divergence (to be maximized).
        """
        # Compute model mean and covariance at matching inputs
        # Use model.forward() directly to avoid GPyTorch's training input check
        self.model.train()
        output = self.model.forward(self.matching_inputs)

        # Add likelihood noise to get full covariance
        output = self.likelihood(output)

        mu_model = output.mean
        Sigma_model = output.covariance_matrix

        # Compute KL divergence
        kl_div = compute_gaussian_kl_divergence(
            mu_p=self._mu_tilde,
            Sigma_p=self._Sigma_tilde,
            mu_q=mu_model,
            Sigma_q=Sigma_model,
            handle_degenerate=True,
        )

        # Return negative KL (to be maximized)
        neg_kl = -kl_div

        # Add log probs of hyperparameter priors
        neg_kl = neg_kl + self._add_prior_log_probs()

        return neg_kl

    def _add_prior_log_probs(self) -> Tensor:
        """Add log probabilities from hyperparameter priors."""
        prior_log_prob = torch.tensor(
            0.0,
            dtype=self.model.train_inputs[0].dtype,
            device=self.model.train_inputs[0].device,
        )
        for _, module, prior, closure, _ in self.model.named_priors():
            prior_log_prob = prior_log_prob + prior.log_prob(closure(module)).sum()
        return prior_log_prob


# =============================================================================
# Phase 7: Helper Functions
# =============================================================================


def compute_mle_estimates(
    datasets: list[ExperimentDataset],
) -> tuple[Tensor, Tensor]:
    """Compute MLE estimates μ̃ and Σ̃ for matching-input datasets.

    From paper Eq. 4:
        μ̃ = (1/N) Y @ 1_N = mean(Y, dim=tasks)
        Σ̃ = (1/N) (Y - μ̃ @ 1_N^T) @ (Y - μ̃ @ 1_N^T)^T

    Args:
        datasets: List of ExperimentDataset instances. All must have
            the same input locations.

    Returns:
        Tuple of (mu_tilde, Sigma_tilde):
        - mu_tilde: `M`-dim tensor of empirical mean at each input location.
        - Sigma_tilde: `M x M` tensor of empirical covariance matrix.
    """
    # Stack Y values from all datasets: shape (N_tasks, M)
    # where M is the number of input locations
    Y_stack = torch.stack([d.Y.squeeze(-1) for d in datasets], dim=0)

    # Compute empirical mean: (M,)
    mu_tilde = Y_stack.mean(dim=0)

    # Compute centered data: (N_tasks, M)
    Y_centered = Y_stack - mu_tilde.unsqueeze(0)

    # Compute empirical covariance: (M, M)
    # Using (N-1) for unbiased estimate
    N = Y_stack.shape[0]
    Sigma_tilde = Y_centered.T @ Y_centered / (N - 1)

    return mu_tilde, Sigma_tilde


def compute_gaussian_kl_divergence(
    mu_p: Tensor,
    Sigma_p: Tensor,
    mu_q: Tensor,
    Sigma_q: Tensor,
    handle_degenerate: bool = True,
) -> Tensor:
    """Compute KL divergence D_KL(N(μ_p, Σ_p) || N(μ_q, Σ_q)).

    For non-degenerate case (Eq. 7):
        D_KL = 1/2 * (tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
                      + log|Σ_q|/|Σ_p| - M)

    For degenerate Σ_p (Eq. 6): project onto support via SVD.

    Args:
        mu_p: Mean of distribution p.
        Sigma_p: Covariance of distribution p.
        mu_q: Mean of distribution q.
        Sigma_q: Covariance of distribution q.
        handle_degenerate: If True, handle degenerate Sigma_p via SVD projection.

    Returns:
        KL divergence (scalar).
    """
    M = mu_p.shape[0]
    dtype = mu_p.dtype

    if handle_degenerate:
        # Check if Sigma_p is degenerate (rank < M)
        # Use eigenvalue decomposition to determine rank
        eigvals = torch.linalg.eigvalsh(Sigma_p)
        # Count eigenvalues above a threshold
        tol = eigvals.max() * M * torch.finfo(dtype).eps
        rank = (eigvals > tol).sum().item()

        if rank < M:
            # Degenerate case: use SVD projection
            return _compute_kl_with_low_rank_support(mu_p, Sigma_p, mu_q, Sigma_q, rank)

    # Non-degenerate case: standard KL divergence formula
    # Compute Cholesky of Sigma_q for stable inversion
    L_q = psd_safe_cholesky(Sigma_q)

    # Solve for Sigma_q^{-1} @ Sigma_p
    # Sigma_q^{-1} @ Sigma_p = L_q^{-T} @ L_q^{-1} @ Sigma_p
    Sigma_q_inv_Sigma_p = torch.cholesky_solve(Sigma_p, L_q)

    # Trace term: tr(Sigma_q^{-1} @ Sigma_p)
    trace_term = torch.trace(Sigma_q_inv_Sigma_p)

    # Quadratic term: (μ_q - μ_p)^T Sigma_q^{-1} (μ_q - μ_p)
    mu_diff = mu_q - mu_p
    mu_diff_solved = torch.cholesky_solve(mu_diff.unsqueeze(-1), L_q).squeeze(-1)
    quad_term = mu_diff @ mu_diff_solved

    # Log determinant terms
    # log|Sigma_q| = 2 * sum(log(diag(L_q)))
    log_det_q = 2 * torch.log(torch.diag(L_q)).sum()

    # For Sigma_p, compute log determinant via Cholesky
    L_p = psd_safe_cholesky(Sigma_p)
    log_det_p = 2 * torch.log(torch.diag(L_p)).sum()

    log_det_term = log_det_q - log_det_p

    # KL divergence
    kl = 0.5 * (trace_term + quad_term + log_det_term - M)

    return kl


def _compute_kl_with_low_rank_support(
    mu_p: Tensor,
    Sigma_p: Tensor,
    mu_q: Tensor,
    Sigma_q: Tensor,
    rank: int,
) -> Tensor:
    """Handle degenerate estimated covariance via SVD projection.

    When rank(Σ_p) = r < M:
    1. Compute SVD: Σ_p = V @ Λ @ V^T where Λ ∈ R^{r×r}
    2. Set A = V @ √Λ and compute pseudoinverse A⁺
    3. Project: μ_proj = A⁺ @ (μ - μ̃), Σ_proj = A⁺ @ Σ @ (A⁺)^T
    4. Compute D_KL(N(0, I), N(μ_proj, Σ_proj))

    Args:
        mu_p: Mean of distribution p.
        Sigma_p: Degenerate covariance of distribution p.
        mu_q: Mean of distribution q.
        Sigma_q: Covariance of distribution q.
        rank: Rank of Sigma_p.

    Returns:
        KL divergence (scalar).
    """
    # Eigendecomposition of Sigma_p
    eigvals, eigvecs = torch.linalg.eigh(Sigma_p)

    # Take top r eigenvalues/vectors
    eigvals_r = eigvals[-rank:]
    V_r = eigvecs[:, -rank:]

    # A = V_r @ sqrt(Λ_r)
    sqrt_eigvals = torch.sqrt(eigvals_r)

    # Pseudoinverse A⁺ = (A^T A)^{-1} A^T = diag(1/sqrt(λ)) @ V_r^T
    A_pinv = (V_r / sqrt_eigvals.unsqueeze(0)).T  # (r, M)

    # Project mean difference
    mu_diff = mu_q - mu_p
    mu_proj = A_pinv @ mu_diff  # (r,)

    # Project Sigma_q
    Sigma_proj = A_pinv @ Sigma_q @ A_pinv.T  # (r, r)

    # Now compute KL(N(0, I_r) || N(mu_proj, Sigma_proj))
    r = rank

    # Cholesky of Sigma_proj
    L_proj = psd_safe_cholesky(Sigma_proj)

    # Trace term: tr(Sigma_proj^{-1} @ I) = tr(Sigma_proj^{-1})
    Sigma_proj_inv = torch.cholesky_inverse(L_proj)
    trace_term = torch.trace(Sigma_proj_inv)

    # Quadratic term: mu_proj^T @ Sigma_proj^{-1} @ mu_proj
    mu_solved = torch.cholesky_solve(mu_proj.unsqueeze(-1), L_proj).squeeze(-1)
    quad_term = mu_proj @ mu_solved

    # Log determinant term: log|Sigma_proj| - log|I| = log|Sigma_proj|
    log_det_proj = 2 * torch.log(torch.diag(L_proj)).sum()

    # KL divergence
    kl = 0.5 * (trace_term + quad_term + log_det_proj - r)

    return kl


def validate_matching_inputs(datasets: list[ExperimentDataset]) -> bool:
    """Check if all datasets have observations at the same input locations.

    Args:
        datasets: List of ExperimentDataset instances.

    Returns:
        True if all datasets have the same input locations, False otherwise.
    """
    if len(datasets) == 0:
        return True

    reference_X = datasets[0].X
    for dataset in datasets[1:]:
        if dataset.X.shape != reference_X.shape:
            return False
        if not torch.allclose(dataset.X, reference_X):
            return False

    return True


# =============================================================================
# Phase 8: HyperBO Output Transform
# =============================================================================


@dataclass
class HyperBOTransformParams:
    """Parameters for the HyperBO output transformation.

    These parameters are computed from the training data and stored
    for use in both forward (transform) and inverse operations.

    Args:
        median: Median of the original values.
        maximum: Maximum of the original values.
        scale: Scaling factor (default 4.0).
        shift: Shift factor (default -2.0).
    """

    median: float
    maximum: float
    scale: float = 4.0
    shift: float = -2.0


def compute_hyperbo_transform_params(
    Y: Tensor,
    scale: float = 4.0,
    shift: float = -2.0,
) -> HyperBOTransformParams:
    """Compute transformation parameters from training data.

    From the HyperBO paper:
        y ← softplus(y − y_median) / softplus(y_max − y_median) * scale + shift

    Args:
        Y: Tensor of target values to compute statistics from.
        scale: Scaling factor (default 4.0, maps to range of width 4).
        shift: Shift factor (default -2.0, centers range around 0).

    Returns:
        HyperBOTransformParams containing median, maximum, scale, and shift.
    """
    Y_flat = Y.flatten()
    median = torch.median(Y_flat).item()
    maximum = Y_flat.max().item()

    return HyperBOTransformParams(
        median=median,
        maximum=maximum,
        scale=scale,
        shift=shift,
    )


def hyperbo_transform(
    Y: Tensor,
    params: HyperBOTransformParams,
) -> Tensor:
    """Apply the HyperBO output transformation.

    From the HyperBO paper (Appendix C):
        y ← softplus(y − y_median) / softplus(y_max − y_median) * scale + shift

    This transformation:
    - Centers data around the median
    - Uses softplus for smooth, bounded behavior
    - Maps to approximately [shift, scale + shift] range (e.g., [-2, 2])

    Properties:
    - At y = y_median: softplus(0) / softplus(y_max - y_median) * scale + shift
    - At y = y_max: scale + shift (e.g., 2.0)
    - Smooth and differentiable everywhere
    - Robust to outliers (softplus dampens extreme values)

    Args:
        Y: Tensor of values to transform.
        params: Transformation parameters (median, maximum, scale, shift).

    Returns:
        Transformed tensor with values in approximately [shift, scale + shift].
    """
    # Compute softplus(y - median) / softplus(y_max - median)
    y_centered = Y - params.median
    y_max_centered = params.maximum - params.median

    # softplus(x) = log(1 + exp(x))
    numerator = torch.nn.functional.softplus(y_centered)
    denominator = torch.nn.functional.softplus(
        torch.tensor(y_max_centered, dtype=Y.dtype, device=Y.device)
    )

    # Apply scaling and shifting
    Y_transformed = numerator / denominator * params.scale + params.shift

    return Y_transformed


def hyperbo_untransform(
    Y_transformed: Tensor,
    params: HyperBOTransformParams,
) -> Tensor:
    """Inverse of the HyperBO output transformation.

    Given: y_t = softplus(y - median) / softplus(y_max - median) * scale + shift

    Inverse:
        softplus(y - median) = (y_t - shift) / scale * softplus(y_max - median)
        y - median = softplus_inverse(...)
        y = median + log(exp(...) - 1)  [inverse softplus]

    Note: softplus_inverse(x) = log(exp(x) - 1) for x > 0

    Args:
        Y_transformed: Tensor of transformed values.
        params: Transformation parameters used in the forward transform.

    Returns:
        Tensor of original-scale values.
    """
    y_max_centered = params.maximum - params.median

    # Compute denominator: softplus(y_max - median)
    denominator = torch.nn.functional.softplus(
        torch.tensor(
            y_max_centered, dtype=Y_transformed.dtype, device=Y_transformed.device
        )
    )

    # Invert scaling and shifting: get softplus(y - median)
    softplus_y_centered = (Y_transformed - params.shift) / params.scale * denominator

    # Clamp to valid range for inverse softplus (must be positive)
    softplus_y_centered = torch.clamp(softplus_y_centered, min=1e-10)

    # Inverse softplus: log(exp(x) - 1)
    # Use stable computation: for large x, this is approximately x
    # For small x, use the exact formula
    y_centered = torch.where(
        softplus_y_centered > 20.0,
        softplus_y_centered,  # For large values, softplus_inverse ≈ identity
        torch.log(torch.expm1(softplus_y_centered)),  # expm1(x) = exp(x) - 1
    )

    # Add back the median
    Y_original = y_centered + params.median

    return Y_original


class HyperBOOutputTransform:
    """Callable wrapper for HyperBO output transformation.

    This class provides a convenient interface for applying the HyperBO
    transformation to learning curves or other bounded positive metrics.

    Usage:
        # Compute transform parameters from training data
        transform = HyperBOOutputTransform.from_data(Y_train)

        # Apply transformation
        Y_transformed = transform(Y)

        # Inverse transformation (e.g., for predictions)
        Y_original = transform.untransform(Y_transformed)

    Args:
        params: HyperBOTransformParams containing median, max, scale, shift.
    """

    def __init__(self, params: HyperBOTransformParams) -> None:
        self.params = params

    @classmethod
    def from_data(
        cls,
        Y: Tensor,
        scale: float = 4.0,
        shift: float = -2.0,
    ) -> "HyperBOOutputTransform":
        """Create a transform from training data.

        Args:
            Y: Training target values to compute statistics from.
            scale: Scaling factor (default 4.0).
            shift: Shift factor (default -2.0).

        Returns:
            HyperBOOutputTransform instance.
        """
        params = compute_hyperbo_transform_params(Y, scale=scale, shift=shift)
        return cls(params)

    def __call__(self, Y: Tensor) -> Tensor:
        """Apply the forward transformation.

        Args:
            Y: Tensor of values to transform.

        Returns:
            Transformed tensor.
        """
        return hyperbo_transform(Y, self.params)

    def transform(self, Y: Tensor) -> Tensor:
        """Apply the forward transformation (alias for __call__).

        Args:
            Y: Tensor of values to transform.

        Returns:
            Transformed tensor.
        """
        return self(Y)

    def untransform(self, Y_transformed: Tensor) -> Tensor:
        """Apply the inverse transformation.

        Args:
            Y_transformed: Tensor of transformed values.

        Returns:
            Tensor in original scale.
        """
        return hyperbo_untransform(Y_transformed, self.params)

    def __repr__(self) -> str:
        return (
            f"HyperBOOutputTransform(median={self.params.median:.4f}, "
            f"max={self.params.maximum:.4f}, scale={self.params.scale}, "
            f"shift={self.params.shift})"
        )
