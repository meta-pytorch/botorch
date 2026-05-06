#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Local Entropy Search (LES) acquisition function as introduced in [Stenger2026]_.

LES is an information-theoretic local Bayesian optimization method based on
posterior samples of local descent / ascent sequences from a fixed incumbent.
The implementation here follows a practical approximation:

1. Draw ``L`` posterior function samples using Matheron paths.
2. Optimize each sampled path from the same incumbent for a fixed number ``P`` of
   steps.
3. Discretize and optionally subsample each sequence, then condition the model
   on sampled function values at the retained support points as virtual
   observations.
4. Compute mutual information between the noisy observation at a candidate and
   the local sequence variable as predictive entropy minus expected conditional
   entropy over the conditioned models.

Based on the original implementation [Stenger2026]_ and the notebook by Paul Brunzema
[Brunzema2026]_.

For reproducing the original paper results exactly, refer to the original codebase
in [Stenger2026]_. The implementation here showed very similar performance to the
original LES code in the within-model comparisons.

References:

.. [Stenger2026]
   D. Stenger, A. Lindicke, A. von Rohr, and S. Trimpe.
   Local Entropy Search over Descent Sequences for Bayesian Optimization.
   In The Fourteenth International Conference on Learning Representations (ICLR),
   2026. https://openreview.net/forum?id=cPxmLZmFa7
.. [Brunzema2026]
   P. Brunzema.
   Local Entropy Search Acquisition Function in BoTorch.
   https://github.com/brunzema/les_botorch/

Contributor: avrohr
"""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.pathwise.posterior_samplers import (
    draw_matheron_paths,
    MatheronPath,
)
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.models import ExactGP
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer


class LocalEntropySearch(AcquisitionFunction):
    r"""Local Entropy Search acquisition function.

    In the notation of [Stenger2026]_, the number of sampled paths is ``L`` and
    the number of descent / ascent steps per path is ``P``. These correspond to
    ``num_path_samples`` and ``num_descent_steps`` in this implementation.
    """

    def __init__(
        self,
        model: ExactGP,
        x_incumbent: Tensor,
        num_path_samples: int = 128,
        num_descent_steps: int = 128,
        learning_rate: float = 1e-2,
        optimizer_cls: type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        maximize: bool = True,
        bounds: Tensor | None = None,
        sequence_subsample_size: int | None = 8,
        sequence_discretization_size: int | None = 128,
        conditional_model_chunk_size: int | None = 8,
        convergence_tol: float | None = 1e-3,
        min_variance: float = 1e-12,
        virtual_observation_noise: float = 1e-6,
    ) -> None:
        r"""Initialize LocalEntropySearch.

        Args:
            model: A fitted unbatched exact single-output BoTorch GP model
                supporting ``condition_on_observations``.
            x_incumbent: Incumbent design point as a ``1 x d`` tensor or ``d`` tensor.
            num_path_samples: Number of posterior function paths for LES. This
                corresponds to ``L`` in [Stenger2026]_.
            num_descent_steps: Number of optimization steps for each sampled
                path. This corresponds to ``P`` in [Stenger2026]_.
            learning_rate: Step size of the sequence optimizer.
            optimizer_cls: Torch optimizer class used for sequence updates.
            optimizer_kwargs: Extra keyword arguments passed to ``optimizer_cls``.
            maximize: If True, follow ascent sequences. If False, follow descent
                sequences.
            bounds: Optional ``2 x d`` bounds tensor used to clamp sequence points.
            sequence_subsample_size: If provided, sample this many support points
                from the discretized sequence.
            sequence_discretization_size: If provided, first discretize each
                sequence to this many approx. equally spaced-based support points.
                Defaults to ``128``. To use the full sequence as the
                discretization source, pass ``None``.
            conditional_model_chunk_size: If provided, condition fantasy models in
                chunks of at most this many paths and aggregate chunk-wise in
                ``forward`` to reduce peak memory usage. Defaults to ``8``. To
                build a single batched conditional model across all paths, pass
                ``None``.
            convergence_tol: If provided, terminate sequence optimization early
                once all paths move less than this tolerance in one update step.
                Defaults to ``1e-3``.
            min_variance: Lower bound used to clamp predictive variances before taking
                logs in entropy calculations.
            virtual_observation_noise: Noise level used for virtual observations
                when conditioning on sequence points. This is interpreted in transformed
                outcome variance units when the model has an outcome transform.
        """
        super().__init__(model=model)

        if not isinstance(model, Model) or not isinstance(model, ExactGP):
            raise ValueError(
                "LocalEntropySearch currently supports only exact single-output "
                "BoTorch GP models."
            )
        if getattr(model, "num_outputs", 1) != 1:
            raise ValueError(
                "LocalEntropySearch currently supports only single-output models."
            )
        if len(model.batch_shape) > 0:
            raise ValueError(
                "LocalEntropySearch currently does not support batched GP models."
            )
        if num_path_samples < 1:
            raise ValueError(
                f"num_path_samples must be >= 1, but got {num_path_samples}."
            )
        if num_descent_steps < 1:
            raise ValueError(
                f"num_descent_steps must be >= 1, but got {num_descent_steps}."
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, but got {learning_rate}."
            )
        if min_variance <= 0:
            raise ValueError(f"min_variance must be positive, but got {min_variance}.")
        if sequence_subsample_size is not None and sequence_subsample_size < 1:
            raise ValueError(
                "sequence_subsample_size must be >= 1 when provided, but got "
                f"{sequence_subsample_size}."
            )
        if (
            sequence_discretization_size is not None
            and sequence_discretization_size < 1
        ):
            raise ValueError(
                "sequence_discretization_size must be >= 1 when provided, but got "
                f"{sequence_discretization_size}."
            )
        if (
            conditional_model_chunk_size is not None
            and conditional_model_chunk_size < 1
        ):
            raise ValueError(
                "conditional_model_chunk_size must be >= 1 when provided, but got "
                f"{conditional_model_chunk_size}."
            )
        if convergence_tol is not None and convergence_tol < 0:
            raise ValueError(
                "convergence_tol must be >= 0 when provided, but got "
                f"{convergence_tol}."
            )
        if virtual_observation_noise <= 0:
            raise ValueError(
                "virtual_observation_noise must be positive, but got "
                f"{virtual_observation_noise}."
            )

        train_X = model.train_inputs[0]
        x_incumbent = x_incumbent.to(device=train_X.device, dtype=train_X.dtype)
        if x_incumbent.ndim == 1:
            x_incumbent = x_incumbent.unsqueeze(0)
        if x_incumbent.ndim != 2 or x_incumbent.shape[0] != 1:
            raise ValueError(
                "x_incumbent must have shape `d` or `1 x d`, but got "
                f"{tuple(x_incumbent.shape)}."
            )
        if x_incumbent.shape[-1] != train_X.shape[-1]:
            raise ValueError(
                "x_incumbent dimension mismatch: expected d="
                f"{train_X.shape[-1]}, got {x_incumbent.shape[-1]}."
            )

        if bounds is not None:
            bounds = bounds.to(device=train_X.device, dtype=train_X.dtype)
            if bounds.shape != torch.Size([2, train_X.shape[-1]]):
                raise ValueError(
                    "bounds must have shape `2 x d`, but got "
                    f"{tuple(bounds.shape)} for d={train_X.shape[-1]}."
                )
            if torch.any(x_incumbent < bounds[0]) or torch.any(x_incumbent > bounds[1]):
                raise ValueError(
                    "x_incumbent must lie within bounds when bounds are provided."
                )

        self.num_path_samples = num_path_samples
        self.num_descent_steps = num_descent_steps
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.maximize = maximize
        self.sequence_subsample_size = sequence_subsample_size
        self.sequence_discretization_size = sequence_discretization_size
        self.conditional_model_chunk_size = conditional_model_chunk_size
        self.convergence_tol = convergence_tol
        self.min_variance = min_variance
        self.virtual_observation_noise = virtual_observation_noise

        self.register_buffer("x_incumbent", x_incumbent)
        if bounds is None:
            self.bounds = None
        else:
            self.register_buffer("bounds", bounds)

        self.sequence_X_per_path: list[Tensor] = []
        self.sequence_Y_per_path: list[Tensor] = []
        self.conditional_models = ModuleList()
        self.conditional_model_num_paths: list[int] = []

        self._build_conditioned_models()

    def _build_conditioned_models(self) -> None:
        # Exact GPs require initialized prediction caches before fantasizing.
        with torch.no_grad():
            _ = self.model.posterior(self.x_incumbent)

        X_sequence, Y_sequence = self._generate_pathwise_sequences()
        self.sequence_X_per_path = list(X_sequence.unbind(dim=0))
        self.sequence_Y_per_path = list(Y_sequence.unbind(dim=0))

        num_paths = X_sequence.shape[0]
        chunk_size = self.conditional_model_chunk_size or num_paths
        self.conditional_models = ModuleList()
        self.conditional_model_num_paths = []
        for start in range(0, num_paths, chunk_size):
            end = min(start + chunk_size, num_paths)
            Y_chunk = Y_sequence[start:end]
            noise = torch.full_like(Y_chunk, self.virtual_observation_noise)
            self.conditional_models.append(
                self.model.condition_on_observations(
                    X=X_sequence[start:end],
                    Y=Y_chunk,
                    noise=noise,
                )
            )
            self.conditional_model_num_paths.append(end - start)

    def _generate_pathwise_sequences(self) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            matheron_paths = draw_matheron_paths(
                model=self.model,
                sample_shape=torch.Size([self.num_path_samples]),
            )

        with torch.enable_grad():
            X_var = (
                self.x_incumbent.expand(self.num_path_samples, -1)
                .unsqueeze(-2)
                .clone()
                .detach()
            )
            X_var.requires_grad_(True)
            optimizer = self.optimizer_cls(
                [X_var],
                lr=self.learning_rate,
                **self.optimizer_kwargs,
            )

            X_history: list[Tensor] = [X_var.detach().clone()]
            prev_X = X_var.detach().clone()

            for _ in range(self.num_descent_steps):
                optimizer.zero_grad(set_to_none=True)
                path_values = matheron_paths(X_var).squeeze(-1).squeeze(-1)
                objective = (
                    path_values.sum() if not self.maximize else -path_values.sum()
                )
                objective.backward()

                optimizer.step()

                if self.bounds is not None:
                    with torch.no_grad():
                        X_var.clamp_(min=self.bounds[0], max=self.bounds[1])

                with torch.no_grad():
                    X_history.append(X_var.detach().clone())
                    if self.convergence_tol is not None:
                        # Shared stopping criterion: break only when every path's
                        # step is below tolerance, keeping equal sequence lengths.
                        step_sizes = torch.linalg.norm(
                            (X_var - prev_X).squeeze(-2),
                            dim=-1,
                        )
                        if torch.all(step_sizes <= self.convergence_tol):
                            break
                        prev_X.copy_(X_var)

        X_steps = torch.stack(X_history, dim=0)  # (T, L, 1, d)
        X_paths = X_steps[:, :, 0, :].transpose(0, 1).contiguous()  # (L, T, d)
        support_indices = self._get_sequence_support_indices_batch(X_paths=X_paths)
        support_X = torch.gather(
            X_paths,
            dim=1,
            index=support_indices.unsqueeze(-1).expand(-1, -1, X_paths.shape[-1]),
        )  # (L, K, d)
        support_Y = self._evaluate_support_values(
            paths=matheron_paths,
            X_support=support_X,
        )  # (L, K)

        return support_X, support_Y.unsqueeze(-1)

    def _get_sequence_support_indices_batch(
        self,
        X_paths: Tensor,
    ) -> Tensor:
        num_paths, n, _ = X_paths.shape
        discretization_size = n
        if self.sequence_discretization_size is not None:
            discretization_size = min(self.sequence_discretization_size, n)
        if discretization_size >= n:
            discretized_indices = (
                torch.arange(n, device=X_paths.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(num_paths, n)
            )
        else:
            # Keep paper-aligned support points along cumulative path distance.
            cumsum_deltas = torch.cumsum(
                torch.linalg.norm(X_paths[:, 1:] - X_paths[:, :-1], dim=-1),
                dim=1,
            )
            thresholds = (
                torch.linspace(
                    0.0,
                    1.0,
                    steps=discretization_size + 1,
                    device=X_paths.device,
                    dtype=X_paths.dtype,
                )[1:].unsqueeze(0)
                * cumsum_deltas[:, -1:].contiguous()
            )
            discretized_indices = (
                torch.searchsorted(
                    cumsum_deltas,
                    thresholds,
                    right=False,
                )
                .add(1)
                .clamp_max(n - 1)
                .to(dtype=torch.long)
            )

        if self.sequence_subsample_size is None:
            return discretized_indices
        k = min(self.sequence_subsample_size, discretized_indices.shape[1])
        if k == discretized_indices.shape[1]:
            return discretized_indices

        # Sample indices from the discretized sequence and preserve path order.
        chosen = torch.argsort(
            torch.rand(
                discretized_indices.shape[0],
                discretized_indices.shape[1],
                device=X_paths.device,
            ),
            dim=1,
        )[:, :k]
        chosen = torch.sort(chosen, dim=1).values
        return torch.gather(discretized_indices, dim=1, index=chosen)

    def _evaluate_support_values(
        self,
        paths: MatheronPath,
        X_support: Tensor,
    ) -> Tensor:
        r"""Evaluate aligned path samples at selected support points.

        Args:
            paths: Matheron posterior sample paths with ``L`` samples.
            X_support: Tensor of shape ``L x K x d`` with support points per path.

        Returns:
            Tensor of shape ``L x K`` with aligned path values.
        """
        return paths(X_support)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate LES for query points ``X``.

        Args:
            X: A ``batch_shape x 1 x d`` tensor of candidate points.

        Returns:
            A ``batch_shape`` tensor of LES values.
        """
        entropy = self._predictive_entropy(model=self.model, X=X)
        conditional_entropy_sum = torch.zeros_like(entropy)
        total_paths = 0
        for model_chunk, num_paths in zip(
            self.conditional_models, self.conditional_model_num_paths, strict=True
        ):
            X_batched = X.unsqueeze(-3).expand(
                *X.shape[:-2],
                num_paths,
                *X.shape[-2:],
            )
            conditional_entropies = self._predictive_entropy(
                model=model_chunk,
                X=X_batched,
            )
            conditional_entropy_sum = conditional_entropy_sum + (
                conditional_entropies.mean(dim=-1) * num_paths
            )
            total_paths += num_paths
        conditional_entropy_mean = conditional_entropy_sum / total_paths
        return entropy - conditional_entropy_mean

    def _predictive_entropy(self, model: Model, X: Tensor) -> Tensor:
        posterior = model.posterior(X=X, observation_noise=True)
        variance = (
            posterior.variance.squeeze(-1).squeeze(-1).clamp_min(self.min_variance)
        )
        factor = variance.new_tensor(2.0 * torch.pi * torch.e)
        return 0.5 * torch.log(factor * variance)
