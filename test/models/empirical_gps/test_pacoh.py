#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for PACOH-GP model, SVGD utilities, and pre-training.

Design: Minimize public test methods by using private helpers and loops.
Each public test method covers multiple configurations via subTest.
"""

from __future__ import annotations

import os
import tempfile

import torch
from botorch.models.empirical_gps.pacoh import (
    PACOHGPConfig,
    PACOHGPModel,
    PACOHPriorContainer,
    pretrain_pacoh_gp,
)
from botorch.models.empirical_gps.svgd import svgd_kernel, svgd_update
from botorch.models.empirical_gps.utils import BatchedLinear, ExperimentDataset
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.utils.testing import BotorchTestCase


class TestBatchedLinear(BotorchTestCase):
    """Tests for the BatchedLinear utility."""

    def test_unbatched_matches_linear(self) -> None:
        """BatchedLinear with empty batch_shape behaves like nn.Linear."""
        in_f, out_f = 4, 3
        bl = BatchedLinear(in_f, out_f)
        self.assertEqual(bl.weight.shape, (out_f, in_f))
        self.assertEqual(bl.bias.shape, (out_f,))

        x = torch.randn(5, in_f)
        out = bl(x)
        self.assertEqual(out.shape, (5, out_f))
        self.assertTrue(torch.isfinite(out).all())

    def test_batched_forward(self) -> None:
        """BatchedLinear with batch_shape broadcasts correctly."""
        K, in_f, out_f = 3, 4, 2
        bl = BatchedLinear(in_f, out_f, batch_shape=torch.Size([K]))
        self.assertEqual(bl.weight.shape, (K, out_f, in_f))
        self.assertEqual(bl.bias.shape, (K, out_f))

        x = torch.randn(K, 5, in_f)
        out = bl(x)
        self.assertEqual(out.shape, (K, 5, out_f))
        self.assertTrue(torch.isfinite(out).all())

    def test_no_bias(self) -> None:
        """BatchedLinear without bias."""
        bl = BatchedLinear(4, 3, bias=False)
        self.assertIsNone(bl.bias)
        x = torch.randn(5, 4)
        out = bl(x)
        self.assertEqual(out.shape, (5, 3))


class TestSVGD(BotorchTestCase):
    """Tests for SVGD kernel and update functions."""

    def test_svgd_kernel_shapes(self) -> None:
        """Verify kernel matrix and gradient shapes."""
        K, D = 5, 10
        particles = torch.randn(K, D)
        kernel_matrix, kernel_grads = svgd_kernel(particles)

        self.assertEqual(kernel_matrix.shape, (K, K))
        self.assertEqual(kernel_grads.shape, (K, K, D))
        # Kernel should be symmetric and positive
        self.assertTrue(torch.allclose(kernel_matrix, kernel_matrix.T, atol=1e-6))
        self.assertTrue((kernel_matrix >= 0).all())
        # Diagonal should be 1 (K(x, x) = 1 for RBF)
        self.assertTrue(torch.allclose(kernel_matrix.diag(), torch.ones(K), atol=1e-6))

    def test_svgd_kernel_with_length_scale(self) -> None:
        """Verify kernel with explicit length scale."""
        K, D = 4, 3
        particles = torch.randn(K, D)
        km1, _ = svgd_kernel(particles, length_scale=1.0)
        km2, _ = svgd_kernel(particles, length_scale=0.1)
        # Smaller length scale → more peaked kernel → smaller off-diagonal
        off_diag_mask = ~torch.eye(K, dtype=torch.bool)
        self.assertGreater(
            km1[off_diag_mask].mean().item(),
            km2[off_diag_mask].mean().item(),
        )

    def test_svgd_update_moves_particles(self) -> None:
        """Verify SVGD update produces different particles."""
        K, D = 5, 10
        particles = torch.randn(K, D)
        score_grads = torch.randn(K, D)
        updated = svgd_update(particles, score_grads, step_size=0.1)

        self.assertEqual(updated.shape, (K, D))
        self.assertFalse(torch.allclose(updated, particles))
        self.assertTrue(torch.isfinite(updated).all())


class TestPACOHGP(BotorchTestCase):
    """Tests for PACOH-GP pre-training and prediction."""

    def _make_sinusoidal_datasets(
        self,
        n_tasks: int = 5,
        n_points: int = 20,
        input_dim: int = 1,
        dtype: torch.dtype = torch.double,
    ) -> list[ExperimentDataset]:
        """Generate synthetic sinusoidal datasets for testing."""
        datasets = []
        for i in range(n_tasks):
            X = torch.linspace(0, 1, n_points, dtype=dtype).unsqueeze(-1)
            if input_dim > 1:
                X = X.expand(-1, input_dim)
            amp = 0.7 + 0.6 * (i / max(n_tasks - 1, 1))
            phase = i * 0.5
            Y = (amp * torch.sin(2 * 3.14159 * X[..., 0] + phase)).unsqueeze(-1)
            Y = Y + 0.05 * torch.randn_like(Y)
            datasets.append(ExperimentDataset(X=X, Y=Y))
        return datasets

    def test_pretrain_returns_valid_container(self) -> None:
        """Pre-training returns a PACOHPriorContainer with correct structure."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=3, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=5,
            learning_rate=1e-3,
        )

        self.assertIsInstance(prior, PACOHPriorContainer)
        self.assertEqual(prior.num_particles, 3)
        self.assertEqual(len(prior.particle_states), 3)
        self.assertEqual(prior.input_dim, 1)
        self.assertEqual(prior.num_datasets, 3)
        self.assertIsNotNone(prior.final_loss)

        # Each particle state should have the same keys
        keys = set(prior.particle_states[0].keys())
        for ps in prior.particle_states[1:]:
            self.assertEqual(set(ps.keys()), keys)

    def test_pretrain_with_task_minibatching(self) -> None:
        """Pre-training works with task mini-batching."""
        datasets = self._make_sinusoidal_datasets(n_tasks=5, n_points=10)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=3,
            task_batch_size=2,
        )

        self.assertEqual(prior.num_particles, 2)
        self.assertEqual(prior.num_datasets, 5)

    def test_from_pretrained_creates_valid_model(self) -> None:
        """PACOHGPModel.from_pretrained creates a working model."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=3, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=5,
        )

        # Create model for a new task
        train_X = torch.linspace(0, 0.5, 5, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.sin(train_X.squeeze(-1))

        model = PACOHGPModel.from_pretrained(prior, train_X, train_Y)

        # Check structure
        self.assertEqual(model.num_particles, 3)
        self.assertTrue(model._is_ensemble)

        # All params should be frozen
        for p in model.parameters():
            self.assertFalse(p.requires_grad)

    def test_posterior_returns_gaussian_mixture(self) -> None:
        """Posterior returns GaussianMixturePosterior with correct shapes."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=3, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=5,
        )

        train_X = torch.linspace(0, 0.5, 5, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.sin(train_X.squeeze(-1))
        model = PACOHGPModel.from_pretrained(prior, train_X, train_Y)

        model.eval()
        model.likelihood.eval()

        test_X = torch.linspace(0.5, 1.0, 4, dtype=torch.double).unsqueeze(-1)
        with torch.no_grad():
            posterior = model.posterior(test_X)

        self.assertIsInstance(posterior, GaussianMixturePosterior)

        # mixture_mean and mixture_variance should be finite with correct size
        mix_mean = posterior.mixture_mean
        mix_var = posterior.mixture_variance
        # The mean has shape (..., q, 1) or (..., q) depending on output dim
        self.assertEqual(mix_mean.numel(), 4)
        self.assertEqual(mix_var.numel(), 4)
        self.assertTrue(torch.isfinite(mix_mean).all())
        self.assertTrue((mix_var > 0).all())

    def test_posterior_with_observation_noise(self) -> None:
        """Posterior with observation noise produces larger variance."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=5,
        )

        train_X = torch.linspace(0, 0.5, 5, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.sin(train_X.squeeze(-1))
        model = PACOHGPModel.from_pretrained(prior, train_X, train_Y)
        model.eval()
        model.likelihood.eval()

        test_X = torch.linspace(0.5, 1.0, 4, dtype=torch.double).unsqueeze(-1)
        with torch.no_grad():
            post_no_noise = model.posterior(test_X, observation_noise=False)
            post_noise = model.posterior(test_X, observation_noise=True)

        self.assertTrue(
            (post_noise.mixture_variance >= post_no_noise.mixture_variance - 1e-6).all()
        )

    def test_container_save_load_roundtrip(self) -> None:
        """PACOHPriorContainer save/load roundtrip preserves state."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))

        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=3,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            prior.save(path)
            loaded = PACOHPriorContainer.load(path)

            self.assertEqual(loaded.num_particles, prior.num_particles)
            self.assertEqual(loaded.input_dim, prior.input_dim)
            self.assertEqual(loaded.num_datasets, prior.num_datasets)

            # Verify particle states match
            for ps_orig, ps_loaded in zip(
                prior.particle_states, loaded.particle_states
            ):
                for key in ps_orig:
                    self.assertTrue(
                        torch.allclose(ps_orig[key], ps_loaded[key]),
                        f"Mismatch for key {key}",
                    )
        finally:
            os.unlink(path)

    def test_empty_datasets_raises(self) -> None:
        """Pre-training with empty datasets raises ValueError."""
        with self.assertRaises(ValueError):
            pretrain_pacoh_gp(datasets=[], input_dim=1)

    def test_input_dim_mismatch_raises(self) -> None:
        """from_pretrained with wrong input_dim raises ValueError."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))
        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=3,
        )

        wrong_X = torch.randn(5, 3, dtype=torch.double)  # input_dim=3, expected 1
        wrong_Y = torch.randn(5, dtype=torch.double)
        with self.assertRaises(ValueError):
            PACOHGPModel.from_pretrained(prior, wrong_X, wrong_Y)

    def test_load_particles_weights_match(self) -> None:
        """Verify load_particles actually loads the correct weights, not random."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))
        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=1,
            config=config,
            num_iterations=3,
        )

        train_X = torch.linspace(0, 0.5, 5, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.sin(train_X.squeeze(-1))
        model = PACOHGPModel.from_pretrained(prior, train_X, train_Y)

        # Check that each particle's weights are actually in the batched model
        model_state = model.state_dict()
        for key in prior.particle_states[0]:
            if key in model_state:
                batched_param = model_state[key]
                for k in range(prior.num_particles):
                    expected = prior.particle_states[k][key]
                    actual = batched_param[k]
                    self.assertTrue(
                        torch.allclose(expected, actual, atol=1e-6),
                        f"Particle {k}, key {key}: loaded weight doesn't match",
                    )

    def test_hyper_prior_regularization_effect(self) -> None:
        """Stronger hyper-prior (smaller σ_P) keeps particles closer to origin."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)

        # Weak regularization (large σ_P)
        config_weak = PACOHGPConfig(
            num_particles=2, hidden_dims=(8,), hyper_prior_std=10.0
        )
        prior_weak = pretrain_pacoh_gp(
            datasets=datasets, input_dim=1, config=config_weak, num_iterations=20
        )

        # Strong regularization (moderate σ_P — not too small to avoid NaN)
        config_strong = PACOHGPConfig(
            num_particles=2, hidden_dims=(8,), hyper_prior_std=0.1
        )
        prior_strong = pretrain_pacoh_gp(
            datasets=datasets, input_dim=1, config=config_strong, num_iterations=20
        )

        # Compute average parameter norm for each
        def avg_param_norm(prior: PACOHPriorContainer) -> float:
            total = 0.0
            for ps in prior.particle_states:
                total += sum(v.norm().item() for v in ps.values())
            return total / prior.num_particles

        norm_weak = avg_param_norm(prior_weak)
        norm_strong = avg_param_norm(prior_strong)

        # Strong regularization should yield smaller parameter norms
        self.assertGreater(norm_weak, norm_strong)

    def test_svgd_convergence_toward_mode(self) -> None:
        """SVGD with known score function moves particles toward the mode."""
        D = 2
        K = 10
        # Target: N(3, I) → score = -(x - 3), mode at x=3
        target_mean = torch.full((D,), 3.0)
        particles = torch.randn(K, D)
        initial_dist = (particles - target_mean).norm(dim=-1).mean()

        for _ in range(50):
            score_grads = -(particles - target_mean)
            particles = svgd_update(particles, score_grads, step_size=0.1)

        final_dist = (particles - target_mean).norm(dim=-1).mean()
        # Particles should be closer to the mode after optimization
        self.assertLess(final_dist.item(), initial_dist.item() * 0.5)

    def test_pacoh_score_formula_components(self) -> None:
        """Verify PACOH score formula: hyper-prior penalizes large params."""
        from botorch.models.empirical_gps.pacoh import (
            _compute_batched_pacoh_score,
            PACOHGPModel,
        )

        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10)
        dummy_X = torch.zeros(1, 1, dtype=torch.double)
        dummy_Y = torch.zeros(1, dtype=torch.double)
        model = PACOHGPModel(dummy_X, dummy_Y, num_particles=1, hidden_dims=(8,))
        model.train()

        # Score with tight hyper-prior should be lower than with loose one
        for p in model.parameters():
            p.requires_grad_(True)
        score_tight = _compute_batched_pacoh_score(
            model,
            datasets,
            hyper_prior_std=0.01,
            lambda_coeff=3.0,
            beta_coeff=10.0,
            n_tasks=3,
        )
        score_loose = _compute_batched_pacoh_score(
            model,
            datasets,
            hyper_prior_std=100.0,
            lambda_coeff=3.0,
            beta_coeff=10.0,
            n_tasks=3,
        )
        # Tight prior penalizes more → lower score
        self.assertLess(score_tight.item(), score_loose.item())

    def test_multi_dim_input(self) -> None:
        """PACOH works with multi-dimensional inputs."""
        datasets = self._make_sinusoidal_datasets(n_tasks=3, n_points=10, input_dim=3)
        config = PACOHGPConfig(num_particles=2, hidden_dims=(8,))
        prior = pretrain_pacoh_gp(
            datasets=datasets,
            input_dim=3,
            config=config,
            num_iterations=3,
        )

        train_X = torch.randn(5, 3, dtype=torch.double)
        train_Y = torch.randn(5, dtype=torch.double)
        model = PACOHGPModel.from_pretrained(prior, train_X, train_Y)
        model.eval()
        model.likelihood.eval()

        test_X = torch.randn(4, 3, dtype=torch.double)
        with torch.no_grad():
            posterior = model.posterior(test_X)
        self.assertIsInstance(posterior, GaussianMixturePosterior)
        self.assertTrue(torch.isfinite(posterior.mixture_mean).all())
