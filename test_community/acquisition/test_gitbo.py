#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.gitbo import (
    compute_active_subspace,
    gitbo_step,
    quantile_ucb,
    sample_subspace_candidates,
)
from botorch_community.models.prior_fitted_network import PFNModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import nn, Tensor


class GradDummyPFN(nn.Module):
    def __init__(self, n_buckets: int = 100):
        """A dummy PFN whose logits depend differentiably on the test inputs.

        Args:
            n_buckets: Number of buckets for the output distribution.
        """
        super().__init__()
        self.n_buckets = n_buckets
        self.criterion = MagicMock()
        self.criterion.borders = torch.linspace(0, 1, n_buckets + 1)
        self.style_encoder = None
        self.y_style_encoder = None

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        test_x: Tensor,
        style: Tensor | None = None,
        y_style: Tensor | None = None,
    ) -> Tensor:
        bucket_scores = torch.linspace(
            -1.0, 1.0, self.n_buckets, dtype=test_x.dtype, device=test_x.device
        )
        return torch.sin(test_x * 3.0).sum(dim=-1, keepdim=True) * bucket_scores


def _get_gp(
    n: int = 6, d: int = 3, fit: bool = False, **tkwargs
) -> tuple[SingleTaskGP, Tensor, Tensor]:
    train_X = torch.rand(n, d, **tkwargs)
    train_Y = (train_X[:, :2] ** 2).sum(dim=-1, keepdim=True)
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    if fit:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    else:
        model.eval()
    return model, train_X, train_Y


def _get_pfn(n: int = 6, d: int = 3, **tkwargs) -> tuple[PFNModel, Tensor, Tensor]:
    train_X = torch.rand(n, d, **tkwargs)
    train_Y = torch.rand(n, 1, **tkwargs)
    model = PFNModel(train_X, train_Y, model=GradDummyPFN())
    return model, train_X, train_Y


class TestQuantileUCB(BotorchTestCase):
    def test_shapes_and_batch_limit(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, _, _ = _get_gp(**tkwargs)
                X = torch.rand(17, 3, **tkwargs)
                scores, grads = quantile_ucb(model, X)
                self.assertEqual(scores.shape, torch.Size([17]))
                self.assertEqual(grads.shape, torch.Size([17, 3]))
                self.assertTrue(torch.isfinite(scores).all())
                self.assertTrue(torch.isfinite(grads).all())
                # chunked evaluation is exact (quantile-UCB is deterministic)
                scores_c, grads_c = quantile_ucb(model, X, batch_limit=7)
                self.assertAllClose(scores, scores_c)
                self.assertAllClose(grads, grads_c)
                # no-gradient mode
                scores_ng, grads_ng = quantile_ucb(
                    model, X, compute_mean_gradients=False
                )
                self.assertIsNone(grads_ng)
                self.assertAllClose(scores, scores_ng)

    def test_raises(self):
        model, _, _ = _get_gp(device=self.device)
        X = torch.rand(5, 3, device=self.device)
        with self.assertRaisesRegex(ValueError, "X must be"):
            quantile_ucb(model, X.unsqueeze(0))
        with self.assertRaisesRegex(ValueError, "quantile must be"):
            quantile_ucb(model, X, quantile=1.2)
        with self.assertRaisesRegex(ValueError, "batch_limit must be"):
            quantile_ucb(model, X, batch_limit=0)

    def test_gradients_match_finite_differences_gp(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        model, _, _ = _get_gp(**tkwargs)
        X = torch.rand(4, 3, **tkwargs)
        _, grads = quantile_ucb(model, X)

        def posterior_mean(X_eval: Tensor) -> Tensor:
            with torch.no_grad():
                return model.posterior(X_eval.unsqueeze(-2)).mean.reshape(-1)

        eps = 1e-5
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_plus, X_minus = X.clone(), X.clone()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps
                fd = (posterior_mean(X_plus)[i] - posterior_mean(X_minus)[i]) / (
                    2 * eps
                )
                self.assertAllClose(grads[i, j], fd, atol=1e-4)

    def test_score_exceeds_mean(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        model, _, _ = _get_gp(**tkwargs)
        X = torch.rand(16, 3, **tkwargs)
        scores, _ = quantile_ucb(model, X, quantile=0.975)
        with torch.no_grad():
            mean = model.posterior(X.unsqueeze(-2)).mean.reshape(-1)
        self.assertTrue((scores > mean).all())

    def test_pfn_gradients_and_batching(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        model, _, _ = _get_pfn(**tkwargs)
        X = torch.rand(8, 3, **tkwargs)
        scores, grads = quantile_ucb(model, X, eval_in_q_batch=True)
        self.assertEqual(scores.shape, torch.Size([8]))
        self.assertEqual(grads.shape, torch.Size([8, 3]))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((grads.abs().sum(dim=-1) > 0).all())
        # the q-batch and the (N, 1, d) layouts agree on gradients and scores
        scores_b, grads_b = quantile_ucb(model, X, eval_in_q_batch=False)
        self.assertAllClose(scores, scores_b, atol=1e-5)
        self.assertAllClose(grads, grads_b, atol=1e-5)
        # gradients match finite differences of the Riemann posterior mean
        eps = 1e-3
        for i in range(3):
            for j in range(X.shape[1]):
                X_plus, X_minus = X.clone(), X.clone()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps
                with torch.no_grad():
                    mean_plus = model.posterior(X_plus).mean.reshape(-1)
                    mean_minus = model.posterior(X_minus).mean.reshape(-1)
                fd = (mean_plus[i] - mean_minus[i]) / (2 * eps)
                self.assertAllClose(grads[i, j], fd, atol=1e-3)


class TestActiveSubspace(BotorchTestCase):
    def test_known_span_recovery(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        d, r = 5, 2
        basis = torch.linalg.qr(torch.randn(d, r, **tkwargs)).Q
        coeffs = torch.tensor(
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]], **tkwargs
        )
        gradients = coeffs @ basis.transpose(-2, -1)
        subspace, eigenvalues = compute_active_subspace(gradients, rank=r)
        self.assertEqual(subspace.shape, torch.Size([d, r]))
        self.assertEqual(eigenvalues.shape, torch.Size([d]))
        self.assertAllClose(
            subspace @ subspace.transpose(-2, -1),
            basis @ basis.transpose(-2, -1),
            atol=1e-8,
        )
        # eigenvalues are descending: 2.0, 0.5, then zeros
        self.assertAllClose(
            eigenvalues[:2], torch.tensor([2.0, 0.5], **tkwargs), atol=1e-8
        )
        self.assertTrue((eigenvalues.diff() <= 1e-12).all())

    def test_rank_modes(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        d = 5
        basis = torch.linalg.qr(torch.randn(d, 2, **tkwargs)).Q
        coeffs = torch.tensor(
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]], **tkwargs
        )
        gradients = coeffs @ basis.transpose(-2, -1)
        # integer rank larger than d is clamped
        subspace, _ = compute_active_subspace(gradients, rank=10)
        self.assertEqual(subspace.shape, torch.Size([d, d]))
        # percent-variance mode: eigenvalue ratios are 0.8 and 1.0
        subspace, _ = compute_active_subspace(gradients, rank=0.3)
        self.assertEqual(subspace.shape, torch.Size([d, 1]))
        subspace, _ = compute_active_subspace(gradients, rank=0.99)
        self.assertEqual(subspace.shape, torch.Size([d, 2]))
        for bad_rank in (0, -1, -0.5):
            with self.assertRaisesRegex(ValueError, "rank must be positive"):
                compute_active_subspace(gradients, rank=bad_rank)
        # all-zero gradients in percent mode fall back to rank 1
        subspace, _ = compute_active_subspace(torch.zeros(4, d, **tkwargs), rank=0.5)
        self.assertEqual(subspace.shape, torch.Size([d, 1]))
        with self.assertRaisesRegex(ValueError, "gradients must be"):
            compute_active_subspace(gradients.unsqueeze(0), rank=2)

    def test_sample_subspace_candidates(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"device": self.device, "dtype": dtype}
                d, r = 6, 2
                subspace = torch.linalg.qr(torch.randn(d, r, **tkwargs)).Q
                origin = torch.full((d,), 0.5, **tkwargs)
                bounds = torch.stack(
                    [torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)]
                )
                X = sample_subspace_candidates(
                    subspace, origin, bounds, num_candidates=64, scale=0.1
                )
                self.assertEqual(X.shape, torch.Size([64, d]))
                self.assertTrue((X >= bounds[0]).all() and (X <= bounds[1]).all())
                # with an interior origin and small scale, samples stay in the
                # affine subspace: residual after projection is zero
                diff = X - origin
                proj = diff @ subspace @ subspace.transpose(-2, -1)
                self.assertAllClose(diff, proj, atol=1e-5)
                # scale=0 collapses onto the origin
                X0 = sample_subspace_candidates(
                    subspace, origin, bounds, num_candidates=8, scale=0.0
                )
                self.assertAllClose(X0, origin.expand(8, d))


class TestGITBOStep(BotorchTestCase):
    def test_first_iteration_sobol(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        model, train_X, _ = _get_gp(**tkwargs)
        bounds = torch.stack([torch.zeros(3, **tkwargs), torch.ones(3, **tkwargs)])
        result = gitbo_step(
            model, train_X, gradients=None, bounds=bounds, num_candidates=32
        )
        self.assertIsNone(result.subspace)
        self.assertIsNone(result.eigenvalues)
        self.assertEqual(result.candidate_set.shape, torch.Size([32, 3]))
        self.assertEqual(result.candidate.shape, torch.Size([1, 3]))
        self.assertEqual(result.gradients.shape, torch.Size([32, 3]))
        self.assertTrue(
            torch.equal(
                result.candidate,
                result.candidate_set[result.acq_values.argmax()].unsqueeze(0),
            )
        )
        with self.assertRaisesRegex(ValueError, "bounds must be"):
            gitbo_step(model, train_X, gradients=None, bounds=bounds[0])

    def test_degenerate_gradients_fall_back_to_sobol(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        model, train_X, _ = _get_gp(**tkwargs)
        bounds = torch.stack([torch.zeros(3, **tkwargs), torch.ones(3, **tkwargs)])
        result = gitbo_step(
            model,
            train_X,
            gradients=torch.zeros(32, 3, **tkwargs),
            bounds=bounds,
            num_candidates=32,
        )
        self.assertIsNone(result.subspace)
        self.assertIsNone(result.eigenvalues)

    def test_two_iteration_loop_gp(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        d = 4
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        train_X = torch.rand(5, d, **tkwargs)
        train_Y = -((train_X - 0.5) ** 2).sum(dim=-1, keepdim=True)
        gradients = None
        for iteration in range(2):
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            result = gitbo_step(
                model,
                train_X,
                gradients,
                bounds,
                num_candidates=64,
                rank=2,
                scale=0.2,
            )
            new_Y = -((result.candidate - 0.5) ** 2).sum(dim=-1, keepdim=True)
            train_X = torch.cat([train_X, result.candidate])
            train_Y = torch.cat([train_Y, new_Y])
            gradients = result.gradients
            if iteration == 0:
                self.assertIsNone(result.subspace)
            else:
                self.assertEqual(result.subspace.shape, torch.Size([d, 2]))
                self.assertEqual(result.eigenvalues.shape, torch.Size([d]))
                self.assertTrue((result.candidate_set >= bounds[0]).all())
                self.assertTrue((result.candidate_set <= bounds[1]).all())
            self.assertTrue(torch.isfinite(result.acq_values).all())
        self.assertEqual(train_X.shape, torch.Size([7, d]))

    def test_two_iteration_loop_pfn(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        d = 3
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        train_X = torch.rand(5, d, **tkwargs)
        train_Y = torch.rand(5, 1, **tkwargs)
        pfn_module = GradDummyPFN()  # loaded once, reused across iterations
        gradients = None
        for iteration in range(2):
            model = PFNModel(train_X, train_Y, model=pfn_module)
            result = gitbo_step(
                model,
                train_X,
                gradients,
                bounds,
                num_candidates=32,
                rank=2,
                eval_in_q_batch=True,
                batch_limit=16,
            )
            train_X = torch.cat([train_X, result.candidate])
            train_Y = torch.cat([train_Y, torch.rand(1, 1, **tkwargs)])
            gradients = result.gradients
            if iteration == 1:
                self.assertEqual(result.subspace.shape, torch.Size([d, 2]))
            self.assertTrue(torch.isfinite(result.acq_values).all())
            self.assertTrue(torch.isfinite(result.gradients).all())
        self.assertEqual(train_X.shape, torch.Size([7, d]))
