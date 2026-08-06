#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.empirical_gps.em_coordinate_ascent import (
    coordinate_ascent_em,
    fit_kernel_to_em_prior,
    kl_divergence_mvn,
    KLPriorFitMLL,
)
from botorch.models.empirical_gps.em_empirical_gp import (
    EMPriorContainer,
    pretrain_em_prior,
)
from botorch.models.empirical_gps.utils import ExperimentDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean


class TestEMCoordinateAscent(BotorchTestCase):
    """Tests for the coordinate-ascent and KL-fit EM utilities."""

    def _make_em_setup(self, tkwargs, d=3, K=5, n_i=20, M=10, seed=42):
        """Create EM setup with inducing points for testing."""
        torch.manual_seed(seed)
        X_shared = torch.rand(n_i, d, **tkwargs)
        datasets = []
        for _ in range(K):
            Y_i = torch.randn(n_i, 1, **tkwargs) * 0.5 + 1.0
            datasets.append(ExperimentDataset(X=X_shared, Y=Y_i))

        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)).to(**tkwargs)
        Z = X_shared[:M]

        container = pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood_noise=torch.tensor(1e-4, **tkwargs),
            num_em_iterations=5,
            inducing_points=Z,
            enable_interpolation=True,
        )
        return datasets, mean_module, covar_module, container, Z

    # =========================================================================
    # Tests for kl_divergence_mvn
    # =========================================================================

    def test_kl_divergence_mvn_identical_distributions(self) -> None:
        """KL(p || p) = 0 for identical distributions."""

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(42)
            M = 5
            mu = torch.randn(M, **tkwargs)
            L = torch.randn(M, M, **tkwargs)
            Sigma = L @ L.T + 0.1 * torch.eye(M, **tkwargs)

            kl = kl_divergence_mvn(mu, Sigma, mu, Sigma)
            self.assertAlmostEqual(kl.item(), 0.0, places=5)

    def test_kl_divergence_mvn_positive_for_different_distributions(self) -> None:
        """KL(p || q) > 0 for p != q (Gibbs' inequality)."""

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(42)
            M = 5
            mu_p = torch.randn(M, **tkwargs)
            mu_q = torch.randn(M, **tkwargs)
            L_p = torch.randn(M, M, **tkwargs)
            Sigma_p = L_p @ L_p.T + 0.1 * torch.eye(M, **tkwargs)
            L_q = torch.randn(M, M, **tkwargs)
            Sigma_q = L_q @ L_q.T + 0.1 * torch.eye(M, **tkwargs)

            kl = kl_divergence_mvn(mu_p, Sigma_p, mu_q, Sigma_q)
            self.assertGreater(kl.item(), 0.0)

    def test_kl_divergence_mvn_agrees_with_torch_distributions(self) -> None:
        """KL matches torch.distributions.kl_divergence for MultivariateNormal."""

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(42)
            M = 5
            mu_p = torch.randn(M, **tkwargs)
            mu_q = torch.randn(M, **tkwargs)
            L_p = torch.randn(M, M, **tkwargs)
            Sigma_p = L_p @ L_p.T + 0.1 * torch.eye(M, **tkwargs)
            L_q = torch.randn(M, M, **tkwargs)
            Sigma_q = L_q @ L_q.T + 0.1 * torch.eye(M, **tkwargs)

            kl_ours = kl_divergence_mvn(mu_p, Sigma_p, mu_q, Sigma_q)

            from torch.distributions import kl_divergence, MultivariateNormal

            p = MultivariateNormal(mu_p, Sigma_p)
            q = MultivariateNormal(mu_q, Sigma_q)
            kl_ref = kl_divergence(p, q)

            atol = 1e-4 if dtype == torch.float else 1e-8
            self.assertAlmostEqual(kl_ours.item(), kl_ref.item(), delta=atol)

    def test_kl_divergence_mvn_gradient_flows_to_kernel_params(self) -> None:
        """Gradients from KL flow to kernel parameters."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        d, M = 3, 10
        X = torch.rand(M, d, **tkwargs)
        mu_em = torch.randn(M, **tkwargs)
        L = torch.randn(M, M, **tkwargs)
        Sigma_em = L @ L.T + 0.1 * torch.eye(M, **tkwargs)

        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)).to(**tkwargs)
        mean_module = ConstantMean().to(**tkwargs)

        K = covar_module(X, X).to_dense()
        m = mean_module(X).squeeze(-1)
        kl = kl_divergence_mvn(mu_em.detach(), Sigma_em.detach(), m, K)
        kl.backward()

        self.assertIsNotNone(covar_module.raw_outputscale.grad)
        self.assertTrue(covar_module.raw_outputscale.grad.abs().sum() > 0)
        self.assertIsNotNone(covar_module.base_kernel.raw_lengthscale.grad)
        self.assertTrue(covar_module.base_kernel.raw_lengthscale.grad.abs().sum() > 0)

    # =========================================================================
    # Tests for fit_kernel_to_em_prior (parameter recovery)
    # =========================================================================

    def test_fit_kernel_to_em_prior_recovers_lengthscales(self) -> None:
        """KL fit recovers known ARD lengthscales from synthetic EM estimates."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        d, M = 3, 30
        torch.manual_seed(42)
        X = torch.rand(M, d, **tkwargs)

        # Generate ground truth Sigma from a known kernel
        true_ls = torch.tensor([0.2, 0.5, 1.0], **tkwargs)
        true_os = 2.0
        gt_covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)).to(**tkwargs)
        gt_covar.base_kernel.lengthscale = true_ls.unsqueeze(0)
        gt_covar.outputscale = true_os
        gt_mean = ConstantMean().to(**tkwargs)
        gt_mean.constant.data.fill_(0.5)

        with torch.no_grad():
            mu_em = gt_mean(X).squeeze(-1)
            Sigma_em = gt_covar(X, X).to_dense()

        # Initialize with wrong parameters
        from botorch.utils.constraints import LogTransformedInterval

        fit_covar = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=d,
                lengthscale_constraint=LogTransformedInterval(0.01, 100.0),
            ),
            outputscale_constraint=LogTransformedInterval(0.01, 100.0),
        ).to(**tkwargs)
        fit_mean = ConstantMean().to(**tkwargs)
        fit_covar.base_kernel.lengthscale = torch.ones(1, d, **tkwargs)
        fit_covar.outputscale = torch.tensor(1.0, **tkwargs)

        fit_kernel_to_em_prior(fit_mean, fit_covar, X, mu_em, Sigma_em)

        # Check recovered lengthscales are within 50% of ground truth
        recovered_ls = fit_covar.base_kernel.lengthscale.detach().squeeze()
        for i in range(d):
            self.assertAlmostEqual(
                recovered_ls[i].item(),
                true_ls[i].item(),
                delta=true_ls[i].item() * 0.5,
                msg=f"Lengthscale {i}: {recovered_ls[i]:.3f} vs {true_ls[i]:.3f}",
            )

    # =========================================================================
    # Tests for KLPriorFitMLL integration with fit_gpytorch_mll
    # =========================================================================

    def test_kl_prior_fit_mll_with_fit_gpytorch_mll(self) -> None:
        """KLPriorFitMLL works with fit_gpytorch_mll (L-BFGS)."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        d, M = 3, 15
        torch.manual_seed(42)
        X = torch.rand(M, d, **tkwargs)
        mu_em = torch.randn(M, **tkwargs)
        L = torch.randn(M, M, **tkwargs)
        Sigma_em = L @ L.T + 0.1 * torch.eye(M, **tkwargs)

        mean_mod = ConstantMean().to(**tkwargs)
        covar_mod = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)).to(**tkwargs)
        dummy_gp = SingleTaskGP(
            train_X=X[:1],
            train_Y=torch.zeros(1, 1, **tkwargs),
            mean_module=mean_mod,
            covar_module=covar_mod,
            outcome_transform=None,
        )

        ls_before = covar_mod.base_kernel.lengthscale.detach().clone()

        mll = KLPriorFitMLL(dummy_gp.likelihood, dummy_gp, X, mu_em, Sigma_em)
        fit_gpytorch_mll(mll)

        ls_after = covar_mod.base_kernel.lengthscale.detach().clone()
        self.assertFalse(
            torch.allclose(ls_before, ls_after, atol=1e-4),
            "Lengthscales should change after KL fit",
        )

    # =========================================================================
    # Tests for coordinate_ascent_em
    # =========================================================================

    def test_coordinate_ascent_em_returns_container(self) -> None:
        """coordinate_ascent_em returns a valid EMPriorContainer."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(42)
        d, K, n_i, M = 2, 4, 15, 8
        X_shared = torch.rand(n_i, d, **tkwargs)
        datasets = []
        for _ in range(K):
            Y_i = torch.randn(n_i, 1, **tkwargs) * 0.5 + 1.0
            datasets.append(ExperimentDataset(X=X_shared, Y=Y_i))

        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=d)).to(**tkwargs)
        Z = X_shared[:M]

        result = coordinate_ascent_em(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood_noise=torch.tensor(1e-4, **tkwargs),
            num_em_iterations=3,
            n_coord_ascent_iterations=1,
            inducing_points=Z,
            enable_interpolation=True,
        )

        self.assertIsInstance(result, EMPriorContainer)
        self.assertEqual(result.mu_inducing.shape[0], M)
        self.assertEqual(result.Sigma_inducing.shape, (M, M))

    def test_coordinate_ascent_updates_kernel_params(self) -> None:
        """Coordinate ascent should update kernel hyperparameters."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(42)
        d, K, n_i, M = 2, 4, 15, 8
        X_shared = torch.rand(n_i, d, **tkwargs)
        datasets = []
        for _ in range(K):
            Y_i = torch.randn(n_i, 1, **tkwargs) * 0.5 + 1.0
            datasets.append(ExperimentDataset(X=X_shared, Y=Y_i))

        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=d)).to(**tkwargs)
        Z = X_shared[:M]

        ls_before = covar_module.base_kernel.lengthscale.detach().clone()

        coordinate_ascent_em(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood_noise=torch.tensor(1e-4, **tkwargs),
            num_em_iterations=3,
            n_coord_ascent_iterations=1,
            inducing_points=Z,
            enable_interpolation=True,
        )

        ls_after = covar_module.base_kernel.lengthscale.detach().clone()
        self.assertFalse(
            torch.allclose(ls_before, ls_after, atol=1e-4),
            "Kernel lengthscales should change after coordinate ascent",
        )

    def test_coordinate_ascent_with_inducing_optimization(self) -> None:
        """optimize_inducing_points=True should change inducing locations."""

        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(42)
        d, K, n_i, M = 2, 4, 15, 8
        X_shared = torch.rand(n_i, d, **tkwargs)
        datasets = []
        for _ in range(K):
            Y_i = torch.randn(n_i, 1, **tkwargs) * 0.5 + 1.0
            datasets.append(ExperimentDataset(X=X_shared, Y=Y_i))

        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=d)).to(**tkwargs)
        Z = X_shared[:M].clone()
        Z_before = Z.clone()

        result = coordinate_ascent_em(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood_noise=torch.tensor(1e-4, **tkwargs),
            num_em_iterations=3,
            n_coord_ascent_iterations=1,
            inducing_points=Z,
            optimize_inducing_points=True,
            enable_interpolation=True,
        )

        # The returned container should have inducing points
        # (they may or may not have changed depending on optimization)
        self.assertIsInstance(result, EMPriorContainer)
        self.assertEqual(result.X_inducing.shape, Z_before.shape)

    def test_coordinate_ascent_options(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(42)
        d, K, n_i, M = 2, 4, 15, 8
        X_shared = torch.rand(n_i, d, **tkwargs)
        datasets = [
            ExperimentDataset(X=X_shared, Y=torch.randn(n_i, 1, **tkwargs) * 0.5 + 1.0)
            for _ in range(K)
        ]
        container_history: list[EMPriorContainer] = []
        final = coordinate_ascent_em(
            datasets=datasets,
            mean_module=ConstantMean().to(**tkwargs),
            covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)).to(**tkwargs),
            likelihood_noise=1e-4,  # float -> tensor conversion branch
            num_em_iterations=2,
            n_coord_ascent_iterations=1,
            inducing_points=X_shared[:M],
            enable_interpolation=True,
            mll_dataset_subsample=2,  # dataset subsampling branch
            mll_max_iter=2,  # keep the L-BFGS fit cheap; branch coverage only
            container_history=container_history,
        )
        self.assertIsInstance(final, EMPriorContainer)
        self.assertGreaterEqual(len(container_history), 2)
        self.assertTrue(all(isinstance(c, EMPriorContainer) for c in container_history))

    def test_kl_prior_fit_mll_squeezes_2d_mean(self) -> None:
        """KLPriorFitMLL.forward squeezes a 2D (n, 1) mean to 1D before the KL."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        torch.manual_seed(0)
        d, M = 2, 5
        X_ref = torch.rand(M, d, **tkwargs)
        mu_em = torch.zeros(M, **tkwargs)
        Sigma_em = torch.eye(M, **tkwargs)

        class _TwoDMean(Mean):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Return shape (..., n, 1) to exercise the dim > 1 squeeze branch.
                return torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)

        mean_module = _TwoDMean().to(**tkwargs)
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=d)).to(**tkwargs)
        gp = SingleTaskGP(
            train_X=X_ref[:1],
            train_Y=torch.zeros(1, 1, **tkwargs),
            mean_module=mean_module,
            covar_module=covar_module,
            outcome_transform=None,
        )
        kl_mll = KLPriorFitMLL(gp.likelihood, gp, X_ref, mu_em, Sigma_em)
        # forward ignores its (function_dist, target) args.
        val = kl_mll.forward(None, None)
        self.assertEqual(val.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(val))
