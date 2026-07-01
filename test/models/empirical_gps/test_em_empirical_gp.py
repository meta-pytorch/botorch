#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.empirical_gps.em_empirical_gp import (  # noqa: E501
    _e_step,
    _m_step,
    build_shared_gp_model_list,
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    EMPriorContainer,
    pretrain_em_prior,
)
from botorch.models.empirical_gps.utils import build_unique_inputs, ExperimentDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean


class TestEMEmpiricalGaussianProcess(BotorchTestCase):
    """Tests for EM-based Empirical Gaussian Process."""

    # =========================================================================
    # Test Helpers
    # =========================================================================

    def _make_datasets(
        self,
        K: int,
        n_i: int,
        tkwargs: dict,
        seed: int = 42,
        target_mean: float = 3.0,
        noise_std: float = 1.0,
    ) -> list[ExperimentDataset]:
        """Create K datasets with n_i observations each.

        Args:
            K: Number of datasets.
            n_i: Observations per dataset.
            tkwargs: Tensor kwargs (device, dtype).
            seed: Random seed for reproducibility.
            target_mean: Mean value for Y observations.
            noise_std: Standard deviation of noise added to Y.

        Returns:
            List of ExperimentDataset objects.
        """
        torch.manual_seed(seed)
        datasets = []
        for _ in range(K):
            X_i = torch.rand(n_i, 1, **tkwargs)
            Y_i = torch.randn(n_i, 1, **tkwargs) * noise_std + target_mean
            datasets.append(ExperimentDataset(X=X_i, Y=Y_i))
        return datasets

    def _make_model(
        self,
        datasets: list[ExperimentDataset],
        tkwargs: dict,
        train_X: torch.Tensor | None = None,
        train_Y: torch.Tensor | None = None,
        **model_kwargs,
    ) -> EMEmpiricalGaussianProcess:
        """Create an EMEmpiricalGaussianProcess model.

        Args:
            datasets: List of ExperimentDataset objects.
            tkwargs: Tensor kwargs (device, dtype).
            train_X: Training inputs (default: random 3x1).
            train_Y: Training targets (default: random 3x1).
            **model_kwargs: Additional kwargs for EMEmpiricalGaussianProcess.

        Returns:
            EMEmpiricalGaussianProcess model.
        """
        if train_X is None:
            train_X = torch.rand(3, 1, **tkwargs)
        if train_Y is None:
            train_Y = torch.randn(3, 1, **tkwargs)

        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(MaternKernel()).to(**tkwargs)

        default_kwargs = {"num_em_iterations": 3}
        default_kwargs.update(model_kwargs)

        return EMEmpiricalGaussianProcess(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            train_X=train_X,
            train_Y=train_Y,
            **default_kwargs,
        )

    # =========================================================================
    # Core EM Algorithm Tests
    # =========================================================================

    def _test_e_step_and_m_step(self) -> None:
        """Test E-step and M-step properties.

        This test verifies:
            1. E-step produces K conditional distributions with correct shapes
            2. E-step reduces uncertainty (posterior variance <= prior variance)
            3. E-step produces PSD covariance matrices
            4. M-step mean is the average of conditional means (ML update)
            5. M-step produces PSD covariance matrix
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, N, n_i = 3, 5, 3

            # Create datasets with shared locations
            X_all = torch.linspace(0, 1, N, **tkwargs).unsqueeze(-1)
            datasets = []
            for i in range(K):
                indices = torch.arange(i, i + n_i) % N
                X_i = X_all[indices]
                Y_i = torch.randn(n_i, 1, **tkwargs) + 5.0
                datasets.append(ExperimentDataset(X=X_i, Y=Y_i))

            unique_inputs = build_unique_inputs(datasets, X_all)

            # Initialize prior
            mu = torch.zeros(N, **tkwargs)
            kernel = ScaleKernel(MaternKernel()).to(**tkwargs)
            Sigma = kernel(X_all, X_all).to_dense().detach()

            # E-step
            cond_means, cond_covs = _e_step(
                datasets,
                mu,
                Sigma,
                experiment_indices=unique_inputs.experiment_indices,
            )

            # Property: K conditional distributions with correct shapes
            self.assertEqual(len(cond_means), K)
            self.assertEqual(len(cond_covs), K)

            for i in range(K):
                self.assertEqual(cond_means[i].shape, (N,))
                self.assertEqual(cond_covs[i].shape, (N, N))

                # Property: Uncertainty reduction (diagonal <= prior diagonal)
                prior_var = torch.diag(Sigma)
                post_var = torch.diag(cond_covs[i])
                self.assertTrue((post_var <= prior_var + 1e-5).all())

                # Property: PSD covariance
                eigvals = torch.linalg.eigvalsh(cond_covs[i])
                self.assertTrue((eigvals >= -1e-5).all())

            # M-step (ML updates)
            mu_new, Sigma_new = _m_step(cond_means, cond_covs)

            # Property: mu_new is average of conditional means
            expected_mu = sum(cond_means) / K
            self.assertAllClose(mu_new, expected_mu, atol=1e-5)

            # Property: Sigma_new is PSD
            eigvals = torch.linalg.eigvalsh(Sigma_new)
            self.assertTrue((eigvals >= -1e-5).all())

    def _test_em_iterations_deterministic(self) -> None:
        """Test EM iterations are deterministic and produce valid outputs.

        This test verifies:
            1. Running EM twice with same inputs gives identical results
            2. Output shapes are correct (mu: (N,), Sigma: (N, N))
            3. Final covariance is positive semi-definite
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, N, n_i = 3, 5, 3

            X_all = torch.linspace(0, 1, N, **tkwargs).unsqueeze(-1)
            datasets = []
            for i in range(K):
                indices = torch.arange(i, i + n_i) % N
                X_i = X_all[indices]
                Y_i = torch.randn(n_i, 1, **tkwargs) + 5.0
                datasets.append(ExperimentDataset(X=X_i, Y=Y_i))

            unique_inputs = build_unique_inputs(datasets, X_all)

            mu_init = torch.zeros(N, **tkwargs)
            kernel = ScaleKernel(MaternKernel()).to(**tkwargs)
            Sigma_init = kernel(X_all, X_all).to_dense().detach()

            def run_em(mu, Sigma, num_iters, datasets, unique_inputs):
                for _ in range(num_iters):
                    cond_means, cond_covs = _e_step(
                        datasets,
                        mu,
                        Sigma,
                        experiment_indices=unique_inputs.experiment_indices,
                    )
                    mu, Sigma = _m_step(cond_means, cond_covs)
                return mu, Sigma

            mu1, Sigma1 = run_em(
                mu_init.clone(), Sigma_init.clone(), 3, datasets, unique_inputs
            )
            mu2, Sigma2 = run_em(
                mu_init.clone(), Sigma_init.clone(), 3, datasets, unique_inputs
            )

            # Property: Deterministic
            self.assertAllClose(mu1, mu2, atol=1e-10)
            self.assertAllClose(Sigma1, Sigma2, atol=1e-10)

            # Property: Valid shapes and PSD
            self.assertEqual(mu1.shape, (N,))
            self.assertEqual(Sigma1.shape, (N, N))
            eigvals = torch.linalg.eigvalsh(Sigma1)
            self.assertTrue((eigvals >= -1e-5).all())

    def _test_em_output_symmetry_and_psd(self) -> None:
        """Test EM outputs are symmetric and PSD at each step.

        This test verifies:
            1. Initial covariance from kernel is symmetric and PSD
            2. E-step conditional covariances are symmetric and PSD
            3. M-step output covariance is symmetric and PSD
            4. Multiple EM iterations maintain symmetry and PSD properties
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        asym_tol, psd_tol = 1e-15, 1e-15
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        unique_inputs = build_unique_inputs(datasets, None)
        X_all = unique_inputs.X_all
        N_unique = X_all.shape[0]

        kernel = ScaleKernel(MaternKernel()).to(**tkwargs)
        Sigma_init = kernel(X_all, X_all).to_dense().detach()
        mu_init = torch.zeros(N_unique, **tkwargs)

        # Check 1: Initial covariance
        asym_init = torch.abs(Sigma_init - Sigma_init.T).max()
        self.assertLess(asym_init, asym_tol)
        self.assertGreaterEqual(torch.linalg.eigvalsh(Sigma_init).min(), psd_tol)

        # Check 2 & 3: E-step and M-step outputs
        likelihood_noise = torch.tensor(0.1, **tkwargs)
        cond_means, cond_covs = _e_step(
            datasets,
            mu_init,
            Sigma_init,
            likelihood_noise=likelihood_noise,
            experiment_indices=unique_inputs.experiment_indices,
        )
        for i, cond_cov in enumerate(cond_covs):
            asym = torch.abs(cond_cov - cond_cov.T).max()
            self.assertLess(asym, asym_tol, f"E-step cond_cov[{i}] not symmetric")
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(cond_cov).min(),
                psd_tol,
                f"E-step cond_cov[{i}] not PSD",
            )

        mu_m, Sigma_m = _m_step(cond_means, cond_covs)
        asym_m = torch.abs(Sigma_m - Sigma_m.T).max()
        self.assertLess(asym_m, asym_tol, "M-step Sigma not symmetric")
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(Sigma_m).min(), psd_tol, "M-step Sigma not PSD"
        )

        # Check 4: Multiple EM iterations maintain symmetry/PSD
        for num_iters in [1, 3, 5]:
            mu, Sigma = mu_init.clone(), Sigma_init.clone()
            for _ in range(num_iters):
                cond_means, cond_covs = _e_step(
                    datasets,
                    mu,
                    Sigma,
                    likelihood_noise=likelihood_noise,
                    experiment_indices=unique_inputs.experiment_indices,
                )
                mu, Sigma = _m_step(cond_means, cond_covs)

            asym = torch.abs(Sigma - Sigma.T).max()
            self.assertLess(
                asym, asym_tol, f"After {num_iters} iters: Sigma not symmetric"
            )
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(Sigma).min(),
                psd_tol,
                f"After {num_iters} iters: Sigma not PSD",
            )

    def test_core_em_algorithm(self) -> None:
        """Test core EM algorithm components: E-step, M-step, and iterations."""
        self._test_e_step_and_m_step()
        self._test_em_iterations_deterministic()
        self._test_em_output_symmetry_and_psd()

    # =========================================================================
    # Model Learning and Output Tests
    # =========================================================================

    def _test_model_learns_empirical_prior(self) -> None:
        """Test model learns prior reflecting data.

        This test verifies that with complete observations and tiny noise,
        EM converges to the empirical mean and covariance in one iteration.
        This is a mathematically exact property that validates the EM implementation.
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 50, 15
        target_mean = 5.0

        # Use shared X locations (complete observations)
        torch.manual_seed(42)
        X_shared = torch.linspace(0.3, 0.7, n_i, **tkwargs).unsqueeze(-1)

        # Create a GP prior with non-trivial covariance structure
        # Sample Y_i ~ N(mean_vec, Sigma_true) for each dataset
        kernel = ScaleKernel(MaternKernel()).to(**tkwargs)
        Sigma_true = kernel(X_shared, X_shared).to_dense().detach()
        # Add small jitter for numerical stability in Cholesky
        jitter = 1e-10
        Sigma_true = Sigma_true + jitter * torch.eye(n_i, **tkwargs)
        L_true = torch.linalg.cholesky(Sigma_true)
        mean_vec = torch.full((n_i,), target_mean, **tkwargs)

        datasets = []
        for _ in range(K):
            z = torch.randn(n_i, **tkwargs)
            Y_i = mean_vec + L_true @ z
            datasets.append(ExperimentDataset(X=X_shared.clone(), Y=Y_i.unsqueeze(-1)))

        # Compute empirical statistics directly
        Y_stack = torch.stack([d.Y.squeeze(-1) for d in datasets])  # (K, n_i)
        empirical_mean = Y_stack.mean(dim=0)
        diffs = Y_stack - empirical_mean.unsqueeze(0)
        empirical_cov = (diffs.T @ diffs) / K

        # Test EM directly: with complete observations and tiny noise,
        # one EM iteration from any initialization should give empirical stats
        unique_inputs = build_unique_inputs(datasets, X_shared)
        likelihood_noise = torch.tensor(1e-12, **tkwargs)

        # Regardless of initialization, EM should converge to empirical stats in one
        # iteration
        mu_init = torch.randn(n_i, **tkwargs)
        Sigma_init = torch.randn(n_i, n_i, **tkwargs)
        Sigma_init = Sigma_init @ Sigma_init.T

        cond_means, cond_covs = _e_step(
            datasets,
            mu_init,
            Sigma_init,
            likelihood_noise=likelihood_noise,
            experiment_indices=unique_inputs.experiment_indices,
        )
        mu_em, Sigma_em = _m_step(cond_means, cond_covs)

        # With tiny noise and complete observations, EM should converge to
        # empirical statistics exactly (up to numerical precision)
        self.assertAllClose(mu_em, empirical_mean, atol=1e-8)
        self.assertAllClose(Sigma_em, empirical_cov, atol=1e-8)

        # Also verify full model produces the same result.
        # Test both interpolation modes: on the observed grid, results should match.
        for enable_interpolation in [False, True]:
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-15))
            likelihood.noise = 1e-12
            likelihood.raw_noise.requires_grad_(False)

            model = EMEmpiricalGaussianProcess(
                datasets=datasets,
                mean_module=ConstantMean().to(**tkwargs),
                covar_module=ScaleKernel(MaternKernel()).to(**tkwargs),
                train_X=X_shared[:1],
                train_Y=torch.zeros(1, 1, **tkwargs),
                likelihood=likelihood,
                num_em_iterations=1,
                enable_interpolation=enable_interpolation,
            )

            # Verify internal EM results match empirical statistics
            self.assertAllClose(model._mu_inducing, empirical_mean, atol=1e-8)
            self.assertAllClose(model._Sigma_inducing, empirical_cov, atol=1e-8)

            # Verify forward() returns the empirical prior at inducing locations.
            # NOTE: Call forward() directly to get the prior without conditioning
            # on train data (model(X) would condition on train_X/train_Y).
            model.eval()
            prior_dist = model.forward(X_shared)
            self.assertAllClose(
                prior_dist.mean,
                empirical_mean,
                atol=1e-8,
            )
            self.assertAllClose(
                prior_dist.covariance_matrix,
                empirical_cov,
                atol=1e-8,
            )

    def _test_model_with_map_priors(self) -> None:
        """Test model with covariance and mean priors (MAP estimation).

        This test verifies:
            1. iw_nu is auto-computed as M + 2 when use_covar_prior=True
            2. Configuration flags are stored correctly on the model
            3. MAP models produce valid finite, PSD outputs
            4. MAP prior regularizes the covariance toward the kernel prior
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)

        # Test with covariance prior (let iw_nu be auto-computed)
        model_cov = self._make_model(datasets, tkwargs, use_covar_prior=True)
        # Verify iw_nu was auto-computed as M + 2
        self.assertEqual(model_cov.iw_nu, model_cov._N_inducing + 2)
        self.assertTrue(model_cov.use_covar_prior)
        self.assertFalse(model_cov.use_mean_prior)

        # Test with both priors (let iw_nu be auto-computed)
        model_both = self._make_model(
            datasets, tkwargs, use_mean_prior=True, use_covar_prior=True
        )
        self.assertTrue(model_both.use_mean_prior)
        self.assertTrue(model_both.use_covar_prior)

        # Both should produce valid outputs (finite mean, PSD covariance)
        model_cov.eval()
        model_both.eval()
        test_X = torch.rand(3, 1, **tkwargs)
        for model in (model_cov, model_both):
            output = model(test_X)
            self.assertTrue(torch.isfinite(output.mean).all())
            covar = output.covariance_matrix
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(covar).min().item(),
                -1e-8,
                "MAP model output not PSD",
            )

        # Verify MAP regularization: with a strong IW prior (large nu),
        # the EM covariance should stay closer to the kernel prior than ML.
        model_ml = self._make_model(datasets, tkwargs, num_em_iterations=5)
        model_strong_prior = self._make_model(
            datasets,
            tkwargs,
            use_covar_prior=True,
            iw_nu=float(model_ml._N_inducing + 100),
            num_em_iterations=5,
        )
        # Compute kernel prior covariance at inducing points
        K_prior = (
            model_ml.initial_covar_module(model_ml._X_inducing, model_ml._X_inducing)
            .to_dense()
            .detach()
        )
        dist_ml = (model_ml._Sigma_inducing - K_prior).norm()
        dist_map = (model_strong_prior._Sigma_inducing - K_prior).norm()
        self.assertLess(
            dist_map.item(),
            dist_ml.item(),
            "Strong IW prior should pull Sigma closer to kernel prior than ML",
        )

    def _test_forward_output_symmetry_and_psd(self) -> None:
        """Test model forward() output is symmetric and PSD.

        This test verifies:
            1. Model output covariance is symmetric
            2. Model output covariance is positive semi-definite
            3. Properties hold for both direct indexing and interpolation paths
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        asym_tol, psd_tol = 1e-10, -1e-8
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        unique_inputs = build_unique_inputs(datasets, None)

        # Test interpolation path (default)
        model_interp = self._make_model(datasets, tkwargs)
        model_interp.eval()
        test_X = torch.rand(5, 1, **tkwargs)
        output_interp = model_interp(test_X)
        covar_interp = output_interp.covariance_matrix

        asym_interp = torch.abs(covar_interp - covar_interp.T).max()
        self.assertLess(asym_interp, asym_tol, "Interpolation output not symmetric")
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(covar_interp).min().item(),
            psd_tol,
            "Interpolation output not PSD",
        )

        # Test direct indexing path
        train_X = unique_inputs.X_all[:3]
        train_Y = torch.randn(3, 1, **tkwargs)
        model_direct = self._make_model(
            datasets,
            tkwargs,
            train_X=train_X,
            train_Y=train_Y,
            enable_interpolation=False,
        )
        model_direct.eval()
        X_historical = model_direct._X_inducing[:4]
        output_direct = model_direct(X_historical)
        covar_direct = output_direct.covariance_matrix

        asym_direct = torch.abs(covar_direct - covar_direct.T).max()
        self.assertLess(asym_direct, asym_tol, "Direct indexing output not symmetric")
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(covar_direct).min().item(),
            psd_tol,
            "Direct indexing output not PSD",
        )

    def test_model_learning_and_output(self) -> None:
        """Test model learning behavior and output properties."""
        self._test_model_learns_empirical_prior()
        self._test_model_with_map_priors()
        self._test_forward_output_symmetry_and_psd()

    # =========================================================================
    # Gradient Flow and MLL Tests
    # =========================================================================

    def _test_differentiability_and_mll(self) -> None:
        """Test gradient flow through forward and MLL.

        This test verifies:
            1. Gradients exist and are finite on hyperparameters after backward()
            2. MLL returns a finite scalar value
            3. Analytical gradients match finite difference approximation
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, n_i = 3, 3

            datasets = self._make_datasets(K, n_i, tkwargs, target_mean=0.0)
            train_X = torch.rand(2, 1, **tkwargs)
            train_Y = torch.randn(2, 1, **tkwargs)
            model = self._make_model(
                datasets, tkwargs, train_X=train_X, train_Y=train_Y, num_em_iterations=2
            )

            # Test gradient flow through forward
            model.train()
            output = model(train_X)
            loss = -output.log_prob(train_Y.squeeze(-1))
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.assertTrue(
                        torch.isfinite(param.grad).all(),
                        f"Non-finite gradient on {name}",
                    )

            # Test MLL
            mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
            model.zero_grad()
            mll_value = mll(model(train_X), train_Y.squeeze(-1))

            self.assertEqual(mll_value.shape, ())
            self.assertTrue(torch.isfinite(mll_value))

            # Finite difference gradient check
            mll_value.backward()
            for _name, param in model.named_parameters():
                if (
                    param.requires_grad
                    and param.grad is not None
                    and param.numel() == 1
                ):
                    eps = 1e-5
                    analytical_grad = param.grad.clone()
                    original_value = param.data.clone()

                    param.data = original_value + eps
                    mll_plus = mll(model(train_X), train_Y.squeeze(-1))
                    param.data = original_value - eps
                    mll_minus = mll(model(train_X), train_Y.squeeze(-1))
                    param.data = original_value

                    numerical_grad = (mll_plus - mll_minus) / (2 * eps)
                    numerical_grad = numerical_grad.to(dtype=analytical_grad.dtype)
                    grad_atol = 0.05 if dtype == torch.float else 0.01
                    grad_rtol = 0.3 if dtype == torch.float else 0.05
                    self.assertAllClose(
                        analytical_grad.squeeze(),
                        numerical_grad,
                        atol=grad_atol,
                        rtol=grad_rtol,
                    )
                    break

    def test_gradient_flow_and_mll(self) -> None:
        """Test differentiability and marginal log-likelihood computation."""
        self._test_differentiability_and_mll()

    # =========================================================================
    # Inducing Points Tests
    # =========================================================================

    def _test_inducing_point_approximation(self) -> None:
        """Test the inducing point approximation produces valid outputs.

        This test verifies:
            1. Cached Cholesky factor is lower triangular
            2. Cached delta_mu contains finite values
            3. Eval mode produces PSD, symmetric covariance matrices
            4. Training mode still works (re-runs EM with gradient flow)
            5. Interpolation at inducing points exactly recovers EM mean
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4
        psd_tol = -1e-8

        datasets = self._make_datasets(K, n_i, tkwargs)
        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)
        model = self._make_model(
            datasets,
            tkwargs,
            train_X=train_X,
            train_Y=train_Y,
            use_covar_prior=True,
        )

        # Verify cached quantities
        L = model._cached_L_kernel_inducing
        self.assertLess(torch.triu(L, diagonal=1).abs().max().item(), 1e-10)
        self.assertTrue(torch.isfinite(model._cached_delta_mu).all())

        # Eval mode produces valid output
        model.eval()
        test_X = torch.rand(5, 1, **tkwargs)
        output = model(test_X)

        self.assertTrue(torch.isfinite(output.mean).all())
        covar = output.covariance_matrix
        self.assertGreaterEqual(torch.linalg.eigvalsh(covar).min().item(), psd_tol)
        self.assertLess((covar - covar.T).abs().max().item(), 1e-10)

        # Training mode works (must use train_X due to GPyTorch requirement)
        model.train()
        dist_train = model(train_X)
        self.assertTrue(torch.isfinite(dist_train.mean).all())
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(dist_train.covariance_matrix).min().item(), psd_tol
        )

        # Exact interpolation at inducing points recovers EM prior mean/covariance
        # Note: We call forward() directly because model(X) returns the posterior
        # (conditioned on train data), not the prior.
        model.eval()
        output_at_inducing = model.forward(model._X_inducing)
        self.assertAllClose(output_at_inducing.mean, model._mu_inducing, atol=1e-5)
        self.assertAllClose(
            output_at_inducing.covariance_matrix, model._Sigma_inducing, atol=1e-5
        )

    def _test_custom_inducing_points(self) -> None:
        """Test model with custom inducing points.

        This test verifies:
            1. Subset of inducing points works correctly
            2. Grid inducing points (different from historical X) work correctly
            3. _use_inducing_points flag is set to True
            4. Outputs are valid (finite mean, PSD covariance)
            5. Shape validation raises ValueError for invalid inputs
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, n_i = 5, 4
            psd_tol = -1e-6

            datasets = self._make_datasets(K, n_i, tkwargs)

            # Subset of inducing points
            unique_inputs = build_unique_inputs(datasets, None)
            n_inducing = max(2, unique_inputs.X_all.shape[0] // 2)
            inducing_subset = unique_inputs.X_all[:n_inducing].clone()

            model_subset = self._make_model(
                datasets, tkwargs, inducing_points=inducing_subset
            )
            self.assertEqual(model_subset._N_inducing, n_inducing)
            self.assertTrue(model_subset._use_inducing_points)

            model_subset.eval()
            test_X = torch.rand(5, 1, **tkwargs)
            output = model_subset(test_X)
            self.assertTrue(torch.isfinite(output.mean).all())
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(output.covariance_matrix).min().item(), psd_tol
            )

            # Grid inducing points
            custom_inducing = torch.linspace(0, 1, 8, **tkwargs).unsqueeze(-1)
            model_custom = self._make_model(
                datasets, tkwargs, inducing_points=custom_inducing
            )
            self.assertEqual(model_custom._N_inducing, 8)
            self.assertAllClose(model_custom._X_inducing, custom_inducing)

            model_custom.eval()
            output_custom = model_custom(test_X)
            self.assertTrue(torch.isfinite(output_custom.mean).all())

            # Shape validation errors
            with self.assertRaises(ValueError):
                self._make_model(
                    datasets, tkwargs, inducing_points=torch.rand(5, **tkwargs)
                )
            with self.assertRaises(ValueError):
                self._make_model(
                    datasets, tkwargs, inducing_points=torch.rand(5, 3, **tkwargs)
                )

    def _test_inducing_points_backward_compatibility(self) -> None:
        """Test that inducing_points=None uses direct indexing correctly.

        This test verifies:
            1. Default (inducing_points=None) sets _use_inducing_points=False
            2. Default uses _X_obs as inducing points
            3. Explicit inducing_points sets _use_inducing_points=True
            4. Both modes produce valid outputs (finite, PSD)

        Note: When inducing_points is explicitly set (even to historical X),
        the model uses interpolation which may give slightly different results
        due to numerical precision.
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, n_i = 5, 4

            datasets = self._make_datasets(K, n_i, tkwargs)

            # Default (no inducing points)
            model_default = self._make_model(datasets, tkwargs)
            self.assertFalse(model_default._use_inducing_points)
            self.assertAllClose(model_default._X_inducing, model_default._X_obs)

            # Explicit inducing_points = unique inputs
            model_explicit = self._make_model(
                datasets, tkwargs, inducing_points=model_default._X_obs.clone()
            )
            self.assertTrue(model_explicit._use_inducing_points)

            # Both should produce valid outputs
            model_default.eval()
            model_explicit.eval()
            test_X = torch.rand(5, 1, **tkwargs)

            out_default = model_default(test_X)
            out_explicit = model_explicit(test_X)

            # Outputs should both be valid (finite, PSD)
            self.assertTrue(torch.isfinite(out_default.mean).all())
            self.assertTrue(torch.isfinite(out_explicit.mean).all())
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(out_default.covariance_matrix).min().item(), -1e-6
            )
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(out_explicit.covariance_matrix).min().item(),
                -1e-6,
            )

    def _test_inducing_points_differentiability(self) -> None:
        """Test gradient flow with custom inducing points.

        This test verifies:
            1. Gradients exist and are finite after backward() through forward
            2. MLL returns finite value with custom inducing points
            3. Gradients flow through MLL backward()
        """
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            K, n_i = 3, 3

            datasets = self._make_datasets(K, n_i, tkwargs, target_mean=0.0)
            train_X = torch.rand(2, 1, **tkwargs)
            train_Y = torch.randn(2, 1, **tkwargs)
            inducing_points = torch.linspace(0, 1, 5, **tkwargs).unsqueeze(-1)

            model = self._make_model(
                datasets,
                tkwargs,
                train_X=train_X,
                train_Y=train_Y,
                num_em_iterations=2,
                inducing_points=inducing_points,
            )

            # Gradient flow through forward
            model.train()
            output = model(train_X)
            loss = -output.log_prob(train_Y.squeeze(-1))
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.assertTrue(
                        torch.isfinite(param.grad).all(),
                        f"Non-finite gradient on {name}",
                    )

            # MLL with inducing points
            mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
            model.zero_grad()
            mll_value = mll(model(train_X), train_Y.squeeze(-1))
            self.assertTrue(torch.isfinite(mll_value))

            mll_value.backward()
            has_grad = any(
                p.grad is not None for p in model.parameters() if p.requires_grad
            )
            self.assertTrue(has_grad, "No gradients found after MLL backward")

    def test_inducing_points(self) -> None:
        """Test inducing point approximation, custom points, and differentiability."""
        self._test_inducing_point_approximation()
        self._test_custom_inducing_points()
        self._test_inducing_points_backward_compatibility()
        self._test_inducing_points_differentiability()

        # Direct assertion so this aggregator validates behavior itself.
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._make_datasets(K=3, n_i=4, tkwargs=tkwargs)
        inducing = torch.linspace(0.0, 1.0, 5, **tkwargs).unsqueeze(-1)
        model = self._make_model(datasets, tkwargs, inducing_points=inducing)
        self.assertTrue(model._use_inducing_points)
        self.assertEqual(model._N_inducing, 5)

    def test_ill_conditioned_inducing_points_psd(self) -> None:
        # Near-duplicate inducing points make K(Z, Z) nearly singular -- the
        # regime where the naive covariance K + W dSigma W^T catastrophically
        # cancels. The stable Lambda + W Sigma W^T decomposition must still yield
        # a finite, PSD interpolated covariance.
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._make_datasets(K=5, n_i=4, tkwargs=tkwargs)
        base = torch.linspace(0.4, 0.6, 4, **tkwargs)
        inducing = torch.cat([base, base + 1e-6]).unsqueeze(-1)
        model = self._make_model(datasets, tkwargs, inducing_points=inducing)
        model.eval()
        test_X = torch.linspace(0.0, 1.0, 7, **tkwargs).unsqueeze(-1)
        out = model.forward(test_X)
        self.assertTrue(torch.isfinite(out.mean).all())
        covar = out.covariance_matrix
        self.assertTrue(torch.isfinite(covar).all())
        self.assertGreaterEqual(torch.linalg.eigvalsh(covar).min().item(), -1e-6)

    def test_extrapolation_beyond_inducing_range_psd(self) -> None:
        # Query points outside the inducing range exercise the kernel correction
        # (Nystrom residual) for extrapolation; the covariance must stay finite
        # and PSD there.
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._make_datasets(K=5, n_i=4, tkwargs=tkwargs)
        inducing = torch.linspace(0.0, 1.0, 6, **tkwargs).unsqueeze(-1)
        model = self._make_model(datasets, tkwargs, inducing_points=inducing)
        model.eval()
        test_X = torch.tensor([[-1.0], [-0.5], [1.5], [2.0]], **tkwargs)
        out = model.forward(test_X)
        self.assertTrue(torch.isfinite(out.mean).all())
        covar = out.covariance_matrix
        self.assertTrue(torch.isfinite(covar).all())
        self.assertGreaterEqual(torch.linalg.eigvalsh(covar).min().item(), -1e-6)

    # =========================================================================
    # Direct Indexing Tests (enable_interpolation=False)
    # =========================================================================

    def _test_direct_indexing_mode(self) -> None:
        """Test forward with enable_interpolation=False uses direct indexing.

        This test verifies:
            1. Query at historical observations returns valid output
            2. Query at non-historical locations raises ValueError
            3. Error message mentions "not found in historical observations"
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        # Use historical X as train_X so GPyTorch is happy
        unique_inputs = build_unique_inputs(datasets, None)
        train_X = unique_inputs.X_all[:3]
        train_Y = torch.randn(3, 1, **tkwargs)
        model = self._make_model(
            datasets,
            tkwargs,
            train_X=train_X,
            train_Y=train_Y,
            enable_interpolation=False,
        )
        model.eval()

        # Query at historical observations should work
        X_historical = model._X_inducing[:3]
        output = model(X_historical)
        self.assertEqual(output.mean.shape, (3,))
        self.assertTrue(torch.isfinite(output.mean).all())

        # Query at non-historical locations should raise
        X_new = torch.rand(2, 1, **tkwargs) + 10.0  # Far from historical
        with self.assertRaises(ValueError) as ctx:
            model(X_new)
        self.assertIn("not found in historical observations", str(ctx.exception))

    def _test_index_prior_helper(self) -> None:
        """Test _get_prior_at_indices helper returns correct subsets.

        This test verifies:
            1. Output shapes are correct (mu: (n,), Sigma: (n, n))
            2. Values match direct indexing into _mu_inducing and _Sigma_inducing
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        model = self._make_model(datasets, tkwargs)

        # Get indices for first 3 points
        indices = torch.tensor([0, 1, 2], device=self.device)
        mu, Sigma = model._get_prior_at_indices(indices)

        # Verify shapes
        self.assertEqual(mu.shape, (3,))
        self.assertEqual(Sigma.shape, (3, 3))

        # Verify values match direct indexing
        self.assertAllClose(mu, model._mu_inducing[indices])
        self.assertAllClose(Sigma, model._Sigma_inducing[indices][:, indices])

    def _test_find_indices_in_historical_error(self) -> None:
        """Test _find_indices_in_historical raises on missing points.

        This test verifies:
            1. ValueError is raised when query points are not in historical data
            2. Error message includes the count of missing points
        """
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 3, 3

        datasets = self._make_datasets(K, n_i, tkwargs)
        model = self._make_model(datasets, tkwargs)

        # Points far from any historical observation
        X_missing = torch.tensor([[100.0], [200.0]], **tkwargs)
        with self.assertRaises(ValueError) as ctx:
            model._find_indices_in_historical(X_missing)

        self.assertIn("2 query point(s) not found", str(ctx.exception))

    def test_direct_indexing(self) -> None:
        """Test direct indexing mode (enable_interpolation=False)."""
        self._test_direct_indexing_mode()
        self._test_index_prior_helper()
        self._test_find_indices_in_historical_error()

        # Direct assertion so this aggregator validates behavior itself.
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._make_datasets(K=3, n_i=4, tkwargs=tkwargs)
        model = self._make_model(datasets, tkwargs, enable_interpolation=False)
        self.assertFalse(model.enable_interpolation)

    # =========================================================================
    # Pre-training Workflow Tests
    # =========================================================================

    def _make_modules(self, tkwargs: dict) -> tuple[ConstantMean, ScaleKernel]:
        """Create mean and covariance modules."""
        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(MaternKernel()).to(**tkwargs)
        return mean_module, covar_module

    def _make_pretrained_prior(
        self,
        datasets: list[ExperimentDataset],
        tkwargs: dict,
        **em_kwargs,
    ) -> EMPriorContainer:
        """Create a pre-trained EMPriorContainer using existing patterns."""
        mean_module, covar_module = self._make_modules(tkwargs)
        default_kwargs = {"num_em_iterations": 3}
        default_kwargs.update(em_kwargs)
        return pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            **default_kwargs,
        )

    def test_pretrain_em_prior_and_container(self) -> None:
        """Test EMPriorContainer creation, shapes, symmetry, and PSD properties."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        prior = self._make_pretrained_prior(datasets, tkwargs)

        # Shape checks
        M = prior.X_inducing.shape[0]
        self.assertEqual(prior.mu_inducing.shape, (M,))
        self.assertEqual(prior.Sigma_inducing.shape, (M, M))
        self.assertEqual(prior.L_kernel_inducing.shape, (M, M))
        self.assertEqual(prior.delta_mu.shape, (M,))

        # Symmetry check
        self.assertLess(
            (prior.Sigma_inducing - prior.Sigma_inducing.T).abs().max().item(),
            1e-10,
        )

        # PSD check
        eigvals = torch.linalg.eigvalsh(prior.Sigma_inducing)
        self.assertTrue((eigvals >= -1e-10).all())

        # Tensors should be detached (no gradient tracking)
        self.assertFalse(prior.mu_inducing.requires_grad)
        self.assertFalse(prior.Sigma_inducing.requires_grad)
        self.assertFalse(prior.X_inducing.requires_grad)

    def test_pretrained_vs_internal_equivalence(self) -> None:
        """Test that pretrained and internal EM give equivalent results."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)

        # Create shared modules and likelihood
        mean_module, covar_module = self._make_modules(tkwargs)
        likelihood = GaussianLikelihood().to(**tkwargs)

        # Create pre-trained prior with same noise as likelihood
        prior = pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            num_em_iterations=3,
            likelihood_noise=likelihood.noise,  # Use same noise as internal
        )

        # Model with pre-trained prior
        model_pretrained = EMEmpiricalGaussianProcess(
            train_X=train_X,
            train_Y=train_Y,
            em_prior=prior,
            likelihood=likelihood,
        )

        # Model with internal EM (same modules and params)
        model_internal = EMEmpiricalGaussianProcess(
            train_X=train_X,
            train_Y=train_Y,
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            num_em_iterations=3,
        )

        # Compare outputs in eval mode
        model_pretrained.eval()
        model_internal.eval()
        test_X = torch.rand(5, 1, **tkwargs)

        with torch.no_grad():
            out_pretrained = model_pretrained(test_X)
            out_internal = model_internal(test_X)

        # Means should match closely (same EM algorithm)
        self.assertAllClose(out_pretrained.mean, out_internal.mean, atol=1e-5)
        self.assertAllClose(
            out_pretrained.covariance_matrix,
            out_internal.covariance_matrix,
            atol=1e-5,
        )

    def test_pretrained_prior_skips_em_in_training_forward(self) -> None:
        """Verify pretrained model doesn't re-run EM in training mode."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)

        prior = self._make_pretrained_prior(datasets, tkwargs)
        model = EMEmpiricalGaussianProcess(
            train_X=train_X, train_Y=train_Y, em_prior=prior
        )

        # Capture initial state
        mu_before = model._mu_inducing.clone()
        Sigma_before = model._Sigma_inducing.clone()

        # Forward in training mode (should NOT re-run EM)
        model.train()
        _ = model(train_X)

        # mu/Sigma should not change (no EM re-run)
        self.assertTrue(torch.equal(model._mu_inducing, mu_before))
        self.assertTrue(torch.equal(model._Sigma_inducing, Sigma_before))

    def test_pretrained_prior_reuse_multiple_test_sets(self) -> None:
        """Core use case: pre-train once, reuse for multiple test sets."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        prior = self._make_pretrained_prior(datasets, tkwargs)

        # Create multiple models with same prior, different test sets
        torch.manual_seed(123)
        for _ in range(3):
            train_X = torch.rand(4, 1, **tkwargs)
            train_Y = torch.randn(4, 1, **tkwargs)

            model = EMEmpiricalGaussianProcess(
                train_X=train_X, train_Y=train_Y, em_prior=prior
            )
            model.eval()

            test_X = torch.rand(5, 1, **tkwargs)
            posterior = model(test_X)

            # Basic sanity checks
            self.assertEqual(posterior.mean.shape, (5,))
            self.assertTrue(torch.isfinite(posterior.mean).all())
            self.assertGreaterEqual(
                torch.linalg.eigvalsh(posterior.covariance_matrix).min().item(), -1e-6
            )

        # All models should share the same inducing points from prior
        self.assertAllClose(model._X_inducing, prior.X_inducing)

    def test_from_pretrained_factory_method(self) -> None:
        """Test the from_pretrained factory method."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        K, n_i = 5, 4

        datasets = self._make_datasets(K, n_i, tkwargs)
        prior = self._make_pretrained_prior(datasets, tkwargs)

        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)

        # Create model via factory method
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=prior, train_X=train_X, train_Y=train_Y
        )

        # Should behave the same as constructor with em_prior
        self.assertTrue(model._using_pretrained_prior)
        model.eval()
        test_X = torch.rand(5, 1, **tkwargs)
        output = model(test_X)
        self.assertTrue(torch.isfinite(output.mean).all())


class TestBuildSharedGPModelList(BotorchTestCase):
    """Tests for build_shared_gp_model_list and fit_gpytorch_mll compatibility."""

    def _make_datasets(
        self,
        K: int,
        n_per_dataset: list[int],
        d: int,
        tkwargs: dict,
        seed: int = 42,
    ) -> list[ExperimentDataset]:
        """Create K heterogeneous datasets with different sizes."""
        torch.manual_seed(seed)
        datasets = []
        for i in range(K):
            n_i = n_per_dataset[i] if i < len(n_per_dataset) else n_per_dataset[-1]
            X_i = torch.rand(n_i, d, **tkwargs)
            Y_i = torch.randn(n_i, 1, **tkwargs)
            datasets.append(ExperimentDataset(X=X_i, Y=Y_i))
        return datasets

    def test_shared_parameters(self) -> None:
        """Test that all GPs in the ModelList share the same parameter objects."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        datasets = self._make_datasets(
            K=3, n_per_dataset=[8, 10, 12], d=2, tkwargs=tkwargs
        )
        mean = ConstantMean().to(**tkwargs)
        kernel = ScaleKernel(RBFKernel()).to(**tkwargs)

        model_list, mll = build_shared_gp_model_list(
            datasets=datasets,
            mean_module=mean,
            covar_module=kernel,
            observation_noise=1e-2,
        )

        # Verify all GPs share the SAME mean_module instance
        for gp in model_list.models:
            self.assertIs(gp.mean_module, mean)
            self.assertIs(gp.covar_module, kernel)

        # Verify RAW parameter objects are identical (same memory location)
        # Note: lengthscale/outputscale are computed properties, so we check
        # the underlying raw parameters instead
        raw_lengthscales = [
            gp.covar_module.base_kernel.raw_lengthscale for gp in model_list.models
        ]
        for raw_ls in raw_lengthscales[1:]:
            self.assertIs(raw_ls, raw_lengthscales[0])

        raw_outputscales = [gp.covar_module.raw_outputscale for gp in model_list.models]
        for raw_os in raw_outputscales[1:]:
            self.assertIs(raw_os, raw_outputscales[0])

    def test_fit_gpytorch_mll_compatibility(self) -> None:
        """Test that build_shared_gp_model_list works with fit_gpytorch_mll."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        datasets = self._make_datasets(
            K=5, n_per_dataset=[8, 12, 10, 6, 14], d=2, tkwargs=tkwargs
        )
        mean = ConstantMean().to(**tkwargs)
        kernel = ScaleKernel(RBFKernel()).to(**tkwargs)

        # Store initial hyperparameters
        lengthscale_before = kernel.base_kernel.lengthscale.clone().detach()
        outputscale_before = kernel.outputscale.clone().detach()

        model_list, mll = build_shared_gp_model_list(
            datasets=datasets,
            mean_module=mean,
            covar_module=kernel,
            observation_noise=1e-2,
        )

        # This should work without errors (the main test!)
        fit_gpytorch_mll(mll)

        # Verify hyperparameters have changed
        lengthscale_after = kernel.base_kernel.lengthscale
        outputscale_after = kernel.outputscale

        params_changed = not torch.allclose(
            lengthscale_before, lengthscale_after
        ) or not torch.allclose(outputscale_before, outputscale_after)
        self.assertTrue(params_changed, "Hyperparameters should change after fitting")

        # All hyperparameters should be finite
        self.assertTrue(torch.isfinite(lengthscale_after).all())
        self.assertTrue(torch.isfinite(outputscale_after).all())

    def test_gradients_accumulate_from_all_datasets(self) -> None:
        """Test that gradients from all K datasets flow to shared parameters."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        datasets = self._make_datasets(
            K=3, n_per_dataset=[10, 10, 10], d=2, tkwargs=tkwargs
        )
        mean = ConstantMean().to(**tkwargs)
        kernel = ScaleKernel(RBFKernel()).to(**tkwargs)

        model_list, mll = build_shared_gp_model_list(
            datasets=datasets,
            mean_module=mean,
            covar_module=kernel,
        )

        # Compute MLL and backward
        # Note: GPyTorch's SumMarginalLogLikelihood expects:
        # - outputs: list of MultivariateNormal distributions (one per model)
        # - targets: list of tensors (one per model)
        mll.train()
        outputs = model_list(*[d.X for d in datasets])
        targets = [d.Y.squeeze(-1) for d in datasets]
        loss = -mll(outputs, targets)
        loss.backward()

        # Verify gradients exist on shared kernel parameters
        self.assertIsNotNone(kernel.base_kernel.raw_lengthscale.grad)
        self.assertIsNotNone(kernel.raw_outputscale.grad)

        # Gradients should be finite
        self.assertTrue(torch.isfinite(kernel.base_kernel.raw_lengthscale.grad).all())
        self.assertTrue(torch.isfinite(kernel.raw_outputscale.grad).all())

    def test_freeze_pretrained_parameters(self) -> None:
        """Test that freeze_pretrained_parameters() freezes pre-trained quantities."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        datasets = self._make_datasets(
            K=3, n_per_dataset=[5, 5, 5], d=1, tkwargs=tkwargs
        )
        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(MaternKernel()).to(**tkwargs)

        # Pre-train
        em_prior = pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            num_em_iterations=3,
        )

        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)

        # Create model with freeze_pretrained=True (default)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=True,
        )

        # Check that mean_module parameters are frozen
        for name, param in model.initial_mean_module.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"initial_mean_module.{name} should be frozen",
            )

        # Check that covar_module parameters are frozen
        for name, param in model.initial_covar_module.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"initial_covar_module.{name} should be frozen",
            )

        # Check that inducing point quantities are frozen
        self.assertFalse(
            model._mu_inducing.requires_grad,
            "_mu_inducing should be frozen",
        )
        self.assertFalse(
            model._Sigma_inducing.requires_grad,
            "_Sigma_inducing should be frozen",
        )

    def test_from_pretrained_unfrozen_allows_finetuning(self) -> None:
        """Test that freeze_pretrained=False allows fine-tuning."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        datasets = self._make_datasets(
            K=3, n_per_dataset=[5, 5, 5], d=1, tkwargs=tkwargs
        )
        mean_module = ConstantMean().to(**tkwargs)
        covar_module = ScaleKernel(MaternKernel()).to(**tkwargs)

        # Pre-train
        em_prior = pretrain_em_prior(
            datasets=datasets,
            mean_module=mean_module,
            covar_module=covar_module,
            num_em_iterations=3,
        )

        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)

        # Create model with freeze_pretrained=False
        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=em_prior,
            train_X=train_X,
            train_Y=train_Y,
            freeze_pretrained=False,
        )

        # Check that mean_module parameters are trainable
        trainable_mean_params = [
            p for p in model.initial_mean_module.parameters() if p.requires_grad
        ]
        self.assertGreater(
            len(trainable_mean_params),
            0,
            "initial_mean_module should have trainable parameters when unfrozen",
        )

        # Check that covar_module parameters are trainable
        trainable_covar_params = [
            p for p in model.initial_covar_module.parameters() if p.requires_grad
        ]
        self.assertGreater(
            len(trainable_covar_params),
            0,
            "initial_covar_module should have trainable parameters when unfrozen",
        )


# =============================================================================
# Tests for KL Prior Fit and Kernel Optimization
# =============================================================================


class TestPretrainedMLL(BotorchTestCase):
    """Tests for EMEmpiricalMarginalLogLikelihood with a pretrained prior."""

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
    def test_pretrained_mll_gradient_flow(self) -> None:
        """Pretrained MLL mode: gradients flow through kernel but not EM estimates."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets, mean_module, covar_module, container, Z = self._make_em_setup(tkwargs)

        likelihood = GaussianLikelihood()
        likelihood.noise = 1e-2
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        model = EMEmpiricalGaussianProcess(
            train_X=datasets[0].X[:1].to(**tkwargs),
            train_Y=datasets[0].Y[:1].to(**tkwargs),
            em_prior=container,
            likelihood=likelihood,
        )

        mll = EMEmpiricalMarginalLogLikelihood(likelihood, model)
        model.train()
        dummy_dist = model(model.train_inputs[0])
        loss = mll(dummy_dist, model.train_targets)
        loss.backward()

        # Gradients should flow through kernel parameters
        has_grad = False
        for p in model.covar_module.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "Kernel parameters should have non-zero gradients")

        # EM estimates should be detached (no gradient)
        self.assertFalse(
            model._mu_inducing.requires_grad,
            "mu_inducing should not require gradients in pretrained mode",
        )

    def test_pretrained_mll_skips_em(self) -> None:
        """Pretrained MLL mode skips EM re-run."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets, mean_module, covar_module, container, Z = self._make_em_setup(tkwargs)

        likelihood = GaussianLikelihood()
        model = EMEmpiricalGaussianProcess(
            train_X=datasets[0].X[:1].to(**tkwargs),
            train_Y=datasets[0].Y[:1].to(**tkwargs),
            em_prior=container,
            likelihood=likelihood,
        )

        # Record the mu_inducing before MLL call
        mu_before = model._mu_inducing.clone()

        mll = EMEmpiricalMarginalLogLikelihood(likelihood, model)
        model.train()
        with torch.no_grad():
            dummy_dist = model(model.train_inputs[0])
            mll(dummy_dist, model.train_targets)

        # mu_inducing should be unchanged (EM was not re-run)
        self.assertTrue(
            torch.allclose(model._mu_inducing, mu_before),
            "mu_inducing should not change in pretrained mode (EM skipped)",
        )


class TestInducingPoints(BotorchTestCase):
    """Tests for learnable vs fixed inducing-point storage."""

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
    # Tests for learnable inducing points
    # =========================================================================

    def test_learnable_inducing_points_is_parameter(self) -> None:
        """When learnable_inducing_points=True, _X_inducing is a Parameter."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets, mean_module, covar_module, container, Z = self._make_em_setup(tkwargs)

        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=container,
            train_X=datasets[0].X[:1].to(**tkwargs),
            train_Y=datasets[0].Y[:1].to(**tkwargs),
            freeze_pretrained=False,
            learnable_inducing_points=True,
        )

        # _X_inducing is a Parameter and is trainable when learnable.
        self.assertIsInstance(model._X_inducing, torch.nn.Parameter)
        self.assertEqual(model._X_inducing.shape, Z.shape)
        self.assertTrue(model._X_inducing.requires_grad)

    def test_learnable_inducing_points_not_frozen(self) -> None:
        """Learnable inducing points stay trainable after freezing."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets, mean_module, covar_module, container, Z = self._make_em_setup(tkwargs)

        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=container,
            train_X=datasets[0].X[:1].to(**tkwargs),
            train_Y=datasets[0].Y[:1].to(**tkwargs),
            freeze_pretrained=True,
            learnable_inducing_points=True,
        )

        # Inducing points should still be learnable after freezing
        self.assertTrue(
            model._X_inducing.requires_grad,
            "Learnable inducing points should not be frozen",
        )

    def test_non_learnable_inducing_points_not_trainable(self) -> None:
        """When learnable_inducing_points=False (default), _X_inducing is frozen."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets, mean_module, covar_module, container, Z = self._make_em_setup(tkwargs)

        model = EMEmpiricalGaussianProcess.from_pretrained(
            em_prior=container,
            train_X=datasets[0].X[:1].to(**tkwargs),
            train_Y=datasets[0].Y[:1].to(**tkwargs),
        )

        # Stored as a Parameter but with gradients disabled (i.e. fixed).
        self.assertIsInstance(model._X_inducing, torch.nn.Parameter)
        self.assertFalse(model._X_inducing.requires_grad)
        self.assertEqual(model._X_inducing.shape, Z.shape)

    # =========================================================================


class TestEMCoverage(BotorchTestCase):
    """Targeted tests covering EM model edge-case branches."""

    def _datasets(
        self, tkwargs: dict, K: int = 3, n_i: int = 4
    ) -> list[ExperimentDataset]:
        torch.manual_seed(0)
        return [
            ExperimentDataset(
                X=torch.rand(n_i, 1, **tkwargs),
                Y=torch.randn(n_i, 1, **tkwargs) + 3.0,
            )
            for _ in range(K)
        ]

    def test_container_save_load(self) -> None:
        import os
        import tempfile

        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        container = pretrain_em_prior(
            datasets,
            ConstantMean().to(**tkwargs),
            ScaleKernel(MaternKernel()).to(**tkwargs),
            num_em_iterations=2,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prior.pt")
            container.save(path)
            loaded = EMPriorContainer.load(path)
        self.assertAllClose(loaded.mu_inducing, container.mu_inducing)
        self.assertAllClose(loaded.Sigma_inducing, container.Sigma_inducing)

    def test_return_iteration_history(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        history: list[tuple[torch.Tensor, torch.Tensor]] = []
        container = pretrain_em_prior(
            datasets,
            ConstantMean().to(**tkwargs),
            ScaleKernel(MaternKernel()).to(**tkwargs),
            num_em_iterations=3,
            iteration_history=history,
        )
        self.assertIsInstance(container, EMPriorContainer)
        self.assertGreaterEqual(len(history), 1)
        mu, _ = history[-1]
        self.assertAllClose(mu, container.mu_inducing)

    def test_init_modes_and_float_noise(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        mean = ConstantMean().to(**tkwargs)
        covar = ScaleKernel(MaternKernel()).to(**tkwargs)
        # "naive" init mode + float (non-Tensor) likelihood noise.
        container = pretrain_em_prior(
            datasets,
            mean,
            covar,
            num_em_iterations=2,
            init_mode="naive",
            likelihood_noise=1e-3,
        )
        self.assertIsInstance(container, EMPriorContainer)
        # Unknown init mode raises.
        with self.assertRaisesRegex(ValueError, "Unknown init_mode"):
            pretrain_em_prior(
                datasets, mean, covar, num_em_iterations=1, init_mode="bad"
            )

    def test_early_stopping(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        # A huge tolerance forces the convergence break after one iteration.
        container = pretrain_em_prior(
            datasets,
            ConstantMean().to(**tkwargs),
            ScaleKernel(MaternKernel()).to(**tkwargs),
            num_em_iterations=50,
            em_convergence_tol=1e10,
        )
        self.assertIsInstance(container, EMPriorContainer)

    def test_validation_errors(self) -> None:
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        mean = ConstantMean().to(**tkwargs)
        covar = ScaleKernel(MaternKernel()).to(**tkwargs)
        train_X = torch.rand(3, 1, **tkwargs)
        train_Y = torch.randn(3, 1, **tkwargs)
        # Non-Gaussian likelihood is rejected.
        with self.assertRaisesRegex(ValueError, "only supports GaussianLikelihood"):
            EMEmpiricalGaussianProcess(
                train_X=train_X,
                train_Y=train_Y,
                datasets=datasets,
                mean_module=mean,
                covar_module=covar,
                likelihood=FixedNoiseGaussianLikelihood(
                    noise=torch.full((3,), 1e-2, **tkwargs)
                ),
            )
        # Neither em_prior nor datasets provided.
        with self.assertRaisesRegex(ValueError, "Either em_prior or datasets"):
            EMEmpiricalGaussianProcess(train_X=train_X, train_Y=train_Y)
        # Datasets provided but mean/covar modules missing.
        with self.assertRaisesRegex(ValueError, "mean_module and covar_module"):
            EMEmpiricalGaussianProcess(
                train_X=train_X, train_Y=train_Y, datasets=datasets
            )
        # iw_nu too small for the Inverse-Wishart prior.
        with self.assertRaisesRegex(ValueError, "iw_nu must be"):
            pretrain_em_prior(
                datasets,
                mean,
                covar,
                num_em_iterations=1,
                use_covar_prior=True,
                iw_nu=1.0,
            )

    def test_warm_start_em(self) -> None:
        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        model = EMEmpiricalGaussianProcess(
            train_X=torch.rand(3, 1, **tkwargs),
            train_Y=torch.randn(3, 1, **tkwargs),
            datasets=datasets,
            mean_module=ConstantMean().to(**tkwargs),
            covar_module=ScaleKernel(MaternKernel()).to(**tkwargs),
            num_em_iterations=2,
            warm_start_em=True,
        )
        # Drive EM re-runs through the MLL; ExactGP.__call__ forbids arbitrary
        # inputs in train mode. Warm-start reuses the detached previous state.
        mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
        mll(None, None)
        out = mll(None, None)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(model._mu_inducing).all())
        self.assertTrue(torch.isfinite(model._Sigma_inducing).all())

    def test_named_priors_in_mll(self) -> None:
        from gpytorch.priors import GammaPrior

        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        covar = ScaleKernel(
            MaternKernel(lengthscale_prior=GammaPrior(3.0, 6.0)),
            outputscale_prior=GammaPrior(2.0, 0.15),
        ).to(**tkwargs)
        model = EMEmpiricalGaussianProcess(
            train_X=torch.rand(3, 1, **tkwargs),
            train_Y=torch.randn(3, 1, **tkwargs),
            datasets=datasets,
            mean_module=ConstantMean().to(**tkwargs),
            covar_module=covar,
            num_em_iterations=2,
        )
        mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
        out = mll(None, None)
        self.assertTrue(torch.isfinite(out).all())

    def test_two_dim_output_mean_branches(self) -> None:
        from gpytorch.means import Mean

        class _TwoDimOutputMean(Mean):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Return a trailing singleton dim to exercise squeeze branches."""
                return torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)

        tkwargs = {"device": self.device, "dtype": torch.double}
        datasets = self._datasets(tkwargs)
        mean = _TwoDimOutputMean().to(**tkwargs)
        covar = ScaleKernel(MaternKernel()).to(**tkwargs)
        Z = torch.linspace(0, 1, 5, **tkwargs).unsqueeze(-1)
        # Inducing points + 2D-output mean exercises the e-step/interpolation
        # `dim() > 1` squeeze branches during pre-training.
        pretrain_em_prior(datasets, mean, covar, inducing_points=Z, num_em_iterations=2)
        # Cold-start training forward exercises _get_em_initialization and
        # _update_cache squeeze branches.
        model = EMEmpiricalGaussianProcess(
            train_X=torch.rand(3, 1, **tkwargs),
            train_Y=torch.randn(3, 1, **tkwargs),
            datasets=datasets,
            mean_module=mean,
            covar_module=covar,
            inducing_points=Z,
            num_em_iterations=2,
        )
        # Drive the cold-start EM re-run + cache update + interpolation via MLL.
        mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
        out = mll(None, None)
        self.assertTrue(torch.isfinite(out).all())
