#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Dual observation-space Preferential Gaussian Process supporting strict pairwise
preferences, indifference ties, and dual-rejection ("neither fits") human feedback.

By solving the Newton-Raphson Laplace approximation in the ``m``-dimensional
dual observation space ``d = M f`` (Rasmussen & Williams, 2006, Alg. 3.1/3.2)
rather than ``n``-dimensional item space, inference scales as ``O(m^3)`` when
``m << n`` — enabling low-latency preferential Bayesian optimization over large
pre-embedded discrete candidate catalogs alongside semiparametric explicit
multi-objective linear kernels.

References:

.. [Chu2005preference]
    W. Chu and Z. Ghahramani. Preference learning with Gaussian processes.
    Proceedings of the 22nd International Conference on Machine Learning, 2005.

.. [Rasmussen2006gp]
    C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine
    Learning. MIT Press, 2006.

Contributor: M-Colley
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.utils.probability.utils import (
    log_ndtr,
    standard_normal_log_hazard,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from linear_operator.operators import RootLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

SQRT_2 = math.sqrt(2.0)


class DualPairwiseGP(Model):
    r"""Dual-space Laplace Preferential GP with tie and dual-rejection feedback.

    Models latent utility ``u(x)`` across ``n`` candidate points with optional
    semiparametric explicit objective terms ``w^T y(x)`` and supports three
    human-in-the-loop feedback modalities:
    1. ``comparisons``: strict pairwise preferences ``u(x_i) > u(x_j)``.
    2. ``ties``: indifference / equality observations ``u(x_i) ~= u(x_j)``.
    3. ``rejected_pairs``: joint rejection where both ``u(x_i)`` and ``u(x_j)``
       fall below ``reject_threshold``.
    """

    def __init__(
        self,
        datapoints: Tensor,
        comparisons: Tensor | None = None,
        *,
        ties: Tensor | None = None,
        rejected_pairs: Tensor | None = None,
        explicit_objectives: Tensor | None = None,
        objective_prior_mean: Tensor | None = None,
        objective_prior_var: float = 0.35,
        covar_module: ScaleKernel | None = None,
        likelihood_type: str = "probit",
        tie_noise_variance: float = 0.15,
        reject_threshold: float = -0.75,
        jitter: float = 1e-6,
        max_newton_iter: int = 25,
        newton_tol: float = 1e-9,
    ) -> None:
        r"""Initialize a ``DualPairwiseGP`` preference model.

        Args:
            datapoints: A ``n x d`` tensor of candidate features (e.g. subspace-reduced
                multimodal embeddings or design vectors).
            comparisons: Optional ``m_pref x 2`` integer tensor where row ``[i, j]``
                indicates ``datapoints[i]`` is preferred over ``datapoints[j]``.
            ties: Optional ``m_tie x 2`` integer tensor of tied/indifferent pairs.
            rejected_pairs: Optional ``m_rej x 2`` integer tensor of jointly rejected
                candidate pairs ("neither option fits").
            explicit_objectives: Optional ``n x k`` tensor of known competing
                attributes incorporated via a linear kernel ``y Sigma_w y^T``.
            objective_prior_mean: Optional ``k``-dim prior mean weight vector on
                ``explicit_objectives``. Defaults to zeros.
            objective_prior_var: Scalar prior variance for explicit objective weights.
            covar_module: Optional GPyTorch ``ScaleKernel`` over ``datapoints``.
                Defaults to ``ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))``.
            likelihood_type: ``"probit"`` or ``"logit"``.
            tie_noise_variance: Gaussian observation variance on ``u(x_i) - u(x_j)``
                for ``ties``.
            reject_threshold: Soft upper-bound utility threshold applied to each
                candidate in ``rejected_pairs``.
            jitter: Diagonal jitter for numerical Cholesky stability.
            max_newton_iter: Maximum damped Newton-Raphson iterations in dual space.
            newton_tol: Infinity-norm optimality tolerance on dual gradient residual.
        """
        super().__init__()
        if datapoints.ndim != 2 or datapoints.shape[0] < 2:
            raise UnsupportedError(
                "`DualPairwiseGP` expects `datapoints` of shape `(n, d)` with `n >= 2`."
            )
        if likelihood_type not in ("probit", "logit"):
            raise UnsupportedError(
                f"Unsupported `likelihood_type={likelihood_type!r}`; "
                "expected `'probit'` or `'logit'`."
            )
        if tie_noise_variance <= 0.0:
            raise UnsupportedError("`tie_noise_variance` must be strictly positive.")

        self.register_buffer("datapoints", datapoints)
        self._num_outputs = 1
        self.likelihood_type = likelihood_type
        self.tie_noise_variance = tie_noise_variance
        self.reject_threshold = reject_threshold
        self._jitter = jitter
        self._max_newton_iter = max_newton_iter
        self._newton_tol = newton_tol
        self._objective_prior_var = objective_prior_var

        n, d = datapoints.shape
        tkwargs = {"dtype": datapoints.dtype, "device": datapoints.device}

        if explicit_objectives is not None:
            if explicit_objectives.ndim != 2 or explicit_objectives.shape[0] != n:
                raise UnsupportedError(
                    "`explicit_objectives` must have shape `(n, k)` matching "
                    "`datapoints`."
                )
            self.register_buffer(
                "explicit_objectives", explicit_objectives.to(**tkwargs)
            )
            k_obj = explicit_objectives.shape[1]
            if objective_prior_mean is None:
                prior_w = torch.zeros(k_obj, **tkwargs)
            else:
                prior_w = objective_prior_mean.to(**tkwargs).reshape(k_obj)
            self.register_buffer("objective_prior_mean", prior_w)
        else:
            self.explicit_objectives = None
            self.objective_prior_mean = None

        if covar_module is None:
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=d, **tkwargs)
            base_kernel.initialize(lengthscale=0.35)
            covar_module = ScaleKernel(base_kernel, **tkwargs)
            covar_module.initialize(outputscale=1.5)
        self.covar_module = covar_module.to(**tkwargs)

        self.register_buffer("_incidence_M", None)
        self.register_buffer("_row_offsets", None)
        self.register_buffer("_is_tie_row", None)
        self.register_buffer("_alpha", None)
        self.register_buffer("_sqrt_w", None)
        self.register_buffer("_lb_chol", None)
        self.register_buffer("_prior_mean_train", None)
        self.register_buffer("_k_train", None)

        self.set_preferences(
            comparisons=comparisons,
            ties=ties,
            rejected_pairs=rejected_pairs,
        )

    @property
    def num_outputs(self) -> int:
        r"""Number of model outputs (1 scalar utility)."""
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model."""
        return torch.Size()

    @property
    def inferred_objective_weights(self) -> Tensor | None:
        r"""Posterior MAP estimate of linear weights on ``explicit_objectives``."""
        if self.explicit_objectives is None:
            return None
        if self._alpha is None or self._alpha.numel() == 0:
            return self.objective_prior_mean.clone()
        mt_alpha = self._incidence_M.transpose(-1, -2) @ self._alpha
        delta_w = self._objective_prior_var * (
            self.explicit_objectives.transpose(-1, -2) @ mt_alpha
        )
        return self.objective_prior_mean + delta_w

    def set_preferences(
        self,
        comparisons: Tensor | None = None,
        ties: Tensor | None = None,
        rejected_pairs: Tensor | None = None,
    ) -> None:
        r"""Update human preference observations and solve the dual Laplace posterior.

        Args:
            comparisons: Optional ``m_pref x 2`` strict preference index pairs.
            ties: Optional ``m_tie x 2`` indifferent/tied index pairs.
            rejected_pairs: Optional ``m_rej x 2`` jointly rejected index pairs.
        """
        n = self.datapoints.shape[0]
        tkwargs = {"dtype": self.datapoints.dtype, "device": self.datapoints.device}

        rows: list[Tensor] = []
        offsets: list[float] = []
        tie_flags: list[bool] = []

        def _validate_pairs(pairs: Tensor, label: str) -> Tensor:
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise UnsupportedError(f"`{label}` must have shape `(m, 2)`.")
            if pairs.numel() > 0 and (pairs.min() < 0 or pairs.max() >= n):
                raise UnsupportedError(
                    f"`{label}` indices out of bounds for {n} datapoints."
                )
            return pairs.long()

        if comparisons is not None and comparisons.numel() > 0:
            comp = _validate_pairs(comparisons, "comparisons")
            if torch.any(comp[:, 0] == comp[:, 1]):
                raise UnsupportedError(
                    "`comparisons` cannot strictly prefer an item over itself."
                )
            for r_idx in range(comp.shape[0]):
                row = torch.zeros(n, **tkwargs)
                row[comp[r_idx, 0]] = 1.0
                row[comp[r_idx, 1]] = -1.0
                rows.append(row)
                offsets.append(0.0)
                tie_flags.append(False)

        if ties is not None and ties.numel() > 0:
            tie_t = _validate_pairs(ties, "ties")
            for r_idx in range(tie_t.shape[0]):
                i_idx, j_idx = int(tie_t[r_idx, 0]), int(tie_t[r_idx, 1])
                if i_idx == j_idx:
                    continue
                row = torch.zeros(n, **tkwargs)
                row[i_idx] = 1.0
                row[j_idx] = -1.0
                rows.append(row)
                offsets.append(0.0)
                tie_flags.append(True)

        if rejected_pairs is not None and rejected_pairs.numel() > 0:
            rej_t = _validate_pairs(rejected_pairs, "rejected_pairs")
            for r_idx in range(rej_t.shape[0]):
                i_idx, j_idx = int(rej_t[r_idx, 0]), int(rej_t[r_idx, 1])
                row_i = torch.zeros(n, **tkwargs)
                row_i[i_idx] = -1.0
                rows.append(row_i)
                offsets.append(self.reject_threshold)
                tie_flags.append(False)
                if j_idx != i_idx:
                    row_j = torch.zeros(n, **tkwargs)
                    row_j[j_idx] = -1.0
                    rows.append(row_j)
                    offsets.append(self.reject_threshold)
                    tie_flags.append(False)

        if rows:
            self._incidence_M = torch.stack(rows, dim=0)
            self._row_offsets = torch.tensor(offsets, **tkwargs)
            self._is_tie_row = torch.tensor(
                tie_flags, dtype=torch.bool, device=self.datapoints.device
            )
        else:
            self._incidence_M = torch.zeros((0, n), **tkwargs)
            self._row_offsets = torch.zeros(0, **tkwargs)
            self._is_tie_row = torch.zeros(
                0, dtype=torch.bool, device=self.datapoints.device
            )

        self._fit_dual_laplace()

    def _compute_train_kernel_and_prior(self) -> tuple[Tensor, Tensor]:
        n = self.datapoints.shape[0]
        tkwargs = {"dtype": self.datapoints.dtype, "device": self.datapoints.device}
        k_vis = self.covar_module(self.datapoints, self.datapoints).to_dense()
        if self.explicit_objectives is not None:
            k_obj = self._objective_prior_var * (
                self.explicit_objectives @ self.explicit_objectives.transpose(-1, -2)
            )
            k_tot = k_vis + k_obj
            mu0 = self.explicit_objectives @ self.objective_prior_mean
        else:
            k_tot = k_vis
            mu0 = torch.zeros(n, **tkwargs)
        k_tot = k_tot + self._jitter * torch.eye(n, **tkwargs)
        return k_tot, mu0

    def _eval_dual_likelihood(self, d_vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Evaluate log-likelihood, gradient, and negative Hessian wrt ``d``."""
        shifted = d_vec + self._row_offsets
        log_p = torch.zeros_like(d_vec)
        grad = torch.zeros_like(d_vec)
        neg_hess = torch.zeros_like(d_vec)

        ineq_mask = ~self._is_tie_row
        if torch.any(ineq_mask):
            s_ineq = shifted[ineq_mask]
            if self.likelihood_type == "probit":
                z = s_ineq / SQRT_2
                lp = log_ndtr(z)
                hazard = standard_normal_log_hazard(-z).exp()
                g = hazard / SQRT_2
                w = (0.5 * hazard * (hazard + z)).clamp(min=1e-9)
            else:
                lp = torch.nn.functional.logsigmoid(s_ineq)
                p = torch.sigmoid(s_ineq)
                g = 1.0 - p
                w = (p * (1.0 - p)).clamp(min=1e-9)
            log_p[ineq_mask] = lp
            grad[ineq_mask] = g
            neg_hess[ineq_mask] = w

        if torch.any(self._is_tie_row):
            d_tie = d_vec[self._is_tie_row]
            var = self.tie_noise_variance
            log_p[self._is_tie_row] = -0.5 * (d_tie * d_tie) / var - 0.5 * math.log(
                2.0 * math.pi * var
            )
            grad[self._is_tie_row] = -d_tie / var
            neg_hess[self._is_tie_row] = 1.0 / var

        return log_p, grad, neg_hess

    def _fit_dual_laplace(self) -> None:
        with torch.no_grad():
            k_train, mu0 = self._compute_train_kernel_and_prior()
            self._k_train = k_train
            self._prior_mean_train = mu0

            m = self._incidence_M.shape[0]
            tkwargs = {"dtype": self.datapoints.dtype, "device": self.datapoints.device}
            if m == 0:
                self._alpha = torch.zeros(0, **tkwargs)
                self._sqrt_w = torch.zeros(0, **tkwargs)
                self._lb_chol = torch.zeros((0, 0), **tkwargs)
                return

            M = self._incidence_M
            mu_d = M @ mu0
            kd = M @ k_train @ M.transpose(-1, -2) + 1e-8 * torch.eye(m, **tkwargs)

            alpha = torch.zeros(m, **tkwargs)
            d_vec = mu_d.clone()
            log_p, grad, w_diag = self._eval_dual_likelihood(d_vec)
            psi = log_p.sum() - 0.5 * torch.dot(alpha, d_vec - mu_d)
            eye_m = torch.eye(m, **tkwargs)

            for _ in range(self._max_newton_iter):
                if torch.max(torch.abs(grad - alpha)).item() < self._newton_tol:
                    break
                sqrt_w = torch.sqrt(w_diag)
                b_mat = eye_m + (sqrt_w.unsqueeze(-1) * kd) * sqrt_w.unsqueeze(-2)
                lb = psd_safe_cholesky(b_mat, jitter=1e-9)

                b_rhs = w_diag * (d_vec - mu_d) + grad
                wkdb = sqrt_w * (kd @ b_rhs)
                solved = torch.cholesky_solve(wkdb.unsqueeze(-1), lb).squeeze(-1)
                alpha_target = b_rhs - sqrt_w * solved

                step_size = 1.0
                accepted = False
                for _ls in range(8):
                    alpha_try = (1.0 - step_size) * alpha + step_size * alpha_target
                    d_try = mu_d + kd @ alpha_try
                    lp_try, g_try, w_try = self._eval_dual_likelihood(d_try)
                    psi_try = lp_try.sum() - 0.5 * torch.dot(alpha_try, d_try - mu_d)
                    if psi_try >= psi - 1e-11:
                        alpha = alpha_try
                        d_vec = d_try
                        grad = g_try
                        w_diag = w_try
                        psi = psi_try
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted:
                    break

            _, _, w_final = self._eval_dual_likelihood(d_vec)
            sqrt_w_final = torch.sqrt(w_final)
            b_final = eye_m + (
                sqrt_w_final.unsqueeze(-1) * kd
            ) * sqrt_w_final.unsqueeze(-2)
            self._alpha = alpha
            self._sqrt_w = sqrt_w_final
            self._lb_chol = psd_safe_cholesky(b_final, jitter=1e-9)

    def _cross_cov_and_prior(
        self, X: Tensor, explicit_objectives_X: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute prior mean, cross-covariance ``K(X, X_train)``, and ``K(X, X)``."""
        tkwargs = {"dtype": self.datapoints.dtype, "device": self.datapoints.device}
        X_t = X.to(**tkwargs)
        batch_shape = X_t.shape[:-2]
        q = X_t.shape[-2]

        train_exp = self.datapoints.expand(*batch_shape, *self.datapoints.shape)
        k_x_train = self.covar_module(X_t, train_exp).to_dense()
        k_xx = self.covar_module(X_t, X_t).to_dense()

        if self.explicit_objectives is not None and explicit_objectives_X is not None:
            y_new = explicit_objectives_X.to(**tkwargs)
            y_train = self.explicit_objectives.expand(
                *batch_shape, *self.explicit_objectives.shape
            )
            k_x_train = k_x_train + self._objective_prior_var * (
                y_new @ y_train.transpose(-1, -2)
            )
            k_xx = k_xx + self._objective_prior_var * (y_new @ y_new.transpose(-1, -2))
            mu0_x = y_new @ self.objective_prior_mean
        else:
            mu0_x = torch.zeros(*batch_shape, q, **tkwargs)

        eye_q = torch.eye(q, **tkwargs).expand(*batch_shape, q, q)
        k_xx = k_xx + self._jitter * eye_q
        return mu0_x, k_x_train, k_xx

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
        explicit_objectives_X: Tensor | None = None,
    ) -> Posterior:
        r"""Compute joint Gaussian posterior over query points ``X``.

        Args:
            X: A ``(batch_shape x) q x d`` tensor of query points.
            output_indices: Unused (single-output latent utility model).
            observation_noise: Ignored (scale fixed by probit/logit link).
            posterior_transform: Optional ``PosteriorTransform``.
            explicit_objectives_X: Optional ``(batch_shape x) q x k`` explicit
                objective values corresponding to ``X`` when ``explicit_objectives``
                was supplied at initialization. If ``X`` equals ``self.datapoints``
                and ``explicit_objectives_X`` is ``None``, ``self.explicit_objectives``
                is reused automatically.

        Returns:
            A ``GPyTorchPosterior`` over the ``q`` query points.
        """
        if output_indices is not None:
            raise UnsupportedError(
                "`output_indices` is not supported for DualPairwiseGP."
            )

        squeeze_batch = False
        if X.ndim == 2:
            X_in = X.unsqueeze(0)
            obj_in = (
                explicit_objectives_X.unsqueeze(0)
                if explicit_objectives_X is not None and explicit_objectives_X.ndim == 2
                else explicit_objectives_X
            )
            squeeze_batch = True
        else:
            X_in = X
            obj_in = explicit_objectives_X

        if (
            obj_in is None
            and self.explicit_objectives is not None
            and X_in.shape[-2:] == self.datapoints.shape
            and torch.allclose(
                X_in.to(self.datapoints),
                self.datapoints.expand_as(X_in),
                atol=1e-7,
            )
        ):
            obj_in = self.explicit_objectives.expand(
                *X_in.shape[:-2], *self.explicit_objectives.shape
            )

        mu0_x, k_x_train, k_xx = self._cross_cov_and_prior(X_in, obj_in)

        if self._alpha is None or self._alpha.numel() == 0:
            mean = mu0_x
            cov = k_xx
        else:
            mt = self._incidence_M.transpose(-1, -2)
            k_star_mt = k_x_train @ mt  # (..., q, m)
            mean = mu0_x + k_star_mt @ self._alpha

            rhs = self._sqrt_w.unsqueeze(-1) * k_star_mt.transpose(
                -1, -2
            )  # (..., m, q)
            lb_exp = self._lb_chol.expand(*rhs.shape[:-2], *self._lb_chol.shape)
            v_mat = torch.linalg.solve_triangular(lb_exp, rhs, upper=False)
            cov = k_xx - v_mat.transpose(-1, -2) @ v_mat

        if squeeze_batch:
            mean = mean.squeeze(0)
            cov = cov.squeeze(0)

        chol = psd_safe_cholesky(cov, jitter=self._jitter)
        mvn = MultivariateNormal(mean=mean, covariance_matrix=RootLinearOperator(chol))
        post = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior=post, X=X)
        return post
