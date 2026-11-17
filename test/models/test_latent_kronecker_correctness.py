# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless

import torch
from botorch.models.latent_kronecker_gp import LatentKroneckerGP
from botorch.models.transforms.outcome import Log
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import LinearKernel, MaternKernel
from linear_operator.operators import KroneckerProductLinearOperator


class TestLatentKroneckerPosteriorCorrectness(BotorchTestCase):
    def _model(self, missing=False, standardize=False, log=False):
        X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double, device=self.device)
        T = X.new_tensor([[0.0], [0.5], [1.0]])
        Y = 10 + torch.arange(9, dtype=X.dtype, device=X.device).reshape(3, 3)
        if missing:
            Y[1, 1] = torch.nan
            Y[2, 0] = torch.nan
        kwargs = {} if standardize else {"outcome_transform": Log() if log else None}
        model = LatentKroneckerGP(X, T, Y, **kwargs).eval()
        model.likelihood.noise = 0.1
        return model, X, T, Y

    def _conditional(self, model, X, T):
        train_X, train_T = model.train_inputs
        kxx = model.covar_module_X(train_X)
        ktt = model.covar_module_T(train_T)
        train = KroneckerProductLinearOperator(kxx, ktt).to_dense()
        mask = model.mask_valid
        train = train[..., mask, :][..., :, mask]
        train = train + torch.diag_embed(
            model.likelihood.noise.expand_as(model.train_targets)
        )
        cross = KroneckerProductLinearOperator(
            model.covar_module_X(train_X, X), model.covar_module_T(train_T, T)
        ).to_dense()[..., mask, :]
        test = KroneckerProductLinearOperator(
            model.covar_module_X(X), model.covar_module_T(T)
        ).to_dense()
        mean_train, mean_test = (
            model._get_mean(train_X, train_T, mask),
            model._get_mean(X, T),
        )
        mean = mean_test + (
            cross.transpose(-1, -2)
            @ torch.linalg.solve(
                train, (model.train_targets - mean_train).unsqueeze(-1)
            )
        ).squeeze(-1)
        covariance = test - cross.transpose(-1, -2) @ torch.linalg.solve(train, cross)
        return mean, covariance

    def test_structured_moment_shape_matches_draws(self):
        model, X, T, _ = self._model()
        query = X.new_tensor([[0.2], [0.8]])
        posterior = model.posterior(query, T[:2])
        self.assertEqual(posterior.mean.shape, torch.Size([2, 2]))
        self.assertEqual(posterior.variance.shape, torch.Size([2, 2]))
        self.assertEqual(
            posterior.rsample(torch.Size([3])).shape, torch.Size([3, 2, 2])
        )

    def test_affine_outcomes_moments_and_distribution_use_original_space(self):
        model, X, T, _ = self._model(missing=True, standardize=True)
        query = X.new_tensor([[0.2], [1.5]])
        mean, covariance = self._conditional(model, query, T)
        scale, location = (
            model.outcome_transform.stdvs.squeeze(-1),
            model.outcome_transform.means.squeeze(-1),
        )
        expected_mean = mean * scale + location
        expected_covariance = covariance * scale.square().unsqueeze(-1)
        posterior = model.posterior(query, T)
        self.assertAllClose(posterior.mean.flatten(-2), expected_mean)
        self.assertAllClose(
            posterior.variance.flatten(-2),
            expected_covariance.diagonal(dim1=-2, dim2=-1),
        )
        self.assertAllClose(posterior.distribution.mean, expected_mean)
        self.assertAllClose(
            posterior.distribution.covariance_matrix, expected_covariance
        )

    def test_matheron_full_covariance_from_all_normal_basis_vectors(self):
        # Sampling is affine in the base vector. Its exact covariance is J J^T,
        # so normal basis vectors test the full law without Monte Carlo error.
        for missing in (False, True):
            for novel_times in (False, True):
                with self.subTest(missing=missing, novel_times=novel_times):
                    model, X, T, _ = self._model(missing=missing)
                    query = X.new_tensor([[2.0], [3.0]])
                    time = T[:2] if not novel_times else X.new_tensor([[0.2], [0.8]])
                    mean, covariance = self._conditional(model, query, time)
                    posterior = model.posterior(query, time)
                    total = posterior.base_sample_shape[-1]
                    zero = X.new_zeros(1, total)
                    basis = torch.eye(total, dtype=X.dtype, device=X.device)
                    center = (
                        posterior.rsample_from_base_samples(torch.Size([1]), zero)
                        .flatten(-2)
                        .squeeze(0)
                    )
                    influence = (
                        posterior.rsample_from_base_samples(
                            torch.Size([total]), basis
                        ).flatten(-2)
                        - center
                    )
                    self.assertAllClose(center, mean, atol=1e-6, rtol=1e-6)
                    self.assertAllClose(
                        influence.T @ influence, covariance, atol=2e-6, rtol=2e-6
                    )

    def test_time_only_batch_in_extended_shape(self):
        model, X, T, _ = self._model()
        query = X.new_tensor([[0.2], [0.8]])
        time = X.new_tensor([[[0.2], [0.8]], [[0.3], [0.9]]])
        posterior = model.posterior(query, time)
        self.assertEqual(posterior.output_batch_shape, torch.Size([2]))
        self.assertEqual(
            posterior._extended_shape(torch.Size([3])), torch.Size([3, 2, 2, 2])
        )
        self.assertEqual(
            posterior.rsample(torch.Size([3])).shape, torch.Size([3, 2, 2, 2])
        )

    def test_log_outcomes_have_log_normal_moments_and_one_sample_transform(self):
        model, X, T, _ = self._model(log=True, missing=True)
        query = X.new_tensor([[0.2], [1.5]])
        mean, covariance = self._conditional(model, query, T[:2])
        variance = covariance.diagonal(dim1=-2, dim2=-1)
        expected_mean = (mean + 0.5 * variance).exp()
        expected_variance = variance.expm1() * (2 * mean + variance).exp()
        posterior = model.posterior(query, T[:2])
        self.assertAllClose(posterior.mean.flatten(-2), expected_mean)
        self.assertAllClose(posterior.variance.flatten(-2), expected_variance)
        zero = X.new_zeros(1, *posterior.base_sample_shape)
        value = posterior.rsample_from_base_samples(torch.Size([1]), zero)
        self.assertAllClose(
            value.flatten(-2).squeeze(0), mean.exp(), atol=1e-6, rtol=1e-6
        )

    def test_covariance_factor_linear_operators_are_supported(self):
        model, X, T, _ = self._model(missing=True)
        model.covar_module_X = (LinearKernel() + MaternKernel()).to(X)
        query, time = X.new_tensor([[1.3], [2.0]]), X.new_tensor([[0.2], [0.8]])
        mean, covariance = self._conditional(model, query, time)
        posterior = model.posterior(query, time)
        size = posterior.base_sample_shape[-1]
        center = (
            posterior.rsample_from_base_samples(torch.Size([1]), X.new_zeros(1, size))
            .flatten(-2)
            .squeeze(0)
        )
        samples = posterior.rsample_from_base_samples(
            torch.Size([size]), torch.eye(size, device=X.device, dtype=X.dtype)
        ).flatten(-2)
        self.assertAllClose(center, mean, atol=1e-6, rtol=1e-6)
        influence = samples - center
        self.assertAllClose(influence.T @ influence, covariance, atol=2e-6, rtol=2e-6)

    def test_corrected_pathwise_query_derivative_matches_finite_difference(self):
        model, X, T, _ = self._model(missing=True)
        query = X.new_tensor([[1.3], [2.0]]).requires_grad_()
        time = X.new_tensor([[0.2], [0.8]])
        posterior = model.posterior(query, time)
        base = torch.linspace(
            -1, 1, posterior.base_sample_shape[-1], device=X.device, dtype=X.dtype
        ).unsqueeze(0)
        loss = posterior.rsample_from_base_samples(torch.Size([1]), base).square().sum()
        gradient = torch.autograd.grad(loss, query)[0]
        step = 1e-5
        values = []
        for sign in (1, -1):
            p = model.posterior(query.detach() + sign * step, time)
            values.append(
                p.rsample_from_base_samples(torch.Size([1]), base).square().sum()
            )
        self.assertAllClose(
            gradient.sum(), (values[0] - values[1]) / (2 * step), atol=3e-5, rtol=3e-5
        )


@skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestLatentKroneckerPosteriorCorrectnessCUDA(
    TestLatentKroneckerPosteriorCorrectness
):
    device = torch.device("cuda")
