# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest import skipUnless
from unittest.mock import patch

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import InputDataWarning
from botorch.models import SingleTaskGP
from botorch.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch.models.relevance_pursuit import (
    backward_relevance_pursuit,
    forward_relevance_pursuit,
    get_posterior_over_support,
)
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise
from gpytorch.mlls import ExactMarginalLogLikelihood


class TestRelevancePursuitCorrectness(BotorchTestCase):
    def _data(self, n=5):
        X = torch.linspace(0, 1, n, dtype=torch.double, device=self.device).unsqueeze(
            -1
        )
        Y = 10 + (X * 5).sin()
        return X, Y

    def _noise(self, batch=(), support=None):
        base = HomoskedasticNoise(batch_shape=torch.Size(batch)).to(
            device=self.device, dtype=torch.double
        )
        base.noise = 0.1
        return SparseOutlierNoise(
            base, dim=5, outlier_indices=support, convex_parameterization=False
        )

    def test_loo_expansion_centers_nonzero_mean(self):
        X, Y = self._data()
        likelihood = SparseOutlierGaussianLikelihood(
            HomoskedasticNoise().to(X),
            dim=5,
            outlier_indices=[2],
            convex_parameterization=False,
        )
        model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None)
        model.mean_module.constant = 10.0
        noise = model.likelihood.noise_covar
        with torch.no_grad():
            noise.raw_rho.fill_(0.2)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        marginal = likelihood(model(X), X)
        covariance, centered = marginal.covariance_matrix, Y.squeeze(-1) - marginal.mean
        expected = []
        for i in range(5):
            other = torch.arange(5, device=self.device) != i
            cross = covariance[i, other]
            remaining = covariance[other][:, other]
            error = centered[i] - cross @ torch.linalg.solve(remaining, centered[other])
            variance = covariance[i, i] - cross @ torch.linalg.solve(remaining, cross)
            expected.append((error.square() - variance - noise.rho[i]).clamp_min(0))
        expected = torch.stack(expected)[~noise.is_active]
        self.assertAllClose(noise._optimal_rhos(mll), expected)
        original = noise._optimal_rhos(mll)
        model.set_train_data(targets=model.train_targets + 20)
        model.mean_module.constant = 30.0
        self.assertAllClose(noise._optimal_rhos(mll), original)

    def test_batched_rho_gathers_last_axis_and_gradients(self):
        for batch in [(2,), (2, 3)]:
            noise = self._noise(batch, [2, 4])
            with torch.no_grad():
                noise.raw_rho.copy_(
                    torch.arange(noise.raw_rho.numel(), device=self.device).reshape_as(
                        noise.raw_rho
                    )
                    / 10
                )
            expected = noise.raw_rho.new_zeros(*batch, 5)
            expected[..., [2, 4]] = noise.raw_rho
            self.assertAllClose(noise.rho, expected)
            weights = torch.arange(expected.numel(), device=self.device).reshape_as(
                expected
            )
            (noise.rho * weights).sum().backward()
            self.assertAllClose(noise.raw_rho.grad, weights[..., [2, 4]].to(expected))

    def test_batched_sparse_dense_support_edits_preserve_values_and_trainability(self):
        for batch in [(2,), (2, 3)]:
            for frozen in (False, True):
                noise = self._noise(batch, [2, 4])
                with torch.no_grad():
                    noise.raw_rho.fill_(0.3)
                noise.raw_rho.requires_grad_(not frozen)
                expected = noise.raw_rho.new_zeros(*batch, 5)
                expected[..., [2, 4]] = 0.3
                noise.to_dense()
                self.assertAllClose(noise.raw_rho, expected)
                self.assertEqual(noise.raw_rho.requires_grad, not frozen)
                noise.to_sparse()
                self.assertEqual(noise.raw_rho.shape, (*batch, 2))
                noise.expand_support([1])
                self.assertEqual(noise.raw_rho.shape, (*batch, 3))
                self.assertEqual(noise.raw_rho.requires_grad, not frozen)
                noise.contract_support([2])
                expected[..., 2] = 0
                self.assertAllClose(noise.rho, expected)
                noise.full_support()
                self.assertAllClose(noise.raw_rho, expected)
                noise.contract_support([4])
                expected[..., 4] = 0
                self.assertAllClose(noise.raw_rho, expected)
                noise.remove_support().to_sparse()
                self.assertEqual(noise.raw_rho.shape, (*batch, 0))
                self.assertEqual(noise.raw_rho.requires_grad, not frozen)

    def test_batched_loo_and_gradient_objectives_keep_inactive_last_axis(self):
        X, Y = self._data()
        X, Y = X.expand(2, -1, -1), Y.expand(2, -1, -1).clone()
        Y[1] += 0.5
        likelihood = SparseOutlierGaussianLikelihood(
            HomoskedasticNoise(batch_shape=torch.Size([2])).to(X),
            dim=5,
            outlier_indices=[2],
            convex_parameterization=False,
        )
        model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None)
        model.mean_module.constant = 10.0
        noise = likelihood.noise_covar
        mll = ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        marginal = likelihood(model(X), X)
        inverse = torch.linalg.inv(marginal.covariance_matrix)
        diagonal = inverse.diagonal(dim1=-2, dim2=-1)
        alpha = (inverse @ (model.train_targets - marginal.mean).unsqueeze(-1)).squeeze(
            -1
        )
        expected = (
            (alpha / diagonal).square() - diagonal.reciprocal() - noise.rho
        ).clamp_min(0)
        self.assertAllClose(noise._optimal_rhos(mll), expected[..., ~noise.is_active])
        noise.to_dense()
        expected = torch.autograd.grad(
            mll(model(X), model.train_targets, X).sum(), noise.raw_rho
        )[0]
        noise.loo = False
        self.assertAllClose(
            noise.expansion_objective(mll), expected[..., ~noise.is_active]
        )

    def test_integral_tensor_support_is_preserved(self):
        for dtype in (torch.int16, torch.int64):
            indices = torch.tensor([2, 4], device=self.device, dtype=dtype)
            noise = self._noise(support=indices)
            self.assertEqual(noise.support, [2, 4])
            self.assertTrue(all(type(i) is int for i in noise.support))
            noise.contract_support(indices[:1])
            self.assertEqual(noise.support, [4])
            noise.expand_support(list(indices[:1]))
            self.assertEqual(noise.support, [4, 2])
        for indices in (
            torch.tensor([1.5], device=self.device),
            torch.tensor([True], device=self.device),
        ):
            with self.assertRaisesRegex(ValueError, "integers"):
                self._noise(support=indices)

    def test_invalid_initial_support_and_external_support_alias(self):
        for support in ([1, 1], [-1], [5], [True], [1.5]):
            with self.subTest(support=support), self.assertRaises(ValueError):
                self._noise(support=support)
        indices = [2]
        noise = self._noise(support=indices)
        indices.append(4)
        self.assertEqual(noise.support, [2])

    def test_invalid_expansion_is_atomic(self):
        for dense in (False, True):
            for indices in ([1, 1], [-1], [5], [True], [1.5], [1, 2]):
                noise = self._noise(support=[2])
                if dense:
                    noise.to_dense()
                parameter, values, support = (
                    noise.raw_rho,
                    noise.raw_rho.clone(),
                    list(noise.support),
                )
                with (
                    self.subTest(dense=dense, indices=indices),
                    self.assertRaises(ValueError),
                ):
                    noise.expand_support(indices)
                self.assertIs(noise.raw_rho, parameter)
                self.assertEqual(noise.support, support)
                self.assertAllClose(noise.raw_rho, values)

    def test_invalid_contraction_is_atomic(self):
        for dense in (False, True):
            for indices in ([2, 2], [2, 3], [-1], [5], [True], [2.0]):
                noise = self._noise(support=[2, 4])
                if dense:
                    noise.to_dense()
                parameter, values, support = (
                    noise.raw_rho,
                    noise.raw_rho.clone(),
                    list(noise.support),
                )
                with (
                    self.subTest(dense=dense, indices=indices),
                    self.assertRaises(ValueError),
                ):
                    noise.contract_support(indices)
                self.assertIs(noise.raw_rho, parameter)
                self.assertEqual(noise.support, support)
                self.assertAllClose(noise.raw_rho, values)

    def test_cached_training_inputs_are_independent(self):
        X, _ = self._data()
        noise = self._noise(support=[2])
        with torch.no_grad():
            noise.raw_rho.fill_(0.4)
        noise.train()
        robust = noise(X, shape=torch.Size([5])).diagonal()
        noise.eval()
        self.assertAllClose(noise(X, shape=torch.Size([5])).diagonal(), robust)
        X.add_(0.1)
        with self.assertWarnsRegex(InputDataWarning, "not equal"):
            observed = noise(X, shape=torch.Size([5])).diagonal()
        self.assertAllClose(observed, noise.base_noise(X).diagonal())

    def test_fixed_noise_mismatched_event_is_shaped_zero(self):
        X, Y = self._data()
        noise = SparseOutlierNoise(
            FixedGaussianNoise(torch.full_like(Y.squeeze(-1), 0.1)), dim=5
        )
        for batch in [(), (2,)]:
            Q = X[:2].expand(*batch, 2, 1)
            with self.assertWarnsRegex(InputDataWarning, "last dimension"):
                actual = noise(Q, shape=Q.shape[:-1])
            self.assertEqual(actual.shape, (*batch, 2, 2))
            self.assertAllClose(actual.to_dense(), Q.new_zeros(*batch, 2, 2))
            self.assertEqual(actual.dtype, X.dtype)
            self.assertEqual(actual.device, X.device)

    def test_fixed_noise_noisy_posterior_matches_latent_outside_training_event(self):
        X, Y = self._data()
        model = RobustRelevancePursuitSingleTaskGP(
            X, Y, train_Yvar=torch.full_like(Y, 0.1), outcome_transform=None
        )
        Q = X[:2] + 0.01
        latent = model.posterior(Q)
        with self.assertWarnsRegex(InputDataWarning, "last dimension"):
            observed = model.posterior(Q, observation_noise=True)
        self.assertAllClose(observed.mean, latent.mean)
        self.assertAllClose(
            observed.distribution.covariance_matrix,
            latent.distribution.covariance_matrix,
        )

    def test_small_sample_default_prior_gives_finite_support_posterior(self):
        for n in range(1, 5):
            X, Y = self._data(n)
            model = RobustRelevancePursuitSingleTaskGP(X, Y, outcome_transform=None)
            self.assertGreater(model.prior_mean_of_support, 0)
            sizes, probability = get_posterior_over_support(
                SparseOutlierNoise,
                [model.to_standard_model()],
                prior_mean_of_support=model.prior_mean_of_support,
            )
            self.assertAllClose(sizes, X.new_tensor([0]))
            self.assertAllClose(probability, X.new_tensor([1]))

    def test_invalid_support_prior_is_rejected(self):
        X, Y = self._data()
        for mean in [0, -1, float("nan"), float("inf")]:
            with (
                self.subTest(mean=mean),
                self.assertRaisesRegex(ValueError, "positive"),
            ):
                RobustRelevancePursuitSingleTaskGP(X, Y, prior_mean_of_support=mean)
            model = RobustRelevancePursuitSingleTaskGP(X, Y)
            with (
                self.subTest(mean=mean),
                self.assertRaisesRegex(ValueError, "positive"),
            ):
                get_posterior_over_support(
                    SparseOutlierNoise,
                    [model.to_standard_model()],
                    prior_mean_of_support=mean,
                )

    def test_failed_custom_fit_restores_model_and_likelihood(self):
        X, Y = self._data()
        model = RobustRelevancePursuitSingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        original_likelihood = mll.likelihood

        def fail(**kwargs):
            self.assertIsNot(mll.model, model)
            mll.likelihood = GaussianLikelihood().to(X)
            raise RuntimeError("injected optimizer failure")

        with self.assertRaisesRegex(RuntimeError, "injected optimizer failure"):
            model.custom_fit(
                mll, numbers_of_outliers=[0, 1], relevance_pursuit_optimizer=fail
            )
        self.assertIs(mll.model, model)
        self.assertIs(mll.likelihood, original_likelihood)
        self.assertIs(model.likelihood, original_likelihood)

    def test_failed_model_comparison_restores_mll(self):
        X, Y = self._data()
        model = RobustRelevancePursuitSingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        original_likelihood = mll.likelihood

        def completed(**kwargs):
            return kwargs["sparse_module"], [deepcopy(mll.model)]

        with patch(
            "botorch.models.robust_relevance_pursuit_model.get_posterior_over_support",
            side_effect=RuntimeError("injected comparison failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "comparison failure"):
                model.custom_fit(
                    mll, numbers_of_outliers=[0], relevance_pursuit_optimizer=completed
                )
        self.assertIs(mll.model, model)
        self.assertIs(mll.likelihood, original_likelihood)

    def test_successful_custom_fit_uses_selected_likelihood(self):
        X, Y = self._data()
        model = RobustRelevancePursuitSingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        def completed(**kwargs):
            selected = deepcopy(mll.model)
            selected.likelihood.noise_covar.expand_support([1])
            return kwargs["sparse_module"], [selected]

        model.custom_fit(
            mll, numbers_of_outliers=[1], relevance_pursuit_optimizer=completed
        )
        self.assertIs(mll.model, model)
        self.assertIs(mll.likelihood, model.likelihood)
        self.assertEqual(model.likelihood.noise_covar.support, [1])

    def test_failed_sparse_gradient_restores_representation_and_trainability(self):
        X, Y = self._data()
        for dense in (False, True):
            for frozen in (False, True):
                for stage in ("model", "mll", "backward", "none"):
                    with self.subTest(dense=dense, frozen=frozen, stage=stage):
                        likelihood = SparseOutlierGaussianLikelihood(
                            HomoskedasticNoise().to(X),
                            dim=5,
                            outlier_indices=[2],
                            convex_parameterization=False,
                        )
                        model = SingleTaskGP(
                            X, Y, likelihood=likelihood, outcome_transform=None
                        )
                        noise = likelihood.noise_covar
                        if dense:
                            noise.to_dense()
                        noise.raw_rho.requires_grad_(not frozen)
                        noise.raw_rho.grad = torch.full_like(noise.raw_rho, 0.7)
                        parameter = noise.raw_rho
                        values, gradient, support = (
                            parameter.clone(),
                            parameter.grad.clone(),
                            list(noise.support),
                        )
                        mll = ExactMarginalLogLikelihood(likelihood, model)
                        if stage == "none":
                            other = SingleTaskGP(X, Y, outcome_transform=None)
                            mll = ExactMarginalLogLikelihood(other.likelihood, other)
                            context = patch.object(noise, "_expansion_modifier", None)
                            exception, message = ValueError, "gradient.*is None"
                        elif stage == "backward":
                            context = patch.object(
                                torch.Tensor,
                                "backward",
                                side_effect=RuntimeError("injected backward failure"),
                            )
                            exception, message = RuntimeError, "injected"
                        else:
                            target = model if stage == "model" else mll
                            context = patch.object(
                                target,
                                "forward",
                                side_effect=RuntimeError("injected evaluation failure"),
                            )
                            exception, message = RuntimeError, "injected"
                        with context, self.assertRaisesRegex(exception, message):
                            noise._sparse_parameter_gradient(mll)
                        self.assertIs(noise.raw_rho, parameter)
                        self.assertEqual(noise.is_sparse, not dense)
                        self.assertEqual(noise.support, support)
                        self.assertEqual(noise.raw_rho.requires_grad, not frozen)
                        self.assertAllClose(noise.raw_rho, values)
                        self.assertAllClose(noise.raw_rho.grad, gradient)

    def test_batched_discrete_ranking_rejects_without_mutation(self):
        noise = self._noise(batch=(2,), support=[2])
        parameter, support = noise.raw_rho, list(noise.support)
        with patch.object(
            noise,
            "expansion_objective",
            return_value=torch.ones(2, 4, device=self.device),
        ):
            with self.assertRaisesRegex(UnsupportedError, "batch"):
                noise.support_expansion(None)
        self.assertIs(noise.raw_rho, parameter)
        self.assertEqual(noise.support, support)
        with self.assertRaisesRegex(UnsupportedError, "batch"):
            noise.support_contraction(None)
        self.assertIs(noise.raw_rho, parameter)
        self.assertEqual(noise.support, support)

        for pursuit in (forward_relevance_pursuit, backward_relevance_pursuit):
            with self.assertRaisesRegex(UnsupportedError, "batch"):
                pursuit(noise, None)
            self.assertIs(noise.raw_rho, parameter)
            self.assertEqual(noise.support, support)


@skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestRelevancePursuitCorrectnessCUDA(TestRelevancePursuitCorrectness):
    device = torch.device("cuda")
