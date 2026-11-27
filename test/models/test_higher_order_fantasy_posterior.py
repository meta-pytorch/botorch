# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import skipUnless

import torch
from botorch.models import HigherOrderGP
from botorch.models.transforms.input import Normalize
from botorch.utils.testing import BotorchTestCase
from gpytorch.settings import skip_posterior_variances
from linear_operator.operators import KroneckerProductLinearOperator


class TestHigherOrderFantasyPosterior(BotorchTestCase):
    def _fantasy(self, batch=False, standardize=False, normalize=False):
        X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=self.device)
        Y = (
            (X * 5 + torch.arange(4, dtype=X.dtype, device=X.device))
            .sin()
            .reshape(3, 2, 2)
        )
        new_X, Q = X.new_tensor([[0.6]]), X.new_tensor([[0.25], [0.75]])
        if batch:
            X, Y = X.expand(2, -1, -1).clone(), Y.expand(2, -1, -1, -1).clone()
            Y[1] = Y[1] * 1.2 + 0.1
            new_X = new_X.expand(2, -1, -1).clone()
        transform = None
        if normalize:
            X, new_X, Q = [value * 3 + 2 for value in (X, new_X, Q)]
            transform = Normalize(1, bounds=X.new_tensor([[2.0], [5.0]]))
        kwargs = {} if standardize else {"outcome_transform": None}
        model = HigherOrderGP(X, Y, input_transform=transform, **kwargs)
        posterior = model.posterior(new_X)
        values = (
            posterior.mean.detach()
            .unsqueeze(0)
            .expand(3, *posterior.mean.shape)
            .clone()
        )
        delta = torch.arange(3, device=X.device, dtype=X.dtype).reshape(
            3, *([1] * (values.ndim - 1))
        )
        values = values + 0.2 * delta
        if standardize:
            values = model.outcome_transform(values)[0]
        fantasy = model.get_fantasy_model(model.transform_inputs(new_X), values)
        return fantasy, Q

    def _law(self, model, Q):
        X = model.train_inputs[0]
        Q = model.transform_inputs(Q)
        latent = [
            cm(coordinates)
            for cm, coordinates in zip(model.covar_modules[1:], model.latent_parameters)
        ]
        train = KroneckerProductLinearOperator(
            model.covar_modules[0](X), *latent
        ).to_dense()
        cross = KroneckerProductLinearOperator(
            model.covar_modules[0](X, Q), *latent
        ).to_dense()
        test = KroneckerProductLinearOperator(
            model.covar_modules[0](Q), *latent
        ).to_dense()
        train = train + torch.diag_embed(
            model.likelihood.noise.expand(*train.shape[:-2], train.shape[-1])
        )
        mean = (
            cross.transpose(-1, -2)
            @ torch.linalg.solve(train, model.train_targets.unsqueeze(-1))
        ).squeeze(-1)
        covariance = test - cross.transpose(-1, -2) @ torch.linalg.solve(train, cross)
        if hasattr(model, "outcome_transform"):
            mean = mean * model.outcome_transform.stdvs.squeeze(
                -1
            ) + model.outcome_transform.means.squeeze(-1)
            covariance = covariance * model.outcome_transform.stdvs.square()
        return mean, covariance

    def test_generic_fantasy_cache_posterior_matches_dense_law(self):
        for standardize in (False, True):
            for normalize in (False, True):
                with self.subTest(standardize=standardize, normalize=normalize):
                    model, Q = self._fantasy(
                        standardize=standardize, normalize=normalize
                    )
                    cache = model.prediction_strategy.lik_train_train_covar
                    self.assertFalse(hasattr(cache, "linear_op"))
                    mean, covariance = self._law(model, Q)
                    posterior = model.posterior(Q)
                    self.assertEqual(posterior.mean.shape, torch.Size([3, 2, 2, 2]))
                    self.assertAllClose(
                        posterior.mean.flatten(-3), mean, atol=2e-6, rtol=2e-6
                    )
                    self.assertAllClose(
                        posterior.variance.flatten(-3),
                        covariance.diagonal(dim1=-2, dim2=-1),
                        atol=2e-6,
                        rtol=2e-6,
                    )

    def test_model_fantasy_and_query_batches_broadcast(self):
        model, Q = self._fantasy(batch=True, standardize=True, normalize=True)
        Q = Q.expand(4, 1, 1, -1, -1).clone()
        mean, covariance = self._law(model, Q)
        posterior = model.posterior(Q)
        self.assertEqual(posterior.mean.shape, torch.Size([4, 3, 2, 2, 2, 2]))
        self.assertAllClose(posterior.mean.flatten(-3), mean, atol=2e-6, rtol=2e-6)
        self.assertAllClose(
            posterior.variance.flatten(-3),
            covariance.diagonal(dim1=-2, dim2=-1),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_matheron_noise_broadcast_matches_exact_fantasy_covariance(self):
        model, Q = self._fantasy(batch=True)
        mean, covariance = self._law(model, Q)
        posterior = model.posterior(Q)
        size = posterior.base_sample_shape[-1]
        batch = posterior.base_sample_shape[:-1]
        normal = (
            torch.eye(size, device=Q.device, dtype=Q.dtype)
            .reshape(size, *([1] * len(batch)), size)
            .expand(size, *batch, size)
        )
        zero = Q.new_zeros(1, *batch, size)
        center = (
            posterior.rsample_from_base_samples(torch.Size([1]), zero)
            .flatten(-3)
            .squeeze(0)
        )
        value = (
            posterior.rsample_from_base_samples(torch.Size([size]), normal).flatten(-3)
            - center
        )
        actual = torch.einsum("s...i,s...j->...ij", value, value)
        self.assertAllClose(center, mean, atol=2e-6, rtol=2e-6)
        self.assertAllClose(actual, covariance, atol=2e-5, rtol=2e-5)

    def test_empty_and_multiaxis_sample_shapes_match_flattened_normals(self):
        model, Q = self._fantasy(batch=True)
        posterior = model.posterior(Q)
        for shape in (torch.Size(), torch.Size([2, 3])):
            with self.subTest(sample_shape=shape):
                normal = torch.randn(
                    shape + posterior.base_sample_shape, dtype=Q.dtype, device=Q.device
                )
                expected = posterior.rsample_from_base_samples(
                    torch.Size([shape.numel()]),
                    normal.reshape(shape.numel(), *posterior.base_sample_shape),
                )
                expected = expected.reshape(shape + posterior._extended_shape())
                actual = posterior.rsample_from_base_samples(shape, normal)
                self.assertEqual(actual.shape, expected.shape)
                self.assertAllClose(actual, expected)
                self.assertEqual(posterior.rsample(shape).shape, expected.shape)

    def test_skip_variance_keeps_fantasy_output_batch_dtype_and_device(self):
        model, Q = self._fantasy(batch=True)
        with skip_posterior_variances(True):
            posterior = model.posterior(Q)
        self.assertEqual(posterior.mean.shape, torch.Size([3, 2, 2, 2, 2]))
        self.assertEqual(posterior.variance.shape, posterior.mean.shape)
        self.assertEqual(posterior.variance.dtype, Q.dtype)
        self.assertEqual(posterior.variance.device, Q.device)

    def test_fantasy_query_gradient_matches_dense_conditional(self):
        model, Q = self._fantasy(batch=True)
        Q.requires_grad_()
        mean, covariance = self._law(model, Q)
        expected = torch.autograd.grad(
            mean.square().sum() + covariance.diagonal(dim1=-2, dim2=-1).sum(), Q
        )[0]
        posterior = model.posterior(Q)
        actual = torch.autograd.grad(
            posterior.mean.square().sum() + posterior.variance.sum(), Q
        )[0]
        self.assertAllClose(actual, expected, atol=3e-5, rtol=3e-5)


@skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class TestHigherOrderFantasyPosteriorCUDA(TestHigherOrderFantasyPosterior):
    device = torch.device("cuda")
