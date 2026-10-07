#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.gitbo import gitbo_step, quantile_ucb
from botorch_community.models.tabpfn_v2 import TabPFNv2Model
from torch import nn, Tensor


class DummyBarDistribution(nn.Module):
    def __init__(self, n_buckets: int = 64):
        """A stub bar distribution exposing standardized-space borders.

        Args:
            n_buckets: Number of buckets for the output distribution.
        """
        super().__init__()
        self.register_buffer("borders", torch.linspace(-3.0, 3.0, n_buckets + 1))


class DummyTabPFNv2(nn.Module):
    def __init__(self, n_buckets: int = 64):
        """A dummy TabPFN v2 with differentiable logits over the test block.

        Mimics the TabPFN v2 interface: sequence-first ``x`` containing the
        training block followed by the test block, with the number of
        training points inferred from the length of ``y``.

        Args:
            n_buckets: Number of buckets for the output distribution.
        """
        super().__init__()
        self.n_buckets = n_buckets

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        only_return_standard_out: bool = True,
        **kwargs,
    ) -> Tensor:
        n_train = y.shape[0]
        test_x = x[n_train:]  # (q, b, d)
        bucket_scores = torch.linspace(
            -1.0, 1.0, self.n_buckets, dtype=test_x.dtype, device=test_x.device
        )
        return torch.sin(test_x * 3.0).sum(dim=-1, keepdim=True) * bucket_scores


def _get_model(
    n: int = 8, d: int = 4, y_mean: float = 100.0, y_std: float = 20.0, **tkwargs
) -> tuple[TabPFNv2Model, Tensor, Tensor]:
    train_X = torch.rand(n, d, **tkwargs)
    train_Y = y_mean + y_std * torch.randn(n, 1, **tkwargs)
    model = TabPFNv2Model(
        train_X,
        train_Y,
        model=DummyTabPFNv2(),
        bar_distribution=DummyBarDistribution(),
    )
    return model, train_X, train_Y


class TestTabPFNv2Model(BotorchTestCase):
    def test_raises_without_bar_distribution(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = torch.rand(5, 1, device=self.device)
        with self.assertRaisesRegex(ValueError, "bar_distribution"):
            TabPFNv2Model(train_X, train_Y, model=DummyTabPFNv2())

    def test_downloads_model_when_not_provided(self):
        train_X = torch.rand(5, 3, device=self.device)
        train_Y = torch.rand(5, 1, device=self.device)
        with patch(
            "botorch_community.models.tabpfn_v2.download_tabpfn_v2_regressor",
            return_value=(DummyTabPFNv2(), DummyBarDistribution()),
        ) as mock_download:
            model = TabPFNv2Model(train_X, train_Y, accept_license=True)
        mock_download.assert_called_once_with(accept_license=True)
        with torch.no_grad():
            posterior = model.posterior(torch.rand(4, 3, device=self.device))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_posterior_in_raw_units(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        model, train_X, train_Y = _get_model(**tkwargs)
        # borders are mapped from standardized to raw units
        expected_borders = (
            torch.linspace(-3.0, 3.0, 65, **tkwargs) * train_Y.std() + train_Y.mean()
        )
        self.assertAllClose(model.borders, expected_borders)
        with torch.no_grad():
            posterior = model.posterior(torch.rand(6, 4, **tkwargs))
        mean = posterior.mean.reshape(-1)
        # raw-unit posterior: means live inside the raw-space borders, far
        # from the standardized [-3, 3] range
        self.assertTrue((mean >= expected_borders.min()).all())
        self.assertTrue((mean <= expected_borders.max()).all())
        self.assertGreater(mean.abs().min().item(), 10.0)
        # icdf at a high quantile exceeds the mean
        ucb = posterior.icdf(0.975).reshape(-1)
        self.assertTrue((ucb > mean).all())

    def test_gradients_and_batch_layouts(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        model, _, _ = _get_model(**tkwargs)
        X = torch.rand(7, 4, **tkwargs)
        scores, grads = quantile_ucb(model, X, eval_in_q_batch=True)
        self.assertEqual(scores.shape, torch.Size([7]))
        self.assertEqual(grads.shape, torch.Size([7, 4]))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((grads.abs().sum(dim=-1) > 0).all())
        scores_b, grads_b = quantile_ucb(model, X, eval_in_q_batch=False)
        self.assertAllClose(scores, scores_b, atol=1e-4)
        self.assertAllClose(grads, grads_b, atol=1e-4)

    def test_constant_train_y(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        train_X = torch.rand(5, 3, **tkwargs)
        train_Y = torch.full((5, 1), 7.0, **tkwargs)
        model = TabPFNv2Model(
            train_X,
            train_Y,
            model=DummyTabPFNv2(),
            bar_distribution=DummyBarDistribution(),
        )
        with torch.no_grad():
            posterior = model.posterior(torch.rand(4, 3, **tkwargs))
        self.assertTrue(torch.isfinite(posterior.mean).all())

    def test_two_iteration_gitbo_loop(self):
        tkwargs = {"device": self.device, "dtype": torch.float}
        d = 5
        bounds = torch.stack([torch.zeros(d, **tkwargs), torch.ones(d, **tkwargs)])
        train_X = torch.rand(6, d, **tkwargs)
        train_Y = 50.0 - 100.0 * ((train_X[:, :2] - 0.5) ** 2).sum(-1, keepdim=True)
        network = DummyTabPFNv2()  # loaded once, reused across iterations
        bardist = DummyBarDistribution()
        gradients = None
        for iteration in range(2):
            model = TabPFNv2Model(
                train_X, train_Y, model=network, bar_distribution=bardist
            )
            result = gitbo_step(
                model,
                train_X,
                gradients,
                bounds,
                num_candidates=32,
                rank=2,
                eval_in_q_batch=True,
            )
            train_X = torch.cat([train_X, result.candidate])
            new_Y = 50.0 - 100.0 * ((result.candidate[:, :2] - 0.5) ** 2).sum(
                -1, keepdim=True
            )
            train_Y = torch.cat([train_Y, new_Y])
            gradients = result.gradients
            if iteration == 1:
                self.assertEqual(result.subspace.shape, torch.Size([d, 2]))
            self.assertTrue(torch.isfinite(result.acq_values).all())
        self.assertEqual(train_X.shape, torch.Size([8, d]))
