#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.posteriors.gpytorch import scalarize_posterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.testing import BotorchTestCase, get_test_posterior


class TestPosteriorList(BotorchTestCase):
    def test_scalarize_posterior_two_posteriors(self) -> None:
        """
        Test that when a PosteriorList has two posteriors, result of
        ``scalarize_posterior`` matches quantitative expectations, analyitically
        computed by hand.
        """
        m = 1
        for batch_shape, lazy, dtype in itertools.product(
            ([], [3]), (False, True), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            posterior = get_test_posterior(batch_shape, m=m, lazy=lazy, **tkwargs)
            posterior_list = PosteriorList(posterior, posterior)
            scalarized_posterior = scalarize_posterior(
                posterior, weights=torch.ones(1, **tkwargs)
            )

            scalarized_posterior_list = scalarize_posterior(
                posterior_list, weights=torch.arange(1, 3, **tkwargs)
            )
            # 1 * orig mean + 2 * orig mean
            self.assertTrue(
                torch.allclose(
                    scalarized_posterior.mean * 3, scalarized_posterior_list.mean
                )
            )
            # 1 * orig var + 2^2 * orig var
            self.assertTrue(
                torch.allclose(
                    scalarized_posterior.variance * 5,
                    scalarized_posterior_list.variance,
                )
            )

    def test_scalarize_posterior_one_posterior(self) -> None:
        """
        Test that when a PosteriorList has one posterior, result of
        ``scalarize_posterior`` matches result of calling ``scalarize_posterior``
        on that posterior.
        """
        m = 1
        for batch_shape, lazy, dtype in itertools.product(
            ([], [3]), (False, True), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            offset = torch.rand(1).item()
            weights = torch.randn(m, **tkwargs)
            # Make sure the weights are not too small.
            while torch.any(weights.abs() < 0.1):
                weights = torch.randn(m, **tkwargs)
            # test q=1
            posterior = get_test_posterior(batch_shape, m=m, lazy=lazy, **tkwargs)
            posterior_list = PosteriorList(posterior)
            new_posterior = scalarize_posterior(posterior, weights, offset)
            new_post_from_list = scalarize_posterior(posterior_list, weights, offset)
            self.assertEqual(new_posterior.mean.shape, new_post_from_list.mean.shape)
            self.assertAllClose(new_posterior.mean, new_post_from_list.mean)
            self.assertTrue(
                torch.allclose(new_posterior.variance, new_post_from_list.variance)
            )

    def test_scalarize_posterior_raises_not_implemented(self) -> None:
        """
        Test that ``scalarize_posterior`` raises ``NotImplementedError`` when provided
        input shapes that are not supported for ``PosteriorList``.
        """
        for batch_shape, m, lazy, dtype in itertools.product(
            ([], [3]), (1, 2), (False, True), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            offset = torch.rand(1).item()
            weights = torch.randn(m, **tkwargs)
            # Make sure the weights are not too small.
            while torch.any(weights.abs() < 0.1):
                weights = torch.randn(m, **tkwargs)
            # test q=1
            posterior = get_test_posterior(batch_shape, m=m, lazy=lazy, **tkwargs)
            posterior_list = PosteriorList(posterior)
            if m > 1:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "scalarize_posterior only works with a PosteriorList if each "
                    "sub-posterior has one outcome.",
                ):
                    scalarize_posterior(posterior_list, weights, offset)

            # test q=2, interleaved
            q = 2
            posterior = get_test_posterior(
                batch_shape, q=q, m=m, lazy=lazy, interleaved=True, **tkwargs
            )
            posterior_list = PosteriorList(posterior)
            with self.assertRaisesRegex(
                NotImplementedError,
                "scalarize_posterior only works with a PosteriorList if each "
                "sub-posterior has q=1.",
            ):
                scalarize_posterior(posterior_list, weights, offset)

            # test q=2, non-interleaved
            # test independent special case as well
            for independent in (False, True) if m > 1 else (False,):
                posterior = get_test_posterior(
                    batch_shape,
                    q=q,
                    m=m,
                    lazy=lazy,
                    interleaved=False,
                    independent=independent,
                    **tkwargs,
                )
                posterior_list = PosteriorList(posterior)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "scalarize_posterior only works with a PosteriorList if each "
                    "sub-posterior has q=1.",
                ):
                    scalarize_posterior(posterior_list, weights, offset)
