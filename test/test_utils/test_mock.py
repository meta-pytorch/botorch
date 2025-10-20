#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import re
import warnings
from unittest.mock import patch

import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.core import scipy_minimize
from botorch.optim.initializers import gen_batch_initial_conditions, initialize_q_batch
from botorch.optim.optimize import optimize_acqf
from botorch.optim.optimize_mixed import (
    continuous_step,
    discrete_step,
    get_nearest_neighbors,
    optimize_acqf_mixed_alternating,
)
from botorch.test_utils.mock import mock_optimize, mock_optimize_context_manager
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction


MAX_ITER_MSG = re.compile(
    # Note that the message changed with scipy 1.15, hence the different matching here.
    "TOTAL NO. (of|OF) ITERATIONS REACHED LIMIT"
)


class SinAcqusitionFunction(MockAcquisitionFunction):
    """Simple acquisition function with known numerical properties."""

    def __init__(self, *args, **kwargs):  # noqa: D107
        return

    def __call__(self, X):
        return torch.sin(2 * X[..., 0].max(dim=-1).values)


class TestMock(BotorchTestCase):
    def test_mock_optimize_context_manager(self) -> None:
        with self.subTest("gen_candidates_scipy"):
            with mock_optimize_context_manager():
                cand, value = gen_candidates_scipy(
                    initial_conditions=torch.tensor([[0.0]]),
                    acquisition_function=SinAcqusitionFunction(),
                )
            # When not using `mock_optimize`, the value is 1.0. With it, the value is
            # around 0.9875
            self.assertLess(value.item(), 0.99)

        with self.subTest("scipy_minimize"):
            x = torch.tensor([0.0])

            def closure():
                return torch.sin(x), [torch.cos(x)]

            with mock_optimize_context_manager():
                result = scipy_minimize(closure=closure, parameters={"x": x})
            self.assertTrue(MAX_ITER_MSG.search(result.message))

        with self.subTest("optimize_acqf"):
            with mock_optimize_context_manager():
                cand, value = optimize_acqf(
                    acq_function=SinAcqusitionFunction(),
                    bounds=torch.tensor([[-2.0], [2.0]]),
                    q=1,
                    num_restarts=32,
                    batch_initial_conditions=torch.tensor([[0.0]]),
                )
            self.assertLess(value.item(), 0.99)

        with self.subTest("gen_batch_initial_conditions"):
            with (
                mock_optimize_context_manager(),
                patch(
                    "botorch.optim.initializers.initialize_q_batch",
                    wraps=initialize_q_batch,
                ) as mock_init_q_batch,
            ):
                cand, value = optimize_acqf(
                    acq_function=SinAcqusitionFunction(),
                    bounds=torch.tensor([[-2.0], [2.0]]),
                    q=1,
                    num_restarts=32,
                    raw_samples=16,
                )
            self.assertEqual(mock_init_q_batch.call_args[1]["n"], 2)

    def test_mock_optimize_mixed_alternating(self) -> None:
        with (
            patch(
                "botorch.optim.optimize_mixed.discrete_step",
                wraps=discrete_step,
            ) as mock_discrete,
            patch(
                "botorch.optim.optimize_mixed.continuous_step",
                wraps=continuous_step,
            ) as mock_continuous,
            patch(
                "botorch.optim.optimize_mixed.get_nearest_neighbors",
                wraps=get_nearest_neighbors,
            ) as mock_neighbors,
        ):
            optimize_acqf_mixed_alternating(
                acq_function=SinAcqusitionFunction(),
                bounds=torch.tensor([[-2.0, 0.0], [2.0, 5.0]]),
                discrete_dims={1: list(range(6))},
                num_restarts=1,
            )
        # These should be called at most `MAX_ITER_ALTER` times for each random
        # restart, which is mocked to 1.
        mock_discrete.assert_called_once()
        mock_continuous.assert_called_once()
        # This should be called at most `MAX_ITER_DISCRETE` in each call of
        # `mock_discrete`, which should total to 1.
        mock_neighbors.assert_called_once()

    @mock_optimize
    def test_decorator(self) -> None:
        model = SingleTaskGP(
            train_X=torch.tensor([[0.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0]], dtype=torch.double),
        )
        acqf = qKnowledgeGradient(model=model, num_fantasies=64)
        # this is called within gen_one_shot_kg_initial_conditions
        with (
            patch(
                "botorch.optim.initializers.gen_batch_initial_conditions",
                wraps=gen_batch_initial_conditions,
            ) as mock_gen_batch_ics,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
            cand, value = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[-2.0], [2.0]], dtype=torch.double),
                q=1,
                num_restarts=32,
                raw_samples=16,
            )

        called_with = mock_gen_batch_ics.call_args[1]
        self.assertEqual(called_with["num_restarts"], 2)
        self.assertEqual(called_with["raw_samples"], 4)

    def test_raises_when_unused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "No mocks were called"):
            with mock_optimize_context_manager():
                pass
