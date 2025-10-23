#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import re
import warnings
from functools import partial
from itertools import product
from typing import Any, Callable
from unittest import mock

import numpy as np
import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.exceptions import InputDataError, UnsupportedError
from botorch.exceptions.errors import CandidateGenerationError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.initializers import (
    gen_one_shot_hvkg_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.optimize import (
    _combine_initial_conditions,
    _filter_infeasible,
    _filter_invalid,
    _gen_batch_initial_conditions_local_search,
    _generate_neighbors,
    gen_batch_initial_conditions,
    optimize_acqf,
    optimize_acqf_cyclic,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_list,
    optimize_acqf_mixed,
    OptimizeAcqfInputs,
)
from botorch.optim.parameter_constraints import (
    _arrayify,
    _make_f_and_grad_nonlinear_inequality_constraints,
    project_to_feasible_space_via_slsqp,
)
from botorch.optim.utils.timeout import minimize_with_timeout
from botorch.test_utils.mock import mock_optimize
from botorch.utils.testing import BotorchTestCase, MockAcquisitionFunction
from scipy.optimize import OptimizeResult
from torch import Tensor


class MockOneShotAcquisitionFunction(
    MockAcquisitionFunction, OneShotAcquisitionFunction
):
    def __init__(self, num_fantasies=2):
        r"""
        Args:
            num_fantasies: The number of fantasies.
        """
        super().__init__()
        self.num_fantasies = num_fantasies

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_fantasies, :]

    def forward(self, X):
        pass


class SquaredAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model=None):  # noqa: D107
        super().__init__(model=model)

    def forward(self, X):
        # we take the norm and sum over the q-batch dim
        if len(X.shape) > 2:
            return torch.linalg.norm(X, dim=-1).sum(-1)
        else:
            return torch.linalg.norm(X, dim=-1).squeeze(-1)


class MockOneShotEvaluateAcquisitionFunction(MockOneShotAcquisitionFunction):
    def evaluate(self, X: Tensor, bounds: Tensor):
        return X.sum()


class SinOneOverXAcqusitionFunction(MockAcquisitionFunction):
    """
    Acquisition function for sin(1/x).

    This is useful for testing because it behaves pathologically only near zero,
    so optimization is likely to fail when initializing near zero but not
    elsewhere.
    """

    def __call__(self, X):
        return torch.sin(1 / X[..., 0].max(dim=-1).values)


def rounding_func(X: Tensor) -> Tensor:
    batch_shape, d = X.shape[:-1], X.shape[-1]
    X_round = torch.stack([x.round() for x in X.view(-1, d)])
    return X_round.view(*batch_shape, d)


class TestCombineInitialConditions(BotorchTestCase):
    def test_combine_both_conditions(self):
        provided = torch.randn(1, 3, 4)
        generated = torch.randn(2, 3, 4)

        result = _combine_initial_conditions(
            provided_initial_conditions=provided,
            generated_initial_conditions=generated,
        )

        assert result.shape == (3, 3, 4)  # Combined shape

    def test_only_generated_conditions(self):
        generated = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = _combine_initial_conditions(
            provided_initial_conditions=None,
            generated_initial_conditions=generated,
        )

        assert torch.equal(result, generated)

    def test_no_conditions_raises_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Either `batch_initial_conditions` or `raw_samples` must be set.",
        ):
            _combine_initial_conditions(
                provided_initial_conditions=None,
                generated_initial_conditions=None,
            )


class TestOptimizeAcqf(BotorchTestCase):
    @mock.patch("botorch.generation.gen.gen_candidates_torch")
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_joint(
        self,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
        mock_gen_candidates_torch,
    ):
        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}
        mock_acq_function = MockAcquisitionFunction()
        cnt = 0

        for dtype in (torch.float, torch.double):
            for mock_gen_candidates in (
                mock_gen_candidates_scipy,
                mock_gen_candidates_torch,
            ):
                mock_gen_batch_initial_conditions.return_value = torch.zeros(
                    num_restarts, q, 3, device=self.device, dtype=dtype
                )
                base_cand = torch.arange(3, device=self.device, dtype=dtype).expand(
                    1, q, 3
                )
                mock_candidates = torch.cat(
                    [i * base_cand for i in range(num_restarts)], dim=0
                )
                mock_acq_values = num_restarts - torch.arange(
                    num_restarts, device=self.device, dtype=dtype
                )
                mock_gen_candidates.return_value = (mock_candidates, mock_acq_values)
                bounds = torch.stack(
                    [
                        torch.zeros(3, device=self.device, dtype=dtype),
                        4 * torch.ones(3, device=self.device, dtype=dtype),
                    ]
                )
                mock_gen_candidates.reset_mock()
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    gen_candidates=mock_gen_candidates,
                )
                mock_gen_candidates.assert_called_once()
                self.assertTrue(torch.equal(candidates, mock_candidates[0]))
                self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))
                cnt += 1
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test case where provided initial conditions equal to raw_samples
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    return_best_only=False,
                    batch_initial_conditions=torch.zeros(
                        num_restarts, q, 3, device=self.device, dtype=dtype
                    ),
                    gen_candidates=mock_gen_candidates,
                )
                self.assertTrue(torch.equal(candidates, mock_candidates))
                self.assertTrue(torch.equal(acq_vals, mock_acq_values))
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test generation with batch initial conditions less than num_restarts
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts + 1,
                    raw_samples=raw_samples,
                    options=options,
                    return_best_only=False,
                    batch_initial_conditions=torch.zeros(
                        num_restarts, q, 3, device=self.device, dtype=dtype
                    ),
                    gen_candidates=mock_gen_candidates,
                )
                cnt += 1
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test fixed features
                fixed_features = {0: 0.1}
                mock_candidates[:, 0] = 0.1
                mock_gen_candidates.return_value = (mock_candidates, mock_acq_values)
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    fixed_features=fixed_features,
                    gen_candidates=mock_gen_candidates,
                )
                self.assertEqual(
                    mock_gen_candidates.call_args[1]["fixed_features"], fixed_features
                )
                self.assertTrue(torch.equal(candidates, mock_candidates[0]))
                cnt += 1
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

                # test trivial case when all features are fixed
                candidates, acq_vals = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    fixed_features={0: 0.1, 1: 0.2, 2: 0.3},
                    gen_candidates=mock_gen_candidates,
                )
                self.assertTrue(
                    torch.equal(
                        candidates,
                        torch.tensor(
                            [0.1, 0.2, 0.3], device=self.device, dtype=dtype
                        ).expand(3, 3),
                    )
                )
                self.assertEqual(mock_gen_batch_initial_conditions.call_count, cnt)

        # test OneShotAcquisitionFunction
        mock_acq_function = MockOneShotAcquisitionFunction()
        candidates, acq_vals = optimize_acqf(
            acq_function=mock_acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            gen_candidates=mock_gen_candidates,
        )
        self.assertTrue(
            torch.equal(
                candidates, mock_acq_function.extract_candidates(mock_candidates[0])
            )
        )
        self.assertTrue(torch.equal(acq_vals, mock_acq_values[0]))

        # verify ValueError
        with self.assertRaisesRegex(ValueError, "Must specify"):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                options=options,
                gen_candidates=mock_gen_candidates,
            )

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch(
        "botorch.optim.optimize.gen_candidates_scipy", wraps=gen_candidates_scipy
    )
    @mock.patch(
        "botorch.generation.gen.gen_candidates_torch", wraps=gen_candidates_torch
    )
    def test_optimize_acqf_sequential(
        self,
        mock_gen_candidates_torch,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
        timeout_sec=None,
    ):
        for mock_gen_candidates, timeout_sec in product(
            [mock_gen_candidates_scipy, mock_gen_candidates_torch], [None, 1e-4]
        ):
            q = 3
            num_restarts = 2
            raw_samples = 10
            options = {}
            for dtype, use_rounding in ((torch.float, True), (torch.double, False)):
                mock_acq_function = MockAcquisitionFunction()
                mock_gen_batch_initial_conditions.side_effect = [
                    torch.zeros(num_restarts, 1, 3, device=self.device, dtype=dtype)
                    for _ in range(q)
                ]
                gcs_return_vals = [
                    (
                        torch.tensor(
                            [[[1.1, 2.1, 4.0]]], device=self.device, dtype=dtype
                        ),
                        torch.tensor([i], device=self.device, dtype=dtype),
                    )
                    for i in range(q)
                ]
                mock_gen_candidates.side_effect = gcs_return_vals
                bounds = torch.stack(
                    [
                        torch.zeros(3, device=self.device, dtype=dtype),
                        4 * torch.ones(3, device=self.device, dtype=dtype),
                    ]
                )
                if mock_gen_candidates is mock_gen_candidates_scipy:
                    # x[2] * 4 >= 5
                    inequality_constraints = [
                        (
                            torch.tensor([2], dtype=torch.long, device=self.device),
                            torch.tensor([4], device=self.device),
                            torch.tensor(5, device=self.device),
                        )
                    ]
                    equality_constraints = [
                        (
                            torch.tensor([0, 1], dtype=torch.long, device=self.device),
                            torch.ones(2, device=self.device),
                            torch.tensor(16.0, device=self.device),
                        )
                    ]
                    equality_constraints = None
                # gen_candidates_torch does not support constraints
                else:
                    inequality_constraints = None
                    equality_constraints = None

                mock_gen_candidates.reset_mock()
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    post_processing_func=rounding_func if use_rounding else None,
                    sequential=True,
                    timeout_sec=timeout_sec,
                    gen_candidates=mock_gen_candidates,
                )
                self.assertEqual(mock_gen_candidates.call_count, q)
                base_candidates = torch.cat(
                    [cands[0] for cands, _ in gcs_return_vals], dim=-2
                )
                if use_rounding:
                    expected_candidates = base_candidates.round()
                    expected_val = mock_acq_function(expected_candidates.unsqueeze(-2))
                else:
                    expected_candidates = base_candidates
                    expected_val = torch.cat(
                        [acqval for _candidate, acqval in gcs_return_vals]
                    )
                self.assertTrue(torch.equal(candidates, expected_candidates))
                self.assertTrue(torch.equal(acq_value, expected_val))
            # verify error when using a OneShotAcquisitionFunction
            with self.assertRaises(NotImplementedError):
                optimize_acqf(
                    acq_function=mock.Mock(spec=OneShotAcquisitionFunction),
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    sequential=True,
                )
            # Verify error for passing in incorrect bounds
            with self.assertRaisesRegex(
                ValueError,
                "bounds should be a `2 x d` tensor",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds.T,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    sequential=True,
                )

            # Verify error when using sequential=True in
            # conjunction with user-supplied batch_initial_conditions
            with self.assertRaisesRegex(
                UnsupportedError,
                "`batch_initial_conditions` is not supported for sequential "
                "optimization. Either avoid specifying `batch_initial_conditions` "
                "to use the custom initializer or use the `ic_generator` kwarg to "
                "generate initial conditions for the case of "
                "nonlinear inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    batch_initial_conditions=torch.zeros((1, 1, 3)),
                    sequential=True,
                )

    @mock.patch(
        "botorch.optim.optimize.gen_candidates_scipy", wraps=gen_candidates_scipy
    )
    def test_optimize_acq_function_sequence(
        self,
        mock_gen_candidates_scipy,
    ):
        acq_function_sequence = [MockAcquisitionFunction() for _ in range(3)]
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Validation
        with self.assertRaisesRegex(
            ValueError,
            "Either `acq_function` or `acq_function_sequence` must be specified",
        ):
            optimize_acqf(
                acq_function=None,
                bounds=bounds,
                q=3,
                num_restarts=2,
                raw_samples=10,
                sequential=True,
                acq_function_sequence=None,
            )
        with self.assertRaisesRegex(
            ValueError,
            "acq_function_sequence requires sequential optimization",
        ):
            optimize_acqf(
                acq_function=mock.MagicMock(),
                bounds=bounds,
                q=3,
                num_restarts=2,
                raw_samples=10,
                sequential=False,
                acq_function_sequence=acq_function_sequence,
            )
        with self.assertRaisesRegex(
            ValueError,
            "acq_function_sequence must have length q",
        ):
            optimize_acqf(
                acq_function=mock.MagicMock(),
                bounds=bounds,
                q=2,
                num_restarts=2,
                raw_samples=10,
                sequential=True,
                acq_function_sequence=acq_function_sequence,
            )
        with self.assertRaisesRegex(
            ValueError,
            "acq_function_sequence requires q > 1",
        ):
            optimize_acqf(
                acq_function=mock.MagicMock(),
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=10,
                sequential=True,
                acq_function_sequence=acq_function_sequence[:1],
            )
        # Test that uses sequence of acquisitions
        acq_function = mock.MagicMock()
        acq_function.X_pending = None
        acq_function_sequence[2].X_pending = torch.ones(2, 3)
        _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=3,
            num_restarts=2,
            raw_samples=10,
            sequential=True,
            acq_function_sequence=acq_function_sequence,
        )
        self.assertEqual(mock_gen_candidates_scipy.call_count, 3)
        self.assertEqual(acq_function_sequence[0]._call_args["set_X_pending"], [None])
        for i in range(1, 2):
            set_X_args = acq_function_sequence[i]._call_args["set_X_pending"]
            self.assertEqual(len(set_X_args), 2)
            self.assertEqual(len(set_X_args[0]), i)
            self.assertIsNone(set_X_args[1])
        set_X_args = acq_function_sequence[2]._call_args["set_X_pending"]
        self.assertEqual(len(set_X_args), 2)
        self.assertEqual(len(set_X_args[0]), 4)
        self.assertTrue(
            torch.equal(set_X_args[0][:2, :], torch.ones(2, 3))
        )  # base X_pending
        self.assertTrue(torch.equal(set_X_args[1], torch.ones(2, 3)))  # reset
        acq_function.assert_not_called()

    @mock.patch(
        "botorch.generation.gen.minimize_with_timeout",
        wraps=minimize_with_timeout,
    )
    @mock.patch("botorch.optim.utils.timeout.optimize.minimize")
    def test_optimize_acqf_timeout(
        self, mock_minimize, mock_minimize_with_timeout
    ) -> None:
        """
        Check that the right value of `timeout_sec` is passed to `minimize_with_timeout`
        """

        num_restarts = 2
        q = 3
        dim = 4

        for timeout_sec, sequential, expected_call_count, expected_timeout_arg in [
            (1.0, True, num_restarts * q, 1.0 / (num_restarts * q)),
            (0.0, True, num_restarts * q, 0.0),
            (1.0, False, num_restarts, 1.0 / num_restarts),
            (0.0, False, num_restarts, 0.0),
        ]:
            with self.subTest(
                timeout_sec=timeout_sec,
                sequential=sequential,
                expected_call_count=expected_call_count,
                expected_timeout_arg=expected_timeout_arg,
            ):
                mock_minimize.return_value = OptimizeResult(
                    {
                        "x": np.zeros(dim if sequential else dim * q),
                        "success": True,
                        "status": 0,
                    },
                )

                optimize_acqf(
                    timeout_sec=timeout_sec,
                    q=q,
                    sequential=sequential,
                    num_restarts=num_restarts,
                    acq_function=SinOneOverXAcqusitionFunction(),
                    bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                    raw_samples=7,
                    options={"batch_limit": 1},
                )
                self.assertEqual(
                    mock_minimize_with_timeout.call_count, expected_call_count
                )
                timeout_times = torch.tensor(
                    [
                        elt.kwargs["timeout_sec"]
                        for elt in mock_minimize_with_timeout.mock_calls
                    ]
                )
                self.assertGreaterEqual(timeout_times.min(), 0)
                self.assertAllClose(
                    timeout_times,
                    torch.full_like(timeout_times, expected_timeout_arg),
                    rtol=float("inf"),
                    atol=1e-8,
                )
                mock_minimize_with_timeout.reset_mock()

    def test_optimize_acqf_sequential_notimplemented(self):
        # Sequential acquisition function optimization only supported
        # when return_best_only=True
        with self.assertRaises(NotImplementedError):
            optimize_acqf(
                acq_function=MockAcquisitionFunction(),
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                q=3,
                num_restarts=2,
                raw_samples=10,
                return_best_only=False,
                sequential=True,
            )

    def test_optimize_acqf_sequential_q_constraint_notimplemented(self):
        # Sequential optimization is not supported with constraints across q-dim.
        shared_args: dict[str, Any] = {
            "acq_function": MockAcquisitionFunction(),
            "bounds": torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
            "num_restarts": 2,
            "raw_samples": 10,
            "q": 3,
            "sequential": True,
        }
        with self.assertRaisesRegex(
            UnsupportedError, "Inter-point constraints .* linear equality"
        ):
            optimize_acqf(
                equality_constraints=[
                    (
                        torch.tensor(
                            [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                        ),
                        torch.tensor(
                            [1.0, -1.0], device=self.device, dtype=torch.float64
                        ),
                        0,
                    ),
                ],
                **shared_args,
            )
        with self.assertRaisesRegex(
            UnsupportedError, "Inter-point constraints .* linear inequality"
        ):
            optimize_acqf(
                inequality_constraints=[
                    (
                        torch.tensor(
                            [[0, 0], [1, 0]], device=self.device, dtype=torch.int64
                        ),
                        torch.tensor(
                            [1.0, -1.0], device=self.device, dtype=torch.float64
                        ),
                        0,
                    ),
                ],
                **shared_args,
            )
        with self.assertRaisesRegex(
            UnsupportedError, "Inter-point constraints .* non-linear inequality"
        ):
            optimize_acqf(
                nonlinear_inequality_constraints=[
                    (lambda X: X.sum(dim=(-1, -2)), False)
                ],
                ic_generator=lambda *args, **kwargs: torch.zeros(1, 1, 3),
                **shared_args,
            )

    def test_optimize_acqf_batch_limit(self) -> None:
        num_restarts = 5
        raw_samples = 16
        dim = 4
        q = 4
        batch_limit = 2

        options = {"batch_limit": batch_limit}
        initial_conditions = [(1, 2, dim), (3, 1, dim), (3, q, dim), (1, dim), None]
        expected_acqf_shapes = [1, 3, num_restarts, 1, num_restarts]
        expected_candidates_shapes = [
            (1, 2, dim),
            (3, 1, dim),
            (num_restarts, q, dim),
            (1, dim),
            (num_restarts, q, dim),
        ]

        for gen_candidates, (
            ic_shape,
            expected_acqf_shape,
            expected_candidates_shape,
        ) in product(
            [gen_candidates_scipy, gen_candidates_torch],
            zip(
                initial_conditions,
                expected_acqf_shapes,
                expected_candidates_shapes,
                strict=True,
            ),
        ):
            ics = torch.ones(ic_shape) if ic_shape is not None else None
            with self.subTest(gen_candidates=gen_candidates, initial_conditions=ics):
                _candidates, acq_value_list = optimize_acqf(
                    acq_function=SinOneOverXAcqusitionFunction(),
                    bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    return_best_only=False,
                    gen_candidates=gen_candidates,
                    batch_initial_conditions=ics,
                )

                self.assertEqual(acq_value_list.shape, (expected_acqf_shape,))
                self.assertEqual(_candidates.shape, expected_candidates_shape)

        for ic_shape, expected_shape in [((2, 1, dim), 2), ((2, dim), 1)]:
            with self.subTest(gen_candidates=gen_candidates):
                ics = torch.ones((ic_shape))
                with self.assertWarnsRegex(
                    RuntimeWarning, "botorch will default to old behavior"
                ):
                    _candidates, acq_value_list = optimize_acqf(
                        acq_function=SinOneOverXAcqusitionFunction(),
                        bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                        q=q,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options=options,
                        return_best_only=False,
                        gen_candidates=gen_candidates,
                        batch_initial_conditions=ics,
                    )
                self.assertEqual(acq_value_list.shape, (expected_shape,))

    def test_optimize_acqf_runs_given_batch_initial_conditions(self):
        num_restarts, raw_samples, dim = 1, 2, 3

        opt_x = 2 / np.pi
        # -x[i] * 1 >= -opt_x * 1.01 => x[i] <= opt_x * 1.01
        inequality_constraints = [
            (torch.tensor([i]), -torch.tensor([1]), -opt_x * 1.01) for i in range(dim)
        ] + [
            # x[i] * 1 >= opt_x * .99
            (torch.tensor([i]), torch.tensor([1]), opt_x * 0.99)
            for i in range(dim)
        ]
        q = 1

        ic_shapes = [(1, 2, dim), (1, dim)]

        torch.manual_seed(0)
        for shape in ic_shapes:
            with self.subTest(shape=shape):
                # start near one (of many) optima
                initial_conditions = (opt_x * 1.01) * torch.ones(shape)
                batch_candidates, acq_value_list = optimize_acqf(
                    acq_function=SinOneOverXAcqusitionFunction(),
                    bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    batch_initial_conditions=initial_conditions,
                    inequality_constraints=inequality_constraints,
                )
                self.assertAllClose(
                    batch_candidates,
                    opt_x * torch.ones_like(batch_candidates),
                    # must be at least 50% closer to the optimum than it started
                    atol=0.004,
                    rtol=0.005,
                )
                self.assertAlmostEqual(acq_value_list.item(), 1, places=3)

    def test_optimize_acqf_wrong_ic_shape_inequality_constraints(self) -> None:
        dim = 3
        ic_shapes = [(1, 2, dim + 1), (1, 2, dim, 1), (1, dim + 1), (1, 1), (dim,)]

        for shape in ic_shapes:
            with self.subTest(shape=shape):
                initial_conditions = torch.ones(shape)
                expected_error = (
                    rf"batch_initial_conditions.shape\[-1\] must be {dim}\."
                    if len(shape) in (2, 3)
                    else r"batch_initial_conditions must be 2\-dimensional or "
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    optimize_acqf(
                        acq_function=MockAcquisitionFunction(),
                        bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                        q=4,
                        batch_initial_conditions=initial_conditions,
                        num_restarts=1,
                    )

    def test_optimize_acqf_warns_on_opt_failure(self):
        """
        Test error handling in `scipy.optimize.minimize`.

        Expected behavior is that a warning is raised when optimization fails
        in `scipy.optimize.minimize`, and then it restarts and tries again.

        This is a test case cooked up to fail. It is trying to optimize
        sin(1/x), which is pathological near zero, given a starting point near
        zero.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        initial_conditions = 1e-8 * torch.ones((num_restarts, raw_samples, dim))
        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=torch.stack([-1 * torch.ones(dim), torch.ones(dim)]),
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                batch_initial_conditions=initial_conditions,
            )
        message_regex = re.compile(
            r"Optimization failed in `gen_candidates_scipy` with the following "
            r"warning\(s\):\n\[OptimizationWarning\('Optimization failed within "
            r"`scipy.optimize.minimize` with status 2 and message "
            r"ABNORMAL(: |_TERMINATION_IN_LNSRCH).'\)]\nBecause you specified "
            r"`batch_initial_conditions` larger than required `num_restarts`, "
            r"optimization will not be retried with new initial conditions and "
            r"will proceed with the current solution. Suggested remediation: "
            r"Try again with different `batch_initial_conditions`, don't provide "
            r"`batch_initial_conditions`, or increase `num_restarts`."
        )
        expected_warning_raised = any(
            issubclass(w.category, RuntimeWarning)
            and message_regex.search(str(w.message))
            for w in ws
        )
        self.assertTrue(expected_warning_raised)

    def test_optimize_acqf_successfully_restarts_on_opt_failure(self):
        """
        Test that `optimize_acqf` can succeed after restarting on opt failure.

        With the given seed (5), `optimize_acqf` will choose an initial
        condition that causes failure in the first run of
        `gen_candidates_scipy`, then re-tries with a new starting point and
        succeed.

        Also tests that this can be turned off by setting
        `retry_on_optimization_warning = False`.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        bounds = torch.stack(
            [
                -1 * torch.ones(dim, dtype=torch.double),
                torch.ones(dim, dtype=torch.double),
            ]
        )
        torch.manual_seed(5)

        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
            )
        message_regex = re.compile(
            r"Optimization failed in `gen_candidates_scipy` with the following "
            r"warning\(s\):\n\[OptimizationWarning\('Optimization failed within "
            r"`scipy.optimize.minimize` with status 2 and message ABNORMAL(: |"
            r"_TERMINATION_IN_LNSRCH).'\)\]\nTrying again with a new set of "
            r"initial conditions."
        )
        expected_warning_raised = any(
            issubclass(w.category, RuntimeWarning)
            and message_regex.search(str(w.message))
            for w in ws
        )
        self.assertTrue(expected_warning_raised)
        # check if it succeeded on restart -- the maximum value of sin(1/x) is 1
        self.assertAlmostEqual(acq_value_list.item(), 1.0)

        # Test with retry_on_optimization_warning = False.
        torch.manual_seed(5)
        with warnings.catch_warnings(record=True) as ws:
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
                retry_on_optimization_warning=False,
            )
        expected_warning_raised = any(
            issubclass(w.category, RuntimeWarning)
            and message_regex.search(str(w.message))
            for w in ws
        )
        self.assertFalse(expected_warning_raised)

    def test_optimize_acqf_warns_on_second_opt_failure(self):
        """
        Test that `optimize_acqf` warns if it fails on a second optimization try.

        With the given seed (230), `optimize_acqf` will choose an initial
        condition that causes failure in the first run of
        `gen_candidates_scipy`, then re-tries and still does not succeed. Since
        this doesn't happen with seeds 0 - 229, this test might be broken by
        future refactorings affecting calls to `torch`.
        """
        num_restarts, raw_samples, dim = 1, 1, 1

        bounds = torch.stack(
            [
                -1 * torch.ones(dim, dtype=torch.double),
                torch.ones(dim, dtype=torch.double),
            ]
        )

        with warnings.catch_warnings(record=True) as ws:
            torch.manual_seed(230)
            batch_candidates, acq_value_list = optimize_acqf(
                acq_function=SinOneOverXAcqusitionFunction(),
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                # shorten the line search to make it faster and make failure
                # more likely
                options={"maxls": 2},
            )

        message_1_regex = re.compile(
            r"Optimization failed in `gen_candidates_scipy` with the following "
            r"warning\(s\):\n\[OptimizationWarning\('Optimization failed within "
            r"`scipy.optimize.minimize` with status 2 and message ABNORMAL(: |"
            r"_TERMINATION_IN_LNSRCH).'\)\]\nTrying again with a new set of "
            r"initial conditions."
        )

        message_2 = (
            "Optimization failed on the second try, after generating a new set "
            "of initial conditions."
        )
        first_expected_warning_raised = any(
            issubclass(w.category, RuntimeWarning)
            and message_1_regex.search(str(w.message))
            for w in ws
        )
        second_expected_warning_raised = any(
            issubclass(w.category, RuntimeWarning) and message_2 in str(w.message)
            for w in ws
        )
        self.assertTrue(first_expected_warning_raised)
        self.assertTrue(second_expected_warning_raised)

    def test_optimize_acqf_nonlinear_constraints(self):
        num_restarts = 2
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mock_acq_function = SquaredAcquisitionFunction()
            bounds = torch.stack(
                [torch.zeros(3, **tkwargs), 4 * torch.ones(3, **tkwargs)]
            )

            # Make sure we find the global optimum [4, 4, 4] without constraints
            with torch.random.fork_rng():
                torch.manual_seed(0)
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    sequential=True,
                    raw_samples=16,
                )
            self.assertAllClose(candidates, 4 * torch.ones(1, 3, **tkwargs))

            # Constrain the sum to be <= 4 in which case the solution is a
            # permutation of [4, 0, 0]
            def nlc1(x):
                return 4 - x.sum(dim=-1)

            batch_initial_conditions = torch.tensor([[[0.5, 0.5, 3]]], **tkwargs)
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=1,
                nonlinear_inequality_constraints=[(nlc1, True)],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=1,
            )
            self.assertTrue(
                torch.allclose(
                    torch.sort(candidates).values,
                    torch.tensor([[0, 0, 4]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor([4], **tkwargs), atol=1e-3)
            )

            # Constrain all variables to be >= 1. The global optimum is 2.45 and
            # is attained by some permutation of [1, 1, 2]
            def nlc2(x):
                return x[..., 0] - 1

            def nlc3(x):
                return x[..., 1] - 1

            def nlc4(x):
                return x[..., 2] - 1

            # test it with q=1
            with torch.random.fork_rng():
                torch.manual_seed(0)
                batch_initial_conditions = 1 + 0.33 * torch.rand(
                    num_restarts, 1, 3, **tkwargs
                )
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=1,
                nonlinear_inequality_constraints=[
                    (nlc1, True),
                    (nlc2, True),
                    (nlc3, True),
                    (nlc4, True),
                ],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=num_restarts,
            )
            self.assertTrue(
                torch.allclose(
                    torch.sort(candidates).values,
                    torch.tensor([[1, 1, 2]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor(2.45, **tkwargs), atol=1e-3)
            )

            # test it with q=2
            with torch.random.fork_rng():
                torch.manual_seed(0)
                batch_initial_conditions = 1 + 0.33 * torch.rand(
                    num_restarts, 2, 3, **tkwargs
                )
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=2,
                nonlinear_inequality_constraints=[
                    (nlc1, True),
                    (nlc2, True),
                    (nlc3, True),
                    (nlc4, True),
                ],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=num_restarts,
                return_best_only=True,
            )

            for candidate in candidates:
                self.assertTrue(
                    torch.allclose(
                        torch.sort(candidate).values,
                        torch.tensor([[1.0, 1.0, 2.0]], **tkwargs),
                    )
                )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor(2.45 * 2, **tkwargs), atol=1e-3)
            )

            with torch.random.fork_rng():
                torch.manual_seed(0)
                batch_initial_conditions = torch.rand(num_restarts, 1, 3, **tkwargs)
                batch_initial_conditions[..., 0] = 2

            # test with fixed features
            candidates, acq_value = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=bounds,
                q=1,
                nonlinear_inequality_constraints=[(nlc1, True), (nlc2, True)],
                batch_initial_conditions=batch_initial_conditions,
                num_restarts=num_restarts,
                fixed_features={0: 2},
            )
            self.assertEqual(candidates[0, 0], 2.0)
            self.assertTrue(
                torch.allclose(
                    torch.sort(candidates).values,
                    torch.tensor([[0, 2, 2]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.allclose(acq_value, torch.tensor(2.8284, **tkwargs), atol=1e-3)
            )

            # Test that an ic_generator object with the same API as
            # gen_batch_initial_conditions returns candidates of the
            # required shape.
            with mock.patch(
                "botorch.optim.optimize.gen_batch_initial_conditions"
            ) as ic_generator:
                ic_generator.return_value = batch_initial_conditions
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=3,
                    num_restarts=1,
                    raw_samples=16,
                    nonlinear_inequality_constraints=[(nlc1, True)],
                    ic_generator=ic_generator,
                )
                self.assertEqual(candidates.size(), torch.Size([1, 3]))

            # batch_initial_conditions must be feasible
            with self.assertRaisesRegex(
                ValueError,
                "`batch_initial_conditions` must satisfy the non-linear "
                "inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[(nlc1, True)],
                    num_restarts=num_restarts,
                    batch_initial_conditions=4 * torch.ones(1, 1, 3, **tkwargs),
                )
            # Explicitly setting batch_limit to be >1 should raise
            with self.assertRaisesRegex(
                ValueError,
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[nlc1],
                    batch_initial_conditions=torch.rand(5, 1, 3, **tkwargs),
                    num_restarts=5,
                    options={"batch_limit": 5},
                )
            # If there are non-linear inequality constraints an initial condition
            # generator object `ic_generator` must be supplied.
            with self.assertRaisesRegex(
                RuntimeError,
                "`ic_generator` must be given if "
                "there are non-linear inequality constraints.",
            ):
                optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=1,
                    nonlinear_inequality_constraints=[(nlc1, True)],
                    num_restarts=1,
                    raw_samples=16,
                )

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    @mock.patch("botorch.optim.optimize.gen_candidates_scipy")
    def test_optimize_acqf_non_linear_constraints_sequential(
        self,
        mock_gen_candidates_scipy,
        mock_gen_batch_initial_conditions,
    ):
        def nlc(x):
            return 4 * x[..., 2] - 5

        q = 3
        num_restarts = 2
        raw_samples = 10
        options = {}

        for dtype in (torch.float, torch.double):
            mock_acq_function = MockAcquisitionFunction()
            mock_gen_batch_initial_conditions.side_effect = [
                torch.zeros(num_restarts, 1, 3, device=self.device, dtype=dtype)
                for _ in range(q)
            ]
            gcs_return_vals = [
                (
                    torch.tensor([[[1.0, 2.0, 3.0]]], device=self.device, dtype=dtype),
                    torch.tensor([i], device=self.device, dtype=dtype),
                )
                # for nonlinear inequality constraints the batch_limit variable is
                # currently set to 1 by default and hence gen_candidates_scipy is
                # called num_restarts*q times
                for i in range(num_restarts * q)
            ]
            mock_gen_candidates_scipy.side_effect = gcs_return_vals
            expected_candidates = torch.cat(
                [cands[0] for cands, _ in gcs_return_vals[::num_restarts]], dim=-2
            )
            bounds = torch.stack(
                [
                    torch.zeros(3, device=self.device, dtype=dtype),
                    4 * torch.ones(3, device=self.device, dtype=dtype),
                ]
            )
            with warnings.catch_warnings(record=True) as ws:
                candidates, acq_value = optimize_acqf(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    nonlinear_inequality_constraints=[(nlc, True)],
                    sequential=True,
                    ic_generator=mock_gen_batch_initial_conditions,
                    gen_candidates=mock_gen_candidates_scipy,
                )
                self.assertEqual(len(ws), 0)
            self.assertTrue(torch.equal(candidates, expected_candidates))
            # Extract the relevant entries from gcs_return_vals to
            # perform comparison with.
            self.assertTrue(
                torch.equal(
                    acq_value,
                    torch.cat(
                        [
                            expected_acq_value
                            for _candidate, expected_acq_value in gcs_return_vals[
                                num_restarts - 1 :: num_restarts
                            ]
                        ]
                    ),
                ),
            )

    def test_constraint_caching(self):
        def nlc(x):
            return 4 - x.sum(dim=-1)

        class FunWrapperWithCallCount:
            def __init__(self):
                self.call_count = 0

            def __call__(self, x, f):
                self.call_count += 1
                X = torch.from_numpy(x).view(-1).contiguous().requires_grad_(True)
                loss = f(X).sum()
                gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
                return loss.item(), gradf

        f_np_wrapper = FunWrapperWithCallCount()
        f_obj, f_grad = _make_f_and_grad_nonlinear_inequality_constraints(
            f_np_wrapper=f_np_wrapper, nlc=nlc
        )
        x1, x2 = np.array([1.0, 0.5, 0.25]), np.array([1.0, 0.5, 0.5])
        # Call f_obj once, this requires calling f_np_wrapper
        self.assertEqual(f_obj(x1), 2.25)
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_obj again, we should use the cached value this time
        self.assertEqual(f_obj(x1), 2.25)
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_grad, we should use the cached value here as well
        self.assertTrue(np.array_equal(f_grad(x1), -np.ones(3)))
        self.assertEqual(f_np_wrapper.call_count, 1)
        # Call f_grad with a new input
        self.assertTrue(np.array_equal(f_grad(x2), -np.ones(3)))
        self.assertEqual(f_np_wrapper.call_count, 2)
        # Call f_obj on the new input, should use the cache
        self.assertEqual(f_obj(x2), 2.0)
        self.assertEqual(f_np_wrapper.call_count, 2)

    def _test_optimize_acqf_infeasible_candidates(
        self, mock_gen_batch_initial_conditions, q, num_restarts, ics
    ):
        mock_acq_function = MockAcquisitionFunction()
        mock_gen_batch_initial_conditions.side_effect = [ics for _ in range(2)]
        with mock.patch(
            "botorch.generation.gen.minimize_with_timeout",
            return_value=OptimizeResult(x=ics.view(-1).cpu().numpy()),
        ):
            candidates, _ = optimize_acqf(
                acq_function=mock_acq_function,
                bounds=torch.tensor(
                    [[0.0], [3.0]], dtype=ics.dtype, device=self.device
                ),
                q=q,
                num_restarts=num_restarts,
                raw_samples=1,
                inequality_constraints=[
                    (  # X[..., 0] >= 2.0, which is infeasible.
                        torch.tensor([0], device=self.device),
                        torch.tensor([1.0], dtype=ics.dtype, device=self.device),
                        2.0,
                    )
                ],
                sequential=False,
                gen_candidates=gen_candidates_scipy,
                initial_conditions=ics,
                acquisition_function=MockAcquisitionFunction(),
            )
            return candidates

    @mock.patch("botorch.optim.optimize.project_to_feasible_space_via_slsqp")
    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    def test_optimize_acqf_all_infeasible_candidates(
        self, mock_gen_batch_initial_conditions, mock_project
    ) -> None:
        # Check for error when all batches of candidates are infeasible w.r.t
        # parameter constraints and projecting fails (e.g. returns the same
        # points)
        mock_project.side_effect = lambda X, **kwargs: X
        q = 3
        num_restarts = 2
        for dtype in (torch.float, torch.double):
            with self.assertRaisesRegex(
                CandidateGenerationError, "infeasible candidates. 2 out of 2"
            ):
                self._test_optimize_acqf_infeasible_candidates(
                    mock_gen_batch_initial_conditions=mock_gen_batch_initial_conditions,
                    q=q,
                    num_restarts=num_restarts,
                    ics=torch.rand(num_restarts, q, 1, dtype=dtype, device=self.device),
                )

    @mock.patch("botorch.optim.optimize.gen_batch_initial_conditions")
    def test_optimize_acqf_some_infeasible_candidates(
        self, mock_gen_batch_initial_conditions
    ) -> None:
        # Check that no error is raised when the first batch of candidates
        # contains points that are infeasible w.r.t parameter constraints, but
        # second batch is feasible
        q = 3
        num_restarts = 2

        for dtype in (torch.float, torch.double):
            ics = torch.rand(num_restarts, q, 1, dtype=dtype, device=self.device)
            ics[1, ..., 0] = 3.0  # make second batch feasible
            candidates = self._test_optimize_acqf_infeasible_candidates(
                mock_gen_batch_initial_conditions=mock_gen_batch_initial_conditions,
                q=q,
                num_restarts=num_restarts,
                ics=ics,
            )
            self.assertTrue(torch.equal(candidates, ics[1]))

    def _setup_projection_test(
        self,
        d: int,
        dtype: torch.dtype,
    ) -> tuple[MockAcquisitionFunction, torch.Tensor]:
        """Helper method to set up common projection test components."""
        mock_acq_function = MockAcquisitionFunction()
        bounds = torch.zeros(2, d, dtype=dtype, device=self.device)
        bounds[1] = 2
        return mock_acq_function, bounds

    def _verify_projection_called_and_constraints_satisfied(
        self,
        mock_project_slsqp: mock.Mock,
        candidates: torch.Tensor,
        bounds: torch.Tensor,
        constraint_checks: list[Callable] | None = None,
    ) -> None:
        """Helper method to verify calls and constraint satisfaction."""
        # Verify that project_to_feasible_space_via_slsqp was called
        mock_project_slsqp.assert_called_once()

        # Verify that projected candidates are within bounds
        self.assertTrue(torch.all(candidates >= bounds[0] - 1e-6))
        self.assertTrue(torch.all(candidates <= bounds[1] + 1e-6))

        # Run custom constraint checks if provided
        if constraint_checks:
            for check in constraint_checks:
                check(candidates)

    def _run_optimize_acqf_with_projection(
        self,
        mock_minimize: mock.Mock,
        mock_acq_function: MockAcquisitionFunction,
        bounds: Tensor,
        infeasible_candidates: Tensor,
        q: int,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
        should_raise_error: bool = False,
        error_regex: str | None = None,
        post_processing_func: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor | None:
        """Helper method to run optimize_acqf with common setup."""
        # Mock scipy minimize to return infeasible candidates
        mock_minimize.return_value = OptimizeResult(
            x=infeasible_candidates.view(-1).cpu().numpy(),
            success=True,
            status=0,
        )

        kwargs = {
            "acq_function": mock_acq_function,
            "bounds": bounds,
            "q": q,
            "num_restarts": 1,
            "raw_samples": 2,
            "batch_initial_conditions": torch.rand(
                1, q, bounds.shape[1], dtype=bounds.dtype, device=self.device
            ),
            "post_processing_func": post_processing_func,
        }

        if inequality_constraints is not None:
            kwargs["inequality_constraints"] = inequality_constraints
        if equality_constraints is not None:
            kwargs["equality_constraints"] = equality_constraints
        if nonlinear_inequality_constraints is not None:
            kwargs["nonlinear_inequality_constraints"] = (
                nonlinear_inequality_constraints
            )
            kwargs["options"] = {"batch_limit": 1}
            kwargs["ic_generator"] = lambda *args, **kwargs: torch.rand(
                1, q, bounds.shape[1], dtype=bounds.dtype, device=self.device
            )
        if should_raise_error:
            with self.assertRaisesRegex(CandidateGenerationError, error_regex):
                optimize_acqf(**kwargs)
        else:
            candidates, _ = optimize_acqf(**kwargs)
            return candidates

    @mock.patch(
        "botorch.optim.optimize.project_to_feasible_space_via_slsqp",
        wraps=project_to_feasible_space_via_slsqp,
    )
    @mock.patch("botorch.generation.gen.minimize_with_timeout")
    def test_optimize_acqf_projection_applied(
        self, mock_minimize: mock.Mock, mock_project_slsqp: mock.Mock
    ) -> None:
        """Test that project_to_feasible_space_via_slsqp is correctly applied."""
        for dtype in (torch.float, torch.double):
            mock_project_slsqp.reset_mock()
            # Create infeasible candidates that violate constraint x[0] + x[1] >= 1.5
            infeasible_candidates = torch.tensor(
                [[0.5, 0.5], [0.3, 0.7]], device=self.device, dtype=dtype
            )

            mock_acq_function, bounds = self._setup_projection_test(d=2, dtype=dtype)

            # Define constraints: x[0] + x[1] >= 1.5
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]

            candidates = self._run_optimize_acqf_with_projection(
                mock_minimize,
                mock_acq_function,
                bounds,
                infeasible_candidates,
                q=2,
                inequality_constraints=inequality_constraints,
            )

            def check_inequality_constraint(candidates):
                for i in range(candidates.shape[0]):
                    constraint_value = candidates[i, 0] + candidates[i, 1]
                    self.assertGreaterEqual(constraint_value, 1.5 - 1e-6)

            self._verify_projection_called_and_constraints_satisfied(
                mock_project_slsqp=mock_project_slsqp,
                candidates=candidates,
                bounds=bounds,
                constraint_checks=[check_inequality_constraint],
            )

    @mock.patch(
        "botorch.optim.parameter_constraints.project_to_feasible_space_via_slsqp"
    )
    @mock.patch("botorch.generation.gen.minimize_with_timeout")
    def test_optimize_acqf_projection_not_applied_with_nonlinear_constraints(
        self, mock_minimize: mock.Mock, mock_project_slsqp: mock.Mock
    ) -> None:
        """Test that projection is not applied when nonlinear constraints exist."""
        for dtype in (torch.float, torch.double):
            mock_project_slsqp.reset_mock()
            infeasible_candidates = torch.tensor(
                [[0.5, 0.5]], dtype=dtype, device=self.device
            )

            mock_acq_function, bounds = self._setup_projection_test(d=2, dtype=dtype)

            # Define both linear and nonlinear constraints
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]

            def callable_constraint(x):
                return x.sum(dim=-1)

            nonlinear_inequality_constraints = [(callable_constraint, True)]

            # Test that optimization should raise error due to infeasible candidates
            # when nonlinear constraints are present (projection not applied)
            self._run_optimize_acqf_with_projection(
                mock_minimize,
                mock_acq_function,
                bounds,
                infeasible_candidates,
                q=1,
                inequality_constraints=inequality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                should_raise_error=True,
                error_regex="infeasible candidates",
            )

            # Verify that project_to_feasible_space_via_slsqp was NOT called
            mock_project_slsqp.assert_not_called()

    @mock.patch(
        "botorch.optim.optimize.project_to_feasible_space_via_slsqp",
        wraps=project_to_feasible_space_via_slsqp,
    )
    @mock.patch("botorch.generation.gen.minimize_with_timeout")
    def test_optimize_acqf_projection_with_equality_constraints(
        self, mock_minimize: mock.Mock, mock_project_slsqp: mock.Mock
    ) -> None:
        """Test projection with equality constraints."""
        # Create candidates that violate equality constraint x[0] + x[1] = 1.5
        for dtype in (torch.float, torch.double):
            mock_project_slsqp.reset_mock()
            infeasible_candidates = torch.tensor(
                [[1.0, 1.0]], dtype=dtype, device=self.device
            )

            mock_acq_function, bounds = self._setup_projection_test(d=2, dtype=dtype)

            # Define equality constraint: x[0] + x[1] = 1.5
            equality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.5,
                )
            ]

            candidates = self._run_optimize_acqf_with_projection(
                mock_minimize,
                mock_acq_function,
                bounds,
                infeasible_candidates,
                q=1,
                equality_constraints=equality_constraints,
            )

            def check_equality_constraint(candidates):
                constraint_value = (candidates[0, 0] + candidates[0, 1]).item()
                self.assertAllClose(constraint_value, 1.5, atol=1e-6)

            self._verify_projection_called_and_constraints_satisfied(
                mock_project_slsqp=mock_project_slsqp,
                candidates=candidates,
                bounds=bounds,
                constraint_checks=[check_equality_constraint],
            )

    @mock.patch(
        "botorch.optim.optimize.project_to_feasible_space_via_slsqp",
        wraps=project_to_feasible_space_via_slsqp,
    )
    @mock.patch("botorch.generation.gen.minimize_with_timeout")
    def test_optimize_acqf_projection_inequality_constraints_only(
        self, mock_minimize: mock.Mock, mock_project_slsqp: mock.Mock
    ) -> None:
        """Test projection with inequality constraints only."""
        for dtype in (torch.float, torch.double):
            mock_project_slsqp.reset_mock()
            # Create infeasible candidates that violate multiple inequality constraints
            infeasible_candidates = torch.tensor(
                [[0.2, 0.3, 0.4], [0.1, 0.5, 0.6]], dtype=dtype, device=self.device
            )

            mock_acq_function, bounds = self._setup_projection_test(d=3, dtype=dtype)

            # Define multiple inequality constraints:
            # x[0] + x[1] >= 1.0 and x[1] + x[2] >= 1.2
            inequality_constraints = [
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.0,
                ),
                (
                    torch.tensor([1, 2], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    1.2,
                ),
            ]

            candidates = self._run_optimize_acqf_with_projection(
                mock_minimize,
                mock_acq_function,
                bounds,
                infeasible_candidates,
                q=2,
                inequality_constraints=inequality_constraints,
            )

            def check_inequality_constraints(candidates):
                for i in range(candidates.shape[0]):
                    # First constraint: x[0] + x[1] >= 1.0
                    constraint1_value = (candidates[i, 0] + candidates[i, 1]).item()
                    self.assertGreaterEqual(constraint1_value, 1.0 - 1e-6)

                    # Second constraint: x[1] + x[2] >= 1.2
                    constraint2_value = (candidates[i, 1] + candidates[i, 2]).item()
                    self.assertGreaterEqual(constraint2_value, 1.2 - 1e-6)

            self._verify_projection_called_and_constraints_satisfied(
                mock_project_slsqp=mock_project_slsqp,
                candidates=candidates,
                bounds=bounds,
                constraint_checks=[check_inequality_constraints],
            )
        # test that post_processing_func is applied after projection
        candidates = self._run_optimize_acqf_with_projection(
            mock_minimize,
            mock_acq_function,
            bounds,
            infeasible_candidates,
            q=2,
            inequality_constraints=[
                (
                    torch.tensor([0, 1], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    2.5,
                )
            ],
            post_processing_func=lambda x: x.ceil(),
        )
        expected_candidates = torch.ones_like(candidates)
        expected_candidates[:, :2] = 2
        self.assertTrue(torch.equal(candidates, expected_candidates))

    @mock.patch(
        "botorch.optim.optimize.project_to_feasible_space_via_slsqp",
        wraps=project_to_feasible_space_via_slsqp,
    )
    @mock.patch("botorch.generation.gen.minimize_with_timeout")
    def test_optimize_acqf_projection_mixed_constraints(
        self, mock_minimize: mock.Mock, mock_project_slsqp: mock.Mock
    ) -> None:
        """Test projection with both inequality and equality constraints."""
        for dtype in (torch.float, torch.double):
            mock_project_slsqp.reset_mock()
            # Create infeasible candidates that violate both types of constraints
            infeasible_candidates = torch.tensor(
                [[0.5, 1.2, 0.8], [1.1, 0.7, 1.0]], dtype=dtype, device=self.device
            )

            mock_acq_function, bounds = self._setup_projection_test(d=3, dtype=dtype)

            # Define mixed constraints:
            # Inequality: x[0] + x[2] >= 2.0
            # Equality: x[0] + x[1] + x[2] = 3.5
            inequality_constraints = [
                (
                    torch.tensor([0, 2], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0], dtype=dtype, device=self.device),
                    2.0,
                )
            ]
            equality_constraints = [
                (
                    torch.tensor([0, 1, 2], dtype=torch.long, device=self.device),
                    torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=self.device),
                    3.5,
                )
            ]

            candidates = self._run_optimize_acqf_with_projection(
                mock_minimize,
                mock_acq_function,
                bounds,
                infeasible_candidates,
                q=2,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )

            def check_mixed_constraints(candidates):
                for i in range(candidates.shape[0]):
                    # Inequality constraint: x[0] + x[2] >= 2.0
                    inequality_value = candidates[i, 0] + candidates[i, 2]
                    self.assertGreaterEqual(inequality_value, 2.0 - 1e-6)

                    # Equality constraint: x[0] + x[1] + x[2] = 3.5
                    equality_value = (
                        candidates[i, 0] + candidates[i, 1] + candidates[i, 2]
                    ).item()
                    self.assertAllClose(equality_value, 3.5, atol=1e-6)

            self._verify_projection_called_and_constraints_satisfied(
                mock_project_slsqp, candidates, bounds, [check_mixed_constraints]
            )


class TestAllOptimizers(BotorchTestCase):
    @mock_optimize
    def test_negative_fixed_features(self) -> None:
        optim_funcs = {
            "optimize_acqf": partial(
                optimize_acqf,
                acq_function=MockAcquisitionFunction(),
                q=5,
            ),
            "optimize_acqf_cyclic": partial(
                optimize_acqf_cyclic,
                acq_function=MockAcquisitionFunction(),
                q=3,
            ),
            "optimize_acqf_mixed": partial(
                optimize_acqf_mixed,
                acq_function=MockAcquisitionFunction(),
                q=5,
            ),
            "optimize_acqf_list": partial(
                optimize_acqf_list,
                acq_function_list=[MockAcquisitionFunction()],
            ),
        }

        cases = [
            {
                "num_features": 2,
                "fixed_features_match": [
                    {"orig_idx": 0, "neg_idx": -2, "val": 0.1},
                ],
            },
            {
                "num_features": 2,
                "fixed_features_match": [
                    {"orig_idx": 1, "neg_idx": -1, "val": 0.1},
                ],
            },
            {
                "num_features": 5,
                "fixed_features_match": [
                    {"orig_idx": 0, "neg_idx": -5, "val": 0.1},
                    {"orig_idx": 3, "neg_idx": 3, "val": 0.2},
                    {"orig_idx": 4, "neg_idx": -1, "val": 0.3},
                ],
            },
            {
                "num_features": 7,
                "fixed_features_match": [
                    {"orig_idx": 0, "neg_idx": -7, "val": 0.1},
                    {"orig_idx": 1, "neg_idx": -6, "val": 0.2},
                    {"orig_idx": 2, "neg_idx": -5, "val": 0.3},
                    {"orig_idx": 3, "neg_idx": -4, "val": 0.4},
                    {"orig_idx": 4, "neg_idx": -3, "val": 0.5},
                    {"orig_idx": 5, "neg_idx": -2, "val": 0.6},
                ],
            },
            {
                "num_features": 3,
                "fixed_features_match": [
                    {"orig_idx": 0, "neg_idx": 0, "val": 0.1},
                    {"orig_idx": 1, "neg_idx": 1, "val": 0.2},
                ],
            },
        ]

        for name, func in optim_funcs.items():
            for case in cases:
                num_features = case["num_features"]
                fixed_features_match = case["fixed_features_match"]

                # optimize_acqf_mixed raises an error if fixed_features_list
                # is the same length as num_features
                if name == "optimize_acqf_mixed":
                    if len(fixed_features_match) == num_features:
                        fixed_features_match.pop(0)

                # Common opt args
                common_kwargs = {
                    "bounds": torch.tensor(
                        [[0.0] * num_features, [1.0] * num_features], device=self.device
                    ),
                    "num_restarts": 2,
                    "raw_samples": 10,
                }

                # Verify that this is the correct mapping of indices
                test_list = list(range(num_features))
                for a in fixed_features_match:
                    self.assertEqual(test_list[a["orig_idx"]], test_list[a["neg_idx"]])

                # Create fixed_features with non-negative indices and negative indices
                nonneg_fixed_features = {
                    a["orig_idx"]: a["val"] for a in fixed_features_match
                }
                neg_fixed_features = {
                    a["neg_idx"]: a["val"] for a in fixed_features_match
                }

                # Setup separate kwargs for the two fixed_features
                nonneg_func_kwargs = common_kwargs.copy()
                neg_func_kwargs = common_kwargs.copy()

                # If optimize_acqf_mixed, then need to pass fixed_features_list
                if name in ["optimize_acqf_mixed"]:
                    nonneg_func_kwargs["fixed_features_list"] = [nonneg_fixed_features]
                    neg_func_kwargs["fixed_features_list"] = [neg_fixed_features]
                else:
                    nonneg_func_kwargs["fixed_features"] = nonneg_fixed_features
                    neg_func_kwargs["fixed_features"] = neg_fixed_features

                # Run the optimization with both fixed_features
                nonneg_x, acq = func(**nonneg_func_kwargs)
                neg_x, acq = func(**neg_func_kwargs)

                # Verify that the fixed_features are the same and equal to val
                # for the original index values
                nonneg_x = nonneg_x[0]
                neg_x = neg_x[0]

                for a in fixed_features_match:
                    self.assertAlmostEqual(nonneg_x[a["orig_idx"]], a["val"], places=4)
                    self.assertAlmostEqual(neg_x[a["orig_idx"]], a["val"], places=4)


class TestOptimizeAcqfCyclic(BotorchTestCase):
    @mock.patch("botorch.optim.optimize._optimize_acqf")  # noqa: C901
    # TODO: make sure this runs without mock
    def test_optimize_acqf_cyclic(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        num_cycles = 2
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        inequality_constraints = [
            [torch.tensor([2], dtype=int), torch.tensor([4.0]), torch.tensor(5.0)]
        ]
        mock_acq_function = MockAcquisitionFunction()
        for q, dtype in itertools.product([1, 3], (torch.float, torch.double)):
            tkwargs["dtype"] = dtype
            inequality_constraints = [
                (
                    # indices can't be floats or doubles
                    inequality_constraints[0][0],
                    inequality_constraints[0][1].to(**tkwargs),
                    inequality_constraints[0][2].to(**tkwargs),
                )
            ]
            mock_optimize_acqf.reset_mock()
            bounds = bounds.to(**tkwargs)
            candidate_rvs = []
            acq_val_rvs = []
            for cycle_j in range(num_cycles):
                gcs_return_vals = [
                    (torch.rand(1, 3, **tkwargs), torch.rand(1, **tkwargs))
                    for _ in range(q)
                ]
                if cycle_j == 0:
                    # return `q` candidates for first call
                    candidate_rvs.append(
                        torch.cat([rv[0] for rv in gcs_return_vals], dim=-2)
                    )
                    acq_val_rvs.append(torch.cat([rv[1] for rv in gcs_return_vals]))
                else:
                    # return 1 candidate for subsequent calls
                    for rv in gcs_return_vals:
                        candidate_rvs.append(rv[0])
                        acq_val_rvs.append(rv[1])
            mock_optimize_acqf.side_effect = list(zip(candidate_rvs, acq_val_rvs))
            orig_candidates = candidate_rvs[0].clone()
            # wrap the set_X_pending method for checking that call arguments
            with mock.patch.object(
                MockAcquisitionFunction,
                "set_X_pending",
                wraps=mock_acq_function.set_X_pending,
            ) as mock_set_X_pending:
                candidates, _ = optimize_acqf_cyclic(
                    acq_function=mock_acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                    inequality_constraints=inequality_constraints,
                    post_processing_func=rounding_func,
                    cyclic_options={"maxiter": num_cycles},
                )
            # check that X_pending is set correctly in cyclic optimization
            if q > 1:
                x_pending_call_args_list = mock_set_X_pending.call_args_list
                idxr = torch.ones(q, dtype=torch.bool, device=self.device)
                for i in range(len(x_pending_call_args_list) - 1):
                    idxr[i] = 0
                    self.assertTrue(
                        torch.equal(
                            x_pending_call_args_list[i][0][0], orig_candidates[idxr]
                        )
                    )
                    idxr[i] = 1
                    orig_candidates[i] = candidate_rvs[i + 1]
                # check reset to base_X_pendingg
                self.assertIsNone(x_pending_call_args_list[-1][0][0])
            else:
                mock_set_X_pending.assert_not_called()
            # check final candidates
            expected_candidates = (
                torch.cat(candidate_rvs[-q:], dim=0) if q > 1 else candidate_rvs[0]
            )
            self.assertTrue(torch.equal(candidates, expected_candidates))
            # check call arguments for optimize_acqf
            call_args_list = mock_optimize_acqf.call_args_list
            expected_call_args = {
                "acq_function": mock_acq_function,
                "bounds": bounds,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "inequality_constraints": inequality_constraints,
                "equality_constraints": None,
                "fixed_features": None,
                "post_processing_func": rounding_func,
                "return_best_only": True,
                "sequential": True,
            }
            orig_candidates = candidate_rvs[0].clone()
            for i in range(len(call_args_list)):
                if i == 0:
                    # first cycle
                    expected_call_args.update(
                        {"batch_initial_conditions": None, "q": q}
                    )
                else:
                    expected_call_args.update(
                        {"batch_initial_conditions": orig_candidates[i - 1 : i], "q": 1}
                    )
                    orig_candidates[i - 1] = candidate_rvs[i]
                for k, v in call_args_list[i][1].items():
                    if torch.is_tensor(v):
                        self.assertTrue(torch.equal(expected_call_args[k], v))
                    elif k == "acq_function":
                        self.assertIsInstance(
                            mock_acq_function, MockAcquisitionFunction
                        )
                    else:
                        self.assertEqual(expected_call_args[k], v)


class TestOptimizeAcqfList(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    @mock.patch("botorch.optim.optimize.optimize_acqf_mixed")
    def test_optimize_acqf_list(self, mock_optimize_acqf, mock_optimize_acqf_mixed):
        num_restarts = 2
        raw_samples = 10
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        inequality_constraints = [
            [torch.tensor([3]), torch.tensor([4]), torch.tensor(5)]
        ]
        # reinitialize so that dtype
        mock_acq_function_1 = MockAcquisitionFunction()
        mock_acq_function_2 = MockAcquisitionFunction()
        mock_acq_function_list = [mock_acq_function_1, mock_acq_function_2]
        fixed_features_list = [None, [{0: 0.5}]]
        for ffl in fixed_features_list:
            for num_acqf, dtype in itertools.product(
                [1, 2], (torch.float, torch.double)
            ):
                for m in mock_acq_function_list:
                    # clear previous X_pending
                    m.set_X_pending(None)
                tkwargs["dtype"] = dtype
                inequality_constraints[0] = [
                    t.to(**tkwargs) for t in inequality_constraints[0]
                ]
                mock_optimize_acqf.reset_mock()
                mock_optimize_acqf_mixed.reset_mock()
                bounds = bounds.to(**tkwargs)
                candidate_rvs = []
                acq_val_rvs = []
                gcs_return_vals = [
                    (torch.rand(1, 3, **tkwargs), torch.rand(1, **tkwargs))
                    for _ in range(num_acqf)
                ]
                for rv in gcs_return_vals:
                    candidate_rvs.append(rv[0])
                    acq_val_rvs.append(rv[1])
                side_effect = list(zip(candidate_rvs, acq_val_rvs))
                mock_optimize_acqf.side_effect = side_effect
                mock_optimize_acqf_mixed.side_effect = side_effect
                orig_candidates = candidate_rvs[0].clone()
                # Wrap the set_X_pending method for checking that call arguments
                with (
                    mock.patch.object(
                        MockAcquisitionFunction,
                        "set_X_pending",
                        wraps=mock_acq_function_1.set_X_pending,
                    ) as mock_set_X_pending_1,
                    mock.patch.object(
                        MockAcquisitionFunction,
                        "set_X_pending",
                        wraps=mock_acq_function_2.set_X_pending,
                    ) as mock_set_X_pending_2,
                ):
                    candidates, _ = optimize_acqf_list(
                        acq_function_list=mock_acq_function_list[:num_acqf],
                        bounds=bounds,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options=options,
                        inequality_constraints=inequality_constraints,
                        post_processing_func=rounding_func,
                        fixed_features_list=ffl,
                    )
                    # check that X_pending is set correctly in sequential optimization
                    if num_acqf > 1:
                        x_pending_call_args_list = mock_set_X_pending_2.call_args_list
                        idxr = torch.ones(
                            num_acqf, dtype=torch.bool, device=self.device
                        )
                        for i in range(len(x_pending_call_args_list) - 1):
                            idxr[i] = 0
                            self.assertTrue(
                                torch.equal(
                                    x_pending_call_args_list[i][0][0],
                                    orig_candidates[idxr],
                                )
                            )
                            idxr[i] = 1
                            orig_candidates[i] = candidate_rvs[i + 1]
                    else:
                        mock_set_X_pending_1.assert_not_called()
                # check final candidates
                expected_candidates = (
                    torch.cat(candidate_rvs[-num_acqf:], dim=0)
                    if num_acqf > 1
                    else candidate_rvs[0]
                )
                self.assertTrue(torch.equal(candidates, expected_candidates))
                # check call arguments for optimize_acqf
                if ffl is None:
                    call_args_list = mock_optimize_acqf.call_args_list
                    expected_call_args = {
                        "acq_function": None,
                        "bounds": bounds,
                        "q": 1,
                        "num_restarts": num_restarts,
                        "raw_samples": raw_samples,
                        "options": options,
                        "inequality_constraints": inequality_constraints,
                        "equality_constraints": None,
                        "fixed_features": None,
                        "post_processing_func": rounding_func,
                        "batch_initial_conditions": None,
                        "return_best_only": True,
                        "sequential": False,
                    }
                else:
                    call_args_list = mock_optimize_acqf_mixed.call_args_list
                    expected_call_args = {
                        "acq_function": None,
                        "bounds": bounds,
                        "q": 1,
                        "num_restarts": num_restarts,
                        "raw_samples": raw_samples,
                        "options": options,
                        "inequality_constraints": inequality_constraints,
                        "equality_constraints": None,
                        "post_processing_func": rounding_func,
                        "batch_initial_conditions": None,
                        "fixed_features_list": ffl,
                    }
                for i in range(len(call_args_list)):
                    expected_call_args["acq_function"] = mock_acq_function_list[i]
                    for k, v in call_args_list[i][1].items():
                        if torch.is_tensor(v):
                            self.assertTrue(torch.equal(expected_call_args[k], v))
                        elif k == "acq_function":
                            self.assertIsInstance(
                                mock_acq_function_list[i], MockAcquisitionFunction
                            )
                        else:
                            self.assertEqual(expected_call_args[k], v)

    def test_optimize_acqf_list_empty_list(self):
        with self.assertRaises(ValueError):
            optimize_acqf_list(
                acq_function_list=[],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )

    def test_optimize_acqf_list_fixed_features(self):
        with self.assertRaises(ValueError):
            optimize_acqf_list(
                acq_function_list=[
                    MockAcquisitionFunction(),
                    MockAcquisitionFunction(),
                ],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                fixed_features_list=[{0: 0.5}],
                fixed_features={0: 0.5},
            )


class TestOptimizeAcqfMixed(BotorchTestCase):
    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    def test_optimize_acqf_mixed_q1(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        q = 1
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        mock_acq_function = MockAcquisitionFunction()
        for num_ff, dtype, return_best_only in itertools.product(
            [1, 3], (torch.float, torch.double), (True, False)
        ):
            tkwargs["dtype"] = dtype
            mock_optimize_acqf.reset_mock()
            bounds = bounds.to(**tkwargs)

            candidate_rvs = []
            acq_val_rvs = []
            for _ in range(num_ff):
                candidate_rvs.append(torch.rand(num_restarts, 1, 3, **tkwargs))
                acq_val_rvs.append(torch.rand(num_restarts, **tkwargs))
            fixed_features_list = [{i: i * 0.1} for i in range(num_ff)]
            side_effect = list(zip(candidate_rvs, acq_val_rvs))
            mock_optimize_acqf.side_effect = side_effect

            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=q,
                fixed_features_list=fixed_features_list,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                return_best_only=return_best_only,
                post_processing_func=rounding_func,
            )
            # compute expected output
            best_acq_values = torch.tensor(
                [torch.max(acq_values) for acq_values in acq_val_rvs]
            )
            best_batch_idx = torch.argmax(best_acq_values)

            if return_best_only:
                best_batch_candidates = candidate_rvs[best_batch_idx]
                best_batch_acq_values = acq_val_rvs[best_batch_idx]
                best_idx = torch.argmax(best_batch_acq_values)
                expected_candidates = best_batch_candidates[best_idx]
                expected_acq_value = best_batch_acq_values[best_idx]
                self.assertEqual(expected_candidates.dim(), 2)

            else:
                expected_candidates = candidate_rvs[best_batch_idx]
                expected_acq_value = acq_val_rvs[best_batch_idx]
                self.assertEqual(expected_candidates.dim(), 3)
                self.assertEqual(expected_acq_value.dim(), 1)

            self.assertTrue(torch.equal(candidates, expected_candidates))
            self.assertTrue(torch.equal(acq_value, expected_acq_value))
            # check call arguments for optimize_acqf
            call_args_list = mock_optimize_acqf.call_args_list
            expected_call_args = {
                "acq_function": None,
                "bounds": bounds,
                "q": q,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": options,
                "inequality_constraints": None,
                "equality_constraints": None,
                "fixed_features": None,
                "gen_candidates": None,
                "post_processing_func": rounding_func,
                "batch_initial_conditions": None,
                "return_best_only": False,
                "sequential": False,
                "ic_generator": None,
                "timeout_sec": None,
                "retry_on_optimization_warning": True,
                "nonlinear_inequality_constraints": None,
            }
            for i in range(len(call_args_list)):
                expected_call_args["fixed_features"] = fixed_features_list[i]
                for k, v in call_args_list[i][1].items():
                    if torch.is_tensor(v):
                        self.assertTrue(torch.equal(expected_call_args[k], v))
                    elif k == "acq_function":
                        self.assertIsInstance(v, MockAcquisitionFunction)
                    else:
                        self.assertEqual(expected_call_args[k], v)

    @mock.patch("botorch.optim.optimize.optimize_acqf")  # noqa: C901
    def test_optimize_acqf_mixed_q2(self, mock_optimize_acqf):
        num_restarts = 2
        raw_samples = 10
        q = 2
        options = {}
        tkwargs = {"device": self.device}
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        mock_acq_functions = [
            MockAcquisitionFunction(),
            MockOneShotEvaluateAcquisitionFunction(),
        ]
        for num_ff, dtype, mock_acq_function in itertools.product(
            [1, 3], (torch.float, torch.double), mock_acq_functions
        ):
            tkwargs["dtype"] = dtype
            mock_optimize_acqf.reset_mock()
            bounds = bounds.to(**tkwargs)

            fixed_features_list = [{i: i * 0.1} for i in range(num_ff)]
            candidate_rvs, exp_candidates, acq_val_rvs = [], [], []
            # generate mock side effects and compute expected outputs
            for _ in range(q):
                candidate_rvs_q = [
                    torch.rand(num_restarts, 1, 3, **tkwargs) for _ in range(num_ff)
                ]
                acq_val_rvs_q = [
                    torch.rand(num_restarts, **tkwargs) for _ in range(num_ff)
                ]

                best_acq_values = torch.tensor(
                    [torch.max(acq_values) for acq_values in acq_val_rvs_q]
                )
                best_batch_idx = torch.argmax(best_acq_values)

                best_batch_candidates = candidate_rvs_q[best_batch_idx]
                best_batch_acq_values = acq_val_rvs_q[best_batch_idx]
                best_idx = torch.argmax(best_batch_acq_values)

                exp_candidates.append(best_batch_candidates[best_idx])

                candidate_rvs += candidate_rvs_q
                acq_val_rvs += acq_val_rvs_q
            side_effect = list(zip(candidate_rvs, acq_val_rvs))
            mock_optimize_acqf.side_effect = side_effect

            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=q,
                fixed_features_list=fixed_features_list,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                post_processing_func=rounding_func,
            )

            expected_candidates = torch.cat(exp_candidates, dim=-2)
            if isinstance(mock_acq_function, MockOneShotEvaluateAcquisitionFunction):
                expected_acq_value = mock_acq_function.evaluate(
                    expected_candidates, bounds=bounds
                )
            else:
                expected_acq_value = mock_acq_function(expected_candidates)
            self.assertTrue(torch.equal(candidates, expected_candidates))
            self.assertTrue(torch.equal(acq_value, expected_acq_value))

    def test_optimize_acqf_mixed_empty_ff(self):
        mock_acq_function = MockAcquisitionFunction()
        with self.assertRaisesRegex(
            ValueError, expected_regex="fixed_features_list must be non-empty."
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )

    def test_optimize_acqf_mixed_inter_point_inequality_constraints(self):
        mock_acq_function = MockAcquisitionFunction()
        with self.assertRaisesRegex(
            UnsupportedError,
            expected_regex="Inter-point constraints are not supported for sequential "
            "optimization. But the 0th linear inequality constraint is defined "
            "as inter-point.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[{0: 0.0}],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                inequality_constraints=[
                    (  # Inter-point constraint: X[0, 0] - X[1, 0] >= 0
                        torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
                        torch.tensor([1.0, -1.0]),
                        0.0,
                    )
                ],
            )

    def test_optimize_acqf_mixed_inter_point_equality_constraints(self):
        mock_acq_function = MockAcquisitionFunction()
        with self.assertRaisesRegex(
            UnsupportedError,
            expected_regex="Inter-point constraints are not supported for sequential "
            "optimization. But the 0th linear equality constraint is defined "
            "as inter-point.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[{0: 0.0}],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                equality_constraints=[
                    (  # Inter-point constraint: X[0, 0] - X[1, 0] == 0
                        torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
                        torch.tensor([1.0, -1.0]),
                        0.0,
                    )
                ],
            )

    def test_optimize_acqf_mixed_inter_point_nonlinear_constraints(self):
        mock_acq_function = MockAcquisitionFunction()
        with self.assertRaisesRegex(
            UnsupportedError,
            expected_regex="Inter-point constraints are not supported for sequential "
            "optimization. But the 0th non-linear inequality constraint is defined "
            "as inter-point.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[{0: 0.0}],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                nonlinear_inequality_constraints=[
                    (  # Inter-point constraint: sum of all points >= 0
                        lambda X: X.sum(dim=(-1, -2)),
                        False,  # False indicates inter-point constraint
                    )
                ],
            )

    def test_optimize_acqf_mixed_return_best_only_q2(self):
        mock_acq_function = MockAcquisitionFunction()
        with self.assertRaisesRegex(
            NotImplementedError,
            expected_regex="`return_best_only=False` is only supported for q=1.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=2,
                fixed_features_list=[{0: 0.0}],
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
                return_best_only=False,
            )

    def test_optimize_acqf_one_shot_large_q(self):
        mock_acq_function = MockOneShotAcquisitionFunction()
        fixed_features_list = [{i: i * 0.1} for i in range(2)]
        with self.assertRaisesRegex(UnsupportedError, "OneShotAcquisitionFunction"):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=2,
                fixed_features_list=fixed_features_list,
                bounds=torch.stack([torch.zeros(3), 4 * torch.ones(3)]),
                num_restarts=2,
                raw_samples=10,
            )

    def test_optimize_acqf_mixed_ff_with_constraint(self):
        mock_acq_function = MockAcquisitionFunction()
        bounds = torch.stack([torch.zeros(3), 4 * torch.ones(3)])
        ineq_constraints = [
            (torch.zeros(1, dtype=torch.long), torch.ones(1), 1)
        ]  # x[0] >= 1
        with self.assertWarnsRegex(
            OptimizationWarning,
            "Candidate generation failed for 1 combinations of `fixed_features`. "
            "To suppress this warning, make sure all equality/inequality "
            "constraints can be satisfied by all `fixed_features` in "
            "`fixed_features_list`.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[{0: 0}, {0: 1}],
                bounds=bounds,
                num_restarts=2,
                raw_samples=10,
                inequality_constraints=ineq_constraints,
            )
        # No fixed features satisfy the constraint
        with self.assertRaisesRegex(
            CandidateGenerationError,
            "Candidate generation failed for all `fixed_features`.",
        ):
            optimize_acqf_mixed(
                acq_function=mock_acq_function,
                q=1,
                fixed_features_list=[{0: 0}],
                bounds=bounds,
                num_restarts=2,
                raw_samples=10,
                inequality_constraints=ineq_constraints,
            )


class TestOptimizeAcqfDiscrete(BotorchTestCase):
    def test_optimize_acqf_discrete(self):
        for q, dtype in itertools.product((1, 2), (torch.float, torch.double)):
            tkwargs = {"device": self.device, "dtype": dtype}

            mock_acq_function = SquaredAcquisitionFunction()
            mock_acq_function.set_X_pending(None)
            # ensure proper raising of errors if no choices
            with self.assertRaisesRegex(InputDataError, "`choices` must be non-empty."):
                optimize_acqf_discrete(
                    acq_function=mock_acq_function,
                    q=q,
                    choices=torch.empty(0, 2),
                )

            choices = torch.rand(5, 2, **tkwargs)

            exp_acq_vals = mock_acq_function(choices)

            # test unique
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=q,
                choices=choices,
            )
            best_idcs = torch.topk(exp_acq_vals, q).indices
            expected_candidates = choices[best_idcs]
            expected_acq_value = exp_acq_vals[best_idcs].reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test non-unique (test does not properly use pending points)
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function, q=q, choices=choices, unique=False
            )
            best_idx = torch.argmax(exp_acq_vals)
            expected_candidates = choices[best_idx].repeat(q, 1)
            expected_acq_value = exp_acq_vals[best_idx].repeat(q).reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test max_batch_limit
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function, q=q, choices=choices, max_batch_size=3
            )
            best_idcs = torch.topk(exp_acq_vals, q).indices
            expected_candidates = choices[best_idcs]
            expected_acq_value = exp_acq_vals[best_idcs].reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

            # test max_batch_limit & unique
            candidates, acq_value = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=q,
                choices=choices,
                unique=False,
                max_batch_size=3,
            )
            best_idx = torch.argmax(exp_acq_vals)
            expected_candidates = choices[best_idx].repeat(q, 1)
            expected_acq_value = exp_acq_vals[best_idx].repeat(q).reshape_as(acq_value)
            self.assertAllClose(acq_value, expected_acq_value)
            self.assertAllClose(candidates, expected_candidates)

        acqf = MockOneShotAcquisitionFunction()
        with self.assertRaisesRegex(UnsupportedError, "one-shot acquisition"):
            optimize_acqf_discrete(
                acq_function=acqf,
                q=1,
                choices=torch.tensor([[0.5], [0.2]]),
            )

    def test_optimize_acqf_discrete_X_avoid_and_constraints(self):
        # Check that choices are filtered correctly using X_avoid and constraints.
        tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        mock_acq_function = SquaredAcquisitionFunction()
        choices = torch.rand(2, 2, **tkwargs)
        with self.assertRaisesRegex(InputDataError, "No feasible points"):
            optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=1,
                choices=choices,
                X_avoid=choices,
            )
        with self.assertWarnsRegex(OptimizationWarning, "Requested q=2 candidates"):
            candidates, _ = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=2,
                choices=choices,
                X_avoid=choices[:1],
            )
        self.assertAllClose(candidates, choices[1:])
        constraints = [
            (  # X[..., 0] >= 1.0
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.tensor([1.0], **tkwargs),
                1.0,
            )
        ]
        choices[0, 0] = 1.0
        with self.assertWarnsRegex(OptimizationWarning, "Requested q=2 candidates"):
            candidates, _ = optimize_acqf_discrete(
                acq_function=mock_acq_function,
                q=2,
                choices=choices,
                inequality_constraints=constraints,
            )
        self.assertAllClose(candidates, choices[:1])

    def test_optimize_acqf_discrete_local_search(self):
        for q, dtype in itertools.product((1, 2), (torch.float, torch.double)):
            tkwargs = {"device": self.device, "dtype": dtype}

            mock_acq_function = SquaredAcquisitionFunction()
            mock_acq_function.set_X_pending(None)
            discrete_choices = [
                torch.tensor([0, 1, 6], **tkwargs),
                torch.tensor([2, 3, 4], **tkwargs),
                torch.tensor([5, 6, 9], **tkwargs),
            ]

            # make sure we can find the global optimum
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 4, 9], **tkwargs))
            )
            if q > 1:  # there are three local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 3, 9], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([1, 4, 9], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([6, 4, 6], **tkwargs))
                )

            # same but with unique=False
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
                unique=False,
            )
            expected_candidates = torch.tensor([[6, 4, 9], [6, 4, 9]], **tkwargs)
            self.assertAllClose(candidates, expected_candidates[:q])

            # test X_avoid and batch_initial_conditions
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                X_avoid=torch.tensor([[6, 4, 9]], **tkwargs),
                batch_initial_conditions=torch.tensor([[0, 2, 5]], **tkwargs).unsqueeze(
                    1
                ),
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 3, 9], **tkwargs))
            )
            if q > 1:  # there are two local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 2, 9], **tkwargs))
                )

            # test inequality constraints
            inequality_constraints = [
                (
                    torch.tensor([2], device=self.device),
                    -1 * torch.ones(1, **tkwargs),
                    -6 * torch.ones(1, **tkwargs),
                )
            ]
            candidates, acq_value = optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=discrete_choices,
                raw_samples=1,
                num_restarts=1,
                inequality_constraints=inequality_constraints,
            )
            self.assertTrue(
                torch.allclose(candidates[0], torch.tensor([6, 4, 6], **tkwargs))
            )
            if q > 1:  # there are three local minima
                self.assertTrue(
                    torch.allclose(candidates[1], torch.tensor([6, 4, 5], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([6, 3, 6], **tkwargs))
                    or torch.allclose(candidates[1], torch.tensor([1, 4, 6], **tkwargs))
                )

            # make sure we break if there are no neighbors
            optimize_acqf_discrete_local_search(
                acq_function=mock_acq_function,
                q=q,
                discrete_choices=[
                    torch.tensor([0, 1], **tkwargs),
                    torch.tensor([1], **tkwargs),
                ],
                raw_samples=1,
                num_restarts=1,
            )

            # test _filter_infeasible
            X = torch.tensor([[0, 2, 5], [0, 2, 6], [0, 2, 9]], **tkwargs)
            X_filtered = _filter_infeasible(
                X=X, inequality_constraints=inequality_constraints
            )
            self.assertAllClose(X[:2], X_filtered)

            # test _filter_invalid
            X_filtered = _filter_invalid(X=X, X_avoid=X[1].unsqueeze(0))
            self.assertAllClose(X[[0, 2]], X_filtered)
            X_filtered = _filter_invalid(X=X, X_avoid=X[[0, 2]])
            self.assertAllClose(X[1].unsqueeze(0), X_filtered)

            # test _generate_neighbors
            X_loc = _generate_neighbors(
                x=torch.tensor([0, 2, 6], **tkwargs).unsqueeze(0),
                discrete_choices=discrete_choices,
                X_avoid=torch.tensor([[0, 3, 6], [0, 2, 5]], **tkwargs),
                inequality_constraints=inequality_constraints,
            )
            self.assertTrue(
                torch.allclose(
                    X_loc, torch.tensor([[1, 2, 6], [6, 2, 6], [0, 4, 6]], **tkwargs)
                )
            )

            # test ValueError for batch_initial_conditions shape
            with self.assertRaisesRegex(ValueError, "must have shape `n x 1 x d`"):
                candidates, _acq_value = optimize_acqf_discrete_local_search(
                    acq_function=mock_acq_function,
                    q=q,
                    discrete_choices=discrete_choices,
                    X_avoid=torch.tensor([[6, 4, 9]], **tkwargs),
                    batch_initial_conditions=torch.tensor([[0, 2, 5]], **tkwargs),
                )

            # test _gen_batch_initial_conditions_local_search
            with self.assertRaisesRegex(RuntimeError, "Failed to generate"):
                _gen_batch_initial_conditions_local_search(
                    discrete_choices=discrete_choices,
                    raw_samples=1,
                    X_avoid=torch.zeros(0, 3, **tkwargs),
                    inequality_constraints=[],
                    min_points=30,
                )

            X = _gen_batch_initial_conditions_local_search(
                discrete_choices=discrete_choices,
                raw_samples=1,
                X_avoid=torch.zeros(0, 3, **tkwargs),
                inequality_constraints=[],
                min_points=20,
            )
            self.assertEqual(len(X), 20)
            self.assertAllClose(torch.unique(X, dim=0), X)

    def test_no_precision_loss_with_fixed_features(self) -> None:
        acqf = SquaredAcquisitionFunction()

        val = 1e-1
        fixed_features_list = [{0: val}]

        bounds = torch.stack(
            [torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)]
        )
        candidate, _ = optimize_acqf_mixed(
            acqf,
            bounds=bounds,
            q=1,
            num_restarts=1,
            raw_samples=1,
            fixed_features_list=fixed_features_list,
        )
        self.assertEqual(candidate[0, 0].item(), val)


class TestOptimizeAcqfInputs(BotorchTestCase):
    def test_get_ic_generator(self):
        X = torch.rand(4, 3)
        Y1 = torch.rand(4, 1)
        Y2 = torch.rand(4, 1)
        m1 = SingleTaskGP(X, Y1)
        m2 = SingleTaskGP(X, Y2)
        model = ModelListGP(m1, m2)
        bounds = torch.zeros(2, 3)
        bounds[1] = 1
        kwargs = {
            "raw_samples": 2,
            "options": None,
            "inequality_constraints": None,
            "equality_constraints": None,
            "nonlinear_inequality_constraints": None,
            "fixed_features": None,
            "post_processing_func": None,
            "batch_initial_conditions": None,
            "return_best_only": False,
            "gen_candidates": gen_candidates_scipy,
            "sequential": False,
        }
        acqf = qExpectedImprovement(model=m1, best_f=0.0)
        opt_inputs = OptimizeAcqfInputs(
            acq_function=acqf, bounds=bounds, q=1, num_restarts=1, **kwargs
        )
        ic_generator = opt_inputs.get_ic_generator()
        self.assertIs(ic_generator, gen_batch_initial_conditions)
        acqf = qHypervolumeKnowledgeGradient(model=model, ref_point=torch.zeros(2))
        opt_inputs = OptimizeAcqfInputs(
            acq_function=acqf, bounds=bounds, q=1, num_restarts=1, **kwargs
        )
        ic_generator = opt_inputs.get_ic_generator()
        self.assertIs(ic_generator, gen_one_shot_hvkg_initial_conditions)

        acqf = qKnowledgeGradient(model=m1)
        opt_inputs = OptimizeAcqfInputs(
            acq_function=acqf, bounds=bounds, q=1, num_restarts=1, **kwargs
        )
        ic_generator = opt_inputs.get_ic_generator()
        self.assertIs(ic_generator, gen_one_shot_kg_initial_conditions)

        def my_gen():
            pass

        opt_inputs = OptimizeAcqfInputs(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=1,
            ic_generator=my_gen,
            **kwargs,
        )
        ic_generator = opt_inputs.get_ic_generator()
        self.assertIs(ic_generator, my_gen)
