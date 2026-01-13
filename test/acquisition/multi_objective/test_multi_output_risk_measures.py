#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    IndependentCVaR,
    IndependentVaR,
    MARS,
    MultiOutputExpectation,
    MultiOutputRiskMeasureMCObjective,
    MultiOutputWorstCase,
    MVaR,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.input import InputPerturbation
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


class NotSoAbstractMORiskMeasure(MultiOutputRiskMeasureMCObjective):
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.sum(dim=-2)


class TestMultiOutputRiskMeasureMCObjective(BotorchTestCase):
    def test_multi_output_risk_measure_mc_objective(self):
        # abstract raises
        with self.assertRaises(TypeError):
            MultiOutputRiskMeasureMCObjective(n_w=3)

        for dtype in (torch.float, torch.double):
            samples = torch.tensor(
                [
                    [
                        [1.0, 1.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 3.4],
                        [1.0, 1.2],
                        [5.0, 5.6],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            obj = NotSoAbstractMORiskMeasure(n_w=3)
            # test _prepare_samples
            expected_samples = samples.view(1, 2, 3, 2)
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # test batches
            samples = torch.rand(5, 3, 6, 3, device=self.device, dtype=dtype)
            expected_samples = samples.view(5, 3, 2, 3, 3)
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, expected_samples))
            # negating with preprocessing function
            obj = NotSoAbstractMORiskMeasure(
                n_w=3,
                preprocessing_function=WeightedMCMultiOutputObjective(
                    weights=torch.tensor(
                        [-1.0, -1.0, -1.0], device=self.device, dtype=dtype
                    )
                ),
            )
            prepared_samples = obj._prepare_samples(samples)
            self.assertTrue(torch.equal(prepared_samples, -expected_samples))


class TestMultiOutputExpectation(BotorchTestCase):
    def test_mo_expectation(self):
        obj = MultiOutputExpectation(n_w=3)
        for dtype in (torch.float, torch.double):
            obj = MultiOutputExpectation(n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 1.2],
                        [0.5, 0.5],
                        [1.5, 2.2],
                        [3.0, 1.2],
                        [1.0, 7.1],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[1.0, 1.3], [3.0, 4.7]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ first output negated
            obj.preprocessing_function = WeightedMCMultiOutputObjective(
                torch.tensor([-1.0, 1.0], device=self.device, dtype=dtype)
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[-1.0, 1.3], [-3.0, 4.7]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestIndependentCVaR(BotorchTestCase):
    def test_independent_cvar(self):
        obj = IndependentCVaR(alpha=0.5, n_w=3)
        self.assertEqual(obj.alpha_idx, 1)
        with self.assertRaises(ValueError):
            IndependentCVaR(alpha=3, n_w=3)
        for dtype in (torch.float, torch.double):
            obj = IndependentCVaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 1.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [1.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[0.75, 0.95], [2.0, 3.5]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ first output negated
            obj.preprocessing_function = WeightedMCMultiOutputObjective(
                torch.tensor([-1.0, 1.0], device=self.device, dtype=dtype)
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.allclose(
                    rm_samples,
                    torch.tensor(
                        [[[-1.5, 0.95], [-4.0, 3.5]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestIndependentVaR(BotorchTestCase):
    def test_independent_var(self):
        for dtype in (torch.float, torch.double):
            obj = IndependentVaR(alpha=0.5, n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 3.2],
                        [0.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [1.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[1.0, 2.2], [3.0, 5.8]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ weights
            obj.preprocessing_function = WeightedMCMultiOutputObjective(
                torch.tensor([0.5, -1.0], device=self.device, dtype=dtype)
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[0.5, -2.2], [1.5, -5.8]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestMultiOutputWorstCase(BotorchTestCase):
    def test_multi_output_worst_case(self):
        for dtype in (torch.float, torch.double):
            obj = MultiOutputWorstCase(n_w=3)
            samples = torch.tensor(
                [
                    [
                        [1.0, 3.2],
                        [5.5, 0.7],
                        [2.0, 2.2],
                        [3.0, 1.2],
                        [5.0, 7.2],
                        [5.0, 5.8],
                    ]
                ],
                device=self.device,
                dtype=dtype,
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[1.0, 0.7], [3.0, 1.2]]], device=self.device, dtype=dtype
                    ),
                )
            )
            # w/ weights
            obj.preprocessing_function = WeightedMCMultiOutputObjective(
                torch.tensor([-1.0, 2.0], device=self.device, dtype=dtype)
            )
            rm_samples = obj(samples)
            self.assertTrue(
                torch.equal(
                    rm_samples,
                    torch.tensor(
                        [[[-5.5, 1.4], [-5.0, 2.4]]], device=self.device, dtype=dtype
                    ),
                )
            )


class TestMVaR(BotorchTestCase):
    def test_mvar(self):
        with self.assertRaises(ValueError):
            MVaR(n_w=5, alpha=3.0)

        def set_equals(t1: Tensor, t2: Tensor) -> bool:
            r"""Check if two ``k x m``-dim tensors are equivalent after possibly
            reordering the ``k`` dimension. Ignores duplicate entries.
            """
            t1 = t1.unique(dim=0)
            t2 = t2.unique(dim=0)
            if t1.shape != t2.shape:
                return False
            equals_sum = (t1.unsqueeze(-2) == t2).all(dim=-1).sum(dim=-1)
            return torch.equal(equals_sum, torch.ones_like(equals_sum))

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            mvar = MVaR(n_w=5, alpha=0.6)
            # a simple negatively correlated example
            Y = torch.stack(
                [torch.linspace(1, 5, 5), torch.linspace(5, 1, 5)],
                dim=-1,
            ).to(**tkwargs)
            expected_set = torch.stack(
                [torch.linspace(1, 3, 3), torch.linspace(3, 1, 3)],
                dim=-1,
            ).to(Y)
            # check that both versions produce the correct set
            mvar_counting = mvar.get_mvar_set_via_counting(Y)[
                0
            ]  # returns a batch list of k x m
            mvar_vectorized = mvar.get_mvar_set_vectorized(Y)[
                0
            ]  # returns a batch list of k x m
            self.assertTrue(set_equals(mvar_counting, mvar_vectorized))
            self.assertTrue(set_equals(mvar_counting, expected_set))
            # check that the ``filter_dominated`` works correctly
            mvar = MVaR(
                n_w=5,
                alpha=0.4,
                filter_dominated=False,
            )
            # negating the input to treat large values as undesirable
            Y = -torch.tensor(
                [
                    [1, 4],
                    [2, 3],
                    [3, 2],
                    [4, 1],
                    [3.5, 3.5],
                ],
                **tkwargs,
            )
            mvar_counting = mvar.get_mvar_set_via_counting(Y)[0]
            mvar_vectorized = mvar.get_mvar_set_vectorized(Y)[0]
            self.assertTrue(set_equals(mvar_counting, mvar_vectorized))
            # negating here as well
            expected_w_dominated = -torch.tensor(
                [
                    [2, 4],
                    [3, 3],
                    [3.5, 3],
                    [3, 3.5],
                    [4, 2],
                ],
                **tkwargs,
            )
            self.assertTrue(set_equals(mvar_counting, expected_w_dominated))
            expected_non_dominated = expected_w_dominated[
                is_non_dominated(expected_w_dominated)
            ]
            mvar.filter_dominated = True
            mvar_counting = mvar.get_mvar_set_via_counting(Y)[0]
            mvar_vectorized = mvar.get_mvar_set_vectorized(Y)[0]
            self.assertTrue(set_equals(mvar_counting, mvar_vectorized))
            self.assertTrue(set_equals(mvar_counting, expected_non_dominated))

            # test batched w/ random input
            mvar = MVaR(
                n_w=10,
                alpha=0.5,
                filter_dominated=False,
            )
            Y = torch.rand(4, 10, 2, **tkwargs)
            mvar_counting = mvar.get_mvar_set_via_counting(Y)
            mvar_vectorized = mvar.get_mvar_set_vectorized(Y)
            # check that the two agree
            self.assertTrue(
                all(set_equals(mvar_counting[i], mvar_vectorized[i]) for i in range(4))
            )
            # check that the MVaR is dominated by ``alpha`` fraction (maximization).
            dominated_count = (
                (Y[0].unsqueeze(-2) >= mvar_counting[0]).all(dim=-1).sum(dim=0)
            )
            expected_count = (
                torch.ones(
                    mvar_counting[0].shape[0], device=self.device, dtype=torch.long
                )
                * 5
            )
            self.assertTrue(torch.equal(dominated_count, expected_count))

            # test forward pass
            for use_counting in (True, False):
                # with ``expectation=True``
                mvar = MVaR(
                    n_w=10,
                    alpha=0.5,
                    expectation=True,
                    use_counting=use_counting,
                )
                samples = torch.rand(2, 20, 2, **tkwargs)
                mvar_exp = mvar(samples)
                expected = [
                    mvar.get_mvar_set_via_counting(Y)[0].mean(dim=0)
                    for Y in samples.view(4, 10, 2)
                ]
                self.assertTrue(
                    torch.allclose(mvar_exp, torch.stack(expected).view(2, 2, 2))
                )

            # m > 2
            samples = torch.rand(2, 20, 3, **tkwargs)
            mvar_exp = mvar(samples)
            expected = [
                mvar.get_mvar_set_vectorized(Y)[0].mean(dim=0)
                for Y in samples.view(4, 10, 3)
            ]
            self.assertTrue(torch.equal(mvar_exp, torch.stack(expected).view(2, 2, 3)))

            # check that cpu code also works with m=3
            samples = samples.view(4, 10, 3)
            self.assertTrue(
                set_equals(
                    torch.cat(mvar.get_mvar_set_vectorized(samples)),
                    torch.cat(mvar.get_mvar_set_via_counting(samples)),
                )
            )

            # with ``expectation=False``
            mvar = MVaR(
                n_w=10,
                alpha=0.5,
                expectation=False,
                pad_to_n_w=True,
            )
            samples = torch.rand(2, 20, 2, **tkwargs)
            mvar_vals = mvar(samples)
            self.assertTrue(mvar_vals.shape == samples.shape)
            expected = [
                mvar.get_mvar_set_via_counting(Y)[0] for Y in samples.view(4, 10, 2)
            ]
            for i in range(4):
                batch_idx = i // 2
                q_idx_start = 10 * (i % 2)
                expected_ = expected[i]
                # check that the actual values are there
                self.assertTrue(
                    set_equals(
                        mvar_vals[
                            batch_idx, q_idx_start : q_idx_start + expected_.shape[0]
                        ],
                        expected_,
                    )
                )
                # check for correct padding
                self.assertTrue(
                    torch.equal(
                        mvar_vals[
                            batch_idx,
                            q_idx_start + expected_.shape[0] : q_idx_start + 10,
                        ],
                        mvar_vals[
                            batch_idx, q_idx_start + expected_.shape[0] - 1
                        ].expand(10 - expected_.shape[0], -1),
                    )
                )

            # Test the no-exact alpha level points case.
            # This happens when there are duplicates in the input.
            Y = torch.ones(10, 2, **tkwargs)
            mvar_counting = mvar.get_mvar_set_via_counting(Y)[0]
            mvar_vectorized = mvar.get_mvar_set_vectorized(Y)[0]
            self.assertTrue(torch.equal(mvar_counting, Y[:1]))
            self.assertTrue(torch.equal(mvar_vectorized, Y[:1]))

            # Check that the output has gradients.
            self.assertTrue(mvar(Y.requires_grad_()).requires_grad)


class TestMARS(BotorchTestCase):
    def test_init(self):
        # Init w/ defaults.
        mars = MARS(
            alpha=0.5,
            n_w=3,
            chebyshev_weights=[0.5, 0.5],
        )
        self.assertEqual(mars.alpha, 0.5)
        self.assertEqual(mars.n_w, 3)
        self.assertTrue(torch.equal(mars.chebyshev_weights, torch.tensor([0.5, 0.5])))
        self.assertIsNone(mars.baseline_Y)
        self.assertIsNone(mars.ref_point)
        self.assertIsInstance(
            mars.preprocessing_function, IdentityMCMultiOutputObjective
        )
        self.assertIsInstance(mars.mvar, MVaR)
        self.assertEqual(mars.mvar.alpha, 0.5)
        self.assertEqual(mars.mvar.n_w, 3)
        # Errors with Chebyshev weights.
        with self.assertRaisesRegex(UnsupportedError, "Negative"):
            MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[-0.5, 0.5],
            )
        with self.assertRaisesRegex(UnsupportedError, "Batched"):
            MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[[0.5], [0.5]],
            )
        # With optional arguments.
        baseline_Y = torch.rand(3, 2)
        ref_point = [3.0, 5.0]

        def dummy_func(Y):
            return Y

        mars = MARS(
            alpha=0.5,
            n_w=3,
            chebyshev_weights=[0.5, 0.5],
            baseline_Y=baseline_Y,
            ref_point=ref_point,
            preprocessing_function=dummy_func,
        )
        self.assertTrue(torch.equal(mars.baseline_Y, baseline_Y))
        self.assertTrue(torch.equal(mars.ref_point, torch.tensor(ref_point)))
        self.assertIs(mars.preprocessing_function, dummy_func)

    def test_set_baseline_Y(self):
        mars = MARS(
            alpha=0.5,
            n_w=3,
            chebyshev_weights=[0.5, 0.5],
        )
        perturbation = InputPerturbation(
            perturbation_set=torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        )
        model = GenericDeterministicModel(f=lambda X: X, num_outputs=2)
        model.input_transform = perturbation
        X_baseline = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        mars.set_baseline_Y(model=model, X_baseline=X_baseline)
        self.assertTrue(torch.equal(mars.baseline_Y, torch.tensor([[1.5, 1.5]])))
        # With Y_samples.
        mars._baseline_Y = None
        Y_samples = model.posterior(X_baseline).mean
        with warnings.catch_warnings(record=True) as ws:
            mars.set_baseline_Y(model=model, X_baseline=X_baseline, Y_samples=Y_samples)
        self.assertTrue(torch.equal(mars.baseline_Y, torch.tensor([[1.5, 1.5]])))
        self.assertTrue(any(w.category == BotorchWarning for w in ws))
        # With pre-processing function.
        mars = MARS(
            alpha=0.5,
            n_w=3,
            chebyshev_weights=[0.5, 0.5],
            preprocessing_function=lambda Y: -Y,
        )
        mars.set_baseline_Y(model=model, X_baseline=X_baseline)
        self.assertTrue(torch.equal(mars.baseline_Y, torch.tensor([[-0.5, -0.5]])))

    def test_get_Y_normalization_bounds(self):
        # Error if batched.
        with self.assertRaisesRegex(UnsupportedError, "Batched"):
            MARS._get_Y_normalization_bounds(Y=torch.rand(3, 5, 2))
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            # Empty Y.
            bounds = MARS._get_Y_normalization_bounds(Y=torch.empty(0, 3, **tkwargs))
            expected = torch.zeros(2, 3, **tkwargs)
            expected[1] = 1.0
            self.assertAllClose(bounds, expected)

            # Single point in pareto_Y.
            bounds = MARS._get_Y_normalization_bounds(Y=torch.zeros(1, 3, **tkwargs))
            self.assertAllClose(bounds, expected)

            # With reference point.
            bounds = MARS._get_Y_normalization_bounds(
                Y=torch.zeros(1, 3, **tkwargs), ref_point=-torch.ones(3)
            )
            self.assertAllClose(bounds, expected - 1)

            # Check that dominated points are ignored.
            Y = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]], **tkwargs)
            expected = expected[:, :2]
            expected[0] = 0.5
            bounds = MARS._get_Y_normalization_bounds(Y=Y)
            self.assertAllClose(bounds, expected)

            # Multiple pareto with ref point.
            # Nothing better than ref.
            bounds = MARS._get_Y_normalization_bounds(
                Y=Y, ref_point=torch.ones(2) * 0.75
            )
            self.assertAllClose(bounds, expected)

            # W/ points better than ref.
            Y = torch.tensor(
                [[0.5, 1.0], [1.0, 0.5], [0.8, 0.8], [0.9, 0.7]], **tkwargs
            )
            bounds = MARS._get_Y_normalization_bounds(
                Y=Y, ref_point=torch.ones(2) * 0.6
            )
            expected = torch.tensor([[0.6, 0.6], [0.9, 0.8]], **tkwargs)
            self.assertAllClose(bounds, expected)

    def test_chebyshev_objective(self):
        # Check that the objective is destroyed on setters.
        mars = MARS(
            alpha=0.5,
            n_w=3,
            chebyshev_weights=[0.5, 0.5],
            baseline_Y=torch.empty(0, 2),
        )
        self.assertIsNone(mars._chebyshev_objective)
        # Gets constructed on property access.
        self.assertIsNotNone(mars.chebyshev_objective)
        self.assertIsNotNone(mars._chebyshev_objective)
        # Destored on updating the weights.
        mars.chebyshev_weights = [0.5, 0.3]
        self.assertIsNone(mars._chebyshev_objective)
        # Destroyed on setting baseline_Y.
        mars.chebyshev_objective
        mars.baseline_Y = None
        self.assertIsNone(mars._chebyshev_objective)

        # Error if baseline_Y is not set.
        with self.assertRaisesRegex(RuntimeError, "baseline_Y"):
            MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[0.5, 0.5],
            ).chebyshev_objective

        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            # Without ref point or pre-processing.
            mars = MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[0.5, 0.5],
                baseline_Y=torch.tensor([[0.0, 0.5], [0.5, 0.0]], **tkwargs),
            )
            obj = mars.chebyshev_objective
            Y = torch.ones(2, 2, **tkwargs)
            self.assertAllClose(obj(Y), torch.ones(2, **tkwargs))
            # With pre-processing.
            mars = MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[0.5, 0.5],
                baseline_Y=torch.tensor([[0.0, 0.5], [0.5, 0.0]], **tkwargs),
                preprocessing_function=lambda Y: -Y,
            )
            obj = mars.chebyshev_objective
            Y = -torch.ones(2, 2, **tkwargs)
            self.assertAllClose(obj(Y), torch.ones(2, **tkwargs))
            # With ref point.
            mars = MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[0.5, 0.5],
                baseline_Y=torch.tensor([[0.0, 0.5], [0.5, 0.0]], **tkwargs),
                ref_point=[1.0, 1.0],
            )
            obj = mars.chebyshev_objective
            Y = torch.ones(2, 2, **tkwargs)
            self.assertAllClose(obj(Y), torch.zeros(2, **tkwargs))

    def test_end_to_end(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            mars = MARS(
                alpha=0.5,
                n_w=3,
                chebyshev_weights=[0.5, 0.5],
                ref_point=[1.0, 1.0],
                baseline_Y=torch.randn(5, 2, **tkwargs),
            )
            samples = torch.randn(5, 9, 2, **tkwargs)
            mars_vals = mars(samples)
            self.assertEqual(mars_vals.shape, torch.Size([5, 3]))
            self.assertEqual(mars_vals.dtype, dtype)
            self.assertEqual(mars_vals.device.type, self.device.type)
