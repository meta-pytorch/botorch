#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
When adding tests for a new input constructor, please add a new case to
`TestInstantiationFromInputConstructor.setUp`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import reduce

from random import randint
from unittest import mock
from unittest.mock import MagicMock

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfFeasibility,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
)
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.input_constructors import (
    _field_is_shared,
    _register_acqf_input_constructor,
    acqf_input_constructor,
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    construct_inputs_mf_base,
    get_acqf_input_constructor,
    get_best_f_analytic,
    get_best_f_mc,
    optimize_objective,
)
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)

from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
    TAU_MAX,
    TAU_RELU,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qLowerConfidenceBound,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)

from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MultiOutputExpectation,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.models import MultiTaskGP, SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.test_utils.mock import mock_optimize
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import Tensor


class DummyAcquisitionFunction(AcquisitionFunction): ...


class InputConstructorBaseTestCase(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.mock_model = MockModel(
            posterior=MockPosterior(mean=None, variance=None, base_shape=(1,))
        )

        X1 = torch.rand(3, 2)
        X2 = torch.rand(3, 2)
        Y1 = torch.rand(3, 1)
        Y2 = torch.rand(3, 1)
        feature_names = ["X1", "X2"]
        outcome_names = ["Y"]

        self.blockX_blockY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            )
        }
        self.blockX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X1, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.multiX_multiY = {
            0: SupervisedDataset(
                X1, Y1, feature_names=feature_names, outcome_names=outcome_names
            ),
            1: SupervisedDataset(
                X2, Y2, feature_names=feature_names, outcome_names=outcome_names
            ),
        }
        self.bounds = 2 * [(0.0, 1.0)]


class TestInputConstructorUtils(InputConstructorBaseTestCase):
    def test_field_is_shared(self) -> None:
        self.assertTrue(_field_is_shared(self.blockX_multiY, "X"))
        self.assertFalse(_field_is_shared(self.blockX_multiY, "Y"))
        with self.assertRaisesRegex(AttributeError, "has no field"):
            self.assertFalse(_field_is_shared(self.blockX_multiY, "foo"))

    def test_get_best_f_analytic(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "Currently only block designs are supported."
        ):
            get_best_f_analytic(training_data=self.multiX_multiY)

        best_f = get_best_f_analytic(training_data=self.blockX_blockY)
        self.assertEqual(best_f, get_best_f_analytic(self.blockX_blockY[0]))

        best_f_expected = self.blockX_blockY[0].Y.squeeze().max()
        self.assertEqual(best_f, best_f_expected)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Analytic acquisition functions currently only work with "
            "multi-output models if provided with a",
        ):
            get_best_f_analytic(training_data=self.blockX_multiY)
        weights = torch.rand(2)

        post_tf = ScalarizedPosteriorTransform(weights=weights)
        best_f_tf = get_best_f_analytic(
            training_data=self.blockX_multiY, posterior_transform=post_tf
        )

        multi_Y = torch.cat([d.Y for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = post_tf.evaluate(multi_Y).max()
        self.assertEqual(best_f_tf, best_f_expected)

    def test_get_best_f_mc(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "Currently only block designs are supported."
        ):
            get_best_f_mc(training_data=self.multiX_multiY)

        best_f = get_best_f_mc(training_data=self.blockX_blockY)
        self.assertEqual(best_f, get_best_f_mc(self.blockX_blockY[0]))

        best_f_expected = self.blockX_blockY[0].Y.max()
        self.assertAllClose(best_f, best_f_expected)
        with self.assertRaisesRegex(UnsupportedError, "require an objective"):
            get_best_f_mc(training_data=self.blockX_multiY)
        obj = LinearMCObjective(weights=torch.rand(2))
        best_f = get_best_f_mc(training_data=self.blockX_multiY, objective=obj)

        multi_Y = torch.cat([d.Y for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = (multi_Y @ obj.weights).max()
        self.assertAllClose(best_f, best_f_expected)
        post_tf = ScalarizedPosteriorTransform(weights=torch.ones(2))
        best_f = get_best_f_mc(
            training_data=self.blockX_multiY, posterior_transform=post_tf
        )
        best_f_expected = multi_Y.sum(dim=-1).max()
        self.assertAllClose(best_f, best_f_expected)

    @mock_optimize
    def test_optimize_objective(self) -> None:
        torch.manual_seed(randint(a=0, b=100))
        n = 4
        d = 3
        x = torch.rand(n, d, dtype=torch.double, device=self.device)
        y = torch.rand(n, 1, dtype=torch.double, device=self.device)
        model = SingleTaskGP(train_X=x, train_Y=y)

        bounds = torch.tensor(
            [[0.0, -0.01, -0.02], [1.0, 1.01, 1.02]],
            dtype=torch.double,
            device=self.device,
        )

        with self.subTest("scalarObjective_acquisitionFunction"):
            acq_function = UpperConfidenceBound(model=model, beta=0.1)
            with mock.patch(
                "botorch.acquisition.input_constructors.optimize_acqf",
                wraps=optimize_acqf,
            ) as mock_optimize_acqf:
                optimize_objective(
                    model=model,
                    bounds=bounds,
                    q=1,
                    acq_function=acq_function,
                )
            kwargs = mock_optimize_acqf.call_args[1]
            self.assertIs(kwargs["acq_function"], acq_function)

        with self.subTest("Passing optimizer"):
            # Not testing for a more specific error message because the
            # exception comes from Scipy and they might change it
            with self.assertRaises(RuntimeWarning):
                optimize_objective(
                    model=model,
                    bounds=bounds,
                    q=1,
                    acq_function=acq_function,
                    optimizer_options={"method": "throwing darts"},
                )

        A = torch.rand(1, bounds.shape[-1], dtype=torch.double, device=self.device)
        b = torch.zeros([1, 1], dtype=torch.double, device=self.device)
        idx = A[0].nonzero(as_tuple=False).squeeze()
        inequality_constraints = ((idx, -A[0, idx], -b[0, 0]),)

        m = 2
        y = torch.rand((n, m), dtype=torch.double, device=self.device)
        model = SingleTaskGP(train_X=x, train_Y=y)

        with self.subTest("scalarObjective_linearConstraints"):
            post_tf = ScalarizedPosteriorTransform(
                weights=torch.rand(m, dtype=torch.double, device=self.device)
            )
            with mock.patch(
                "botorch.acquisition.input_constructors.optimize_acqf",
                wraps=optimize_acqf,
            ) as mock_optimize_acqf:
                _ = optimize_objective(
                    model=model,
                    bounds=bounds,
                    q=1,
                    posterior_transform=post_tf,
                    linear_constraints=(A, b),
                    fixed_features=None,
                )

            kwargs = mock_optimize_acqf.call_args[1]
            self.assertIsInstance(kwargs["acq_function"], PosteriorMean)
            self.assertTrue(torch.equal(kwargs["bounds"], bounds))
            self.assertEqual(len(kwargs["inequality_constraints"]), 1)
            for a, b in zip(
                kwargs["inequality_constraints"][0], inequality_constraints[0]
            ):
                self.assertTrue(torch.equal(a, b))

        with self.subTest("mcObjective_fixedFeatures"):
            objective = LinearMCObjective(
                weights=torch.rand(m, dtype=torch.double, device=self.device)
            )
            with mock.patch(
                "botorch.acquisition.input_constructors.optimize_acqf",
                wraps=optimize_acqf,
            ) as mock_optimize_acqf:
                _ = optimize_objective(
                    model=model,
                    bounds=bounds,
                    q=1,
                    objective=objective,
                    fixed_features={0: 0.5},
                )

            kwargs = mock_optimize_acqf.call_args[1]
            self.assertIsInstance(
                kwargs["acq_function"], FixedFeatureAcquisitionFunction
            )
            self.assertIsInstance(kwargs["acq_function"].acq_func, qSimpleRegret)
            self.assertTrue(torch.equal(kwargs["bounds"], bounds[:, 1:]))

    def test__allow_only_specific_variable_kwargs__raises(self) -> None:
        input_constructor = get_acqf_input_constructor(ExpectedImprovement)
        with self.assertRaisesRegex(
            TypeError,
            "Unexpected keyword argument `hat` when constructing input arguments",
        ):
            input_constructor(
                model=self.mock_model, training_data=self.blockX_blockY, hat="car"
            )

    def test__register_acqf_input_constructor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            get_acqf_input_constructor(DummyAcquisitionFunction)

        dummy_constructor = MagicMock()

        _register_acqf_input_constructor(
            acqf_cls=DummyAcquisitionFunction,
            input_constructor=dummy_constructor,
        )
        input_constructor = get_acqf_input_constructor(DummyAcquisitionFunction)
        self.assertIs(input_constructor, dummy_constructor)

        # Clean up changes to the global registry (leads to failure of other tests).
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyAcquisitionFunction)


class TestAnalyticAcquisitionFunctionInputConstructors(InputConstructorBaseTestCase):
    def test_acqf_input_constructor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            get_acqf_input_constructor(DummyAcquisitionFunction)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            acqf_input_constructor(ExpectedImprovement)(lambda x: x)

    def test_construct_inputs_posterior_mean(self) -> None:
        c = get_acqf_input_constructor(PosteriorMean)
        mock_model = self.mock_model
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        # test instantiation
        acqf = PosteriorMean(**kwargs)
        self.assertIs(acqf.model, mock_model)

        post_tf = ScalarizedPosteriorTransform(weights=torch.rand(1))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            posterior_transform=post_tf,
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertIs(kwargs["posterior_transform"], post_tf)
        # test instantiation
        acqf = PosteriorMean(**kwargs)
        self.assertIs(acqf.model, mock_model)

    def test_construct_inputs_best_f(self) -> None:
        for acqf_cls in [
            ExpectedImprovement,
            LogExpectedImprovement,
            ProbabilityOfImprovement,
            LogProbabilityOfImprovement,
        ]:
            with self.subTest(acqf_cls=acqf_cls):
                c = get_acqf_input_constructor(acqf_cls)
                mock_model = self.mock_model
                kwargs = c(
                    model=mock_model, training_data=self.blockX_blockY, maximize=False
                )
                best_f_expected = self.blockX_blockY[0].Y.squeeze().max()
                self.assertIs(kwargs["model"], mock_model)
                self.assertIsNone(kwargs["posterior_transform"])
                self.assertEqual(kwargs["best_f"], best_f_expected)
                self.assertFalse(kwargs["maximize"])
                acqf = acqf_cls(**kwargs)
                self.assertIs(acqf.model, mock_model)

                kwargs = c(
                    model=mock_model, training_data=self.blockX_blockY, best_f=0.1
                )
                self.assertIs(kwargs["model"], mock_model)
                self.assertIsNone(kwargs["posterior_transform"])
                self.assertEqual(kwargs["best_f"], 0.1)
                self.assertTrue(kwargs["maximize"])
                acqf = acqf_cls(**kwargs)
                self.assertIs(acqf.model, mock_model)

    def test_construct_inputs_ucb(self) -> None:
        c = get_acqf_input_constructor(UpperConfidenceBound)
        mock_model = self.mock_model
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["beta"], 0.2)
        self.assertTrue(kwargs["maximize"])
        acqf = UpperConfidenceBound(**kwargs)
        self.assertIs(mock_model, acqf.model)

        kwargs = c(
            model=mock_model, training_data=self.blockX_blockY, beta=0.1, maximize=False
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertEqual(kwargs["beta"], 0.1)
        self.assertFalse(kwargs["maximize"])
        acqf = UpperConfidenceBound(**kwargs)
        self.assertIs(mock_model, acqf.model)

    def test_construct_inputs_noisy_ei(self) -> None:
        for acqf_cls in [NoisyExpectedImprovement, LogNoisyExpectedImprovement]:
            with self.subTest(acqf_cls=acqf_cls):
                c = get_acqf_input_constructor(acqf_cls)
                mock_model = SingleTaskGP(
                    train_X=torch.rand((2, 2)),
                    train_Y=torch.rand((2, 1)),
                    train_Yvar=torch.rand((2, 1)),
                )
                kwargs = c(model=mock_model, training_data=self.blockX_blockY)
                self.assertEqual(kwargs["model"], mock_model)
                self.assertTrue(
                    torch.equal(kwargs["X_observed"], self.blockX_blockY[0].X)
                )
                self.assertEqual(kwargs["num_fantasies"], 20)
                self.assertTrue(kwargs["maximize"])
                acqf = acqf_cls(**kwargs)
                self.assertTrue(acqf.maximize)

                kwargs = c(
                    model=mock_model,
                    training_data=self.blockX_blockY,
                    num_fantasies=10,
                    maximize=False,
                )
                self.assertEqual(kwargs["model"], mock_model)
                self.assertTrue(
                    torch.equal(kwargs["X_observed"], self.blockX_blockY[0].X)
                )
                self.assertEqual(kwargs["num_fantasies"], 10)
                self.assertFalse(kwargs["maximize"])
                acqf = acqf_cls(**kwargs)
                self.assertFalse(acqf.maximize)

                with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
                    c(model=mock_model, training_data=self.multiX_multiY)

    def test_construct_inputs_LogCEI(self) -> None:
        c = get_acqf_input_constructor(LogConstrainedExpectedImprovement)
        mock_model = self.mock_model
        constraints_tuple = [torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0]])]
        constraints = {1: (None, 2.0)}
        best_f_expected = self.blockX_blockY[0].Y.squeeze().max()
        objective_index = 0
        # test that best_f is inferred from training data
        # test constraint tuple
        kwargs = c(
            model=mock_model,
            objective_index=objective_index,
            training_data=self.blockX_blockY,
            constraints_tuple=constraints_tuple,
            maximize=False,
        )
        self.assertEqual(
            set(kwargs.keys()),
            {"model", "best_f", "objective_index", "constraints", "maximize"},
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertEqual(kwargs["objective_index"], objective_index)
        self.assertEqual(kwargs["constraints"], constraints)
        self.assertEqual(kwargs["best_f"], best_f_expected)
        self.assertFalse(kwargs["maximize"])
        # test that best_f overrides default from training data
        # test that negative constraints work
        constraints_tuple = [torch.tensor([[0.0, -1.0]]), torch.tensor([[-2.0]])]
        constraints = {1: (2.0, None)}
        kwargs = c(
            model=mock_model,
            objective_index=objective_index,
            training_data=self.blockX_blockY,
            best_f=0.1,
            constraints_tuple=constraints_tuple,
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertEqual(kwargs["objective_index"], objective_index)
        self.assertEqual(kwargs["constraints"], constraints)
        self.assertEqual(kwargs["best_f"], 0.1)
        self.assertTrue(kwargs["maximize"])
        # test that constraints on multiple outcomes raises an exception
        with self.assertRaisesRegex(
            BotorchError,
            "LogConstrainedExpectedImprovement only support constraints on single"
            " outcomes.",
        ):
            c(
                model=mock_model,
                objective_index=objective_index,
                training_data=self.blockX_blockY,
                constraints_tuple=[torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0]])],
            )
        # test that if objective_index coincides with constraints raises a value error
        with self.assertRaisesRegex(
            ValueError,
            "Output corresponding to objective should not be a constraint.",
        ):
            kwargs = c(
                model=mock_model,
                objective_index=1,
                training_data=self.blockX_blockY,
                constraints_tuple=[torch.tensor([[0.0, -1.0]]), torch.tensor([[-2.0]])],
            )
            LogConstrainedExpectedImprovement(**kwargs)

    def test_construct_inputs_eubo(self) -> None:
        """test input constructor for analytical EUBO and MC qEUBO"""

        # Set up
        # create dummy modellist gp
        n = 10
        X = torch.linspace(0, 0.95, n).unsqueeze(dim=-1)
        Y1, Y2 = torch.sin(X * (2 * math.pi)), torch.cos(X * (2 * math.pi))
        # 3 tasks
        train_X = torch.cat(
            [torch.nn.functional.pad(X, (1, 0), value=i) for i in range(3)]
        )
        train_Y = torch.cat([Y1, Y2])  # train_Y is a 1d tensor with shape (2n,)
        # model list of 2, so model.num_outputs is 4
        model = ModelListGP(
            *[MultiTaskGP(train_X, train_Y, task_feature=0) for i in range(2)]
        )
        self.assertEqual(model.num_outputs, 6)

        mock_pref_model = self.mock_model
        # assume we only have a preference model with 2 outcomes
        mock_pref_model.dim = 2
        mock_pref_model.datapoints = torch.tensor([])

        for eubo_acqf in (
            AnalyticExpectedUtilityOfBestOption,
            qExpectedUtilityOfBestOption,
        ):
            c = get_acqf_input_constructor(eubo_acqf)

            # test model only (i.e., PBO) construction
            kwargs = c(model=mock_pref_model)
            self.assertIsNone(kwargs["outcome_model"])

            # test basic construction
            kwargs = c(model=model, pref_model=mock_pref_model)
            self.assertIsInstance(kwargs["outcome_model"], FixedSingleSampleModel)
            self.assertIs(kwargs["pref_model"], mock_pref_model)
            if eubo_acqf is AnalyticExpectedUtilityOfBestOption:
                self.assertIsNone(kwargs["previous_winner"])
            # test instantiation
            eubo_acqf(**kwargs)

            # test sample_multiplier
            torch.manual_seed(123)
            kwargs = c(
                model=model,
                pref_model=mock_pref_model,
                sample_multiplier=1e6,
            )
            # w by default is drawn from std normal and very unlikely to be > 10.0
            self.assertTrue((kwargs["outcome_model"].w.abs() > 10.0).all())
            # Check w has the right dimension that agrees with the preference model
            self.assertEqual(kwargs["outcome_model"].w.shape[-1], mock_pref_model.dim)

        # test previous_winner
        c = get_acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
        previous_winner = torch.randn(mock_pref_model.dim)
        kwargs = c(
            model=model,
            pref_model=mock_pref_model,
            previous_winner=previous_winner,
        )
        self.assertTrue(torch.equal(kwargs["previous_winner"], previous_winner))
        # test instantiation
        AnalyticExpectedUtilityOfBestOption(**kwargs)


class TestMCAcquisitionFunctionInputConstructors(InputConstructorBaseTestCase):
    def test_construct_inputs_mc_base(self) -> None:
        c = get_acqf_input_constructor(qSimpleRegret)
        mock_model = self.mock_model
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        acqf = qSimpleRegret(**kwargs)
        self.assertIs(acqf.model, mock_model)

        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective=objective,
            X_pending=X_pending,
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        acqf = qSimpleRegret(**kwargs)
        self.assertIs(acqf.model, mock_model)
        # test constraints
        constraints = [lambda Y: Y[..., 0]]
        with self.assertRaisesRegex(ValueError, "Constraints require an X_baseline."):
            c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective=objective,
                X_pending=X_pending,
                constraints=constraints,
            )
        with mock.patch(
            "botorch.acquisition.input_constructors.get_infeasible_cost",
            return_value=2.0,
        ):
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective=objective,
                X_pending=X_pending,
                constraints=constraints,
                X_baseline=X_pending,
            )
        acqf = qSimpleRegret(**kwargs)
        self.assertIsNone(acqf._constraints)
        self.assertIsInstance(acqf.objective, ConstrainedMCObjective)
        self.assertIs(acqf.objective.objective, objective)
        self.assertIs(acqf.objective.constraints, constraints)
        self.assertEqual(acqf.objective.infeasible_cost.item(), 2.0)

        # TODO: Test passing through of sampler

    def test_construct_inputs_qLogPOF(self) -> None:
        c = get_acqf_input_constructor(qLogProbabilityOfFeasibility)
        mock_model = self.mock_model

        def constraint(Y: Tensor) -> Tensor:
            return Y[..., 0] - 0.5

        kwargs = c(model=mock_model, constraints=[constraint])
        self.assertEqual(
            set(kwargs.keys()),
            {
                "model",
                "constraints",
                "posterior_transform",
                "X_pending",
                "sampler",
                "eta",
                "fat",
                "tau_max",
            },
        )

        self.assertIs(kwargs["model"], mock_model)
        self.assertListEqual(kwargs["constraints"], [constraint])
        self.assertIsNone(kwargs["posterior_transform"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertIsInstance(kwargs["eta"], float)
        self.assertGreater(kwargs["eta"], 0.0)
        self.assertTrue(kwargs["fat"])
        self.assertIsInstance(kwargs["tau_max"], float)
        self.assertGreater(kwargs["tau_max"], 0.0)

    def test_construct_inputs_LogPOF(self) -> None:
        c = get_acqf_input_constructor(LogProbabilityOfFeasibility)
        mock_model = self.mock_model
        constraints_tuple = [torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0]])]
        constraints = {1: (None, 2.0)}
        kwargs = c(model=mock_model, constraints_tuple=constraints_tuple)
        self.assertEqual(set(kwargs.keys()), {"model", "constraints"})
        self.assertIs(kwargs["model"], mock_model)
        self.assertEqual(kwargs["constraints"], constraints)
        constraints_tuple = [torch.tensor([[0.0, -1.0]]), torch.tensor([[-2.0]])]
        kwargs = c(model=mock_model, constraints_tuple=constraints_tuple)
        constraints = {1: (2.0, None)}
        self.assertEqual(kwargs["constraints"], constraints)
        # test that constraints on multiple outcomes raises an exception
        with self.assertRaisesRegex(
            BotorchError,
            "LogProbabilityOfFeasibility only support constraints on single"
            " outcomes.",
        ):
            c(
                model=mock_model,
                constraints_tuple=[torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0]])],
            )

    def test_construct_inputs_qEI(self) -> None:
        c = get_acqf_input_constructor(qExpectedImprovement)
        mock_model = self.mock_model
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertIsNone(kwargs["constraints"])
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        acqf = qExpectedImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)

        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
        )
        self.assertIs(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        acqf = qExpectedImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)

        multi_Y = torch.cat([d.Y for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = objective(multi_Y).max()
        self.assertEqual(kwargs["best_f"], best_f_expected)
        # Check explicitly specifying `best_f`.
        best_f_expected = best_f_expected - 1  # Random value.
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            best_f=best_f_expected,
        )
        self.assertEqual(kwargs["best_f"], best_f_expected)
        acqf = qExpectedImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)
        self.assertEqual(acqf.best_f, best_f_expected)

        # test passing constraints
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            best_f=best_f_expected,
            constraints=constraints,
        )
        self.assertIs(kwargs["constraints"], constraints)
        acqf = qExpectedImprovement(**kwargs)
        self.assertEqual(acqf.best_f, best_f_expected)

        # testing qLogEI input constructor
        log_constructor = get_acqf_input_constructor(qLogExpectedImprovement)
        log_kwargs = log_constructor(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective=objective,
            X_pending=X_pending,
            best_f=best_f_expected,
            constraints=constraints,
        )
        # includes strict superset of kwargs tested above
        self.assertLessEqual(kwargs.items(), log_kwargs.items())
        self.assertIn("fat", log_kwargs)
        self.assertIn("tau_max", log_kwargs)
        self.assertEqual(log_kwargs["tau_max"], TAU_MAX)
        self.assertIn("tau_relu", log_kwargs)
        self.assertEqual(log_kwargs["tau_relu"], TAU_RELU)
        self.assertIs(log_kwargs["constraints"], constraints)
        acqf = qLogExpectedImprovement(**log_kwargs)
        self.assertIs(acqf.model, mock_model)
        self.assertIs(acqf.objective, objective)

    def test_construct_inputs_qNEI(self) -> None:
        c = get_acqf_input_constructor(qNoisyExpectedImprovement)
        mock_model = SingleTaskGP(
            train_X=torch.rand((2, 2)), train_Y=torch.rand((2, 1))
        )
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertIs(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertTrue(kwargs["prune_baseline"])
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.blockX_blockY[0].X))
        self.assertIsNone(kwargs["constraints"])
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        acqf = qNoisyExpectedImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)

        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(model=mock_model, training_data=self.multiX_multiY)

        X_baseline = torch.rand(2, 2)
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            X_baseline=X_baseline,
            prune_baseline=False,
            constraints=constraints,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertFalse(kwargs["prune_baseline"])
        self.assertTrue(torch.equal(kwargs["X_baseline"], X_baseline))
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        self.assertIs(kwargs["constraints"], constraints)
        acqf = qNoisyExpectedImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)

        # testing qLogNEI input constructor
        log_constructor = get_acqf_input_constructor(qLogNoisyExpectedImprovement)

        log_kwargs = log_constructor(
            model=mock_model,
            training_data=self.blockX_blockY,
            X_baseline=X_baseline,
            prune_baseline=False,
            constraints=constraints,
        )
        # includes strict superset of kwargs tested above
        self.assertLessEqual(kwargs.items(), log_kwargs.items())
        self.assertIn("fat", log_kwargs)
        self.assertIn("tau_max", log_kwargs)
        self.assertEqual(log_kwargs["tau_max"], TAU_MAX)
        self.assertIn("tau_relu", log_kwargs)
        self.assertEqual(log_kwargs["tau_relu"], TAU_RELU)
        self.assertIs(log_kwargs["constraints"], constraints)
        acqf = qLogNoisyExpectedImprovement(**log_kwargs)
        self.assertIs(acqf.model, mock_model)

    def test_construct_inputs_qPI(self) -> None:
        c = get_acqf_input_constructor(qProbabilityOfImprovement)
        mock_model = self.mock_model
        kwargs = c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["objective"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-3)
        self.assertIsNone(kwargs["constraints"])
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        acqf = qProbabilityOfImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)

        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            tau=1e-2,
        )
        self.assertEqual(kwargs["model"], mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["tau"], 1e-2)
        self.assertIsInstance(kwargs["eta"], float)
        self.assertLess(kwargs["eta"], 1)
        multi_Y = torch.cat([d.Y for d in self.blockX_multiY.values()], dim=-1)
        best_f_expected = objective(multi_Y).max()
        self.assertEqual(kwargs["best_f"], best_f_expected)
        acqf = qProbabilityOfImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)
        self.assertIs(acqf.objective, objective)

        # Check explicitly specifying `best_f`.
        best_f_expected = best_f_expected - 1  # Random value.
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_multiY,
            objective=objective,
            X_pending=X_pending,
            tau=1e-2,
            best_f=best_f_expected,
            constraints=constraints,
        )
        self.assertEqual(kwargs["best_f"], best_f_expected)
        self.assertIs(kwargs["constraints"], constraints)
        acqf = qProbabilityOfImprovement(**kwargs)
        self.assertIs(acqf.model, mock_model)
        self.assertIs(acqf.objective, objective)


class TestQUpperConfidenceBoundInputConstructor(InputConstructorBaseTestCase):
    acqf_class = qUpperConfidenceBound

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.c = get_acqf_input_constructor(self.acqf_class)

    def test_confidence_bound(self) -> None:
        mock_model = self.mock_model
        kwargs = self.c(model=mock_model, training_data=self.blockX_blockY)
        self.assertEqual(kwargs["model"], mock_model)
        self.assertIsNone(kwargs["X_pending"])
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["beta"], 0.2)
        acqf = self.acqf_class(**kwargs)
        self.assertIs(acqf.model, mock_model)

    def test_confidence_bound_with_objective(self) -> None:
        X_pending = torch.rand(2, 2)
        objective = LinearMCObjective(torch.rand(2))
        kwargs = self.c(
            model=self.mock_model,
            training_data=self.blockX_blockY,
            objective=objective,
            X_pending=X_pending,
            beta=0.1,
        )
        self.assertEqual(kwargs["model"], self.mock_model)
        self.assertTrue(torch.equal(kwargs["objective"].weights, objective.weights))
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertIsNone(kwargs["sampler"])
        self.assertEqual(kwargs["beta"], 0.1)
        acqf = self.acqf_class(**kwargs)
        self.assertIs(acqf.model, self.mock_model)

    def test_confidence_bound_with_constraints_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Constraints require an X_baseline."):
            self.c(
                model=self.mock_model,
                training_data=self.blockX_blockY,
                constraints=torch.rand(2, 2),
            )

    def test_confidence_bound_with_constraints(self) -> None:
        # these are needed for computing the infeasible cost
        self.mock_model._posterior._mean = torch.zeros(2, 2)
        self.mock_model._posterior._variance = torch.ones(2, 2)

        X_baseline = torch.rand(2, 2)
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        for objective in (LinearMCObjective(torch.rand(2)), None):
            with self.subTest(objective=objective):
                kwargs = self.c(
                    model=self.mock_model,
                    training_data=self.blockX_blockY,
                    objective=objective,
                    constraints=constraints,
                    X_baseline=X_baseline,
                )
                final_objective = kwargs["objective"]
                self.assertIsInstance(final_objective, ConstrainedMCObjective)
                if objective is None:
                    self.assertIsInstance(
                        final_objective.objective, IdentityMCObjective
                    )
                else:
                    self.assertIs(final_objective.objective, objective)
                self.assertIs(final_objective.constraints, constraints)
                # test that we can construct the acquisition function
                self.acqf_class(**kwargs)


class TestQLowerConfidenceBoundInputConstructor(
    TestQUpperConfidenceBoundInputConstructor
):
    acqf_class = qLowerConfidenceBound


class TestMultiObjectiveAcquisitionFunctionInputConstructors(
    InputConstructorBaseTestCase
):
    def test_construct_inputs_EHVI(self) -> None:
        c = get_acqf_input_constructor(ExpectedHypervolumeImprovement)
        mock_model = mock.Mock()
        objective_thresholds = torch.rand(6)

        # test error on non-block designs
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(
                model=mock_model,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
            )

        # test error on unsupported outcome constraints
        with self.assertRaises(NotImplementedError):
            c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                constraints=mock.Mock(),
            )

        # test with Y_pmean supplied explicitly
        with self.subTest("explicit Y_pmean"):
            Y_pmean = torch.rand(3, 6)
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                Y_pmean=Y_pmean,
            )
            self.assertSetEqual(
                set(kwargs.keys()), {"model", "ref_point", "partitioning"}
            )
            self.assertEqual(kwargs["model"], mock_model)
            self.assertTrue(torch.equal(kwargs["ref_point"], objective_thresholds))
            partitioning = kwargs["partitioning"]
            alpha_expected = get_default_partitioning_alpha(6)
            self.assertIsInstance(partitioning, NondominatedPartitioning)
            self.assertEqual(partitioning.alpha, alpha_expected)
            self.assertTrue(
                torch.equal(partitioning._neg_ref_point, -objective_thresholds)
            )

            Y_pmean = torch.rand(3, 2)
            objective_thresholds = torch.rand(2)
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                Y_pmean=Y_pmean,
            )
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, FastNondominatedPartitioning)
            self.assertTrue(torch.equal(partitioning.ref_point, objective_thresholds))

        n = 3
        m = 4
        k = 2
        with self.subTest("posterior_transform"):
            model = SingleTaskGP(train_X=torch.rand((n, k)), train_Y=torch.rand((n, m)))
            #  This is a strange posterior transform to test with a hypervolume acqf
            posterior_transform = ScalarizedPosteriorTransform(weights=torch.rand(m))
            objective_thresholds = torch.rand(m)
            Y_pmean = torch.rand(n, m)
            kwargs = c(
                model=model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                Y_pmean=Y_pmean,
                posterior_transform=posterior_transform,
            )
            self.assertSetEqual(
                set(kwargs.keys()),
                {"model", "ref_point", "partitioning", "posterior_transform"},
            )
            acqf = ExpectedHypervolumeImprovement(**kwargs)
            acqf(torch.rand((1, k)))

        with self.subTest("Without Y_pmean"):
            mean = torch.rand(1, m)
            variance = torch.ones(1, 1)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
            )
            self.assertSetEqual(
                set(kwargs.keys()), {"model", "ref_point", "partitioning"}
            )
            self.assertTrue(torch.equal(kwargs["ref_point"], objective_thresholds))
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, FastNondominatedPartitioning)
            self.assertTrue(torch.equal(partitioning.ref_point, objective_thresholds))
            self.assertTrue(torch.equal(partitioning._neg_Y, -mean))

    def test_construct_inputs_qEHVI(self) -> None:
        c = get_acqf_input_constructor(qExpectedHypervolumeImprovement)
        objective_thresholds = torch.rand(2)

        with self.subTest("defaults"):
            mm = SingleTaskGP(torch.rand(1, 2), torch.rand(1, 2))
            mean = mm.posterior(self.blockX_blockY[0].X).mean
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
            )
            self.assertSetEqual(
                set(kwargs.keys()),
                {
                    "model",
                    "ref_point",
                    "partitioning",
                    "sampler",
                    "X_pending",
                    "constraints",
                    "eta",
                    "objective",
                },
            )
            self.assertIsNone(kwargs["objective"])
            ref_point_expected = objective_thresholds
            self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, FastNondominatedPartitioning)
            self.assertTrue(torch.equal(partitioning.ref_point, ref_point_expected))
            self.assertTrue(torch.equal(partitioning._neg_Y, -mean))
            sampler = kwargs["sampler"]
            self.assertIsInstance(sampler, SobolQMCNormalSampler)
            self.assertEqual(sampler.sample_shape, torch.Size([128]))
            self.assertIsNone(kwargs["X_pending"])
            self.assertIsNone(kwargs["constraints"])
            self.assertEqual(kwargs["eta"], 1e-3)

        with self.subTest("custom objective"):
            weights = torch.rand(2)
            obj = WeightedMCMultiOutputObjective(weights=weights)
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=obj,
                alpha=0.05,
            )
            self.assertEqual(kwargs["model"], mm)
            self.assertIsInstance(kwargs["objective"], WeightedMCMultiOutputObjective)
            ref_point_expected = objective_thresholds * weights
            self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, NondominatedPartitioning)
            self.assertEqual(partitioning.alpha, 0.05)
            self.assertTrue(
                torch.equal(partitioning._neg_ref_point, -ref_point_expected)
            )

        with self.subTest("risk measures"):
            for use_preprocessing in (True, False):
                obj = MultiOutputExpectation(
                    n_w=3,
                    preprocessing_function=(
                        WeightedMCMultiOutputObjective(torch.tensor([-1.0, -1.0]))
                        if use_preprocessing
                        else None
                    ),
                )
                kwargs = c(
                    model=mm,
                    training_data=self.blockX_blockY,
                    objective_thresholds=objective_thresholds,
                    objective=obj,
                )
                expected_obj_t = (
                    -objective_thresholds if use_preprocessing else objective_thresholds
                )
                self.assertIs(kwargs["objective"], obj)
                self.assertTrue(torch.equal(kwargs["ref_point"], expected_obj_t))
                partitioning = kwargs["partitioning"]
                self.assertIsInstance(partitioning, FastNondominatedPartitioning)
                self.assertTrue(torch.equal(partitioning.ref_point, expected_obj_t))

        with self.subTest("IID sampler"):
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                qmc=False,
                mc_samples=64,
            )
            sampler = kwargs["sampler"]
            self.assertIsInstance(sampler, IIDNormalSampler)
            self.assertEqual(sampler.sample_shape, torch.Size([64]))

        # Test outcome constraints and custom inputs
        with self.subTest("outcome constraints and custom imports"):
            mean = torch.tensor([[1.0, 0.25], [0.5, 1.0]])
            variance = torch.ones(1, 1)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            weights = torch.rand(2)
            obj = WeightedMCMultiOutputObjective(weights=weights)
            outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
            constraints = get_outcome_constraint_transforms(
                outcome_constraints=outcome_constraints
            )
            X_pending = torch.rand(1, 2)
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=obj,
                constraints=constraints,
                X_pending=X_pending,
                alpha=0.05,
                eta=1e-2,
            )
            self.assertIsInstance(kwargs["objective"], WeightedMCMultiOutputObjective)
            ref_point_expected = objective_thresholds * weights
            self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
            partitioning = kwargs["partitioning"]
            self.assertIsInstance(partitioning, NondominatedPartitioning)
            self.assertEqual(partitioning.alpha, 0.05)
            self.assertTrue(
                torch.equal(partitioning._neg_ref_point, -ref_point_expected)
            )
            Y_expected = mean[:1] * weights
            self.assertTrue(torch.equal(partitioning._neg_Y, -Y_expected))
            self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
            self.assertIs(kwargs["constraints"], constraints)
            self.assertEqual(kwargs["eta"], 1e-2)

        with (
            self.subTest("block designs"),
            self.assertRaisesRegex(ValueError, "Field `X` must be shared"),
        ):
            c(
                model=mm,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
                objective=obj,
                constraints=constraints,
                X_pending=X_pending,
                alpha=0.05,
                eta=1e-2,
            )

        # Test custom sampler
        with self.subTest("custom sampler"):
            custom_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([16]), seed=1234
            )
            kwargs = c(
                model=mm,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                sampler=custom_sampler,
            )
            sampler = kwargs["sampler"]
            self.assertIsInstance(sampler, SobolQMCNormalSampler)
            self.assertEqual(sampler.sample_shape, torch.Size([16]))
            self.assertEqual(sampler.seed, 1234)

    def test_construct_inputs_qNEHVI(self) -> None:
        self._test_construct_inputs_qNEHVI(qNoisyExpectedHypervolumeImprovement)

    def test_construct_inputs_qLogNEHVI(self) -> None:
        self._test_construct_inputs_qNEHVI(qLogNoisyExpectedHypervolumeImprovement)

    def _test_construct_inputs_qNEHVI(self, acqf_class: type[AcquisitionFunction]):
        c = get_acqf_input_constructor(acqf_class)
        objective_thresholds = torch.rand(2)

        # Test defaults
        kwargs = c(
            model=SingleTaskGP(torch.rand(1, 2), torch.rand(1, 2)),
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        ref_point_expected = objective_thresholds
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.blockX_blockY[0].X))
        self.assertIsInstance(kwargs["sampler"], SobolQMCNormalSampler)
        self.assertEqual(kwargs["sampler"].sample_shape, torch.Size([128]))
        self.assertIsInstance(kwargs["objective"], IdentityMCMultiOutputObjective)
        self.assertIsNone(kwargs["constraints"])
        self.assertIsNone(kwargs["X_pending"])
        self.assertEqual(kwargs["eta"], 1e-3)
        self.assertTrue(kwargs["prune_baseline"])
        self.assertEqual(kwargs["alpha"], 0.0)
        self.assertTrue(kwargs["cache_pending"])
        self.assertEqual(kwargs["max_iep"], 0)
        self.assertTrue(kwargs["incremental_nehvi"])
        self.assertTrue(kwargs["cache_root"])

        if acqf_class == qLogNoisyExpectedHypervolumeImprovement:
            self.assertEqual(kwargs["tau_relu"], TAU_RELU)
            self.assertEqual(kwargs["tau_max"], TAU_MAX)
            self.assertEqual(kwargs["fat"], True)
        else:
            self.assertNotIn("tau_relu", kwargs)
            self.assertNotIn("tau_max", kwargs)
            self.assertEqual(kwargs["fat"], False)

        # Test check for block designs
        mock_model = mock.Mock()
        mock_model.num_outputs = 2
        with self.assertRaisesRegex(ValueError, "Field `X` must be shared"):
            c(
                model=mock_model,
                training_data=self.multiX_multiY,
                objective_thresholds=objective_thresholds,
            )

        # Test custom inputs
        weights = torch.rand(2)
        objective = WeightedMCMultiOutputObjective(weights=weights)
        X_baseline = torch.rand(2, 2)
        sampler = IIDNormalSampler(sample_shape=torch.Size([4]))
        outcome_constraints = (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.5]]))
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=outcome_constraints
        )
        X_pending = torch.rand(1, 2)
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
            objective=objective,
            X_baseline=X_baseline,
            sampler=sampler,
            constraints=constraints,
            X_pending=X_pending,
            eta=1e-2,
            prune_baseline=True,
            alpha=0.0,
            cache_pending=False,
            max_iep=1,
            incremental_nehvi=False,
            cache_root=False,
        )
        ref_point_expected = objective(objective_thresholds)
        self.assertTrue(torch.equal(kwargs["ref_point"], ref_point_expected))
        self.assertTrue(torch.equal(kwargs["X_baseline"], X_baseline))
        sampler_ = kwargs["sampler"]
        self.assertIsInstance(sampler_, IIDNormalSampler)
        self.assertEqual(sampler_.sample_shape, torch.Size([4]))
        self.assertEqual(kwargs["objective"], objective)
        self.assertIs(kwargs["constraints"], constraints)
        self.assertTrue(torch.equal(kwargs["X_pending"], X_pending))
        self.assertEqual(kwargs["eta"], 1e-2)
        self.assertTrue(kwargs["prune_baseline"])
        self.assertEqual(kwargs["alpha"], 0.0)
        self.assertFalse(kwargs["cache_pending"])
        self.assertEqual(kwargs["max_iep"], 1)
        self.assertFalse(kwargs["incremental_nehvi"])
        self.assertFalse(kwargs["cache_root"])

        # Test with risk measures.
        with self.assertRaisesRegex(UnsupportedError, "feasibility-weighted"):
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=MultiOutputExpectation(n_w=3),
                constraints=constraints,
            )
        for use_preprocessing in (True, False):
            obj = MultiOutputExpectation(
                n_w=3,
                preprocessing_function=(
                    WeightedMCMultiOutputObjective(torch.tensor([-1.0, -1.0]))
                    if use_preprocessing
                    else None
                ),
            )
            kwargs = c(
                model=mock_model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=obj,
            )
            expected_obj_t = (
                -objective_thresholds if use_preprocessing else objective_thresholds
            )
            self.assertIs(kwargs["objective"], obj)
            self.assertTrue(torch.equal(kwargs["ref_point"], expected_obj_t))

        # Test default alpha for many objectives/
        mock_model.num_outputs = 5
        kwargs = c(
            model=mock_model,
            training_data=self.blockX_blockY,
            objective_thresholds=objective_thresholds,
        )
        self.assertEqual(kwargs["alpha"], 0.0)

    def test_construct_inputs_qLogNParEGO(self) -> None:
        # Focusing on the unique attributes since the rest are same as qLogNEI.
        c = get_acqf_input_constructor(qLogNParEGO)
        kwargs = c(model=mock.Mock(), training_data=self.blockX_blockY)
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.blockX_blockY[0].X))
        self.assertIsNone(kwargs["scalarization_weights"])
        self.assertIsNone(kwargs["objective"])
        self.assertNotIn("posterior_transform", kwargs)
        # With custom objective & weights.
        kwargs = c(
            model=mock.Mock(),
            training_data=self.blockX_blockY,
            scalarization_weights=torch.zeros(2),
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        )
        self.assertAllClose(kwargs["scalarization_weights"], torch.zeros(2))
        self.assertIsInstance(kwargs["objective"], IdentityMCMultiOutputObjective)


class TestKGandESAcquisitionFunctionInputConstructors(InputConstructorBaseTestCase):
    def test_construct_inputs_kg(self) -> None:
        func = get_acqf_input_constructor(qKnowledgeGradient)

        with self.subTest("test_with_current_value"):
            current_value = torch.tensor(1.23)

            with mock.patch(
                target="botorch.acquisition.input_constructors.optimize_objective",
                return_value=(None, current_value),
            ):
                kwargs = func(
                    model=mock.Mock(),
                    training_data=self.blockX_blockY,
                    objective=LinearMCObjective(torch.rand(2)),
                    bounds=self.bounds,
                    num_fantasies=33,
                    with_current_value=True,
                )

                self.assertEqual(kwargs["num_fantasies"], 33)
                self.assertEqual(kwargs["current_value"], current_value)

        with self.subTest("test_without_current_value"):
            kwargs = func(
                model=mock.Mock(),
                training_data=self.blockX_blockY,
                objective=LinearMCObjective(torch.rand(2)),
                bounds=self.bounds,
                num_fantasies=33,
                with_current_value=False,
            )
            self.assertNotIn("current_value", kwargs)

    def test_construct_inputs_mfhvkg(self) -> None:
        get_kwargs = get_acqf_input_constructor(
            qMultiFidelityHypervolumeKnowledgeGradient
        )

        model = mock.Mock()
        objective = IdentityMCMultiOutputObjective()
        objective_thresholds = torch.rand(2)

        with self.assertRaisesRegex(
            NotImplementedError, "Trace observations are not currently supported"
        ):
            get_kwargs(
                model=model,
                training_data=self.blockX_blockY,
                objective_thresholds=objective_thresholds,
                objective=objective,
                bounds=self.bounds,
                num_fantasies=33,
                num_pareto=11,
                target_fidelities={0: 0.987},
                fidelity_weights={0: 0.654},
                cost_intercept=0.321,
                num_trace_observations=5,
            )

    @mock.patch("botorch.acquisition.input_constructors._get_hv_value_function")
    def test_construct_inputs_hvkg(self, mock_get_hv_value_function) -> None:
        current_value = torch.tensor(1.23)
        objective_thresholds = torch.rand(2)

        for acqf_cls in (
            qHypervolumeKnowledgeGradient,
            qMultiFidelityHypervolumeKnowledgeGradient,
        ):
            get_kwargs = get_acqf_input_constructor(acqf_cls)

            model = mock.Mock()
            objective = IdentityMCMultiOutputObjective()

            input_constructor_extra_kwargs = {}
            if acqf_cls == qMultiFidelityHypervolumeKnowledgeGradient:
                input_constructor_extra_kwargs.update(
                    target_fidelities={0: 0.987},
                    fidelity_weights={0: 0.654},
                    cost_intercept=0.321,
                )

            with mock.patch(
                target="botorch.acquisition.input_constructors.optimize_acqf",
                return_value=(None, current_value),
            ) as mock_optimize_acqf:
                kwargs = get_kwargs(
                    model=model,
                    training_data=self.blockX_blockY,
                    objective_thresholds=objective_thresholds,
                    objective=objective,
                    bounds=self.bounds,
                    num_fantasies=33,
                    num_pareto=11,
                    **input_constructor_extra_kwargs,
                )

                self.assertEqual(
                    mock_get_hv_value_function.call_args.kwargs["model"], model
                )
                self.assertEqual(
                    mock_get_hv_value_function.call_args.kwargs["objective"], objective
                )
                self.assertTrue(
                    torch.equal(
                        mock_get_hv_value_function.call_args.kwargs["ref_point"],
                        objective_thresholds,
                    )
                )

                # check that `optimize_acqf` is called with the desired value function
                if acqf_cls == qMultiFidelityHypervolumeKnowledgeGradient:
                    self.assertIsInstance(
                        mock_optimize_acqf.call_args.kwargs["acq_function"],
                        FixedFeatureAcquisitionFunction,
                    )
                else:
                    self.assertEqual(
                        mock_optimize_acqf.call_args.kwargs["acq_function"],
                        mock_get_hv_value_function(),
                    )

            self.assertLessEqual(
                {
                    "model",
                    "ref_point",
                    "num_fantasies",
                    "num_pareto",
                    "objective",
                    "current_value",
                },
                set(kwargs.keys()),
            )
            self.assertEqual(kwargs["num_fantasies"], 33)
            self.assertEqual(kwargs["num_pareto"], 11)
            self.assertEqual(kwargs["current_value"], current_value)
            self.assertTrue(torch.equal(kwargs["ref_point"], objective_thresholds))

            with self.subTest("custom objective"):
                weights = torch.rand(2)
                objective = WeightedMCMultiOutputObjective(weights=weights)
                with mock.patch(
                    target="botorch.acquisition.input_constructors.optimize_acqf",
                    return_value=(None, current_value),
                ) as mock_optimize_acqf:
                    kwargs = get_kwargs(
                        model=model,
                        training_data=self.blockX_blockY,
                        objective_thresholds=objective_thresholds,
                        objective=objective,
                        bounds=self.bounds,
                        num_fantasies=33,
                        num_pareto=11,
                        **input_constructor_extra_kwargs,
                    )
                self.assertIsInstance(
                    kwargs["objective"], WeightedMCMultiOutputObjective
                )
                self.assertTrue(
                    torch.equal(kwargs["ref_point"], objective_thresholds * weights)
                )

            with self.subTest("risk measures"):
                for use_preprocessing in (True, False):
                    objective = MultiOutputExpectation(
                        n_w=3,
                        preprocessing_function=(
                            WeightedMCMultiOutputObjective(torch.tensor([-1.0, -1.0]))
                            if use_preprocessing
                            else None
                        ),
                    )
                    with mock.patch(
                        target="botorch.acquisition.input_constructors.optimize_acqf",
                        return_value=(None, current_value),
                    ) as mock_optimize_acqf:
                        kwargs = get_kwargs(
                            model=model,
                            training_data=self.blockX_blockY,
                            objective_thresholds=objective_thresholds,
                            objective=objective,
                            bounds=self.bounds,
                            num_fantasies=33,
                            num_pareto=11,
                            **input_constructor_extra_kwargs,
                        )
                    expected_obj_t = (
                        -objective_thresholds
                        if use_preprocessing
                        else objective_thresholds
                    )
                    self.assertTrue(torch.equal(kwargs["ref_point"], expected_obj_t))

    def test_construct_inputs_mes(self) -> None:
        func = get_acqf_input_constructor(qMaxValueEntropy)
        n, d, m = 5, 2, 1
        model = SingleTaskGP(train_X=torch.ones((n, d)), train_Y=torch.zeros((n, m)))
        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            posterior_transform=ScalarizedPosteriorTransform(torch.rand(m)),
            bounds=self.bounds,
            candidate_size=17,
            maximize=False,
        )

        self.assertFalse(kwargs["maximize"])
        self.assertGreaterEqual(kwargs["candidate_set"].min(), 0.0)
        self.assertLessEqual(kwargs["candidate_set"].max(), 1.0)
        self.assertEqual(
            [int(s) for s in kwargs["candidate_set"].shape], [17, len(self.bounds)]
        )

        acqf = qMaxValueEntropy(**kwargs)
        self.assertIs(acqf.model, model)

    def test_construct_inputs_mf_base(self) -> None:
        target_fidelities = {0: 0.123}
        fidelity_weights = {0: 0.456}
        cost_intercept = 0.789
        num_trace_observations = 0

        with self.subTest("test_fully_specified"):
            kwargs = construct_inputs_mf_base(
                target_fidelities=target_fidelities,
                fidelity_weights=fidelity_weights,
                cost_intercept=cost_intercept,
                num_trace_observations=num_trace_observations,
            )

            X = torch.rand(3, 2)
            self.assertIsInstance(kwargs["expand"], Callable)
            self.assertTrue(
                torch.equal(
                    kwargs["expand"](X),
                    expand_trace_observations(
                        X=X,
                        fidelity_dims=sorted(target_fidelities),
                        num_trace_obs=num_trace_observations,
                    ),
                )
            )

            self.assertIsInstance(kwargs["project"], Callable)
            self.assertTrue(
                torch.equal(
                    kwargs["project"](X),
                    project_to_target_fidelity(X, target_fidelities=target_fidelities),
                )
            )

            cm = kwargs["cost_aware_utility"].cost_model
            w = torch.tensor(list(fidelity_weights.values()), dtype=cm.weights.dtype)
            self.assertEqual(cm.fixed_cost, cost_intercept)
            self.assertAllClose(cm.weights, w)

        with self.subTest("test_missing_fidelity_weights"):
            kwargs = construct_inputs_mf_base(
                target_fidelities=target_fidelities,
                cost_intercept=cost_intercept,
            )
            cm = kwargs["cost_aware_utility"].cost_model
            self.assertAllClose(cm.weights, torch.ones_like(cm.weights))

        with self.subTest("test_mismatched_weights"):
            with self.assertRaisesRegex(
                RuntimeError, "Must provide the same indices for"
            ):
                construct_inputs_mf_base(
                    target_fidelities={0: 1.0},
                    fidelity_weights={1: 0.5},
                    cost_intercept=cost_intercept,
                )

    def test_construct_inputs_mfkg(self) -> None:
        current_value = torch.tensor(1.23)

        constructor_args = {
            "model": self.mock_model,
            "training_data": self.blockX_blockY,
            "bounds": self.bounds,
            "target_fidelities": {0: 0.987},
            "objective": None,
            "fidelity_weights": {0: 0.654},
            "cost_intercept": 0.321,
            "num_fantasies": 123,
        }

        input_constructor = get_acqf_input_constructor(qMultiFidelityKnowledgeGradient)
        with mock.patch(
            target="botorch.acquisition.input_constructors.optimize_acqf",
            return_value=(None, current_value),
        ) as mock_optimize_acqf:
            inputs_mfkg = input_constructor(**constructor_args)

            mock_optimize_acqf_kwargs = mock_optimize_acqf.call_args.kwargs

            self.assertIsInstance(
                mock_optimize_acqf_kwargs["acq_function"],
                FixedFeatureAcquisitionFunction,
            )
            self.assertLessEqual(
                {
                    "model",
                    "objective",
                    "current_value",
                    "project",
                    "expand",
                    "cost_aware_utility",
                    "posterior_transform",
                    "num_fantasies",
                },
                set(inputs_mfkg.keys()),
            )
            self.assertEqual(
                inputs_mfkg["num_fantasies"], constructor_args["num_fantasies"]
            )
            self.assertEqual(inputs_mfkg["current_value"], current_value)

    def test_construct_inputs_mfmes(self) -> None:
        target_fidelities = {0: 0.987}
        constructor_args = {
            "model": None,
            "training_data": self.blockX_blockY,
            "bounds": self.bounds,
            "candidate_size": 17,
            "target_fidelities": target_fidelities,
            "fidelity_weights": {0: 0.654},
            "cost_intercept": 0.321,
        }
        input_constructor = get_acqf_input_constructor(qMultiFidelityMaxValueEntropy)
        with (
            mock.patch(
                target=(
                    "botorch.acquisition.input_constructors.construct_inputs_mf_base"
                ),
                return_value={"foo": 0},
            ),
            mock.patch(
                target="botorch.acquisition.input_constructors.construct_inputs_qMES",
                return_value={"bar": 1},
            ),
        ):
            inputs_mfmes = input_constructor(**constructor_args)
        inputs_test = {"foo": 0, "bar": 1, "num_fantasies": 64}
        self.assertEqual(inputs_mfmes, inputs_test)

    def test_construct_inputs_jes(self) -> None:
        func = get_acqf_input_constructor(qJointEntropySearch)
        # we need to run optimize_posterior_samples, so we sort of need
        # a real model as there is no other (apparent) option
        model = SingleTaskGP(self.blockX_blockY[0].X, self.blockX_blockY[0].Y)

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
            num_optima=17,
            posterior_transform=ScalarizedPosteriorTransform(
                torch.rand(1, dtype=self.blockX_blockY[0].Y.dtype)
            ),
        )

        self.assertEqual(self.blockX_blockY[0].X.dtype, kwargs["optimal_inputs"].dtype)
        self.assertEqual(len(kwargs["optimal_inputs"]), 17)
        self.assertEqual(len(kwargs["optimal_outputs"]), 17)
        # asserting that, for the non-batch case, the optimal inputs are
        # of shape N x D and outputs are N x 1
        self.assertEqual(len(kwargs["optimal_inputs"].shape), 2)
        self.assertEqual(len(kwargs["optimal_outputs"].shape), 2)
        qJointEntropySearch(**kwargs)

    def test_construct_inputs_bald(self) -> None:
        func = get_acqf_input_constructor(qBayesianActiveLearningByDisagreement)
        num_samples = 3
        model = SaasFullyBayesianSingleTaskGP(
            self.blockX_blockY[0].X, self.blockX_blockY[0].Y
        )

        model.load_mcmc_samples(
            {
                "lengthscale": torch.rand(
                    num_samples,
                    1,
                    self.blockX_blockY[0].X.shape[-1],
                    dtype=torch.double,
                ),
                "outputscale": torch.rand(num_samples, dtype=torch.double),
                "mean": torch.randn(num_samples, dtype=torch.double),
                "noise": torch.rand(num_samples, 1, dtype=torch.double),
            }
        )

        kwargs = func(
            model=model,
            training_data=self.blockX_blockY,
            bounds=self.bounds,
        )

        qBayesianActiveLearningByDisagreement(**kwargs)

    def test_construct_inputs_nipv(self) -> None:
        X = torch.rand(3, 2)
        Y = torch.rand(3, 1)
        blockX_blockY = {
            0: SupervisedDataset(X, Y, feature_names=["X1", "X2"], outcome_names=["Y"])
        }

        func = get_acqf_input_constructor(qNegIntegratedPosteriorVariance)
        model = SingleTaskGP(blockX_blockY[0].X, blockX_blockY[0].Y)
        kwargs = func(
            model=model,
            bounds=self.bounds,
            num_mc_points=17,
            X_pending=torch.rand(3, 2, dtype=torch.double),
            training_data=blockX_blockY,
        )
        nipv = qNegIntegratedPosteriorVariance(**kwargs)
        self.assertEqual(nipv.mc_points.shape, torch.Size([17, 2]))


class TestInstantiationFromInputConstructor(InputConstructorBaseTestCase):
    """End-to-end tests, ensuring that the input constructors are functional."""

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        # {key: (list of acquisition functions, arguments they accept)}
        constraints_tuple_dict = {
            "constraints_tuple": (torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0]])),
        }
        self.cases = {
            "PosteriorMean-type": (
                [
                    PosteriorMean,
                    UpperConfidenceBound,
                    qUpperConfidenceBound,
                    qLowerConfidenceBound,
                ],
                {"model": self.mock_model, **constraints_tuple_dict},
            ),
        }
        st_soo_model = SingleTaskGP(
            train_X=torch.rand((4, 2)),
            train_Y=torch.rand((4, 1)),
            train_Yvar=torch.ones((4, 1)),
        )
        self.cases["EI-type"] = (
            [
                ExpectedImprovement,
                LogExpectedImprovement,
                ProbabilityOfImprovement,
                LogProbabilityOfImprovement,
                NoisyExpectedImprovement,
                LogNoisyExpectedImprovement,
                qExpectedImprovement,
                qLogExpectedImprovement,
                qNoisyExpectedImprovement,
                qLogNoisyExpectedImprovement,
                qProbabilityOfImprovement,
            ],
            {
                "model": st_soo_model,
                "training_data": self.blockX_blockY,
                **constraints_tuple_dict,
            },
        )

        self.cases["LogPoF"] = (
            [LogProbabilityOfFeasibility],
            {
                "model": st_soo_model,
                "constraints": {0: [-5, 5]},
                **constraints_tuple_dict,
            },
        )

        self.cases["LogCEI"] = (
            [LogConstrainedExpectedImprovement],
            {
                "model": st_soo_model,
                "objective_index": 0,
                "training_data": self.blockX_blockY,
                "constraints": {0: [-5, 5]},
                **constraints_tuple_dict,
            },
        )

        def constraint(X: Tensor) -> Tensor:
            return X[..., 0].abs() - 5

        self.cases["qLogPoF"] = (
            [qLogProbabilityOfFeasibility],
            {
                "model": st_soo_model,
                "constraints": [constraint],
                **constraints_tuple_dict,
            },
        )

        bounds = torch.ones((1, 2))
        kg_model = SingleTaskGP(train_X=torch.rand((3, 1)), train_Y=torch.rand((3, 1)))
        self.cases["Look-ahead"] = (
            [qMaxValueEntropy, qKnowledgeGradient],
            {
                "model": kg_model,
                "training_data": self.blockX_blockY,
                "bounds": bounds,
                **constraints_tuple_dict,
            },
        )
        self.cases["MF look-ahead"] = (
            [qMultiFidelityMaxValueEntropy],
            {
                "model": kg_model,
                "training_data": self.blockX_blockY,
                "bounds": bounds,
                "target_fidelities": {0: 0.987},
                "num_fantasies": 30,
                **constraints_tuple_dict,
            },
        )
        bounds = torch.ones((2, 2))
        kg_model = SingleTaskGP(train_X=torch.rand((3, 2)), train_Y=torch.rand((3, 1)))
        self.cases["MF look-ahead (KG)"] = (
            [qMultiFidelityKnowledgeGradient],
            {
                "model": kg_model,
                "training_data": self.blockX_blockY,
                "bounds": bounds,
                "target_fidelities": {0: 0.987},
                "num_fantasies": 30,
                **constraints_tuple_dict,
            },
        )

        objective_thresholds = torch.tensor([0.1, 0.2])
        st_moo_model = SingleTaskGP(
            train_X=torch.rand((3, 2)), train_Y=torch.rand((3, 2))
        )
        self.cases["EHVI-type"] = (
            [
                qNoisyExpectedHypervolumeImprovement,
                qLogNoisyExpectedHypervolumeImprovement,
                ExpectedHypervolumeImprovement,
                qExpectedHypervolumeImprovement,
                qLogExpectedHypervolumeImprovement,
                qLogNParEGO,
            ],
            {
                "model": st_moo_model,
                "objective_thresholds": objective_thresholds,
                "training_data": self.blockX_blockY,
                **constraints_tuple_dict,
            },
        )

        X = torch.rand(3, 2)
        Y1 = torch.rand(3, 1)
        Y2 = torch.rand(3, 1)
        m1 = SingleTaskGP(X, Y1)
        m2 = SingleTaskGP(X, Y2)
        model_list = ModelListGP(m1, m2)
        self.cases["HV Look-ahead"] = (
            [qHypervolumeKnowledgeGradient],
            {
                "model": model_list,
                "training_data": self.blockX_blockY,
                "bounds": bounds,
                "objective_thresholds": objective_thresholds,
                **constraints_tuple_dict,
            },
        )
        self.cases["MF HV Look-ahead"] = (
            [qMultiFidelityHypervolumeKnowledgeGradient],
            {
                "model": model_list,
                "training_data": self.blockX_blockY,
                "bounds": bounds,
                "target_fidelities": {0: 0.987},
                "num_fantasies": 30,
                "objective_thresholds": objective_thresholds,
                **constraints_tuple_dict,
            },
        )

        pref_model = self.mock_model
        pref_model.dim = 2
        pref_model.datapoints = torch.tensor([])

        self.cases["EUBO"] = (
            [AnalyticExpectedUtilityOfBestOption, qExpectedUtilityOfBestOption],
            {"model": st_moo_model, "pref_model": pref_model, **constraints_tuple_dict},
        )
        self.cases["qJES"] = (
            [qJointEntropySearch],
            {
                "model": SingleTaskGP(self.blockX_blockY[0].X, self.blockX_blockY[0].Y),
                "bounds": self.bounds,
                **constraints_tuple_dict,
            },
        )
        self.cases["qSimpleRegret"] = (
            [qSimpleRegret],
            {
                "model": SingleTaskGP(self.blockX_blockY[0].X, self.blockX_blockY[0].Y),
                "training_data": self.blockX_blockY,
                "objective": LinearMCObjective(torch.rand(2)),
                **constraints_tuple_dict,
            },
        )
        self.cases["BayesianActiveLearning"] = (
            [qBayesianActiveLearningByDisagreement],
            {
                "model": SaasFullyBayesianSingleTaskGP(
                    self.blockX_blockY[0].X, self.blockX_blockY[0].Y
                ),
                **constraints_tuple_dict,
            },
        )
        self.cases["ActiveLearning"] = (
            [qNegIntegratedPosteriorVariance],
            {
                "model": SingleTaskGP(self.blockX_blockY[0].X, self.blockX_blockY[0].Y),
                "training_data": self.blockX_blockY,
                "bounds": self.bounds,
                **constraints_tuple_dict,
            },
        )

    @mock_optimize
    def test_constructors_can_instantiate(self) -> None:
        for key, (classes, input_constructor_kwargs) in self.cases.items():
            with self.subTest(
                key, classes=classes, input_constructor_kwargs=input_constructor_kwargs
            ):
                for cls_ in classes:
                    acqf_kwargs = get_acqf_input_constructor(cls_)(
                        **input_constructor_kwargs
                    )
                    # no assertions; we are just testing that this doesn't error
                    cls_(**acqf_kwargs)

    def test_all_cases_covered(self) -> None:
        all_classes_tested = reduce(
            lambda x, y: x + y, [cls_list for cls_list, _ in self.cases.values()]
        )
        for acqf_cls in ACQF_INPUT_CONSTRUCTOR_REGISTRY.keys():
            with self.subTest(acqf_cls=acqf_cls):
                self.assertIn(acqf_cls, all_classes_tested)
