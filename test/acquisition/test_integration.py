#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from warnings import catch_warnings, simplefilter

import torch
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
)
from botorch.acquisition.objective import LearnedObjective
from botorch.exceptions.warnings import InputDataWarning, NumericsWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class TestObjectiveAndConstraintIntegration(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.q = 3
        self.d = 2
        self.tkwargs = {"device": self.device, "dtype": torch.double}

    def _get_acqf_inputs(self, train_batch_shape: torch.Size, m: int) -> dict:
        train_x = torch.rand((*train_batch_shape, 5, self.d), **self.tkwargs)
        y = torch.rand((*train_batch_shape, 5, m), **self.tkwargs)

        training_data = SupervisedDataset(
            X=train_x,
            Y=y,
            feature_names=[f"x{i}" for i in range(self.d)],
            outcome_names=[f"y{i}" for i in range(m)],
        )
        utility = y.sum(-1).unsqueeze(-1)

        with catch_warnings():
            simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(train_x, y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll=mll)

        with catch_warnings():
            simplefilter("ignore", category=InputDataWarning)
            pref_model = SingleTaskGP(y, utility)
        pref_mll = ExactMarginalLogLikelihood(pref_model.likelihood, pref_model)
        fit_gpytorch_mll(mll=pref_mll)
        return {
            "training_data": training_data,
            "model": model,
            "pref_model": pref_model,
            "train_x": train_x,
        }

    def _base_test_with_learned_objective(
        self,
        train_batch_shape: torch.Size,
        prune_baseline: bool,
        test_batch_shape: torch.Size,
    ) -> None:
        acq_inputs = self._get_acqf_inputs(train_batch_shape=train_batch_shape, m=4)

        pref_sample_shapes = [1, 8]
        test_acqf_classes_and_kws = [
            (qExpectedImprovement, {}),
            (qProbabilityOfImprovement, {}),
            (qLogExpectedImprovement, {}),
            (qNoisyExpectedImprovement, {"prune_baseline": prune_baseline}),
            (qLogNoisyExpectedImprovement, {"prune_baseline": prune_baseline}),
        ]

        for (acqf_cls, kws), pref_sample_shape in product(
            test_acqf_classes_and_kws, pref_sample_shapes
        ):
            with self.subTest(
                train_batch_shape=train_batch_shape,
                test_batch_shape=test_batch_shape,
                prune_baseline=prune_baseline,
                acqf_cls=acqf_cls,
                pref_sample_shape=pref_sample_shape,
            ):
                objective = LearnedObjective(
                    pref_model=acq_inputs["pref_model"],
                    sample_shape=torch.Size([pref_sample_shape]),
                )
                test_x = torch.rand(
                    (*test_batch_shape, *train_batch_shape, self.q, self.d),
                    **self.tkwargs,
                )
                input_constructor = get_acqf_input_constructor(acqf_cls=acqf_cls)

                inputs = input_constructor(
                    objective=objective,
                    model=acq_inputs["model"],
                    training_data=acq_inputs["training_data"],
                    X_baseline=acq_inputs["train_x"],
                    sampler=SobolQMCNormalSampler(torch.Size([4])),
                    **kws,
                )
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    acqf = acqf_cls(**inputs)
                acq_val = acqf(test_x)
                self.assertEqual(acq_val.shape.numel(), test_x.shape[:-2].numel())

    def test_with_learned_objective_train_data_not_batched(self) -> None:
        train_batch_shape = []
        test_batch_shapes = [[], [1], [2]]
        for test_batch_shape in test_batch_shapes:
            self._base_test_with_learned_objective(
                train_batch_shape=torch.Size(train_batch_shape),
                prune_baseline=True,
                test_batch_shape=torch.Size(test_batch_shape),
            )

    def test_with_learned_objective_train_data_1d_batch(self) -> None:
        train_batch_shape = [1]
        test_batch_shapes = [[], [1], [2]]
        for test_batch_shape in test_batch_shapes:
            self._base_test_with_learned_objective(
                train_batch_shape=torch.Size(train_batch_shape),
                # Batched inputs ``X_baseline`` are currently unsupported by
                # prune_inferior_points
                prune_baseline=False,
                test_batch_shape=torch.Size(test_batch_shape),
            )

    def test_with_learned_objective_train_data_batched(self) -> None:
        train_batch_shape = [3]
        test_batch_shapes = [[], [1], [2]]
        for test_batch_shape in test_batch_shapes:
            self._base_test_with_learned_objective(
                train_batch_shape=torch.Size(train_batch_shape),
                # Batched inputs ``X_baseline`` are currently unsupported by
                # prune_inferior_points
                prune_baseline=False,
                test_batch_shape=torch.Size(test_batch_shape),
            )

    def _base_test_without_learned_objective(
        self,
        train_batch_shape: torch.Size,
        prune_baseline: bool,
        test_batch_shape: torch.Size,
    ) -> None:
        inputs = self._get_acqf_inputs(train_batch_shape=train_batch_shape, m=1)
        constraints = [lambda y: y[..., 0]]
        test_x = torch.rand(
            (*test_batch_shape, *train_batch_shape, self.q, self.d), **self.tkwargs
        )

        input_constructor_kwargs = {
            "model": inputs["model"],
            "training_data": inputs["training_data"],
            "X_baseline": inputs["train_x"],
            "sampler": SobolQMCNormalSampler(torch.Size([4])),
        }

        for acqf_cls, kws in [
            (qNoisyExpectedImprovement, {"prune_baseline": prune_baseline}),
            (qLogNoisyExpectedImprovement, {"prune_baseline": prune_baseline}),
            (qExpectedImprovement, {}),
            (qLogExpectedImprovement, {}),
            (qProbabilityOfImprovement, {}),
        ]:
            input_constructor = get_acqf_input_constructor(acqf_cls=acqf_cls)

            with self.subTest(
                "no objective or constraints",
                train_batch_shape=train_batch_shape,
                prune_baseline=prune_baseline,
                test_batch_shape=test_batch_shape,
                acqf_cls=acqf_cls,
            ):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    acqf = acqf_cls(
                        **input_constructor(**input_constructor_kwargs, **kws)
                    )
                acq_val = acqf(test_x)
                self.assertEqual(acq_val.shape.numel(), test_x.shape[:-2].numel())

            with self.subTest(
                "constrained",
                train_batch_shape=train_batch_shape,
                prune_baseline=prune_baseline,
                test_batch_shape=test_batch_shape,
                acqf_cls=acqf_cls,
            ):
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    acqf = acqf_cls(
                        **input_constructor(**input_constructor_kwargs, **kws)
                    )
                    acqf = acqf_cls(
                        **input_constructor(
                            constraints=constraints, **input_constructor_kwargs, **kws
                        )
                    )
                self.assertEqual(acq_val.shape.numel(), test_x.shape[:-2].numel())
                acq_val = acqf(test_x)

    def test_without_learned_objective(self) -> None:
        train_batch_shapes = [[], [1], [2]]
        test_batch_shapes = [[], [1], [3]]
        for train_batch_shape, test_batch_shape in product(
            train_batch_shapes, test_batch_shapes
        ):
            # Batched inputs ``X_baseline`` are currently unsupported by
            # prune_inferior_points
            prune_baseline_ = [False] if len(train_batch_shape) > 0 else [False, True]
            for prune_baseline in prune_baseline_:
                self._base_test_without_learned_objective(
                    train_batch_shape=torch.Size(train_batch_shape),
                    prune_baseline=prune_baseline,
                    test_batch_shape=torch.Size(test_batch_shape),
                )


class TestInputConstructorIntegration(BotorchTestCase):
    def _base_test_input_consructor(
        self, test_batch_shape: torch.Size, train_batch_shape: torch.Size
    ) -> None:
        m = 1
        d = 2
        q = 3

        train_x = torch.rand(
            (*train_batch_shape, 5, d), device=self.device, dtype=torch.double
        )
        y = torch.rand(
            (*train_batch_shape, 5, m), device=self.device, dtype=torch.double
        )

        training_data = SupervisedDataset(
            X=train_x,
            Y=y,
            feature_names=[f"x{i}" for i in range(d)],
            outcome_names=[f"y{i}" for i in range(m)],
        )

        with catch_warnings():
            simplefilter("ignore", category=InputDataWarning)
            model = SingleTaskGP(train_x, y)

        test_x = torch.rand(
            (*test_batch_shape, q, d), device=self.device, dtype=torch.double
        )

        for acqf_cls, kws in [
            (qNoisyExpectedImprovement, {"prune_baseline": False}),
            (qLogNoisyExpectedImprovement, {"prune_baseline": False}),
            (qExpectedImprovement, {}),
            (qProbabilityOfImprovement, {}),
            (qLogExpectedImprovement, {}),
        ]:
            with self.subTest(acqf_cls=acqf_cls):
                input_constructor = get_acqf_input_constructor(acqf_cls=acqf_cls)
                with catch_warnings():
                    simplefilter("ignore", category=NumericsWarning)
                    acqf = acqf_cls(
                        **input_constructor(
                            model=model,
                            training_data=training_data,
                            X_baseline=train_x,
                            sampler=SobolQMCNormalSampler(torch.Size([4])),
                            **kws,
                        )
                    )
                acq_val = acqf(test_x)
                self.assertEqual(acq_val.numel(), torch.Size(test_batch_shape).numel())

    def test_input_constructor_not_batched(self) -> None:
        self._base_test_input_consructor(
            test_batch_shape=torch.Size([]),
            train_batch_shape=torch.Size([]),
        )

    def test_input_constructor_batched(self) -> None:
        self._base_test_input_consructor(
            test_batch_shape=torch.Size([6]),
            train_batch_shape=torch.Size([]),
        )
        self._base_test_input_consructor(
            test_batch_shape=torch.Size([6]),
            train_batch_shape=torch.Size([6]),
        )
