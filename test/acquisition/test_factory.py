#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from unittest import mock

import torch
from botorch.acquisition import logei, monte_carlo
from botorch.acquisition.factory import get_acquisition_function
from botorch.acquisition.multi_objective import (
    logei as moo_logei,
    MCMultiOutputObjective,
    monte_carlo as moo_monte_carlo,
)
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import compute_best_feasible_objective
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class DummyMCObjective(MCAcquisitionObjective):
    def forward(self, samples: Tensor, X=None) -> Tensor:
        return samples.sum(-1)


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples: Tensor, X=None) -> Tensor:
        return samples


class TestGetAcquisitionFunction(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.model = MockModel(MockPosterior())
        self.objective = DummyMCObjective()
        self.X_observed = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.X_pending = torch.tensor([[1.0, 3.0, 4.0]])
        self.mc_samples = 250
        self.qmc = True
        self.ref_point = [0.0, 0.0]
        self.mo_objective = DummyMCMultiOutputObjective()
        self.Y = torch.tensor([[1.0, 2.0]])  # (2 x 1)-dim multi-objective outcomes
        self.seed = 1

    @mock.patch(f"{monte_carlo.__name__}.qExpectedImprovement")
    def test_GetQEI(self, mock_acqf):
        self._test_GetQEI(acqf_name="qEI", mock_acqf=mock_acqf)

    @mock.patch(f"{logei.__name__}.qLogExpectedImprovement")
    def test_GetQLogEI(self, mock_acqf):
        self._test_GetQEI(acqf_name="qLogEI", mock_acqf=mock_acqf)

    def _test_GetQEI(self, acqf_name: str, mock_acqf):
        n = len(self.X_observed)
        mean = torch.arange(n, dtype=torch.double).view(-1, 1)
        var = torch.ones_like(mean)
        self.model = MockModel(MockPosterior(mean=mean, variance=var))
        common_kwargs = {
            "model": self.model,
            "objective": self.objective,
            "X_observed": self.X_observed,
            "X_pending": self.X_pending,
            "mc_samples": self.mc_samples,
            "seed": self.seed,
        }
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            marginalize_dim=0,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model,
            best_f=best_f,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            constraints=None,
            eta=1e-3,
        )
        # test batched model
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2, 1)))
        common_kwargs.update({"model": self.model})
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name, **common_kwargs
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        # test batched model without marginalize dim
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

        # test w/ posterior transform
        pm = torch.tensor([1.0, 2.0])
        mvn = MultivariateNormal(pm, torch.eye(2))
        self.model._posterior.distribution = mvn
        self.model._posterior._mean = pm.unsqueeze(-1)
        common_kwargs.update({"model": self.model})
        pt = ScalarizedPosteriorTransform(weights=torch.tensor([-1]))
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            posterior_transform=pt,
            marginalize_dim=0,
        )
        self.assertEqual(mock_acqf.call_args[-1]["best_f"].item(), -1.0)

        # with constraints
        upper_bound = self.Y[0, 0] + 1 / 2  # = 1.5
        constraints = [lambda samples: samples[..., 0] - upper_bound]
        eta = math.pi * 1e-2  # testing non-standard eta

        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            marginalize_dim=0,
            constraints=constraints,
            eta=eta,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        best_feasible_f = compute_best_feasible_objective(
            samples=mean,
            obj=self.objective(mean),
            constraints=constraints,
            model=self.model,
            objective=self.objective,
            X_baseline=self.X_observed,
        )
        mock_acqf.assert_called_with(
            model=self.model,
            best_f=best_feasible_f,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            constraints=constraints,
            eta=eta,
        )

    @mock.patch(f"{monte_carlo.__name__}.qProbabilityOfImprovement")
    def test_GetQPI(self, mock_acqf):
        # basic test
        n = len(self.X_observed)
        mean = torch.arange(n, dtype=torch.double).view(-1, 1)
        var = torch.ones_like(mean)
        self.model = MockModel(MockPosterior(mean=mean, variance=var))
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        best_f = self.objective(self.model.posterior(self.X_observed).mean).max().item()
        mock_acqf.assert_called_once_with(
            model=self.model,
            best_f=best_f,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            tau=1e-3,
            constraints=None,
            eta=1e-3,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with different tau, non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            tau=1.0,
        )
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["tau"], 1.0)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            tau=1.0,
        )
        # test batched model
        self.model = MockModel(MockPosterior(mean=torch.zeros(1, 2, 1)))
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertEqual(acqf, mock_acqf.return_value)

        # with constraints
        n = len(self.X_observed)
        mean = torch.arange(n, dtype=torch.double).view(-1, 1)
        var = torch.ones_like(mean)
        self.model = MockModel(MockPosterior(mean=mean, variance=var))
        upper_bound = self.Y[0, 0] + 1 / 2  # = 1.5
        constraints = [lambda samples: samples[..., 0] - upper_bound]
        eta = math.pi * 1e-2  # testing non-standard eta
        acqf = get_acquisition_function(
            acquisition_function_name="qPI",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            marginalize_dim=0,
            constraints=constraints,
            eta=eta,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        best_feasible_f = compute_best_feasible_objective(
            samples=mean,
            obj=self.objective(mean),
            constraints=constraints,
            model=self.model,
            objective=self.objective,
            X_baseline=self.X_observed,
        )
        mock_acqf.assert_called_with(
            model=self.model,
            best_f=best_feasible_f,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            tau=1e-3,
            constraints=constraints,
            eta=eta,
        )

    @mock.patch(f"{monte_carlo.__name__}.qNoisyExpectedImprovement")
    def test_GetQNEI(self, mock_acqf):
        self._test_GetQNEI(acqf_name="qNEI", mock_acqf=mock_acqf)

    @mock.patch(f"{logei.__name__}.qLogNoisyExpectedImprovement")
    def test_GetQLogNEI(self, mock_acqf):
        self._test_GetQNEI(acqf_name="qLogNEI", mock_acqf=mock_acqf)

    def _test_GetQNEI(self, acqf_name: str, mock_acqf):
        # basic test
        n = len(self.X_observed)
        mean = torch.arange(n, dtype=torch.double).view(-1, 1)
        var = torch.ones_like(mean)
        self.model = MockModel(MockPosterior(mean=mean, variance=var))
        common_kwargs = {
            "model": self.model,
            "objective": self.objective,
            "X_observed": self.X_observed,
            "X_pending": self.X_pending,
            "mc_samples": self.mc_samples,
            "seed": self.seed,
        }
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            marginalize_dim=0,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        self.assertEqual(mock_acqf.call_count, 1)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertEqual(kwargs["marginalize_dim"], 0)
        self.assertIsNone(kwargs["cache_root"])
        # test with cache_root = False
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            marginalize_dim=0,
            cache_root=False,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(kwargs["cache_root"], False)
        # test with non-qmc, no X_pending
        common_kwargs.update({"X_pending": None})
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
        )
        self.assertEqual(mock_acqf.call_count, 3)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))
        self.assertEqual(kwargs["X_pending"], None)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_baseline"], self.X_observed))

        # with constraints
        upper_bound = self.Y[0, 0] + 1 / 2  # = 1.5
        constraints = [lambda samples: samples[..., 0] - upper_bound]
        eta = math.pi * 1e-2  # testing non-standard eta
        common_kwargs.update({"X_pending": self.X_pending})
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            marginalize_dim=0,
            constraints=constraints,
            eta=eta,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_with(
            model=self.model,
            X_baseline=self.X_observed,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            prune_baseline=True,
            marginalize_dim=0,
            cache_root=None,
            constraints=constraints,
            eta=eta,
        )

    @mock.patch(f"{monte_carlo.__name__}.qSimpleRegret")
    def test_GetQSR(self, mock_acqf):
        # basic test
        acqf = get_acquisition_function(
            acquisition_function_name="qSR",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            model=self.model,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qSR",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
        )
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

    @mock.patch(f"{monte_carlo.__name__}.qUpperConfidenceBound")
    def test_GetQUCB(self, mock_acqf):
        # make sure beta is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name="qUCB",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )
        acqf = get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            beta=0.3,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            model=self.model,
            beta=0.3,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))
        # test with different tau, non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name="qUCB",
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            beta=0.2,
        )
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["beta"], 0.2)
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        self.assertTrue(torch.equal(kwargs["X_pending"], self.X_pending))

    @mock.patch(f"{moo_monte_carlo.__name__}.qExpectedHypervolumeImprovement")
    def test_GetQEHVI(self, mock_acqf):
        self._test_GetQEHVI(acqf_name="qEHVI", mock_acqf=mock_acqf)

    @mock.patch(f"{moo_logei.__name__}.qLogExpectedHypervolumeImprovement")
    def test_GetQLogEHVI(self, mock_acqf):
        self._test_GetQEHVI(acqf_name="qLogEHVI", mock_acqf=mock_acqf)

    def _test_GetQEHVI(self, acqf_name: str, mock_acqf):
        # make sure ref_point is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name=acqf_name,
                model=self.model,
                objective=self.mo_objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                Y=self.Y,
            )
        # make sure Y is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name=acqf_name,
                model=self.model,
                objective=self.mo_objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                ref_point=self.ref_point,
            )
        # posterior transforms are not supported
        with self.assertRaises(NotImplementedError):
            acqf = get_acquisition_function(
                acquisition_function_name=acqf_name,
                model=self.model,
                objective=self.mo_objective,
                posterior_transform=ScalarizedPosteriorTransform(weights=torch.rand(2)),
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
                ref_point=self.ref_point,
            )
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            constraints=None,
            eta=1e-3,
            model=self.model,
            objective=self.mo_objective,
            ref_point=self.ref_point,
            partitioning=mock.ANY,
            sampler=mock.ANY,
            X_pending=self.X_pending,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)

        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["ref_point"], self.ref_point)
        sampler = kwargs["sampler"]
        self.assertIsInstance(kwargs["objective"], DummyMCMultiOutputObjective)
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, FastNondominatedPartitioning)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)
        # test that approximate partitioning is used when alpha > 0
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
            alpha=0.1,
        )
        _, kwargs = mock_acqf.call_args
        partitioning = kwargs["partitioning"]
        self.assertIsInstance(partitioning, NondominatedPartitioning)
        self.assertEqual(partitioning.alpha, 0.1)
        # test constraints
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.mo_objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            constraints=[lambda Y: Y[..., -1]],
            eta=1e-2,
            seed=2,
            ref_point=self.ref_point,
            Y=self.Y,
        )
        _, kwargs = mock_acqf.call_args
        partitioning = kwargs["partitioning"]
        self.assertEqual(partitioning.pareto_Y.shape[0], 0)
        self.assertEqual(kwargs["eta"], 1e-2)

    @mock.patch(f"{moo_monte_carlo.__name__}.qNoisyExpectedHypervolumeImprovement")
    def test_GetQNEHVI(self, mock_acqf):
        self._test_GetQNEHVI(acqf_name="qNEHVI", mock_acqf=mock_acqf)

    @mock.patch(f"{moo_logei.__name__}.qLogNoisyExpectedHypervolumeImprovement")
    def test_GetQLogNEHVI(self, mock_acqf):
        self._test_GetQNEHVI(acqf_name="qLogNEHVI", mock_acqf=mock_acqf)

    def _test_GetQNEHVI(self, acqf_name: str, mock_acqf):
        # make sure ref_point is specified
        with self.assertRaises(ValueError):
            acqf = get_acquisition_function(
                acquisition_function_name=acqf_name,
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=self.seed,
            ref_point=self.ref_point,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_once_with(
            constraints=None,
            eta=1e-3,
            model=self.model,
            X_baseline=self.X_observed,
            objective=self.objective,
            ref_point=self.ref_point,
            sampler=mock.ANY,
            prune_baseline=True,
            alpha=0.0,
            X_pending=self.X_pending,
            marginalize_dim=None,
            cache_root=None,
        )
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        sampler = kwargs["sampler"]
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 1)
        # test with non-qmc
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
        )
        self.assertEqual(mock_acqf.call_count, 2)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["ref_point"], self.ref_point)
        sampler = kwargs["sampler"]
        ref_point = kwargs["ref_point"]
        self.assertEqual(ref_point, self.ref_point)
        self.assertEqual(sampler.sample_shape, torch.Size([self.mc_samples]))
        self.assertEqual(sampler.seed, 2)

        # test passing alpha
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            model=self.model,
            objective=self.objective,
            X_observed=self.X_observed,
            X_pending=self.X_pending,
            mc_samples=self.mc_samples,
            seed=2,
            ref_point=self.ref_point,
            alpha=0.01,
        )
        self.assertEqual(mock_acqf.call_count, 3)
        args, kwargs = mock_acqf.call_args
        self.assertEqual(kwargs["alpha"], 0.01)

    def test_GetUnknownAcquisitionFunction(self):
        with self.assertRaises(NotImplementedError):
            get_acquisition_function(
                acquisition_function_name="foo",
                model=self.model,
                objective=self.objective,
                X_observed=self.X_observed,
                X_pending=self.X_pending,
                mc_samples=self.mc_samples,
                seed=self.seed,
            )

    @mock.patch(f"{logei.__name__}.qLogProbabilityOfFeasibility")
    def test_GetQLogPF(self, mock_acqf):
        acqf_name = "qLogPF"
        # basic test
        n = len(self.X_observed)
        mean = torch.arange(n, dtype=torch.double).view(-1, 1)
        var = torch.ones_like(mean)
        self.model = MockModel(MockPosterior(mean=mean, variance=var))
        common_kwargs = {
            "model": self.model,
            "objective": self.objective,
            "X_observed": self.X_observed,
            "X_pending": self.X_pending,
            "mc_samples": self.mc_samples,
            "seed": self.seed,
        }
        # with constraints
        upper_bound = self.Y[0, 0] + 1 / 2  # = 1.5
        constraints = [lambda samples: samples[..., 0] - upper_bound]
        eta = math.pi * 1e-2  # testing non-standard eta
        acqf = get_acquisition_function(
            acquisition_function_name=acqf_name,
            **common_kwargs,
            constraints=constraints,
            eta=eta,
        )
        self.assertEqual(acqf, mock_acqf.return_value)
        mock_acqf.assert_called_with(
            model=self.model,
            constraints=constraints,
            sampler=mock.ANY,
            objective=self.objective,
            posterior_transform=None,
            X_pending=self.X_pending,
            eta=eta,
        )
