#!/usr/bin/env python3

import unittest
from unittest import mock

import torch
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
)
from botorch.acquisition.sampler import IIDNormalSampler
from botorch.benchmarks.config import AcquisitionFunctionConfig, OptimizeConfig
from botorch.benchmarks.optimize import (
    _fit_model_and_get_best_point,
    _get_fitted_model,
    greedy,
    run_benchmark,
    run_closed_loop,
)
from botorch.benchmarks.output import (
    BenchmarkOutput,
    ClosedLoopOutput,
    _ModelBestPointOutput,
)
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.utils.transforms import squeeze_last_dim
from gpytorch.likelihoods import (
    GaussianLikelihood,
    HeteroskedasticNoise,
    _GaussianLikelihoodBase,
)
from torch import Tensor

from ..mock import MockModel, MockPosterior


def gen_x_uniform(b: int, q: int, bounds: Tensor) -> Tensor:
    """Generate `b` random `q`-batches with elements within the specified bounds.

    Args:
        n: The number of `q`-batches to sample.
        q: The size of the `q`-batches.
        bounds: A `2 x d` tensor where bounds[0] (bounds[1]) contains the lower
            (upper) bounds for each column.

    Returns:
        A `b x q x d` tensor with elements uniformly sampled from the box
            specified by bounds.

    """
    x_ranges = torch.sum(bounds * torch.tensor([[-1.0], [1.0]]).to(bounds), dim=0)
    return bounds[0] + torch.rand((b, q, bounds.shape[1])).to(bounds) * x_ranges


def get_bounds(cuda, dtype):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    return torch.tensor([-1.0, 1.0], dtype=dtype, device=device).view(2, 1)


def get_gen_x(bounds):
    def gen_x(b, q):
        return gen_x_uniform(b, q, bounds=bounds)

    return gen_x


class TestGenXUniform(unittest.TestCase):
    def setUp(self):
        self.bounds = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 4.0, 5.0, 7.0]])
        self.d = self.bounds.shape[-1]

    def test_GenXUniform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            bnds = self.bounds.to(dtype=dtype, device=device)
            X = gen_x_uniform(2, 3, bnds)
            self.assertTrue(X.shape == torch.Size([2, 3, self.d]))
            X_flat = X.view(-1, self.d)
            self.assertTrue(torch.all(X_flat.max(0)[0] <= bnds[1]))
            self.assertTrue(torch.all(X_flat.min(0)[0] >= bnds[0]))

    def test_GenXUniform_cuda(self):
        if torch.cuda.is_available():
            self.test_GenXUniform(cuda=True)


class TestRunClosedLoop(unittest.TestCase):
    def setUp(self):
        self.optim_config = OptimizeConfig(
            joint_optimization=True,
            initial_points=5,
            q=2,
            n_batch=1,
            model_fit_options={"maxiter": 3},
            num_starting_points=1,
            num_raw_samples=2,
            max_retries=0,
        )
        self.acq_func_config = AcquisitionFunctionConfig(
            name="qEI", args={"mc_samples": 20, "qmc": False}
        )

        def test_func(X):
            f_X = torch.sin(X)
            f_X_noisy = f_X + torch.randn_like(f_X)
            return f_X_noisy, torch.tensor([]).to(X)

        self.func = test_func

    @mock.patch("botorch.benchmarks.optimize.joint_optimize")
    @mock.patch("botorch.benchmarks.optimize._get_fitted_model")
    def test_run_closed_loop(
        self, mock_get_fitted_model, mock_joint_optimize, cuda=False
    ):
        for dtype in (torch.float, torch.double):
            bounds = get_bounds(cuda=cuda, dtype=dtype)
            tkwargs = {"dtype": dtype, "device": bounds.device}

            def gen_x(b, q):
                return gen_x_uniform(b, q, bounds=bounds)

            mock_joint_optimize.side_effect = [
                gen_x(self.optim_config.num_starting_points, self.optim_config.q)
                for _ in range(self.optim_config.n_batch)
            ]
            X = torch.zeros((self.optim_config.initial_points, 1), **tkwargs)
            Y, Ycov = self.func(X)
            mean1 = torch.ones(self.optim_config.initial_points, **tkwargs)
            samples1 = torch.zeros(1, self.optim_config.initial_points, 1, **tkwargs)
            mm1 = MockModel(MockPosterior(mean=mean1, samples=samples1))
            samples2 = torch.zeros(1, self.optim_config.q, 1, **tkwargs)
            mm2 = MockModel(MockPosterior(samples=samples2))
            mock_get_fitted_model.side_effect = [mm1, mm2]
            # basic test for output shapes and types
            best_point = X[0].view(-1, 1)
            obj = torch.tensor(0.0, **tkwargs)
            feas = torch.tensor(1.0, **tkwargs)
            output = ClosedLoopOutput(
                Xs=[X],
                Ys=[Y],
                Ycovs=[Ycov],
                best=[best_point],
                best_model_objective=[obj],
                best_model_feasibility=[feas],
                costs=[1.0],
                runtimes=[1.0],
            )
            output = run_closed_loop(
                func=self.func,
                acq_func_config=self.acq_func_config,
                optim_config=self.optim_config,
                model=mm1,
                output=output,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
            self.assertEqual(len(output.Xs), 2)
            self.assertEqual(output.Xs[0].shape[0], self.optim_config.initial_points)
            self.assertEqual(output.Xs[1].shape[0], self.optim_config.q)
            self.assertEqual(len(output.Ys), 2)
            self.assertEqual(output.Ys[0].shape[0], self.optim_config.initial_points)
            self.assertEqual(output.Ys[1].shape[0], self.optim_config.q)
            self.assertEqual(len(output.best), 2)
            self.assertEqual(output.best[1].shape, best_point.shape)
            self.assertEqual(len(output.best_model_objective), 2)
            self.assertEqual(output.best_model_objective[1].shape, obj.shape)
            self.assertEqual(len(output.best_model_feasibility), 2)
            self.assertEqual(output.best_model_feasibility[1].shape, feas.shape)
            self.assertEqual(
                output.costs, [1.0 for _ in range(self.optim_config.n_batch + 1)]
            )
            self.assertEqual(len(output.runtimes), 2)

    def test_run_closed_loop_cuda(self):
        if torch.cuda.is_available():
            self.test_run_closed_loop(cuda=True)


class TestRunBenchmark(unittest.TestCase):
    def setUp(self):
        self.optim_config = OptimizeConfig(
            joint_optimization=True,
            initial_points=5,
            q=2,
            n_batch=2,
            model_fit_options={"maxiter": 3},
            num_starting_points=1,
            num_raw_samples=2,
            max_retries=0,
        )
        self.acq_func_configs = {
            "test_qEI": AcquisitionFunctionConfig(
                name="qEI", args={"mc_samples": 20, "qmc": False}
            )
        }
        self.func = lambda X: (X + 0.25, torch.tensor([]))
        self.global_optimum = 5.0

    @mock.patch("botorch.benchmarks.optimize._fit_model_and_get_best_point")
    @mock.patch("botorch.benchmarks.optimize.run_closed_loop")
    def test_run_benchmark(
        self, mock_run_closed_loop, mock_fit_model_and_get_best_point, cuda=False
    ):
        for dtype in [torch.float, torch.double]:
            bounds = get_bounds(cuda, dtype=dtype)
            tkwargs = {"dtype": dtype, "device": bounds.device}
            init_X = torch.tensor([-1.0, 0.0], **tkwargs).view(-1, 1)
            Xs = [init_X, torch.tensor([1.0, 0.0], **tkwargs).view(-1, 1)]
            Ys = []
            best_model_objective = []
            best_model_feasibility = []
            best = []
            costs = []
            for X in Xs:
                Ys.append(X + 0.5)
                Ycovs = torch.zeros_like(Ys[-1])
                best_obj, best_idx = torch.max(X, dim=0)
                best_model_objective.append(best_obj.item())
                best.append(X[best_idx])
                best_model_feasibility.append(1.0)
                costs.append(1.0)
            closed_loop_output = ClosedLoopOutput(
                Xs=Xs,
                Ys=Ys,
                Ycovs=Ycovs,
                best=best,
                best_model_objective=best_model_objective,
                best_model_feasibility=best_model_feasibility,
                costs=costs,
                runtimes=[1.0],
            )
            mock_run_closed_loop.return_value = closed_loop_output
            model = MockModel(posterior=MockPosterior())
            mock_fit_model_and_get_best_point.return_value = _ModelBestPointOutput(
                model=model,
                best_point=best[0],
                obj=best_model_objective[0],
                feas=1.0,
                retry=0,
            )
            outputs_dict = run_benchmark(
                func=self.func,
                acq_func_configs=self.acq_func_configs,
                optim_config=self.optim_config,
                initial_model=model,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                num_runs=2,
                true_func=self.func,
                global_optimum=self.global_optimum,
            )
            outputs = outputs_dict["test_qEI"]
            self.assertTrue(isinstance(outputs, BenchmarkOutput))
            # Check 2 trials
            self.assertEqual(len(outputs.Xs), 2)
            # Check iterations
            self.assertEqual(len(outputs.Xs[0]), 2)
            expected_best_true_objective = torch.tensor([0.25, 1.25], **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs.best_true_objective[0], expected_best_true_objective
                )
            )
            expected_best_true_feasibility = torch.ones(2, **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs.best_true_feasibility[0], expected_best_true_feasibility
                )
            )
            expected_regrets_trial = torch.tensor(
                [(self.global_optimum - self.func(X)[0]).sum().item() for X in Xs]
            ).to(Xs[0])
            self.assertEqual(len(outputs.regrets[0]), len(expected_regrets_trial))
            self.assertTrue(
                torch.equal(outputs.regrets[0][0], expected_regrets_trial[0])
            )
            self.assertTrue(
                torch.equal(
                    outputs.cumulative_regrets[0],
                    torch.cumsum(expected_regrets_trial, dim=0),
                )
            )
            # test modifying the objective
            weights = torch.full((1,), 0.1, **tkwargs)
            outputs_dict2 = run_benchmark(
                func=self.func,
                acq_func_configs=self.acq_func_configs,
                optim_config=self.optim_config,
                initial_model=model,
                objective=LinearMCObjective(weights),
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                true_func=self.func,
            )
            expected_best_true_objective2 = torch.tensor([0.025, 0.125], **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs_dict2["test_qEI"].best_true_objective[0],
                    expected_best_true_objective2,
                )
            )
            # test constraints
            objective = ConstrainedMCObjective(
                objective=squeeze_last_dim,
                constraints=[lambda Y: torch.ones_like(Y[..., 0])],
            )
            outputs_dict3 = run_benchmark(
                func=self.func,
                acq_func_configs=self.acq_func_configs,
                optim_config=self.optim_config,
                initial_model=model,
                objective=objective,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                true_func=self.func,
            )
            expected_best_true_feasibility = torch.zeros(2, **tkwargs)
            self.assertTrue(
                torch.equal(
                    outputs_dict3["test_qEI"].best_true_feasibility[0],
                    expected_best_true_feasibility,
                )
            )

    def test_run_benchmark_cuda(self):
        if torch.cuda.is_available():
            self.test_run_benchmark(cuda=True)


class TestGreedy(unittest.TestCase):
    def test_greedy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        sampler = IIDNormalSampler(num_samples=2)
        for dtype in (torch.float, torch.double):
            X = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
            X = X.to(device=device, dtype=dtype)
            model = MockModel(MockPosterior(samples=X.view(1, -1, 1) * 2))
            # basic test
            best_point, best_obj, best_feas = greedy(
                X=X, model=model, objective=IdentityMCObjective(), sampler=sampler
            )
            best_point_exp = torch.tensor([3.0]).to(X)
            best_obj_exp = torch.tensor(6.0).to(X)
            best_feas_exp = torch.tensor(1.0).to(X)
            self.assertTrue(torch.equal(best_point, best_point_exp))
            # interestingly, these are not exactly the same on the GPU
            print(f"best_obj: {best_obj}")
            print(f"best_obj_exp: {best_obj_exp}")
            self.assertTrue(torch.allclose(best_obj, best_obj_exp))
            self.assertTrue(torch.allclose(best_feas, best_feas_exp))
            # test objective
            weights = torch.full((1,), 0.5).to(X)
            best_point2, best_obj2, best_feas2 = greedy(
                X=X,
                model=model,
                objective=LinearMCObjective(weights=weights),
                sampler=sampler,
            )
            self.assertTrue(torch.equal(best_point2, best_point_exp))
            self.assertTrue(torch.allclose(best_obj2, 0.5 * best_obj_exp))
            self.assertTrue(torch.allclose(best_feas2, best_feas_exp))
            # test constraints
            objective = ConstrainedMCObjective(
                objective=squeeze_last_dim,
                constraints=[lambda Y: torch.ones_like(Y[..., 0])],
            )
            _, _, best_feas3 = greedy(
                X=X, model=model, objective=objective, sampler=sampler
            )
            best_feas3_exp = torch.tensor(0.0).to(X)
            self.assertTrue(torch.allclose(best_feas3, best_feas3_exp))

    def test_greedy_cuda(self):
        if torch.cuda.is_available():
            self.test_greedy(cuda=True)

    def test_greedy_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        sampler = IIDNormalSampler(num_samples=2)
        for dtype in (torch.float, torch.double):
            X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unsqueeze(-1)
            X = X.to(device=device, dtype=dtype)
            model = MockModel(MockPosterior(samples=X.view(2, -1, 1) * 2))
            # basic test
            best_point, best_obj, best_feas = greedy(
                X=X, model=model, objective=IdentityMCObjective(), sampler=sampler
            )
            best_point_exp = torch.tensor([[3.0], [6.0]]).to(X)
            best_obj_exp = torch.tensor([6.0, 12.0]).to(X)
            best_feas_exp = torch.tensor([1.0, 1.0]).to(X)
            self.assertTrue(torch.equal(best_point, best_point_exp))
            # interestingly, these are not exactly the same on the GPU
            self.assertTrue(torch.allclose(best_obj, best_obj_exp))
            self.assertTrue(torch.allclose(best_feas, best_feas_exp))
            # test objective
            weights = torch.full((1,), 0.5).to(X)

            best_point2, best_obj2, best_feas2 = greedy(
                X=X,
                model=model,
                objective=LinearMCObjective(weights=weights),
                sampler=sampler,
            )
            self.assertTrue(torch.equal(best_point2, best_point_exp))
            self.assertTrue(torch.allclose(best_obj2, 0.5 * best_obj_exp))
            self.assertTrue(torch.allclose(best_feas2, best_feas_exp))
            # test constraints
            objective = ConstrainedMCObjective(
                objective=squeeze_last_dim,
                constraints=[lambda Y: torch.ones_like(Y[..., 0])],
            )
            _, __, best_feas3 = greedy(
                X=X, model=model, objective=objective, sampler=sampler
            )
            best_feas3_exp = torch.tensor([0.0, 0.0]).to(X)
            self.assertTrue(torch.allclose(best_feas3, best_feas3_exp))

    def test_greedy_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_greedy_batch(cuda=True)


class TestGetFittedModel(unittest.TestCase):
    @mock.patch("botorch.benchmarks.optimize.fit_gpytorch_model")
    def test_get_fitted_model(self, mock_fit_model, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            init_X = torch.rand((5, 2), dtype=dtype, device=device)
            init_Y = torch.rand(5, dtype=dtype, device=device)
            init_Y_se = torch.rand(5, dtype=dtype, device=device)
            initial_model = SingleTaskGP(train_X=init_X, train_Y=init_Y)
            train_X = torch.rand((5, 2), dtype=dtype, device=device)
            train_Y = torch.rand(5, dtype=dtype, device=device)
            train_Y_se = torch.rand(5, dtype=dtype, device=device)
            model = _get_fitted_model(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_se=None,
                model=initial_model,
                options={"maxiter": 1},
                warm_start=False,
            )
            self.assertIsInstance(model, SingleTaskGP)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertTrue(torch.equal(model.train_inputs[0], train_X))
            self.assertTrue(torch.equal(model.train_targets, train_Y))
            initial_model2 = HeteroskedasticSingleTaskGP(
                train_X=init_X, train_Y=init_Y, train_Y_se=init_Y_se
            )
            model2 = _get_fitted_model(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_se=train_Y_se,
                model=initial_model2,
                options={"maxiter": 1},
                warm_start=False,
            )
            self.assertIsInstance(model2, HeteroskedasticSingleTaskGP)
            self.assertIsInstance(model2.likelihood, _GaussianLikelihoodBase)
            self.assertIsInstance(model2.likelihood.noise_covar, HeteroskedasticNoise)
            self.assertTrue(torch.equal(model2.train_inputs[0], train_X))
            self.assertTrue(torch.equal(model2.train_targets, train_Y))
        self.assertEqual(mock_fit_model.call_count, 4)

    def test_get_fitted_model_cuda(self):
        if torch.cuda.is_available():
            self.test_get_fitted_model(cuda=True)


class TestFitModelAndGetBestPoint(unittest.TestCase):
    @mock.patch("botorch.benchmarks.optimize.greedy")
    @mock.patch("botorch.benchmarks.optimize._get_fitted_model")
    def test_fit_model_and_get_best_point(
        self, mock_get_fitted_model, mock_greedy, cuda=False
    ):
        objective = IdentityMCObjective()
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            train_X = torch.rand((5, 2), dtype=dtype, device=device)
            train_Y = torch.rand(5, dtype=dtype, device=device)
            train_Y_se = torch.rand(5, dtype=dtype, device=device)
            model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
            mock_get_fitted_model.return_value = model
            exp_best_point = train_X[0]
            exp_obj = torch.tensor(2.0, dtype=dtype, device=device)
            exp_feas = torch.tensor(1.0, dtype=dtype, device=device)
            mock_greedy.return_value = (exp_best_point, exp_obj, exp_feas)
            model_fit_options = {"maxiter": 1}
            model_and_best_point_output = _fit_model_and_get_best_point(
                train_X=train_X,
                train_Y=train_Y,
                train_Y_se=train_Y_se,
                model=model,
                max_retries=0,
                model_fit_options={"maxiter": 1},
                warm_start=False,
                verbose=False,
                objective=objective,
                retry=0,
            )

            call_args = mock_get_fitted_model.call_args[-1]
            self.assertTrue(torch.equal(call_args["train_X"], train_X))
            self.assertTrue(torch.equal(call_args["train_Y"], train_Y))
            self.assertTrue(torch.equal(call_args["train_Y_se"], train_Y_se))
            self.assertEqual(call_args["model"], model)
            self.assertEqual(call_args["options"], model_fit_options)
            self.assertEqual(call_args["warm_start"], False)
            greedy_call_args = mock_greedy.call_args[-1]
            self.assertTrue(torch.equal(greedy_call_args["X"], train_X))
            self.assertEqual(greedy_call_args["model"], model)
            self.assertEqual(greedy_call_args["objective"], objective)
            self.assertTrue(
                torch.equal(model_and_best_point_output.best_point, exp_best_point)
            )
            self.assertTrue(torch.equal(model_and_best_point_output.obj, exp_obj))
            self.assertTrue(torch.equal(model_and_best_point_output.feas, exp_feas))
            self.assertEqual(model_and_best_point_output.retry, 0)

    def test_fit_model_and_get_best_point_cuda(self):
        if torch.cuda.is_available():
            self.test_fit_model_and_get_best_point(cuda=True)
