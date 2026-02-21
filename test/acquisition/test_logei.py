#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from itertools import product
from math import pi
from unittest import mock

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    LogImprovementMCAcquisitionFunction,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
)
from botorch.acquisition.input_constructors import ACQF_INPUT_CONSTRUCTOR_REGISTRY
from botorch.acquisition.logei import qLogProbabilityOfFeasibility
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions import BotorchWarning, UnsupportedError
from botorch.exceptions.errors import BotorchError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.low_rank import sample_cached_cholesky
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.test_helpers import DummyNonScalarizingPosteriorTransform
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import standardize
from torch import Tensor


def infeasible_con(samples: Tensor) -> Tensor:
    return torch.ones_like(samples[..., 0])


def feasible_con(samples: Tensor) -> Tensor:
    return -torch.ones_like(samples[..., 0])


class DummyLogImprovementAcquisitionFunction(LogImprovementMCAcquisitionFunction):
    def _sample_forward(self, X):
        pass


class TestLogImprovementAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            LogImprovementMCAcquisitionFunction()
        # raise if model is multi-output, but no outcome transform or objective
        # are given
        no = "botorch.utils.testing.MockModel.num_outputs"
        with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 2
            mm = MockModel(MockPosterior())
            with self.assertRaises(UnsupportedError):
                DummyLogImprovementAcquisitionFunction(model=mm)
        # raise if model is multi-output, but outcome transform does not
        # scalarize and no objetive is given
        with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 2
            mm = MockModel(MockPosterior())
            ptf = DummyNonScalarizingPosteriorTransform()
            with self.assertRaises(UnsupportedError):
                DummyLogImprovementAcquisitionFunction(
                    model=mm, posterior_transform=ptf
                )

        mm = MockModel(MockPosterior())
        objective = ConstrainedMCObjective(
            IdentityMCObjective(),
            constraints=[lambda samples: torch.zeros_like(samples[..., 0])],
        )
        with self.assertRaisesRegex(
            BotorchError,
            "Log-Improvement should not be used with `ConstrainedMCObjective`.",
        ):
            DummyLogImprovementAcquisitionFunction(model=mm, objective=objective)


class TestQLogExpectedImprovement(BotorchTestCase):
    def test_q_log_expected_improvement(self):
        self.assertIn(qLogExpectedImprovement, ACQF_INPUT_CONSTRUCTOR_REGISTRY.keys())
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # the event shape is ``b x q x t`` = 1 x 1 x 1
            samples = torch.zeros(1, 1, 1, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # X is ``q x d`` = 1 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(1, 1, **tkwargs)

            # basic test
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
            acqf = qExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=0, sampler=sampler)
            self.assertFalse(acqf._fat)  # different default behavior
            self.assertTrue(log_acqf._fat)
            # test initialization
            for k in ["objective", "sampler"]:
                self.assertIn(k, acqf._modules)
                self.assertIn(k, log_acqf._modules)

            res = acqf(X)
            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.item(), 0.0)
            exp_log_res = log_acqf(X).exp().item()
            # Due to the smooth approximation, the value at zero should be close to, but
            # not exactly zero, and upper-bounded by the tau hyperparameter.
            self.assertTrue(0 < exp_log_res)
            self.assertTrue(exp_log_res <= log_acqf.tau_relu)

            # test shifting best_f value downward to see non-zero improvement
            best_f = -1
            acqf = qExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            res, exp_log_res = acqf(X), log_acqf(X).exp()
            expected_val = -best_f

            self.assertEqual(res.dtype, dtype)
            self.assertEqual(res.device.type, self.device.type)
            self.assertEqual(res.item(), expected_val)
            # Further away from zero, the value is numerically indistinguishable with
            # single precision arithmetic.
            self.assertEqual(exp_log_res.dtype, dtype)
            self.assertEqual(exp_log_res.device.type, self.device.type)
            self.assertTrue(expected_val <= exp_log_res.item())
            self.assertTrue(exp_log_res.item() <= expected_val + log_acqf.tau_relu)

            # test shifting best_f value upward to see advantage of LogEI
            best_f = 1
            acqf = qExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            log_acqf = qLogExpectedImprovement(model=mm, best_f=best_f, sampler=sampler)
            res, log_res = acqf(X), log_acqf(X)
            exp_log_res = log_res.exp()
            expected_val = 0
            self.assertEqual(res.item(), expected_val)
            self.assertTrue(expected_val <= exp_log_res.item())
            self.assertTrue(exp_log_res.item() <= expected_val + log_acqf.tau_relu)
            # However, the log value is large and negative with non-vanishing gradients
            self.assertGreater(-1, log_res.item())
            self.assertGreater(log_res.item(), -100)

            # testing with illegal taus
            with self.assertRaisesRegex(ValueError, "tau_max is not a scalar:"):
                qLogExpectedImprovement(
                    model=mm, best_f=0, tau_max=torch.tensor([1, 2])
                )
            with self.assertRaisesRegex(ValueError, "tau_relu is non-positive:"):
                qLogExpectedImprovement(model=mm, best_f=0, tau_relu=-2)

    # # TODO: Test different objectives (incl. constraints)


class TestQLogNoisyExpectedImprovement(BotorchTestCase):
    def test_q_log_noisy_expected_improvement(self):
        self.assertIn(
            qLogNoisyExpectedImprovement, ACQF_INPUT_CONSTRUCTOR_REGISTRY.keys()
        )
        for dtype in (torch.float, torch.double):
            # the event shape is ``b x q x t`` = 1 x 2 x 1
            samples_noisy = torch.tensor([0.0, 1.0], device=self.device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 2, 1)
            # X_baseline is ``q' x d`` = 1 x 1
            X_baseline = torch.zeros(1, 1, device=self.device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is ``q x d`` = 1 x 1
            X = torch.zeros(1, 1, device=self.device, dtype=dtype)

            # basic test
            sampler = IIDNormalSampler(sample_shape=torch.Size([2]))
            kwargs = {
                "model": mm_noisy,
                "X_baseline": X_baseline,
                "sampler": sampler,
                "prune_baseline": False,
                "cache_root": False,
            }
            acqf = qNoisyExpectedImprovement(**kwargs)
            log_acqf = qLogNoisyExpectedImprovement(**kwargs)

            res = acqf(X)
            self.assertEqual(res.item(), 1.0)
            log_res = log_acqf(X)
            self.assertEqual(log_res.dtype, dtype)
            self.assertEqual(log_res.device.type, self.device.type)
            self.assertAllClose(log_res.exp().item(), 1.0)

            # test incremental
            # Check that adding a pending point is equivalent to adding a point to
            # X_baseline
            samples_noisy_pending = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device, dtype=dtype
            )
            samples_noisy_pending = samples_noisy_pending.view(1, 3, 1)
            mm_noisy_pending = MockModel(MockPosterior(samples=samples_noisy_pending))
            X2 = torch.zeros(
                1, 1, 1, device=self.device, dtype=dtype, requires_grad=True
            )
            for cache_root in (True, False):
                kwargs = {
                    "model": mm_noisy_pending,
                    "X_baseline": X_baseline,
                    "sampler": sampler,
                    "prune_baseline": False,
                    "cache_root": cache_root,
                    "incremental": True,
                }
                log_acqf = qLogNoisyExpectedImprovement(**kwargs)
                log_acqf.set_X_pending(X)
                self.assertIsNone(log_acqf.X_pending)
                self.assertTrue(
                    torch.equal(log_acqf.X_baseline, torch.cat([X_baseline, X], dim=0))
                )
                af_val1 = log_acqf(X2)
                kwargs = {
                    "model": mm_noisy_pending,
                    "X_baseline": torch.cat([X_baseline, X], dim=-2),
                    "sampler": sampler,
                    "prune_baseline": False,
                    "cache_root": cache_root,
                    "incremental": False,
                }
                log_acqf2 = qLogNoisyExpectedImprovement(**kwargs)
                af_val2 = log_acqf2(X2)
                self.assertAllClose(af_val1.item(), af_val2.item())
            # test reseting X_pending
            log_acqf.set_X_pending(None)
            self.assertTrue(torch.equal(log_acqf.X_baseline, X_baseline))

    def test_prune_baseline(self):
        no = "botorch.utils.testing.MockModel.num_outputs"
        prune = "botorch.acquisition.logei.prune_inferior_points"
        constraints = [lambda Y: Y[..., 1] + 0.1]
        # only the last sample if feasible and it has the worst objective value
        mc_obj = GenericMCObjective(objective=lambda Y, X: Y[..., 0])

        for dtype in (torch.float, torch.double):
            samples = torch.tensor(
                [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
                device=self.device,
                dtype=dtype,
            )
            mm = MockModel(
                MockPosterior(
                    samples=samples,
                )
            )
            X_baseline = torch.zeros(3, 1, device=self.device, dtype=dtype)
            with mock.patch(no, new_callable=mock.PropertyMock) as mock_num_outputs:
                mock_num_outputs.return_value = 2
                with mock.patch(prune, wraps=prune_inferior_points) as mock_prune:
                    acqf = qLogNoisyExpectedImprovement(
                        model=mm,
                        X_baseline=X_baseline,
                        prune_baseline=True,
                        cache_root=False,
                        objective=mc_obj,
                        constraints=constraints,
                    )
                mock_prune.assert_called_once()
                self.assertIs(mock_prune.call_args[1]["constraints"], constraints)
                self.assertTrue(torch.equal(acqf.X_baseline, X_baseline[[-1]]))
                # test marginalize_dim
                samples2 = torch.stack([samples, samples * 2], dim=0)
                mm = MockModel(
                    MockPosterior(
                        samples=samples2,
                    )
                )
                with mock.patch(prune, wraps=prune_inferior_points) as mock_prune:
                    acqf = qLogNoisyExpectedImprovement(
                        model=mm,
                        X_baseline=X_baseline,
                        prune_baseline=True,
                        cache_root=False,
                        marginalize_dim=-3,
                        objective=mc_obj,
                        constraints=constraints,
                    )
                mock_prune.assert_called_once()
                _, kwargs = mock_prune.call_args
                self.assertIs(kwargs["constraints"], constraints)
                self.assertTrue(torch.equal(acqf.X_baseline, X_baseline[[-1]]))
                self.assertEqual(kwargs["marginalize_dim"], -3)

    def test_cache_root(self):
        sample_cached_path = (
            "botorch.acquisition.cached_cholesky.sample_cached_cholesky"
        )
        raw_state_dict = {
            "likelihood.noise_covar.raw_noise": torch.tensor(
                [[0.0895], [0.2594]], dtype=torch.float64
            ),
            "mean_module.raw_constant": torch.tensor(
                [-0.4545, -0.1285], dtype=torch.float64
            ),
            "covar_module.raw_outputscale": torch.tensor(
                [1.4876, 1.4897], dtype=torch.float64
            ),
            "covar_module.base_kernel.raw_lengthscale": torch.tensor(
                [[[-0.7202, -0.2868]], [[-0.8794, -1.2877]]], dtype=torch.float64
            ),
        }
        # test batched models (e.g. for MCMC)
        for train_batch_shape, m, dtype in product(
            (torch.Size([]), torch.Size([3])), (1, 2), (torch.float, torch.double)
        ):
            state_dict = deepcopy(raw_state_dict)
            for k, v in state_dict.items():
                if m == 1:
                    v = v[0]
                if len(train_batch_shape) > 0:
                    v = v.unsqueeze(0).expand(*train_batch_shape, *v.shape)
                state_dict[k] = v
            tkwargs = {"device": self.device, "dtype": dtype}
            if m == 2:
                objective = GenericMCObjective(lambda Y, X: Y.sum(dim=-1))
            else:
                objective = None
            for k, v in state_dict.items():
                state_dict[k] = v.to(**tkwargs)
            all_close_kwargs = (
                {
                    "atol": 1e-1,
                    "rtol": 0.0,
                }
                if dtype == torch.float
                else {"atol": 1e-4, "rtol": 0.0}
            )
            torch.manual_seed(1234)
            train_X = torch.rand(*train_batch_shape, 3, 2, **tkwargs)
            train_Y = (
                torch.sin(train_X * 2 * pi)
                + torch.randn(*train_batch_shape, 3, 2, **tkwargs)
            )[..., :m]
            train_Y = standardize(train_Y)
            model = SingleTaskGP(
                train_X,
                train_Y,
            )
            if len(train_batch_shape) > 0:
                X_baseline = train_X[0]
            else:
                X_baseline = train_X
            model.load_state_dict(state_dict, strict=False)
            sampler = IIDNormalSampler(sample_shape=torch.Size([5]), seed=0)
            torch.manual_seed(0)
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_baseline,
                sampler=sampler,
                objective=objective,
                prune_baseline=False,
                cache_root=True,
            )

            orig_base_samples = acqf.base_sampler.base_samples.detach().clone()
            sampler2 = IIDNormalSampler(sample_shape=torch.Size([5]), seed=0)
            sampler2.base_samples = orig_base_samples
            torch.manual_seed(0)
            acqf_no_cache = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_baseline,
                sampler=sampler2,
                objective=objective,
                prune_baseline=False,
                cache_root=False,
            )
            for q, batch_shape in product(
                (1, 3), (torch.Size([]), torch.Size([3]), torch.Size([4, 3]))
            ):
                acqf.q_in = -1
                acqf_no_cache.q_in = -1
                test_X = (
                    0.3 + 0.05 * torch.randn(*batch_shape, q, 2, **tkwargs)
                ).requires_grad_(True)
                with mock.patch(
                    sample_cached_path, wraps=sample_cached_cholesky
                ) as mock_sample_cached:
                    torch.manual_seed(0)
                    val = acqf(test_X).exp()
                    mock_sample_cached.assert_called_once()
                val.sum().backward()
                base_samples = acqf.sampler.base_samples.detach().clone()
                X_grad = test_X.grad.clone()
                test_X2 = test_X.detach().clone().requires_grad_(True)
                acqf_no_cache.sampler.base_samples = base_samples
                with mock.patch(
                    sample_cached_path, wraps=sample_cached_cholesky
                ) as mock_sample_cached:
                    torch.manual_seed(0)
                    val2 = acqf_no_cache(test_X2).exp()
                    mock_sample_cached.assert_not_called()
                self.assertAllClose(val, val2, **all_close_kwargs)
                val2.sum().backward()
                self.assertAllClose(X_grad, test_X2.grad, **all_close_kwargs)
            # test we fall back to standard sampling for
            # ill-conditioned covariances
            acqf._baseline_L = torch.zeros_like(acqf._baseline_L)
            with warnings.catch_warnings(record=True) as ws, torch.no_grad():
                acqf(test_X)
            self.assertEqual(sum(issubclass(w.category, BotorchWarning) for w in ws), 1)

        # test w/ posterior transform
        X_baseline = torch.rand(2, 1)
        model = SingleTaskGP(X_baseline, torch.randn(2, 1))
        pt = ScalarizedPosteriorTransform(weights=torch.tensor([-1]))
        with mock.patch.object(
            qLogNoisyExpectedImprovement,
            "_compute_root_decomposition",
        ) as mock_cache_root:
            acqf = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_baseline,
                sampler=IIDNormalSampler(sample_shape=torch.Size([1])),
                posterior_transform=pt,
                prune_baseline=False,
                cache_root=True,
            )
            tf_post = model.posterior(X_baseline, posterior_transform=pt)
            self.assertTrue(
                torch.allclose(
                    tf_post.mean, mock_cache_root.call_args[-1]["posterior"].mean
                )
            )

        # testing constraints
        n, d, m = 8, 1, 3
        X_baseline = torch.rand(n, d)
        model = SingleTaskGP(X_baseline, torch.randn(n, m))  # batched model
        nei_args = {
            "model": model,
            "X_baseline": X_baseline,
            "prune_baseline": False,
            "cache_root": True,
            "posterior_transform": ScalarizedPosteriorTransform(weights=torch.ones(m)),
            "sampler": SobolQMCNormalSampler(torch.Size([5])),
        }
        acqf = qLogNoisyExpectedImprovement(**nei_args)
        X = torch.randn_like(X_baseline)
        for con in [feasible_con, infeasible_con]:
            with self.subTest(con=con):
                target = "botorch.acquisition.utils.get_infeasible_cost"
                infcost = torch.tensor([3], device=self.device, dtype=dtype)
                with mock.patch(target, return_value=infcost):
                    cacqf = qLogNoisyExpectedImprovement(**nei_args, constraints=[con])

                _, obj = cacqf._get_samples_and_objectives(X)
                best_feas_f = cacqf.compute_best_f(obj)
                if con is feasible_con:
                    self.assertAllClose(best_feas_f, acqf.compute_best_f(obj))
                else:
                    self.assertAllClose(
                        best_feas_f, torch.full_like(obj[..., 0], -infcost.item())
                    )
        # TODO: Test different objectives (incl. constraints)


class TestQLogProbabilityOfFeasibility(BotorchTestCase):
    def test_q_log_probability_of_feasibility(self):
        self.assertIn(
            qLogProbabilityOfFeasibility, ACQF_INPUT_CONSTRUCTOR_REGISTRY.keys()
        )
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                self._test_q_log_probability_of_feasibility(dtype=dtype)

    def _test_q_log_probability_of_feasibility(self, dtype=torch.double):
        tkwargs = {"device": self.device, "dtype": dtype}
        sample_shape = torch.Size([2])
        samples = torch.zeros(1, 1, **tkwargs)
        mm = MockModel(MockPosterior(samples=samples))
        # X is ``q x d`` = 1 x 1. X is a dummy and unused b/c of mocking
        X = torch.zeros(1, 1, **tkwargs)

        # basic test
        torch.manual_seed(1234)
        sampler = IIDNormalSampler(sample_shape=sample_shape)

        def satisfied_constraint(Y: Tensor) -> Tensor:
            return torch.full_like(Y[..., 0], 10.0)

        def barely_satisfied_constraint(Y: Tensor) -> Tensor:
            return torch.full_like(Y[..., 0], 3 * eta)

        def uncertain_constraint(Y: Tensor) -> Tensor:
            return torch.full_like(Y[..., 0], 0.0)

        def barely_unsatisfied_constraint(Y: Tensor) -> Tensor:
            return torch.full_like(Y[..., 0], -torch.pi * eta)

        acqf = qLogProbabilityOfFeasibility(
            model=mm,
            constraints=[satisfied_constraint],
            sampler=sampler,
        )
        # test initialization
        self.assertTrue(acqf._fat)  # default behavior
        self.assertIn("sampler", acqf._modules)
        self.assertIn("objective", acqf._modules)
        # objective is always set to IdentityMCObjective
        self.assertIsInstance(acqf.objective, IdentityMCObjective)
        self.assertTrue(acqf._log)

        constraints_list = [
            [barely_satisfied_constraint, uncertain_constraint],
            [barely_unsatisfied_constraint, barely_satisfied_constraint],
            [
                barely_unsatisfied_constraint,
                uncertain_constraint,
                barely_satisfied_constraint,
            ],
            [uncertain_constraint],
            [satisfied_constraint],
        ]

        eta = torch.pi * 1e-3  # not the default
        fat = True
        for constraints in constraints_list:
            acqf = qLogProbabilityOfFeasibility(
                model=mm,
                constraints=constraints,
                sampler=sampler,
                eta=eta,
            )
            res = acqf(X)
            log_sample_feasibility = compute_smoothed_feasibility_indicator(
                constraints=constraints,
                samples=samples.expand(*sample_shape, *samples.shape),
                eta=eta,
                log=True,
                fat=fat,
            )
            log_feasibility = acqf._sample_reduction(
                acqf._q_reduction(log_sample_feasibility)
            )
            self.assertAllClose(res, log_feasibility)


class TestIsLog(BotorchTestCase):
    def test_is_log(self):
        # the flag is False by default
        self.assertFalse(AcquisitionFunction._log)

        # single objective case
        X, Y = torch.rand(3, 2), torch.randn(3, 1)
        model = SingleTaskGP(train_X=X, train_Y=Y, train_Yvar=torch.rand_like(Y))

        # (q)LogEI
        for acqf_class in [LogExpectedImprovement, qLogExpectedImprovement]:
            acqf = acqf_class(model=model, best_f=0.0)
            self.assertTrue(acqf._log)

        # (q)EI
        for acqf_class in [ExpectedImprovement, qExpectedImprovement]:
            acqf = acqf_class(model=model, best_f=0.0)
            self.assertFalse(acqf._log)

        # (q)LogNEI
        for acqf_class in [LogNoisyExpectedImprovement, qLogNoisyExpectedImprovement]:
            # avoiding keywords since they differ: X_observed vs. X_baseline
            acqf = acqf_class(model, X)
            self.assertTrue(acqf._log)

        # (q)NEI
        for acqf_class in [NoisyExpectedImprovement, qNoisyExpectedImprovement]:
            acqf = acqf_class(model, X)
            self.assertFalse(acqf._log)

        # probability of feasibility
        self.assertTrue(qLogProbabilityOfFeasibility._log)

        # multi-objective case
        model_list = ModelListGP(model, model)
        ref_point = [4, 2]  # the meaning of life

        # qLogNEHVI
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model_list, X_baseline=X, ref_point=ref_point
        )
        self.assertTrue(acqf._log)
