#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from unittest.mock import MagicMock, patch
from warnings import catch_warnings

import numpy as np
import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import core, fit
from botorch.optim.core import OptimizationResult, OptimizationStatus
from botorch.utils.context_managers import module_rollback_ctx, TensorCheckpoint
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.optimize import OptimizeResult

MAX_ITER_MSG_REGEX = re.compile(
    # Note that the message changed with scipy 1.15, hence the different matching here.
    "TOTAL NO. (of|OF) ITERATIONS REACHED LIMIT"
)


class TestFitGPyTorchMLLScipy(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.mlls = {}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 30).unsqueeze(-1)
            train_Y = torch.sin((6 * math.pi) * train_X)
            train_Y = train_Y + 0.01 * torch.randn_like(train_Y)

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=1),
        )
        self.mlls[SingleTaskGP, 1] = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gpytorch_mll_scipy(self):
        for mll in self.mlls.values():
            for dtype in (torch.float32, torch.float64):
                self._test_fit_gpytorch_mll_scipy(mll.to(dtype=dtype))

    def _test_fit_gpytorch_mll_scipy(self, mll):
        options = {"disp": False, "maxiter": 2}
        ckpt = {
            k: TensorCheckpoint(v.detach().clone(), v.device, v.dtype)
            for k, v in mll.state_dict().items()
        }
        with self.subTest("main"), module_rollback_ctx(mll, checkpoint=ckpt):
            with catch_warnings(record=True) as ws:
                result = fit.fit_gpytorch_mll_scipy(mll, options=options)

            # Test only parameters requiring gradients have changed
            self.assertTrue(
                all(
                    param.equal(ckpt[name].values) != param.requires_grad
                    for name, param in mll.named_parameters()
                )
            )

            # Test stopping due to maxiter without optimization warning.
            self.assertEqual(result.status, OptimizationStatus.STOPPED)
            self.assertTrue(MAX_ITER_MSG_REGEX.search(result.message))
            self.assertFalse(
                any(issubclass(w.category, OptimizationWarning) for w in ws)
            )

            # Test iteration tracking
            self.assertIsInstance(result, OptimizationResult)
            self.assertLessEqual(result.step, options["maxiter"])

        # Test that user provided bounds are respected
        with self.subTest("bounds"), module_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_mll_scipy(
                mll,
                bounds={"likelihood.noise_covar.raw_noise": (123, 456)},
                options=options,
            )

            self.assertTrue(
                mll.likelihood.noise_covar.raw_noise >= 123
                and mll.likelihood.noise_covar.raw_noise <= 456
            )

            for name, param in mll.named_parameters():
                self.assertNotEqual(param.requires_grad, param.equal(ckpt[name].values))

        # Test handling of scipy optimization failures and parameter assignments
        mock_x = []
        assignments = {}
        for name, param in mll.named_parameters():
            if not param.requires_grad:
                continue  # pragma: no cover

            values = assignments[name] = torch.rand_like(param)
            mock_x.append(values.view(-1))

        with (
            module_rollback_ctx(mll, checkpoint=ckpt),
            patch.object(core, "minimize_with_timeout") as mock_minimize_with_timeout,
        ):
            mock_minimize_with_timeout.return_value = OptimizeResult(
                x=torch.concat(mock_x).tolist(),
                success=False,
                status=0,
                fun=float("nan"),
                jac=None,
                nfev=1,
                njev=1,
                nhev=1,
                nit=1,
                message=b"ABNORMAL_TERMINATION_IN_LNSRCH",
            )
            with catch_warnings(record=True) as ws:
                fit.fit_gpytorch_mll_scipy(mll, options=options)

            # Test that warning gets raised
            self.assertTrue(
                any("ABNORMAL_TERMINATION_IN_LNSRCH" in str(w.message) for w in ws)
            )

            # Test that parameter values get assigned correctly
            self.assertTrue(
                all(
                    param.equal(assignments[name])
                    for name, param in mll.named_parameters()
                    if param.requires_grad
                )
            )

        # Test ``closure_kwargs``
        with self.subTest("closure_kwargs"):
            mock_closure = MagicMock(side_effect=StopIteration("foo"))
            with self.assertRaisesRegex(StopIteration, "foo"):
                fit.fit_gpytorch_mll_scipy(
                    mll, closure=mock_closure, closure_kwargs={"ab": "cd"}
                )
            mock_closure.assert_called_once_with(ab="cd")

    def test_fit_with_nans(self) -> None:
        """Test the branch of NdarrayOptimizationClosure that handles errors."""

        from botorch.optim.closures import NdarrayOptimizationClosure

        def closure():
            raise RuntimeError("singular")

        for dtype in [torch.float32, torch.float64]:
            parameters = {"x": torch.tensor([0.0], dtype=dtype)}

            wrapper = NdarrayOptimizationClosure(closure=closure, parameters=parameters)

            def _assert_np_array_is_float64_type(array) -> bool:
                # e.g. "float32" in "torch.float32"
                self.assertEqual(str(array.dtype), "float64")

            _assert_np_array_is_float64_type(wrapper()[0])
            _assert_np_array_is_float64_type(wrapper()[1])
            _assert_np_array_is_float64_type(wrapper.state)
            _assert_np_array_is_float64_type(wrapper._get_gradient_ndarray())

            # Any mll will do
            mll = next(iter(self.mlls.values()))
            # will error if dtypes are wrong
            fit.fit_gpytorch_mll_scipy(mll, closure=wrapper, parameters=parameters)


class TestFitGPyTorchMLLTorch(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mlls = {}
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_Y = torch.sin((2 * math.pi) * train_X)
            train_Y = train_Y + 0.1 * torch.randn_like(train_Y)

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=1),
        )
        self.mlls[SingleTaskGP, 1] = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gpytorch_mll_torch(self):
        for mll in self.mlls.values():
            for dtype in (torch.float32, torch.float64):
                self._test_fit_gpytorch_mll_torch(mll.to(dtype=dtype))

    def _test_fit_gpytorch_mll_torch(self, mll):
        ckpt = {
            k: TensorCheckpoint(v.detach().clone(), v.device, v.dtype)
            for k, v in mll.state_dict().items()
        }
        with self.subTest("main"), module_rollback_ctx(mll, checkpoint=ckpt):
            with catch_warnings(record=True):
                result = fit.fit_gpytorch_mll_torch(mll, step_limit=2)

            self.assertIsInstance(result, OptimizationResult)
            self.assertLessEqual(result.step, 2)

            # Test only parameters requiring gradients have changed
            self.assertTrue(
                all(
                    param.requires_grad != param.equal(ckpt[name].values)
                    for name, param in mll.named_parameters()
                )
            )

        # Test that user provided bounds are respected
        with self.subTest("bounds"), module_rollback_ctx(mll, checkpoint=ckpt):
            fit.fit_gpytorch_mll_torch(
                mll,
                bounds={"likelihood.noise_covar.raw_noise": (123, 456)},
            )

            self.assertTrue(
                mll.likelihood.noise_covar.raw_noise >= 123
                and mll.likelihood.noise_covar.raw_noise <= 456
            )

        # Test ``closure_kwargs``
        with self.subTest("closure_kwargs"):
            mock_closure = MagicMock(side_effect=StopIteration("foo"))
            with self.assertRaisesRegex(StopIteration, "foo"):
                fit.fit_gpytorch_mll_torch(
                    mll, closure=mock_closure, closure_kwargs={"ab": "cd"}
                )
            mock_closure.assert_called_once_with(ab="cd")


class TestFitGPyTorchMLLScipyIndependent(BotorchTestCase):
    def test_multi_output(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 3, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        result = fit.fit_gpytorch_mll_scipy_independent(mll)
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.status, OptimizationStatus.SUCCESS)
        # Verify model can produce predictions after fitting.
        model.eval()
        test_X = torch.rand(3, 2, dtype=torch.double)
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([3, 3]))

    def test_single_output_fallback(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 1, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        result = fit.fit_gpytorch_mll_scipy_independent(mll, options={"maxiter": 2})
        self.assertIsInstance(result, OptimizationResult)

    def test_with_fit_gpytorch_mll(self):
        from botorch.fit import fit_gpytorch_mll

        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with catch_warnings(record=True):
            fit_gpytorch_mll(
                mll,
                optimizer=fit.fit_gpytorch_mll_scipy_independent,
                optimizer_kwargs={"options": {"maxiter": 2}},
                max_attempts=1,
            )
        self.assertFalse(model.training)

    def test_callback_passed_through(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        callback_called = []

        def my_callback(*args, **kwargs):
            callback_called.append(True)

        fit.fit_gpytorch_mll_scipy_independent(
            mll,
            callback=my_callback,
            options={"maxiter": 1},
        )
        self.assertTrue(len(callback_called) > 0)

    def test_ensemble_model(self):
        from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP

        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 3, dtype=torch.double)
            train_Y = torch.rand(10, 1, dtype=torch.double)
        model = EnsembleMapSaasSingleTaskGP(train_X, train_Y, num_taus=3)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        result = fit.fit_gpytorch_mll_scipy_independent(mll, options={"maxiter": 3})
        self.assertIsInstance(result, OptimizationResult)
        # Verify the model can produce predictions.
        model.eval()
        test_X = torch.rand(4, 3, dtype=torch.double)
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([3, 4, 1]))

    def test_timeout_sec_warning(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        with self.assertWarns(OptimizationWarning):
            fit.fit_gpytorch_mll_scipy_independent(
                mll,
                timeout_sec=60.0,
                options={"maxiter": 1},
            )

    def test_gtol_option_mapping(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        # gtol should be mapped to pgtol for fmin_l_bfgs_b_batched
        result = fit.fit_gpytorch_mll_scipy_independent(
            mll,
            options={"gtol": 1e-3, "maxiter": 2},
        )
        self.assertIsInstance(result, OptimizationResult)

    def test_unrecognized_options_warning(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        with self.assertWarns(OptimizationWarning):
            fit.fit_gpytorch_mll_scipy_independent(
                mll,
                options={"maxiter": 1, "disp": True, "unknown_opt": 42},
            )

    def test_stopped_status(self):
        """Test STOPPED status when some outputs hit maxiter (warnflag=1)."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        def mock_fmin(func, x0, bounds, **kwargs):
            return (
                x0,
                np.zeros(x0.shape[0]),
                [
                    {"success": False, "nit": 1, "warnflag": 1}
                    for _ in range(x0.shape[0])
                ],
            )

        with patch(
            "botorch.optim.batched_lbfgs_b.fmin_l_bfgs_b_batched",
            side_effect=mock_fmin,
        ):
            result = fit.fit_gpytorch_mll_scipy_independent(mll)

        self.assertEqual(result.status, OptimizationStatus.STOPPED)

    def test_failure_status(self):
        """Test FAILURE status when outputs fail without hitting maxiter."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        def mock_fmin(func, x0, bounds, **kwargs):
            return (
                x0,
                np.zeros(x0.shape[0]),
                [
                    {"success": False, "nit": 1, "warnflag": 2}
                    for _ in range(x0.shape[0])
                ],
            )

        with patch(
            "botorch.optim.batched_lbfgs_b.fmin_l_bfgs_b_batched",
            side_effect=mock_fmin,
        ):
            result = fit.fit_gpytorch_mll_scipy_independent(mll)

        self.assertEqual(result.status, OptimizationStatus.FAILURE)
