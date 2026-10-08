#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from copy import deepcopy
from unittest.mock import MagicMock, patch
from warnings import catch_warnings

import numpy as np
import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize, Warp
from botorch.optim import core, fit
from botorch.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched
from botorch.optim.closures import get_loss_closure, get_loss_closure_with_grads
from botorch.optim.core import OptimizationResult, OptimizationStatus
from botorch.optim.utils import get_parameters
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
    """Tests for the batched independent fitting path of fit_gpytorch_mll_scipy."""

    def test_dispatches_to_independent_for_multi_output(self):
        """Verify that fit_gpytorch_mll_scipy dispatches to
        _fit_gpytorch_mll_scipy_independent when using a batched
        multi-output model (aug_batch_shape.numel() > 1)."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 3, dtype=torch.double)  # 3 outputs
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        with patch(
            "botorch.optim.fit._fit_gpytorch_mll_scipy_independent"
        ) as mock_independent:
            mock_independent.return_value = OptimizationResult(
                fval=0.0, step=1, status=OptimizationStatus.SUCCESS
            )
            fit.fit_gpytorch_mll_scipy(mll)
            mock_independent.assert_called_once()

    def test_dispatches_to_independent_for_ensemble_model(self):
        """Verify that fit_gpytorch_mll_scipy dispatches to
        _fit_gpytorch_mll_scipy_independent when using an ensemble
        model (aug_batch_shape.numel() > 1)."""
        from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP

        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 3, dtype=torch.double)
            train_Y = torch.rand(10, 1, dtype=torch.double)
        model = EnsembleMapSaasSingleTaskGP(train_X, train_Y, num_taus=3)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        with patch(
            "botorch.optim.fit._fit_gpytorch_mll_scipy_independent"
        ) as mock_independent:
            mock_independent.return_value = OptimizationResult(
                fval=0.0, step=1, status=OptimizationStatus.SUCCESS
            )
            fit.fit_gpytorch_mll_scipy(mll)
            mock_independent.assert_called_once()

    def test_does_not_dispatch_to_independent_for_single_output(self):
        """Verify that fit_gpytorch_mll_scipy does NOT dispatch to
        _fit_gpytorch_mll_scipy_independent for single-output models."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 1, dtype=torch.double)  # single output
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        with patch(
            "botorch.optim.fit._fit_gpytorch_mll_scipy_independent"
        ) as mock_independent:
            # The standard path should be used, not the independent path
            fit.fit_gpytorch_mll_scipy(mll, options={"maxiter": 2})
            mock_independent.assert_not_called()

    def test_does_not_dispatch_to_independent_when_closure_provided(self):
        """Verify that fit_gpytorch_mll_scipy does NOT dispatch to
        _fit_gpytorch_mll_scipy_independent when a custom closure is provided,
        even for multi-output models."""
        from botorch.optim.closures import get_loss_closure_with_grads
        from botorch.optim.utils import get_parameters

        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 3, dtype=torch.double)  # multi-output
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        closure = get_loss_closure_with_grads(
            mll, parameters=get_parameters(mll, requires_grad=True)
        )

        with patch(
            "botorch.optim.fit._fit_gpytorch_mll_scipy_independent"
        ) as mock_independent:
            fit.fit_gpytorch_mll_scipy(mll, closure=closure, options={"maxiter": 2})
            mock_independent.assert_not_called()

    def test_multi_output(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 3, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        result = fit.fit_gpytorch_mll_scipy(mll)
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
        result = fit.fit_gpytorch_mll_scipy(mll, options={"maxiter": 2})
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

        fit.fit_gpytorch_mll_scipy(
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
        result = fit.fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})
        self.assertIsInstance(result, OptimizationResult)
        # Verify the model can produce predictions.
        model.eval()
        test_X = torch.rand(4, 3, dtype=torch.double)
        posterior = model.posterior(test_X)
        self.assertEqual(posterior.mean.shape, torch.Size([3, 4, 1]))

    def test_shared_parameters_use_joint_fitting(self):
        # The parameters of a learnable input transform are shared across outputs,
        # so the outputs' hyperparameters cannot be optimized independently. This
        # must also hold if the parameter shape happens to match the batch shape.
        for d in (2, 3):
            with torch.random.fork_rng():
                torch.manual_seed(0)
                train_X = torch.rand(15, d, dtype=torch.double)
                train_Y = torch.rand(15, 2, dtype=torch.double)
            model = SingleTaskGP(
                train_X, train_Y, input_transform=Warp(d=d, indices=list(range(d)))
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.train()
            parameters = get_parameters(mll, requires_grad=True)
            self.assertFalse(fit._is_batch_independent(model, parameters))
            self.assertTrue(
                fit._is_batch_independent(
                    model,
                    {
                        n: p
                        for n, p in parameters.items()
                        if not n.startswith("model.input_transform")
                    },
                )
            )
            # Fitting gives the same result as explicitly using the joint closure.
            state_dict = deepcopy(mll.state_dict())
            with patch.object(
                fit,
                "_fit_gpytorch_mll_scipy_independent",
                wraps=fit._fit_gpytorch_mll_scipy_independent,
            ) as mock_independent:
                result = fit.fit_gpytorch_mll_scipy(mll, options={"maxiter": 10})
            mock_independent.assert_not_called()
            loss = get_loss_closure(mll)().sum().item()
            mll.load_state_dict(state_dict)
            result_joint = fit.fit_gpytorch_mll_scipy(
                mll,
                closure=get_loss_closure_with_grads(mll, parameters=parameters),
                options={"maxiter": 10},
            )
            loss_joint = get_loss_closure(mll)().sum().item()
            self.assertEqual(result.fval, result_joint.fval)
            self.assertEqual(loss, loss_joint)

    def test_unsupported_arguments_use_joint_fitting(self):
        # Arguments that are not supported by ``fmin_l_bfgs_b_batched`` must not be
        # ignored, so the joint optimizer is used for them.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        for kwargs in (
            {"timeout_sec": 60.0},
            {"options": {"maxiter": 1, "disp": False}},
            {"options": {"maxfun": 5}},
            {"method": "SLSQP"},
            {"callback": lambda parameters, result: None},
            {"closure_kwargs": {}},
        ):
            with (
                patch.object(
                    fit, "_fit_gpytorch_mll_scipy_independent"
                ) as mock_independent,
                patch.object(
                    fit,
                    "scipy_minimize",
                    return_value=OptimizationResult(
                        fval=0.0, step=1, status=OptimizationStatus.SUCCESS
                    ),
                ) as mock_joint,
            ):
                fit.fit_gpytorch_mll_scipy(mll, **kwargs)
            mock_independent.assert_not_called()
            mock_joint.assert_called_once()
            for key in ("timeout_sec", "options", "method", "callback"):
                if key in kwargs:
                    self.assertEqual(mock_joint.call_args.kwargs[key], kwargs[key])

    def test_timeout_sec(self):
        # ``timeout_sec`` is respected by using the joint optimizer, rather than
        # being ignored with an ``OptimizationWarning`` (which ``fit_gpytorch_mll``
        # would consider a failed fitting attempt).
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        with catch_warnings(record=True) as ws:
            result = fit.fit_gpytorch_mll_scipy(
                mll,
                timeout_sec=60.0,
                options={"maxiter": 1},
            )
        self.assertFalse(any(issubclass(w.category, OptimizationWarning) for w in ws))
        self.assertEqual(result.status, OptimizationStatus.STOPPED)

    def test_callback(self):
        # The callback is called as ``callback(parameters, result)``, as documented.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        calls = []

        def callback(parameters, result):
            calls.append((parameters, result))

        fit.fit_gpytorch_mll_scipy(mll, callback=callback, options={"maxiter": 2})
        self.assertGreater(len(calls), 0)
        for parameters, result in calls:
            self.assertEqual(
                set(parameters), set(get_parameters(mll, requires_grad=True))
            )
            self.assertIsInstance(result, OptimizationResult)

    def test_ftol_option(self):
        # ``fmin_l_bfgs_b_batched`` does not accept ``ftol`` together with ``factr``.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        with patch(
            "botorch.optim.batched_lbfgs_b.fmin_l_bfgs_b_batched",
            wraps=fmin_l_bfgs_b_batched,
        ) as mock_fmin:
            result = fit.fit_gpytorch_mll_scipy(
                mll, options={"ftol": 1e-6, "gtol": 1e-3, "maxiter": 5}
            )
        mock_fmin.assert_called_once()
        call_kwargs = mock_fmin.call_args.kwargs
        self.assertEqual(call_kwargs["ftol"], 1e-6)
        self.assertIsNone(call_kwargs["factr"])
        self.assertEqual(call_kwargs["pgtol"], 1e-3)
        self.assertEqual(call_kwargs["maxiter"], 5)
        self.assertIn(
            result.status, (OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED)
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
        result = fit.fit_gpytorch_mll_scipy(
            mll,
            options={"gtol": 1e-3, "maxiter": 2},
        )
        self.assertIsInstance(result, OptimizationResult)

    def _test_status(
        self, statuses: list[int], expected_status: OptimizationStatus
    ) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        def mock_fmin(func, x0, bounds, **kwargs):
            # ``fmin_l_bfgs_b_batched`` returns ``OptimizeResult``s with a ``status``
            # of 0 (converged), 1 (iteration limit reached) or 2 (other).
            return (
                x0,
                np.zeros(x0.shape[0]),
                [
                    OptimizeResult(success=status == 0, nit=1, status=status)
                    for status in statuses
                ],
            )

        with (
            patch(
                "botorch.optim.batched_lbfgs_b.fmin_l_bfgs_b_batched",
                side_effect=mock_fmin,
            ),
            catch_warnings(record=True) as ws,
        ):
            result = fit.fit_gpytorch_mll_scipy(mll)

        self.assertEqual(result.status, expected_status)
        # As for joint fitting, failures result in an ``OptimizationWarning``, which
        # makes ``fit_gpytorch_mll`` retry the fit.
        self.assertEqual(
            any(issubclass(w.category, OptimizationWarning) for w in ws),
            expected_status == OptimizationStatus.FAILURE,
        )

    def test_success_status(self):
        self._test_status([0, 0], OptimizationStatus.SUCCESS)

    def test_stopped_status(self):
        """Test STOPPED status when some outputs hit maxiter."""
        self._test_status([1, 1], OptimizationStatus.STOPPED)
        self._test_status([0, 1], OptimizationStatus.STOPPED)
        # Reaching ``maxiter`` without mocks.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.rand(10, 2, dtype=torch.double)
            train_Y = torch.rand(10, 2, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        with catch_warnings(record=True) as ws:
            result = fit.fit_gpytorch_mll_scipy(mll, options={"maxiter": 1})
        self.assertEqual(result.status, OptimizationStatus.STOPPED)
        self.assertFalse(any(issubclass(w.category, OptimizationWarning) for w in ws))

    def test_failure_status(self):
        """Test FAILURE status when any output fails without hitting maxiter."""
        self._test_status([2, 2], OptimizationStatus.FAILURE)
        self._test_status([0, 2], OptimizationStatus.FAILURE)
        self._test_status([1, 2], OptimizationStatus.FAILURE)
