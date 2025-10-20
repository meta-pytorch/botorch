#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from collections.abc import Callable, Iterable
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from itertools import filterfalse, product
from unittest.mock import MagicMock, patch
from warnings import catch_warnings, warn, WarningMessage

import torch
from botorch import fit
from botorch.exceptions.errors import ModelFittingError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import OptimizationResult, OptimizationStatus
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.optim.utils import get_data_loader
from botorch.utils.context_managers import module_rollback_ctx, TensorCheckpoint
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from linear_operator.utils.errors import NotPSDError

MAX_ITER_MSG_REGEX = re.compile(
    # Note that the message changed with scipy 1.15, hence the different matching here.
    "TOTAL NO. (of|OF) ITERATIONS REACHED LIMIT"
)


class MockOptimizer:
    def __init__(
        self,
        randomize_requires_grad: bool = True,
        warnings: Iterable[WarningMessage] = (),
        exception: BaseException | None = None,
    ):
        r"""Class used to mock `optimizer` argument to `fit_gpytorch_mll."""
        self.randomize_requires_grad = randomize_requires_grad
        self.warnings = warnings
        self.exception = exception
        self.call_count = 0
        self.state_dicts = []

    def __call__(self, mll, closure: Callable | None = None) -> OptimizationResult:
        self.call_count += 1
        for w in self.warnings:
            warn(str(w.message), w.category)

        if self.randomize_requires_grad:
            with torch.no_grad():
                for param in mll.parameters():
                    if param.requires_grad:
                        param[...] = torch.rand_like(param)

        if self.exception is not None:
            raise self.exception

        self.state_dicts.append(deepcopy(mll.state_dict()))
        return OptimizationResult(
            fval=torch.rand(1).item(),
            step=1,
            status=OptimizationStatus.SUCCESS,
            message="Mock Success!",
            runtime=1.0,
        )


class TestFitAPI(BotorchTestCase):
    r"""Unit tests for general fitting API"""

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)
            train_Y = train_F + 0.1 * torch.randn_like(train_F)

            model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
            )
            self.mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def test_fit_gpytorch_mll(self):
        # Test that `optimizer` is only passed when non-None
        with patch.object(fit, "FitGPyTorchMLL") as mock_dispatcher:
            fit_gpytorch_mll(self.mll, optimizer=None)
            mock_dispatcher.assert_called_once_with(
                self.mll,
                type(self.mll.likelihood),
                type(self.mll.model),
                closure=None,
                closure_kwargs=None,
                optimizer_kwargs=None,
            )

            fit_gpytorch_mll(self.mll, optimizer="foo")
            mock_dispatcher.assert_called_with(
                self.mll,
                type(self.mll.likelihood),
                type(self.mll.model),
                closure=None,
                closure_kwargs=None,
                optimizer="foo",
                optimizer_kwargs=None,
            )


class TestFitFallback(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)

            self.mlls = {}
            self.checkpoints = {}
            for fixed_noise, output_dim in product([True, False], [1, 2]):
                train_Y = train_F.repeat(1, output_dim)
                train_Y = train_Y + 0.1 * torch.randn_like(train_Y)
                model = SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=torch.full_like(train_Y, 0.1) if fixed_noise else None,
                    input_transform=Normalize(d=1),
                )
                self.assertIsInstance(model.covar_module, RBFKernel)

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                for dtype in (torch.float32, torch.float64):
                    key = fixed_noise, output_dim
                    self.mlls[key] = mll.to(dtype=dtype)
                    self.checkpoints[key] = {
                        k: TensorCheckpoint(
                            values=v.detach().clone(), device=v.device, dtype=v.dtype
                        )
                        for k, v in mll.state_dict().items()
                    }

    def test_main(self):
        for case, mll in self.mlls.items():
            self._test_main(mll, self.checkpoints[case])

    def test_warnings(self):
        for case, mll in self.mlls.items():
            self._test_warnings(mll, self.checkpoints[case])

    def test_exceptions(self):
        for case, mll in self.mlls.items():
            self._test_exceptions(mll, self.checkpoints[case])

    def _test_main(self, mll, ckpt):
        r"""Main test for `_fit_fallback`."""
        optimizer = MockOptimizer()
        optimizer.warnings = [
            WarningMessage("test_runtime_warning", RuntimeWarning, __file__, 0),
        ]
        for should_fail in (True, False):
            optimizer.call_count = 0
            with catch_warnings(), module_rollback_ctx(mll, checkpoint=ckpt):
                try:
                    fit._fit_fallback(
                        mll,
                        None,
                        None,
                        max_attempts=2,
                        optimizer=optimizer,
                        warning_handler=lambda w: not should_fail,
                    )
                except ModelFittingError:
                    failed = True
                else:
                    failed = False

                # Test control flow
                self.assertEqual(failed, should_fail)
                self.assertEqual(optimizer.call_count, 2 if should_fail else 1)

                # Test terminal state
                self.assertEqual(failed, mll.training)
                for key, vals in mll.state_dict().items():
                    if failed:
                        self.assertTrue(vals.equal(ckpt[key].values))
                    else:
                        try:
                            param = mll.get_parameter(key)
                            self.assertNotEqual(
                                param.equal(ckpt[key].values), param.requires_grad
                            )
                        except AttributeError:
                            pass

        # Test `closure_kwargs`
        with self.subTest("closure_kwargs"):
            mock_closure = MagicMock(side_effect=StopIteration("foo"))
            with self.assertRaisesRegex(StopIteration, "foo"):
                fit._fit_fallback(
                    mll, None, None, closure=mock_closure, closure_kwargs={"ab": "cd"}
                )
            mock_closure.assert_called_once_with(ab="cd")

    def _test_warnings(self, mll, ckpt):
        r"""Test warning handling for `_fit_fallback`."""
        optimizer = MockOptimizer(randomize_requires_grad=False)
        optimizer.warnings = [
            WarningMessage("test_runtime_warning", RuntimeWarning, __file__, 0),
            WarningMessage(
                "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
                OptimizationWarning,
                __file__,
                0,
            ),
            WarningMessage(
                "Optimization timed out after X", OptimizationWarning, __file__, 0
            ),
        ]

        warning_handlers = {
            "default": fit.DEFAULT_WARNING_HANDLER,
            "none": lambda w: False,
            "all": lambda w: True,
        }
        for case, warning_handler in warning_handlers.items():
            with ExitStack() as es:
                logs = es.enter_context(
                    self.assertLogs(level="DEBUG")
                    if case == "default"
                    else nullcontext()
                )
                ws = es.enter_context(catch_warnings(record=True))

                try:
                    fit._fit_fallback(
                        mll,
                        None,
                        None,
                        max_attempts=2,
                        optimizer=optimizer,
                        warning_handler=warning_handler,
                    )
                except ModelFittingError:
                    failed = True
                else:
                    failed = False

                # Test that warnings were resolved in the expected fashion
                self.assertEqual(failed, case == "none")
                with catch_warnings(record=True) as rethrown:
                    unresolved = list(filterfalse(warning_handler, optimizer.warnings))
                    self.assertEqual(failed, len(unresolved) > 0)

                self.assertEqual(
                    {str(w.message) for w in ws},
                    {str(w.message) for w in rethrown + unresolved},
                )
                if logs:  # test that default filter logs certain warnings
                    self.assertTrue(
                        any(MAX_ITER_MSG_REGEX.search(log) for log in logs.output)
                    )

        # Test default of retrying upon encountering an uncaught OptimizationWarning
        optimizer.warnings.append(
            WarningMessage("test_optim_warning", OptimizationWarning, __file__, 0)
        )

        with self.assertRaises(ModelFittingError), catch_warnings():
            fit._fit_fallback(
                mll,
                None,
                None,
                max_attempts=1,
                optimizer=optimizer,
            )

    def _test_exceptions(self, mll, ckpt):
        r"""Test exception handling for `_fit_fallback`."""
        optimizer = MockOptimizer(exception=NotPSDError("not_psd"))
        with catch_warnings():
            # Test behavior when encountering a caught exception
            with (
                self.assertLogs(logger="botorch", level="DEBUG") as logs,
                self.assertRaises(ModelFittingError),
            ):
                fit._fit_fallback(
                    mll,
                    None,
                    None,
                    max_attempts=1,
                    optimizer=optimizer,
                )

            self.assertTrue(any("not_psd" in log for log in logs.output))
            self.assertTrue(  # test state rollback
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )

            # Test behavior when encountering an uncaught exception
            with self.assertRaisesRegex(NotPSDError, "not_psd"):
                fit._fit_fallback(
                    mll,
                    None,
                    None,
                    max_attempts=1,
                    optimizer=optimizer,
                    caught_exception_types=(),
                )

            self.assertTrue(  # test state rollback
                all(v.equal(ckpt[k].values) for k, v in mll.state_dict().items())
            )

    def test_pick_best_of_all_attempts(self) -> None:
        mll = next(iter(self.mlls.values()))
        optimizer = MockOptimizer()
        max_attempts = 10
        with patch("botorch.fit.logger.debug") as mock_log:
            fit._fit_fallback(
                mll,
                None,
                None,
                max_attempts=max_attempts,
                pick_best_of_all_attempts=True,
                optimizer=optimizer,
            )
        # Check that optimizer is called 3 times.
        self.assertEqual(optimizer.call_count, max_attempts)
        # Check that we log after each call.
        self.assertEqual(mock_log.call_count, max_attempts)
        # We have an increasing sequence of best MLL values.
        mll_vals = []
        for call in mock_log.call_args_list:
            message = call.args[0]
            mll_val = message.split(" ")[-1][:-1]
            mll_vals.append(float(mll_val))
        self.assertEqual(mll_vals, sorted(mll_vals))
        # Check that the returned MLL is in eval mode.
        self.assertFalse(mll.training)
        # Check that the state dict matches the state dict of best attempt.
        final_statedict = mll.state_dict()
        best_idx = mll_vals.index(max(mll_vals))
        best_state_dict = optimizer.state_dicts[best_idx]
        for key, val in final_statedict.items():
            self.assertAllClose(val, best_state_dict[key])


class TestFitFallbackApproximate(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            train_X = torch.linspace(0, 1, 10).unsqueeze(-1)
            train_F = torch.sin(2 * math.pi * train_X)
            train_Y = train_F + 0.1 * torch.randn_like(train_F)

            model = SingleTaskVariationalGP(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=Normalize(d=1),
                outcome_transform=Standardize(m=1),
            )
            self.mll = mll = VariationalELBO(model.likelihood, model.model, num_data=10)
            self.data_loader = get_data_loader(mll.model, batch_size=1)
            self.closure = get_loss_closure_with_grads(
                mll=mll,
                parameters={n: p for n, p in mll.named_parameters() if p.requires_grad},
                data_loader=self.data_loader,
            )

    def test_main(self):
        # Test parameter updates
        with module_rollback_ctx(self.mll) as ckpt:
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                closure=self.closure,
                optimizer_kwargs={"step_limit": 3},
            )
            for name, param in self.mll.named_parameters():
                self.assertFalse(param.equal(ckpt[name].values))

        # Test dispatching pattern
        kwargs = {"full_batch_limit": float("inf")}
        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(self.mll, None, None, full_batch_limit=1)
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=None,
                optimizer=fit_gpytorch_mll_torch,
            )

        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(self.mll, None, None, **kwargs)
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=None,
                optimizer=fit_gpytorch_mll_scipy,
            )

        with patch.object(fit, "_fit_fallback") as mock_fallback:
            fit._fit_fallback_approximate(
                self.mll, None, None, closure=self.closure, **kwargs
            )

            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure=self.closure,
                optimizer=fit_gpytorch_mll_torch,
            )

        with (
            patch.object(fit, "_fit_fallback") as mock_fallback,
            patch.object(fit, "get_loss_closure_with_grads") as mock_get_closure,
        ):
            mock_get_closure.return_value = "foo"
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                data_loader=self.data_loader,
                **kwargs,
            )
            params = {n: p for n, p in self.mll.named_parameters() if p.requires_grad}
            mock_get_closure.assert_called_once_with(
                mll=self.mll,
                data_loader=self.data_loader,
                parameters=params,
            )
            mock_fallback.assert_called_once_with(
                self.mll,
                None,
                None,
                closure="foo",
                optimizer=fit_gpytorch_mll_torch,
            )

        # Test exception handling
        with self.assertRaisesRegex(
            UnsupportedError, "Only one of `data_loader` or `closure` may be passed."
        ):
            fit._fit_fallback_approximate(
                self.mll,
                None,
                None,
                closure=self.closure,
                data_loader=self.data_loader,
            )
