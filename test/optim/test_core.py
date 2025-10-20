#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from functools import partial
from unittest.mock import MagicMock, patch

import torch
from botorch.optim import core
from botorch.optim.closures import ForwardBackwardClosure, NdarrayOptimizationClosure
from botorch.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    scipy_minimize,
    torch_minimize,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.utils.testing import BotorchTestCase
from numpy import allclose
from scipy.optimize import OptimizeResult
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.sgd import SGD

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # pragma: no cover


class ToyModule(Module):
    def __init__(self, b: Parameter, x: Parameter, dummy: Parameter):
        r"""Toy module for unit testing."""
        super().__init__()
        self.x = x
        self.b = b
        self.dummy = dummy

    def forward(self) -> Tensor:
        return (self.x - self.b).square().sum()

    @property
    def free_parameters(self) -> dict[str, Tensor]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


def norm_squared(x, delay: float = 0.0):
    if x.grad is not None:
        x.grad.zero_()
    loss = x.square().sum()
    loss.backward()
    if delay:
        time.sleep(delay)
    return loss, [x.grad]


class TestScipyMinimize(BotorchTestCase):
    def setUp(self):
        super().setUp()
        module = ToyModule(
            x=Parameter(torch.tensor(0.5, device=self.device)),
            b=Parameter(torch.tensor(0.0, device=self.device), requires_grad=False),
            dummy=Parameter(torch.tensor(1.0, device=self.device)),
        ).to(self.device)

        self.closures = {}
        for dtype in ("float32", "float64"):
            m = module.to(dtype=getattr(torch, dtype))
            self.closures[dtype] = ForwardBackwardClosure(m, m.free_parameters)

    def test_basic(self):
        x = Parameter(torch.rand([]))
        closure = partial(norm_squared, x)
        result = scipy_minimize(closure, {"x": x})
        self.assertEqual(result.status, OptimizationStatus.SUCCESS)
        self.assertTrue(allclose(result.fval, 0.0))

    def test_timeout(self):
        x = Parameter(torch.tensor(1.0))
        # adding a small delay here to combat some timing issues on windows
        closure = partial(norm_squared, x, delay=1e-2)
        result = scipy_minimize(closure, {"x": x}, timeout_sec=1e-4)
        self.assertEqual(result.status, OptimizationStatus.STOPPED)
        self.assertTrue("Optimization timed out after" in result.message)

    def test_main(self):
        def _callback(parameters, result, out) -> None:
            out.append(result)

        for closure in self.closures.values():
            for with_wrapper in (True, False):
                with torch.no_grad():
                    cache = {}  # cache random starting values
                    for name, param in closure.parameters.items():
                        init = cache[name] = torch.rand_like(param)
                        param.data.copy_(init)

                closure_arg = (
                    NdarrayOptimizationClosure(closure, closure.parameters)
                    if with_wrapper
                    else closure
                )
                result = scipy_minimize(
                    closure=closure_arg,
                    parameters=closure.parameters,
                    bounds={"x": (0, 1)},
                )
                self.assertIsInstance(result, OptimizationResult)
                self.assertEqual(result.status, OptimizationStatus.SUCCESS)
                self.assertTrue(allclose(result.fval, 0.0))
                self.assertTrue(closure.parameters["dummy"].equal(cache["dummy"]))
                self.assertFalse(closure.parameters["x"].equal(cache["x"]))

        # Test `bounds` and `callback`
        with torch.no_grad():  # closure.forward is a ToyModule instance
            closure.forward.b.fill_(0.0)
            closure.forward.x.fill_(0.5)

        step_results = []
        result = scipy_minimize(
            closure=closure,
            parameters=closure.parameters,
            bounds={"x": (0.1, 1.0)},
            callback=partial(_callback, out=step_results),
        )
        self.assertTrue(allclose(0.01, result.fval))
        self.assertTrue(allclose(0.1, closure.forward.x.detach().cpu().item()))

        self.assertEqual(result.step, len(step_results))
        self.assertEqual(result.step, step_results[-1].step)
        self.assertEqual(result.fval, step_results[-1].fval)

    def test_post_processing(self):
        closure = next(iter(self.closures.values()))
        wrapper = NdarrayOptimizationClosure(closure, closure.parameters)

        # Scipy changed return values and messages in v1.15, so we check both
        # old and new versions here.
        status_msgs = [
            # scipy >=1.15
            (OptimizationStatus.FAILURE, "ABNORMAL_TERMINATION_IN_LNSRCH"),
            (OptimizationStatus.STOPPED, "TOTAL NO. of ITERATIONS REACHED LIMIT"),
            # scipy <1.15
            (OptimizationStatus.FAILURE, "ABNORMAL "),
            (OptimizationStatus.STOPPED, "TOTAL NO. OF ITERATIONS REACHED LIMIT"),
        ]

        with patch.object(core, "minimize_with_timeout") as mock_minimize_with_timeout:
            for status, msg in status_msgs:
                mock_minimize_with_timeout.return_value = OptimizeResult(
                    x=wrapper.state,
                    fun=1.0,
                    nit=3,
                    success=False,
                    message=msg,
                )
                result = core.scipy_minimize(wrapper, closure.parameters)
                self.assertEqual(result.status, status)
                self.assertEqual(
                    result.fval, mock_minimize_with_timeout.return_value.fun
                )
                self.assertEqual(
                    result.message, msg if isinstance(msg, str) else msg.decode("ascii")
                )


class TestTorchMinimize(BotorchTestCase):
    def setUp(self):
        super().setUp()
        module = ToyModule(
            x=Parameter(torch.tensor(0.5, device=self.device)),
            b=Parameter(torch.tensor(0.0, device=self.device), requires_grad=False),
            dummy=Parameter(torch.tensor(1.0, device=self.device)),
        ).to(self.device)

        self.closures = {}
        for dtype in ("float32", "float64"):
            m = module.to(dtype=getattr(torch, dtype))
            self.closures[dtype] = ForwardBackwardClosure(m, m.free_parameters)

    def test_basic(self):
        x = Parameter(torch.tensor([0.02]))
        closure = partial(norm_squared, x)
        result = torch_minimize(closure, {"x": x}, step_limit=100)
        self.assertEqual(result.status, OptimizationStatus.STOPPED)
        self.assertTrue(allclose(result.fval, 0.0))

    def test_timeout(self):
        x = Parameter(torch.tensor(1.0))
        # adding a small delay here to combat some timing issues on windows
        closure = partial(norm_squared, x, delay=1e-3)
        result = torch_minimize(closure, {"x": x}, timeout_sec=1e-4)
        self.assertEqual(result.status, OptimizationStatus.STOPPED)
        self.assertTrue("stopped due to timeout after" in result.message)

    def test_main(self):
        def _callback(parameters, result, out) -> None:
            out.append(result)

        for closure in self.closures.values():
            # Test that we error out if no termination conditions are given
            with self.assertRaisesRegex(RuntimeError, "No termination conditions"):
                torch_minimize(closure=closure, parameters=closure.parameters)

            # Test single step behavior
            for optimizer in (
                SGD(params=list(closure.parameters.values()), lr=0.1),  # instance
                partial(SGD, lr=0.1),  # factory
            ):
                cache = {n: p.detach().clone() for n, p in closure.parameters.items()}
                grads = [g if g is None else g.detach().clone() for g in closure()[1]]
                result = torch_minimize(
                    closure=closure,
                    parameters=closure.parameters,
                    optimizer=optimizer,
                    step_limit=1,
                )
                self.assertIsInstance(result, OptimizationResult)
                self.assertEqual(result.fval, closure()[0])
                self.assertEqual(result.step, 1)
                self.assertEqual(result.status, OptimizationStatus.STOPPED)
                self.assertTrue(closure.parameters["dummy"].equal(cache["dummy"]))
                self.assertFalse(closure.parameters["x"].equal(cache["x"]))
                for (name, param), g in zip(closure.parameters.items(), grads):
                    self.assertTrue(
                        param.allclose(cache[name] - (0 if g is None else 0.1 * g))
                    )

            # Test local convergence
            with torch.no_grad():  # closure.forward is a ToyModule instance
                closure.forward.b.fill_(0.0)
                closure.forward.x.fill_(0.02)

            result = torch_minimize(closure, closure.parameters, step_limit=100)
            self.assertTrue(allclose(0.0, result.fval))
            self.assertEqual(result.step, 100)

            # Test `bounds` and `callback`
            with torch.no_grad():  # closure.forward is a ToyModule instance
                closure.forward.b.fill_(0.0)
                closure.forward.x.fill_(0.11)

            step_results = []
            result = torch_minimize(
                closure=closure,
                parameters=closure.parameters,
                bounds={"x": (0.1, 1.0)},
                callback=partial(_callback, out=step_results),
                step_limit=100,
            )
            self.assertTrue(allclose(0.01, result.fval))
            self.assertEqual(result.step, len(step_results))

            # Test `stopping_criterion`
            max3_stopping_criterion = ExpMAStoppingCriterion(maxiter=3, n_window=5)
            result = torch_minimize(
                closure=closure,
                parameters=closure.parameters,
                stopping_criterion=max3_stopping_criterion,
            )
            self.assertEqual(result.step, 3)
            self.assertEqual(result.status, OptimizationStatus.STOPPED)

            # Test passing `scheduler`
            mock_scheduler = MagicMock(spec=LRScheduler)
            mock_scheduler.step = MagicMock(side_effect=RuntimeError("foo"))
            with self.assertRaisesRegex(RuntimeError, "foo"):
                torch_minimize(
                    closure=closure,
                    parameters=closure.parameters,
                    scheduler=mock_scheduler,
                    step_limit=1,
                )
            mock_scheduler.step.assert_called_once()

            # Test passing `scheduler` as a factory
            optimizer = SGD(list(closure.parameters.values()), lr=1e-3)
            mock_factory = MagicMock(side_effect=RuntimeError("foo"))
            with self.assertRaisesRegex(RuntimeError, "foo"):
                torch_minimize(
                    closure=closure,
                    parameters=closure.parameters,
                    optimizer=optimizer,
                    scheduler=mock_factory,
                    step_limit=1,
                )
            mock_factory.assert_called_once_with(optimizer)
