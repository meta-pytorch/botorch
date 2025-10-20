#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Core abstractions and generic optimizers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import auto, Enum
from itertools import count
from sys import maxsize
from time import monotonic
from typing import Any

import numpy.typing as npt

from botorch.optim.closures import NdarrayOptimizationClosure
from botorch.optim.stopping import StoppingCriterion
from botorch.optim.utils.numpy_utils import get_bounds_as_ndarray
from botorch.optim.utils.timeout import minimize_with_timeout
from numpy import asarray, float64 as np_float64
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # pragma: no cover


_LBFGSB_MAXITER_MAXFUN_REGEX = re.compile(  # regex for maxiter and maxfun messages
    # Note that the messages changed with scipy 1.15, hence the different matching here.
    "TOTAL NO. (of|OF) "
    + "(ITERATIONS REACHED LIMIT|(f AND g|F,G) EVALUATIONS EXCEEDS LIMIT)"
)


class OptimizationStatus(int, Enum):
    RUNNING = auto()  # incomplete
    SUCCESS = auto()  # optimizer converged
    FAILURE = auto()  # terminated abnormally
    STOPPED = auto()  # stopped due to user provided criterion


@dataclass
class OptimizationResult:
    step: int
    fval: float | int
    status: OptimizationStatus
    runtime: float | None = None
    message: str | None = None


def scipy_minimize(
    closure: (
        Callable[[], tuple[Tensor, Sequence[Tensor | None]]]
        | NdarrayOptimizationClosure
    ),
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    callback: Callable[[dict[str, Tensor], OptimizationResult], None] | None = None,
    x0: npt.NDArray | None = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic scipy.optimize.minimize-based optimization routine.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient tensors or
            NdarrayOptimizationClosure instance.
        parameters: A dictionary of tensors to be optimized.
        bounds: A dictionary mapping parameter names to lower and upper bounds.
        callback: A callable taking `parameters` and an OptimizationResult as arguments.
        x0: An optional initialization vector passed to scipy.optimize.minimize.
        method: Solver type, passed along to scipy.optimize.minimize.
        options: Dictionary of solver options, passed along to scipy.optimize.minimize.
        timeout_sec: Timeout in seconds to wait before aborting the optimization loop
            if not converged (will return the best found solution thus far).

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    start_time = monotonic()
    wrapped_closure = (
        closure
        if isinstance(closure, NdarrayOptimizationClosure)
        else NdarrayOptimizationClosure(closure, parameters)
    )
    if bounds is None:
        bounds_np = None
    else:
        bounds_np = get_bounds_as_ndarray(parameters, bounds)

    if callback is None:
        wrapped_callback = None
    else:
        call_counter = count(1)  # callbacks are typically made at the end of each iter

        def wrapped_callback(x: npt.NDArray):
            result = OptimizationResult(
                step=next(call_counter),
                fval=float(wrapped_closure(x)[0]),
                status=OptimizationStatus.RUNNING,
                runtime=monotonic() - start_time,
            )
            return callback(parameters, result)  # pyre-ignore [29]

    raw = minimize_with_timeout(
        wrapped_closure,
        wrapped_closure.state if x0 is None else x0.astype(np_float64, copy=False),
        jac=True,
        bounds=bounds_np,
        method=method,
        options=options,
        callback=wrapped_callback,
        timeout_sec=timeout_sec,
    )

    # Post-processing and outcome handling
    wrapped_closure.state = asarray(raw.x)  # set parameter state to optimal values
    msg = raw.message if isinstance(raw.message, str) else raw.message.decode("ascii")
    if raw.success:
        status = OptimizationStatus.SUCCESS
    else:
        status = (  # Check whether we stopped due to reaching maxfun or maxiter
            OptimizationStatus.STOPPED
            if _LBFGSB_MAXITER_MAXFUN_REGEX.search(msg)
            or "Optimization timed out after" in msg
            else OptimizationStatus.FAILURE
        )

    return OptimizationResult(
        fval=raw.fun,
        step=raw.nit,
        status=status,
        message=msg,
        runtime=monotonic() - start_time,
    )


def torch_minimize(
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]],
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    callback: Callable[[dict[str, Tensor], OptimizationResult], None] | None = None,
    optimizer: Optimizer | Callable[[list[Tensor]], Optimizer] = Adam,
    scheduler: LRScheduler | Callable[[Optimizer], LRScheduler] | None = None,
    step_limit: int | None = None,
    timeout_sec: float | None = None,
    stopping_criterion: StoppingCriterion | None = None,
) -> OptimizationResult:
    r"""Generic torch.optim-based optimization routine.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient tensors.
            Responsible for setting relevant parameters' `grad` attributes.
        parameters: A dictionary of tensors to be optimized.
        bounds: An optional dictionary of bounds for elements of `parameters`.
        callback: A callable taking `parameters` and an OptimizationResult as arguments.
        optimizer: A `torch.optim.Optimizer` instance or a factory that takes
            a list of parameters and returns an `Optimizer` instance.
        scheduler: A `torch.optim.lr_scheduler._LRScheduler` instance or a factory
            that takes a `Optimizer` instance and returns a `_LRSchedule` instance.
        step_limit: Integer specifying a maximum number of optimization steps.
            One of `step_limit`, `stopping_criterion`, or `timeout_sec` must be passed.
        timeout_sec: Timeout in seconds before terminating the optimization loop.
            One of `step_limit`, `stopping_criterion`, or `timeout_sec` must be passed.
        stopping_criterion: A StoppingCriterion for the optimization loop.

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    result: OptimizationResult
    start_time = monotonic()

    if step_limit is None:
        if stopping_criterion is None and timeout_sec is None:
            raise RuntimeError("No termination conditions were given.")
        step_limit = maxsize

    if not isinstance(optimizer, Optimizer):
        optimizer = optimizer(list(parameters.values()))

    if not (scheduler is None or isinstance(scheduler, LRScheduler)):
        scheduler = scheduler(optimizer)

    if stopping_criterion is not None:
        # Reset stopping criterion to ensure clean state for new optimization run
        stopping_criterion.reset()

    _bounds = (
        {}
        if bounds is None
        else {name: limits for name, limits in bounds.items() if name in parameters}
    )
    for step in range(1, step_limit + 1):
        fval = closure()[0].detach()
        runtime = monotonic() - start_time
        result = OptimizationResult(
            step=step,
            fval=fval.cpu().item(),
            status=OptimizationStatus.RUNNING,
            runtime=runtime,
        )

        # TODO: Update stopping_criterion API to return a message.
        if stopping_criterion and stopping_criterion(fval):
            result.status = OptimizationStatus.STOPPED
            result.message = "`torch_minimize` stopped due to `stopping_criterion`."

        if timeout_sec is not None and runtime >= timeout_sec:
            result.status = OptimizationStatus.STOPPED
            result.message = (
                f"`torch_minimize` stopped due to timeout after {runtime} seconds."
            )

        if callback:
            callback(parameters, result)

        if result.status != OptimizationStatus.RUNNING:
            break

        optimizer.step()
        for name, (lower, upper) in _bounds.items():
            parameters[name].data = parameters[name].clamp(min=lower, max=upper)

        if scheduler:
            scheduler.step()

    if result.status != OptimizationStatus.RUNNING:
        return replace(result, runtime=monotonic() - start_time)

    # Account for final parameter update when stopping due to step_limit
    return OptimizationResult(
        step=step,
        fval=closure()[0].detach().cpu().item(),
        status=OptimizationStatus.STOPPED,
        runtime=monotonic() - start_time,
        message=f"`torch_minimize` stopped after reaching step_limit={step_limit}.",
    )
