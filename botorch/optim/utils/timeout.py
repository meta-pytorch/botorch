#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from botorch.exceptions.errors import OptimizationTimeoutError
from scipy import optimize
from scipy.optimize._constraints import old_bound_to_new
from threadpoolctl import threadpool_limits


def minimize_with_timeout(
    fun: Callable[[npt.NDArray, ...], float],
    x0: npt.NDArray,
    args: tuple[Any, ...] = (),
    method: str | None = None,
    jac: str | Callable | bool | None = None,
    hess: str | Callable | optimize.HessianUpdateStrategy | None = None,
    hessp: Callable | None = None,
    bounds: Sequence[tuple[float, float]] | optimize.Bounds | None = None,
    constraints=(),  # Typing this properly is a s**t job
    tol: float | None = None,
    callback: Callable | None = None,
    options: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> optimize.OptimizeResult:
    r"""Wrapper around scipy.optimize.minimize to support timeout.

    This method calls scipy.optimize.minimize with all arguments forwarded
    verbatim. The only difference is that if provided a ``timeout_sec`` argument,
    it will automatically stop the optimization after the timeout is reached.

    Internally, this is achieved by automatically constructing a wrapper callback
    method that is injected to the scipy.optimize.minimize call and that keeps
    track of the runtime and the optimization variables at the current iteration.
    """
    if timeout_sec is not None:
        start_time = time.monotonic()
        callback_data = {"num_iterations": 0}  # update from withing callback below

        def timeout_callback(xk: npt.NDArray) -> bool:
            runtime = time.monotonic() - start_time
            callback_data["num_iterations"] += 1
            if runtime > timeout_sec:
                raise OptimizationTimeoutError(current_x=xk, runtime=runtime)
            return False

        if callback is None:
            wrapped_callback = timeout_callback

        elif callable(method):
            raise NotImplementedError(
                "Custom callable not supported for `method` argument."
            )

        elif method == "trust-constr":  # special signature

            def wrapped_callback(
                xk: npt.NDArray, state: optimize.OptimizeResult
            ) -> bool:
                # order here is important to make sure base callback gets executed
                return callback(xk, state) or timeout_callback(xk=xk)

        else:

            def wrapped_callback(xk: npt.NDArray) -> None:
                timeout_callback(xk=xk)
                callback(xk)

    else:
        wrapped_callback = callback

    # SciPy's L-BFGS-B checks its iteration budget after completing the first step,
    # so ``maxiter=0`` would still advance ``x0`` by one full iteration.
    is_lbfgsb = (isinstance(method, str) and method.lower() == "l-bfgs-b") or (
        method is None and bounds is not None and not constraints
    )
    if (
        is_lbfgsb
        and not constraints
        and options is not None
        and options.get("maxiter") == 0
    ):
        x = np.asarray(x0, dtype=np.float64).copy()
        if bounds is not None:
            if isinstance(bounds, optimize.Bounds):
                lb, ub = np.asarray(bounds.lb), np.asarray(bounds.ub)
            else:
                lb, ub = np.asarray(old_bound_to_new(bounds))
            if (lb > ub).any():
                raise ValueError(
                    "LBFGSB - one of the lower bounds is greater than an upper bound."
                )
            x = np.clip(x, lb, ub)
        with threadpool_limits(limits=1, user_api="blas"):
            if jac is True:
                fval, grad = fun(x, *args)
                njev = 1
            else:
                fval = fun(x, *args)
                grad = jac(x, *args) if callable(jac) else None
                njev = 1 if callable(jac) else 0
        return optimize.OptimizeResult(
            fun=float(np.asarray(fval).item()),
            jac=None if grad is None else np.asarray(grad, dtype=np.float64),
            nfev=1,
            njev=njev,
            nit=0,
            status=1,
            message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            x=x,
            success=False,
        )

    try:
        # Only raise these warnings as errors within this call, rather than
        # globally modifying the warning filters.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Method .* cannot handle")
            # To prevent slowdowns after scipy 1.15.
            # See https://github.com/scipy/scipy/issues/22438.
            with threadpool_limits(limits=1, user_api="blas"):
                return optimize.minimize(
                    fun=fun,
                    x0=x0,
                    args=args,
                    method=method,
                    jac=jac,
                    hess=hess,
                    hessp=hessp,
                    bounds=bounds,
                    constraints=constraints,
                    tol=tol,
                    callback=wrapped_callback,
                    options=options,
                )
    except OptimizationTimeoutError as e:
        msg = f"Optimization timed out after {e.runtime} seconds."
        current_fun = fun(e.current_x, *args)
        if jac is True:
            # As in ``scipy.optimize.minimize``, ``fun`` returns the function value
            # and the gradient if ``jac`` is True.
            current_fun = current_fun[0]

        return optimize.OptimizeResult(
            fun=current_fun,
            x=e.current_x,
            nit=callback_data["num_iterations"],
            success=False,  # same as when maxiter is reached
            status=1,  # same as when L-BFGS-B reaches maxiter
            message=msg,
        )
