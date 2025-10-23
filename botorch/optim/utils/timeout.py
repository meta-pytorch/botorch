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

import numpy.typing as npt
from botorch.exceptions.errors import OptimizationTimeoutError
from scipy import optimize
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
    verbatim. The only difference is that if provided a `timeout_sec` argument,
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

    try:
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
        current_fun, *_ = fun(e.current_x, *args)

        return optimize.OptimizeResult(
            fun=current_fun,
            x=e.current_x,
            nit=callback_data["num_iterations"],
            success=False,  # same as when maxiter is reached
            status=1,  # same as when L-BFGS-B reaches maxiter
            message=msg,
        )
