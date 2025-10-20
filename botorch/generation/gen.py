#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Candidate generation utilities.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Mapping, NoReturn

import numpy as np
import numpy.typing as npt
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import OptimizationGradientError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.generation.utils import _remove_fixed_features_from_optimization
from botorch.logging import logger
from botorch.optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
    make_scipy_linear_constraints,
    make_scipy_nonlinear_inequality_constraints,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (
    check_scipy_version_at_least,
    columnwise_clamp,
    fix_features,
    minimize_with_timeout,
)
from scipy.optimize import OptimizeResult
from threadpoolctl import threadpool_limits
from torch import Tensor
from torch.optim import Optimizer

if check_scipy_version_at_least(minor=13) and not check_scipy_version_at_least(
    minor=17
):
    # We only import the batched lbfgs_b code here, as it might otherwise
    # lead to import errors, if the wrong scipy version is used
    from botorch.optim.batched_lbfgs_b import (
        fmin_l_bfgs_b_batched,
        translate_bounds_for_lbfgsb,
    )

TGenCandidates = Callable[[Tensor, AcquisitionFunction, Any], tuple[Tensor, Tensor]]


def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: float | Tensor | None = None,
    upper_bounds: float | Tensor | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
    options: dict[str, Any] | None = None,
    fixed_features: Mapping[int, float | Tensor] | None = None,
    timeout_sec: float | None = None,
    use_parallel_mode: bool | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using `scipy.optimize.minimize`.

    Optimizes an acquisition function starting from a set of initial candidates
    using `scipy.optimize.minimize` via a numpy converter.
    We use SLSQP, if constraints are present, and LBFGS-B otherwise.
    As `scipy.optimize.minimize` does not support optimizating a batch of problems, we
    treat optimizing a set of candidates as a single optimization problem by
    summing together their acquisition values.

    Args:
        initial_conditions: Starting points for optimization, with shape
            (b) x q x d.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver.
        options: Options used to control the optimization including "method"
            and "maxiter". Select method for `scipy.optimize.minimize` using the
            "method" key. By default uses L-BFGS-B for box-constrained problems
            and SLSQP if inequality or equality constraints are present. If
            `with_grad=False`, then we use a two-point finite difference estimate
            of the gradient.
        fixed_features: Mapping[int, float | Tensor] | None,
            all generated candidates will have features fixed to these values.
            If passing tensors as values, they should have either shape `b` or
            `b x q` to fix the same feature to different values in the batch.
            Assumes values to be compatible with lower_bounds and upper_bounds!
        timeout_sec: Timeout (in seconds) for `scipy.optimize.minimize` routine -
            if provided, optimization will stop after this many seconds and return
            the best solution found so far.
        use_parallel_mode: If None uses the parallel implementation of l-bfgs-b,
            if possible.
            If True, forces the use of the parallel implementation and fails if not
            possible.
            If using parallel mode, each item in the batch dimension is treated as a
            separate optimization problem, we enforce the shape of the initial
            conditions to be `b x q x d` or `q x d`, and we assume the
            `acquisition_function` does not treat elements differently in the batch
            dimension (it is simply a batched function).
            If False, forces the use of the serial implementation through
            `scipy.optimize.minimize`.

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    if use_parallel_mode:
        if initial_conditions.ndim not in (2, 3):
            raise UnsupportedError(
                "Initial conditions must have either 2 or 3 dimensions. "
                f"Received shape {initial_conditions.shape}"
            )
    options = options or {}
    options = {**options, "maxiter": options.get("maxiter", 2000)}

    original_initial_conditions_shape = initial_conditions.shape

    if initial_conditions.ndim == 2:
        initial_conditions = initial_conditions.unsqueeze(0)

    initial_conditions_all_features = initial_conditions
    if fixed_features:
        initial_conditions = initial_conditions[
            ...,
            [i for i in range(initial_conditions.shape[-1]) if i not in fixed_features],
        ]
        if isinstance(lower_bounds, Tensor):
            lower_bounds = lower_bounds[
                [i for i in range(len(lower_bounds)) if i not in fixed_features]
            ]
        if isinstance(upper_bounds, Tensor):
            upper_bounds = upper_bounds[
                [i for i in range(len(upper_bounds)) if i not in fixed_features]
            ]

    clamped_candidates = columnwise_clamp(
        X=initial_conditions,
        lower=lower_bounds,
        upper=upper_bounds,
        raise_on_violation=True,
    )

    def f(x):
        return -acquisition_function(x)

    is_constrained = (
        nonlinear_inequality_constraints
        or equality_constraints
        or inequality_constraints
    )
    method = options.get("method", "SLSQP" if is_constrained else "L-BFGS-B")
    with_grad = options.get("with_grad", True)
    minimize_options = {
        k: v
        for k, v in options.items()
        if k
        not in [
            "method",
            "callback",
            "with_grad",
            "max_optimization_problem_aggregation_size",
        ]
    }

    why_not_fast_path = get_reasons_against_fast_path(
        method=method,
        with_grad=with_grad,
        minimize_options=minimize_options,
        timeout_sec=timeout_sec,
    )

    f_np_wrapper = _get_f_np_wrapper(
        clamped_candidates.shape,
        initial_conditions.device,
        initial_conditions.dtype,
        with_grad,
    )

    if not why_not_fast_path and use_parallel_mode is not False:
        if is_constrained:
            raise RuntimeWarning("Method L-BFGS-B cannot handle constraints.")

        batched_x0 = _arrayify(clamped_candidates).reshape(len(clamped_candidates), -1)

        l_bfgs_b_bounds = translate_bounds_for_lbfgsb(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            num_features=clamped_candidates.shape[-1],
            q=clamped_candidates.shape[1],
        )

        with threadpool_limits(limits=1, user_api="blas"):
            xs, fs, results = fmin_l_bfgs_b_batched(
                func=partial(f_np_wrapper, f=f, fixed_features=fixed_features),
                # args is not necessary, done via the partial instead
                x0=batched_x0,
                # method=method, # method is not necessary as it is only l-bfgs-b
                # jac=with_grad, this is assumed to be true
                bounds=l_bfgs_b_bounds,
                # constraints=constraints,
                callback=options.get("callback", None),
                pass_batch_indices=True,
                **minimize_options,
            )
        for res in results:
            _process_scipy_result(res=res, options=options)

    else:
        # In this case we optimize multiple initial conditions in a single
        # problem, up to max_optimization_problem_aggregation_size at a time.
        max_optimization_problem_aggregation_size = options.get(
            "max_optimization_problem_aggregation_size", len(clamped_candidates)
        )

        if use_parallel_mode is not False:
            msg = (
                "Not using the parallel implementation of l-bfgs-b, as: "
                + ", and ".join(why_not_fast_path)
            )
            if use_parallel_mode:
                raise NotImplementedError(msg)
            else:
                logger.debug(msg)

        if (
            fixed_features
            and any(
                torch.is_tensor(ff) and ff.ndim > 0 for ff in fixed_features.values()
            )
            and max_optimization_problem_aggregation_size != 1
        ):
            raise UnsupportedError(
                "Batch shaped fixed features are not "
                "supported, when optimizing more than one optimization "
                "problem at a time."
            )

        all_xs = []
        split_candidates = clamped_candidates.split(
            max_optimization_problem_aggregation_size
        )
        for i, candidates_ in enumerate(split_candidates):
            if fixed_features:
                fixed_features_ = {
                    k: ff[i : i + 1].item()
                    # from the test above, we know that we only treat one candidate
                    # at a time thus we can use index i
                    if torch.is_tensor(ff) and ff.ndim > 0
                    else ff
                    for k, ff in fixed_features.items()
                }
            else:
                fixed_features_ = None

            _no_fixed_features = _remove_fixed_features_from_optimization(
                fixed_features=fixed_features_,
                acquisition_function=acquisition_function,
                initial_conditions=None,
                d=initial_conditions_all_features.shape[-1],
                lower_bounds=None,
                upper_bounds=None,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            )
            bounds = make_scipy_bounds(
                X=candidates_,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            f_np_wrapper_ = partial(
                f_np_wrapper,
                fixed_features=fixed_features_,
            )

            x0 = candidates_.flatten()

            constraints = make_scipy_linear_constraints(
                shapeX=candidates_.shape,
                inequality_constraints=_no_fixed_features.inequality_constraints,
                equality_constraints=_no_fixed_features.equality_constraints,
            )
            if _no_fixed_features.nonlinear_inequality_constraints:
                # Make sure `batch_limit` is 1 for now.
                if not (len(candidates_.shape) == 3 and candidates_.shape[0] == 1):
                    raise ValueError(
                        "`batch_limit` must be 1 when non-linear inequality "
                        "constraints are given."
                    )
                nl_ineq_constraints = (
                    _no_fixed_features.nonlinear_inequality_constraints
                )
                constraints += make_scipy_nonlinear_inequality_constraints(
                    nonlinear_inequality_constraints=nl_ineq_constraints,
                    f_np_wrapper=f_np_wrapper_,
                    x0=x0,
                    shapeX=candidates_.shape,
                )

            x0 = _arrayify(x0)

            res = minimize_with_timeout(
                fun=f_np_wrapper_,
                args=(f,),
                x0=x0,
                method=method,
                jac=with_grad,
                bounds=bounds,
                constraints=constraints,
                callback=options.get("callback", None),
                options=minimize_options,
                timeout_sec=timeout_sec / len(split_candidates)
                if timeout_sec is not None
                else None,
            )
            _process_scipy_result(res=res, options=options)
            xs = res.x.reshape(candidates_.shape)
            all_xs.append(xs)
        xs = np.concatenate(all_xs)

    candidates = torch.from_numpy(xs).view_as(clamped_candidates).to(initial_conditions)

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )

    clamped_candidates = fix_features(
        X=clamped_candidates,
        fixed_features=fixed_features,
        replace_current_value=False,
    )
    clamped_candidates = clamped_candidates.reshape(original_initial_conditions_shape)

    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition


def _get_f_np_wrapper(shapeX, device, dtype, with_grad):
    if with_grad:

        def f_np_wrapper(
            x: npt.NDArray,
            f: Callable,
            fixed_features: Mapping[int, float | Tensor] | None,
            batch_indices: list[int] | None = None,
        ) -> tuple[float | np.NDArray, np.NDArray]:
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(device=device, dtype=dtype)
                # We reshape in this way, as in parallel mode the batch dimension might
                # change during optimization: some examples might finish earlier than
                # others.
                .view(-1, *shapeX[1:])
                .contiguous()
                .requires_grad_(True)
            )
            if fixed_features is not None:
                if batch_indices is not None:
                    this_fixed_features = {
                        k: ff[batch_indices]
                        if torch.is_tensor(ff) and ff.ndim > 0
                        else ff
                        for k, ff in fixed_features.items()
                    }
                else:
                    this_fixed_features = fixed_features
            else:
                this_fixed_features = None

            X_fix = fix_features(
                X, fixed_features=this_fixed_features, replace_current_value=False
            )
            # we compute the loss on the whole batch, under the assumption that f
            # treats multiple inputs in the 0th dimension as independent
            # inputs in a batch
            losses = f(X_fix)
            loss = losses.sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
            gradf = gradf.reshape(*x.shape)
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. "
                    "This often indicates numerical issues."
                )
                if x.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise OptimizationGradientError(msg, current_x=x)
            fval = (
                losses.detach().view(-1).cpu().numpy()
                if batch_indices is not None
                else loss.detach().item()
            )  # the view(-1) seems necessary as f might return a single scalar
            return fval, gradf

    else:
        # This function (that is used if no grads are avail) can also be batched,
        # we just did not batch it so far as the priority is not as high.
        def f_np_wrapper(
            x: npt.NDArray, f: Callable, fixed_features: dict[int, float] | None
        ):
            X = (
                torch.from_numpy(x)
                .to(device=device, dtype=dtype)
                .view(-1, *shapeX[1:])
                .contiguous()
            )
            with torch.no_grad():
                X_fix = fix_features(
                    X=X, fixed_features=fixed_features, replace_current_value=False
                )
                loss = f(X_fix).sum()
            fval = loss.detach().item()
            return fval

    return f_np_wrapper


def get_reasons_against_fast_path(
    method: str,
    with_grad: bool,
    minimize_options: dict[str, Any],
    timeout_sec: float | None,
) -> list[str]:
    # this if-statement is a homage to pytorch's nn.MultiheadAttentionby by @swolchok
    why_not_fast_path = []
    if not method == "L-BFGS-B":
        why_not_fast_path.append(f"method={method}, method needs to be L-BFGS-B")
    if not with_grad:
        why_not_fast_path.append("with_grad=False, it needs to be True")
    if extra_keys := set(minimize_options.keys()) - {
        "maxiter",
        "disp",
        "iprint",
        "max_cor",
        "ftol",
        "pgtol",
        "factr",
        "tol",
        "maxls",
    }:
        why_not_fast_path.append(f"options={extra_keys} are not accepted")
    if timeout_sec is not None:
        why_not_fast_path.append(f"timeout_sec={timeout_sec}, it needs to be None")
    if not check_scipy_version_at_least(minor=13) or check_scipy_version_at_least(
        minor=17
    ):  # pragma: no cover
        # In SciPy 1.15.0, the fortran implementation of L-BFGS-B was
        # translated to C changing its interface slightly.
        # Additionally, we don't know what the future might hold in scipy,
        # thus we use this function to use less optimized code for too new
        # scipy versions.
        why_not_fast_path.append(
            "Scipy version is not in the range from 1.13.0 to 1.15.x."
        )
    return why_not_fast_path


def gen_candidates_torch(
    initial_conditions: Tensor,
    acquisition_function: AcquisitionFunction,
    lower_bounds: float | Tensor | None = None,
    upper_bounds: float | Tensor | None = None,
    optimizer: type[Optimizer] = torch.optim.Adam,
    options: dict[str, float | str] | None = None,
    callback: Callable[[int, Tensor, Tensor], NoReturn] | None = None,
    fixed_features: Mapping[int, float | Tensor] | None = None,
    timeout_sec: float | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using a `torch.optim` optimizer.

    Optimizes an acquisition function starting from a set of initial candidates
    using an optimizer from `torch.optim`.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        optimizer (Optimizer): The pytorch optimizer to use to perform
            candidate search.
        options: Options used to control the optimization. Includes
            maxiter: Maximum number of iterations
        callback: A callback function accepting the current iteration, loss,
            and gradients as arguments. This function is executed after computing
            the loss and gradients, but before calling the optimizer.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If a float is passed it is fixed across [b,q], if a tensor is passed:
            it might either be of shape [b,q] or [b], in which case the same value
            is used across the q dimension.
            Assumes values to be compatible with lower_bounds and upper_bounds!
        timeout_sec: Timeout (in seconds) for optimization. If provided,
            `gen_candidates_torch` will stop after this many seconds and return
            the best solution found so far.

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    start_time = time.monotonic()
    options = options or {}
    # We remove max_optimization_problem_aggregation_size as it does not affect
    # the 1st order optimizers implemented in this method.
    # Here, it does not matter whether one combines multiple optimizations into
    # one or not.
    options.pop("max_optimization_problem_aggregation_size", None)
    _clamp = partial(columnwise_clamp, lower=lower_bounds, upper=upper_bounds)
    clamped_candidates = _clamp(initial_conditions)
    if fixed_features:
        clamped_candidates = clamped_candidates[
            ...,
            [i for i in range(clamped_candidates.shape[-1]) if i not in fixed_features],
        ]
    clamped_candidates = clamped_candidates.requires_grad_(True)
    _optimizer = optimizer(params=[clamped_candidates], lr=options.get("lr", 0.025))

    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(**options)
    while not stop:
        i += 1
        with torch.no_grad():
            X = _clamp(clamped_candidates).requires_grad_(True)

        loss = -acquisition_function(fix_features(X, fixed_features)).sum()
        grad = torch.autograd.grad(loss, X)[0]
        if callback:
            callback(i, loss, grad)

        def assign_grad():
            _optimizer.zero_grad()
            clamped_candidates.grad = grad
            return loss

        _optimizer.step(assign_grad)
        stop = stopping_criterion(fvals=loss.detach())
        if timeout_sec is not None:
            runtime = time.monotonic() - start_time
            if runtime > timeout_sec:
                stop = True
                logger.info(f"Optimization timed out after {runtime} seconds.")

    clamped_candidates = _clamp(clamped_candidates)
    clamped_candidates = fix_features(clamped_candidates, fixed_features)
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition


def get_best_candidates(batch_candidates: Tensor, batch_values: Tensor) -> Tensor:
    r"""Extract best (q-batch) candidate from batch of candidates

    Args:
        batch_candidates: A `b x q x d` tensor of `b` q-batch candidates, or a
            `b x d` tensor of `b` single-point candidates.
        batch_values: A tensor with `b` elements containing the value of the
            respective candidate (higher is better).

    Returns:
        A tensor of size `q x d` (if q-batch mode) or `d` from batch_candidates
        with the highest associated value.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
        >>> best_candidates = get_best_candidates(batch_candidates, batch_acq_values)
    """
    best = torch.argmax(batch_values.view(-1), dim=0)
    return batch_candidates[best]


def _process_scipy_result(res: OptimizeResult, options: dict[str, Any]) -> None:
    r"""Process scipy optimization result to produce relevant logs and warnings."""
    if "success" not in res.keys() or "status" not in res.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("always", category=OptimizationWarning)
            warnings.warn(
                "Optimization failed within `scipy.optimize.minimize` with no "
                "status returned to `res.`",
                OptimizationWarning,
                stacklevel=3,
            )
    elif not res.success:
        if (
            "ITERATIONS REACHED LIMIT" in res.message
            or "Iteration limit reached" in res.message
        ):
            logger.info(
                "`scipy.optimize.minimize` exited by reaching the iteration limit of "
                f"`maxiter: {options.get('maxiter')}`."
            )
        elif "EVALUATIONS EXCEEDS LIMIT" in res.message:
            logger.info(
                "`scipy.optimize.minimize` exited by reaching the function evaluation "
                f"limit of `maxfun: {options.get('maxfun')}`."
            )
        elif "Optimization timed out after" in res.message:
            logger.info(res.message)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("always", category=OptimizationWarning)
                warnings.warn(
                    f"Optimization failed within `scipy.optimize.minimize` with status "
                    f"{res.status} and message {res.message}.",
                    OptimizationWarning,
                    stacklevel=3,
                )
