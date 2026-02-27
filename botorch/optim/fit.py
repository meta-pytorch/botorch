#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tools for model fitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
from warnings import warn

from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    scipy_minimize,
    torch_minimize,
)
from botorch.optim.stopping import ExpMAStoppingCriterion, StoppingCriterion
from botorch.optim.utils import get_parameters_and_bounds, TorchAttr
from botorch.utils.types import DEFAULT
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

TBoundsDict = dict[str, tuple[float | None, float | None]]
TScipyObjective = Callable[
    [ndarray, MarginalLogLikelihood, dict[str, TorchAttr]], tuple[float, ndarray]
]
TModToArray = Callable[
    [Module, TBoundsDict | None, set[str] | None],
    tuple[ndarray, dict[str, TorchAttr], ndarray | None],
]
TArrayToMod = Callable[[Module, ndarray, dict[str, TorchAttr]], Module]


def fit_gpytorch_mll_scipy(
    mll: MarginalLogLikelihood,
    parameters: dict[str, Tensor] | None = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Tensor], OptimizationResult], None] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic scipy.optimize-based fitting routine for GPyTorch MLLs.

    The model and likelihood in mll must already be in train mode.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized. Defaults
            to all parameters of ``mll`` that require gradients.
        bounds: A dictionary of user-specified bounds for ``parameters``. Used to update
            default parameter bounds obtained from ``mll``.
        closure: Callable that returns a tensor and an iterable of gradient
            tensors. Responsible for setting the ``grad`` attributes of
            ``parameters``. If no closure is provided, one will be obtained
            by calling ``get_loss_closure_with_grads``.
        closure_kwargs: Keyword arguments passed to ``closure``.
        method: Solver type, passed along to scipy.optimize.minimize.
        options: Dictionary of solver options, passed along to scipy.optimize.minimize.
        callback: Optional callback taking ``parameters`` and an
            ``OptimizationResult`` as its sole arguments.
        timeout_sec: Timeout in seconds after which to terminate the fitting loop
            (note that timing out can result in bad fits!).

    Returns:
        The final OptimizationResult.
    """
    # Resolve ``parameters`` and update default bounds
    _parameters, _bounds = get_parameters_and_bounds(mll)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}
    if parameters is None:
        parameters = {n: p for n, p in _parameters.items() if p.requires_grad}

    if closure is None:
        closure = get_loss_closure_with_grads(mll, parameters=parameters)

    if closure_kwargs is not None:
        closure = partial(closure, **closure_kwargs)

    result = scipy_minimize(
        closure=closure,
        parameters=parameters,
        bounds=bounds,
        method=method,
        options=options,
        callback=callback,
        timeout_sec=timeout_sec,
    )
    if result.status not in [OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED]:
        warn(
            f"`scipy_minimize` terminated with status {result.status}, displaying"
            f" original message from `scipy.optimize.minimize`: {result.message}",
            OptimizationWarning,
            stacklevel=2,
        )

    return result


def fit_gpytorch_mll_torch(
    mll: MarginalLogLikelihood,
    parameters: dict[str, Tensor] | None = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    step_limit: int | None = None,
    stopping_criterion: StoppingCriterion | None = DEFAULT,
    optimizer: Optimizer | Callable[..., Optimizer] = Adam,
    scheduler: _LRScheduler | Callable[..., _LRScheduler] | None = None,
    callback: Callable[[dict[str, Tensor], OptimizationResult], None] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic torch.optim-based fitting routine for GPyTorch MLLs.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized. Defaults
            to all parameters of ``mll`` that require gradients.
        bounds: A dictionary of user-specified bounds for ``parameters``. Used to update
            default parameter bounds obtained from ``mll``.
        closure: Callable that returns a tensor and an iterable of gradient
            tensors. Responsible for setting the ``grad`` attributes of
            ``parameters``. If no closure is provided, one will be obtained
            by calling ``get_loss_closure_with_grads``.
        closure_kwargs: Keyword arguments passed to ``closure``.
        step_limit: Optional upper bound on the number of optimization steps.
        stopping_criterion: A StoppingCriterion for the optimization loop.
        optimizer: A ``torch.optim.Optimizer`` instance or a factory that takes
            a list of parameters and returns an ``Optimizer`` instance.
        scheduler: A ``torch.optim.lr_scheduler._LRScheduler`` instance or a factory
            that takes an ``Optimizer`` instance and returns an ``_LRSchedule``.
        callback: Optional callback taking ``parameters`` and an
            OptimizationResult as its sole arguments.
        timeout_sec: Timeout in seconds after which to terminate the fitting loop
            (note that timing out can result in bad fits!).

    Returns:
        The final OptimizationResult.
    """
    if stopping_criterion == DEFAULT:
        stopping_criterion = ExpMAStoppingCriterion()

    # Resolve ``parameters`` and update default bounds
    param_dict, bounds_dict = get_parameters_and_bounds(mll)
    if parameters is None:
        parameters = {n: p for n, p in param_dict.items() if p.requires_grad}

    if closure is None:
        closure = get_loss_closure_with_grads(mll, parameters)

    if closure_kwargs is not None:
        closure = partial(closure, **closure_kwargs)

    return torch_minimize(
        closure=closure,
        parameters=parameters,
        bounds=bounds_dict if bounds is None else {**bounds_dict, **bounds},
        optimizer=optimizer,
        scheduler=scheduler,
        step_limit=step_limit,
        stopping_criterion=stopping_criterion,
        callback=callback,
        timeout_sec=timeout_sec,
    )


def fit_gpytorch_mll_scipy_independent(
    mll: MarginalLogLikelihood,
    parameters: dict[str, Tensor] | None = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Tensor], OptimizationResult], None] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Fit a batched model by independently optimizing each batch element's
    hyperparameters using parallel L-BFGS-B.

    For ``BatchedMultiOutputGPyTorchModel`` instances with a non-trivial
    ``_aug_batch_shape`` (e.g., multi-output ``SingleTaskGP`` or
    ``EnsembleMapSaasSingleTaskGP``), this runs ``fmin_l_bfgs_b_batched`` to
    optimize each batch element's hyperparameters independently. This converts
    the single high-dimensional optimization problem into multiple
    lower-dimensional problems that are easier to solve.

    For non-batched models, falls back to ``fit_gpytorch_mll_scipy``.

    The model and likelihood in mll must already be in train mode.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized. Defaults
            to all parameters of ``mll`` that require gradients.
        bounds: A dictionary of user-specified bounds for ``parameters``. Used
            to update default parameter bounds obtained from ``mll``.
        closure: Ignored. The batched closure is constructed internally.
            Accepted for API compatibility with ``fit_gpytorch_mll_scipy``.
        closure_kwargs: Ignored. Accepted for API compatibility.
        method: Ignored (always uses L-BFGS-B). Accepted for API compatibility.
        options: Dictionary of solver options passed to
            ``fmin_l_bfgs_b_batched`` (e.g., ``maxiter``, ``pgtol``).
        callback: Optional callback passed to ``fmin_l_bfgs_b_batched``.
        timeout_sec: Timeout in seconds. Not currently supported for batched
            fitting; a warning is issued if provided.

    Returns:
        The final OptimizationResult. The ``fval`` field contains the sum of
        per-batch-element negative MLL values.
    """
    from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel

    # Fall back to standard scipy fitting for non-batched models
    model = mll.model
    if (
        not isinstance(model, BatchedMultiOutputGPyTorchModel)
        or model._aug_batch_shape.numel() <= 1
    ):
        return fit_gpytorch_mll_scipy(
            mll=mll,
            parameters=parameters,
            bounds=bounds,
            closure=closure,
            closure_kwargs=closure_kwargs,
            method=method,
            options=options,
            callback=callback,
            timeout_sec=timeout_sec,
        )

    if timeout_sec is not None:
        warn(
            "timeout_sec is not supported for batched independent fitting "
            "and will be ignored.",
            OptimizationWarning,
            stacklevel=2,
        )

    from botorch.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched
    from botorch.optim.closures import (
        BatchedNDarrayOptimizationClosure,
        get_loss_closure,
    )
    from botorch.optim.utils.numpy_utils import get_per_element_bounds

    # Resolve parameters and bounds
    _parameters, _bounds = get_parameters_and_bounds(mll)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}
    if parameters is None:
        parameters = {n: p for n, p in _parameters.items() if p.requires_grad}

    batch_shape = model._aug_batch_shape

    # Build forward closure (returns per-batch neg MLL, NOT summed)
    forward = get_loss_closure(mll)

    # Build batched closure
    batched_closure = BatchedNDarrayOptimizationClosure(
        forward=forward,
        parameters=parameters,
        batch_shape=batch_shape,
    )

    # Extract per-element bounds
    bounds_np = get_per_element_bounds(parameters, bounds, batch_shape)

    # Get initial state
    x0 = batched_closure.state  # (batch_size, per_element_size)

    # Resolve options for fmin_l_bfgs_b_batched
    _recognized_options = {
        "gtol",
        "maxiter",
        "maxcor",
        "ftol",
        "pgtol",
        "maxls",
        "factr",
    }
    lbfgsb_options: dict[str, Any] = {}
    if options is not None:
        # Map scipy-style option names to fmin_l_bfgs_b_batched kwargs
        for key, value in options.items():
            if key == "gtol":
                lbfgsb_options["pgtol"] = value
            elif key in ("maxiter", "maxcor", "ftol", "pgtol", "maxls", "factr"):
                lbfgsb_options[key] = value
        unrecognized = set(options.keys()) - _recognized_options
        if unrecognized:
            warn(
                f"Unrecognized options for batched independent fitting "
                f"will be ignored: {sorted(unrecognized)}.",
                OptimizationWarning,
                stacklevel=2,
            )

    # Run batched L-BFGS-B
    xs, fs, results = fmin_l_bfgs_b_batched(
        func=batched_closure,
        x0=x0,
        bounds=bounds_np,
        pass_batch_indices=True,
        callback=callback,
        **lbfgsb_options,
    )

    # Write optimal state back to model parameters
    batched_closure.state = xs

    # Determine overall status from individual results
    all_success = all(r.get("success", False) for r in results)
    max_nit = max(r.get("nit", 0) for r in results)

    if all_success:
        status = OptimizationStatus.SUCCESS
    else:
        # Check if any hit maxiter
        any_maxiter = any(r.get("warnflag", 0) == 1 for r in results)
        status = (
            OptimizationStatus.STOPPED if any_maxiter else OptimizationStatus.FAILURE
        )

    return OptimizationResult(
        fval=float(fs.sum()),
        step=max_nit,
        status=status,
        message=(
            f"Batched L-BFGS-B: {sum(r.get('success', False) for r in results)}"
            f"/{len(results)} outputs converged."
        ),
    )
