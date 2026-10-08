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

# Options of the L-BFGS-B solver that are supported by ``fmin_l_bfgs_b_batched``,
# mapped to the name of the corresponding ``fmin_l_bfgs_b_batched`` argument.
_BATCHED_LBFGSB_OPTIONS = {
    "maxiter": "maxiter",
    "maxcor": "maxcor",
    "maxls": "maxls",
    "ftol": "ftol",
    "factr": "factr",
    "gtol": "pgtol",
    "pgtol": "pgtol",
}


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

    For ``BatchedMultiOutputGPyTorchModel`` instances with a non-trivial
    ``_aug_batch_shape`` (e.g., multi-output ``SingleTaskGP`` or
    ``EnsembleMapSaasSingleTaskGP``) whose parameters are all batched along
    ``_aug_batch_shape``, this automatically runs ``fmin_l_bfgs_b_batched``
    to optimize each batch element's hyperparameters independently. This
    converts the single high-dimensional optimization problem into multiple
    lower-dimensional problems that are easier to solve. Batched independent
    fitting is not used if any parameter is shared across batch elements
    (e.g., the parameters of a learnable input transform such as ``Warp``),
    or if a ``closure``, ``closure_kwargs``, ``callback``, ``timeout_sec``,
    a ``method`` other than L-BFGS-B, or ``options`` that are not supported
    by ``fmin_l_bfgs_b_batched`` are provided.

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
        options: Dictionary of solver options, passed along to
            scipy.optimize.minimize or ``fmin_l_bfgs_b_batched``.
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

    if (
        closure is None
        and closure_kwargs is None
        and callback is None
        and timeout_sec is None
        and method == "L-BFGS-B"
        and set(options or {}).issubset(_BATCHED_LBFGSB_OPTIONS)
        and _is_batch_independent(model=mll.model, parameters=parameters)
    ):
        result = _fit_gpytorch_mll_scipy_independent(
            mll=mll, parameters=parameters, bounds=bounds, options=options
        )
    else:
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


def _is_batch_independent(model: Module, parameters: dict[str, Tensor]) -> bool:
    r"""Check whether the MLL optimization decomposes into independent problems.

    This is the case if ``model`` is a ``BatchedMultiOutputGPyTorchModel`` with a
    non-trivial ``_aug_batch_shape`` and every parameter in ``parameters`` is
    batched along ``_aug_batch_shape``, so that each batch element of the loss
    only depends on the corresponding slice of each parameter. Parameters of
    the model's input transform (e.g., of a learnable ``Warp`` transform) are
    considered shared across batch elements, even if their shape happens to
    match ``_aug_batch_shape``.

    Args:
        model: The model whose hyperparameters are to be fit.
        parameters: The dictionary of parameters to be optimized.

    Returns:
        True if the parameters of different batch elements can be optimized
        independently.
    """
    # Avoid circular import: models.gpytorch imports from optim.
    from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel

    if not isinstance(model, BatchedMultiOutputGPyTorchModel):
        return False
    batch_shape = model._aug_batch_shape
    if batch_shape.numel() <= 1:
        return False
    input_transform = getattr(model, "input_transform", None)
    shared_ids = (
        set()
        if input_transform is None
        else {id(p) for p in input_transform.parameters()}
    )
    return all(
        p.shape[: len(batch_shape)] == batch_shape and id(p) not in shared_ids
        for p in parameters.values()
    )


def _fit_gpytorch_mll_scipy_independent(
    mll: MarginalLogLikelihood,
    parameters: dict[str, Tensor],
    bounds: dict[str, tuple[float | None, float | None]],
    options: dict[str, Any] | None = None,
) -> OptimizationResult:
    r"""Fit a batched model by independently optimizing each batch element's
    hyperparameters using parallel L-BFGS-B.

    This is an internal helper called by ``fit_gpytorch_mll_scipy`` when the
    optimization problem decomposes into independent problems per batch element
    (see ``_is_batch_independent``).

    Args:
        mll: MarginalLogLikelihood to be maximized.
        parameters: Dictionary of parameters to be optimized, each of which is
            batched along ``mll.model._aug_batch_shape``.
        bounds: A dictionary of bounds for ``parameters``.
        options: Dictionary of L-BFGS-B options (see ``_BATCHED_LBFGSB_OPTIONS``
            for the supported options), e.g., ``maxiter`` or ``gtol``.

    Returns:
        The final OptimizationResult. The ``fval`` field contains the sum of
        per-batch-element negative MLL values.
    """
    # Avoid circular imports: closures and batched_lbfgs_b import from optim.
    from botorch.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched
    from botorch.optim.closures import (
        BatchedNDarrayOptimizationClosure,
        get_loss_closure,
    )
    from botorch.optim.utils.numpy_utils import get_per_element_bounds

    batch_shape = mll.model._aug_batch_shape

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

    # Map scipy-style option names to fmin_l_bfgs_b_batched kwargs
    lbfgsb_options = {
        _BATCHED_LBFGSB_OPTIONS[key]: value for key, value in (options or {}).items()
    }
    if "ftol" in lbfgsb_options:
        # ``fmin_l_bfgs_b_batched`` does not accept ``ftol`` together with ``factr``
        # (which has a non-trivial default value), and ``ftol`` takes precedence.
        lbfgsb_options["factr"] = None

    # Run batched L-BFGS-B
    xs, fs, results = fmin_l_bfgs_b_batched(
        func=batched_closure,
        x0=x0,
        bounds=bounds_np,
        pass_batch_indices=True,
        **lbfgsb_options,
    )

    # Write optimal state back to model parameters
    batched_closure.state = xs

    # Determine overall status from individual results. The ``status`` of each
    # result is 0 if converged, 1 if the iteration limit was reached, and 2 if the
    # optimization terminated for another (abnormal) reason.
    max_nit = max(r.get("nit", 0) for r in results)
    statuses = [r["status"] for r in results]
    if any(s not in (0, 1) for s in statuses):
        status = OptimizationStatus.FAILURE
    elif any(s == 1 for s in statuses):
        status = OptimizationStatus.STOPPED
    else:
        status = OptimizationStatus.SUCCESS

    return OptimizationResult(
        fval=float(fs.sum()),
        step=max_nit,
        status=status,
        message=(
            f"Batched L-BFGS-B: {sum(r.get('success', False) for r in results)}"
            f"/{len(results)} outputs converged."
        ),
    )
