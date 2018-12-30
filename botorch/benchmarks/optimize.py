#!/usr/bin/env python3

from time import time
from typing import Callable, List, NamedTuple, Optional, Tuple

import gpytorch
import torch
from botorch import fit_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

from ..acquisition.batch_modules import BatchAcquisitionFunction
from ..acquisition.utils import get_acquisition_function
from ..models.model import Model
from ..optim.random_restarts import random_restarts
from .output import BenchmarkOutput, ClosedLoopOutput


class OptimizeConfig(NamedTuple):
    """Config for closed loop optimization"""

    acquisition_function_name: str = "qEI"
    initial_points: int = 10
    q: int = 5
    n_batch: int = 10
    candidate_gen_max_iter: int = 25
    model_max_iter: int = 50
    num_starting_points: int = 1
    max_retries: int = 0  # number of retries, in the case of exceptions


def _get_fitted_model(
    train_X: Tensor, train_Y: Tensor, train_Y_se: Tensor, model: Model, max_iter: int
) -> Model:
    """
    Helper function that returns a model fitted to the provided data.

    Args:
        train_X: A `n x d` Tensor of points
        train_Y: A `n x (t)` Tensor of outcomes
        train_Y_se: A `n x (t)` Tensor of observed standard errors for each outcome
        model: an initialized Model. This model must have a likelihood attribute.
        max_iter: The maximum number of iterations
    Returns:
        Model: a fitted model
    """
    # TODO: copy over the state_dict from the existing model before
    # optimization begins
    model.reinitialize(train_X=train_X, train_Y=train_Y, train_Y_se=train_Y_se)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(dtype=train_X.dtype, device=train_X.device)
    mll = fit_model(mll, options={"maxiter": max_iter})
    return model


def greedy(
    X: Tensor,
    model: Model,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    mc_samples: int = 10000,
) -> Tuple[Tensor, float, float]:
    """
    Fetch the best point, best objective, and feasibility based on the joint
    posterior of the evaluated points.

    Args:
        X: q x d tensor of points
        model: model: A fitted model.
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        mc_samples: The number of Monte-Carlo samples to draw from the model
            posterior.
    Returns:
        Tensor: `1 x d` best point
        float: best objective
        float: feasibility of best point

    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(X)
        # mc_samples x b x q x (t)
        samples = posterior.rsample(sample_shape=torch.Size([mc_samples])).unsqueeze(1)
    # TODO: handle non-positive definite objectives
    obj = objective(samples).clamp_min_(0)  # pyre-ignore [16]
    obj_raw = objective(samples)
    feas_raw = torch.ones_like(obj_raw)
    if constraints is not None:
        for constraint in constraints:
            feas_raw.mul_((constraint(samples) < 0).type_as(obj))  # pyre-ignore [16]
        obj.mul_(feas_raw)
    _, best_idx = torch.max(obj.mean(dim=0), dim=-1)
    return (
        X[best_idx].view(-1, X.shape[-1]).detach(),
        obj_raw.mean(dim=0)[0, best_idx].item(),  # pyre-ignore [16]
        feas_raw.mean(dim=0)[0, best_idx].item(),  # pyre-ignore [16]
    )


def run_closed_loop(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    model: Model,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> ClosedLoopOutput:
    """
    Uses Bayesian Optimization to optimize func.

    Args:
        func: function to optimize (maximize by default)
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        config: configuration for the optimization
        model: an initialized Model. This model must have a likelihood attribute.
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        verbose: whether to provide verbose output
        seed: if seed is provided, do deterministic optimization where the function to
            optimize is fixed and not stochastic.
    Returns:
        ClosedLoopOutput: outputs from optimization

    # TODO: Add support for multi-task models.
    """
    # TODO: remove exception handling wrappers when model fitting is stabilized
    Xs = []
    Ys = []
    Ycovs = []
    best = []
    best_model_objective = []
    best_model_feasibility = []
    costs = []
    runtime = 0.0
    retry = 0
    refit_model = False

    best_point: Tensor
    obj: float
    feas: float
    train_X = None
    train_Y = None
    train_Y_se = None

    start_time = time()
    X = gen_function(config.initial_points)
    for iteration in range(config.n_batch):
        if verbose:
            print("Iteration:", iteration + 1)
        if iteration > 0:
            failed = True
            while retry <= config.max_retries and failed:
                try:
                    # If an exception occured during at evaluation time, then refit
                    # the model.
                    if refit_model:
                        model = _get_fitted_model(
                            train_X=train_X,
                            train_Y=train_Y,
                            train_Y_se=train_Y_se,
                            model=model,
                            max_iter=config.model_max_iter,
                        )
                        best_point, obj, feas = greedy(
                            X=train_X,
                            model=model,
                            objective=objective,
                            constraints=constraints,
                        )
                        best[-1] = best_point
                        best_model_objective[-1] = obj
                        best_model_feasibility[-1] = feas
                        costs[-1] = 1.0
                        refit_model = False
                    # type check for pyre
                    assert isinstance(model, Model)
                    acq_func: BatchAcquisitionFunction = get_acquisition_function(
                        acquisition_function_name=config.acquisition_function_name,
                        model=model,
                        X_observed=train_X,
                        objective=objective,
                        constraints=constraints,
                        X_pending=None,
                        seed=seed,
                    )
                    if verbose:
                        print("---- acquisition optimization")
                    candidates = random_restarts(
                        gen_function=gen_function,
                        acq_function=acq_func,
                        q=config.q,
                        num_starting_points=config.num_starting_points,
                        multiplier=100,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                        options={"maxiter": config.candidate_gen_max_iter},
                    )
                    X = acq_func.extract_candidates(candidates).detach()
                    failed = False
                    refit_model = False
                except Exception:
                    retry += 1
                    refit_model = True
                    if verbose:
                        print("---- Failed {} times ----".format(retry))
                    if retry > config.max_retries:
                        raise
        if verbose:
            print("---- evaluate")
        Y, Ycov = func(X)
        Xs.append(X)
        Ys.append(Y)
        Ycovs.append(Ycov)
        train_X = torch.cat(Xs, dim=0)
        train_Y = torch.cat(Ys, dim=0)
        train_Y_se = torch.cat(Ycovs, dim=0).sqrt()
        failed = True
        # handle errors in model fitting and model evaluation (when selecting best
        # point).
        while retry <= config.max_retries and failed:
            try:
                if verbose:
                    print("---- train")
                model = _get_fitted_model(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Y_se=train_Y_se,
                    model=model,
                    max_iter=config.model_max_iter,
                )
                if verbose:
                    print("---- identify")
                best_point, obj, feas = greedy(
                    X=train_X, model=model, objective=objective, constraints=constraints
                )
                failed = False
            except Exception:
                retry += 1
                if verbose:
                    print("---- Failed {} times ----".format(retry))
                if retry > config.max_retries:
                    raise
        best.append(best_point)
        best_model_objective.append(obj)
        best_model_feasibility.append(feas)
        costs.append(1.0)
    runtime = time() - start_time
    return ClosedLoopOutput(
        Xs=Xs,
        Ys=Ys,
        Ycovs=Ycovs,
        best=best,
        best_model_objective=best_model_objective,
        best_model_feasibility=best_model_feasibility,
        costs=costs,
        runtime=runtime,
    )


def run_benchmark(
    func: Callable[[Tensor], List[Tensor]],
    gen_function: Callable[[int], Tensor],
    config: OptimizeConfig,
    model: Model,
    objective: Callable[[Tensor], Tensor] = lambda Y: Y,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    lower_bounds: Optional[Tensor] = None,
    upper_bounds: Optional[Tensor] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
    num_runs: Optional[int] = 1,
    true_func: Optional[Callable[[Tensor], List[Tensor]]] = None,
    global_optimum: Optional[float] = None,
) -> BenchmarkOutput:
    """
    Uses Bayesian Optimization to optimize func multiple times.

    Args:
        func: function to optimize (maximize by default)
        gen_function: A function n -> X_cand producing n (typically random)
            feasible candidates as a `n x d` tensor X_cand
        config: configuration for the optimization
        model: an initialized Model. This model must have a likelihood attribute.
        objective: A callable mapping a Tensor of size `b x q x (t)`
            to a Tensor of size `b x q`, where `t` is the number of
            outputs (tasks) of the model. Note: the callable must support broadcasting.
            If omitted, use the identity map (applicable to single-task models only).
            Assumed to be non-negative when the constraints are used!
        constraints: A list of callables, each mapping a Tensor of size
            `b x q x t` to a Tensor of size `b x q`, where negative values imply
            feasibility. Note: the callable must support broadcasting. Only
            relevant for multi-task models (`t` > 1).
        lower_bounds: minimum values for each column of initial_candidates
        upper_bounds: maximum values for each column of initial_candidates
        verbose: whether to provide verbose output
        seed: if seed is provided, do deterministic optimization where the function to
            optimize is fixed and not stochastic. Note: this seed is incremented
            each run.
        num_runs: number of runs of bayesian optimization
        true_func: true noiseless function being optimized
        global_optimum: the global optimum of func after applying the objective
            transformation. If provided, this is used to compute regret.
    Returns:
        BenchmarkOutput: outputs from optimization
    """
    outputs = BenchmarkOutput(
        Xs=[],
        Ys=[],
        Ycovs=[],
        best=[],
        best_model_objective=[],
        best_model_feasibility=[],
        costs=[],
        runtime=[],
        best_true_objective=[],
        best_true_feasibility=[],
        regrets=[],
        cumulative_regrets=[],
        weights=[],
    )
    for run in range(num_runs):
        run_output = run_closed_loop(
            func=func,
            gen_function=gen_function,
            config=config,
            model=model,
            objective=objective,
            constraints=constraints,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            verbose=verbose,
            seed=seed + run if seed is not None else seed,  # increment seed each run
        )
        if verbose:
            print("---- Finished {} loops ----".format(run + 1))
        # compute true objective base on best point (greedy from model)
        best = torch.cat(run_output.best, dim=0)
        if true_func is not None:
            f_best = true_func(best)[0]
            best_true_objective = objective(f_best).view(-1)
            best_true_feasibility = torch.ones_like(best_true_objective)
            if constraints is not None:
                for constraint in constraints:
                    best_true_feasibility.mul_(  # pyre-ignore [16]
                        (constraint(f_best) < 0)  # pyre-ignore [16]
                        .type_as(best)
                        .view(-1)
                    )
            outputs.best_true_objective.append(best_true_objective)
            outputs.best_true_feasibility.append(best_true_feasibility)
            if global_optimum is not None:
                regrets = torch.tensor(
                    [
                        (global_optimum - objective(true_func(X)[0]))  # pyre-ignore [6]
                        .sum()
                        .item()
                        for X in run_output.Xs
                    ]
                ).type_as(best)
                # check that objective is never > than global_optimum
                assert torch.all(regrets >= 0)
                # compute regret on best point (greedy from model)
                cumulative_regrets = torch.cumsum(regrets, dim=0)
                outputs.regrets.append(regrets)
                outputs.cumulative_regrets.append(cumulative_regrets)
        for f in run_output._fields:
            getattr(outputs, f).append(getattr(run_output, f))
    return outputs
