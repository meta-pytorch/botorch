# noqa
# flake8: noqa
"""
This is a port of the L-BFGS-B implementation from SciPy s.t. it supports batched
evaluations. That is, the objective function's output value (and its gradient)
can be evaluated at a batch of points at once.
This yields optimization speedups for acquisition function optimization,
where multiple independent problems with the same structure are optimized in parallel.

This file is written such that it explicitly supports all scipy versions
from 1.13 to 1.15 (likely 1.16, too, based on its pre-release version).
This file might break for higher versions, as it uses internal APIs.
There is a major revision of the core optimization code in 1.15, as it is
ported from FORTRAN to C, we handle the API changes, though, and are
compatible with both.
"""

## License for the Python wrapper
## ==============================

## Heavily modified to allow batched optimization by Samuel Müller (2025) <sammuller@meta.com>

## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.

## Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy

import typing as tp

import numpy as np

from numpy import array, asarray, zeros
from scipy.optimize import _lbfgsb
from scipy.optimize._constraints import old_bound_to_new
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct
from scipy.optimize._optimize import (
    _call_callback_maybe_halt,
    _check_unknown_options,
    _wrap_callback,
    OptimizeResult,
)

from .utils import check_scipy_version_at_least


__all__ = ["fmin_l_bfgs_b_batched"]


status_messages = {
    0: "START",
    1: "NEW_X",
    2: "RESTART",
    3: "FG",
    4: "CONVERGENCE",
    5: "STOP",
    6: "WARNING",
    7: "ERROR",
    8: "ABNORMAL",
}


task_messages = {
    0: "",
    301: "",
    302: "",
    401: "NORM OF PROJECTED GRADIENT <= PGTOL",
    402: "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    501: "CPU EXCEEDING THE TIME LIMIT",
    502: "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    503: "PROJECTED GRADIENT IS SUFFICIENTLY SMALL",
    504: "TOTAL NO. OF ITERATIONS REACHED LIMIT",
    505: "CALLBACK REQUESTED HALT",
    601: "ROUNDING ERRORS PREVENT PROGRESS",
    602: "STP = STPMAX",
    603: "STP = STPMIN",
    604: "XTOL TEST SATISFIED",
    701: "NO FEASIBLE SOLUTION",
    702: "FACTR < 0",
    703: "FTOL < 0",
    704: "GTOL < 0",
    705: "XTOL < 0",
    706: "STP < STPMIN",
    707: "STP > STPMAX",
    708: "STPMIN < 0",
    709: "STPMAX < STPMIN",
    710: "INITIAL G >= 0",
    711: "M <= 0",
    712: "N <= 0",
    713: "INVALID NBD",
}

uses_c_implementation = check_scipy_version_at_least(minor=15)


class OptimState:
    def __init__(
        self,
        bounds: list[tuple[float]] | None,
        maxls: int,
        x0: np.ndarray,
        n: int,
        m: int,
    ):
        standard_int = np.int32 if uses_c_implementation else _lbfgsb.types.intvar.dtype
        self.nbd = zeros(n, standard_int)
        self.low_bnd = zeros(n, np.float64)
        self.upper_bnd = zeros(n, np.float64)
        self.bounds_map = {
            (-np.inf, np.inf): 0,
            (1, np.inf): 1,
            (1, 1): 2,
            (-np.inf, 1): 3,
        }

        self.x = array(x0, np.float64)
        self.f = array(0.0, np.int32 if uses_c_implementation else np.float64)
        self.g = zeros((n,), np.int32 if uses_c_implementation else np.float64)
        self.wa = zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64)
        self.iwa = zeros(3 * n, standard_int)
        self.task = (
            zeros(2, dtype=np.int32) if uses_c_implementation else zeros(1, "S60")
        )
        self.csave = zeros(1, "S60")  # only used for fortran implementation
        self.ln_task = zeros(2, dtype=np.int32)  # only used for c implementation
        self.lsave = zeros(4, standard_int)
        self.isave = zeros(44, standard_int)
        self.dsave = zeros(29, np.float64)

        self.state_str = None

        if not uses_c_implementation:  # pragma: no cover
            self.task[:] = "START"

        self.n_iterations = 0
        self.fun_calls = 0

        if bounds is not None:
            for i in range(0, n):
                l, u = bounds[:, i]
                if not np.isinf(l):
                    self.low_bnd[i] = l
                    l = 1
                if not np.isinf(u):
                    self.upper_bnd[i] = u
                    u = 1
                self.nbd[i] = self.bounds_map[l, u]

        if not maxls > 0:
            raise ValueError("maxls must be positive.")


def fmin_l_bfgs_b_batched(
    func,
    x0,
    bounds=None,
    maxcor=10,
    factr=1e7,
    ftol=None,
    pgtol=1e-5,
    tol=None,
    maxiter=15000,
    disp=None,
    callback=None,
    maxls=20,
    pass_batch_indices=False,
):
    """
    Minimize multiple inputs to a batched function `func` using the L-BFGS-B algorithm.
    We minimize multiple inputs to the function at once (by providing a 2d array
    of shape [b, n]).
    We assume that the function `func` is batched, i.e. it will return a 1d array
    of shape [b,] of independent function values, when passed a 2d array of
    shape [b, n].

    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimize.
    x0 : ndarray
        Initial guess of shape [b, n].
    bounds : list, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None or +-inf for one of ``min`` or
        ``max`` when there is no bound in that direction.
    maxcor : int, optional
        The maximum number of variable metric corrections
        used to define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms in an
        approximation to it.)
    factr : float, optional
        The iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
        where ``eps`` is the machine precision, which is automatically
        generated by the code. Typical values for `factr` are: 1e12 for
        low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
        high accuracy. See Notes for relationship to `ftol`, which is exposed
        (instead of `factr`) by the `scipy.optimize.minimize` interface to
        L-BFGS-B.
    ftol: float, optional
        Set ftol directly, meaning the iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``
    pgtol : float, optional
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n} <= pgtol``
        where ``proj g_i`` is the i-th component of the projected gradient.
    tol : float, optional
        An alias for `pgtol` to be compatible with the `scipy.optimize.minimize`.
    maxiter : int, optional
        Maximum number of iterations.
    disp: int, optional
        This is depcreated and only here for backwards compatibility.
    callback : callable, optional
        Called after each iteration for each batch item, as ``callback(xk)``,
        where ``xk`` is the current parameter vector.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    pass_batch_indices : bool
        If True, fun is called with an additional kwargs `batch_indices`,
        which is a list that is as long as the current batch is wide,
        and indexes into the original batch specified via `x0`.

    Returns
    -------
    x : array_like
        Estimated position of the minimum.
    f : float
        Value of `func` at the minimum.
    d : dict
        Information dictionary.

        * d['warnflag'] is

          - 0 if converged,
          - 1 if too many function evaluations or too many iterations,
          - 2 if stopped for another reason, given in d['task']

        * d['grad'] is the gradient at the minimum (should be 0 ish)
        * d['funcalls'] is the number of function calls made.
        * d['nit'] is the number of iterations.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'L-BFGS-B' `method` in particular. Note that the
        `ftol` option is made available via that interface, while `factr` is
        provided via this interface, where `factr` is the factor multiplying
        the default machine floating-point precision to arrive at `ftol`:
        ``ftol = factr * numpy.finfo(float).eps``.

    Notes
    -----
    License of L-BFGS-B (FORTRAN code):

    The version included here (in fortran code) is 3.0
    (released April 25, 2011). It was written by Ciyou Zhu, Richard Byrd,
    and Jorge Nocedal <nocedal@ece.nwu.edu>. It carries the following
    condition for use:

    This software is freely available, but we expect that all publications
    describing work using this software, or all commercial products using it,
    quote at least one of the references given below. This software is released
    under the BSD License.

    SciPy uses a C-translated and modified version of the Fortran code,
    L-BFGS-B v3.0 (released April 25, 2011, BSD-3 licensed). Original Fortran
    version was written by Ciyou Zhu, Richard Byrd, Jorge Nocedal and,
    Jose Luis Morales.

    References
    ----------
    * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing, 16, 5, pp. 1190-1208.
    * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (2011),
      ACM Transactions on Mathematical Software, 38, 1.

    Examples
    --------
    Solve a linear regression problem via `fmin_l_bfgs_b`. To do this, first we define
    an objective function ``f(m, b) = (y - y_model)**2``, where `y` describes the
    observations and `y_model` the prediction of the linear model as
    ``y_model = m*x + b``. The bounds for the parameters, ``m`` and ``b``, are
    arbitrarily chosen as ``(0,5)`` and ``(5,10)`` for this example.

    >>> import numpy as np
    >>> from scipy.optimize import fmin_l_bfgs_b
    >>> X = np.arange(0, 10, 1)
    >>> M = 2
    >>> B = 3
    >>> Y = M * X + B
    >>> def func(parameters, *args):
    ...     x = args[0]
    ...     y = args[1]
    ...     m, b = parameters
    ...     y_model = m*x + b
    ...     error = sum(np.power((y - y_model), 2))
    ...     return error

    >>> initial_values = np.array([0.0, 1.0])

    >>> x_opt, f_opt, info = fmin_l_bfgs_b(func, x0=initial_values, args=(X, Y),
    ...                                    approx_grad=True)
    >>> x_opt, f_opt
    array([1.99999999, 3.00000006]), 1.7746231151323805e-14  # may vary

    The optimized parameters in ``x_opt`` agree with the ground truth parameters
    ``m`` and ``b``. Next, let us perform a bound contrained optimization using
    the `bounds` parameter.

    >>> bounds = [(0, 5), (5, 10)]
    >>> x_opt, f_op, info = fmin_l_bfgs_b(func, x0=initial_values, args=(X, Y),
    ...                                   approx_grad=True, bounds=bounds)
    >>> x_opt, f_opt
    array([1.65990508, 5.31649385]), 15.721334516453945  # may vary
    """
    if disp is not None:  # pragma: no cover
        print("The option `disp` is deprecated and will be removed in a future.")
    if ftol is None:
        ftol = factr * np.finfo(float).eps
    else:
        assert (
            factr is None
        ), "ftol and factr cannot be used together, set factr explicitly to None."

    # build options
    callback = _wrap_callback(callback)
    opts = {
        "maxcor": maxcor,
        "ftol": ftol,
        "gtol": tol if tol is not None else pgtol,
        "maxiter": maxiter,
        "callback": callback,
        "maxls": maxls,
        "pass_batch_indices": pass_batch_indices,
    }

    results = _minimize_lbfgsb(func, x0, bounds=bounds, **opts)
    fs = [res["fun"] for res in results]
    xs = [res["x"] for res in results]

    return np.stack(xs), np.stack(fs), results


def _minimize_lbfgsb(
    fun: tp.Callable[np.ndarray, tp.Tuple[np.ndarray, np.ndarray]],
    x0,
    bounds=None,
    maxcor=10,
    ftol=2.2204460492503131e-09,
    gtol=1e-5,
    maxiter=15000,
    callback=None,
    maxls=20,
    pass_batch_indices=False,
    **unknown_options,
):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.

    Options
    -------
    fun: callable[np.ndarray] -> tuple[np.ndarray, np.ndarray]
        accepts a batch of inputs [b,d] and returns a batch of outputs [b]
        and gradients [b,d] as a tuple
    bounds: list[tuple[float, float]], optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None or +-inf for one of ``min`` or
        ``max`` when there is no bound in that direction.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        The default is set to be a classical machine epsilon
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``proj g_i`` is the i-th component of the
        projected gradient.
    maxiter : int
        Maximum number of iterations.
    callback : callable, optional
        A "wrapped" callback function called after each iteration
        for each batch item, as ``callback(xk)``,
        where ``xk`` is the current parameter vector. It is called on the
        original xk, thus it should not change xk inplace.
        It can halt the algorithm by raising a StopIteration exception.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    pass_batch_indices : bool
        If True, fun is called with an additional kwargs `batch_indices`,
        which is a list that is as long as the current batch is wide,
        and indexes into the original batch specified via `x0`.

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.

    """

    _check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps

    x0s = asarray(x0)
    assert x0s.ndim == 2, "x0 must be a 2-D array"
    (b, n) = x0s.shape

    # historically old-style bounds were/are expected by lbfgsb.
    # That's still the case but we'll deal with new-style from here on,
    # it's easier
    if bounds is None:
        pass
    elif len(bounds) != n:
        raise ValueError(f"length of x0 != length of bounds, {n} != len({bounds})")
    else:
        bounds = np.array(old_bound_to_new(bounds))

        # check bounds
        if (bounds[0] > bounds[1]).any():
            raise ValueError(
                "LBFGSB - one of the lower bounds is greater than an upper bound."
            )

        # initial vector must lie within the bounds. Otherwise ScalarFunction and
        # approx_derivative will cause problems
        x0s = np.clip(x0s, bounds[0], bounds[1])

    # _prepare_scalar_function can use bounds=None to represent no bounds

    func_and_grad = fun

    states = [OptimState(bounds, maxls, x0, n, m) for x0 in x0s]
    dones = np.zeros(b, bool)
    do_forward = np.zeros(b, bool)

    while 1:
        # prep
        for i in range(b):
            while (
                ~dones[i] & ~do_forward[i]
            ):  # setulb sometimes needs to be called multiple times
                # until it needs new info or is done
                state = states[i]
                # g may become float32 if a user provides a function that calculates
                # the Jacobian in float32 (see gh-18730). The underlying Fortran/C code
                # expects float64, so upcast it
                state.g = state.g.astype(np.float64)
                # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
                if uses_c_implementation:
                    _lbfgsb.setulb(
                        m,
                        state.x,
                        state.low_bnd,
                        state.upper_bnd,
                        state.nbd,
                        state.f,
                        state.g,
                        factr,
                        pgtol,
                        state.wa,
                        state.iwa,
                        state.task,
                        state.lsave,
                        state.isave,
                        state.dsave,
                        maxls,
                        state.ln_task,
                    )
                else:  # pragma: no cover
                    _lbfgsb.setulb(
                        m,
                        state.x,
                        state.low_bnd,
                        state.upper_bnd,
                        state.nbd,
                        state.f,
                        state.g,
                        factr,
                        pgtol,
                        state.wa,
                        state.iwa,
                        state.task,
                        -1,  # iprint, default is -1 (not supported by the C impl)
                        state.csave,
                        state.lsave,
                        state.isave,
                        state.dsave,
                        maxls,
                    )

                if not uses_c_implementation:  # pragma: no cover
                    task_str = state.task.tobytes()

                if (
                    state.task[0] == 3
                    if uses_c_implementation
                    else task_str.startswith(b"FG")
                ):
                    # The minimization routine wants f and g at the current x.
                    # Note that interruptions due to maxfun are postponed
                    # until the completion of the current minimization iteration.
                    # Overwrite f and g:
                    # state.f, state.g = func_and_grad(
                    #     state.x
                    # )  # todo potentially use [:] assignment?
                    do_forward[i] = True
                elif (
                    state.task[0] == 1
                    if uses_c_implementation
                    else task_str.startswith(b"NEW_X")
                ):
                    # new iteration
                    state.n_iterations += 1

                    intermediate_result = OptimizeResult(x=state.x, fun=state.f)
                    if _call_callback_maybe_halt(callback, intermediate_result):
                        if uses_c_implementation:
                            state.task[0] = 5
                            state.task[1] = 505
                        else:  # pragma: no cover
                            state.task[:] = "STOP: CALLBACK REQUESTED HALT"
                    if state.n_iterations >= maxiter:
                        if uses_c_implementation:
                            state.task[0] = 5
                            state.task[1] = 504
                        else:  # pragma: no cover
                            state.task[:] = (
                                "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
                            )
                else:
                    dones[i] = True

        if np.any(do_forward):  # only the do_forward stuff is worked on
            total_x = np.stack(
                [state.x for state, do_fw in zip(states, do_forward) if do_fw]
            )
            if pass_batch_indices:
                batch_indices = [i for i, do_fw in enumerate(do_forward) if do_fw]
                total_f, total_g = func_and_grad(total_x, batch_indices=batch_indices)
            else:
                total_f, total_g = func_and_grad(total_x)

            for func_i, i in enumerate(
                do_forward.nonzero()[0]
            ):  # taking the 0 as we are interested in the first (and only) dim
                states[i].f = total_f[func_i]
                states[i].g = total_g[func_i]
                states[i].fun_calls += 1

            do_forward[:] = False

        if np.all(dones):
            break

    results = []

    for state in states:
        if not uses_c_implementation:  # pragma: no cover
            task_str = state.task.tobytes().strip(b"\x00").strip()
        if (
            state.task[0] == 4
            if uses_c_implementation
            else task_str.startswith(b"CONV")
        ):
            warnflag = 0
        elif state.n_iterations >= maxiter:
            warnflag = 1
        else:
            warnflag = 2

        # These two portions of the workspace are described in the mainlb
        # subroutine in lbfgsb.f (See line 363), if you are on an older
        # scipy version (< 1.14) and in the function docstring in "__lbfgsb.c",
        # ws and wy arguments, otherwise.
        s = state.wa[0 : m * n].reshape(m, n)
        y = state.wa[m * n : 2 * m * n].reshape(m, n)

        # See lbfgsb.f line 160 for this portion of the workspace.
        # isave(31) = the total number of BFGS updates prior the current iteration;
        n_bfgs_updates = state.isave[30]

        n_corrs = min(n_bfgs_updates, maxcor)
        hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

        if uses_c_implementation:
            msg = status_messages[state.task[0]] + ": " + task_messages[state.task[1]]
        else:  # pragma: no cover
            msg = task_str.decode()

        results.append(
            OptimizeResult(
                fun=state.f,
                jac=state.g,
                nfev=state.fun_calls,
                njev=None,
                nit=state.n_iterations,
                status=warnflag,
                message=msg,
                x=state.x,
                success=(warnflag == 0),
                hess_inv=hess_inv,
            )
        )
    return results


# extra helper function
def translate_bounds_for_lbfgsb(
    lower_bounds: tp.Union[tp.Sequence[tp.Optional[float]], float, None],
    upper_bounds: tp.Union[tp.Sequence[tp.Optional[float]], float, None],
    num_features: int,
    q: int,
):
    """
    Translates the bounds to the format expected by L-BFGS-B.

    Parameters
    ----------
    lower_bounds : tensor(n,) or float or None
        Lower bounds for the parameters. If None, then the lower bounds
        are unbounded. If float is provided, then that value is used as
        the lower bound for all parameters.
    upper_bounds : tensor(n,) or None
        Upper bounds for the parameters. If None, then the upper bounds
        are unbounded. If float is provided, then that value is used as
        the lower bound for all parameters.
    num_features : int
        Number of features in the model.
    q : int
        Number of repetitions to be optimized jointly in each item.
    fixed_features : dict, optional
        Dictionary mapping indices to the values that they are fixed to.
        These indices will not be included in the returned bounds.
    """
    bounds = [lower_bounds, upper_bounds]
    for i in range(2):
        if bounds[i] is None:
            bounds[i] = num_features * [None]
        elif not isinstance(bounds[i], tp.Iterable):
            bounds[i] = num_features * [bounds[i]]
        else:
            bounds[i] = list(bounds[i])
        if len(bounds[i]) == num_features:
            bounds[i] = sum([bounds[i] for _ in range(q)], [])
        assert (
            len(bounds[i]) == num_features * q
        ), f"Instead got {len(bounds[i])} != {num_features} * {q}."
    return list(zip(*bounds))
