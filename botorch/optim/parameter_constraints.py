#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utility functions for constrained optimization.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from botorch.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch.optim.utils import columnwise_clamp
from scipy.optimize import Bounds, minimize
from torch import Tensor


ScipyConstraintDict = dict[
    str, Union[str, Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]
]


def get_constraint_tolerance(dtype: torch.dtype) -> float:
    r"""Get the constraint tolerance for a given dtype.

    Args:
        dtype: The dtype to use.

    Returns:
        The constraint tolerance for the given dtype.
    """
    if dtype == torch.double:
        return 1e-8
    elif dtype == torch.float:
        return 1e-6
    elif dtype == torch.half:
        return 1e-4
    raise ValueError(f"Unsupported dtype {dtype}.")


def make_scipy_bounds(
    X: Tensor,
    lower_bounds: float | Tensor | None = None,
    upper_bounds: float | Tensor | None = None,
) -> Bounds | None:
    r"""Creates a scipy Bounds object for optimization

    Args:
        X: `... x d` tensor
        lower_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.

    Returns:
        A scipy `Bounds` object if either lower_bounds or upper_bounds is not
        None, and None otherwise.

    Example:
        >>> X = torch.rand(5, 2)
        >>> scipy_bounds = make_scipy_bounds(X, 0.1, 0.8)
    """
    if lower_bounds is None and upper_bounds is None:
        return None

    def _expand(bounds: float | Tensor, X: Tensor, lower: bool) -> Tensor:
        if bounds is None:
            ebounds = torch.full_like(X, float("-inf" if lower else "inf"))
        else:
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
        return _arrayify(ebounds).flatten()

    lb = _expand(bounds=lower_bounds, X=X, lower=True)
    ub = _expand(bounds=upper_bounds, X=X, lower=False)
    return Bounds(lb=lb, ub=ub, keep_feasible=True)


def make_scipy_linear_constraints(
    shapeX: torch.Size,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> list[ScipyConstraintDict]:
    r"""Generate scipy constraints from torch representation.

    Args:
        shapeX: The shape of the torch.Tensor to optimize over (i.e. `(b) x q x d`)
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`, where
            `indices` is a single-dimensional index tensor (long dtype) containing
            indices into the last dimension of `X`, `coefficients` is a
            single-dimensional tensor of coefficients of the same length, and
            rhs is a scalar.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) == rhs` (with `indices`
            and `coefficients` of the same form as in `inequality_constraints`).

    Returns:
        A list of dictionaries containing callables for constraint function
        values and Jacobians and a string indicating the associated constraint
        type ("eq", "ineq"), as expected by `scipy.optimize.minimize`.

    This function assumes that constraints are the same for each input batch,
    and broadcasts the constraints accordingly to the input batch shape. This
    function does support constraints across elements of a q-batch if the
    indices are a 2-d Tensor.

    Example:
        The following will enforce that `x[1] + 0.5 x[3] >= -0.1` for each `x`
        in both elements of the q-batch, and each of the 3 t-batches:

        >>> constraints = make_scipy_linear_constraints(
        >>>     torch.Size([3, 2, 4]),
        >>>     [(torch.tensor([1, 3]), torch.tensor([1.0, 0.5]), -0.1)],
        >>> )

        The following will enforce that `x[0, 1] + 0.5 x[1, 3] >= -0.1` where
        x[0, :] is the first element of the q-batch and x[1, :] is the second
        element of the q-batch, for each of the 3 t-batches:

        >>> constraints = make_scipy_linear_constraints(
        >>>     torch.size([3, 2, 4])
        >>>     [(torch.tensor([[0, 1], [1, 3]), torch.tensor([1.0, 0.5]), -0.1)],
        >>> )
    """
    constraints = []
    if inequality_constraints is not None:
        for indcs, coeffs, rhs in inequality_constraints:
            constraints += _make_linear_constraints(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX, eq=False
            )
    if equality_constraints is not None:
        for indcs, coeffs, rhs in equality_constraints:
            constraints += _make_linear_constraints(
                indices=indcs, coefficients=coeffs, rhs=rhs, shapeX=shapeX, eq=True
            )
    return constraints


def eval_lin_constraint(
    x: npt.NDArray, flat_idxr: list[int], coeffs: npt.NDArray, rhs: float
) -> np.float64:
    r"""Evaluate a single linear constraint.

    Args:
        x: The input array.
        flat_idxr: The indices in `x` to consider.
        coeffs: The coefficients corresponding to the indices.
        rhs: The right-hand-side of the constraint.

    Returns:
        The evaluted constraint: `\sum_i (coeffs[i] * x[i]) - rhs`
    """
    return np.sum(x[flat_idxr] * coeffs, -1) - rhs


def lin_constraint_jac(
    x: npt.NDArray, flat_idxr: list[int], coeffs: npt.NDArray, n: int
) -> npt.NDArray:
    r"""Return the Jacobian associated with a linear constraint.

    Args:
        x: The input array.
        flat_idxr: The indices for the elements of x that appear in the constraint.
        coeffs: The coefficients corresponding to the indices.
        n: number of elements

    Returns:
        The Jacobian.
    """
    # TODO: Use sparse representation (not sure if scipy optim supports that)
    jac = np.zeros(n)
    jac[flat_idxr] = coeffs
    return jac


def _arrayify(X: Tensor) -> npt.NDArray:
    r"""Convert a torch.Tensor (any dtype or device) to a numpy (double) array.

    Args:
        X: The input tensor.

    Returns:
        A numpy array of double dtype with the same shape and data as `X`.
    """
    return X.cpu().detach().contiguous().double().clone().numpy()


def _validate_linear_constraints_shape_input(shapeX: torch.Size) -> torch.Size:
    """
    Validate `shapeX` input to `_make_linear_constraints`.

    Check that it has either 2 or 3 dimensions, and add a scalar batch
    dimension if it is only 2d.
    """

    if len(shapeX) not in (2, 3):
        raise UnsupportedError(
            f"`shapeX` must be `(b) x q x d` (at least two-dimensional). It is "
            f"{shapeX}."
        )
    if len(shapeX) == 2:
        shapeX = torch.Size([1, *shapeX])
    return shapeX


def _validate_linear_constraints_indices_input(indices: Tensor, q: int, d: int) -> None:
    if indices.dim() > 2:
        raise UnsupportedError(
            "Linear constraints supported only on individual candidates and "
            "across q-batches, not across general batch shapes."
        )
    elif indices.dim() == 2:
        if indices[:, 0].max() > q - 1:
            raise RuntimeError(f"Index out of bounds for {q}-batch")
        if indices[:, 1].max() > d - 1:
            raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
    elif indices.dim() == 1:
        if indices.max() > d - 1:
            raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
    else:
        raise ValueError("`indices` must be at least one-dimensional")


def _make_linear_constraints(
    indices: Tensor,
    coefficients: Tensor,
    rhs: float,
    shapeX: torch.Size,
    eq: bool = False,
) -> list[ScipyConstraintDict]:
    r"""Create linear constraints to be used by `scipy.optimize.minimize`.

    Encodes constraints of the form
    `\sum_i (coefficients[i] * X[..., indices[i]]) ? rhs`
    where `?` can be designated either as `>=` by setting `eq=False`, or as
    `=` by setting `eq=True`.

    If indices is one-dimensional, the constraints are broadcasted across
    all elements of the q-batch. If indices is two-dimensional, then
    constraints are applied across elements of a q-batch. In either case,
    constraints are created for all t-batches.

    Args:
        indices: A tensor of shape `c` or `c x 2`, where c is the number of terms
            in the constraint. If single-dimensional, contains the indices of
            the dimensions of the feature space that occur in the linear
            constraint. If two-dimensional, contains pairs of indices of the
            q-batch (0) and the feature space (1) that occur in the linear
            constraint.
        coefficients: A single-dimensional tensor of coefficients with the same
            number of elements as `indices`.
        rhs: The right hand side of the constraint.
        shapeX: The shape of the torch tensor to construct the constraints for
            (i.e. `(b) x q x d`). Must have two or three dimensions.
        eq: If True, return an equality constraint, o/w return an inequality
            constraint (indicated by "eq" / "ineq" value of the `type` key).

    Returns:
        A list of constraint dictionaries with the following keys

        - "type": Indicates the type of the constraint ("eq" if `eq=True`, "ineq" o/w)
        - "fun": A callable evaluating the constraint value on `x`, a flattened
            version of the input tensor `X`, returning a scalar.
        - "jac": A callable evaluating the constraint's Jacobian on `x`, a flattened
            version of the input tensor `X`, returning a numpy array.

    >>> shapeX = torch.Size([3, 5, 4])
    >>> constraints = _make_linear_constraints(
    ...     indices=torch.tensor([1., 2.]),
    ...     coefficients=torch.tensor([-0.5, 1.3]),
    ...     rhs=0.49,
    ...     shapeX=shapeX,
    ...     eq=True
    ... )
    >>> len(constraints)
    15
    >>> constraints[0].keys()
    dict_keys(['type', 'fun', 'jac'])
    >>> x = np.arange(60).reshape(shapeX)
    >>> constraints[0]["fun"](x)
    1.61  # 1 * -0.5 + 2 * 1.3 - 0.49
    >>> constraints[0]["jac"](x)
    [0., -0.5, 1.3, 0., 0., ...]
    >>> constraints[1]["fun"](x)  #
    4.81
    """

    shapeX = _validate_linear_constraints_shape_input(shapeX)

    b, q, d = shapeX
    _validate_linear_constraints_indices_input(indices, q, d)
    n = shapeX.numel()
    constraints: list[ScipyConstraintDict] = []
    coeffs = _arrayify(coefficients)
    ctype = "eq" if eq else "ineq"

    offsets = [q * d, d]
    if indices.dim() == 2:
        # indices has two dimensions (potential constraints across q-batch elements)
        # rule is [i, j, k] is at
        # i * offsets[0] + j * offsets[1] + k
        for i in range(b):
            list_ind = (idx.tolist() for idx in indices)
            idxr = [i * offsets[0] + idx[0] * offsets[1] + idx[1] for idx in list_ind]
            fun = partial(
                eval_lin_constraint, flat_idxr=idxr, coeffs=coeffs, rhs=float(rhs)
            )
            jac = partial(lin_constraint_jac, flat_idxr=idxr, coeffs=coeffs, n=n)
            constraints.append({"type": ctype, "fun": fun, "jac": jac})
    elif indices.dim() == 1:
        # indices is one-dim - broadcast constraints across q-batches and t-batches
        for i in range(b):
            for j in range(q):
                idxr = (i * offsets[0] + j * offsets[1] + indices).tolist()
                fun = partial(
                    eval_lin_constraint, flat_idxr=idxr, coeffs=coeffs, rhs=float(rhs)
                )
                jac = partial(lin_constraint_jac, flat_idxr=idxr, coeffs=coeffs, n=n)
                constraints.append({"type": ctype, "fun": fun, "jac": jac})
    return constraints


def _make_nonlinear_constraints(
    f_np_wrapper: Callable, nlc: Callable, is_intrapoint: bool, shapeX: torch.Size
) -> list[ScipyConstraintDict]:
    """Create nonlinear constraints to be used by `scipy.optimize.minimize`.

    Args:
        f_np_wrapper: A wrapper function that given a constraint evaluates
            the value and gradient (using autograd) of a numpy input and returns both
            the objective and the gradient.
        nlc: Callable representing a constraint of the form `callable(x) >= 0`. In case
            of an intra-point constraint, `callable()`takes in an one-dimensional tensor
            of shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar.
        is_intrapoint: A Boolean indicating if a constraint is an intra-point or
            inter-point constraint (see the docstring of the `inequality_constraints`
            argument to `optimize_acqf()`).
        shapeX: Shape of the three-dimensional batch X, that should be optimized.

    Returns:
        A list of constraint dictionaries with the following keys

        - "type": Indicates the type of the constraint, here always "ineq".
        - "fun": A callable evaluating the constraint value on `x`, a flattened
            version of the input tensor `X`, returning a scalar.
        - "jac": A callable evaluating the constraint's Jacobian on `x`, a flattened
            version of the input tensor `X`, returning a numpy array.
    """
    shapeX = _validate_linear_constraints_shape_input(shapeX)
    b, q, _ = shapeX
    constraints = []

    def get_intrapoint_constraint(b: int, q: int, nlc: Callable) -> Callable:
        return lambda x: nlc(x[b, q])

    def get_interpoint_constraint(b: int, nlc: Callable) -> Callable:
        return lambda x: nlc(x[b])

    if is_intrapoint:
        for i in range(b):
            for j in range(q):
                f_obj, f_grad = _make_f_and_grad_nonlinear_inequality_constraints(
                    f_np_wrapper=f_np_wrapper,
                    nlc=get_intrapoint_constraint(b=i, q=j, nlc=nlc),
                )
                constraints.append({"type": "ineq", "fun": f_obj, "jac": f_grad})
    else:
        for i in range(b):
            f_obj, f_grad = _make_f_and_grad_nonlinear_inequality_constraints(
                f_np_wrapper=f_np_wrapper,
                nlc=get_interpoint_constraint(b=i, nlc=nlc),
            )
            constraints.append({"type": "ineq", "fun": f_obj, "jac": f_grad})

    return constraints


def _generate_unfixed_nonlin_constraints(
    constraints: list[tuple[Callable[[Tensor], Tensor], bool]] | None,
    fixed_features: dict[int, float],
    dimension: int,
) -> list[Callable[[Tensor], Tensor]] | None:
    """Given a dictionary of fixed features, returns a list of callables for
    nonlinear inequality constraints expecting only a tensor with the non-fixed
    features as input.
    """
    if not constraints:
        return constraints

    selector = []
    idx_X, idx_f = 0, dimension - len(fixed_features)
    for i in range(dimension):
        if i in fixed_features.keys():
            selector.append(idx_f)
            idx_f += 1
        else:
            selector.append(idx_X)
            idx_X += 1

    values = torch.tensor(list(fixed_features.values()), dtype=torch.double)

    def _wrap_nonlin_constraint(
        constraint: Callable[[Tensor], Tensor],
    ) -> Callable[[Tensor], Tensor]:
        def new_nonlin_constraint(X: Tensor) -> Tensor:
            ivalues = values.to(X).expand(*X.shape[:-1], len(fixed_features))
            X_perm = torch.cat([X, ivalues], dim=-1)
            return constraint(X_perm[..., selector])

        return new_nonlin_constraint

    return [
        (_wrap_nonlin_constraint(constraint=nlc), is_intrapoint)
        for nlc, is_intrapoint in constraints
    ]


def _generate_unfixed_lin_constraints(
    constraints: list[tuple[Tensor, Tensor, float]] | None,
    fixed_features: dict[int, float],
    dimension: int,
    eq: bool,
) -> list[tuple[Tensor, Tensor, float]] | None:
    # If constraints is None or an empty list, then return itself
    if not constraints:
        return constraints

    # replace_index generates the new indices for the unfixed dimensions
    # after eliminating the fixed dimensions.
    # Example: dimension = 5, ff.keys() = [1, 3], replace_index = {0: 0, 2: 1, 4: 2}
    unfixed_keys = sorted(set(range(dimension)) - set(fixed_features))
    unfixed_keys = torch.tensor(unfixed_keys).to(constraints[0][0])
    replace_index = torch.arange(dimension - len(fixed_features)).to(constraints[0][0])

    new_constraints = []
    # parse constraints one-by-one
    for constraint_id, (indices, coefficients, rhs) in enumerate(constraints):
        new_rhs = rhs
        new_indices = []
        new_coefficients = []
        # the following unsqueeze is done to facilitate a simpler for-loop.
        indices_2dim = indices if indices.ndim == 2 else indices.unsqueeze(-1)
        for coefficient, index in zip(coefficients, indices_2dim):
            ffval_or_None = fixed_features.get(index[-1].item())
            # if ffval_or_None is None, then the index is not fixed
            if ffval_or_None is None:
                new_indices.append(index)
                new_coefficients.append(coefficient)
            # otherwise, we "remove" the constraints corresponding to that index
            else:
                new_rhs = new_rhs - coefficient.item() * ffval_or_None

        # all indices were fixed, so the constraint is gone.
        if len(new_indices) == 0:
            if (eq and new_rhs != 0) or (not eq and new_rhs > 0):
                prefix = "Eq" if eq else "Ineq"
                raise CandidateGenerationError(
                    f"{prefix}uality constraint {constraint_id} not met "
                    "with fixed_features."
                )
        else:
            # However, one key transformation has to be noted.
            # new_indices is with respect to the older (fuller) domain, and so it will
            # have to be converted using replace_index.
            new_indices = torch.stack(new_indices, dim=0)
            # generate new index location after the removal of fixed_features indices
            new_indices_dim_d = new_indices[:, -1].unsqueeze(-1)
            new_indices_dim_d = replace_index[
                torch.nonzero(new_indices_dim_d == unfixed_keys, as_tuple=True)[1]
            ]
            new_indices[:, -1] = new_indices_dim_d
            # squeeze(-1) is a no-op if dim -1 is not singleton
            new_indices.squeeze_(-1)
            # convert new_coefficients to Tensor
            new_coefficients = torch.stack(new_coefficients)
            new_constraints.append((new_indices, new_coefficients, new_rhs))
    return new_constraints


def _make_f_and_grad_nonlinear_inequality_constraints(
    f_np_wrapper: Callable, nlc: Callable
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Create callables for objective + grad for the nonlinear inequality constraints.
    The Scipy interface requires specifying separate callables and we use caching to
    avoid evaluating the same input twice. This caching only works if
    the returned functions are evaluated on the same input in immediate
    sequence (i.e., calling `f_obj(X_1)`, `f_grad(X_1)` will result in a
    single forward pass, while `f_obj(X_1)`, `f_grad(X_2)`, `f_obj(X_1)`
    will result in three forward passes).
    """

    def f_obj_and_grad(x):
        obj, grad = f_np_wrapper(x, f=nlc)
        return obj, grad

    cache = {"X": None, "obj": None, "grad": None}

    def f_obj(X):
        X_c = cache["X"]
        if X_c is None or not np.array_equal(X_c, X):
            cache["X"] = X.copy()
            cache["obj"], cache["grad"] = f_obj_and_grad(X)
        return cache["obj"]

    def f_grad(X):
        X_c = cache["X"]
        if X_c is None or not np.array_equal(X_c, X):
            cache["X"] = X.copy()
            cache["obj"], cache["grad"] = f_obj_and_grad(X)
        return cache["grad"]

    return f_obj, f_grad


def nonlinear_constraint_is_feasible(
    nonlinear_inequality_constraint: Callable,
    is_intrapoint: bool,
    x: Tensor,
    tolerance: float | None = None,
) -> Tensor:
    """Checks if a nonlinear inequality constraint is fulfilled (within tolerance).

    Args:
        nonlinear_inequality_constraint: Callable to evaluate the
            constraint.
        intra: If True, the constraint is an intra-point constraint that
            is applied pointwise and is broadcasted over the q-batch. Else, the
            constraint has to evaluated over the whole q-batch and is a an
            inter-point constraint.
        x: Tensor of shape (batch x q x d).
        tolerance: Rather than using the exact `const(x) >= 0` constraint, this helper
            checks feasibility of `const(x) >= -tolerance`. This avoids marking the
            candidates as infeasible due to tiny violations.

    Returns:
        A boolean tensor of shape (batch) indicating if the constraint is
        satified by the corresponding batch of `x`.
    """
    if tolerance is None:
        tolerance = get_constraint_tolerance(dtype=x.dtype)

    def check_x(x: Tensor) -> bool:
        return _arrayify(nonlinear_inequality_constraint(x)).item() >= -tolerance

    x_flat = x.view(-1, *x.shape[-2:])
    is_feasible = torch.ones(x_flat.shape[0], dtype=torch.bool, device=x.device)
    for i, x_ in enumerate(x_flat):
        if is_intrapoint:
            is_feasible[i] &= all(check_x(x__) for x__ in x_)
        else:
            is_feasible[i] &= check_x(x_)
    return is_feasible.view(x.shape[:-2])


def make_scipy_nonlinear_inequality_constraints(
    nonlinear_inequality_constraints: list[tuple[Callable, bool]],
    f_np_wrapper: Callable,
    x0: Tensor,
    shapeX: torch.Size,
) -> list[dict]:
    r"""Generate Scipy nonlinear inequality constraints from callables.

    Args:
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
        f_np_wrapper: A wrapper function that given a constraint evaluates the value
             and gradient (using autograd) of a numpy input and returns both the
             objective and the gradient.
        x0: The starting point for SLSQP. We return this starting point in (rare)
            cases where SLSQP fails and thus require it to be feasible.
        shapeX: Shape of the three-dimensional batch X, that should be optimized.

    Returns:
        A list of dictionaries containing callables for constraint function
        values and Jacobians and a string indicating the associated constraint
        type ("eq", "ineq"), as expected by `scipy.optimize.minimize`.
    """

    scipy_nonlinear_inequality_constraints = []
    for constraint in nonlinear_inequality_constraints:
        if not isinstance(constraint, tuple):
            raise ValueError(
                f"A nonlinear constraint has to be a tuple, got {type(constraint)}."
            )
        if len(constraint) != 2:
            raise ValueError(
                "A nonlinear constraint has to be a tuple of length 2, "
                f"got length {len(constraint)}."
            )
        nlc, is_intrapoint = constraint
        if not nonlinear_constraint_is_feasible(
            nlc, is_intrapoint=is_intrapoint, x=x0.reshape(shapeX)
        ).all():
            raise ValueError(
                "`batch_initial_conditions` must satisfy the non-linear inequality "
                "constraints."
            )

        scipy_nonlinear_inequality_constraints += _make_nonlinear_constraints(
            f_np_wrapper=f_np_wrapper,
            nlc=nlc,
            is_intrapoint=is_intrapoint,
            shapeX=shapeX,
        )
    return scipy_nonlinear_inequality_constraints


def evaluate_feasibility(
    X: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
    tolerance: float | None = None,
) -> Tensor:
    r"""Evaluate feasibility of candidate points (within a tolerance).

    Args:
        X: The candidate tensor of shape `batch x q x d`.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example. When q=1, or when
            applying the same constraint to each candidate in the batch
            (intra-point constraint), `indices` should be a 1-d tensor.
            For inter-point constraints, in which the constraint is applied to the
            whole batch of candidates, `indices` must be a 2-d tensor, where
            in each row `indices[i] =(k_i, l_i)` the first index `k_i` corresponds
            to the `k_i`-th element of the `q`-batch and the second index `l_i`
            corresponds to the `l_i`-th feature of that element.
        equality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an equality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`. See the docstring of
            `make_scipy_linear_constraints` for an example.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument.
        tolerance: The tolerance used to check the feasibility of constraints.
            For inequality constraints, we check if `const(X) >= rhs - tolerance`.
            For equality constraints, we check if `abs(const(X) - rhs) < tolerance`.
            For non-linear inequality constraints, we check if `const(X) >= -tolerance`.
            This avoids marking the candidates as infeasible due to tiny violations.

    Returns:
        A boolean tensor of shape `batch` indicating if the corresponding candidate of
        shape `q x d` is feasible.
    """
    if tolerance is None:
        tolerance = get_constraint_tolerance(dtype=X.dtype)

    is_feasible = torch.ones(X.shape[:-2], device=X.device, dtype=torch.bool)
    if inequality_constraints is not None:
        for idx, coef, rhs in inequality_constraints:
            if idx.ndim == 1:
                # Intra-point constraints.
                is_feasible &= (
                    (X[..., idx] * coef).sum(dim=-1) >= rhs - tolerance
                ).all(dim=-1)
            else:
                # Inter-point constraints.
                is_feasible &= (X[..., idx[:, 0], idx[:, 1]] * coef).sum(
                    dim=-1
                ) >= rhs - tolerance
    if equality_constraints is not None:
        for idx, coef, rhs in equality_constraints:
            if idx.ndim == 1:
                # Intra-point constraints.
                is_feasible &= (
                    ((X[..., idx] * coef).sum(dim=-1) - rhs).abs() < tolerance
                ).all(dim=-1)
            else:
                # Inter-point constraints.
                is_feasible &= (
                    (X[..., idx[:, 0], idx[:, 1]] * coef).sum(dim=-1) - rhs
                ).abs() < tolerance
    if nonlinear_inequality_constraints is not None:
        for const, intra in nonlinear_inequality_constraints:
            is_feasible &= nonlinear_constraint_is_feasible(
                nonlinear_inequality_constraint=const,
                is_intrapoint=intra,
                x=X,
                tolerance=tolerance,
            )
    return is_feasible


def project_to_feasible_space_via_slsqp(
    X: Tensor,
    bounds: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> Tensor:
    """Project X onto the feasible space by solving a quadratic program.

    This uses SLSQP with gradients to solve the quadratic program.
    NOTE: A proper specialized QP solver would be a better choice here,
    but we'd like to avoid adding dependency on additional packages.
    SLSQP should be able to solve this reliably and quickly since the
    dimension is typically low and the number of constraints is typically
    limited.

    Args:
        X: A `(batch_shape x) n x d`-dim tensor of inptus.
        bounds: A `2 x d`-dim tensor of lower and upper bounds.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and
            `coefficients` should be torch tensors. See the docstring of
            `make_scipy_linear_constraints` for an example.
        equality_constraints: A list of tuples (indices, coefficients, rhs).

    Returns:
        A `(batch_shape x) n x d`-dim tensor of  projected values.
    """
    if inequality_constraints is None and equality_constraints is None:
        return X
    bounds_scipy = make_scipy_bounds(
        X=X, lower_bounds=bounds[0], upper_bounds=bounds[1]
    )
    constraints = make_scipy_linear_constraints(
        shapeX=X.shape,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )
    # Define squared distance objective
    X_np = X.flatten().detach().cpu().numpy()

    def objective(x: np.ndarray):
        return 0.5 * np.sum((x - X_np) ** 2)

    def grad_objective(x: np.ndarray):
        return x - X_np

    x0 = (
        columnwise_clamp(X=X, lower=bounds[0], upper=bounds[1], raise_on_violation=True)
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    # NOTE: A proper specialized QP solver would be a better choice here,
    # but we'd like to avoid adding dependency on additional packages.
    # SLSQP should be able to solve this reliably and quickly since the
    # dimension is typically low and the number of constraints is typically
    # limited.
    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        jac=grad_objective,
        bounds=bounds_scipy,
        constraints=constraints,
        tol=get_constraint_tolerance(dtype=X.dtype),
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return torch.from_numpy(result.x).to(X).view(X.shape)
