#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Assorted helper methods and objects for working with BoTorch models."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack

import torch
from botorch import settings
from botorch.exceptions import InputDataError, InputDataWarning
from botorch.settings import _Flag
from gpytorch import settings as gpt_settings
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.module import Module
from torch import Tensor


def _make_X_full(X: Tensor, output_indices: list[int], tf: int) -> Tensor:
    r"""Helper to construct input tensor with task indices.

    Args:
        X: The raw input tensor (without task information).
        output_indices: The output indices to generate (passed in via ``posterior``).
        tf: The task feature index.

    Returns:
        Tensor: The full input tensor for the multi-task model, including task
            indices.
    """
    index_shape = X.shape[:-1] + torch.Size([1])
    indexers = (
        torch.full(index_shape, fill_value=i, device=X.device, dtype=X.dtype)
        for i in output_indices
    )
    X_l, X_r = X[..., :tf], X[..., tf:]
    return torch.cat(
        [torch.cat([X_l, indexer, X_r], dim=-1) for indexer in indexers], dim=-2
    )


def multioutput_to_batch_mode_transform(
    train_X: Tensor,
    train_Y: Tensor,
    num_outputs: int,
    train_Yvar: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    r"""Transforms training inputs for a multi-output model.

    Used for multi-output models that internally are represented by a
    batched single output model, where each output is modeled as an
    independent batch.

    Args:
        train_X: A ``n x d`` or ``input_batch_shape x n x d`` (batch mode) tensor of
            training features.
        train_Y: A ``n x m`` or ``target_batch_shape x n x m`` (batch mode) tensor of
            training observations.
        num_outputs: number of outputs
        train_Yvar: A ``n x m`` or ``target_batch_shape x n x m`` tensor of observed
            measurement noise.

    Returns:
        3-element tuple containing

        - A ``input_batch_shape x m x n x d`` tensor of training features.
        - A ``target_batch_shape x m x n`` tensor of training observations.
        - A ``target_batch_shape x m x n`` tensor observed measurement noise.
    """
    # make train_Y ``batch_shape x m x n``
    train_Y = train_Y.transpose(-1, -2)
    # expand train_X to ``batch_shape x m x n x d``
    train_X = train_X.unsqueeze(-3).expand(
        train_X.shape[:-2] + torch.Size([num_outputs]) + train_X.shape[-2:]
    )
    if train_Yvar is not None:
        # make train_Yvar ``batch_shape x m x n``
        train_Yvar = train_Yvar.transpose(-1, -2)
    return train_X, train_Y, train_Yvar


def add_output_dim(X: Tensor, original_batch_shape: torch.Size) -> tuple[Tensor, int]:
    r"""Insert the output dimension at the correct location.

    The trailing batch dimensions of X must match the original batch dimensions
    of the training inputs, but can also include extra batch dimensions.

    Args:
        X: A ``(new_batch_shape) x (original_batch_shape) x n x d`` tensor of
            features.
        original_batch_shape: the batch shape of the model's training inputs.

    Returns:
        2-element tuple containing

        - A ``(new_batch_shape) x (original_batch_shape) x m x n x d`` tensor of
            features.
        - The index corresponding to the output dimension.
    """
    X_batch_shape = X.shape[:-2]
    if len(X_batch_shape) > 0 and len(original_batch_shape) > 0:
        # check that X_batch_shape supports broadcasting or augments
        # original_batch_shape with extra batch dims
        try:
            torch.broadcast_shapes(X_batch_shape, original_batch_shape)
        except RuntimeError:
            raise RuntimeError(
                "The trailing batch dimensions of X must match the trailing "
                f"batch dimensions of the training inputs. Got {X.shape=} "
                f"and {original_batch_shape=}."
            )
    # insert ``m`` dimension
    X = X.unsqueeze(-3)
    output_dim_idx = max(len(original_batch_shape), len(X_batch_shape))
    return X, output_dim_idx


def check_no_nans(Z: Tensor) -> None:
    r"""Check that tensor does not contain NaN values.

    Raises an InputDataError if ``Z`` contains NaN values.

    Args:
        Z: The input tensor.
    """
    if torch.any(torch.isnan(Z)).item():
        raise InputDataError("Input data contains NaN values.")


def check_min_max_scaling(
    X: Tensor,
    strict: bool = False,
    atol: float = 1e-2,
    raise_on_fail: bool = False,
    ignore_dims: list[int] | None = None,
) -> None:
    r"""Check that tensor is normalized to the unit cube.

    Args:
        X: A ``batch_shape x n x d`` input tensor. Typically the training inputs
            of a model.
        strict: If True, require ``X`` to be scaled to the unit cube (rather than
            just to be contained within the unit cube).
        atol: The tolerance for the boundary check. Only used if ``strict=True``.
        raise_on_fail: If True, raise an exception instead of a warning.
        ignore_dims: Subset of dimensions where the min-max scaling check is omitted.
    """
    ignore_dims = ignore_dims or []
    check_dims = list(set(range(X.shape[-1])) - set(ignore_dims))
    if len(check_dims) == 0:
        return None

    with torch.no_grad():
        X_check = X[..., check_dims]
        Xmin = torch.min(X_check, dim=-1).values
        Xmax = torch.max(X_check, dim=-1).values
        msg = None
        if strict and max(torch.abs(Xmin).max(), torch.abs(Xmax - 1).max()) > atol:
            msg = "scaled"
        if torch.any(Xmin < -atol) or torch.any(Xmax > 1 + atol):
            msg = "contained"
        if msg is not None:
            # NOTE: If you update this message, update the warning filters as well.
            # See https://github.com/meta-pytorch/botorch/pull/2508.
            msg = (
                f"Data (input features) is not {msg} to the unit cube. "
                "Please consider min-max scaling the input data."
            )
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning, stacklevel=2)


def check_standardization(
    Y: Tensor,
    atol_mean: float = 1e-2,
    atol_std: float = 1e-2,
    raise_on_fail: bool = False,
) -> None:
    r"""Check that tensor is standardized (zero mean, unit variance).

    Args:
        Y: The input tensor of shape ``batch_shape x n x m``. Typically the
            train targets of a model. Standardization is checked across the
            ``n``-dimension.
        atol_mean: The tolerance for the mean check.
        atol_std: The tolerance for the std check.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Ymean = torch.mean(Y, dim=-2)
        mean_not_zero = torch.abs(Ymean).max() > atol_mean
        if Y.shape[-2] <= 1:
            if mean_not_zero:
                # NOTE: If you update this message, update the warning filters as well.
                # See https://github.com/meta-pytorch/botorch/pull/2508.
                msg = (
                    f"Data (outcome observations) is not standardized (mean = {Ymean})."
                    " Please consider scaling the input to zero mean and unit variance."
                )
                if raise_on_fail:
                    raise InputDataError(msg)
                warnings.warn(msg, InputDataWarning, stacklevel=2)
        else:
            Ystd = torch.std(Y, dim=-2)
            std_not_one = torch.abs(Ystd - 1).max() > atol_std
            if mean_not_zero or std_not_one:
                # NOTE: If you update this message, update the warning filters as well.
                # See https://github.com/meta-pytorch/botorch/pull/2508.
                msg = (
                    "Data (outcome observations) is not standardized "
                    f"(std = {Ystd}, mean = {Ymean})."
                    "Please consider scaling the input to zero mean and unit variance."
                )
                if raise_on_fail:
                    raise InputDataError(msg)
                warnings.warn(msg, InputDataWarning, stacklevel=2)


def validate_input_scaling(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    raise_on_fail: bool = False,
    ignore_X_dims: list[int] | None = None,
    check_nans_only: bool = False,
) -> None:
    r"""Helper function to validate input data to models.

    Args:
        train_X: A ``n x d`` or ``batch_shape x n x d`` (batch mode) tensor of
            training features.
        train_Y: A ``n x m`` or ``batch_shape x n x m`` (batch mode) tensor of
            training observations.
        train_Yvar: A ``batch_shape x n x m`` or ``batch_shape x n x m`` (batch mode)
            tensor of observed measurement noise.
        raise_on_fail: If True, raise an error instead of emitting a warning
            (only for normalization/standardization checks, an error is always
            raised if NaN values are present).
        ignore_X_dims: For this subset of dimensions from ``{1, ..., d}``, ignore the
            min-max scaling check.
        check_nans_only: If True, only check for NaN values. Skips min-max scaling
            and standardization checks. This is used when the model is provided
            with a custom covariance module, to avoid potentially irrelevant
            warnings.

    This function is typically called inside the constructor of standard BoTorch
    models. It validates the following:
    (i) none of the inputs contain NaN values
    (ii) the training data (``train_X``) is normalized to the unit cube for all
    dimensions except those in ``ignore_X_dims``.
    (iii) the training targets (``train_Y``) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (``train_Yvar``) at this point.
    """
    if settings.validate_input_scaling.off():
        return
    check_no_nans(train_X)
    check_no_nans(train_Y)
    if train_Yvar is not None:
        check_no_nans(train_Yvar)
        if torch.any(train_Yvar < 0):
            raise InputDataError("Input data contains negative variances.")
    if not check_nans_only:
        check_min_max_scaling(
            X=train_X, raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
        )
        check_standardization(Y=train_Y, raise_on_fail=raise_on_fail)


def mod_batch_shape(module: Module, names: list[str], b: int) -> None:
    r"""Recursive helper to modify gpytorch modules' batch shape attribute.

    Modifies the module in-place.

    Args:
        module: The module to be modified.
        names: The list of names to access the attribute. If the full name of
            the module is ``"module.sub_module.leaf_module"``, this will be
            ``["sub_module", "leaf_module"]``.
        b: The new size of the last element of the module's ``batch_shape``
            attribute.
    """
    if len(names) == 0:
        return
    m = getattr(module, names[0])
    if len(names) == 1 and hasattr(m, "batch_shape") and len(m.batch_shape) > 0:
        m.batch_shape = m.batch_shape[:-1] + torch.Size([b] if b > 0 else [])
    else:
        mod_batch_shape(module=m, names=names[1:], b=b)


@contextmanager
def gpt_posterior_settings():
    r"""Context manager for settings used for computing model posteriors."""
    with ExitStack() as es:
        if gpt_settings.debug.is_default():
            es.enter_context(gpt_settings.debug(False))
        if gpt_settings.fast_pred_var.is_default():
            es.enter_context(gpt_settings.fast_pred_var())
        es.enter_context(
            gpt_settings.detach_test_caches(settings.propagate_grads.off())
        )
        yield


def detect_duplicates(
    X: Tensor,
    rtol: float = 0,
    atol: float = 1e-8,
) -> Iterator[tuple[int, int]]:
    """Returns an iterator over index pairs ``(duplicate index, original index)``
    for all duplicate entries of ``X``. Supporting 2-d Tensor only.

    Args:
        X: the datapoints tensor with potential duplicated entries
        rtol: relative tolerance
        atol: absolute tolerance
    """
    if len(X.shape) != 2:
        raise ValueError("X must have 2 dimensions.")

    tols = atol
    if rtol:
        rval = X.abs().max(dim=-1, keepdim=True).values
        tols = tols + rtol * rval.max(rval.transpose(-1, -2))

    n = X.shape[-2]
    dist = torch.full((n, n), float("inf"), device=X.device, dtype=X.dtype)
    dist[torch.triu_indices(n, n, offset=1).unbind()] = torch.nn.functional.pdist(
        X, p=float("inf")
    )
    return (
        (i, int(j))
        # pyre-fixme[19]: Expected 1 positional argument.
        for diff, j, i in zip(*(dist - tols).min(dim=-2), range(n))
        if diff < 0
    )


def consolidate_duplicates(
    X: Tensor, Y: Tensor, rtol: float = 0.0, atol: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop duplicated Xs and update the indices tensor Y accordingly.
    Supporting 2d Tensor only as in batch mode block design is not guaranteed.

    Args:
        X: the datapoints tensor
        Y: the index tensor to be updated (e.g., pairwise comparisons)
        rtol: relative tolerance
        atol: absolute tolerance

    Returns:
        consolidated_X: the consolidated X
        consolidated_Y: the consolidated Y (e.g., pairwise comparisons indices)
        new_indices: new index of each original item in X, a tensor of size X.shape[-2]
    """
    if len(X.shape) != 2:
        raise ValueError("X must have 2 dimensions.")

    n = X.shape[-2]
    dup_map = dict(detect_duplicates(X=X, rtol=rtol, atol=atol))

    # Handle edge cases conservatively
    # If a item is in both dup set and kept set, do not remove it
    common_set = set(dup_map.keys()).intersection(dup_map.values())
    for k in list(dup_map.keys()):
        if k in common_set or dup_map[k] in common_set:
            del dup_map[k]

    if dup_map:
        dup_indices, kept_indices = zip(*dup_map.items())

        unique_indices = sorted(set(range(n)) - set(dup_indices))

        # After dropping the duplicates,
        # the kept ones' indices may also change by being shifted up
        new_idx_map = dict(zip(unique_indices, range(len(unique_indices))))
        new_indices_for_dup = (new_idx_map[idx] for idx in kept_indices)
        new_idx_map.update(dict(zip(dup_indices, new_indices_for_dup)))
        consolidated_X = X[list(unique_indices), :]
        consolidated_Y = torch.tensor(
            [[new_idx_map[item.item()] for item in row] for row in Y.unbind()],
            dtype=torch.long,
            device=Y.device,
        )
        new_indices = (
            torch.arange(n, dtype=torch.long)
            .apply_(lambda x: new_idx_map[x])
            .to(Y.device)
        )
        return consolidated_X, consolidated_Y, new_indices
    else:
        return X, Y, torch.arange(n, device=Y.device, dtype=Y.dtype)


class fantasize(_Flag):
    r"""A flag denoting whether we are currently in a ``fantasize`` context."""

    _state: bool = False


def get_task_value_remapping(
    observed_task_values: Tensor,
    all_task_values: Tensor,
    dtype: torch.dtype,
    default_task_value: int | None,
) -> Tensor | None:
    """Construct an mapping of observed task values to contiguous int-valued floats.

    Args:
        observed_task_values: A sorted long-valued tensor of task values.
        all_task_values: A sorted long-valued tensor of task values.
        dtype: The dtype of the model inputs (e.g. ``X``), which the new
            task values should have mapped to (e.g. float, double).
        default_task_value: The default task value to use for missing task values.

    Returns:
        A tensor of shape ``task_values.max() + 1`` that maps task values
        to new task values. The indexing operation ``mapper[task_value]``
        will produce a tensor of new task values, of the same shape as
        the original. The elements of the ``mapper`` tensor that do not
        appear in the original ``task_values`` are mapped to ``nan``. The
        return value will be ``None``, when the task values are contiguous
        integers starting from zero.
    """
    if dtype not in (torch.float, torch.double):
        raise ValueError(f"dtype must be torch.float or torch.double, but got {dtype}.")
    task_range = torch.arange(
        len(observed_task_values),
        dtype=all_task_values.dtype,
        device=all_task_values.device,
    )
    mapper = None

    if default_task_value is None:
        fill_value = float("nan")
    else:
        mask = observed_task_values == default_task_value
        if not mask.any():
            fill_value = float("nan")
        else:
            idx = mask.nonzero().item()
            fill_value = task_range[idx]
    # if not all tasks are observed or they are not contiguous integers
    # then map them to contiguous integers
    if not torch.equal(task_range, all_task_values):
        # Create a tensor that maps task values to new task values.
        # The number of tasks should be small, so this should be quite efficient.
        mapper = torch.full(
            (int(all_task_values.max().item()) + 1,),
            fill_value,
            dtype=dtype,
            device=all_task_values.device,
        )
        mapper[observed_task_values] = task_range.to(dtype=dtype)
    return mapper


def extract_targets_and_noise_single_output(model) -> tuple[Tensor, Tensor | None]:
    r"""Extract targets and noise variance for single-output models (m=1).

    Args:
        model: A GPyTorch model.

    Returns:
        A tuple of (Y, Yvar) where Y and Yvar have shape ``batch_shape x n x 1``.
    """
    Y = model.train_targets.unsqueeze(-1)
    Yvar = None
    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        Yvar = model.likelihood.noise_covar.noise.unsqueeze(-1)
    return Y, Yvar


def restore_targets_and_noise_single_output(
    model, Y: Tensor, Yvar: Tensor | None, strict: bool
) -> None:
    r"""Restore targets and noise variance for single-output models (m=1).

    Args:
        model: A GPyTorch model.
        Y: Targets tensor in shape ``batch_shape x n x 1``.
        Yvar: Optional noise variance tensor in shape ``batch_shape x n x 1``.
        strict: Whether to strictly enforce shape constraints.
    """
    Y = Y.squeeze(-1)
    if Yvar is not None and isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
        Yvar = Yvar.squeeze(-1)
        model.likelihood.noise_covar.noise = Yvar
    model.set_train_data(targets=Y, strict=strict)
