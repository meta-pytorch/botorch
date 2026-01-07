#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Utilities for fitting and manipulating models."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from re import Pattern
from typing import Any, NamedTuple
from warnings import warn

import torch
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


def get_data_loader(
    model: GPyTorchModel, batch_size: int = 1024, **kwargs: Any
) -> DataLoader:
    dataset = TensorDataset(*model.train_inputs, model.train_targets)
    return DataLoader(
        dataset=dataset, batch_size=min(batch_size, len(model.train_targets)), **kwargs
    )


def get_parameters(
    module: Module,
    requires_grad: bool | None = None,
    name_filter: Callable[[str], bool] | None = None,
) -> dict[str, Tensor]:
    r"""Helper method for obtaining a module's parameters and their respective ranges.

    Args:
        module: The target module from which parameters are to be extracted.
        requires_grad: Optional Boolean used to filter parameters based on whether
            or not their require_grad attribute matches the user provided value.
        name_filter: Optional Boolean function used to filter parameters by name.

    Returns:
        A dictionary of parameters.
    """
    parameters = {}
    for name, param in module.named_parameters():
        if requires_grad is not None and param.requires_grad != requires_grad:
            continue

        if name_filter and not name_filter(name):
            continue

        parameters[name] = param

    return parameters


def get_parameters_and_bounds(
    module: Module,
    requires_grad: bool | None = None,
    name_filter: Callable[[str], bool] | None = None,
    default_bounds: tuple[float, float] = (-float("inf"), float("inf")),
) -> tuple[dict[str, Tensor], dict[str, tuple[float | None, float | None]]]:
    r"""Helper method for obtaining a module's parameters and their respective ranges.

    Args:
        module: The target module from which parameters are to be extracted.
        name_filter: Optional Boolean function used to filter parameters by name.
        requires_grad: Optional Boolean used to filter parameters based on whether
            or not their require_grad attribute matches the user provided value.
        default_bounds: Default lower and upper bounds for constrained parameters
            with `None` typed bounds.

    Returns:
        A dictionary of parameters and a dictionary of parameter bounds.
    """
    if hasattr(module, "named_parameters_and_constraints"):
        bounds = {}
        params = {}
        for name, param, constraint in module.named_parameters_and_constraints():
            if (requires_grad is None or (param.requires_grad == requires_grad)) and (
                name_filter is None or name_filter(name)
            ):
                params[name] = param
                if constraint is None:
                    continue

                bounds[name] = tuple(
                    default if bound is None else constraint.inverse_transform(bound)
                    for (bound, default) in zip(constraint, default_bounds)
                )

        return params, bounds

    params = get_parameters(
        module, requires_grad=requires_grad, name_filter=name_filter
    )
    return params, {}


def get_name_filter(
    patterns: Iterator[Pattern | str],
) -> Callable[[str | tuple[str, Any, ...]], bool]:
    r"""Returns a binary function that filters strings (or iterables whose first
    element is a string) according to a bank of excluded patterns. Typically, used
    in conjunction with generators such as `module.named_parameters()`.

    Args:
        patterns: A collection of regular expressions or strings that
            define the set of names to be excluded.

    Returns:
        A binary function indicating whether or not an item should be filtered.
    """
    names = set()
    _patterns = set()
    for pattern in patterns:
        if isinstance(pattern, str):
            names.add(pattern)
        elif isinstance(pattern, Pattern):
            _patterns.add(pattern)
        else:
            raise TypeError(
                "Expected `patterns` to contain `str` or `re.Pattern` typed elements, "
                f"but found {type(pattern)}."
            )

    def name_filter(item: str | tuple[str, Any, ...]) -> bool:
        name = item if isinstance(item, str) else next(iter(item))
        if name in names:
            return False

        for pattern in _patterns:
            if pattern.search(name):
                return False

        return True

    return name_filter


def sample_all_priors(model: GPyTorchModel, max_retries: int = 100) -> None:
    r"""Sample from hyperparameter priors (in-place).

    Args:
        model: A GPyTorchModel.
    """
    for _, module, prior, closure, setting_closure in model.named_priors():
        if setting_closure is None:
            raise RuntimeError(
                "Must provide inverse transform to be able to sample from prior."
            )
        for i in range(max_retries):
            try:
                # Set sample shape, so that the prior samples have the same shape
                # as `closure(module)` without having to be repeated.
                prior_shape = prior._extended_shape()
                if prior_shape.numel() == 1:
                    # For a univariate prior we can sample the size of the closure.
                    # Otherwise we will sample exactly the same value for all
                    # lengthscales where we commonly specify a univariate prior.
                    setting_closure(module, prior.sample(closure(module).shape))
                else:
                    closure_shape = closure(module).shape
                    sample_shape = closure_shape[: -len(prior_shape)]
                    setting_closure(module, prior.sample(sample_shape=sample_shape))
                break
            except NotImplementedError:
                warn(
                    f"`rsample` not implemented for {type(prior)}. Skipping.",
                    BotorchWarning,
                )
                break
            except RuntimeError as e:
                if "out of bounds of its current constraints" in str(e):
                    if i == max_retries - 1:
                        raise RuntimeError(
                            "Failed to sample a feasible parameter value "
                            f"from the prior after {max_retries} attempts."
                        )
                else:
                    raise e
