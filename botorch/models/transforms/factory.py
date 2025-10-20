#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import OrderedDict

from botorch.models.transforms.input import (
    ChainedInputTransform,
    Normalize,
    OneHotToNumeric,
    Round,
)
from torch import Tensor


def get_rounding_input_transform(
    one_hot_bounds: Tensor,
    integer_indices: list[int] | None = None,
    categorical_features: dict[int, int] | None = None,
    initialization: bool = False,
    return_numeric: bool = False,
    approximate: bool = False,
) -> ChainedInputTransform:
    """Get a rounding input transform.

    The rounding function will take inputs from the unit cube,
    unnormalize the integers raw search space, round the inputs,
    and normalize them back to the unit cube.

    Categoricals are assumed to be one-hot encoded. Integers are
    currently assumed to be contiguous ranges (e.g. [1,2,3] and not
    [1,5,7]).

    TODO: support non-contiguous sets of integers by modifying
    the rounding function.

    Args:
        one_hot_bounds: The raw search space bounds where categoricals are
            encoded in one-hot representation and the integer parameters
            are not normalized.
        integer_indices: The indices of the integer parameters.
        categorical_features: A dictionary mapping indices to cardinalities
            for the categorical features.
        initialization: A boolean indicating whether this exact rounding
            function is for initialization. For initialization, the bounds
            for are expanded such that the end point of a range is selected
            with same probability that an interior point is selected, after
            rounding.
        return_numeric: A boolean indicating whether to return numeric or
            one-hot encoded categoricals. Returning a nummeric
            representation is helpful if the downstream code (e.g. kernel)
            expects a numeric representation of the categoricals.
        approximate: A boolean indicating whether to use an approximate
            rounding function.

    Returns:
        The rounding function ChainedInputTransform.
    """
    has_integers = integer_indices is not None and len(integer_indices) > 0
    has_categoricals = (
        categorical_features is not None and len(categorical_features) > 0
    )
    if not (has_integers or has_categoricals):
        raise ValueError(
            "A rounding function is a no-op "
            "if there are no integer or categorical parammeters."
        )
    if initialization and has_integers:
        # this gives the extreme integer values (end points)
        # the same probability as the interior values of the range
        init_one_hot_bounds = one_hot_bounds.clone()
        init_one_hot_bounds[0, integer_indices] -= 0.4999
        init_one_hot_bounds[1, integer_indices] += 0.4999
    else:
        init_one_hot_bounds = one_hot_bounds

    tfs = OrderedDict()
    if has_integers:
        # unnormalize to integer space
        tfs["unnormalize_tf"] = Normalize(
            d=init_one_hot_bounds.shape[1],
            bounds=init_one_hot_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=True,
            reverse=True,
        )
    # round
    tfs["round"] = Round(
        approximate=approximate,
        transform_on_train=False,
        transform_on_fantasize=True,
        integer_indices=integer_indices,
        categorical_features=categorical_features,
    )
    if has_integers:
        # renormalize to unit cube
        tfs["normalize_tf"] = Normalize(
            d=one_hot_bounds.shape[1],
            bounds=one_hot_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=True,
            reverse=False,
        )
    if return_numeric and has_categoricals:
        tfs["one_hot_to_numeric"] = OneHotToNumeric(
            # this is the dimension using one-hot encoded representation
            dim=one_hot_bounds.shape[-1],
            categorical_features=categorical_features,
            transform_on_train=True,
            transform_on_eval=True,
            transform_on_fantasize=True,
        )
    tf = ChainedInputTransform(**tfs)
    tf.to(dtype=one_hot_bounds.dtype, device=one_hot_bounds.device)
    tf.eval()
    return tf
