#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.models.utils.assorted import (
    _make_X_full,
    add_output_dim,
    check_min_max_scaling,
    check_no_nans,
    check_standardization,
    consolidate_duplicates,
    detect_duplicates,
    extract_targets_and_noise_single_output,
    fantasize,
    get_data_for_optimization_help,
    gpt_posterior_settings,
    mod_batch_shape,
    multioutput_to_batch_mode_transform,
    restore_targets_and_noise_single_output,
    validate_input_scaling,
)


__all__ = [
    "_make_X_full",
    "add_output_dim",
    "check_no_nans",
    "check_min_max_scaling",
    "check_standardization",
    "fantasize",
    "get_train_inputs",
    "get_train_targets",
    "get_data_for_optimization_help",
    "gpt_posterior_settings",
    "multioutput_to_batch_mode_transform",
    "mod_batch_shape",
    "validate_input_scaling",
    "detect_duplicates",
    "consolidate_duplicates",
    "extract_targets_and_noise_single_output",
    "restore_targets_and_noise_single_output",
]


# Lazy import to avoid circular dependencies
def __getattr__(name):
    if name == "get_train_inputs":
        from botorch.models.utils.helpers import get_train_inputs

        return get_train_inputs
    elif name == "get_train_targets":
        from botorch.models.utils.helpers import get_train_targets

        return get_train_targets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
