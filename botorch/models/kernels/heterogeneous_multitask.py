#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

r"""
Kernels for multi-task GPs with heterogeneous search spaces.
"""

import torch
from botorch.models.map_saas import add_saas_prior
from botorch.utils.constraints import LogTransformedInterval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from torch import Tensor
from torch.nn import ModuleList


LOG_OUTPUTSCALE_CONSTRAINT = LogTransformedInterval(1e-2, 1e4, initial_value=10)


class DeltaKernel(Kernel):
    r"""A kernel that evaluates `x1 == x2 == 1`."""

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        r"""Evaluate the kernel.

        Args:
            x1: A tensor of shape `batch_shape x q1 x 1`.
            x2: A tensor of shape `batch_shape x q2 x 1`.

        Returns:
            A tensor of shape `batch_shape x q1 x q2` containing 1 where
            x1 == x2 == 1 and 0 otherwise.
        """
        assert x1.shape[-1] == x2.shape[-1] == 1, "DeltaKernel expects 1D inputs!"
        x2_ = x2.transpose(-2, -1)
        return torch.where((x1 == x2_) & (x2_ == 1), 1.0, 0.0).to(x1)


class CombinatorialCovarModule(ScaleKernel):
    r"""A kernel suitable for a {0, 1}^d domain and used for combinatorial design."""

    def __init__(self, ard_num_dims: int | None = None) -> None:
        r"""Initialize the kernel.

        Args:
            ard_num_dims: The number of feature dimensions for ARD.
        """
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            lengthscale_constraint=None,
            lengthscale_prior=None,
        )
        if ard_num_dims is not None and ard_num_dims > 1:
            add_saas_prior(base_kernel)

        super().__init__(
            base_kernel=base_kernel,
            outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT,
        )


class MultiTaskConditionalKernel(Kernel):
    r"""A base kernel for multi-task GPs with heterogeneous search
    spaces for tasks.

    This kernel was introduced in [Deshwal2024Heterogeneous]_.

    * This kernel conditionally combines multiple sub-kernels to calculate covariances.
    * The kernel operates on `full_feature_dim + 1` dimensional inputs, with the `+ 1`
        dimension representing the task feature.
    * Given a list of indices representing the active feature dimensions for each task,
        the feature space is split into several non-overlapping subsets and a base
        kernel gets constructed for each of these subset dimensions.
    * The task feature is embedded into a binary tensor, which, together with a
        `DeltaKernel`, determines which of the sub-kernels are added together for
        the given inputs.
    * There is an additional Combinatorial kernel that operates over the binary
        embedding of task features.
    """

    def __init__(
        self,
        feature_indices: list[list[int]],
        task_feature_index: int = -1,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
    ) -> None:
        r"""Initialize the kernel.

        Args:
            feature_indices: A list of lists of integers specifying the indices
                that select the features of a given task from the full tensor of
                features. The `i`th element of the list should contain `d_i`
                integers. These are the active indices for the given task.
            task_feature_index: Index of the task feature in the input tensor.
            use_saas_prior: If True, use SAAS prior for the Matern kernels.
            use_combinatorial_kernel: If True, use combinatorial kernel over the
                binary embedding of task features.
        """
        super().__init__()
        self.task_feature_index: int = task_feature_index
        self.use_saas_prior: bool = use_saas_prior
        self.use_combinatorial_kernel: bool = use_combinatorial_kernel
        active_index_map, binary_map = map_subsets(
            subsets=find_subsets(feature_indices=feature_indices),
            feature_indices=feature_indices,
        )
        self.active_index_map: dict[tuple[int], list[int]] = active_index_map
        self.binary_map: list[list[int]] = binary_map
        self.kernels: ModuleList[Kernel] = self.construct_individual_kernels()
        self.combinatorial_kernel: CombinatorialCovarModule = CombinatorialCovarModule(
            ard_num_dims=len(self.kernels)
        )
        self.delta_kernel: DeltaKernel = DeltaKernel()

    def construct_individual_kernels(self) -> ModuleList:
        """Constructs the individual kernels corresponding to subsets
        of active feature dimensions.
        """
        kernels = ModuleList()
        for active_indices in self.active_index_map:
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=len(active_indices),
                active_dims=active_indices,
            )
            if self.use_saas_prior:
                add_saas_prior(base_kernel)
            kernels.append(
                ScaleKernel(
                    base_kernel,
                    outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT,
                )
            )
        return kernels

    def map_task_to_binary(self, x_task: Tensor) -> Tensor:
        """Maps a tensor of task features to a binary tensor representing
        which kernels are active for the given task.

        Args:
            x_task: A tensor of task features of shape `batch x q`.

        Returns:
            A binary tensor of shape `batch x q x len(self.kernels)`.
            NOTE: The tensor has the same dtype as the input tensor.
            Returning a non-float tensor leads to downstream errors.
        """
        binary_map = torch.as_tensor(
            self.binary_map, dtype=x_task.dtype, device=x_task.device
        )
        return binary_map[x_task.long()]

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        r"""Evaluate the kernel on the given inputs.

        Args:
            x1: A `batch_shape x q1 x d`-dim tensor of inputs.
            x2: A `batch_shape x q2 x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x q1 x q2`-dim tensor of kernel values.
        """
        x1_binary = self.map_task_to_binary(x1[..., self.task_feature_index])
        x2_binary = self.map_task_to_binary(x2[..., self.task_feature_index])
        # This is a list of `batch_shape x q1 x q2`-dim tensors.
        kernel_evals = [k(x1, x2, **params) for k in self.kernels]
        # This is a `batch_shape x q1 x q2`-dim tensor.
        if self.use_combinatorial_kernel:
            base_evals = self.combinatorial_kernel(x1_binary, x2_binary, **params)
        else:
            base_evals = torch.zeros(
                *x1.shape[:-1], x2.shape[-2], dtype=x1.dtype, device=x1.device
            )
        # This is a list of `batch_shape x q1 x q2`-dim tensors.
        delta_evals = [
            self.delta_kernel(x1_b, x2_b, **params)
            for x1_b, x2_b in zip(
                x1_binary.split(1, dim=-1), x2_binary.split(1, dim=-1)
            )
        ]
        # Combine all kernels together to get the covariance.
        covar = base_evals
        for k, d in zip(kernel_evals, delta_evals):
            covar = covar + k * d
        return covar


def find_subsets(feature_indices: list[list[int]]) -> list[set[int]]:
    """Find the subsets of indices for which to construct sub-kernels.
    The goal is to find subsets of indices that are common across
    as many possible tasks.

    The main idea behind this implementation is to keep a running list
    of subsets. We will compare each index list with the elements of subsets,
    break them up and add to the list as needed.

    Args:
        feature_indices: A list of lists of integers specifying the indices
            mapping the features from a given task to the full tensor of features.

    Returns:
        A list of subsets of indices. All indices in the input must appear
        in exactly one of these subsets. When the subsets in the output
        are mapped to the inputs they are subsets of, each mapping should be
        unique. See the examples for more details.

    Examples:
        If input contains only one iterable, the output should be same
        as the input, cast to a set.
        If input contains two iterables, the output should be the intersection
        of the two inputs and the differences of the two. I.e., for an input of
        `[[1, 2, 3, 4], [1, 2, 5]]`, the output would be `[{1, 2}, {3, 4}, {5}]`.
        For larger inputs, the same logic applies. The key point is that we want
        the subsets to be as large as possible. For the above example,
        `[{1, 2}, {3}, {4}, {5}]` would not be acceptable since `{3}` and `{4}`
        can be joined together into a single subset of same inputs.
        However, if the inputs included a third iterable `[1, 2, 3]`, then
        `[{1, 2}, {3}, {4}, {5}]` would be the correct output since `3` appears
        in both inputs `[1, 2, 3, 4]` and `[1, 2, 3]`, but `4` only appears in
        the first one.
        The unit tests for this function provides some additional examples.
    """
    old_subsets = [set(feature_indices[0])]
    for idx_list in feature_indices[1:]:
        idx_set = set(idx_list)
        new_subsets = []
        for sub in old_subsets:
            # The idx_set contains a (possibly empty) subset of sub and potentially
            # other elements that are not in it. Break sub into two along the
            # intersection, remove common elements from idx_set and continue.
            common = idx_set.intersection(sub)
            remaining = sub.difference(common)
            if common:
                new_subsets.append(common)
                idx_set = idx_set.difference(common)
            if remaining:
                new_subsets.append(remaining)
        # If there are elements in idx_set that were not in any of the subsets,
        # we check and add those as another subset here.
        if idx_set:
            new_subsets.append(idx_set)
        old_subsets = new_subsets
    return old_subsets


def map_subsets(
    subsets: list[set[int]], feature_indices: list[list[int]]
) -> tuple[dict[tuple[int], list[int]], list[list[int]]]:
    """Map the given list of subsets of indices to the indices of feature lists they
    are subsets of. Additionally, construct a reverse mapping, a list of length
    `len(feature_indices)`, where each element is a list of length `len(subsets)`.
    Reverse mapping can be thought of as a `len(feature_indices) x len(subsets)`
    matrix, where element (i, j) is 1 if `subsets[j]` is contained in
    `feature_indices[i]` and 0 otherwise.

    Args:
        subsets: A list of sets of indices. Obtained using `find_subsets`.
        feature_indices: A list of lists of integers specifying the indices
            mapping the features from a given task to the full tensor of features.

    Returns:
        A tuple of a dictionary and a list:
        A dictionary mapping each subset (cast to tuple) to the indices
            of feature lists it is subsets of.
        A list where each element of the list (with index i) contains a binary
            list (indexed by j) representing whether the corresponding subset
            (`subsets[j]`) is active for (i.e., a subset of) the corresponding
            feature list (`feature_indices[i]`).

    Examples:
        >>> feature_indices = [[1, 2, 3, 4], [1, 2, 5]]
        >>> subsets = find_subsets(feature_indices)
        >>> # `subsets` is `[{1, 2}, {3, 4}, {5}]`.
        >>> map_subsets(subsets, feature_indices)
        >>> # Should produce `{(1, 2): [0, 1], (3, 4): [0], (5): [1]}`
        >>> # and `[[1, 1, 0], [1, 0, 1]]`.
    """
    feature_index_map = {
        tuple(s): [
            i for i, idx_list in enumerate(feature_indices) if s.issubset(idx_list)
        ]
        for s in subsets
    }
    binary_map = [
        [int(s.issubset(idx_list)) for s in subsets] for idx_list in feature_indices
    ]
    return feature_index_map, binary_map
