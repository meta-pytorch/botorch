#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines some helper functions for parsing hierarchical dependencies. We will
use a running example to illustrate the functionality of each helper function::

    ROOT
    ├── C0, C1
    └── P2
        ├── (0) C3
        └── (1) P4
                ├── (0) C5
                └── (1) C6

- C0, C1, C3, C5, and C6 are child nodes (similar to non-hierarchical parameters in Ax).
- P2 and P4 are parent nodes (similar to hierarchical parameters in Ax).
- Each node, except ROOT, corresponds to a dimension in the feature vector X.

Let's say the input X is a vector of the form ``(C0, C1, P2, C3, P4, C5, C6)``. The
features do not necessarily need to follow this particular order, but we will stick to
this order as an example. Then, the hierarchical tree is represented by a list of
dictionaries as follows::

    hierarchical_dependencies = {
        2: {0: [3], 1: [4]},  # P2 == 0 --> C3 is activated; P2 == 1 -> P4 is activated.
        4: {0: [5], 1: [6]},  # P4 == 0 --> C5 is activated; P4 == 1 -> C6 is activated.
    }

Note that, in the above example, ROOT does not actually exist in the representation. It
is an imaginary node that is the parent of all orphan nodes, e.g., C0, C1, and P2. But
all functions in this file supports both rootless trees and rooted trees.
"""


def get_orphan_feature_indices(
    dim: int,
    hierarchical_dependencies: dict[int, dict[int, list[int]]],
) -> list[int]:
    """
    Construct the indices of the orphan nodes by parsing the hierarchical dependencies.
    They are precisely the children of the (imaginary) root node.

    Args:
        dim: The full dimension of the feature vector.
        hierarchical_dependencies: A dictionary specifying the hierarchical structure.

    Returns:
        A list of indices of the orphan nodes sorted in ascending order.
    """
    grouped_non_orphan_feature_indices = [
        children_indices
        for dependents in hierarchical_dependencies.values()
        for children_indices in dependents.values()
    ]
    non_orphan_feature_indices = sum(grouped_non_orphan_feature_indices, [])

    if len(set(non_orphan_feature_indices)) != len(non_orphan_feature_indices):
        raise ValueError(
            f"{non_orphan_feature_indices=} contains duplicate indices. This is likely"
            f"a mistake in the hierarchical dependencies."
        )

    return sorted(set(range(dim)) - set(non_orphan_feature_indices))


def get_index_to_path(
    dim: int,
    hierarchical_dependencies: dict[int, dict[int, list[int]]],
) -> dict[int, list[tuple[int, int]]]:
    """
    Construct a dictionary that maps the index of each node (feature) to its
    root-to-node path.

    Args:
        dim: The full dimension of the feature vector.
        hierarchical_dependencies: A dictionary specifying the hierarchical structure.

    Returns:
        A dictionary that maps the node (or feature) index ``fid`` to its
        root-to-node path. The path is represented by a list of tuples of the
        form ``(index, value)``. If we follow the path by setting
        ``X[index] = value`` for all ``(index, value)`` in the path, then
        ``X[fid]`` is activated.
    """

    def has_children(fid: int) -> bool:
        return fid in hierarchical_dependencies and hierarchical_dependencies[fid]

    fid_to_path = {}

    # depth-first traversal
    def dfs(fid: int, path: list[tuple[int, int]]) -> None:
        """
        Depth-first search starting from ``fid``.

        Args:
            fid: The feature index to be visited.
            path: A list of tuples of the form ``(index, value)``. They are the
                indices and values of the ancestors of ``fid``, which represent
                the path in the hierarchical tree that leads to ``fid``.
        """
        fid_to_path[fid] = path

        if not has_children(fid):
            return

        for parent_value, children_indices in hierarchical_dependencies[fid].items():
            for child in children_indices:
                dfs(child, path + [(fid, parent_value)])

    orphan_feature_indices = get_orphan_feature_indices(dim, hierarchical_dependencies)

    for idx in orphan_feature_indices:
        dfs(idx, [])

    return fid_to_path


def get_blocks_with_paths(
    dim: int,
    hierarchical_dependencies: dict[int, dict[int, list[int]]],
    keep_hierarchical_features: bool = True,
    separate_hierarchical_features: bool = True,
) -> tuple[list[list[int], list[list[tuple[int, int]]]]]:
    """
    A helper function parsing the hierarchical dependencies. This function does two
    things:

    1. Partition the indices ``{0, 1, 2, ..., dim - 1}`` into blocks. Features
       in the same block are always activated together---either they are all
       active or all inactive.
    2. Construct the path from the root to each block. This will be helpful in
       checking if a block is active or not.

    The partition will be used in ``HierarchicalConditionalKernel``, which
    creates a kernel for each block. The trivial partition
    ``{{0}, {1}, ..., {dim - 1}}`` is bad, because then the kernel would be
    *completely* additive. The goal is to construct a partition where each
    block is as large as possible. Two nodes end up in the same block if and
    only if they share the same parent node and are activated by the same
    parent node value.

    Args:
        dim: The full dimension of the feature vector.
        hierarchical_dependencies: A list that specifies the hierarchical dependencies.
        keep_hierarchical_features: If true, the hierarchical features are kept in the
            partition.
        separate_hierarchical_features: If true, hierarchical features are not grouped
            with non-hierarchical features. This flag is relevant only if
            ``keep_hierarchical_features`` is true.

    Returns:
        A tuple of two lists:

        - A partition of the indices ``{0, 1, 2, ..., dim - 1}``. Each block in
          the partition is a list of indices of features that are activated at
          the same time.
        - A list of root-to-block paths in the tree. Each path is represented by
          a list of tuples of the form ``(index, value)``. The block is
          activated if following the path by setting the ``index``-th feature to
          ``value``.
    """

    def has_children(fid: int) -> bool:
        return fid in hierarchical_dependencies and hierarchical_dependencies[fid]

    orphan_feature_indices = get_orphan_feature_indices(dim, hierarchical_dependencies)

    partition_non_orphans = [
        children_indices
        for dependents in hierarchical_dependencies.values()
        for children_indices in dependents.values()
        if children_indices  # If the children list is empty, do not include it.
    ]
    partition = partition_non_orphans + [orphan_feature_indices]

    # Refine the partition if needed. Here, refinement means separate hierarchical
    # nodes from non-hierarchical nodes. Hierarchical nodes are boolean, discrete or
    # categorical, and thus may require a separate kernel for modeling.
    if not keep_hierarchical_features or (
        keep_hierarchical_features and separate_hierarchical_features
    ):
        # This filters out the hierarchical nodes.
        refined_partition = [
            [fid for fid in block if not has_children(fid)] for block in partition
        ]

        if keep_hierarchical_features and separate_hierarchical_features:
            refined_partition += [
                [fid for fid in block if has_children(fid)] for block in partition
            ]

        # Some blocks might be empty due to the refinement. Filter them out.
        refined_partition = [block for block in refined_partition if block]

    else:
        refined_partition = partition

    # Sorting is techcnical not necessary. But it does simplify testing substantially.
    refined_partition = sorted([sorted(block) for block in refined_partition])

    fid_to_path = get_index_to_path(dim, hierarchical_dependencies)

    # For each block, construct the path from the root to the block. Note that nodes in
    # the same block, by definition, are activated by the same path. Hence, it suffices
    # to grab any root-to-node path in the block.
    paths = [fid_to_path[block[0]] for block in refined_partition]

    return refined_partition, paths
