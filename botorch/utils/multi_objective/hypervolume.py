#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Hypervolume Utilities.

References

.. [Fonseca2006]
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

.. [Ishibuchi2011]
    H. Ishibuchi, N. Akedo, and Y. Nojima. A many-objective test problem
    for visually examining diversity maintenance behavior in a decision
    space. Proc. 13th Annual Conf. Genetic Evol. Comput., 2011.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from copy import deepcopy
from itertools import combinations

import torch
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.utils import (
    prune_inferior_points_multi_objective,
)
from botorch.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    UnsupportedError,
)
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.utils import (
    _pad_batch_pareto_frontier,
)
from botorch.utils.objective import compute_feasibility_indicator
from botorch.utils.torch import BufferDict
from botorch.utils.transforms import is_ensemble, match_batch_shape
from torch import Tensor


MIN_Y_RANGE = 1e-7


def infer_reference_point(
    pareto_Y: Tensor,
    max_ref_point: Tensor | None = None,
    scale: float = 0.1,
    scale_max_ref_point: bool = False,
) -> Tensor:
    r"""Get reference point for hypervolume computations.

    This sets the reference point to be `ref_point = nadir - scale * range`
    when there is no `pareto_Y` that is better than `max_ref_point`.
    If there's `pareto_Y` better than `max_ref_point`, the reference point
    will be set to `max_ref_point - scale * range` if `scale_max_ref_point`
    is true and to `max_ref_point` otherwise.

    [Ishibuchi2011]_ find 0.1 to be a robust multiplier for scaling the
    nadir point.

    Note: this assumes maximization of all objectives.

    Args:
        pareto_Y: A `n x m`-dim tensor of Pareto-optimal points.
        max_ref_point: A `m` dim tensor indicating the maximum reference point.
            Some elements can be NaN, except when `pareto_Y` is empty,
            in which case these dimensions will be treated as if no
            `max_ref_point` was provided and set to `nadir - scale * range`.
        scale: A multiplier used to scale back the reference point based on the
            range of each objective.
        scale_max_ref_point: A boolean indicating whether to apply scaling to
            the max_ref_point based on the range of each objective.

    Returns:
        A `m`-dim tensor containing the reference point.
    """
    if pareto_Y.shape[0] == 0:
        if max_ref_point is None:
            raise BotorchError("Empty pareto set and no max ref point provided")
        if max_ref_point.isnan().any():
            raise BotorchError("Empty pareto set and max ref point includes NaN.")
        if scale_max_ref_point:
            return max_ref_point - scale * max_ref_point.abs()
        return max_ref_point
    if max_ref_point is not None:
        non_nan_idx = ~max_ref_point.isnan()
        # Count all points exceeding non-NaN reference point as being better.
        better_than_ref = (pareto_Y[:, non_nan_idx] > max_ref_point[non_nan_idx]).all(
            dim=-1
        )
    else:
        non_nan_idx = torch.ones(
            pareto_Y.shape[-1], dtype=torch.bool, device=pareto_Y.device
        )
        better_than_ref = torch.ones(
            pareto_Y.shape[:1], dtype=torch.bool, device=pareto_Y.device
        )
    if max_ref_point is not None and better_than_ref.any() and non_nan_idx.all():
        Y_range = pareto_Y[better_than_ref].max(dim=0).values - max_ref_point
        if scale_max_ref_point:
            return max_ref_point - scale * Y_range
        return max_ref_point
    elif pareto_Y.shape[0] == 1:
        # no points better than max_ref_point and only a single observation
        # subtract MIN_Y_RANGE to handle the case that pareto_Y is a singleton
        # with objective value of 0.
        Y_range = pareto_Y.abs().clamp_min(MIN_Y_RANGE).view(-1)
        ref_point = pareto_Y.view(-1) - scale * Y_range
    else:
        # no points better than max_ref_point and multiple observations
        # make sure that each dimension of the nadir point is no greater than
        # the max_ref_point
        nadir = pareto_Y.min(dim=0).values
        if max_ref_point is not None:
            nadir[non_nan_idx] = torch.min(
                nadir[non_nan_idx], max_ref_point[non_nan_idx]
            )
        ideal = pareto_Y.max(dim=0).values
        # handle case where all values for one objective are the same
        Y_range = (ideal - nadir).clamp_min(MIN_Y_RANGE)
        ref_point = nadir - scale * Y_range
    # Set not-nan indices - if any - to max_ref_point.
    if non_nan_idx.any() and not non_nan_idx.all() and better_than_ref.any():
        if scale_max_ref_point:
            ref_point[non_nan_idx] = (max_ref_point - scale * Y_range)[non_nan_idx]
        else:
            ref_point[non_nan_idx] = max_ref_point[non_nan_idx]
    return ref_point


class Hypervolume:
    r"""Hypervolume computation dimension sweep algorithm from [Fonseca2006]_.

    Adapted from Simon Wessing's implementation of the algorithm
    (Variant 3, Version 1.2) in [Fonseca2006]_ in PyMOO:
    https://github.com/msu-coinlab/pymoo/blob/master/pymoo/vendor/hv.py

    Maximization is assumed.

    TODO: write this in C++ for faster looping.
    """

    def __init__(self, ref_point: Tensor) -> None:
        r"""Initialize hypervolume object.

        Args:
            ref_point: `m`-dim Tensor containing the reference point.

        """
        self.ref_point = ref_point

    @property
    def ref_point(self) -> Tensor:
        r"""Get reference point (for maximization).

        Returns:
            A `m`-dim tensor containing the reference point.
        """
        return -self._ref_point

    @ref_point.setter
    def ref_point(self, ref_point: Tensor) -> None:
        r"""Set the reference point for maximization

        Args:
            ref_point:  A `m`-dim tensor containing the reference point.
        """
        self._ref_point = -ref_point

    def compute(self, pareto_Y: Tensor) -> float:
        r"""Compute hypervolume.

        Args:
            pareto_Y: A `n x m`-dim tensor of pareto optimal outcomes

        Returns:
            The hypervolume.
        """
        if pareto_Y.shape[-1] != self._ref_point.shape[0]:
            raise BotorchTensorDimensionError(
                "pareto_Y must have the same number of objectives as ref_point. "
                f"Got {pareto_Y.shape[-1]}, expected {self._ref_point.shape[0]}."
            )
        elif pareto_Y.ndim != 2:
            raise BotorchTensorDimensionError(
                f"pareto_Y must have exactly two dimensions, got {pareto_Y.ndim}."
            )
        # This assumes maximization, but internally flips the sign of the pareto front
        # and the reference point and computes hypervolume for the minimization problem.
        pareto_Y = -pareto_Y
        better_than_ref = (pareto_Y <= self._ref_point).all(dim=-1)
        pareto_Y = pareto_Y[better_than_ref]
        # shift the pareto front so that reference point is all zeros
        pareto_Y = pareto_Y - self._ref_point
        self._initialize_multilist(pareto_Y)
        bounds = torch.full_like(self._ref_point, float("-inf"))
        return self._hv_recursive(
            i=self._ref_point.shape[0] - 1, n_pareto=pareto_Y.shape[0], bounds=bounds
        )

    def _hv_recursive(self, i: int, n_pareto: int, bounds: Tensor) -> float:
        r"""Recursive method for hypervolume calculation.

        This assumes minimization (internally).

        In contrast to the paper, this code assumes that the reference point
        is the origin. This enables pruning a few operations.

        Args:
            i: objective index
            n_pareto: number of pareto points
            bounds: objective bounds

        Returns:
            The hypervolume.
        """
        hvol = torch.tensor(0.0, dtype=bounds.dtype, device=bounds.device)
        sentinel = self.list.sentinel
        if n_pareto == 0:
            # base case: one dimension
            return hvol.item()
        elif i == 0:
            # base case: one dimension
            return -sentinel.next[0].data[0].item()
        elif i == 1:
            # two dimensions, end recursion
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol.item()
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (
                q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]
            ):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1 : i + 1] = q.area[:i] * -(q.data[:i])
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol.item()

    def _initialize_multilist(self, pareto_Y: Tensor) -> None:
        r"""Sets up the multilist data structure needed for calculation.

        Note: this assumes minimization.

        Args:
            pareto_Y: A `n x m`-dim tensor of pareto optimal objectives.

        """
        m = pareto_Y.shape[-1]
        nodes = [
            Node(m=m, dtype=pareto_Y.dtype, device=pareto_Y.device, data=point)
            for point in pareto_Y
        ]
        self.list = MultiList(m=m, dtype=pareto_Y.dtype, device=pareto_Y.device)
        for i in range(m):
            sort_by_dimension(nodes, i)
            self.list.extend(nodes, i)


def sort_by_dimension(nodes: list[Node], i: int) -> None:
    r"""Sorts the list of nodes in-place by the specified objective.

    Args:
        nodes: A list of Nodes
        i: The index of the objective to sort by

    """
    # build a list of tuples of (point[i], node)
    decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
    # sort by this value
    decorated.sort()
    # write back to original list
    nodes[:] = [node for (_, _, node) in decorated]


class Node:
    r"""Node in the MultiList data structure."""

    def __init__(
        self,
        m: int,
        dtype: torch.dtype,
        device: torch.device,
        data: Tensor | None = None,
    ) -> None:
        r"""Initialize MultiList.

        Args:
            m: The number of objectives
            dtype: The dtype
            device: The device
            data: The tensor data to be stored in this Node.
        """
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = torch.zeros(m, dtype=dtype, device=device)
        self.volume = torch.zeros_like(self.area)


class MultiList:
    r"""A special data structure used in hypervolume computation.

    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """

    def __init__(self, m: int, dtype: torch.dtype, device: torch.device) -> None:
        r"""Initialize `m` doubly linked lists.

        Args:
            m: number of doubly linked lists
            dtype: the dtype
            device: the device

        """
        self.m = m
        self.sentinel = Node(m=m, dtype=dtype, device=device)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node: Node, index: int) -> None:
        r"""Appends a node to the end of the list at the given index.

        Args:
            node: the new node
            index: the index where the node should be appended.
        """
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes: list[Node], index: int) -> None:
        r"""Extends the list at the given index with the nodes.

        Args:
            nodes: list of nodes to append at the given index.
            index: the index where the nodes should be appended.

        """
        for node in nodes:
            self.append(node=node, index=index)

    def remove(self, node: Node, index: int, bounds: Tensor) -> Node:
        r"""Removes and returns 'node' from all lists in [0, 'index'].

        Args:
            node: The node to remove
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim tensor bounds on the objectives
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
        bounds.data = torch.min(bounds, node.data)
        return node

    def reinsert(self, node: Node, index: int, bounds: Tensor) -> None:
        r"""Re-inserts the node at its original position.

        Re-inserts the node at its original position in all lists in [0, 'index']
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        Args:
            node: The node
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim tensor bounds on the objectives

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds.data = torch.min(bounds, node.data)


class SubsetIndexCachingMixin:
    """A Mixin class that adds q-subset index computations and caching."""

    def __init__(self):
        """Initializes the class with q_out = -1 and an empty q_subset_indices dict."""
        self.q_out: int = -1
        self.q_subset_indices: BufferDict[str, Tensor] = BufferDict()

    def compute_q_subset_indices(
        self, q_out: int, device: torch.device
    ) -> BufferDict[str, Tensor]:
        r"""Returns and caches a dict of indices equal to subsets of `{1, ..., q_out}`.

        This means that consecutive calls to `self.compute_q_subset_indices` with
        the same `q_out` do not recompute the indices for all (2^q_out - 1) subsets.

        NOTE: This will use more memory than regenerating the indices
        for each i and then deleting them, but it will be faster for
        repeated evaluations (e.g. during optimization).

        Args:
            q_out: The batch size of the objectives. This is typically equal
                to the q-batch size of `X`. However, if using a set valued
                objective (e.g., MVaR) that produces `s` objective values for
                each point on the q-batch of `X`, we need to properly account
                for each objective while calculating the hypervolume contributions
                by using `q_out = q * s`.

        Returns:
            A dict that maps "q choose i" to all size-i subsets of `{1, ..., q_out}`.
        """
        if q_out != self.q_out:
            self.q_subset_indices = compute_subset_indices(q_out, device=device)
            self.q_out = q_out
        return self.q_subset_indices


def compute_subset_indices(
    q: int, device: torch.device | None = None
) -> BufferDict[str, Tensor]:
    r"""Compute all (2^q - 1) distinct subsets of {1, ..., `q`}.

    Args:
        q: An integer defininig the set {1, ..., `q`} whose subsets to compute.

    Returns:
        A dict that maps "q choose i" to all size-i subsets of {1, ..., `q_out`}.
    """
    indices = torch.arange(q, dtype=torch.long, device=device)
    return BufferDict(
        {
            f"q_choose_{i}": torch.tensor(
                list(combinations(indices, i)), dtype=torch.long, device=device
            )
            for i in range(1, q + 1)
        }
    )


class NoisyExpectedHypervolumeMixin(CachedCholeskyMCSamplerMixin):
    def __init__(
        self,
        model: Model,
        ref_point: list[float] | Tensor,
        X_baseline: Tensor,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool | None = None,
        marginalize_dim: int | None = None,
    ):
        """Initialize a mixin that contains functions for the batched Pareto-frontier
        partitioning used by the noisy hypervolume-improvement-based acquisition
        functions, i.e. qNEHVI and qLogNEHVI.

        Args:
            model: A fitted model.
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying `objective` to the samples).
            X_baseline: A `r x d`-dim Tensor of `r` design points that have already
                been observed. These points are considered as potential approximate
                pareto-optimal design points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`. NOTE: A box decomposition is
                of the Pareto front is created for each MC sample, an operation that
                scales as `O(n^m)` and thus becomes particularly costly for `m` > 2.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to `IdentityMCMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acqusition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation, but
                have not yet been evaluated.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the pareto optimal and better than the
                reference point. This can significantly improve computation time and
                is generally recommended. In order to customize pruning parameters,
                instead manually call `prune_inferior_points_multi_objective` on
                `X_baseline` before instantiating the acquisition function.
            alpha: The hyperparameter controlling the approximate non-dominated
                partitioning. The default value of 0.0 means an exact partitioning
                is used. As the number of objectives `m` increases, consider increasing
                this parameter in order to limit computational complexity.
            cache_pending: A boolean indicating whether to use cached box
                decompositions (CBD) for handling pending points. This is
                generally recommended.
            max_iep: The maximum number of pending points before the box
                decompositions will be recomputed.
            incremental_nehvi: A boolean indicating whether to compute the
                incremental NEHVI from the `i`th point where `i=1, ..., q`
                under sequential greedy optimization, or the full qNEHVI over
                `q` points.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
            marginalize_dim: A batch dimension that should be marginalized. For example,
                this is useful when using a batched fully Bayesian model.
        """
        super().__init__(model=model, cache_root=cache_root, sampler=sampler)
        if len(ref_point) < 2:
            raise ValueError(
                "NoisyExpectedHypervolumeMixin supports m>=2 outcomes "
                f"but ref_point has length {len(ref_point)}, which is smaller than 2."
            )
        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}
        ref_point = torch.as_tensor(ref_point, **tkwargs)
        self.register_buffer("ref_point", ref_point)

        if X_baseline.ndim > 2:
            raise UnsupportedError(
                f"NoisyExpectedHypervolumeMixin does not support batched "
                f"X_baseline. Expected 2 dims, got {X_baseline.ndim}."
            )
        if prune_baseline:
            X_baseline = prune_inferior_points_multi_objective(
                model=model,
                X=X_baseline,
                objective=objective,
                constraints=constraints,
                ref_point=ref_point,
                marginalize_dim=marginalize_dim,
            )

        self.alpha = alpha
        self.q_in = -1
        self.q_out = -1

        self.partitioning = None
        # set partitioning class and args
        self.p_kwargs = {}
        if self.alpha > 0:
            self.p_kwargs["alpha"] = self.alpha
            self.p_class = NondominatedPartitioning
        else:
            self.p_class = FastNondominatedPartitioning
        self.register_buffer("_X_baseline", X_baseline)
        self.register_buffer("_X_baseline_and_pending", X_baseline)
        self.register_buffer(
            "cache_pending",
            torch.tensor(cache_pending, dtype=bool),
        )
        self.register_buffer(
            "_prev_nehvi",
            torch.tensor(0.0, **tkwargs),
        )
        self.register_buffer(
            "_max_iep",
            torch.tensor(max_iep, dtype=torch.long),
        )
        self.register_buffer(
            "incremental_nehvi",
            torch.tensor(incremental_nehvi, dtype=torch.bool),
        )
        # Base sampler is initialized in _set_cell_bounds.
        self.base_sampler = None

        # is this called twice, once here, once in MultiObjectiveMCAcquisitionFunction?
        if X_pending is not None:
            # This will call self._set_cell_bounds if the number of pending
            # points is greater than self._max_iep.
            self.set_X_pending(X_pending)
        # In the case that X_pending is not None, but there are fewer than
        # max_iep pending points, the box decompositions are not performed in
        # set_X_pending. Therefore, we need to perform a box decomposition over
        # f(X_baseline) here.
        if X_pending is None or X_pending.shape[-2] <= self._max_iep:
            self._set_cell_bounds(num_new_points=X_baseline.shape[0])

        # Set q_in=-1 to so that self.sampler is updated at the next forward call.
        self.q_in = -1

    @property
    def X_baseline(self) -> Tensor:
        r"""Return X_baseline augmented with pending points cached using CBD."""
        return self._X_baseline_and_pending

    def _compute_initial_hvs(self, obj: Tensor, feas: Tensor | None = None) -> None:
        r"""Compute hypervolume dominated by f(X_baseline) under each sample.

        Args:
            obj: A `sample_shape x batch_shape x n x m`-dim tensor of samples
                of objectives.
            feas: `sample_shape x batch_shape x n`-dim tensor of samples
                of feasibility indicators.
        """
        initial_hvs = []
        for i, sample in enumerate(obj):
            if self.constraints is not None:
                sample = sample[feas[i]]
            dominated_partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=sample,
            )
            hv = dominated_partitioning.compute_hypervolume()
            initial_hvs.append(hv)
        self.register_buffer(
            "_initial_hvs",
            torch.tensor(initial_hvs, dtype=obj.dtype, device=obj.device).view(
                self._batch_sample_shape, *obj.shape[-2:]
            ),
        )

    def _set_cell_bounds(self, num_new_points: int) -> None:
        r"""Compute the box decomposition under each posterior sample.

        Args:
            num_new_points: The number of new points (beyond the points
                in X_baseline) that were used in the previous box decomposition.
                In the first box decomposition, this should be the number of points
                in X_baseline.
        """
        if self.X_baseline.shape[0] > 0:
            with torch.no_grad():
                posterior = self.model.posterior(self.X_baseline)
            # Reset sampler, accounting for possible one-to-many transform.
            self.q_in = -1
            if self.base_sampler is None:
                # Initialize the base sampler if needed.
                samples = self.get_posterior_samples(posterior)
                self.base_sampler = deepcopy(self.sampler)
            else:
                samples = self.base_sampler(posterior)
            n_w = posterior._extended_shape()[-2] // self.X_baseline.shape[-2]
            self._set_sampler(q_in=num_new_points * n_w, posterior=posterior)
            # cache posterior
            if self._cache_root:
                # Note that this implicitly uses LinearOperator's caching to check if
                # the proper root decomposition has already been cached to
                # `posterior.mvn.lazy_covariance_matrix`, which it may have been in
                # the call to `self.base_sampler`, and computes it if not found
                self._baseline_L = self._compute_root_decomposition(posterior=posterior)
            obj = self.objective(samples, X=self.X_baseline)

        else:
            sample_shape = (
                self.sampler.sample_shape
                if self.sampler is not None
                else self._default_sample_shape
            )
            obj = torch.empty(
                *sample_shape,
                0,
                self.ref_point.shape[-1],
                dtype=self.ref_point.dtype,
                device=self.ref_point.device,
            )

        # compute feasibility indicator if there are constraints
        if self.constraints is None or self.X_baseline.shape[0] == 0:
            feas = None
        else:
            feas = compute_feasibility_indicator(
                constraints=self.constraints, samples=samples
            )

        self._batch_sample_shape = obj.shape[:-2]
        # collapse batch dimensions
        # use numel() rather than view(-1) to handle case of no baseline points
        new_batch_shape = self._batch_sample_shape.numel()
        obj = obj.view(new_batch_shape, *obj.shape[-2:])
        if feas is not None:
            feas = feas.view(new_batch_shape, *feas.shape[-1:])

        if self.partitioning is None and not self.incremental_nehvi:
            self._compute_initial_hvs(obj=obj, feas=feas)

        if self.ref_point.shape[-1] > 2:
            # the partitioning algorithms run faster on the CPU
            # due to advanced indexing
            ref_point_cpu = self.ref_point.cpu()
            obj_cpu = obj.cpu()
            if feas is not None:
                feas_cpu = feas.cpu()
                obj_cpu = [obj_cpu[i][feas_cpu[i]] for i in range(obj.shape[0])]
            partitionings = []
            for sample in obj_cpu:
                partitioning = self.p_class(
                    ref_point=ref_point_cpu, Y=sample, **self.p_kwargs
                )
                partitionings.append(partitioning)
            self.partitioning = BoxDecompositionList(*partitionings)
        else:
            # use batched partitioning
            obj = _pad_batch_pareto_frontier(
                Y=obj,
                ref_point=self.ref_point.unsqueeze(0).expand(
                    obj.shape[0], self.ref_point.shape[-1]
                ),
                feasibility_mask=feas,
            )
            self.partitioning = self.p_class(
                ref_point=self.ref_point, Y=obj, **self.p_kwargs
            )
        cell_bounds = self.partitioning.get_hypercell_bounds().to(self.ref_point)
        cell_bounds = cell_bounds.view(
            2, *self._batch_sample_shape, *cell_bounds.shape[-2:]
        )  # 2 x batch_shape x sample_shape x num_cells x m
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is None:
            self.X_pending = None
        else:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    BotorchWarning,
                    stacklevel=2,
                )
            self.X_pending = X_pending.detach().clone()
            if self.cache_pending:
                X_baseline = torch.cat([self._X_baseline, X_pending], dim=-2)
                # Number of new points is the total number of points minus
                # (the number of previously cached pending points plus the
                # of number of baseline points).
                num_new_points = X_baseline.shape[0] - self.X_baseline.shape[0]
                if num_new_points > 0:
                    if num_new_points > self._max_iep:
                        # Set the new baseline points to include pending points.
                        self.register_buffer("_X_baseline_and_pending", X_baseline)
                        # Recompute box decompositions.
                        self._set_cell_bounds(num_new_points=num_new_points)
                        if not self.incremental_nehvi:
                            self._prev_nehvi = (
                                (self._hypervolumes - self._initial_hvs)
                                .clamp_min(0.0)
                                .mean()
                            )
                        # Set q_in=-1 to so that self.sampler is updated at the next
                        # forward call.
                        self.q_in = -1

    @property
    def _hypervolumes(self) -> Tensor:
        r"""Compute hypervolume over X_baseline under each posterior sample.

        Returns:
            A `sample_shape`-dim tensor of hypervolumes.
        """
        return (
            self.partitioning.compute_hypervolume()
            .to(self.ref_point)  # for m > 2, the partitioning is on the CPU
            .view(self._batch_sample_shape)
        )

    def _compute_posterior_samples_and_concat_pending(
        self, X: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Get samples from the posterior, and concatenate uncached pending points.

        Args:
            X: `batch_shape x q x d` X Tensor pased into the `forward` method of an acqf

        Returns:
            A tuple containing samples of the latent function from the posterior, and
            the `batch_shape x (q + num_uncached_pending) x d` X tensor including any
            pending observations that have not been cached.
        """
        # Manually concatenate pending points only if:
        # - pending points are not cached, or
        # - number of pending points is less than max_iep
        if self.X_pending is not None:
            num_pending = self.X_pending.shape[-2]
            num_X_baseline = self._X_baseline.shape[-2]
            num_X_baseline_and_cached_pending = self.X_baseline.shape[-2]
            num_uncached_pending = (
                (num_pending + num_X_baseline - num_X_baseline_and_cached_pending)
                if self.cache_pending
                else num_pending
            )
            X_pending_uncached = self.X_pending[
                ..., num_pending - num_uncached_pending :, :
            ]
            X = torch.cat([X, match_batch_shape(X_pending_uncached, X)], dim=-2)
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # NOTE: To ensure that we correctly sample `f(X)` from the joint distribution
        # `f((X_baseline, X)) ~ P(f | D)`, it is critical to compute the joint posterior
        # over X *and* X_baseline -- which also contains pending points whenever there
        # are any --  since the baseline and pending values `f(X_baseline)` are
        # generally pre-computed and cached before the `forward` call, see the docs of
        # `cache_pending` for details.
        # TODO: Improve the efficiency by not re-computing the X_baseline-X_baseline
        # covariance matrix, but only the covariance of
        # 1) X and X, and
        # 2) X and X_baseline.
        posterior = self.model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
        event_shape_lag = 1 if is_ensemble(self.model) else 2
        n_w = (
            posterior._extended_shape()[X_full.dim() - event_shape_lag]
            // X_full.shape[-2]
        )
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        return self._get_f_X_samples(posterior=posterior, q_in=q_in), X


def get_hypervolume_maximizing_subset(
    n: int, Y: Tensor, ref_point: Tensor
) -> tuple[Tensor, Tensor]:
    """Find an approximately hypervolume-maximizing subset of size `n`.

    This greedily selects points from Y to maximize the hypervolume of
    the subset sequentially. This has bounded error since hypervolume is
    submodular.

    Args:
        n: The size of the subset to return.
        Y: A `n' x m`-dim tensor of outcomes.
        ref_point: A `m`-dim tensor containing the reference point.

    Returns:
        A two-element tuple containing
            - A `n x m`-dim tensor of outcomes.
            - A `n`-dim tensor of indices of the outcomes in the original set.
    """
    if Y.ndim != 2:
        raise NotImplementedError(
            "Only two dimensions are supported (no additional) batch dims."
        )
    elif Y.shape[0] < n:
        raise ValueError(
            f"Y has fewer points ({Y.shape[0]}) than the requested subset size ({n})."
        )
    Y_subset = torch.zeros(0, Y.shape[1], dtype=Y.dtype, device=Y.device)
    selected_indices = []
    remaining_idcs = set(range(Y.shape[0]))
    best_hv = 0.0
    for _ in range(n):
        # Add each point and compute the hypervolume
        best_idx = None
        for i in remaining_idcs:
            partitioning = DominatedPartitioning(
                ref_point=ref_point, Y=torch.cat((Y_subset, Y[i : i + 1]), dim=0)
            )
            hv = partitioning.compute_hypervolume().item()
            if hv > best_hv:
                best_idx = i
                best_hv = hv
        if best_idx is None:
            # no arm improved HV, so select a random arm. This will only happen if Y is
            # not a Pareto frontier, where all points are better than the reference
            # point
            best_idx = random.choice(list(remaining_idcs))
        remaining_idcs.remove(best_idx)
        selected_indices.append(best_idx)
        Y_subset = torch.cat((Y_subset, Y[best_idx : best_idx + 1]), dim=0)
    return Y_subset, torch.tensor(selected_indices, dtype=torch.long, device=Y.device)
