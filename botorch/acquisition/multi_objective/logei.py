# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-objective variants of the LogEI family of acquisition functions, see
[Ament2023logei]_ for details.

A fused C++ kernel is available that accelerates the inner loop of
``_compute_log_qehvi`` by ~1.5-3x on CPU.  It is loaded lazily on first
acquisition function construction — either from a pre-compiled extension
(Buck builds) or via JIT compilation (pip installs).  If loading fails the
pure-Python implementation is used transparently.

For pip installs, JIT compilation requires a C++ compiler (``gcc`` or
``clang``) to be available on the system.  The ``ninja`` build system is
included as a dependency.  The kernel is compiled once on first use (~7 s)
and cached by PyTorch for subsequent imports.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable

import torch
from botorch.acquisition.logei import TAU_MAX, TAU_RELU
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.logging import logger
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import (
    NoisyExpectedHypervolumeMixin,
    SubsetIndexCachingMixin,
)
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import (
    fatmin,
    log_fatplus,
    log_softplus,
    logdiffexp,
    logmeanexp,
    logplusexp,
    logsumexp,
    smooth_amin,
)
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor

_C = None  # Fused C++ kernel module, loaded lazily by _try_load_fused_kernel().
_load_attempted = False  # Sentinel to avoid retrying after a failed load.


def _try_load_fused_kernel() -> None:
    """Load the fused C++ kernel if available, storing it in module-level ``_C``.

    Tries the pre-compiled Buck extension first, then falls back to JIT
    compilation for pip/source installs.  Called from ``__init__`` of the
    acquisition functions so the cost is deferred until actually needed.
    Only attempts loading once; subsequent calls are no-ops.
    """
    global _C, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:  # pragma: no cover
        import logei_fused_ext

        _C = logei_fused_ext
        return
    except ImportError:
        pass
    try:
        from torch.utils.cpp_extension import load as _load_ext

        _cpp_src = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "csrc", "logei_fused.cpp"
        )
        _C = _load_ext(
            name="logei_fused_ext",
            sources=[os.path.abspath(_cpp_src)],
            extra_cflags=["-O3", "-march=native"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Failed to compile fused qLogEHVI C++ extension: {exc}. "
            "Falling back to the pure-Python implementation. "
            "To get ~3x speedup, ensure a C++ compiler and the `ninja` "
            "package are installed.",
            stacklevel=2,
        )
        logger.debug("C++ extension compilation error", exc_info=True)


class _FusedLogAreas(torch.autograd.Function):
    """Custom autograd function wrapping the fused C++ kernel.

    Fuses steps 1-4 of ``_compute_log_qehvi`` into a single C++ call.
    In the pure-Python path these steps are:

    1. ``_log_improvement`` — log smoothed ReLU of (obj - cell_lower),
    2. ``_smooth_min`` — smooth minimum over subset elements,
    3. ``_log_cell_lengths`` — smooth minimum with log cell side lengths,
    4. ``sum`` over objectives to get log areas.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        obj_subsets: Tensor,
        cell_lower: Tensor,
        cell_upper: Tensor,
        tau_relu: float,
        tau_max: float,
    ) -> Tensor:
        batch_shape = obj_subsets.shape[:-3]
        B = max(1, int(torch.Size(batch_shape).numel()))
        obj_flat = obj_subsets.reshape(B, *obj_subsets.shape[-3:])

        # Reshape cell bounds to 3-D (B_cells, nc, m) for the C++ kernel.
        # Non-batched (2-D) bounds are left as-is; the kernel handles both.
        nc, m_cells = cell_lower.shape[-2], cell_lower.shape[-1]
        if cell_lower.ndim > 2:
            cell_lower = cell_lower.reshape(-1, nc, m_cells)
            cell_upper = cell_upper.reshape(-1, nc, m_cells)

        result = _C.forward(obj_flat, cell_lower, cell_upper, tau_relu, tau_max)

        ctx.save_for_backward(obj_flat, cell_lower, cell_upper)
        ctx.tau_relu = tau_relu
        ctx.tau_max = tau_max
        ctx.orig_shape = obj_subsets.shape

        return result.reshape(*batch_shape, result.shape[-2], result.shape[-1])

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor | None, ...]:
        obj_flat, cell_lower, cell_upper = ctx.saved_tensors
        B = obj_flat.shape[0]

        grad_flat = grad_output.reshape(B, *grad_output.shape[-2:])

        grad_obj = _C.backward(
            grad_flat,
            obj_flat,
            cell_lower,
            cell_upper,
            ctx.tau_relu,
            ctx.tau_max,
        )

        # Gradients for: obj_subsets, cell_lower, cell_upper, tau_relu, tau_max.
        # Only obj_subsets needs a gradient; the rest are non-differentiable.
        return grad_obj.reshape(ctx.orig_shape), None, None, None, None


class qLogExpectedHypervolumeImprovement(
    MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin
):
    _log: bool = True

    def __init__(
        self,
        model: Model,
        ref_point: list[float] | Tensor,
        partitioning: NondominatedPartitioning,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-2,
        fat: bool = True,
        tau_relu: float = TAU_RELU,
        tau_max: float = TAU_MAX,
    ) -> None:
        r"""Parallel Log Expected Hypervolume Improvement supporting m>=2 outcomes.

        See [Ament2023logei]_ for details and the methodology behind the LogEI family of
        acquisition function. Line-by-line differences to the original differentiable
        expected hypervolume formulation of [Daulton2020qehvi]_ are described via inline
        comments in ``forward``.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> acq = qLogExpectedHypervolumeImprovement(model, ref_point, partitioning)
            >>> value = acq(test_X)

        Args:
            model: A fitted model.
            ref_point: A list or tensor with ``m`` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying ``objective`` to the samples).
            partitioning: A ``NondominatedPartitioning`` module that provides the non-
                dominated front and a partitioning of the non-dominated space in hyper-
                rectangles. If constraints are present, this partitioning must only
                include feasible points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using ``get_sampler``.
            objective: The MCMultiOutputObjective under which the samples are evaluated.
                Defaults to ``IdentityMultiOutputObjective()``.
            constraints: A list of callables, each mapping a Tensor of dimension
                ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                ``sample_shape x batch-shape x q``, where negative values imply
                feasibility. The acquisition function will compute expected feasible
                hypervolume.
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design
                points that have points that have been submitted for function
                evaluation but have not yet been evaluated. Concatenated into ``X``
                upon forward call. Copied and set to have no gradient.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same eta is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                eta value.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU and the maximum.
            tau_relu: Temperature parameter controlling the sharpness of the
                approximation to the ReLU over the ``q`` candidate points. For further
                details, see the comments above the definition of ``TAU_RELU``.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the ``max`` operator over the ``q`` candidate points.
                For further details, see the comments above the definition of
                ``TAU_MAX``.
        """
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The dimensionality of the reference point must match the number of "
                f"outcomes. Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.tau_relu = tau_relu
        self.tau_max = tau_max
        self.fat = fat
        _try_load_fused_kernel()

    def _compute_log_qehvi(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        When the fused C++ kernel is available and the cell bounds are not
        batched (the common case for ``qLogEHVI``), steps 1-4 of the inner
        loop are replaced by a single fused call for ~1.5-3x speedup on CPU.
        Otherwise the pure-Python path is used.

        Args:
            samples: A ``sample_shape x batch_shape x q' x m``-dim tensor of samples.
            X: A ``batch_shape x q x d``-dim tensor of inputs.

        Returns:
            A ``batch_shape x (model_batch_shape)``-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)  # mc_samples x batch_shape x q x m
        q = obj.shape[-2]
        if self.constraints is not None:
            log_feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                log=True,
                fat=self.fat,
            )
        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]  # mc_samples x batch_shape
        # areas tensor is ``mc_samples x batch_shape x num_cells x 2``-dim
        log_areas_per_segment = torch.full(
            size=(
                *batch_shape,
                self.cell_lower_bounds.shape[-2],  # num_cells
                2,  # for even and odd terms
            ),
            fill_value=-torch.inf,
            dtype=obj.dtype,
            device=device,
        )

        # The fused C++ kernel supports non-batched (qLogEHVI) and batched
        # (qLogNEHVI, fully Bayesian) cell bounds, with fat=True on CPU.
        # When obj has extra t-batch dims that broadcast against the cell
        # bounds, we expand cell bounds to match before flattening.
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        use_fused = _C is not None and self.fat and obj.device.type == "cpu"

        if use_fused and cell_batch_ndim > 0:
            # Expand cell bounds to match obj batch shape for 1:1 mapping.
            # Insert 1-dims at t-batch positions, then expand + reshape to 3-D.
            n_tbatch = len(batch_shape) - cell_batch_ndim
            if n_tbatch > 0:
                # cell shape: (mc, *model_batch, nc, m)
                # target:     (mc, *t_batch, *model_batch, nc, m)
                expand_shape = (
                    self.cell_lower_bounds.shape[0],  # mc_samples
                    *[1] * n_tbatch,  # 1s for t-batch dims
                    *self.cell_lower_bounds.shape[1:],  # model_batch + nc + m
                )
                fused_cell_lower = self.cell_lower_bounds.view(expand_shape).expand(
                    *batch_shape, *self.cell_lower_bounds.shape[-2:]
                )
                fused_cell_upper = self.cell_upper_bounds.view(expand_shape).expand(
                    *batch_shape, *self.cell_upper_bounds.shape[-2:]
                )
            else:
                fused_cell_lower = self.cell_lower_bounds
                fused_cell_upper = self.cell_upper_bounds
        elif use_fused:
            fused_cell_lower = self.cell_lower_bounds
            fused_cell_upper = self.cell_upper_bounds

        if not use_fused:
            # conditionally adding mc_samples dim if cell_batch_ndim > 0
            # adding ones to shape equal in number to batch_shape_ndim -
            # cell_batch_ndim
            # adding cell_bounds batch shape w/o 1st dimension
            sample_batch_view_shape = torch.Size(
                [
                    batch_shape[0] if cell_batch_ndim > 0 else 1,
                    *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                    *self.cell_lower_bounds.shape[1:-2],
                ]
            )
            view_shape = (
                *sample_batch_view_shape,
                self.cell_upper_bounds.shape[-2],  # num_cells
                1,  # adding for q_choose_i dimension
                self.cell_upper_bounds.shape[-1],  # num_objectives
            )

        for i in range(1, self.q_out + 1):
            q_choose_i = q_subset_indices[f"q_choose_{i}"]  # q_choose_i x i
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )  # mc_samples x batch_shape x q_choose_i x i x m

            if use_fused:
                # Fused C++ kernel: steps 1-4 in one pass.
                # Input:  (*batch, n_subsets, i, m)
                # Output: (*batch, num_cells, n_subsets)
                log_areas_i = _FusedLogAreas.apply(
                    obj_subsets,
                    fused_cell_lower,
                    fused_cell_upper,
                    self.tau_relu,
                    self.tau_max,
                )
            else:
                # Pure-Python path (steps 1-4).
                # 1) computes log smoothed improvement over cell lower bounds.
                log_improvement_i = self._log_improvement(obj_subsets, view_shape)

                # 2) take the minimum log improvement over all i subsets.
                log_improvement_i = self._smooth_min(
                    log_improvement_i,
                    dim=-2,
                )

                # 3) compute the log lengths of the cells' sides.
                log_lengths_i = self._log_cell_lengths(log_improvement_i, view_shape)

                # 4) take product over hyperrectangle side lengths.
                log_areas_i = log_lengths_i.sum(dim=-1)

            # 5) if constraints are present, apply a differentiable approximation of
            # the indicator function.
            if self.constraints is not None:
                log_feas_subsets = log_feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(log_feas_weights.shape[:-1] + q_choose_i.shape)
                log_areas_i = log_areas_i + log_feas_subsets.unsqueeze(-3).sum(dim=-1)

            # 6) sum over all subsets of size i, i.e. reduce over q_choose_i-dim
            # after, log_areas_i is mc_samples x batch_shape x num_cells
            log_areas_i = logsumexp(log_areas_i, dim=-1)  # areas_i.sum(dim=-1)

            # 7) Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            log_areas_per_segment[..., i % 2] = logplusexp(
                log_areas_per_segment[..., i % 2],
                log_areas_i,
            )

        # 8) subtract even from odd log area terms
        log_areas_per_segment = logdiffexp(
            log_a=log_areas_per_segment[..., 0], log_b=log_areas_per_segment[..., 1]
        )

        # 9) sum over segments (n_cells-dim) and average over MC samples
        return logmeanexp(logsumexp(log_areas_per_segment, dim=-1), dim=0)

    def _log_improvement(
        self, obj_subsets: Tensor, view_shape: tuple | torch.Size
    ) -> Tensor:
        # smooth out the clamp and take the log (previous step 3)
        # subtract cell lower bounds, clamp min at zero, but first
        # make obj_subsets broadcastable with cell bounds:
        # mc_samples x batch_shape x (num_cells = 1) x q_choose_i x i x m
        obj_subsets = obj_subsets.unsqueeze(-4)
        # making cell bounds broadcastable with obj_subsets:
        # (mc_samples = 1) x (batch_shape = 1) x num_cells x 1 x (i = 1) x m
        cell_lower_bounds = self.cell_lower_bounds.view(view_shape).unsqueeze(-3)
        Z = obj_subsets - cell_lower_bounds
        log_Zi = self._log_smooth_relu(Z)
        return log_Zi  # mc_samples x batch_shape x num_cells x q_choose_i x i x m

    def _log_cell_lengths(
        self, log_improvement_i: Tensor, view_shape: tuple | torch.Size
    ) -> Tensor:
        cell_upper_bounds = self.cell_upper_bounds.clamp_max(
            1e10 if log_improvement_i.dtype == torch.double else 1e8
        )  # num_cells x num_objectives
        # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
        log_cell_lengths = (
            (cell_upper_bounds - self.cell_lower_bounds).log().view(view_shape)
        )  # (mc_samples = 1) x (batch_shape = 1) x n_cells x (q_choose_i = 1) x m
        # mc_samples x batch_shape x num_cells x q_choose_i x m
        return self._smooth_minimum(
            log_improvement_i,
            log_cell_lengths,
        )

    def _log_smooth_relu(self, X: Tensor) -> Tensor:
        f = log_fatplus if self.fat else log_softplus
        return f(X, tau=self.tau_relu)

    def _smooth_min(self, X: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        f = fatmin if self.fat else smooth_amin
        return f(X, tau=self.tau_max, dim=dim)

    def _smooth_minimum(self, X: Tensor, Y: Tensor) -> Tensor:
        XY = torch.stack(torch.broadcast_tensors(X, Y), dim=-1)
        return self._smooth_min(XY, dim=-1, keepdim=False)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        return self._compute_log_qehvi(samples=samples, X=X)


class qLogNoisyExpectedHypervolumeImprovement(
    NoisyExpectedHypervolumeMixin,
    qLogExpectedHypervolumeImprovement,
):
    _log: bool = True

    def __init__(
        self,
        model: Model,
        ref_point: list[float] | Tensor,
        X_baseline: Tensor,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool | None = None,
        tau_relu: float = TAU_RELU,
        tau_max: float = 1e-3,  # TAU_MAX,
        fat: bool = True,
        marginalize_dim: int | None = None,
    ) -> None:
        r"""
        q-Log Noisy Expected Hypervolume Improvement supporting m>=2 outcomes.

        Based on the differentiable hypervolume formulation of [Daulton2021nehvi]_.

        Example:
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> ref_point = [0.0, 0.0]
            >>> qNEHVI = qLogNoisyExpectedHypervolumeImprovement(
            ...     model, ref_point, train_X
            ... )
            >>> qnehvi = qNEHVI(test_X)

        Args:
            model: A fitted model.
            ref_point: A list or tensor with ``m`` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (i.e. after
                applying ``objective`` to the samples).
            X_baseline: A ``r x d``-dim Tensor of ``r`` design points that have already
                been observed. These points are considered as potential approximate
                pareto-optimal design points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using ``get_sampler``.
                Note: a pareto front is created for each mc sample, which can be
                computationally intensive for ``m`` > 2.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to ``IdentityMultiOutputObjective()``.
            constraints: A list of callables, each mapping a Tensor of dimension
                ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                ``sample_shape x batch-shape x q``, where negative values imply
                feasibility. The acquisition function will compute expected feasible
                hypervolume.
            X_pending: A ``batch_shape x m x d``-dim Tensor of ``m`` design points that
                have points that have been submitted for function evaluation, but
                have not yet been evaluated.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints. In case of a float the
                same ``eta`` is used for every constraint in constraints. In case of a
                tensor the length of the tensor must match the number of provided
                constraints. The i-th constraint is then estimated with the i-th
                ``eta`` value.
            prune_baseline: If True, remove points in ``X_baseline`` that are
                highly unlikely to be the pareto optimal and better than the
                reference point. This can significantly improve computation time and
                is generally recommended. In order to customize pruning parameters,
                instead manually call ``prune_inferior_points_multi_objective`` on
                ``X_baseline`` before instantiating the acquisition function.
            alpha: The hyperparameter controlling the approximate non-dominated
                partitioning. The default value of 0.0 means an exact partitioning
                is used. As the number of objectives ``m`` increases, consider
                increasing this parameter in order to limit computational complexity.
            cache_pending: A boolean indicating whether to use cached box
                decompositions (CBD) for handling pending points. This is
                generally recommended.
            max_iep: The maximum number of pending points before the box
                decompositions will be recomputed.
            incremental_nehvi: A boolean indicating whether to compute the
                incremental NEHVI from the ``i``th point where ``i=1, ..., q``
                under sequential greedy optimization, or the full qNEHVI over
                ``q`` points.
            cache_root: A boolean indicating whether to cache the root
                decomposition over ``X_baseline`` and use low-rank updates.
            marginalize_dim: A batch dimension that should be marginalized.
        """
        MultiObjectiveMCAcquisitionFunction.__init__(
            self,
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
        )
        SubsetIndexCachingMixin.__init__(self)
        NoisyExpectedHypervolumeMixin.__init__(
            self,
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )
        # parameters that are used by qLogEHVI
        self.tau_relu = tau_relu
        self.tau_max = tau_max
        self.fat = fat
        _try_load_fused_kernel()

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        # Get samples from the posterior, and manually concatenate pending points that
        # have not yet been cached. Shared with qNEHVI.
        samples, X = self._compute_posterior_samples_and_concat_pending(X)
        # Add previous nehvi from pending points.
        nehvi = self._compute_log_qehvi(samples=samples, X=X)
        if self.incremental_nehvi:
            return nehvi
        return logplusexp(nehvi, self._prev_nehvi.log())
