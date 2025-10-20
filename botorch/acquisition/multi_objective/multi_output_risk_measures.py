#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-output extensions of the risk measures, implemented as Monte-Carlo
objectives. Except for MVaR, the risk measures are computed over each
output dimension independently. In contrast, MVaR is computed using the
joint distribution of the outputs, and provides more accurate risk estimates.

References

.. [Prekopa2012MVaR]
    A. Prekopa. Multivariate value at risk and related topics.
    Annals of Operations Research, 2012.

.. [Cousin2013MVaR]
    A. Cousin and E. Di Bernardino. On multivariate extensions of Value-at-Risk.
    Journal of Multivariate Analysis, 2013.

.. [Daulton2022MARS]
    S. Daulton, S, Cakmak, M. Balandat, M. Osborne, E. Zhou, and E. Bakshy.
    Robust multi-objective Bayesian optimization under input noise.
    Proceedings of the 39th International Conference on Machine Learning, 2022.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from math import ceil

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.risk_measures import CVaR, RiskMeasureMCObjective, VaR
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize
from torch import Tensor


class MultiOutputRiskMeasureMCObjective(
    RiskMeasureMCObjective, MCMultiOutputObjective, ABC
):
    r"""Objective transforming the multi-output posterior samples to samples
    of a multi-output risk measure.

    The risk measure is calculated over joint q-batch samples from the posterior.
    If the q-batch includes samples corresponding to multiple inputs, it is assumed
    that first `n_w` samples correspond to first input, second `n_w` samples
    correspond to second input, etc.
    """

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure. This can be used to
                remove non-objective outcomes or to align all outcomes for
                maximization. For constrained optimization, this should also
                apply feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch x m'`-dim tensor.
        """
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)

    def _prepare_samples(self, samples: Tensor) -> Tensor:
        r"""Prepare samples for risk measure calculations by scaling and
        separating out the q-batch dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.

        Returns:
            A `sample_shape x batch_shape x q x n_w x m'`-dim tensor of
            prepared samples.
        """
        samples = self.preprocessing_function(samples)
        return samples.view(*samples.shape[:-2], -1, self.n_w, samples.shape[-1])

    @abstractmethod
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the risk measure corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of risk measure samples.
        """
        pass  # pragma: no cover


class MultiOutputExpectation(MultiOutputRiskMeasureMCObjective):
    r"""A multi-output MC expectation risk measure.

    For unconstrained problems, we recommend using the `ExpectationPosteriorTransform`
    instead. `ExpectationPosteriorTransform` directly transforms the posterior
    distribution over `q * n_w` to a posterior of `q` expectations, significantly
    reducing the cost of posterior sampling as a result.
    """

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the expectation of the given samples. Expectation is
        calculated over each `n_w` samples in the q-batch dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of expectation samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.mean(dim=-2)


class IndependentCVaR(CVaR, MultiOutputRiskMeasureMCObjective):
    r"""The multi-output Conditional Value-at-Risk risk measure that operates on
    each output dimension independently. Since this does not consider the joint
    distribution of the outputs (i.e., that the outputs were evaluated on same
    perturbed input and are not independent), the risk estimates provided by
    `IndependentCVaR` in general are more optimistic than the definition of CVaR
    would suggest.

    The Conditional Value-at-Risk measures the expectation of the worst outcomes
    (small rewards or large losses) with a total probability of `1 - alpha`. It
    is commonly defined as the conditional expectation of the reward function,
    with the condition that the reward is smaller than the corresponding
    Value-at-Risk (also defined below).

    NOTE: Due to the use of a discrete `w_set` of samples, the VaR and CVaR
    calculated here are (possibly biased) Monte-Carlo approximations of the
    true risk measures.
    """

    def _get_sorted_prepared_samples(self, samples: Tensor) -> Tensor:
        r"""Get the prepared samples that are sorted over the `n_w` dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.

        Returns:
            A `sample_shape x batch_shape x q x n_w x m'`-dim tensor of sorted samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.sort(dim=-2, descending=True).values

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the CVaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of CVaR samples.
        """
        sorted_samples = self._get_sorted_prepared_samples(samples)
        return sorted_samples[..., self.alpha_idx :, :].mean(dim=-2)


class IndependentVaR(IndependentCVaR):
    r"""The multi-output Value-at-Risk risk measure that operates on each output
    dimension independently. For the same reasons as `IndependentCVaR`, the risk
    estimates provided by this are in general more optimistic than the definition
    of VaR would suggest.

    Value-at-Risk measures the smallest possible reward (or largest possible loss)
    after excluding the worst outcomes with a total probability of `1 - alpha`. It
    is commonly used in financial risk management, and it corresponds to the
    `1 - alpha` quantile of a given random variable.
    """

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the VaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of VaR samples.
        """
        sorted_samples = self._get_sorted_prepared_samples(samples)
        return sorted_samples[..., self.alpha_idx, :]


class MultiOutputWorstCase(MultiOutputRiskMeasureMCObjective):
    r"""The multi-output worst-case risk measure."""

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the worst-case measure corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of worst-case samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.min(dim=-2).values


class MVaR(MultiOutputRiskMeasureMCObjective):
    r"""The multivariate Value-at-Risk as introduced in [Prekopa2012MVaR]_.

    MVaR is defined as the non-dominated set of points in the extended domain
    of the random variable that have multivariate CDF greater than or equal to
    `alpha`. Note that MVaR is set valued and the size of the set depends on the
    particular realizations of the random variable. [Cousin2013MVaR]_ instead
    propose to use the expectation of the set-valued MVaR as the multivariate
    VaR. We support this alternative with an `expectation` flag.

    This supports approximate gradients as discussed in [Daulton2022MARS]_.
    """

    _verify_output_shape = False

    def __init__(
        self,
        n_w: int,
        alpha: float,
        expectation: bool = False,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
        *,
        pad_to_n_w: bool = False,
        filter_dominated: bool = True,
        use_counting: bool = False,
    ) -> None:
        r"""The multivariate Value-at-Risk.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            alpha: The risk level of MVaR, float in `(0.0, 1.0]`. Each MVaR value
                dominates `alpha` fraction of all observations.
            expectation: If True, returns the expectation of the MVaR set as is
                done in [Cousin2013MVaR]_. Otherwise, it returns the union of all
                values in the MVaR set. Default: False.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure. This can be used to
                remove non-objective outcomes or to align all outcomes for
                maximization. For constrained optimization, this should also
                apply feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch x m'`-dim tensor.
            pad_to_n_w: If True, instead of padding up to `k'`, which is the size of
                the largest MVaR set across all batches, we pad the MVaR set up to
                `n_w`. This produces a return tensor of known size, however, it may
                in general be much larger than the alternative. See `forward` for
                more details on the return shape.
                NOTE: this is only relevant if `expectation=False`.
            filter_dominated: If True, returns the non-dominated subset of
                alpha level points (this is MVaR as defined by [Prekopa2012MVaR]_).
                Disabling this will make it faster, and may be preferable if
                the dominated points will be filtered out later, e.g., while
                calculating the hypervolume. Disabling this is not recommended
                if `expectation=True`.
            use_counting: If True, uses `get_mvar_set_via_counting` for finding the
                MVaR set. This is method is less memory intensive than the vectorized
                implementation, which is beneficial when `n_w` is quite large.
        """
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        if not 0 < alpha <= 1:
            raise ValueError("`alpha` must be in (0.0, 1.0]")
        self.alpha = alpha
        self.expectation = expectation
        self.pad_to_n_w = pad_to_n_w
        self.filter_dominated = filter_dominated
        self.use_counting = use_counting

    def get_mvar_set_via_counting(self, Y: Tensor) -> list[Tensor]:
        r"""Find MVaR set based on the definition in [Prekopa2012MVaR]_.

        This first calculates the CDF for each point on the extended domain of the
        random variable (the grid defined by the given samples), then takes the
        values with CDF equal to (rounded if necessary) `alpha`. The non-dominated
        subset of these form the MVaR set.

        This implementation processes each batch of `Y` in a for loop using a counting
        based implementation. It requires less memory than the vectorized implementation
        and should be used with large (>128) `n_w` values.

        Args:
            Y: A `batch x n_w x m`-dim tensor of outcomes.

        Returns:
            A `batch` length list of `k x m`-dim tensor of MVaR values, where `k`
            depends on the corresponding batch inputs. Note that MVaR values in general
            are not in-sample points.
        """
        if Y.dim() == 3:
            return sum((self.get_mvar_set_via_counting(y_) for y_ in Y), [])
        m = Y.shape[-1]
        # Generate sets of all unique values in each output dimension.
        # Note that points in MVaR are bounded from above by the
        # independent VaR of each objective. Hence, we only need to
        # consider the unique outcomes that are less than or equal to
        # the VaR of the independent objectives
        var_alpha_idx = ceil(self.alpha * self.n_w) - 1
        Y_sorted = Y.topk(Y.shape[0] - var_alpha_idx, dim=0, largest=False).values
        unique_outcomes_list = []
        for i in range(m):
            sorted_i = Y_sorted[:, i].cpu().clone(memory_format=torch.contiguous_format)
            unique_outcomes_list.append(sorted_i.unique().tolist()[::-1])
        # Convert this into a list of m dictionaries mapping values to indices.
        unique_outcomes = [
            dict(zip(outcomes, range(len(outcomes))))
            for outcomes in unique_outcomes_list
        ]
        # Initialize a tensor counting the number of points in Y that a given grid point
        # is dominated by. This will essentially be a non-normalized CDF.
        counter_tensor = torch.zeros(
            [len(outcomes) for outcomes in unique_outcomes],
            dtype=torch.long,
            device=Y.device,
        )
        # populate the tensor, counting the dominated points.
        # we only need to consider points in Y where at least one
        # objective is less than the max objective value in
        # unique_outcomes_list
        max_vals = torch.tensor(
            [o[0] for o in unique_outcomes_list], dtype=Y.dtype, device=Y.device
        )
        mask = (Y < max_vals).any(dim=-1)
        counter_tensor += self.n_w - mask.sum()
        Y_pruned = Y[mask]
        for y_ in Y_pruned:
            starting_idcs = [unique_outcomes[i].get(y_[i].item(), 0) for i in range(m)]
            slices = tuple(slice(s_idx, None) for s_idx in starting_idcs)
            counter_tensor[slices] += 1

        # Get the count alpha-level points should have.
        alpha_count = ceil(self.alpha * self.n_w)
        # Get the alpha level indices.
        alpha_level_indices = (counter_tensor == alpha_count).nonzero(as_tuple=False)
        # If there are no exact alpha level points, get the smallest alpha' > alpha
        # and find the corresponding alpha level indices.
        if alpha_level_indices.numel() == 0:
            min_greater_than_alpha = counter_tensor[counter_tensor > alpha_count].min()
            alpha_level_indices = (counter_tensor == min_greater_than_alpha).nonzero(
                as_tuple=False
            )
        unique_outcomes = [
            torch.as_tensor(list(outcomes.keys()), device=Y.device, dtype=Y.dtype)
            for outcomes in unique_outcomes
        ]
        alpha_level_points = torch.stack(
            [
                unique_outcomes[i][alpha_level_indices[:, i]]
                for i in range(len(unique_outcomes))
            ],
            dim=-1,
        )
        # MVaR is simply the non-dominated subset of alpha level points.
        if self.filter_dominated:
            mask = is_non_dominated(alpha_level_points)
            mvar = alpha_level_points[mask]
        else:
            mvar = alpha_level_points
        return [mvar]

    def get_mvar_set_vectorized(self, Y: Tensor) -> list[Tensor]:
        r"""Find MVaR set based on the definition in [Prekopa2012MVaR]_.

        This first calculates the CDF for each point on the extended domain of the
        random variable (the grid defined by the given samples), then takes the
        values with CDF equal to (rounded if necessary) `alpha`. The non-dominated
        subset of these form the MVaR set.

        This implementation uses computes the CDF of each point using highly vectorized
        operations. As such, it may use large amounts of memory, particularly when the
        batch size and/or `n_w` are large. It is typically faster than the alternative
        implementation when computing MVaR of a large batch of points with small to
        moderate (<128 for m=2, <64 for m=3) `n_w`.

        Args:
            Y: A `batch x n_w x m`-dim tensor of observations.

        Returns:
            A `batch` length list of `k x m`-dim tensor of MVaR values, where `k`
            depends on the corresponding batch inputs. Note that MVaR values in general
            are not in-sample points.
        """
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
        batch, m = Y.shape[0], Y.shape[-1]
        # Note that points in MVaR are bounded from above by the
        # independent VaR of each objective. Hence, we only need to
        # consider the unique outcomes that are less than or equal to
        # the VaR of the independent objectives
        var_alpha_idx = ceil(self.alpha * self.n_w) - 1
        n_points = Y.shape[-2] - var_alpha_idx
        Y_sorted = Y.topk(n_points, dim=-2, largest=False).values
        # `y_grid` is the grid formed by all inputs in each batch.
        if m == 2:
            # This is significantly faster but only works with m=2.
            y_grid = torch.stack(
                [
                    Y_sorted[..., 0].repeat_interleave(repeats=n_points, dim=-1),
                    Y_sorted[..., 1].repeat(1, n_points),
                ],
                dim=-1,
            )
        else:
            y_grid = torch.stack(
                [
                    torch.stack(
                        torch.meshgrid(
                            [Y_sorted[b, :, i] for i in range(m)], indexing="ij"
                        ),
                        dim=-1,
                    ).view(-1, m)
                    for b in range(batch)
                ],
                dim=0,
            )
        # Get the non-normalized CDF.
        cdf = (Y.unsqueeze(-2) >= y_grid.unsqueeze(-3)).all(dim=-1).sum(dim=-2)
        # Get the alpha level points
        alpha_count = ceil(self.alpha * self.n_w)
        # NOTE: Need to loop here since mvar may have different shapes.
        mvar = []
        for b in range(batch):
            alpha_level_points = y_grid[b][cdf[b] == alpha_count]
            # If there are no exact alpha level points, get the smallest alpha' > alpha
            # and find the corresponding alpha level indices.
            if alpha_level_points.numel() == 0:
                min_greater_than_alpha = cdf[b][cdf[b] > alpha_count].min()
                alpha_level_points = y_grid[b][cdf[b] == min_greater_than_alpha]
            # MVaR is the non-dominated subset of alpha level points.
            if self.filter_dominated:
                mask = is_non_dominated(alpha_level_points)
                mvar.append(alpha_level_points[mask])
            else:
                mvar.append(alpha_level_points)
        return mvar

    def make_differentiable(self, prepared_samples: Tensor, mvars: Tensor) -> Tensor:
        r"""An experimental approach for obtaining the gradient of the MVaR via
        component-wise mapping to original samples. See [Daulton2022MARS]_.

        Args:
            prepared_samples: A `(sample_shape * batch_shape * q) x n_w x m`-dim tensor
                of posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            mvars: A `(sample_shape * batch_shape * q) x k x m`-dim tensor
                of padded MVaR values.
        Returns:
            The same `mvars` with entries mapped to inputs to produce gradients.
        """
        samples = prepared_samples.unsqueeze(-2).repeat(1, 1, mvars.shape[-2], 1)
        mask = samples == mvars.unsqueeze(-3)
        samples[~mask] = 0
        return samples.sum(dim=-3) / mask.sum(dim=-3)

    def forward(
        self,
        samples: Tensor,
        X: Tensor | None = None,
    ) -> Tensor:
        r"""Calculate the MVaR corresponding to the given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim tensor of MVaR values,
            if `self.expectation=True`.
            Otherwise, this returns a `sample_shape x batch_shape x (q * k') x m'`-dim
            tensor, where `k'` is the maximum `k` across all batches that is returned
            by `get_mvar_set_...`. Each `(q * k') x m'` corresponds to the `k` MVaR
            values for each `q` batch of `n_w` inputs, padded up to `k'` by repeating
            the last element. If `self.pad_to_n_w`, we set `k' = self.n_w`, producing
            a deterministic return shape.
        """
        batch_shape, m = samples.shape[:-2], samples.shape[-1]
        prepared_samples = self._prepare_samples(samples)
        # This is -1 x n_w x m.
        prepared_samples = prepared_samples.reshape(-1, *prepared_samples.shape[-2:])
        with torch.no_grad():
            if self.use_counting:
                mvar_set = self.get_mvar_set_via_counting(prepared_samples)
            else:
                mvar_set = self.get_mvar_set_vectorized(prepared_samples)
        # Set the `pad_size` to either `self.n_w` or the size of the largest MVaR set.
        pad_size = self.n_w if self.pad_to_n_w else max([_.shape[0] for _ in mvar_set])
        padded_mvar_list = []
        for mvar_ in mvar_set:
            if self.expectation:
                padded_mvar_list.append(mvar_.mean(dim=0))
            else:
                # Repeat the last entry to make `mvar_set` `pad_size x m`.
                repeats_needed = pad_size - mvar_.shape[0]
                padded_mvar_list.append(
                    torch.cat([mvar_, mvar_[-1].expand(repeats_needed, m)], dim=0)
                )
        mvars = torch.stack(padded_mvar_list, dim=0)
        if samples.requires_grad:
            mvars = self.make_differentiable(
                prepared_samples=prepared_samples, mvars=mvars
            )
        return mvars.view(*batch_shape, -1, m)


class MARS(VaR, MultiOutputRiskMeasureMCObjective):
    r"""MVaR Approximation based on Random Scalarizations as introduced
    in [Daulton2022MARS]_.

    This approximates MVaR via VaR of Chebyshev scalarizations, where each
    scalarization corresponds to a point in the MVaR set. As implemented,
    this uses one set of scalarization weights to approximate a single MVaR value.
    Note that due to the normalization within the Chebyshev scalarization,
    the output of this risk measure may not be on the same scale as its inputs.
    """

    _is_mo: bool = False

    def __init__(
        self,
        alpha: float,
        n_w: int,
        chebyshev_weights: Tensor | list[float],
        baseline_Y: Tensor | None = None,
        ref_point: Tensor | list[float] | None = None,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            alpha: The risk level, float in `(0.0, 1.0]`.
            n_w: The size of the perturbation set to calculate the risk measure over.
            chebyshev_weights: The weights to use in the Chebyshev scalarization.
                The Chebyshev scalarization is applied before computing VaR.
                The weights must be non-negative. See `preprocessing_function` to
                support minimization objectives.
            baseline_Y: An `n' x d`-dim tensor of baseline outcomes to use in
                determining the normalization bounds for Chebyshev scalarization.
                It is recommended to set this via `set_baseline_Y` helper.
            ref_point: An optional MVaR reference point to use in determining
                the normalization bounds for Chebyshev scalarization.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure. This can be used to
                remove non-objective outcomes or to align all outcomes for
                maximization. For constrained optimization, this should also
                apply feasibility-weighting to samples.
        """
        if preprocessing_function is None:
            preprocessing_function = IdentityMCMultiOutputObjective()
        super().__init__(
            alpha=alpha,
            n_w=n_w,
            preprocessing_function=preprocessing_function,
        )
        self.chebyshev_weights = torch.as_tensor(chebyshev_weights)
        self.baseline_Y = baseline_Y
        self.register_buffer(
            "ref_point", torch.as_tensor(ref_point) if ref_point is not None else None
        )
        self.mvar = MVaR(n_w=self.n_w, alpha=self.alpha)
        self._chebyshev_objective = None

    def set_baseline_Y(
        self,
        model: Model | None,
        X_baseline: Tensor | None,
        Y_samples: Tensor | None = None,
    ) -> None:
        r"""Set the `baseline_Y` based on the MVaR predictions of the `model`
        for `X_baseline`.

        Args:
            model: The model being used for MARS optimization. Must have a compatible
                `InputPerturbation` transform attached. Ignored if `Y_samples` is given.
            X_baseline: An `n x d`-dim tensor of previously evaluated points.
                Ignored if `Y_samples` is given.
            Y_samples: An optional `(n * n_w) x d`-dim tensor of predictions. If given,
                instead of sampling from the model, these are used.
        """
        if Y_samples is None:
            with torch.no_grad():
                Y = model.posterior(X_baseline.unsqueeze(-2)).mean.squeeze(-2)
        else:
            if model is not None or X_baseline is not None:
                warnings.warn(
                    "`model` and `X_baseline` are ignored when `Y_samples` is "
                    "provided to `MARS.set_baseline_Y`.",
                    BotorchWarning,
                    stacklevel=2,
                )
            Y = Y_samples
        Y = self.preprocessing_function(Y)
        Y = self.mvar(Y).view(-1, Y.shape[-1])
        Y = Y[is_non_dominated(Y)]
        self.baseline_Y = Y

    @property
    def chebyshev_weights(self) -> Tensor:
        r"""The weights used in Chebyshev scalarization."""
        return self._chebyshev_weights

    @chebyshev_weights.setter
    def chebyshev_weights(self, chebyshev_weights: Tensor | list[float]) -> None:
        r"""Update the Chebyshev weights.

        Invalidates the cached Chebyshev objective.

        Args:
            chebyshev_weights: The weights to use in the Chebyshev scalarization.
                The Chebyshev scalarization is applied before computing VaR.
                The weights must be non-negative. See `preprocessing_function` to
                support minimization objectives.
        """
        self._chebyshev_objective = None
        chebyshev_weights = torch.as_tensor(chebyshev_weights)
        if torch.any(chebyshev_weights < 0):
            raise UnsupportedError("Negative weights are not supported in MARS.")
        if chebyshev_weights.dim() != 1:
            raise UnsupportedError("Batched weights are not supported in MARS.")
        self.register_buffer("_chebyshev_weights", chebyshev_weights)

    @property
    def baseline_Y(self) -> Tensor | None:
        r"""Baseline outcomes used in determining the normalization bounds."""
        return self._baseline_Y

    @baseline_Y.setter
    def baseline_Y(self, baseline_Y: Tensor | None) -> None:
        r"""Update the baseline outcomes.

        Invalidates the cached Chebyshev objective.

        Args:
            baseline_Y: An `n' x d`-dim tensor of baseline outcomes to use in
                determining the normalization bounds for Chebyshev scalarization.
                It is recommended to set this via `set_baseline_Y` helper.
        """
        self._chebyshev_objective = None
        self.register_buffer("_baseline_Y", baseline_Y)

    @property
    def chebyshev_objective(self) -> Callable[[Tensor, Tensor | None], Tensor]:
        r"""The objective for applying the Chebyshev scalarization."""
        if self._chebyshev_objective is None:
            self._construct_chebyshev_objective()
        return self._chebyshev_objective

    def _construct_chebyshev_objective(self) -> None:
        r"""Construct a Chebyshev scalarization. Outcomes are first normalized to [0,1],
        then the Chebyshev scalarization is applied.

        NOTE: This is a modified version of the `get_chebyshev_scalarization` helper.
        It doesn't support negative weights. All objectives should be aligned for
        maximization using `preprocessing_function`.
        """
        if self.baseline_Y is None:
            raise RuntimeError(
                "baseline_Y must be set before constructing the Chebyshev objective."
            )
        ref_point = self.ref_point
        if ref_point is not None:
            ref_point = ref_point.to(self.baseline_Y)
        Y_bounds = self._get_Y_normalization_bounds(
            Y=self.baseline_Y, ref_point=ref_point
        )
        if ref_point is not None:
            ref_point = normalize(ref_point.unsqueeze(0), bounds=Y_bounds).squeeze(0)

        def chebyshev_obj(Y: Tensor, X: Tensor | None = None) -> Tensor:
            Y = self.preprocessing_function(Y)
            Y = normalize(Y, bounds=Y_bounds)
            if ref_point is not None:
                Y = Y - ref_point
            product = torch.einsum("...m,m->...m", Y, self.chebyshev_weights.to(Y))
            return product.min(dim=-1).values

        self._chebyshev_objective = chebyshev_obj

    def _prepare_samples(self, samples: Tensor) -> Tensor:
        r"""Prepare samples for VaR computation by applying the Chebyshev scalarization
        and separating out the q-batch dimension.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.

        Returns:
            A `sample_shape x batch_shape x q x n_w`-dim tensor of prepared samples.
        """
        samples = self.chebyshev_objective(samples)
        return samples.view(*samples.shape[:-1], -1, self.n_w)

    @staticmethod
    def _get_Y_normalization_bounds(
        Y: Tensor,
        ref_point: Tensor | None = None,
    ) -> Tensor:
        r"""Get normalization bounds for scalarizations.

        Args:
            Y: A `n x m`-dim tensor of outcomes.
            ref_point: The reference point.

        Returns:
            A `2 x m`-dim tensor containing the normalization bounds.
        """
        if ref_point is not None:
            ref_point = ref_point.to(Y)

        if Y.ndim != 2:
            raise UnsupportedError("Batched Y is not supported.")

        if Y.shape[-2] == 0:
            # If there are no observations, return standard bounds.
            Y_bounds = torch.zeros(2, Y.shape[-1], dtype=Y.dtype, device=Y.device)
            Y_bounds[1] = 1.0
            return Y_bounds

        pareto_Y = Y[is_non_dominated(Y)]
        if pareto_Y.shape[-2] == 1:
            if ref_point is not None and (pareto_Y > ref_point).all():
                Y_bounds = torch.cat([ref_point.unsqueeze(0), pareto_Y], dim=0)
            else:
                # If there is only one observation, set the bounds to be [Y_m, Y_m + 1]
                # for each objective m. This ensures we do not divide by zero.
                Y_bounds = torch.cat([pareto_Y, pareto_Y + 1], dim=0)
        else:
            if ref_point is None:
                better_than_ref = torch.ones(
                    pareto_Y.shape[0], device=pareto_Y.device, dtype=torch.long
                )
            else:
                better_than_ref = (pareto_Y > ref_point).all(dim=-1)
            if ref_point is not None and better_than_ref.any():
                nadir = ref_point
                pareto_Y = pareto_Y[better_than_ref]
            else:
                nadir = pareto_Y.min(dim=-2).values
            ideal = pareto_Y.max(dim=-2).values
            Y_bounds = torch.stack([nadir, ideal])

        # If any of the lower bounds is equal to the upper bound, increase the
        # upper bound to prevent division by zero.
        Y_range = Y_bounds.max(dim=0).values - Y_bounds.min(dim=0).values
        mask = Y_range <= 0
        Y_bounds[1, mask] = Y_bounds[1, mask] + 1.0
        return Y_bounds
