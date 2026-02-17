#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import botorch.models.model as model
import torch
from botorch.logging import _get_logger
from botorch.utils.sampling import manual_seed, unnormalize
from torch import Tensor


logger = _get_logger(name="Feasibility")


def get_feasible_samples(
    samples: Tensor,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
) -> tuple[Tensor, float]:
    r"""
    Checks which of the samples satisfy all of the inequality constraints.

    Args:
        samples: A ``sample size x d`` size tensor of feature samples,
            where d is a feature dimension.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
    Returns:
        2-element tuple containing

        - Samples satisfying the linear constraints.
        - Estimated proportion of samples satisfying the linear constraints.
    """

    if inequality_constraints is None:
        return samples, 1.0

    nsamples = samples.size(0)

    feasible = torch.ones(nsamples, device=samples.device, dtype=torch.bool)

    for indices, coefficients, rhs in inequality_constraints:
        lhs = samples.index_select(1, indices) @ coefficients.to(dtype=samples.dtype)
        feasible &= lhs >= rhs

    feasible_samples = samples[feasible]

    p_linear = feasible_samples.size(0) / nsamples

    return feasible_samples, p_linear


def get_outcome_feasibility_probability(
    model: model.Model,
    X: Tensor,
    outcome_constraints: list[Callable[[Tensor], Tensor]],
    threshold: float = 0.1,
    nsample_outcome: int = 1000,
    seed: int | None = None,
) -> float:
    r"""
    Monte Carlo estimate of the feasible volume with respect to the outcome constraints.

    Args:
        model: The model used for sampling the posterior.
        X: A tensor of dimension ``batch-shape x 1 x d``, where d is feature dimension.
        outcome_constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply feasibility.
        threshold: A lower limit for the probability of posterior samples feasibility.
        nsample_outcome: The number of samples from the model posterior.
        seed: The seed for the posterior sampler. If omitted, use a random seed.

    Returns:
        Estimated proportion of features for which posterior samples satisfy
        given outcome constraints with probability above or equal to
        the given threshold.
    """
    if outcome_constraints is None:
        return 1.0

    from botorch.sampling.get_sampler import get_sampler

    seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    posterior = model.posterior(X)  # posterior consists of batch_shape marginals
    sampler = get_sampler(
        posterior=posterior, sample_shape=torch.Size([nsample_outcome]), seed=seed
    )
    # size of samples: (num outcome samples, batch_shape, 1, outcome dim)
    samples = sampler(posterior)

    feasible = torch.ones(samples.shape[:-1], dtype=torch.bool, device=samples.device)

    # a sample passes if each constraint applied to the sample
    # produces a non-negative tensor
    for oc in outcome_constraints:
        # broadcasted evaluation of the outcome constraints
        feasible &= oc(samples) <= 0

    # proportion of feasibile samples for each of the elements of X
    # summation is done across feasible outcome samples
    p_feas = feasible.sum(0).float() / feasible.size(0)

    # proportion of features leading to the posterior outcome
    # satisfying the given outcome constraints
    # with at probability above a given threshold
    p_outcome = (p_feas >= threshold).sum().item() / X.size(0)

    return p_outcome


def estimate_feasible_volume(
    bounds: Tensor,
    model: model.Model,
    outcome_constraints: list[Callable[[Tensor], Tensor]],
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    nsample_feature: int = 1000,
    nsample_outcome: int = 1000,
    threshold: float = 0.1,
    verbose: bool = False,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    r"""
    Monte Carlo estimate of the feasible volume with respect
    to feature constraints and outcome constraints.

    Args:
        bounds: A ``2 x d`` tensor of lower and upper bounds
            for each column of ``X``.
        model: The model used for sampling the outcomes.
        outcome_constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply
            feasibility.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
        nsample_feature: The number of feature samples satisfying the bounds.
        nsample_outcome: The number of outcome samples from the model posterior.
        threshold: A lower limit for the probability of outcome feasibility
        seed: The seed for both feature and outcome samplers. If omitted,
            use a random seed.
        verbose: An indicator for whether to log the results.

    Returns:
        2-element tuple containing:

        - Estimated proportion of volume in feature space that is
            feasible wrt the bounds and the inequality constraints (linear).
        - Estimated proportion of feasible features for which
            posterior samples (outcome) satisfies the outcome constraints
            with probability above the given threshold.
    """

    seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    with manual_seed(seed=seed):
        samples_nlzd = torch.rand(
            (nsample_feature, bounds.size(1)), dtype=dtype, device=device
        )
        box_samples = unnormalize(samples_nlzd, bounds, update_constant_bounds=False)

    features, p_feature = get_feasible_samples(
        samples=box_samples, inequality_constraints=inequality_constraints
    )  # each new feature sample is a row

    p_outcome = get_outcome_feasibility_probability(
        model=model,
        X=features.unsqueeze(-2),
        outcome_constraints=outcome_constraints,
        threshold=threshold,
        nsample_outcome=nsample_outcome,
        seed=seed,
    )

    if verbose:  # pragma: no cover
        logger.info(
            "Proportion of volume that satisfies linear constraints: "
            + f"{p_feature:.4e}"
        )
        if p_feature <= 0.01:
            logger.warning(
                "The proportion of satisfying volume is very low and may lead to "
                + "very long run times. Consider making your constraints less "
                + "restrictive."
            )
        logger.info(
            "Proportion of linear-feasible volume that also satisfies each "
            + f"outcome constraint with probability > 0.1: {p_outcome:.4e}"
        )
        if p_outcome <= 0.001:
            logger.warning(
                "The proportion of volume that also satisfies the outcome constraint "
                + "is very low. Consider making your parameter and outcome constraints "
                + "less restrictive."
            )
    return p_feature, p_outcome
