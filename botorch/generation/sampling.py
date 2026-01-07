#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampling-based generation strategies.

A SamplingStrategy returns samples from the input points (i.e. Tensors in feature
space), rather than the value for a set of tensors, as acquisition functions do.
The q-batch dimension has similar semantics as for acquisition functions in that the
points across the q-batch are considered jointly for sampling (where as for
q-acquisition functions we evaluate the joint value of the q-batch).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import batched_multinomial
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.nn import Module


class SamplingStrategy(Module, ABC):
    """Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times.
        # The contiguous call is needed due to a pytorch issue:
        # https://github.com/pytorch/pytorch/issues/151978
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1)).contiguous()
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)


class BoltzmannSampling(SamplingStrategy):
    r"""Sample from a set of points according to a tempered acquisition value.

    Given an acquisition function `acq_func`, this sampling strategies draws
    samples from a `batch_shape x N x d`-dim tensor `X` according to a multinomial
    distribution over its indices given by

        weight(X[..., i, :]) ~ exp(eta * standardize(acq_func(X[..., i, :])))

    where `standardize(Y)` standardizes `Y` to zero mean and unit variance. As the
    temperature parameter `eta -> 0`, this approaches uniform sampling, while as
    `eta -> infty`, this approaches selecting the maximizer(s) of the acquisition
    function `acq_func`.

    Example:
        >>> UCB = UpperConfidenceBound(model, beta=0.1)
        >>> BMUCB = BoltzmannSampling(UCB, eta=0.5)
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = BMUCB(X, num_samples=5)
    """

    def __init__(
        self, acq_func: AcquisitionFunction, eta: float = 1.0, replacement: bool = True
    ) -> None:
        r"""Boltzmann Acquisition Value Sampling.

        Args:
            acq_func: The acquisition function; to be evaluated in batch at the
                individual points of a q-batch (not jointly, as is the case for
                acquisition functions). Can be analytic or Monte-Carlo.
            eta: The temperature parameter in the softmax.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.acq_func = acq_func
        self.eta = eta
        self.replacement = replacement

    def forward(self, X: Tensor, num_samples: int = 1) -> Tensor:
        r"""Sample from a tempered value of the acquisition function value.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
                Note that if a batched model is used in the underlying acquisition
                function, then its batch shape must be broadcastable to `batch_shape`.
            num_samples: The number of samples to draw.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        # TODO: Can we get the model batch shape property from the model?
        # we move the `N` dimension to the front for evaluating the acquisition function
        # so that X_eval has shape `N x batch_shape x 1 x d`
        X_eval = X.permute(-2, *range(X.ndim - 2), -1).unsqueeze(-2)
        acqval = self.acq_func(X_eval)  # N x batch_shape
        # now move the `N` dimension back (this is the number of categories)
        acqval = acqval.permute(*range(1, X.ndim - 1), 0)  # batch_shape x N
        weights = torch.exp(self.eta * standardize(acqval))  # batch_shape x N
        idcs = batched_multinomial(
            weights=weights, num_samples=num_samples, replacement=self.replacement
        )
        # now do some gathering acrobatics to select the right elements from X
        return torch.gather(X, -2, idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1)))


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: ModelListGP | MultiTaskGP,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.constraint_model = constraint_model

    def _convert_samples_to_scores(self, Y_samples, C_samples) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = -total_violation
        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            # Note: `posterior_transform` is only used for the objective
            posterior_transform=self.posterior_transform,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Loop over the constraint models (if possible) to reduce peak memory usage.
        constraint_models = (
            self.constraint_model.models
            if isinstance(self.constraint_model, ModelListGP)
            else [self.constraint_model]
        )
        C_samples_list = []
        for c_model in constraint_models:
            c_posterior = c_model.posterior(X=X, observation_noise=observation_noise)
            C_samples_list.append(
                c_posterior.rsample(sample_shape=torch.Size([num_samples]))
            )
        C_samples = torch.cat(C_samples_list, dim=-1)

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples, C_samples=C_samples
        )
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)
