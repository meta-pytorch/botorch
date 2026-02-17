#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Objective Modules to be used with acquisition functions."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.sampling import IIDNormalSampler
from botorch.utils import apply_constraints
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators.dense_linear_operator import to_linear_operator
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from botorch.posteriors.posterior import Posterior  # pragma: no cover
    from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover

DEFAULT_NUM_PREF_SAMPLES = 16


class PosteriorTransform(Module, ABC):
    """Abstract base class for objectives that transform the posterior."""

    @abstractmethod
    def evaluate(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A ``batch_shape x q x m``-dim tensor of outcomes.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly.

        Returns:
            A ``batch_shape x q' [x m']``-dim tensor of transformed outcomes.
        """
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, posterior: Posterior, X: Tensor | None = None) -> Posterior:
        r"""Compute the transformed posterior.

        Args:
            posterior: The posterior to be transformed.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly.

        Returns:
            The transformed posterior object.
        """
        pass  # pragma: no cover


# import DeterministicModel after PosteriorTransform to avoid circular import
from botorch.models.deterministic import DeterministicModel  # noqa


class ScalarizedPosteriorTransform(PosteriorTransform):
    r"""An affine posterior transform for scalarizing multi-output posteriors.

    For a Gaussian posterior at a single point (``q=1``) with mean ``mu`` and
    covariance matrix ``Sigma``, this yields a single-output posterior with mean
    ``weights^T * mu`` and variance ``weights^T Sigma w``.

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.5, 0.25])
        >>> posterior_transform = ScalarizedPosteriorTransform(weights)
        >>> EI = ExpectedImprovement(
        ... model, best_f=0.1, posterior_transform=posterior_transform
        ... )
    """

    scalarize: bool = True

    def __init__(self, weights: Tensor, offset: float = 0.0) -> None:
        r"""
        Args:
            weights: A one-dimensional tensor with ``m`` elements representing the
                linear weights on the outputs.
            offset: An offset to be added to posterior mean.
        """
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        super().__init__()
        self.register_buffer("weights", weights)
        self.offset = offset

    def evaluate(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the transform on a set of outcomes.

        Args:
            Y: A ``batch_shape x q x m``-dim tensor of outcomes.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly. Ignored here.

        Returns:
            A ``batch_shape x q``-dim tensor of transformed outcomes.
        """
        return self.offset + Y @ self.weights

    def forward(
        self, posterior: GPyTorchPosterior | PosteriorList, X: Tensor | None = None
    ) -> GPyTorchPosterior:
        r"""Compute the posterior of the affine transformation.

        Args:
            posterior: A posterior with the same number of outputs as the
                elements in ``self.weights``.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly. Ignored here.

        Returns:
            A single-output posterior.
        """
        return scalarize_posterior(
            posterior=posterior, weights=self.weights, offset=self.offset
        )


class ExpectationPosteriorTransform(PosteriorTransform):
    r"""Transform the ``batch x (q * n_w) x m`` posterior into a ``batch x q x m``
    posterior of the expectation. The expectation is calculated over each
    consecutive ``n_w`` block of points in the posterior.

    This is intended for use with ``InputPerturbation`` or ``AppendFeatures`` for
    optimizing the expectation over ``n_w`` points. This should not be used when
    there are constraints present, since this does not take into account
    the feasibility of the objectives.

    Note: This is different than ``ScalarizedPosteriorTransform`` in that
    this operates over the q-batch dimension.
    """

    def __init__(self, n_w: int, weights: Tensor | None = None) -> None:
        r"""A posterior transform calculating the expectation over the q-batch
        dimension.

        Args:
            n_w: The number of points in the q-batch of the posterior to compute
                the expectation over. This corresponds to the size of the
                ``feature_set`` of ``AppendFeatures`` or the size of the
                ``perturbation_set`` of ``InputPerturbation``.
            weights: An optional ``n_w x m``-dim tensor of weights. Can be used to
                compute a weighted expectation. Weights are normalized before use.
        """
        super().__init__()
        if weights is not None:
            if weights.dim() != 2 or weights.shape[0] != n_w:
                raise ValueError("`weights` must be a tensor of size `n_w x m`.")
            if torch.any(weights < 0):
                raise ValueError("`weights` must be non-negative.")
        else:
            weights = torch.ones(n_w, 1)
        # Normalize the weights.
        weights = weights / weights.sum(dim=0)
        self.register_buffer("weights", weights)
        self.n_w = n_w

    def evaluate(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the expectation of a set of outcomes.

        Args:
            Y: A ``batch_shape x (q * n_w) x m``-dim tensor of outcomes.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly. Ignored here.

        Returns:
            A ``batch_shape x q x m``-dim tensor of expectation outcomes.
        """
        batch_shape, m = Y.shape[:-2], Y.shape[-1]
        weighted_Y = Y.view(*batch_shape, -1, self.n_w, m) * self.weights.to(Y)
        return weighted_Y.sum(dim=-2)

    def forward(
        self, posterior: GPyTorchPosterior, X: Tensor | None = None
    ) -> GPyTorchPosterior:
        r"""Compute the posterior of the expectation.

        Args:
            posterior: An ``m``-outcome joint posterior over ``q * n_w`` points.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the transform depends on the inputs explicitly. Ignored here.

        Returns:
            An ``m``-outcome joint posterior over ``q`` expectations.
        """
        org_mvn = posterior.distribution
        if getattr(org_mvn, "_interleaved", False):
            raise UnsupportedError(
                "`ExpectationPosteriorTransform` does not support "
                "interleaved posteriors."
            )
        # Initialize the weight matrix of shape compatible with the mvn.
        org_event_shape = org_mvn.event_shape
        batch_shape = org_mvn.batch_shape
        q = org_event_shape[0] // self.n_w
        m = 1 if len(org_event_shape) == 1 else org_event_shape[-1]
        tkwargs = {"device": org_mvn.loc.device, "dtype": org_mvn.loc.dtype}
        weights = torch.zeros(q * m, q * self.n_w * m, **tkwargs)
        # Make sure self.weights has the correct dtype/device and shape.
        self.weights = self.weights.to(org_mvn.loc).expand(self.n_w, m)
        # Fill in the non-zero entries of the weight matrix.
        # We want each row to have non-zero weights for the corresponding
        # ``n_w`` sized diagonal. The ``m`` outcomes are not interleaved.
        for i in range(q * m):
            weights[i, self.n_w * i : self.n_w * (i + 1)] = self.weights[:, i // q]
        # Trasform the mean.
        new_loc = (
            (weights @ org_mvn.loc.unsqueeze(-1))
            .view(*batch_shape, m, q)
            .transpose(-1, -2)
        )
        # Transform the covariance matrix.
        org_cov = (
            org_mvn.lazy_covariance_matrix
            if org_mvn.islazy
            else org_mvn.covariance_matrix
        )
        new_cov = weights @ (org_cov @ weights.t())
        if m == 1:
            new_mvn = MultivariateNormal(
                new_loc.squeeze(-1), to_linear_operator(new_cov)
            )
        else:
            # Using MTMVN since we pass a single loc and covar for all ``m`` outputs.
            new_mvn = MultitaskMultivariateNormal(
                new_loc, to_linear_operator(new_cov), interleaved=False
            )
        return GPyTorchPosterior(distribution=new_mvn)


class MCAcquisitionObjective(Module, ABC):
    r"""Abstract base class for MC-based objectives.

    Args:
        _verify_output_shape: If True and ``X`` is given, check that the q-batch
            shape of the objectives agrees with that of X.
        _is_mo: A boolean denoting whether the objectives are multi-output.
    """

    _verify_output_shape: bool = True
    _is_mo: bool = False

    @abstractmethod
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim Tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            Tensor: A ``sample_shape x batch_shape x q``-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives.

        Example:
            >>> # ``__call__`` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass  # pragma: no cover

    def __call__(
        self, samples: Tensor, X: Tensor | None = None, *args, **kwargs
    ) -> Tensor:
        output = super().__call__(samples=samples, X=X, *args, **kwargs)
        # q-batch dimension is at -1 for single-output objectives and at
        # -2 for multi-output objectives.
        q_batch_idx = -2 if self._is_mo else -1
        if (
            X is not None
            and self._verify_output_shape
            and output.shape[q_batch_idx] != X.shape[-2]
        ):
            raise RuntimeError(
                "The q-batch shape of the objective values does not agree with "
                f"the q-batch shape of X. Got {output.shape[q_batch_idx]} and "
                f"{X.shape[-2]}. This may happen if you used a one-to-many input "
                "transform but forgot to use a corresponding objective."
            )
        return output


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return samples.squeeze(-1)


class LinearMCObjective(MCAcquisitionObjective):
    r"""Linear objective constructed from a weight tensor.

    For input ``samples`` and ``mc_obj = LinearMCObjective(weights)``, this produces
    ``mc_obj(samples) = sum_{i} weights[i] * samples[..., i]``

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) -> None:
        r"""
        Args:
            weights: A one-dimensional tensor with ``m`` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the linear objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return torch.einsum("...m, m", [samples, self.weights])


class GenericMCObjective(MCAcquisitionObjective):
    r"""Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(
                lambda Y, X: torch.sqrt(Y).sum(dim=-1),
            )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor, Tensor | None], Tensor]) -> None:
        r"""
        Args:
            objective: A callable ``f(samples, X)`` mapping a
                ``sample_shape x batch-shape x q x m``-dim Tensor ``samples`` and
                an optional ``batch-shape x q x d``-dim Tensor ``X`` to a
                ``sample_shape x batch-shape x q``-dim Tensor of objective values.
        """
        super().__init__()
        self.objective = objective

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim Tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of objective values.
        """
        return self.objective(samples, X=X)


class ConstrainedMCObjective(GenericMCObjective):
    r"""Feasibility-weighted objective.

    An Objective allowing to maximize some scalable objective on the model
    outputs subject to a number of constraints. Constraint feasibilty is
    approximated by a sigmoid function.

        mc_acq(X) = (
        (objective(X) + infeasible_cost) * \prod_i (1  - sigmoid(constraint_i(X)))
        ) - infeasible_cost

    See ``botorch.utils.objective.apply_constraints`` for details on the constraint
    handling.

    Example:
        >>> bound = 0.0
        >>> objective = lambda Y: Y[..., 0]
        >>> # apply non-negativity constraint on f(x)[1]
        >>> constraint = lambda Y: bound - Y[..., 1]
        >>> constrained_objective = ConstrainedMCObjective(objective, [constraint])
        >>> samples = sampler(posterior)
        >>> objective = constrained_objective(samples)

    TODO: Deprecate this as default way to handle constraints with MC acquisition
    functions once we have data on how well SampleReducingMCAcquisitionFunction works.
    """

    def __init__(
        self,
        objective: Callable[[Tensor, Tensor | None], Tensor],
        constraints: list[Callable[[Tensor], Tensor]],
        infeasible_cost: Tensor | float = 0.0,
        eta: Tensor | float = 1e-3,
    ) -> None:
        r"""
        Args:
            objective: A callable ``f(samples, X)`` mapping a
                ``sample_shape x batch-shape x q x m``-dim Tensor ``samples`` and
                an optional ``batch-shape x q x d``-dim Tensor ``X`` to a
                ``sample_shape x batch-shape x q``-dim Tensor of objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                ``sample_shape x batch-shape x q``, where negative values imply
                feasibility.
            infeasible_cost: The cost of a design if all associated samples are
                infeasible.
            eta: The temperature parameter of the sigmoid function approximating
                the constraint. Can be either a float or a 1-dim tensor. In case
                of a float the same eta is used for every constraint in
                constraints. In case of a tensor the length of the tensor must
                match the number of provided constraints. The i-th constraint is
                then estimated with the i-th eta value.
        """
        super().__init__(objective=objective)
        self.constraints = constraints
        if type(eta) is not Tensor:
            eta = torch.full((len(constraints),), eta)
        self.register_buffer("eta", eta)
        self.register_buffer("infeasible_cost", torch.as_tensor(infeasible_cost))

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A ``sample_shape x batch_shape x q x m``-dim Tensors of
                samples from a model posterior.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        return apply_constraints(
            obj=obj,
            constraints=self.constraints,
            samples=samples,
            infeasible_cost=self.infeasible_cost,
            eta=self.eta,
        )


LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN = (
    "pref_model has double-precision data, but single-precision data "
    "was passed to the LearnedObjective. Upcasting to double."
)


class LearnedObjective(MCAcquisitionObjective):
    r"""Learned preference objective constructed from a preference model.

    For input ``samples``, it samples each individual sample again from the
    latent preference posterior distribution using ``pref_model`` and return the
    posterior mean.

    Example:
        >>> train_X = torch.rand(2, 2)
        >>> train_comps = torch.LongTensor([[0, 1]])
        >>> pref_model = PairwiseGP(train_X, train_comps)
        >>> learned_pref_obj = LearnedObjective(pref_model)
        >>> samples = sampler(posterior)
        >>> objective = learned_pref_obj(samples)
    """

    def __init__(
        self,
        pref_model: Model,
        sample_shape: torch.Size | None = None,
        seed: int | None = None,
    ):
        r"""
        Args:
            pref_model: A BoTorch model, which models the latent preference/utility
                function. Given an input tensor of size
                ``sample_size x batch_shape x q x d``, its ``posterior`` method should
                return a ``Posterior`` object with single outcome representing the
                utility values of the input.
            sample_shape: Determines the number of preference-model samples drawn
                *per outcome-model sample* when the ``LearnedObjective`` is called.
                Note that this is an additional layer of sampling relative to what
                is needed when evaluating most MC acquisition functions in order to
                account for uncertainty in the preference model. If ``None``, it will
                default to ``torch.Size([16])``, so that 16 samples will be drawn
                from the preference model at each outcome sample. This number is
                relatively high because sampling from the preference model is general
                cheap relative to generating the outcome model posterior.
        """
        super().__init__()
        self.pref_model = pref_model
        if isinstance(pref_model, DeterministicModel):
            assert sample_shape is None
            self.sampler = None
        else:
            if sample_shape is None:
                sample_shape = torch.Size([DEFAULT_NUM_PREF_SAMPLES])
            # using an IIDNormalSampler instead of a SobolQMCNormalSampler by default
            # because SobolQMCNormalSampler can support up to 21201 total samples and
            # becomes noticeably slower than uniform sampling when the sample size is
            # large.
            self.sampler = IIDNormalSampler(sample_shape=sample_shape, seed=seed)
            self.sampler.batch_range_override = (1, -1)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Sample each element of samples.

        Args:
            samples: A ``sample_size x batch_shape x q x d``-dim Tensors of
                samples from a model posterior.

        Returns:
            A ``(sample_size * num_samples) x batch_shape x q``-dim Tensor of
            objective values sampled from utility posterior using ``pref_model``.
        """
        if samples.dtype != torch.float64 and any(
            d == torch.float64 for d in self.pref_model.dtypes_of_buffers
        ):
            warnings.warn(
                LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN,
                InputDataWarning,
                stacklevel=2,
            )
            samples = samples.to(torch.float64)

        if samples.ndim < 3:
            raise ValueError("samples should have at least 3 dimensions.")

        posterior = self.pref_model.posterior(samples)
        if isinstance(self.pref_model, DeterministicModel):
            # return preference posterior mean
            return posterior.mean.squeeze(-1)
        else:
            # return preference posterior augmented samples
            samples = self.sampler(posterior).squeeze(-1)
            return samples.reshape(-1, *samples.shape[2:])  # batch_shape x N
