#!/usr/bin/env python3

"""
Single-valued analytical acquisition functions.
These functions do not rely on MC-sampling.
"""

from typing import Union

import torch
from torch import Tensor
from torch.distributions import Normal

from ...models import Model
from ..batch_utils import batch_mode_transform


@batch_mode_transform
def expected_improvement(
    X: Tensor, model: Model, best_f: Union[float, Tensor]
) -> Tensor:
    """Single-outcome expected improvement (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted single-output model (must be in batch mode if X is)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the expected improvement values of the respective design
        points

    """
    if torch.is_tensor(best_f):
        best_f = best_f.unsqueeze(-1)
    posterior = model.posterior(X)
    mean, sigma = posterior.mean, posterior.variance.sqrt()
    if mean.shape[-1] != 1:
        raise RuntimeError("Analytical EI can only be used with single-outcome models")
    mean = mean.squeeze(-1)
    sigma = sigma.squeeze(-1).clamp_min(1e-9)
    u = (mean - best_f) / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


def posterior_mean(X: Tensor, model: Model) -> Tensor:
    """Single-outcome posterior mean

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted model (must be in batch mode if X is)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the posterior mean of the respective design points

    """
    mean = model.posterior(X).mean
    if mean.shape[-1] != 1:
        raise RuntimeError("Posterior mean can only be used with single-outcome models")
    return mean.squeeze(-1)


@batch_mode_transform
def probability_of_improvement(
    X: Tensor, model: Model, best_f: Union[float, Tensor]
) -> Tensor:
    """Single-outcome probability of improvement (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted model (must be in batch mode if X is)
        best_f: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the best function value observed so far (assumed noiseless)

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the expected improvement values of the respective design
        points

    """
    if torch.is_tensor(best_f):
        best_f.unsqueeze(-1)
    posterior = model.posterior(X)
    mean, sigma = posterior.mean, posterior.variance.sqrt()
    if mean.shape[-1] != 1:
        raise RuntimeError("Analytical PI can only be used with single-outcome models")
    mean = mean.squeeze(-1)
    sigma = sigma.squeeze(-1).clamp_min(1e-9)
    u = (mean - best_f) / sigma
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    return normal.cdf(u)


@batch_mode_transform
def upper_confidence_bound(
    X: Tensor, model: Model, beta: Union[float, Tensor]
) -> Tensor:
    """Single-outcome upper confidence bound (assumes maximization)

    Args:
        X: Either a `n x d` or a `b x n x d` (batch mode) Tensor of `n` individual
            design points in `d` dimensions each. Design points are evaluated
            independently (i.e. covariance across the different points is not
            considered)
        model: A fitted model (must be in batch mode if X is)
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance

    Returns:
        A one-dim tensor with `n` elements or a `b x n` tensor (batch mode)
        corresponding to the upper confidence bound values of the respective design
        points

    """
    if torch.is_tensor(beta):
        beta = beta.unsqueeze(-1)
    posterior = model.posterior(X)
    mean, variance = posterior.mean, posterior.variance
    if mean.shape[-1] != 1:
        raise RuntimeError("Analytical UCB can only be used with single-outcome models")
    return mean.squeeze(-1) + (beta * variance.squeeze(-1)).sqrt()


def max_value_entropy_search(X: Tensor, model: Model, num_samples: int) -> Tensor:
    raise NotImplementedError()
