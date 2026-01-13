#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Pairwise likelihood for pairwise preference model (e.g., PairwiseGP).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from botorch.utils.probability.utils import (
    log_ndtr,
    log_phi,
    standard_normal_log_hazard,
)
from gpytorch.likelihoods import Likelihood
from torch import Tensor
from torch.distributions import Bernoulli


class PairwiseLikelihood(Likelihood, ABC):
    """
    Pairwise likelihood base class for pairwise preference GP (e.g., PairwiseGP).
    """

    def __init__(self, max_plate_nesting: int = 1):
        """
        Initialized like a ``gpytorch.likelihoods.Likelihood``.

        Args:
            max_plate_nesting: Defaults to 1.
        """
        super().__init__(max_plate_nesting)

    def forward(self, utility: Tensor, D: Tensor) -> Bernoulli:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return a Bernoulli distribution object representing the likelihood of
        the user prefer v over u.

        Note that this is not used by the ``PairwiseGP`` model,
        """
        return Bernoulli(probs=self.p(utility=utility, D=D))

    @abstractmethod
    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        """Given the difference in (estimated) utility util_diff = f(v) - f(u),
        return the probability of the user prefer v over u.

        Args:
            utility: A Tensor of shape ``(batch_size) x n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.
            log: if true, return log probability
        """

    def log_p(self, utility: Tensor, D: Tensor) -> Tensor:
        """return the log of p"""
        return torch.log(self.p(utility=utility, D=D))

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the sum of negative log gradient with respect to each item's latent
            utility values. Useful for models using laplace approximation.

        Args:
            utility: A Tensor of shape ``(batch_size x) n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.

        Returns:
            A ``(batch_size x) n`` Tensor representing the sum of negative log gradient
            values of the likelihood over all comparisons (i.e., the m dimension)
            with respect to each item.
        """
        raise NotImplementedError

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the sum of negative log hessian with respect to each item's latent
            utility values. Useful for models using laplace approximation.

        Args:
            utility: A Tensor of shape ``(batch_size) x n``, the utility at MAP point
            D: D is ``(batch_size x) m x n`` matrix with all elements being zero in last
                dimension except at two positions D[..., i] = 1 and D[..., j] = -1
                respectively, representing item i is preferred over item j.

        Returns:
            A ``(batch_size x) n x n`` Tensor representing the sum of
            negative log hessian values of the likelihood over all
            comparisons (i.e., the m dimension) with respect to each item.
        """
        raise NotImplementedError


class PairwiseProbitLikelihood(PairwiseLikelihood):
    """Pairwise likelihood using probit function

    Given two items v and u with utilities f(v) and f(u), the probability that we
    prefer v over u with probability std_normal_cdf((f(v) - f(u))/sqrt(2)). Note
    that this formulation implicitly assume the noise term is fixed at 1.
    """

    # Clamping z values for better numerical stability. See self._calc_z for detail
    # norm_cdf(z=3) ~= 0.999, top 0.1% percent
    _zlim = 3

    def _calc_z(self, utility: Tensor, D: Tensor) -> Tensor:
        """Calculate the z score given estimated utility values and
        the comparison matrix D.
        """
        scaled_util = (utility / math.sqrt(2)).unsqueeze(-1)
        z = D.to(scaled_util) @ scaled_util
        z = z.clamp(-self._zlim, self._zlim).squeeze(-1)
        return z

    def _calc_z_derived(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Calculate auxiliary statistics derived from z, including log pdf,
        log cdf, and the hazard function (pdf divided by cdf)

        Args:
            z: A Tensor of arbitrary shape.

        Returns:
            Tensors with standard normal logpdf(z), logcdf(z), and hazard function
            values evaluated at -z.
        """
        return log_phi(z), log_ndtr(z), standard_normal_log_hazard(-z).exp()

    def p(self, utility: Tensor, D: Tensor, log: bool = False) -> Tensor:
        z = self._calc_z(utility=utility, D=D)
        std_norm = torch.distributions.normal.Normal(
            torch.zeros(1, dtype=z.dtype, device=z.device),
            torch.ones(1, dtype=z.dtype, device=z.device),
        )
        return std_norm.cdf(z)

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        # Compute the sum over of grad. of negative Log-LH wrt utility f.
        # Original grad should be of dimension m x n, as in (6) from
        # [Chu2005preference]_. The sum over the m dimension of grad. of
        # negative log likelihood with respect to the utility
        z = self._calc_z(utility, D)
        _, _, h = self._calc_z_derived(z)
        h_factor = h / math.sqrt(2)
        grad = (h_factor.unsqueeze(-2) @ (-D)).squeeze(-2)
        return grad

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        # Original hess should be of dimension m x n x n, as in (7) from
        # [Chu2005preference]_ Sum over the first dimension and return a tensor of
        # shape n x n.
        # The sum over the m dimension of hessian of negative log likelihood
        # with respect to the utility
        DT = D.transpose(-1, -2)
        z = self._calc_z(utility, D)
        _, _, h = self._calc_z_derived(z)
        mul_factor = h * (h + z) / 2
        mul_factor = mul_factor.unsqueeze(-2).expand(*DT.size())
        # multiply the hessian value by preference signs
        # (+1 if preferred or -1 otherwise) and sum over the m dimension
        hess = DT * mul_factor @ D
        return hess


class PairwiseLogitLikelihood(PairwiseLikelihood):
    """Pairwise likelihood using logistic (i.e., sigmoid) function

    Given two items v and u with utilities f(v) and f(u), the probability that we
    prefer v over u with probability sigmoid(f(v) - f(u)). Note
    that this formulation implicitly assume the beta term in logistic function is
    fixed at 1.
    """

    # Clamping logit values for better numerical stability.
    # See self._calc_logit for detail logistic(8) ~= 0.9997, top 0.03% percent
    _logit_lim = 8

    def _calc_logit(self, utility: Tensor, D: Tensor) -> Tensor:
        logit = D.to(utility) @ utility.unsqueeze(-1)
        logit = logit.clamp(-self._logit_lim, self._logit_lim).squeeze(-1)
        return logit

    def log_p(self, utility: Tensor, D: Tensor) -> Tensor:
        logit = self._calc_logit(utility=utility, D=D)
        return torch.nn.functional.logsigmoid(logit)

    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        logit = self._calc_logit(utility=utility, D=D)
        return torch.sigmoid(logit)

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        indices_shape = utility.shape[:-1] + (-1,)
        winner_indices = (D == 1).nonzero(as_tuple=True)[-1].reshape(indices_shape)
        loser_indices = (D == -1).nonzero(as_tuple=True)[-1].reshape(indices_shape)
        ex = torch.exp(torch.gather(utility, -1, winner_indices))
        ey = torch.exp(torch.gather(utility, -1, loser_indices))
        unsigned_grad = ey / (ex + ey)
        grad = (unsigned_grad.unsqueeze(-2) @ (-D)).squeeze(-2)
        return grad

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        DT = D.transpose(-1, -2)
        # calculating f(v) - f(u) given u > v information in D
        neg_logit = -(D @ utility.unsqueeze(-1)).squeeze(-1)
        term = torch.sigmoid(neg_logit)
        mul_factor = term - (term) ** 2
        mul_factor = mul_factor.unsqueeze(-2).expand(*DT.size())
        # multiply the hessian value by preference signs
        # (+1 if preferred or -1 otherwise) and sum over the m dimension
        hess = DT * mul_factor @ D
        return hess
