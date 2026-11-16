#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Likelihood for top-choice (best-of-k) preference observations."""

from __future__ import annotations

from typing import Any

import torch
from botorch.utils.sampling import draw_sobol_normal_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor
from torch.distributions import Categorical


class TopChoiceSoftmaxLikelihood(Likelihood):
    r"""Softmax (multinomial logit) likelihood for top-choice observations.

    Each observation is a comparison of ``k`` alternatives in which one alternative
    is chosen as the best. Given the latent utilities ``f_1, ..., f_k`` of the
    alternatives, the probability of choosing alternative ``i`` is

        ``P(choice = i | f) = exp(f_i) / sum_j exp(f_j)``.

    The likelihood is invariant to adding a constant to the utilities of a
    comparison. For ``k = 2``, it is the logistic (Bradley-Terry) pairwise comparison
    likelihood.

    The utilities of the ``k`` alternatives of ``m`` comparisons are flattened along
    the last dimension in comparison-major order, i.e., the utilities of the ``i``-th
    comparison are at positions ``i * k, ..., i * k + k - 1``.
    """

    def __init__(
        self,
        num_alternatives: int,
        num_samples: int = 256,
        seed: int | None = None,
    ) -> None:
        r"""Initialize the likelihood.

        Args:
            num_alternatives: The number of alternatives per comparison, ``k >= 2``.
            num_samples: The number of quasi-Monte Carlo samples used to estimate
                the expected log-likelihood.
            seed: The seed of the scrambled Sobol base samples. The same base
                samples are used in every evaluation, so that the expected
                log-likelihood is a deterministic function of the variational
                distribution, as required by deterministic optimizers such as
                L-BFGS-B. If None, a random seed is drawn at initialization.
        """
        super().__init__()
        self.num_alternatives = num_alternatives
        self.num_samples = num_samples
        self.seed = (
            seed if seed is not None else int(torch.randint(0, 1_000_000, (1,)).item())
        )

    def forward(
        self, function_samples: Tensor, *args: Any, **kwargs: Any
    ) -> Categorical:
        r"""Return the distribution of the choices given the utilities.

        Args:
            function_samples: A ``batch_shape x (m * k)`` tensor of utilities,
                flattened in comparison-major order.

        Returns:
            A ``Categorical`` distribution with batch shape ``batch_shape x m`` over
            the ``k`` alternatives of each comparison.
        """
        return Categorical(
            logits=function_samples.unflatten(-1, (-1, self.num_alternatives))
        )

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Estimate the expected log-likelihood of each comparison via QMC.

        The log-likelihood factorizes over the comparisons and each factor depends
        only on the ``k`` utilities of its comparison, so only the ``k x k`` diagonal
        blocks of the covariance of ``function_dist`` are factorized. Note that the
        full ``(m * k) x (m * k)`` covariance matrix is evaluated to extract them.

        Args:
            observations: A ``batch_shape x m`` tensor of the indices, in
                ``{0, ..., k - 1}``, of the chosen alternative of each comparison.
            function_dist: A ``MultivariateNormal`` with event shape ``m * k`` over
                the utilities, flattened in comparison-major order. Its covariance
                must include the covariances between the utilities of each
                comparison, i.e., it must not be a diagonal approximation.

        Returns:
            A ``batch_shape x m`` tensor of expected log-likelihoods.
        """
        k = self.num_alternatives
        mean = function_dist.mean.unflatten(-1, (-1, k))  # batch_shape x m x k
        idx = torch.arange(mean.shape[-2] * k, device=mean.device).view(-1, k)
        # batch_shape x m x k x k. Indexing the dense covariance is much cheaper than
        # indexing the lazy covariance with tensor indices.
        covar = function_dist.covariance_matrix[
            ..., idx.unsqueeze(-1), idx.unsqueeze(-2)
        ]
        base_samples = draw_sobol_normal_samples(
            d=k,
            n=self.num_samples,
            device=mean.device,
            dtype=mean.dtype,
            seed=self.seed,
        )  # num_samples x k
        # num_samples x batch_shape x m x k
        samples = mean + (psd_safe_cholesky(covar) @ base_samples.mT).movedim(-1, 0)
        log_probs = samples.log_softmax(dim=-1)
        index = observations.long().expand(log_probs.shape[:-1]).unsqueeze(-1)
        return log_probs.gather(dim=-1, index=index).squeeze(-1).mean(dim=0)
