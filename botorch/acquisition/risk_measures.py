#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Risk Measures implemented as Monte-Carlo objectives, based on Bayesian
optimization of risk measures as introduced in [Cakmak2020risk]_. For a
broader discussion of Monte-Carlo methods for VaR and CVaR risk measures,
see also [Hong2014review]_.

.. [Cakmak2020risk]
    S. Cakmak, R. Astudillo, P. Frazier, and E. Zhou. Bayesian Optimization of
    Risk Measures. Advances in Neural Information Processing Systems 33, 2020.

.. [Hong2014review]
    L. J. Hong, Z. Hu, and G. Liu. Monte carlo methods for value-at-risk and
    conditional value-at-risk: a review. ACM Transactions on Modeling and
    Computer Simulation, 2014.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import ceil

import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from torch import Tensor


class RiskMeasureMCObjective(MCAcquisitionObjective, ABC):
    r"""Objective transforming the posterior samples to samples of a risk measure.

    The risk measure is calculated over joint q-batch samples from the posterior.
    If the q-batch includes samples corresponding to multiple inputs, it is assumed
    that first ``n_w`` samples correspond to first input, second ``n_w`` samples
    correspond to second input etc.

    The risk measures are commonly defined for minimization by considering the
    upper tail of the distribution, i.e., treating larger values as being undesirable.
    BoTorch by default assumes a maximization objective, so the default behavior here
    is to calculate the risk measures w.r.t. the lower tail of the distribution.
    This can be changed by passing a preprocessing function with
    ``weights=torch.tensor([-1.0])``.
    """

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a ``batch x m``-dim
                tensor of samples, this should return a ``batch``-dim tensor.
        """
        super().__init__()
        self.n_w = n_w
        if preprocessing_function is None:
            if self._is_mo:
                preprocessing_function = IdentityMCMultiOutputObjective()
            else:
                preprocessing_function = IdentityMCObjective()
        self.preprocessing_function = preprocessing_function

    def _prepare_samples(self, samples: Tensor) -> Tensor:
        r"""Prepare samples for risk measure calculations by scalarizing and
        separating out the q-batch dimension.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.

        Returns:
            A ``sample_shape x batch_shape x q x n_w``-dim tensor of prepared samples.
        """
        if samples.shape[-1] > 1 and isinstance(
            self.preprocessing_function, IdentityMCObjective
        ):
            raise RuntimeError(
                "Multi-output samples should be scalarized using a "
                "`preprocessing_function`."
            )
        samples = self.preprocessing_function(samples)
        return samples.view(*samples.shape[:-1], -1, self.n_w)

    @abstractmethod
    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the risk measure corresponding to the given samples.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Ignored.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of risk measure samples.
        """
        pass  # pragma: no cover


class CVaR(RiskMeasureMCObjective):
    r"""The Conditional Value-at-Risk risk measure.

    The Conditional Value-at-Risk measures the expectation of the worst outcomes
    (small rewards or large losses) with a total probability of ``1 - alpha``. It
    is commonly defined as the conditional expectation of the reward function,
    with the condition that the reward is smaller than the corresponding
    Value-at-Risk (also defined below).

    Note: Due to the use of a discrete ``w_set`` of samples, the VaR and CVaR
        calculated here are (possibly biased) Monte-Carlo approximations of
        the true risk measures.
    """

    def __init__(
        self,
        alpha: float,
        n_w: int,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            alpha: The risk level, float in ``(0.0, 1.0]``.
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a ``batch x m``-dim
                tensor of samples, this should return a ``batch``-dim tensor.
        """
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0.0, 1.0]")
        self.alpha = alpha
        self.alpha_idx = ceil(n_w * alpha) - 1

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the CVaR corresponding to the given samples.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Ignored.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of CVaR samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return torch.topk(
            prepared_samples,
            k=prepared_samples.shape[-1] - self.alpha_idx,
            largest=False,
            dim=-1,
        ).values.mean(dim=-1)


class VaR(CVaR):
    r"""The Value-at-Risk risk measure.

    Value-at-Risk measures the smallest possible reward (or largest possible loss)
    after excluding the worst outcomes with a total probability of ``1 - alpha``. It
    is commonly used in financial risk management, and it corresponds to the
    ``1 - alpha`` quantile of a given random variable.
    """

    def __init__(
        self,
        alpha: float,
        n_w: int,
        preprocessing_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            alpha: The risk level, float in ``(0.0, 1.0]``.
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a ``batch x m``-dim
                tensor of samples, this should return a ``batch``-dim tensor.
        """
        super().__init__(
            n_w=n_w,
            alpha=alpha,
            preprocessing_function=preprocessing_function,
        )
        self._q = 1 - self.alpha_idx / n_w

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the VaR corresponding to the given samples.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Ignored.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of VaR samples.
        """
        prepared_samples = self._prepare_samples(samples)
        # this is equivalent to sorting along dim=-1 in descending order
        # and taking the values at index self.alpha_idx. E.g.
        # >>> sorted_res = prepared_samples.sort(dim=-1, descending=True)
        # >>> sorted_res.values[..., self.alpha_idx]
        # Using quantile is far more memory efficient since ``torch.sort``
        # produces values and indices tensors with shape
        # ``sample_shape x batch_shape x (q * n_w) x m``
        return torch.quantile(
            input=prepared_samples,
            q=self._q,
            dim=-1,
            keepdim=False,
            interpolation="lower",
        )


class WorstCase(RiskMeasureMCObjective):
    r"""The worst-case risk measure."""

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the worst-case measure corresponding to the given samples.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Ignored.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of worst-case samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.min(dim=-1).values


class Expectation(RiskMeasureMCObjective):
    r"""The expectation risk measure.

    For unconstrained problems, we recommend using the ``ExpectationPosteriorTransform``
    instead. ``ExpectationPosteriorTransform`` directly transforms the posterior
    distribution over ``q * n_w`` to a posterior of ``q`` expectations, significantly
    reducing the cost of posterior sampling as a result.
    """

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        r"""Calculate the expectation corresponding to the given samples.
        This calculates the expectation / mean / average of each ``n_w`` samples
        across the q-batch dimension. If ``self.weights`` is given, the samples
        are scalarized across the output dimension before taking the expectation.

        Args:
            samples: A ``sample_shape x batch_shape x (q * n_w) x m``-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                ``n_w`` block of samples correspond to the same input.
            X: A ``batch_shape x q x d``-dim tensor of inputs. Ignored.

        Returns:
            A ``sample_shape x batch_shape x q``-dim tensor of expectation samples.
        """
        prepared_samples = self._prepare_samples(samples)
        return prepared_samples.mean(dim=-1)
