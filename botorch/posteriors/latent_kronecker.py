#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import IdentityLinearOperator, ZeroLinearOperator
from torch import Tensor


r"""
References

.. [wilson2020sampling]
    J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. Efficiently
    sampling functions from Gaussian process posteriors. International Conference on
    Machine Learning (2020).

.. [wilson2021pathwise]
    J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. Pathwise
    Conditioning of Gaussian Processes. Journal of Machine Learning Research (2021).
"""


class LatentKroneckerGPPosterior(GPyTorchPosterior):
    r"""
    Dummy posterior class for a LatentKroneckerGP model.
    Internally calls model._rsample_from_base_samples to draw posterior samples via
    pathwise conditioning aka Matheron's rule [wilson2020sampling, wilson2021pathwise].

    This is necessary because BoTorch instantiates the posterior object before creating
    base samples, whereas pathwise conditioning requires the base samples first to
    calculate the posterior samples. To cache expensive computations, which only have
    to be performed once for the same base samples, the results are stored in the model
    instead of the posterior object, because a new posterior object is created in each
    acquisition function call.
    """

    def __init__(
        self,
        model: GPyTorchModel,
        X: Tensor,
        T: Tensor,
    ) -> None:
        r"""A dummy posterior for LatentKroneckerGP models.

        Args:
            model: The LatentKroneckerGP model to which this posterior belongs to.
            X: A ``(batch_shape) x q x d``-dim Tensor, where ``d`` is the dimension
                of the feature space and ``q`` is the number of points considered
                jointly, on which the posterior shall be evaluated.
            T: A ``(batch_shape) x t x 1``-dim Tensor of ``T``-locations at which to
                evaluate the posterior. If None, defaults to using ``self.train_T``.
        """
        self._dtype = X.dtype
        self._device = X.device
        self.batch_shape = model.batch_shape
        self.output_batch_shape = torch.broadcast_shapes(
            model.batch_shape, X.shape[:-2]
        )
        output_dim = X.shape[-2] * T.shape[-2]
        mean = ZeroLinearOperator(
            *self.output_batch_shape, output_dim, dtype=X.dtype, device=X.device
        )
        covar = IdentityLinearOperator(
            output_dim,
            batch_shape=self.output_batch_shape,
            dtype=X.dtype,
            device=X.device,
        )
        dummy_mvn = MultivariateNormal(mean=mean, covariance_matrix=covar)
        super().__init__(distribution=dummy_mvn)
        self.model = model
        self.X = X
        self.T = T
        self._is_mt = True

    @property
    def base_sample_shape(self):
        r"""The shape of a base sample used for constructing posterior samples.

        Overwrites the standard ``base_sample_shape`` call to inform samplers that
        ``n_train_full + n_train + n_test`` samples are needed rather than n samples.
        """
        n_train_full = (
            self.model.train_inputs[0].shape[-2] * self.model.train_T.shape[-2]
        )
        n_train = self.model.train_targets.shape[-1]
        n_test = self.X.shape[-2] * self.T.shape[-2]
        return self.batch_shape + torch.Size([n_train_full + n_train + n_test])

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        ``base_sample_shape``. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -1)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the distribution with
        the given ``sample_shape``.
        """
        x_shape = self.X.shape[-2:-1]
        t_shape = self.T.shape[-2:-1]
        return sample_shape + self.output_batch_shape + x_shape + t_shape

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Since this posterior is a dummy object, call the model to perform sampling.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: A Tensor of ``N(0, I)`` base samples of shape
                ``sample_shape x base_sample_shape``, typically obtained from
                a ``Sampler``. This is used for deterministic optimization.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )

        return self.model._rsample_from_base_samples(self.X, self.T, base_samples)

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        base_samples = torch.randn(
            sample_shape + self.base_sample_shape,
            dtype=self.X.dtype,
            device=self.X.device,
        )
        return self.rsample_from_base_samples(sample_shape, base_samples)
