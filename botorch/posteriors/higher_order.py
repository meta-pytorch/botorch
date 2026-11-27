#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import LinearOperator
from torch import Tensor


class HigherOrderGPPosterior(GPyTorchPosterior):
    r"""
    Posterior class for a Higher order Gaussian process model [Zhe2019hogp]_. Extends
    the standard GPyTorch posterior class by overwriting the rsample method.
    The posterior variance is handled internally by the HigherOrderGP model.
    HOGP is a tensorized GP model so the posterior covariance grows to be extremely
    large, but is highly structured, which means that we can exploit Kronecker
    identities to sample from the posterior using Matheron's rule as described in
    [Doucet2010sampl]_.

    In general, this posterior should ONLY be used for HOGP models
    that have highly structured covariances. It should also only be used internally when
    called from the HigherOrderGP.posterior(...) method. At this time, the posterior
    does not support gradients with respect to the training data.
    """

    def __init__(
        self,
        distribution: MultivariateNormal,
        joint_covariance_matrix: LinearOperator,
        train_train_covar: LinearOperator,
        test_train_covar: LinearOperator,
        train_targets: Tensor,
        output_shape: torch.Size,
        num_outputs: int,
    ) -> None:
        r"""A Posterior for HigherOrderGP models.

        Args:
            distribution: Posterior multivariate normal distribution.
            joint_covariance_matrix: Joint test train covariance matrix over the entire
                tensor.
            train_train_covar: Covariance matrix of train points in the data space.
            test_train_covar: Covariance matrix of test x train points
                in the data space.
            train_targets: Training responses vectorized.
            output_shape: Shape output training responses.
            num_outputs: Batch shaping of model.
        """
        super().__init__(distribution=distribution)
        self.joint_covariance_matrix = joint_covariance_matrix
        self.train_train_covar = train_train_covar
        self.test_train_covar = test_train_covar
        self.train_targets = train_targets
        self.output_shape = output_shape
        self._is_mt = True
        self.num_outputs = num_outputs

    @property
    def base_sample_shape(self):
        r"""The shape of a base sample used for constructing posterior samples.

        Overwrites the standard ``base_sample_shape`` call to inform samplers that
        ``n + 2 n_train`` samples need to be drawn rather than n samples.
        """
        joint_covar = self.joint_covariance_matrix
        batch_shape = joint_covar.shape[:-2]
        sampling_shape = torch.Size(
            [joint_covar.shape[-2] + self.train_train_covar.shape[-2]]
        )
        return batch_shape + sampling_shape

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
        r"""Returns the shape of the samples produced by the posterior with
        the given ``sample_shape``.
        """
        return sample_shape + self.output_shape

    def _prepare_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor = None
    ) -> Tensor:
        sample_shape = torch.Size(sample_shape)
        covariance_matrix = self.joint_covariance_matrix
        joint_size = covariance_matrix.shape[-1]
        batch_shape = covariance_matrix.batch_shape

        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("sample_shape disagrees with shape of base_samples.")

            appended_shape = joint_size + self.train_train_covar.shape[-1]
            if appended_shape != base_samples.shape[-1]:
                # get base_samples to the correct shape by expanding as sample shape,
                # batch shape, then rest of dimensions. We have to add first the sample
                # shape, then the batch shape of the model, and then finally the shape
                # of the test data points squeezed into a single dimension, accessed
                # from the test_train_covar.
                base_sample_shapes = (
                    sample_shape + batch_shape + self.test_train_covar.shape[-2:-1]
                )
                if base_samples.nelement() == base_sample_shapes.numel():
                    base_samples = base_samples.reshape(base_sample_shapes)

                    new_base_samples = torch.randn(
                        *sample_shape,
                        *batch_shape,
                        appended_shape - base_samples.shape[-1],
                        device=base_samples.device,
                        dtype=base_samples.dtype,
                    )
                    base_samples = torch.cat((base_samples, new_base_samples), dim=-1)
                else:
                    raise BotorchTensorDimensionError(
                        "The base samples are not compatible with base sample shape. "
                        f"Received base samples of shape {base_samples.shape}, "
                        f"expected {base_sample_shapes}."
                    )

        if base_samples is None:
            # TODO: Allow qMC sampling
            base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                joint_size,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )

            noise_base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                self.train_train_covar.shape[-1],
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )
        else:
            # finally split up the base samples
            noise_base_samples = base_samples[..., joint_size:]
            base_samples = base_samples[..., :joint_size]

        def flatten_samples(value: Tensor) -> Tensor:
            tail = value.shape[len(sample_shape) :]
            missing = len(batch_shape) + 1 - len(tail)
            if missing > 0:
                value = value.reshape(*sample_shape, *([1] * missing), *tail)
            value = value.expand(*sample_shape, *batch_shape, value.shape[-1])
            return value.reshape(
                sample_shape.numel(), *batch_shape, value.shape[-1]
            ).movedim(0, -1)

        return flatten_samples(base_samples), flatten_samples(noise_base_samples)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor | None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        As the posterior covariance is difficult to draw from in this model,
        we implement Matheron's rule as described in [Doucet2010sampl]-. This may not
        work entirely correctly for deterministic base samples unless base samples
        are provided that are of shape ``n + 2 * n_train`` because the sampling method
        draws ``2 * n_train`` samples as well as the standard ``n``.
        samples.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: An (optional) Tensor of ``N(0, I)`` base samples of
                appropriate dimension, typically obtained from a ``Sampler``.
                This is used for deterministic optimization.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        base_samples, noise_base_samples = self._prepare_base_samples(
            sample_shape, base_samples
        )

        # base samples now have trailing sample dimension
        covariance_matrix = self.joint_covariance_matrix
        covar_root = covariance_matrix.root_decomposition().root

        samples = covar_root.matmul(base_samples[..., : covar_root.shape[-1], :])

        # now pluck out Y_x and X_x
        noiseless_train_marginal_samples = samples[
            ..., : self.train_train_covar.shape[-1], :
        ]
        test_marginal_samples = samples[..., self.train_train_covar.shape[-1] :, :]
        # Recover the observation-noise diagonal without depending on training
        # cache wrapper types or guessing matching batch dimensions by size.
        count = self.train_train_covar.shape[-1]
        noise_variance = (
            self.train_train_covar.diagonal()
            - covariance_matrix.diagonal()[..., :count]
        )
        noise_std = noise_variance.clamp_min(0).sqrt().unsqueeze(-1)

        # we need to add noise into the noiseless samples
        noise_marginal_samples = noise_std * noise_base_samples

        train_marginal_samples = (
            noiseless_train_marginal_samples + noise_marginal_samples
        )

        # compute y - Y_x
        train_rhs = self.train_targets - train_marginal_samples

        # K_{train, train}^{-1} (y - Y_x)
        # internally, this solve is done using Kronecker algebra and is fast.
        kinv_rhs = self.train_train_covar.solve(train_rhs)
        # multiply by cross-covariance
        test_updated_samples = self.test_train_covar.matmul(kinv_rhs)

        # add samples
        test_cond_samples = test_marginal_samples + test_updated_samples
        test_cond_samples = test_cond_samples.permute(
            test_cond_samples.ndim - 1, *range(0, test_cond_samples.ndim - 1)
        )

        # reshape samples to be the actual size of the train targets
        return test_cond_samples.reshape(*sample_shape, *self.output_shape)

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
        return self.rsample_from_base_samples(
            sample_shape=sample_shape, base_samples=None
        )
