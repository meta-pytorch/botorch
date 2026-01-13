# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import LinearOperator, to_linear_operator
from torch import Tensor


class MultitaskGPPosterior(GPyTorchPosterior):
    def __init__(
        self,
        distribution: MultivariateNormal,
        joint_covariance_matrix: LinearOperator,
        test_train_covar: LinearOperator,
        train_diff: Tensor,
        test_mean: Tensor,
        train_train_covar: LinearOperator,
        train_noise: LinearOperator | Tensor,
        test_noise: LinearOperator | Tensor | None = None,
    ):
        r"""
        Posterior class for a Kronecker Multi-task GP model using with ICM kernel.
        Extends the standard GPyTorch posterior class by overwriting the rsample
        method. In general, this posterior should ONLY be used for MTGP models
        that have structured covariances. It should also only be used internally when
        called from the ``KroneckerMultiTaskGP.posterior(...)`` method.

        Args:
            distribution: Posterior multivariate normal distribution.
            joint_covariance_matrix: Joint test train covariance matrix over the entire
                tensor.
            test_train_covar: Covariance matrix of test x train points in the data
                space.
            train_diff: Difference between train mean and train responses.
            test_mean: Test mean response.
            train_train_covar: Covariance matrix of train points in the data space.
            train_noise: Training noise covariance.
            test_noise: Only used if posterior should contain observation noise.
                Testing noise covariance.
        """
        super().__init__(distribution=distribution)
        self._is_mt = True

        self.joint_covariance_matrix = joint_covariance_matrix
        self.test_train_covar = test_train_covar
        self.train_diff = train_diff
        self.test_mean = test_mean
        self.train_train_covar = train_train_covar
        self.train_noise = train_noise
        self.test_noise = test_noise
        self.observation_noise = self.test_noise is not None

        self.num_train = self.train_diff.shape[-2]
        # The following assumes test_train_covar is a SumLinearOperator. TODO: Improve
        self.num_tasks = self.test_train_covar.linear_ops[-1].shape[-1]

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples.

        Overwrites the standard ``base_sample_shape`` call to inform samplers that
        ``n + 2 n_train`` samples need to be drawn rather than n samples.
        """
        batch_shape = self.joint_covariance_matrix.shape[:-2]
        sampling_shape = (
            self.joint_covariance_matrix.shape[-2] + self.train_noise.shape[-2]
        )
        if self.observation_noise:
            sampling_shape = sampling_shape + self.test_noise.shape[-2]
        return batch_shape + torch.Size((sampling_shape,))

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        ``base_sample_shape``. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -1)

    def _prepare_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor = None
    ) -> tuple[Tensor, Tensor]:
        covariance_matrix = self.joint_covariance_matrix
        joint_size = covariance_matrix.shape[-1]
        batch_shape = covariance_matrix.batch_shape

        # pre-allocated this as None
        test_noise_base_samples = None
        if base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError(
                    "sample_shape disagrees with shape of base_samples."
                    f"provided base sample shape is {base_samples.shape} while"
                    f"the expected shape is {sample_shape}."
                )

            if base_samples.shape[-1] != 1:
                base_samples = base_samples.unsqueeze(-1)
            unsqueezed_dim = -2

            appended_shape = joint_size + self.train_train_covar.shape[-1]
            if self.observation_noise:
                appended_shape = appended_shape + self.test_noise.shape[-1]

            if appended_shape != base_samples.shape[unsqueezed_dim]:
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
                        dtype=base_samples.dtype,
                        device=base_samples.device,
                    )
                    base_samples = torch.cat((base_samples, new_base_samples), dim=-1)
                    base_samples = base_samples.unsqueeze(-1)
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
                1,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )

            noise_base_samples = torch.randn(
                *sample_shape,
                *batch_shape,
                self.train_train_covar.shape[-1],
                1,
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )
            if self.observation_noise:
                test_noise_base_samples = torch.randn(
                    *sample_shape,
                    *self.test_noise.shape[:-1],
                    1,
                    device=covariance_matrix.device,
                    dtype=covariance_matrix.dtype,
                )
        else:
            # finally split up the base samples
            noise_base_samples = base_samples[..., joint_size:, :]
            base_samples = base_samples[..., :joint_size, :]
            if self.observation_noise:
                test_noise_base_samples = noise_base_samples[
                    ..., -self.test_noise.shape[-1] :, :
                ]
                noise_base_samples = noise_base_samples[
                    ..., : -self.test_noise.shape[-1], :
                ]

        return base_samples, noise_base_samples, test_noise_base_samples

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor | None,
        train_diff: Tensor | None = None,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.
            base_samples: An (optional) Tensor of ``N(0, I)`` base samples of
                appropriate dimension, typically obtained from a ``Sampler``.
                This is used for deterministic optimization.
            train_diff: Difference between train mean and train responses to assume
                during sampling.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        if train_diff is None:
            train_diff = self.train_diff

        (
            base_samples,
            noise_base_samples,
            test_noise_base_samples,
        ) = self._prepare_base_samples(
            sample_shape=sample_shape, base_samples=base_samples
        )
        joint_samples = self._draw_from_base_covar(
            self.joint_covariance_matrix, base_samples
        )
        noise_samples = self._draw_from_base_covar(self.train_noise, noise_base_samples)

        # pluck out the train + test samples and add the likelihood's noise to the
        # train side. This should be fine for higher rank likelihoods.
        n_obs = self.num_tasks * self.num_train
        n_test = joint_samples.shape[-1] - n_obs
        obs_samples, test_samples = torch.split(joint_samples, [n_obs, n_test], dim=-1)
        updated_obs_samples = obs_samples + noise_samples
        obs_minus_samples = (
            train_diff.reshape(*train_diff.shape[:-2], -1) - updated_obs_samples
        )
        train_covar_plus_noise = self.train_train_covar + self.train_noise
        obs_solve = _permute_solve(
            train_covar_plus_noise, obs_minus_samples.unsqueeze(-1)
        )

        # and multiply the test-observed matrix against the result of the solve
        updated_samples = self.test_train_covar.matmul(obs_solve).squeeze(-1)

        # finally, we add the conditioned samples to the prior samples
        final_samples = test_samples + updated_samples

        # add in likelihood noise if necessary
        if self.observation_noise:
            test_noise_samples = self._draw_from_base_covar(
                self.test_noise, test_noise_base_samples
            )
            final_samples = final_samples + test_noise_samples

        # and reshape
        final_samples = final_samples.reshape(
            *final_samples.shape[:-1], self.test_mean.shape[-2], self.num_tasks
        )
        final_samples = final_samples + self.test_mean

        return final_samples

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

    def _draw_from_base_covar(
        self, covar: Tensor | LinearOperator, base_samples: Tensor
    ) -> Tensor:
        # Now reparameterize those base samples
        if not isinstance(covar, LinearOperator):
            covar = to_linear_operator(covar)
        covar_root = covar.root_decomposition().root
        # If necessary, adjust base_samples for rank of root decomposition
        if covar_root.shape[-1] < base_samples.shape[-2]:
            base_samples = base_samples[..., : covar_root.shape[-1], :]
        elif covar_root.shape[-1] > base_samples.shape[-2]:
            raise RuntimeError("Incompatible dimension of `base_samples`")
        # the mean is included in the posterior forwards so is not included here
        res = covar_root.matmul(base_samples)

        return res.squeeze(-1)


def _permute_solve(A: LinearOperator, b: Tensor) -> LinearOperator:
    r"""Solve the batched linear system AX = b, where b is a batched column
    vector. The solve is carried out after permuting the largest batch
    dimension of b to the final position, which results in a more efficient
    matrix-matrix solve.

    This ideally should be handled upstream (in GPyTorch, linear_operator or
    PyTorch), after which any uses of this method can be replaced with
    ``A.solve(b)``.

    Args:
        A: LinearOperator of shape (n, n)
        b: Tensor of shape (..., n, 1)

    Returns:
        LinearOperator of shape (..., n, 1)
    """
    # permute dimensions to move largest batch dimension to the end (more efficient
    # than unsqueezing)
    perm = list(range(b.ndim))
    if b.ndim > 2:
        largest_batch_dim, _ = max(enumerate(b.shape[:-2]), key=lambda t: t[1])
        perm[-1], perm[largest_batch_dim] = perm[largest_batch_dim], perm[-1]
    b_p = b.permute(*perm)

    x_p = A.solve(b_p)

    # Undo permutation
    inverse_perm = torch.argsort(torch.tensor(perm))
    x = x_p.permute(*inverse_perm)

    return x
