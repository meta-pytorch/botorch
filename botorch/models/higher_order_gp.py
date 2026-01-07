#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
References

.. [Zhe2019hogp]
    S. Zhe, W. Xing, and R. M. Kirby. Scalable high-order gaussian process regression.
    Proceedings of Machine Learning Research, volume 89, Apr 2019.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import gpt_posterior_settings
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.posteriors import (
    GPyTorchPosterior,
    HigherOrderGPPosterior,
    TransformedPosterior,
)
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP
from gpytorch.priors.torch_priors import MultivariateNormalPrior
from gpytorch.settings import fast_pred_var, skip_posterior_variances
from linear_operator.operators import (
    BatchRepeatLinearOperator,
    DiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    ZeroLinearOperator,
)
from linear_operator.settings import _fast_solves
from torch import Tensor
from torch.nn import ModuleList, Parameter, ParameterList


class FlattenedStandardize(Standardize):
    r"""
    Standardize outcomes in a structured multi-output settings by reshaping the
    batched output dimensions to be a vector. Specifically, an output dimension
    of [a x b x c] will be squeezed to be a vector of [a * b * c].
    """

    def __init__(
        self,
        output_shape: torch.Size,
        batch_shape: torch.Size | None = None,
        min_stdv: float = 1e-8,
    ):
        r"""
        Args:
            output_shape: A `n x output_shape`-dim tensor of training targets.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        if batch_shape is None:
            batch_shape = torch.Size()

        super().__init__(m=1, outputs=None, batch_shape=batch_shape, min_stdv=min_stdv)

        self.output_shape = output_shape
        self.batch_shape = batch_shape

    def _squeeze_to_single_output(self, tsr: Tensor) -> Tensor:
        dim_ct = tsr.ndim - len(self.output_shape) - 1
        return tsr.reshape(*tsr.shape[:dim_ct], -1, 1)

    def _return_to_output_shape(self, tsr: Tensor) -> Tensor:
        out = tsr.reshape(*tsr.shape[:-2], -1, *self.output_shape)
        return out

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)

        Y, Yvar = super().forward(Y, Yvar)
        Y_out = self._return_to_output_shape(Y)

        if Yvar is not None:
            Yvar_out = self._return_to_output_shape(Yvar)
        else:
            Yvar_out = None
        return Y_out, Yvar_out

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        Y = self._squeeze_to_single_output(Y)
        if Yvar is not None:
            Yvar = self._squeeze_to_single_output(Yvar)

        Y, Yvar = super().untransform(Y=Y, Yvar=Yvar, X=X)

        Y = self._return_to_output_shape(Y)
        if Yvar is not None:
            Yvar = self._return_to_output_shape(Yvar)
        return Y, Yvar

    def untransform_posterior(
        self, posterior: HigherOrderGPPosterior, X: Tensor | None = None
    ) -> TransformedPosterior:
        # TODO: return a HigherOrderGPPosterior once rescaling constant
        # muls * LinearOperators won't force a dense decomposition rather than a
        # Kronecker structured one.
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: self._return_to_output_shape(
                self.means + self.stdvs * self._squeeze_to_single_output(s)
            ),
            mean_transform=lambda m, v: self._return_to_output_shape(
                self.means + self.stdvs * self._squeeze_to_single_output(m)
            ),
            variance_transform=lambda m, v: self._return_to_output_shape(
                self._stdvs_sq * self._squeeze_to_single_output(v)
            ),
        )


class HigherOrderGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""
    A model for high-dimensional output regression.

    As described in [Zhe2019hogp]_. “Higher-order” means that the predictions
    are matrices (tensors) with at least two dimensions, such as images or
    grids of images, or measurements taken from a region of at least two
    dimensions.
    The posterior uses Matheron's rule [Doucet2010sampl]_
    as described in [Maddox2021bohdo]_.

    `HigherOrderGP` differs from a "vector” multi-output model in that it uses
    Kronecker algebra to obtain parsimonious covariance matrices for these
    outputs (see `KroneckerMultiTaskGP` for more information). For example,
    imagine a 10 x 20 x 30 grid of images. If we were to vectorize the
    resulting 6,000 data points in order to use them in a non-higher-order GP,
    they would have a 6,000 x 6,000 covariance matrix, with 36 million entries.
    The Kronecker structure allows representing this as a product of 10x10,
    20x20, and 30x30 covariance matrices, with only 1,400 entries.

    NOTE: This model requires the use of specialized Kronecker solves in
    linear operator, which are disabled by default in BoTorch. These are enabled
    by default in the `HigherOrderGP.posterior` call. However, they need to be
    manually enabled by the user during model fitting. Note also that we're using
    `fit_gpytorch_mll_torch()` here instead of `fit_gpytorch_mll()` since the
    approximate computations result in a non-smooth MLL that the default
    L-BFGS-B optimizer invoked by `fit_gpytorch_mll()` does not handle well.

    Example:
        >>> from linear_operator.settings import _fast_solves
        >>> model = HigherOrderGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> with _fast_solves(True):
        >>>     fit_gpytorch_mll_torch(mll)
        >>> samples = model.posterior(test_X).rsample()
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood | None = None,
        covar_modules: list[Kernel] | None = None,
        num_latent_dims: list[int] | None = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ):
        r"""
        Args:
            train_X: A `batch_shape x n x d`-dim tensor of training inputs.
            train_Y: A `batch_shape x n x output_shape`-dim tensor of training targets.
            likelihood: Gaussian likelihood for the model.
            covar_modules: List of kernels for each output structure.
            num_latent_dims: Sizes for the latent dimensions.
            learn_latent_pars: If true, learn the latent parameters.
            latent_init: [default or gp] how to initialize the latent parameters.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform. Note that `.train()` will
                be called on the outcome transform during instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        if input_transform is not None:
            input_transform.to(train_X)

        # infer the dimension of `output_shape`.
        num_output_dims = train_Y.dim() - train_X.dim() + 1
        batch_shape = train_X.shape[:-2]
        if len(batch_shape) > 1:
            raise NotImplementedError(
                "HigherOrderGP currently only supports 1-dim `batch_shape`."
            )
        if outcome_transform == DEFAULT:
            outcome_transform = FlattenedStandardize(
                output_shape=train_Y.shape[-num_output_dims:],
                batch_shape=batch_shape,
            )
        if outcome_transform is not None:
            if isinstance(outcome_transform, Standardize) and not isinstance(
                outcome_transform, FlattenedStandardize
            ):
                warnings.warn(
                    "HigherOrderGP does not support the outcome_transform "
                    "`Standardize`! Using `FlattenedStandardize` with `output_shape="
                    f"{train_Y.shape[- num_output_dims:]} and batch_shape="
                    f"{batch_shape} instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                outcome_transform = FlattenedStandardize(
                    output_shape=train_Y.shape[-num_output_dims:],
                    batch_shape=batch_shape,
                )
            outcome_transform.train()
            train_Y, _ = outcome_transform(train_Y, X=train_X)

        self._aug_batch_shape = batch_shape
        self._num_dimensions = num_output_dims + 1
        self._num_outputs = train_Y.shape[0] if batch_shape else 1
        self.target_shape = train_Y.shape[-num_output_dims:]
        self._input_batch_shape = batch_shape

        if likelihood is None:
            likelihood = get_gaussian_likelihood_with_lognormal_prior(
                batch_shape=self._aug_batch_shape
            )
        else:
            self._is_custom_likelihood = True

        super().__init__(
            train_X,
            train_Y.view(*self._aug_batch_shape, -1),
            likelihood=likelihood,
        )

        if covar_modules is not None:
            self.covar_modules = ModuleList(covar_modules)
        else:
            self.covar_modules = ModuleList(
                [
                    get_covar_module_with_dim_scaled_prior(
                        ard_num_dims=1 if dim > 0 else train_X.shape[-1],
                        batch_shape=self._aug_batch_shape,
                    )
                    for dim in range(self._num_dimensions)
                ]
            )

        if num_latent_dims is None:
            num_latent_dims = [1] * (self._num_dimensions - 1)

        self.to(train_X)

        self._initialize_latents(
            latent_init=latent_init,
            num_latent_dims=num_latent_dims,
            learn_latent_pars=learn_latent_pars,
            device=train_Y.device,
            dtype=train_Y.dtype,
        )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def _initialize_latents(
        self,
        latent_init: str,
        num_latent_dims: list[int],
        learn_latent_pars: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.latent_parameters = ParameterList()
        if latent_init == "default":
            for dim_num in range(len(self.covar_modules) - 1):
                self.latent_parameters.append(
                    Parameter(
                        torch.rand(
                            *self._aug_batch_shape,
                            self.target_shape[dim_num],
                            num_latent_dims[dim_num],
                            device=device,
                            dtype=dtype,
                        ),
                        requires_grad=learn_latent_pars,
                    )
                )
        elif latent_init == "gp":
            for dim_num, covar in enumerate(self.covar_modules[1:]):
                latent_covar = covar(
                    torch.linspace(
                        0.0,
                        1.0,
                        self.target_shape[dim_num],
                        device=device,
                        dtype=dtype,
                    )
                ).add_jitter(1e-4)
                latent_dist = MultivariateNormal(
                    torch.zeros(
                        *self._aug_batch_shape,
                        self.target_shape[dim_num],
                        device=device,
                        dtype=dtype,
                    ),
                    latent_covar,
                )
                sample_shape = torch.Size((num_latent_dims[dim_num],))
                latent_sample = latent_dist.sample(sample_shape=sample_shape)
                latent_sample = latent_sample.reshape(
                    *self._aug_batch_shape,
                    self.target_shape[dim_num],
                    num_latent_dims[dim_num],
                )
                self.latent_parameters.append(
                    Parameter(
                        latent_sample,
                        requires_grad=learn_latent_pars,
                    )
                )
                self.register_prior(
                    "latent_parameters_" + str(dim_num),
                    MultivariateNormalPrior(
                        latent_dist.loc,
                        latent_dist.covariance_matrix.detach().clone(),
                        transform=lambda x: x.squeeze(-1),
                    ),
                    lambda module, dim_num=dim_num: self.latent_parameters[dim_num],
                )

    def forward(self, X: Tensor) -> MultivariateNormal:
        if self.training:
            X = self.transform_inputs(X)

        covariance_list = []
        covariance_list.append(self.covar_modules[0](X))

        for cm, param in zip(self.covar_modules[1:], self.latent_parameters):
            if not self.training:
                with torch.no_grad():
                    covariance_list.append(cm(param))
            else:
                covariance_list.append(cm(param))

        # check batch_shapes
        if covariance_list[0].batch_shape != covariance_list[1].batch_shape:
            for i in range(1, len(covariance_list)):
                cm = covariance_list[i]
                covariance_list[i] = BatchRepeatLinearOperator(
                    cm, covariance_list[0].batch_shape
                )
        kronecker_covariance = KroneckerProductLinearOperator(*covariance_list)

        # TODO: expand options for the mean module via batch shaping?
        mean = torch.zeros(
            *covariance_list[0].batch_shape,
            kronecker_covariance.shape[-1],
            device=kronecker_covariance.device,
            dtype=kronecker_covariance.dtype,
        )
        return MultivariateNormal(mean, kronecker_covariance)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        # we need to squeeze the targets in order to preserve the shaping
        inputs_batch_dims = len(inputs.shape[:-2])
        target_shape = (*inputs.shape[:-2], -1)
        if (inputs_batch_dims + self._num_dimensions) < targets.ndim:
            target_shape = (targets.shape[0], *target_shape)
        reshaped_targets = targets.view(*target_shape)

        return super().get_fantasy_model(inputs, reshaped_targets, **kwargs)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: torch.Tensor | None = None, **kwargs: Any
    ) -> HigherOrderGP:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `m` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m_d`-dim Tensor, where `m_d` is the shaping
                of the model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.
            noise: If not None, a tensor of the same shape as `Y` representing
                the noise variance associated with each observation.
            kwargs: Passed to `condition_on_observations`.

        Returns:
            A `BatchedMultiOutputGPyTorchModel` object of the same type with
            `n + n'` training examples, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        if hasattr(self, "outcome_transform"):
            # we need to apply transforms before shifting batch indices around
            Y, noise = self.outcome_transform(Y=Y, Yvar=noise, X=X)
        # Do not check shapes when fantasizing as they are not expected to match.
        if fantasize_flag.off():
            self._validate_tensor_args(X=X, Y=Y, Yvar=noise, strict=False)

        # we don't need to do un-squeezing because Y already is batched
        # we don't support fixed noise here yet
        # if noise is not None:
        #     kwargs.update({"noise": noise})
        fantasy_model = super(
            BatchedMultiOutputGPyTorchModel, self
        ).condition_on_observations(X=X, Y=Y, noise=noise, **kwargs)
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
            : (-1 if self._num_outputs == 1 else -2)
        ]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        return fantasy_model

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior:
        self.eval()  # make sure we're calling a posterior

        if posterior_transform is not None:
            # this could be very costly, disallow for now
            raise NotImplementedError(
                "Posterior transforms currently not supported for "
                f"{self.__class__.__name__}"
            )

        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        no_pred_variance = skip_posterior_variances._state

        with ExitStack() as es:
            es.enter_context(gpt_posterior_settings())
            es.enter_context(fast_pred_var(True))
            es.enter_context(_fast_solves(True))

            # we need to skip posterior variances here
            es.enter_context(skip_posterior_variances(True))
            mvn = self(X)
            if observation_noise is not False:
                # TODO: ensure that this still works for structured noise solves.
                mvn = self.likelihood(mvn, X)

            # lazy covariance matrix includes the interpolated version of the full
            # covariance matrix so we can actually grab that instead.
            if X.ndimension() > self.train_inputs[0].ndimension():
                X_batch_shape = X.shape[:-2]
                train_inputs = self.train_inputs[0].reshape(
                    *[1] * len(X_batch_shape), *self.train_inputs[0].shape
                )
                train_inputs = train_inputs.repeat(
                    *X_batch_shape, *[1] * self.train_inputs[0].ndimension()
                )
            else:
                train_inputs = self.train_inputs[0]

            # we now compute the data covariances for the training data, the testing
            # data, the joint covariances, and the test train cross-covariance
            train_train_covar = self.prediction_strategy.lik_train_train_covar.detach()
            base_train_train_covar = train_train_covar.linear_op

            data_train_covar = base_train_train_covar.linear_ops[0]
            data_covar = self.covar_modules[0]
            data_train_test_covar = data_covar(X, train_inputs)
            data_test_test_covar = data_covar(X)
            data_joint_covar = data_train_covar.cat_rows(
                cross_mat=data_train_test_covar,
                new_mat=data_test_test_covar,
            )

            # we detach the latents so that they don't cause gradient errors
            # TODO: Can we enable backprop through the latent covariances?
            batch_shape = data_train_test_covar.batch_shape
            latent_covar_list = []
            for latent_covar in base_train_train_covar.linear_ops[1:]:
                if latent_covar.batch_shape != batch_shape:
                    latent_covar = BatchRepeatLinearOperator(latent_covar, batch_shape)
                latent_covar_list.append(latent_covar.detach())

            joint_covar = KroneckerProductLinearOperator(
                data_joint_covar, *latent_covar_list
            )
            test_train_covar = KroneckerProductLinearOperator(
                data_train_test_covar, *latent_covar_list
            )

            # compute the posterior variance if necessary
            if no_pred_variance:
                pred_variance = mvn.variance
            else:
                pred_variance = self.make_posterior_variances(joint_covar)

            # mean and variance get reshaped into the target shape
            new_mean = mvn.mean.reshape(*X.shape[:-1], *self.target_shape)
            if not no_pred_variance:
                new_variance = pred_variance.reshape(*X.shape[:-1], *self.target_shape)
                new_variance = DiagLinearOperator(new_variance)
            else:
                new_variance = ZeroLinearOperator(
                    *X.shape[:-1], *self.target_shape, self.target_shape[-1]
                )

            mvn = MultivariateNormal(new_mean, new_variance)

            # return a specialized Posterior to allow for sampling
            # cloning the full covar allows backpropagation through it
            posterior = HigherOrderGPPosterior(
                distribution=mvn,
                train_targets=self.train_targets.unsqueeze(-1),
                train_train_covar=train_train_covar,
                test_train_covar=test_train_covar,
                joint_covariance_matrix=joint_covar.clone(),
                output_shape=X.shape[:-1] + self.target_shape,
                num_outputs=self._num_outputs,
            )
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
            return posterior

    def make_posterior_variances(
        self, joint_covariance_matrix: LinearOperator
    ) -> Tensor:
        r"""
        Computes the posterior variances given the data points X. As currently
        implemented, it computes another forwards call with the stacked data to get out
        the joint covariance across all data points.
        """
        # TODO: use the exposed joint covariances from the prediction strategy
        data_joint_covariance = joint_covariance_matrix.linear_ops[0].evaluate_kernel()
        num_train = self.train_inputs[0].shape[-2]
        test_train_covar = data_joint_covariance[..., num_train:, :num_train]
        train_train_covar = data_joint_covariance[..., :num_train, :num_train]
        test_test_covar = data_joint_covariance[..., num_train:, num_train:]

        jcm_linops = joint_covariance_matrix.linear_ops[1:]
        full_train_train_covar = KroneckerProductLinearOperator(
            train_train_covar, *jcm_linops
        )
        full_test_test_covar = KroneckerProductLinearOperator(
            test_test_covar, *jcm_linops
        )
        full_test_train_covar_tuple = (test_train_covar,) + jcm_linops

        train_evals, train_evecs = full_train_train_covar.eigh()
        # (\kron \Lambda_i + \sigma^2 I)^{-1}
        train_inv_evals = DiagLinearOperator(
            1.0 / (train_evals + self.likelihood.noise)
        )

        # compute K_i S_i \hadamard K_i S_i
        test_train_hadamard = KroneckerProductLinearOperator(
            *[
                lt1.matmul(lt2).to_dense() ** 2
                for lt1, lt2 in zip(full_test_train_covar_tuple, train_evecs.linear_ops)
            ]
        )

        # and compute the column sums of
        #  (\kron K_i S_i * K_i S_i) \tilde{\Lambda}^{-1}
        test_train_pred_covar = test_train_hadamard.matmul(train_inv_evals).sum(dim=-1)

        pred_variances = full_test_test_covar.diagonal() - test_train_pred_covar
        return pred_variances
