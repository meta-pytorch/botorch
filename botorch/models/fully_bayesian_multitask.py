# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Multi-task Gaussian Process Regression models with fully Bayesian inference."""

from collections.abc import Mapping
from typing import Any, NoReturn, TypeVar

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import (
    matern52_kernel,
    MCMC_DIM,
    MIN_INFERRED_NOISE_LEVEL,
    PyroModel,
    reshape_and_detach,
    SaasPyroModel,
)
from botorch.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    MultiTaskGPyTorchModel,
)
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.means.multitask_mean import MultitaskMean
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Self

# Can replace with Self type once 3.11 is the minimum version
TFullyBayesianMultiTaskGP = TypeVar(
    "TFullyBayesianMultiTaskGP", bound="FullyBayesianMultiTaskGP"
)


class MultiTaskPyroMixin:
    r"""Mixin with universal multi-task logic for PyroModel subclasses.

    Stores task-related attributes (``task_feature``, ``num_tasks``,
    ``task_rank``) and adjusts ``ard_num_dims`` to exclude the task column.
    Overrides ``sample_mean`` to return per-task means and
    ``_prepare_features`` to strip the task column.

    Place before the ``PyroModel`` subclass in the MRO.
    """

    def set_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        task_feature: int | None = None,
        task_rank: int | None = None,
        all_tasks: list[int] | None = None,
    ) -> None:
        """Set training data and configure multi-task attributes.

        Args:
            train_X: Training inputs (n x (d + 1)), including a task column.
            train_Y: Training targets (n x 1).
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            task_feature: The index of the task feature column.
            task_rank: The number of learned task embeddings. Defaults to
                the number of tasks.
            all_tasks: A list of all task indices. If omitted, all tasks will be
                inferred from the task feature column of the training data.
        """
        super().set_inputs(train_X, train_Y, train_Yvar)
        task_feature = task_feature % train_X.shape[-1]
        self.task_feature = task_feature
        if all_tasks is None:
            all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
        self.num_tasks = len(all_tasks)
        self.task_rank = task_rank or self.num_tasks
        self.ard_num_dims = self.train_X.shape[-1] - 1

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample per-task mean constants.

        Returns a vector of shape ``(num_tasks,)`` with one mean per task.
        """
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ).expand(torch.Size([self.num_tasks])),
        )

    def _get_task_indices_and_base_idxr(self, **tkwargs: Any) -> tuple[Tensor, Tensor]:
        r"""Compute the task indices and the base feature index selector.

        Returns:
            A tuple of ``(task_indices, base_idxr)`` where ``task_indices`` are
            long-typed task assignments and ``base_idxr`` selects the non-task
            columns.
        """
        base_idxr = torch.arange(self.ard_num_dims, device=tkwargs["device"])
        base_idxr[self.task_feature :] += 1
        task_indices = self.train_X[..., self.task_feature].to(
            device=tkwargs["device"], dtype=torch.long
        )
        return task_indices, base_idxr

    def _prepare_features(self, X: Tensor, **tkwargs: Any) -> Tensor:
        """Strip the task column from X, selecting only base features."""
        _, base_idxr = self._get_task_indices_and_base_idxr(**tkwargs)
        return X[..., base_idxr]


class LatentFeatureMultiTaskPyroMixin(MultiTaskPyroMixin):
    r"""Mixin that adds ICM-style multi-task capabilities via latent features.

    Extends ``MultiTaskPyroMixin`` with an ICM task covariance using learned
    latent task embeddings and a Matern-5/2 task kernel. Place before the
    ``PyroModel`` subclass in the MRO::

        class MultitaskSaasPyroModel(LatentFeatureMultiTaskPyroMixin, SaasPyroModel):
            ...

    Overrides the dispatch methods ``_maybe_multitask_transform``,
    ``_build_mean_module``, ``_build_multitask_covariance``,
    and ``get_dummy_mcmc_samples``.
    """

    def sample_latent_features(self, **tkwargs: Any) -> Tensor:
        r"""Sample latent task feature embeddings."""
        return pyro.sample(
            "latent_features",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ).expand(torch.Size([self.num_tasks, self.task_rank])),
        )

    def sample_task_lengthscale(
        self, concentration: float = 6.0, rate: float = 3.0, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the task kernel lengthscale."""
        return pyro.sample(
            "task_lengthscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ).expand(torch.Size([self.task_rank])),
        )

    def _build_task_covar(self, **tkwargs: Any) -> tuple[Tensor, Tensor]:
        r"""Sample latent features and task lengthscale and build n x n task covar.

        Returns:
            A tuple of ``(task_covar, task_indices)`` where ``task_covar`` is an
            ``n x n`` task covariance matrix and ``task_indices`` are the task
            assignments.
        """
        task_indices, _ = self._get_task_indices_and_base_idxr(**tkwargs)
        task_latent_features = self.sample_latent_features(**tkwargs)[task_indices]
        task_lengthscale = self.sample_task_lengthscale(**tkwargs)
        task_covar = matern52_kernel(
            X=task_latent_features, lengthscale=task_lengthscale
        )
        return task_covar, task_indices

    def _maybe_multitask_transform(
        self, K_noiseless: Tensor, mean: Tensor, **tkwargs: Any
    ) -> tuple[Tensor, Tensor]:
        r"""Multiply K by task covariance and index mean by task assignments."""
        task_covar, task_indices = self._build_task_covar(**tkwargs)
        K_noiseless = K_noiseless.mul(task_covar)
        return K_noiseless, mean[task_indices]

    def _build_mean_module(
        self,
        mcmc_samples: dict[str, Tensor],
        batch_shape: torch.Size,
        **tkwargs: Any,
    ) -> Mean:
        """Build a ``MultitaskMean`` with per-task constants from MCMC samples."""
        mean_module = MultitaskMean(
            base_means=ConstantMean(batch_shape=batch_shape),
            num_tasks=self.num_tasks,
        ).to(**tkwargs)
        for i in range(self.num_tasks):
            mean_module.base_means[i].constant.data = reshape_and_detach(
                target=mean_module.base_means[i].constant.data,
                new_value=mcmc_samples["mean"][:, i],
            )
        return mean_module

    def _build_multitask_covariance(
        self,
        mcmc_samples: dict[str, Tensor],
        covar_module: Kernel,
        batch_shape: torch.Size,
        **tkwargs: Any,
    ) -> Kernel:
        """Build task IndexKernel and combine with data covariance."""
        data_indices = torch.arange(self.train_X.shape[-1] - 1)
        data_indices[self.task_feature :] += 1
        covar_module.active_dims = data_indices.to(device=tkwargs["device"])

        latent_covar_module = MaternKernel(
            nu=2.5,
            ard_num_dims=self.task_rank,
            batch_shape=batch_shape,
        ).to(**tkwargs)
        latent_covar_module.lengthscale = reshape_and_detach(
            target=latent_covar_module.lengthscale,
            new_value=mcmc_samples["task_lengthscale"],
        )
        latent_features = mcmc_samples["latent_features"]
        task_covar = latent_covar_module(latent_features)
        task_covar_module = IndexKernel(
            num_tasks=self.num_tasks,
            rank=self.task_rank,
            batch_shape=latent_features.shape[:-2],
            active_dims=torch.tensor([self.task_feature], device=tkwargs["device"]),
        )
        task_covar_module.covar_factor = Parameter(
            task_covar.cholesky().to_dense().detach()
        )
        task_covar_module = task_covar_module.to(**tkwargs)
        task_covar_module.var = torch.zeros_like(task_covar_module.var)
        covar_module = covar_module * task_covar_module
        return covar_module

    def get_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        **tkwargs: Any,
    ) -> dict[str, Tensor]:
        """Return dummy MCMC samples for state dict loading.

        Calls ``super()`` for base model keys, then reshapes ``mean`` to
        ``(S, num_tasks)`` and adds ``task_lengthscale`` and
        ``latent_features``.
        """
        mcmc_samples = super().get_dummy_mcmc_samples(
            num_mcmc_samples=num_mcmc_samples, **tkwargs
        )
        mcmc_samples["mean"] = torch.ones(num_mcmc_samples, self.num_tasks, **tkwargs)
        mcmc_samples["task_lengthscale"] = torch.ones(
            num_mcmc_samples, self.task_rank, **tkwargs
        )
        mcmc_samples["latent_features"] = torch.ones(
            num_mcmc_samples, self.num_tasks, self.task_rank, **tkwargs
        )
        return mcmc_samples


class MultitaskSaasPyroModel(LatentFeatureMultiTaskPyroMixin, SaasPyroModel):
    r"""
    Multi-task SAAS model using latent task features. Backward-compatible
    subclass that composes ``LatentFeatureMultiTaskPyroMixin`` with
    ``SaasPyroModel``.
    """

    pass


class FullyBayesianMultiTaskGP(MultiTaskGP):
    r"""A fully Bayesian multi-task GP model.

    This model assumes that the inputs have been normalized to [0, 1]^d and that the
    output has been stratified standardized to have zero mean and unit variance for
    each task.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
        >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
        >>> train_X = torch.cat([
        >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
        >>> ])
        >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
        >>> train_Yvar = 0.01 * torch.ones_like(train_Y)
        >>> mt_gp = FullyBayesianMultiTaskGP(
        >>>     train_X, train_Y, task_feature=-1,
        >>>     pyro_model=MultitaskSaasPyroModel(),
        >>> )
        >>> fit_fully_bayesian_model_nuts(mt_gp)
        >>> posterior = mt_gp.posterior(test_X)
    """

    _is_fully_bayesian = True
    _is_ensemble = True

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        pyro_model: PyroModel | None = None,
        validate_task_values: bool = True,
    ) -> None:
        r"""Initialize the fully Bayesian multi-task GP model.

        Args:
            train_X: Training inputs (n x (d + 1))
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). If None, we infer the noise.
                Note that the inferred noise is common across all tasks.
            task_feature: The index of the task feature (``-d <= task_feature <= d``).
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The num of learned task embeddings to be used in the task kernel.
                If omitted, use a full rank (i.e. number of tasks) kernel.
            all_tasks: A list of all task indices. If omitted, all tasks will be
                inferred from the task feature column of the training data. Used to
                inform the model about the total number of tasks, including any
                unobserved tasks.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale).
                Note that ``.train()`` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied to the inputs ``X``
                in the model's forward pass.
            pyro_model: A ``PyroModel`` that inherits from ``MultiTaskPyroMixin``.
            validate_task_values: If True, validate that the task values supplied in the
                input are expected tasks values. If false, unexpected task values
                will be mapped to the first output_task if supplied.
        """
        if not (
            train_X.ndim == train_Y.ndim == 2
            and len(train_X) == len(train_Y)
            and train_Y.shape[-1] == 1
        ):
            raise ValueError(
                "Expected train_X to have shape n x d and train_Y to have shape n x 1"
            )
        if train_Yvar is not None and train_Y.shape != train_Yvar.shape:
            raise ValueError(
                "Expected train_Yvar to be None or have the same shape as train_Y"
            )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            outcome_transform.train()  # Ensure we learn parameters here on init
            train_Y, train_Yvar = outcome_transform(
                Y=train_Y, Yvar=train_Yvar, X=transformed_X
            )
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=task_feature,
            output_tasks=output_tasks,
            rank=rank,
            # We already transformed the data above, this avoids applying the
            # default ``Standardize`` transform twice. As outcome_transform is
            # set on ``self`` below, it will be applied to the posterior in the
            # ``posterior`` method of ``MultiTaskGP``.
            outcome_transform=None,
            all_tasks=all_tasks,
            validate_task_values=validate_task_values,
        )
        self.to(train_X)
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        if pyro_model is None:
            pyro_model = MultitaskSaasPyroModel()
        if not isinstance(pyro_model, MultiTaskPyroMixin):
            raise ValueError("pyro_model must be a multi-task model.")
        x_before, task_idcs, x_after = self._split_inputs(transformed_X)
        pyro_model.set_inputs(
            train_X=torch.cat([x_before, task_idcs, x_after], dim=-1),
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            task_feature=task_feature,
            task_rank=self._rank,
            all_tasks=all_tasks,
        )
        self.pyro_model: PyroModel = pyro_model
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def train(self, mode: bool = True, reset: bool = True) -> TFullyBayesianMultiTaskGP:
        r"""Puts the model in ``train`` mode.

        Args:
            mode: A boolean indicating whether to put the model in training mode.
            reset: A boolean indicating whether to reset the model to its initial
                state. If ``mode`` is False, this argument is ignored.

        Returns:
            The model itself.
        """
        super().train(mode=mode)
        if mode and reset:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
        return self

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.kernels[0].base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return self.covar_module.kernels[0].batch_shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that ``FullyBayesianMultiTaskGP`` does not support batching
        over input data at this point.
        """
        self._check_if_fitted()
        return torch.Size([self.num_mcmc_samples])

    def fantasize(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError("Fantasize is not implemented!")

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def load_mcmc_samples(self, mcmc_samples: dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by ``fit_fully_bayesian_model_nuts`` when the model
        has been fitted in order to create a batched MultiTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
            _,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def eval(self) -> Self:
        r"""Puts the model in eval mode.

        Circumvents the need to call MultiTaskGP.eval(), which computes the
        task_covar_matrix for non-observed tasks. This is not needed for fully
        Bayesian models, since the non-observed tasks' covar factors are instead
        sampled.

        Returns:
            The model itself.
        """
        self._check_if_fitted()
        return MultiTaskGPyTorchModel.eval(self)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Returns:
            A ``GaussianMixturePosterior`` object. Includes observation noise
                if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X.unsqueeze(MCMC_DIM),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = GaussianMixturePosterior(distribution=posterior.distribution)
        return posterior

    def forward(self, X: Tensor) -> MultivariateNormal:
        self._check_if_fitted()
        return super().forward(X)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling ``load_state_dict`` currently doesn't
        play well with the ``FullyBayesianMultiTaskGP`` since the
        ``mean_module``, ``covar_module`` and ``likelihood`` aren't initialized
        until the model has been fitted. The reason for this is that we don't
        know the number of MCMC samples until NUTS is called. Given the state
        dict, we can initialize a new model with some dummy samples and then
        load the state dict into this model. The dummy samples are obtained
        from ``pyro_model.get_dummy_mcmc_samples()``.
        """
        raw_mean = state_dict["mean_module.base_means.0.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        mcmc_samples = self.pyro_model.get_dummy_mcmc_samples(
            num_mcmc_samples=num_mcmc_samples, **tkwargs
        )
        self.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> BatchedMultiOutputGPyTorchModel:
        """Conditions on additional observations for a Fully Bayesian model (either
        identical across models or unique per-model).

        Args:
            X: A ``batch_shape x num_samples x d``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.
            Y: A ``batch_shape x num_samples x 1``-dim Tensor, where ``d`` is
                the dimension of the feature space and ``batch_shape`` is the number of
                sampled models.

        Returns:
            BatchedMultiOutputGPyTorchModel: A fully bayesian model conditioned on
              given observations. The returned model has ``batch_shape`` copies of the
              training data in case of identical observations (and ``batch_shape``
              training datasets otherwise).
        """
        if X.ndim == 2 and Y.ndim == 2:
            # To avoid an error in GPyTorch when inferring the batch dimension, we add
            # the explicit batch shape here. The result is that the conditioned model
            # will have 'batch_shape' copies of the training data.
            X = X.repeat(self.batch_shape + (1, 1))
            Y = Y.repeat(self.batch_shape + (1, 1))

        elif X.ndim < Y.ndim:
            # We need to duplicate the training data to enable correct batch
            # size inference in gpytorch.
            X = X.repeat(*(Y.shape[:-2] + (1, 1)))

        return super().condition_on_observations(X, Y, **kwargs)


class SaasFullyBayesianMultiTaskGP(FullyBayesianMultiTaskGP):
    r"""A fully Bayesian multi-task GP model with the SAAS prior by default."""
