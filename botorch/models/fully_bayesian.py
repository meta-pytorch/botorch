# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Gaussian Process Regression models with fully Bayesian inference.

Fully Bayesian models use Bayesian inference over model hyperparameters, such
as lengthscales and noise variance, learning a posterior distribution for the
hyperparameters using the No-U-Turn-Sampler (NUTS). This is followed by
sampling a small set of hyperparameters (often ~16) from the posterior
that we will use for model predictions and for computing acquisition function
values. By contrast, our “standard” models (e.g.
``SingleTaskGP``) learn only a single best value for each hyperparameter using
MAP. The fully Bayesian method generally results in a better and more
well-calibrated model, but is more computationally intensive. For a full
description, see [Eriksson2021saasbo].

We use a lightweight PyTorch implementation of a Matern-5/2 kernel as there are
some performance issues with running NUTS on top of standard GPyTorch models.
The resulting hyperparameter samples are loaded into a batched GPyTorch model
after fitting.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import log, sqrt
from typing import Any, TypeVar

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    Warp,
)
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.utils import kumaraswamy_warp, subset_transform
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.kernel import dist, Kernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.means.multitask_mean import MultitaskMean
from gpytorch.models.exact_gp import ExactGP
from pyro.ops.integrator import register_exception_handler
from torch import Tensor
from torch.nn.parameter import Parameter

# Can replace with Self type once 3.11 is the minimum version
TFullyBayesianSingleTaskGP = TypeVar(
    "TFullyBayesianSingleTaskGP", bound="FullyBayesianSingleTaskGP"
)

_sqrt5 = math.sqrt(5)


def _handle_torch_linalg(exception: Exception) -> bool:
    return type(exception) is torch.linalg.LinAlgError


def _handle_valerr_in_dist_init(exception: Exception) -> bool:
    if type(exception) is not ValueError:
        return False
    return "satisfy the constraint PositiveDefinite()" in str(exception)


register_exception_handler("torch_linalg", _handle_torch_linalg)
register_exception_handler("valerr_in_dist_init", _handle_valerr_in_dist_init)


def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    dist = compute_dists(X=X, lengthscale=lengthscale)
    sqrt5_dist = _sqrt5 * dist
    return sqrt5_dist.add(1 + 5 / 3 * (dist**2)) * torch.exp(-sqrt5_dist)


def linear_kernel(X: Tensor, weight_variance: Tensor) -> Tensor:
    """Linear kernel."""
    Xw = X * weight_variance.sqrt()
    return Xw @ Xw.t()


def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances."""
    scaled_X = X / lengthscale
    return dist(scaled_X, scaled_X, x1_eq_x2=True)


def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape ``new_value`` to match ``target``."""
    return new_value.detach().clone().view(target.shape).to(target)


class PyroModel:
    r"""
    Base class for a Pyro model; used to assist in learning hyperparameters.

    This class and its subclasses are not a standard BoTorch models; instead
    the subclasses are used as inputs to a ``SaasFullyBayesianSingleTaskGP``,
    which should then have its hyperparameters fit with
    ``fit_fully_bayesian_model_nuts``. (By default, its subclass ``SaasPyroModel``
    is used).  A ``PyroModel``'s ``sample`` method should specify lightweight
    PyTorch functionality, which will be used for fast model fitting with NUTS.
    The utility of ``PyroModel`` is in enabling fast fitting with NUTS, since we
    would otherwise need to use GPyTorch, which is computationally infeasible
    in combination with Pyro.
    """

    _prior_mode: bool = False
    _noiseless_eps_for_sampleability: float = 1e-7

    def __init__(
        self,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
        eps: float = 1e-7,
        is_multitask: bool = False,
    ) -> None:
        r"""Initialize the PyroModel.

        Args:
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.
            eps: A small value that is used to ensure inputs are not 0 or 1,
                when using input warping.
            is_multitask: A boolean indicating whether to use multi-task mode.
        """
        self.use_input_warping = use_input_warping
        self.indices = indices_to_warp
        self._eps = eps
        self.is_multitask = is_multitask

    @subset_transform
    def warp(self, X: Tensor, c0: Tensor, c1: Tensor) -> Tensor:
        r"""Warp the input through a Kumaraswamy CDF."""
        return kumaraswamy_warp(X=X, c0=c0, c1=c1, eps=self._eps)

    def _maybe_input_warp(self, X: Tensor, **tkwargs: Any) -> Tensor:
        if self.use_input_warping:
            c0, c1 = self.sample_concentrations(**tkwargs)
            # unnormalize X from [0, 1] to [eps, 1-eps]
            return self.warp(X=self.train_X, c0=c0, c1=c1)
        else:
            return self.train_X

    def set_inputs(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        task_feature: int | None = None,
        task_rank: int | None = None,
    ) -> None:
        """Set the training data.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            task_feature: The index of the task feature column. Required when
                ``is_multitask=True``.
            task_rank: The number of learned task embeddings. Defaults to
                the number of tasks when ``is_multitask=True``.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.ard_num_dims = self.train_X.shape[-1]
        if self.is_multitask:
            if task_feature is None:
                raise ValueError(
                    "task_feature is required when is_multitask=True."
                )
            task_feature = task_feature % train_X.shape[-1]
            self.task_feature = task_feature
            all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
            self.num_tasks = len(all_tasks)
            self.task_rank = task_rank or self.num_tasks
            # exclude the task column from the ARD dimensions
            self.ard_num_dims = self.train_X.shape[-1] - 1

    @abstractmethod
    def sample(self) -> None:
        r"""Sample from the model."""
        pass  # pragma: no cover

    @abstractmethod
    def postprocess_mcmc_samples(
        self,
        mcmc_samples: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Post-process the final MCMC samples."""
        pass  # pragma: no cover

    @abstractmethod
    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood]:
        pass  # pragma: no cover

    @abstractmethod
    def get_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        r"""Return dummy MCMC samples for initializing the model in
        ``load_state_dict``.

        Each subclass provides its kernel-specific keys (e.g. lengthscale,
        weight_variance).  The base implementation is responsible for common
        keys shared across all models (noise, input-warping concentrations,
        and multitask embeddings).

        Args:
            num_mcmc_samples: Number of MCMC samples.
            dim: Number of input dimensions (excluding the task feature for
                multitask models).
            dtype: Tensor dtype.
            device: Tensor device.

        Returns:
            A dictionary mapping sample names to tensors.
        """
        pass  # pragma: no cover

    def _get_base_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        r"""Return dummy samples for keys common to all PyroModel subclasses.

        Handles noise, input-warping concentrations, and multitask embeddings.
        Subclasses should call this and merge with their kernel-specific keys.
        """
        tkwargs: dict[str, Any] = {"dtype": dtype, "device": device}
        mcmc_samples: dict[str, Tensor] = {}
        if self.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        if self.use_input_warping:
            mcmc_samples["c0"] = torch.ones(num_mcmc_samples, dim, **tkwargs)
            mcmc_samples["c1"] = torch.ones(num_mcmc_samples, dim, **tkwargs)
        if self.is_multitask:
            mcmc_samples["task_lengthscale"] = torch.ones(
                num_mcmc_samples, self.task_rank, **tkwargs
            )
            mcmc_samples["latent_features"] = torch.ones(
                num_mcmc_samples, self.num_tasks, self.task_rank, **tkwargs
            )
        return mcmc_samples

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant.

        Returns a scalar when single-task, or a ``(num_tasks,)`` vector when
        ``is_multitask=True``.
        """
        shape = torch.Size([self.num_tasks]) if self.is_multitask else torch.Size([])
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ).expand(shape),
        )

    def sample_concentrations(self, **tkwargs: Any) -> tuple[Tensor, Tensor]:
        r"""Sample concentrations for input warping.

        The prior has a mean value of 1 for each concentration and is very
        concentrated around the mean.
        """
        d = len(self.indices) if self.indices is not None else self.ard_num_dims
        c0 = pyro.sample(
            "c0",
            pyro.distributions.LogNormal(
                torch.tensor([0.0] * d, **tkwargs),
                torch.tensor([0.1**0.5] * d, **tkwargs),
            ),
        )
        c1 = pyro.sample(
            "c1",
            pyro.distributions.LogNormal(
                torch.tensor([0.0] * d, **tkwargs),
                torch.tensor([0.1**0.5] * d, **tkwargs),
            ),
        )

        return c0, c1

    def sample_latent_features(self, **tkwargs: Any) -> Tensor:
        r"""Sample latent task feature embeddings for multi-task models."""
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
        r"""Sample the task kernel lengthscale for multi-task models."""
        return pyro.sample(
            "task_lengthscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ).expand(torch.Size([self.task_rank])),
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
        r"""If multi-task, multiply K by task covar and index mean by task.

        Otherwise passes through unchanged. Analogous to ``_maybe_input_warp``.

        Returns:
            A tuple of ``(K_noiseless, mean_expanded)`` ready for
            ``sample_observations``.
        """
        if not self.is_multitask:
            return K_noiseless, mean
        task_covar, task_indices = self._build_task_covar(**tkwargs)
        K_noiseless = K_noiseless.mul(task_covar)
        return K_noiseless, mean[task_indices]

    def _load_multitask_components(
        self,
        mcmc_samples: dict[str, Tensor],
        data_covar_module: Kernel,
        batch_shape: torch.Size,
        **tkwargs: Any,
    ) -> tuple[MultitaskMean, Kernel]:
        r"""Build MultitaskMean and task IndexKernel from MCMC samples.

        Sets ``active_dims`` on the data covariance module to exclude the task
        feature column, constructs a ``MultitaskMean`` with per-task constants,
        and builds an ``IndexKernel`` from the sampled latent features and task
        lengthscales.

        Returns:
            A tuple of ``(mean_module, covar_module)`` where ``covar_module``
            is ``data_covar_module * task_covar_module``.
        """
        mean_module = MultitaskMean(
            base_means=ConstantMean(batch_shape=batch_shape),
            num_tasks=self.num_tasks,
        ).to(**tkwargs)
        for i in range(self.num_tasks):
            mean_module.base_means[i].constant.data = reshape_and_detach(
                target=mean_module.base_means[i].constant.data,
                new_value=mcmc_samples["mean"][:, i],
            )

        data_indices = torch.arange(self.train_X.shape[-1] - 1)
        data_indices[self.task_feature :] += 1
        data_covar_module.active_dims = data_indices.to(device=tkwargs["device"])

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
        covar_module = data_covar_module * task_covar_module
        return mean_module, covar_module

    def sample_observations(
        self,
        mean: Tensor,
        K_noiseless: Tensor,
        noise: Tensor,
        **tkwargs: Any,
    ) -> None:
        r"""Sample the observations Y (or prior samples in prior mode).

        Args:
            mean: The mean constant.
            K_noiseless: The kernel matrix without noise.
            noise: The noise variance.
            **tkwargs: dtype and device keyword arguments.
        """
        if self.train_Y.shape[-2] == 0:
            # Do not attempt to sample Y if the data is empty.
            return

        n = self.train_X.shape[0]
        K = K_noiseless + noise * torch.eye(n, **tkwargs)

        if self._prior_mode:
            self.f_prior_sample = pyro.sample(
                "f",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(n),
                    covariance_matrix=K_noiseless
                    + self._noiseless_eps_for_sampleability * torch.eye(n, **tkwargs),
                    # sadly need to add a little bit of noise to be possible
                    # to sample from this
                ),
            )
            self.Y_prior_sample = pyro.sample(
                "Y",
                pyro.distributions.Normal(
                    loc=self.f_prior_sample,
                    scale=noise.sqrt(),
                ),
            )
        else:
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(n),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )


class MaternPyroModel(PyroModel):
    r"""Implementation of the a fully Bayesian model with a dimension-scaling prior.

    ``MaternPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``FullyBayesianSingleTaskGP``.
    """

    _outputscale_prior_concentration: float | None = None
    _outputscale_prior_rate: float | None = None

    def sample(self) -> None:
        r"""Sample from the Matern pyro model.

        This samples the mean, noise variance, (optional) outputscale, and
        lengthscales according to a dimension-scaled prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(
            concentration=self._outputscale_prior_concentration,
            rate=self._outputscale_prior_rate,
            **tkwargs,
        )
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        X_tf = self._maybe_input_warp(self.train_X, **tkwargs)
        if self.is_multitask:
            _, base_idxr = self._get_task_indices_and_base_idxr(**tkwargs)
            X_tf = X_tf[..., base_idxr]
        K_noiseless = outputscale * matern52_kernel(X=X_tf, lengthscale=lengthscale)
        K_noiseless, mean = self._maybe_multitask_transform(
            K_noiseless, mean, **tkwargs
        )
        self.sample_observations(
            mean=mean, K_noiseless=K_noiseless, noise=noise, **tkwargs
        )

    def sample_lengthscale(self, dim: int, **tkwargs: Any) -> Tensor:
        r"""Sample the lengthscale."""
        return pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                loc=torch.full((dim,), sqrt(2) + log(dim) * 0.5, **tkwargs),
                scale=torch.full((dim,), sqrt(3), **tkwargs),
            ),
        )

    def sample_outputscale(
        self,
        concentration: float | None = None,
        rate: float | None = None,
        **tkwargs: Any,
    ) -> Tensor:
        r"""Sample the outputscale.

        If the concentration or rate arguments are None, then an outputscale
        of 1 is used.

        Args:
            concentration: The concentration parameter for a GammaPrior.
            rate: The rate parameter for a GammaPrior.

        Returns:
            The outputscale.
        """
        if concentration is None or rate is None:
            return torch.ones(1, **tkwargs)
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def postprocess_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def get_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        r"""Return dummy MCMC samples for a Matern model."""
        tkwargs: dict[str, Any] = {"dtype": dtype, "device": device}
        mean_shape = (
            (num_mcmc_samples, self.num_tasks)
            if self.is_multitask
            else (num_mcmc_samples,)
        )
        mcmc_samples = {
            "mean": torch.ones(*mean_shape, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            **self._get_base_dummy_mcmc_samples(num_mcmc_samples, dim, dtype, device),
        }
        return mcmc_samples

    def _get_covar_module(
        self,
        use_scale_kernel: bool,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        """Get the covar module to load samples into.

        Args:
            use_scale_kernel: A boolean indicating whether to use an outputscale other
                than 1.
            batch_shape: The batch shape (number of mcmc samples).
            dtype: The dtype.
            device: The device.

        Returns:
            The covar module.
        """
        covar_module = MaternKernel(
            ard_num_dims=self.ard_num_dims,
            batch_shape=batch_shape,
        )
        if use_scale_kernel:
            covar_module = ScaleKernel(covar_module, batch_shape=batch_shape)
        return covar_module.to(dtype=dtype, device=device)

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, Warp | None]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        # For multitask, pass a scalar mean slice to build covar/likelihood,
        # then replace mean_module with MultitaskMean below.
        mean_mcmc = mcmc_samples
        if self.is_multitask:
            mean_mcmc = {**mcmc_samples, "mean": mcmc_samples["mean"][:, 0]}

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        outputscale = mean_mcmc.get("outputscale")
        covar_module = self._get_covar_module(
            use_scale_kernel=outputscale is not None, batch_shape=batch_shape, **tkwargs
        )
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape ``num_mcmc_samples x N``
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        if isinstance(covar_module, ScaleKernel):
            covar_module.outputscale = reshape_and_detach(
                target=covar_module.outputscale,
                new_value=mcmc_samples["outputscale"],
            )
            base_kernel = covar_module.base_kernel
        else:
            base_kernel = covar_module
        base_kernel.lengthscale = reshape_and_detach(
            target=base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mean_mcmc["mean"],
        )
        if self.use_input_warping:
            indices = (
                list(range(self.ard_num_dims)) if self.indices is None else self.indices
            )
            bounds = torch.zeros(2, self.ard_num_dims, **tkwargs)
            bounds[1] = 1
            warping_function = Warp(
                d=self.ard_num_dims,
                batch_shape=batch_shape,
                indices=indices,
                bounds=bounds,
            ).to(**tkwargs)
            warping_function.concentration0.data = reshape_and_detach(
                target=warping_function.concentration0,
                new_value=mcmc_samples["c0"],
            )
            warping_function.concentration1.data = reshape_and_detach(
                target=warping_function.concentration1,
                new_value=mcmc_samples["c1"],
            )
        else:
            warping_function = None

        if self.is_multitask:
            mean_module, covar_module = self._load_multitask_components(
                mcmc_samples=mcmc_samples,
                data_covar_module=covar_module,
                batch_shape=batch_shape,
                **tkwargs,
            )
        return mean_module, covar_module, likelihood, warping_function


class SaasPyroModel(MaternPyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    ``SaasPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``SaasFullyBayesianSingleTaskGP``. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    ``SaasPyroModel`` unless they want to customize its attributes (such as
    ``covar_module``).
    """

    _outputscale_prior_concentration: float | None = 2.0
    _outputscale_prior_rate: float | None = 0.15

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            inv_length_sq.rsqrt(),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            mcmc_samples["kernel_tausq"].unsqueeze(-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = inv_length_sq.rsqrt()
        # Delete ``kernel_tausq`` and ``_kernel_inv_length_sq`` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples

    def get_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        r"""Return dummy MCMC samples for a SAAS model (adds outputscale)."""
        mcmc_samples = super().get_dummy_mcmc_samples(
            num_mcmc_samples, dim, dtype, device
        )
        mcmc_samples["outputscale"] = torch.ones(
            num_mcmc_samples, dtype=dtype, device=device
        )
        return mcmc_samples


class LinearPyroModel(PyroModel):
    r"""Implementation of a Bayesian Linear pyro model.

    ``LinearPyroModel`` is not a standard BoTorch model; instead, it is used as
    an input to ``FullyBayesianLinearSingleTaskGP``. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    ``LinearPyroModel`` unless they want to customize its attributes (such as
    ``covar_module``).
    """

    def sample(self) -> None:
        r"""Sample from the model."""
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        mean = self.sample_mean(**tkwargs)
        weight_variance = self.sample_weight_variance(**tkwargs)
        X_tf = self._maybe_input_warp(X=self.train_X, **tkwargs)
        if self.is_multitask:
            _, base_idxr = self._get_task_indices_and_base_idxr(**tkwargs)
            X_tf = X_tf[..., base_idxr]
        X_tf = X_tf - 0.5  # center transformed data at 0 (for linear model)
        K_noiseless = linear_kernel(X=X_tf, weight_variance=weight_variance)
        noise = self.sample_noise(**tkwargs)
        K_noiseless, mean = self._maybe_multitask_transform(
            K_noiseless, mean, **tkwargs
        )
        self.sample_observations(
            mean=mean, K_noiseless=K_noiseless, noise=noise, **tkwargs
        )

    def sample_weight_variance(self, alpha: float = 0.1, **tkwargs: Any) -> Tensor:
        r"""Sample the weight variance.

        This is a hierarchical prior is a half-Cauchy prior on the prior weight
        covariance, which is diagonal with different values for each input
        dimension. The prior samples a global level of sparsity (tau) and which
        scales the HalfCauchy prior on the weight variance. Since the weight prior
        is centered at zero, a prior variance of 0, would correspond to the
        dimension being irrelevant. This choice of prior is motivated by Saas
        priors.
        """
        tau_sq = pyro.sample(
            "tau_sq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        weight_variance_sq = pyro.sample(
            "_weight_variance_sq",
            pyro.distributions.HalfCauchy(torch.ones(self.ard_num_dims, **tkwargs)),
        )
        return pyro.deterministic(
            "weight_variance", (tau_sq * weight_variance_sq).sqrt()
        )

    def postprocess_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true weight variance and removes tausq (global shrinkage).
        """
        mcmc_samples["weight_variance"] = (
            mcmc_samples["tau_sq"].unsqueeze(-1) * mcmc_samples["_weight_variance_sq"]
        ).sqrt()
        del mcmc_samples["tau_sq"], mcmc_samples["_weight_variance_sq"]
        return mcmc_samples

    def get_dummy_mcmc_samples(
        self,
        num_mcmc_samples: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        r"""Return dummy MCMC samples for a Linear model."""
        tkwargs: dict[str, Any] = {"dtype": dtype, "device": device}
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "weight_variance": torch.ones(num_mcmc_samples, dim, **tkwargs),
            **self._get_base_dummy_mcmc_samples(num_mcmc_samples, dim, dtype, device),
        }
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood, InputTransform]:
        r"""Load the MCMC samples into their corresponding modules."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["weight_variance"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = LinearKernel(
            batch_shape=batch_shape,
            ard_num_dims=self.ard_num_dims,
        ).to(**tkwargs)

        bounds = torch.zeros(2, self.ard_num_dims, **tkwargs)
        bounds[1] = 1
        input_tf = Normalize(
            d=self.ard_num_dims,
            bounds=bounds,
            center=0.0,
            # batch shape passed here when using input warping
            # which is applied first and adds a batch dimension
            batch_shape=batch_shape if self.use_input_warping else torch.Size([]),
        )
        indices = (
            list(range(self.ard_num_dims)) if self.indices is None else self.indices
        )
        if self.use_input_warping:
            warping_function = Warp(
                d=self.ard_num_dims,
                batch_shape=batch_shape,
                indices=indices,
                bounds=bounds,
            ).to(**tkwargs)
            warping_function.concentration0.data = reshape_and_detach(
                target=warping_function.concentration0,
                new_value=mcmc_samples["c0"],
            )
            warping_function.concentration1.data = reshape_and_detach(
                target=warping_function.concentration1,
                new_value=mcmc_samples["c1"],
            )
            input_tf = ChainedInputTransform(warp=warping_function, normalize=input_tf)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape ``num_mcmc_samples x N``
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.variance = reshape_and_detach(
            target=covar_module.variance,
            new_value=mcmc_samples["weight_variance"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood, input_tf


class AbstractFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel, ABC):
    r"""An abstract fully Bayesian single-task GP model.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.
    """

    _is_fully_bayesian = True
    _is_ensemble = True
    _pyro_model_class: type[PyroModel] = PyroModel

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] = None,
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale).
                Note that ``.train()`` will be called on the outcome transform during
                instantiation of the model.
            input_transform: An input transform that is applied in the model's
                forward pass.
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.
        """
        if not (
            train_X.ndim == train_Y.ndim == 2
            and len(train_X) == len(train_Y)
            and train_Y.shape[-1] == 1
        ):
            raise ValueError(
                "Expected train_X to have shape n x d and train_Y to have shape n x 1"
            )
        if train_Yvar is not None:
            if train_Y.shape != train_Yvar.shape:
                raise ValueError(
                    "Expected train_Yvar to be None or have the same shape as train_Y"
                )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, train_Yvar = outcome_transform(
                Y=train_Y, Yvar=train_Yvar, X=transformed_X
            )
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._num_outputs: int = train_Y.shape[-1]
        self._input_batch_shape: torch.Size = train_X.shape[:-2]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        self.pyro_model = self._pyro_model_class(
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )
        self.pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        if outcome_transform is not None:
            self.outcome_transform: OutcomeTransform = outcome_transform
        if input_transform is not None:
            self.input_transform: InputTransform = input_transform

    def _check_if_fitted(self) -> None:
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return self.covar_module.batch_shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that ``SaasFullyBayesianSingleTaskGP`` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    @property
    def _aug_batch_shape(self) -> torch.Size:
        r"""The batch shape of the model, augmented to include the output dim."""
        aug_batch_shape = self.batch_shape
        if self.num_outputs > 1:
            aug_batch_shape += torch.Size([self.num_outputs])
        return aug_batch_shape

    def train(
        self: TFullyBayesianSingleTaskGP, mode: bool = True, reset: bool = True
    ) -> TFullyBayesianSingleTaskGP:
        r"""Puts the model in ``train`` mode.

        Args:
            mode: A boolean indicating whether to put the model in training mode.
            reset: A boolean indicating whether to reset the model to its initial
                state if mode is True. If ``mode`` is False, this argument is ignored.

        Returns:
            The model itself.
        """
        super().train(mode=mode)
        if mode and reset:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None
        return self

    def load_mcmc_samples(self, mcmc_samples: dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by ``fit_fully_bayesian_model_nuts`` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (self.mean_module, self.covar_module, self.likelihood, input_transform) = (
            self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        )
        if input_transform is not None:
            if hasattr(self, "input_transform"):
                tfs = [self.input_transform]
                if isinstance(input_transform, ChainedInputTransform):
                    tfs.extend(list(input_transform.values()))
                else:
                    tfs.append(input_transform)
                self.input_transform = ChainedInputTransform(
                    **{f"tf{i}": tf for i, tf in enumerate(tfs)}
                )
            else:
                self.input_transform = input_transform

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' ``forward`` methods, there is no
        ``if self.training`` block, because it ought to be unreachable:
        If ``self.train()`` has been called, then ``self.covar_module`` will
        be None, ``check_if_fitted()`` will fail, and the rest of this method
        will not run.
        """
        self._check_if_fitted()
        if self.training:
            X = self.transform_inputs(X=X)
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A ``(batch_shape) x q x d``-dim Tensor, where ``d`` is the dimension
                of the feature space and ``q`` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape ``(batch_shape) x q x m``).
            posterior_transform: An optional PosteriorTransform.

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

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        *,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> dict[str, BotorchContainer | Tensor | None]:
        r"""Construct ``SingleTaskGP`` keyword arguments from a ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset``, with attributes ``train_X``,
                ``train_Y``, and, optionally, ``train_Yvar``.
            use_input_warping: A boolean indicating whether to use input warping.
            indices_to_warp: An optional list of indices to warp. The default
                is to warp all inputs.

        Returns:
            A dict of keyword arguments that can be used to initialize a
            ``FullyBayesianLinearSingleTaskGP``, with keys ``train_X``, ``train_Y``,
            ``use_input_warping``, ``indices_to_warp``, and, optionally, ``train_Yvar``.
        """
        return {
            **super().construct_inputs(training_data=training_data),
            "use_input_warping": use_input_warping,
            "indices_to_warp": indices_to_warp,
        }


class FullyBayesianSingleTaskGP(AbstractFullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``. A model with a Matern-5/2 kernel
    and dimension-scaled priors on the hyperparameters from [Hvarfner2024vanilla]_
    is used by default.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> fully_bayesian_gp = FullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(fully_bayesian_gp)
        >>> posterior = fully_bayesian_gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = MaternPyroModel

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        if isinstance(self.covar_module, ScaleKernel):
            base_kernel = self.covar_module.base_kernel
        else:
            base_kernel = self.covar_module
        lengthscale = base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> None:
        r"""Custom logic for loading the state dict.

        The standard approach of calling ``load_state_dict`` currently doesn't
        play well with the ``FullyBayesianSingleTaskGP`` since the
        ``mean_module``, ``covar_module`` and ``likelihood`` aren't initialized
        until the model has been fitted. The reason for this is that we don't
        know the number of MCMC samples until NUTS is called. Given the state
        dict, we can initialize a new model with some dummy samples and then
        load the state dict into this model. This delegates to the PyroModel's
        ``get_dummy_mcmc_samples`` method.
        """
        raw_mean = state_dict["mean_module.raw_constant"]
        mcmc_samples = self.pyro_model.get_dummy_mcmc_samples(
            num_mcmc_samples=len(raw_mean),
            dim=self.pyro_model.train_X.shape[-1],
            dtype=raw_mean.dtype,
            device=raw_mean.device,
        )
        self.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)


class SaasFullyBayesianSingleTaskGP(FullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``. The SAAS model
    [Eriksson2021saasbo]_ with a Matern-5/2 kernel is used by default.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model
    as it isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = SaasPyroModel


class FullyBayesianLinearSingleTaskGP(AbstractFullyBayesianSingleTaskGP):
    r"""A fully Bayesian single-task GP model with a linear kernel.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an ``input_transform`` and ``outcome_transform``.

    You are expected to use ``fit_fully_bayesian_model_nuts`` to fit this model as it
    isn't compatible with ``fit_gpytorch_mll``.

    Example:
        >>> gp = FullyBayesianLinearSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
        >>> posterior = gp.posterior(test_X)
    """

    _pyro_model_class: type[PyroModel] = LinearPyroModel

    @property
    def median_weight_variance(self) -> Tensor:
        r"""Median weight variance across the MCMC samples."""
        self._check_if_fitted()
        weight_variance = self.covar_module.variance.clone()
        return weight_variance.median(0).values.squeeze(0)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> None:
        r"""Custom logic for loading the state dict.

        Delegates to the PyroModel's ``get_dummy_mcmc_samples`` to construct
        the dummy samples needed for initialization.
        """
        weight_variance = state_dict["covar_module.raw_variance"]
        num_mcmc_samples = len(weight_variance)
        dim = self.pyro_model.train_X.shape[-1]
        mcmc_samples = self.pyro_model.get_dummy_mcmc_samples(
            num_mcmc_samples=num_mcmc_samples,
            dim=dim,
            dtype=weight_variance.dtype,
            device=weight_variance.device,
        )
        self.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)
