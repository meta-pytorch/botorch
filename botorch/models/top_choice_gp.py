#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Variational GP for preference learning from top-choice (best-of-k) observations.

Each observation is a comparison that presents ``k >= 2`` datapoints and records the
one chosen as the best. The latent utility of the datapoints is modeled with a GP and
the choices with a softmax likelihood, see ``TopChoiceSoftmaxLikelihood``. For
``k = 2``, this is a pairwise comparison model that follows the ``PairwiseGP`` API.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import (
    _get_single_precision_warning,
    InputDataWarning,
    OptimizationWarning,
)
from botorch.fit import _fit_fallback
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.likelihoods.top_choice import TopChoiceSoftmaxLikelihood
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.optim.core import OptimizationResult
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.posteriors.posterior import Posterior
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import MarginalLogLikelihood, VariationalELBO
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from linear_operator.operators import LinearOperator, TriangularLinearOperator
from torch import Tensor
from torch.nn import Parameter


class VariationalTopChoiceGP(GPyTorchModel, ApproximateGP):
    r"""Variational GP for preference learning from top-choice observations.

    Notation: ``m`` comparisons are made among ``n`` datapoints of dimension ``d``.
    Each comparison presents ``k >= 2`` of the datapoints and records the one chosen
    as the best.

    The latent utility ``f`` has a zero-mean GP prior, and the probability of choosing
    datapoint ``x_i`` among the datapoints ``{x_j}`` of a comparison is the softmax
    ``exp(f(x_i)) / sum_j exp(f(x_j))``. The outputscale of the kernel controls the
    softmax temperature (its square root acts as an inverse temperature), i.e., how
    noisy the choices are.

    For ``k = 2``, the likelihood is the logistic pairwise comparison likelihood, i.e.
    the model is a variational counterpart of ``PairwiseGP`` with
    ``PairwiseLogitLikelihood``, and it follows the ``PairwiseGP`` API: comparisons
    are rows of datapoint indices with the preferred datapoint first,
    ``condition_on_observations`` takes comparisons as ``Y``, and ``load_state_dict``
    supports loading the hyperparameters of a model fit to different data. Unlike
    ``PairwiseGP``, batched data, models without data, and ``fantasize`` are not
    supported.

    Inference uses a variational GP whose inducing points are the unique datapoints
    that appear in at least one comparison, so that the only approximation is the
    Gaussian variational family (unless the input transform has learnable
    parameters, which are fit while the inducing points stay fixed). Fitting takes
    minutes for about a thousand comparisons. The memory cost is quadratic in
    ``m * k`` for fitting, which computes the joint posterior covariance of all
    ``m * k`` training inputs, and proportional to the batch size times the squared
    number of inducing points for batched posteriors, e.g., when optimizing
    acquisition functions.

    The training inputs are the ``m * k`` (transformed) datapoints of the comparisons,
    flattened in comparison-major order, and the training targets are the ``m``
    positions of the chosen datapoint within each comparison (always 0).

    Example:
        >>> datapoints = torch.rand(10, 3, dtype=torch.float64)
        >>> # Each row is a comparison; its first element is the chosen datapoint.
        >>> comparisons = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        >>> model = VariationalTopChoiceGP(datapoints, comparisons)
        >>> mll = TopChoiceVariationalELBO(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
    """

    _num_outputs: int = 1
    # The training inputs are always transformed, see ``train_inputs``.
    _has_transformed_inputs: bool = True

    def __init__(
        self,
        datapoints: Tensor,
        comparisons: Tensor,
        likelihood: TopChoiceSoftmaxLikelihood | None = None,
        covar_module: ScaleKernel | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""Initialize the model.

        Args:
            datapoints: An ``n x d`` tensor of the compared datapoints.
            comparisons: An ``m x k`` tensor (``k >= 2``) of indices into
                ``datapoints``. Each row is a comparison. Its first element is the
                index of the chosen datapoint; the order of the remaining elements
                is irrelevant.
            likelihood: A ``TopChoiceSoftmaxLikelihood`` with ``num_alternatives=k``.
                Defaults to one with the default number of samples.
            covar_module: The covariance module, which must be a ``ScaleKernel``
                since its outputscale controls the effective softmax temperature.
                Defaults to a ``ScaleKernel`` with a ``GammaPrior(2.0, 0.15)``
                outputscale prior around an RBF kernel with a dimension-scaled
                lengthscale prior, see ``get_covar_module_with_dim_scaled_prior``.
                Since consistent choices only weakly constrain the scale of the
                utility, the outputscale and any other scale parameter of the
                kernel, e.g., the variances of a ``LinearKernel``, need priors that
                penalize both large values and values near zero, such as the default
                outputscale prior. With a flat prior, such as the outputscale prior
                of the default kernel of ``PairwiseGP``, or without a prior, fitting
                can fail.
            input_transform: An input transform. Its coefficients that are set from
                data, such as the bounds of ``Normalize``, are set from
                ``datapoints`` at initialization. Its learnable parameters, such as
                those of ``Warp``, are fit with the other hyperparameters.
        """
        if datapoints is None or comparisons is None:
            raise UnsupportedError(
                "VariationalTopChoiceGP requires `datapoints` and `comparisons`; "
                "models without data are not supported."
            )
        if datapoints.dim() != 2:
            raise UnsupportedError(
                "`datapoints` must be an `n x d` tensor; batched datapoints are not "
                "supported."
            )
        _validate_comparisons(comparisons, num_datapoints=datapoints.shape[-2])
        if datapoints.dtype != torch.float64:
            warnings.warn(
                _get_single_precision_warning(str(datapoints.dtype)),
                InputDataWarning,
                stacklevel=2,
            )
        k = comparisons.shape[-1]
        if likelihood is None:
            likelihood = TopChoiceSoftmaxLikelihood(num_alternatives=k)
        elif not isinstance(likelihood, TopChoiceSoftmaxLikelihood):
            raise UnsupportedError(
                "VariationalTopChoiceGP requires a `TopChoiceSoftmaxLikelihood`, "
                "which is the logistic pairwise comparison likelihood for `k = 2`."
            )
        elif likelihood.num_alternatives != k:
            raise ValueError(
                f"`likelihood.num_alternatives` ({likelihood.num_alternatives}) must "
                f"equal the number of datapoints per comparison ({k})."
            )
        if covar_module is None:
            covar_module = ScaleKernel(
                get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=datapoints.shape[-1]
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        elif not isinstance(covar_module, ScaleKernel):
            raise UnsupportedError(
                "VariationalTopChoiceGP requires a `ScaleKernel`, whose outputscale "
                "controls the effective softmax temperature."
            )

        # The variational strategy depends on the data and is set in
        # ``_set_train_data``.
        ApproximateGP.__init__(self, variational_strategy=None)
        if input_transform is not None:
            with torch.no_grad():
                # Sets the coefficients of the transform that depend on the data.
                self.transform_inputs(X=datapoints, input_transform=input_transform)
            self.input_transform = input_transform
        self.likelihood = likelihood
        # As for ``PairwiseGP``, the prior mean is zero: the likelihood is invariant
        # to shifting all utilities, so a constant mean would not be identified.
        self.mean_module = ZeroMean()
        self.covar_module = covar_module.to(datapoints)
        self._set_train_data(datapoints=datapoints, comparisons=comparisons.long())
        self.to(datapoints)

    @property
    def datapoints(self) -> Tensor:
        r"""The ``n x d`` tensor of (untransformed) datapoints."""
        return self._datapoints

    @property
    def comparisons(self) -> Tensor:
        r"""The ``m x k`` tensor of comparisons."""
        return self._comparisons

    @property
    def dim(self) -> int:
        r"""The dimension ``d`` of the datapoints."""
        return self.datapoints.shape[-1]

    @property
    def train_inputs(self) -> tuple[Tensor]:
        r"""The ``m * k`` transformed datapoints of the comparisons.

        They are flattened in comparison-major order, i.e., the datapoints of the
        ``i``-th comparison are at positions ``i * k, ..., i * k + k - 1``.
        """
        return (
            self._transform_datapoints(self.datapoints)[self.comparisons].flatten(0, 1),
        )

    @property
    def _original_train_inputs(self) -> Tensor:
        r"""The untransformed counterpart of ``train_inputs``."""
        return self.datapoints[self.comparisons].flatten(0, 1)

    @property
    def train_targets(self) -> Tensor:
        r"""The positions of the chosen datapoints in the ``m`` comparisons (zeros)."""
        return torch.zeros_like(self.comparisons[:, 0])

    @property
    def utility(self) -> Tensor:
        r"""The posterior mean of the utility at the ``n`` datapoints."""
        with torch.no_grad():
            return self(self._transform_datapoints(self.datapoints)).mean

    @classmethod
    def construct_inputs(cls, training_data: SupervisedDataset) -> dict[str, Tensor]:
        r"""Construct ``Model`` keyword arguments from a ``RankingDataset``.

        The top-ranked (rank 0) item of each ranking is used as the chosen datapoint
        of the comparison, and the ranks of the other items are ignored. For pairwise
        rankings, this is equivalent to ``PairwiseGP.construct_inputs``.

        Args:
            training_data: A ``RankingDataset`` whose rankings each have a unique
                top-ranked item.

        Returns:
            A dict of keyword arguments that can be used to initialize a
            ``VariationalTopChoiceGP``.
        """
        if not isinstance(training_data, RankingDataset):
            raise UnsupportedError(
                "Only `RankingDataset` is supported for `VariationalTopChoiceGP`. "
                f"Received {type(training_data)}."
            )
        ranks = training_data.Y
        if ((ranks == 0).sum(dim=-1) != 1).any():
            raise ValueError("Each ranking must have a unique top-ranked item.")
        order = ranks.argsort(dim=-1, stable=True)
        return {
            "datapoints": training_data._X.values,
            "comparisons": training_data._X.indices.gather(dim=-1, index=order),
        }

    def __call__(
        self, inputs: Tensor | None, prior: bool = False, **kwargs: Any
    ) -> MultivariateNormal:
        r"""Compute the variational posterior (or the prior) at the inputs.

        Unlike ``ApproximateGP.__call__``, this computes the full covariance also in
        train mode: GPyTorch otherwise computes only the variances when there are
        fewer inducing points than inputs, but the softmax likelihood depends on the
        covariances between the utilities of each comparison.
        """
        kwargs.setdefault("diag", False)
        return super().__call__(inputs, prior=prior, **kwargs)

    def forward(self, X: Tensor) -> MultivariateNormal:
        r"""Compute the GP prior at the (transformed) inputs ``X``.

        Args:
            X: A ``batch_shape x N x d`` tensor of inputs.
        """
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> Posterior:
        r"""Compute the posterior over the latent utility.

        As for ``PairwiseGP``, gradients of the posterior are only supported with
        respect to ``X``.

        Args:
            X: A ``batch_shape x q x d`` tensor of inputs.
            output_indices: Not supported for this single-output model; must be None.
            observation_noise: Ignored, since the noise of the choices is modeled by
                the softmax likelihood rather than as additive noise on the utility.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``GPyTorchPosterior`` over the utility at ``X``, or the result of
            applying ``posterior_transform`` to it.
        """
        if output_indices is not None:
            raise UnsupportedError(
                "`output_indices` is not supported by the single-output "
                "VariationalTopChoiceGP."
            )
        # Freezing the parameters keeps their autograd history out of the eval-mode
        # caches of the variational strategy, as ``_InputDetachedVariationalStrategy``
        # does for the history of ``X``. Otherwise, gradients with respect to ``X``
        # could not be computed more than once, and the model could not be
        # deep-copied.
        with _frozen(self.parameters()):
            return super().posterior(X=X, posterior_transform=posterior_transform)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> VariationalTopChoiceGP:
        r"""Condition the model on new comparisons.

        As for ``PairwiseGP``, ``Y`` must contain comparisons rather than utility
        values, and the posterior is updated with the hyperparameters fixed: the
        returned model has the hyperparameters of this model, and its variational
        distribution is fit to the augmented data.

        Args:
            X: An ``n' x d`` tensor of new (untransformed) datapoints.
            Y: An ``m' x k`` tensor of new comparisons, i.e. of indices into ``X``.
                The first element of each row is the index of the chosen datapoint.
            kwargs: Not used.

        Returns:
            A new model conditioned on the augmented datapoints and comparisons.
        """
        if (
            X.dim() != 2
            or Y.dim() != 2
            or Y.is_floating_point()
            or Y.shape[-1] != self.likelihood.num_alternatives
        ):
            raise UnsupportedError(
                "VariationalTopChoiceGP can only be conditioned on an `n' x d` tensor "
                "of datapoints `X` and an `m' x k` integer tensor of comparisons `Y` "
                "with the same `k` as the training comparisons."
            )
        _validate_comparisons(Y, num_datapoints=X.shape[-2])
        # The caches of the variational strategy may contain non-leaf tensors, which
        # cannot be deep-copied.
        clear_cache_hook(self.variational_strategy)
        new_model = deepcopy(self)
        new_model._set_train_data(
            datapoints=torch.cat([self.datapoints, X.to(self.datapoints)], dim=-2),
            comparisons=torch.cat(
                [self.comparisons, Y.to(self.comparisons) + self.datapoints.shape[-2]],
                dim=-2,
            ),
        )
        new_model._fit_variational_distribution()
        return new_model

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = False,
        keep_transforms: bool = True,
        assign: bool = False,
    ) -> None:
        r"""Load a state dict.

        If the state dict is from a model with the same datapoints, comparisons, and
        inducing points, all parameters are loaded. Otherwise, as for ``PairwiseGP``,
        only the hyperparameters are loaded and the variational distribution is fit
        to the current data with these hyperparameters fixed. This supports, e.g.,
        cross-validation with the hyperparameters of a model fit to all data. Since
        raw parameter values are loaded, the hyperparameters of a ``PairwiseGP`` are
        only loaded correctly if ``covar_module`` has the same constraints as its
        covariance module, e.g., if it is a copy of it.

        Args:
            state_dict: The state dict.
            strict: Whether the keys of ``state_dict`` must match the keys of this
                model's state dict. Only supported for a state dict with the same
                datapoints, comparisons, and inducing points.
            keep_transforms: Passed to ``GPyTorchModel.load_state_dict``.
            assign: Passed to ``GPyTorchModel.load_state_dict``.
        """
        current = self.state_dict()
        if all(
            name in state_dict
            and torch.equal(state_dict[name].to(current[name]), current[name])
            for name in (
                "_datapoints",
                "_comparisons",
                "variational_strategy.inducing_points",
            )
        ):
            super().load_state_dict(
                state_dict,
                strict=strict,
                keep_transforms=keep_transforms,
                assign=assign,
            )
            return
        if strict:
            raise UnsupportedError(
                "`strict=True` is only supported for a state dict with the same "
                "datapoints, comparisons, and inducing points."
            )
        # The load pre-hook of GPyTorch's ``VariationalStrategy`` requires its state
        # dict to be non-empty, so the hyperparameters are loaded on top of the
        # current state, whose data-dependent part is re-initialized below.
        current.update(
            (name, value)
            for name, value in state_dict.items()
            if not name.startswith(
                ("variational_strategy.", "_datapoints", "_comparisons")
            )
        )
        super().load_state_dict(
            current,
            strict=False,
            keep_transforms=keep_transforms,
            assign=assign,
        )
        # Re-initialize the inducing points with the loaded input transform.
        self._set_train_data(datapoints=self.datapoints, comparisons=self.comparisons)
        self._fit_variational_distribution()

    def custom_fit(
        self, mll: MarginalLogLikelihood, **kwargs: Any
    ) -> MarginalLogLikelihood:
        r"""Fit the model. This is called by ``fit_gpytorch_mll``.

        This runs the generic fitting routine of ``fit_gpytorch_mll``, in which each
        attempt first fits the variational distribution with the hyperparameters
        fixed, since a joint fit that starts from the prior can end at an
        over-smoothed local optimum. This initialization uses the default ELBO and
        optimizer settings, and it is skipped if the variational parameters are
        frozen. Unlike for other approximate GPs, the routine uses L-BFGS-B
        regardless of the number of comparisons (unless another ``optimizer`` is
        passed), since the ELBO of this model is a deterministic full-batch objective.

        Args:
            mll: A variational objective, e.g., a ``TopChoiceVariationalELBO``.
            kwargs: Keyword arguments of the generic fitting routine of
                ``fit_gpytorch_mll``, e.g., ``optimizer`` or ``max_attempts``.

        Returns:
            The fitted ``mll``.
        """
        if not isinstance(mll, _ApproximateMarginalLogLikelihood):
            raise UnsupportedError(
                "VariationalTopChoiceGP must be fit with a variational objective such "
                f"as `TopChoiceVariationalELBO`, not {type(mll).__name__}."
            )
        optimizer = kwargs.pop("optimizer", None) or fit_gpytorch_mll_scipy

        def warm_started_optimizer(
            mll: MarginalLogLikelihood, **optimizer_kwargs: Any
        ) -> OptimizationResult:
            if any(p.requires_grad for p in self.variational_strategy.parameters()):
                with warnings.catch_warnings():
                    # The warm start is only an initialization of the joint fit.
                    warnings.simplefilter("ignore", category=OptimizationWarning)
                    self._fit_variational_distribution()
            return optimizer(mll.train(), **optimizer_kwargs)

        return _fit_fallback(mll=mll, optimizer=warm_started_optimizer, **kwargs)

    def _transform_datapoints(self, datapoints: Tensor) -> Tensor:
        r"""Transform datapoints as training inputs, without updating the transform."""
        input_transform = getattr(self, "input_transform", None)
        if input_transform is None:
            return datapoints
        # In train mode, ``preprocess_transform`` only prevents the outermost
        # transform from updating its coefficients, but not nested transforms such
        # as the ``Normalize`` of ``Warp``.
        modes = {module: module.training for module in input_transform.modules()}
        input_transform.eval()
        try:
            return input_transform.preprocess_transform(datapoints)
        finally:
            for module, training in modes.items():
                module.training = training

    def _set_train_data(self, datapoints: Tensor, comparisons: Tensor) -> None:
        r"""Set the data, and initialize the inducing points and the variational
        distribution."""
        self.register_buffer("_datapoints", datapoints.detach())
        self.register_buffer("_comparisons", comparisons)
        with torch.no_grad():
            inducing_points = torch.unique(
                self._transform_datapoints(datapoints)[comparisons.flatten()], dim=0
            )
        self.variational_strategy = _InputDetachedVariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2]
            ),
            learn_inducing_locations=False,
        ).to(datapoints)

    def _fit_variational_distribution(self) -> None:
        r"""Fit the variational distribution with the hyperparameters fixed.

        This is the counterpart of the Laplace approximation update of ``PairwiseGP``.
        Unlike ``fit_gpytorch_mll``, this keeps the result of an unsuccessful
        optimization, which is typically close to optimal, instead of retrying the
        deterministic optimization.
        """
        variational_parameters = set(self.variational_strategy.parameters())
        mll = TopChoiceVariationalELBO(self.likelihood, self).train()
        with _frozen(p for p in self.parameters() if p not in variational_parameters):
            fit_gpytorch_mll_scipy(mll)
        self.eval()

    def _revert_to_original_inputs(self) -> None:
        r"""Do nothing, since ``train_inputs`` are always transformed.

        The base class implementation calls ``set_train_data``, which only exists for
        exact GPs.
        """


class TopChoiceVariationalELBO(VariationalELBO):
    r"""The variational ELBO for fitting a ``VariationalTopChoiceGP``.

    GPyTorch divides the expected log-likelihood term of the ELBO by the number of
    training inputs, ``m * k``, and the KL divergence term by ``num_data``. This is a
    ``VariationalELBO`` with ``num_data = m * k``, so that the objective is
    proportional to the ELBO. Like ``PairwiseLaplaceMarginalLogLikelihood`` for
    ``PairwiseGP``, it can be constructed from the likelihood and the model alone,
    e.g. by Ax.
    """

    def __init__(
        self,
        likelihood: TopChoiceSoftmaxLikelihood,
        model: VariationalTopChoiceGP,
        beta: float = 1.0,
        combine_terms: bool = True,
    ) -> None:
        r"""Initialize the ELBO.

        Args:
            likelihood: The likelihood of ``model``.
            model: The ``VariationalTopChoiceGP``.
            beta: The weight of the KL divergence term, see ``VariationalELBO``.
            combine_terms: Whether to sum the terms of the ELBO, see
                ``VariationalELBO``.
        """
        super().__init__(
            likelihood=likelihood,
            model=model,
            num_data=model.comparisons.numel(),
            beta=beta,
            combine_terms=combine_terms,
        )


class _InputDetachedVariationalStrategy(VariationalStrategy):
    r"""A ``VariationalStrategy`` whose eval-mode cache does not keep the autograd
    history of the inputs.

    GPyTorch takes the covariance of the inducing points from the joint covariance of
    the inducing points and the inputs, so in eval mode, its cached Cholesky factor
    would otherwise keep the autograd history of the first inputs that require
    gradients. Only the inputs of the lazily evaluated kernel matrix are detached,
    not the kernel hyperparameters.
    """

    def _cholesky_factor(
        self, induc_induc_covar: LinearOperator
    ) -> TriangularLinearOperator:
        if not self.training:
            induc_induc_covar = induc_induc_covar.detach()
        return super()._cholesky_factor(induc_induc_covar)


@contextmanager
def _frozen(parameters: Iterable[Parameter]) -> Iterator[None]:
    r"""Temporarily set ``requires_grad=False`` for the parameters that require it."""
    trainable = [p for p in parameters if p.requires_grad]
    for p in trainable:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in trainable:
            p.requires_grad_(True)


def _validate_comparisons(comparisons: Tensor, num_datapoints: int) -> None:
    r"""Check that ``comparisons`` is an ``m x k`` tensor of datapoint indices."""
    if comparisons.dim() != 2 or len(comparisons) == 0 or comparisons.shape[-1] < 2:
        raise ValueError(
            "`comparisons` must be an `m x k` tensor with `m >= 1` and `k >= 2`."
        )
    if comparisons.min() < 0 or comparisons.max() >= num_datapoints:
        raise ValueError(
            "`comparisons` must contain indices in `[0, n)`, where `n` is the "
            "number of datapoints."
        )
