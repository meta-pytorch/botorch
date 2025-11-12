#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Abstract model class for all GPyTorch-based botorch models.

To implement your own, simply inherit from both the provided classes and a
GPyTorch Model class such as an ExactGP.
"""

from __future__ import annotations

import itertools
import warnings
from abc import ABC
from copy import deepcopy
from typing import Any, Mapping, TYPE_CHECKING

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch.exceptions.warnings import (
    _get_single_precision_warning,
    BotorchTensorDimensionWarning,
    BotorchWarning,
    InputDataWarning,
)
from botorch.models.model import Model, ModelList
from botorch.models.utils import (
    _make_X_full,
    add_output_dim,
    extract_targets_and_noise_single_output,
    gpt_posterior_settings,
    mod_batch_shape,
    multioutput_to_batch_mode_transform,
    restore_targets_and_noise_single_output,
)
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.multitask import separate_mtmvn
from botorch.utils.transforms import is_ensemble
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from linear_operator.operators import BlockDiagLinearOperator, CatLinearOperator
from torch import broadcast_shapes, Tensor

if TYPE_CHECKING:
    from botorch.posteriors.posterior_list import PosteriorList  # pragma: no cover
    from botorch.posteriors.transformed import TransformedPosterior  # pragma: no cover
    from gpytorch.likelihoods import Likelihood  # pragma: no cover


class GPyTorchModel(Model, ABC):
    r"""Abstract base class for models based on GPyTorch models.

    The easiest way to use this is to subclass a model from a GPyTorch model
    class (e.g. an `ExactGP`) and this `GPyTorchModel`. See e.g. `SingleTaskGP`.
    """

    likelihood: Likelihood

    @staticmethod
    def _validate_tensor_args(
        X: Tensor, Y: Tensor, Yvar: Tensor | None = None, strict: bool = True
    ) -> None:
        r"""Checks that `Y` and `Yvar` have an explicit output dimension if strict.
        Checks that the dtypes of the inputs match, and warns if using float.

        This also checks that `Yvar` has the same trailing dimensions as `Y`. Note
        we only infer that an explicit output dimension exists when `X` and `Y` have
        the same `batch_shape`.

        Args:
            X: A `batch_shape x n x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n` is the number of points per batch, and
                `batch_shape` is the batch shape (potentially empty).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
            Yvar: A `batch_shape' x n x m` tensor of observed measurement noise.
                Note: this will be None when using a model that infers the noise
                level (e.g. a `SingleTaskGP`).
            strict: A boolean indicating whether to check that `Y` and `Yvar`
                have an explicit output dimension.
        """
        if X.dim() != Y.dim():
            if (X.dim() - Y.dim() == 1) and (X.shape[:-1] == Y.shape):
                message = (
                    "An explicit output dimension is required for targets."
                    f" Expected Y with dimension {X.dim()} (got {Y.dim()=})."
                )
            else:
                message = (
                    "Expected X and Y to have the same number of dimensions"
                    f" (got X with dimension {X.dim()} and Y with dimension"
                    f" {Y.dim()})."
                )
            if strict:
                raise BotorchTensorDimensionError(message)
            else:
                warnings.warn(
                    "Non-strict enforcement of botorch tensor conventions. The "
                    "following error would have been raised with strict enforcement: "
                    f"{message}",
                    BotorchTensorDimensionWarning,
                    stacklevel=2,
                )
        # Yvar may not have the same batch dimensions, but the trailing dimensions
        # of Yvar should be the same as the trailing dimensions of Y.
        if Yvar is not None and Y.shape[-(Yvar.dim()) :] != Yvar.shape:
            raise BotorchTensorDimensionError(
                "An explicit output dimension is required for observation noise."
                f" Expected Yvar with shape: {Y.shape[-Yvar.dim() :]} (got"
                f" {Yvar.shape})."
            )
        # Check the dtypes.
        if X.dtype != Y.dtype or (Yvar is not None and Y.dtype != Yvar.dtype):
            raise InputDataError(
                "Expected all inputs to share the same dtype. Got "
                f"{X.dtype} for X, {Y.dtype} for Y, and "
                f"{Yvar.dtype if Yvar is not None else None} for Yvar."
            )
        if X.dtype != torch.float64:
            warnings.warn(
                _get_single_precision_warning(str(X.dtype)),
                InputDataWarning,
                stacklevel=3,  # Warn at model constructor call.
            )

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return self.train_inputs[0].shape[:-2]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    # pyre-fixme[14]: Inconsistent override.
    # `botorch.models.gpytorch.GPyTorchModel.posterior` overrides method defined
    # in `Model` inconsistently. Could not find parameter `output_indices` in
    # overriding signature.
    def posterior(
        self,
        X: Tensor,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`). It is
                assumed to be in the outcome-transformed space if an outcome
                transform is used.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            # NOTE: BoTorch's GPyTorchModels also inherit from GPyTorch's ExactGP, thus
            # self(X) calls GPyTorch's ExactGP's __call__, which computes the posterior,
            # rather than e.g. SingleTaskGP's forward, which computes the prior.
            mvn = self(X)
            if observation_noise is not False:
                if isinstance(observation_noise, torch.Tensor):
                    # TODO: Make sure observation noise is transformed correctly
                    self._validate_tensor_args(X=X, Y=observation_noise)
                    if observation_noise.size(-1) == 1:
                        observation_noise = observation_noise.squeeze(-1)
                    mvn = self.likelihood(mvn, X, noise=observation_noise)
                else:
                    mvn = self.likelihood(mvn, X)
        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.
            noise: If not `None`, a tensor of the same shape as `Y` representing
                the associated noise variance.
            kwargs: Passed to `self.get_fantasy_model`.

        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X[:, :1]) + torch.cos(train_X[:, 1:])
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> model.eval()
            >>> test_X = torch.rand(10, 2)
            # Need to evaluate once to fill test independent caches
            # so that condition_on_observations works.
            >>> model(test_X)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.sin(new_X[:, :1]) + torch.cos(new_X[:, 1:])
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        # pass the transformed data to get_fantasy_model below
        # (unless we've already transformed if BatchedMultiOutputGPyTorchModel)
        X_original = X.clone()
        X = self.transform_inputs(X)

        Yvar = noise
        if hasattr(self, "outcome_transform"):
            # And do the same for the outcome transform, if it exists.
            if not isinstance(self, BatchedMultiOutputGPyTorchModel):
                # `noise` is assumed to already be outcome-transformed.
                Y, _ = self.outcome_transform(Y=Y, Yvar=Yvar, X=X)
        # Validate using strict=False, since we cannot tell if Y has an explicit
        # output dimension. Do not check shapes when fantasizing as they are
        # not expected to match.
        if fantasize_flag.off():
            self._validate_tensor_args(X=X, Y=Y, Yvar=Yvar, strict=False)
        if Y.size(-1) == 1:
            Y = Y.squeeze(-1)
            if Yvar is not None:
                kwargs.update({"noise": Yvar.squeeze(-1)})
        # get_fantasy_model will properly copy any existing outcome transforms
        # (since it deepcopies the original model))
        fantasy_model = self.get_fantasy_model(inputs=X, targets=Y, **kwargs)

        # If we use an input transform, the fantasized data will not get added to
        # the training data by default. We need to manually add it.
        if hasattr(fantasy_model, "input_transform"):
            # Broadcast tensors to compatible shape before concatenating
            expand_shape = torch.broadcast_shapes(
                X_original.shape[:-2], fantasy_model._original_train_inputs.shape[:-2]
            )
            X_expanded = X_original.expand(expand_shape + X_original.shape[-2:])
            orig_expanded = fantasy_model._original_train_inputs.expand(
                expand_shape + fantasy_model._original_train_inputs.shape[-2:]
            )
            fantasy_model._original_train_inputs = torch.cat(
                [orig_expanded, X_expanded],
                dim=-2,
            ).detach()
        return fantasy_model

    def _extract_targets_and_noise(self) -> tuple[Tensor, Tensor | None]:
        r"""Extract targets and noise variance in the correct shape.

        Returns a tuple of (Y, Yvar) where Y and Yvar have shape
        [batch_shape] x n x m, with batch_shape included only if the
        training data initially contained it.
        """
        if self.num_outputs > 1:
            Y = self.train_targets.transpose(-1, -2)
            Yvar = None
            if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                Yvar = self.likelihood.noise_covar.noise.transpose(-1, -2)
        else:
            Y, Yvar = extract_targets_and_noise_single_output(self)
        return Y, Yvar

    def _restore_targets_and_noise(
        self, Y: Tensor, Yvar: Tensor | None, strict: bool
    ) -> None:
        r"""Restore targets and noise variance to the model.

        Args:
            Y: Targets tensor in shape [batch_shape] x n x m.
            Yvar: Optional noise variance tensor in shape [batch_shape] x n x m.
            strict: Whether to strictly enforce shape constraints.
        """
        if self.num_outputs > 1:
            Y = Y.transpose(-1, -2)
            if Yvar is not None and isinstance(
                self.likelihood, FixedNoiseGaussianLikelihood
            ):
                Yvar = Yvar.transpose(-1, -2)
                self.likelihood.noise_covar.noise = Yvar
            self.set_train_data(targets=Y, strict=strict)
        else:
            restore_targets_and_noise_single_output(self, Y, Yvar, strict)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        keep_transforms: bool = True,
        assign: bool = False,
    ) -> None:
        r"""Load the model state.

        Args:
            state_dict: A dict containing the state of the model.
            strict: A boolean indicating whether to strictly enforce that the keys.
            keep_transforms: A boolean indicating whether to keep the input and outcome
                transforms. Doing so is useful when loading a model that was trained on
                a full set of data, and is later loaded with a subset of the data.
            assign: When set to ``False``, the properties of the tensors in the current
                module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
                for which the value from the module is preserved. Default: ``False``.
        """
        if assign:
            first_item = next(iter(state_dict.values()))
            self.to(first_item)
        if not keep_transforms:
            super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
            return

        should_outcome_transform = (
            hasattr(self, "train_targets")
            and getattr(self, "outcome_transform", None) is not None
        )

        with torch.no_grad():
            untransformed_Y, untransformed_Yvar = self._extract_targets_and_noise()
            X = self.train_inputs[0]

            if should_outcome_transform:
                try:
                    untransformed_Y, untransformed_Yvar = (
                        self.outcome_transform.untransform(
                            Y=untransformed_Y,
                            Yvar=untransformed_Yvar,
                            X=X,
                        )
                    )
                except NotImplementedError:
                    warnings.warn(
                        "Outcome transform does not support untransforming."
                        "Cannot load the state dict with transforms preserved."
                        "Setting keep_transforms=False.",
                        BotorchWarning,
                        stacklevel=3,
                    )
                    super().load_state_dict(
                        state_dict=state_dict, strict=strict, assign=assign
                    )
                    return

        super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

        if getattr(self, "input_transform", None) is not None:
            self.input_transform.eval()

        if should_outcome_transform:
            self.outcome_transform.eval()
            retransformed_Y, retransformed_Yvar = self.outcome_transform(
                Y=untransformed_Y, Yvar=untransformed_Yvar, X=X
            )
            self._restore_targets_and_noise(retransformed_Y, retransformed_Yvar, strict)


# pyre-fixme[13]: uninitialized attributes _num_outputs, _input_batch_shape,
# _aug_batch_shape
class BatchedMultiOutputGPyTorchModel(GPyTorchModel):
    r"""Base class for batched multi-output GPyTorch models with independent outputs.

    This model should be used when the same training data is used for all outputs.
    Outputs are modeled independently by using a different batch for each output.
    """

    _num_outputs: int
    _input_batch_shape: torch.Size
    _aug_batch_shape: torch.Size

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> tuple[torch.Size, torch.Size]:
        r"""Get the raw batch shape and output-augmented batch shape of the inputs.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.

        Returns:
            2-element tuple containing

            - The `input_batch_shape`
            - The output-augmented batch shape: `input_batch_shape x (m)`
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape

    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor) -> None:
        r"""Store the number of outputs and the batch shape.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
        """
        self._num_outputs = train_Y.shape[-1]
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return self._input_batch_shape

    def _transform_tensor_args(
        self, X: Tensor, Y: Tensor, Yvar: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        r"""Transforms tensor arguments: for single output models, the output
        dimension is squeezed and for multi-output models, the output dimension is
        transformed into the left-most batch dimension.

        Args:
            X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
            Yvar: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                observed measurement noise. Note: this will be None when using a model
                that infers the noise level (e.g. a `SingleTaskGP`).

        Returns:
            3-element tuple containing

            - A `input_batch_shape x (m) x n x d` tensor of training features.
            - A `target_batch_shape x (m) x n` tensor of training observations.
            - A `target_batch_shape x (m) x n` tensor observed measurement noise
                (or None).
        """
        if self._num_outputs > 1:
            return multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, train_Yvar=Yvar, num_outputs=self._num_outputs
            )
        return X, Y.squeeze(-1), None if Yvar is None else Yvar.squeeze(-1)

    def _apply_noise(
        self,
        X: Tensor,
        mvn: MultivariateNormal,
        observation_noise: bool | Tensor = False,
    ) -> MultivariateNormal:
        """Adds the observation noise to the posterior.

        Args:
            X: A tensor of shape `batch_shape x q x d`.
            mvn: A `MultivariateNormal` object representing the posterior over the true
                latent function.
            num_outputs: The number of outputs of the model.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).

        Returns:
            The posterior predictive.
        """
        if observation_noise is False:
            return mvn
        # noise_shape is `broadcast(test_batch_shape, model.batch_shape) x m x q`
        noise_shape = mvn.batch_shape + mvn.event_shape
        if torch.is_tensor(observation_noise):
            # TODO: Validate noise shape
            # make observation_noise's shape match noise_shape
            if self.num_outputs > 1:
                obs_noise = observation_noise.transpose(-1, -2)
            else:
                obs_noise = observation_noise.squeeze(-1)
            mvn = self.likelihood(
                mvn,
                X,
                noise=obs_noise.expand(noise_shape),
            )
        elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
            # Use the mean of the previous noise values (TODO: be smarter here).
            observation_noise = self.likelihood.noise.mean(dim=-1, keepdim=True)
            mvn = self.likelihood(
                mvn,
                X,
                noise=observation_noise.expand(noise_shape),
            )
        else:
            mvn = self.likelihood(mvn, X)
        return mvn

    # pyre-ignore[14]: Inconsistent override. Could not find parameter
    # `Keywords(typing.Any)` in overriding signature.
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if specified.
        """
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            # insert a dimension for the output dimension
            if self._num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            # NOTE: BoTorch's GPyTorchModels also inherit from GPyTorch's ExactGP, thus
            # self(X) calls GPyTorch's ExactGP's __call__, which computes the posterior,
            # rather than e.g. SingleTaskGP's forward, which computes the prior.
            mvn = self(X)
            mvn = self._apply_noise(X=X, mvn=mvn, observation_noise=observation_noise)
            if self._num_outputs > 1:
                if torch.jit.is_tracing():
                    mvn = MultitaskMultivariateNormal.from_batch_mvn(
                        mvn, task_dim=output_dim_idx
                    )
                else:
                    mean_x = mvn.mean
                    covar_x = mvn.lazy_covariance_matrix
                    output_indices = output_indices or range(self._num_outputs)
                    mvns = [
                        MultivariateNormal(
                            mean_x.select(dim=output_dim_idx, index=t),
                            covar_x[(slice(None),) * output_dim_idx + (t,)],
                        )
                        for t in output_indices
                    ]
                    mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    # pyre-ignore[14]: Inconsistent override. Could not find parameter `noise`.
    def condition_on_observations(
        self, X: Tensor, Y: Tensor, **kwargs: Any
    ) -> BatchedMultiOutputGPyTorchModel:
        r"""Condition the model on new observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `m` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, its is assumed that the missing batch dimensions are
                the same for all `Y`.

        Returns:
            A `BatchedMultiOutputGPyTorchModel` object of the same type with
            `n + n'` training examples, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.cat(
            >>>     [torch.sin(train_X[:, 0]), torch.cos(train_X[:, 1])], -1
            >>> )
            >>> model = SingleTaskGP(train_X, train_Y)
            >>> new_X = torch.rand(5, 2)
            >>> new_Y = torch.cat([torch.sin(new_X[:, 0]), torch.cos(new_X[:, 1])], -1)
            >>> model = model.condition_on_observations(X=new_X, Y=new_Y)
        """
        noise = kwargs.get("noise")
        if hasattr(self, "outcome_transform"):
            # We need to apply transforms before shifting batch indices around.
            # `noise` is assumed to already be outcome-transformed.
            Y, _ = self.outcome_transform(Y, X=X)
        # Do not check shapes when fantasizing as they are not expected to match.
        if fantasize_flag.off():
            self._validate_tensor_args(X=X, Y=Y, Yvar=noise, strict=False)
        inputs = X
        if self._num_outputs > 1:
            inputs, targets, noise = multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, num_outputs=self._num_outputs, train_Yvar=noise
            )
            # `multioutput_to_batch_mode_transform` removes the output dimension,
            # which is necessary for `condition_on_observations`
            targets = targets.unsqueeze(-1)
            if noise is not None:
                noise = noise.unsqueeze(-1)
        else:
            inputs = X
            targets = Y
        if noise is not None:
            kwargs.update({"noise": noise})
        fantasy_model = super().condition_on_observations(X=inputs, Y=targets, **kwargs)
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
            : (-1 if self._num_outputs == 1 else -2)
        ]
        if not self._is_fully_bayesian:
            fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        return fantasy_model

    def subset_output(self, idcs: list[int]) -> BatchedMultiOutputGPyTorchModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        try:
            subset_batch_dict = self._subset_batch_dict
        except AttributeError:
            raise NotImplementedError(
                "`subset_output` requires the model to define a `_subset_batch_dict` "
                "attribute that lists the indices of the output dimensions in each "
                "model parameter that needs to be subset."
            )

        m = len(idcs)
        new_model = deepcopy(self)

        subset_everything = self.num_outputs == m and idcs == list(range(m))
        if subset_everything:
            return new_model

        tidxr = torch.tensor(idcs, device=new_model.train_targets.device)
        idxr = tidxr if m > 1 else idcs[0]
        new_tail_bs = torch.Size([m]) if m > 1 else torch.Size()

        new_model._num_outputs = m
        new_model._aug_batch_shape = new_model._aug_batch_shape[:-1] + new_tail_bs
        new_model.train_inputs = tuple(
            ti[..., idxr, :, :] for ti in new_model.train_inputs
        )
        new_model.train_targets = new_model.train_targets[..., idxr, :]

        # adjust batch shapes of parameters/buffers if necessary
        for full_name, p in itertools.chain(
            new_model.named_parameters(), new_model.named_buffers()
        ):
            if full_name in subset_batch_dict:
                idx = subset_batch_dict[full_name]
                new_data = p.index_select(dim=idx, index=tidxr)
                if m == 1:
                    new_data = new_data.squeeze(idx)
                p.data = new_data
            mod_name = full_name.split(".")[:-1]
            mod_batch_shape(new_model, mod_name, m if m > 1 else 0)

        # subset outcome transform if present
        try:
            subset_octf = new_model.outcome_transform.subset_output(idcs=idcs)
            new_model.outcome_transform = subset_octf
        except AttributeError:
            pass

        # Subset fixed noise likelihood if present.
        if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
            full_noise = new_model.likelihood.noise_covar.noise
            new_noise = full_noise[..., idcs if len(idcs) > 1 else idcs[0], :]
            new_model.likelihood.noise_covar.noise = new_noise

        return new_model


class ModelListGPyTorchModel(ModelList, GPyTorchModel, ABC):
    r"""Abstract base class for models based on multi-output GPyTorch models.

    This is meant to be used with a gpytorch ModelList wrapper for independent
    evaluation of submodels. Those submodels can themselves be multi-output
    models, in which case the task covariances will be ignored.
    """

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        batch_shapes = {m.batch_shape for m in self.models}
        if len(batch_shapes) > 1:
            msg = (
                f"Component models of {self.__class__.__name__} have different "
                "batch shapes"
            )
            try:
                broadcast_shape = torch.broadcast_shapes(*batch_shapes)
                warnings.warn(msg + ". Broadcasting batch shapes.", stacklevel=2)
                return broadcast_shape
            except RuntimeError:
                raise NotImplementedError(msg + " that are not broadcastble.")
        return next(iter(batch_shapes))

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        return ModelList.load_state_dict(
            self, state_dict=state_dict, strict=strict, assign=assign
        )

    # pyre-fixme[14]: Inconsistent override in return types
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior | PosteriorList:
        r"""Computes the posterior over model outputs at the provided points.
        If any model returns a MultitaskMultivariateNormal posterior, then that
        will be split into individual MVNs per task, with inter-task covariance
        ignored.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                respective likelihoods to the posterior. If a Tensor of shape
                `(batch_shape) x q x m`, use it directly as the observation
                noise (with `observation_noise[...,i]` added to the posterior
                of the `i`-th model).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            - If no `posterior_transform` is provided and the component models have no
                `outcome_transform`, or if the component models only use linear outcome
                transforms like `Standardize` (i.e. not `Log`), returns a
                `GPyTorchPosterior` or `GaussianMixturePosterior` object,
                representing `batch_shape` joint distributions over `q` points
                and the outputs selected by `output_indices` each. Includes
                measurement noise if `observation_noise` is specified.
            - If no `posterior_transform` is provided and component models have
                nonlinear transforms like `Log`, returns a `PosteriorList` with
                sub-posteriors of type `TransformedPosterior`
            - If `posterior_transform` is provided, that posterior transform will be
               applied and will determine the return type. This could potentially be
               any subclass of `Posterior`, but common choices give a
               `GPyTorchPosterior`.
        """

        # Nonlinear transforms untransform to a `TransformedPosterior`,
        # which can't be made into a `GPyTorchPosterior`
        returns_untransformed = any(
            hasattr(mod, "outcome_transform") and (not mod.outcome_transform._is_linear)
            for mod in self.models
        )
        # NOTE: We're not passing in the posterior transform here. We'll apply it later.
        posterior = ModelList.posterior(
            self,
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
        )
        if not returns_untransformed:
            mvns = [p.distribution for p in posterior.posteriors]
            if any(isinstance(m, MultitaskMultivariateNormal) for m in mvns):
                mvn_list = []
                for mvn in mvns:
                    if len(mvn.event_shape) == 2:
                        # We separate MTMVNs into independent-across-task MVNs for
                        # the convenience of using BlockDiagLinearOperator below.
                        # (b x q x m x m) -> list of m (b x q x 1 x 1)
                        mvn_list.extend(separate_mtmvn(mvn))
                    else:
                        mvn_list.append(mvn)
                mean = torch.stack([mvn.mean for mvn in mvn_list], dim=-1)
                covars = CatLinearOperator(
                    *[mvn.lazy_covariance_matrix.unsqueeze(-3) for mvn in mvn_list],
                    dim=-3,
                )  # List of m (b x q x 1 x 1) -> (b x q x m x 1 x 1)
                mvn = MultitaskMultivariateNormal(
                    mean=mean,
                    covariance_matrix=BlockDiagLinearOperator(covars, block_dim=-3).to(
                        X
                    ),  # (b x q x m x 1 x 1) -> (b x q x m x m)
                    interleaved=False,
                )
            else:
                mvns = self._broadcast_mvns(mvns=mvns)
                mvn = (
                    mvns[0]
                    if len(mvns) == 1
                    else MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)
                )
            # Return the result as a GPyTorchPosterior/GaussianMixturePosterior.
            if any(is_ensemble(m) for m in self.models):
                # Mixing fully Bayesian and other GP models is currently
                # not supported.
                posterior = GaussianMixturePosterior(distribution=mvn)
            else:
                posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        raise NotImplementedError()

    def _broadcast_mvns(self, mvns: list[MultivariateNormal]) -> MultivariateNormal:
        """Broadcasts the batch shapes of the given MultivariateNormals.

        The MVNs will have a batch shape of `input_batch_shape x model_batch_shape`.
        If the model batch shapes are broadcastable, we will broadcast the mvns to
        a batch shape of `input_batch_shape x self.batch_shape`.

        Args:
            mvns: A list of MultivariateNormals.

        Returns:
            A list of MultivariateNormals with broadcasted batch shapes.
        """
        mvn_batch_shapes = {mvn.batch_shape for mvn in mvns}
        if len(mvn_batch_shapes) == 1:
            # All MVNs have the same batch shape. We can return as is.
            return mvns
        # This call will error out if they're not broadcastable.
        # If they're broadcastable, it'll log a warning.
        target_model_shape = self.batch_shape
        max_batch = max(mvn_batch_shapes, key=len)
        max_len = len(max_batch)
        input_batch_len = max_len - len(target_model_shape)
        for i in range(len(mvns)):  # Loop over index since we modify contents.
            while len(mvns[i].batch_shape) < max_len:
                # MVN is missing batch dimensions. Unsqueeze as needed.
                mvns[i] = mvns[i].unsqueeze(input_batch_len)
            if mvns[i].batch_shape != max_batch:
                # Expand to match the batch shapes.
                mvns[i] = mvns[i].expand(max_batch)
        return mvns


class MultiTaskGPyTorchModel(GPyTorchModel, ABC):
    r"""Abstract base class for multi-task models based on GPyTorch models.

    This class provides the `posterior` method to models that implement a
    "long-format" multi-task GP in the style of `MultiTaskGP`.
    """

    def _extract_targets_and_noise(self) -> tuple[Tensor, Tensor | None]:
        r"""Extract targets and noise variance for multi-task models.

        Returns a tuple of (Y, Yvar) where Y and Yvar have shape
        [batch_shape] x n x m, with batch_shape included only if the
        training data initially contained it.
        """
        return extract_targets_and_noise_single_output(self)

    def _restore_targets_and_noise(
        self, Y: Tensor, Yvar: Tensor | None, strict: bool
    ) -> None:
        r"""Restore targets and noise variance for multi-task models.

        Args:
            Y: Targets tensor in shape [batch_shape] x n x m.
            Yvar: Optional noise variance tensor in shape [batch_shape] x n x m.
            strict: Whether to strictly enforce shape constraints.
        """
        restore_targets_and_noise_single_output(self, Y, Yvar, strict)

    def _apply_noise(
        self,
        X: Tensor,
        mvn: MultivariateNormal,
        observation_noise: bool | Tensor,
    ) -> MultivariateNormal:
        """Adds the observation noise to the posterior.

        If the likelihood is a `FixedNoiseGaussianLikelihood`, then
        the average noise per task is computed, and a diagonal noise
        matrix is added to the posterior covariance matrix, where
        the noise per input is the average noise for its respective
        task. If the likelihood is a Gaussian likelihood, then
        currently there is a shared inferred noise level for all
        tasks.

        TODO: implement support for task-specific inferred noise levels.

        Args:
            X: A tensor of shape `batch_shape x q x d + 1`,
                where `d` is the dimension of the feature space and the `+ 1`
                dimension is the task feature / index.
            mvn: A `MultivariateNormal` object representing the posterior over the true
                latent function.
            num_outputs: The number of outputs of the model.
            observation_noise: If True, add observation noise from the respective
                likelihood. Tensor input is currently not supported.

        Returns:
            The posterior predictive.
        """
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(
                "Passing a tensor of observations is not supported by MultiTaskGP."
            )
        elif observation_noise is False:
            return mvn
        elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
            # get task features for test points
            test_task_features = X[..., self._task_feature]
            test_task_features = self._map_tasks(test_task_features).long()
            unique_test_task_features = test_task_features.unique()
            # get task features for training points
            train_task_features = self.train_inputs[0][..., self._task_feature]
            train_task_features = self._map_tasks(train_task_features).long()
            noise_by_task = torch.zeros(
                *self.batch_shape, self.num_tasks, dtype=X.dtype, device=X.device
            )
            for task_feature in unique_test_task_features:
                mask = train_task_features == task_feature
                noise_by_task[..., task_feature] = self.likelihood.noise[
                    ..., mask
                ].mean(dim=-1)
            # noise_shape is `broadcast(test_batch_shape, model.batch_shape) x q`
            noise_shape = (
                broadcast_shapes(X.shape[:-2], self.batch_shape) + X.shape[-2:-1]
            )
            # Expand and gather ensures we pick correct noise dimensions for
            # batch evaluations of batched models.
            observation_noise = noise_by_task.expand(*noise_shape[:-1], -1).gather(
                dim=-1, index=test_task_features.expand(noise_shape)
            )
            return self.likelihood(
                mvn,
                X,
                noise=observation_noise,
            )
        return self.likelihood(mvn, X)

    # pyre-ignore[14]: Inconsistent override. Could not find parameter
    # `Keywords(typing.Any)` in overriding signature.
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A tensor of shape `batch_shape x q x d` or `batch_shape x q x (d + 1)`,
                where `d` is the dimension of the feature space (not including task
                indices) and `q` is the number of points considered jointly. The `+ 1`
                dimension is the optional task feature / index. If given, the model
                produces the outputs for the given task indices. If omitted, the
                model produces outputs for tasks in in `self._output_tasks` (specified
                as `output_tasks` while constructing the model), which can overwritten
                using `output_indices`.
            output_indices: A list of task values over which to compute the posterior.
                Only used if `X` does not include the task feature. If omitted,
                defaults to `self._output_tasks`.
            observation_noise: If True, add observation noise from the respective
                likelihoods. If a Tensor, specifies the observation noise levels
                to add.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points. If the task features are included in `X`,
            the posterior will be single output. Otherwise, the posterior will be
            single or multi output corresponding to the tasks included in
            either the `output_indices` or `self._output_tasks`.
        """
        includes_task_feature = X.shape[-1] == self.num_non_task_features + 1
        if includes_task_feature:
            if output_indices is not None:
                raise ValueError(
                    "`output_indices` must be None when `X` includes task features."
                )
            task_features = X[..., self._task_feature].unique()
            num_outputs = 1
            X_full = X
        else:
            # Add the task features to construct the full X for evaluation.
            task_features = torch.tensor(
                self._output_tasks if output_indices is None else output_indices,
                dtype=torch.long,
                device=X.device,
            )
            num_outputs = len(task_features)
            X_full = _make_X_full(
                X=X, output_indices=task_features.tolist(), tf=self._task_feature
            )
        # Make sure all task feature values are valid.
        task_features = self._map_tasks(task_values=task_features)
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X_full = self.transform_inputs(X_full)
        with gpt_posterior_settings():
            mvn = self(X_full)
            mvn = self._apply_noise(
                X=X_full,
                mvn=mvn,
                observation_noise=observation_noise,
            )
        # If single-output, return the posterior of a single-output model
        if num_outputs == 1:
            posterior = GPyTorchPosterior(distribution=mvn)
        else:
            # Otherwise, make a MultitaskMultivariateNormal out of this
            mtmvn = MultitaskMultivariateNormal(
                mean=mvn.mean.view(*mvn.mean.shape[:-1], num_outputs, -1).transpose(
                    -1, -2
                ),
                covariance_matrix=mvn.lazy_covariance_matrix,
                interleaved=False,
            )
            posterior = GPyTorchPosterior(distribution=mtmvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def subset_output(self, idcs: list[int]) -> MultiTaskGPyTorchModel:
        r"""Returns a new model that only outputs a subset of the outputs.

        Args:
            idcs: A list of output indices, corresponding to the outputs to keep.

        Returns:
            A new model that only outputs the requested outputs.
        """
        raise UnsupportedError(
            "Subsetting outputs is not supported by `MultiTaskGPyTorchModel`."
        )
