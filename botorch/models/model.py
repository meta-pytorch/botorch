#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Abstract base module for all BoTorch models.

This module contains ``Model``, the abstract base class for all BoTorch models,
and ``ModelList``, a container for a list of Models.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Self, TYPE_CHECKING

import numpy as np
import torch
from botorch import settings
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    InputDataError,
)
from botorch.logging import shape_to_str
from botorch.models.utils.assorted import fantasize as fantasize_flag
from botorch.posteriors import Posterior, PosteriorList
from botorch.sampling.base import MCSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform  # pragma: no cover


class Model(Module, ABC):
    r"""Abstract base class for BoTorch models.

    The ``Model`` base class cannot be used directly; it only defines an API for other
    BoTorch models.

    ``Model`` subclasses ``torch.nn.Module``. While a ``Module`` is most typically
    encountered as a representation of a neural network layer, it can be used more
    generally: see
    documentation <https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html>_
    on custom NN Modules.

    ``Module`` provides several pieces of useful functionality: A ``Model``'s attributes of
    ``Tensor`` or ``Module`` type are automatically registered so they can be moved and/or
    cast with the ``to`` method, automatically differentiated, and used with CUDA.

    Attributes:
        _has_transformed_inputs: A boolean denoting whether ``train_inputs`` are currently
            stored as transformed or not.
        _original_train_inputs: A Tensor storing the original train inputs for use in
            ``_revert_to_original_inputs``. Note that this is necessary since
            transform / untransform cycle introduces numerical errors which lead
            to upstream errors during training.
        _is_fully_bayesian: Returns ``True`` if this is a fully Bayesian model.
        _is_ensemble: Returns ``True`` if this model consists of multiple models
            that are stored in an additional batch dimension. This is true for the fully
            Bayesian models.
    """  # noqa: E501

    _has_transformed_inputs: bool = False
    _original_train_inputs: Tensor | None = None
    _is_fully_bayesian = False
    _is_ensemble = False

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            ``self.transform_inputs(X)`` after the ``self.eval()`` call and before
            any ``model.forward`` or ``model.likelihood`` calls.

        Args:
            X: A ``b x q x d``-dim Tensor, where ``d`` is the dimension of the
                feature space, ``q`` is the number of points considered jointly,
                and ``b`` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: For models with an inferred noise level, if True,
                include observation noise. For models with an observed noise level,
                this must be a ``model_batch_shape x 1 x m``-dim tensor or
                a ``model_batch_shape x n' x m``-dim tensor containing the average
                noise for each batch and output. ``noise`` must be in the
                outcome-transformed space if an outcome transform is used.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``Posterior`` object, representing a batch of ``b`` joint distributions
            over ``q`` points and ``m`` outputs each.
        """
        pass  # pragma: no cover

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with ``m`` outputs, a ``test_batch_shape x q x d``-shaped
        input ``X`` to the ``posterior`` method returns a Posterior object over
        an output of shape ``broadcast(test_batch_shape, model.batch_shape) x q x m``.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define batch_shape property")

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"{cls_name} does not define num_outputs property")

    def subset_output(self, idcs: list[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            A ``Model`` object of the same type and with the same parameters as
            the current model, subset to the specified output indices.
        """
        raise NotImplementedError

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.

        Args:
            X: A ``batch_shape x n' x d``-dim Tensor, where ``d`` is the dimension of
                the feature space, ``n'`` is the number of points per batch, and
                ``batch_shape`` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A ``batch_shape' x n' x m``-dim Tensor, where ``m`` is the number of
                model outputs, ``n'`` is the number of points per batch, and
                ``batch_shape'`` is the batch shape of the observations.
                ``batch_shape'`` must be broadcastable to ``batch_shape`` using
                standard broadcasting semantics. If ``Y`` has fewer batch dimensions
                than ``X``, it is assumed that the missing batch dimensions are
                the same for all ``Y``.

        Returns:
            A ``Model`` object of the same type, representing the original model
            conditioned on the new observations ``(X, Y)`` (and possibly noise
            observations passed in via kwargs).
        """
        raise NotImplementedError(
            f"`condition_on_observations` not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
    ) -> dict[str, BotorchContainer | Tensor]:
        """
        Construct ``Model`` keyword arguments from a ``SupervisedDataset``.

        Args:
            training_data: A ``SupervisedDataset``, with attributes ``train_X``,
                ``train_Y``, and, optionally, ``train_Yvar``.

        Returns:
            A dict of keyword arguments that can be used to initialize a ``Model``,
            with keys ``train_X``, ``train_Y``, and, optionally, ``train_Yvar``.
        """
        if not isinstance(training_data, SupervisedDataset):
            raise TypeError(
                "Expected `training_data` to be a `SupervisedDataset`, but got "
                f"{type(training_data)}."
            )
        parsed_data = {"train_X": training_data.X, "train_Y": training_data.Y}
        if training_data.Yvar is not None:
            parsed_data["train_Yvar"] = training_data.Yvar
        return parsed_data

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Module | None = None,
    ) -> Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(
                        self.train_inputs[0]
                    )
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _revert_to_original_inputs(self) -> None:
        r"""Revert training inputs back to original."""
        if hasattr(self, "input_transform") and self._has_transformed_inputs:
            self.set_train_data(self._original_train_inputs, strict=False)
            self._has_transformed_inputs = False

    def eval(self) -> Model:
        r"""Puts the model in ``eval`` mode and sets the transformed inputs."""
        self._set_transformed_inputs()
        return super().eval()

    def train(self, mode: bool = True) -> Model:
        r"""Put the model in ``train`` mode. Reverts to the original inputs if
        in ``train`` mode (``mode=True``) or sets transformed inputs if in
        ``eval`` mode (``mode=False``).

        Args:
            mode: A boolean denoting whether to put in ``train`` or ``eval`` mode.
                If ``False``, model is put in ``eval`` mode.
        """
        if mode:
            self._revert_to_original_inputs()
        else:
            self._set_transformed_inputs()
        return super().train(mode=mode)

    @property
    def dtypes_of_buffers(self) -> set[torch.dtype]:
        return {t.dtype for t in self.buffers() if t is not None}


class FantasizeMixin(ABC):
    """
    Mixin to add a ``fantasize`` method to a ``Model``.

    Example:
        class BaseModel:
            def __init__(self, ...):
            def condition_on_observations(self, ...):
            def posterior(self, ...):
            def transform_inputs(self, ...):

        class ModelThatCanFantasize(BaseModel, FantasizeMixin):
            def __init__(self, args):
                super().__init__(args)

        model = ModelThatCanFantasize(...)
        model.fantasize(X)
    """

    @abstractmethod
    def condition_on_observations(self, X: Tensor, Y: Tensor) -> Self:
        """
        Classes that inherit from ``FantasizeMixin`` must implement
        a ``condition_on_observations`` method.
        """

    @abstractmethod
    def posterior(
        self,
        X: Tensor,
        *args,
        observation_noise: bool = False,
    ) -> Posterior:
        """
        Classes that inherit from ``FantasizeMixin`` must implement
        a ``posterior`` method.
        """

    @abstractmethod
    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Module | None = None,
    ) -> Tensor:
        """
        Classes that inherit from ``FantasizeMixin`` must implement
        a ``transform_inputs`` method.
        """

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Tensor | None = None,
        **kwargs: Any,
    ) -> Self:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at ``X``, including observation noise.
        If ``observation_noise`` is a Tensor, use it directly as the observation
        noise to add.
        (2) sample from this posterior (using ``sampler``) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A ``batch_shape x n' x d``-dim Tensor, where ``d`` is the dimension of
                the feature space, ``n'`` is the number of points per batch, and
                ``batch_shape`` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at ``X``.
            observation_noise: A ``model_batch_shape x 1 x m``-dim tensor or
                a ``model_batch_shape x n' x m``-dim tensor containing the average
                noise for each batch and output, where ``m`` is the number of outputs.
                ``noise`` must be in the outcome-transformed space if an outcome
                transform is used.
                If None and using an inferred noise likelihood, the noise will be the
                inferred noise level. If using a fixed noise likelihood, the mean across
                the observation noise in the training data is used as observation noise.
            kwargs: Will be passed to ``model.condition_on_observations``

        Returns:
            The constructed fantasy model.
        """
        if not isinstance(observation_noise, Tensor) and observation_noise is not None:
            raise DeprecationError(
                "`fantasize` no longer accepts a boolean for `observation_noise`."
            )
        elif observation_noise is None and isinstance(
            self.likelihood, FixedNoiseGaussianLikelihood
        ):
            if self.num_outputs > 1:
                # make noise ... x n x m
                observation_noise = self.likelihood.noise.transpose(-1, -2)
            else:
                observation_noise = self.likelihood.noise.unsqueeze(-1)
            observation_noise = observation_noise.mean(dim=-2, keepdim=True)
        # if the inputs are empty, expand the inputs
        if X.shape[-2] == 0:
            output_shape = (
                sampler.sample_shape
                + X.shape[:-2]
                + self.batch_shape
                + torch.Size([0, self.num_outputs])
            )
            Y = torch.empty(output_shape, dtype=X.dtype, device=X.device)
            if observation_noise is not None:
                kwargs["noise"] = observation_noise.expand(Y.shape[1:])
            return self.condition_on_observations(
                X=X,
                Y=Y,
                **kwargs,
            )
        propagate_grads = kwargs.pop("propagate_grads", False)
        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
                post_X = self.posterior(
                    X,
                    observation_noise=(
                        True if observation_noise is None else observation_noise
                    ),
                )
            Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
            if observation_noise is not None:
                kwargs["noise"] = observation_noise.expand(Y_fantasized.shape[1:])
            return self.condition_on_observations(X=X, Y=Y_fantasized, **kwargs)


class ModelList(Model):
    r"""A multi-output Model represented by a list of independent models.

    All BoTorch models are acceptable as inputs. The cost of this flexibility is
    that ``ModelList`` does not support all methods that may be implemented by its
    component models. One use case for ``ModelList`` is combining a regression
    model and a deterministic model in one multi-output container model, e.g.
    for cost-aware or multi-objective optimization where one of the outcomes is
    a deterministic function of the inputs.
    """

    def __init__(self, *models: Model) -> None:
        r"""
        Args:
            *models: A variable number of models.

        Example:
            >>> m_1 = SingleTaskGP(train_X, train_Y)
            >>> m_2 = GenericDeterministicModel(lambda x: x.sum(dim=-1))
            >>> m_12 = ModelList(m_1, m_2)
            >>> m_12.posterior(test_X)
        """
        super().__init__()
        self.models = ModuleList(models)

    def _get_group_subset_indices(self, idcs: list[int] | None) -> dict[int, list[int]]:
        r"""Convert global subset indices to indices for the individual models.

        Args:
            idcs: A list of indices to which the ``ModelList`` model is to be
                subset to.

        Returns:
            A dictionary mapping model indices to subset indices of the
                respective model in the ``ModelList``.
        """
        if idcs is None:
            return dict.fromkeys(range(len(self.models)))
        output_sizes = [model.num_outputs for model in self.models]
        cum_output_sizes = np.cumsum(output_sizes)
        idcs = [idx % cum_output_sizes[-1] for idx in idcs]
        group_indices: dict[int, list[int]] = defaultdict(list)
        for idx in idcs:
            grp_idx = np.argwhere(idx < cum_output_sizes)[0].item()
            sub_idx = idx - int(np.sum(output_sizes[:grp_idx]))
            group_indices[grp_idx].append(sub_idx)
        return group_indices

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Callable[[PosteriorList], Posterior] | None = None,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Note: The input transforms should be applied here using
            ``self.transform_inputs(X)`` after the ``self.eval()`` call and before
            any ``model.forward`` or ``model.likelihood`` calls.

        Args:
            X: A ``b x q x d``-dim Tensor, where ``d`` is the dimension of the
                feature space, ``q`` is the number of points considered jointly,
                and ``b`` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                respective likelihoods to the posterior. If a Tensor of shape
                ``(batch_shape) x q x m``, use it directly as the observation
                noise (with ``observation_noise[...,i]`` added to the posterior
                of the ``i``-th model). ``observation_noise`` is assumed
                to be in the outcome-transformed space, if an outcome transform
                is used by the model.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``Posterior`` object, representing a batch of ``b`` joint distributions
            over ``q`` points and ``m`` outputs each.
        """
        group_indices = self._get_group_subset_indices(idcs=output_indices)
        posteriors = []
        for i, idcs in group_indices.items():
            if isinstance(observation_noise, Tensor):
                if idcs is None:
                    start_idx = sum(m.num_outputs for m in self.models[:i])
                    end_idx = start_idx + self.models[i].num_outputs
                    idcs = list(range(start_idx, end_idx))
                obs_noise = observation_noise[..., idcs]
            else:
                obs_noise = observation_noise
            posteriors.append(
                self.models[i].posterior(
                    X=X, output_indices=idcs, observation_noise=obs_noise
                )
            )
        posterior = PosteriorList(*posteriors)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior=posterior, X=X)
        return posterior

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with ``m`` outputs, a ``test_batch_shape x q x d``-shaped
        input ``X`` to the ``posterior`` method returns a Posterior object over
        an output of shape ``broadcast(test_batch_shape, model.batch_shape) x q x m``.
        """
        batch_shape = self.models[0].batch_shape
        if all(batch_shape == m.batch_shape for m in self.models[1:]):
            return batch_shape
        # TODO: Allow broadcasting of model batch shapes
        raise NotImplementedError(
            f"`{self.__class__.__name__}.batch_shape` is only supported if all "
            "constituent models have the same `batch_shape`."
        )

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model.

        Equal to the sum of the number of outputs of the individual models
        in the ModelList.
        """
        return sum(model.num_outputs for model in self.models)

    def subset_output(self, idcs: list[int]) -> Model:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to. Relative to the
                overall number of outputs of the model.

        Returns:
            A ``Model`` (either a ``ModelList`` or one of the submodels) with
            the outputs subset to the indices in ``idcs``.

        Internally, this drops (if single-output) or subsets (if multi-output)
        the constitutent models and returns them as a ``ModelList``. If the
        result is a single (possibly subset) model from the list, returns this
        model (instead of forming a degenerate singe-model ``ModelList``).
        For instance, if ``m = ModelList(m1, m2)`` with ``m1`` a two-output model
        and ``m2`` a single-output model, then ``m.subset_output([1]) `` will return
        the model ``m1`` subset to its second output.
        """
        group_indices = self._get_group_subset_indices(idcs=idcs)
        subset_models = []
        for grp_idx, sub_idcs in group_indices.items():
            subset_model = self.models[grp_idx]
            if sub_idcs is not None and subset_model.num_outputs != len(sub_idcs):
                subset_model = subset_model.subset_output(idcs=sub_idcs)
            subset_models.append(subset_model)
        if len(subset_models) == 1:
            return subset_models[0]
        return self.__class__(*subset_models)

    def transform_inputs(self, X: Tensor) -> list[Tensor]:
        r"""Individually transform the inputs for each model.

        Args:
            X: A tensor of inputs.

        Returns:
            A list of tensors of transformed inputs.
        """
        transformed_X_list = []
        for model in self.models:
            try:
                transformed_X_list.append(model.input_transform(X))
            except AttributeError:
                transformed_X_list.append(X)
        return transformed_X_list

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        keep_transforms: bool = True,
        assign: bool = False,
    ) -> None:
        """Initialize the fully Bayesian models before loading the state dict."""
        for i, m in enumerate(self.models):
            filtered_dict = {
                k.replace(f"models.{i}.", ""): v
                for k, v in state_dict.items()
                if k.startswith(f"models.{i}.")
            }
            m.load_state_dict(filtered_dict, strict=strict, assign=assign)

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Tensor | None = None,
        evaluation_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Model:
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at ``X`` (including observation noise if
        ``observation_noise=True``).
        (2) sample from this posterior (using ``sampler``) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A ``batch_shape x n' x d``-dim Tensor, where ``d`` is the dimension of
                the feature space, ``n'`` is the number of points per batch, and
                ``batch_shape`` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at ``X``. If
                evaluation_mask is not None, this must be a ``ListSampler``.
            observation_noise: A ``model_batch_shape x 1 x m``-dim tensor or
                a ``model_batch_shape x n' x m``-dim tensor containing the average
                noise for each batch and output, where ``m`` is the number of outputs.
                ``noise`` must be in the outcome-transformed space if an outcome
                transform is used. If None, then the noise will be the inferred
                noise level.
            evaluation_mask: A ``n' x m``-dim tensor of booleans indicating which
                outputs should be fantasized for a given design. This uses the same
                evaluation mask for all batches.

        Returns:
            The constructed fantasy model.
        """
        if evaluation_mask is not None:
            if evaluation_mask.ndim != 2 or evaluation_mask.shape != torch.Size(
                [X.shape[-2], self.num_outputs]
            ):
                raise BotorchTensorDimensionError(
                    f"Expected evaluation_mask of shape `{X.shape[0]} "
                    f"x {self.num_outputs}`, but got "
                    f"{shape_to_str(evaluation_mask.shape)}."
                )
            if not isinstance(sampler, ListSampler):
                raise ValueError("Decoupled fantasization requires a list of samplers.")

        fant_models = []
        X_i = X
        if observation_noise is None:
            observation_noise_i = observation_noise
        for i in range(self.num_outputs):
            # get the inputs to fantasize at for output i
            if evaluation_mask is not None:
                mask_i = evaluation_mask[:, i]
                X_i = X[..., mask_i, :]
                # TODO (T158701749): implement a QMC DecoupledSampler that draws all
                # samples from a single Sobol sequence or consider requiring that the
                # sampling is IID to ensure good coverage.
                sampler_i = sampler.samplers[i]
                if observation_noise is not None:
                    observation_noise_i = observation_noise[..., mask_i, i : i + 1]
            else:
                sampler_i = (
                    sampler.samplers[i] if isinstance(sampler, ListSampler) else sampler
                )
                if observation_noise is not None:
                    observation_noise_i = observation_noise[..., i : i + 1]

            fant_model = self.models[i].fantasize(
                X=X_i,
                sampler=sampler_i,
                observation_noise=observation_noise_i,
                **kwargs,
            )
            fant_models.append(fant_model)
        return self.__class__(*fant_models)


class ModelDict(ModuleDict):
    r"""A lightweight container mapping model names to models."""

    def __init__(self, **models: Model) -> None:
        r"""Initialize a ``ModelDict``.

        Args:
            models: An arbitrary number of models. Each model can be any type
                of BoTorch ``Model``, including multi-output models and ``ModelList``.
        """
        if any(not isinstance(m, Model) for m in models.values()):
            raise InputDataError(
                f"Expected all models to be a BoTorch `Model`. Got {models}."
            )
        super().__init__(modules=models)
