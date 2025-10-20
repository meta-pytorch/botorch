#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Input Transformations.

These classes implement a variety of transformations for
input parameters including: learned input warping functions,
rounding functions, and log transformations. The input transformation
is typically part of a Model and applied within the model.forward()
method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any
from warnings import warn

import numpy as np
import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import UserInputWarning
from botorch.models.transforms.utils import (
    interaction_features,
    inv_kumaraswamy_warp,
    kumaraswamy_warp,
    subset_transform,
)
from botorch.models.utils import fantasize
from botorch.utils.rounding import approximate_round, OneHotArgmaxSTE, RoundSTE
from botorch.utils.sampling import draw_sobol_samples
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import Prior
from torch import LongTensor, nn, Tensor
from torch.nn import Module, ModuleDict
from torch.nn.functional import one_hot


class InputTransform(Module, ABC):
    r"""Abstract base class for input transforms.

    Properties:
        is_one_to_many: A boolean denoting whether the transform produces
            multiple values for each input.
        transform_on_train: A boolean indicating whether to apply the
            transform in train() mode.
        transform_on_eval: A boolean indicating whether to apply the
            transform in eval() mode.
        transform_on_fantasize: A boolean indicating whether to apply
            the transform when called from within a `fantasize` call.
    """

    is_one_to_many: bool = False
    transform_on_eval: bool
    transform_on_train: bool
    transform_on_fantasize: bool

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n' x d`-dim tensor of transformed inputs.
        """
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if fantasize.off() or self.transform_on_fantasize:
                return self.transform(X)
        return X

    @abstractmethod
    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method."
        )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Note: The reason that a custom equals method is defined rather than
        defining an __eq__ method is because defining an __eq__ method sets
        the __hash__ method to None. Hashing modules is currently used in
        pytorch. See https://github.com/pytorch/pytorch/issues/7733.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        other_state_dict = other.state_dict()
        return (
            type(self) is type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
            and (self.transform_on_fantasize == other.transform_on_fantasize)
            and all(
                torch.allclose(v, other_state_dict[k].to(v))
                for k, v in self.state_dict().items()
            )
        )

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        if self.transform_on_train:
            # We need to disable learning of bounds / affine coefficients here.
            # See why: https://github.com/meta-pytorch/botorch/issues/1078.
            if hasattr(self, "learn_coefficients"):
                learn_coefficients = self.learn_coefficients
                self.learn_coefficients = False
                result = self.transform(X)
                self.learn_coefficients = learn_coefficients
                return result
            else:
                return self.transform(X)
        return X


class BatchBroadcastedInputTransform(InputTransform, ModuleDict):
    r"""An input transform representing a list of transforms to be broadcasted."""

    def __init__(
        self,
        transforms: list[InputTransform],
        broadcast_index: int = -3,
    ) -> None:
        r"""A transform list that is broadcasted across a batch dimension specified by
        `broadcast_index`. This is allows using a batched Gaussian process model when
        the input transforms are different for different batch dimensions.

        Args:
            transforms: The transforms to broadcast across the first batch dimension.
                The transform at position i in the list will be applied to `X[i]` for
                a given input tensor `X` in the forward pass.
            broadcast_index: The tensor index at which the transforms are broadcasted.

        Example:
            >>> tf1 = Normalize(d=2)
            >>> tf2 = InputStandardize(d=2)
            >>> tf = BatchBroadcastedTransformList(transforms=[tf1, tf2])
        """
        super().__init__()
        self.transform_on_train = False
        self.transform_on_eval = False
        self.transform_on_fantasize = False
        self.transforms = transforms
        if broadcast_index in (-2, -1):
            raise ValueError(
                "The broadcast index cannot be -2 and -1, as these indices are reserved"
                " for non-batch, data and input dimensions."
            )
        self.broadcast_index = broadcast_index
        self.is_one_to_many = self.transforms[0].is_one_to_many
        if not all(tf.is_one_to_many == self.is_one_to_many for tf in self.transforms):
            raise ValueError(  # output shapes of transforms must be the same
                "All transforms must have the same is_one_to_many property."
            )
        for tf in self.transforms:
            self.transform_on_train |= tf.transform_on_train
            self.transform_on_eval |= tf.transform_on_eval
            self.transform_on_fantasize |= tf.transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Individual transforms are applied in sequence and results are returned as
        a batched tensor.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return torch.stack(
            [t.forward(Xi) for Xi, t in self._Xs_and_transforms(X)],
            dim=self.broadcast_index,
        )

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        return torch.stack(
            [t.untransform(Xi) for Xi, t in self._Xs_and_transforms(X)],
            dim=self.broadcast_index,
        )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and all(t1.equals(t2) for t1, t2 in zip(self.transforms, other.transforms))
            and (self.broadcast_index == other.broadcast_index)
        )

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        return torch.stack(
            [t.preprocess_transform(Xi) for Xi, t in self._Xs_and_transforms(X)],
            dim=self.broadcast_index,
        )

    def _Xs_and_transforms(self, X: Tensor) -> Iterable[tuple[Tensor, InputTransform]]:
        r"""Returns an iterable of sub-tensors of X and their associated transforms.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            An iterable containing tuples of sub-tensors of X and their transforms.
        """
        Xs = X.unbind(dim=self.broadcast_index)
        return zip(Xs, self.transforms)


class ChainedInputTransform(InputTransform, ModuleDict):
    r"""An input transform representing the chaining of individual transforms."""

    def __init__(self, **transforms: InputTransform) -> None:
        r"""Chaining of input transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.

        Example:
            >>> tf1 = Normalize(d=2)
            >>> tf2 = Normalize(d=2)
            >>> tf = ChainedInputTransform(tf1=tf1, tf2=tf2)
            >>> list(tf.keys())
            ['tf1', 'tf2']
            >>> tf["tf1"]
            Normalize()

        """
        super().__init__(OrderedDict(transforms))
        self.transform_on_train = False
        self.transform_on_eval = False
        self.transform_on_fantasize = False
        for tf in transforms.values():
            self.is_one_to_many |= tf.is_one_to_many
            self.transform_on_train |= tf.transform_on_train
            self.transform_on_eval |= tf.transform_on_eval
            self.transform_on_fantasize |= tf.transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Individual transforms are applied in sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        for tf in self.values():
            X = tf.forward(X)
        return X

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        for tf in reversed(self.values()):
            X = tf.untransform(X)
        return X

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return super().equals(other=other) and all(
            t1.equals(t2) for t1, t2 in zip(self.values(), other.values())
        )

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        for tf in self.values():
            X = tf.preprocess_transform(X)
        return X


class ReversibleInputTransform(InputTransform, ABC):
    r"""An abstract class for a reversible input transform.

    Properties:
        reverse: A boolean indicating if the functionality of transform
            and untransform methods should be swapped.
    """

    reverse: bool

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return self._untransform(X) if self.reverse else self._transform(X)

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        return self._transform(X) if self.reverse else self._untransform(X)

    @abstractmethod
    def _transform(self, X: Tensor) -> Tensor:
        r"""Forward transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Reverse transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass  # pragma: no cover

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return super().equals(other=other) and (self.reverse == other.reverse)


class AffineInputTransform(ReversibleInputTransform):
    def __init__(
        self,
        d: int,
        coefficient: Tensor,
        offset: Tensor,
        indices: list[int] | Tensor | None = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        r"""Apply affine transformation to input:

            `output = (input - offset) / coefficient`

        Args:
            d: The dimension of the input space.
            coefficient: Tensor of linear coefficients, shape must to be
                broadcastable with `(batch_shape x n x d)`-dim input tensors.
            offset: Tensor of offset coefficients, shape must to be
                broadcastable with `(batch_shape x n x d)`-dim input tensors.
            indices: The indices of the inputs to transform. If omitted,
                take all dimensions of the inputs into account. Either a list of ints
                or a Tensor of type `torch.long`.
            batch_shape: The batch shape of the inputs (assuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                transformation per batch, otherwise uses a single transformation.
            transform_on_train: A boolean indicating whether to apply the
                transform in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        if (indices is not None) and (len(indices) == 0):
            raise ValueError("`indices` list is empty!")
        if (indices is not None) and (len(indices) > 0):
            indices = torch.as_tensor(
                indices, dtype=torch.long, device=coefficient.device
            )
            if len(indices) > d:
                raise ValueError("Can provide at most `d` indices!")
            if (indices > d - 1).any():
                raise ValueError("Elements of `indices` have to be smaller than `d`!")
            if len(indices.unique()) != len(indices):
                raise ValueError("Elements of `indices` tensor must be unique!")
            self.register_buffer("indices", indices)
        torch.broadcast_shapes(coefficient.shape, offset.shape)

        self._d = d
        self.register_buffer("_coefficient", coefficient)
        self.register_buffer("_offset", offset)
        self.batch_shape = batch_shape
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @property
    def coefficient(self) -> Tensor:
        r"""The tensor of linear coefficients."""
        coeff = self._coefficient
        return coeff if self.learn_coefficients and self.training else coeff.detach()

    @property
    def offset(self) -> Tensor:
        r"""The tensor of offset coefficients."""
        offset = self._offset
        return offset if self.learn_coefficients and self.training else offset.detach()

    @property
    def learn_coefficients(self) -> bool:
        return getattr(self, "_learn_coefficients", False)

    @learn_coefficients.setter
    def learn_coefficients(self, value: bool) -> None:
        r"""A boolean denoting whether to learn the coefficients
        from inputs during model training.
        """
        self._learn_coefficients = value

    @subset_transform
    def _transform(self, X: Tensor) -> Tensor:
        r"""Apply affine transformation to input.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        self._check_shape(X)
        if self.learn_coefficients and self.training:
            self._update_coefficients(X)
        self._to(X)
        return (X - self.offset) / self.coefficient

    @subset_transform
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Apply inverse of affine transformation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        self._to(X)
        return self.coefficient * X + self.offset

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if hasattr(self, "indices") != hasattr(other, "indices"):
            return False
        isequal = (
            super().equals(other=other)
            and (self._d == other._d)
            and torch.allclose(self.coefficient, other.coefficient)
            and torch.allclose(self.offset, other.offset)
            and self.learn_coefficients == other.learn_coefficients
        )
        if hasattr(self, "indices"):
            isequal = isequal and bool((self.indices == other.indices).all())
        return isequal

    def _check_shape(self, X: Tensor) -> None:
        """Checking input dimensions, included to increase code sharing
        among the derived classes Normalize and InputStandardize.
        """
        if X.size(-1) != self.offset.size(-1):
            raise BotorchTensorDimensionError(
                f"Wrong input dimension. Received {X.size(-1)}, "
                f"expected {self.offset.size(-1)}."
            )
        if X.ndim < 2:
            raise BotorchTensorDimensionError(
                f"`X` must have at least 2 dimensions, but has {X.ndim}."
            )

        n = len(self.batch_shape) + 2
        if self.training and X.ndim < n:
            raise ValueError(
                f"`X` must have at least {n} dimensions, {n - 2} batch and 2 innate"
                f" , but has {X.ndim}."
            )

        torch.broadcast_shapes(self.coefficient.shape, self.offset.shape, X.shape)

    def _to(self, X: Tensor) -> None:
        r"""Makes coefficient and offset have same device and dtype as X."""
        self._coefficient = self.coefficient.to(X)
        self._offset = self.offset.to(X)

    def _update_coefficients(self, X: Tensor) -> None:
        r"""Updates affine coefficients. Implemented by subclasses,
        e.g. Normalize and InputStandardize.
        """
        raise NotImplementedError(
            "Only subclasses of AffineInputTransform implement "
            "_update_coefficients, e.g. Normalize and InputStandardize."
        )


class Normalize(AffineInputTransform):
    r"""Normalize the inputs have unit range and be centered at 0.5 (by default).

    If no explicit bounds are provided this module is stateful: If in train mode,
    calling `forward` updates the module state (i.e. the normalizing bounds). If
    in eval mode, calling `forward` simply applies the normalization using the
    current module state.
    """

    def __init__(
        self,
        d: int,
        indices: list[int] | Tensor | None = None,
        bounds: Tensor | None = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_range: float = 1e-8,
        learn_bounds: bool | None = None,
        almost_zero: float = 1e-12,
        center: float = 0.5,
    ) -> None:
        r"""Normalize the inputs to the unit cube.

        Args:
            d: The dimension of the input space.
            indices: The indices of the inputs to normalize. If omitted,
                take all dimensions of the inputs into account.
            bounds: If provided, use these bounds to normalize the inputs. If
                omitted, learn the bounds in train mode.
            batch_shape: The batch shape of the inputs (assuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            min_range: If the range of an input dimension is smaller than `min_range`,
                that input dimension will not be normalized. This is equivalent to
                using bounds of `[0, 1]` for this dimension, and helps avoid division
                by zero errors and related numerical issues. See the example below.
                NOTE: This only applies if `learn_bounds=True`.
            learn_bounds: Whether to learn the bounds in train mode. Defaults
                to False if bounds are provided, otherwise defaults to True.
            center: The center of the range for each parameter. Default: 0.5.

        Example:
            >>> t = Normalize(d=2)
            >>> t(torch.tensor([[3., 2.], [3., 6.]]))
            ... tensor([[3., 2.],
            ...         [3., 6.]])
            >>> t.eval()
            ... Normalize()
            >>> t(torch.tensor([[3.5, 2.8]]))
            ... tensor([[3.5, 0.2]])
            >>> t.bounds
            ... tensor([[0., 2.],
            ...         [1., 6.]])
            >>> t.coefficient
            ... tensor([[1., 4.]])
        """
        if learn_bounds is not None:
            self.learn_coefficients = learn_bounds
        else:
            self.learn_coefficients = bounds is None
        transform_dimension = d if indices is None else len(indices)
        if bounds is not None:
            if indices is not None and bounds.size(-1) == d:
                bounds = bounds[..., indices]
            if bounds.size(-1) != transform_dimension:
                raise BotorchTensorDimensionError(
                    "Dimensions of provided `bounds` are incompatible with "
                    f"transform_dimension = {transform_dimension}!"
                )
            offset = bounds[..., 0:1, :]
            coefficient = bounds[..., 1:2, :] - offset
            if coefficient.ndim > 2:
                batch_shape = coefficient.shape[:-2]
        else:
            coefficient = torch.ones(*batch_shape, 1, transform_dimension)
            offset = torch.zeros(*batch_shape, 1, transform_dimension)
            if self.learn_coefficients is False:
                warn(
                    "learn_bounds is False and no bounds were provided. The bounds "
                    "will not be updated and the transform will be a no-op.",
                    UserInputWarning,
                )
        self.center = center
        super().__init__(
            d=d,
            coefficient=coefficient,
            offset=offset + (0.5 - center) * coefficient,
            indices=indices,
            batch_shape=batch_shape,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
        )
        self.min_range = min_range

    @property
    def ranges(self):
        return self.coefficient

    @property
    def mins(self):
        return self.offset

    @property
    def bounds(self) -> Tensor:
        r"""The bounds used for normalizing the inputs."""
        return torch.cat([self.offset, self.offset + self.coefficient], dim=-2)

    @property
    def learn_bounds(self) -> bool:
        return self.learn_coefficients

    def _update_coefficients(self, X) -> None:
        """Computes the normalization bounds and updates the affine
        coefficients, which determine the base class's behavior.
        """
        # Aggregate mins and ranges over extra batch and marginal dims
        batch_ndim = min(len(self.batch_shape), X.ndim - 2)  # batch rank of `X`
        reduce_dims = (*range(X.ndim - batch_ndim - 2), X.ndim - 2)
        offset = torch.amin(X, dim=reduce_dims).unsqueeze(-2)
        coefficient = torch.amax(X, dim=reduce_dims).unsqueeze(-2) - offset
        almost_zero = coefficient < self.min_range
        self._coefficient = torch.where(almost_zero, 1.0, coefficient)
        self._offset = (
            torch.where(almost_zero, 0.0, offset)
            + (0.5 - self.center) * self._coefficient
        )

    def get_init_args(self) -> dict[str, Any]:
        r"""Get the arguments necessary to construct an exact copy of the transform."""
        return {
            "d": self._d,
            "indices": getattr(self, "indices", None),
            "bounds": self.bounds,
            "batch_shape": self.batch_shape,
            "transform_on_train": self.transform_on_train,
            "transform_on_eval": self.transform_on_eval,
            "transform_on_fantasize": self.transform_on_fantasize,
            "reverse": self.reverse,
            "min_range": self.min_range,
            "learn_bounds": self.learn_bounds,
        }


class InputStandardize(AffineInputTransform):
    r"""Standardize inputs (zero mean, unit variance).

    In train mode, calling `forward` updates the module state
    (i.e. the mean/std normalizing constants). If in eval mode, calling `forward`
    simply applies the standardization using the current module state.
    """

    def __init__(
        self,
        d: int,
        indices: list[int] | Tensor | None = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_std: float = 1e-8,
    ) -> None:
        r"""Standardize inputs (zero mean, unit variance).

        Args:
            d: The dimension of the input space.
            indices: The indices of the inputs to standardize. If omitted,
                take all dimensions of the inputs into account.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            min_std: If the standard deviation of an input dimension is smaller than
                `min_std`, that input dimension will not be standardized. This is
                equivalent to using a standard deviation of 1.0 and a mean of 0.0 for
                this dimension, and helps avoid division by zero errors and related
                numerical issues.
        """
        transform_dimension = d if indices is None else len(indices)
        super().__init__(
            d=d,
            coefficient=torch.ones(*batch_shape, 1, transform_dimension),
            offset=torch.zeros(*batch_shape, 1, transform_dimension),
            indices=indices,
            batch_shape=batch_shape,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            reverse=reverse,
        )
        self.min_std = min_std
        self.learn_coefficients = True

    @property
    def stds(self):
        return self.coefficient

    @property
    def means(self):
        return self.offset

    def _update_coefficients(self, X: Tensor) -> None:
        """Computes the normalization bounds and updates the affine
        coefficients, which determine the base class's behavior.
        """
        # Aggregate means and standard deviations over extra batch and marginal dims
        batch_ndim = min(len(self.batch_shape), X.ndim - 2)  # batch rank of `X`
        reduce_dims = (*range(X.ndim - batch_ndim - 2), X.ndim - 2)
        coefficient, offset = (
            values.unsqueeze(-2)
            for values in torch.std_mean(X, dim=reduce_dims, unbiased=True)
        )
        almost_zero = coefficient < self.min_std
        self._coefficient = torch.where(almost_zero, 1.0, coefficient)
        self._offset = torch.where(almost_zero, 0.0, offset)


class Round(InputTransform):
    r"""A discretization transformation for discrete inputs.

    If `approximate=False` (the default), uses PyTorch's `round`.

    If `approximate=True`, a differentiable approximate rounding function is
    used, with a temperature parameter of `tau`. This method is a piecewise
    approximation of a rounding function where each piece is a hyperbolic
    tangent function.

    For integers, this will typically be used in conjunction
    with normalization as follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization). 1. These are unnormalized back to the raw input space.
    2. The integers are rounded. 3. All values are normalized to the unit
    cube.

    In train() mode, the inputs can either (a) be normalized to the unit
    cube or (b) provided using their raw values. In the case of (a)
    transform_on_train should be set to True, so that the normalized inputs
    are unnormalized before rounding. In the case of (b) transform_on_train
    should be set to False, so that the raw inputs are rounded and then
    normalized to the unit cube.

    By default, the straight through estimators are used for the gradients as
    proposed in [Daulton2022bopr]_. This transformation supports differentiable
    approximate rounding (currently only for integers). The rounding function
    is approximated with a piece-wise function where each piece is a hyperbolic
    tangent function.

    For categorical parameters, the input must be one-hot encoded.

    Example:
        >>> bounds = torch.tensor([[0, 5], [0, 1], [0, 1]]).t()
        >>> integer_indices = [0]
        >>> categorical_features = {1: 2}
        >>> unnormalize_tf = Normalize(
        >>>     d=d,
        >>>     bounds=bounds,
        >>>     transform_on_eval=True,
        >>>     transform_on_train=True,
        >>>     reverse=True,
        >>> )
        >>> round_tf = Round(integer_indices, categorical_features)
        >>> normalize_tf = Normalize(d=d, bounds=bounds)
        >>> tf = ChainedInputTransform(
        >>>     tf1=unnormalize_tf, tf2=round_tf, tf3=normalize_tf
        >>> )
    """

    def __init__(
        self,
        integer_indices: list[int] | LongTensor | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        approximate: bool = False,
        tau: float = 1e-3,
    ) -> None:
        r"""Initialize transform.

        Args:
            integer_indices: The indices of the integer inputs.
            categorical_features: A dictionary mapping the starting index of each
                categorical feature to its cardinality. This assumes that categoricals
                are one-hot encoded.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            approximate: A boolean indicating whether approximate or exact
                rounding should be used. Default: False.
            tau: The temperature parameter for approximate rounding.
        """
        if approximate and categorical_features is not None:
            raise NotImplementedError
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        integer_indices = integer_indices if integer_indices is not None else []
        self.register_buffer(
            "integer_indices", torch.as_tensor(integer_indices, dtype=torch.long)
        )
        self.categorical_features = categorical_features or {}
        self.approximate = approximate
        self.tau = tau

    def transform(self, X: Tensor) -> Tensor:
        r"""Discretize the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of discretized inputs.
        """
        X_rounded = X.clone()
        # round integers
        X_int = X_rounded[..., self.integer_indices]
        if self.approximate:
            X_int = approximate_round(X_int, tau=self.tau)
        else:
            X_int = RoundSTE.apply(X_int)
        X_rounded[..., self.integer_indices] = X_int
        # discrete categoricals to the category with the largest value
        # in the continuous relaxation of the one-hot encoding
        for start, card in self.categorical_features.items():
            end = start + card
            X_rounded[..., start:end] = OneHotArgmaxSTE.apply(X[..., start:end])
        return X_rounded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and (self.integer_indices == other.integer_indices).all()
            and self.categorical_features == other.categorical_features
            and self.approximate == other.approximate
            and self.tau == other.tau
        )

    def get_init_args(self) -> dict[str, Any]:
        r"""Get the arguments necessary to construct an exact copy of the transform."""
        return {
            "integer_indices": self.integer_indices,
            "categorical_features": self.categorical_features,
            "transform_on_train": self.transform_on_train,
            "transform_on_eval": self.transform_on_eval,
            "transform_on_fantasize": self.transform_on_fantasize,
            "approximate": self.approximate,
            "tau": self.tau,
        }


class Log10(ReversibleInputTransform):
    r"""A base-10 log transformation."""

    def __init__(
        self,
        indices: list[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to log transform.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse

    @subset_transform
    def _transform(self, X: Tensor) -> Tensor:
        r"""Log transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return X.log10()

    @subset_transform
    def _untransform(self, X: Tensor) -> Tensor:
        r"""Reverse the log transformation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        return 10.0**X


class Warp(ReversibleInputTransform, GPyTorchModule):
    r"""A transform that uses learned input warping functions.

    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.

    TODO: implement support using independent warping functions
    for each output in batched multi-output and multi-task models.

    For now, ModelListGPs should be used to learn independent warping
    functions for each output.
    """

    # TODO: make minimum value dtype-dependent
    _min_concentration_level = 1e-4

    def __init__(
        self,
        d: int,
        indices: list[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Prior | None = None,
        concentration0_prior: Prior | None = None,
        batch_shape: torch.Size | None = None,
        bounds: Tensor | None = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            concentration1_prior: A prior distribution on the concentration1 parameter
                of the Kumaraswamy distribution.
            concentration0_prior: A prior distribution on the concentration0 parameter
                of the Kumaraswamy distribution.
            batch_shape: An optional batch shape, for learning independent warping
                parameters for each batch of inputs. This should match the input batch
                shape of the model (i.e., `train_X.shape[:-2]`).
                NOTE: This is only supported for single-output models.
            bounds: A `2 x d`-dim tensor of lower and upper bounds for the inputs.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse
        self.batch_shape = batch_shape or torch.Size([])
        self._eps = eps
        # Note: we don't pass batch shape here and assume that the
        # bounds apply to all batch dims (if batch dims are present).
        # The warping function can add a batch dim if needed.
        self._normalize = Normalize(d=d, indices=indices, bounds=bounds)
        if len(self.batch_shape) > 0:
            # Note: this follows the gpytorch shape convention for lengthscales
            # There is ongoing discussion about the extra `1`.
            # TODO: update to follow new gpytorch convention resulting from
            # https://github.com/cornellius-gp/gpytorch/issues/1317
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape
        for i in (0, 1):
            p_name = f"concentration{i}"
            self.register_parameter(
                p_name,
                nn.Parameter(torch.full(batch_shape + self.indices.shape, 1.0)),
            )
        if concentration0_prior is not None:
            self.register_prior(
                "concentration0_prior",
                concentration0_prior,
                lambda m: m.concentration0,
                lambda m, v: m._set_concentration(i=0, value=v),
            )
        if concentration1_prior is not None:
            self.register_prior(
                "concentration1_prior",
                concentration1_prior,
                lambda m: m.concentration1,
                lambda m, v: m._set_concentration(i=1, value=v),
            )
        for i in (0, 1):
            p_name = f"concentration{i}"
            constraint = GreaterThan(
                self._min_concentration_level,
                transform=None,
                # set the initial value to be the identity transformation
                initial_value=1.0,
            )
            self.register_constraint(param_name=p_name, constraint=constraint)

    def _set_concentration(self, i: int, value: float | Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.concentration0)
        self.initialize(**{f"concentration{i}": value})

    @subset_transform
    def _warp_transform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy CDF.

        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.

        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor
                of transformed inputs.
        """
        return kumaraswamy_warp(
            X=X, c0=self.concentration0, c1=self.concentration1, eps=self._eps
        )

    def _transform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy CDF.

        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.

        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor of transformed
                inputs.
        """
        # Normalize to unit cube
        X = self._normalize(X=X)
        # normalize to [eps, 1-eps], IDEA: could use Normalize and ChainedTransform.
        return self._warp_transform(X=X)

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy inverse CDF.

        Args:
            X: A `input_batch_shape x batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `input_batch_shape x batch_shape x n x d`-dim tensor of transformed
                inputs.
        """
        if len(self.batch_shape) > 0:
            if self.batch_shape != X.shape[-2 - len(self.batch_shape) : -2]:
                raise BotorchTensorDimensionError(
                    "The right most batch dims of X must match self.batch_shape: "
                    f"({self.batch_shape})."
                )
        untransformed_X = self._warp_untransform(X=X)
        return self._normalize.untransform(X=untransformed_X)

    @subset_transform
    def _warp_untransform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy inverse CDF.

        Args:
            X: A `input_batch_shape x batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `input_batch_shape x batch_shape x n x d`-dim tensor of transformed
                inputs.
        """
        return inv_kumaraswamy_warp(
            X=X, c0=self.concentration0, c1=self.concentration1, eps=self._eps
        )


class AppendFeatures(InputTransform):
    r"""A transform that appends the input with a given set of features either
    provided beforehand or generated on the fly via a callable.

    As an example, the predefined set of features can be used with
    `RiskMeasureMCObjective` to optimize risk measures as described in
    [Cakmak2020risk]_. A tutorial notebook implementing the rhoKG acqusition
    function introduced in [Cakmak2020risk]_ can be found at
    https://botorch.org/docs/tutorials/risk_averse_bo_with_environmental_variables.

    The steps for using this to obtain samples of a risk measure are as follows:

    -   Train a model on `(x, w)` inputs and the corresponding observations;

    -   Pass in an instance of `AppendFeatures` with the `feature_set` denoting the
        samples of `W` as the `input_transform` to the trained model;

    -   Call `posterior(...).rsample(...)` on the model with `x` inputs only to
        get the joint posterior samples over `(x, w)`s, where the `w`s come
        from the `feature_set`;

    -   Pass these posterior samples through the `RiskMeasureMCObjective` of choice to
        get the samples of the risk measure.

    Note: The samples of the risk measure obtained this way are in general biased
    since the `feature_set` does not fully represent the distribution of the
    environmental variable.

    Possible examples for using a callable include statistical models that are built on
    PyTorch, built-in mathematical operations such as torch.sum, or custom scripted
    functions. By this, this input transform allows for advanced feature engineering
    and transfer learning models within the optimization loop.

    Example:
        >>> # We consider 1D `x` and 1D `w`, with `W` having a
        >>> # uniform distribution over [0, 1]
        >>> model = SingleTaskGP(
        ...     train_X=torch.rand(10, 2),
        ...     train_Y=torch.randn(10, 1),
        ...     input_transform=AppendFeatures(feature_set=torch.rand(10, 1))
        ... )
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
        >>> test_x = torch.rand(3, 1)
        >>> # `posterior_samples` is a `10 x 30 x 1`-dim tensor
        >>> posterior_samples = model.posterior(test_x).rsamples(torch.size([10]))
        >>> risk_measure = VaR(alpha=0.8, n_w=10)
        >>> # `risk_measure_samples` is a `10 x 3`-dim tensor of samples of the
        >>> # risk measure VaR
        >>> risk_measure_samples = risk_measure(posterior_samples)
    """

    is_one_to_many: bool = True

    def __init__(
        self,
        feature_set: Tensor | None = None,
        f: Callable[[Tensor], Tensor] | None = None,
        indices: list[int] | None = None,
        fkwargs: dict[str, Any] | None = None,
        skip_expand: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Append `feature_set` to each input or generate a set of features to
        append on the fly via a callable.

        Args:
            feature_set: An `n_f x d_f`-dim tensor denoting the features to be
                appended to the inputs. Default: None.
            f: A callable mapping a `batch_shape x q x d`-dim input tensor `X`
                to a `batch_shape x q x n_f x d_f`-dimensional output tensor.
                Default: None.
            indices: List of indices denoting the indices of the features to be
                passed into f. Per default all features are passed to `f`.
                Default: None.
            fkwargs: Dictionary of keyword arguments passed to the callable `f`.
                Default: None.
            skip_expand: A boolean indicating whether to expand the input tensor
                before appending features. This is intended for use with an
                `InputPerturbation`. If `True`, the input tensor will be expected
                to be of shape `batch_shape x (q * n_f) x d`. Not implemented
                in combination with a callable.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if (feature_set is None) and (f is None):
            raise ValueError(
                "Either a `feature_set` or a callable `f` has to be provided."
            )
        if (feature_set is not None) and (f is not None):
            raise ValueError(
                "Only one can be used: either `feature_set` or callable `f`."
            )
        if feature_set is not None:
            if feature_set.dim() != 2:
                raise ValueError("`feature_set` must be an `n_f x d_f`-dim tensor!")
            self.register_buffer("feature_set", feature_set)
            self._f = None
        if f is not None:
            if skip_expand:
                raise ValueError(
                    "`skip_expand` option is not supported in case of using a callable"
                )
            if (indices is not None) and (len(indices) == 0):
                raise ValueError("`indices` list is empty!")
            if indices is not None:
                indices = torch.tensor(indices, dtype=torch.long)
                if len(indices.unique()) != len(indices):
                    raise ValueError("Elements of `indices` tensor must be unique!")
                self.indices = indices
            else:
                self.indices = slice(None)
            self._f = f
            self.fkwargs = fkwargs or {}

        self.skip_expand = skip_expand
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by appending `feature_set` to each input or
        by generating a set of features to be appended on the fly via a callable.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_f x (d + d_f)`-dim tensor with `feature_set` appended as the last `d_f`
        dimensions. For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_f) x (d + d_f)`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_f: (i + 1) * n_f, :]`.

        Note: Adding the `feature_set` on the `q-batch` dimension is necessary to avoid
        introducing additional bias by evaluating the inputs on independent GP
        sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs. If `self.skip_expand` is
                `True`, then `X` should be of shape `batch_shape x (q * n_f) x d`,
                typically obtained by passing a `batch_shape x q x d` shape input
                through an `InputPerturbation` with `n_f` perturbation values.

        Returns:
            A `batch_shape x (q * n_f) x (d + d_f)`-dim tensor of appended inputs.
        """
        if self._f is not None:
            expanded_features = self._f(X[..., self.indices], **self.fkwargs)
            n_f = expanded_features.shape[-2]
        else:
            n_f = self.feature_set.shape[-2]

        if self.skip_expand:
            expanded_X = X.view(*X.shape[:-2], -1, n_f, X.shape[-1])
        else:
            expanded_X = X.unsqueeze(dim=-2).expand(*X.shape[:-1], n_f, -1)

        if self._f is None:
            expanded_features = self.feature_set.expand(*expanded_X.shape[:-1], -1)

        appended_X = torch.cat([expanded_X, expanded_features], dim=-1)
        return appended_X.view(*X.shape[:-2], -1, appended_X.shape[-1])


class InteractionFeatures(AppendFeatures):
    r"""A transform that appends the first-order interaction terms $x_i * x_j, i < j$,
    for all or a subset of the input variables."""

    def __init__(
        self,
        indices: list[int] | None = None,
    ) -> None:
        r"""Initializes the InteractionFeatures transform.

        Args:
            indices: Indices of the subset of dimensions to compute interaction
                features on.
        """

        super().__init__(
            f=interaction_features,
            indices=indices,
            transform_on_train=True,
            transform_on_eval=True,
            transform_on_fantasize=True,
        )


class FilterFeatures(InputTransform):
    r"""A transform that filters the input with a given set of features indices.

    As an example, this can be used in a multiobjective optimization with `ModelListGP`
    in which the specific models only share subsets of features (feature selection).
    A reason could be that it is known that specific features do not have any impact on
    a specific objective but they need to be included in the model for another one.
    """

    def __init__(
        self,
        feature_indices: Tensor,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        r"""Filter features from a model.

        Args:
            feature_set: An one-dim tensor denoting the indices of the features to be
                kept and fed to the model.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
        """
        super().__init__()
        if feature_indices.dim() != 1:
            raise ValueError("`feature_indices` must be a one-dimensional tensor!")
        if feature_indices.dtype != torch.int64:
            raise ValueError("`feature_indices` tensor must be int64/long!")
        if (feature_indices < 0).any():
            raise ValueError(
                "Elements of `feature_indices` have to be larger/equal to zero!"
            )
        if len(feature_indices.unique()) != len(feature_indices):
            raise ValueError("Elements of `feature_indices` tensor must be unique!")
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.register_buffer("feature_indices", feature_indices)

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by keeping only the in `feature_indices` specified
        feature indices and filtering out the others.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x q x e`-dim tensor of filtered inputs,
                where `e` is the length of `feature_indices`.
        """
        return X[..., self.feature_indices]

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if len(self.feature_indices) != len(other.feature_indices):
            return False
        return super().equals(other=other)


class InputPerturbation(InputTransform):
    r"""A transform that adds the set of perturbations to the given input.

    Similar to `AppendFeatures`, this can be used with `RiskMeasureMCObjective`
    to optimize risk measures. See `AppendFeatures` for additional discussion
    on optimizing risk measures.

    A tutorial notebook using this with `qNoisyExpectedImprovement` can be found at
    https://botorch.org/docs/tutorials/risk_averse_bo_with_input_perturbations.
    """

    is_one_to_many: bool = True

    def __init__(
        self,
        perturbation_set: Tensor | Callable[[Tensor], Tensor],
        bounds: Tensor | None = None,
        indices: list[int] | None = None,
        multiplicative: bool = False,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Add `perturbation_set` to each input.

        Args:
            perturbation_set: An `n_p x d`-dim tensor denoting the perturbations
                to be added to the inputs. Alternatively, this can be a callable that
                returns `batch x n_p x d`-dim tensor of perturbations for input of
                shape `batch x d`. This is useful for heteroscedastic perturbations.
            bounds: A `2 x d`-dim tensor of lower and upper bounds for each
                column of the input. If given, the perturbed inputs will be
                clamped to these bounds.
            indices: A list of indices specifying a subset of inputs on which to apply
                the transform. Note that `len(indices)` should be equal to the second
                dimension of `perturbation_set` and `bounds`. The dimensionality of
                the input `X.shape[-1]` can be larger if we only transform a subset.
            multiplicative: A boolean indicating whether the input perturbations
                are additive or multiplicative. If True, inputs will be multiplied
                with the perturbations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if isinstance(perturbation_set, Tensor):
            if perturbation_set.dim() != 2:
                raise ValueError("`perturbation_set` must be an `n_p x d`-dim tensor!")
            self.register_buffer("perturbation_set", perturbation_set)
        else:
            self.perturbation_set = perturbation_set
        if bounds is not None:
            if (
                isinstance(perturbation_set, Tensor)
                and bounds.shape[-1] != perturbation_set.shape[-1]
            ):
                raise ValueError(
                    "`bounds` must have the same number of columns (last dimension) as "
                    f"the `perturbation_set`! Got {bounds.shape[-1]} and "
                    f"{perturbation_set.shape[-1]}."
                )
            self.register_buffer("bounds", bounds)
        else:
            self.bounds = None
        self.register_buffer("_perturbations", None)
        self.indices = indices
        self.multiplicative = multiplicative
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by adding `perturbation_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_p x d`-dim tensor with the `perturbation_set` added to the input.
        For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_p) x d`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_w: (i + 1) * n_w, :]`.

        Note: Adding the `perturbation_set` on the `q-batch` dimension is necessary
        to avoid introducing additional bias by evaluating the inputs on independent
        GP sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_p) x d`-dim tensor of perturbed inputs.
        """
        # NOTE: If we had access to n_p without evaluating _perturbations when the
        # perturbation_set is a function, we could move this into `_transform`.
        # Further, we could remove the two `transpose` calls below if one were
        # willing to accept a different ordering of the transformed output.
        self._perturbations = self._expanded_perturbations(X)
        # make space for n_p dimension, switch n_p with n after transform, and flatten.
        return self._transform(X.unsqueeze(-3)).transpose(-3, -2).flatten(-3, -2)

    @subset_transform
    def _transform(self, X: Tensor):
        p = self._perturbations
        Y = X * p if self.multiplicative else X + p
        if self.bounds is not None:
            return torch.maximum(torch.minimum(Y, self.bounds[1]), self.bounds[0])
        return Y

    @property
    def batch_shape(self):
        """Returns a shape tuple such that `subset_transform` pre-allocates
        a (b x n_p x n x d) - dim tensor, where `b` is the batch shape of the
        input `X` of the transform and `n_p` is the number of perturbations.
        NOTE: this function is dependent on calling `_expanded_perturbations(X)`
        because `n_p` is inaccessible otherwise if `perturbation_set` is a function.
        """
        return self._perturbations.shape[:-2]

    def _expanded_perturbations(self, X: Tensor) -> Tensor:
        p = self.perturbation_set
        if isinstance(p, Tensor):
            p = p.expand(X.shape[-2], *p.shape)  # p is batch_shape x n x n_p x d
        else:
            p = p(X) if self.indices is None else p(X[..., self.indices])
        return p.transpose(-3, -2)  # p is batch_shape x n_p x n x d


class NumericToCategoricalEncoding(InputTransform):
    """Transform categorical parameters from an integer/numeric representation
    to a vector based representation like one-hot encoding or a descriptor
    encoding.

    The vector encoding is inserted at the position of the categorical feature
    in the input tensor. This is demonstrated in the example below in which a
    categorical feature of cardinality 3 at position 1 in the original
    representation is one-hot encoded.

    Example:

        >>> import torch
        >>> from torch.nn.functional import one_hot
        >>> from functools import partial
        >>> from botorch.models.transforms.input import NumericToCategoricalEncoding
        >>> tf = NumericToCategoricalEncoding(
        ...     dim=3,
        ...     categorical_features={1: 3},
        ...     encoders={1: partial(one_hot, num_classes=3)},
        ... )
        >>> X = torch.tensor([[0.5, 2, 1.2], [1.1, 0, 0.8]])
        >>> tf.transform(X)
        tensor([[0.5000, 0.0000, 0.0000, 1.0000, 1.2000],
                [1.1000, 1.0000, 0.0000, 0.0000, 0.8000]])
    """

    def __init__(
        self,
        dim: int,
        categorical_features: dict[int, int],
        encoders: dict[int, Callable[[Tensor], Tensor]],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        r"""Initialize.

        Args:
            dim: The dimension of the numerically encoded input.
            categorical_features: A dictionary mapping the index of each
                categorical feature to its cardinality which has to be
                greater than 1. This assumes that categoricals
                are integer encoded.
            encoders: A dictionary mapping the index of each categorical feature to
                a callable that encodes the categorical feature into a vector
                representation.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

        self.encoders = encoders
        self.categorical_features = categorical_features

        if len(self.categorical_features) == 0:
            raise ValueError(
                "At least one categorical feature for encoding must be provided."
            )

        if (num_cat := len(self.categorical_features)) > dim:
            raise ValueError(
                f"The number of categorical features ({num_cat}) exceeds the "
                f"provided dimension ({dim})."
            )

        for idx, card in self.categorical_features.items():
            if card <= 1 or not isinstance(card, int):
                raise ValueError(
                    f"Categorical feature at index {idx} has cardinality {card}. "
                    f"All categorical features must be an integer and have cardinality "
                    "greater than 1."
                )

        # check that the encoders match the categorical features
        if (enc_keys := set(self.encoders)) != (
            cf_keys := set(self.categorical_features)
        ):
            raise ValueError(
                f"The keys of `encoders` ({enc_keys}) must match the keys of "
                f"of `categorical_features`  ({cf_keys})."
            )

        self.ordinal_idx = list(
            self.categorical_features.keys()
        )  # indices of categorical features before encoding

        self.numerical_idx = list(
            set(range(dim)) - set(self.ordinal_idx)
        )  # indices of numerical features before encoding

        self.new_numerical_idx = []  # indices of numerical features after encoding
        self.encoded_idx = []  # indices of categorical features after encoding

        offset = 0
        for idx in range(dim):
            if idx in self.numerical_idx:
                self.new_numerical_idx.append(idx + offset)
            else:
                card = self.categorical_features[idx]
                self.encoded_idx.append(
                    np.arange(
                        idx + offset, idx + offset + card
                    ).tolist()  # indices of categorical features after encoding
                )
                offset += card - 1  # adjust offset for next categorical feature

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the categorical inputs into a vector representation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d'`-dim tensor with
            `d' = d + sum(categorical_features.values())` in which the
            integer-encoded categoricals are transformed to a vector representation.
        """
        s = list(X.shape)
        s[-1] = len(self.numerical_idx) + len(np.concatenate(self.encoded_idx))
        X_encoded = torch.zeros(size=s, device=X.device, dtype=X.dtype)
        X_encoded[..., self.new_numerical_idx] = X[..., self.numerical_idx]
        for i, idx in enumerate(self.categorical_features.keys()):
            encoded_val = self.encoders[idx](X[..., idx].long()).to(X_encoded)
            X_encoded[..., self.encoded_idx[i]] = encoded_val
        return X_encoded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            type(self) is type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
            and (self.transform_on_fantasize == other.transform_on_fantasize)
            and self.categorical_features == other.categorical_features
        )


class OneHotToNumeric(InputTransform):
    r"""Transform categorical parameters from a one-hot to a numeric representation."""

    def __init__(
        self,
        dim: int,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        r"""Initialize.

        Args:
            dim: The dimension of the one-hot-encoded input.
            categorical_features: A dictionary mapping the starting index of each
                categorical feature to its cardinality. This assumes that categoricals
                are one-hot encoded.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        categorical_features = categorical_features or {}
        # sort by starting index
        self.categorical_features = OrderedDict(
            sorted(categorical_features.items(), key=lambda x: x[0])
        )
        if len(self.categorical_features) > 0:
            self.onehot_idx = [
                np.arange(start, start + card)
                for start, card in self.categorical_features.items()
            ]
            idx = np.concatenate(self.onehot_idx)

            if len(idx) != len(set(idx)):
                raise ValueError("Categorical features overlap.")
            if max(idx) >= dim:
                raise ValueError("Categorical features exceed the provided dimension.")
            self.numerical_idx = list(set(range(dim)) - set(idx))

            offset = 0
            self.ordinal_idx = []
            for start, card in self.categorical_features.items():
                self.ordinal_idx.append(start - offset)
                offset += card - 1

            reduced_dim = len(self.ordinal_idx) + len(self.numerical_idx)
            self.new_numerical_idx = list(
                set(range(reduced_dim)) - set(self.ordinal_idx)
            )

            self.numeric_dim = len(self.new_numerical_idx) + len(
                self.categorical_features
            )

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the categorical inputs into integer representation.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d'`-dim tensor of where the one-hot encoded
            categoricals are transformed to integer representation.
        """
        if len(self.categorical_features) > 0:
            X_numeric = X[..., : self.numeric_dim].clone()
            # copy the numerical dims over
            X_numeric[..., self.new_numerical_idx] = X[..., self.numerical_idx]
            for i in range(len(self.categorical_features)):
                X_numeric[..., self.ordinal_idx[i]] = X[..., self.onehot_idx[i]].argmax(
                    dim=-1
                )
            return X_numeric
        return X

    def untransform(self, X: Tensor) -> Tensor:
        r"""Transform the categoricals from integer representation to one-hot.

        Args:
            X: A `batch_shape x n x d'`-dim tensor of transformed inputs, where
                the categoricals are represented as integers.

        Returns:
            A `batch_shape x n x d`-dim tensor of inputs, where the categoricals
            have been transformed to one-hot representation.
        """
        if len(self.categorical_features) > 0:
            s = list(X.shape)
            s[-1] = len(self.numerical_idx) + len(np.concatenate(self.onehot_idx))
            X_onehot = torch.zeros(size=s).to(X)
            X_onehot[..., self.numerical_idx] = X[..., self.new_numerical_idx]
            for i in range(len(self.categorical_features)):
                X_onehot[..., self.onehot_idx[i]] = one_hot(
                    X[..., self.ordinal_idx[i]].long(),
                    num_classes=len(self.onehot_idx[i]),
                ).to(X_onehot)
            return X_onehot
        return X

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            type(self) is type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
            and (self.transform_on_fantasize == other.transform_on_fantasize)
            and self.categorical_features == other.categorical_features
        )


class AbstractProbabilisticReparameterizationInputTransform(InputTransform, ABC):
    r"""An abstract input transform to prepare inputs for PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete values are created.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__()
        if integer_indices is None and categorical_features is None:
            raise ValueError(
                "integer_indices and/or categorical_features must be provided."
            )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        discrete_indices = []
        self.categorical_features = categorical_features
        categorical_start_idx = (
            one_hot_bounds.shape[1]
            if self.categorical_features is None
            else min(self.categorical_features.keys())
        )

        if integer_indices is not None and len(integer_indices) > 0:
            error_msg = (
                f"{self.__class__.__name__} requires that the integer "
                "parameters are to the right of continuous parameters, and the "
                "left of categorical parameters."
            )
            leftmost_int, rightmost_int = min(integer_indices), max(integer_indices)
            if categorical_start_idx - rightmost_int != 1:
                # continuous parameters between ints and cats, or overlapping indices
                raise ValueError(error_msg)
            if set(range(leftmost_int, rightmost_int + 1)) != set(integer_indices):
                # non-contiguous integer indices
                raise ValueError(error_msg)
            self.register_buffer(
                "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
            )
            discrete_indices += integer_indices
        else:
            self.integer_indices = None

        if self.categorical_features is not None:
            # check that the trailing dimensions are categoricals
            end = categorical_start_idx
            err_msg = (
                f"{self.__class__.__name__} requires that the categorical "
                "parameters are the rightmost elements."
            )
            for start, card in self.categorical_features.items():
                # the end of one one-hot representation should be followed
                # by the start of the next
                if end != start:
                    raise ValueError(err_msg)
                end = start + card
            if end != one_hot_bounds.shape[1]:
                # check end
                raise ValueError(err_msg)
        categorical_starts = []
        categorical_ends = []
        if self.categorical_features is not None:
            start = None
            for i, n_categories in categorical_features.items():
                if start is None:
                    start = i
                end = start + n_categories
                categorical_starts.append(start)
                categorical_ends.append(end)
                discrete_indices += list(range(start, end))
                start = end
        self.register_buffer(
            "discrete_indices",
            torch.tensor(
                discrete_indices, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        self.register_buffer(
            "categorical_starts",
            torch.tensor(
                categorical_starts, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        self.register_buffer(
            "categorical_ends",
            torch.tensor(
                categorical_ends, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        if integer_indices is None:
            self.register_buffer(
                "integer_bounds",
                torch.tensor([], dtype=torch.long, device=one_hot_bounds.device),
            )
        else:
            self.register_buffer("integer_bounds", one_hot_bounds[:, integer_indices])
        self.tau = tau

    def get_rounding_prob(self, X: Tensor) -> Tensor:
        X_prob = X.detach().clone()
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X_prob[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            if self.tau is not None:
                X_prob[..., self.integer_indices] = torch.sigmoid(
                    (X_int_abs - offset - 0.5) / self.tau
                )
            else:
                X_prob[..., self.integer_indices] = X_int_abs - offset
        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X_prob[..., start:end]
            if self.tau is not None:
                X_prob[..., start:end] = torch.softmax(
                    (X_categ - 0.5) / self.tau, dim=-1
                )
            else:
                X_prob[..., start:end] = X_categ / X_categ.sum(dim=-1)
        return X_prob[..., self.discrete_indices]

    def _check_input_shape(self, X: Tensor) -> None:
        r"""Check that the input shape is valid for this transform."""
        if X.shape[-3] > 1:
            raise ValueError(
                f"Input to {self.__class__.__name__} must have a dimension of size 1 "
                f"at index -3 (got shape {X.shape})."
            )
        if X.shape[-2] > 1:
            raise ValueError(
                f"Input transform {self.__class__.__name__} does not support `n` > 1 "
                f"(got shape {X.shape})."
            )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and torch.equal(self.integer_indices, other.integer_indices)
            and self.tau == other.tau
        )


class AnalyticProbabilisticReparameterizationInputTransform(
    AbstractProbabilisticReparameterizationInputTransform
):
    r"""An input transform to prepare inputs for analytic PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete values are created.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor = None,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            tau=tau,
        )
        # create cartesian product of discrete options
        discrete_options = []
        dim = one_hot_bounds.shape[1]
        # get number of discrete parameters
        num_discrete_params = 0
        if self.integer_indices is not None:
            num_discrete_params += self.integer_indices.shape[0]
        if self.categorical_features is not None:
            num_discrete_params += len(self.categorical_features)
        # add zeros for continuous params to simplify code
        for _ in range(dim - len(self.discrete_indices)):
            discrete_options.append(
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=one_hot_bounds.device,
                )
            )
        if integer_indices is not None:
            # FIXME: this assumes that the integer dimensions are after the continuous
            # if we want to enforce this, we should test for it similarly to
            # categoricals
            for i in range(self.integer_bounds.shape[-1]):
                discrete_options.append(
                    torch.arange(
                        self.integer_bounds[0, i],
                        self.integer_bounds[1, i] + 1,
                        dtype=torch.long,
                        device=one_hot_bounds.device,
                    )
                )
        categorical_start_idx = len(discrete_options)
        if categorical_features is not None:
            for idx in sorted(categorical_features.keys()):
                cardinality = categorical_features[idx]
                discrete_options.append(
                    torch.arange(
                        cardinality, dtype=torch.long, device=one_hot_bounds.device
                    )
                )
        # categoricals are in numeric representation
        all_discrete_options = torch.cartesian_prod(*discrete_options)
        # one-hot encode the categoricals
        if categorical_features is not None and len(categorical_features) > 0:
            X_categ = torch.empty(
                *all_discrete_options.shape[:-1], sum(categorical_features.values())
            )
            start = 0
            for i, (idx, cardinality) in enumerate(
                sorted(categorical_features.items(), key=lambda kv: kv[0])
            ):
                start = idx - categorical_start_idx
                X_categ[..., start : start + cardinality] = one_hot(
                    all_discrete_options[..., -len(categorical_features) + i],
                    num_classes=cardinality,
                ).to(X_categ)
            all_discrete_options = torch.cat(
                [all_discrete_options[..., : -len(categorical_features)], X_categ],
                dim=-1,
            )
        self.register_buffer("all_discrete_options", all_discrete_options)

    def get_probs(self, X: Tensor) -> Tensor:
        """
        Args:
            X: a `batch_shape x n x d`-dim tensor

        Returns:
            A `batch_shape x n_discrete x n`-dim tensors of probabilities of each
                discrete config under X.
        """
        # note this method should be differentiable
        X_prob = torch.ones(
            *X.shape[:-2],
            self.all_discrete_options.shape[0],
            X.shape[-2],
            dtype=X.dtype,
            device=X.device,
        )
        # n_discrete x batch_shape x n x d
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 2)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-2], self.all_discrete_options.shape[0], *X.shape[-2:])
        X = X.unsqueeze(-3)
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            # note we don't actually need the sigmoid here
            X_prob_int = torch.sigmoid((X_int_abs - offset - 0.5) / self.tau)
            # X_prob_int = X_int_abs - offset
            for int_idx, idx in enumerate(self.integer_indices):
                offset_i = offset[..., int_idx]
                all_discrete_i = all_discrete_options[..., idx]
                diff = (offset_i + 1) - all_discrete_i
                round_up_mask = diff == 0
                round_down_mask = diff == 1
                neither_mask = ~(round_up_mask | round_down_mask)
                prob = X_prob_int[..., int_idx].expand(round_up_mask.shape)
                # need to be careful with in-place ops here for autograd
                X_prob[round_up_mask] = X_prob[round_up_mask] * prob[round_up_mask]
                X_prob[round_down_mask] = X_prob[round_down_mask] * (
                    1 - prob[round_down_mask]
                )
                X_prob[neither_mask] = X_prob[neither_mask] * 0

        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X[..., start:end]
            X_prob_c = torch.softmax((X_categ - 0.5) / self.tau, dim=-1).expand(
                *X_categ.shape[:-3], all_discrete_options.shape[-3], *X_categ.shape[-2:]
            )
            for i in range(X_prob_c.shape[-1]):
                mask = all_discrete_options[..., start + i] == 1
                X_prob[mask] = X_prob[mask] * X_prob_c[..., i][mask]

        return X_prob

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x 1 x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x num_discrete_options x n x d`-dim tensor of rounded inputs.
        """
        self._check_input_shape(X)

        n_discrete = self.discrete_indices.shape[0]
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 3)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:])
        if X.shape[-1] > n_discrete:
            X = X.expand(
                *X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:]
            )
            return torch.cat(
                [X[..., :-n_discrete], all_discrete_options[..., -n_discrete:]], dim=-1
            )
        return all_discrete_options


class MCProbabilisticReparameterizationInputTransform(
    AbstractProbabilisticReparameterizationInputTransform
):
    r"""An input transform to prepare inputs for Monte Carlo PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete ordinal values are sampled.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            tau=tau,
        )
        self.mc_samples = mc_samples
        self.resample = resample

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x 1 x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x mc_samples x n x d`-dim tensor of rounded inputs.
        """
        self._check_input_shape(X)

        X_expanded = X.expand(*X.shape[:-3], self.mc_samples, *X.shape[-2:]).clone()
        X_prob = self.get_rounding_prob(X=X)
        if self.integer_indices is not None:
            X_int = X[..., self.integer_indices].detach()
            assert X.ndim > 1
            if X.ndim == 2:
                X.unsqueeze(-1)
            if (
                not hasattr(self, "base_samples")
                or self.base_samples.shape[-2:] != X_int.shape[-2:]
                or self.resample
            ):
                # construct sobol base samples
                bounds = torch.zeros(
                    2, X_int.shape[-1], dtype=X_int.dtype, device=X_int.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X_int.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )
            X_int_abs = X_int.abs()
            # perform exact rounding
            is_negative = X_int < 0
            offset = X_int_abs.floor()
            prob = X_prob[..., : self.integer_indices.shape[0]]
            rounding_component = (prob >= self.base_samples).to(
                dtype=X.dtype,
            )
            X_abs_rounded = offset + rounding_component
            X_int_new = (-1) ** is_negative.to(offset) * X_abs_rounded
            # clamp to bounds
            X_expanded[..., self.integer_indices] = torch.minimum(
                torch.maximum(X_int_new, self.integer_bounds[0]), self.integer_bounds[1]
            )

        # sample for categoricals
        if self.categorical_features is not None and len(self.categorical_features) > 0:
            if (
                not hasattr(self, "base_samples_categorical")
                or self.base_samples_categorical.shape[-2] != X.shape[-2]
                or self.resample
            ):
                bounds = torch.zeros(
                    2, len(self.categorical_features), dtype=X.dtype, device=X.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples_categorical",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )

            # sample from multinomial as argmin_c [sample_c * exp(-x_c)]
            sample_d_start_idx = 0
            X_categ_prob = X_prob
            if self.integer_indices is not None:
                n_ints = self.integer_indices.shape[0]
                if n_ints > 0:
                    X_categ_prob = X_prob[..., n_ints:]

            for i, cardinality in enumerate(self.categorical_features.values()):
                sample_d_end_idx = sample_d_start_idx + cardinality
                start = self.categorical_starts[i]
                end = self.categorical_ends[i]
                cum_prob = X_categ_prob[
                    ..., sample_d_start_idx:sample_d_end_idx
                ].cumsum(dim=-1)
                categories = (
                    (
                        (cum_prob > self.base_samples_categorical[..., i : i + 1])
                        .long()
                        .cumsum(dim=-1)
                        == 1
                    )
                    .long()
                    .argmax(dim=-1)
                )
                # one-hot encode
                X_expanded[..., start:end] = one_hot(
                    categories, num_classes=cardinality
                ).to(X)
                sample_d_start_idx = sample_d_end_idx

        return X_expanded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and (self.resample == other.resample)
            and torch.equal(self.base_samples, other.base_samples)
        )
