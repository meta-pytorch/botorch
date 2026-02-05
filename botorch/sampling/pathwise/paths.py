#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from botorch.exceptions.errors import UnsupportedError
from botorch.sampling.pathwise.features import FeatureMap
from botorch.sampling.pathwise.utils import (
    ModuleDictMixin,
    ModuleListMixin,
    TInputTransform,
    TOutputTransform,
    TransformedModuleMixin,
)
from torch import Tensor
from torch.nn import Module, Parameter


class SamplePath(ABC, TransformedModuleMixin, Module):
    r"""Abstract base class for Botorch sample paths."""

    @abstractmethod
    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is considered as a batch dimension.

        Args:
            ensemble_as_batch: Whether the ensemble dimension is considered as a batch
                dimension or not.
        """
        pass  # pragma: no cover


class PathDict(SamplePath, ModuleDictMixin[SamplePath]):
    r"""A dictionary of SamplePaths."""

    def __init__(
        self,
        paths: Mapping[str, SamplePath] | None = None,
        reducer: Callable[[list[Tensor]], Tensor] | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a PathDict instance.

        Args:
            paths: An optional mapping of strings to sample paths.
            reducer: An optional callable used to combine each path's outputs.
                Must be provided if output_transform is specified.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
                Can only be specified if reducer is provided.
        """
        if reducer is None and output_transform is not None:
            raise UnsupportedError(
                "`output_transform` must be preceded by a `reducer`."
            )

        SamplePath.__init__(self)
        ModuleDictMixin.__init__(self, attr_name="paths", modules=paths)
        self.reducer = reducer
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor | dict[str, Tensor]:
        outputs = [path(x, **kwargs) for path in self.values()]
        return (
            dict(zip(self, outputs)) if self.reducer is None else self.reducer(outputs)
        )

    @property
    def paths(self):
        """Access the internal module dict."""
        return self._paths_dict

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is considered as a batch dimension.

        Args:
            ensemble_as_batch: Whether the ensemble dimension is considered as a batch
                dimension or not.
        """
        for path in self.paths.values():
            path.set_ensemble_as_batch(ensemble_as_batch)


class PathList(SamplePath, ModuleListMixin[SamplePath]):
    r"""A list of SamplePaths."""

    def __init__(
        self,
        paths: Iterable[SamplePath] | None = None,
        reducer: Callable[[list[Tensor]], Tensor] | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
    ) -> None:
        r"""Initializes a PathList instance.

        Args:
            paths: An optional iterable of sample paths.
            reducer: An optional callable used to combine each path's outputs.
                Must be provided if output_transform is specified.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
                Can only be specified if reducer is provided.
        """
        if reducer is None and output_transform is not None:
            raise UnsupportedError(
                "`output_transform` must be preceded by a `reducer`."
            )

        SamplePath.__init__(self)
        ModuleListMixin.__init__(self, attr_name="paths", modules=paths)
        self.reducer = reducer
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor | list[Tensor]:
        outputs = [path(x, **kwargs) for path in self]
        return outputs if self.reducer is None else self.reducer(outputs)

    @property
    def paths(self):
        """Access the internal module list."""
        return self._paths_list

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is considered as a batch dimension.

        Args:
            ensemble_as_batch: Whether the ensemble dimension is considered as a batch
                dimension or not.
        """
        for path in self.paths:
            path.set_ensemble_as_batch(ensemble_as_batch)


class GeneralizedLinearPath(SamplePath):
    r"""A sample path in the form of a generalized linear model."""

    def __init__(
        self,
        feature_map: FeatureMap,
        weight: Parameter | Tensor,
        bias_module: Module | None = None,
        input_transform: TInputTransform | None = None,
        output_transform: TOutputTransform | None = None,
        is_ensemble: bool = False,
        ensemble_as_batch: bool = False,
    ):
        r"""Initializes a GeneralizedLinearPath instance.

        .. code-block:: text

            path(x) = output_transform(bias_module(z) + feature_map(z)^T weight),
            where z = input_transform(x).

        Args:
            feature_map: A map used to featurize the module's inputs.
            weight: A tensor of weights used to combine input features. When
                generated with ``draw_kernel_feature_paths``, ``weight`` is a Tensor
                with the shape ``sample_shape x batch_shape x num_outputs``.
            bias_module: An optional module used to define additive offsets.
            input_transform: An optional input transform for the module.
            output_transform: An optional output transform for the module.
            is_ensemble: Whether the associated model is an ensemble model or not.
            ensemble_as_batch: Whether the ensemble dimension is added as a batch
                dimension or not. If ``True``, the ensemble dimension is treated as a
                batch dimension, which allows for the joint optimization of all members
                of the ensemble.
        """
        super().__init__()
        self.feature_map = feature_map
        # Register weight as buffer if not a Parameter
        if not isinstance(weight, Parameter):
            self.register_buffer("weight", weight)
        self.weight = weight
        self.bias_module = bias_module
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.is_ensemble = is_ensemble
        self.ensemble_as_batch = ensemble_as_batch

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Evaluates the path.

        Args:
            x: The input tensor of shape ``batch_shape x [num_ensemble x] q x d``,
                where ``num_ensemble`` is the number of ensemble members and is
                required to *only* be included if ``is_ensemble=True`` and
                ``ensemble_as_batch=True``.
            kwargs: Additional keyword arguments passed to the feature map.

        Returns:
            A tensor of shape ``batch_shape x [num_ensemble x] q x m``, where
                ``m`` is the number of outputs, where ``num_ensemble`` is only
                included if ``is_ensemble`` is ``True``, and regardless of whether
                ``ensemble_as_batch`` is ``True`` or not.
        """
        if self.is_ensemble and not self.ensemble_as_batch:
            # assuming that the ensembling dimension is added after (n, d), but
            # before the other batch dimensions, starting from the left.
            x = x.unsqueeze(-3)
        features = self.feature_map(x, **kwargs)
        output = (features @ self.weight.unsqueeze(-1)).squeeze(-1)
        ndim = len(self.feature_map.output_shape)
        if ndim > 1:  # sum over the remaining feature dimensions
            output = output.sum(dim=list(range(-ndim + 1, 0)))

        return output if self.bias_module is None else output + self.bias_module(x)

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is considered as a batch dimension.

        Args:
            ensemble_as_batch: Whether the ensemble dimension is considered as a batch
                dimension or not.
        """
        self.ensemble_as_batch = ensemble_as_batch
