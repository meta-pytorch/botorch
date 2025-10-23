#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Deterministic Models: Simple wrappers that allow the usage of deterministic
mappings via the BoTorch Model and Posterior APIs.

Deterministic models are useful for expressing known input-output relationships
within the BoTorch Model API. This is useful e.g. for multi-objective
optimization with known objective functions (e.g. the number of parameters of a
Neural Network in the context of Neural Architecture Search is usually a known
function of the architecture configuration), or to encode cost functions for
cost-aware acquisition utilities. Cost-aware optimization is desirable when
evaluations have a cost that is heterogeneous, either in the inputs `X` or in a
particular fidelity parameter that directly encodes the fidelity of the
observation. `GenericDeterministicModel` supports arbitrary deterministic
functions, while `AffineFidelityCostModel` is a particular cost model for
multi-fidelity optimization. Other use cases of deterministic models include
representing approximate GP sample paths, e.g. Matheron paths obtained
with `get_matheron_path_model`, which allows them to be substituted in acquisition
functions or in other places where a `Model` is expected.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model, ModelList
from botorch.utils.transforms import is_ensemble
from torch import Size, Tensor


class DeterministicModel(EnsembleModel):
    """Abstract base class for deterministic models."""

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the (deterministic) model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x n x m`-dimensional output tensor (the outcome
            dimension `m` must be explicit if `m=1`).
        """
        pass  # pragma: no cover

    def _forward(self, X: Tensor) -> Tensor:
        r"""Compatibilizes the `DeterministicModel` with `EnsemblePosterior`"""
        return self.forward(X=X).unsqueeze(-3)


class GenericDeterministicModel(DeterministicModel):
    r"""A generic deterministic model constructed from a callable.

    Example:
        >>> f = lambda x: x.sum(dim=-1, keep_dims=True)
        >>> model = GenericDeterministicModel(f)
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        num_outputs: int = 1,
        batch_shape: torch.Size | None = None,
    ) -> None:
        r"""
        Args:
            f: A callable mapping a `batch_shape x n x d`-dim input tensor `X`
                to a `batch_shape x n x m`-dimensional output tensor (the
                outcome dimension `m` must be explicit, even if `m=1`).
            num_outputs: The number of outputs `m`.
        """
        super().__init__()
        self._f = f
        self._num_outputs = num_outputs
        self._batch_shape = batch_shape

    @property
    def batch_shape(self) -> torch.Size | None:
        r"""The batch shape of the model."""
        return self._batch_shape

    def subset_output(self, idcs: list[int]) -> GenericDeterministicModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """

        def f_subset(X: Tensor) -> Tensor:
            return self._f(X)[..., idcs]

        return self.__class__(f=f_subset, num_outputs=len(idcs))

    def forward(self, X: Tensor) -> Tensor:
        r"""Compute the (deterministic) model output at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `batch_shape x n x m`-dimensional output tensor.
        """
        Y = self._f(X)
        batch_shape = Y.shape[:-2]
        # allowing for old behavior of not specifying the batch_shape
        if self.batch_shape is not None:
            try:
                torch.broadcast_shapes(self.batch_shape, batch_shape)
            except RuntimeError:
                raise ValueError(
                    "GenericDeterministicModel was initialized with batch_shape="
                    f"{self.batch_shape=} but the output of f has a batch_shape="
                    f"{batch_shape=} that is not broadcastable with it."
                )
        return Y


class AffineDeterministicModel(DeterministicModel):
    r"""An affine deterministic model."""

    def __init__(self, a: Tensor, b: Tensor | float = 0.01) -> None:
        r"""Affine deterministic model from weights and offset terms.

        A simple model of the form

            y[..., m] = b[m] + sum_{i=1}^d a[i, m] * X[..., i]

        Args:
            a: A `d x m`-dim tensor of linear weights, where `m` is the number
                of outputs (must be explicit if `m=1`)
            b: The affine (offset) term. Either a float (for single-output
                models or if the offset is shared), or a `m`-dim tensor (with
                different offset values for for the `m` different outputs).
        """
        if not a.ndim == 2:
            raise ValueError("a must be two-dimensional")
        if not torch.is_tensor(b):
            b = torch.tensor([b])
        if not b.ndim == 1:
            raise ValueError("b nust be one-dimensional")
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b.expand(a.size(-1)))
        self._num_outputs = a.size(-1)

    def subset_output(self, idcs: list[int]) -> AffineDeterministicModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        a_sub = self.a.detach()[..., idcs].clone()
        b_sub = self.b.detach()[..., idcs].clone()
        return self.__class__(a=a_sub, b=b_sub)

    def forward(self, X: Tensor) -> Tensor:
        return self.b + torch.einsum("...d,dm", X, self.a)


class PosteriorMeanModel(DeterministicModel):
    """A deterministic model that always returns the posterior mean."""

    def __init__(self, model: Model) -> None:
        r"""
        Args:
            model: The base model.
        """
        super().__init__()
        self.model = model

    def forward(self, X: Tensor) -> Tensor:
        return self.model.posterior(X).mean

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self.model.num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model."""
        return self.model.batch_shape


class FixedSingleSampleModel(DeterministicModel):
    r"""
    A deterministic model defined by a single sample `w`.

    Given a base model `f` and a fixed sample `w`, the model always outputs

        y = f_mean(x) + f_stddev(x) * w

    We assume the outcomes are uncorrelated here.
    """

    def __init__(
        self,
        model: Model,
        w: Tensor | None = None,
        dim: int | None = None,
        jitter: float | None = 1e-8,
        dtype: torch.dtype | None = None,
        device: torch.dtype | None = None,
    ) -> None:
        r"""
        Args:
            model: The base model.
            w: A 1-d tensor with length model.num_outputs.
                If None, draw it from a standard normal distribution.
            dim: dimensionality of w.
                If None and w is not provided, draw w samples of size model.num_outputs.
            jitter: jitter value to be added for numerical stability, 1e-8 by default.
            dtype: dtype for w if specified
            device: device for w if specified
        """
        super().__init__()
        self.model = model
        self._num_outputs = model.num_outputs
        self.jitter = jitter
        if w is None:
            self.w = (
                torch.randn(model.num_outputs, dtype=dtype, device=device)
                if dim is None
                else torch.randn(dim, dtype=dtype, device=device)
            )
        else:
            self.w = w

    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        return post.mean + torch.sqrt(post.variance + self.jitter) * self.w.to(X)


class MatheronPathModel(DeterministicModel):
    r"""A deterministic model that returns a Matheron path sample.

    A Matheron path is a continuous sample path of a GP, obtained by drawing
    random Fourier features from a GP prior and a pathwise update rule based on
    the observed data.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> matheron_model = MatheronPathModel(model)
        >>> output = matheron_model(test_X)
    """

    def __init__(
        self,
        model: Model,
        sample_shape: Size | None = None,
        ensemble_as_batch: bool = False,
        seed: int | None = None,
    ) -> None:
        r"""
        Args:
            model: The base model.
            sample_shape: The shape of the sample paths to be drawn, if an ensemble
                of sample paths is desired. If this is specified, the resulting
                deterministic model will behave as if the `sample_shape` is
                prepended to the model's `batch_shape`.
            ensemble_as_batch: If True, and model is an ensemble model, the resulting
                model will treat the ensemble dimension as a batch dimension.
            seed: Random seed for reproducible path generation. If None,
                no specific seed is set.
        """
        super().__init__()
        self.model = model

        # Validate model compatibility
        if isinstance(model, ModelList) and len(model.models) != model.num_outputs:
            raise UnsupportedError(
                "A model-list of multi-output models is not supported."
            )

        # Initialize path generation parameters
        self.sample_shape = Size() if sample_shape is None else sample_shape
        self.ensemble_as_batch = ensemble_as_batch

        # NOTE circular import in pathwise/utils.py otherwise
        from botorch.sampling.pathwise import draw_matheron_paths

        # Generate the Matheron path once during initialization
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._path = draw_matheron_paths(
                    model,
                    sample_shape=self.sample_shape,
                )
        else:
            self._path = draw_matheron_paths(model, sample_shape=self.sample_shape)
        self._path.set_ensemble_as_batch(ensemble_as_batch)
        self._is_ensemble = is_ensemble(model) or len(self.sample_shape) > 0

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Matheron path at X.

        Args:
            X: A `batch_shape x n x d`-dim input tensor `X`.

        Returns:
            A `[sample_shape x] batch_shape x n x m`-dimensional output tensor.
        """
        if self.model.num_outputs == 1:
            # For single-output, add the output dimension
            return self._path(X).unsqueeze(-1)
        elif isinstance(self.model, ModelList):
            # For model list, stack the path outputs
            return torch.stack(self._path(X), dim=-1)
        else:
            # For multi-output models
            return self._path(X.unsqueeze(-3)).transpose(-1, -2)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self.model.num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model."""
        return self.sample_shape + self.model.batch_shape
