#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A Posterior represented by a list of independent Posteriors.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior, MCMC_DIM
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class PosteriorList(Posterior):
    r"""A Posterior represented by a list of independent Posteriors.

    When at least one of the posteriors is a ``GaussianMixturePosterior``, the other
    posteriors are expanded to match the size of the ``GaussianMixturePosterior``.
    """

    def __init__(self, *posteriors: Posterior) -> None:
        r"""A Posterior represented by a list of independent Posteriors.

        Args:
            *posteriors: A variable number of single-outcome posteriors.

        Example:
            >>> p_1 = model_1.posterior(test_X)
            >>> p_2 = model_2.posterior(test_X)
            >>> p_12 = PosteriorList(p_1, p_2)

        Note: This is typically produced automatically in ``ModelList``; it should
        generally not be necessary for the end user to invoke it manually.
        """
        self.posteriors = list(posteriors)

    @cached_property
    def _is_gaussian_mixture(self) -> bool:
        r"""Check if any of the posteriors is a ``GaussianMixturePosterior``."""
        return any(isinstance(p, GaussianMixturePosterior) for p in self.posteriors)

    def _get_mcmc_batch_dimension(self) -> int:
        """Return the number of MCMC samples in the corresponding batch dimension."""
        mcmc_samples = [
            p.mean.shape[MCMC_DIM]
            for p in self.posteriors
            if isinstance(p, GaussianMixturePosterior)
        ]
        if len(set(mcmc_samples)) > 1:
            raise NotImplementedError(
                "All MCMC batch dimensions must have the same size, got shapes: "
                f"{mcmc_samples}."
            )
        return mcmc_samples[0]

    @staticmethod
    def _reshape_tensor(X: Tensor, mcmc_samples: int) -> Tensor:
        """Reshape a tensor without an MCMC batch dimension to match the shape."""
        X = X.unsqueeze(MCMC_DIM)
        return X.expand(*X.shape[:MCMC_DIM], mcmc_samples, *X.shape[MCMC_DIM + 1 :])

    def _reshape_and_cat(self, tensors: list[Tensor]):
        r"""Reshape, if needed, and concatenate (across dim=-1) a list of tensors."""
        if self._is_gaussian_mixture:
            mcmc_samples = self._get_mcmc_batch_dimension()
            return torch.cat(
                [
                    (
                        x
                        if isinstance(p, GaussianMixturePosterior)
                        else self._reshape_tensor(x, mcmc_samples=mcmc_samples)
                    )
                    for x, p in zip(tensors, self.posteriors)
                ],
                dim=-1,
            )
        else:
            return torch.cat(tensors, dim=-1)

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        devices = {p.device for p in self.posteriors}
        if len(devices) > 1:
            raise NotImplementedError(  # pragma: no cover
                "Multi-device posteriors are currently not supported. "
                "The devices of the constituent posteriors are: {devices}."
            )
        return next(iter(devices))

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        dtypes = {p.dtype for p in self.posteriors}
        if len(dtypes) > 1:
            raise NotImplementedError(
                "Multi-dtype posteriors are currently not supported. "
                "The dtypes of the constituent posteriors are: {dtypes}."
            )
        return next(iter(dtypes))

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given ``sample_shape``.

        If there's at least one ``GaussianMixturePosterior``, the MCMC dimension
        is included the ``_extended_shape``.
        """
        if self._is_gaussian_mixture:
            mcmc_shape = torch.Size([self._get_mcmc_batch_dimension()])
            extend_dim = MCMC_DIM + 1  # The dimension to inject MCMC shape.
        extended_shapes = []
        for p in self.posteriors:
            es = p._extended_shape(sample_shape=sample_shape)
            if self._is_gaussian_mixture and not isinstance(
                p, GaussianMixturePosterior
            ):
                # Extend the shapes of non-fully Bayesian ones to match.
                extended_shapes.append(es[:extend_dim] + mcmc_shape + es[extend_dim:])
            else:
                extended_shapes.append(es)
        batch_shapes = [es[:-1] for es in extended_shapes]
        if len(set(batch_shapes)) > 1:
            raise NotImplementedError(
                "`PosteriorList` is only supported if the constituent posteriors "
                f"all have the same `batch_shape`. Batch shapes: {batch_shapes}."
            )
        # Last dimension is the output dimension (concatenation dimension).
        return batch_shapes[0] + torch.Size([sum(es[-1] for es in extended_shapes)])

    @property
    def mean(self) -> Tensor:
        r"""The mean of the posterior as a ``(b) x n x m``-dim Tensor.

        This is only supported if all posteriors provide a mean.
        """
        return self._reshape_and_cat(tensors=[p.mean for p in self.posteriors])

    @property
    def variance(self) -> Tensor:
        r"""The variance of the posterior as a ``(b) x n x m``-dim Tensor.

        This is only supported if all posteriors provide a variance.
        """
        return self._reshape_and_cat(tensors=[p.variance for p in self.posteriors])

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A ``torch.Size`` object specifying the sample shape. To
                draw ``n`` samples, set to ``torch.Size([n])``. To draw ``b`` batches
                of ``n`` samples each, set to ``torch.Size([b, n])``.

        Returns:
            Samples from the posterior, a tensor of shape
            ``self._extended_shape(sample_shape=sample_shape)``.
        """
        samples = [p.rsample(sample_shape=sample_shape) for p in self.posteriors]
        return self._reshape_and_cat(tensors=samples)

    def __getattr__(self, name: str) -> Any:
        r"""A catch-all for attributes not defined on the posterior level.

        Raises an attribute error.
        """
        raise AttributeError(
            f"`PosteriorList` does not define the attribute {name}. "
            "Consider accessing the attributes of the individual posteriors instead."
        )
