#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Quasi Monte-Carlo sampling from Normal distributions.

References:

.. [Pages2018numprob]
    G. Pages. Numerical Probability: An Introduction with Applications to
    Finance. Universitext. Springer International Publishing, 2018.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine


class NormalQMCEngine:
    r"""Engine for qMC sampling from a Multivariate Normal ``N(0, I_d)``.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set ``inv_transform=True``.

    Example:
        >>> engine = NormalQMCEngine(3)
        >>> samples = engine.draw(16)
    """

    def __init__(
        self, d: int, seed: int | None = None, inv_transform: bool = False
    ) -> None:
        r"""Engine for drawing qMC samples from a multivariate normal ``N(0, I_d)``.

        Args:
            d: The dimension of the samples.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        self._d = d
        self._seed = seed
        self._inv_transform = inv_transform
        if inv_transform:
            sobol_dim = d
        else:
            # to apply Box-Muller, we need an even number of dimensions
            sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True, seed=seed)

    def draw(
        self,
        n: int = 1,
        out: Tensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor | None:
        r"""Draw ``n`` qMC samples from the standard Normal.

        Args:
            n: The number of samples to draw. As a best practice, use powers of 2.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.
            dtype: The desired torch data type (ignored if ``out`` is provided).
                If None, uses ``torch.get_default_dtype()``.

        Returns:
            A ``n x d`` tensor of samples if ``out=None`` and ``None`` otherwise.
        """
        dtype = torch.get_default_dtype() if dtype is None else dtype
        # get base samples
        samples = self._sobol_engine.draw(n, dtype=dtype)
        if self._inv_transform:
            # apply inverse transform (values too close to 0/1 result in inf values)
            v = 0.5 + (1 - torch.finfo(samples.dtype).eps) * (samples - 0.5)
            samples_tf = torch.erfinv(2 * v - 1) * math.sqrt(2)
        else:
            # apply Box-Muller transform (note: [1] indexes starting from 1)
            even = torch.arange(0, samples.shape[-1], 2)
            Rs = (-2 * torch.log(samples[:, even])).sqrt()
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = torch.cos(thetas)
            sin = torch.sin(thetas)
            samples_tf = torch.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            samples_tf = samples_tf[:, : self._d]
        if out is None:
            return samples_tf
        else:
            out.copy_(samples_tf)


class MultivariateNormalQMCEngine:
    r"""Engine for qMC sampling from a multivariate Normal ``N(\mu, \Sigma)``.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set ``inv_transform=True``.

    Example:
        >>> mean = torch.tensor([1.0, 2.0])
        >>> cov = torch.tensor([[1.0, 0.25], [0.25, 2.0]])
        >>> engine = MultivariateNormalQMCEngine(mean, cov)
        >>> samples = engine.draw(16)
    """

    def __init__(
        self,
        mean: Tensor,
        cov: Tensor,
        seed: int | None = None,
        inv_transform: bool = False,
    ) -> None:
        r"""Engine for qMC sampling from a multivariate Normal ``N(\mu, \Sigma)``.

        Args:
            mean: The mean vector.
            cov: The covariance matrix.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        # validate inputs
        if not cov.shape[0] == cov.shape[1]:
            raise ValueError("Covariance matrix is not square.")
        if not mean.shape[0] == cov.shape[0]:
            raise ValueError("Dimension mismatch between mean and covariance.")
        if not torch.allclose(cov, cov.transpose(-1, -2)):
            raise ValueError("Covariance matrix is not symmetric.")
        self._mean = mean
        self._normal_engine = NormalQMCEngine(
            d=mean.shape[0], seed=seed, inv_transform=inv_transform
        )
        # compute Cholesky decomp; if it fails, do the eigendecomposition
        try:
            self._corr_matrix = torch.linalg.cholesky(cov).transpose(-1, -2)
        except RuntimeError:
            eigval, eigvec = torch.linalg.eigh(cov)
            tol = 1e-8 if eigval.dtype == torch.double else 1e-6
            if torch.any(eigval < -tol):
                raise ValueError("Covariance matrix not PSD.")
            eigval_root = eigval.clamp_min(0.0).sqrt()
            self._corr_matrix = (eigvec * eigval_root).transpose(-1, -2)

    def draw(self, n: int = 1, out: Tensor | None = None) -> Tensor | None:
        r"""Draw ``n`` qMC samples from the multivariate Normal.

        Args:
            n: The number of samples to draw. As a best practice, use powers of 2.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.

        Returns:
            A ``n x d`` tensor of samples if ``out=None`` and ``None`` otherwise.
        """
        dtype = out.dtype if out is not None else self._mean.dtype
        device = out.device if out is not None else self._mean.device
        base_samples = self._normal_engine.draw(n, dtype=dtype).to(device=device)
        corr_mat = self._corr_matrix.to(dtype=dtype, device=device)
        mean = self._mean.to(dtype=dtype, device=device)
        qmc_samples = base_samples @ corr_mat + mean
        if out is None:
            return qmc_samples
        else:
            out.copy_(qmc_samples)
