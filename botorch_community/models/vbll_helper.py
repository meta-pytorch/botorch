#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The following code is from the repository vbll (https://github.com/VectorInstitute/vbll)
which is under the MIT license.
Paper: "Variational Bayesian Last Layers" by Harrison et al., ICLR 2024
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from botorch.logging import logger
from torch import Tensor


def tp(M):
    return M.transpose(-1, -2)


class Normal(torch.distributions.Normal):
    def __init__(self, loc: Tensor, chol: Tensor):
        """Normal distribution.

        Args:
            loc (_type_): _description_
            chol (_type_): _description_
        """
        super().__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def precision(self):
        return torch.diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(
                self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min=1e-12))
            )
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError(
                "Distribution addition only implemented for diag covs"
            )

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc: Tensor, cholesky: Tensor):
        """Dense Normal distribution. Note that this is a multivariate normal used for
        the distribution of the weights in the last layer of a neural network.

        Args:
            loc: Location of the distribution.
            cholesky: Lower triangular Cholesky factor of the covariance matrix.
        """
        super().__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self):
        logger.warning(
            "Direct matrix inverse for dense covariances is O(N^3),"
            "consider using eg inverse weighted inner product"
        )
        Eye = torch.eye(
            self.scale_tril.shape[-1],
            device=self.scale_tril.device,
            dtype=self.scale_tril.dtype,
        )
        W = torch.linalg.solve_triangular(self.scale_tril, Eye, upper=False)
        return tp(W) @ W

    @property
    def logdet_covariance(self):
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (self.scale_tril**2).sum(-1).sum(-1)  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (
            torch.linalg.solve_triangular(self.scale_tril, b, upper=False) ** 2
        ).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    def __init__(self, loc: Tensor, cov_factor: Tensor, diag: Tensor):
        """Low Rank Normal distribution. Note that this is a multivariate normal used
        for the distribution of the weights in the last layer of a neural network.

        Args:
            loc: Location of the distribution.
            cov_factor: Low rank factor of the covariance matrix.
            diag: Diagonal of the covariance matrix.
        """
        super().__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def inverse_covariance(self):
        raise NotImplementedError()

    @property
    def logdet_covariance(self):
        # Apply Matrix determinant lemma
        term1 = torch.log(self.cov_diag).sum(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor / self.cov_diag.unsqueeze(-1))
        term2 = torch.linalg.det(
            arg1 + torch.eye(arg1.shape[-1], dtype=torch.float64)
        ).log()
        return term1 + term2

    @property
    def trace_covariance(self):
        # trace of sum is sum of traces
        trace_diag = self.cov_diag.sum(-1)
        trace_lowrank = (self.cov_factor**2).sum(-1).sum(-1)
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        diag_term = (self.cov_diag.unsqueeze(-1) * (b**2)).sum(-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return LowRankNormal(
            self.loc.squeeze(idx),
            self.cov_factor.squeeze(idx),
            self.cov_diag.squeeze(idx),
        )


class DenseNormalPrec(torch.distributions.MultivariateNormal):
    """A DenseNormal parameterized by the mean and the cholesky decomp of the precision
    matrix. Low Rank Normal distribution. Note that this is a multivariate normal used
    for the distribution of the weights in the last layer of a neural network.

    This function also includes a recursive_update function which performs a recursive
    linear regression update with effecient cholesky factor updates.
    """

    def __init__(self, loc: Tensor, cholesky: Tensor, validate_args=False):
        """A DenseNormal parameterized by the mean and the cholesky decomp of the
        precision.

        Args:
            loc: Location of the distribution.
            cholesky: Lower triangular Cholesky factor of the precision matrix.
            validate_args: Whether to validate the input arguments.
        """
        prec = cholesky @ tp(cholesky)
        super().__init__(loc, precision_matrix=prec, validate_args=validate_args)
        self.tril = cholesky

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        logger.warning(
            "Direct matrix inverse for dense covariances is O(N^3)"
            "consider using eg inverse weighted inner product"
        )
        return torch.cholesky_inverse(self.tril)

    @property
    def inverse_covariance(self):
        return self.precision_matrix

    @property
    def logdet_covariance(self):
        return -2.0 * torch.diagonal(self.tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        return (
            (torch.inverse(self.tril) ** 2).sum(-1).sum(-1)
        )  # compute as frob norm squared

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormalPrec(self.loc.squeeze(idx), self.tril.squeeze(idx))


def get_parameterization(p):
    COV_PARAM_DICT = {
        "dense": DenseNormal,
        "dense_precision": DenseNormalPrec,
        "diagonal": Normal,
        "lowrank": LowRankNormal,
    }

    try:
        return COV_PARAM_DICT[p]
    except KeyError:
        raise ValueError(f"Invalid covariance parameterization: {p!r}")


# following functions/classes are from
# https://github.com/VectorInstitute/vbll/blob/main/vbll/layers/regression.py
def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean**2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)
    return 0.5 * (mse_term + trace_term + logdet_term)  # currently exclude constant


@dataclass
class VBLLReturn:
    predictive: Normal | DenseNormal
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]


class Regression(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        regularization_weight,
        parameterization="dense",
        mean_initialization=None,
        prior_scale=1.0,
        wishart_scale=1e-2,
        cov_rank=None,
        clamp_noise_init=True,
        dof=1.0,
    ):
        """
        Variational Bayesian Linear Regression

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        regularization_weight : float
            Weight on regularization term in ELBO
        parameterization : str
            Parameterization of covariance matrix.
            Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
        mean_initialization : str or None
            Initialization method for the mean of the weights.
            Supports {'kaiming', None}. If None, weights are initialized from
            a standard normal distribution. Defaults to None.
        prior_scale : float
            Scale of prior covariance matrix
        wishart_scale : float
            Scale of Wishart prior on noise covariance
        cov_rank : int or None
            For 'lowrank' parameterization, the rank of the covariance matrix.
        clamp_noise_init : bool
            Whether to clamp the noise initialization to be positive.
        dof : float
            Degrees of freedom of Wishart prior on noise covariance
        """
        super().__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.0) / 2.0
        self.regularization_weight = regularization_weight
        self.dtype = torch.float64  # NOTE: not in the original source code

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale * (1.0 / in_features)

        # noise distribution
        self.noise_mean = nn.Parameter(
            torch.zeros(out_features, dtype=self.dtype), requires_grad=False
        )
        self.noise_logdiag = nn.Parameter(
            torch.randn(out_features, dtype=self.dtype) * (np.log(wishart_scale))
        )

        # ensure that log noise is positive
        if clamp_noise_init:
            self.noise_logdiag.data = torch.clamp(self.noise_logdiag.data, min=0)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)

        if mean_initialization is None:
            self.W_mean = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
            )
        elif mean_initialization == "kaiming":
            self.W_mean = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                * np.sqrt(2.0 / in_features)
            )
        elif isinstance(mean_initialization, str):
            raise ValueError(
                f"Unknown initialization method: {mean_initialization!r}. "
                f"Supported methods: 'kaiming'"
            )
        else:
            raise TypeError(
                f"mean_initialization must be a string or None, "
                f"got {type(mean_initialization).__name__}"
            )

        if parameterization == "diagonal":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
        elif parameterization == "dense":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features, dtype=self.dtype)
                / in_features
            )
        elif parameterization == "dense_precision":
            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                + 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, in_features, dtype=self.dtype)
                * 0.0
            )
        elif parameterization == "lowrank":
            if cov_rank is None:
                raise ValueError("Must specify cov_rank for lowrank parameterization")

            self.W_logdiag = nn.Parameter(
                torch.randn(out_features, in_features, dtype=self.dtype)
                - 0.5 * np.log(in_features)
            )
            self.W_offdiag = nn.Parameter(
                torch.randn(out_features, in_features, cov_rank, dtype=self.dtype)
                / in_features
            )

    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def forward(self, x):
        out = VBLLReturn(
            self.predictive(x), self._get_train_loss_fn(x), self._get_val_loss_fn(x)
        )
        return out

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):
        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[..., None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5 * (
                (W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None]))
                * noise.trace_precision
            )

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (
                self.dof * noise.logdet_precision
                - 0.5 * self.wishart_scale * noise.trace_precision
            )
            total_elbo = torch.mean(pred_likelihood - trace_term)
            regularization_term = self.regularization_weight * (wishart_term - kl_term)
            total_elbo = total_elbo + regularization_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1)  # sum over output dims
            return -logprob.mean(0)  # mean over batch dim

        return loss_fn
