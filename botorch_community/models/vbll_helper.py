#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Variational Bayesian Last Layers — enhanced port.

Original: vbll (https://github.com/VectorInstitute/vbll), MIT license.
Paper: "Variational Bayesian Last Layers" by Harrison et al., ICLR 2024

Enhancements:
- Use torch consistently for numerics (no numpy for log/dtypes).
- Device/dtype aware operations (torch.as_tensor with matching device/dtype).
- Improved numerical stability with torch.clamp.
- Convenience helpers: predictive sampling, posterior mean/covariance.
- Clearer typing and docstrings.
- Minor robustness fixes and shape checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from botorch.logging import logger


def tp(M: Tensor) -> Tensor:
    """Transpose the last two dimensions of a tensor."""
    return M.transpose(-1, -2)


class Normal(torch.distributions.Normal):
    """
    Diagonal Gaussian wrapper. 'scale' is interpreted as the std-dev vector.
    """

    def __init__(self, loc: Tensor, scale: Tensor):
        # ensure shape broadcastability but keep behavior identical to torch.Normal
        super().__init__(loc, scale)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def var(self) -> Tensor:
        return self.scale ** 2

    @property
    def chol_covariance(self) -> Tensor:
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self) -> Tensor:
        return self.var

    @property
    def covariance(self) -> Tensor:
        return torch.diag_embed(self.var)

    @property
    def precision(self) -> Tensor:
        return torch.diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self) -> Tensor:
        # 2 * sum log(scale)
        return 2.0 * torch.log(torch.clamp(self.scale, min=1e-30)).sum(dim=-1)

    @property
    def logdet_precision(self) -> Tensor:
        return -self.logdet_covariance

    @property
    def trace_covariance(self) -> Tensor:
        return self.var.sum(dim=-1)

    @property
    def trace_precision(self) -> Tensor:
        return (1.0 / self.var).sum(dim=-1)

    def covariance_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        """
        Compute b^T Cov b for diagonal covariance. Expects last dim of b == 1.
        b shape: (..., feat, 1)
        """
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = (self.var.unsqueeze(-1) * (b ** 2)).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = ((b ** 2) / self.var.unsqueeze(-1)).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp: Union["Normal", Tensor]) -> "Normal":
        if isinstance(inp, Normal):
            new_var = self.var + inp.var
            new_scale = torch.sqrt(torch.clamp(new_var, min=1e-12))
            return Normal(self.mean + inp.mean, new_scale)
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError("Distribution addition only implemented for diag covs")

    def __matmul__(self, inp: Tensor) -> "Normal":
        # linear projection: returns Normal of projected quantity
        if inp.shape[-2] != self.loc.shape[-1] or inp.shape[-1] != 1:
            raise ValueError("Input to matmul must have shape (..., feat, 1) matching loc")
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        new_scale = torch.sqrt(torch.clamp(new_cov, min=1e-12))
        return Normal(self.loc @ inp, new_scale)

    def squeeze(self, idx: int) -> "Normal":
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    """
    Dense multivariate normal with full lower-triangular scale_tril.
    """

    def __init__(self, loc: Tensor, cholesky: Tensor):
        super().__init__(loc, scale_tril=cholesky)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def chol_covariance(self) -> Tensor:
        return self.scale_tril

    @property
    def covariance(self) -> Tensor:
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self) -> Tensor:
        logger.warning(
            "Direct matrix inverse for dense covariances is O(N^3); prefer specialized ops"
        )
        Eye = torch.eye(self.scale_tril.shape[-1], device=self.scale_tril.device, dtype=self.scale_tril.dtype)
        W = torch.linalg.solve_triangular(self.scale_tril, Eye, upper=False)
        return tp(W) @ W

    @property
    def logdet_covariance(self) -> Tensor:
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(dim=-1)

    @property
    def trace_covariance(self) -> Tensor:
        # Frobenius norm squared of L equals trace of LL^T
        return (self.scale_tril ** 2).sum(dim=(-2, -1))

    def covariance_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = (torch.linalg.solve_triangular(self.scale_tril, b, upper=False) ** 2).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp: Tensor) -> Normal:
        if inp.shape[-2] != self.loc.shape[-1] or inp.shape[-1] != 1:
            raise ValueError("Input to matmul must have shape (..., feat, 1) matching loc")
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        new_scale = torch.sqrt(torch.clamp(new_cov, min=1e-12))
        return Normal(self.loc @ inp, new_scale)

    def squeeze(self, idx: int) -> "DenseNormal":
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    """Low-rank multivariate normal: cov = UU^T + diag(cov_diag)"""

    def __init__(self, loc: Tensor, cov_factor: Tensor, diag: Tensor):
        super().__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError("Cholesky not available for LowRankMultivariateNormal")

    @property
    def inverse_covariance(self):
        raise NotImplementedError("Inverse not implemented for LowRankNormal")

    @property
    def logdet_covariance(self) -> Tensor:
        # Matrix determinant lemma det(D + U U^T) = det(D) * det(I + D^{-1/2} U U^T D^{-1/2})
        cov_diag = self.cov_diag
        device = cov_diag.device
        dtype = cov_diag.dtype
        cov_diag = torch.clamp(cov_diag, min=1e-30)
        term1 = torch.log(cov_diag).sum(dim=-1)
        # build small matrix
        Dinv = (1.0 / cov_diag).unsqueeze(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor * Dinv)
        # ensure arg1 is float/double consistent
        I = torch.eye(arg1.shape[-1], device=device, dtype=arg1.dtype)
        term2 = torch.logdet(arg1 + I)
        return term1 + term2

    @property
    def trace_covariance(self) -> Tensor:
        trace_diag = self.cov_diag.sum(dim=-1)
        trace_lowrank = (self.cov_factor ** 2).sum(dim=(-2, -1))
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        diag_term = (self.cov_diag.unsqueeze(-1) * (b ** 2)).sum(dim=-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(dim=-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        raise NotImplementedError("Precision-weighted inner product for low-rank not implemented")

    def __matmul__(self, inp: Tensor) -> Normal:
        if inp.shape[-2] != self.loc.shape[-1] or inp.shape[-1] != 1:
            raise ValueError("Input to matmul must have shape (..., feat, 1) matching loc")
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        new_scale = torch.sqrt(torch.clamp(new_cov, min=1e-12))
        return Normal(self.loc @ inp, new_scale)

    def squeeze(self, idx: int) -> "LowRankNormal":
        return LowRankNormal(self.loc.squeeze(idx), self.cov_factor.squeeze(idx), self.cov_diag.squeeze(idx))


class DenseNormalPrec(torch.distributions.MultivariateNormal):
    """
    Dense Normal parameterized by mean and Cholesky of precision matrix.
    Internally we construct precision = L L^T (where L here is the provided tril),
    and pass precision_matrix to base class.
    """

    def __init__(self, loc: Tensor, cholesky: Tensor, validate_args: bool = False):
        prec = cholesky @ tp(cholesky)
        super().__init__(loc, precision_matrix=prec, validate_args=validate_args)
        self.tril = cholesky

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError("chol_covariance undefined for DenseNormalPrec")

    @property
    def covariance(self) -> Tensor:
        logger.warning("Direct inverse is O(N^3); prefer specialized ops")
        # Use solve_triangular to invert tril: inv_cov = (L^{-1})^T (L^{-1})
        invL = torch.cholesky_inverse(self.tril)  # returns full inverse of tril? use torch.cholesky_inverse for triangular
        return invL

    @property
    def inverse_covariance(self) -> Tensor:
        return self.precision_matrix

    @property
    def logdet_covariance(self) -> Tensor:
        # For precision tril T: logdet(cov) = -2 * sum log diag(T)
        return -2.0 * torch.diagonal(self.tril, dim1=-2, dim2=-1).log().sum(dim=-1)

    @property
    def trace_covariance(self) -> Tensor:
        # approximate as sum of squared inverse elements (not optimal but consistent)
        return (torch.inverse(self.tril) ** 2).sum(dim=(-2, -1))

    def covariance_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = (torch.linalg.solve(self.tril, b) ** 2).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b: Tensor, reduce_dim: bool = True) -> Tensor:
        if b.shape[-1] != 1:
            raise ValueError("b must have last dim 1")
        prod = ((tp(self.tril) @ b) ** 2).sum(dim=-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp: Tensor) -> Normal:
        if inp.shape[-2] != self.loc.shape[-1] or inp.shape[-1] != 1:
            raise ValueError("Input to matmul must have shape (..., feat, 1) matching loc")
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim=False)
        new_scale = torch.sqrt(torch.clamp(new_cov, min=1e-12))
        return Normal(self.loc @ inp, new_scale)

    def squeeze(self, idx: int) -> "DenseNormalPrec":
        return DenseNormalPrec(self.loc.squeeze(idx), self.tril.squeeze(idx))


def get_parameterization(p: str):
    COV_PARAM_DICT = {
        "dense": DenseNormal,
        "dense_precision": DenseNormalPrec,
        "diagonal": Normal,
        "lowrank": LowRankNormal,
    }
    if p not in COV_PARAM_DICT:
        raise ValueError(f"Invalid covariance parameterization: {p!r}")
    return COV_PARAM_DICT[p]


def gaussian_kl(p: Union[Normal, DenseNormal, LowRankNormal, DenseNormalPrec], q_scale: float) -> Tensor:
    """
    KL between variational posterior p (with zero-mean prior scaled by q_scale).
    q_scale may be float or tensor; we coerce to p.mean dtype/device.
    """
    feat_dim = p.mean.shape[-1]
    dtype = p.mean.dtype
    device = p.mean.device
    q_scale_t = torch.as_tensor(float(q_scale), dtype=dtype, device=device)

    mse_term = (p.mean ** 2).sum(dim=-1).sum(dim=-1) / q_scale_t
    trace_term = (p.trace_covariance / q_scale_t).sum(dim=-1)
    logdet_term = (feat_dim * torch.log(q_scale_t) - p.logdet_covariance).sum(dim=-1)
    return 0.5 * (mse_term + trace_term + logdet_term)


@dataclass
class VBLLReturn:
    predictive: Union[Normal, DenseNormal, LowRankNormal, DenseNormalPrec]
    train_loss_fn: Callable[[Tensor], Tensor]
    val_loss_fn: Callable[[Tensor], Tensor]


class Regression(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        regularization_weight: float,
        parameterization: str = "dense",
        mean_initialization: Optional[str] = None,
        prior_scale: float = 1.0,
        wishart_scale: float = 1e-2,
        cov_rank: Optional[int] = None,
        clamp_noise_init: bool = True,
        dof: float = 1.0,
    ):
        super().__init__()

        self.wishart_scale = float(wishart_scale)
        self.dof = float((dof + out_features + 1.0) / 2.0)
        self.regularization_weight = float(regularization_weight)
        self.dtype = torch.get_default_dtype()

        # prior scale adjusted by input dimension
        self.prior_scale = float(prior_scale) * (1.0 / float(in_features))

        # noise distribution params (diagonal)
        self.noise_mean = nn.Parameter(torch.zeros(out_features, dtype=self.dtype), requires_grad=False)
        # initialize log-diagonal of noise; use torch.randn scaled by wishart_scale
        self.noise_logdiag = nn.Parameter(torch.randn(out_features, dtype=self.dtype) * (torch.log(torch.tensor(wishart_scale, dtype=self.dtype))))
        if clamp_noise_init:
            with torch.no_grad():
                self.noise_logdiag.data = torch.clamp(self.noise_logdiag.data, min=0.0)

        # last-layer distribution type
        self.W_dist = get_parameterization(parameterization)

        # initialize mean
        if mean_initialization is None:
            self.W_mean = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype))
        elif mean_initialization == "kaiming":
            self.W_mean = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype) * torch.sqrt(torch.tensor(2.0 / in_features, dtype=self.dtype)))
        else:
            raise ValueError(f"Unknown initialization method: {mean_initialization!r}")

        # covariance parameterization-specific params
        if parameterization == "diagonal":
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype) - 0.5 * torch.log(torch.tensor(in_features, dtype=self.dtype)))
            self.W_offdiag = None
        elif parameterization == "dense":
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype) - 0.5 * torch.log(torch.tensor(in_features, dtype=self.dtype)))
            # create a full lower-triangular container per output row: stored as full matrix and later tril is taken
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features, dtype=self.dtype) / float(in_features))
        elif parameterization == "dense_precision":
            # here offdiag will encode cholesky of precision; initialize small
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype) + 0.5 * torch.log(torch.tensor(in_features, dtype=self.dtype)))
            self.W_offdiag = nn.Parameter(torch.zeros(out_features, in_features, in_features, dtype=self.dtype))
        elif parameterization == "lowrank":
            if cov_rank is None:
                raise ValueError("Must specify cov_rank for lowrank parameterization")
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype) - 0.5 * torch.log(torch.tensor(in_features, dtype=self.dtype)))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank, dtype=self.dtype) / float(in_features))
        else:
            raise ValueError(f"Unknown parameterization {parameterization}")

        self.parameterization = parameterization

    def W(self) -> Union[Normal, DenseNormal, LowRankNormal, DenseNormalPrec]:
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist is Normal:
            return Normal(self.W_mean, cov_diag)
        elif self.W_dist is DenseNormal or self.W_dist is DenseNormalPrec:
            # build lower-triangular cholesky tril: ensure diagonal uses cov_diag
            # if W_offdiag has shape (..., D, D) we use tril(W_offdiag) + diag(cov_diag)
            tril_base = torch.tril(self.W_offdiag, diagonal=-1) if self.W_offdiag is not None else torch.zeros_like(torch.diag_embed(cov_diag))
            tril = tril_base + torch.diag_embed(cov_diag)
            if self.W_dist is DenseNormal:
                return DenseNormal(self.W_mean, tril)
            else:
                # DenseNormalPrec expects tril of precision; we accept passed tril as precision-cholesky
                return DenseNormalPrec(self.W_mean, tril)
        elif self.W_dist is LowRankNormal:
            return LowRankNormal(self.W_mean, self.W_offdiag, cov_diag)
        else:
            raise RuntimeError("Unsupported W distribution type")

    def noise(self) -> Normal:
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    def forward(self, x: Tensor) -> VBLLReturn:
        return VBLLReturn(self.predictive(x), self._get_train_loss_fn(x), self._get_val_loss_fn(x))

    def predictive(self, x: Tensor) -> Union[Normal, DenseNormal, LowRankNormal, DenseNormalPrec]:
        # x is expected with shape (..., feat)
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def sample_predictive(self, x: Tensor, num_samples: int = 1) -> Tensor:
        """
        Draw samples from the predictive posterior:
        returns tensor with shape (num_samples, batch..., out_features)
        """
        pred = self.predictive(x)
        # Distribution supports sample with given sample_shape
        samples = pred.rsample(sample_shape=(num_samples,))
        return samples

    def _get_train_loss_fn(self, x: Tensor) -> Callable[[Tensor], Tensor]:
        def loss_fn(y: Tensor) -> Tensor:
            W = self.W()
            noise = self.noise()
            pred_mean = (W.mean @ x[..., None]).squeeze(-1)
            pred_density = Normal(pred_mean, noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            # covariance-weighted inner product (averages over features)
            # x.unsqueeze(-2)[..., None] ensures shape (..., feat, 1)
            b = x.unsqueeze(-2)[..., None]
            trace_term = 0.5 * (W.covariance_weighted_inner_prod(b) * noise.trace_precision)

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)

            total_elbo = torch.mean(pred_likelihood - trace_term)
            regularization_term = self.regularization_weight * (wishart_term - kl_term)
            total_elbo = total_elbo + regularization_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x: Tensor) -> Callable[[Tensor], Tensor]:
        def loss_fn(y: Tensor) -> Tensor:
            logprob = self.predictive(x).log_prob(y).sum(dim=-1)  # sum over output dims
            return -logprob.mean(dim=0)
        return loss_fn
