#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Preference Learning with Gaussian Process

.. [Chu2005preference]
    Wei Chu, and Zoubin Ghahramani. Preference learning with Gaussian processes.
    Proceedings of the 22nd international conference on Machine learning. 2005.

.. [Brochu2010tutorial]
    Eric Brochu, Vlad M. Cora, and Nando De Freitas.
    A tutorial on Bayesian optimization of expensive cost functions,
    with application to active user modeling and hierarchical reinforcement learning.
    arXiv preprint arXiv:1012.2599 (2010).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.exceptions.warnings import _get_single_precision_warning, InputDataWarning
from botorch.models.likelihoods.pairwise import (
    PairwiseLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.model import FantasizeMixin, Model
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.assorted import consolidate_duplicates
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from gpytorch import settings
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.models.gp import GP
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from linear_operator.operators import LinearOperator, RootLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from scipy import optimize
from torch import float32, float64, Tensor
from torch.nn.modules.module import _IncompatibleKeys


# Helper functions
def _check_strict_input(
    inputs: Iterable[Tensor], t_inputs: list[Tensor], target_or_inputs: str
):
    for input_, t_input in zip(inputs, t_inputs or (None,)):
        for attr in {"shape", "dtype", "device"}:
            expected_attr = getattr(t_input, attr, None)
            found_attr = getattr(input_, attr, None)
            if expected_attr != found_attr:
                raise RuntimeError(
                    f"Cannot modify {attr} of {target_or_inputs} "
                    f"(expected {expected_attr}, found {found_attr})."
                )


def _scaled_psd_safe_cholesky(
    matrix: Tensor, scale: Tensor, jitter: float | None = None
) -> Tensor:
    r"""scale matrix by 1/outputscale before cholesky for better numerical stability"""
    matrix = matrix / scale
    chol = psd_safe_cholesky(matrix, jitter=jitter)
    chol = chol * scale.sqrt()
    return chol


def _ensure_psd_with_jitter(
    matrix: Tensor,
    scale: float | Tensor = 1.0,
    jitter: float = 1e-8,
    max_tries: int = 3,
) -> Tensor:
    scaled_matrix = matrix / scale
    new_jitter = 0
    for i in range(max_tries):
        scaled_matrix = scaled_matrix + new_jitter * torch.diag_embed(
            torch.ones(
                scaled_matrix.shape[:-1],
                device=scaled_matrix.device,
                dtype=scaled_matrix.dtype,
            )
        )
        _, info = torch.linalg.cholesky_ex(scaled_matrix)
        psd = (info == 0).all()
        if psd:
            break
        else:
            new_jitter = jitter * (10**i) - new_jitter
    if not psd:
        raise NotPSDError(
            "Matrix not positive definite after repeatedly adding jitter "
            f"up to {jitter * (10**i):.1e}."
        )
    return scaled_matrix * scale


# Why we subclass GP even though it provides no functionality:
# if this subclassing is removed, we get the following GPyTorch error:
# "RuntimeError: All MarginalLogLikelihood objects must be given a GP object as
# a model. If you are using a more complicated model involving a GP, pass the
# underlying GP object as the model, not a full PyTorch module."
class PairwiseGP(Model, GP, FantasizeMixin):
    r"""Probit GP for preference learning with Laplace approximation

    A probit-likelihood GP that learns via pairwise comparison data, using a
    Laplace approximation of the posterior of the estimated utility values. By
    default it uses a scaled RBF kernel.

    Implementation is based on [Chu2005preference]_.
    Also see [Brochu2010tutorial]_ for additional reference.

    Note that in [Chu2005preference]_ the likelihood of a pairwise comparison
    is :math:`\left(\frac{f(x_1) - f(x_2)}{\sqrt{2}\sigma}\right)`, i.e. a scale is
    used in the denominator. To maintain consistency with usage of kernels
    elsewhere in BoTorch, we instead do not include :math:`\sigma` in the code
    (implicitly setting it to 1) and use ScaleKernel to scale the function.

    In the example below, the user/decision maker has stated that they prefer
    the first item over the second item and the third item over the second item,
    generating comparisons [0, 1] and [2, 1].
    Example:
        >>> from botorch.models import PairwiseGP
        >>> import torch
        >>> datapoints = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> comparisons = torch.Tensor([[0, 1], [2, 1]])
        >>> model = PairwiseGP(datapoints, comparisons)
    """

    _buffer_names = [
        "consolidated_datapoints",
        "consolidated_comparisons",
        "D",
        "DT",
        "utility",
        "covar_chol",
        "likelihood_hess",
        "hlcov_eye",
        "covar",
        "covar_inv",
        "unconsolidated_datapoints",
        "unconsolidated_comparisons",
        "consolidated_indices",
    ]

    def __init__(
        self,
        datapoints: Tensor | None,
        comparisons: Tensor | None,
        likelihood: PairwiseLikelihood | None = None,
        covar_module: ScaleKernel | None = None,
        input_transform: InputTransform | None = None,
        *,
        jitter: float = 1e-6,
        xtol: float | None = None,
        consolidate_rtol: float = 0.0,
        consolidate_atol: float = 1e-4,
        maxfev: int | None = None,
    ) -> None:
        r"""
        Args:
            datapoints: Either `None` or a `batch_shape x n x d` tensor of
                training features. If either `datapoints` or `comparisons` is
                `None`, construct a prior-only model.
            comparisons: Either `None` or a `batch_shape x m x 2` tensor of
                training comparisons; comparisons[i] is a noisy indicator
                suggesting the utility value of comparisons[i, 0]-th is greater
                than comparisons[i, 1]-th. If either `comparisons` or
                `datapoints` is `None`, construct a prior-only model.
            likelihood: A PairwiseLikelihood.
            covar_module: Covariance module.
            input_transform: An input transform that is applied in the model's
                forward pass.
            jitter: Value added to diagonal for numerical stability in
                `psd_safe_cholesky`.
            xtol: Stopping creteria in scipy.optimize.fsolve used to find f_map
                in `PairwiseGP._update`. If None, default behavior is handled by
                `PairwiseGP._update`.
            consolidate_rtol: `rtol` passed to `consolidate_duplicates`.
            consolidate_atol: `atol` passed to `consolidate_duplicates`.
            maxfev: The maximum number of calls to the function in
                scipy.optimize.fsolve. If None, default behavior is handled by
                `PairwiseGP._update`.
        """
        super().__init__()
        # Input data validation
        if datapoints is not None and datapoints.dtype == torch.float32:
            warnings.warn(
                _get_single_precision_warning(str(datapoints.dtype)),
                category=InputDataWarning,
                stacklevel=2,
            )

        # Set optional parameters
        self._jitter = jitter
        self._xtol = xtol
        self._consolidate_rtol = consolidate_rtol
        self._consolidate_atol = consolidate_atol
        self._maxfev = maxfev

        if input_transform is not None:
            input_transform.to(datapoints)
            # input transformation is applied in set_train_data
            self.input_transform = input_transform

        # Compatibility variables with fit_gpytorch_*: Dummy likelihood
        # Likelihood is tightly tied with this model and
        # it doesn't make much sense to keep it separate
        self.likelihood = (
            PairwiseProbitLikelihood() if likelihood is None else likelihood
        )

        for key in self._buffer_names:
            self.register_buffer(key, None)

        self.train_inputs = []
        self.train_targets = None
        self.utility = None

        self.pred_cov_fac_need_update = True
        self.dim = None
        self.unconsolidated_datapoints = None
        self.unconsolidated_comparisons = None
        self.consolidated_datapoints = None
        self.consolidated_comparisons = None
        self.consolidated_indices = None

        # See set_train_data for additional compatibility variables.
        # Not that the datapoints here are not transformed even if input_transform
        # is not None to avoid double transformation during model fitting.
        # self.transform_inputs is called in `forward`
        self.set_train_data(datapoints, comparisons, update_model=False)

        # Set hyperparameters
        # Do not set the batch_shape explicitly so mean_module can operate in both mode
        # once fsolve used in _update can run in batch mode, we should explicitly set
        # the bacth shape here
        self.mean_module = ConstantMean()
        # Do not optimize constant mean prior
        for param in self.mean_module.parameters():
            param.requires_grad = False

        # set covariance module
        # the default outputscale here is only a rule of thumb, meant to keep
        # estimates away from scale value that would make Phi(f(x)) saturate
        # at 0 or 1
        if covar_module is None:
            os_lb, os_ub = 1e-2, 1e2
            ls_prior = GammaPrior(concentration=2.4, rate=2.7)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            covar_module = ScaleKernel(
                RBFKernel(
                    batch_shape=self.batch_shape,
                    ard_num_dims=self.dim,
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=GreaterThan(
                        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
                    ),
                    dtype=torch.float64,
                ),
                outputscale_prior=SmoothedBoxPrior(a=os_lb, b=os_ub),
                # make sure we won't get extreme values for the output scale
                outputscale_constraint=Interval(
                    lower_bound=os_lb * 0.5,
                    upper_bound=os_ub * 2.0,
                    initial_value=1.0,
                ),
                dtype=torch.float64,
            )
        if not isinstance(covar_module, ScaleKernel):
            raise UnsupportedError("PairwiseGP must be used with a ScaleKernel.")
        self.covar_module = covar_module

        self._x0 = None  # will store temporary results for warm-starting
        if self.datapoints is not None and self.comparisons is not None:
            self.to(dtype=self.datapoints.dtype, device=self.datapoints.device)
            # Find f_map for initial parameters with transformed datapoints
            transformed_dp = self.transform_inputs(self.datapoints)
            self._update(transformed_dp)

        self.to(self.datapoints)

    def __deepcopy__(self, memo) -> PairwiseGP:
        attrs = (
            "consolidated_datapoints",
            "consolidated_comparisons",
            "covar",
            "covar_inv",
            "covar_chol",
            "likelihood_hess",
            "utility",
            "hlcov_eye",
            "unconsolidated_datapoints",
            "unconsolidated_comparisons",
            "consolidated_indices",
        )

        if any(getattr(self, attr) is not None for attr in attrs):
            # Temporarily remove non-leaf tensors so that pytorch allows deepcopy
            old_attr = {}
            for attr in attrs:
                old_attr[attr] = getattr(self, attr)
                setattr(self, attr, None)
            new_model = deepcopy(self, memo)
            # now set things back
            for attr in attrs:
                setattr(self, attr, old_attr[attr])
            return new_model
        else:
            dcp = self.__deepcopy__
            # make sure we don't fall into the infinite recursive loop
            self.__deepcopy__ = None
            new_model = deepcopy(self, memo)
            self.__deepcopy__ = dcp
            return new_model

    def _has_no_data(self):
        r"""Return true if the model does not have both datapoints and comparisons"""
        return (
            self.datapoints is None
            or len(self.datapoints.size()) == 0
            or self.comparisons is None
        )

    def _calc_covar(self, X1: Tensor, X2: Tensor) -> Tensor | LinearOperator:
        r"""Calculate the covariance matrix given two sets of datapoints"""
        covar = self.covar_module(X1, X2).to_dense()
        # making sure covar is PSD when it's a covariance matrix
        if X1 is X2:
            scale = self.covar_module.outputscale.unsqueeze(-1).unsqueeze(-1).detach()
            covar = _ensure_psd_with_jitter(
                matrix=covar,
                scale=scale,
                jitter=self._jitter,
            )
        return covar

    def _update_covar(self, datapoints: Tensor) -> None:
        r"""Update values derived from the data and hyperparameters

        covar, covar_chol, and covar_inv will be of shape batch_shape x n x n

        Args:
            datapoints: (Transformed) datapoints for finding f_max
        """
        self.covar = self._calc_covar(datapoints, datapoints)
        scale = self.covar_module.outputscale.unsqueeze(-1).unsqueeze(-1).detach()
        self.covar_chol = _scaled_psd_safe_cholesky(
            matrix=self.covar,
            scale=scale,
            jitter=self._jitter,
        )
        self.covar_inv = torch.cholesky_inverse(self.covar_chol)

    def _prior_mean(self, X: Tensor) -> Tensor | LinearOperator:
        r"""Return point prediction using prior only

        Args:
            X: A `batch_size x n' x d`-dim Tensor at which to evaluate prior

        Returns:
            Prior mean prediction
        """
        return self.mean_module(X)

    def _prior_predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Predict utility based on prior info only

        Args:
            X: A `batch_size x n' x d`-dim Tensor at which to evaluate prior

        Returns:
            pred_mean: predictive mean
            pred_covar: predictive covariance
        """
        pred_mean = self._prior_mean(X)
        pred_covar = self._calc_covar(X, X)
        return pred_mean, pred_covar

    def _grad_posterior_f(
        self,
        utility: Tensor | npt.NDArray,
        datapoints: Tensor,
        D: Tensor,
        covar_chol: Tensor,
        covar_inv: Tensor | None = None,
        ret_np: bool = False,
    ) -> Tensor | npt.NDArray:
        r"""Compute the gradient of S loss wrt to f/utility in [Chu2005preference]_.

        For finding f_map, which is negative of the log posterior, i.e., -log(p(f|D))
        Derivative of (10) in [Chu2005preference]_.
        Also see [Brochu2010tutorial]_ page 26. This is needed for estimating f_map.

        Args:
            utility: A Tensor of shape `batch_size x n`
            datapoints: A Tensor of shape `batch_size x n x d` as in self.datapoints
            D: A Tensor of shape `batch_size x m x n` as in self.D
            covar_chol: A Tensor of shape `batch_size x n x n`, as in self.covar_chol
            covar_inv: `None` or a Tensor of shape `batch_size x n x n`, as in
                self.covar_inv. This is not used but is needed so that
                PairwiseGP._grad_posterior_f has the same signature as
                PairwiseGP._hess_posterior_f.
            ret_np: return a numpy array if True, otherwise a Tensor
        """
        prior_mean = self._prior_mean(datapoints)

        if ret_np:
            utility = torch.tensor(utility, dtype=self.datapoints.dtype)
            prior_mean = prior_mean.cpu()

        # NOTE: During the optimization, it can occur that b, p, and g_ are NaNs, though
        # in the cases that occured during testing, the optimization routine escaped and
        # terminated successfully without NaNs in the result.
        b = self.likelihood.negative_log_gradient_sum(utility=utility, D=D)
        # g_ = covar_inv x (utility - pred_prior)
        p = (utility - prior_mean).unsqueeze(-1).to(covar_chol)
        g_ = torch.cholesky_solve(p, covar_chol).squeeze(-1)
        g = g_ + b
        if ret_np:
            return g.cpu().numpy()
        return g

    def _hess_posterior_f(
        self,
        utility: Tensor | npt.NDArray,
        datapoints: Tensor,
        D: Tensor,
        covar_chol: Tensor,
        covar_inv: Tensor,
        ret_np: bool = False,
    ) -> Tensor | npt.NDArray:
        r"""Compute the hessian of S loss wrt utility for finding f_map.

        which is negative of the log posterior, i.e., -log(p(f|D))
        Following [Chu2005preference]_ section 2.2.1.
        This is needed for estimating f_map

        Args:
            utility: A Tensor of shape `batch_size x n`
            datapoints: A Tensor of shape `batch_size x n x d`, as in
                self.datapoints. This is not used but is needed so that
                `_hess_posterior_f` has the same signature as
                `_grad_posterior_f`.
            D: A Tensor of shape `batch_size x m x n` as in self.D
            covar_chol: A Tensor of shape `batch_size x n x n`, as in
                self.covar_chol. This is not used but is needed so that
                `_hess_posterior_f` has the same signature as
                `_grad_posterior_f`.
            covar_inv: A Tensor of shape `batch_size x n x n`, as in self.covar_inv
            ret_np: return a numpy array if true, otherwise a Tensor
        """
        if ret_np:
            utility = torch.tensor(utility, dtype=self.datapoints.dtype)

        hl = self.likelihood.negative_log_hessian_sum(utility=utility, D=D)
        hess = hl + covar_inv
        return hess.numpy() if ret_np else hess

    def _update_utility_derived_values(self) -> None:
        r"""
        Set self.hlcov_eye to self.likelihood_hess @ self.covar + I.

        `self.hlcov_eye` is a utility-derived value not used during
        optimization. This quantity is used so that we will be able to compute
        the predictive covariance (in PairwiseGP.forward in posterior mode) with
        better numerical stability using the substitution method:

        Let `pred_cov_fac = (covar + hl^-1)`, which is needed for calculating
        the predictive covariance = `K - k.T @ pred_cov_fac^-1 @ k`.
        Instead of inverting `pred_cov_fac`, let `hlcov_eye = (hl @ covar + I)`
        Then we can obtain `pred_cov_fac^-1 @ k` by solving for p in
        `(hl @ k) p = hlcov_eye`
        `hlcov_eye p = hl @ k`
        """
        hl = self.likelihood_hess  # "C" from page 27, [Brochu2010tutorial]_
        hlcov = hl @ self.covar
        eye = torch.eye(
            hlcov.size(-1), dtype=self.datapoints.dtype, device=self.datapoints.device
        ).expand(hlcov.shape)
        self.hlcov_eye = hlcov + eye

        self.pred_cov_fac_need_update = False

    def _update(self, datapoints: Tensor, **kwargs) -> None:
        r"""Update the model by updating the covar matrix and MAP utility values

        Update the model by
        1. Re-evaluating the covar matrix as the data or hyperparams may have changed
        2. Approximating maximum a posteriori of the utility function f using fsolve

        Should be called after data or hyperparameters are changed to update
        f_map and related values

        self._xtol and self._maxfev are passed to fsolve as xtol and maxfev
        to control stopping criteria

        Args:
            datapoints: (transformed) datapoints for finding f_max
        """

        xtol = 1e-6 if self._xtol is None else self._xtol
        maxfev = 100 if self._maxfev is None else self._maxfev

        # Using the latest param for covariance before calculating f_map
        self._update_covar(datapoints)

        # scipy newton raphson
        with torch.no_grad():
            # warm start
            init_x0_size = self.batch_shape + torch.Size([self.n])
            if self._x0 is None or torch.Size(self._x0.shape) != init_x0_size:
                sqrt_scale = (
                    self.covar_module.outputscale.sqrt()
                    .unsqueeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                # Heuristic intialization using winning count with perturbation
                # to avoid extreme or unprobable likelihood values
                win_count = self.D.sum(dim=-2).detach().cpu().numpy()
                wc_mean, wc_std = (
                    win_count.mean(axis=-1, keepdims=True),
                    win_count.std(axis=-1, keepdims=True).clip(min=1e-6),
                )
                x0 = (win_count - wc_mean) / wc_std
                # adding random perturbation to in case get stuck at strange init values
                x0 = x0 + 0.05 * np.random.standard_normal(init_x0_size)
                # scale x0 to be on roughly the right scale
                x0 = x0 * sqrt_scale
            else:
                x0 = self._x0

            if len(self.batch_shape) > 0:
                # batch mode, do optimize.fsolve sequentially on CPU
                # TODO: enable vectorization/parallelization here
                x0 = x0.reshape(-1, self.n)
                dp_v = datapoints.view(-1, self.n, self.dim).cpu()
                D_v = self.D.view(-1, self.m, self.n).cpu()
                ch_v = self.covar_chol.view(-1, self.n, self.n).cpu()
                ci_v = self.covar_inv.view(-1, self.n, self.n).cpu()
                x = np.empty(x0.shape)
                for i in range(x0.shape[0]):
                    fsolve_args = (dp_v[i], D_v[i], ch_v[i], ci_v[i], True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        x[i] = optimize.fsolve(
                            x0=x0[i],
                            func=self._grad_posterior_f,
                            fprime=self._hess_posterior_f,
                            xtol=xtol,
                            maxfev=maxfev,
                            args=fsolve_args,
                            **kwargs,
                        )
                x = x.reshape(*init_x0_size)
            else:
                # fsolve only works on CPU
                fsolve_args = (
                    datapoints.cpu(),
                    self.D.cpu(),
                    self.covar_chol.cpu(),
                    self.covar_inv.cpu(),
                    True,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    x = optimize.fsolve(
                        x0=x0,
                        func=self._grad_posterior_f,
                        fprime=self._hess_posterior_f,
                        xtol=xtol,
                        maxfev=maxfev,
                        args=fsolve_args,
                        **kwargs,
                    )

            self._x0 = x.copy()  # save for warm-starting
            f = torch.tensor(x, dtype=datapoints.dtype, device=datapoints.device)

        # To perform hyperparameter optimization, this needs to be recalculated
        # when calling forward() in order to obtain correct gradients
        # self.likelihood_hess is updated here is for the rare case where we
        # do not want to call forward()
        self.likelihood_hess = self.likelihood.negative_log_hessian_sum(
            utility=f, D=self.D
        )

        # Lazy update hlcov_eye, which is used in calculating posterior during training
        self.pred_cov_fac_need_update = True
        # fill in dummy values for hlcov_eye so that load_state_dict can function
        hlcov_eye_size = torch.Size((*self.likelihood_hess.shape[:-2], self.n, self.n))
        self.hlcov_eye = torch.empty(hlcov_eye_size)

        # Take two newton step on the posterior MAP point to fill
        # in gradients for pytorch. Using 2 instead of 1 since empirically sometimes
        # the first step results in gradients in the order of 1e-7 while the 2nd step
        # allows it go down further to the order of 1e-12 and stay there.
        self.utility = self._util_newton_updates(
            dp=datapoints, x0=f.clone().requires_grad_(True), max_iter=2
        )

    def _transform_batch_shape(self, X: Tensor, X_new: Tensor) -> tuple[Tensor, Tensor]:
        r"""Transform X and X_new into the same shape

        Transform the batch shape of X to be compatible
        with `X_new` to calculate the posterior.
        If X has the same batch size as `X_new`, return it as is.
        If one is in batch mode and the other one is not, convert both
        into batch mode.
        If both are in batch mode, this will only work if X_batch_shape
        can propagate to X_new_batch_shape

        Args:
            X: A `batch_shape x q x d`-dim or `(1 x) q x d`-dim Tensor
            X_new: A `batch_shape x q x d`-dim Tensor

        Returns:
            Transformed X and X_new pair
        """
        X_bs = X.shape[:-2]  # X batch shape
        X_new_bs = X_new.shape[:-2]  # X_new batch shape
        if X_new_bs == X_bs:
            # if batch shapes match, there's no need to transform
            # X_new may or may not have batch_shape dimensions
            return X, X_new
        elif len(X_new_bs) < len(X_bs):
            # if X_new has fewer dimension, try to expand it to X's shape
            return X, X_new.expand(X_bs + X_new.shape[-2:])
        else:
            # if X has fewer dimension, try to expand it to X_new's shape
            return X.expand(X_new_bs + X.shape[-2:]), X_new

    def _util_newton_updates(
        self, dp: Tensor, x0: Tensor, max_iter: int = 1, xtol: float | None = None
    ) -> Tensor:
        r"""Make `max_iter` newton updates on utility.

        This is used in `forward` to calculate and fill in gradient into tensors.
        Instead of doing utility -= H^-1 @ g, use substition method.
        See more explanation in _update_utility_derived_values.
        By default only need to run one iteration just to fill the the gradients.

        Args:
            dp: (Transformed) datapoints. A Tensor of shape `batch_size x n x d`
                as in self.datapoints
            x0: A `batch_size x n` dimension tensor, initial values.
            max_iter: Max number of iterations.
            xtol: Stop creteria. If `None`, do not stop until
                finishing `max_iter` updates.
        """
        xtol = float("-Inf") if xtol is None else xtol
        D, ch = self.D, self.covar_chol
        covar = self.covar
        diff = float("Inf")
        i = 0
        x = x0
        eye = None
        while i < max_iter and diff > xtol:
            hl = self.likelihood.negative_log_hessian_sum(utility=x, D=D)
            self.likelihood_hess = hl
            cov_hl = covar @ hl
            if eye is None:
                eye = torch.diag_embed(
                    torch.ones(
                        cov_hl.shape[:-1], device=cov_hl.device, dtype=cov_hl.dtype
                    )
                )
            cov_hl = cov_hl + eye  # add 1 to cov_hl
            g = self._grad_posterior_f(
                utility=x,
                datapoints=dp,
                D=D,
                covar_chol=ch,
            )
            cov_g = covar @ g.unsqueeze(-1)
            x_update = torch.linalg.solve(cov_hl, cov_g).squeeze(-1)
            x_next = x - x_update
            diff = torch.linalg.norm(x - x_next)
            x = x_next
            i += 1

        return x

    def _consolidate_duplicates(
        self, datapoints: Tensor, comparisons: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Consolidate and cache datapoints and comparisons"""
        # check if consolidated datapoints/comparisons are cached
        if (
            (datapoints is not self.unconsolidated_datapoints)
            or (comparisons is not self.unconsolidated_comparisons)
            or (self.consolidated_datapoints is None)
            or (self.consolidated_comparisons is None)
        ):
            self.unconsolidated_datapoints, self.unconsolidated_comparisons = (
                datapoints,
                comparisons,
            )

            if len(datapoints.shape) > 2 or len(comparisons.shape) > 2:
                # Do not perform consolidation in batch mode as block design
                # cannot be guaranteed
                self.consolidated_datapoints = datapoints
                self.consolidated_comparisons = comparisons
                self.consolidated_indices = None
            else:
                (
                    self.consolidated_datapoints,
                    self.consolidated_comparisons,
                    self.consolidated_indices,
                ) = consolidate_duplicates(
                    datapoints,
                    comparisons,
                    rtol=self._consolidate_rtol,
                    atol=self._consolidate_atol,
                )

        return self.consolidated_datapoints, self.consolidated_comparisons

    # ============== public APIs ==============
    @property
    def datapoints(self) -> Tensor:
        r"""Alias for consolidated datapoints"""
        return self.consolidated_datapoints

    @property
    def comparisons(self) -> Tensor:
        r"""Alias for consolidated comparisons"""
        return self.consolidated_comparisons

    @property
    def unconsolidated_utility(self) -> Tensor:
        r"""Utility of the unconsolidated datapoints"""
        if self.consolidated_indices is None:
            # self.consolidated_indices is None in batch mode
            return self.utility
        else:
            return self.utility[self.consolidated_indices]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        r"""The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the internal
        representation of the model (as e.g. in BatchedMultiOutputGPyTorchModel).
        For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        if self.datapoints is None:
            # this could happen in prior mode
            return torch.Size()
        else:
            return self.datapoints.shape[:-2]

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
    ) -> dict[str, Tensor]:
        r"""
        Construct `Model` keyword arguments from a `RankingDataset`.

        Args:
            training_data: A `RankingDataset`, with attributes `train_X`,
                `train_Y`, and, optionally, `train_Yvar`.

        Returns:
            A dict of keyword arguments that can be used to initialize a
            `PairwiseGP`, including `datapoints` and `comparisons`.
        """
        if not isinstance(training_data, RankingDataset):
            raise UnsupportedError(
                "Only `RankingDataset` is supported for `PairwiseGP`. Received "
                f"{type(training_data)}."
            )
        datapoints = training_data._X.values
        comparisons = training_data._X.indices
        comp_order = training_data.Y
        comparisons = torch.gather(input=comparisons, dim=-1, index=comp_order)

        return {
            "datapoints": datapoints,
            "comparisons": comparisons,
        }

    def set_train_data(
        self,
        datapoints: Tensor | None = None,
        comparisons: Tensor | None = None,
        strict: bool = False,
        update_model: bool = True,
    ) -> None:
        r"""Set datapoints and comparisons and update model properties if needed

        Args:
            datapoints: Either `None` or a `batch_shape x n x d` dimension
                tensor X. If there are input transformations, assume the
                datapoints are not transformed. If either `datapoints` or
                `comparisons` is `None`, construct a prior-only model.
            comparisons: Either `None` or a tensor of size `batch_shape x m x
                2`. (i, j) means f_i is preferred over f_j. If either
                `comparisons` or `datapoints` is `None`, construct a prior-only
                model.
            strict: `strict` argument as in gpytorch.models.exact_gp for compatibility
                when using fit_gpytorch_mll with input_transform.
            update_model: True if we want to refit the model (see _update) after
                re-setting the data.
        """
        # When datapoints and/or comparisons are None, we are constructing
        # a prior-only model
        if datapoints is None or comparisons is None:
            return

        # following gpytorch.models.exact_gp.set_train_data
        if datapoints is not None:
            if torch.is_tensor(datapoints):
                inputs = [datapoints]

            inputs = tuple(
                input_.unsqueeze(-1) if input_.ndimension() == 1 else input_
                for input_ in inputs
            )
            if strict:
                _check_strict_input(inputs, self.train_inputs, "inputs")

            datapoints = inputs[0]
            # Compatibility variables with fit_gpytorch_*
            # alias for datapoints ("train_inputs")
            self.train_inputs = inputs

        if comparisons is not None:
            if strict:
                _check_strict_input([comparisons], [self.train_targets], "targets")

            # convert to long so that it can be used as index and
            # compatible with Tensor.scatter_
            comparisons = comparisons.long()
            # Compatibility variables with fit_gpytorch_*
            # alias for comparisons ("train_targets" here)
            self.train_targets = comparisons

        # self.datapoints and self.comparisons are being updated here
        self._consolidate_duplicates(datapoints, comparisons)

        # Compatibility variables with optimize_acqf
        self._dtype = self.datapoints.dtype
        self._num_outputs = 1  # 1 latent value output per observation

        self.dim = self.datapoints.shape[-1]  # feature dimensions
        self.n = self.datapoints.shape[-2]  # num datapoints
        self.m = self.comparisons.shape[-2]  # num pairwise comparisons
        # D is batch_size x m x n or num_comparison x num_datapoints.
        # D_k_i is the s_k(x_i) value as in equation (6) in [Chu2005preference]_
        # D will usually be very sparse as well
        # TODO swap out scatter_ so that comparisons could be int instead of long
        # TODO: make D a sparse matrix once pytorch has better support for
        #       sparse tensors
        D_size = torch.Size((*(self.batch_shape), self.m, self.n))
        self.D = torch.zeros(
            D_size, dtype=self.datapoints.dtype, device=self.datapoints.device
        )
        comp_view = self.comparisons.view(-1, self.m, 2).long()
        for i, sub_D in enumerate(self.D.view(-1, self.m, self.n)):
            sub_D.scatter_(1, comp_view[i, :, [0]], 1)
            sub_D.scatter_(1, comp_view[i, :, [1]], -1)
        self.DT = self.D.transpose(-1, -2)

        if update_model:
            transformed_dp = self.transform_inputs(self.datapoints)
            self._update(transformed_dp)

        self.to(self.datapoints)

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = False
    ) -> _IncompatibleKeys:
        r"""Removes data related buffers from the `state_dict` and calls
        `super().load_state_dict` with `strict=False`.

        Args:
            state_dict: The state dict.
            strict: Boolean specifying whether or not given and instance-bound
                state_dicts should have identical keys. Only implemented for
                `strict=False` since buffers will filters out when calling
                `_load_from_state_dict`.

        Returns:
            A named tuple `_IncompatibleKeys`, containing the `missing_keys`
            and `unexpected_keys`.
        """
        if strict:
            raise UnsupportedError("Passing strict=True is not supported.")

        return super().load_state_dict(state_dict=state_dict, strict=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict={
                k: v for k, v in state_dict.items() if k not in self._buffer_names
            },
            prefix=prefix,
            local_metadata=local_metadata,
            strict=False,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def forward(self, datapoints: Tensor) -> MultivariateNormal:
        r"""Calculate a posterior or prior prediction.

        During training mode, forward implemented solely for gradient-based
        hyperparam opt. Essentially what it does is to re-calculate the utility
        f using its analytical form at f_map so that we are able to obtain
        gradients of the hyperparameters.

        Args:
            datapoints: A `batch_shape x n x d` Tensor,
                should be the same as self.datapoints during training

        Returns:
            A MultivariateNormal object, being one of the followings:
                1. Posterior centered at MAP points for training data (training mode)
                2. Prior predictions (prior mode)
                3. Predictive posterior (eval mode)
        """
        # Training mode: optimizing
        if self.training:
            if self._has_no_data():
                raise RuntimeError(
                    "datapoints and comparisons cannot be None in training mode. "
                    "Call .eval() for prior predictions, "
                    "or call .set_train_data() to add training data."
                )

            if datapoints is not self.unconsolidated_datapoints:
                raise RuntimeError("Must train on training data")

            # We pass in the untransformed datapoints into set_train_data
            # as we will be setting self.datapoints as the untransformed datapoints
            # self.transform_inputs will be called inside before calling _update()
            self.set_train_data(
                datapoints=datapoints,
                comparisons=self.unconsolidated_comparisons,
                update_model=True,
            )

            transformed_dp = self.transform_inputs(self.datapoints)

            hl = self.likelihood_hess
            covar = self.covar

            # Apply matrix inversion lemma on eq. in page 27 of [Brochu2010tutorial]_
            # (A + B)^-1 = A^-1 - A^-1 @ (I + BA^-1)^-1 @ BA^-1
            # where A = covar_inv, B = hl
            hl_cov = hl @ covar
            eye = torch.eye(
                hl_cov.size(-1),
                dtype=self.datapoints.dtype,
                device=self.datapoints.device,
            ).expand(hl_cov.shape)
            hl_cov_I = hl_cov + eye  # add I to hl_cov
            output_covar = covar - covar @ torch.linalg.solve(hl_cov_I, hl_cov)
            output_mean = self.utility

        # Prior mode
        elif settings.prior_mode.on() or self._has_no_data():
            transformed_new_dp = self.transform_inputs(datapoints)
            # if we don't have any data yet, use prior GP to make predictions
            output_mean, output_covar = self._prior_predict(transformed_new_dp)

        # Posterior mode
        else:
            transformed_dp = self.transform_inputs(self.datapoints)
            transformed_new_dp = self.transform_inputs(datapoints).to(transformed_dp)

            # self.utility might be None if exception was raised and _update
            # was failed to be called during hyperparameter optimization
            # procedures (e.g., fit_gpytorch_mll_scipy)
            if self.utility is None:
                self._update(transformed_dp)

            if self.pred_cov_fac_need_update:
                self._update_utility_derived_values()

            X, X_new = self._transform_batch_shape(transformed_dp, transformed_new_dp)
            covar_chol, _ = self._transform_batch_shape(self.covar_chol, X_new)
            hl, _ = self._transform_batch_shape(self.likelihood_hess, X_new)
            hlcov_eye, _ = self._transform_batch_shape(self.hlcov_eye, X_new)

            # otherwise compute predictive mean and covariance
            covar_xnew_x = self._calc_covar(X_new, X)
            covar_x_xnew = covar_xnew_x.transpose(-1, -2)
            covar_xnew = self._calc_covar(X_new, X_new)
            p = self.utility - self._prior_mean(X)

            covar_inv_p = torch.cholesky_solve(p.unsqueeze(-1), covar_chol)
            pred_mean = (covar_xnew_x @ covar_inv_p).squeeze(-1)
            pred_mean = pred_mean + self._prior_mean(X_new)

            # Using the terminology from [Brochu2010tutorial]_ page 27:
            # hl = C; hlcov_eye = CK + I; k = covar_x_xnew
            #
            # To compute the predictive covariance, one term we need is
            # k^T (K + C^{-1})^{-1} k.
            # Rather than performing two matrix inversions, we can compute this
            # in a more efficient and numerically stable way by using
            # fac = hlcov_eye^-1 @ hl @ covar_x_xnew
            #     = (CK + I)^-1 @ C @ k
            #     = (K + C^-1)^{-1}
            # This is the substitution method.
            fac = torch.linalg.solve(hlcov_eye, hl @ covar_x_xnew)
            pred_covar = covar_xnew - (covar_xnew_x @ fac)

            output_mean, output_covar = pred_mean, pred_covar

        scale = self.covar_module.outputscale.unsqueeze(-1).unsqueeze(-1).detach()
        post = MultivariateNormal(
            mean=output_mean,
            # output_covar is sometimes non-PSD
            # perform a cholesky decomposition to check and amend
            covariance_matrix=RootLinearOperator(
                _scaled_psd_safe_cholesky(
                    matrix=output_covar,
                    scale=scale,
                    jitter=self._jitter,
                )
            ),
        )
        return post

    # ============== botorch.models.model.Model interfaces ==============
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `batch_shape x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered jointly.
            output_indices: As defined in parent Model class, not used for this model.
            observation_noise: Ignored (since noise is not identifiable from scale
                in probit models).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `Posterior` object, representing joint
                distributions over `q` points.
        """
        self.eval()  # make sure model is in eval mode

        if output_indices is not None:
            raise RuntimeError(
                "output_indices is not None. PairwiseGP should not be a"
                "multi-output model."
            )

        post = self(X)
        posterior = GPyTorchPosterior(post)
        if posterior_transform is not None:
            return posterior_transform(posterior=posterior, X=X)
        return posterior

    def condition_on_observations(self, X: Tensor, Y: Tensor) -> Model:
        r"""Condition the model on new observations.

        Note that unlike other BoTorch models, PairwiseGP requires Y to be
        pairwise comparisons.

        Args:
            X: A `batch_shape x n x d` dimension tensor X
            Y: A tensor of size `batch_shape x m x 2`. (i, j) means
                f_i is preferred over f_j
            kwargs: Not used.

        Returns:
            A (deepcopied) `Model` object of the same type, representing the
            original model conditioned on the new observations `(X, Y)`.
        """
        new_model = deepcopy(self)

        if self._has_no_data():
            # If the model previously has no data, set X and Y as the data directly
            new_model.set_train_data(X, Y, update_model=True)
        else:
            # Can only condition on pairwise comparisons instead of the directly
            # observed values. Raise a RuntimeError if Y is not a tensor presenting
            # pairwise comparisons
            if Y.dtype in (float32, float64) or Y.shape[-1] != 2:
                raise RuntimeError(
                    "Conditioning on non-pairwise comparison observations."
                )

            # Reshaping datapoints and comparisons by batches
            Y_new_batch_shape = Y.shape[:-2]
            new_datapoints = self.datapoints.expand(
                Y_new_batch_shape + self.datapoints.shape[-2:]
            )
            new_comparisons = self.comparisons.expand(
                Y_new_batch_shape + self.comparisons.shape[-2:]
            )
            # Reshape X since Y may have additional batch dim. from fantasy models
            X = X.expand(Y_new_batch_shape + X.shape[-2:])

            new_datapoints = torch.cat((new_datapoints, X.to(new_datapoints)), dim=-2)

            shifted_comp = Y.to(new_comparisons) + self.n
            new_comparisons = torch.cat((new_comparisons, shifted_comp), dim=-2)

            # TODO: be smart about how we can update covar matrix here
            new_model.set_train_data(new_datapoints, new_comparisons, update_model=True)

        return new_model


class PairwiseLaplaceMarginalLogLikelihood(MarginalLogLikelihood):
    r"""Laplace-approximated marginal log likelihood/evidence for PairwiseGP

    See (12) from [Chu2005preference]_.
    """

    def __init__(self, likelihood, model: GP):
        """
        Args:
            likelihood: Used as in args to GPyTorch MarginalLogLikelihood
            model: Used as in args to GPyTorch MarginalLogLikelihood
        """
        super().__init__(likelihood, model)

    def forward(self, post: Posterior, comp: Tensor) -> Tensor:
        r"""Calculate approximated log evidence, i.e., log(P(D|theta))

        Note that post will be based on the consolidated/deduped datapoints for
        numerical stability, but comp will still be the unconsolidated comparisons
        so that it's still compatible with fit_gpytorch_*.

        Args:
            post: training posterior distribution from self.model (after consolidation)
            comp: Comparisons pairs (before consolidation)

        Returns:
            The approximated evidence, i.e., the marginal log likelihood
        """

        model = self.model
        likelihood = self.likelihood
        if comp is not model.unconsolidated_comparisons:
            raise RuntimeError("Must train on training data")

        f_map = post.mean.squeeze(-1)

        log_likelihood = likelihood.log_p(utility=f_map, D=model.D)
        neg_log_likelihood_sum = -(torch.sum(log_likelihood, dim=-1))

        # 1/2 f_map^T @ covar_inv @ f_map
        inv_prod = torch.cholesky_solve(f_map.unsqueeze(-1), model.covar_chol)
        log_prior = 0.5 * (f_map.unsqueeze(-2) @ inv_prod).squeeze(-1).squeeze(-1)
        log_posterior = neg_log_likelihood_sum + log_prior
        # log_posterior is the S loss function in [Chu2005preference]_
        log_posterior = -log_posterior.clamp(min=0)

        mll = model.covar @ model.likelihood_hess
        mll = mll + torch.diag_embed(
            torch.ones(mll.shape[:-1], device=mll.device, dtype=mll.dtype)
        )
        mll = -0.5 * torch.logdet(mll)

        mll = mll + log_posterior

        # Sum up mll first so that when adding parameter prior probs it won't
        # propagate and double count
        mll = mll.sum()

        # Add log probs of priors on the (functions of) parameters
        for _, module, prior, closure, _ in self.named_priors():
            mll = mll.add(prior.log_prob(closure(module)).sum())

        return mll
