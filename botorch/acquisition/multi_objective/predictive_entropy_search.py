#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for predictive entropy search for multi-objective Bayesian
optimization (PES). The code does not support constraint handling.

NOTE: The PES acquisition might not be differentiable. As a result, we recommend
optimizing the acquisition function using finite differences.

References:

.. [Garrido-Merchan2019]
    E. Garrido-Merchan and D. Hernandez-Lobato. Predictive Entropy Search for
    Multi-objective Bayesian Optimization with Constraints. Neurocomputing. 2019.
    The computation follows the procedure described in the supplementary material:
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219308525

"""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import InputDataError
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.utils import check_no_nans
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor
from torch.distributions import Normal


class qMultiObjectivePredictiveEntropySearch(AcquisitionFunction):
    r"""The acquisition function for Predictive Entropy Search. The code supports
    both single and multiple objectives as well as batching.

    This acquisition function approximates the mutual information between the
    observation at a candidate point ``X`` and the Pareto optimal input using the
    moment-matching procedure known as expectation propagation (EP).

    See the Appendix of [Garrido-Merchan2019]_ for the description of the EP
    procedure.

    IMPORTANT NOTES:
    (i) The PES acquisition function estimated using EP is sometimes not
    differentiable, and therefore we advise using a finite-difference estimate of
    the gradient as opposed to the gradients identified using automatic
    differentiation, which occasionally outputs ``nan`` values.

    The source of this differentiability is in the ``_update_damping`` function, which
    finds the damping factor ``a`` that is used to update the EP parameters
    ``a * param_new + (1 - a) * param_old``. The damping factor has to ensure
    that the updated covariance matrices, ``a * cov_f_new + (1 - a) cov_f_old``, is
    positive semi-definiteness. We follow the original paper, which identifies
    ``a`` via a successive halving scheme i.e. we check ``a=1`` then ``a=0.5`` etc. This
    procedure means ``a`` is a function of the test input ``X``. This function is not
    differentiable  in ``X``.

    (ii) EP could potentially fail for a number of reasons:

        (a) When the sampled Pareto optimal points ``x_p`` is poor compared to the
        training or testing data ``x_n``.

        (b) When the training or testing data ``x_n`` is close the Pareto optimal
        points ``x_p``.

        (c) When the convergence threshold is set too small.


        Problem (a) occurs because we have to compute the variable:
        ``alpha = (mean(x_n) - mean(x_p)) / std(x_n - x_p)``, which becomes very
        large when ``x_n`` is better than ``x_p`` with high-probability. This leads to a
        log(0) error when we compute ``log(1 - cdf(alpha))``. We have preemptively
        clamped some values depending on ``alpha`` in order to mitigate this.

        Problem (b) occurs because we have to compute matrix inverses for the
        two-dimensional marginals (x_n, x_p). To address this we manually add jitter
        to the diagonal of the covariance matrix i.e. ``ep_jitter`` when training and
        ``test_jitter`` when testing. The default choice is not always appropriate
        because the same jitter is used for the inversion of the covariance
        and precision matrix, which are on different scales.

        TODO: come up with strategy to adaptively update the jitter.

        Problem (c) occurs because a smaller threshold usually means that more EP
        iterations are required. Running too many EP iterations could lead to
        invertibility problems such as in problem (b). Setting a larger threshold
        or reducing the number of EP iterations could alleviate this.

    (iii) The estimated acquisition value could be negative.
    """

    def __init__(
        self,
        model: Model,
        pareto_sets: Tensor,
        maximize: bool = True,
        X_pending: Tensor | None = None,
        max_ep_iterations: int = 250,
        ep_jitter: float = 1e-4,
        test_jitter: float = 1e-4,
        threshold: float = 1e-2,
    ) -> None:
        r"""Multi-objective predictive entropy search acquisition function.

        Args:
            model: A fitted batched model with ``M`` number of outputs.
            pareto_sets: A ``num_pareto_samples x P x d``-dim tensor containing the
                Pareto optimal set of inputs, where ``P`` is the number of pareto
                optimal points. The points in each sample have to be discrete
                otherwise expectation propagation will fail.
            maximize: If true, we consider a maximization problem.
            X_pending: A ``m x d``-dim Tensor of ``m`` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            max_ep_iterations: The maximum number of expectation propagation
                iterations. (The minimum number of iterations is set at 3.)
            ep_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update during the training
                phase.
            test_jitter: The amount of jitter added for the matrix inversion that
                occurs during the expectation propagation update in the testing
                phase.
            threshold: The convergence threshold for expectation propagation. This
                assesses the relative change in the mean and covariance. We default
                to one percent change i.e. ``threshold = 1e-2``.
        """
        super().__init__(model=model)

        self.model = model
        self.maximize = maximize
        self.set_X_pending(X_pending)

        if model.num_outputs > 1 or isinstance(model, ModelListGP):
            train_X = self.model.train_inputs[0][0]
        else:
            train_X = self.model.train_inputs[0]

        # Batch GP models (e.g. fantasized models) are not currently supported
        if train_X.ndim > 2:
            raise NotImplementedError(
                "Batch GP models (e.g. fantasized models) are not supported."
            )

        if pareto_sets.ndim != 3 or pareto_sets.shape[-1] != train_X.shape[-1]:
            raise UnsupportedError(
                "The Pareto set should have a shape of "
                "`num_pareto_samples x num_pareto_points x input_dim`."
            )
        else:
            self.pareto_sets = pareto_sets

        # add the pareto set to the existing training data
        self.num_pareto_samples = pareto_sets.shape[0]

        self.augmented_X = torch.cat(
            [train_X.repeat(self.num_pareto_samples, 1, 1), self.pareto_sets], dim=-2
        )
        self.max_ep_iterations = max_ep_iterations
        self.ep_jitter = ep_jitter
        self.test_jitter = test_jitter
        self.threshold = threshold
        self._expectation_propagation()

    def _expectation_propagation(self) -> None:
        r"""Perform expectation propagation to obtain the covariance factors that
        depend on the Pareto sets.

        The updates are performed in the natural parameter space. For a multivariate
        normal distribution with mean mu and covariance Sigma, we call Sigma^{-1}
        the natural covariance and Sigma^{-1} mu the natural mean.
        """
        ###########################################################################
        # INITIALIZATION
        ###########################################################################
        M = self.model.num_outputs

        if self.model.num_outputs > 1 or isinstance(self.model, ModelListGP):
            train_X = self.model.train_inputs[0][0]
        else:
            train_X = self.model.train_inputs[0]

        tkwargs = {"dtype": train_X.dtype, "device": train_X.device}
        N = len(train_X)
        num_pareto_samples = self.num_pareto_samples
        P = self.pareto_sets.shape[-2]

        # initialize the predictive natural mean and variances
        (
            pred_nat_mean,
            pred_nat_cov,
            pred_mean,
            pred_cov,
        ) = _initialize_predictive_matrices(
            X=self.augmented_X,
            model=self.model,
            observation_noise=False,
            jitter=self.ep_jitter,
            natural=True,
        )

        pred_f_mean = pred_mean[..., 0:M, :]
        pred_f_nat_mean = pred_nat_mean[..., 0:M, :]
        pred_f_cov = pred_cov[..., 0:M, :, :]
        pred_f_nat_cov = pred_nat_cov[..., 0:M, :, :]

        # initialize the marginals
        # ``num_pareto_samples x M x (N + P)``
        mean_f = pred_f_mean.clone()
        nat_mean_f = pred_f_nat_mean.clone()
        # ``num_pareto_samples x M x (N + P) x (N + P)``
        cov_f = pred_f_cov.clone()
        nat_cov_f = pred_f_nat_cov.clone()

        # initialize omega the function which encodes the fact that the pareto points
        # are optimal in the feasible space i.e. any point in the feasible space
        # should not dominate the Pareto efficient points.

        # ``num_pareto_samples x M x (N + P) x P x 2``
        omega_f_nat_mean = torch.zeros((num_pareto_samples, M, N + P, P, 2), **tkwargs)
        # ``num_pareto_samples x M x (N + P) x P x 2 x 2``
        omega_f_nat_cov = torch.zeros(
            (num_pareto_samples, M, N + P, P, 2, 2), **tkwargs
        )

        ###########################################################################
        # EXPECTATION PROPAGATION
        ###########################################################################
        damping = torch.ones(num_pareto_samples, M, **tkwargs)

        iteration = 0
        while (torch.sum(damping) > 0) and (iteration < self.max_ep_iterations):
            # Compute the new natural mean and covariance
            ####################################################################
            # OBJECTIVE FUNCTION: OMEGA UPDATE
            ####################################################################
            omega_f_nat_mean_new, omega_f_nat_cov_new = _safe_update_omega(
                mean_f=mean_f,
                cov_f=cov_f,
                omega_f_nat_mean=omega_f_nat_mean,
                omega_f_nat_cov=omega_f_nat_cov,
                N=N,
                P=P,
                M=M,
                maximize=self.maximize,
                jitter=self.ep_jitter,
            )

            ####################################################################
            # OBJECTIVE FUNCTION: MARGINAL UPDATE
            ####################################################################
            nat_mean_f_new, nat_cov_f_new = _update_marginals(
                pred_f_nat_mean=pred_f_nat_mean,
                pred_f_nat_cov=pred_f_nat_cov,
                omega_f_nat_mean=omega_f_nat_mean_new,
                omega_f_nat_cov=omega_f_nat_cov_new,
                N=N,
                P=P,
            )
            ########################################################################
            # OBJECTIVE FUNCTION: DAMPING UPDATE
            ########################################################################
            # update damping of objectives
            damping, cholesky_nat_cov_f = _update_damping(
                nat_cov=nat_cov_f,
                nat_cov_new=nat_cov_f_new,
                damping_factor=damping,
                jitter=self.ep_jitter,
            )
            check_no_nans(cholesky_nat_cov_f)
            ########################################################################
            # OBJECTIVE FUNCTION: DAMPED UPDATE
            ########################################################################
            # Damp update of omega
            omega_f_nat_mean = _damped_update(
                old_factor=omega_f_nat_mean,
                new_factor=omega_f_nat_mean_new,
                damping_factor=damping,
            )

            omega_f_nat_cov = _damped_update(
                old_factor=omega_f_nat_cov,
                new_factor=omega_f_nat_cov_new,
                damping_factor=damping,
            )
            # update the mean and covariance
            nat_mean_f = _damped_update(
                old_factor=nat_mean_f, new_factor=nat_mean_f_new, damping_factor=damping
            )
            nat_cov_f = _damped_update(
                old_factor=nat_cov_f, new_factor=nat_cov_f_new, damping_factor=damping
            )

            # compute cholesky inverse
            cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f)
            mean_f_new = torch.einsum("...ij,...j->...i", cov_f_new, nat_mean_f)
            check_no_nans(cov_f_new)
            ########################################################################
            # OBJECTIVE FUNCTION: CONVERGENCE UPDATE
            ########################################################################
            # Set the damping to zero when the change in the mean and
            # covariance is less than the threshold
            damping, delta_mean_f, delta_cov_f = _update_damping_when_converged(
                mean_old=mean_f,
                mean_new=mean_f_new,
                cov_old=cov_f,
                cov_new=cov_f_new,
                damping_factor=damping,
                threshold=self.threshold,
                iteration=iteration,
            )
            cov_f = cov_f_new
            mean_f = mean_f_new
            iteration = iteration + 1

        ############################################################################
        # SAVE OMEGA AND PHI FACTORS
        ############################################################################
        check_no_nans(omega_f_nat_mean)
        check_no_nans(omega_f_nat_cov)
        # save phi and omega for the forward
        self._omega_f_nat_mean = omega_f_nat_mean
        self._omega_f_nat_cov = omega_f_nat_cov

    def _compute_information_gain(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiObjectivePredictiveEntropySearch on the candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of Predictive Entropy Search values at the
            given design points ``X``.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        batch_shape = X.shape[0:-2]
        q = X.shape[-2]
        M = self.model.num_outputs

        if M > 1 or isinstance(self.model, ModelListGP):
            N = len(self.model.train_inputs[0][0])
        else:
            N = len(self.model.train_inputs[0])
        P = self.pareto_sets.shape[-2]
        num_pareto_samples = self.num_pareto_samples
        ###########################################################################
        # AUGMENT X WITH THE SAMPLED PARETO SET
        ###########################################################################
        new_shape = batch_shape + torch.Size([num_pareto_samples]) + X.shape[-2:]
        expanded_X = X.unsqueeze(-3).expand(new_shape)
        expanded_ps = self.pareto_sets.expand(X.shape[0:-2] + self.pareto_sets.shape)
        # ``batch_shape x num_pareto_samples x (q + P) x d``
        aug_X = torch.cat([expanded_X, expanded_ps], dim=-2)

        ###########################################################################
        # COMPUTE THE POSTERIORS AND OBSERVATION NOISE
        ###########################################################################
        # compute predictive distribution without observation noise
        (
            pred_nat_mean,
            pred_nat_cov,
            pred_mean,
            pred_cov,
        ) = _initialize_predictive_matrices(
            X=aug_X,
            model=self.model,
            observation_noise=True,
            jitter=self.test_jitter,
            natural=True,
        )

        pred_f_mean = pred_mean[..., 0:M, :]
        pred_f_nat_mean = pred_nat_mean[..., 0:M, :]
        pred_f_cov = pred_cov[..., 0:M, :, :]
        pred_f_nat_cov = pred_nat_cov[..., 0:M, :, :]

        (_, _, _, pred_cov_noise) = _initialize_predictive_matrices(
            X=aug_X,
            model=self.model,
            observation_noise=True,
            jitter=self.test_jitter,
            natural=False,
        )

        pred_f_cov_noise = pred_cov_noise[..., 0:M, :, :]
        observation_noise = pred_f_cov_noise - pred_f_cov
        ###########################################################################
        # INITIALIZE THE EP FACTORS
        ###########################################################################
        # ``batch_shape x num_pareto_samples x M x (q + P) x P x 2``
        omega_f_nat_mean = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2]), **tkwargs
        )
        # ``batch_shape x num_pareto_samples x M x (q + P) x P x 2 x 2``
        omega_f_nat_cov = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2, 2]), **tkwargs
        )
        ###########################################################################
        # RUN EP ONCE
        ###########################################################################
        # run update omega once
        omega_f_nat_mean, omega_f_nat_cov = _safe_update_omega(
            mean_f=pred_f_mean,
            cov_f=pred_f_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
            M=M,
            maximize=self.maximize,
            jitter=self.test_jitter,
        )
        ###########################################################################
        # ADD THE CACHE FACTORS BACK
        ###########################################################################
        omega_f_nat_mean, omega_f_nat_cov = _augment_factors_with_cached_factors(
            q=q,
            N=N,
            omega_f_nat_mean=omega_f_nat_mean,
            cached_omega_f_nat_mean=self._omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            cached_omega_f_nat_cov=self._omega_f_nat_cov,
        )
        ###########################################################################
        # COMPUTE THE MARGINAL
        ###########################################################################
        nat_mean_f, nat_cov_f = _update_marginals(
            pred_f_nat_mean=pred_f_nat_mean,
            pred_f_nat_cov=pred_f_nat_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
        )
        ###########################################################################
        # COMPUTE THE DAMPED UPDATE
        ###########################################################################
        # # update damping of objectives
        damping = torch.ones(
            batch_shape + torch.Size([num_pareto_samples, M]), **tkwargs
        )
        damping, cholesky_nat_cov_f_new = _update_damping(
            nat_cov=pred_f_nat_cov,
            nat_cov_new=nat_cov_f,
            damping_factor=damping,
            jitter=self.test_jitter,
        )

        # invert matrix
        cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f_new)
        check_no_nans(cov_f_new)

        ###########################################################################
        # COMPUTE THE LOG DETERMINANTS
        ###########################################################################
        # compute the initial log determinant term
        log_det_pred_f_cov_noise = _compute_log_determinant(cov=pred_f_cov_noise, q=q)
        # compute the post log determinant term
        log_det_cov_f = _compute_log_determinant(cov=cov_f_new + observation_noise, q=q)

        ###########################################################################
        # COMPUTE THE ACQUISITION FUNCTION
        ###########################################################################
        q_pes_f = log_det_pred_f_cov_noise - log_det_cov_f
        check_no_nans(q_pes_f)

        return 0.5 * q_pes_f

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiObjectivePredictiveEntropySearch on the candidate set ``X``.

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A ``batch_shape'``-dim Tensor of acquisition values at the given design
            points ``X``.
        """
        return self._compute_information_gain(X)


def log_cdf_robust(x: Tensor) -> Tensor:
    r"""Computes the logarithm of the normal cumulative density robustly. This uses
    the approximation log(1-z) ~ -z when z is small:

    if x > 5:
        log(cdf(x)) = log(1-cdf(-x)) approx -cdf(-x)
    else:
        log(cdf(x)).

    Args:
        x: a ``x_shape``-dim Tensor.

    Returns
        A ``x_shape``-dim Tensor.
    """
    CLAMP_LB = torch.finfo(x.dtype).eps
    NEG_INF = torch.finfo(x.dtype).min
    normal = Normal(torch.zeros_like(x), torch.ones_like(x))
    cdf_x = normal.cdf(x)
    neg_cdf_neg_x = -normal.cdf(-x)
    log_cdf_x = torch.where(x < 5, torch.log(cdf_x), neg_cdf_neg_x)

    return log_cdf_x.clamp(NEG_INF, -CLAMP_LB)


def _initialize_predictive_matrices(
    X: Tensor,
    model: Model,
    observation_noise: bool = True,
    jitter: float = 1e-4,
    natural: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Initializes the natural predictive mean and covariance matrix. For a
    multivariate normal distribution with mean mu and covariance Sigma, the natural
    mean is Sigma^{-1} mu and the natural covariance is Sigma^{-1}.

    Args:
        X: A ``batch_shape x R x d``-dim Tensor.
        model: The fitted model.
        observation_noise: If true, the posterior is computed with observation noise.
        jitter: The jitter added to the covariance matrix.
        natural: If true, we compute the natural statistics as well.

    Return:
        A four-element tuple containing

        - pred_nat_mean: A ``batch_shape x num_outputs x R ``-dim Tensor containing the
            predictive natural mean vectors.
        - pred_nat_cov: A ``batch_shape x num_outputs x R x R``-dim Tensor containing
            the predictive natural covariance matrices.
        - pred_mean: A ``batch_shape x num_outputs x R``-dim Tensor containing the
            predictive mean vectors.
        - pred_cov: A ``batch_shape x num_outputs x R x R``-dim Tensor containing the
            predictive covariance matrices.
    """
    tkwargs = {"dtype": X.dtype, "device": X.device}
    # compute the predictive mean and covariances at X
    posterior = model.posterior(X, observation_noise=observation_noise)

    # ``batch_shape x (R * num_outputs) x (R * num_outputs)``
    init_pred_cov = posterior.mvn.covariance_matrix
    num_outputs = model.num_outputs
    R = int(init_pred_cov.shape[-1] / num_outputs)
    pred_cov = [
        init_pred_cov[..., (m * R) : ((m + 1) * R), (m * R) : ((m + 1) * R)].unsqueeze(
            -1
        )
        for m in range(num_outputs)
    ]
    # ``batch_shape x R x R x num_outputs`` (before swap axes)
    # ``batch_shape x num_outputs x R * R``
    pred_cov = torch.cat(pred_cov, axis=-1).swapaxes(-2, -1).swapaxes(-3, -2)
    identity = torch.diag_embed(torch.ones(pred_cov.shape[:-1], **tkwargs))
    pred_cov = pred_cov + jitter * identity

    # ``batch_shape x num_outputs x R``
    pred_mean = posterior.mean.swapaxes(-2, -1)

    #############################################################
    if natural:
        # natural parameters
        # ``batch_shape x num_outputs x R x R``
        cholesky_pred_cov, _ = torch.linalg.cholesky_ex(pred_cov)
        pred_nat_cov = torch.cholesky_inverse(cholesky_pred_cov)

        # ``batch_shape x num_outputs x R``
        pred_nat_mean = torch.einsum("...ij,...j->...i", pred_nat_cov, pred_mean)

        return pred_nat_mean, pred_nat_cov, pred_mean, pred_cov
    else:
        return None, None, pred_mean, pred_cov


def _get_omega_f_contribution(
    mean: Tensor, cov: Tensor, N: int, P: int, M: int
) -> tuple[Tensor, Tensor]:
    r"""Extract the mean vector and covariance matrix corresponding to the ``2 x 2``
    multivariate normal blocks in the objective model between the points in ``X`` and
    the Pareto optimal set.

    [There is likely a more efficient way to do this.]

    Args:
        mean: A ``batch_shape x M x (N + P)``-dim Tensor containing the natural
            mean matrix for the objectives.
        cov: A ``batch_shape x M x (N + P) x (N + P)``-dim Tensor containing
            the natural mean matrix for the objectives.
        N: The number of design points.
        P: The number of Pareto optimal points.
        M: The number of objectives.

    Return:
        A two-element tuple containing

        - mean_fX_fS: A ``batch_shape x M x (N + P) x P x 2``-dim Tensor containing the
            means of the inputs and Pareto optimal points.
        - cov_fX_fS: A ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor containing
            the covariances between the inputs and Pareto optimal points.
    """
    tkwargs = {"dtype": mean.dtype, "device": mean.device}
    batch_shape = mean.shape[:-2]
    # ``batch_shape x M x (N + P) x P x 2 x 2``
    cov_fX_fS = torch.zeros(batch_shape + torch.Size([M, N + P, P, 2, 2]), **tkwargs)
    # ``batch_shape x M x (N + P) x P x 2``
    mean_fX_fS = torch.zeros(batch_shape + torch.Size([M, N + P, P, 2]), **tkwargs)

    # ``batch_shape x M x (N + P) x P``
    mean_fX_fS[..., 0] = mean.unsqueeze(-1).expand(mean.shape + torch.Size([P]))
    # ``batch_shape x M x (N + P) x P``
    mean_fX_fS[..., 1] = (
        mean[..., N:].unsqueeze(-2).expand(mean.shape + torch.Size([P]))
    )
    # ``batch_shape x M x (N + P) x P``
    cov_fX_fS[..., 0, 0] = (
        cov[..., range(N + P), range(N + P)]
        .unsqueeze(-1)
        .expand(batch_shape + torch.Size([M, N + P, P]))
    )
    # ``batch_shape x M x (N + P) x P``
    cov_fX_fS[..., 1, 1] = (
        cov[..., range(N, N + P), range(N, N + P)]
        .unsqueeze(-2)
        .expand(batch_shape + torch.Size([M, N + P, P]))
    )

    for p in range(P):
        # ``batch_shape x M x (N + P)``
        cov_p = cov[..., range(N + P), N + p]
        cov_fX_fS[..., p, 0, 1] = cov_p
        cov_fX_fS[..., p, 1, 0] = cov_p

    return mean_fX_fS, cov_fX_fS


def _replace_pareto_diagonal(A: Tensor) -> Tensor:
    """Replace the pareto diagonal with identity matricx.

    The Pareto diagonal of the omega factor shouldn't be updated because does not
    contribute anything: ``omega(x_p, x_p) = 1`` for any pareto optimal input ``x_p``.

    Args:
        A: a ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor.

    Returns:
        A ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor, where the Pareto
        diagonal is padded with identity matrices.
    """
    tkwargs = {"dtype": A.dtype, "device": A.device}
    batch_shape = A.shape[:-5]
    P = A.shape[-3]
    N = A.shape[-4] - P
    M = A.shape[-5]
    identity = torch.diag_embed(torch.ones(batch_shape + torch.Size([M, 2]), **tkwargs))
    for p in range(P):
        A[..., N + p, p, :, :] = identity

    return A


def _update_omega(
    mean_f: Tensor,
    cov_f: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
    M: int,
    maximize: bool = True,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""Computes the new omega factors by matching the moments.

    Args:
        mean_f: A ``batch_shape x M x (N + P)``-dim Tensor containing the mean vector
            for the objectives.
        cov_f: A ``batch_shape x M x (N + P) x (N + P)``-dim Tensor containing the
            covariance matrix for the objectives.
        omega_f_nat_mean: A ``batch_shape x M x (N + P) x P x 2``-dim Tensor containing
            the omega natural mean factors for the objective matrix.
        omega_f_nat_cov: A ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor
            containing the omega natural covariance factors for the objective matrix.
        N: The number of design points.
        M: The number of Pareto optimal points.
        M: The number of objectives.
        maximize: If true, we consider the Pareto maximum domination relation.
        jitter: The jitter for the matrix inverse.

    Return:
        A two-element tuple containing

        - omega_f_nat_mean_new: A ``batch_shape x M x (N + P) x P x 2`` containing the
            new omega natural mean factors for the objective matrix.
        - omega_f_nat_cov_new: A ``batch_shape x M x (N + P) x P x 2 x 2`` containing
            the new omega natural covariance factors for the objective matrix.
    """
    tkwargs = {"dtype": mean_f.dtype, "device": mean_f.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
    NEG_INF = torch.finfo(tkwargs["dtype"]).min
    weight = 1.0 if maximize else -1.0
    ###############################################################################
    # EXTRACT THE NECESSARY COMPONENTS
    ###############################################################################
    # ``batch_shape x M x (N + P) x P x 2``-dim mean
    # ``batch_shape x M x (N + P) x P x 2 x 2``-dim covariance
    mean_fX_fS, cov_fX_fS = _get_omega_f_contribution(mean_f, cov_f, N, P, M)
    identity = torch.diag_embed(torch.ones(cov_fX_fS.shape[:-1], **tkwargs))
    # remove the Pareto diagonal
    cov_fX_fS = _replace_pareto_diagonal(cov_fX_fS + jitter * identity)
    nat_cov_fX_fS = torch.inverse(cov_fX_fS)
    nat_mean_fX_fS = torch.einsum("...ij,...j->...i", nat_cov_fX_fS, mean_fX_fS)

    ###############################################################################
    # COMPUTE THE CAVITIES
    ###############################################################################
    # cavity distribution
    # natural parameters
    cav_nat_mean_f = nat_mean_fX_fS - omega_f_nat_mean
    cav_nat_cov_f = nat_cov_fX_fS - omega_f_nat_cov

    # transform to standard parameters
    # remove the Pareto diagonal
    cav_nat_cov_f = _replace_pareto_diagonal(cav_nat_cov_f)
    identity = torch.diag_embed(torch.ones(cav_nat_cov_f.shape[:-1], **tkwargs))
    cav_cov_f = torch.inverse(cav_nat_cov_f + jitter * identity)

    cav_mean_f = torch.einsum("...ij,...j->...i", cav_cov_f, cav_nat_mean_f)

    ###############################################################################
    # COMPUTE THE NORMALIZATION CONSTANT
    ###############################################################################
    # ``batch_shape x M x (N + P) x P``
    # Equation 29
    cav_var_fX_minus_fS = (
        cav_cov_f[..., 0, 0] + cav_cov_f[..., 1, 1] - 2 * cav_cov_f[..., 0, 1]
    ).clamp_min(CLAMP_LB)
    cav_std_fX_minus_fS = torch.sqrt(cav_var_fX_minus_fS).clamp_min(CLAMP_LB)

    # ``batch_shape x M x (N + P) x P``
    cav_mean_fX_minus_fS = weight * (cav_mean_f[..., 0] - cav_mean_f[..., 1])

    # Equation 30
    cav_alpha = cav_mean_fX_minus_fS / cav_std_fX_minus_fS
    # compute alpha pdf and cdf
    normal_alpha = Normal(torch.zeros_like(cav_alpha), torch.ones_like(cav_alpha))
    # ``batch_shape x M x (N + P) x P``
    cav_alpha_log_cdf = log_cdf_robust(cav_alpha)
    # ``batch_shape x M x (N + P) x P``
    cav_alpha_log_pdf = normal_alpha.log_prob(cav_alpha).clamp_min(NEG_INF)
    # ``batch_shape x (N + P) x P``
    cav_sum_alpha_log_cdf = torch.sum(cav_alpha_log_cdf, dim=-3).clamp_min(NEG_INF)

    # compute normalization constant Z
    # Equation 35
    cav_log_zeta = torch.log1p(-torch.exp(cav_sum_alpha_log_cdf)).clamp_min(NEG_INF)

    # Need to clamp log values to prevent ``exp(-inf) = nan``
    cav_logZ = cav_log_zeta

    # Equation 40 [first bit]
    # ``batch_shape x (N + P) x P``
    cav_log_rho = -cav_logZ + cav_sum_alpha_log_cdf

    # Equation 40 [second bit]
    # ``batch_shape x M x (N + P) x P``
    cav_log_rho = cav_log_rho.unsqueeze(-3) - cav_alpha_log_cdf + cav_alpha_log_pdf
    cav_rho = -torch.exp(cav_log_rho).clamp(NEG_INF, -NEG_INF)
    ###############################################################################
    # COMPUTE THE PARTIAL DERIVATIVES
    ###############################################################################
    # ``batch_shape x M x (N + P) x P x 2``
    # Final vector: ``[1, -1]``
    ones_mean = torch.ones(cav_mean_f.shape, **tkwargs)
    ones_mean[..., 1] = -ones_mean[..., 1]

    # ``batch_shape x M x (N + P) x P x 2 x 2``
    # Final matrix: ``[[1, -1], [-1, 1]]``
    ones_cov = torch.ones(cav_cov_f.shape, **tkwargs)
    ones_cov[..., 0, 1] = -ones_cov[..., 0, 1]
    ones_cov[..., 1, 0] = -ones_cov[..., 1, 0]

    # first partial derivation of the log Z with respect to the mean
    # assuming maximization (this is also where the sign will change)
    # Equation 41
    cav_dlogZ_dm = cav_rho / cav_std_fX_minus_fS
    cav_dlogZ_dm = weight * cav_dlogZ_dm.unsqueeze(-1) * ones_mean

    # second partial derivation of the log Z with respect to the mean
    # Equation 42
    cav_d2logZ_dm2 = -cav_rho * (cav_rho + cav_alpha) / cav_var_fX_minus_fS
    cav_d2logZ_dm2 = cav_d2logZ_dm2.unsqueeze(-1).unsqueeze(-1) * ones_cov

    ###############################################################################
    # COMPUTE THE NEW MEAN AND COVARIANCE
    ###############################################################################
    # compute the new mean and covariance
    cav_updated_mean_f = cav_mean_f + torch.einsum(
        "...ij,...j->...i", cav_cov_f, cav_dlogZ_dm
    )
    cav_updated_cov_f = cav_cov_f + torch.einsum(
        "...ij,...jk,...kl->...il", cav_cov_f, cav_d2logZ_dm2, cav_cov_f
    )
    # transform to natural parameters
    # remove the Pareto diagonal
    cav_updated_cov_f = _replace_pareto_diagonal(cav_updated_cov_f)

    identity = torch.diag_embed(torch.ones(cav_updated_cov_f.shape[:-1], **tkwargs))
    cav_updated_nat_cov_f = torch.inverse(cav_updated_cov_f + jitter * identity)

    cav_updated_nat_mean_f = torch.einsum(
        "...ij,...j->...i", cav_updated_nat_cov_f, cav_updated_mean_f
    )

    # match the moments to compute the gain
    omega_f_nat_mean_new = cav_updated_nat_mean_f - cav_nat_mean_f
    omega_f_nat_cov_new = cav_updated_nat_cov_f - cav_nat_cov_f

    # it is also possible to calculate the update directly as in the original paper:
    # identity = torch.diag_embed(torch.ones(cav_d2logZ_dm2.shape[:-1], **tkwargs))
    # denominator = torch.inverse(cav_cov_f @ cav_d2logZ_dm2 + identity)
    # omega_f_nat_cov_new = - cav_d2logZ_dm2 @ denominator
    # omega_f_nat_mean_new = torch.einsum(
    #     '...ij,...j->...i', denominator,
    #     cav_dlogZ_dm - torch.einsum('...ij,...j->...i', cav_d2logZ_dm2, cav_mean_f)
    # )

    return omega_f_nat_mean_new, omega_f_nat_cov_new


def _safe_update_omega(
    mean_f: Tensor,
    cov_f: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
    M: int,
    maximize: bool = True,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    r"""Try to update the new omega factors by matching the moments. If the update
    is not possible then this returns the initial omega factors.

    Args:
        mean_f: A ``batch_shape x M x (N + P)``-dim Tensor containing the mean vector
            for the objectives.
        cov_f: A ``batch_shape x M x (N + P) x (N + P)``-dim Tensor containing the
            covariance matrix for the objectives.
        omega_f_nat_mean: A ``batch_shape x M x (N + P) x P x 2``-dim Tensor containing
            the omega natural mean factors for the objective matrix.
        omega_f_nat_cov: A ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor
            containing the omega natural covariance factors for the objective
            matrix.
        N: The number of design points.
        M: The number of Pareto optimal points.
        M: The number of objectives.
        maximize: If true, we consider the Pareto maximum domination relation.
        jitter: The jitter for the matrix inverse.

    Return:
        A two-element tuple containing

        - omega_f_nat_mean_new: A ``batch_shape x M x (N + P) x P x 2`` containing the
            new omega natural mean factors for the objective matrix.
        - omega_f_nat_cov_new: A ``batch_shape x M x (N + P) x P x 2 x 2`` containing
            the new omega natural covariance factors for the objective matrix.
    """
    try:
        omega_f_nat_mean_new, omega_f_nat_cov_new = _update_omega(
            mean_f=mean_f,
            cov_f=cov_f,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=N,
            P=P,
            M=M,
            maximize=maximize,
            jitter=jitter,
        )
        check_no_nans(omega_f_nat_mean_new)
        check_no_nans(omega_f_nat_cov_new)
        return omega_f_nat_mean_new, omega_f_nat_cov_new

    except (RuntimeError, InputDataError):
        return omega_f_nat_mean, omega_f_nat_cov


def _update_marginals(
    pred_f_nat_mean: Tensor,
    pred_f_nat_cov: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
) -> tuple[Tensor, Tensor]:
    r"""Computes the new marginal by summing up all the natural factors.

    Args:
        pred_f_nat_mean: A ``batch_shape x M x (N + P)``-dim Tensor containing the
            natural predictive mean matrix for the objectives.
        pred_f_nat_cov: A ``batch_shape x M x (N + P) x (N + P)``-dim Tensor containing
            the natural predictive covariance matrix for the objectives.
        omega_f_nat_mean: A ``batch_shape x M x (N + P) x P x 2``-dim Tensor containing
            the omega natural mean factors for the objective matrix.
        omega_f_nat_cov: A ``batch_shape x M x (N + P) x P x 2 x 2``-dim Tensor
            containing the omega natural covariance factors for the objective matrix.
        N: The number of design points.
        P: The number of Pareto optimal points.

    Returns:
        A two-element tuple containing

        - nat_mean_f: A ``batch_shape x M x (N + P)``-dim Tensor containing the updated
            natural mean matrix for the objectives.
        - nat_cov_f: A ``batch_shape x M x (N + P) x (N + P)``-dim Tensor containing
            the updated natural predictive covariance matrix for the objectives.
    """

    # ``batch_shape x M x (N + P)``
    nat_mean_f = pred_f_nat_mean.clone()
    # `batch_shape x M x (N + P) x (N + P)
    nat_cov_f = pred_f_nat_cov.clone()

    ################################################################################
    # UPDATE THE OBJECTIVES
    ################################################################################
    # remove Pareto diagonal
    # zero out the diagonal
    omega_f_nat_mean[..., range(N, N + P), range(P), :] = 0
    omega_f_nat_cov[..., range(N, N + P), range(P), :, :] = 0

    # ``batch_shape x M x (N + P)``
    # sum over the pareto dim
    nat_mean_f = nat_mean_f + omega_f_nat_mean[..., 0].sum(dim=-1)
    # ``batch_shape x M x P``
    # sum over the data dim
    nat_mean_f[..., N:] = nat_mean_f[..., N:] + omega_f_nat_mean[..., 1].sum(dim=-2)

    # ``batch_shape x M x (N + P)``
    nat_cov_f[..., range(N + P), range(N + P)] = nat_cov_f[
        ..., range(N + P), range(N + P)
    ] + omega_f_nat_cov[..., 0, 0].sum(dim=-1)
    # ``batch_shape x M x P``
    nat_cov_f[..., range(N, N + P), range(N, N + P)] = nat_cov_f[
        ..., range(N, N + P), range(N, N + P)
    ] + omega_f_nat_cov[..., 1, 1].sum(dim=-2)

    for p in range(P):
        # ``batch_shape x M x (N + P)``
        nat_cov_f[..., range(N + P), N + p] = (
            nat_cov_f[..., range(N + P), N + p] + omega_f_nat_cov[..., p, 0, 1]
        )

        # ``batch_shape x M x (N + P)``
        nat_cov_f[..., N + p, range(N + P)] = (
            nat_cov_f[..., N + p, range(N + P)] + omega_f_nat_cov[..., p, 1, 0]
        )

    return nat_mean_f, nat_cov_f


def _damped_update(
    old_factor: Tensor,
    new_factor: Tensor,
    damping_factor: Tensor,
) -> Tensor:
    r"""Computes the damped updated for natural factor.

    Args:
        old_factor: A ``batch_shape x param_shape``-dim Tensor containing the old
            natural factor.
        new_factor: A ``batch_shape x param_shape``-dim Tensor containing the new
            natural factor.
        damping_factor: A ``batch_shape``-dim Tensor containing the damping factor.

    Returns:
        A ``batch_shape x param_shape``-dim Tensor containing the updated natural
        factor.
    """
    bs = damping_factor.shape
    fs = old_factor.shape

    df = damping_factor
    for _ in range(len(fs[len(bs) :])):
        df = df.unsqueeze(-1)

    return df * new_factor + (1 - df) * old_factor


def _update_damping(
    nat_cov: Tensor,
    nat_cov_new: Tensor,
    damping_factor: Tensor,
    jitter: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Updates the damping factor whilst ensuring the covariance matrix is positive
    definite by trying a Cholesky decomposition.

    Args:
        nat_cov: A ``batch_shape x R x R``-dim Tensor containing the old natural
            covariance matrix.
        nat_cov_new: A ``batch_shape x R x R``-dim Tensor containing the new natural
            covariance matrix.
        damping_factor: A``batch_shape``-dim Tensor containing the damping factor.
        jitter: The amount of jitter added before matrix inversion.

    Returns:
        A two-element tuple containing

        - A ``batch_shape x param_shape``-dim Tensor containing the updated damping
            factor.
        - A ``batch_shape x R x R``-dim Tensor containing the Cholesky factor.
    """
    tkwargs = {"dtype": nat_cov.dtype, "device": nat_cov.device}
    df = damping_factor
    jitter = jitter * torch.diag_embed(torch.ones(nat_cov.shape[:-1], **tkwargs))
    _, info = torch.linalg.cholesky_ex(nat_cov + jitter)

    if torch.sum(info) > 1:
        raise ValueError(
            "The previous covariance is not positive semi-definite. "
            "This usually happens if the predictive covariance is "
            "ill-conditioned and the added jitter is insufficient."
        )

    damped_nat_cov = _damped_update(
        old_factor=nat_cov, new_factor=nat_cov_new, damping_factor=df
    )
    cholesky_factor, info = torch.linalg.cholesky_ex(damped_nat_cov)
    contains_nans = torch.any(torch.isnan(cholesky_factor)).item()

    run = 0
    while torch.sum(info) > 1 or contains_nans:
        # propose an alternate damping factor which is half the original
        df_alt = 0.5 * df
        # hard threshold at 1e-3
        df_alt = torch.where(
            df_alt > 1e-3, df_alt, torch.zeros(df_alt.shape, **tkwargs)
        )
        # only change the damping factor where psd failure occurs
        df_new = torch.where(info == 0, df, df_alt)

        # new damped covariance
        damped_nat_cov = _damped_update(nat_cov, nat_cov_new, df_new)

        # try cholesky decomposition
        cholesky_factor, info = torch.linalg.cholesky_ex(damped_nat_cov + jitter)
        contains_nans = torch.any(torch.isnan(cholesky_factor)).item()
        df = df_new
        run = run + 1

    return df, cholesky_factor


def _update_damping_when_converged(
    mean_old: Tensor,
    mean_new: Tensor,
    cov_old: Tensor,
    cov_new: Tensor,
    damping_factor: Tensor,
    iteration: Tensor,
    threshold: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Set the damping factor to 0 once converged. Convergence is determined by the
    relative change in the entries of the mean and covariance matrix.

    Args:
        mean_old: A ``batch_shape x R``-dim Tensor containing the old natural mean
            matrix for the objective.
        mean_new: A ``batch_shape x R``-dim Tensor containing the new natural mean
            matrix for the objective.
        cov_old: A ``batch_shape x R x R``-dim Tensor containing the old natural
            covariance matrix for the objective.
        cov_new: A ``batch_shape x R x R``-dim Tensor containing the new natural
            covariance matrix for the objective.
        iteration: The current iteration number
        damping_factor: A ``batch_shape``-dim Tensor containing the damping factor.

    Returns:
        - A ``batch_shape x param_shape``-dim Tensor containing the updated damping
        factor.
        - Difference between ``mean_new`` and ``mean_old``
        - Difference between ``cov_new`` and ``cov_old``
    """
    df = damping_factor.clone()
    delta_mean = mean_new - mean_old
    delta_cov = cov_new - cov_old
    am = torch.amax(abs(mean_old), dim=-1)
    ac = torch.amax(abs(cov_old), dim=(-2, -1))

    if iteration > 2:
        mask_mean = torch.amax(abs(delta_mean), dim=-1) < threshold * am
        mask_cov = torch.amax(abs(delta_cov), dim=(-2, -1)) < threshold * ac
        mask = torch.logical_and(mask_mean, mask_cov)
        df[mask] = 0

    return df, delta_mean, delta_cov


def _augment_factors_with_cached_factors(
    q: int,
    N: int,
    omega_f_nat_mean: Tensor,
    cached_omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    cached_omega_f_nat_cov: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Incorporate the cached Pareto updated factors in the forward call and
    augment them with the previously computed factors.

    Args:
        q: The batch size.
        N: The number of training points.
        omega_f_nat_mean: A ``batch_shape x num_pareto_samples x M x (q + P) x P x 2``
            -dim Tensor containing the omega natural mean for the objective at ``X``.
        cached_omega_f_nat_mean: A ``num_pareto_samples x M x (N + P) x P x 2``-dim
            Tensor containing the omega natural mean for the objective at ``X``.
        omega_f_nat_cov: A `batch_shape x num_pareto_samples x M x (q + P) x P x 2
            x 2` -dim Tensor containing the omega natural covariance for the
            objective at ``X``.
        cached_omega_f_nat_cov: A ``num_pareto_samples x M x (N + P) x P x 2 x 2``-dim
            Tensor containing the omega covariance mean for the objective at ``X``.

    Returns:
        A two-element tuple containing

        - omega_f_nat_mean_new: A `batch_shape x num_pareto_samples x M x (q + P)
            x P x 2`-dim Tensor containing the omega natural mean for the objective
            at ``X``.
        - omega_f_nat_cov_new: A `batch_shape x num_pareto_samples x M x (q + P) x
            P x 2 x 2`-dim Tensor containing the omega natural covariance for the
            objective at ``X``.
    """
    ##############################################################################
    # omega_f_nat_mean
    ##############################################################################
    # retrieve the natural mean contribution of the Pareto block omega(x_p, x_p) for
    # the objective
    exp_cached_omega_f_nat_mean = cached_omega_f_nat_mean[..., N:, :, :].expand(
        omega_f_nat_mean[..., q:, :, :].shape
    )
    omega_f_nat_mean[..., q:, :, :] = exp_cached_omega_f_nat_mean
    ##############################################################################
    # omega_f_nat_cov
    ##############################################################################
    # retrieve the natural covariance contribution of the Pareto block
    # omega(x_p, x_p) for the objective
    exp_omega_f_nat_cov = cached_omega_f_nat_cov[..., N:, :, :, :].expand(
        omega_f_nat_cov[..., q:, :, :, :].shape
    )
    omega_f_nat_cov[..., q:, :, :, :] = exp_omega_f_nat_cov

    return omega_f_nat_mean, omega_f_nat_cov


def _compute_log_determinant(cov: Tensor, q: int) -> Tensor:
    r"""Computes the sum of the log determinants of a block diagonal covariance
    matrices averaged over the Pareto samples.

    Args:
        cov: A ``batch_shape x num_pareto_samples x num_outputs x (q + P) x (q + P)``
            -dim Tensor containing the covariance matrices.
        q: The batch size.

    Return:
        log_det_cov: A ``batch_shape``-dim Tensor containing the sum of the
            determinants for each output.
    """
    log_det_cov = torch.logdet(cov[..., 0:q, 0:q])
    check_no_nans(log_det_cov)

    return log_det_cov.sum(dim=-1).mean(dim=-1)
