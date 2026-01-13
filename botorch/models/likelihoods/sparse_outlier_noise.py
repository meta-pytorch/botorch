# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from warnings import warn

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.model import Model
from botorch.models.relevance_pursuit import RelevancePursuitMixin
from botorch.utils.constraints import NonTransformedInterval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, Noise
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import Prior
from linear_operator.operators import DiagLinearOperator, LinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor
from torch.nn.parameter import Parameter


class SparseOutlierGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
        self,
        base_noise: Noise | FixedGaussianNoise,
        dim: int,
        outlier_indices: list[int] | None = None,
        rho_prior: Prior | None = None,
        rho_constraint: NonTransformedInterval | None = None,
        batch_shape: torch.Size | None = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ) -> None:
        """A likelihood that models the noise of a GP with SparseOutlierNoise, a noise
        model in the Relevance Pursuit family of models, permitting additional "robust"
        variance for a small set of outlier data points. Notably, the indices of the
        outlier data points are inferred during the optimization of the associated log
        marginal likelihood via the Relevance Pursuit algorithm.

        For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

        NOTE: Letting base_noise also use the non-transformed constraints, will lead
        to more stable optimization, but is orthogonal implementation-wise. If the base
        noise is a HomoskedasticNoise, one can pass the non-transformed constraint as
        the ``noise_constraint``.

        Example:
            >>> base_noise = HomoskedasticNoise(
            >>>    noise_constraint=NonTransformedInterval(
            >>>        1e-5, 1e-1, initial_value=1e-3
            >>>    )
            >>> )
            >>> likelihood = SparseOutlierGaussianLikelihood(
            >>>    base_noise=base_noise,
            >>>    dim=X.shape[0],
            >>> )
            >>> model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
            >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
            >>> # NOTE: ``likelihood.noise_covar`` is the ``RelevancePursuitMixin``
            >>> sparse_module = likelihood.noise_covar
            >>> backward_relevance_pursuit(sparse_module, mll)

        Args:
            base_noise: The base noise model.
            dim: The number of training observations, which determines the maximum
                number of data-point-specific noise variances of the noise model.
            outlier_indices: The indices of the outliers.
            rho_prior: Prior for ``self.noise_covar``'s rho parameter.
            rho_constraint: Constraint for ``self.noise_covar``'s rho
                parameter. Needs to be a ``NonTransformedInterval`` because exact
                sparsity cannot be represented using smooth transforms like a
                softplus or sigmoid.
            batch_shape: The batch shape of the learned noise parameter (default: []).
            convex_parameterization: Whether to use the convex parameterization of rho,
                which generally improves optimization results and is thus recommended.
            loo: Whether to use leave-one-out (LOO) update equations that can compute
                the optimal values of each individual rho, keeping all else equal.
        """
        noise_covar = SparseOutlierNoise(
            base_noise=base_noise,
            dim=dim,
            outlier_indices=outlier_indices,
            rho_prior=rho_prior,
            rho_constraint=rho_constraint,
            batch_shape=batch_shape,
            convex_parameterization=convex_parameterization,
            loo=loo,
        )
        super().__init__(noise_covar=noise_covar)

    # pyre-ignore[14]: Inconsistent override because the super class accepts ``*params``
    def marginal(
        self,
        function_dist: MultivariateNormal,
        X: Tensor | list[Tensor] | None = None,
        **kwargs: Any,
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        # this scales the rhos by the diagonal of the "non-robust" covariance matrix
        diag_K = covar.diagonal() if self.noise_covar.convex_parameterization else None
        noise_covar = self.noise_covar.forward(
            X=X, shape=mean.shape, diag_K=diag_K, **kwargs
        )
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)

    def expected_log_prob(
        self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        raise NotImplementedError(
            "SparseOutlierGaussianLikelihood does not yet support variational inference"
            ", but this is not a fundamental limitation. It will require an adding "
            "the `expected_log_prob` method to SparseOutlierGaussianLikelihood."
        )


class SparseOutlierNoise(Noise, RelevancePursuitMixin):
    def __init__(
        self,
        base_noise: Noise | FixedGaussianNoise,
        dim: int,
        outlier_indices: list[int] | None = None,
        rho_prior: Prior | None = None,
        rho_constraint: NonTransformedInterval | None = None,
        batch_shape: torch.Size | None = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ):
        """A noise model in the Relevance Pursuit family of models, permitting
        additional "robust" variance for a small set of outlier data points.
        See also ``SparseOutlierGaussianLikelihood``, which leverages this noise model.

        For details, see [Ament2024pursuit]_ or https://arxiv.org/abs/2410.24222.

        Example:
            >>> base_noise = HomoskedasticNoise(
            >>>    noise_constraint=NonTransformedInterval(
            >>>        1e-5, 1e-1, initial_value=1e-3
            >>>    )
            >>> )
            >>> likelihood = SparseOutlierGaussianLikelihood(
            >>>    base_noise=base_noise,
            >>>    dim=X.shape[0],
            >>> )
            >>> model = SingleTaskGP(train_X=X, train_Y=Y, likelihood=likelihood)
            >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
            >>> # NOTE: ``likelihood.noise_covar`` is the ``SparseOutlierNoise``
            >>> sparse_module = likelihood.noise_covar
            >>> backward_relevance_pursuit(sparse_module, mll)

        Args:
            base_noise: The base noise model.
            dim: The number of training observations, which determines the maximum
                number of data-point-specific noise variances of the noise model.
            outlier_indices: The indices of the outliers.
            rho_prior: Prior for the rho parameter.
            rho_constraint: Constraint for the rho parameter. Needs to be a
                NonTransformedInterval because exact sparsity cannot be represented
                using smooth transforms like a softplus or sigmoid.
            batch_shape: The batch shape of the learned noise parameter (default: []).
            convex_parameterization: Whether to use the convex parameterization of rho,
                which generally improves optimization results and is thus recommended.
            loo: Whether to use leave-one-out (LOO) update equations that can compute
                the optimal values of each individual rho, keeping all else equal.
        """
        super().__init__()
        RelevancePursuitMixin.__init__(self, dim=dim, support=outlier_indices)

        if batch_shape is None:
            batch_shape = base_noise.noise.shape[:-1]

        self.base_noise = base_noise
        device = base_noise.noise.device
        if rho_constraint is None:
            cvx_upper_bound = 1 - 1e-3  # < 1 to avoid singularities
            rho_constraint = NonTransformedInterval(
                lower_bound=0.0,
                upper_bound=cvx_upper_bound if convex_parameterization else torch.inf,
                initial_value=0.0,
            )
        else:
            if not isinstance(rho_constraint, NonTransformedInterval):
                raise ValueError(
                    "`rho_constraint` must be a `NonTransformedInterval` if it "
                    "is not None."
                )

            if rho_constraint.lower_bound < 0:
                raise ValueError(
                    "SparseOutlierNoise requires rho_constraint.lower_bound >= 0."
                )

            if convex_parameterization and rho_constraint.upper_bound > 1:
                raise ValueError(
                    "Convex parameterization requires rho_constraint.upper_bound <= 1."
                )

        # NOTE: Prefer to keep the initialization of the sparse_parameter in the
        # derived classes of the Mixin, because it might require additional logic
        # that we don't want to put into RelevancePursuitMixin.
        num_outliers = len(self.support)
        self.register_parameter(
            "raw_rho",
            parameter=Parameter(
                torch.zeros(
                    *batch_shape,
                    num_outliers,
                    dtype=base_noise.noise.dtype,
                    device=device,
                )
            ),
        )

        if rho_prior is not None:

            def _rho_param(m):
                return m.rho

            # this closure is only needed for some features, e.g. sampling from the
            # prior, and requires additional thought in this case, as it will also has
            # to modify the support of the RelevancePursuitMixin
            _set_rho_closure = None

            self.register_prior("rho_prior", rho_prior, _rho_param, _set_rho_closure)

        self.register_constraint("raw_rho", rho_constraint)
        # only publicly exposing getter of convex parameterization
        # since post-hoc modification can lead to inconsistencies
        # with the rho constraints.
        self._convex_parameterization = convex_parameterization
        self.loo = loo
        self._cached_train_inputs = None

    @property
    def sparse_parameter(self) -> Parameter:
        return self.raw_rho

    def set_sparse_parameter(self, value: Parameter) -> None:
        """Sets the sparse parameter.

        NOTE: We can't use the property setter @sparse_parameter.setter because of
        the special way PyTorch treats Parameter types, including custom setters.
        """
        self.raw_rho = torch.nn.Parameter(value.to(self.raw_rho))

    @property
    def convex_parameterization(self) -> bool:
        return self._convex_parameterization

    @staticmethod
    def _from_model(model: Model) -> RelevancePursuitMixin:
        sparse_module = model.likelihood.noise_covar
        if not isinstance(sparse_module, SparseOutlierNoise):
            raise ValueError(
                "The model's likelihood does not have a SparseOutlierNoise noise "
                f"as its noise_covar module, but instead a {type(sparse_module)}."
            )
        return sparse_module

    @property
    def _convex_rho(self) -> Tensor:
        """Transforms the raw_rho parameter such that ``rho ~= 1 / (1 - raw_rho) - 1``,
        which is a diffeomorphism from [0, 1] to [0, inf] whose derivative is nowhere
        zero. This transforms the marginal log likelihood to be a convex function of
        the ``self.raw_rho`` Parameter, when the covariance matrix is well conditioned.

        NOTE: The convex parameterization also includes a scaling of the rho values by
        the diagonal of the covariance matrix, which is carried out in the ``marginal``
        call in the SparseOutlierGaussianLikelihood.
        """
        # pyre-ignore[7]: It is not have an incompatible return type, pyre just doesn't
        # recognize that the result gets promoted to a Tensor.
        return 1 / (1 - self.raw_rho) - 1

    @property
    def rho(self) -> Tensor:
        """Dense representation of the data-point-specific variances,
        corresponding to the latent ``self.raw_rho`` values, which might be
        represented sparsely or in the convex parameterization. The last dimension
        is equal to the number of training points ``self.dim``.

        NOTE: ``rho`` differs from ``self.sparse_parameter`` in that the latter
        returns the parameter in its sparse representation when ``self.is_sparse``
        is true, and in its latent convex paramzeterization when
        ``self.convex_parameterization`` is true, while ``rho`` always returns the
        data-point-specific variances, embedded in a dense tensor. The dense
        representation is used to propagate gradients to the sparse rhos in the
        support.

        Returns:
            A ``batch_shape x self.dim``-dim Tensor of robustness variances.
        """
        # NOTE: don't need to do transform / untransform since we are
        # enforcing NonTransformedIntervals.
        rho_outlier = self._convex_rho if self.convex_parameterization else self.raw_rho
        if not self.is_sparse:  # in the dense representation, we're done.
            return rho_outlier

        # If rho_outlier is in the sparse representation, we need to pad the
        # rho values with zeros at the correct positions. The difference
        # between this and calling RelevancePursuit's ``to_dense`` is that
        # the latter will propagate gradients through all rhos, whereas
        # the path here only propagates gradients to the sparse set of
        # outliers, which is important for the optimization of the support.
        rho_inlier = torch.zeros(
            1, dtype=rho_outlier.dtype, device=rho_outlier.device
        ).expand(rho_outlier.shape[:-1] + (1,))
        rho = torch.cat(
            [rho_outlier, rho_inlier], dim=-1
        )  # batch_shape x (num_outliers + 1)

        return rho[..., self._rho_selection_indices]

    @property
    def _rho_selection_indices(self) -> Tensor:
        # num_train is cached in the forward pass in training mode
        # if an index is not in the outlier indices, we get the zeros from the
        # last index of "rho"
        # is this related to a sparse to dense mapping used in RP?
        rho_selection_indices = torch.full(
            self.raw_rho.shape[:-1] + (self.dim,),
            -1,
            dtype=torch.long,
            device=self.raw_rho.device,
        )
        for i, j in enumerate(self.support):
            rho_selection_indices[j] = i

        return rho_selection_indices

    # pyre-ignore[14]: Inconsistent override because the super class accepts ``*params``
    def forward(
        self,
        X: Tensor | list[Tensor] | None = None,
        shape: torch.Size | None = None,
        diag_K: Tensor | None = None,
        **kwargs: Any,
    ) -> LinearOperator | Tensor:
        """Computes the covariance matrix of the sparse outlier noise model.

        Args:
            X: The training inputs, used to determine if the model is applied to the
                training data, in which case the outlier variances are applied, or not.
                NOTE: By default, BoTorch passes the transformed training inputs to
                the likelihood during both training and inference.
            shape: The shape of the covariance matrix, which is used to broadcast the
                rho values to the correct shape.
            diag_K: The diagonal of the covariance matrix, which is used to scale the
                rho values in the convex parameterization.
            kwargs: Any additional parameters of the base noise model, same as for
                GPyTorch's noise model. Note that this implementation does not support
                non-kwarg ``params`` arguments, which are used in GPyTorch's
                noise models.

        Returns:
            A ``batch_shape x self.dim``-dim Tensor of robustness variances.
        """
        noise_covar = self.base_noise(X, shape=shape, **kwargs)
        # rho should always be applied to the training set, irrespective of whether or
        # not we are in training mode, so we will check if we should apply the rhos,
        # based on cached training inputs.
        rho = self.rho
        # NOTE: Even though it is not strictly required for many likelihoods, BoTorch
        # and GPyTorch generally pass the training inputs to the likelihood, e.g.:
        # 1) in fit_gpytorch_mll:
        # (
        #   github.com/meta-pytorch/botorch/blob/3ca48d0ac5865a017ac6b2294807b432d6472bcf/
        #   botorch/optim/closures/model_closures.py#L185
        # )
        # 2) in the exact prediction strategy:
        # (
        #   github.com/cornellius-gp/gpytorch/blob/
        #   d501c284d05a1186868dc3fb20e0fa6ad32d32ac/
        #   gpytorch/models/exact_prediction_strategies.py#L387
        # )
        # 3) In the model's ``posterior`` method, if ``observation_noise`` is
        # True, in which case the test inputs will be passed to the likelihood:
        # (
        #    https://github.com/meta-pytorch/botorch/blob/
        #    4190f74363757ad97bfb0b437402b749ae50ba4c/
        #    botorch/models/gpytorch.py#L198
        # )
        # Note that this module will raise a warning in this case to inform the user
        # that the robust rhos are not applied to the test data.
        if X is not None:
            if isinstance(X, list):
                if len(X) != 1:
                    raise UnsupportedError(
                        "SparseOutlierNoise only supports a single training input "
                        f"Tensor, but got a list of length {len(X)}."
                    )
                X = X[0]
            if noise_covar.shape[-1] != rho.shape[-1]:
                apply_robust_variances = False
                warning_reason = (
                    "the last dimension of the base noise covariance "
                    f"({noise_covar.shape[-1]}) "
                    "is not compatible with the last dimension of rho "
                    f"({rho.shape[-1]})."
                )
            elif self.training or self._cached_train_inputs is None:
                apply_robust_variances = True
                self._cached_train_inputs = X
                warning_reason = ""  # will not warn when applying robust variances
            else:
                apply_robust_variances = torch.equal(X, self._cached_train_inputs)
                warning_reason = (
                    "the passed train_inputs are not equal to the cached ones."
                )
        else:
            apply_robust_variances = False
            warning_reason = "the training inputs were not passed to the likelihood."

        if apply_robust_variances:
            if diag_K is not None:
                rho = (diag_K + noise_covar.diagonal()) * rho  # convex parameterization
            noise_covar = noise_covar + DiagLinearOperator(rho)
        else:
            warn(
                f"SparseOutlierNoise: Robust rho not applied because {warning_reason} "
                + "This can happen when the model posterior is evaluated on test data.",
                InputDataWarning,
                stacklevel=2,
            )
        return noise_covar

    # relevance pursuit method expansion and contraction related methods
    def expansion_objective(self, mll: ExactMarginalLogLikelihood) -> Tensor:
        """Computes an objective value for all the inactive parameters, i.e.
        self.sparse_parameter[~self.is_active] since we can't add already active
        parameters to the support. This value will be used to select the parameters.

        Args:
            mll: The marginal likelihood, containing the model to optimize.

        Returns:
            The expansion objective value for all the inactive parameters.
        """
        f = self._optimal_rhos if self.loo else self._sparse_parameter_gradient
        return f(mll)

    def _optimal_rhos(self, mll: ExactMarginalLogLikelihood) -> Tensor:
        """Computes the optimal rho deltas for the given model.

        Args:
            mll: The marginal likelihood, containing the model to optimize.

        Returns:
            A ``batch_shape x self.dim``-dim Tensor of optimal rho deltas.
        """
        # train() is important, since we want to evaluate the prior with mll.model(X),
        # but in eval(), __call__ gives the posterior.
        mll.train()  # NOTE: this changes model.train_inputs to be unnormalized.
        X, Y = mll.model.train_inputs[0], mll.model.train_targets
        F = mll.model(X)
        TX = mll.model.transform_inputs(X)
        L = mll.likelihood(F, TX)  # likelihood expects transformed inputs
        S = L.covariance_matrix  # (Kernel Matrix + Noise Matrix)

        # NOTE: The following computation is mathematically equivalent to the formula
        # in this comment, but leverages the positive-definiteness of S via its
        # Cholesky factorization.
        # S_inv = S.inverse()
        # diag_S_inv = S_inv.diagonal(dim1=-1, dim2=-2)
        # loo_var = 1 / S_inv.diagonal(dim1=-1, dim2=-2)
        # loo_mean = Y - (S_inv @ Y) / diag_S_inv

        chol = psd_safe_cholesky(S, upper=True)
        eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
        inv_root = torch.linalg.solve_triangular(chol, eye, upper=True)

        # test: inv_root.square().sum(dim=-1) - S.inverse().diag()
        diag_S_inv = inv_root.square().sum(dim=-1)
        loo_var = 1 / diag_S_inv
        S_inv_Y = torch.cholesky_solve(Y.unsqueeze(-1), chol, upper=True).squeeze(-1)
        loo_mean = Y - S_inv_Y / diag_S_inv

        loo_error = loo_mean - Y
        optimal_rho_deltas = loo_error.square() - loo_var
        return (optimal_rho_deltas - self.rho).clamp(0)[~self.is_active]
