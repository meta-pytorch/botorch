#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import warnings

from functools import partial
from unittest.mock import patch

import gpytorch
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch.models.relevance_pursuit import (
    _get_initial_value,
    backward_relevance_pursuit,
    forward_relevance_pursuit,
    get_posterior_over_support,
    RelevancePursuitMixin,
)
from botorch.models.robust_relevance_pursuit_model import (
    FRACTIONS_OF_OUTLIERS,
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import Normalize
from botorch.test_functions.base import constant_outlier_generator, CorruptedTestProblem

from botorch.test_functions.synthetic import Ackley
from botorch.utils.constraints import NonTransformedInterval
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import Interval

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.means import ZeroMean
from gpytorch.mlls import PredictiveLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from pyre_extensions import none_throws
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class DummyRelevancePursuitModule(Module, RelevancePursuitMixin):
    """Module to test support modification methods of RelevancePursuitMixin."""

    def __init__(
        self,
        dim: int,
        support: list[int] | None,
    ) -> None:
        """Initializes test module.

        Args:
            dim: The total number of features.
            support: The indices of the features in the support, subset of range(dim).
        """
        Module.__init__(self)
        self._par = Parameter(torch.zeros(len(support) if support is not None else 0))
        RelevancePursuitMixin.__init__(self, dim=dim, support=support)

    @property
    def sparse_parameter(self) -> Parameter:
        return self._par

    def set_sparse_parameter(self, value: Parameter) -> None:
        self._par = torch.nn.Parameter(value.to(self._par))


class TestRobustGP(BotorchTestCase):
    def _make_dataset(
        self,
        n: int,
        num_outliers: int,
        dtype: torch.dtype,
        seed: int = 1,
    ) -> tuple[Tensor, Tensor, list[int]]:
        torch.manual_seed(seed)

        X = torch.rand(n, 1, dtype=dtype, device=self.device)
        F = torch.sin(2 * torch.pi * (2 * X)).sum(dim=-1, keepdim=True)
        sigma = 1e-2
        Y = F + torch.randn_like(F) * sigma
        outlier_indices = list(range(n - num_outliers, n))
        Y[outlier_indices] = -Y[outlier_indices]
        return X, Y, outlier_indices

    def _get_robust_model(
        self,
        X: Tensor,
        Y: Tensor,
        likelihood: SparseOutlierGaussianLikelihood,
    ) -> SingleTaskGP:
        min_lengthscale = 0.1
        lengthscale_constraint = NonTransformedInterval(
            min_lengthscale, torch.inf, initial_value=0.2
        )
        d = X.shape[-1]

        kernel = ScaleKernel(
            RBFKernel(ard_num_dims=d, lengthscale_constraint=lengthscale_constraint),
            outputscale_constraint=NonTransformedInterval(
                0.01, 10.0, initial_value=0.1
            ),
        ).to(dtype=X.dtype, device=self.device)

        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            mean_module=ZeroMean(),
            covar_module=kernel,
            input_transform=Normalize(d=X.shape[-1]),
            likelihood=likelihood,
        )
        model.to(dtype=X.dtype, device=self.device)
        return model

    def test_robust_gp_end_to_end(self) -> None:
        self._test_robust_gp_end_to_end(convex_parameterization=False, mll_tol=1e-8)

    def test_robust_convex_gp_end_to_end(self) -> None:
        self._test_robust_gp_end_to_end(convex_parameterization=True, mll_tol=1e-8)

    def _test_robust_gp_end_to_end(
        self,
        convex_parameterization: bool,
        mll_tol: float,
    ) -> None:
        """End-to-end robust GP test."""
        n = 32
        dtype = torch.double
        num_outliers = 5
        X, Y, outlier_indices = self._make_dataset(
            n=n, num_outliers=num_outliers, dtype=dtype, seed=1
        )

        # The definition of "outliers" depends on the model capacity, so what is an
        # outlier w.r.t. a simple model might not be an outlier w.r.t. a complex model.
        # For this reason, it is necessary to bound the lengthscale of the GP from below
        # as otherwise arbitrarily complex outlier deviations can be modeled well by the
        # GP, if the data is not sampled finely enough.
        min_lengthscale = 0.1
        lengthscale_constraint = NonTransformedInterval(
            min_lengthscale, torch.inf, initial_value=0.2
        )

        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=X.shape[-1], lengthscale_constraint=lengthscale_constraint
            ),
            outputscale_constraint=NonTransformedInterval(
                0.01, 10.0, initial_value=0.1
            ),
        )

        prior_mean_of_support = int(0.2 * n)
        model = RobustRelevancePursuitSingleTaskGP(
            train_X=X,
            train_Y=Y,
            input_transform=Normalize(d=X.shape[-1]),
            cache_model_trace=True,  # to check the model trace after optimization
            covar_module=covar_module,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
        )

        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        numbers_of_outliers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16]
        rp_kwargs = {
            "numbers_of_outliers": numbers_of_outliers,  # skipping some
            "optimizer_kwargs": {"options": {"maxiter": 1024}},
        }
        fit_gpytorch_mll(mll, **rp_kwargs)
        model_trace = none_throws(model.model_trace)

        # Bayesian model comparison
        support_size, bmc_probabilities = get_posterior_over_support(
            SparseOutlierNoise, model_trace, prior_mean_of_support=prior_mean_of_support
        )
        self.assertEqual(len(support_size), len(numbers_of_outliers) + 1)
        self.assertEqual(len(support_size), len(bmc_probabilities))
        self.assertAlmostEqual(bmc_probabilities.sum().item(), 1.0)
        map_index = torch.argmax(bmc_probabilities)

        with self.assertRaisesRegex(
            ValueError,
            "`log_support_prior` and `prior_mean_of_support` cannot both be None",
        ):
            get_posterior_over_support(SparseOutlierNoise, model_trace)

        map_model = model_trace[map_index]
        sparse_module = map_model.likelihood.noise_covar
        undetected_outliers = set(outlier_indices) - set(sparse_module.support)
        self.assertEqual(len(undetected_outliers), 0)

        with patch.object(
            SparseOutlierNoise,
            "forward",
            wraps=sparse_module.forward,
        ) as sparse_module_fwd:
            # testing that posterior inference on training set does not throw warnings,
            # which means that the passed inputs are the equal to the cached ones.
            with warnings.catch_warnings(record=True) as warnings_log:
                map_model.posterior(X)
                self.assertEqual(warnings_log, [])
            # Testing that the noise module's forward receives transformed inputs
            X_in_call = sparse_module_fwd.call_args.kwargs["X"]
            self.assertIsInstance(X_in_call, list)
            self.assertEqual(len(X_in_call), 1)
            X_in_call = X_in_call[0]
            X_max = X_in_call.amax(dim=0)
            X_min = X_in_call.amin(dim=0)
            self.assertAllClose(X_max, torch.ones_like(X_max))
            self.assertAllClose(X_min, torch.zeros_like(X_min))

    def test_robust_relevance_pursuit(self) -> None:
        for optimizer, convex_parameterization, dtype in itertools.product(
            [forward_relevance_pursuit, backward_relevance_pursuit],
            [True, False],
            [torch.float32, torch.float64],
        ):
            with self.subTest(
                optimizer=optimizer,
                convex_parameterization=convex_parameterization,
                dtype=dtype,
            ):
                # testing the loo functionality only with the forward algorithm
                # and the convex parameterization, to save test runtime.
                loo = (
                    optimizer is forward_relevance_pursuit
                ) and convex_parameterization
                self._test_robust_relevance_pursuit(
                    optimizer=optimizer,
                    convex_parameterization=convex_parameterization,
                    dtype=dtype,
                    loo=loo,
                )

    def _test_robust_relevance_pursuit(
        self,
        optimizer: forward_relevance_pursuit | backward_relevance_pursuit,
        convex_parameterization: bool,
        dtype: torch.dtype,
        loo: bool,
    ) -> None:
        """
        Test executing with different combinations of arguments, without checking the
        model fit end-to-end.
        """
        n = 32
        dtype = torch.double
        X, Y, _ = self._make_dataset(n=n, num_outliers=6, dtype=dtype, seed=1)
        min_noise = 1e-6  # minimum noise variance constraint
        max_noise = 1e-2
        base_noise = HomoskedasticNoise(
            noise_constraint=NonTransformedInterval(
                min_noise, max_noise, initial_value=1e-3
            )
        ).to(dtype=dtype, device=self.device)

        with self.assertRaisesRegex(
            ValueError,
            "`rho_constraint` must be a `NonTransformedInterval` if it is not None.",
        ):
            SparseOutlierGaussianLikelihood(
                base_noise=base_noise,
                dim=X.shape[0],
                convex_parameterization=convex_parameterization,
                rho_constraint=Interval(0.0, 1.0),  # pyre-ignore[6]
            )

        with self.assertRaisesRegex(ValueError, "rho_constraint.lower_bound >= 0"):
            SparseOutlierGaussianLikelihood(
                base_noise=base_noise,
                dim=X.shape[0],
                convex_parameterization=convex_parameterization,
                rho_constraint=NonTransformedInterval(-1.0, 1.0),
            )

        if convex_parameterization:
            with self.assertRaisesRegex(ValueError, "rho_constraint.upper_bound <= 1"):
                SparseOutlierGaussianLikelihood(
                    base_noise=base_noise,
                    dim=X.shape[0],
                    convex_parameterization=convex_parameterization,
                    rho_constraint=NonTransformedInterval(0.0, 2.0),
                    loo=loo,
                )
        else:  # with the canonical parameterization, any upper bound on rho is valid.
            likelihood_with_other_bounds = SparseOutlierGaussianLikelihood(
                base_noise=base_noise,
                dim=X.shape[0],
                convex_parameterization=convex_parameterization,
                rho_constraint=NonTransformedInterval(0.0, 2.0),
                loo=loo,
            )
            noise_w_other_bounds = likelihood_with_other_bounds.noise_covar
            self.assertEqual(noise_w_other_bounds.raw_rho_constraint.lower_bound, 0.0)
            self.assertEqual(noise_w_other_bounds.raw_rho_constraint.upper_bound, 2.0)

        rp_likelihood = SparseOutlierGaussianLikelihood(
            base_noise=base_noise,
            dim=X.shape[0],
            convex_parameterization=convex_parameterization,
            loo=loo,
        )

        # testing conversion to and from dense representation
        sparse_noise = rp_likelihood.noise_covar
        sparse_noise.to_dense()
        self.assertFalse(sparse_noise.is_sparse)
        dense_rho = sparse_noise.rho
        sparse_noise.to_sparse()
        self.assertTrue(sparse_noise.is_sparse)
        sparse_rho = sparse_noise.rho
        self.assertAllClose(dense_rho, sparse_rho)

        with self.assertRaisesRegex(NotImplementedError, "variational inference"):
            rp_likelihood.expected_log_prob(target=None, input=None)  # pyre-ignore[6]

        # testing prior initialization
        likelihood_with_prior = SparseOutlierGaussianLikelihood(
            base_noise=base_noise,
            dim=X.shape[0],
            convex_parameterization=convex_parameterization,
            rho_prior=gpytorch.priors.NormalPrior(loc=1 / 2, scale=0.1),
            loo=loo,
        )
        self.assertIsInstance(
            likelihood_with_prior.noise_covar.rho_prior, gpytorch.priors.NormalPrior
        )

        # combining likelihood with rho prior and full GP model
        # this will test the prior code paths when computing the marginal likelihood
        model = self._get_robust_model(X=X, Y=Y, likelihood=likelihood_with_prior)

        # testing the _from_model method
        with self.assertRaisesRegex(
            ValueError,
            "The model's likelihood does not have a SparseOutlierNoise noise",
        ):
            SparseOutlierNoise._from_model(SingleTaskGP(train_X=X, train_Y=Y))

        self.assertEqual(
            SparseOutlierNoise._from_model(model), likelihood_with_prior.noise_covar
        )

        # Test that there is a warning because
        # model.likelihood.noise_covar._cached_train_inputs is None
        # and the shape of the test inputs are not compatible with the noise module.
        X_test = torch.rand(3, 1, dtype=dtype, device=self.device)
        with self.assertWarnsRegex(
            InputDataWarning,
            "Robust rho not applied because the last dimension of the base noise "
            "covariance",
        ):
            model.likelihood.noise_covar.forward(X_test)

        with self.assertRaisesRegex(
            UnsupportedError, "only supports a single training input Tensor"
        ):
            model.likelihood.noise_covar.forward([X, X])

        # executing once successfully so that _cached_train_inputs is populated
        self.assertIsNone(model.likelihood.noise_covar._cached_train_inputs)
        model.posterior(X, observation_noise=True)
        self.assertIsNotNone(model.likelihood.noise_covar._cached_train_inputs)

        X_test = torch.rand_like(X)  # same size as training inputs but not the same
        with self.assertWarnsRegex(
            InputDataWarning,
            "Robust rho not applied because the passed train_inputs are not equal to"
            " the cached ones.",
        ):
            model.posterior(X_test, observation_noise=True)

        # optimization via forward or backward relevance pursuit
        sparse_module = model.likelihood.noise_covar
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

        with self.assertWarnsRegex(
            InputDataWarning,
            "not applied because the training inputs were not passed to the likelihood",
        ):
            # the training inputs would have to be passed as arguments to the mll after
            # the train targets.
            mll(mll.model(*mll.model.train_inputs), mll.model.train_targets)

        extra_kwargs = {}
        if convex_parameterization:
            # initial_support doesn't have any special effect in the convex
            # parameterization, adding this here so we don't have to test every
            # combination of arguments.
            extra_kwargs["initial_support"] = torch.randperm(
                X.shape[0], device=self.device
            )[: n // 2]  # initializing with half the support

        # testing the reset_parameters functionality in conjunction with the convex
        # parameterization to limit test runtime, but they are orthogonal.
        # NOTE: the forward algorithm can give rise to an early termination warning,
        # due to reaching a stationary point of the likelihood before adding all indices
        reset_parameters = convex_parameterization
        sparse_module, model_trace = optimizer(
            sparse_module=sparse_module,
            mll=mll,
            record_model_trace=True,
            reset_parameters=reset_parameters,
            reset_dense_parameters=reset_parameters,
            optimizer_kwargs={"options": {"maxiter": 2}},
            **extra_kwargs,
        )
        model_trace = none_throws(model_trace)
        # Bayesian model comparison
        prior_mean_of_support = 2.0
        support_size, bmc_probabilities = get_posterior_over_support(
            SparseOutlierNoise, model_trace, prior_mean_of_support=prior_mean_of_support
        )
        if optimizer is backward_relevance_pursuit:
            expected_length = len(X) + 1
            if "initial_support" in extra_kwargs:
                expected_length = len(X) // 2 + 1
            self.assertEqual(len(model_trace), expected_length)
            self.assertEqual(support_size.max(), expected_length - 1)
            self.assertEqual(support_size[-1].item(), 0)  # includes zero
            # results of forward are sorted in decreasing order of support size
            self.assertAllClose(support_size, support_size.sort(descending=True).values)

        elif optimizer is forward_relevance_pursuit:
            # the forward algorithm will only add until no additional rho has a
            # non-negative gradient, which can happen before the full support is added.
            min_expected_length = 10
            self.assertGreaterEqual(len(model_trace), min_expected_length)
            lower_bound = len(X) // 2 if "initial_support" in extra_kwargs else 0
            self.assertGreaterEqual(support_size.min().item(), lower_bound)
            self.assertEqual(
                support_size[0].item(),
                n // 2 if "initial_support" in extra_kwargs else 0,
            )

            # results of forward are sorted in increasing order of support size
            self.assertAllClose(
                support_size, support_size.sort(descending=False).values
            )

    def test_robust_relevance_pursuit_single_task_gp(self) -> None:
        """Test for `RobustRelevancePursuitSingleTaskGP`, whose main purpose is to
        automatically dispatch to the relevance pursuit algorithm when optimized with
        `fit_gpytorch_mll`.
        """
        for optimizer, dtype in itertools.product(
            [forward_relevance_pursuit, backward_relevance_pursuit],
            [torch.float32, torch.float64],
        ):
            with self.subTest(
                optimizer=optimizer,
                dtype=dtype,
            ):
                self._test_robust_relevance_pursuit_single_task_gp(
                    optimizer=optimizer,
                    dtype=dtype,
                )

    def _test_robust_relevance_pursuit_single_task_gp(
        self,
        optimizer: forward_relevance_pursuit | backward_relevance_pursuit,
        dtype: torch.dtype,
    ) -> None:
        n = 32
        dtype = torch.double
        X, Y, _ = self._make_dataset(n=n, num_outliers=6, dtype=dtype, seed=1)
        # test for model class
        robust_model = RobustRelevancePursuitSingleTaskGP(
            train_X=X,
            train_Y=Y,
        )
        # test that the training mode is not affected by the conversion
        # to the standard model
        robust_model.eval()
        standard_model = robust_model.to_standard_model()
        self.assertIsInstance(standard_model, SingleTaskGP)
        self.assertFalse(standard_model.training)
        self.assertFalse(robust_model.training)

        robust_model.train()
        mll = ExactMarginalLogLikelihood(
            likelihood=robust_model.likelihood, model=robust_model
        )
        mll = fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": 2}},
            timeout_sec=10.0,
            relevance_pursuit_optimizer=optimizer,
        )
        # check the default sparsity levels
        expected_numbers_of_outliers = torch.tensor(
            [int(p * n) for p in FRACTIONS_OF_OUTLIERS],
            dtype=dtype,
            device=self.device,
        )
        if optimizer is backward_relevance_pursuit:
            inferred_numbers_of_outliers = robust_model.bmc_support_sizes
            expected_numbers_of_outliers = expected_numbers_of_outliers.sort(
                descending=True
            ).values
        else:
            # the forward algorithm can terminate early when the mll doesn't strictly
            # increase when including an additional index in the support. Further, the
            # last iteration of the forward algorithm might have only added a smaller
            # number of elements than expected, fewer elements had a strictly
            # positive gradient.
            inferred_numbers_of_outliers = robust_model.bmc_support_sizes[:-1]
            expected_numbers_of_outliers = expected_numbers_of_outliers[
                : len(inferred_numbers_of_outliers)
            ]

        self.assertAllClose(inferred_numbers_of_outliers, expected_numbers_of_outliers)

        # test that multiple dispatch throws an error when attempting to fit
        # an approximate marginal log liklihood with a RobustRelevancePursuitModel
        approx_mll = PredictiveLogLikelihood(
            likelihood=robust_model.likelihood,
            model=robust_model,
            num_data=n,
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "Relevance Pursuit does not yet support approximate inference",
        ):
            fit_gpytorch_mll(approx_mll)

    def test_basic_relevance_pursuit_module(self) -> None:
        dim = 3
        module = DummyRelevancePursuitModule(dim=dim, support=[0])
        # testing basic sparse -> dense conversions
        self.assertTrue(module.is_sparse)
        module.to_dense()
        self.assertFalse(module.is_sparse)
        self.assertEqual(len(module.sparse_parameter), dim)

        for representation in ["sparse", "dense"]:
            module = DummyRelevancePursuitModule(dim=dim, support=[0])
            getattr(module, "to_" + representation)()

            # testing that we can't remove indices that don't exist in the support
            with self.assertRaisesRegex(ValueError, "is not in support"):
                module.contract_support(indices=[1])

            # testing that we can't add already existing indices to the support
            with self.assertRaisesRegex(ValueError, "already in the support"):
                module.expand_support(indices=[0])

            # successfully contract in dense representation
            module.contract_support(indices=[0])
            self.assertEqual(len(module.support), 0)
            self.assertFalse(
                module.is_sparse if representation == "dense" else not module.is_sparse
            )
            self.assertAllClose(
                module.sparse_parameter, torch.zeros_like(module.sparse_parameter)
            )

            module = DummyRelevancePursuitModule(dim=dim, support=[0, 2])
            getattr(module, "to_" + representation)()
            module.remove_support()
            self.assertEqual(len(module.support), 0)
            self.assertAllClose(
                module.sparse_parameter, torch.zeros_like(module.sparse_parameter)
            )

            # unrelated model
            model = SingleTaskGP(train_X=torch.rand(3, 1), train_Y=torch.rand(3, 1))
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            with self.assertRaisesRegex(
                ValueError, "The gradient of the sparse_parameter is None."
            ):
                module.expansion_objective(mll)

            # can't "expand" support by a negative number of indices
            self.assertFalse(module.support_expansion(mll, n=-1))

        # initalizing model so that train_X is dependent on the sparse_parameter
        module = DummyRelevancePursuitModule(dim=dim, support=None)
        module.to_dense()
        train_X = module.sparse_parameter.unsqueeze(-1)
        model = SingleTaskGP(train_X=train_X, train_Y=torch.rand_like(train_X))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        # not expanding because the gradient is zero, but not None.
        self.assertFalse(module.support_expansion(mll, n=1))

        # testing support expansion with a gradient modifier, which assigns
        # a positive value to the last element of the sparse_parameter.
        def expansion_modifier(g: Tensor | None, module=module) -> Tensor:
            g = torch.zeros_like(module.sparse_parameter)
            g[-1] = 1.0  # this means we are adding the last inactive element
            return g

        expanded = module.support_expansion(mll, n=1, modifier=expansion_modifier)
        self.assertTrue(expanded)
        self.assertTrue(dim - 1 in module.support)

        # testing support contraction
        # not contracting since we can't contract by a negative number of indices
        self.assertFalse(module.support_contraction(mll, n=-1))
        self.assertTrue(
            module.support_contraction(mll, n=1, modifier=expansion_modifier)
        )
        self.assertEqual(len(module.support), 0)

        def contraction_modifier(g: Tensor | None, module) -> Tensor:
            g = torch.ones_like(module.sparse_parameter)
            g[-1] = 0.0  # this means we are removing the last element
            return g

        module = DummyRelevancePursuitModule(dim=dim, support=None)
        module.full_support()
        module._contraction_modifier = partial(contraction_modifier, module=module)

        # unrelated model
        model = SingleTaskGP(train_X=torch.rand(3, 1), train_Y=torch.rand(3, 1))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

        _, model_trace = backward_relevance_pursuit(
            sparse_module=module,
            mll=mll,
            record_model_trace=True,
        )
        self.assertEqual(len(model_trace), dim + 1)

        module = DummyRelevancePursuitModule(dim=dim, support=None)
        module.full_support()
        module._contraction_modifier = partial(contraction_modifier, module=module)

        with self.assertWarnsRegex(Warning, "contraction from sparsity.*unsuccessful."):
            _, model_trace = backward_relevance_pursuit(
                sparse_module=module,
                mll=mll,
                record_model_trace=True,
                sparsity_levels=[dim + 1, dim, dim - 1],
            )
        self.assertEqual(len(model_trace), 1)

        # the exit condition of the forward algorithm is already tested in the tests of
        # the robust model

        # testing parameter initialization helper
        value = _get_initial_value(
            value=None,
            lower=torch.tensor(-torch.inf, device=self.device),
            upper=torch.tensor(torch.inf, device=self.device),
        )
        self.assertEqual(value, torch.tensor(0.0, device=self.device))
        value = _get_initial_value(
            value=torch.tensor(-3.0, device=self.device),
            lower=torch.tensor(-2.0, device=self.device),
            upper=torch.tensor(torch.inf, device=self.device),
        )
        self.assertEqual(value, torch.tensor(-1.0, device=self.device))
        value = _get_initial_value(
            value=None,
            lower=torch.tensor(-torch.inf, device=self.device),
            upper=torch.tensor(2.0, device=self.device),
        )
        self.assertEqual(value, torch.tensor(1.0, device=self.device))
        value = _get_initial_value(
            value=None,
            lower=torch.tensor(0.0, device=self.device),
            upper=torch.tensor(torch.pi, device=self.device),
        )
        self.assertAllClose(value, torch.tensor(1.0, device=self.device))

    def test_experimental_utils(self) -> None:
        base_f = Ackley(dim=3)
        outlier_value = 100.0
        outlier_generator = partial(constant_outlier_generator, constant=outlier_value)

        # no outliers
        f = CorruptedTestProblem(
            base_test_problem=base_f,
            outlier_generator=outlier_generator,
            outlier_fraction=0.0,
        )
        n, d = 16, base_f.dim
        X = torch.randn(n, d)
        Y = f(X)
        self.assertAllClose(Y, base_f(X))

        # all outliers
        f = CorruptedTestProblem(
            base_test_problem=base_f,
            outlier_generator=outlier_generator,
            outlier_fraction=1.0,
        )
        n, d = 16, base_f.dim
        X = torch.randn(n, d)
        Y = f(X)
        self.assertTrue(((Y - base_f(X)).abs() > 1).all())

        # testing seeds
        num_seeds = 3
        f = CorruptedTestProblem(
            base_test_problem=base_f,
            outlier_generator=outlier_generator,
            outlier_fraction=1 / 2,
            seeds=range(num_seeds),
        )
        n, d = 8, base_f.dim
        X = torch.randn(n, d)
        Y_last = base_f(X)
        for _ in range(num_seeds):
            Y = f(X)
            # with these seeds we should have at least 1 outlier and less than n,
            # which shows that the masking works correctly.
            num_outliers = (Y == outlier_value).sum()
            self.assertGreater(num_outliers, 1)
            self.assertLess(num_outliers, n)
            # testing that the outliers are not the same, even if we evaluate on the
            # same input.
            self.assertTrue((Y_last - Y).norm() > 1.0)
            Y_last = Y

        # after num_seeds has been exhausted, the evaluation will error.
        with self.assertRaises(StopIteration):
            f(X)
