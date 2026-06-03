#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from unittest.mock import patch

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.exceptions.warnings import InputDataWarning, OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.latent_kronecker_gp import LatentKroneckerGP
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase, get_random_data
from botorch.utils.types import DEFAULT
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from linear_operator import settings
from linear_operator.utils.linear_cg import linear_cg as _linear_cg_fn
from linear_operator.utils.warnings import NumericalWarning


def _get_data_with_missing_entries(
    n_train: int, d: int, t: int, batch_shape: torch.Size, tkwargs: dict
):
    train_X, train_Y = get_random_data(
        batch_shape=batch_shape, m=t, d=d, n=n_train, **tkwargs
    )

    train_T = torch.linspace(0, 1, t, **tkwargs).repeat(*batch_shape, 1).unsqueeze(-1)

    # randomly set half of the training outputs to nan
    mask = torch.ones(n_train * t, dtype=torch.bool, device=tkwargs["device"])
    mask[torch.randperm(n_train * t)[: n_train * t // 2]] = False
    train_Y[..., ~mask.reshape(n_train, t)] = torch.nan

    return train_X, train_T, train_Y, mask


class TestLatentKroneckerGP(BotorchTestCase):
    def _make_model(
        self,
        n_train=10,
        d=2,
        t=3,
        batch_shape=None,
        use_transforms=True,
        tkwargs=None,
        **model_kwargs,
    ):
        """Create data, transforms, and model for testing."""
        if batch_shape is None:
            batch_shape = torch.Size([])
        if tkwargs is None:
            tkwargs = {"device": self.device, "dtype": torch.double}
        train_X, train_T, train_Y, mask = _get_data_with_missing_entries(
            n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
        )
        intf = Normalize(d=d, batch_shape=batch_shape) if use_transforms else None
        octf = DEFAULT if use_transforms else None
        model = LatentKroneckerGP(
            train_X=train_X,
            train_T=train_T,
            train_Y=train_Y,
            input_transform=intf,
            outcome_transform=octf,
            **model_kwargs,
        )
        model.to(**tkwargs)
        return model, train_X, train_T, train_Y, mask, intf, octf

    # --- Init helpers ---

    def _test_default_init(self):
        n_train = 10
        for batch_shape, d, t, dtype, use_transforms in itertools.product(
            (torch.Size([]), torch.Size([1]), torch.Size([2, 3])),
            (1, 2),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, train_X, train_T, train_Y, mask, intf, octf = self._make_model(
                n_train=n_train,
                d=d,
                t=t,
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

            train_Y_flat = train_Y.reshape(*batch_shape, -1)[..., mask]
            if use_transforms:
                self.assertIsInstance(model.input_transform, Normalize)
                self.assertIsInstance(model.outcome_transform, Standardize)
                train_Y_flat = model.outcome_transform(train_Y_flat.unsqueeze(-1))[
                    0
                ].squeeze(-1)
            else:
                self.assertFalse(hasattr(model, "input_transform"))
                self.assertFalse(hasattr(model, "outcome_transform"))
            self.assertAllClose(model.train_inputs[0], train_X, atol=0.0)
            self.assertEqual(len(model.train_inputs), 2)
            self.assertAllClose(model.train_inputs[1], train_T, atol=0.0)
            self.assertIs(model.train_T, model.train_inputs[1])
            self.assertAllClose(model.train_targets, train_Y_flat, atol=0.0)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertIsInstance(model.mean_module_X, ZeroMean)
            self.assertIsInstance(model.mean_module_T, ZeroMean)
            self.assertIsInstance(model.covar_module_X, MaternKernel)
            self.assertIsInstance(model.covar_module_T, ScaleKernel)
            self.assertIsInstance(model.covar_module_T.base_kernel, MaternKernel)

    def _test_custom_init(self):
        n_train = 10
        for batch_shape, d, t, dtype in itertools.product(
            (torch.Size([]), torch.Size([1]), torch.Size([2]), torch.Size([2, 3])),
            (1, 2),
            (1, 3),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            # Error: incorrect T shape
            train_T_incorrect_shape = train_T.clone()[..., :-1, :]
            expected_shape = torch.Size([*batch_shape, train_Y.shape[-1], 1])
            err_msg = (
                f"Expected train_T with shape {expected_shape} "
                f"but got {train_T_incorrect_shape.shape}."
            )
            with self.assertRaises(BotorchTensorDimensionError) as e:
                LatentKroneckerGP(
                    train_X=train_X, train_T=train_T_incorrect_shape, train_Y=train_Y
                )
            self.assertEqual(err_msg, str(e.exception))

            # Error: non-broadcastable T
            train_T_not_broadcastable = (
                train_T.clone().unsqueeze(0).repeat(2, *([1] * len(train_T.shape)))
            )
            with self.assertRaises(RuntimeError):
                LatentKroneckerGP(
                    train_X=train_X, train_T=train_T_not_broadcastable, train_Y=train_Y
                )

            # Error: inhomogeneous missing pattern (only for non-trivial batch)
            if sum(batch_shape) > 1:
                train_Y_inhomogeneous = train_Y.clone()
                train_Y_inhomogeneous[..., 0, :, 0] = 0.0
                train_Y_inhomogeneous[..., 1, :, 0] = torch.nan
                err_msg = (
                    "Pattern of missing values in train_Y "
                    "must be equal across batch_shape."
                )
                with self.assertRaises(ValueError) as e:
                    LatentKroneckerGP(
                        train_X=train_X, train_T=train_T, train_Y=train_Y_inhomogeneous
                    )
                self.assertEqual(err_msg, str(e.exception))

            # Custom modules
            likelihood = GaussianLikelihood(batch_shape=batch_shape)
            mean_module_X = ConstantMean(batch_shape=batch_shape)
            mean_module_T = ConstantMean(batch_shape=batch_shape)
            covar_module_X = RBFKernel(ard_num_dims=d, batch_shape=batch_shape)
            covar_module_T = RBFKernel(ard_num_dims=1, batch_shape=batch_shape)

            model, *_ = self._make_model(
                n_train=n_train,
                d=d,
                t=t,
                batch_shape=batch_shape,
                use_transforms=False,
                tkwargs=tkwargs,
                likelihood=likelihood,
                mean_module_X=mean_module_X,
                mean_module_T=mean_module_T,
                covar_module_X=covar_module_X,
                covar_module_T=covar_module_T,
            )

            self.assertEqual(model.likelihood, likelihood)
            self.assertEqual(model.mean_module_X, mean_module_X)
            self.assertEqual(model.mean_module_T, mean_module_T)
            self.assertEqual(model.covar_module_X, covar_module_X)
            self.assertEqual(model.covar_module_T, covar_module_T)

            # Verify all model state is on the correct device
            device_type = self.device.type
            for tensor in (*model.train_inputs, model.train_targets, model.mask_valid):
                self.assertEqual(tensor.device.type, device_type)
            for p in model.parameters():
                self.assertEqual(p.device.type, device_type)

    # --- Training helpers ---

    def _test_gp_train(self):
        n_train = 10
        for batch_shape, d, t, dtype, use_transforms in itertools.product(
            (torch.Size([]), torch.Size([1]), torch.Size([2, 3])),
            (1, 2),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, *_ = self._make_model(
                n_train=n_train,
                d=d,
                t=t,
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(**tkwargs)
            with warnings.catch_warnings(), model.use_iterative_methods():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

    # --- Eval helpers ---

    def _test_gp_eval_shapes(
        self,
        batch_shape: torch.Size,
        use_transforms: bool,
        tkwargs: dict,
    ):
        n_test, d, t = 7, 1, 2
        model, train_X, train_T, *_ = self._make_model(
            d=d,
            t=t,
            batch_shape=batch_shape,
            use_transforms=use_transforms,
            tkwargs=tkwargs,
        )
        model.eval()
        test_T = train_T[..., :-1, :]

        for test_shape in (
            torch.Size([]),
            torch.Size([3]),
            torch.Size([*batch_shape]),
            torch.Size([2, *batch_shape]),
        ):
            test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

            try:
                broadcast_shape = torch.broadcast_shapes(test_shape, batch_shape)
            except RuntimeError as e:
                with self.assertRaisesRegex(RuntimeError, str(e)):
                    model.posterior(test_X, test_T)
                continue
            pred_shape = torch.Size([*broadcast_shape, n_test, t - 1])

            posterior = model.posterior(test_X, test_T)
            self.assertEqual(posterior.batch_range, (0, -1))
            for sample_shape in (
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
                None,
            ):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample(sample_shape=sample_shape)
                expected_sample_shape = (
                    torch.Size([1]) if sample_shape is None else sample_shape
                )
                expected_shape = torch.Size([*expected_sample_shape, *pred_shape])
                self.assertEqual(pred_samples.shape, expected_shape)
                self.assertEqual(
                    pred_samples.shape,
                    posterior._extended_shape(torch.Size(expected_sample_shape)),
                )
                base_samples = torch.randn(
                    *expected_sample_shape,
                    *posterior.base_sample_shape,
                    **tkwargs,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample_from_base_samples(
                        expected_sample_shape, base_samples
                    )
                self.assertEqual(
                    pred_samples.shape,
                    torch.Size([*expected_sample_shape, *pred_shape]),
                )
                if len(expected_sample_shape) > 0:
                    incorrect_base_samples = torch.randn(
                        5, *posterior.base_sample_shape, **tkwargs
                    )
                    with self.assertRaises(RuntimeError):
                        posterior.rsample_from_base_samples(
                            expected_sample_shape, incorrect_base_samples
                        )

    def _test_gp_eval_values(self):
        n_train, n_test, d, t = 10, 7, 1, 1
        for batch_shape, dtype, use_transforms in itertools.product(
            (torch.Size([]), torch.Size([1]), torch.Size([2, 3])),
            (torch.float, torch.double),
            (False, True),
        ):
            torch.manual_seed(12345)
            tkwargs = {"device": self.device, "dtype": dtype}
            model, train_X, train_T, *_, intf, octf = self._make_model(
                n_train=n_train,
                d=d,
                t=t,
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )
            model.eval()

            for test_shape in (
                torch.Size([]),
                torch.Size([3]),
                torch.Size([*batch_shape]),
                torch.Size([2, *batch_shape]),
            ):
                test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

                try:
                    broadcast_shape = torch.broadcast_shapes(test_shape, batch_shape)
                except RuntimeError as e:
                    with self.assertRaisesRegex(RuntimeError, str(e)):
                        model.posterior(test_X)
                    continue
                pred_shape = torch.Size([*broadcast_shape, n_test, t])

                posterior = model.posterior(test_X)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample(sample_shape=(2048,))
                self.assertEqual(pred_samples.shape, torch.Size([2048, *pred_shape]))

                # GPyTorch predictions
                with warnings.catch_warnings(), model.use_iterative_methods():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred = model(intf(test_X)) if intf is not None else model(test_X)
                pred_mean, pred_var = pred.mean, pred.variance
                pred_mean = pred_mean.reshape(*pred_mean.shape[:-1], n_test, t)
                pred_var = pred_var.reshape(*pred_var.shape[:-1], n_test, t)
                pred_mean, pred_var = (
                    model.outcome_transform.untransform(pred_mean, pred_var)
                    if octf is not None
                    else (pred_mean, pred_var)
                )
                self.assertEqual(pred_mean.shape, pred_shape)
                self.assertEqual(pred_var.shape, pred_shape)

                self.assertLess(
                    (pred_mean - pred_samples.mean(dim=0)).norm() / pred_mean.norm(),
                    0.1,
                )
                self.assertLess(
                    (pred_var - pred_samples.var(dim=0)).norm() / pred_var.norm(), 0.1
                )

    # --- Posterior helpers ---

    def _test_not_implemented(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        model, train_X, train_T, train_Y, *_ = self._make_model(
            d=1, t=1, use_transforms=False, tkwargs=tkwargs
        )
        cls_name = model.__class__.__name__

        transform = ScalarizedPosteriorTransform(torch.tensor([1.0], **tkwargs))
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Posterior transforms currently not supported for {cls_name}",
        ):
            model.posterior(train_X, posterior_transform=transform)

        with self.assertRaisesRegex(
            NotImplementedError,
            f"Observation noise currently not supported for {cls_name}",
        ):
            model.posterior(train_X, observation_noise=True)

        with self.assertRaisesRegex(
            NotImplementedError,
            f"Conditioning currently not supported for {cls_name}",
        ):
            model.condition_on_observations(train_X, train_Y)

        model, train_X, *_ = self._make_model(
            d=1,
            t=1,
            use_transforms=False,
            tkwargs=tkwargs,
            likelihood=FixedNoiseGaussianLikelihood(torch.tensor([1.0]), **tkwargs),
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Only GaussianLikelihood currently supported for {cls_name}",
        ):
            model.posterior(train_X)

    def _test_posterior_mean_and_variance(self):
        """Test posterior mean/variance match model(X) predictions."""
        for dtype in (torch.float, torch.double):
            for batch_shape in (torch.Size([]), torch.Size([2])):
                tkwargs = {"device": self.device, "dtype": dtype}
                n_test, d, t = 5, 2, 3
                model, train_X, train_T, *_, intf, _ = self._make_model(
                    d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
                )
                model.eval()

                test_X = torch.rand(*batch_shape, n_test, d, **tkwargs)
                cg_tol = 1e-6 if dtype == torch.double else 1e-4
                atol = cg_tol

                # Verify model(X) and model(X, T) produce same results
                with warnings.catch_warnings(), model.use_iterative_methods(tol=cg_tol):
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_single = model(intf(test_X))
                    pred_double = model(intf(test_X), train_T)
                self.assertAllClose(pred_single.mean, pred_double.mean, atol=atol)

                # Verify posterior.mean matches model(X).mean
                with warnings.catch_warnings(), model.use_iterative_methods(tol=cg_tol):
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    model_mean = model(intf(test_X)).mean
                    posterior = model.posterior(test_X, train_T)
                n_x, n_t = test_X.shape[-2], train_T.shape[-2]
                model_mean = model_mean.reshape(*model_mean.shape[:-1], n_x, n_t)
                self.assertAllClose(
                    model_mean,
                    posterior.mean.reshape(model_mean.shape),
                    atol=atol,
                )

                # Verify posterior variance is positive and finite
                self.assertTrue((posterior.variance > 0).all())
                self.assertTrue(posterior.variance.isfinite().all())

                # Test _set_transformed_inputs fallback with single train input
                saved_inputs = model.train_inputs
                model.train_inputs = (saved_inputs[0],)
                model._set_transformed_inputs()
                model.train_inputs = saved_inputs

                # Test forward() with single arg in training mode
                model.train()
                with warnings.catch_warnings(), model.use_iterative_methods(tol=cg_tol):
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_1 = model.forward(train_X)
                    pred_2 = model.forward(train_X, train_T)
                self.assertAllClose(pred_1.mean, pred_2.mean, atol=atol)

    def _test_solver_dispatch(self):
        """Test that posterior uses Cholesky vs CG based on settings."""
        tkwargs = {"device": self.device, "dtype": torch.double}
        data_kwargs = {"d": 2, "t": 3, "use_transforms": False, "tkwargs": tkwargs}
        test_X = torch.rand(5, 2, **tkwargs)

        # Test 1: Force CG by bypassing the Cholesky size threshold (default=800).
        model, _, train_T, *_ = self._make_model(**data_kwargs)
        model.eval()

        with (
            warnings.catch_warnings(),
            model.use_iterative_methods(),
            settings.max_cholesky_size(0),
            patch(
                "linear_operator.utils.linear_cg",
                wraps=_linear_cg_fn,
            ) as mock_cg,
        ):
            warnings.filterwarnings("ignore", category=NumericalWarning)
            # Verify settings are correctly configured for CG
            self.assertTrue(settings._fast_covar_root_decomposition.off())
            self.assertTrue(settings._fast_log_prob.on())
            self.assertTrue(settings._fast_solves.on())
            self.assertEqual(settings.cg_tolerance.value(), 0.01)
            self.assertEqual(settings.max_cg_iterations.value(), 10000)
            model.posterior(test_X, train_T)
        self.assertTrue(mock_cg.called, "Expected CG solver to be used")

        # Test 2: Force Cholesky by disabling fast solves.
        # Fresh model to avoid cached prediction strategy.
        model, _, train_T, *_ = self._make_model(**data_kwargs)
        model.eval()

        with (
            warnings.catch_warnings(),
            settings.fast_computations(solves=False),
            patch(
                "linear_operator.utils.linear_cg",
                wraps=_linear_cg_fn,
            ) as mock_cg,
        ):
            warnings.filterwarnings("ignore", category=NumericalWarning)
            self.assertTrue(settings._fast_solves.off())
            model.posterior(test_X, train_T)
        self.assertFalse(mock_cg.called, "Expected Cholesky solver, not CG")

    # --- Data construction helpers ---

    def _test_construct_inputs(self) -> None:
        # This test relies on the fact that the random (missing) data generation
        # does not remove all occurrences of a particular X or T value. Therefore,
        # we fix the random seed and set n_train and t to slightly larger values.
        torch.manual_seed(12345)
        n_train, t = 15, 10
        for batch_shape, d, dtype in itertools.product(
            (torch.Size([]), torch.Size([1]), torch.Size([2]), torch.Size([2, 3])),
            (1, 2),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_T, train_Y, mask = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            train_X_supervised = torch.cat(
                [
                    train_X.repeat_interleave(t, dim=-2),
                    train_T.repeat(*([1] * len(batch_shape)), n_train, 1),
                ],
                dim=-1,
            )
            train_Y_supervised = train_Y.reshape(*batch_shape, n_train * t, 1)

            idx = torch.randperm(n_train * t, device=self.device)
            train_X_supervised = train_X_supervised[..., idx, :][..., mask[idx], :]
            train_Y_supervised = train_Y_supervised[..., idx, :][..., mask[idx], :]

            dataset = SupervisedDataset(
                X=train_X_supervised,
                Y=train_Y_supervised,
                Yvar=train_Y_supervised,
                feature_names=[f"x_{i}" for i in range(d)] + ["step"],
                outcome_names=["y"],
            )

            with self.assertWarnsRegex(
                InputDataWarning,
                "Ignoring Yvar values in provided training data, because "
                "they are currently not supported by LatentKroneckerGP.",
            ):
                model_inputs = LatentKroneckerGP.construct_inputs(dataset)

            self.assertAllClose(model_inputs["train_X"], train_X, atol=0.0)
            self.assertAllClose(model_inputs["train_T"], train_T, atol=0.0)
            self.assertAllClose(
                model_inputs["train_Y"], train_Y, atol=0.0, equal_nan=True
            )

    # === Public test methods ===

    def test_init(self):
        self._test_default_init()
        self._test_custom_init()

    def test_train(self):
        self._test_gp_train()

    def test_eval(self):
        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            for dtype in (torch.float, torch.double):
                tkwargs = {"device": self.device, "dtype": dtype}
                for use_transforms in (False, True):
                    self._test_gp_eval_shapes(
                        batch_shape=batch_shape,
                        use_transforms=use_transforms,
                        tkwargs=tkwargs,
                    )
        self._test_gp_eval_values()

    def test_posterior(self):
        self._test_not_implemented()
        self._test_posterior_mean_and_variance()
        self._test_solver_dispatch()

    def test_construct_inputs(self):
        self._test_construct_inputs()
