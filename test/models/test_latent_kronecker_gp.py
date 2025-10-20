#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

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
    def test_default_init(self):
        for (
            batch_shape,
            n_train,
            d,
            t,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # t
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_T, train_Y, mask = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_T=train_T,
                train_Y=train_Y,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)

            # test init
            train_Y_flat = train_Y.reshape(*batch_shape, -1)[..., mask]
            if use_transforms:
                self.assertIsInstance(model.input_transform, Normalize)
                self.assertIsInstance(model.outcome_transform, Standardize)
            else:
                self.assertFalse(hasattr(model, "input_transform"))
                self.assertFalse(hasattr(model, "outcome_transform"))
            train_Y_flat = (
                model.outcome_transform(train_Y_flat.unsqueeze(-1))[0].squeeze(-1)
                if use_transforms
                else train_Y_flat
            )
            self.assertAllClose(model.train_inputs[0], train_X, atol=0.0)
            self.assertAllClose(model.train_T, train_T, atol=0.0)
            self.assertAllClose(model.train_targets, train_Y_flat, atol=0.0)
            self.assertIsInstance(model.likelihood, GaussianLikelihood)
            self.assertIsInstance(model.mean_module_X, ZeroMean)
            self.assertIsInstance(model.mean_module_T, ZeroMean)
            self.assertIsInstance(model.covar_module_X, MaternKernel)
            self.assertIsInstance(model.covar_module_T, ScaleKernel)
            self.assertIsInstance(model.covar_module_T.base_kernel, MaternKernel)

    def test_custom_init(self):
        # test whether custom likelihoods and mean/covar modules are set correctly.
        for batch_shape, n_train, d, t, dtype in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 3),  # t
            (torch.float, torch.double),  # dtype
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            train_T_incorrect_shape = train_T.clone()[..., :-1, :]
            expected_shape = torch.Size([*batch_shape, train_Y.shape[-1], 1])
            err_msg = f"Expected train_T with shape {expected_shape} "
            err_msg += f"but got {train_T_incorrect_shape.shape}."
            with self.assertRaises(BotorchTensorDimensionError) as e:
                LatentKroneckerGP(
                    train_X=train_X, train_T=train_T_incorrect_shape, train_Y=train_Y
                )
            self.assertEqual(err_msg, str(e.exception))

            train_T_not_broadcastable = train_T.clone().unsqueeze(0)
            train_T_not_broadcastable = train_T_not_broadcastable.repeat(
                2, *([1] * len(train_T.shape))
            )
            with self.assertRaises(RuntimeError) as e:
                LatentKroneckerGP(
                    train_X=train_X, train_T=train_T_not_broadcastable, train_Y=train_Y
                )

            # only test if batch_shape is not empty or singleton
            if sum(batch_shape) > 1:
                train_Y_inhomogeneous = train_Y.clone()
                train_Y_inhomogeneous[..., 0, :, 0] = 0.0
                train_Y_inhomogeneous[..., 1, :, 0] = torch.nan
                err_msg = "Pattern of missing values in train_Y "
                err_msg += "must be equal across batch_shape."
                with self.assertRaises(ValueError) as e:
                    LatentKroneckerGP(
                        train_X=train_X, train_T=train_T, train_Y=train_Y_inhomogeneous
                    )
                self.assertEqual(err_msg, str(e.exception))

            likelihood = GaussianLikelihood(batch_shape=batch_shape)
            mean_module_X = ConstantMean(batch_shape=batch_shape)
            mean_module_T = ConstantMean(batch_shape=batch_shape)
            covar_module_X = RBFKernel(ard_num_dims=d, batch_shape=batch_shape)
            covar_module_T = RBFKernel(ard_num_dims=1, batch_shape=batch_shape)

            model = LatentKroneckerGP(
                train_X=train_X,
                train_T=train_T,
                train_Y=train_Y,
                likelihood=likelihood,
                mean_module_X=mean_module_X,
                mean_module_T=mean_module_T,
                covar_module_X=covar_module_X,
                covar_module_T=covar_module_T,
            )
            model.to(**tkwargs)

            self.assertEqual(model.likelihood, likelihood)
            self.assertEqual(model.mean_module_X, mean_module_X)
            self.assertEqual(model.mean_module_T, mean_module_T)
            self.assertEqual(model.covar_module_X, covar_module_X)
            self.assertEqual(model.covar_module_T, covar_module_T)

            # check devices
            def _get_index(device):
                return device.index if device.index is not None else 0

            device_type = self.device.type
            device_idx = _get_index(self.device)

            self.assertEqual(model.train_inputs[0].device.type, device_type)
            self.assertEqual(_get_index(model.train_inputs[0].device), device_idx)
            self.assertEqual(model.train_targets.device.type, device_type)
            self.assertEqual(_get_index(model.train_targets.device), device_idx)
            self.assertEqual(model.mask_valid.device.type, device_type)
            self.assertEqual(_get_index(model.mask_valid.device), device_idx)
            for p in model.parameters():
                self.assertEqual(p.device.type, device_type)
                self.assertEqual(_get_index(p.device), device_idx)

    def test_gp_train(self):
        for (
            batch_shape,
            n_train,
            d,
            t,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (1, 2),  # d
            (1, 2),  # t
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_T=train_T,
                train_Y=train_Y,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)

            # test optim
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(**tkwargs)
            with warnings.catch_warnings(), model.use_iterative_methods():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

    def _test_gp_eval_shapes(
        self,
        batch_shape: torch.Size,
        use_transforms: bool,
        tkwargs: dict,
    ):
        n_train = 10
        n_test = 7
        d = 1
        t = 2

        if use_transforms:
            intf = Normalize(d=d, batch_shape=batch_shape)
            octf = DEFAULT
        else:
            intf = None
            octf = None

        train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
            n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
        )

        test_T = train_T[..., :-1, :]

        model = LatentKroneckerGP(
            train_X=train_X,
            train_T=train_T,
            train_Y=train_Y,
            input_transform=intf,
            outcome_transform=octf,
        )
        model.to(**tkwargs)
        model.eval()

        for test_shape in (
            torch.Size([]),
            torch.Size([3]),
            torch.Size([*batch_shape]),
            torch.Size([2, *batch_shape]),
        ):
            test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

            # we expect an error if test_shape and batch_shape cannot be broadcasted
            try:
                broadcast_shape = torch.broadcast_shapes(test_shape, batch_shape)
            except RuntimeError as e:
                with self.assertRaisesRegex(RuntimeError, str(e)):
                    model.posterior(test_X, test_T)
                continue
            pred_shape = torch.Size([*broadcast_shape, n_test, t - 1])

            # custom posterior samples
            posterior = model.posterior(test_X, test_T)
            self.assertEqual(posterior.batch_range, (0, -1))
            for sample_shape in (
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ):
                # test posterior.rsample
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample(sample_shape=sample_shape)
                self.assertEqual(
                    pred_samples.shape, torch.Size([*sample_shape, *pred_shape])
                )
                self.assertEqual(
                    pred_samples.shape,
                    posterior._extended_shape(torch.Size(sample_shape)),
                )
                # test posterior.rsample_from_base_samples
                base_samples = torch.randn(
                    *sample_shape,
                    *posterior.base_sample_shape,
                    **tkwargs,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NumericalWarning)
                    pred_samples = posterior.rsample_from_base_samples(
                        sample_shape, base_samples
                    )
                self.assertEqual(
                    pred_samples.shape, torch.Size([*sample_shape, *pred_shape])
                )
                if len(sample_shape) > 0:
                    # test incorrect base sample shape
                    incorrect_base_samples = torch.randn(
                        5,
                        *posterior.base_sample_shape,
                        **tkwargs,
                    )
                    with self.assertRaises(RuntimeError):
                        posterior.rsample_from_base_samples(
                            sample_shape, incorrect_base_samples
                        )

    def test_gp_eval_shapes_float_with_tf(self):
        use_transforms = True
        tkwargs = {"device": self.device, "dtype": torch.float}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_double_with_tf(self):
        use_transforms = True
        tkwargs = {"device": self.device, "dtype": torch.double}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_float_without_tf(self):
        use_transforms = False
        tkwargs = {"device": self.device, "dtype": torch.float}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_shapes_double_without_tf(self):
        use_transforms = False
        tkwargs = {"device": self.device, "dtype": torch.double}

        for batch_shape in (
            torch.Size([]),
            torch.Size([1]),
            torch.Size([2, 3]),
        ):
            self._test_gp_eval_shapes(
                batch_shape=batch_shape,
                use_transforms=use_transforms,
                tkwargs=tkwargs,
            )

    def test_gp_eval_values(self):
        for (
            batch_shape,
            n_train,
            n_test,
            d,
            t,
            dtype,
            use_transforms,
        ) in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2, 3]),
            ),
            (10,),  # n_train
            (7,),  # n_test
            (1,),  # d
            (1,),  # t
            (torch.float, torch.double),  # dtype
            (False, True),  # use_transforms
        ):
            torch.manual_seed(12345)
            tkwargs = {"device": self.device, "dtype": dtype}

            if use_transforms:
                intf = Normalize(d=d, batch_shape=batch_shape)
                octf = DEFAULT
            else:
                intf = None
                octf = None

            train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
                n_train=n_train, d=d, t=t, batch_shape=batch_shape, tkwargs=tkwargs
            )

            model = LatentKroneckerGP(
                train_X=train_X,
                train_T=train_T,
                train_Y=train_Y,
                input_transform=intf,
                outcome_transform=octf,
            )
            model.to(**tkwargs)
            model.eval()

            for test_shape in (
                torch.Size([]),
                torch.Size([3]),
                torch.Size([*batch_shape]),
                torch.Size([2, *batch_shape]),
            ):
                test_X = torch.rand(*test_shape, n_test, d, **tkwargs)

                # we expect an error if test_shape and batch_shape cannot be broadcasted
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

                # check custom predictions and GPyTorch are roughly the same
                self.assertLess(
                    (pred_mean - pred_samples.mean(dim=0)).norm() / pred_mean.norm(),
                    0.1,
                )
                self.assertLess(
                    (pred_var - pred_samples.var(dim=0)).norm() / pred_var.norm(), 0.1
                )

    def test_iterative_methods(self):
        batch_shape = torch.Size([])
        tkwargs = {"device": self.device, "dtype": torch.double}

        train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
            n_train=10, d=1, t=1, batch_shape=batch_shape, tkwargs=tkwargs
        )

        model = LatentKroneckerGP(train_X=train_X, train_T=train_T, train_Y=train_Y)
        model.to(**tkwargs)

        with model.use_iterative_methods():
            self.assertTrue(settings._fast_covar_root_decomposition.off())
            self.assertTrue(settings._fast_log_prob.on())
            self.assertTrue(settings._fast_solves.on())
            self.assertEqual(settings.cg_tolerance.value(), 0.01)
            self.assertEqual(settings.max_cg_iterations.value(), 10000)

    def test_not_implemented(self):
        batch_shape = torch.Size([])
        tkwargs = {"device": self.device, "dtype": torch.double}

        train_X, train_T, train_Y, _ = _get_data_with_missing_entries(
            n_train=10, d=1, t=1, batch_shape=batch_shape, tkwargs=tkwargs
        )

        model = LatentKroneckerGP(train_X=train_X, train_T=train_T, train_Y=train_Y)
        model.to(**tkwargs)

        cls_name = model.__class__.__name__

        transform = ScalarizedPosteriorTransform(torch.tensor([1.0], **tkwargs))
        err_msg = f"Posterior transforms currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X, posterior_transform=transform)

        err_msg = f"Observation noise currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X, observation_noise=True)

        err_msg = f"Conditioning currently not supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.condition_on_observations(train_X, train_Y)

        likelihood = FixedNoiseGaussianLikelihood(
            torch.tensor([1.0]), batch_shape=batch_shape, **tkwargs
        )
        model = LatentKroneckerGP(
            train_X=train_X, train_T=train_T, train_Y=train_Y, likelihood=likelihood
        )
        model.to(**tkwargs)

        err_msg = f"Only GaussianLikelihood currently supported for {cls_name}"
        with self.assertRaisesRegex(NotImplementedError, err_msg):
            model.posterior(train_X)

    def test_construct_inputs(self) -> None:
        # This test relies on the fact that the random (missing) data generation
        # does not remove all occurrences of a particular X or T value. Therefore,
        # we fix the random seed and set n_train and t to slightly larger values.

        torch.manual_seed(12345)
        for batch_shape, n_train, d, t, dtype in itertools.product(
            (  # batch_shape
                torch.Size([]),
                torch.Size([1]),
                torch.Size([2]),
                torch.Size([2, 3]),
            ),
            (15,),  # n_train
            (1, 2),  # d
            (10,),  # t
            (torch.float, torch.double),  # dtype
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

            # randomly permute data to test robustness to non-contiguous data
            idx = torch.randperm(n_train * t, device=self.device)
            train_X_supervised = train_X_supervised[..., idx, :][..., mask[idx], :]
            train_Y_supervised = train_Y_supervised[..., idx, :][..., mask[idx], :]

            dataset = SupervisedDataset(
                X=train_X_supervised,
                Y=train_Y_supervised,
                Yvar=train_Y_supervised,  # just to check warning
                feature_names=[f"x_{i}" for i in range(d)] + ["step"],
                outcome_names=["y"],
            )

            w_msg = "Ignoring Yvar values in provided training data, because "
            w_msg += "they are currently not supported by LatentKroneckerGP."
            with self.assertWarnsRegex(InputDataWarning, w_msg):
                model_inputs = LatentKroneckerGP.construct_inputs(dataset)

            # this test generates train_X and train_T in sorted order
            # the data is randomly permuted before passing to construct_inputs
            # construct_inputs sorts the data, so we expect the results to be equal
            self.assertAllClose(model_inputs["train_X"], train_X, atol=0.0)
            self.assertAllClose(model_inputs["train_T"], train_T, atol=0.0)
            self.assertAllClose(
                model_inputs["train_Y"], train_Y, atol=0.0, equal_nan=True
            )
