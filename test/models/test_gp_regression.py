#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import warnings

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Log
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.test_helpers import get_pvar_expected
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior


class TestGPRegressionBase(BotorchTestCase):
    def _get_model_and_data(
        self,
        batch_shape,
        m,
        outcome_transform=None,
        input_transform=None,
        extra_model_kwargs=None,
        **tkwargs,
    ):
        extra_model_kwargs = extra_model_kwargs or {}
        train_X, train_Y = get_random_data(batch_shape=batch_shape, m=m, **tkwargs)
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "outcome_transform": outcome_transform,
            "input_transform": input_transform,
        }
        model = SingleTaskGP(**model_kwargs, **extra_model_kwargs)
        return model, model_kwargs

    def _get_extra_model_kwargs(self):
        return {
            "mean_module": ZeroMean(),
            "covar_module": RBFKernel(use_ard=False),
            "likelihood": GaussianLikelihood(),
        }

    def test_gp(self, double_only: bool = False):
        bounds = torch.tensor([[-1.0], [1.0]])
        for batch_shape, m, dtype, use_octf, use_intf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.double,) if double_only else (torch.float, torch.double),
            (False, True),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            # Putting the outcome transform into eval mode to ensure that it is put into
            # train mode inside the constructor
            octf = (
                Standardize(m=m, batch_shape=batch_shape).eval() if use_octf else None
            )
            intf = (
                Normalize(d=1, bounds=bounds.to(**tkwargs), transform_on_train=True)
                if use_intf
                else None
            )
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                m=m,
                outcome_transform=octf,
                input_transform=intf,
                **tkwargs,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(**tkwargs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 1}}, max_attempts=1
                )

            # test init
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, RBFKernel)
            rbf_kernel = model.covar_module
            self.assertIsInstance(rbf_kernel, RBFKernel)
            self.assertIsInstance(rbf_kernel.lengthscale_prior, LogNormalPrior)
            if use_octf:
                self.assertIsInstance(model.outcome_transform, Standardize)
                # Ensure that the outcome transform was put into train mode.
                self.assertFalse(torch.all(model.outcome_transform.means == 0))
            if use_intf:
                self.assertIsInstance(model.input_transform, Normalize)
                # permute output dim
                train_X, train_Y, _ = model._transform_tensor_args(
                    X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                )
                # check that the train inputs have been transformed and set on the model
                self.assertTrue(torch.equal(model.train_inputs[0], intf(train_X)))

            # test param sizes
            params = dict(model.named_parameters())
            for p in params:
                self.assertEqual(params[p].numel(), m * math.prod(batch_shape))

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([3, 1]), **tkwargs)
            expected_shape = batch_shape + torch.Size([3, m])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # test adding observation noise
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            pvar = posterior_pred.variance
            pvar_exp = get_pvar_expected(posterior=posterior, model=model, X=X, m=m)
            self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

            # Tensor valued observation noise.
            obs_noise = torch.rand(X.shape, **tkwargs)
            posterior_pred = model.posterior(X, observation_noise=obs_noise)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            self.assertEqual(posterior_pred.variance.shape, expected_shape)
            if use_octf:
                _, obs_noise = model.outcome_transform.untransform(obs_noise, obs_noise)
            self.assertAllClose(posterior_pred.variance, posterior.variance + obs_noise)

            # test batch evaluation
            X = torch.rand(2, *batch_shape, 3, 1, **tkwargs)
            expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, m])

            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            # test adding observation noise in batch mode
            posterior_pred = model.posterior(X, observation_noise=True)
            self.assertIsInstance(posterior_pred, GPyTorchPosterior)
            self.assertEqual(posterior_pred.mean.shape, expected_shape)
            pvar = posterior_pred.variance
            pvar_exp = get_pvar_expected(posterior=posterior, model=model, X=X, m=m)
            self.assertAllClose(pvar, pvar_exp, rtol=1e-4, atol=1e-5)

            # test batch evaluation with broadcasting
            for input_batch_shape in ([], [3], [1]):
                X = torch.rand(*input_batch_shape, 3, 1, **tkwargs)

                if input_batch_shape == [3] and len(batch_shape) > 0:
                    if m == 1:
                        # Combine both possible messages into a regex pattern
                        bdcast_msg_1 = "Shape mismatch: objects cannot be broadcast to "
                        "a single shape"
                        bdcast_msg_2 = "Attempting to broadcast a dimension of length "
                        f"{input_batch_shape[0]} at -1! "
                        "Mismatching argument at index 1 had "
                        f"torch.Size([{input_batch_shape[0]}]); "
                        "but expected shape should be broadcastable to "
                        f"[{batch_shape[0]}]"
                        msg_pattern = r"({}|{})".format(bdcast_msg_1, bdcast_msg_2)
                    else:
                        msg_pattern = (
                            "The trailing batch dimensions of X must match "
                            "the trailing batch dimensions of the training inputs."
                        )
                    with self.assertRaisesRegex(RuntimeError, msg_pattern):
                        model.posterior(X, observation_noise=True)
                    continue
                else:
                    posterior = model.posterior(X, observation_noise=True)
                if input_batch_shape == [1] and len(batch_shape) > 0:
                    new_dims = []
                else:
                    new_dims = input_batch_shape
                expected_shape = batch_shape + torch.Size(new_dims + [3, m])
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, expected_shape)

    def test_default_transforms(self):
        for batch_shape, m, dtype, octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            ("Default", "None", "Log"),  # Outcome transform
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X, train_Y = get_random_data(batch_shape=batch_shape, m=m, **tkwargs)

            model_kwargs = {}
            if octf == "None":
                model_kwargs["outcome_transform"] = None
            elif octf == "Log":
                model_kwargs["outcome_transform"] = Log()
                train_Y = train_Y.abs() + 1

            model = SingleTaskGP(train_X=train_X, train_Y=train_Y, **model_kwargs)

            # Check outcome transform
            if octf == "Default":
                self.assertIsInstance(model.outcome_transform, Standardize)
                self.assertEqual(model.outcome_transform._batch_shape, batch_shape)
                self.assertEqual(model.outcome_transform._m, m)
            elif octf == "None":
                self.assertFalse(hasattr(model, "outcome_transform"))
            else:
                self.assertIsInstance(model.outcome_transform, Log)
            # Make sure there is no input transform
            self.assertFalse(hasattr(model, "input_transform"))

    def test_custom_init(self):
        extra_model_kwargs = self._get_extra_model_kwargs()
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                m=m,
                extra_model_kwargs=extra_model_kwargs,
                **tkwargs,
            )
            self.assertEqual(model.mean_module, extra_model_kwargs["mean_module"])
            self.assertEqual(model.covar_module, extra_model_kwargs["covar_module"])
            if "likelihood" in extra_model_kwargs:
                self.assertEqual(model.likelihood, extra_model_kwargs["likelihood"])

    def test_condition_on_observations(self):
        for batch_shape, m, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=m, outcome_transform=octf, **tkwargs
            )
            # evaluate model
            model.posterior(torch.rand(torch.Size([4, 1]), **tkwargs))
            # test condition_on_observations
            fant_shape = torch.Size([2])
            # fantasize at different input points
            X_fant, Y_fant = get_random_data(
                batch_shape=fant_shape + batch_shape, m=m, n=3, **tkwargs
            )
            c_kwargs = (
                {"noise": torch.full_like(Y_fant, 0.01)}
                if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                else {}
            )
            cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
            # fantasize at same input points (check proper broadcasting)
            c_kwargs_same_inputs = (
                {"noise": torch.full_like(Y_fant[0], 0.01)}
                if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                else {}
            )
            cm_same_inputs = model.condition_on_observations(
                X_fant[0], Y_fant, **c_kwargs_same_inputs
            )

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, 1, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(batch_shape + torch.Size([4, 1]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(fant_shape + batch_shape + torch.Size([4, 1]), **tkwargs),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape, fant_shape + batch_shape + torch.Size([4, m])
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + batch_shape + torch.Size([4, m]),
                )

                # check that fantasies of batched model are correct
                if len(batch_shape) > 0 and test_X.dim() == 2:
                    state_dict_non_batch = {
                        key: (val[0] if val.numel() > 1 else val)
                        for key, val in model.state_dict().items()
                    }
                    model_kwargs_non_batch = {
                        "train_X": model_kwargs["train_X"][0],
                        "train_Y": model_kwargs["train_Y"][0],
                    }
                    if "train_Yvar" in model_kwargs:
                        model_kwargs_non_batch["train_Yvar"] = model_kwargs[
                            "train_Yvar"
                        ][0]
                    if model_kwargs["outcome_transform"] is not None:
                        model_kwargs_non_batch["outcome_transform"] = Standardize(m=m)
                    else:
                        model_kwargs_non_batch["outcome_transform"] = None
                    model_non_batch = type(model)(**model_kwargs_non_batch)
                    model_non_batch.load_state_dict(state_dict_non_batch)
                    model_non_batch.eval()
                    model_non_batch.likelihood.eval()
                    model_non_batch.posterior(torch.rand(torch.Size([4, 1]), **tkwargs))
                    c_kwargs = (
                        {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                        if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                        else {}
                    )
                    cm_non_batch = model_non_batch.condition_on_observations(
                        X_fant[0][0], Y_fant[:, 0, :], **c_kwargs
                    )
                    non_batch_posterior = cm_non_batch.posterior(test_X)
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.mean[:, 0, ...],
                            non_batch_posterior.mean,
                            atol=1e-3,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            posterior_same_inputs.distribution.covariance_matrix[
                                :, 0, :, :
                            ],
                            non_batch_posterior.distribution.covariance_matrix,
                            atol=1e-3,
                        )
                    )

    def test_fantasize(self):
        for batch_shape, m, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            model, _ = self._get_model_and_data(
                batch_shape=batch_shape, m=m, outcome_transform=octf, **tkwargs
            )
            # fantasize
            X_f = torch.rand(torch.Size(batch_shape + torch.Size([4, 1])), **tkwargs)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            self.assertIsInstance(fm, model.__class__)

        # check that input transforms are applied to X.
        tkwargs = {"device": self.device, "dtype": torch.float}
        intf = Normalize(d=1, bounds=torch.tensor([[0], [10]], **tkwargs))
        model, _ = self._get_model_and_data(
            batch_shape=torch.Size(),
            m=1,
            input_transform=intf,
            **tkwargs,
        )
        X_f = torch.rand(4, 1, **tkwargs)
        fm = model.fantasize(
            X_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([3]))
        )
        self.assertTrue(
            torch.allclose(fm.train_inputs[0][:, -4:], intf(X_f).expand(3, -1, -1))
        )

    def test_subset_model(self):
        for batch_shape, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double), (True, False)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=2, batch_shape=batch_shape) if use_octf else None
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=2, outcome_transform=octf, **tkwargs
            )
            subset_model = model.subset_output([0])
            X = torch.rand(torch.Size(batch_shape + torch.Size([3, 1])), **tkwargs)
            p = model.posterior(X)
            p_sub = subset_model.posterior(X)
            self.assertTrue(
                torch.allclose(p_sub.mean, p.mean[..., [0]], atol=1e-4, rtol=1e-4)
            )
            self.assertTrue(
                torch.allclose(
                    p_sub.variance, p.variance[..., [0]], atol=1e-4, rtol=1e-4
                )
            )
            # test subsetting each of the outputs (follows a different code branch)
            subset_all_model = model.subset_output([0, 1])
            p_sub_all = subset_all_model.posterior(X)
            self.assertAllClose(p_sub_all.mean, p.mean)
            # subsetting should still return a copy
            self.assertNotEqual(model, subset_all_model)

    def test_construct_inputs(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=1, **tkwargs
            )
            X = model_kwargs["train_X"]
            Y = model_kwargs["train_Y"]
            training_data = SupervisedDataset(
                X,
                Y,
                feature_names=[f"x{i}" for i in range(X.shape[-1])],
                outcome_names=["y"],
            )
            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))

    def test_set_transformed_inputs(self):
        # This intended to catch https://github.com/meta-pytorch/botorch/issues/1078.
        # More general testing of _set_transformed_inputs is done under ModelListGP.
        X = torch.rand(5, 2)
        Y = X**2
        for tf_class in [Normalize, InputStandardize]:
            intf = tf_class(d=2)
            model = SingleTaskGP(X, Y, input_transform=intf)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
            tf_X = intf(X)
            self.assertEqual(X.shape, tf_X.shape)


class TestSingleTaskGP(TestGPRegressionBase):
    model_class = SingleTaskGP


class TestSingleTaskGPFixedNoise(TestSingleTaskGP):
    def _get_model_and_data(
        self,
        batch_shape,
        m,
        outcome_transform=None,
        input_transform=None,
        extra_model_kwargs=None,
        **tkwargs,
    ):
        extra_model_kwargs = extra_model_kwargs or {}
        train_X, train_Y = get_random_data(batch_shape=batch_shape, m=m, **tkwargs)
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": torch.full_like(train_Y, 0.01),
            "input_transform": input_transform,
            "outcome_transform": outcome_transform,
        }
        model = SingleTaskGP(**model_kwargs, **extra_model_kwargs)
        return model, model_kwargs

    def _get_extra_model_kwargs(self):
        return {
            "mean_module": ZeroMean(),
            "covar_module": RBFKernel(use_ard=False),
        }

    def test_fixed_noise_likelihood(self):
        for batch_shape, m, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (1, 2), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=m, **tkwargs
            )
            self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
            self.assertTrue(
                torch.equal(
                    model.likelihood.noise.contiguous().view(-1),
                    model_kwargs["train_Yvar"].contiguous().view(-1),
                )
            )

    def test_construct_inputs(self):
        for batch_shape, dtype in itertools.product(
            (torch.Size(), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape, m=1, **tkwargs
            )
            X = model_kwargs["train_X"]
            Y = model_kwargs["train_Y"]
            Yvar = model_kwargs["train_Yvar"]
            training_data = SupervisedDataset(
                X,
                Y,
                Yvar=Yvar,
                feature_names=[f"x{i}" for i in range(X.shape[-1])],
                outcome_names=["y"],
            )
            data_dict = model.construct_inputs(training_data)
            self.assertTrue(X.equal(data_dict["train_X"]))
            self.assertTrue(Y.equal(data_dict["train_Y"]))
            self.assertTrue(Yvar.equal(data_dict["train_Yvar"]))

    def test_fantasized_noise(self):
        for batch_shape, m, dtype, use_octf in itertools.product(
            (torch.Size(), torch.Size([2])),
            (1, 2),
            (torch.float, torch.double),
            (False, True),
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
            model, _ = self._get_model_and_data(
                batch_shape=batch_shape, m=m, outcome_transform=octf, **tkwargs
            )
            # fantasize
            X_f = torch.rand(torch.Size(batch_shape + torch.Size([4, 1])), **tkwargs)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            noise = (
                model.likelihood.noise.unsqueeze(-1)
                if m == 1
                else model.likelihood.noise.transpose(-1, -2)
            )
            avg_noise = noise.mean(dim=-2, keepdim=True)
            fm_noise = (
                fm.likelihood.noise.unsqueeze(-1)
                if m == 1
                else fm.likelihood.noise.transpose(-1, -2)
            )

            self.assertTrue((fm_noise[..., -4:, :] == avg_noise).all())
            # pass tensor of noise
            # noise is assumed to be outcome transformed
            # batch shape x n' x m
            obs_noise = torch.full(
                X_f.shape[:-1] + torch.Size([m]), 0.1, dtype=dtype, device=self.device
            )
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=obs_noise)
            fm_noise = (
                fm.likelihood.noise.unsqueeze(-1)
                if m == 1
                else fm.likelihood.noise.transpose(-1, -2)
            )
            self.assertTrue((fm_noise[..., -4:, :] == obs_noise).all())
            # test batch shape x 1 x m
            obs_noise = torch.full(
                X_f.shape[:-2] + torch.Size([1, m]),
                0.1,
                dtype=dtype,
                device=self.device,
            )
            fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=obs_noise)
            fm_noise = (
                fm.likelihood.noise.unsqueeze(-1)
                if m == 1
                else fm.likelihood.noise.transpose(-1, -2)
            )
            self.assertTrue(
                (
                    fm_noise[..., -4:, :]
                    == obs_noise.expand(X_f.shape[:-1] + torch.Size([m]))
                ).all()
            )
