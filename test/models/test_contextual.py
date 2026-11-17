#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.contextual import LCEAGP, SACGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.contextual_lcea import LCEAKernel
from botorch.models.kernels.contextual_sac import SACKernel
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


def _gen_datasets(
    infer_noise: bool = False,
    **tkwargs,
) -> tuple[dict[int, SupervisedDataset], tuple[Tensor, Tensor, Tensor]]:
    train_X = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], **tkwargs
    )
    train_Y = torch.tensor([[1.0], [2.0], [3.0]], **tkwargs)
    train_Yvar = None if infer_noise else torch.full_like(train_Y, 0.01)

    datasets = SupervisedDataset(
        X=train_X,
        Y=train_Y,
        Yvar=train_Yvar,
        feature_names=[f"x{i}" for i in range(train_X.shape[-1])],
        outcome_names=["y"],
    )
    return datasets, (train_X, train_Y, train_Yvar)


class TestContextualGP(BotorchTestCase):
    def test_construct_without_observed_noise(self) -> None:
        dataset, _ = _gen_datasets(
            infer_noise=True, device=self.device, dtype=torch.double
        )
        for cls, decomposition in [
            (SACGP, {"a": [0, 1], "b": [2, 3]}),
            (LCEAGP, {"a": ["x0", "x1"], "b": ["x2", "x3"]}),
        ]:
            with self.subTest(model=cls.__name__):
                kwargs = cls.construct_inputs(dataset, decomposition=decomposition)
                self.assertIn("train_Yvar", kwargs)
                self.assertIsNone(kwargs["train_Yvar"])
                model = cls(**kwargs)
                self.assertTrue(model.posterior(dataset.X[:1]).mean.isfinite().all())

    def test_SACGP(self):
        for dtype, infer_noise in ((torch.float, False), (torch.double, True)):
            tkwargs = {"device": self.device, "dtype": dtype}
            datasets, (train_X, train_Y, train_Yvar) = _gen_datasets(
                infer_noise, **tkwargs
            )
            self.decomposition = {"1": [0, 3], "2": [1, 2]}

            model = SACGP(train_X, train_Y, train_Yvar, self.decomposition)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 1}})

            self.assertIsInstance(model, SingleTaskGP)
            self.assertIsInstance(
                model.likelihood,
                GaussianLikelihood if infer_noise else FixedNoiseGaussianLikelihood,
            )
            self.assertDictEqual(model.decomposition, self.decomposition)
            self.assertIsInstance(model.mean_module, ConstantMean)
            self.assertIsInstance(model.covar_module, SACKernel)

            # test number of named parameters
            num_of_mean = 0
            num_of_lengthscales = 0
            num_of_outputscales = 0
            for param_name, param in model.named_parameters():
                if param_name == "mean_module.raw_constant":
                    num_of_mean += param.data.shape.numel()
                elif "raw_lengthscale" in param_name:
                    num_of_lengthscales += param.data.shape.numel()
                elif "raw_outputscale" in param_name:
                    num_of_outputscales += param.data.shape.numel()
            self.assertEqual(num_of_mean, 1)
            self.assertEqual(num_of_lengthscales, 2)
            self.assertEqual(num_of_outputscales, 2)

            test_x = torch.rand(5, 4, device=self.device, dtype=dtype)
            posterior = model(test_x)
            self.assertIsInstance(posterior, MultivariateNormal)

    def test_SACGP_construct_inputs(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            datasets, (train_X, train_Y, train_Yvar) = _gen_datasets(**tkwargs)
            self.decomposition = {"1": [0, 3], "2": [1, 2]}
            model = SACGP(train_X, train_Y, train_Yvar, self.decomposition)
            data_dict = model.construct_inputs(
                training_data=datasets, decomposition=self.decomposition
            )

            self.assertTrue(train_X.equal(data_dict["train_X"]))
            self.assertTrue(train_Y.equal(data_dict["train_Y"]))
            self.assertTrue(train_Yvar.equal(data_dict["train_Yvar"]))
            self.assertDictEqual(data_dict["decomposition"], self.decomposition)

    def test_LCEAGP(self):
        for dtype, infer_noise in ((torch.float, False), (torch.double, True)):
            tkwargs = {"device": self.device, "dtype": dtype}
            datasets, (train_X, train_Y, train_Yvar) = _gen_datasets(
                infer_noise, **tkwargs
            )
            # Test setting attributes
            decomposition = {"1": [0, 1], "2": [2, 3]}
            # test instantiate model
            model = LCEAGP(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                decomposition=decomposition,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 1}})

            self.assertIsInstance(model, LCEAGP)
            self.assertIsInstance(model.covar_module, LCEAKernel)
            self.assertDictEqual(model.decomposition, decomposition)
            self.assertIsInstance(
                model.likelihood,
                GaussianLikelihood if infer_noise else FixedNoiseGaussianLikelihood,
            )

            test_x = torch.rand(5, 4, device=self.device, dtype=dtype)
            posterior = model(test_x)
            self.assertIsInstance(posterior, MultivariateNormal)

    def test_LCEAGP_construct_inputs(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            datasets, (train_X, train_Y, train_Yvar) = _gen_datasets(**tkwargs)
            decomposition = {"1": ["x0", "x1"], "2": ["x2", "x3"]}
            decomposition_index = {"1": [0, 1], "2": [2, 3]}

            data_dict = LCEAGP.construct_inputs(
                training_data=datasets,
                decomposition=decomposition,
                train_embedding=False,
            )

            self.assertTrue(train_X.equal(data_dict["train_X"]))
            self.assertTrue(train_Y.equal(data_dict["train_Y"]))
            self.assertTrue(train_Yvar.equal(data_dict["train_Yvar"]))
            self.assertDictEqual(data_dict["decomposition"], decomposition_index)
            self.assertFalse(data_dict["train_embedding"])
