#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy

import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.target_aware_gp import TargetAwareEnsembleGP
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import HalfCauchyPrior
from torch.nn.modules import ModuleDict


class TestTargetAwareEnsembleGP(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tkwargs = {"device": self.device, "dtype": torch.double}
        self.train_X = torch.rand(10, 1, **self.tkwargs)
        self.train_Y = torch.rand(10, 1, **self.tkwargs)
        self.train_Yvar = torch.full_like(self.train_Y, 0.01, **self.tkwargs)
        self.base_model_dict = {
            "auxiliary_metric1": SingleTaskGP(
                self.train_X, self.train_Y, self.train_Yvar
            ),
            "auxiliary_metric2": SingleTaskGP(
                self.train_X, self.train_Y, self.train_Yvar
            ),
        }
        for m in self.base_model_dict.values():
            mll = ExactMarginalLogLikelihood(m.likelihood, m)
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": 5}}, max_attempts=1
            )
        self.base_model_state_dict = {
            metric_name: m.state_dict()
            for metric_name, m in self.base_model_dict.items()
        }

    def test_init(self) -> None:
        mean_module = ZeroMean()
        covar_module = RBFKernel(use_ard=False)
        bounds = torch.tensor([[-1.0], [1.0]], **self.tkwargs)
        octf = Standardize(m=1)
        intf = Normalize(d=1, bounds=bounds)
        weight_prior = HalfCauchyPrior(torch.tensor(0.1, **self.tkwargs))

        for prior in [None, weight_prior]:
            tw_model = TargetAwareEnsembleGP(
                train_X=self.train_X,
                train_Y=self.train_Y,
                train_Yvar=self.train_Yvar,
                base_model_dict=self.base_model_dict,
                mean_module=mean_module,
                covar_module=covar_module,
                outcome_transform=octf,
                input_transform=intf,
                ensemble_weight_prior=prior,
            )
            self.assertEqual(tw_model.mean_module, mean_module)
            self.assertEqual(tw_model.covar_module, covar_module)
            self.assertIsInstance(tw_model.base_model_dict, ModuleDict)

            self.assertIsInstance(tw_model.outcome_transform, Standardize)
            self.assertIsInstance(tw_model.input_transform, Normalize)

    def test_model(self) -> None:
        # pass a mixed type of base model dict
        new_model_dict = deepcopy(self.base_model_dict)
        m = SaasFullyBayesianSingleTaskGP(self.train_X, self.train_Y, self.train_Yvar)
        fit_fully_bayesian_model_nuts(
            m, warmup_steps=8, num_samples=5, thinning=2, disable_progbar=True
        )
        new_model_dict["auxiliary_metric3"] = m

        for base_model_dict in [self.base_model_dict, new_model_dict]:
            tw_model = TargetAwareEnsembleGP(
                train_X=self.train_X,
                train_Y=self.train_Y,
                train_Yvar=self.train_Yvar,
                base_model_dict=base_model_dict,
            )

            mll = ExactMarginalLogLikelihood(tw_model.likelihood, tw_model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 5}}, max_attempts=4
                )

            # check there is no training done on base_model_dict
            for (
                metric_name,
                base_model_state_dict,
            ) in self.base_model_state_dict.items():
                self.assertDictEqual(
                    tw_model.base_model_dict[metric_name].state_dict(),
                    base_model_state_dict,
                )

            # check posterior
            X_list = [
                torch.rand(torch.Size([3, 1]), **self.tkwargs),
                torch.rand(torch.Size([5, 3, 1]), **self.tkwargs),
            ]
            for X in X_list:
                posterior = tw_model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, X.shape)
                self.assertEqual(posterior.variance.shape, X.shape)

    def test_construct_inputs(self) -> None:
        datasets = SupervisedDataset(
            X=self.train_X,
            Y=self.train_Y,
            Yvar=self.train_Yvar,
            feature_names=[f"x{i}" for i in range(self.train_X.shape[-1])],
            outcome_names=["y"],
        )

        model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )

        mean_module = ZeroMean()
        covar_module = RBFKernel(use_ard=False)
        weight_prior = HalfCauchyPrior(torch.tensor(0.1, **self.tkwargs))
        data_dict = model.construct_inputs(
            training_data=datasets,
            base_model_dict=self.base_model_dict,
            mean_module=mean_module,
            covar_module=covar_module,
            ensemble_weight_prior=weight_prior,
        )
        self.assertTrue(self.train_X.equal(data_dict["train_X"]))
        self.assertTrue(self.train_Y.equal(data_dict["train_Y"]))
        self.assertTrue(self.train_Yvar.equal(data_dict["train_Yvar"]))
        self.assertIs(data_dict["base_model_dict"], self.base_model_dict)
        self.assertEqual(data_dict["mean_module"], mean_module)
        self.assertEqual(data_dict["covar_module"], covar_module)
        self.assertIsInstance(data_dict["ensemble_weight_prior"], HalfCauchyPrior)

    def test_weight_property(self) -> None:
        """Test weight property getter and setter."""
        tw_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )
        # Test getter returns tensor of correct shape
        weight = tw_model.weight
        self.assertEqual(weight.shape, torch.Size([len(self.base_model_dict)]))

        # Test setter: public setter is an identity round-trip
        new_weight = torch.tensor([0.5, 1.5], **self.tkwargs)
        tw_model.weight = new_weight
        self.assertTrue(torch.allclose(tw_model.weight, new_weight))

    def test_train_mode(self) -> None:
        """Test that base models stay in eval mode during training."""
        tw_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )
        # Put model in train mode
        tw_model.train(True)
        self.assertTrue(tw_model.training)
        # Base models should remain in eval mode
        for m in tw_model.base_model_dict.values():
            self.assertFalse(m.training)
            for param in m.parameters():
                self.assertFalse(param.requires_grad)

        # Put model in eval mode
        tw_model.train(False)
        self.assertFalse(tw_model.training)
        for m in tw_model.base_model_dict.values():
            self.assertFalse(m.training)

    def test_weight_threshold(self) -> None:
        """Test that weights below threshold are excluded from forward pass."""
        from botorch.models.target_aware_gp import WEIGHT_THRESHOLD

        tw_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )
        # Set one weight below threshold and one above
        tw_model.weight = torch.tensor(
            [WEIGHT_THRESHOLD / 2, WEIGHT_THRESHOLD * 2], **self.tkwargs
        )
        # Forward pass should work and only use the second model
        X_test = torch.rand(3, 1, **self.tkwargs)
        posterior = tw_model.posterior(X_test)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, X_test.shape)

    def test_state_dict(self) -> None:
        """Test serialization via state_dict and load_state_dict."""
        tw_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )
        # Set a specific weight
        original_weight = torch.tensor([0.3, 0.7], **self.tkwargs)
        tw_model.weight = original_weight

        # Save state dict
        state_dict = tw_model.state_dict()

        # Create new model and load state dict
        new_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=self.base_model_dict,
        )
        new_model.load_state_dict(state_dict)

        # Verify weights match
        self.assertTrue(torch.allclose(new_model.weight, original_weight))

    def test_single_base_model(self) -> None:
        """Test with a single base model (edge case)."""
        single_model_dict = {
            "auxiliary_metric1": SingleTaskGP(
                self.train_X, self.train_Y, self.train_Yvar
            )
        }
        mll = ExactMarginalLogLikelihood(
            single_model_dict["auxiliary_metric1"].likelihood,
            single_model_dict["auxiliary_metric1"],
        )
        fit_gpytorch_mll(
            mll, optimizer_kwargs={"options": {"maxiter": 5}}, max_attempts=1
        )

        tw_model = TargetAwareEnsembleGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            train_Yvar=self.train_Yvar,
            base_model_dict=single_model_dict,
        )
        # Test posterior
        X_test = torch.rand(3, 1, **self.tkwargs)
        posterior = tw_model.posterior(X_test)
        self.assertIsInstance(posterior, GPyTorchPosterior)
        self.assertEqual(posterior.mean.shape, X_test.shape)
