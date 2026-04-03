#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import OptimizationWarning, UnsupportedError
from botorch.exceptions.warnings import _get_single_precision_warning, InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.pairwise import (
    PairwiseLikelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.model import Model
from botorch.models.pairwise_gp import (
    _ensure_psd_with_jitter,
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)
from botorch.models.transforms.input import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling.pairwise_samplers import PairwiseSobolQMCNormalSampler
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior, SmoothedBoxPrior
from linear_operator.utils.errors import NotPSDError
from torch import Tensor


class TestPairwiseGP(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings)
        # single-precision tests are carried out by TestPairwiseGP_float32
        self.dtype = torch.float64

    def _make_rand_mini_data(
        self,
        batch_shape,
        X_dim=2,
    ) -> tuple[Tensor, Tensor]:
        train_X = torch.rand(
            *batch_shape, 2, X_dim, device=self.device, dtype=self.dtype
        )
        train_Y = train_X.sum(dim=-1, keepdim=True)
        train_comp = torch.topk(train_Y, k=2, dim=-2).indices.transpose(-1, -2)

        return train_X, train_comp

    def _get_model_and_data(
        self,
        batch_shape,
        X_dim=2,
        likelihood_cls=None,
    ) -> tuple[Model, dict[str, Tensor | PairwiseLikelihood]]:
        train_X, train_comp = self._make_rand_mini_data(
            batch_shape=batch_shape,
            X_dim=X_dim,
        )

        model_kwargs = {
            "datapoints": train_X,
            "comparisons": train_comp,
            "likelihood": None if likelihood_cls is None else likelihood_cls(),
        }
        model = PairwiseGP(**model_kwargs)
        return model, model_kwargs

    def test_construct_inputs(self) -> None:
        datapoints = torch.rand(3, 2)
        indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        event_shape = torch.Size([2 * datapoints.shape[-1]])
        dataset_X = SliceContainer(
            values=datapoints, indices=indices, event_shape=event_shape
        )
        dataset_Y = torch.tensor([[0, 1], [1, 0]]).expand(indices.shape)
        dataset = RankingDataset(
            X=dataset_X, Y=dataset_Y, feature_names=["a", "b"], outcome_names=["y"]
        )
        model_inputs = PairwiseGP.construct_inputs(dataset)
        comparisons = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
        self.assertSetEqual(set(model_inputs.keys()), {"datapoints", "comparisons"})
        self.assertTrue(torch.equal(model_inputs["datapoints"], datapoints))
        self.assertTrue(torch.equal(model_inputs["comparisons"], comparisons))

        with self.subTest("Input other than RankingDataset"):
            dataset = SupervisedDataset(
                X=datapoints,
                Y=torch.rand(3, 1),
                feature_names=["a", "b"],
                outcome_names=["y"],
            )
            with self.assertRaisesRegex(
                UnsupportedError, "Only `RankingDataset` is supported"
            ):
                PairwiseGP.construct_inputs(dataset)

    def test_pairwise_gp(self) -> None:
        torch.manual_seed(random.randint(0, 10))
        for batch_shape, likelihood_cls in itertools.product(
            (torch.Size(), torch.Size([2])),
            (PairwiseLogitLikelihood, PairwiseProbitLikelihood),
        ):
            tkwargs = {"device": self.device, "dtype": self.dtype}
            X_dim = 2

            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                X_dim=X_dim,
                likelihood_cls=likelihood_cls,
            )
            train_X = model_kwargs["datapoints"]
            train_comp = model_kwargs["comparisons"]

            # test training
            # regular training
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model).to(
                **tkwargs
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                fit_gpytorch_mll(
                    mll, optimizer_kwargs={"options": {"maxiter": 2}}, max_attempts=1
                )
            with self.subTest("prior training"):
                # prior training
                prior_m = PairwiseGP(None, None).to(**tkwargs)
                with self.assertRaises(RuntimeError):
                    prior_m(train_X)

            with self.subTest("forward in training mode with non-training data"):
                custom_m = PairwiseGP(**model_kwargs)
                other_X = torch.rand(batch_shape + torch.Size([3, X_dim]), **tkwargs)
                other_comp = train_comp.clone()
                with self.assertRaises(RuntimeError):
                    custom_m(other_X)
                custom_mll = PairwiseLaplaceMarginalLogLikelihood(
                    custom_m.likelihood, custom_m
                ).to(**tkwargs)
                post = custom_m(train_X)
                with self.assertRaises(RuntimeError):
                    custom_mll(post, other_comp)

            with self.subTest("init"):
                self.assertIsInstance(model.mean_module, ConstantMean)
                self.assertIsInstance(model.covar_module, ScaleKernel)
                self.assertIsInstance(model.covar_module.base_kernel, RBFKernel)
                self.assertIsInstance(
                    model.covar_module.base_kernel.lengthscale_prior, GammaPrior
                )
                self.assertIsInstance(
                    model.covar_module.outputscale_prior, SmoothedBoxPrior
                )
                self.assertEqual(model.num_outputs, 1)
                self.assertEqual(model.batch_shape, batch_shape)

            # test not using a ScaleKernel
            with self.assertRaisesRegex(UnsupportedError, "used with a ScaleKernel"):
                PairwiseGP(**model_kwargs, covar_module=LinearKernel())

            # test custom models
            custom_m = PairwiseGP(
                **model_kwargs, covar_module=ScaleKernel(LinearKernel())
            )
            self.assertIsInstance(custom_m.covar_module, ScaleKernel)
            self.assertIsInstance(custom_m.covar_module.base_kernel, LinearKernel)

            # prior prediction
            prior_m = PairwiseGP(None, None).to(**tkwargs)
            prior_m.eval()
            post = prior_m.posterior(train_X)
            self.assertIsInstance(post, GPyTorchPosterior)

            # test initial utility val
            util_comp = torch.topk(model.utility, k=2, dim=-1).indices.unsqueeze(-2)
            self.assertTrue(torch.all(util_comp == train_comp))

            # test posterior
            # test non batch evaluation
            X = torch.rand(batch_shape + torch.Size([3, X_dim]), **tkwargs)
            expected_shape = batch_shape + torch.Size([3, 1])
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)
            self.assertEqual(posterior.variance.shape, expected_shape)

            # test posterior transform
            post_tf = ScalarizedPosteriorTransform(weights=torch.ones(1))
            posterior_tf = model.posterior(X, posterior_transform=post_tf)
            self.assertTrue(torch.equal(posterior.mean, posterior_tf.mean))

            # expect to raise error when output_indices is not None
            with self.assertRaises(RuntimeError):
                model.posterior(X, output_indices=[0])

            # test re-evaluating utility when it's None
            model.utility = None
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)

            # test batch evaluation
            X = torch.rand(2, *batch_shape, 3, X_dim, **tkwargs)
            expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, 1])

            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, expected_shape)

            # test input_transform
            # the untransfomed one should be stored
            normalize_tf = Normalize(d=2, bounds=torch.tensor([[0, 0], [0.5, 1.5]]))
            model = PairwiseGP(**model_kwargs, input_transform=normalize_tf)
            self.assertTrue(torch.equal(model.datapoints, train_X))

            # test set_train_data strict mode
            model = PairwiseGP(**model_kwargs)
            changed_train_X = train_X.unsqueeze(0)
            changed_train_comp = train_comp.unsqueeze(0)
            # expect to raise error when set data to something different
            with self.assertRaises(RuntimeError):
                model.set_train_data(changed_train_X, changed_train_comp, strict=True)

            # the same datapoints but changed comparison will also raise error
            with self.assertRaises(RuntimeError):
                model.set_train_data(train_X, changed_train_comp, strict=True)

    def test_consolidation(self) -> None:
        for batch_shape, likelihood_cls in itertools.product(
            (torch.Size(), torch.Size([2])),
            (PairwiseLogitLikelihood, PairwiseProbitLikelihood),
        ):
            X_dim = 2

            _, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                X_dim=X_dim,
                likelihood_cls=likelihood_cls,
            )
            train_X = model_kwargs["datapoints"]
            train_comp = model_kwargs["comparisons"]

            # Test consolidation
            i1, i2 = train_X.shape[-2], train_X.shape[-2] + 1
            dup_comp = torch.cat(
                [
                    train_comp,
                    torch.tensor(
                        [[i1, i2]], dtype=train_comp.dtype, device=train_comp.device
                    ).expand(*batch_shape, 1, 2),
                ],
                dim=-2,
            )
            dup_X = torch.cat([train_X, train_X[..., :2, :]], dim=-2)
            model = PairwiseGP(datapoints=dup_X, comparisons=dup_comp)
            self.assertIs(dup_X, model.unconsolidated_datapoints)
            self.assertIs(dup_comp, model.unconsolidated_comparisons)
            if batch_shape:
                self.assertIs(dup_X, model.consolidated_datapoints)
                self.assertIs(dup_comp, model.consolidated_comparisons)
                self.assertIs(model.utility, model.unconsolidated_utility)
            else:
                self.assertFalse(torch.equal(dup_X, model.consolidated_datapoints))
                self.assertFalse(torch.equal(dup_comp, model.consolidated_comparisons))
                self.assertFalse(
                    torch.equal(model.utility, model.unconsolidated_utility)
                )

            # calling forward with duplicated datapoints should work after consolidation
            mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            # make sure model is in training mode
            self.assertTrue(model.training)
            pred = model(dup_X)
            # posterior shape in training should match the consolidated utility
            self.assertEqual(pred.shape(), model.utility.shape)
            if batch_shape:
                # do not perform consolidation in batch mode
                # because the block structure cannot be guaranteed
                self.assertEqual(pred.shape(), dup_X.shape[:-1])
            else:
                self.assertEqual(pred.shape(), train_X.shape[:-1])
            # Pass the original comparisons through mll should work
            mll(pred, dup_comp)

    def test_condition_on_observations(self) -> None:
        for batch_shape, likelihood_cls in itertools.product(
            (torch.Size(), torch.Size([2])),
            (PairwiseLogitLikelihood, PairwiseProbitLikelihood),
        ):
            tkwargs = {"device": self.device, "dtype": self.dtype}
            X_dim = 2

            model, model_kwargs = self._get_model_and_data(
                batch_shape=batch_shape,
                X_dim=X_dim,
                likelihood_cls=likelihood_cls,
            )
            train_X = model_kwargs["datapoints"]
            train_comp = model_kwargs["comparisons"]

            # evaluate model
            model.posterior(torch.rand(torch.Size([4, X_dim]), **tkwargs))

            # test condition_on_observations

            # test condition_on_observations with prior mode
            prior_m = PairwiseGP(None, None).to(**tkwargs)
            cond_m = prior_m.condition_on_observations(train_X, train_comp)
            self.assertIs(cond_m.datapoints, train_X)
            self.assertIs(cond_m.comparisons, train_comp)

            # fantasize at different input points
            fant_shape = torch.Size([2])
            X_fant, comp_fant = self._make_rand_mini_data(
                batch_shape=fant_shape + batch_shape,
                X_dim=X_dim,
            )

            # cannot condition on non-pairwise Ys
            with self.assertRaises(RuntimeError):
                model.condition_on_observations(X_fant, comp_fant[..., 0])
            cm = model.condition_on_observations(X_fant, comp_fant)
            # make sure it's a deep copy
            self.assertTrue(model is not cm)

            # fantasize at same input points (check proper broadcasting)
            cm_same_inputs = model.condition_on_observations(X_fant[0], comp_fant)

            test_Xs = [
                # test broadcasting single input across fantasy and model batches
                torch.rand(4, X_dim, **tkwargs),
                # separate input for each model batch and broadcast across
                # fantasy batches
                torch.rand(batch_shape + torch.Size([4, X_dim]), **tkwargs),
                # separate input for each model and fantasy batch
                torch.rand(
                    fant_shape + batch_shape + torch.Size([4, X_dim]), **tkwargs
                ),
            ]
            for test_X in test_Xs:
                posterior = cm.posterior(test_X)
                self.assertEqual(
                    posterior.mean.shape, fant_shape + batch_shape + torch.Size([4, 1])
                )
                posterior_same_inputs = cm_same_inputs.posterior(test_X)
                self.assertEqual(
                    posterior_same_inputs.mean.shape,
                    fant_shape + batch_shape + torch.Size([4, 1]),
                )

                # check that fantasies of batched model are correct
                if len(batch_shape) > 0 and test_X.dim() == 2:
                    state_dict_non_batch = {
                        key: (val[0] if val.numel() > 1 else val)
                        for key, val in model.state_dict().items()
                    }
                    model_kwargs_non_batch = {
                        "datapoints": model_kwargs["datapoints"][0],
                        "comparisons": model_kwargs["comparisons"][0],
                        "likelihood": likelihood_cls(),
                    }
                    model_non_batch = model.__class__(**model_kwargs_non_batch)
                    model_non_batch.load_state_dict(state_dict_non_batch)
                    model_non_batch.eval()
                    model_non_batch.posterior(
                        torch.rand(torch.Size([4, X_dim]), **tkwargs)
                    )
                    cm_non_batch = model_non_batch.condition_on_observations(
                        X_fant[0][0], comp_fant[:, 0, :]
                    )
                    non_batch_posterior = cm_non_batch.posterior(test_X)
                    self.assertAllClose(
                        posterior_same_inputs.mean[:, 0, ...],
                        non_batch_posterior.mean,
                        atol=1e-3,
                    )
                    self.assertAllClose(
                        posterior_same_inputs.distribution.covariance_matrix[
                            :, 0, :, :
                        ],
                        non_batch_posterior.distribution.covariance_matrix,
                        atol=1e-3,
                    )

    def test_fantasize(self) -> None:
        for batch_shape, likelihood_cls in itertools.product(
            (torch.Size(), torch.Size([2])),
            (PairwiseLogitLikelihood, PairwiseProbitLikelihood),
        ):
            tkwargs = {"device": self.device, "dtype": self.dtype}
            X_dim = 2

            model, _ = self._get_model_and_data(
                batch_shape=batch_shape,
                X_dim=X_dim,
                likelihood_cls=likelihood_cls,
            )

            # fantasize
            X_f = torch.rand(
                torch.Size(batch_shape + torch.Size([4, X_dim])), **tkwargs
            )
            sampler = PairwiseSobolQMCNormalSampler(sample_shape=torch.Size([3]))
            fm = model.fantasize(X=X_f, sampler=sampler)
            self.assertIsInstance(fm, model.__class__)

    def test_load_state_dict(self) -> None:
        model, _ = self._get_model_and_data(batch_shape=[])
        sd = model.state_dict()
        with self.assertRaises(UnsupportedError):
            model.load_state_dict(sd, strict=True)

        # Set instance buffers to None
        for buffer_name in model._buffer_names:
            model.register_buffer(buffer_name, None)

        # Check that instance buffers were not restored
        _ = model.load_state_dict(sd)
        for buffer_name in model._buffer_names:
            self.assertIsNone(model.get_buffer(buffer_name))

    def test_helper_functions(self) -> None:
        for batch_shape in (torch.Size(), torch.Size([2])):
            tkwargs = {"device": self.device, "dtype": self.dtype}
            # M is borderline PSD
            M = torch.ones((*batch_shape, 2, 2), **tkwargs)
            with self.assertRaises(torch._C._LinAlgError):
                torch.linalg.cholesky(M)
            # This should work fine
            _ensure_psd_with_jitter(M)

            bad_M = torch.tensor([[1.0, 2.0], [2.0, 1.0]], **tkwargs).expand(
                (*batch_shape, 2, 2)
            )
            with self.assertRaises(NotPSDError):
                _ensure_psd_with_jitter(bad_M)


class TestPairwiseGP_float32(TestPairwiseGP):
    """Runs tests from TestPairwiseGP in single precision."""

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings)
        self.dtype = torch.float32
        warnings.filterwarnings(
            "ignore",
            category=InputDataWarning,
            message=_get_single_precision_warning(str(torch.float32)),
        )

    def test_init_warns_on_single_precision(self) -> None:
        with self.assertWarnsRegex(
            InputDataWarning,
            expected_regex=_get_single_precision_warning(str(torch.float32)),
        ):
            self._get_model_and_data(batch_shape=torch.Size([]))
