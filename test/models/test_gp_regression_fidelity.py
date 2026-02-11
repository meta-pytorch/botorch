#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase, get_random_data
from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


def _get_random_data_with_fidelity(
    batch_shape: torch.Size, m: int, n_fidelity: int, d: int = 1, n: int = 10, **tkwargs
) -> tuple[Tensor, Tensor]:
    r"""Construct test data.
    For this test, by convention the trailing dimensions are the fidelity dimensions
    """
    train_x, train_y = get_random_data(
        batch_shape=batch_shape, m=m, d=d, n=n, **tkwargs
    )
    s = torch.rand(n, n_fidelity, **tkwargs).repeat(batch_shape + torch.Size([1, 1]))
    train_x = torch.cat((train_x, s), dim=-1)
    train_y = train_y + (1 - s).pow(2).sum(dim=-1).unsqueeze(-1)
    return train_x, train_y


class TestSingleTaskMultiFidelityGP(BotorchTestCase):
    FIDELITY_TEST_PAIRS = (
        # (iteration_fidelity, data_fidelities)
        (None, [1]),
        (1, None),
        (None, [-1]),
        (-1, None),
        (1, [2]),
        (1, [2, 3]),
        (None, [1, 2]),
        (-1, [1, -2]),
    )

    def _get_model_and_data(
        self,
        iteration_fidelity,
        data_fidelities,
        batch_shape,
        m,
        lin_truncated,
        outcome_transform=None,
        input_transform=None,
        **tkwargs,
    ):
        model_kwargs = {}
        n_fidelity = iteration_fidelity is not None
        if data_fidelities is not None:
            n_fidelity += len(data_fidelities)
            model_kwargs["data_fidelities"] = data_fidelities
        train_X, train_Y = _get_random_data_with_fidelity(
            batch_shape=batch_shape, m=m, n_fidelity=n_fidelity, **tkwargs
        )
        model_kwargs.update(
            {
                "train_X": train_X,
                "train_Y": train_Y,
                "iteration_fidelity": iteration_fidelity,
                "linear_truncated": lin_truncated,
                "outcome_transform": outcome_transform,
                "input_transform": input_transform,
            }
        )
        model = SingleTaskMultiFidelityGP(**model_kwargs)
        return model, model_kwargs

    def test_init_error(self) -> None:
        train_X = torch.rand(2, 2, device=self.device)
        train_Y = torch.rand(2, 1)
        for lin_truncated in (True, False):
            no_fidelity_msg = (
                "SingleTaskMultiFidelityGP requires at least one fidelity parameter."
            )
            with self.assertRaisesRegex(UnsupportedError, no_fidelity_msg):
                SingleTaskMultiFidelityGP(
                    train_X, train_Y, linear_truncated=lin_truncated
                )
            with self.assertRaisesRegex(UnsupportedError, no_fidelity_msg):
                SingleTaskMultiFidelityGP(
                    train_X, train_Y, linear_truncated=lin_truncated, data_fidelities=[]
                )

    def test_gp(self) -> None:
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            num_dim = 1 + (iteration_fidelity is not None)
            if data_fidelities is not None:
                num_dim += len(data_fidelities)
            bounds = torch.zeros(2, num_dim)
            bounds[1] = 1
            for (
                batch_shape,
                m,
                dtype,
                lin_trunc,
                use_octf,
                use_intf,
            ) in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
                (False, True),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
                intf = Normalize(d=num_dim, bounds=bounds) if use_intf else None
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    outcome_transform=octf,
                    input_transform=intf,
                    **tkwargs,
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                mll.to(**tkwargs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizationWarning)
                    fit_gpytorch_mll(
                        mll,
                        optimizer_kwargs={"options": {"maxiter": 1}},
                        sequential=False,
                    )

                # test init
                self.assertIsInstance(model.mean_module, ConstantMean)
                self.assertIsInstance(
                    model.covar_module, ScaleKernel if lin_trunc else ProductKernel
                )
                if use_octf:
                    self.assertIsInstance(model.outcome_transform, Standardize)
                if use_intf:
                    self.assertIsInstance(model.input_transform, Normalize)
                    # permute output dim
                    train_X, train_Y, _ = model._transform_tensor_args(
                        X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                    )
                    # check that the train inputs have been transformed and set on the
                    # model
                    self.assertTrue(torch.equal(model.train_inputs[0], intf(train_X)))

                # test param sizes
                params = dict(model.named_parameters())

                if data_fidelities is not None and len(data_fidelities) == 1:
                    for p in params:
                        self.assertEqual(
                            params[p].numel(),
                            m * torch.tensor(batch_shape).prod().item(),
                        )

                # test posterior
                # test non batch evaluation
                X = torch.rand(*batch_shape, 3, num_dim, **tkwargs)
                expected_shape = batch_shape + torch.Size([3, m])
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, expected_shape)
                self.assertEqual(posterior.variance.shape, expected_shape)
                if use_octf:
                    # ensure un-transformation is applied
                    tmp_tf = model.outcome_transform
                    del model.outcome_transform
                    pp_tf = model.posterior(X)
                    model.outcome_transform = tmp_tf
                    expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                    self.assertAllClose(posterior.variance, expected_var)

                # test batch evaluation
                X = torch.rand(2, *batch_shape, 3, num_dim, **tkwargs)
                expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, m])
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, expected_shape)
                self.assertEqual(posterior.variance.shape, expected_shape)
                if use_octf:
                    # ensure un-transformation is applied
                    tmp_tf = model.outcome_transform
                    del model.outcome_transform
                    pp_tf = model.posterior(X)
                    model.outcome_transform = tmp_tf
                    expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                    self.assertAllClose(posterior.variance, expected_var)

    def test_condition_on_observations(self):
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            n_fidelity = iteration_fidelity is not None
            if data_fidelities is not None:
                n_fidelity += len(data_fidelities)
            num_dim = 1 + n_fidelity
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # evaluate model
                model.posterior(torch.rand(torch.Size([4, num_dim]), **tkwargs))
                # test condition_on_observations
                fant_shape = torch.Size([2])
                # fantasize at different input points
                X_fant, Y_fant = _get_random_data_with_fidelity(
                    fant_shape + batch_shape, m, n_fidelity=n_fidelity, n=3, **tkwargs
                )
                c_kwargs = (
                    {"noise": torch.full_like(Y_fant, 0.01)}
                    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                    else {}
                )
                cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
                # fantasize at different same input points
                c_kwargs_same_inputs = (
                    {"noise": torch.full_like(Y_fant[0], 0.01)}
                    if isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
                    else {}
                )
                cm_same_inputs = model.condition_on_observations(
                    X_fant[0], Y_fant, **c_kwargs_same_inputs
                )

                test_Xs = [
                    # test broadcasting single input across fantasy and
                    # model batches
                    torch.rand(4, num_dim, **tkwargs),
                    # separate input for each model batch and broadcast across
                    # fantasy batches
                    torch.rand(batch_shape + torch.Size([4, num_dim]), **tkwargs),
                    # separate input for each model and fantasy batch
                    torch.rand(
                        fant_shape + batch_shape + torch.Size([4, num_dim]), **tkwargs
                    ),
                ]
                for test_X in test_Xs:
                    posterior = cm.posterior(test_X)
                    self.assertEqual(
                        posterior.mean.shape,
                        fant_shape + batch_shape + torch.Size([4, m]),
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

                        model_kwargs_non_batch = {}
                        for k, v in model_kwargs.items():
                            if k in (
                                "iteration_fidelity",
                                "data_fidelities",
                                "linear_truncated",
                                "outcome_transform",
                                "input_transform",
                            ):
                                model_kwargs_non_batch[k] = v
                            else:
                                model_kwargs_non_batch[k] = v[0]

                        model_non_batch = type(model)(**model_kwargs_non_batch)
                        model_non_batch.load_state_dict(state_dict_non_batch)
                        model_non_batch.eval()
                        model_non_batch.likelihood.eval()
                        model_non_batch.posterior(
                            torch.rand(torch.Size([4, num_dim]), **tkwargs)
                        )
                        c_kwargs = (
                            {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                            if isinstance(
                                model.likelihood, FixedNoiseGaussianLikelihood
                            )
                            else {}
                        )
                        mnb = model_non_batch
                        cm_non_batch = mnb.condition_on_observations(
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
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            n_fidelity = iteration_fidelity is not None
            if data_fidelities is not None:
                n_fidelity += len(data_fidelities)
            num_dim = 1 + n_fidelity
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # fantasize
                X_f = torch.rand(
                    torch.Size(batch_shape + torch.Size([4, num_dim])), **tkwargs
                )
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
                fm = model.fantasize(X=X_f, sampler=sampler)
                self.assertIsInstance(fm, model.__class__)

    def test_subset_model(self):
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            num_dim = 1 + (iteration_fidelity is not None)
            if data_fidelities is not None:
                num_dim += len(data_fidelities)
            for batch_shape, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, _ = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=2,
                    lin_truncated=lin_trunc,
                    outcome_transform=None,  # TODO: Subset w/ outcome transform
                    **tkwargs,
                )
                subset_model = model.subset_output([0])
                X = torch.rand(
                    torch.Size(batch_shape + torch.Size([3, num_dim])), **tkwargs
                )
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

    def test_construct_inputs(self):
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            for batch_shape, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=1,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )

                X = kwargs["train_X"]
                training_data = SupervisedDataset(
                    X=X,
                    Y=kwargs["train_Y"],
                    feature_names=[f"x{i}" for i in range(X.shape[-1])],
                    outcome_names=["y"],
                )

                # missing fidelity features
                with self.assertRaisesRegex(TypeError, "argument: 'fidelity_features'"):
                    model.construct_inputs(training_data)

                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue("data_fidelities" in data_dict)
                self.assertEqual(data_dict["data_fidelities"], [1])
                self.assertTrue(kwargs["train_X"].equal(data_dict["train_X"]))
                self.assertTrue(kwargs["train_Y"].equal(data_dict["train_Y"]))

    def test_cont_kernel_factory(self):
        iteration_fidelity, data_fidelities = self.FIDELITY_TEST_PAIRS[0]
        d = 1
        m = 2
        batch_shape = torch.Size([2, 3])
        aug_batch_shape = batch_shape + torch.Size([m])
        tkwargs = {"device": self.device, "dtype": torch.double}

        _, model_kwargs = self._get_model_and_data(
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            batch_shape=batch_shape,
            m=m,
            lin_truncated=False,
            **tkwargs,
        )

        def custom_kernel_factory(
            ard_num_dims: int,
            batch_shape: torch.Size | None = None,
            active_dims: list[int] | None = None,
        ):
            return LinearKernel(
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
                active_dims=active_dims,
            )

        expected_kernels = (RBFKernel, LinearKernel)
        expected_subset_batch_dict = {
            "mean_module.raw_constant": -1,
            "covar_module.kernels.1.raw_power": -2,
            "covar_module.kernels.1.raw_offset": -2,
        }
        if model_kwargs.get("train_Yvar", None) is None:
            expected_subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2

        for cont_kernel_factory, expected_kernel in zip(
            (None, custom_kernel_factory), expected_kernels
        ):
            model = SingleTaskMultiFidelityGP(
                **model_kwargs, cont_kernel_factory=cont_kernel_factory
            )
            non_fid_kernel = model.covar_module.kernels[0]
            self.assertIsInstance(non_fid_kernel, expected_kernel)
            self.assertEqual(non_fid_kernel.ard_num_dims, d)
            self.assertEqual(non_fid_kernel.batch_shape, aug_batch_shape)
            self.assertEqual(
                non_fid_kernel.active_dims,
                torch.as_tensor(list(range(d)), device=self.device),
            )

            extra_subset_batch_dict = (
                {
                    "covar_module.kernels.0.raw_lengthscale": -3,
                }
                if cont_kernel_factory is None
                else {}
            )
            self.assertEqual(
                model._subset_batch_dict,
                {**expected_subset_batch_dict, **extra_subset_batch_dict},
            )

        model_kwargs.update({"linear_truncated": True})
        with self.assertRaises(ValueError):
            model = SingleTaskMultiFidelityGP(
                **model_kwargs, cont_kernel_factory=custom_kernel_factory
            )


class TestFixedNoiseSingleTaskMultiFidelityGP(TestSingleTaskMultiFidelityGP):
    def _get_model_and_data(
        self,
        iteration_fidelity,
        data_fidelities,
        batch_shape,
        m,
        lin_truncated,
        outcome_transform=None,
        input_transform=None,
        **tkwargs,
    ):
        model_kwargs = {}
        n_fidelity = iteration_fidelity is not None
        if data_fidelities is not None:
            n_fidelity += len(data_fidelities)
            model_kwargs["data_fidelities"] = data_fidelities
        train_X, train_Y = _get_random_data_with_fidelity(
            batch_shape=batch_shape, m=m, n_fidelity=n_fidelity, **tkwargs
        )
        train_Yvar = torch.full_like(train_Y, 0.01)
        model_kwargs.update(
            {
                "train_X": train_X,
                "train_Y": train_Y,
                "train_Yvar": train_Yvar,
                "iteration_fidelity": iteration_fidelity,
                "linear_truncated": lin_truncated,
                "outcome_transform": outcome_transform,
                "input_transform": input_transform,
            }
        )
        model = SingleTaskMultiFidelityGP(**model_kwargs)
        return model, model_kwargs

    def test_init_error(self):
        train_X = torch.rand(2, 2, device=self.device)
        train_Y = torch.rand(2, 1)
        train_Yvar = torch.full_like(train_Y, 0.01)
        for lin_truncated in (True, False):
            with self.assertRaises(UnsupportedError):
                SingleTaskMultiFidelityGP(
                    train_X, train_Y, train_Yvar, linear_truncated=lin_truncated
                )

    def test_fixed_noise_likelihood(self):
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
                self.assertTrue(
                    torch.equal(
                        model.likelihood.noise.contiguous().view(-1),
                        model_kwargs["train_Yvar"].contiguous().view(-1),
                    )
                )

    def test_construct_inputs(self):
        for iteration_fidelity, data_fidelities in self.FIDELITY_TEST_PAIRS:
            for batch_shape, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelities=data_fidelities,
                    batch_shape=batch_shape,
                    m=1,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                X = kwargs["train_X"]
                training_data = SupervisedDataset(
                    X=X,
                    Y=kwargs["train_Y"],
                    feature_names=[f"x{i}" for i in range(X.shape[-1])],
                    outcome_names=["y"],
                )
                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue("train_Yvar" not in data_dict)

                # len(Xs) == len(Ys) == 1
                training_data = SupervisedDataset(
                    X=kwargs["train_X"],
                    Y=kwargs["train_Y"],
                    Yvar=torch.full(kwargs["train_Y"].shape[:-1] + (1,), 0.1),
                    feature_names=[f"x{i}" for i in range(X.shape[-1])],
                    outcome_names=["y"],
                )

                # missing fidelity features
                with self.assertRaisesRegex(TypeError, "argument: 'fidelity_features'"):
                    model.construct_inputs(training_data)

                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue("train_Yvar" in data_dict)
                self.assertEqual(data_dict.get("data_fidelities", None), [1])
                self.assertTrue(kwargs["train_X"].equal(data_dict["train_X"]))
                self.assertTrue(kwargs["train_Y"].equal(data_dict["train_Y"]))
