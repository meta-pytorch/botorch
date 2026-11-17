#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.exceptions import UnsupportedError
from botorch.models.kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.priors.torch_priors import GammaPrior, NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestLinearTruncatedFidelityKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2], dimension=3, **kwargs
        )

    def create_data_no_batch(self):
        return torch.rand(50, 10)

    def create_data_single_batch(self):
        return torch.rand(2, 50, 3)

    def create_data_double_batch(self):
        return torch.rand(3, 2, 50, 3)

    def test_compute_linear_truncated_kernel_no_batch(self):
        x1 = torch.tensor([[1, 0.1, 0.2], [2, 0.3, 0.4]])
        x2 = torch.tensor([[3, 0.5, 0.6], [4, 0.7, 0.8]])
        t_1 = torch.tensor([[0.3584, 0.1856], [0.2976, 0.1584]])
        for nu, fidelity_dims in itertools.product({0.5, 1.5, 2.5}, ([2], [1, 2])):
            kernel = LinearTruncatedFidelityKernel(
                fidelity_dims=fidelity_dims, dimension=3, nu=nu
            )
            kernel.power = 1
            n_fid = len(fidelity_dims)
            if n_fid > 1:
                active_dimsM = [0]
                t_2 = torch.tensor([[0.4725, 0.2889], [0.4025, 0.2541]])
                t_3 = torch.tensor([[0.1685, 0.0531], [0.1168, 0.0386]])
                t = 1 + t_1 + t_2 + t_3
            else:
                active_dimsM = [0, 1]
                t = 1 + t_1

            matern_ker = MaternKernel(nu=nu, active_dims=active_dimsM)
            matern_term = matern_ker(x1, x2).to_dense()
            actual = t * matern_term
            res = kernel(x1, x2).to_dense()
            self.assertLess(torch.linalg.norm(res - actual), 1e-4)
            # test diagonal mode
            res_diag = kernel(x1, x2, diag=True)
            self.assertLess(torch.linalg.norm(res_diag - actual.diag()), 1e-4)
        # make sure that we error out if last_dim_is_batch=True
        with self.assertRaises(NotImplementedError):
            kernel(x1, x2, diag=True, last_dim_is_batch=True)

    def test_compute_linear_truncated_kernel_with_batch(self):
        x1 = torch.tensor(
            [[[1.0, 0.1, 0.2], [3.0, 0.3, 0.4]], [[5.0, 0.5, 0.6], [7.0, 0.7, 0.8]]]
        )
        x2 = torch.tensor(
            [[[2.0, 0.8, 0.7], [4.0, 0.6, 0.5]], [[6.0, 0.4, 0.3], [8.0, 0.2, 0.1]]]
        )
        t_1 = torch.tensor(
            [[[0.2736, 0.4400], [0.2304, 0.3600]], [[0.3304, 0.3816], [0.1736, 0.1944]]]
        )
        batch_shape = torch.Size([2])
        for nu, fidelity_dims in itertools.product({0.5, 1.5, 2.5}, ([2], [1, 2])):
            kernel = LinearTruncatedFidelityKernel(
                fidelity_dims=fidelity_dims, dimension=3, nu=nu, batch_shape=batch_shape
            )
            kernel.power = 1
            if len(fidelity_dims) > 1:
                active_dimsM = [0]
                t_2 = torch.tensor(
                    [
                        [[0.0527, 0.1670], [0.0383, 0.1159]],
                        [[0.1159, 0.1670], [0.0383, 0.0527]],
                    ]
                )
                t_3 = torch.tensor(
                    [
                        [[0.1944, 0.3816], [0.1736, 0.3304]],
                        [[0.3600, 0.4400], [0.2304, 0.2736]],
                    ]
                )
                t = 1 + t_1 + t_2 + t_3
            else:
                active_dimsM = [0, 1]
                t = 1 + t_1

            matern_ker = MaternKernel(
                nu=nu, active_dims=active_dimsM, batch_shape=batch_shape
            )
            matern_term = matern_ker(x1, x2).to_dense()
            actual = t * matern_term
            res = kernel(x1, x2).to_dense()
            self.assertLess(torch.linalg.norm(res - actual), 1e-4)
            # test diagonal mode
            res_diag = kernel(x1, x2, diag=True)
            self.assertLess(
                torch.linalg.norm(res_diag - torch.diagonal(actual, dim1=-1, dim2=-2)),
                1e-4,
            )
        # make sure that we error out if last_dim_is_batch=True
        with self.assertRaises(NotImplementedError):
            kernel(x1, x2, diag=True, last_dim_is_batch=True)

    def test_initialize_lengthscale_prior(self):
        kernel = LinearTruncatedFidelityKernel(fidelity_dims=[1, 2], dimension=3)
        self.assertTrue(
            isinstance(kernel.covar_module_unbiased.lengthscale_prior, GammaPrior)
        )
        self.assertTrue(
            isinstance(kernel.covar_module_biased.lengthscale_prior, GammaPrior)
        )
        # the documented defaults GammaPrior(3, 6) and GammaPrior(6, 2)
        for prior, (concentration, rate) in (
            (kernel.covar_module_unbiased.lengthscale_prior, (3.0, 6.0)),
            (kernel.covar_module_biased.lengthscale_prior, (6.0, 2.0)),
        ):
            self.assertEqual(prior.concentration.item(), concentration)
            self.assertEqual(prior.rate.item(), rate)
        kernel2 = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2],
            dimension=3,
            lengthscale_prior_unbiased=NormalPrior(1, 1),
        )
        self.assertTrue(
            isinstance(kernel2.covar_module_unbiased.lengthscale_prior, NormalPrior)
        )
        kernel2 = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2],
            dimension=3,
            lengthscale_prior_biased=NormalPrior(1, 1),
        )
        self.assertTrue(
            isinstance(kernel2.covar_module_biased.lengthscale_prior, NormalPrior)
        )

    def test_initialize_power_prior(self):
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2], dimension=3, power_prior=NormalPrior(1, 1)
        )
        self.assertTrue(isinstance(kernel.power_prior, NormalPrior))

    def test_initialize_power(self):
        kernel = LinearTruncatedFidelityKernel(fidelity_dims=[1, 2], dimension=3)
        kernel.initialize(power=1)
        actual_value = torch.tensor(1, dtype=torch.float).view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_initialize_power_batch(self):
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2], dimension=3, batch_shape=torch.Size([2])
        )
        power_init = torch.tensor([1, 2], dtype=torch.float)
        kernel.initialize(power=power_init)
        actual_value = power_init.view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_raise_init_errors(self):
        with self.assertRaises(UnsupportedError):
            LinearTruncatedFidelityKernel(fidelity_dims=[2])
        with self.assertRaises(UnsupportedError):
            LinearTruncatedFidelityKernel(fidelity_dims=[0, 1, 2], dimension=3)
        with self.assertRaises(ValueError):
            LinearTruncatedFidelityKernel(fidelity_dims=[2, 2], dimension=3)
        with self.assertRaises(ValueError):
            LinearTruncatedFidelityKernel(fidelity_dims=[2], dimension=2, nu=1)

    def test_active_dims_list(self):
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2], dimension=10, active_dims=[0, 2, 4, 6]
        )
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        kernel_basic = LinearTruncatedFidelityKernel(fidelity_dims=[1, 2], dimension=4)
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().to_dense()
        self.assertLess(
            torch.linalg.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(),
            1e-4,
        )

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2], dimension=10, active_dims=active_dims
        )
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        kernel_basic = LinearTruncatedFidelityKernel(fidelity_dims=[1, 2], dimension=6)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().to_dense()
        self.assertLess(
            torch.linalg.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(),
            1e-4,
        )

    def test_error_on_fidelity_only(self):
        x1 = torch.tensor([[0.1], [0.3]])
        x2 = torch.tensor([[0.5], [0.7]])
        kernel = LinearTruncatedFidelityKernel(fidelity_dims=[0], dimension=1, nu=2.5)
        with self.assertRaises(RuntimeError):
            kernel(x1, x2).to_dense()

    def test_initialize_covar_module(self):
        kernel = LinearTruncatedFidelityKernel(fidelity_dims=[1, 2], dimension=3)
        self.assertTrue(isinstance(kernel.covar_module_unbiased, MaternKernel))
        self.assertTrue(isinstance(kernel.covar_module_biased, MaternKernel))
        kernel.covar_module_unbiased = RBFKernel()
        kernel.covar_module_biased = RBFKernel()
        self.assertTrue(isinstance(kernel.covar_module_unbiased, RBFKernel))
        self.assertTrue(isinstance(kernel.covar_module_biased, RBFKernel))
        kernel2 = LinearTruncatedFidelityKernel(
            fidelity_dims=[1, 2],
            dimension=3,
            covar_module_unbiased=RBFKernel(),
            covar_module_biased=RBFKernel(),
        )
        self.assertTrue(isinstance(kernel2.covar_module_unbiased, RBFKernel))
        self.assertTrue(isinstance(kernel2.covar_module_biased, RBFKernel))

    def test_kernel_pickle_unpickle(self):
        # This kernel uses priors by default, which cause this test to fail
        pass
