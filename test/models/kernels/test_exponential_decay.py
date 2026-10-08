#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.constraints import Interval, Positive
from gpytorch.priors.torch_priors import GammaPrior, NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestExponentialDecayKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return ExponentialDecayKernel(**kwargs)

    def test_subset_active_compute_exponential_decay_function(self):
        a = torch.tensor([1.0, 2.0]).view(2, 1)
        a_p = torch.tensor([3.0, 4.0]).view(2, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.tensor([2.0, 4.0]).view(2, 1)
        lengthscale = 1
        power = 1
        offset = 1

        kernel = ExponentialDecayKernel(active_dims=[0])
        kernel.initialize(lengthscale=lengthscale, power=power, offset=offset)
        kernel.eval()

        diff = torch.tensor([[4.0, 6.0], [5.0, 7.0]])
        actual = offset + diff.pow(-power)
        res = kernel(a, b).to_dense()

        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_computes_exponential_decay_function(self):
        a = torch.tensor([1.0, 2.0]).view(2, 1)
        b = torch.tensor([2.0, 4.0]).view(2, 1)
        lengthscale = 1
        power = 1
        offset = 1

        kernel = ExponentialDecayKernel()
        kernel.initialize(lengthscale=lengthscale, power=power, offset=offset)
        kernel.eval()

        diff = torch.tensor([[4.0, 6.0], [5.0, 7.0]])
        actual = offset + torch.tensor([1.0]).div(diff.pow(power))
        res = kernel(a, b).to_dense()

        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_subset_active_exponential_decay_function_batch(self):
        a = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]).view(2, 2, 2)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).view(2, 2, 1)
        lengthscale = 1
        power = 1
        offset = 1

        kernel = ExponentialDecayKernel(batch_shape=torch.Size([2]), active_dims=[0])
        kernel.initialize(lengthscale=lengthscale, power=power, offset=offset)
        kernel.eval()

        actual = torch.zeros(2, 2, 2)

        diff = torch.tensor([[7.0, 8.0], [8.0, 9.0]])
        actual[0, :, :] = offset + torch.tensor([1.0]).div(diff.pow(power))

        diff = torch.tensor([[11.0, 12.0], [12.0, 13.0]])
        actual[1, :, :] = offset + torch.tensor([1.0]).div(diff.pow(power))

        res = kernel(a, b).to_dense()
        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_computes_exponential_decay_function_batch(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(2, 2, 1)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).view(2, 2, 1)
        lengthscale = 1
        power = 1
        offset = 1

        kernel = ExponentialDecayKernel(batch_shape=torch.Size([2]))
        kernel.initialize(lengthscale=lengthscale, power=power, offset=offset)
        kernel.eval()

        actual = torch.zeros(2, 2, 2)

        diff = torch.tensor([[7.0, 8.0], [8.0, 9.0]])
        actual[0, :, :] = offset + diff.pow(-power)

        diff = torch.tensor([[11.0, 12.0], [12.0, 13.0]])
        actual[1, :, :] = offset + diff.pow(-power)

        res = kernel(a, b).to_dense()
        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = ExponentialDecayKernel()
        kernel.initialize(lengthscale=1)
        actual_value = torch.tensor(1.0).view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = ExponentialDecayKernel(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([1.0, 2.0])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.linalg.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_offset(self):
        kernel = ExponentialDecayKernel()
        kernel.initialize(offset=1)
        actual_value = torch.tensor(1.0).view_as(kernel.offset)
        self.assertLess(torch.linalg.norm(kernel.offset - actual_value), 1e-5)

    def test_initialize_offset_batch(self):
        kernel = ExponentialDecayKernel(batch_shape=torch.Size([2]))
        off_init = torch.tensor([1.0, 2.0])
        kernel.initialize(offset=off_init)
        actual_value = off_init.view_as(kernel.offset)
        self.assertLess(torch.linalg.norm(kernel.offset - actual_value), 1e-5)

    def test_initialize_power(self):
        kernel = ExponentialDecayKernel()
        kernel.initialize(power=1)
        actual_value = torch.tensor(1.0).view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_initialize_power_batch(self):
        kernel = ExponentialDecayKernel(batch_shape=torch.Size([2]))
        power_init = torch.tensor([1.0, 2.0])
        kernel.initialize(power=power_init)
        actual_value = power_init.view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_initialize_power_prior(self):
        kernel = ExponentialDecayKernel()
        kernel.power_prior = NormalPrior(1, 1)
        self.assertTrue(isinstance(kernel.power_prior, NormalPrior))
        kernel2 = ExponentialDecayKernel(power_prior=GammaPrior(1, 1))
        self.assertTrue(isinstance(kernel2.power_prior, GammaPrior))

    def test_initialize_offset_prior(self):
        kernel = ExponentialDecayKernel()
        kernel.offset_prior = NormalPrior(1, 1)
        self.assertTrue(isinstance(kernel.offset_prior, NormalPrior))
        kernel2 = ExponentialDecayKernel(offset_prior=GammaPrior(1, 1))
        self.assertTrue(isinstance(kernel2.offset_prior, GammaPrior))

    def test_constraints(self):
        power_constraint = Interval(2.0, 3.0)
        offset_constraint = Interval(0.0, 0.1)
        kernel = ExponentialDecayKernel(
            power_constraint=power_constraint, offset_constraint=offset_constraint
        )
        self.assertIs(kernel.raw_power_constraint, power_constraint)
        self.assertIs(kernel.raw_offset_constraint, offset_constraint)
        self.assertTrue(2.0 <= kernel.power.item() <= 3.0)
        self.assertTrue(0.0 <= kernel.offset.item() <= 0.1)
        # The defaults are positivity constraints.
        kernel = ExponentialDecayKernel()
        self.assertIsInstance(kernel.raw_power_constraint, Positive)
        self.assertIsInstance(kernel.raw_offset_constraint, Positive)
