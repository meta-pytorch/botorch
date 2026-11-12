#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.priors.torch_priors import GammaPrior, NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestDownsamplingKernel(BotorchTestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return DownsamplingKernel(**kwargs)

    def create_data_no_batch(self):
        return torch.rand(50, 1)

    def create_data_single_batch(self):
        return torch.rand(2, 3, 1)

    def create_data_double_batch(self):
        return torch.rand(3, 2, 50, 1)

    def test_active_dims_list(self):
        # this makes no sense for this kernel since d=1
        pass

    def test_active_dims_range(self):
        # this makes no sense for this kernel since d=1
        pass

    def test_subset_active_compute_downsampling_function(self):
        a = torch.tensor([0.1, 0.2]).view(2, 1)
        a_p = torch.tensor([0.3, 0.4]).view(2, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.tensor([0.2, 0.4]).view(2, 1)
        power = 1
        offset = 1

        kernel = DownsamplingKernel(active_dims=[0])
        kernel.initialize(power=power, offset=offset)
        kernel.eval()

        diff = torch.tensor([[0.72, 0.54], [0.64, 0.48]])
        actual = offset + diff.pow(1 + power)
        res = kernel(a, b).to_dense()

        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_computes_downsampling_function(self):
        a = torch.tensor([0.1, 0.2]).view(2, 1)
        b = torch.tensor([0.2, 0.4]).view(2, 1)
        power = 1
        offset = 1

        kernel = DownsamplingKernel()
        kernel.initialize(power=power, offset=offset)
        kernel.eval()

        diff = torch.tensor([[0.72, 0.54], [0.64, 0.48]])
        actual = offset + diff.pow(1 + power)
        res = kernel(a, b).to_dense()

        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_subset_computes_active_downsampling_function_batch(self):
        a = torch.tensor([[0.1, 0.2, 0.2], [0.3, 0.4, 0.2], [0.5, 0.5, 0.5]]).view(
            3, 3, 1
        )
        a_p = torch.tensor([[0.1, 0.2, 0.2], [0.3, 0.4, 0.2], [0.5, 0.5, 0.5]]).view(
            3, 3, 1
        )
        a = torch.cat((a, a_p), 2)
        b = torch.tensor([[0.5, 0.6, 0.1], [0.7, 0.8, 0.2], [0.6, 0.6, 0.5]]).view(
            3, 3, 1
        )
        power = 1
        offset = 1
        kernel = DownsamplingKernel(batch_shape=torch.Size([3]), active_dims=[0])
        kernel.initialize(power=power, offset=offset)
        kernel.eval()
        res = kernel(a, b).to_dense()

        actual = torch.zeros(3, 3, 3)

        diff = torch.tensor([[0.45, 0.36, 0.81], [0.4, 0.32, 0.72], [0.4, 0.32, 0.72]])
        actual[0, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor(
            [[0.21, 0.14, 0.56], [0.18, 0.12, 0.48], [0.24, 0.16, 0.64]]
        )
        actual[1, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor([[0.2, 0.2, 0.25], [0.2, 0.2, 0.25], [0.2, 0.2, 0.25]])
        actual[2, :, :] = offset + diff.pow(1 + power)
        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_computes_downsampling_function_batch(self):
        a = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]]).view(3, 2, 1)
        b = torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.6, 0.6]]).view(3, 2, 1)
        power = 1
        offset = 1

        kernel = DownsamplingKernel(batch_shape=torch.Size([3]))
        kernel.initialize(power=power, offset=offset)
        kernel.eval()
        res = kernel(a, b).to_dense()

        actual = torch.zeros(3, 2, 2)

        diff = torch.tensor([[0.45, 0.36], [0.4, 0.32]])
        actual[0, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor([[0.21, 0.14], [0.18, 0.12]])
        actual[1, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
        actual[2, :, :] = offset + diff.pow(1 + power)
        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_initialize_offset(self):
        kernel = DownsamplingKernel()
        kernel.initialize(offset=1)
        actual_value = torch.tensor(1.0).view_as(kernel.offset)
        self.assertLess(torch.linalg.norm(kernel.offset - actual_value), 1e-5)

    def test_initialize_offset_batch(self):
        kernel = DownsamplingKernel(batch_shape=torch.Size([2]))
        off_init = torch.tensor([1.0, 2.0])
        kernel.initialize(offset=off_init)
        actual_value = off_init.view_as(kernel.offset)
        self.assertLess(torch.linalg.norm(kernel.offset - actual_value), 1e-5)

    def test_initialize_power(self):
        kernel = DownsamplingKernel()
        kernel.initialize(power=1)
        actual_value = torch.tensor(1.0).view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_initialize_power_batch(self):
        kernel = DownsamplingKernel(batch_shape=torch.Size([2]))
        power_init = torch.tensor([1.0, 2.0])
        kernel.initialize(power=power_init)
        actual_value = power_init.view_as(kernel.power)
        self.assertLess(torch.linalg.norm(kernel.power - actual_value), 1e-5)

    def test_last_dim_is_batch(self):
        a = (
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]])
            .view(3, 2)
            .transpose(-1, -2)
        )
        b = (
            torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.6, 0.6]])
            .view(3, 2)
            .transpose(-1, -2)
        )
        power = 1
        offset = 1

        kernel = DownsamplingKernel()
        kernel.initialize(power=power, offset=offset)
        kernel.eval()
        res = kernel(a, b, last_dim_is_batch=True).to_dense()

        actual = torch.zeros(3, 2, 2)

        diff = torch.tensor([[0.45, 0.36], [0.4, 0.32]])
        actual[0, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor([[0.21, 0.14], [0.18, 0.12]])
        actual[1, :, :] = offset + diff.pow(1 + power)

        diff = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
        actual[2, :, :] = offset + diff.pow(1 + power)
        self.assertLess(torch.linalg.norm(res - actual), 1e-5)

    def test_diag_calculation(self):
        a = torch.tensor([0.1, 0.2]).view(2, 1)
        b = torch.tensor([0.2, 0.4]).view(2, 1)
        power = 1
        offset = 1

        kernel = DownsamplingKernel()
        kernel.initialize(power=power, offset=offset)
        kernel.eval()

        diff = torch.tensor([[0.72, 0.54], [0.64, 0.48]])
        actual = offset + diff.pow(1 + power)
        res = kernel(a, b, diag=True)

        self.assertLess(torch.linalg.norm(res - torch.diag(actual)), 1e-5)

    def test_diag_multiple_dims(self):
        # The diagonal of K(x1, x2) for inputs with more than one dimension.
        for batch_shape, dtype in itertools.product(
            (torch.Size([]), torch.Size([2])), (torch.float, torch.double)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            kernel = DownsamplingKernel(batch_shape=batch_shape).to(**tkwargs)
            kernel.initialize(power=0.5, offset=0.2)
            x1 = torch.rand(*batch_shape, 5, 3, **tkwargs)
            x2 = torch.rand(*batch_shape, 5, 3, **tkwargs)
            for x2_ in (x1, x2):
                self.assertAllClose(
                    kernel(x1, x2_, diag=True),
                    kernel(x1, x2_).to_dense().diagonal(dim1=-2, dim2=-1),
                )

    def test_initialize_power_prior(self):
        kernel = DownsamplingKernel()
        kernel.power_prior = NormalPrior(1, 1)
        self.assertTrue(isinstance(kernel.power_prior, NormalPrior))
        kernel2 = DownsamplingKernel(power_prior=GammaPrior(1, 1))
        self.assertTrue(isinstance(kernel2.power_prior, GammaPrior))

    def test_initialize_offset_prior(self):
        kernel = DownsamplingKernel()
        kernel.offset_prior = NormalPrior(1, 1)
        self.assertTrue(isinstance(kernel.offset_prior, NormalPrior))
        kernel2 = DownsamplingKernel(offset_prior=GammaPrior(1, 1))
        self.assertTrue(isinstance(kernel2.offset_prior, GammaPrior))
